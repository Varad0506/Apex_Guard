"""Phase 2: train a calibrated logistic regression pass-probability
classifier and export it as a joblib artifact that overtake_probability.py
loads at runtime, falling back to the Phase 1 heuristic if the artifact is
missing (per the fail-safe contract: 'Model missing -> Degraded verified
mode', never a hard failure).

Usage:
    python scripts/train_overtake_model.py

Track-holdout evaluation, not random row splitting: the two `holdout_*`
tracks in the synthetic dataset are entirely excluded from training and
used only for the generalization report, per the build guide's explicit
warning against random-splitting ticks from the same race/track.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from generate_scenarios import FEATURE_COLUMNS, OUTPUT_PATH

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "overtake_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "overtake_model_metrics.json"

HOLDOUT_TRACK_PREFIX = "holdout_"


def load_data(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or OUTPUT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/generate_scenarios.py first "
            "(or point this script at a real FastF1-derived CSV with the same schema)."
        )
    df = pd.read_csv(path)
    df["drs_available"] = df["drs_available"].astype(int)

    # Real FastF1-derived rows can carry NaNs the synthetic generator never
    # produces (e.g. a lap with missing sector/tyre data upstream). Drop
    # incomplete rows rather than silently feeding NaN into StandardScaler,
    # which would raise deep inside the pipeline instead of here.
    required = FEATURE_COLUMNS + ["pass_success", "track_id"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"load_data: dropped {dropped}/{before} rows with missing values in {required}")
    return df


def train(csv_path: Path | None = None):
    df = load_data(csv_path)

    train_df = df[~df["track_id"].str.startswith(HOLDOUT_TRACK_PREFIX)]
    holdout_df = df[df["track_id"].str.startswith(HOLDOUT_TRACK_PREFIX)]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["pass_success"]
    X_holdout = holdout_df[FEATURE_COLUMNS]
    y_holdout = holdout_df["pass_success"]

    # class_weight="balanced" -- pass_success is ~13% positive in real data
    # (rare event), so an unweighted model tends to just predict "no pass"
    # and still score high accuracy without learning anything useful.
    # CalibratedClassifierCV's sigmoid calibration re-maps the resulting
    # probabilities back toward the true base rate, so this doesn't leave
    # predicted probabilities biased high.
    base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base_model, method="sigmoid", cv=5)),
    ])
    pipeline.fit(X_train, y_train)

    # Baseline comparison: uncalibrated logistic regression, same features.
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)

    def evaluate(model, X, y, label):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return {
            "split": label,
            "n": int(len(y)),
            "accuracy": round(accuracy_score(y, preds), 4),
            "auc": round(roc_auc_score(y, proba), 4),
            "log_loss": round(log_loss(y, proba), 4),
            "brier_score": round(brier_score_loss(y, proba), 4),
        }

    metrics = {
        "calibrated_train_tracks": evaluate(pipeline, X_train, y_train, "train_tracks"),
        "calibrated_holdout_tracks": evaluate(pipeline, X_holdout, y_holdout, "holdout_tracks"),
        "baseline_uncalibrated_holdout_tracks": evaluate(baseline, X_holdout, y_holdout, "holdout_tracks_baseline"),
        "feature_columns": FEATURE_COLUMNS,
        "train_tracks": sorted(train_df["track_id"].unique().tolist()),
        "holdout_tracks": sorted(holdout_df["track_id"].unique().tolist()),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model artifact -> {MODEL_PATH}")
    print(f"Saved metrics        -> {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import argparse
    from generate_scenarios import REAL_OUTPUT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Train on overtake_windows_real.csv instead of the synthetic set.")
    args = parser.parse_args()

    train(REAL_OUTPUT_PATH if args.real else None)
