"""Two ways to produce the overtake-attempt dataset train_overtake_model.py
consumes, both with the identical output schema:

  generate_dataset()      -- SYNTHETIC. Hand-authored ground-truth function
                              with injected noise. Not real telemetry; must
                              never be presented as such. Useful for building
                              and smoke-testing the training pipeline without
                              network access.

  ingest_fastf1_sessions() -- REAL. Pulls actual race telemetry via FastF1
                              (https://github.com/theOehrly/Fast-F1) and
                              derives overtake-attempt windows from it. This
                              sandbox has no route to F1's timing/schedule
                              servers, so this must be run on your own
                              machine -- see the setup steps provided
                              alongside this code. Every feature FastF1
                              doesn't expose directly (grip, overtake
                              difficulty, braking-zone length) is a
                              documented proxy, called out inline at the
                              point it's computed -- not a measured value.

Run `python scripts/generate_scenarios.py --real` after installing fastf1
to use the real path; with no flags it uses the synthetic generator.

Feature set follows the build guide's "Suggested features" list, plus one
engineered addition (`action_energy_level`) needed because ApexGuard scores
pass probability *per candidate action*, not just per race state -- the
guide's feature list alone describes state, not the deploy decision itself.
"""

import csv
import random
import re
from pathlib import Path

import numpy as np

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "training" / "overtake_windows.csv"
REAL_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "training" / "overtake_windows_real.csv"

# DRS status codes FastF1 reports as "open" on the car-data DRS channel.
# 0/1 = off, 8 = eligible/detected but not yet open, 10/12/14 = open
# (the exact code varies by season/car). See fastf1.api.car_data docs.
_DRS_OPEN_CODES = {10, 12, 14}

# Hand-authored tyre-grip decay curve, same shape/spirit as
# rival_estimator.py's stint_penalty heuristic: grip isn't a channel FastF1
# exposes (no team shares real grip telemetry), so this is a documented
# proxy from Compound + TyreLife, not a measured value.
_COMPOUND_BASE_GRIP = {"SOFT": 0.97, "MEDIUM": 0.90, "HARD": 0.84, "INTERMEDIATE": 0.70, "WET": 0.60}


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _tyre_grip_estimate(compound: str, tyre_life_laps: float) -> float:
    base = _COMPOUND_BASE_GRIP.get(str(compound).upper(), 0.85)
    if tyre_life_laps is None or (isinstance(tyre_life_laps, float) and np.isnan(tyre_life_laps)):
        tyre_life_laps = 0.0  # missing TyreLife (e.g. an out-lap) -- assume fresh rather than propagate NaN
    decay = min(0.45, 0.006 * max(0.0, tyre_life_laps))  # ~0.6%/lap, capped
    return round(max(0.35, base - decay), 3)

# 6 synthetic "historical" tracks + 2 held-out tracks for generalization testing,
# per the guide's dataset split table.
TRACKS = {
    "hist_monza_like": {"overtake_difficulty": 0.30, "straight_bias": 650},
    "hist_spa_like": {"overtake_difficulty": 0.35, "straight_bias": 580},
    "hist_bahrain_like": {"overtake_difficulty": 0.45, "straight_bias": 480},
    "hist_silverstone_like": {"overtake_difficulty": 0.50, "straight_bias": 420},
    "hist_suzuka_like": {"overtake_difficulty": 0.62, "straight_bias": 300},
    "hist_singapore_like": {"overtake_difficulty": 0.80, "straight_bias": 220},
    "holdout_jeddah_like": {"overtake_difficulty": 0.40, "straight_bias": 550},
    "holdout_monaco_like": {"overtake_difficulty": 0.92, "straight_bias": 150},
}

ACTION_ENERGY_LEVEL = {
    "HARVEST": -1,
    "HOLD": 0,
    "PARTIAL_DEPLOY": 1,
    "FULL_DEPLOY": 2,
}

FEATURE_COLUMNS = [
    "target_gap_s",
    "relative_speed_kph",
    "drs_available",
    "straight_remaining_m",
    "braking_zone_m",
    "track_overtake_difficulty",
    "ego_tyre_grip_estimate",
    "target_tyre_grip_estimate",
    "target_tyre_uncertainty",
    "laps_remaining",
    "rear_gap_s",
    "rear_drs_risk",
    "cars_within_3s",
    "post_pass_traffic_gap_s",
    "recent_sector_delta_s",
    "action_energy_level",
]


def _ground_truth_pass_probability(row: dict) -> float:
    """Hand-authored logit combining the features in a motorsport-plausible
    way. This stands in for 'reality' when generating synthetic labels --
    a real pipeline has no equivalent function, it just observes outcomes."""
    logit = (
        -1.5 * row["target_gap_s"]
        + 0.028 * row["relative_speed_kph"]
        + (0.85 if row["drs_available"] else 0.0)
        + 0.0009 * row["straight_remaining_m"]
        - 1.3 * row["track_overtake_difficulty"]
        + 0.7 * (row["ego_tyre_grip_estimate"] - row["target_tyre_grip_estimate"])
        - 0.4 * row["rear_drs_risk"]
        + 0.32 * row["action_energy_level"]
        + 0.15 * row["recent_sector_delta_s"] * -1
        - 0.05 * row["cars_within_3s"]
    )
    prob = 1.0 / (1.0 + pow(2.71828, -logit))
    return prob


def generate_dataset(n_per_track: int = 260, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for track_id, meta in TRACKS.items():
        for _ in range(n_per_track):
            action = rng.choice(list(ACTION_ENERGY_LEVEL.keys()))
            row = {
                "track_id": track_id,
                "target_gap_s": round(rng.uniform(0.1, 2.0), 3),
                "relative_speed_kph": round(rng.uniform(-5, 30), 2),
                "drs_available": rng.random() < 0.55,
                "straight_remaining_m": round(max(50, rng.gauss(meta["straight_bias"], 150)), 1),
                "braking_zone_m": round(rng.uniform(90, 180), 1),
                "track_overtake_difficulty": meta["overtake_difficulty"],
                "ego_tyre_grip_estimate": round(rng.uniform(0.55, 0.95), 3),
                "target_tyre_grip_estimate": round(rng.uniform(0.5, 0.95), 3),
                "target_tyre_uncertainty": round(rng.uniform(0.05, 0.45), 3),
                "laps_remaining": rng.randint(1, 55),
                "rear_gap_s": round(rng.uniform(0.1, 3.5), 3),
                "cars_within_3s": rng.randint(0, 6),
                "post_pass_traffic_gap_s": round(rng.uniform(0.2, 3.0), 3),
                "recent_sector_delta_s": round(rng.gauss(0, 0.2), 3),
                "action": action,
                "action_energy_level": ACTION_ENERGY_LEVEL[action],
            }
            row["rear_drs_risk"] = round(max(0.0, 1.0 - row["rear_gap_s"] / 1.5), 3)

            prob = _ground_truth_pass_probability(row)
            noisy_prob = min(0.98, max(0.02, prob + rng.gauss(0, 0.08)))
            row["pass_success"] = 1 if rng.random() < noisy_prob else 0
            rows.append(row)
    return rows


def _time_at_distance(telemetry, distance: float) -> float:
    """Linear-interpolate SessionTime (seconds) at a given track distance
    from a telemetry frame that already has add_distance() applied.
    Distance must be monotonically increasing over the slice, which holds
    for a single-lap telemetry frame."""
    d = telemetry["Distance"].to_numpy(dtype=float)
    t = telemetry["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
    return float(np.interp(distance, d, t))


def _speed_at_distance(telemetry, distance: float) -> float:
    d = telemetry["Distance"].to_numpy(dtype=float)
    s = telemetry["Speed"].to_numpy(dtype=float)
    return float(np.interp(distance, d, s))


def _next_corner_distance(corner_distances: list[float], lap_length: float, from_distance: float) -> float:
    ahead = [c for c in corner_distances if c > from_distance]
    if ahead:
        return min(ahead)
    # No corner left on this lap -- wrap to the first corner of the next lap.
    return min(corner_distances) + lap_length if corner_distances else lap_length


def _drs_window_start_distance(lap_telemetry) -> float | None:
    """First distance in the lap where the DRS channel flips into an
    'open' state -- used as the sampling point for an overtake attempt
    window. Returns None if DRS was never open on this lap (e.g. wet
    session, or DRS disabled that lap)."""
    drs = lap_telemetry["DRS"].to_numpy()
    dist = lap_telemetry["Distance"].to_numpy(dtype=float)
    for i, val in enumerate(drs):
        if int(val) in _DRS_OPEN_CODES:
            return float(dist[i])
    return None


def ingest_fastf1_sessions(
    seasons: list[int],
    sessions: list[str],
    holdout_events: list[str] | None = None,
    cache_dir: str = ".fastf1_cache",
    max_events_per_season: int | None = None,
) -> list[dict]:
    """Real ingestion pipeline over FastF1 (https://github.com/theOehrly/Fast-F1,
    MIT licensed, v3.8.3 as of this writing). Requires network access to
    jolpica-f1 (schedule/results) and F1's live timing feed (telemetry) --
    this sandbox cannot reach either, so this must be run on your own
    machine. See the setup steps in the chat response for exact commands.

    Output schema matches generate_dataset() exactly, so nothing downstream
    (train_overtake_model.py, FEATURE_COLUMNS, write_csv()) needs to change.

    `holdout_events` is a list of event names (fuzzy-matched the same way
    fastf1.get_session's `gp` argument is, e.g. ["Monaco", "Suzuka"]) whose
    rows get the `holdout_` track_id prefix train_overtake_model.py already
    looks for, so real data slots into the existing track-holdout eval.

    Every feature below is derived, not measured -- FastF1/F1 timing does
    not expose "grip", "overtake difficulty", or a live gap-to-every-car
    feed directly. Each approximation is called out inline; treat this as
    a documented modeling choice; refine any of them as you validate
    against real outcomes.
    """
    import fastf1  # imported lazily -- only required for this real path

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)  # required, or every call re-downloads

    rows: list[dict] = []
    holdout_events = holdout_events or []

    for season in seasons:
        schedule = fastf1.get_event_schedule(season)
        events = schedule[schedule["EventFormat"] != "testing"]
        if max_events_per_season is not None:
            events = events.iloc[:max_events_per_season]

        for _, event in events.iterrows():
            is_holdout = any(h.lower() in str(event["EventName"]).lower() for h in holdout_events)
            track_id = ("holdout_" if is_holdout else "") + _slugify(event["EventName"])

            for session_name in sessions:  # e.g. ["R"] for the race, ["R", "Q"] for both
                try:
                    session = fastf1.get_session(season, event["RoundNumber"], session_name)
                    session.load(telemetry=True, laps=True, weather=False)
                    if session.laps is None or session.laps.empty or session.laps["LapNumber"].dropna().empty:
                        raise RuntimeError("no lap data available for this session")
                    circuit_info = session.get_circuit_info()
                    corner_distances = (
                        circuit_info.corners["Distance"].tolist() if circuit_info is not None else []
                    )
                    total_laps = int(session.laps["LapNumber"].max())
                except Exception as exc:  # noqa: BLE001 -- one bad session shouldn't kill the run
                    print(f"  skipping {season} {event['EventName']} {session_name}: {exc}")
                    continue

                lap_numbers = sorted(session.laps["LapNumber"].dropna().unique())
                for lap_number in lap_numbers:
                    laps_this_lap = session.laps[session.laps["LapNumber"] == lap_number]
                    by_position = laps_this_lap.dropna(subset=["Position"]).sort_values("Position")

                    # Build {DriverNumber: (Position, telemetry-with-distance)} once per lap.
                    tel_by_driver = {}
                    for _, lap in by_position.iterrows():
                        try:
                            tel = lap.get_car_data().add_distance()
                        except Exception:  # noqa: BLE001 -- missing telemetry for this driver/lap
                            continue
                        tel_by_driver[lap["DriverNumber"]] = (lap, tel)

                    if len(tel_by_driver) < 2:
                        continue
                    lap_length = max(tel["Distance"].max() for _, tel in tel_by_driver.values())

                    # Pair up every nose-to-tail combination (adjacent race positions).
                    ordered = by_position[by_position["DriverNumber"].isin(tel_by_driver)]
                    ordered_nums = ordered["DriverNumber"].tolist()
                    for leader_num, chaser_num in zip(ordered_nums, ordered_nums[1:]):
                        leader_lap, leader_tel = tel_by_driver[leader_num]
                        chaser_lap, chaser_tel = tel_by_driver[chaser_num]

                        window_distance = _drs_window_start_distance(chaser_tel)
                        if window_distance is None:
                            continue  # DRS never opened for the chaser this lap -- no attempt window

                        try:
                            t_leader = _time_at_distance(leader_tel, window_distance)
                            t_chaser = _time_at_distance(chaser_tel, window_distance)
                        except Exception:  # noqa: BLE001
                            continue
                        gap_s = t_chaser - t_leader
                        if gap_s <= 0 or gap_s > 2.0:
                            continue  # not a credible DRS-range attempt

                        rel_speed = _speed_at_distance(chaser_tel, window_distance) - _speed_at_distance(
                            leader_tel, window_distance
                        )
                        next_corner = _next_corner_distance(corner_distances, lap_length, window_distance)
                        straight_remaining_m = max(0.0, next_corner - window_distance)

                        # rear_gap_s / cars_within_3s: gaps from every other car on
                        # track to the chaser, evaluated at the same distance point.
                        gaps_to_chaser = []
                        for other_num, (_, other_tel) in tel_by_driver.items():
                            if other_num in (leader_num, chaser_num):
                                continue
                            try:
                                t_other = _time_at_distance(other_tel, window_distance)
                            except Exception:  # noqa: BLE001
                                continue
                            gaps_to_chaser.append(t_other - t_chaser)
                        behind = [g for g in gaps_to_chaser if g > 0]
                        rear_gap_s = round(min(behind), 3) if behind else 99.0
                        cars_within_3s = sum(1 for g in gaps_to_chaser if abs(g) <= 3.0)

                        chaser_grip = _tyre_grip_estimate(chaser_lap["Compound"], chaser_lap["TyreLife"])
                        leader_grip = _tyre_grip_estimate(leader_lap["Compound"], leader_lap["TyreLife"])

                        # Rough proxy for "is the leader's pace trending up or down":
                        # first-sector time vs. this lap's average sector time.
                        # Real per-lap trend (vs. the leader's own recent laps)
                        # would be a better signal once you're validating this
                        # against real outcomes -- flagged as a refinement target.
                        sector_delta = 0.0
                        try:
                            sector1_s = leader_lap["Sector1Time"].total_seconds()
                            lap_s = leader_lap["LapTime"].total_seconds()
                            if sector1_s and lap_s:
                                sector_delta = round(sector1_s - lap_s / 3.0, 3)
                        except Exception:  # noqa: BLE001 -- missing sector/lap time for this lap
                            sector_delta = 0.0

                        # Outcome label: did track position actually swap between
                        # exactly these two drivers by the next lap?
                        next_lap = session.laps[
                            (session.laps["LapNumber"] == lap_number + 1)
                            & (session.laps["DriverNumber"].isin([leader_num, chaser_num]))
                        ]
                        pass_success = 0
                        if len(next_lap) == 2:
                            pos_next = next_lap.set_index("DriverNumber")["Position"]
                            if chaser_num in pos_next and leader_num in pos_next:
                                pass_success = int(pos_next[chaser_num] < pos_next[leader_num])

                        row = {
                            "track_id": track_id,
                            "action": "FULL_DEPLOY",  # real timing data doesn't expose the driver's
                            # discrete energy-deployment choice, so every real-data row is logged
                            # as the action actually available in a genuine DRS+attack window; if
                            # per-lap ERS deployment traces become available, replace this constant.
                            "action_energy_level": ACTION_ENERGY_LEVEL["FULL_DEPLOY"],
                            "target_gap_s": round(gap_s, 3),
                            "relative_speed_kph": round(rel_speed, 2),
                            "drs_available": True,
                            "straight_remaining_m": round(straight_remaining_m, 1),
                            "braking_zone_m": 120.0,  # not exposed by FastF1; fixed generic proxy
                            "track_overtake_difficulty": None,  # filled in post-hoc, see below
                            "ego_tyre_grip_estimate": chaser_grip,
                            "target_tyre_grip_estimate": leader_grip,
                            "target_tyre_uncertainty": 0.1,  # real data -- lower than the estimator's prior
                            "laps_remaining": max(0, total_laps - int(lap_number)),
                            "rear_gap_s": rear_gap_s,
                            "cars_within_3s": cars_within_3s,
                            "post_pass_traffic_gap_s": rear_gap_s,  # same field once passed, as first approximation
                            "recent_sector_delta_s": round(sector_delta, 3),
                            "pass_success": pass_success,
                        }
                        row["rear_drs_risk"] = round(max(0.0, 1.0 - row["rear_gap_s"] / 1.5), 3)
                        rows.append(row)

    # track_overtake_difficulty isn't a FastF1 channel -- derive it empirically
    # per track_id as (1 - observed pass rate), exactly the "historical
    # pass-rate-per-DRS-zone" proxy flagged as the right approach in
    # battle_graph.py's module docstring.
    #
    # IMPORTANT: this must be computed from TRAINING tracks' own labels only.
    # A track tagged holdout_ here is deliberately meant to simulate a track
    # the model has never seen -- computing its difficulty from its own
    # pass_success column would leak the label straight into a feature and
    # inflate holdout metrics in a way that won't hold up on a genuinely new
    # track later. Holdout tracks instead get the mean difficulty observed
    # across training tracks, which is the honest prior a real deployment
    # would actually have.
    by_track: dict[str, list[dict]] = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row)

    train_difficulties = []
    for track_id, track_rows in by_track.items():
        if track_id.startswith("holdout_"):
            continue
        pass_rate = sum(r["pass_success"] for r in track_rows) / len(track_rows)
        difficulty = round(min(0.95, max(0.05, 1.0 - pass_rate)), 3)
        for r in track_rows:
            r["track_overtake_difficulty"] = difficulty
        train_difficulties.append(difficulty)

    fallback_difficulty = round(sum(train_difficulties) / len(train_difficulties), 3) if train_difficulties else 0.5
    for track_id, track_rows in by_track.items():
        if track_id.startswith("holdout_"):
            for r in track_rows:
                r["track_overtake_difficulty"] = fallback_difficulty

    return rows


def write_csv(rows: list[dict], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track_id", "action", "pass_success"] + FEATURE_COLUMNS
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic or real (FastF1) overtake-window data.")
    parser.add_argument(
        "--real", action="store_true",
        help="Use ingest_fastf1_sessions() against real FastF1 data instead of the synthetic generator. "
             "Requires `pip install fastf1` and network access to F1's timing/schedule sources.",
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024],
                         help="Seasons to ingest, only used with --real.")
    parser.add_argument("--sessions", type=str, nargs="+", default=["R"],
                         help="Session identifiers to ingest (e.g. R Q), only used with --real.")
    parser.add_argument("--holdout-events", type=str, nargs="+", default=["Monaco", "Suzuka"],
                         help="Event names held out for generalization eval, only used with --real.")
    parser.add_argument("--max-events-per-season", type=int, default=None,
                         help="Cap events per season while testing the pipeline, only used with --real.")
    args = parser.parse_args()

    if args.real:
        rows = ingest_fastf1_sessions(
            seasons=args.seasons,
            sessions=args.sessions,
            holdout_events=args.holdout_events,
            max_events_per_season=args.max_events_per_season,
        )
        write_csv(rows, path=REAL_OUTPUT_PATH)
        print(f"Wrote {len(rows)} REAL overtake-window rows (FastF1) to {REAL_OUTPUT_PATH}")
        print(f"Tracks: {sorted({r['track_id'] for r in rows})}")
    else:
        rows = generate_dataset()
        write_csv(rows)
        print(f"Wrote {len(rows)} synthetic overtake-window rows to {OUTPUT_PATH}")
        print(f"Tracks: {list(TRACKS.keys())}")
