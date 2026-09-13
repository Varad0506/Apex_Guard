"""FastF1 -> ApexGuard replay adapter.

This module deliberately keeps FastF1 optional. The core API can run without
FastF1; installing the package enables real-session replay generation.

A replay is a sampled, frontend-friendly stream of DecisionRequest-shaped
states derived from one ego lap and one target lap from the same FastF1
session. It is a visualization/replay adapter, not a claim of race-grade
vehicle-state reconstruction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "replays"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _require_fastf1():
    try:
        import fastf1  # type: ignore
        return fastf1
    except ImportError as exc:
        raise RuntimeError(
            "FastF1 is not installed. Install it with: pip install fastf1"
        ) from exc


def _first_value(row: Any, names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        try:
            value = row[name]
            if value is not None:
                return float(value)
        except (KeyError, TypeError, ValueError):
            pass
    return default


def _pick_lap(session, driver: str, lap_number: int | None):
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        raise ValueError(f"Driver {driver} has no laps in this session")
    if lap_number is not None:
        selected = laps[laps["LapNumber"] == lap_number]
        if selected.empty:
            raise ValueError(f"Driver {driver} has no lap {lap_number}")
        return selected.iloc[0]
    return laps.pick_fastest()


def _telemetry_for_lap(lap):
    """Return lightweight car telemetry for a lap.

    Do not use ``lap.get_telemetry()`` here: FastF1 builds that combined
    telemetry by merging position data, which can require copying tens of
    thousands of position samples even though the replay only needs speed,
    throttle, brake, DRS and distance.  ``get_car_data().add_distance()``
    provides those channels without the expensive position-data merge.
    """
    telemetry = lap.get_car_data().add_distance()
    if telemetry is None or telemetry.empty:
        raise ValueError("Selected lap has no car telemetry")
    telemetry = telemetry.copy()

    if "Time" in telemetry.columns:
        telemetry["t_s"] = (
            telemetry["Time"] - telemetry["Time"].iloc[0]
        ).dt.total_seconds()
    elif "SessionTime" in telemetry.columns:
        telemetry["t_s"] = (
            telemetry["SessionTime"] - telemetry["SessionTime"].iloc[0]
        ).dt.total_seconds()
    else:
        telemetry["t_s"] = telemetry.index.to_series().astype(float)
    return telemetry


def _sample_series(telemetry, samples: int):
    import numpy as np
    samples = max(20, min(int(samples), 500))
    t = telemetry["t_s"].to_numpy(dtype=float)
    if len(t) < 2:
        return telemetry.iloc[[0]].copy()
    grid = np.linspace(float(t[0]), float(t[-1]), samples)
    rows = []
    numeric_columns = [c for c in telemetry.columns if c not in {"t_s", "Source", "Driver"}]
    for tt in grid:
        idx = int(np.argmin(np.abs(t - tt)))
        row = telemetry.iloc[idx].copy()
        row["t_s"] = float(tt)
        for col in numeric_columns:
            try:
                values = telemetry[col].to_numpy(dtype=float)
                row[col] = float(np.interp(tt, t, values))
            except (TypeError, ValueError):
                pass
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)


def build_replay(
    year: int,
    event: str | int,
    session_name: str,
    driver: str,
    target_driver: str | None = None,
    lap_number: int | None = None,
    target_lap_number: int | None = None,
    samples: int = 180,
    cache: bool = True,
) -> dict:
    """Load a FastF1 session and produce a frontend-ready replay payload."""
    fastf1 = _require_fastf1()
    session = fastf1.get_session(year, event, session_name)
    session.load(telemetry=True, laps=True, weather=False, messages=False)

    ego_lap = _pick_lap(session, driver, lap_number)
    target_driver = target_driver or driver
    if target_driver.upper() == driver.upper():
        # Prefer the next-fastest distinct driver when the caller does not
        # specify an opponent. This makes the endpoint useful for demos.
        drivers = [str(d) for d in session.laps["Driver"].dropna().unique() if str(d).upper() != driver.upper()]
        if not drivers:
            raise ValueError("No distinct target driver is available in this session")
        target_driver = drivers[0]
    target_lap = _pick_lap(session, target_driver, target_lap_number or int(ego_lap["LapNumber"]))

    ego = _sample_series(_telemetry_for_lap(ego_lap), samples)
    target = _sample_series(_telemetry_for_lap(target_lap), samples)

    import numpy as np
    n = min(len(ego), len(target))
    frames = []
    duration = float(ego["t_s"].iloc[n - 1])
    ego_distance = ego["Distance"].to_numpy(dtype=float) if "Distance" in ego else np.linspace(0, 5000, n)
    target_distance = target["Distance"].to_numpy(dtype=float) if "Distance" in target else ego_distance + 100.0
    target_speed = target["Speed"].to_numpy(dtype=float) if "Speed" in target else ego["Speed"].to_numpy(dtype=float)
    ego_speed = ego["Speed"].to_numpy(dtype=float) if "Speed" in ego else np.zeros(n)

    for i in range(n):
        speed_kph = max(0.0, float(ego_speed[i]))
        speed_mps = max(1.0, speed_kph / 3.6)
        distance_gap_m = float(target_distance[i] - ego_distance[i])
        gap_s = max(0.0, distance_gap_m / speed_mps)
        rel_speed = float(speed_kph - target_speed[i])
        drs = bool(ego["DRS"].iloc[i] > 0) if "DRS" in ego.columns else False
        throttle = _first_value(ego.iloc[i], ("Throttle",), 100.0)
        brake = _first_value(ego.iloc[i], ("Brake",), 0.0)
        segment = "DRS_STRAIGHT" if drs else ("STRAIGHT" if speed_kph > 240 else "CORNER")
        remaining = max(0.0, float(ego_distance[-1] - ego_distance[i]))
        frames.append({
            "t_s": round(float(ego["t_s"].iloc[i]), 3),
            "request_id": f"fastf1-{year}-{event}-{session_name}-{driver}-{i}",
            "battle_id": f"fastf1-{year}-{event}-{session_name}-{driver}-{target_driver}",
            "timestamp_ms": int(float(ego["t_s"].iloc[i]) * 1000),
            "telemetry_age_ms": 20,
            "ego": {
                "soc_pct": 65.0,
                "speed_kph": round(speed_kph, 2),
                "throttle_pct": round(max(0.0, min(100.0, throttle)), 2),
                "brake_pct": round(max(0.0, min(100.0, brake)), 2),
                "tyre_grip_estimate": 0.82,
                "laps_remaining": max(0, int(session.total_laps or 0) - int(ego_lap["LapNumber"])),
                "current_mode": "HOLD",
            },
            "target": {
                "gap_s": round(gap_s, 3),
                "relative_speed_kph": round(rel_speed, 2),
                "stint_age_laps": max(0, int(ego_lap["LapNumber"])),
                "recent_sector_delta_s": 0.0,
            },
            "traffic": {
                "rear_gap_s": 2.0,
                "cars_within_3s": 1,
                "post_pass_traffic_gap_s": 1.5,
            },
            "track": {
                "track_id": str(session.event.EventName if hasattr(session.event, "EventName") else event),
                "segment_type": segment,
                "drs_available": drs,
                "straight_remaining_m": round(remaining, 1),
                "braking_zone_m": 140.0,
                "overtake_difficulty": 0.45,
            },
            "rules": {
                "deployment_budget_remaining_kj": 1800.0,
                "minimum_reserve_soc_pct": 15.0,
                "full_deploy_allowed": True,
            },
            "decision_mode": "FAST",
            "telemetry": {
                "ego_distance_m": round(float(ego_distance[i]), 2),
                "target_distance_m": round(float(target_distance[i]), 2),
                "target_speed_kph": round(float(target_speed[i]), 2),
            },
        })

    payload = {
        "source": "FastF1",
        "synthetic_fields": ["soc_pct", "tyre_grip_estimate", "traffic", "rules", "overtake_difficulty"],
        "year": year,
        "event": str(event),
        "session": session_name,
        "driver": driver,
        "target_driver": target_driver,
        "lap": int(ego_lap["LapNumber"]),
        "target_lap": int(target_lap["LapNumber"]),
        "duration_s": round(duration, 3),
        "frames": frames,
    }
    if cache:
        key = f"{year}_{str(event).replace(' ', '_')}_{session_name}_{driver}_{target_driver}_{payload['lap']}.json"
        (CACHE_DIR / key).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["cache_file"] = key
    return payload
