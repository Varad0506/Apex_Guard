"""OpenF1 -> ApexGuard replay adapter.

OpenF1 supplies public historical F1 timing/telemetry from 2023 onward. This
adapter turns observed OpenF1 data into the same DecisionRequest-shaped replay
stream used by the FastF1 adapter and the ApexGuard backtester.

Important: OpenF1 does not expose battery SOC/ERS reserve or all hidden traffic
state. Those fields remain explicitly synthetic/latent in the replay payload.
"""
from __future__ import annotations

import json
import math
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "replays" / "openf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.openf1.org/v1"
# OpenF1 free historical tier is limited to 30 requests/minute. Keep a
# conservative client-side interval so the six-track batch remains reliable.
REQUEST_INTERVAL_S = float(os.getenv("OPENF1_REQUEST_INTERVAL_S", "2.05"))
_LAST_REQUEST_AT = 0.0


def _require_network():
    return urlopen


def _cache_key(endpoint: str, params: dict[str, Any]) -> Path:
    safe = "_".join(f"{k}-{str(v).replace('/', '-') }" for k, v in sorted(params.items()))
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{endpoint}_{safe}")
    return CACHE_DIR / f"{safe[:220]}.json"


def _fetch_json(endpoint: str, params: dict[str, Any], *, cache: bool = True, timeout_s: int = 45, allow_404: bool = False) -> list[dict[str, Any]]:
    path = _cache_key(endpoint, params)
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    global _LAST_REQUEST_AT
    query = urlencode(params)
    url = f"{API_BASE}/{endpoint}?{query}"
    req = Request(url, headers={"User-Agent": "ApexGuard/1.0 historical-backtest"})
    wait = REQUEST_INTERVAL_S - (time.monotonic() - _LAST_REQUEST_AT)
    if wait > 0:
        time.sleep(wait)
    try:
        with _require_network()(req, timeout=timeout_s) as response:
            _LAST_REQUEST_AT = time.monotonic()
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        _LAST_REQUEST_AT = time.monotonic()
        if exc.code == 429:
            time.sleep(max(REQUEST_INTERVAL_S, 2.5))
            with _require_network()(req, timeout=timeout_s) as response:
                _LAST_REQUEST_AT = time.monotonic()
                data = json.loads(response.read().decode("utf-8"))
        elif exc.code == 404 and allow_404:
            # Some OpenF1 historical endpoints can return 404 when the
            # requested optional dataset/filter has no available rows.
            # Treat optional enrichment as empty rather than aborting the
            # entire replay build. Core telemetry remains required.
            data = []
        else:
            raise RuntimeError(f"OpenF1 request failed ({exc.code}) for {endpoint}: {exc.reason}") from exc
    except URLError as exc:
        _LAST_REQUEST_AT = time.monotonic()
        raise RuntimeError(f"OpenF1 network error for {endpoint}: {exc.reason}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected OpenF1 response for {endpoint}")
    if cache:
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ts(value: str | None) -> float | None:
    dt = _parse_dt(value)
    return dt.timestamp() if dt else None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _driver_number(session_key: int, acronym: str, drivers: list[dict[str, Any]]) -> int:
    wanted = acronym.upper()
    for row in drivers:
        if str(row.get("name_acronym", "")).upper() == wanted:
            return int(row["driver_number"])
    raise ValueError(f"Driver {acronym} is not present in OpenF1 session {session_key}")


def _resolve_session(year: int, event: str, session_name: str, cache: bool) -> dict[str, Any]:
    params = {"year": year, "session_name": session_name}
    # Fetch the year/session index first and resolve the circuit locally.
    # This is more robust than sending circuit_short_name as a filter because
    # OpenF1 deployments may reject that filter on /sessions with HTTP 404.
    all_sessions = _fetch_json("sessions", params, cache=cache)
    needle = event.strip().lower().replace(" ", "").replace("-", "")
    matches = [
        row for row in all_sessions
        if needle in str(row.get("circuit_short_name", "")).lower().replace(" ", "").replace("-", "")
        or needle in str(row.get("location", "")).lower().replace(" ", "").replace("-", "")
        or needle in str(row.get("country_name", "")).lower().replace(" ", "").replace("-", "")
        or needle in str(row.get("meeting_name", "")).lower().replace(" ", "").replace("-", "")
    ]
    rows = matches
    if not rows:
        raise ValueError(f"OpenF1 session not found: {year} {event} {session_name}")
    rows.sort(key=lambda r: str(r.get("date_start", "")), reverse=True)
    return rows[0]


def _nearest(rows: list[dict[str, Any]], timestamp_s: float, value_key: str | None = None):
    best = None
    best_delta = math.inf
    for row in rows:
        t = _ts(row.get("date"))
        if t is None:
            continue
        delta = abs(t - timestamp_s)
        if delta < best_delta:
            best, best_delta = row, delta
    return best


def _sample(rows: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: _ts(r.get("date")) or 0.0)
    n = max(20, min(int(samples), 500))
    if len(rows) <= n:
        return rows
    indices = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [rows[i] for i in indices]


def _position_at(rows: list[dict[str, Any]], timestamp_s: float) -> int | None:
    row = _nearest(rows, timestamp_s)
    if not row:
        return None
    try:
        return int(row["position"])
    except (KeyError, TypeError, ValueError):
        return None


def _lap_at(laps: list[dict[str, Any]], timestamp_s: float) -> int:
    best_lap = 1
    for lap in laps:
        start = _ts(lap.get("date_start"))
        end = _ts(lap.get("date_start"))
        duration = _num(lap.get("lap_duration"), 0.0)
        if start is not None:
            end = start + duration if duration > 0 else start
            if timestamp_s >= start:
                try:
                    best_lap = max(best_lap, int(lap.get("lap_number", best_lap)))
                except (TypeError, ValueError):
                    pass
            if start <= timestamp_s <= end:
                try:
                    return int(lap["lap_number"])
                except (KeyError, TypeError, ValueError):
                    pass
    return best_lap


def _stint_age(laps: list[dict[str, Any]], stints: list[dict[str, Any]], timestamp_s: float) -> int:
    lap = _lap_at(laps, timestamp_s)
    active = [s for s in stints if int(s.get("lap_start", 10**9)) <= lap <= int(s.get("lap_end", -1))]
    if not active:
        return max(0, lap - 1)
    stint = active[-1]
    return max(0, lap - int(stint.get("lap_start", lap)) + int(stint.get("tyre_age_at_start", 0)))


def _track_segment(speed_kph: float, throttle: float, brake: float, drs: int) -> str:
    if drs in {8, 9, 10, 12, 14}:
        return "DRS_STRAIGHT"
    if brake >= 50 or speed_kph < 150:
        return "CORNER"
    if throttle >= 90 and speed_kph >= 220:
        return "STRAIGHT"
    return "TRANSITION"


def build_replay(
    year: int,
    event: str,
    session_name: str,
    driver: str,
    target_driver: str | None = None,
    samples: int = 180,
    cache: bool = True,
) -> dict[str, Any]:
    """Build an OpenF1 historical replay compatible with ApexGuard backtesting."""
    session = _resolve_session(year, event, session_name, cache)
    session_key = int(session["session_key"])
    meeting_key = int(session["meeting_key"])

    drivers = _fetch_json("drivers", {"session_key": session_key}, cache=cache)
    ego_number = _driver_number(session_key, driver, drivers)
    target_driver = target_driver or next(
        str(row["name_acronym"]) for row in drivers if int(row["driver_number"]) != ego_number
    )
    target_number = _driver_number(session_key, target_driver, drivers)

    ego_car = _fetch_json("car_data", {"session_key": session_key, "driver_number": ego_number}, cache=cache)
    target_car = _fetch_json("car_data", {"session_key": session_key, "driver_number": target_number}, cache=cache)
    if not ego_car or not target_car:
        raise ValueError("OpenF1 returned no car telemetry for the requested driver pair")

    ego_intervals = _fetch_json("intervals", {"session_key": session_key, "driver_number": ego_number}, cache=cache)
    ego_positions = _fetch_json("position", {"session_key": session_key, "driver_number": ego_number}, cache=cache)
    target_positions = _fetch_json("position", {"session_key": session_key, "driver_number": target_number}, cache=cache)
    ego_laps = _fetch_json("laps", {"session_key": session_key, "driver_number": ego_number}, cache=cache)
    target_laps = _fetch_json("laps", {"session_key": session_key, "driver_number": target_number}, cache=cache)
    target_stints = _fetch_json("stints", {"session_key": session_key, "driver_number": target_number}, cache=cache)
    overtakes = _fetch_json(
        "overtakes",
        {"session_key": session_key, "overtaking_driver_number": ego_number, "overtaken_driver_number": target_number},
        cache=cache,
        allow_404=True,
    )
    race_control = _fetch_json("race_control", {"session_key": session_key}, cache=cache)

    sampled = _sample(ego_car, samples)
    ego_start = _ts(sampled[0].get("date")) or 0.0
    max_lap = max([int(r.get("lap_number", 0)) for r in ego_laps] + [int(r.get("lap_number", 0)) for r in target_laps] + [1])

    # Only these fields are genuinely absent from public OpenF1 telemetry.
    synthetic_fields = [
        "ego.soc_pct", "ego.tyre_grip_estimate", "ego.current_mode",
        "traffic.rear_gap_s", "traffic.cars_within_3s", "traffic.post_pass_traffic_gap_s",
        "track.straight_remaining_m", "track.braking_zone_m", "track.overtake_difficulty",
        "rules.deployment_budget_remaining_kj", "rules.minimum_reserve_soc_pct", "rules.full_deploy_allowed",
    ]
    frames: list[dict[str, Any]] = []
    target_by_time = sorted(target_car, key=lambda r: _ts(r.get("date")) or 0.0)

    for i, row in enumerate(sampled):
        now = _ts(row.get("date")) or ego_start
        t_s = max(0.0, now - ego_start)
        target_row = _nearest(target_by_time, now) or {}
        interval_row = _nearest(ego_intervals, now) or {}
        ego_position = _position_at(ego_positions, now)
        target_position = _position_at(target_positions, now)

        speed = max(0.0, _num(row.get("speed")))
        target_speed = max(0.0, _num(target_row.get("speed"), speed))
        throttle = max(0.0, min(100.0, _num(row.get("throttle"), 100.0)))
        brake = 100.0 if _num(row.get("brake")) > 0 else 0.0
        drs_code = int(_num(row.get("drs"), 0))
        drs = drs_code in {8, 9, 10, 12, 14}

        # OpenF1's interval is the observed gap to the car immediately ahead.
        # We use it as the target gap only when the selected target is ahead;
        # otherwise a conservative synthetic fallback keeps the schema valid.
        observed_interval = interval_row.get("interval")
        target_ahead = target_position is not None and ego_position is not None and target_position < ego_position
        if target_ahead and observed_interval is not None:
            try:
                gap_s = max(0.05, float(observed_interval))
                gap_source = "openf1_interval_observed"
            except (TypeError, ValueError):
                gap_s = 1.25
                gap_source = "synthetic_fallback"
        else:
            gap_s = 1.25
            gap_source = "synthetic_fallback"

        lap = _lap_at(ego_laps, now)
        segment = _track_segment(speed, throttle, brake, drs_code)
        # A simple telemetry-derived opportunity distance proxy. It is not a
        # circuit-map measurement because OpenF1 location has no track-distance field.
        straight_remaining = max(0.0, min(900.0, speed / 3.6 * 2.2)) if segment != "CORNER" else 80.0
        braking_zone = 90.0 if brake > 0 else 140.0

        # Race-control is observed, but the rule engine's detailed legal state
        # remains synthetic because OpenF1 does not expose team-specific rules.
        recent_control = _nearest(race_control, now)
        safety_or_yellow = bool(recent_control and str(recent_control.get("flag", "")).upper() in {"YELLOW", "DOUBLE YELLOW", "RED"})
        aggressive_allowed = not safety_or_yellow

        frames.append({
            "t_s": round(t_s, 3),
            "request_id": f"openf1-{year}-{event}-{session_name}-{driver}-{i}",
            "battle_id": f"openf1-{year}-{event}-{session_name}-{driver}-{target_driver}",
            "timestamp_ms": int(t_s * 1000),
            "telemetry_age_ms": 270,
            "ego": {
                "soc_pct": 65.0,
                "speed_kph": round(speed, 2),
                "throttle_pct": round(throttle, 2),
                "brake_pct": round(brake, 2),
                "tyre_grip_estimate": 0.82,
                "laps_remaining": max(0, max_lap - lap),
                "current_mode": "HOLD",
            },
            "target": {
                "gap_s": round(gap_s, 3),
                "relative_speed_kph": round(speed - target_speed, 2),
                "stint_age_laps": _stint_age(target_laps, target_stints, now),
                "recent_sector_delta_s": 0.0,
            },
            "traffic": {
                "rear_gap_s": 2.0,
                "cars_within_3s": 1,
                "post_pass_traffic_gap_s": 1.5,
            },
            "track": {
                "track_id": str(session.get("circuit_short_name") or event),
                "segment_type": segment,
                "drs_available": drs,
                "straight_remaining_m": round(straight_remaining, 1),
                "braking_zone_m": braking_zone,
                "overtake_difficulty": 0.45,
                "detection_point_active": target_ahead and gap_s <= 1.0,
                "overtake_mode_available": target_ahead and gap_s <= 1.0,
                "ers_key_acceleration_zone": segment in {"STRAIGHT", "DRS_STRAIGHT"},
            },
            "rules": {
                "deployment_budget_remaining_kj": 1800.0,
                "minimum_reserve_soc_pct": 15.0,
                "full_deploy_allowed": aggressive_allowed,
            },
            "decision_mode": "FAST",
            "telemetry": {
                "source_timestamp": row.get("date"),
                "ego_speed_kph_observed": round(speed, 2),
                "target_speed_kph_observed": round(target_speed, 2),
                "ego_position_observed": ego_position,
                "target_position_observed": target_position,
                "interval_source": gap_source,
                "drs_code_observed": drs_code,
                "overtake_observed": bool(_nearest(overtakes, now)),
                "race_control_flag": recent_control.get("flag") if recent_control else None,
            },
        })

    payload = {
        "source": "OpenF1",
        "provider": "openf1",
        "api_session_key": session_key,
        "meeting_key": meeting_key,
        "synthetic_fields": synthetic_fields,
        "observed_fields": [
            "ego.speed_kph", "ego.throttle_pct", "ego.brake_pct", "target.relative_speed_kph",
            "target.gap_s_when_target_ahead", "track.drs_available", "telemetry.ego_position_observed",
            "telemetry.target_position_observed", "telemetry.race_control_flag", "telemetry.overtake_observed",
        ],
        "year": year,
        "event": event,
        "session": session_name,
        "driver": driver.upper(),
        "target_driver": target_driver.upper(),
        "start_lap": _lap_at(ego_laps, ego_start),
        "end_lap": _lap_at(ego_laps, max((_ts(r.get("date")) or ego_start for r in sampled), default=ego_start)),
        "target_start_lap": _lap_at(target_laps, ego_start),
        "target_end_lap": _lap_at(target_laps, max((_ts(r.get("date")) or ego_start for r in sampled), default=ego_start)),
        "duration_s": round(max((f["t_s"] for f in frames), default=0.0), 3),
        "frames": frames,
        "limitations": [
            "OpenF1 is an unofficial community API; this adapter uses it only for historical replay/backtesting.",
            "Public OpenF1 telemetry does not provide battery SOC/ERS reserve, so energy state remains synthetic.",
            "OpenF1 location lacks lateral track placement and track-distance; straight/braking proxies are derived.",
            "The selected target is not guaranteed to be the immediately-ahead car at every timestamp; interval is only treated as target gap when live position data confirms the target is ahead.",
            "Observed overtakes are reported as historical events; they are not treated as counterfactual ground truth for ApexGuard actions.",
        ],
    }

    if cache:
        key = f"{year}_{event.replace(' ', '_')}_{session_name}_{driver}_{target_driver}_openf1.json"
        (CACHE_DIR.parent / key).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["cache_file"] = key
    return payload
