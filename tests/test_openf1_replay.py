from datetime import datetime, timedelta, timezone

import app.replay.openf1_replay as replay


def _dt(i):
    return (datetime(2025, 5, 25, 13, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i)).isoformat()


def test_openf1_build_replay_maps_observed_fields(monkeypatch):
    session = {
        "session_key": 999,
        "meeting_key": 888,
        "circuit_short_name": "Monaco",
        "date_start": _dt(0),
    }
    drivers = [
        {"driver_number": 1, "name_acronym": "VER"},
        {"driver_number": 4, "name_acronym": "NOR"},
    ]
    ego_car = [
        {"date": _dt(0), "speed": 200, "throttle": 90, "brake": 0, "drs": 0},
        {"date": _dt(1), "speed": 230, "throttle": 100, "brake": 0, "drs": 10},
        {"date": _dt(2), "speed": 180, "throttle": 50, "brake": 100, "drs": 0},
    ]
    target_car = [
        {"date": _dt(0), "speed": 195, "throttle": 90, "brake": 0, "drs": 0},
        {"date": _dt(1), "speed": 220, "throttle": 100, "brake": 0, "drs": 10},
        {"date": _dt(2), "speed": 175, "throttle": 45, "brake": 100, "drs": 0},
    ]

    def fake_fetch(endpoint, params, **kwargs):
        if endpoint == "sessions":
            return [session]
        if endpoint == "drivers":
            return drivers
        if endpoint == "car_data" and params["driver_number"] == 1:
            return ego_car
        if endpoint == "car_data" and params["driver_number"] == 4:
            return target_car
        if endpoint == "intervals":
            return [{"date": _dt(0), "interval": 0.8}, {"date": _dt(1), "interval": 0.7}, {"date": _dt(2), "interval": 1.1}]
        if endpoint == "position" and params["driver_number"] == 1:
            return [{"date": _dt(0), "position": 3}, {"date": _dt(2), "position": 3}]
        if endpoint == "position" and params["driver_number"] == 4:
            return [{"date": _dt(0), "position": 2}, {"date": _dt(2), "position": 2}]
        if endpoint == "laps":
            return [{"date_start": _dt(0), "lap_number": 1, "lap_duration": 90}]
        if endpoint == "stints":
            return [{"lap_start": 1, "lap_end": 5, "tyre_age_at_start": 2}]
        if endpoint == "overtakes":
            return []
        if endpoint == "race_control":
            return []
        raise AssertionError(endpoint)

    monkeypatch.setattr(replay, "_fetch_json", fake_fetch)
    payload = replay.build_replay(2025, "Monaco", "Race", "VER", "NOR", samples=20, cache=False)

    assert payload["source"] == "OpenF1"
    assert payload["api_session_key"] == 999
    assert len(payload["frames"]) == 3
    assert "ego.speed_kph" in payload["observed_fields"]
    assert "ego.soc_pct" in payload["synthetic_fields"]
    # Target is ahead, so interval is allowed to populate the target gap.
    assert payload["frames"][0]["target"]["gap_s"] == 0.8
    assert payload["frames"][0]["telemetry"]["interval_source"] == "openf1_interval_observed"
    assert payload["frames"][1]["track"]["drs_available"] is True
