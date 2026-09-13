import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from app.engine import audit

router = APIRouter(tags=["replay"])
SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "data" / "scenarios"
REPLAYS_DIR = Path(__file__).resolve().parents[1] / "data" / "replays"


@router.get("/v1/scenarios")
def list_scenarios():
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        scenarios.append({"id": path.stem, "request_id": data.get("request_id"), "file": path.name})
    return {"scenarios": scenarios}


@router.get("/v1/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    path = SCENARIOS_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    with path.open() as f:
        return json.load(f)


@router.get("/v1/audits/{audit_id}")
def get_audit(audit_id: str):
    record = audit.get_audit_by_id(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found")
    return record


@router.get("/v1/replays")
def list_replays():
    return {"replays": [
        {"id": p.stem, "file": p.name}
        for p in sorted(REPLAYS_DIR.glob("*.json"))
    ]}


@router.get("/v1/replays/fastf1")
def create_fastf1_replay(
    year: int = Query(..., ge=2018, le=2100),
    event: str = Query(...),
    session: str = Query("R"),
    driver: str = Query(...),
    target_driver: str | None = Query(None),
    lap: int | None = Query(None, ge=1),
    target_lap: int | None = Query(None, ge=1),
    samples: int = Query(180, ge=20, le=500),
):
    try:
        from app.replay.fastf1_replay import build_replay
        return build_replay(year, event, session, driver, target_driver, lap, target_lap, samples)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/v1/replays/openf1")
def create_openf1_replay(
    year: int = Query(..., ge=2023, le=2100),
    event: str = Query(...),
    session: str = Query("Race"),
    driver: str = Query(...),
    target_driver: str | None = Query(None),
    samples: int = Query(180, ge=20, le=500),
):
    try:
        from app.replay.openf1_replay import build_replay
        return build_replay(year, event, session, driver, target_driver, samples)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/v1/replays/{replay_id}")
def get_replay(replay_id: str):
    path = REPLAYS_DIR / f"{replay_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Replay '{replay_id}' not found")
    return json.loads(path.read_text(encoding="utf-8"))

@router.get("/v1/backtests")
def list_backtests():
    backtest_dir = REPLAYS_DIR.parent / "evaluations" / "backtests"
    return {"backtests": [
        {"id": p.stem, "file": p.name}
        for p in sorted(backtest_dir.glob("*_backtest.json"))
    ]}


@router.get("/v1/backtests/{backtest_id}")
def get_backtest(backtest_id: str):
    backtest_dir = REPLAYS_DIR.parent / "evaluations" / "backtests"
    path = backtest_dir / f"{backtest_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Backtest '{backtest_id}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/v1/backtests/run")
def run_backtest(replay_id: str = Query(...), max_frames: int | None = Query(None, ge=1, le=500)):
    replay_path = REPLAYS_DIR / f"{replay_id}.json"
    if not replay_path.exists():
        raise HTTPException(status_code=404, detail=f"Replay '{replay_id}' not found")
    try:
        from app.evaluation.backtest import backtest_file
        result = backtest_file(replay_path, max_frames=max_frames)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
