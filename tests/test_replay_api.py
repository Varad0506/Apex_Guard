from fastapi.testclient import TestClient
from app.main import app


def _all_route_paths(routes) -> set[str]:
    """Recursively collect route paths.

    Newer FastAPI/Starlette versions wrap routers added via
    `app.include_router()` in an internal `_IncludedRouter` object that
    doesn't itself expose `.path` or `.routes` -- the actual APIRouter (with
    its real `.routes`) lives on its `.original_router` attribute instead.
    Only leaf route objects (APIRoute/Route/WebSocketRoute) have `.path`. We
    recurse through `.routes` and `.original_router.routes`, and collect
    from anything that has `.path`, so this keeps working whether
    app.routes is flat (older FastAPI) or wrapped (newer FastAPI).
    """
    paths: set[str] = set()
    for r in routes:
        path = getattr(r, "path", None)
        if path is not None:
            paths.add(path)
        nested = getattr(r, "routes", None)
        if nested:
            paths |= _all_route_paths(nested)
        original_router = getattr(r, "original_router", None)
        if original_router is not None:
            original_routes = getattr(original_router, "routes", None)
            if original_routes:
                paths |= _all_route_paths(original_routes)
    return paths


def test_replay_routes_are_registered():
    paths = _all_route_paths(app.routes)
    assert "/v1/replays" in paths
    assert "/v1/replays/fastf1" in paths
    assert "/v1/replays/{replay_id}" in paths
