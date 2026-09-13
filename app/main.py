from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import routes_decision, routes_health, routes_replay, routes_simulate, routes_quantum
from app.models import overtake_probability, policy_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force lazy-loaded models to load at process startup instead of on the
    # first live request -- without this, the first /v1/decide call pays a
    # cold-start penalty (sklearn/pandas/torch import chains) that has no
    # business happening live in front of a judge.
    overtake_probability.model_is_available()
    policy_adapter.policy_is_available()
    yield


app = FastAPI(title=settings.app_name, version=settings.api_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_decision.router)
app.include_router(routes_replay.router)
app.include_router(routes_simulate.router)
app.include_router(routes_quantum.router)


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.api_version,
        "docs": "/docs",
    }
