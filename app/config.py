from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ApexGuard Decision API"
    api_version: str = "0.1.0-phase1"
    verifier_latency_budget_ms: int = 300


settings = Settings()
