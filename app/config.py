"""Application settings, read from environment variables and `.env` (see `.env.example`)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local-only default matching docker-compose.yml.
    database_url: str = "postgresql+psycopg://support:support@localhost:5432/support"

    # Locally signed HS256 JWTs (Q5). Local-only default matching .env.example.
    jwt_secret: str = "change-me-local-dev-only-32-bytes-min"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Mock providers (W1.8-W1.10). The API keys never reach the model.
    mock_payments_url: str = "http://localhost:8001"
    mock_payments_api_key: str = "mock-payments-local-key"
    mock_shipping_url: str = "http://localhost:8002"
    mock_shipping_api_key: str = "mock-shipping-local-key"
    mock_ticketing_url: str = "http://localhost:8003"
    mock_ticketing_api_key: str = "mock-ticketing-local-key"
    # Per-request timeout for calls to the providers above.
    provider_timeout_seconds: float = 5.0

    # Local embedding model for policy retrieval (Q3). The dimensions must match the
    # policy_chunks.embedding vector(384) column.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # One store, one currency (Q6); amounts are in minor units (paise).
    store_currency: str = "INR"
    # A pending approval expires after this long and must be requested again (Q6).
    approval_expiry_minutes: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
