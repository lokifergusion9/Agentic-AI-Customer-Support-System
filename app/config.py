"""Application settings, read from environment variables and `.env` (see `.env.example`)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local-only default matching docker-compose.yml.
    database_url: str = "postgresql+psycopg://support:support@localhost:5432/support"


@lru_cache
def get_settings() -> Settings:
    return Settings()
