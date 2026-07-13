from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.studio.nebius.com/v1"
    nebius_llm_model: str = "Qwen/Qwen3-1.7B"
    nebius_fine_tuned_model: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def active_model(self) -> str:
        """Prefer the fine-tuned model once it's deployed as a Nebius Endpoint."""
        return self.nebius_fine_tuned_model or self.nebius_llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
