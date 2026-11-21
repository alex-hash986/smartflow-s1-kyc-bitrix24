from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Nonnegative = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class ModelRate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_usd_per_million: Nonnegative
    output_usd_per_million: Nonnegative


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        str_strip_whitespace=True,
        env_ignore_empty=True,
        frozen=True,
    )

    app_host: Literal["127.0.0.1"] = "127.0.0.1"
    app_port: int = Field(default=8021, ge=8021, le=65535)
    worker_token: SecretStr | None = None
    base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434/v1")
    api_key: SecretStr | None = None
    fast_llm_model: str | None = None
    smart_llm_model: str | None = None
    embedding_model: str | None = None
    llm_model_rates: dict[str, ModelRate] = Field(default_factory=dict)
    usd_to_rub_rate: Decimal = Field(default=Decimal(100), gt=0, allow_inf_nan=False)
    max_cycle_cost_rub: Nonnegative = Decimal(1)
    bitrix24_webhook_url: SecretStr | None = None
    bitrix24_rate_limit_rps: float = Field(default=2, gt=0, le=2)
    sqlite_path: Path = Path("data/core.db")
    profiles_path: Path = Path("profiles")
