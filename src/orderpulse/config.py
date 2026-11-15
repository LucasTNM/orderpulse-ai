"""Typed configuration management for OrderPulse AI.

Uses strict Pydantic validation and supports loading values from environment
variables or a local ``.env`` file without hard external dependencies.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


def _load_dotenv_into_env(dotenv_path: Path) -> None:
    """Load key=value pairs from a .env file if present without overriding existing env vars."""
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class Settings(BaseModel):
    """Centralized application settings validated via Pydantic."""

    app_name: str = Field(default="OrderPulse AI")
    app_env: str = Field(default="development")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")

    # LLM routing & reasoning
    llm_provider: str = Field(default="deterministic")
    llm_model_router: str = Field(default="llama-3.3-70b-versatile")
    llm_model_agent: str = Field(default="gpt-4o-mini")
    openai_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")

    # Semantic Cache (Redis Vector)
    redis_url: str = Field(default="redis://localhost:6379/0")
    semantic_cache_enabled: bool = Field(default=True)
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    semantic_cache_ttl_seconds: int = Field(default=86400, ge=60)

    # Hybrid RAG + Reranking
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="orderpulse_policies")
    rag_top_k_initial: int = Field(default=15, ge=3, le=50)
    rag_top_k_reranked: int = Field(default=3, ge=1, le=10)

    # Operational Database (Read-Only SQL via MCP)
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://orderpulse:orderpulse@localhost:5432/orderpulse_ops"
    )
    sql_readonly_enforced: bool = Field(default=True)

    # Financial Governance & Abuse Guardrails
    hitl_refund_threshold_brl: float = Field(default=150.00, gt=0.0)
    max_refunds_30d_abuse_threshold: int = Field(default=4, ge=1)

    # Observability (Langfuse)
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="http://localhost:3000")

    @field_validator("semantic_cache_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if value < 0.5:
            raise ValueError("semantic_cache_threshold must be >= 0.5 to prevent false cache hits.")
        return value

    @classmethod
    def from_env(cls, dotenv_file: str | Path = ".env") -> "Settings":
        """Construct Settings from process environment variables and optional .env file."""
        _load_dotenv_into_env(Path(dotenv_file))

        def _bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            app_name=os.getenv("APP_NAME", "OrderPulse AI"),
            app_env=os.getenv("APP_ENV", "development"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            llm_provider=os.getenv("LLM_PROVIDER", "deterministic"),
            llm_model_router=os.getenv("LLM_MODEL_ROUTER", "llama-3.3-70b-versatile"),
            llm_model_agent=os.getenv("LLM_MODEL_AGENT", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            semantic_cache_enabled=_bool("SEMANTIC_CACHE_ENABLED", True),
            semantic_cache_threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95")),
            semantic_cache_ttl_seconds=int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400")),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "orderpulse_policies"),
            rag_top_k_initial=int(os.getenv("RAG_TOP_K_INITIAL", "15")),
            rag_top_k_reranked=int(os.getenv("RAG_TOP_K_RERANKED", "3")),
            postgres_dsn=os.getenv(
                "POSTGRES_DSN",
                "postgresql+asyncpg://orderpulse:orderpulse@localhost:5432/orderpulse_ops",
            ),
            sql_readonly_enforced=_bool("SQL_READONLY_ENFORCED", True),
            hitl_refund_threshold_brl=float(os.getenv("HITL_REFUND_THRESHOLD_BRL", "150.00")),
            max_refunds_30d_abuse_threshold=int(
                os.getenv("MAX_REFUNDS_30D_ABUSE_THRESHOLD", "4")
            ),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    return Settings.from_env()
