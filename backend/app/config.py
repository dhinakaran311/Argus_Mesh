"""
backend/app/config.py
=====================
AbuseRing Sentinel — Application Settings

Reads all configuration from environment variables (loaded from .env).
Uses pydantic-settings for validation and type coercion.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: backend/app/config.py → ../../  = Argus_Mesh/
ROOT_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """All application settings sourced from the .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),  # allow model_version field
    )

    # -------------------------------------------------------------------------
    # Supabase (RESERVED — not currently used)
    # #17: The DataStore in db/datastore.py is pure in-memory CSV-backed.
    # These settings are placeholders for a future real-Supabase migration.
    # The `supabase` package in requirements.txt is similarly unused.
    # -------------------------------------------------------------------------
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_db_url: str = ""

    # -------------------------------------------------------------------------
    # Qdrant
    # -------------------------------------------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "argus-mesh-vectors"

    # -------------------------------------------------------------------------
    # Neo4j
    # -------------------------------------------------------------------------
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""

    # -------------------------------------------------------------------------
    # Groq
    # -------------------------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "compound-beta"          # verified available on this key
    groq_fast_model: str = "compound-beta-mini"  # verified available on this key

    # -------------------------------------------------------------------------
    # HuggingFace Embeddings
    # -------------------------------------------------------------------------
    huggingface_api_key: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # -------------------------------------------------------------------------
    # Backend
    # -------------------------------------------------------------------------
    environment: str = "development"
    log_level: str = "INFO"
    secret_key: str = "changeme"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    def get_cors_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # -------------------------------------------------------------------------
    # ML
    # -------------------------------------------------------------------------
    risk_threshold: float = 0.1
    fp_cost_inr: int = 100
    fn_cost_inr: int = 3000
    model_version: str = "v1"

    # -------------------------------------------------------------------------
    # Data Paths (resolved relative to project root)
    # -------------------------------------------------------------------------
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"

    @property
    def raw_dir(self) -> Path:
        return ROOT_DIR / self.data_raw_path

    @property
    def processed_dir(self) -> Path:
        return ROOT_DIR / self.data_processed_path

    @property
    def models_dir(self) -> Path:
        return ROOT_DIR / "ml" / "models"

    @property
    def evaluation_dir(self) -> Path:
        return ROOT_DIR / "ml" / "evaluation"

    @property
    def graph_queries_dir(self) -> Path:
        return ROOT_DIR / "graph" / "queries"


_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once per process)."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


def clear_settings_cache() -> None:
    """Invalidate the settings cache (called by lifespan on hot reload)."""
    global _settings_cache
    _settings_cache = None
