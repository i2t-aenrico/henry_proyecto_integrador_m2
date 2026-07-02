"""
settings.py — Carga de configuración desde variables de entorno (.env).

Centraliza todas las variables para que el resto de los módulos no lean
os.environ directamente. Facilita testear con configuraciones distintas
y mantiene el patrón usado en el proyecto del Módulo 1 (settings.py con
fábrica de cliente perezosa y cacheada).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    local_embedding_model: str
    llm_provider: str
    anthropic_api_key: str | None
    anthropic_model: str
    openai_llm_model: str
    top_k: int
    index_dir: str


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración cacheada (una sola lectura del entorno)."""
    return Settings(
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        top_k=int(os.getenv("TOP_K", "5")),
        index_dir=os.getenv("INDEX_DIR", "outputs/index"),
    )
