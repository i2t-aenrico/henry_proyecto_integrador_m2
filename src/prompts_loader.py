"""
prompts_loader.py — Expone las constantes de prompts/prompts.py al resto del
pipeline, manteniendo una única fuente de verdad para el prompt del sistema.
"""

import sys
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
sys.path.insert(0, str(_PROMPTS_DIR))

from prompts import (  # noqa: E402
    FUENTES_LEGIBLES,
    SYSTEM_ASISTENTE,
    SYSTEM_ASISTENTE_MULTI,
    TEMPLATE_USUARIO,
    formatear_chunks,
    SYSTEM_EVALUADOR,
    TEMPLATE_EVALUADOR,
)

__all__ = [
    "FUENTES_LEGIBLES",
    "SYSTEM_ASISTENTE",
    "SYSTEM_ASISTENTE_MULTI",
    "TEMPLATE_USUARIO",
    "formatear_chunks",
    "SYSTEM_EVALUADOR",
    "TEMPLATE_EVALUADOR",
]
