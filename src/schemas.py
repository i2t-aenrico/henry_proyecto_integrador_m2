"""
schemas.py — Contratos Pydantic para validar la salida del pipeline de consulta.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRelated(BaseModel):
    chunk_id: str
    score: float
    text: str


class RespuestaRAG(BaseModel):
    user_question: str = Field(min_length=3)
    system_answer: str = Field(min_length=10)
    chunks_related: list[ChunkRelated]


class EvaluacionRespuesta(BaseModel):
    """Reservado para el agente evaluador (bonus, sprint 2)."""
    puntaje: float = Field(ge=0, le=10)
    justificacion: str = Field(min_length=10)
