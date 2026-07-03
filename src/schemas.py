"""
schemas.py — Contratos Pydantic para validar la salida del pipeline de consulta.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRelated(BaseModel):
    """Representa un chunk recuperado tal como aparece en la salida final.

    Attributes:
        chunk_id: identificador del chunk dentro de su dataset (ej: "chunk_005").
        score: similitud coseno con la pregunta (entre -1 y 1).
        text: texto completo del chunk.
        source: nombre del archivo de origen (ej: "faq_macro.txt"), para
            trazabilidad cuando se combinan varios datasets con --dataset all.
    """
    chunk_id: str
    score: float
    text: str
    source: str | None = None


class RespuestaRAG(BaseModel):
    """Contrato de salida del pipeline de consulta (query.py).

    Es el JSON que se le muestra al usuario final: la pregunta original,
    la respuesta generada por el LLM, y los chunks que se usaron como
    contexto para esa respuesta.
    """
    user_question: str = Field(min_length=3)
    system_answer: str = Field(min_length=10)
    chunks_related: list[ChunkRelated]


class EvaluacionRespuesta(BaseModel):
    """Veredicto del agente evaluador (bonus, sprint 2).

    Attributes:
        puntaje: nota de 0 a 10 asignada por el LLM evaluador.
        justificacion: explicación breve del puntaje asignado.
    """
    puntaje: float = Field(ge=0, le=10)
    justificacion: str = Field(min_length=10)
