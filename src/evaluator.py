"""
evaluator.py — Agente evaluador de calidad (bonus, sprint 2).

Recibe user_question, system_answer y chunks_related, y devuelve un
puntaje de 0 a 10 con una justificación, usando un LLM como juez.
Reutiliza el mismo mecanismo de llamada (llm.py) que la generación
principal, pero con un prompt distinto (prompts.SYSTEM_EVALUADOR).
"""

from __future__ import annotations

import json

from llm import call_llm
from prompts_loader import SYSTEM_EVALUADOR, TEMPLATE_EVALUADOR
from schemas import EvaluacionRespuesta


def build_evaluator_prompt(question: str, answer: str, chunks: list[dict]) -> str:
    """Arma el prompt de usuario para el agente evaluador.

    Args:
        question: pregunta original del usuario.
        answer: respuesta ya generada por el asistente (a evaluar).
        chunks: chunks que se usaron como contexto para esa respuesta.

    Returns:
        Texto final del prompt de usuario para el evaluador, listo para
        pasarle a call_llm junto con SYSTEM_EVALUADOR.
    """
    chunks_texto = "\n\n".join(
        f"[{c['chunk_id']} | score={c['score']:.3f}]\n{c['text']}" for c in chunks
    )
    return TEMPLATE_EVALUADOR.format(
        pregunta=question, respuesta=answer, chunks_formateados=chunks_texto
    )


def extract_evaluation(raw_text: str) -> dict:
    """Extrae el JSON {puntaje, justificacion} de la respuesta cruda del LLM.

    Igual que extract_answer en llm.py, limpia los fences de markdown que
    algunos modelos agregan alrededor del JSON antes de parsearlo.

    Args:
        raw_text: texto crudo devuelto por el LLM evaluador.

    Returns:
        Diccionario con las claves "puntaje" y "justificacion", listo para
        validar con el esquema EvaluacionRespuesta.
    """
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def evaluate_answer(question: str, answer: str, chunks: list[dict]) -> tuple[EvaluacionRespuesta, dict]:
    """Evalúa una respuesta ya generada y devuelve el veredicto + métricas de uso.

    Args:
        question: pregunta original del usuario.
        answer: respuesta del asistente a evaluar.
        chunks: chunks usados como contexto para generar esa respuesta.

    Returns:
        Tupla (evaluacion, llm_result): evaluacion es un EvaluacionRespuesta
        validado con Pydantic (puntaje 0-10 y justificación); llm_result es
        el diccionario de métricas de uso de la llamada al LLM evaluador
        (tokens, latencia, proveedor, modelo).
    """
    user_prompt = build_evaluator_prompt(question, answer, chunks)
    llm_result = call_llm(SYSTEM_EVALUADOR, user_prompt)
    data = extract_evaluation(llm_result["text"])
    evaluacion = EvaluacionRespuesta(**data)
    return evaluacion, llm_result
