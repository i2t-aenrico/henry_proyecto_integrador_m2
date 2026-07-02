"""
metrics_writer.py — Registro de métricas y logs por consulta.

Este módulo deja preparada la infraestructura de logging que va a
necesitar el agente evaluador del sprint 2 (bonus): cada consulta se
registra con su pregunta, respuesta, chunks recuperados, si el sistema
consideró que "no pudo responder", tokens, latencia y costo estimado.

Archivos generados:
- metrics/metrics.csv       -> una fila por consulta, para análisis agregado.
- logs/queries.jsonl        -> un JSON por línea con el detalle completo,
                               incluyendo las respuestas NO obtenidas
                               (para poder auditarlas y, en el sprint 2,
                               pasarlas al agente evaluador).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

METRICS_CSV = "metrics/metrics.csv"
QUERIES_LOG = "logs/queries.jsonl"

CSV_HEADERS = [
    "timestamp", "user_question", "answered", "top_chunk_id", "top_score",
    "input_tokens", "output_tokens", "latency_ms", "cost_usd", "provider", "model",
]

# Precios de referencia por proveedor (USD por millón de tokens: entrada, salida).
PRICING = {
    ("anthropic", "claude-sonnet-4-6"): (3.00, 15.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcula el costo aproximado de una llamada según tokens reales de uso."""
    price_in, price_out = PRICING.get((provider, model), (0.0, 0.0))
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


def was_answered(answer_text: str) -> bool:
    """Determina si la respuesta es una respuesta real o una NO-respuesta.

    Se considera "no respondida" cuando el propio LLM indica que no tiene
    información suficiente. Este criterio simple queda documentado para
    que el sprint 2 lo reemplace por el agente evaluador.
    """
    marcadores_no_respuesta = [
        "no tengo información suficiente",
        "no cuento con información",
        "no dispongo de información",
    ]
    texto = answer_text.lower()
    return not any(marcador in texto for marcador in marcadores_no_respuesta)


def registrar_metrica(
    user_question: str,
    system_answer: str,
    chunks_related: list[dict],
    llm_result: dict,
) -> None:
    """Escribe la métrica en CSV y el detalle completo en el log JSONL."""
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(QUERIES_LOG), exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    answered = was_answered(system_answer)
    top_chunk = chunks_related[0] if chunks_related else {}
    cost = estimate_cost(
        llm_result.get("provider", ""),
        llm_result.get("model", ""),
        llm_result.get("input_tokens", 0),
        llm_result.get("output_tokens", 0),
    )

    _append_csv_row(timestamp, user_question, answered, top_chunk, llm_result, cost)
    _append_jsonl_row(timestamp, user_question, system_answer, chunks_related, answered, llm_result, cost)


def _append_csv_row(timestamp, user_question, answered, top_chunk, llm_result, cost) -> None:
    file_exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "user_question": user_question,
            "answered": answered,
            "top_chunk_id": top_chunk.get("chunk_id", ""),
            "top_score": top_chunk.get("score", ""),
            "input_tokens": llm_result.get("input_tokens", 0),
            "output_tokens": llm_result.get("output_tokens", 0),
            "latency_ms": llm_result.get("latency_ms", 0),
            "cost_usd": cost,
            "provider": llm_result.get("provider", ""),
            "model": llm_result.get("model", ""),
        })


def _append_jsonl_row(timestamp, user_question, system_answer, chunks_related, answered, llm_result, cost) -> None:
    row = {
        "timestamp": timestamp,
        "user_question": user_question,
        "system_answer": system_answer,
        "answered": answered,
        "chunks_related": chunks_related,
        "llm": llm_result,
        "cost_usd": cost,
    }
    with open(QUERIES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evaluaciones del agente evaluador (bonus, sprint 2)
# ---------------------------------------------------------------------------

EVALUATIONS_CSV = "metrics/evaluations.csv"
EVALUATIONS_LOG = "logs/evaluations.jsonl"

EVAL_CSV_HEADERS = [
    "timestamp", "user_question", "puntaje", "justificacion",
    "input_tokens", "output_tokens", "latency_ms", "cost_usd",
]


def registrar_evaluacion(user_question: str, evaluacion, llm_result: dict) -> None:
    """Registra el veredicto del agente evaluador en CSV y en el log JSONL."""
    os.makedirs(os.path.dirname(EVALUATIONS_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(EVALUATIONS_LOG), exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    cost = estimate_cost(
        llm_result.get("provider", ""),
        llm_result.get("model", ""),
        llm_result.get("input_tokens", 0),
        llm_result.get("output_tokens", 0),
    )

    file_exists = os.path.exists(EVALUATIONS_CSV)
    with open(EVALUATIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "user_question": user_question,
            "puntaje": evaluacion.puntaje,
            "justificacion": evaluacion.justificacion,
            "input_tokens": llm_result.get("input_tokens", 0),
            "output_tokens": llm_result.get("output_tokens", 0),
            "latency_ms": llm_result.get("latency_ms", 0),
            "cost_usd": cost,
        })

    row = {
        "timestamp": timestamp,
        "user_question": user_question,
        "puntaje": evaluacion.puntaje,
        "justificacion": evaluacion.justificacion,
        "llm": llm_result,
        "cost_usd": cost,
    }
    with open(EVALUATIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
