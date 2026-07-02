"""
query.py — Pipeline de consulta (embedding → búsqueda vectorial →
construcción de prompt → generación con LLM → JSON validado).

Uso:
    uv run python src/query.py -q "¿Cómo saco turno para la licencia de conducir?"
"""

from __future__ import annotations

import argparse
import json
import sys

from embeddings import embed_texts
from evaluator import evaluate_answer
from llm import call_llm, extract_answer
from metrics_writer import registrar_evaluacion, registrar_metrica
from prompts_loader import SYSTEM_ASISTENTE, TEMPLATE_USUARIO, formatear_chunks
from schemas import RespuestaRAG
from settings import get_settings
from vector_store import load_index, search


def embed_query(question: str) -> list[float]:
    """Genera el embedding de la pregunta con el mismo proveedor del índice."""
    return embed_texts([question])[0]


def retrieve_chunks(question: str, index_path: str, top_k: int) -> list[dict]:
    """Recupera los top_k chunks más relevantes para la pregunta."""
    query_vector = embed_query(question)
    index = load_index(index_path)
    return search(query_vector, index, top_k=top_k)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Arma el prompt de usuario con la pregunta y los chunks recuperados."""
    return TEMPLATE_USUARIO.format(
        pregunta=question,
        chunks_formateados=formatear_chunks(chunks),
    )


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, dict]:
    """Llama al LLM y devuelve la respuesta junto con las métricas de uso."""
    user_prompt = build_prompt(question, chunks)
    llm_result = call_llm(SYSTEM_ASISTENTE, user_prompt)
    answer = extract_answer(llm_result["text"])
    return answer, llm_result


def answer_question(question: str, evaluate: bool = False) -> dict:
    """Orquesta el pipeline completo para una pregunta y devuelve el JSON final.

    Si evaluate=True, además llama al agente evaluador (bonus, sprint 2)
    y registra su veredicto en metrics/evaluations.csv y logs/evaluations.jsonl.
    """
    settings = get_settings()
    index_path = f"{settings.index_dir}/faq_index.json"

    chunks = retrieve_chunks(question, index_path, settings.top_k)
    answer, llm_result = generate_answer(question, chunks)

    registrar_metrica(question, answer, chunks, llm_result)

    result = RespuestaRAG(
        user_question=question,
        system_answer=answer,
        chunks_related=chunks,
    )
    output = json.loads(result.model_dump_json())

    if evaluate:
        evaluacion, eval_llm_result = evaluate_answer(question, answer, chunks)
        registrar_evaluacion(question, evaluacion, eval_llm_result)
        output["evaluation"] = json.loads(evaluacion.model_dump_json())

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el FAQ mediante RAG")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Además de responder, evalúa la respuesta con el agente evaluador (bonus).",
    )
    args = parser.parse_args()

    result = answer_question(args.question, evaluate=args.evaluate)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: no se encontró el índice. Ejecutá primero build_index.py", file=sys.stderr)
        sys.exit(1)
