"""
query.py — Pipeline de consulta (embedding → búsqueda vectorial →
construcción de prompt → generación con LLM → JSON validado).

Uso:
    uv run python src/query.py -q "¿Cómo saco turno para la licencia de conducir?"
    uv run python src/query.py -q "¿Cómo pago mi tarjeta?" --dataset faq_macro
    uv run python src/query.py -q "¿Cómo pago mi tarjeta de crédito?" --dataset all
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

from embeddings import embed_texts
from evaluator import evaluate_answer
from llm import call_llm, extract_answer
from metrics_writer import registrar_evaluacion, registrar_metrica
from prompts_loader import (
    SYSTEM_ASISTENTE,
    SYSTEM_ASISTENTE_MULTI,
    TEMPLATE_USUARIO,
    formatear_chunks,
)
from schemas import RespuestaRAG
from settings import get_settings
from vector_store import load_index, search


def embed_query(question: str) -> list[float]:
    """Genera el embedding de la pregunta con el mismo proveedor del índice."""
    return embed_texts([question])[0]


def retrieve_chunks(question: str, index_path: str, top_k: int) -> list[dict]:
    """Recupera los top_k chunks más relevantes para la pregunta, desde un único índice."""
    query_vector = embed_query(question)
    index = load_index(index_path)
    return search(query_vector, index, top_k=top_k)


def retrieve_chunks_all_datasets(question: str, index_dir: str, top_k: int) -> list[dict]:
    """Busca en TODOS los índices disponibles en index_dir y devuelve los
    top_k globales, sin importar de qué dataset provengan.

    Se busca top_k dentro de cada índice individual y luego se combinan y
    reordenan todos los resultados por score, quedándonos con los top_k
    finales. Esto evita perder un buen match de un dataset chico frente a
    uno con muchos chunks.
    """
    index_paths = sorted(glob.glob(f"{index_dir}/*_index.json"))
    if not index_paths:
        raise FileNotFoundError(f"No se encontraron índices en {index_dir}")

    query_vector = embed_query(question)
    combined: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in index_paths:
        index = load_index(path)
        for chunk in search(query_vector, index, top_k=top_k):
            key = (chunk.get("source", ""), chunk["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            combined.append(chunk)

    combined.sort(key=lambda c: c["score"], reverse=True)
    return combined[:top_k]


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Arma el prompt de usuario con la pregunta y los chunks recuperados."""
    return TEMPLATE_USUARIO.format(
        pregunta=question,
        chunks_formateados=formatear_chunks(chunks),
    )


def generate_answer(question: str, chunks: list[dict], multi_source: bool = False) -> tuple[str, dict]:
    """Llama al LLM y devuelve la respuesta junto con las métricas de uso.

    Usa el prompt de sistema multi-fuente cuando la búsqueda combinó varios
    datasets, para que el modelo indique explícitamente de qué organización
    proviene cada dato usado en la respuesta.
    """
    user_prompt = build_prompt(question, chunks)
    system_prompt = SYSTEM_ASISTENTE_MULTI if multi_source else SYSTEM_ASISTENTE
    llm_result = call_llm(system_prompt, user_prompt)
    answer = extract_answer(llm_result["text"])
    return answer, llm_result


def answer_question(question: str, dataset: str = "faq_document", evaluate: bool = False) -> dict:
    """Orquesta el pipeline completo para una pregunta y devuelve el JSON final.

    `dataset` es el nombre del documento fuente (sin extensión) usado al indexar,
    por ejemplo "faq_document", "faq_macro", "faq_yam", etc. Debe coincidir con
    el índice generado por build_index.py para ese archivo.

    Si dataset="all", busca en todos los índices disponibles y arma una única
    respuesta indicando de qué organización proviene la información usada.

    Si evaluate=True, además llama al agente evaluador (bonus, sprint 2)
    y registra su veredicto en metrics/evaluations.csv y logs/evaluations.jsonl.
    """
    settings = get_settings()
    multi_source = dataset == "all"

    if multi_source:
        chunks = retrieve_chunks_all_datasets(question, settings.index_dir, settings.top_k)
    else:
        index_path = f"{settings.index_dir}/{dataset}_index.json"
        chunks = retrieve_chunks(question, index_path, settings.top_k)

    answer, llm_result = generate_answer(question, chunks, multi_source=multi_source)

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
        "--dataset", default="faq_document",
        help="Nombre del documento fuente indexado (sin extensión), ej: "
             "faq_document, faq_macro, faq_yam, faq_hothaus, faq_agn. "
             "Usá 'all' para buscar en todos los índices disponibles a la vez. "
             "Debe existir outputs/index/{dataset}_index.json (generado con build_index.py).",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Además de responder, evalúa la respuesta con el agente evaluador (bonus).",
    )
    args = parser.parse_args()

    result = answer_question(args.question, dataset=args.dataset, evaluate=args.evaluate)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: no se encontró el índice. Ejecutá primero build_index.py", file=sys.stderr)
        sys.exit(1)
