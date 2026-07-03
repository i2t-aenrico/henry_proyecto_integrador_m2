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
    """Genera el embedding de la pregunta con el mismo proveedor del índice.

    Args:
        question: pregunta del usuario en texto plano.

    Returns:
        Vector (lista de floats) que representa la pregunta.
    """
    return embed_texts([question])[0]


def retrieve_chunks(question: str, index_path: str, top_k: int) -> list[dict]:
    """Recupera los top_k chunks más relevantes para la pregunta, desde un único índice.

    Args:
        question: pregunta del usuario.
        index_path: ruta al archivo de índice de un dataset puntual
            (ej: outputs/index/faq_macro_index.json).
        top_k: cantidad máxima de chunks a devolver.

    Returns:
        Lista de chunks ordenados por similitud descendente (ver
        vector_store.search).
    """
    query_vector = embed_query(question)
    index = load_index(index_path)
    return search(query_vector, index, top_k=top_k)


def retrieve_chunks_all_datasets(
    question: str, index_dir: str, top_k: int, min_per_dataset: int = 3
) -> list[dict]:
    """Busca en TODOS los índices disponibles en index_dir.

    A diferencia de un simple top-k global, esta función GARANTIZA que al
    menos `min_per_dataset` chunks de cada dataset entren al resultado final
    (si existen candidatos), en vez de dejar que un dataset con matches muy
    fuertes tape por completo a los demás. Los lugares restantes (hasta
    completar el tamaño final) se rellenan con los mejores candidatos
    globales que hayan quedado afuera de la garantía.

    Esto evita el problema detectado en el benchmark: una pregunta genérica
    ("¿cuánto cuesta reparar algo que compré?") competía en similitud contra
    chunks de un dataset dominante y los chunks realmente relevantes de otros
    datasets (ej. reparaciones de Yam o Hothaus) quedaban afuera del top-k
    global aunque estuvieran entre los mejores de su propio dataset.

    Con min_per_dataset=1 alcanzaba con que el chunk correcto fuera el MEJOR
    match dentro de su propio dataset, algo que no siempre pasa con preguntas
    genéricas (el chunk correcto puede ser el 2do o 3er mejor localmente).
    Por eso se garantizan los top-`min_per_dataset` locales de cada dataset,
    no solo el primero.

    El tamaño final es max(top_k, cantidad_de_datasets * min_per_dataset),
    para que siempre haya lugar para todos los chunks garantizados.

    Args:
        question: pregunta del usuario.
        index_dir: carpeta donde están todos los índices (uno por dataset).
        top_k: tamaño "deseado" del resultado; puede crecer si hace falta
            para alojar la garantía por dataset (ver total_final).
        min_per_dataset: cantidad mínima de chunks locales a garantizar por
            cada dataset, sin importar cómo compitan globalmente en score.

    Returns:
        Lista de chunks combinados de todos los datasets, ordenados por
        score descendente, de tamaño total_final.

    Raises:
        FileNotFoundError: si no hay ningún índice en index_dir (todavía no
            se corrió build_index.py para ningún dataset).
    """
    index_paths = sorted(glob.glob(f"{index_dir}/*_index.json"))
    if not index_paths:
        raise FileNotFoundError(f"No se encontraron índices en {index_dir}")

    query_vector = embed_query(question)
    total_final = max(top_k, len(index_paths) * min_per_dataset)

    pool: dict[tuple[str, str], dict] = {}
    garantizados: list[dict] = []

    for path in index_paths:
        index = load_index(path)
        local_matches = search(query_vector, index, top_k=max(top_k, min_per_dataset))
        for chunk in local_matches:
            key = (chunk.get("source", ""), chunk["chunk_id"])
            pool[key] = chunk  # dedup automático si dos índices compartieran un chunk
        garantizados.extend(local_matches[:min_per_dataset])

    vistos: set[tuple[str, str]] = set()
    garantizados_unicos: list[dict] = []
    for chunk in garantizados:
        key = (chunk.get("source", ""), chunk["chunk_id"])
        if key not in vistos:
            vistos.add(key)
            garantizados_unicos.append(chunk)

    restantes = sorted(
        (chunk for key, chunk in pool.items() if key not in vistos),
        key=lambda c: c["score"],
        reverse=True,
    )
    faltan = max(total_final - len(garantizados_unicos), 0)
    combinados = garantizados_unicos + restantes[:faltan]
    combinados.sort(key=lambda c: c["score"], reverse=True)
    return combinados[:total_final]


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Arma el prompt de usuario con la pregunta y los chunks recuperados.

    Args:
        question: pregunta original del usuario.
        chunks: chunks recuperados por retrieve_chunks o
            retrieve_chunks_all_datasets.

    Returns:
        Texto final del prompt de usuario, listo para pasarle a call_llm.
    """
    return TEMPLATE_USUARIO.format(
        pregunta=question,
        chunks_formateados=formatear_chunks(chunks),
    )


def generate_answer(question: str, chunks: list[dict], multi_source: bool = False) -> tuple[str, dict]:
    """Llama al LLM y devuelve la respuesta junto con las métricas de uso.

    Usa el prompt de sistema multi-fuente cuando la búsqueda combinó varios
    datasets, para que el modelo indique explícitamente de qué organización
    proviene cada dato usado en la respuesta.

    Args:
        question: pregunta original del usuario.
        chunks: chunks recuperados que forman el contexto de la respuesta.
        multi_source: True si los chunks vienen de retrieve_chunks_all_datasets
            (dataset="all"), para elegir SYSTEM_ASISTENTE_MULTI en vez del
            prompt de sistema de un único dominio.

    Returns:
        Tupla (answer, llm_result): answer es el texto de respuesta ya
        extraído del JSON crudo del LLM; llm_result es el diccionario con
        texto crudo, tokens de uso, proveedor, modelo y latencia (ver llm.py).
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

    Args:
        question: pregunta del usuario en texto plano.
        dataset: nombre del dataset a consultar, o "all" para buscar en
            todos a la vez.
        evaluate: si es True, corre además el agente evaluador sobre la
            respuesta generada.

    Returns:
        Diccionario con user_question, system_answer y chunks_related (y,
        si evaluate=True, también la clave evaluation con puntaje y
        justificación). Es el mismo diccionario que se imprime como JSON
        en main() y que se muestra en la interfaz web (app.py).
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
    """Punto de entrada de línea de comandos: parsea argumentos, corre
    answer_question y muestra el resultado como JSON en stdout."""
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
