"""
vector_store.py — Almacenamiento e indexación de embeddings.

Implementa un índice vectorial simple en memoria (numpy), suficiente para
un corpus de decenas de chunks como el de este proyecto. La búsqueda usa
similitud coseno, calculada como producto punto entre vectores normalizados.
El índice se persiste en un archivo JSON: guarda el texto, los metadatos
y el vector de cada chunk, de modo que el pipeline de consulta pueda
cargarlo sin volver a llamar al modelo de embeddings sobre el corpus.
"""

from __future__ import annotations

import json
import os

import numpy as np


def save_index(chunks: list[dict], path: str) -> None:
    """Guarda chunks + embeddings + metadatos en un archivo JSON.

    Crea el directorio destino si no existe. Se llama una vez por dataset
    desde build_index.py; cada dataset genera su propio archivo de índice
    (ver index_path_for en build_index.py).

    Args:
        chunks: lista de registros a persistir, cada uno con al menos
            chunk_id, text, token_count, source y embedding.
        path: ruta del archivo JSON de salida.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(path: str) -> list[dict]:
    """Carga el índice previamente guardado.

    Args:
        path: ruta del archivo JSON de índice a leer.

    Returns:
        Lista de registros de chunk (chunk_id, text, token_count, source,
        embedding), tal como se guardaron con save_index.

    Raises:
        FileNotFoundError: si el archivo no existe (por ejemplo, si todavía
            no se corrió build_index.py para ese dataset).
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def search(query_vector: list[float], index: list[dict], top_k: int = 5) -> list[dict]:
    """Devuelve los top_k chunks más similares a query_vector (similitud coseno).

    Calcula la similitud como producto punto entre vectores normalizados,
    lo cual es matemáticamente equivalente a la similitud coseno. Es una
    búsqueda exacta por fuerza bruta (no aproximada): recorre todos los
    vectores del índice, adecuada para corpus de hasta unas pocas decenas
    de miles de chunks (ver justificación en el README, sección "Método de
    búsqueda").

    Args:
        query_vector: embedding de la pregunta del usuario.
        index: lista de registros de chunk con su campo "embedding"
            (tal como los devuelve load_index).
        top_k: cantidad máxima de resultados a devolver.

    Returns:
        Lista de hasta top_k chunks, ordenados de mayor a menor similitud.
        Cada chunk incluye un campo "score" (similitud coseno, entre -1 y 1)
        y ya NO incluye el campo "embedding" (se descarta para no inflar
        el tamaño de la respuesta ni del JSON final). Devuelve lista vacía
        si el índice está vacío.
    """
    if not index:
        return []
    matrix = np.array([item["embedding"] for item in index], dtype=np.float32)
    query = np.array(query_vector, dtype=np.float32)
    query = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix_normalized = matrix / norms
    scores = matrix_normalized @ query

    ranked_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx in ranked_idx:
        item = dict(index[int(idx)])
        item["score"] = float(scores[int(idx)])
        item.pop("embedding", None)
        results.append(item)
    return results
