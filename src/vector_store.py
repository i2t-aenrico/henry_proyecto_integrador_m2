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
    """Guarda chunks + embeddings + metadatos en un archivo JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(path: str) -> list[dict]:
    """Carga el índice previamente guardado."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def search(query_vector: list[float], index: list[dict], top_k: int = 5) -> list[dict]:
    """Devuelve los top_k chunks más similares a query_vector (similitud coseno)."""
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
