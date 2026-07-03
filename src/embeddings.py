"""
embeddings.py — Generación de embeddings con proveedor intercambiable.

Soporta dos proveedores, elegidos por la variable de entorno
EMBEDDING_PROVIDER (ver settings.py):

- "openai": usa la API de OpenAI (modelo text-embedding-3-small, 1536 dim).
- "local":  usa Sentence-Transformers (modelo all-MiniLM-L6-v2, 384 dim,
            corre 100% en CPU, sin necesidad de API key).

Es obligatorio usar el mismo proveedor para indexar el corpus y para
embeber las consultas del usuario: mezclar proveedores produce espacios
vectoriales incompatibles y la búsqueda por similitud deja de tener sentido.
"""

from __future__ import annotations

import numpy as np

from settings import get_settings

_local_model_cache = None


def _get_local_model():
    """Carga (una sola vez) el modelo de Sentence-Transformers configurado.

    Usa una variable de módulo como caché para no recargar el modelo en
    cada llamada (la carga inicial tarda varios segundos y descarga pesos
    desde Hugging Face la primera vez que se ejecuta).

    Returns:
        Instancia de SentenceTransformer ya cargada en memoria.
    """
    global _local_model_cache
    if _local_model_cache is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        _local_model_cache = SentenceTransformer(settings.local_embedding_model)
    return _local_model_cache


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    """Genera embeddings usando la API de OpenAI.

    Args:
        texts: lista de textos a convertir en vectores (chunks o la
            pregunta del usuario).

    Returns:
        Lista de vectores (uno por texto de entrada), en el mismo orden.
    """
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_texts_local(texts: list[str]) -> list[list[float]]:
    """Genera embeddings localmente con Sentence-Transformers (sin API).

    Los vectores salen normalizados (norma unitaria) directamente del
    modelo, así el producto punto en la búsqueda equivale a similitud
    coseno sin necesidad de normalizar de nuevo en vector_store.py.

    Args:
        texts: lista de textos a convertir en vectores.

    Returns:
        Lista de vectores (uno por texto de entrada), en el mismo orden.
    """
    model = _get_local_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Punto de entrada único: delega en el proveedor configurado en .env.

    Tanto build_index.py (para los chunks) como query.py (para la pregunta
    del usuario) llaman siempre a esta función, nunca a los proveedores
    específicos directamente, para garantizar que ambos usen el mismo
    EMBEDDING_PROVIDER y los vectores sean comparables entre sí.

    Args:
        texts: lista de textos a convertir en vectores.

    Returns:
        Lista de vectores (uno por texto de entrada), en el mismo orden.

    Raises:
        ValueError: si EMBEDDING_PROVIDER no es "openai" ni "local".
    """
    settings = get_settings()
    if settings.embedding_provider == "openai":
        return embed_texts_openai(texts)
    if settings.embedding_provider == "local":
        return embed_texts_local(texts)
    raise ValueError(
        f"EMBEDDING_PROVIDER desconocido: {settings.embedding_provider!r}. "
        "Usa 'openai' o 'local'."
    )


def normalize(vector: list[float]) -> np.ndarray:
    """Normaliza un vector a norma unitaria para usar producto punto = coseno.

    No se usa en el camino principal (embed_texts_local ya normaliza), pero
    queda disponible como utilidad si algún vector llegara sin normalizar
    (por ejemplo, embeddings de OpenAI, que no vienen normalizados de por sí).

    Args:
        vector: vector a normalizar.

    Returns:
        Array de numpy con norma 1, o el vector original si su norma es 0
        (para evitar división por cero).
    """
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr
