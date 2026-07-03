"""
build_index.py — Pipeline de datos (ingestión → chunking → embeddings → índice).

Cada documento fuente se indexa por separado: el nombre del índice se deriva
del nombre del archivo de entrada (data/faq_macro.txt -> outputs/index/faq_macro_index.json),
para poder mantener varios índices independientes sin que se pisen entre sí.

Uso:
    uv run python src/build_index.py
    uv run python src/build_index.py --input data/faq_macro.txt
    uv run python src/build_index.py --input data/faq_yam.txt --output outputs/index/mi_indice.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from chunking import chunk_document
from embeddings import embed_texts
from settings import get_settings
from vector_store import save_index


def load_document(path: str) -> str:
    """Lee el documento fuente de texto plano.

    Args:
        path: ruta al archivo .txt a indexar.

    Returns:
        Contenido completo del archivo como string (codificado en UTF-8).
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_chunk_records(chunks) -> list[dict]:
    """Convierte los objetos Chunk (dataclass) en diccionarios serializables.

    Necesario porque el índice se persiste como JSON, y los objetos Chunk
    de chunking.py no son serializables directamente.

    Args:
        chunks: lista de objetos Chunk devueltos por chunk_document.

    Returns:
        Lista de diccionarios con chunk_id, text, token_count y source
        (sin el embedding todavía; eso lo agrega attach_embeddings).
    """
    return [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "token_count": c.token_count,
            "source": c.source,
        }
        for c in chunks
    ]


def attach_embeddings(records: list[dict]) -> list[dict]:
    """Genera embeddings para todos los chunks y los adjunta a cada registro.

    Hace una única llamada a embed_texts con todos los textos juntos (en
    vez de uno por uno), para aprovechar el procesamiento por lotes tanto
    en la API de OpenAI como en el modelo local de Sentence-Transformers.

    Args:
        records: lista de diccionarios de chunk (salida de build_chunk_records).

    Returns:
        La misma lista de records, mutada in-place, con un campo
        "embedding" (lista de floats) agregado a cada uno.
    """
    texts = [r["text"] for r in records]
    vectors = embed_texts(texts)
    for record, vector in zip(records, vectors):
        record["embedding"] = vector
    return records


def index_path_for(input_path: str, index_dir: str) -> str:
    """Deriva el path del índice a partir del nombre del archivo de entrada.

    data/faq_macro.txt  -> {index_dir}/faq_macro_index.json
    data/faq_document.txt -> {index_dir}/faq_document_index.json

    Esto permite mantener un índice independiente por cada dataset, sin
    que uno pise el archivo de otro al correr build_index.py varias veces.

    Args:
        input_path: ruta del documento fuente indexado (usa el nombre base,
            sin extensión, para armar el nombre del índice).
        index_dir: carpeta donde se guardan todos los índices.

    Returns:
        Ruta completa del archivo de índice para ese dataset.
    """
    stem = Path(input_path).stem
    return f"{index_dir}/{stem}_index.json"


def main(input_path: str = "data/faq_document.txt", output_path: str | None = None) -> None:
    """Orquesta el pipeline de datos completo para un único documento fuente.

    Etapas: (1) leer el archivo, (2) chunkear el texto, (3) generar
    embeddings para todos los chunks, y (4) guardar chunks + embeddings en
    el índice correspondiente a ese dataset.

    Args:
        input_path: ruta al documento .txt a indexar.
        output_path: ruta explícita para el índice de salida. Si es None
            (default), se deriva automáticamente del nombre de input_path
            mediante index_path_for.
    """
    settings = get_settings()
    start = time.time()

    text = load_document(input_path)
    chunks = chunk_document(text, source=Path(input_path).name)
    print(f"[build_index] {len(chunks)} chunks generados desde {input_path}")

    records = build_chunk_records(chunks)
    records = attach_embeddings(records)
    print(f"[build_index] embeddings generados con proveedor={settings.embedding_provider}")

    index_path = output_path or index_path_for(input_path, settings.index_dir)
    save_index(records, index_path)
    elapsed = time.time() - start
    print(f"[build_index] índice guardado en {index_path} ({elapsed:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye el índice vectorial de un FAQ")
    parser.add_argument("--input", default="data/faq_document.txt", help="Documento fuente a indexar")
    parser.add_argument(
        "--output", default=None,
        help="Path del índice de salida. Por defecto se deriva del nombre del --input "
             "(ej: data/faq_macro.txt -> outputs/index/faq_macro_index.json).",
    )
    args = parser.parse_args()
    main(args.input, args.output)
