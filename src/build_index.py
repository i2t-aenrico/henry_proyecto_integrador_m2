"""
build_index.py — Pipeline de datos (ingestión → chunking → embeddings → índice).

Uso:
    uv run python src/build_index.py
    uv run python src/build_index.py --input data/faq_document.txt
"""

from __future__ import annotations

import argparse
import time

from chunking import chunk_document
from embeddings import embed_texts
from settings import get_settings
from vector_store import save_index


def load_document(path: str) -> str:
    """Lee el documento fuente de texto plano."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_chunk_records(chunks) -> list[dict]:
    """Convierte los objetos Chunk en diccionarios serializables."""
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
    """Genera embeddings para todos los chunks y los adjunta a cada registro."""
    texts = [r["text"] for r in records]
    vectors = embed_texts(texts)
    for record, vector in zip(records, vectors):
        record["embedding"] = vector
    return records


def main(input_path: str = "data/faq_document.txt") -> None:
    settings = get_settings()
    start = time.time()

    text = load_document(input_path)
    chunks = chunk_document(text)
    print(f"[build_index] {len(chunks)} chunks generados desde {input_path}")

    records = build_chunk_records(chunks)
    records = attach_embeddings(records)
    print(f"[build_index] embeddings generados con proveedor={settings.embedding_provider}")

    index_path = f"{settings.index_dir}/faq_index.json"
    save_index(records, index_path)
    elapsed = time.time() - start
    print(f"[build_index] índice guardado en {index_path} ({elapsed:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye el índice vectorial del FAQ")
    parser.add_argument("--input", default="data/faq_document.txt")
    args = parser.parse_args()
    main(args.input)
