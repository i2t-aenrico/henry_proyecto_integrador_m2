"""
chunking.py — Fragmentación del documento FAQ en chunks semánticos.

Estrategia elegida: segmentación por párrafo (cada entrada de FAQ es una
pregunta con su respuesta, ya delimitada por líneas en blanco en el
documento fuente). Es la estrategia más coherente semánticamente para un
FAQ, porque cada bloque es autocontenido: no tiene sentido partir una
pregunta de su respuesta.

Reglas de tamaño (requeridas por la consigna): cada chunk debe tener entre
50 y 500 tokens. Como algunos párrafos del documento (título, introducción)
son mas cortos que el minimo, se fusionan con el bloque siguiente. Si algún
párrafo fuera más largo que el máximo, se subdivide con una ventana de
tamaño fijo y solapamiento (fallback), aunque no ocurre en este documento.
"""

from dataclasses import dataclass

MIN_TOKENS = 50
MAX_TOKENS = 500
OVERLAP_TOKENS = 40


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    source: str


def estimate_tokens(text: str) -> int:
    """Aproxima la cantidad de tokens a partir del conteo de palabras.

    Se usa un factor de 1.3 palabras->tokens, una aproximación razonable
    para español sin depender de un tokenizador específico de modelo.

    Args:
        text: texto sobre el que estimar la cantidad de tokens.

    Returns:
        Cantidad estimada de tokens (entero).
    """
    return int(len(text.split()) * 1.3)


def split_into_paragraphs(text: str) -> list[str]:
    """Divide el documento en párrafos usando líneas en blanco como límite.

    Args:
        text: contenido completo del documento fuente.

    Returns:
        Lista de párrafos (strings), sin líneas en blanco ni espacios
        sobrantes al principio/final de cada uno.
    """
    raw_blocks = text.split("\n\n")
    return [b.strip() for b in raw_blocks if b.strip()]


def merge_short_paragraphs(paragraphs: list[str]) -> list[str]:
    """Fusiona párrafos que quedarían por debajo de MIN_TOKENS con el siguiente.

    Recorre los párrafos en orden, acumulándolos en un buffer hasta que la
    cantidad de tokens acumulados alcanza MIN_TOKENS; en ese momento cierra
    el bloque y empieza uno nuevo. Si al final queda un resto corto sin
    llegar al mínimo, se anexa al último bloque ya cerrado (o se agrega
    solo, si todavía no se cerró ningún bloque).

    Args:
        paragraphs: lista de párrafos ya separados por split_into_paragraphs.

    Returns:
        Lista de bloques de texto, cada uno con al menos MIN_TOKENS tokens
        (excepto, potencialmente, un único bloque si el documento entero
        fuera más corto que MIN_TOKENS).
    """
    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        if estimate_tokens(buffer) >= MIN_TOKENS:
            merged.append(buffer)
            buffer = ""
    if buffer:
        # Si sobra un resto corto al final, se anexa al último chunk.
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def split_oversized_paragraph(paragraph: str) -> list[str]:
    """Fallback: ventana de tamaño fijo con solapamiento para párrafos largos.

    Se activa solo si un bloque supera MAX_TOKENS luego del merge de
    párrafos cortos. Parte el texto en ventanas de palabras de tamaño fijo
    con un solapamiento de OVERLAP_TOKENS entre una ventana y la siguiente,
    para no perder contexto en los bordes de cada corte.

    Args:
        paragraph: bloque de texto que excede MAX_TOKENS.

    Returns:
        Lista de fragmentos de texto, cada uno dentro del límite de tamaño.
    """
    words = paragraph.split()
    step = int(MAX_TOKENS / 1.3) - int(OVERLAP_TOKENS / 1.3)
    window = int(MAX_TOKENS / 1.3)
    pieces = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + window])
        if piece:
            pieces.append(piece)
    return pieces


def chunk_document(text: str, source: str = "faq_document.txt") -> list[Chunk]:
    """Ejecuta el pipeline completo de chunking sobre el documento FAQ.

    Etapas: (1) separar en párrafos por líneas en blanco, (2) fusionar los
    párrafos cortos con el siguiente hasta cumplir el mínimo de tokens,
    (3) si algún bloque resultante supera el máximo, subdividirlo con
    ventana + solapamiento (fallback), y (4) envolver cada bloque final en
    un objeto Chunk con su id secuencial y su fuente de origen.

    Args:
        text: contenido completo del documento fuente ya leído.
        source: nombre del archivo de origen, para trazabilidad de cada
            chunk cuando se combinan varios documentos (ver build_index.py).

    Returns:
        Lista de objetos Chunk, cada uno con chunk_id, text, token_count y
        source.
    """
    paragraphs = split_into_paragraphs(text)
    merged = merge_short_paragraphs(paragraphs)

    final_blocks: list[str] = []
    for block in merged:
        if estimate_tokens(block) > MAX_TOKENS:
            final_blocks.extend(split_oversized_paragraph(block))
        else:
            final_blocks.append(block)

    chunks = [
        Chunk(
            chunk_id=f"chunk_{i:03d}",
            text=block,
            token_count=estimate_tokens(block),
            source=source,
        )
        for i, block in enumerate(final_blocks)
    ]
    return chunks
