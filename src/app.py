"""
app.py — Interfaz web simple (Gradio) para probar el asistente de FAQ.

Permite elegir una pregunta y qué base de FAQ consultar (una en particular
o todas a la vez), y muestra la respuesta junto con los chunks usados como
contexto (fuente, similitud y texto).

No reemplaza a query.py: reutiliza el mismo pipeline (answer_question), solo
le agrega una capa visual para no tener que usar la terminal.

Uso:
    uv run python src/app.py

Requiere haber generado antes al menos un índice con build_index.py.
"""

from __future__ import annotations

import glob
import os

import gradio as gr

from prompts_loader import FUENTES_LEGIBLES
from query import answer_question
from settings import get_settings


def _available_datasets() -> list[str]:
    """Lista los datasets ya indexados, a partir de los archivos en outputs/index/.

    Returns:
        Lista de nombres de dataset (ej: ["faq_agn", "faq_document",
        "faq_hothaus", "faq_macro", "faq_yam"]), sin el sufijo "_index.json".
        Vacía si todavía no se corrió build_index.py para ningún archivo.
    """
    settings = get_settings()
    paths = sorted(glob.glob(f"{settings.index_dir}/*_index.json"))
    datasets = []
    for path in paths:
        filename = os.path.basename(path)
        stem = filename[: -len("_index.json")] if filename.endswith("_index.json") else filename
        datasets.append(stem)
    return datasets


def _etiqueta(dataset: str) -> str:
    """Nombre legible para mostrar en el dropdown, ej: faq_macro -> Banco Macro.

    Args:
        dataset: nombre interno del dataset (sin extensión), ej: "faq_macro".

    Returns:
        Nombre legible según FUENTES_LEGIBLES, o el nombre original si no
        hay traducción configurada para ese dataset.
    """
    return FUENTES_LEGIBLES.get(f"{dataset}.txt", dataset)


def _dataset_choices() -> list[tuple[str, str]]:
    """Devuelve pares (etiqueta visible, valor real) para el dropdown de Gradio.

    Returns:
        Lista de tuplas (etiqueta, valor); la primera opción es siempre
        "Todos los FAQ" -> "all", seguida por cada dataset ya indexado.
    """
    datasets = _available_datasets()
    choices = [("Todos los FAQ", "all")]
    choices += [(_etiqueta(d), d) for d in datasets]
    return choices


def _formatear_chunk(chunk: dict) -> str:
    """Convierte un chunk en un bloque de Markdown legible para la interfaz.

    Trunca el texto a 260 caracteres para no saturar el panel de detalle
    cuando hay muchos chunks (por ejemplo, en modo "Todos los FAQ").

    Args:
        chunk: diccionario de chunk (chunk_id, score, text, source).

    Returns:
        String en Markdown con fuente, chunk_id, similitud y un resumen
        del texto.
    """
    fuente = FUENTES_LEGIBLES.get(chunk.get("source", ""), chunk.get("source", "desconocida"))
    texto = chunk["text"]
    resumen = texto if len(texto) <= 260 else texto[:260].rstrip() + "…"
    return f"**{fuente}** · `{chunk['chunk_id']}` · similitud={chunk['score']:.3f}\n\n> {resumen}"


def preguntar(pregunta: str, dataset: str, evaluar: bool) -> tuple[str, str]:
    """Ejecuta el pipeline de RAG y devuelve (respuesta, detalle_de_chunks).

    Es el callback que se dispara al hacer click en "Preguntar" (o Enter en
    el campo de pregunta). Reutiliza directamente answer_question de
    query.py, sin duplicar lógica de retrieval ni de generación.

    Args:
        pregunta: texto ingresado en el campo de pregunta.
        dataset: valor del dropdown ("all" o el nombre de un dataset puntual).
        evaluar: estado del checkbox de evaluación.

    Returns:
        Tupla (respuesta, detalle): respuesta es el texto para el textbox
        de "Respuesta"; detalle es el Markdown con los chunks usados (y,
        si evaluar=True, el veredicto del agente evaluador al final).
    """
    if not pregunta or not pregunta.strip():
        return "Escribí una pregunta antes de consultar.", ""
    if not dataset:
        return "No hay ningún FAQ indexado todavía. Corré build_index.py primero.", ""

    try:
        resultado = answer_question(pregunta.strip(), dataset=dataset, evaluate=evaluar)
    except FileNotFoundError as exc:
        return f"Error: {exc}. Corré build_index.py para generar el índice.", ""

    respuesta = resultado["system_answer"]

    bloques = [_formatear_chunk(c) for c in resultado["chunks_related"]]
    detalle = "\n\n---\n\n".join(bloques) if bloques else "No se recuperó ningún chunk."

    if evaluar and "evaluation" in resultado:
        ev = resultado["evaluation"]
        detalle += (
            f"\n\n---\n\n**Evaluación del agente evaluador:** "
            f"{ev['puntaje']}/10 — {ev['justificacion']}"
        )

    return respuesta, detalle


def main() -> None:
    """Arma el layout de Gradio (Blocks) y levanta el servidor local."""
    choices = _dataset_choices()
    valor_default = choices[0][1] if choices else None

    with gr.Blocks(title="Asistente de FAQ (RAG)") as demo:
        gr.Markdown("## Asistente de preguntas frecuentes (RAG)")
        gr.Markdown(
            "Elegí un FAQ puntual o **Todos los FAQ** para buscar en todas las "
            "bases a la vez; la respuesta indica de qué organización sale cada dato."
        )

        with gr.Row():
            pregunta = gr.Textbox(
                label="Pregunta", placeholder="Escribí tu pregunta...", scale=3
            )
            modelo = gr.Dropdown(
                choices=choices, value=valor_default, label="Modelo / fuente a consultar"
            )

        evaluar = gr.Checkbox(
            label="Evaluar la respuesta (agente evaluador, bonus)",
            value=False,
            info=(
                "Si lo tildás, un segundo LLM revisa la respuesta y le pone una nota "
                "de 0 a 10 según fidelidad a los chunks, cobertura, honestidad y claridad. "
                "Hace una llamada extra al modelo, por eso tarda un poco más."
            ),
        )
        boton = gr.Button("Preguntar", variant="primary")

        respuesta = gr.Textbox(label="Respuesta", lines=8)
        detalle = gr.Markdown(label="Chunks utilizados")

        boton.click(preguntar, inputs=[pregunta, modelo, evaluar], outputs=[respuesta, detalle])
        pregunta.submit(preguntar, inputs=[pregunta, modelo, evaluar], outputs=[respuesta, detalle])

    demo.launch()


if __name__ == "__main__":
    main()
