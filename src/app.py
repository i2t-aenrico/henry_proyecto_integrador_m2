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
    """Lista los datasets ya indexados, a partir de los archivos en outputs/index/."""
    settings = get_settings()
    paths = sorted(glob.glob(f"{settings.index_dir}/*_index.json"))
    datasets = []
    for path in paths:
        filename = os.path.basename(path)
        stem = filename[: -len("_index.json")] if filename.endswith("_index.json") else filename
        datasets.append(stem)
    return datasets


def _etiqueta(dataset: str) -> str:
    """Nombre legible para mostrar en el dropdown, ej: faq_macro -> Banco Macro."""
    return FUENTES_LEGIBLES.get(f"{dataset}.txt", dataset)


def _dataset_choices() -> list[tuple[str, str]]:
    """Devuelve pares (etiqueta visible, valor real) para el dropdown de Gradio."""
    datasets = _available_datasets()
    choices = [("Todos los FAQ", "all")]
    choices += [(_etiqueta(d), d) for d in datasets]
    return choices


def _formatear_chunk(chunk: dict) -> str:
    fuente = FUENTES_LEGIBLES.get(chunk.get("source", ""), chunk.get("source", "desconocida"))
    texto = chunk["text"]
    resumen = texto if len(texto) <= 260 else texto[:260].rstrip() + "…"
    return f"**{fuente}** · `{chunk['chunk_id']}` · similitud={chunk['score']:.3f}\n\n> {resumen}"


def preguntar(pregunta: str, dataset: str, evaluar: bool) -> tuple[str, str]:
    """Ejecuta el pipeline de RAG y devuelve (respuesta, detalle_de_chunks)."""
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
