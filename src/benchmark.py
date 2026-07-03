"""
benchmark.py — Corre un lote de preguntas de prueba contra el pipeline de
consulta completo (embedding de la pregunta + búsqueda vectorial + generación
con el LLM) y mide el tiempo de respuesta de punta a punta de cada una.

Uso:
    uv run python src/benchmark.py
    uv run python src/benchmark.py --input data/preguntas_prueba.json --output outputs/benchmark_results.json
    uv run python src/benchmark.py --evaluate   # además corre el agente evaluador (más lento)

El archivo de entrada es una lista de objetos {"dataset": ..., "question": ...}.
`dataset` puede ser el nombre de un FAQ indexado (ej: "faq_macro") o "all" para
buscar en todos los índices a la vez.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from query import answer_question


def run_benchmark(input_path: str, output_path: str, evaluate: bool = False) -> None:
    """Corre todas las preguntas del archivo de entrada y mide su tiempo de respuesta.

    Para cada pregunta: mide el tiempo total (embedding + búsqueda + LLM)
    con time.perf_counter, llama a answer_question, y registra el
    resultado aunque haya un error (para no cortar todo el lote por una
    sola falla). Al final arma un resumen agregado y lo imprime en consola
    además de guardarlo junto con el detalle en output_path.

    Args:
        input_path: ruta al archivo JSON con la lista de preguntas
            ({"dataset": ..., "question": ...}).
        output_path: ruta donde guardar el JSON con el resumen y el
            detalle completo de cada pregunta.
        evaluate: si es True, corre también el agente evaluador sobre cada
            respuesta (más lento y con más costo de tokens).
    """
    with open(input_path, encoding="utf-8") as f:
        preguntas = json.load(f)

    resultados: list[dict] = []
    tiempos_por_dataset: dict[str, list[float]] = {}

    print(f"[benchmark] {len(preguntas)} preguntas a ejecutar\n")

    for i, item in enumerate(preguntas, start=1):
        pregunta = item["question"]
        dataset = item.get("dataset", "faq_document")

        inicio = time.perf_counter()
        error = None
        resultado = None
        try:
            resultado = answer_question(pregunta, dataset=dataset, evaluate=evaluate)
        except Exception as exc:  # noqa: BLE001 — registramos cualquier falla y seguimos con la siguiente
            error = str(exc)
        duracion_ms = round((time.perf_counter() - inicio) * 1000, 1)

        tiempos_por_dataset.setdefault(dataset, []).append(duracion_ms)

        top_chunk = None
        if resultado and resultado.get("chunks_related"):
            top_chunk = resultado["chunks_related"][0]

        fila = {
            "n": i,
            "dataset": dataset,
            "question": pregunta,
            "duracion_ms": duracion_ms,
            "answer": resultado["system_answer"] if resultado else None,
            "top_chunk_source": top_chunk.get("source") if top_chunk else None,
            "top_chunk_score": top_chunk.get("score") if top_chunk else None,
            "error": error,
        }
        resultados.append(fila)

        estado = "OK" if error is None else f"ERROR: {error}"
        print(f"[{i:02d}/{len(preguntas)}] ({dataset:<12}) {duracion_ms:>8.1f} ms — {estado} — {pregunta[:55]}")

    resumen = _armar_resumen(tiempos_por_dataset)

    salida = {"resumen": resumen, "resultados": resultados}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    _imprimir_resumen(resumen)
    print(f"\n[benchmark] resultados completos guardados en {output_path}")


def _armar_resumen(tiempos_por_dataset: dict[str, list[float]]) -> dict:
    """Calcula estadísticas agregadas de tiempo a partir de los tiempos por dataset.

    Args:
        tiempos_por_dataset: diccionario {dataset: [duraciones_ms, ...]}.

    Returns:
        Diccionario con total_preguntas, tiempo_total_ms, promedio_ms,
        mediana_ms, min_ms, max_ms globales, y un desglose por_dataset con
        las mismas métricas (excepto mediana) calculadas por separado para
        cada dataset.
    """
    todos = [t for tiempos in tiempos_por_dataset.values() for t in tiempos]
    resumen = {
        "total_preguntas": len(todos),
        "tiempo_total_ms": round(sum(todos), 1),
        "promedio_ms": round(statistics.mean(todos), 1) if todos else 0,
        "mediana_ms": round(statistics.median(todos), 1) if todos else 0,
        "min_ms": round(min(todos), 1) if todos else 0,
        "max_ms": round(max(todos), 1) if todos else 0,
        "por_dataset": {},
    }
    for dataset, tiempos in tiempos_por_dataset.items():
        resumen["por_dataset"][dataset] = {
            "cantidad": len(tiempos),
            "promedio_ms": round(statistics.mean(tiempos), 1),
            "min_ms": round(min(tiempos), 1),
            "max_ms": round(max(tiempos), 1),
        }
    return resumen


def _imprimir_resumen(resumen: dict) -> None:
    """Imprime en consola el resumen de tiempos, global y por dataset.

    Args:
        resumen: diccionario devuelto por _armar_resumen.
    """
    print("\n" + "=" * 64)
    print("RESUMEN DE TIEMPOS DE RESPUESTA")
    print("=" * 64)
    print(f"Preguntas ejecutadas : {resumen['total_preguntas']}")
    print(f"Tiempo total         : {resumen['tiempo_total_ms']:.1f} ms")
    print(f"Promedio             : {resumen['promedio_ms']:.1f} ms")
    print(f"Mediana              : {resumen['mediana_ms']:.1f} ms")
    print(f"Mínimo               : {resumen['min_ms']:.1f} ms")
    print(f"Máximo               : {resumen['max_ms']:.1f} ms")
    print("\nPor dataset:")
    for dataset, datos in resumen["por_dataset"].items():
        print(
            f"  {dataset:<14} n={datos['cantidad']:<3} "
            f"prom={datos['promedio_ms']:>8.1f} ms  "
            f"min={datos['min_ms']:>8.1f} ms  "
            f"max={datos['max_ms']:>8.1f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corre un lote de preguntas contra el pipeline de RAG y mide tiempos de respuesta"
    )
    parser.add_argument("--input", default="data/preguntas_prueba.json")
    parser.add_argument("--output", default="outputs/benchmark_results.json")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Además de responder, corre el agente evaluador sobre cada respuesta "
             "(más lento y consume más tokens; no se recomienda para lotes grandes).",
    )
    args = parser.parse_args()
    run_benchmark(args.input, args.output, evaluate=args.evaluate)
