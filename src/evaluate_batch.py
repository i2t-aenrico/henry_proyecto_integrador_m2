"""
evaluate_batch.py — Corre el agente evaluador sobre un archivo de ejemplos
ya generado (outputs/sample_queries.json) y guarda el resultado con los
puntajes agregados.

Uso:
    uv run python src/evaluate_batch.py --input outputs/sample_queries.json
"""

from __future__ import annotations

import argparse
import json

from evaluator import evaluate_answer
from metrics_writer import registrar_evaluacion


def evaluate_file(input_path: str, output_path: str) -> None:
    """Corre el agente evaluador sobre cada item de un archivo de ejemplos.

    Lee una lista de objetos {user_question, system_answer, chunks_related}
    (típicamente outputs/sample_queries.json, generado a mano o con
    query.py), evalúa cada uno con evaluate_answer, registra el veredicto
    en las métricas agregadas (metrics/evaluations.csv y
    logs/evaluations.jsonl) y guarda el archivo completo con el campo
    "evaluation" agregado a cada item.

    Args:
        input_path: ruta al archivo JSON de entrada con los ejemplos a
            evaluar.
        output_path: ruta donde guardar el archivo de salida con las
            evaluaciones ya agregadas a cada item.
    """
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    resultados = []
    for item in items:
        evaluacion, llm_result = evaluate_answer(
            item["user_question"], item["system_answer"], item["chunks_related"]
        )
        registrar_evaluacion(item["user_question"], evaluacion, llm_result)
        item_evaluado = dict(item)
        item_evaluado["evaluation"] = json.loads(evaluacion.model_dump_json())
        resultados.append(item_evaluado)
        print(f"[evaluate_batch] {item['user_question'][:50]!r} -> {evaluacion.puntaje}/10")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"[evaluate_batch] resultados guardados en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un archivo de ejemplos con el agente evaluador")
    parser.add_argument("--input", default="outputs/sample_queries.json")
    parser.add_argument("--output", default="outputs/sample_queries_evaluated.json")
    args = parser.parse_args()
    evaluate_file(args.input, args.output)
