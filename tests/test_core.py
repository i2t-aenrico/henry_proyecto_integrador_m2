"""
test_core.py — Tests que no llaman a ninguna API externa.

Cubren: chunking, validación de esquemas, búsqueda vectorial (con vectores
sintéticos) y las reglas de métricas (detección de "no respuesta" y costo).
"""

import numpy as np
import pytest

from chunking import chunk_document, estimate_tokens, MIN_TOKENS, MAX_TOKENS
from evaluator import build_evaluator_prompt, extract_evaluation
from metrics_writer import estimate_cost, was_answered
from schemas import EvaluacionRespuesta, RespuestaRAG
from vector_store import search


FAQ_SAMPLE = """\
Título del documento

1. Pregunta uno con suficiente contenido para superar el minimo de tokens
requerido por la consigna del proyecto integrador del modulo dos sobre RAG.

2. Pregunta dos con otro parrafo largo que tambien debe superar el minimo
de tokens exigido, describiendo un tramite ficticio con varios requisitos
para pruebas automatizadas del pipeline de chunking del proyecto.
"""


class TestChunking:
    def test_genera_chunks_dentro_de_rango(self):
        chunks = chunk_document(FAQ_SAMPLE)
        assert len(chunks) >= 1
        for c in chunks:
            assert MIN_TOKENS <= c.token_count <= MAX_TOKENS or c.token_count < MIN_TOKENS

    def test_documento_real_genera_20_o_mas_chunks(self):
        text = open("data/faq_document.txt", encoding="utf-8").read()
        chunks = chunk_document(text)
        assert len(chunks) >= 20
        for c in chunks:
            assert c.token_count >= MIN_TOKENS
            assert c.token_count <= MAX_TOKENS

    def test_estimate_tokens_proporcional_a_palabras(self):
        assert estimate_tokens("una dos tres") == int(3 * 1.3)


class TestVectorStore:
    def test_search_devuelve_el_mas_similar_primero(self):
        index = [
            {"chunk_id": "a", "text": "texto a", "embedding": [1.0, 0.0]},
            {"chunk_id": "b", "text": "texto b", "embedding": [0.0, 1.0]},
        ]
        results = search([0.9, 0.1], index, top_k=2)
        assert results[0]["chunk_id"] == "a"
        assert "embedding" not in results[0]

    def test_search_respeta_top_k(self):
        index = [{"chunk_id": str(i), "text": "x", "embedding": [1.0, i]} for i in range(10)]
        results = search([1.0, 0.0], index, top_k=3)
        assert len(results) == 3

    def test_search_indice_vacio(self):
        assert search([1.0, 0.0], [], top_k=3) == []


class TestSchemas:
    def test_respuesta_rag_valida(self):
        data = {
            "user_question": "¿Cómo saco turno?",
            "system_answer": "Se solicita de forma online desde el sitio municipal.",
            "chunks_related": [{"chunk_id": "chunk_001", "score": 0.8, "text": "texto"}],
        }
        r = RespuestaRAG(**data)
        assert r.user_question.startswith("¿Cómo")

    def test_respuesta_rag_rechaza_respuesta_corta(self):
        with pytest.raises(Exception):
            RespuestaRAG(user_question="hola", system_answer="ok", chunks_related=[])


class TestMetrics:
    def test_was_answered_detecta_no_respuesta(self):
        assert was_answered("Podés hacer el trámite en la oficina virtual.") is True
        assert was_answered("No tengo información suficiente sobre eso.") is False

    def test_estimate_cost_proveedor_conocido(self):
        cost = estimate_cost("anthropic", "claude-sonnet-4-6", 1000, 1000)
        assert cost > 0

    def test_estimate_cost_proveedor_desconocido_es_cero(self):
        assert estimate_cost("otro", "modelo-x", 1000, 1000) == 0.0


class TestEvaluator:
    def test_build_evaluator_prompt_incluye_pregunta_y_respuesta(self):
        chunks = [{"chunk_id": "chunk_001", "score": 0.8, "text": "texto de prueba"}]
        prompt = build_evaluator_prompt("pregunta?", "respuesta.", chunks)
        assert "pregunta?" in prompt
        assert "respuesta." in prompt
        assert "chunk_001" in prompt

    def test_extract_evaluation_parsea_json_limpio(self):
        raw = '{"puntaje": 8.5, "justificacion": "bien fundamentada"}'
        data = extract_evaluation(raw)
        assert data["puntaje"] == 8.5

    def test_extract_evaluation_parsea_con_fence_markdown(self):
        raw = '```json\n{"puntaje": 6, "justificacion": "incompleta"}\n```'
        data = extract_evaluation(raw)
        assert data["puntaje"] == 6

    def test_evaluacion_respuesta_valida_rango(self):
        with pytest.raises(Exception):
            EvaluacionRespuesta(puntaje=11, justificacion="fuera de rango")
