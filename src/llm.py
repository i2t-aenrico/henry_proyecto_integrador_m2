"""
llm.py — Llamada al LLM para la etapa de generación del pipeline RAG.

Soporta dos proveedores intercambiables vía LLM_PROVIDER (ver settings.py):
- "anthropic": usa la API de Anthropic (Claude).
- "openai":    usa la API de OpenAI.

Devuelve tanto el texto de respuesta como el uso de tokens reportado por
la API, necesario para calcular métricas de costo y latencia.
"""

from __future__ import annotations

import json
import time

from settings import get_settings


def call_llm(system_prompt: str, user_prompt: str) -> dict:
    """Llama al proveedor configurado y devuelve texto + métricas de uso.

    Delega en _call_anthropic o _call_openai según LLM_PROVIDER (ver
    settings.py), y agrega la latencia medida de punta a punta de la
    llamada (en milisegundos) al resultado.

    Args:
        system_prompt: instrucciones de sistema (rol, reglas, formato de
            salida esperado).
        user_prompt: mensaje de usuario (pregunta + contexto recuperado).

    Returns:
        Diccionario con: text (respuesta cruda del modelo), input_tokens,
        output_tokens, provider, model y latency_ms.

    Raises:
        ValueError: si LLM_PROVIDER no es "anthropic" ni "openai".
    """
    settings = get_settings()
    start = time.time()

    if settings.llm_provider == "anthropic":
        result = _call_anthropic(system_prompt, user_prompt, settings)
    elif settings.llm_provider == "openai":
        result = _call_openai(system_prompt, user_prompt, settings)
    else:
        raise ValueError(f"LLM_PROVIDER desconocido: {settings.llm_provider!r}")

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def _call_anthropic(system_prompt: str, user_prompt: str, settings) -> dict:
    """Ejecuta la llamada concreta a la API de Anthropic (Claude).

    Args:
        system_prompt: instrucciones de sistema.
        user_prompt: mensaje de usuario.
        settings: instancia de Settings con el modelo y la API key a usar.

    Returns:
        Diccionario con text, input_tokens, output_tokens, provider="anthropic"
        y model (sin latency_ms todavía; eso lo agrega call_llm).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return {
        "text": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "provider": "anthropic",
        "model": settings.anthropic_model,
    }


def _call_openai(system_prompt: str, user_prompt: str, settings) -> dict:
    """Ejecuta la llamada concreta a la API de OpenAI (Chat Completions).

    Args:
        system_prompt: instrucciones de sistema.
        user_prompt: mensaje de usuario.
        settings: instancia de Settings con el modelo y la API key a usar.

    Returns:
        Diccionario con text, input_tokens, output_tokens, provider="openai"
        y model (sin latency_ms todavía; eso lo agrega call_llm).
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content
    return {
        "text": text,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "provider": "openai",
        "model": settings.openai_llm_model,
    }


def extract_answer(raw_text: str) -> str:
    """Extrae system_answer del JSON devuelto por el LLM, con fallback robusto.

    El prompt le pide al modelo que devuelva únicamente un JSON con la
    clave "system_answer", pero algunos modelos envuelven la respuesta en
    fences de markdown (```json ... ```). Esta función limpia esos fences
    antes de parsear, y si el parseo falla igual (el modelo no devolvió
    JSON válido), devuelve el texto crudo tal cual para no perder la
    respuesta.

    Args:
        raw_text: texto crudo devuelto por el LLM (llm_result["text"]).

    Returns:
        El valor de la clave "system_answer" si el parseo fue exitoso;
        si no, el texto crudo sin procesar (fallback).
    """
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        return data["system_answer"]
    except (json.JSONDecodeError, KeyError):
        return raw_text.strip()
