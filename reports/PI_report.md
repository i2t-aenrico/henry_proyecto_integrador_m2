# Informe del Proyecto Integrador — Módulo 2

**Proyecto:** Asistente de FAQ sobre trámites de Santa Fe con RAG
**Módulo:** 2 — Bases de datos vectoriales y arquitectura RAG
**Dominio elegido:** Trámites municipales y provinciales de Santa Fe

## 1. Visión de arquitectura

El sistema separa dos pipelines independientes en ejecución pero acoplados en diseño, tal como se define en el material del módulo: el pipeline de datos corre una vez durante la indexación, y el pipeline de consulta corre cada vez que un usuario pregunta algo.

**Pipeline de datos:** carga del documento FAQ, chunking por párrafo, generación de embeddings y guardado del índice.

**Pipeline de consulta:** embedding de la pregunta, búsqueda vectorial, construcción del prompt con procedencia, generación con LLM, validación del JSON de salida y registro de métricas.

## 2. Estrategia de chunking

Se evaluaron tres alternativas: tamaño fijo con solapamiento, por oraciones y por párrafo. Se eligió **chunking por párrafo** porque el documento FAQ ya presenta una estructura natural de bloques pregunta-respuesta separados por líneas en blanco; partir un bloque con una ventana de tamaño fijo separaría la pregunta de su respuesta y degradaría la calidad de la recuperación.

Se aplicaron dos reglas de control de tamaño: los párrafos por debajo de 50 tokens se fusionan con el siguiente bloque, y los que superarían 500 tokens se subdividirían con una ventana de tamaño fijo y solapamiento de 40 tokens (mecanismo de resguardo que no llegó a activarse con el documento actual). El resultado sobre `data/faq_document.txt` (~1625 palabras) fue de 27 chunks, todos entre 62 y 110 tokens, dentro del rango exigido por la consigna.

## 3. Generación de embeddings

Se implementaron los dos proveedores permitidos por la consigna detrás de una interfaz común (`embed_texts`), seleccionable por variable de entorno:

| Proveedor | Modelo | Dimensiones | Ejecución |
| --- | --- | --- | --- |
| OpenAI | text-embedding-3-small | 1536 | vía API, requiere clave |
| Sentence-Transformers | all-MiniLM-L6-v2 | 384 | local, en CPU |

Es indispensable usar el mismo proveedor para indexar el corpus y para embeber las consultas, porque espacios vectoriales de distinta procedencia no son comparables por similitud coseno. Esta regla queda validada en el código: `query.py` reutiliza el proveedor configurado en `settings.py` tanto para construir el índice como para consultar.

## 4. Búsqueda vectorial

Con un corpus de 27 chunks, una estructura de indexación aproximada (ANN) no aporta ventajas de performance frente a una búsqueda exacta por fuerza bruta, y sí agrega una dependencia adicional (FAISS u otra librería). Se implementó búsqueda exacta por producto punto entre vectores normalizados, equivalente a similitud coseno, usando numpy. La función `search()` en `vector_store.py` es el único punto de contacto entre el resto del sistema y el mecanismo de búsqueda, lo que permite reemplazarla por un índice FAISS si el corpus creciera significativamente, sin modificar `build_index.py` ni `query.py`.

## 5. Arquitectura RAG completa

El flujo de consulta sigue las cuatro etapas descriptas en el módulo: (1) embedding de la consulta con el mismo modelo y dimensionalidad que los chunks, (2) búsqueda vectorial de los top-K chunks más similares, (3) ensamblado del contexto recuperado en un prompt con procedencia (identificador y score de cada chunk), y (4) generación de la respuesta con un LLM, que debe fundamentarse únicamente en los chunks recibidos.

El prompt de sistema (`prompts/prompts.py`) instruye explícitamente al modelo a no inventar información y a declarar cuando el contexto recuperado no alcanza para responder. Esta regla es la base del mecanismo de detección de preguntas no respondidas, descripto en la sección de métricas.

**Beneficio de RAG aplicado a este caso:** si mañana cambian los requisitos de un trámite (por ejemplo, un nuevo medio de pago para la licencia de conducir), alcanza con actualizar `data/faq_document.txt` y volver a correr `build_index.py`; no hace falta reentrenar ningún modelo. Además, cada respuesta queda atada a los chunks que la originaron, lo que da trazabilidad sobre qué parte del FAQ sustenta cada afirmación.

## 6. Organización modular del código

El código se organizó en archivos de responsabilidad única, siguiendo el mismo patrón de separación usado en el proyecto del Módulo 1:

| Archivo | Responsabilidad |
| --- | --- |
| `chunking.py` | Fragmentación del documento en chunks |
| `embeddings.py` | Generación de embeddings (OpenAI / local) |
| `vector_store.py` | Persistencia del índice y búsqueda por similitud |
| `build_index.py` | Orquestador del pipeline de datos |
| `query.py` | Orquestador del pipeline de consulta |
| `llm.py` | Llamada al LLM (Anthropic / OpenAI) y extracción de la respuesta |
| `metrics_writer.py` | Registro de métricas y logs por consulta |
| `schemas.py` | Contratos Pydantic de entrada y salida |
| `settings.py` | Configuración centralizada desde variables de entorno |

Todas las funciones tienen un propósito único y no superan las 30 líneas, con nombres descriptivos (`chunk_document`, `embed_texts`, `search`, `call_llm`, `registrar_metrica`).

## 7. Métricas y detección de respuestas no obtenidas

Toda la infraestructura de captura quedó lista desde el primer sprint, para que el agente evaluador (sección 9) tuviera material sobre el cual trabajar sin instrumentar nada de nuevo:

- **`metrics/metrics.csv`**: una fila por consulta con pregunta, si fue respondida o no, chunk principal recuperado, tokens de entrada/salida, latencia y costo estimado.
- **`logs/queries.jsonl`**: el detalle completo de cada consulta, incluyendo las preguntas que el sistema no pudo responder, con todos los chunks recuperados y la respuesta cruda del LLM.
- **`metrics_writer.was_answered()`**: heurística inicial que detecta cuándo el propio LLM declaró no tener información suficiente.

## 8. Desafíos y mejoras posibles

**Desafío principal:** validar que ambos proveedores de embeddings generen espacios vectoriales de calidad comparable para este dominio específico (trámites administrativos en español), dado que un modelo entrenado mayormente en inglés (como `all-MiniLM-L6-v2`) puede perder matices frente a `text-embedding-3-small`. Se recomienda, como mejora futura, correr un benchmark comparativo con las mismas preguntas de prueba contra ambos proveedores y medir la superposición de los chunks recuperados.

**Otras mejoras posibles:**
- Migrar `vector_store.py` a FAISS si el documento FAQ creciera a cientos o miles de entradas.
- Incorporar caching de consultas frecuentes para reducir costo y latencia.
- Contrastar periódicamente el puntaje del agente evaluador con revisión humana, dado que un LLM como juez no es una verdad absoluta.

## 9. Sprint 2 — Agente evaluador (bonus)

Con la infraestructura de métricas ya lista, el sprint 2 agregó el agente evaluador (`src/evaluator.py`): recibe `user_question`, `system_answer` y `chunks_related`, y devuelve un puntaje de 0 a 10 con una justificación (`schemas.EvaluacionRespuesta`), usando un LLM como juez con un prompt propio (`prompts.SYSTEM_EVALUADOR`).

El criterio de evaluación combina cuatro dimensiones: fidelidad a los chunks (sin inventar datos), cobertura de la información relevante disponible, honestidad ante falta de información (reconocer cuando no se puede responder, en vez de inventar) y claridad para un usuario sin conocimiento previo del trámite.

Se agregó el flag `--evaluate` a `query.py` para evaluar una consulta puntual, y el script `evaluate_batch.py` para evaluar en lote un archivo de ejemplos ya generado. Cada evaluación se registra en `metrics/evaluations.csv` y `logs/evaluations.jsonl`, con el mismo criterio de trazabilidad que las consultas.

Como demostración, se evaluaron los tres ejemplos del sprint 1 (`outputs/sample_queries_evaluated.json`): las dos respuestas correctas puntuaron 9+ por estar bien fundamentadas en los chunks recuperados, y la respuesta de "no tengo información suficiente" ante la pregunta sobre el pasaporte también puntuó alto, porque reconocer la falta de información es el comportamiento esperado, no una falla del sistema.
