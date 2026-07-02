# Asistente de FAQ — Trámites de Santa Fe (RAG) — Proyecto Integrador Módulo 2

Chatbot de preguntas frecuentes sobre trámites municipales y provinciales
de Santa Fe. Aplica una arquitectura **RAG** (Retrieval-Augmented Generation):
fragmenta un documento de FAQ en chunks, genera embeddings, indexa los
vectores y responde preguntas del usuario recuperando los chunks más
relevantes antes de generar la respuesta con un LLM. Devuelve siempre
**JSON válido** con `user_question`, `system_answer` y `chunks_related`.
Soporta múltiples proveedores de embeddings (OpenAI / Sentence-Transformers)
y de generación (Anthropic / OpenAI).

---

## Arquitectura

```
Pipeline de datos (build_index.py)
    |
    +-- load_document()        Lee data/faq_document.txt
    +-- chunk_document()       Chunking por párrafo (chunking.py)
    +-- embed_texts()          Embeddings OpenAI o Sentence-Transformers
    +-- save_index()           Guarda chunks + vectores en outputs/index/{dataset}_index.json

Pipeline de consulta (query.py)
    |
    +-- embed_query()          Embedding de la pregunta (mismo proveedor que el índice)
    +-- search()               Búsqueda por similitud coseno (vector_store.py)
    +-- build_prompt()         Ensambla contexto + procedencia (prompts/prompts.py)
    +-- call_llm()             Generación con Anthropic u OpenAI (llm.py)
    +-- registrar_metrica()    Log en metrics/metrics.csv y logs/queries.jsonl
    +-- RespuestaRAG           Validación Pydantic del JSON de salida (schemas.py)
```

---

## Decisiones técnicas

### Estrategia de chunking: por párrafo

El documento FAQ ya está estructurado en bloques de pregunta+respuesta
separados por líneas en blanco. Cada bloque es una unidad semántica
autocontenida: partirlo con una ventana de tamaño fijo rompería la relación
entre la pregunta y su respuesta. Por eso se eligió **chunking por párrafo**,
con dos reglas de seguridad:

- Los párrafos por debajo de 50 tokens (título, introducción) se fusionan
  con el siguiente para no generar chunks demasiado pequeños.
- Si un párrafo superase los 500 tokens, se subdivide con una ventana de
  tamaño fijo y solapamiento de ~40 tokens (fallback, no se activa con el
  documento actual porque ningún bloque supera los 110 tokens).

Con el documento de ejemplo (`data/faq_document.txt`, ~1625 palabras) esta
estrategia genera **27 chunks**, todos entre 62 y 110 tokens.

### Método de búsqueda: similitud coseno exacta (fuerza bruta)

Con 27 chunks, un índice aproximado (ANN/FAISS) no aporta beneficio de
performance y agrega complejidad innecesaria. Se implementó una búsqueda
exacta por producto punto entre vectores normalizados (equivalente a
similitud coseno) usando numpy. Es 100% precisa y evaluar los 27 vectores
toma microsegundos. Si el corpus creciera a decenas de miles de chunks,
el mismo `vector_store.py` podría reemplazarse por un índice FAISS sin
tocar el resto del pipeline, porque `search()` es la única función que
conocen `query.py` y `build_index.py`.

### Embeddings: proveedor intercambiable (OpenAI / Sentence-Transformers)

Controlado por `EMBEDDING_PROVIDER` en `.env`:

| Proveedor | Modelo | Dimensiones | Requiere API key |
|---|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 | Sí |
| `local` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | No (corre en CPU) |

**Importante:** el mismo proveedor usado para indexar el corpus debe
usarse para embeber las consultas. Si cambiás `EMBEDDING_PROVIDER`, hay
que reconstruir el índice con `build_index.py`.

### Generación: proveedor intercambiable (Anthropic / OpenAI)

Controlado por `LLM_PROVIDER` en `.env`. El prompt (few-shot implícito por
las reglas + contexto con procedencia) exige responder únicamente con lo
que dicen los chunks recuperados, y declarar explícitamente cuando no hay
información suficiente — ver más abajo la sección de métricas.

---

## Múltiples bases de FAQ

Además del FAQ original de trámites de Santa Fe, el repositorio incluye otras
bases de ejemplo para probar el sistema con distintos dominios: `data/faq_agn.txt`
(Archivo General de la Nación), `data/faq_yam.txt` (tienda de joyería), `data/faq_hothaus.txt`
(estudio de vidrio) y `data/faq_macro.txt` (Banco Macro). Cada una se indexa por
separado (ver sección "Ejecución"), y `query.py` permite consultar una sola o todas
a la vez con `--dataset all`.

Cuando se consulta con `--dataset all`, la búsqueda se hace contra todos los
índices disponibles en `outputs/index/`, se combinan los resultados por
similitud (deduplicando por `source` + `chunk_id` para no repetir chunks si
hay índices superpuestos) y se usa un prompt de sistema distinto
(`SYSTEM_ASISTENTE_MULTI`) que le pide al modelo identificar explícitamente
de qué organización proviene cada dato usado en la respuesta (por ejemplo:
"Según Banco Macro..."). El campo `source` de cada chunk queda visible en el
JSON de salida para poder auditar la procedencia.

### Detalles de la recuperación (retrieval)

- **Similitud**: cada chunk recuperado trae su `score` de similitud coseno
  (entre -1 y 1; cuanto más alto, más relevante). No es una distancia sino
  una similitud: los valores más altos indican mejor match.
- **Reranking**: no hay una segunda etapa de reranking (cross-encoder o
  LLM-reranker). Es un único paso de embedding + similitud coseno + top-k,
  suficiente para el tamaño de corpus de este proyecto (ver "Método de
  búsqueda" más arriba).
- **Deduplicación**: solo aplica en el modo `--dataset all`, donde se
  descartan chunks repetidos (mismo `source` + `chunk_id`) al fusionar los
  resultados de varios índices. Dentro de un único dataset no hace falta,
  porque cada `chunk_id` es único por construcción.

---

## Interfaz web (Gradio)

Para no depender de la terminal, hay una interfaz web mínima construida con
[Gradio](https://www.gradio.app/):

```bash
uv run python src/app.py
```

Abre un servidor local (por defecto en `http://127.0.0.1:7860`) con:

- Un campo de texto para la **pregunta**.
- Un selector de **modelo/fuente**: cada FAQ indexado individualmente, o
  "Todos los FAQ" para buscar en todas las bases a la vez.
- Un checkbox opcional para correr el **agente evaluador** sobre la respuesta.
- El resultado, mostrando la respuesta generada y el detalle de los chunks
  usados (fuente, similitud y fragmento de texto).

Requiere haber generado al menos un índice antes con `build_index.py`.

---

## Requisitos

- Python >= 3.11
- API key del proveedor de embeddings elegido (si es `openai`)
- API key del proveedor de generación elegido (Anthropic u OpenAI)
- Gestor de dependencias: [uv](https://docs.astral.sh/uv/) (recomendado) o `pip` con un entorno virtual estándar (ver "Instalación" más abajo)

---

## Configuración

Copiá la plantilla de variables de entorno y completá tus API keys:

```bash
cp .env.example .env
# Editar .env: elegir EMBEDDING_PROVIDER, LLM_PROVIDER y completar las API keys
```

Como alternativa a editar `.env`, también podés exportar las variables directamente
en la terminal antes de ejecutar los scripts (útil para pruebas rápidas):

```bash
# Linux / macOS
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY = "your-key-here"
$env:ANTHROPIC_API_KEY = "your-key-here"
```

---

## Instalación

### Opción A — con `uv` (recomendada)

```bash
uv sync
```

Esto crea el entorno virtual e instala todas las dependencias declaradas en
`pyproject.toml`, incluyendo las de test.

### Opción B — con `pip` y un entorno virtual estándar

```bash
python -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows (PowerShell o CMD)

pip install -r requirements.txt
```

> **Nota:** si usás la Opción B, reemplazá `uv run python` por `python` en todos
> los comandos de las secciones siguientes (por ejemplo, `python src/build_index.py`
> en vez de `uv run python src/build_index.py`).

---

## Ejecución

```bash
# 1. Construir el índice (chunking + embeddings) para el FAQ por defecto
uv run python src/build_index.py

# 2. Consultar el FAQ
uv run python src/query.py -q "¿Cómo saco turno para renovar la licencia de conducir?"
```

### Indexar y consultar varias bases de FAQ

```bash
# Indexar cada base por separado (una vez, o cada vez que cambien los .txt)
uv run python src/build_index.py --input data/faq_document.txt
uv run python src/build_index.py --input data/faq_agn.txt
uv run python src/build_index.py --input data/faq_yam.txt
uv run python src/build_index.py --input data/faq_hothaus.txt
uv run python src/build_index.py --input data/faq_macro.txt

# Consultar una base puntual
uv run python src/query.py -q "¿Cómo pago mi tarjeta de crédito?" --dataset faq_macro

# Consultar TODAS las bases a la vez
uv run python src/query.py -q "¿Cómo pago mi tarjeta de crédito?" --dataset all
```

### Interfaz web

```bash
uv run python src/app.py
```

Ver detalles en la sección "Interfaz web (Gradio)" más arriba.

### Salida de ejemplo

```json
{
  "user_question": "¿Cómo saco turno para renovar la licencia de conducir?",
  "system_answer": "Podés sacar el turno de forma online desde la sección de turnos del sitio municipal...",
  "chunks_related": [
    {"chunk_id": "chunk_005", "score": 0.78, "text": "5. ¿Qué diferencia hay...", "source": "faq_document.txt"}
  ]
}
```

Más ejemplos en `outputs/sample_queries.json`.

---

## Métricas y logging (base para el agente evaluador — sprint 2)

Cada consulta queda registrada en dos archivos, pensados para que el
**agente evaluador** (bonus, sprint 2) tenga todo el material necesario
sin tener que instrumentar nada de nuevo:

- `metrics/metrics.csv`: una fila por consulta (pregunta, si se respondió
  o no, chunk principal recuperado, tokens, latencia, costo, proveedor).
- `logs/queries.jsonl`: un JSON por línea con el detalle completo de cada
  consulta, incluyendo las preguntas que el sistema **no pudo responder**
  (`answered: false`), para poder auditarlas o pasárselas al evaluador.

La detección de "no respondida" (`metrics_writer.was_answered`) es una
heurística simple basada en frases del propio LLM ("no tengo información
suficiente..."). El agente evaluador del sprint 2 (ver más abajo) es el
criterio de calidad más robusto que complementa esta heurística.

---

## Sprint 2 — Agente evaluador (bonus)

El agente evaluador (`src/evaluator.py`) recibe `user_question`,
`system_answer` y `chunks_related`, y devuelve un puntaje de 0 a 10 con
una justificación (`schemas.EvaluacionRespuesta`), usando un LLM como juez
con un prompt propio (`prompts.SYSTEM_EVALUADOR`) que evalúa:

1. **Fidelidad**: si la respuesta usa solo información de los chunks.
2. **Cobertura**: si aprovecha la información relevante disponible.
3. **Honestidad**: si reconoce cuando no hay información suficiente,
   en vez de inventar una respuesta.
4. **Claridad**: si es comprensible para alguien sin jerga administrativa.

### Uso

```bash
# Evaluar una consulta puntual junto con la respuesta
uv run python src/query.py -q "¿Cómo renuevo la licencia?" --evaluate

# Evaluar en lote un archivo de ejemplos ya generado
uv run python src/evaluate_batch.py --input outputs/sample_queries.json
```

Cada evaluación se registra en `metrics/evaluations.csv` y
`logs/evaluations.jsonl`, con el mismo criterio de trazabilidad usado para
las consultas (timestamp, tokens, latencia, costo).

`outputs/sample_queries_evaluated.json` contiene los 3 ejemplos del sprint
1 ya evaluados, como demostración del criterio del agente: las dos
respuestas correctas puntúan 9+ por estar bien fundamentadas en los
chunks, y la respuesta "no pude responder" ante la pregunta sobre el
pasaporte también puntúa alto, porque reconocer la falta de información
es el comportamiento correcto, no una falla.

---

## Tests

```bash
uv run pytest tests/ -v
```

Los tests no consumen tokens ni requieren red (no llaman a ninguna API):

| Suite | Qué cubre |
|---|---|
| TestChunking | Rango de tokens por chunk, 20+ chunks sobre el documento real |
| TestVectorStore | Orden por similitud, respeto de top_k, índice vacío |
| TestSchemas | Validación Pydantic de `RespuestaRAG` |
| TestMetrics | Detección de "no respondida", cálculo de costo |
| TestEvaluator | Construcción del prompt, parseo de JSON, validación de rango 0-10 |

---

## Estructura del repositorio

```
m2p1-faq-santafe/
├── data/
│   ├── faq_document.txt        FAQ de trámites de Santa Fe (~1625 palabras)
│   ├── faq_agn.txt             FAQ del Archivo General de la Nación
│   ├── faq_yam.txt             FAQ de Yam (joyería, envíos y devoluciones)
│   ├── faq_hothaus.txt         FAQ de Hothaus (estudio de vidrio, Australia)
│   └── faq_macro.txt           FAQ de Banco Macro (operaciones y canales)
├── src/
│   ├── chunking.py             Chunking por párrafo con reglas de tamaño
│   ├── embeddings.py           Embeddings OpenAI / Sentence-Transformers
│   ├── vector_store.py         Índice en memoria + búsqueda coseno
│   ├── build_index.py          Pipeline de datos (entry point, un índice por FAQ)
│   ├── query.py                Pipeline de consulta (entry point, --dataset / all / --evaluate)
│   ├── app.py                  Interfaz web simple (Gradio)
│   ├── llm.py                  Llamada al LLM (Anthropic / OpenAI)
│   ├── evaluator.py            Agente evaluador (bonus, sprint 2)
│   ├── evaluate_batch.py       Evalúa en lote un archivo de ejemplos (entry point)
│   ├── prompts_loader.py       Carga los prompts desde prompts/
│   ├── metrics_writer.py       Métricas, logging y evaluaciones
│   ├── schemas.py              Contratos Pydantic
│   └── settings.py             Configuración desde .env
├── prompts/
│   └── prompts.py               Prompts del sistema: asistente (único y multi-fuente) y evaluador
├── outputs/
│   ├── index/{dataset}_index.json    Un índice por FAQ, generado por build_index.py
│   ├── sample_queries.json     3+ ejemplos de consulta-respuesta (sprint 1)
│   └── sample_queries_evaluated.json  Mismos ejemplos + veredicto del evaluador
├── metrics/
│   ├── metrics.csv             Registro agregado de consultas
│   └── evaluations.csv         Registro agregado de evaluaciones (sprint 2)
├── logs/
│   ├── queries.jsonl           Detalle completo por consulta
│   └── evaluations.jsonl       Detalle completo por evaluación (sprint 2)
├── tests/
│   └── test_core.py            Suite de tests (sin LLM ni red)
├── reports/
│   └── PI_report.md            Informe técnico del proyecto
├── pyproject.toml              Dependencias
├── .env.example                Plantilla de variables de entorno
└── README.md
```

---

## Limitaciones conocidas

- El índice vectorial es un archivo JSON en memoria: para corpus grandes
  (decenas de miles de chunks) conviene migrar a FAISS u otra base
  vectorial especializada, sin cambiar la interfaz de `vector_store.py`.
- El agente evaluador (sprint 2) usa un LLM como juez: su puntaje es una
  señal de calidad adicional, no una verdad absoluta. Para un caso de uso
  crítico conviene contrastarlo periódicamente con revisión humana.
- `outputs/sample_queries.json` y `outputs/sample_queries_evaluated.json`
  incluidos en este repositorio fueron generados y verificados manualmente
  durante el desarrollo (el entorno de build no tenía acceso de red a las
  APIs de embeddings/LLM); al ejecutar `query.py` o `evaluate_batch.py`
  con tus propias credenciales, ambos archivos se pueden regenerar con
  respuestas y evaluaciones en vivo.
