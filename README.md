# RAG local: políticas de TechCorp

El sistema recibe una pregunta, busca los fragmentos más cercanos en una colección ChromaDB poblada desde `/data` y genera una respuesta **solo** con ese contexto. Si el dato no está en los documentos, no completa con conocimiento general: dice que no tiene acceso a esa información.

## Cómo ejecutarlo

Usá el bloque de **Windows (PowerShell)** o el de **Linux/macOS (bash/zsh)** según tu sistema. Los pasos 3 y 4 son los mismos en ambos, una vez activado el venv.

1. Entorno virtual e instalación (`langchain`, `chromadb`, `openai`, `pydantic`, más los paquetes de embeddings y Chroma). Tests: `pytest`, `pytest-asyncio`, `pytest-mock`, `respx`. Un `pip install -r requirements.txt` alcanza.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea `Activate.ps1`, el `Set-ExecutionPolicy` de arriba vale solo para esa sesión. Alternativa sin activar: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` y después `.\.venv\Scripts\python.exe main.py`.

**Linux/macOS (bash/zsh):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Variables de entorno: copiá `.env.example` a `.env` y completá `OPENAI_API_KEY` (proveedor por defecto).

**Windows (PowerShell):**

```powershell
copy .env.example .env
```

**Linux/macOS (bash/zsh):**

```bash
cp .env.example .env
```

3. Ingesta de `/data` (chunking + ChromaDB en `./vectorstore`). Si el índice ya existe, no reindexa.

```
python ingesta.py --offline
python ingesta.py --offline --force
```

Sin `--offline` usa `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` para indexar y para consultar. La primera corrida puede bajar el modelo.

4. Cadena RAG asíncrona: pregunta cuya respuesta está en los documentos y pregunta trampa.

```
python main.py --offline --provider openai
python main.py --provider openai
python main.py --interactive
python main.py --max-tokens 16
```

`--offline` evita bajar sentence-transformers (embeddings deterministas). El LLM sigue necesitando la API key, salvo que solo corras el chequeo offline:

```
python validacion.py
python -m pytest -v
```

`pytest` cubre el esquema Pydantic, `get_rag_response`, la ingesta, keys faltantes, 429 y `finish_reason=length` con mocks: no gasta cuota.

Deberías ver un `RAGResponse` con `respuesta`, `fuentes` y `fragmentos_recuperados`. En la pregunta trampa, la respuesta es que no hay acceso a esa información.

## Ejemplo de salida

Pregunta en los documentos:

> ¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?

```json
{
  "respuesta": "Un empleado con 5 años de antigüedad accede a 21 días corridos de vacaciones.",
  "fuentes": ["politica_vacaciones.txt"],
  "fragmentos_recuperados": 4
}
```

Pregunta trampa (no está en `/data`):

> ¿Cuál es la política de bonos por rendimiento anual en TechCorp?

```json
{
  "respuesta": "No tengo acceso a esa información en los documentos disponibles.",
  "fuentes": ["politica_vacaciones.txt"],
  "fragmentos_recuperados": 4
}
```

Eso es un `RAGResponse` válido, no un error. Chroma igual devuelve los `k` vecinos más cercanos (aunque no hablen de bonos); el modelo, al no ver el dato en el CONTEXTO, llena `respuesta` con esa frase. No reintenta ni cae en parseo.

Las `fuentes` las arma el código a partir de los metadatos reales de Chroma, no el LLM.

## Archivos del repositorio

| Artefacto | Dónde está |
|-----------|------------|
| Dataset de ejemplo (`.txt`) | `data/` (`politica_vacaciones.txt`, `politica_teletrabajo.txt`, `politica_seguridad_informatica.txt`, `onboarding_nuevos_empleados.txt`) |
| Script de ingesta (`DirectoryLoader`, `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, `chunk_size=500`, `chunk_overlap=70`, `Chroma.from_documents`) | `ingesta.py` |
| Persistencia local | `./vectorstore` · colección `techcorp_policies` |
| Embeddings (mismo modelo al indexar y al consultar) | `embeddings.py` → `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` |
| Retriever `as_retriever(search_type="similarity", search_kwargs={"k": 4})` | `ingesta.py` → `get_retriever()` |
| `RespuestaLLM` + `RAGResponse` (`respuesta` + `fuentes`) | `schemas.py` |
| `PydanticOutputParser(pydantic_object=RespuestaLLM)` y `async def get_rag_response(query: str)` | `chain.py` |
| Cadena LCEL `prompt \| llm \| parser_llm` | `chain.py` → `build_chain()` |
| `get_model()` → `ChatGoogleGenerativeAI` / `ChatOpenAI` / `ChatAnthropic` | `llm_client/models.py` |
| Mini-script (pregunta OK + pregunta trampa) | `main.py` |
| Tests (`conftest.py`, `test_models.py`, `test_retries.py`, `test_ingesta.py`, `test_rag.py`) | `tests/` |
| `pytest.ini` | `testpaths = tests` · `asyncio_mode = auto` · `addopts = -ra -q` |
| `requirements-dev.txt` | `pytest`, `pytest-asyncio`, `pytest-mock`, `respx` |
| `.env.example` | `.env.example` |
| `requirements.txt` | `langchain`, `chromadb`, `openai`, `pydantic`, `langchain-chroma`, `langchain-huggingface`, `tiktoken`, `pytest`, `pytest-asyncio`, `pytest-mock`, `respx` |

No commitees `.env` ni `vectorstore/`. El repo solo versiona `.env.example` y `/data`.

## Cómo se cubre la consigna

| Requisito | Cómo se cumple | Evidencia |
|----------|----------------|-----------|
| Ingesta y chunking | `DirectoryLoader` lee `/data`. `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=70)` (piso de overlap: 50). | `ingesta.py` · `test_ingesta.py` · `evidencias/01-validacion-offline.txt` |
| ChromaDB persistente | Cliente en `./vectorstore`. Si el índice existe, no reindexa. Mismo embedding para indexar y consultar. | `ingesta.py` → `ya_existe_indice` · `test_ingesta_persiste_y_no_reindexa` |
| Retriever | `vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})` | `get_retriever()` · `evidencias/02-ingesta-offline.txt` |
| Generación grounded | Prompt de filtro de veracidad. `chain = prompt \| llm \| parser_llm`. Si no está en el CONTEXTO: "No tengo acceso…" / "No lo sé". | `chain.py` · `SYSTEM_PROMPT` |
| `get_rag_response` async | `await retriever.ainvoke` + `await chain.ainvoke` + parseo a `RAGResponse` con `fuentes`. | `chain.py` · `tests/test_rag.py` · `evidencias/05-pytest.txt` |
| Dos pruebas | Pregunta de vacaciones (en documentos) y pregunta de bonos (trampa). | `main.py` · `test_pregunta_en_documentos` · `test_pregunta_trampa_no_alucina` |
| Credenciales | Keys solo en `.env`. El repo trae `.env.example`. | `.gitignore` · `.env.example` |

## Cómo está armada la cadena

| Pieza | Dónde | Qué hace |
|-------|--------|----------|
| Dataset | `data/` | Cuatro políticas de TechCorp (vacaciones, teletrabajo, seguridad, onboarding). No hay política de bonos. |
| Splitter | `ingesta.py` | Tokens con tiktoken, overlap 70 (mínimo pedido: 50), separadores de párrafo/oración. |
| Vectorstore | `langchain_chroma.Chroma` | `persist_directory="./vectorstore"`, `collection_name="techcorp_policies"`. |
| Retriever | `get_retriever` | top_k = 4. Fuera de 3–5 se rechaza (contexto infinito). |
| Prompt | `ChatPromptTemplate` | Variables `{contexto}`, `{pregunta}`, `{formato}`. |
| Parser | `PydanticOutputParser` | Solo `RespuestaLLM.respuesta`. Las fuentes salen de `metadata["source"]`. |
| Ejecución | `get_rag_response` | Async. Reintenta 3 veces ante 429, red o JSON truncado. |

## Variables de entorno

| Variable | Obligatorio | Para qué |
|----------|-------------|----------|
| `LLM_PROVIDER` | No (default `openai`) | `openai`, `anthropic` o `gemini` |
| `OPENAI_API_KEY` | Si usás OpenAI (default) | Key de la API |
| `OPENAI_MODEL` | No (`gpt-4o-mini`) | Modelo OpenAI |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | Key de la API |
| `ANTHROPIC_MODEL` | No (`claude-sonnet-4-6`) | Modelo Anthropic |
| `GOOGLE_API_KEY` | Si usás Gemini | Key de AI Studio (también acepta `GEMINI_API_KEY`) |
| `GEMINI_MODEL` | No (`gemini-flash-latest`) | Modelo Gemini |
| `LLM_TIMEOUT` | No (`30`) | Timeout HTTP en segundos |

## Evidencias

| Archivo | Qué demuestra |
|---------|---------------|
| `evidencias/01-validacion-offline.txt` | Chunking 500/70, índice Chroma, `get_rag_response` async, `PydanticOutputParser`, `finish_reason=length`. |
| `evidencias/02-ingesta-offline.txt` | `python ingesta.py --offline --force`: fragmentos y retriever k=4. |
| `evidencias/03-openai-rag.txt` | `python main.py --provider openai`: pregunta en documentos (21 días) y pregunta trampa (no alucina). |
| `evidencias/04-anthropic-rag.txt` | `python main.py --provider anthropic`: mismo par de preguntas, `ChatAnthropic`. |
| `evidencias/05-pytest.txt` | `python -m pytest -v`: factory, keys, 429, truncamiento, ingesta y RAG con mocks. |
| `evidencias/06-error-consulta-vacia.txt` | `get_rag_response("  ")` → `ConsultaVaciaError`. |
| `evidencias/07-error-api-key.txt` | Sin `OPENAI_API_KEY` → `Falta OPENAI_API_KEY.` |
| `evidencias/08-error-max-tokens.txt` | `python main.py --provider openai --max-tokens 16`: `finish_reason=length` tras 3 reintentos. |
| `evidencias/09-error-chunk-size.txt` | `chunk_size=200` → `IngestaError`. |
| `evidencias/10-error-429-pytest.txt` | Mock de `GoogleRateLimitError` / 429 (sin pegarle a la API). |

Salida de `python -m pytest -v`:

```
tests/test_ingesta.py::test_cargar_documentos_data PASSED
tests/test_ingesta.py::test_chunk_size_minimo_500 PASSED
tests/test_ingesta.py::test_chunk_size_menor_a_500_falla PASSED
tests/test_ingesta.py::test_overlap_menor_a_50_falla PASSED
tests/test_ingesta.py::test_fragmentar_produce_varios_chunks PASSED
tests/test_ingesta.py::test_ingesta_persiste_y_no_reindexa PASSED
tests/test_ingesta.py::test_retriever_k_4 PASSED
tests/test_ingesta.py::test_data_vacia PASSED
tests/test_ingesta.py::test_top_k_fuera_de_rango PASSED
tests/test_models.py::test_build_model_creates_chat_openai PASSED
tests/test_models.py::test_missing_api_key_raises PASSED
tests/test_models.py::test_invalid_provider_raises PASSED
tests/test_models.py::test_missing_anthropic_y_openai_keys PASSED
tests/test_models.py::test_get_model_anthropic_y_gemini PASSED
tests/test_models.py::test_respuesta_vacia PASSED
tests/test_models.py::test_esquema_valido PASSED
tests/test_rag.py::test_pregunta_en_documentos PASSED
tests/test_rag.py::test_pregunta_trampa_no_alucina PASSED
tests/test_rag.py::test_pregunta_trampa_texto_plano_no_es_error_de_parseo PASSED
tests/test_rag.py::test_retriever_vacio_no_llama_al_llm PASSED
tests/test_rag.py::test_parse_mensaje_acepta_rechazo_en_texto_plano PASSED
tests/test_rag.py::test_formatear_documentos_incluye_fuente PASSED
tests/test_rag.py::test_parser_llm_es_pydantic_output_parser PASSED
tests/test_retries.py::test_transient_errors_are_retryable PASSED
tests/test_retries.py::test_chain_retries_on_transient_error PASSED
tests/test_retries.py::test_non_transient_does_not_retry PASSED
tests/test_retries.py::test_truncation_is_retryable PASSED
tests/test_retries.py::test_finish_reason_length_no_transforma PASSED
tests/test_retries.py::test_process_query_vacio PASSED
tests/test_retries.py::test_get_rag_response_429_queda_controlado PASSED
tests/test_retries.py::test_get_rag_response_incompleto_tras_retry PASSED
============================= 31 passed in 7.09s ==============================
```

## Manejo de errores personalizados

`main.py` e `ingesta.py` capturan `RAGError`, `LLMClientError` y `ValueError` como **Error controlado** y no se caen.

### Pregunta fuera de contexto (no es error)

Si el dato no está en `/data` (pregunta trampa de bonos), `get_rag_response` devuelve `RAGResponse` con `respuesta` = `No tengo acceso a esa información en los documentos disponibles.` Si el modelo escribe esa frase en texto plano en vez de JSON, el parser la acepta: no reintenta 3 veces ni lanza `Invalid json output`. Tests: `test_pregunta_trampa_no_alucina`, `test_pregunta_trampa_texto_plano_no_es_error_de_parseo`, `test_retriever_vacio_no_llama_al_llm`.

### Consulta vacía

`get_rag_response("   ")` lanza `ConsultaVaciaError` antes de embeddear.

```
Error controlado: query vacío: no hay nada que embeddear ni preguntar.
```

Cómo reproducirlo: `python -c "import asyncio; from chain import get_rag_response; asyncio.run(get_rag_response('  '))"` — o el test `test_process_query_vacio`.

### Falta de API key

`get_model("openai")` / `get_model("gemini")` lanzan `LLMClientError` (`Falta OPENAI_API_KEY.` / `Falta GOOGLE_API_KEY.`) sin llamar a la red.

```
Error controlado: Falta OPENAI_API_KEY.
```

Cubierto por `test_missing_api_key_raises`.

### Rate limit / cuota (429)

Errores transitorios (`GoogleRateLimitError`, `RateLimitError`, red). La cadena reintenta 3 veces. Si persiste:

```
Error controlado: No se obtuvo un objeto validado tras 3 intentos: 429 RESOURCE_EXHAUSTED
```

El test `test_get_rag_response_429_queda_controlado` mockea `GoogleRateLimitError` y exige el mensaje de 3 intentos.

### Respuesta truncada (`finish_reason=length`)

`_ensure_parser_input` lanza `IncompleteOutputError` **antes** de parsear un objeto incompleto. `.with_retry()` reintenta. Si sigue cortada:

```
Error controlado: No se obtuvo un objeto validado tras 3 intentos: Salida incompleta: finish_reason=length. Aumentá max_tokens o reintentá.
```

Cómo reproducirlo: `python main.py --provider openai --max-tokens 16`. Tests: `test_finish_reason_length_no_transforma` y `test_get_rag_response_incompleto_tras_retry`.

### Carpeta /data vacía o chunking inválido

Sin `.txt`/`.md` en `/data`, o con `chunk_size` menor a 500 / overlap menor a 50:

```
Error controlado: La carpeta ... no tiene archivos .txt o .md para indexar.
Error controlado: chunk_size debe ser >= 500 tokens.
```

Tests: `test_data_vacia`, `test_chunk_size_menor_a_500_falla`, `test_overlap_menor_a_50_falla`.

### Esquema inválido (antes del LLM)

`RespuestaLLM(respuesta="   ")` falla en Pydantic (`ValidationError`) sin llamar a la API. Test: `test_respuesta_vacia`.

## Checklist de verificación

- [x] Script de ingesta: `/data` → `RecursiveCharacterTextSplitter` (500 tokens, overlap ≥ 50) → ChromaDB en `./vectorstore`
- [x] Mismo modelo de embeddings para indexar y consultar
- [x] No reindexa si el índice ya existe
- [x] Retriever con `k` entre 3 y 5
- [x] `async def get_rag_response(query: str)` + `.ainvoke()`
- [x] Cadena LCEL con `PydanticOutputParser` y `RAGResponse` (texto + fuentes)
- [x] Prompt grounded: si no está en el CONTEXTO, "No tengo acceso…" / "No lo sé"
- [x] Dataset de ejemplo en `data/` y mini-script `main.py` (pregunta OK + trampa)
- [x] `python validacion.py` y `python -m pytest -v` en verde, con mocks (sin API real)
- [x] README con ejemplos específicos de query vacía, key faltante, 429, truncado, `/data` vacía y schema inválido
- [x] Keys solo en `.env` (no versionado)
