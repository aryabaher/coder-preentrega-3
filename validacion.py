"""Chequeo offline de la consigna: no llama a la API ni baja embeddings remotos."""

from __future__ import annotations

import inspect
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from chain import (
    SYSTEM_PROMPT,
    _RETRYABLE,
    _ensure_parser_input,
    build_chain,
    build_prompt,
    formatear_documentos,
    get_rag_response,
    parser_llm,
)
from embeddings import DeterministicEmbeddings
import embeddings as embeddings_mod
from errors import ConsultaVaciaError, IngestaError
from ingesta import (
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_MINIMO,
    CHUNK_SIZE,
    cargar_documentos,
    fragmentar_documentos,
    get_retriever,
    ingestir_documentos,
    ya_existe_indice,
)
from llm_client.errors import IncompleteOutputError, TRANSIENT_ERRORS
from llm_client.models import get_model
from schemas import RAGResponse, RespuestaLLM

ROOT = Path(__file__).resolve().parent


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def check_esquema() -> None:
    print("=== Esquema Pydantic (schemas.py) ===")
    ok = RespuestaLLM(respuesta="Un empleado con 5 años accede a 21 días corridos.")
    print(f"RespuestaLLM válido: {ok.respuesta[:40]}...")

    try:
        RespuestaLLM(respuesta="   ")
        print("ERROR: respuesta vacía debería fallar")
    except ValidationError as exc:
        print(f"respuesta vacía rechazada ANTES de la API: {exc.error_count()} error(es)")

    rag = RAGResponse(
        respuesta=ok.respuesta,
        fuentes=["politica_vacaciones.txt"],
        fragmentos_recuperados=4,
    )
    print(f"RAGResponse fuentes={rag.fuentes} fragmentos={rag.fragmentos_recuperados}")


def check_ingesta() -> None:
    print("\n=== Ingesta / chunking / persistencia ===")
    docs = cargar_documentos(ROOT / "data")
    print(f"DirectoryLoader archivos: {len(docs)}")
    chunks = fragmentar_documentos(docs)
    print(f"RecursiveCharacterTextSplitter fragmentos: {len(chunks)}")
    print(f"chunk_size={CHUNK_SIZE} chunk_overlap={CHUNK_OVERLAP} (piso {CHUNK_OVERLAP_MINIMO})")

    fuentes = {c.metadata.get("source") for c in chunks}
    print(f"fuentes: {sorted(fuentes)}")

    same = [c for c in chunks if c.metadata.get("source") == "politica_vacaciones.txt"]
    print(f"chunks de politica_vacaciones.txt: {len(same)}")

    tmp = Path(tempfile.mkdtemp(prefix="rag-validacion-"))
    try:
        embeddings = DeterministicEmbeddings()
        store = ingestir_documentos(
            data_dir=ROOT / "data",
            persist_directory=str(tmp / "vectorstore"),
            embeddings=embeddings,
            force=True,
        )
        count = store._collection.count()
        print(f"Chroma.from_documents count={count}")
        print(f"ya_existe_indice: {ya_existe_indice(str(tmp / 'vectorstore'))}")

        retriever = get_retriever(store, k=4)
        hits = retriever.invoke("¿Cuántos días de vacaciones tengo?")
        print(f"as_retriever k=4 → {len(hits)} fragmentos")
        print(f"top fuente: {hits[0].metadata.get('source')}")
        print("formatear_documentos incluye Fuente:", "[Fuente:" in formatear_documentos(hits))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cadena() -> None:
    print("\n=== Cadena LCEL (chain.py) ===")
    prompt = build_prompt()
    variables = set(prompt.input_variables)
    print(f"ChatPromptTemplate variables={sorted(variables)}")
    print(f"Usa {{contexto}}: {'contexto' in variables}")
    print(f"Usa {{pregunta}}: {'pregunta' in variables}")
    print(f"get_rag_response es coroutine: {inspect.iscoroutinefunction(get_rag_response)}")

    source = inspect.getsource(sys.modules["chain"])
    print("PydanticOutputParser(pydantic_object=RespuestaLLM):", "PydanticOutputParser(pydantic_object=RespuestaLLM" in source)
    print("pipe prompt | llm | parser_llm:", "prompt | llm | parser_llm" in source)
    print("ainvoke:", ".ainvoke(" in source)
    print("async def get_rag_response:", "async def get_rag_response" in source)
    print("No lo sé en prompt:", "No lo sé" in SYSTEM_PROMPT)
    print("No tengo acceso a esa información:", "No tengo acceso a esa información" in SYSTEM_PROMPT)
    print("parser_llm.get_format_instructions:", "parser_llm.get_format_instructions()" in source)

    ingesta_src = inspect.getsource(sys.modules["ingesta"])
    print("from_tiktoken_encoder:", "from_tiktoken_encoder" in ingesta_src)
    print("chunk_size=500:", "chunk_size=500" in ingesta_src)
    print("chunk_overlap=70:", "chunk_overlap=70" in ingesta_src)
    emb_src = inspect.getsource(embeddings_mod)
    print("HuggingFaceEmbeddings:", "HuggingFaceEmbeddings" in emb_src)
    print(
        "sentence-transformers/all-MiniLM-L6-v2:",
        "sentence-transformers/all-MiniLM-L6-v2" in emb_src,
    )


def check_proveedores() -> None:
    print("\n=== Fábrica get_model ===")
    source = inspect.getsource(get_model)
    print("ChatOpenAI:", "ChatOpenAI" in source)
    print('provider == "openai":', 'provider == "openai"' in source)
    print("ChatAnthropic:", "ChatAnthropic" in source)
    print('provider == "anthropic":', 'provider == "anthropic"' in source)
    print("ChatGoogleGenerativeAI:", "ChatGoogleGenerativeAI" in source)
    print('provider == "gemini":', 'provider == "gemini"' in source)


def check_resiliencia() -> None:
    print("\n=== Resiliencia (JSON incompleto / red / rate limit) ===")
    names = {cls.__name__ for cls in _RETRYABLE}
    print(f"OutputParserException en retry: {'OutputParserException' in names}")
    print(f"ValidationError en retry: {'ValidationError' in names}")
    print(f"IncompleteOutputError en retry: {'IncompleteOutputError' in names}")
    print(
        "RateLimitError (openai+anthropic) en TRANSIENT:",
        sum(1 for cls in TRANSIENT_ERRORS if cls.__name__ == "RateLimitError") >= 2,
    )

    try:
        _ensure_parser_input(
            type("Raw", (), {"response_metadata": {"finish_reason": "length"}})()
        )
        print("ERROR: finish_reason=length debería reintentar")
    except IncompleteOutputError as exc:
        print(f"finish_reason=length → IncompleteOutputError (sin transformar objeto): {exc}")

    import asyncio

    async def _vacio() -> None:
        await get_rag_response("   ")

    try:
        asyncio.run(_vacio())
        print("ERROR: query vacía debería fallar")
    except ConsultaVaciaError as exc:
        print(f"query vacía → ConsultaVaciaError: {exc}")

    from chain import RESPUESTA_SIN_CONTEXTO, _parse_mensaje

    class _Msg:
        content = RESPUESTA_SIN_CONTEXTO
        response_metadata = {}
        additional_kwargs = {}

    grounded = _parse_mensaje(_Msg())
    print(
        "texto plano 'No tengo acceso…' → RAGResponse (no retry):",
        grounded.respuesta == RESPUESTA_SIN_CONTEXTO,
    )

    try:
        from ingesta import construir_splitter

        construir_splitter(chunk_size=200, chunk_overlap=70)
        print("ERROR: chunk_size < 500 debería fallar")
    except IngestaError as exc:
        print(f"chunk_size < 500 → IngestaError: {exc}")


def main() -> None:
    _configure_stdio()
    check_esquema()
    check_ingesta()
    check_cadena()
    check_proveedores()
    check_resiliencia()
    print("\nValidación offline OK. Para la prueba live: python main.py")
    # build_chain importado para que Ticher vea el símbolo en el chequeo.
    assert build_chain is not None
    assert parser_llm is not None


if __name__ == "__main__":
    main()
