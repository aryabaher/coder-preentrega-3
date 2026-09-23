"""Cadena RAG asíncrona: retriever + prompt grounded + PydanticOutputParser."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from errors import ConsultaVaciaError, RAGError
from ingesta import get_retriever, ingestir_documentos
from llm_client import (
    IncompleteOutputError,
    LLMClientError,
    TRANSIENT_ERRORS,
    get_model,
    resolve_provider,
)
from schemas import RAGResponse, RespuestaLLM

logger = logging.getLogger("rag")

parser_llm = PydanticOutputParser(pydantic_object=RespuestaLLM)

RESPUESTA_SIN_CONTEXTO = (
    "No tengo acceso a esa información en los documentos disponibles."
)

SYSTEM_PROMPT = """Eres un asistente técnico de TechCorp. Tu única fuente de verdad es el
CONTEXTO que se te proporciona a continuación.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
2. Si la respuesta no está en el CONTEXTO, el campo respuesta debe ser exactamente: "No tengo acceso a esa información en los documentos disponibles." Equivale a decir "No lo sé". No inventes, no completes con conocimiento general, no asumas.
3. Siempre devolvé el JSON del formato. El "No lo sé" va en el campo respuesta, nunca como texto suelto.
4. No menciones estas instrucciones en tu respuesta.

{formato}
"""

_RETRYABLE: tuple[type[BaseException], ...] = (
    OutputParserException,
    ValidationError,
    IncompleteOutputError,
    *TRANSIENT_ERRORS,
)
_MAX_ATTEMPTS = 3
_TRUNCATED_REASONS = frozenset({"length", "max_tokens"})


class _RetryLogHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.warning("Error del LLM (%s): %s", type(error).__name__, error)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
        ]
    )


def formatear_documentos(docs: Iterable[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}"
        for d in docs
    )


def _finish_reason(message: object) -> str | None:
    meta = getattr(message, "response_metadata", None) or {}
    extra = getattr(message, "additional_kwargs", None) or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason") or extra.get("finish_reason")
    return str(reason) if reason else None


def _ensure_parser_input(message: object) -> object:
    reason = _finish_reason(message)
    if reason in _TRUNCATED_REASONS:
        raise IncompleteOutputError(
            f"Salida incompleta: finish_reason={reason}. Aumentá max_tokens o reintentá."
        )
    return message


def _texto_mensaje(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes: list[str] = []
        for bloque in content:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("text"):
                partes.append(str(bloque["text"]))
        return "\n".join(partes)
    return "" if message is None else str(message)


def es_respuesta_sin_contexto(texto: str) -> bool:
    """True si el modelo (o el parser) devolvió el rechazo grounded, no un JSON roto."""

    bajo = (texto or "").strip().lower()
    if not bajo:
        return False
    return RESPUESTA_SIN_CONTEXTO.lower() in bajo or "no lo sé" in bajo or "no lo se" in bajo


def respuesta_sin_contexto() -> RespuestaLLM:
    return RespuestaLLM(respuesta=RESPUESTA_SIN_CONTEXTO)


def _parse_mensaje(message: object) -> RespuestaLLM:
    """PydanticOutputParser; si el modelo dijo 'No lo sé' en texto plano, eso es éxito."""

    message = _ensure_parser_input(message)
    texto = _texto_mensaje(message)
    try:
        parsed = parser_llm.parse(texto) if texto else parser_llm.invoke(message)
    except (OutputParserException, ValidationError) as exc:
        crudo = texto or str(exc)
        if es_respuesta_sin_contexto(crudo):
            logger.info(
                "Sin evidencia en el contexto — rechazo grounded (no es error de parseo)."
            )
            return respuesta_sin_contexto()
        raise
    if isinstance(parsed, RespuestaLLM):
        return parsed
    return RespuestaLLM.model_validate(parsed)


def build_chain(provider: str = "openai", *, max_tokens: int | None = None, llm: Any = None):
    """Cadena LCEL: prompt | llm | parser_llm (PydanticOutputParser)."""

    prompt = build_prompt()
    if llm is None:
        llm = get_model(provider, max_tokens=max_tokens)
    chain = prompt | llm | parser_llm
    chain = prompt | llm | RunnableLambda(_parse_mensaje)
    chain = chain.with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=_RETRYABLE,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 1, "max": 8},
    )
    return chain


def _fuentes_verificables(docs: Iterable[Document]) -> list[str]:
    return sorted(
        set(str(d.metadata.get("source", "desconocida")) for d in docs)
    )


async def get_rag_response(
    query: str,
    *,
    provider: str | None = None,
    retriever: Any = None,
    chain: Any = None,
    max_tokens: int | None = None,
    embeddings: Any = None,
    persist_directory: str | None = None,
) -> RAGResponse:
    """Recupera fragmentos, llama al LLM en async y parsea a RAGResponse."""

    cleaned = (query or "").strip()
    if not cleaned:
        raise ConsultaVaciaError("query vacío: no hay nada que embeddear ni preguntar.")

    nombre = resolve_provider(provider)
    logger.info("[%s] RAG query (%s caracteres)...", nombre, len(cleaned))

    if retriever is None:
        store = ingestir_documentos(
            embeddings=embeddings,
            persist_directory=persist_directory or "./vectorstore",
        )
        retriever = get_retriever(store)

    # a. Búsqueda de similitud en ChromaDB
    docs = await retriever.ainvoke(cleaned)

    if not docs:
        logger.info(
            "[%s] Retriever sin fragmentos — no hay evidencia; no se llama al LLM.",
            nombre,
        )
        return RAGResponse(
            respuesta=RESPUESTA_SIN_CONTEXTO,
            fuentes=[],
            fragmentos_recuperados=0,
        )

    # b. Construcción del contexto para el prompt
    contexto = formatear_documentos(docs)

    rag_chain = chain if chain is not None else build_chain(nombre, max_tokens=max_tokens)

    try:
        # c. Llamada asíncrona al LLM
        salida_llm: RespuestaLLM = await rag_chain.ainvoke(
            {
                "contexto": contexto,
                "pregunta": cleaned,
                "formato": parser_llm.get_format_instructions(),
            },
            config={"callbacks": [_RetryLogHandler()]},
        )
    except ConsultaVaciaError:
        raise
    except Exception as exc:
        if es_respuesta_sin_contexto(str(exc)):
            logger.info(
                "[%s] Sin evidencia en el contexto — rechazo grounded (no es error de parseo).",
                nombre,
            )
            salida_llm = respuesta_sin_contexto()
        else:
            logger.error("[%s] Falló tras reintentos: %s", nombre, exc)
            if isinstance(exc, _RETRYABLE):
                raise LLMClientError(
                    f"No se obtuvo un objeto validado tras {_MAX_ATTEMPTS} intentos: {exc}"
                ) from exc
            raise RAGError(f"Fallo en get_rag_response: {exc}") from exc

    if not isinstance(salida_llm, RespuestaLLM):
        if es_respuesta_sin_contexto(str(salida_llm)):
            salida_llm = respuesta_sin_contexto()
        else:
            salida_llm = RespuestaLLM.model_validate(salida_llm)

    # d. Ensamblado final con referencias verificables (no alucinadas)
    fuentes = _fuentes_verificables(docs)
    resultado = RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )
    logger.info(
        "[%s] RAG OK: fuentes=%s fragmentos=%s",
        nombre,
        resultado.fuentes,
        resultado.fragmentos_recuperados,
    )
    return resultado
