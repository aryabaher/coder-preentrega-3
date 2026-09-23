"""Reintentos: transitorio recupera, permanente no reintenta, truncamiento es transitorio."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from openai import RateLimitError

from chain import _ensure_parser_input, _RETRYABLE, get_rag_response
from errors import ConsultaVaciaError
from llm_client.errors import IncompleteOutputError, LLMClientError, TRANSIENT_ERRORS


def _rate_limit() -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "rate limit"}})
    return RateLimitError("rate limit", response=response, body=None)


def _retrying(fn):
    return RunnableLambda(fn).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(
            RateLimitError,
            IncompleteOutputError,
            OutputParserException,
            ConnectionError,
        ),
        wait_exponential_jitter=False,
    )


def test_transient_errors_are_retryable():
    assert RateLimitError in TRANSIENT_ERRORS
    assert ConnectionError in TRANSIENT_ERRORS
    assert IncompleteOutputError in TRANSIENT_ERRORS
    assert isinstance(_rate_limit(), _RETRYABLE) or type(_rate_limit()) in TRANSIENT_ERRORS


def test_chain_retries_on_transient_error():
    fake_model = MagicMock()
    fake_model.invoke.side_effect = [
        _rate_limit(),
        _rate_limit(),
        {"text": "ok"},
    ]
    chain = _retrying(lambda _: fake_model.invoke({}))
    assert chain.invoke({}) == {"text": "ok"}
    assert fake_model.invoke.call_count == 3


def test_non_transient_does_not_retry():
    calls = {"n": 0}

    def boom(_):
        calls["n"] += 1
        raise LLMClientError("Falta GOOGLE_API_KEY.")

    chain = _retrying(boom)
    with pytest.raises(LLMClientError, match="GOOGLE_API_KEY"):
        chain.invoke({})
    assert calls["n"] == 1


def test_truncation_is_retryable():
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] < 2:
            raise IncompleteOutputError("finish_reason=length")
        return {"text": "ok"}

    chain = _retrying(flaky)
    assert chain.invoke({}) == {"text": "ok"}
    assert calls["n"] == 2


def test_finish_reason_length_no_transforma() -> None:
    with pytest.raises(IncompleteOutputError, match="finish_reason=length"):
        _ensure_parser_input(
            type("Raw", (), {"response_metadata": {"finish_reason": "length"}})()
        )


@pytest.mark.asyncio
async def test_process_query_vacio() -> None:
    with pytest.raises(ConsultaVaciaError, match="query vacío"):
        await get_rag_response("   ")


@pytest.mark.asyncio
async def test_get_rag_response_429_queda_controlado(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_google_genai.chat_models import GoogleRateLimitError

    class Retriever:
        async def ainvoke(self, *args: object, **kwargs: object):
            from langchain_core.documents import Document

            return [Document(page_content="vacaciones 21 días", metadata={"source": "politica_vacaciones.txt"})]

    class Boom:
        async def ainvoke(self, *args: object, **kwargs: object) -> None:
            raise GoogleRateLimitError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(LLMClientError, match="3 intentos"):
        await get_rag_response(
            "¿Cuántos días de vacaciones corresponden?",
            retriever=Retriever(),
            chain=Boom(),
        )


@pytest.mark.asyncio
async def test_get_rag_response_incompleto_tras_retry() -> None:
    class Retriever:
        async def ainvoke(self, *args: object, **kwargs: object):
            from langchain_core.documents import Document

            return [Document(page_content="vacaciones 21 días", metadata={"source": "politica_vacaciones.txt"})]

    class Flaky:
        async def ainvoke(self, *args: object, **kwargs: object) -> None:
            raise IncompleteOutputError("finish_reason=length")

    with pytest.raises(LLMClientError):
        await get_rag_response(
            "¿Cuántos días de vacaciones corresponden?",
            retriever=Retriever(),
            chain=Flaky(),
        )
