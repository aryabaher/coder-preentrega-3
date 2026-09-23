"""Factory get_model / build_model y contrato Pydantic (sin API)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_client.errors import LLMClientError
from llm_client.models import build_model, get_model
from schemas import RAGResponse, RespuestaLLM


def test_build_model_creates_chat_openai(fake_chat_openai):
    model = build_model("openai")
    assert model is not None
    fake_chat_openai.assert_called_once()


def test_missing_api_key_raises(monkeypatch, mocker):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mocker.patch("llm_client.models.ChatOpenAI")
    with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
        build_model("openai")


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bloop")
    with pytest.raises(LLMClientError, match="Proveedor no soportado"):
        build_model()


def test_missing_anthropic_y_openai_keys(monkeypatch, mocker):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mocker.patch("llm_client.models.ChatAnthropic")
    mocker.patch("llm_client.models.ChatOpenAI")
    with pytest.raises(LLMClientError, match="ANTHROPIC_API_KEY"):
        get_model("anthropic")
    with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
        get_model("openai")


def test_get_model_anthropic_y_gemini(monkeypatch, mocker):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    ant = mocker.patch("llm_client.models.ChatAnthropic")
    gem = mocker.patch("llm_client.models.ChatGoogleGenerativeAI")
    assert get_model("anthropic") is not None
    assert get_model("gemini") is not None
    ant.assert_called_once()
    gem.assert_called_once()


def test_respuesta_vacia() -> None:
    with pytest.raises(ValidationError):
        RespuestaLLM(respuesta="   ")


def test_esquema_valido() -> None:
    ok = RespuestaLLM(respuesta="21 días corridos para 5 años de antigüedad.")
    rag = RAGResponse(
        respuesta=ok.respuesta,
        fuentes=["politica_vacaciones.txt", "politica_vacaciones.txt"],
        fragmentos_recuperados=2,
    )
    assert rag.fuentes == ["politica_vacaciones.txt"]
    assert rag.fragmentos_recuperados == 2
