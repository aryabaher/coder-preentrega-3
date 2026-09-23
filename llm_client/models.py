"""Fábrica de modelos: ChatOpenAI, ChatAnthropic o ChatGoogleGenerativeAI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from llm_client.errors import LLMClientError

_ROOT = Path(__file__).resolve().parent.parent
_SUPPORTED = ("openai", "anthropic", "gemini")
DEFAULT_MAX_TOKENS = 512


def _load_env() -> None:
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT.parent / ".env")
    load_dotenv()


def resolve_provider(provider: str | None = None) -> str:
    _load_env()
    name = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if name not in _SUPPORTED:
        raise LLMClientError(f"Proveedor no soportado: {name!r}. Usá openai, anthropic o gemini.")
    return name


def _secret(*names: str) -> str:
    for key in names:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def get_model(provider: str, *, max_tokens: int | None = None):
    """Instancia el chat model del proveedor (openai / anthropic / gemini)."""

    provider = resolve_provider(provider)
    timeout = float(os.getenv("LLM_TIMEOUT", "30"))
    tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

    if provider == "openai":
        api_key = _secret("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError("Falta OPENAI_API_KEY.")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=tokens,
            timeout=timeout,
            api_key=api_key,
        )
    elif provider == "anthropic":
        api_key = _secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMClientError("Falta ANTHROPIC_API_KEY.")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=0,
            max_tokens=tokens,
            timeout=timeout,
            api_key=api_key,
        )
    elif provider == "gemini":
        api_key = _secret("GOOGLE_API_KEY", "GEMINI_API_KEY")
        if not api_key:
            raise LLMClientError("Falta GOOGLE_API_KEY.")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            temperature=0,
            max_output_tokens=tokens,
            timeout=timeout,
            max_retries=0,
            google_api_key=api_key,
        )
    raise ValueError(f"Proveedor no soportado: {provider}")


def build_model(provider: str | None = None, *, max_tokens: int | None = None):
    """Atajo: resuelve LLM_PROVIDER y llama a get_model."""

    return get_model(resolve_provider(provider), max_tokens=max_tokens)
