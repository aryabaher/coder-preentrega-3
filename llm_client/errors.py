"""Errores controlados y tipos transitorios que sí conviene reintentar."""

from __future__ import annotations

import httpx
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import InternalServerError as AnthropicInternalError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai import errors as genai_errors
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import InternalServerError as OpenAIInternalError
from openai import LengthFinishReasonError as OpenAILengthFinishReasonError
from openai import RateLimitError as OpenAIRateLimitError


class LLMClientError(Exception):
    """Falta de key, proveedor inválido o fallo de la API convertido en error de aplicación."""


class IncompleteOutputError(RuntimeError):
    """El LLM cortó la respuesta (finish_reason) o el objeto quedó incompleto."""


# Red, timeout, 429, 5xx y respuesta truncada (finish_reason=length). Auth / key: no.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    IncompleteOutputError,
    httpx.ConnectError,
    httpx.TimeoutException,
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIRateLimitError,
    OpenAIInternalError,
    OpenAILengthFinishReasonError,
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicRateLimitError,
    AnthropicInternalError,
    genai_errors.ServerError,
    GoogleAPIError,
    GoogleRateLimitError,
)
