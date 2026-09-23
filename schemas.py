"""Contrato de salida del RAG: texto del LLM + fuentes verificables."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar, parseado directamente de su output."""

    respuesta: str = Field(
        description=(
            "Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
            "Si la información no está en el contexto, decir explícitamente que no se "
            "cuenta con esa información."
        ),
        min_length=1,
    )

    @field_validator("respuesta")
    @classmethod
    def respuesta_no_vacia(cls, v: str) -> str:
        limpio = (v or "").strip()
        if not limpio:
            raise ValueError("La respuesta no puede quedar vacía.")
        return limpio


class RAGResponse(BaseModel):
    """Objeto final que devuelve get_rag_response — combina el output del LLM con metadata verificable."""

    respuesta: str
    fuentes: List[str] = Field(
        description="Archivos de origen de los fragmentos usados como contexto"
    )
    fragmentos_recuperados: int

    @field_validator("fuentes")
    @classmethod
    def fuentes_sin_vacios(cls, v: List[str]) -> List[str]:
        return list(dict.fromkeys(item.strip() for item in v if item and str(item).strip()))
