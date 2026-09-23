"""get_rag_response: recuperación + parseo Pydantic, sin API real."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from chain import formatear_documentos, get_rag_response, parser_llm
from schemas import RAGResponse, RespuestaLLM

VACACIONES = Document(
    page_content=(
        "Los empleados con 5 años o más, y hasta 10 años, acceden a 21 días corridos."
    ),
    metadata={"source": "politica_vacaciones.txt"},
)
TELETRABAJO = Document(
    page_content="El esquema estándar es de 3 días de trabajo remoto y 2 presenciales.",
    metadata={"source": "politica_teletrabajo.txt"},
)


class _FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    async def ainvoke(self, query: str):
        return list(self.docs)


class _FakeChain:
    def __init__(self, respuesta: str):
        self.respuesta = respuesta
        self.last_input = None

    async def ainvoke(self, payload, config=None):
        self.last_input = payload
        return RespuestaLLM(respuesta=self.respuesta)


@pytest.mark.asyncio
async def test_pregunta_en_documentos():
    chain = _FakeChain(
        "Un empleado con 5 años de antigüedad accede a 21 días corridos de vacaciones."
    )
    resultado = await get_rag_response(
        "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?",
        retriever=_FakeRetriever([VACACIONES, TELETRABAJO]),
        chain=chain,
    )
    assert isinstance(resultado, RAGResponse)
    assert "21" in resultado.respuesta
    assert "politica_vacaciones.txt" in resultado.fuentes
    assert resultado.fragmentos_recuperados == 2
    assert "CONTEXTO" not in resultado.respuesta
    assert "formato" in chain.last_input
    assert "21 días" in chain.last_input["contexto"]


@pytest.mark.asyncio
async def test_pregunta_trampa_no_alucina():
    texto = "No tengo acceso a esa información en los documentos disponibles."
    resultado = await get_rag_response(
        "¿Cuál es la política de bonos por rendimiento anual en TechCorp?",
        retriever=_FakeRetriever([VACACIONES]),
        chain=_FakeChain(texto),
    )
    assert resultado.respuesta == texto
    assert "bono" not in resultado.respuesta.lower()
    assert resultado.fuentes == ["politica_vacaciones.txt"]


@pytest.mark.asyncio
async def test_pregunta_trampa_texto_plano_no_es_error_de_parseo():
    from langchain_core.exceptions import OutputParserException

    class UnaSolaVez:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            self.calls += 1
            raise OutputParserException(
                "Invalid json output: No tengo acceso a esa información en los documentos disponibles."
            )

    chain = UnaSolaVez()
    resultado = await get_rag_response(
        "¿Cuál es la política de bonos por rendimiento anual en TechCorp?",
        retriever=_FakeRetriever([VACACIONES]),
        chain=chain,
    )
    assert resultado.respuesta == (
        "No tengo acceso a esa información en los documentos disponibles."
    )
    assert chain.calls == 1
    assert resultado.fragmentos_recuperados == 1


@pytest.mark.asyncio
async def test_retriever_vacio_no_llama_al_llm():
    from chain import RESPUESTA_SIN_CONTEXTO

    class Boom:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("no debería llamarse al LLM sin fragmentos")

    resultado = await get_rag_response(
        "¿Cuál es la política de bonos por rendimiento anual en TechCorp?",
        retriever=_FakeRetriever([]),
        chain=Boom(),
    )
    assert resultado.respuesta == RESPUESTA_SIN_CONTEXTO
    assert resultado.fuentes == []
    assert resultado.fragmentos_recuperados == 0


def test_parse_mensaje_acepta_rechazo_en_texto_plano():
    from chain import RESPUESTA_SIN_CONTEXTO, _parse_mensaje

    class Msg:
        content = RESPUESTA_SIN_CONTEXTO
        response_metadata = {}
        additional_kwargs = {}

    parsed = _parse_mensaje(Msg())
    assert parsed.respuesta == RESPUESTA_SIN_CONTEXTO


def test_formatear_documentos_incluye_fuente():
    texto = formatear_documentos([VACACIONES])
    assert "[Fuente: politica_vacaciones.txt]" in texto
    assert "21 días" in texto


def test_parser_llm_es_pydantic_output_parser():
    from langchain_core.output_parsers import PydanticOutputParser

    assert isinstance(parser_llm, PydanticOutputParser)
    assert parser_llm.pydantic_object is RespuestaLLM
    instrucciones = parser_llm.get_format_instructions()
    assert "respuesta" in instrucciones.lower()
