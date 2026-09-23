"""Mini-script: ingesta + pregunta en contexto + pregunta trampa."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from chain import get_rag_response
from embeddings import DeterministicEmbeddings
from errors import RAGError
from ingesta import ingestir_documentos
from llm_client import LLMClientError, resolve_provider

PREGUNTA_OK = (
    "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?"
)
PREGUNTA_TRAMPA = "¿Cuál es la política de bonos por rendimiento anual en TechCorp?"


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


async def _run_case(title: str, query: str, **kwargs) -> None:
    print(f"=== {title} ===")
    print(f"Pregunta: {query}\n")
    try:
        resultado = await get_rag_response(query, **kwargs)
    except (LLMClientError, RAGError, ValueError) as exc:
        print(f"Error controlado: {exc}\n")
        return
    except Exception as exc:
        print(f"Error controlado: {exc}\n")
        return
    print("RESPUESTA:", resultado.respuesta)
    print("FUENTES:", resultado.fuentes)
    print("Fragmentos usados:", resultado.fragmentos_recuperados)
    print(resultado.model_dump_json(indent=2))
    print()


async def run_interactive(**kwargs) -> None:
    print("Modo interactivo — escribí tu pregunta sobre las políticas de TechCorp")
    print("   (escribí 'salir' para terminar)\n")
    while True:
        pregunta_usuario = input("Vos: ").strip()
        if pregunta_usuario.lower() in ("salir", "exit", "quit", ""):
            print("\nListo, terminamos la sesión.")
            break
        await _run_case("Consulta", pregunta_usuario, **kwargs)
        print("-" * 80)


async def run_demo(
    *,
    provider: str | None,
    interactive: bool,
    offline: bool,
    force: bool,
    max_tokens: int | None,
) -> None:
    embeddings = DeterministicEmbeddings() if offline else None
    ingestir_documentos(embeddings=embeddings, force=force)

    kwargs = {
        "provider": provider,
        "embeddings": embeddings,
        "max_tokens": max_tokens,
    }

    if interactive:
        await run_interactive(**kwargs)
        return

    await _run_case("Pregunta en los documentos (camino feliz)", PREGUNTA_OK, **kwargs)
    await _run_case("Pregunta trampa (fuera de contexto)", PREGUNTA_TRAMPA, **kwargs)


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="RAG local: ingesta ChromaDB + get_rag_response asíncrono."
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "gemini"),
        default=None,
        help="Proveedor del LLM. Default: LLM_PROVIDER o openai.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Preguntas libres en consola.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Embeddings deterministas (sin bajar sentence-transformers).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reindexa /data aunque ya exista ./vectorstore.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Tope de tokens de salida. Un valor bajo (p. ej. 16) fuerza finish_reason=length.",
    )
    args = parser.parse_args()
    try:
        if args.provider:
            resolve_provider(args.provider)
    except LLMClientError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)
    asyncio.run(
        run_demo(
            provider=args.provider,
            interactive=args.interactive,
            offline=args.offline,
            force=args.force,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
