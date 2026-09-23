"""Ingesta: lee /data, fragmenta con RecursiveCharacterTextSplitter y persiste en ChromaDB."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import DeterministicEmbeddings, get_embeddings
from errors import IngestaError, PersistenciaError

logger = logging.getLogger("ingesta")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "techcorp_policies"
CHUNK_SIZE = 500
CHUNK_OVERLAP_MINIMO = 50
CHUNK_OVERLAP = 70
TOP_K = 4

# De mayor a menor coherencia. Solo se baja de nivel si el bloque sigue superando CHUNK_SIZE.
SEMANTIC_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]


def _as_path(ruta: str | Path) -> Path:
    path = Path(ruta)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def ya_existe_indice(persist_directory: str = PERSIST_DIR) -> bool:
    """True si el store local ya tiene archivos (no reindexar)."""

    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        return True
    path = _as_path(persist_directory)
    if path == Path(persist_directory).resolve():
        return False
    return path.exists() and any(path.iterdir())


def cargar_documentos(data_dir: str | Path = DATA_DIR) -> List[Document]:
    """Carga todos los .txt / .md de /data con DirectoryLoader + TextLoader."""

    carpeta = _as_path(data_dir)
    if not carpeta.is_dir():
        raise IngestaError(f"No existe la carpeta de documentos: {carpeta}")

    loader = DirectoryLoader(
        str(carpeta),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos_crudos = loader.load()
    md_loader = DirectoryLoader(
        str(carpeta),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos_crudos.extend(md_loader.load())

    if not documentos_crudos:
        raise IngestaError(
            f"La carpeta {carpeta} no tiene archivos .txt o .md para indexar."
        )
    logger.info("Documentos cargados: %s", len(documentos_crudos))
    return documentos_crudos


def construir_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Fragmentación en tokens (tiktoken), no en caracteres. Piso de overlap: 50."""

    if chunk_size < 500:
        raise IngestaError("chunk_size debe ser >= 500 tokens.")
    if chunk_overlap < CHUNK_OVERLAP_MINIMO:
        raise IngestaError("chunk_overlap debe ser >= 50 tokens.")
    if chunk_overlap >= chunk_size:
        raise IngestaError("chunk_overlap debe ser menor que chunk_size.")

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=70,
        separators=SEMANTIC_SEPARATORS,
    )


def _normalizar_fuente(doc: Document) -> Document:
    origen = doc.metadata.get("source", "desconocida")
    doc.metadata["source"] = Path(str(origen)).name
    return doc


def fragmentar_documentos(
    documentos_crudos: Iterable[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Aplica RecursiveCharacterTextSplitter.from_tiktoken_encoder."""

    splitter = construir_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(list(documentos_crudos))
    chunks = [_normalizar_fuente(chunk) for chunk in chunks]
    if not chunks:
        raise IngestaError("El splitter no produjo fragmentos.")
    logger.info("Fragmentos generados: %s", len(chunks))
    return chunks


def abrir_vectorstore(
    *,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    embeddings: Optional[Embeddings] = None,
) -> Chroma:
    """Abre el cliente persistente en ./vectorstore con el mismo embedding de consulta."""

    funcion = embeddings or get_embeddings()
    destino = str(_as_path(persist_directory))
    try:
        Path(destino).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=funcion,
            persist_directory=destino,
        )
    except Exception as exc:
        raise PersistenciaError(
            f"No se pudo abrir persist_directory={destino}: {exc}"
        ) from exc


def ingestir_documentos(
    data_dir: str | Path = DATA_DIR,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    embeddings: Optional[Embeddings] = None,
    *,
    force: bool = False,
) -> Chroma:
    """Indexa /data en ChromaDB. Si el índice ya existe, lo reutiliza."""

    funcion = embeddings or get_embeddings()
    destino = str(_as_path(persist_directory))

    if force and Path(destino).exists():
        shutil.rmtree(destino, ignore_errors=True)

    if ya_existe_indice(destino):
        logger.info("Índice existente detectado — cargando sin reindexar")
        print("Índice existente detectado — cargando sin reindexar")
        vectorstore = abrir_vectorstore(
            persist_directory=destino,
            collection_name=collection_name,
            embeddings=funcion,
        )
        print(f"Documentos en la colección: {vectorstore._collection.count()}")
        return vectorstore

    print("No hay índice previo — indexando documentos por primera vez")
    documentos_crudos = cargar_documentos(data_dir)
    chunks = fragmentar_documentos(documentos_crudos)
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=funcion,
            collection_name=collection_name,
            persist_directory=destino,
        )
    except Exception as exc:
        raise PersistenciaError(f"Chroma.from_documents falló: {exc}") from exc

    print(f"Documentos en la colección: {vectorstore._collection.count()}")
    return vectorstore


def get_retriever(
    vectorstore: Optional[Chroma] = None,
    *,
    k: int = TOP_K,
    **kwargs: Any,
):
    """Capa de recuperación: similarity search con top_k entre 3 y 5."""

    if k < 3 or k > 5:
        raise IngestaError("top_k debe estar entre 3 y 5 (evita contexto infinito).")
    store = vectorstore or ingestir_documentos(**kwargs)
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return retriever


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Ingesta de /data: chunking en tokens + ChromaDB persistente."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Carpeta de .txt/.md (default: data).",
    )
    parser.add_argument(
        "--persist-dir",
        default=PERSIST_DIR,
        help="Carpeta del PersistentClient (default: ./vectorstore).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borra el índice y vuelve a fragmentar / indexar.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usa DeterministicEmbeddings (sin bajar sentence-transformers).",
    )
    args = parser.parse_args(argv)

    embeddings = DeterministicEmbeddings() if args.offline else get_embeddings()
    try:
        store = ingestir_documentos(
            data_dir=args.data_dir,
            persist_directory=args.persist_dir,
            embeddings=embeddings,
            force=args.force,
        )
    except (IngestaError, PersistenciaError) as exc:
        print(f"Error controlado: {exc}")
        return 1

    retriever = get_retriever(store, k=4)
    muestra = retriever.invoke("¿Cuántos días de vacaciones tengo?")
    print(f"Retriever k=4 → {len(muestra)} fragmentos")
    for i, doc in enumerate(muestra, 1):
        print(f"--- Fragmento {i} (fuente: {doc.metadata['source']}) ---")
        print(doc.page_content[:150], "...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
