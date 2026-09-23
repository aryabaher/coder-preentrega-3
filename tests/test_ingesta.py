"""Chunking, DirectoryLoader y persistencia sin reindexar."""

from __future__ import annotations

from pathlib import Path

import pytest

from embeddings import DeterministicEmbeddings
from errors import IngestaError
from ingesta import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    cargar_documentos,
    construir_splitter,
    fragmentar_documentos,
    get_retriever,
    ingestir_documentos,
    ya_existe_indice,
)
import ingesta as ingesta_mod

ROOT = Path(__file__).resolve().parent.parent


def test_cargar_documentos_data():
    docs = cargar_documentos(ROOT / "data")
    nombres = {Path(d.metadata["source"]).name for d in docs}
    assert "politica_vacaciones.txt" in nombres
    assert "politica_teletrabajo.txt" in nombres
    assert "politica_seguridad_informatica.txt" in nombres
    assert "onboarding_nuevos_empleados.txt" in nombres
    assert len(docs) == 4


def test_chunk_size_minimo_500():
    splitter = construir_splitter()
    assert splitter._chunk_size == CHUNK_SIZE == 500
    assert splitter._chunk_overlap == CHUNK_OVERLAP == 70


def test_chunk_size_menor_a_500_falla():
    with pytest.raises(IngestaError, match="500"):
        construir_splitter(chunk_size=200, chunk_overlap=50)


def test_overlap_menor_a_50_falla():
    with pytest.raises(IngestaError, match="50"):
        construir_splitter(chunk_size=500, chunk_overlap=10)


def test_fragmentar_produce_varios_chunks():
    docs = cargar_documentos(ROOT / "data")
    chunks = fragmentar_documentos(docs)
    assert len(chunks) > len(docs)
    vacaciones = [c for c in chunks if c.metadata["source"] == "politica_vacaciones.txt"]
    assert len(vacaciones) >= 2


def test_ingesta_persiste_y_no_reindexa(tmp_path, mocker):
    destino = tmp_path / "vectorstore"
    embeddings = DeterministicEmbeddings()
    store = ingestir_documentos(
        data_dir=ROOT / "data",
        persist_directory=str(destino),
        embeddings=embeddings,
        force=True,
    )
    count = store._collection.count()
    assert count > 0
    assert ya_existe_indice(str(destino))

    spy = mocker.spy(ingesta_mod.Chroma, "from_documents")
    store2 = ingestir_documentos(
        data_dir=ROOT / "data",
        persist_directory=str(destino),
        embeddings=embeddings,
        force=False,
    )
    assert store2._collection.count() == count
    spy.assert_not_called()


def test_retriever_k_4(tmp_path):
    embeddings = DeterministicEmbeddings()
    store = ingestir_documentos(
        data_dir=ROOT / "data",
        persist_directory=str(tmp_path / "vectorstore"),
        embeddings=embeddings,
        force=True,
    )
    retriever = get_retriever(store, k=4)
    hits = retriever.invoke("días corridos de vacaciones por antigüedad")
    assert 1 <= len(hits) <= 4
    blob = " ".join(h.page_content.lower() for h in hits)
    assert "vacaciones" in blob


def test_data_vacia(tmp_path):
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    with pytest.raises(IngestaError, match="no tiene archivos"):
        cargar_documentos(vacia)


def test_top_k_fuera_de_rango(tmp_path):
    embeddings = DeterministicEmbeddings()
    store = ingestir_documentos(
        data_dir=ROOT / "data",
        persist_directory=str(tmp_path / "vs"),
        embeddings=embeddings,
        force=True,
    )
    with pytest.raises(IngestaError, match="top_k"):
        get_retriever(store, k=50)
    with pytest.raises(IngestaError, match="top_k"):
        get_retriever(store, k=1)
