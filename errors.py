"""Errores controlados de ingesta, recuperación y generación grounded."""

from __future__ import annotations


class RAGError(Exception):
    """Fallo de ingesta, retriever o cadena RAG convertido en error de aplicación."""


class ConsultaVaciaError(RAGError):
    """query vacío: no hay nada que embeddear ni preguntar."""


class IngestaError(RAGError):
    """Carpeta /data vacía, archivos ilegibles o el splitter no produjo fragmentos."""


class PersistenciaError(RAGError):
    """No se pudo abrir o escribir el vectorstore local."""
