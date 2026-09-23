"""Embeddings: el mismo modelo para indexar y para consultar."""

from __future__ import annotations

import hashlib
from typing import List

from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Mismo modelo de embeddings para indexar Y para consultar."""

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class DeterministicEmbeddings(Embeddings):
    """Bag-of-words hasheado. Misma dimensión y mismos vectores en cada corrida.

    No baja modelos ni llama APIs. Tests, `validacion.py` e `--offline` lo inyectan.
    """

    _STOP = frozenset(
        "el la los las un una de del y a en con por para que se al lo su "
        "es un una o the a an of to and".split()
    )

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(texto) for texto in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    def _vector(self, texto: str) -> List[float]:
        vec = [0.0] * self.dim
        for crudo in (texto or "").lower().replace("¿", " ").replace("?", " ").split():
            token = "".join(ch for ch in crudo if ch.isalnum())
            if not token or token in self._STOP:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[digest[0] % self.dim] += 1.0
            vec[digest[1] % self.dim] += 0.5
        norma = sum(x * x for x in vec) ** 0.5
        if norma == 0:
            return vec
        return [x / norma for x in vec]
