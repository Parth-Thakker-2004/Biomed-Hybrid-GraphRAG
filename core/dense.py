"""
Dense retrieval: BGE sentence embeddings in a FAISS HNSW index.

Embeddings are L2-normalised, so inner product equals cosine similarity. The raw
embedding matrix is stored next to the index; it allows exact (brute-force) search,
which the evaluation uses to measure how much recall HNSW gives up.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from core.chunking import ClinicalChunk

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_instruction: str = "",
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        batch_size: int = 64,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.batch_size = batch_size
        self.device = resolve_device(device)

        self.chunks: List[ClinicalChunk] = []
        self.embeddings: np.ndarray | None = None
        self.index = None
        self._model = None

    # ------------------------------------------------------------------ #
    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_instruction + query])

    def _build_hnsw(self) -> None:
        import faiss

        index = faiss.IndexHNSWFlat(self.embeddings.shape[1], self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.add(self.embeddings)
        index.hnsw.efSearch = self.ef_search
        self.index = index

    # ------------------------------------------------------------------ #
    def index_chunks(self, chunks: Sequence[ClinicalChunk]) -> None:
        self.chunks = list(chunks)
        self.embeddings = self._encode([c.indexed_text for c in self.chunks], show_progress=True)
        self._build_hnsw()

    def search(
        self, query: str, top_k: int = 20, exact: bool = False
    ) -> List[Tuple[ClinicalChunk, float]]:
        if not self.chunks:
            return []
        query_vec = self.embed_query(query)
        top_k = min(top_k, len(self.chunks))

        if exact or self.index is None:
            sims = (self.embeddings @ query_vec.T).ravel()
            order = np.argsort(-sims)[:top_k]
            return [(self.chunks[i], float(sims[i])) for i in order]

        scores, ids = self.index.search(query_vec, top_k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    # ------------------------------------------------------------------ #
    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        faiss.write_index(self.index, str(directory / "dense.faiss"))
        (directory / "dense_meta.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "query_instruction": self.query_instruction,
                    "m": self.m,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "count": len(self.chunks),
                    "dimension": int(self.embeddings.shape[1]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, directory: Path, chunks: Sequence[ClinicalChunk], device: str = "auto"
    ) -> DenseRetriever:
        import faiss

        meta = json.loads((directory / "dense_meta.json").read_text(encoding="utf-8"))
        if meta["count"] != len(chunks):
            raise ValueError("Dense index is out of date; run scripts/build_indexes.py")
        retriever = cls(
            model_name=meta["model_name"],
            query_instruction=meta["query_instruction"],
            m=meta["m"],
            ef_construction=meta["ef_construction"],
            ef_search=meta["ef_search"],
            device=device,
        )
        retriever.chunks = list(chunks)
        retriever.embeddings = np.load(directory / "embeddings.npy")
        retriever.index = faiss.read_index(str(directory / "dense.faiss"))
        retriever.index.hnsw.efSearch = meta["ef_search"]
        return retriever
