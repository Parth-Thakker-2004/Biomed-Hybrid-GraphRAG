"""
Cross-encoder reranking.

The cross-encoder reads the query and a passage together, which makes it far more
accurate than the bi-encoder or BM25 but too slow to run over the corpus; it only scores
the fused candidate pool. Scores are returned as raw logits (no sigmoid) because the
evidence gate thresholds them directly.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from core.chunking import ClinicalChunk
from core.dense import resolve_device

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 384,
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading cross-encoder %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.device == "cuda":
            model = model.half()  # fp16 roughly halves GPU latency; ranking is unaffected
        self._model = model.to(self.device).eval()
        self._torch = torch

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Relevance logit for each (query, passage) pair."""
        if not passages:
            return []
        self._load()
        torch = self._torch
        scores: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            inputs = self._tokenizer(
                [query] * len(batch),
                list(batch),
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
        return scores

    def rerank(
        self, query: str, chunks: Sequence[ClinicalChunk]
    ) -> List[Tuple[ClinicalChunk, float]]:
        """Returns all candidates sorted by cross-encoder logit, best first."""
        scores = self.score(query, [c.indexed_text for c in chunks])
        return sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
