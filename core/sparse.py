"""
BM25 lexical retrieval over an inverted index.

Implemented directly (no rank-bm25) so the tokenizer can be tuned for pharmacology text:
hyphenated identifiers stay whole ("P-gp", "HMG-CoA"), doses are joined ("50 mg" ->
"50mg"), and plurals are folded so "inhibitors" matches "inhibitor". Scoring only touches
the postings of the query terms, so cost grows with matches rather than corpus size.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from core.chunking import ClinicalChunk

STOPWORDS = frozenset(
    """a about after all also am an and any are as at be because been before being between both
    but by can could did do does doing during each for from further had has have having how
    if in into is it its may more most of on once only or other our out over own same should
    so some such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would
    you your""".split()
)

_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s+(mg|mcg|ug|g|ml|iu|meq|mmol)\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def _stem(token: str) -> str:
    """Folds regular plurals; deliberately conservative to keep drug names intact."""
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    text = _DOSE_RE.sub(r"\1\2", text.lower())
    tokens: List[str] = []
    for token in _TOKEN_RE.findall(text):
        parts = [token]
        if "-" in token or "/" in token:
            parts += re.split(r"[-/]", token)  # "beta-blockers" also indexes "beta", "blockers"
        for part in parts:
            if len(part) > 1 and part not in STOPWORDS:
                tokens.append(_stem(part))
    return tokens


class BiomedicalBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[ClinicalChunk] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.idf: Dict[str, float] = {}
        self._length_norm: List[float] = []

    @property
    def vocabulary(self) -> frozenset:
        return frozenset(self.postings)

    def index_chunks(self, chunks: Sequence[ClinicalChunk]) -> None:
        self.chunks = list(chunks)
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        lengths: List[int] = []

        for idx, chunk in enumerate(self.chunks):
            term_freqs = Counter(tokenize(chunk.indexed_text))
            lengths.append(sum(term_freqs.values()))
            for term, tf in term_freqs.items():
                postings[term].append((idx, tf))

        n = len(self.chunks)
        avg_len = (sum(lengths) / n) if n else 0.0
        self.postings = dict(postings)
        self.idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }
        self._length_norm = [
            self.k1 * (1.0 - self.b + self.b * (length / avg_len)) if avg_len else self.k1
            for length in lengths
        ]

    def search(self, query: str, top_k: int = 20) -> List[Tuple[ClinicalChunk, float]]:
        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in self.postings[term]:
                scores[idx] += idf * tf * (self.k1 + 1.0) / (tf + self._length_norm[idx])

        best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [(self.chunks[idx], score) for idx, score in best]
