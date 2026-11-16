"""
Reciprocal Rank Fusion (Cormack et al., 2009).

    score(d) = sum over channels c of  w_c / (k + rank_c(d))

RRF uses ranks only, so BM25 scores, cosine similarities and graph path order can be
combined without calibrating their scales against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from core.chunking import ClinicalChunk


@dataclass
class FusedCandidate:
    chunk: ClinicalChunk
    rrf_score: float
    ranks: Dict[str, int] = field(default_factory=dict)  # channel -> 1-based rank


class ReciprocalRankFusion:
    def __init__(self, k: int = 60, weights: Mapping[str, float] | None = None):
        self.k = k
        self.weights = dict(weights or {})

    def fuse(
        self, channels: Mapping[str, Sequence[ClinicalChunk]], top_n: int = 20
    ) -> List[FusedCandidate]:
        """Fuses ranked lists; `channels` maps a channel name to chunks, best first."""
        merged: Dict[str, FusedCandidate] = {}
        for name, ranked_chunks in channels.items():
            weight = self.weights.get(name, 1.0)
            for rank, chunk in enumerate(ranked_chunks, start=1):
                candidate = merged.setdefault(chunk.chunk_id, FusedCandidate(chunk, 0.0))
                if name in candidate.ranks:  # duplicate within one channel
                    continue
                candidate.rrf_score += weight / (self.k + rank)
                candidate.ranks[name] = rank
        return sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)[:top_n]
