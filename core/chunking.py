"""
Section-aware chunking of FDA drug labels.

Labels arrive already split into named sections (CONTRAINDICATIONS, DRUG INTERACTIONS,
...). Chunks never cross a section boundary, are built from whole sentences, and overlap
by a few sentences so that a statement split across a boundary is still retrievable.
Every chunk keeps the drug name and section it came from, which is what the retrievers
and the reranker see as context.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\[(])")


@dataclass
class ClinicalChunk:
    """A passage of one label section."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    content: str
    token_count: int
    url: str = ""

    @property
    def indexed_text(self) -> str:
        """Text given to retrievers: label title and section give the passage its subject."""
        return f"{self.title} | {self.section}: {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClinicalChunk:
        return cls(**data)


def count_tokens(text: str) -> int:
    """Cheap word/punctuation count; within ~15% of BERT word-piece counts on label text."""
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class ClinicalSectionChunker:
    def __init__(self, target_tokens: int = 220, overlap_tokens: int = 40):
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def _units(self, text: str) -> List[str]:
        """Sentences, with any oversized 'sentence' (flattened tables) split on words."""
        units: List[str] = []
        for sentence in split_sentences(text):
            if count_tokens(sentence) <= self.target_tokens:
                units.append(sentence)
                continue
            words = sentence.split()
            step = max(1, int(self.target_tokens * 0.75))  # words are ~1.3 tokens each
            units.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
        return units

    def split_section(self, text: str) -> List[str]:
        """Greedy sentence packing into ~target_tokens windows with sentence overlap."""
        windows: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for unit in self._units(text):
            unit_tokens = count_tokens(unit)
            if current and current_tokens + unit_tokens > self.target_tokens:
                windows.append(" ".join(current))
                # Carry trailing sentences into the next window as overlap.
                overlap: List[str] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    prev_tokens = count_tokens(prev)
                    if overlap_tokens + prev_tokens > self.overlap_tokens:
                        break
                    overlap.insert(0, prev)
                    overlap_tokens += prev_tokens
                current, current_tokens = overlap, overlap_tokens
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            windows.append(" ".join(current))
        return windows

    def chunk_document(self, doc: Dict[str, Any]) -> List[ClinicalChunk]:
        """Chunks one corpus document: {doc_id, title, url, sections: {name: text}}."""
        chunks: List[ClinicalChunk] = []
        for section, text in doc.get("sections", {}).items():
            slug = section.lower().replace(" ", "_")
            for i, window in enumerate(self.split_section(text)):
                chunks.append(
                    ClinicalChunk(
                        chunk_id=f"{doc['doc_id']}::{slug}::{i:03d}",
                        doc_id=doc["doc_id"],
                        title=doc.get("title", doc["doc_id"]),
                        section=section,
                        content=window,
                        token_count=count_tokens(window),
                        url=doc.get("url", ""),
                    )
                )
        return chunks

    def chunk_corpus(self, docs: Iterable[Dict[str, Any]]) -> List[ClinicalChunk]:
        return [chunk for doc in docs for chunk in self.chunk_document(doc)]
