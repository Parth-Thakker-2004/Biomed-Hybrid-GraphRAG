"""
Evidence gate: decide whether to answer or refuse.

A retriever always returns its top-k, relevant or not, and a generator will happily write
an answer from whatever it is given. The gate is the component that says "the corpus does
not cover this". It combines two independent signals:

1. Relevance. The best cross-encoder logit must reach a threshold calibrated on a
   held-out dev split.
2. Vocabulary. If the question names something that looks like a drug but the name
   appears nowhere in the corpus, no passage can be about it. This catches the failure
   mode where a question about "Xanodrol and warfarin" retrieves strong warfarin passages
   and passes the relevance check on the strength of the half that is real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import AbstractSet, List, Sequence

from core.entities import EntityLinker
from core.sparse import tokenize

# Common generic-name stems (USAN): a lower-case out-of-vocabulary word with one of these
# endings is treated as a drug name.
DRUG_SUFFIXES = (
    "olol", "pril", "sartan", "statin", "afil", "azole", "mycin", "cillin", "cycline",
    "floxacin", "xaban", "tinib", "mab", "prazole", "tidine", "gliptin", "gliflozin",
    "glutide", "azepam", "zolam", "oxetine", "triptan", "dipine", "vir", "caine", "barbital",
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")


@dataclass
class GateDecision:
    passed: bool
    reason: str  # "ok" | "low_relevance" | "unknown_term" | "no_candidates"
    max_score: float | None = None
    unknown_terms: List[str] = field(default_factory=list)


def find_unknown_drug_terms(
    query: str, vocabulary: AbstractSet[str], linker: EntityLinker
) -> List[str]:
    """Words that look like drug names but occur nowhere in the indexed corpus."""
    linked_spans = [(e.start, e.end) for e in linker.link(query)]
    unknown: List[str] = []
    for match in _WORD_RE.finditer(query):
        word = match.group(0)
        if any(start <= match.start() < end for start, end in linked_spans):
            continue
        tokens = tokenize(word)
        if not tokens or any(token in vocabulary for token in tokens):
            continue
        is_sentence_start = match.start() == 0 or query[: match.start()].rstrip().endswith(("?", ".", "!"))
        capitalised = word[0].isupper() and not is_sentence_start
        if capitalised or word.lower().endswith(DRUG_SUFFIXES):
            unknown.append(word)
    return unknown


class EvidenceGate:
    def __init__(
        self,
        logit_threshold: float,
        vocabulary: AbstractSet[str],
        linker: EntityLinker,
        refuse_on_unknown_drug: bool = True,
    ):
        self.logit_threshold = logit_threshold
        self.vocabulary = vocabulary
        self.linker = linker
        self.refuse_on_unknown_drug = refuse_on_unknown_drug

    def decide(self, query: str, reranker_scores: Sequence[float]) -> GateDecision:
        max_score = max(reranker_scores) if reranker_scores else None
        if self.refuse_on_unknown_drug:
            unknown = find_unknown_drug_terms(query, self.vocabulary, self.linker)
            if unknown:
                return GateDecision(False, "unknown_term", max_score, unknown)
        if max_score is None:
            return GateDecision(False, "no_candidates")
        if max_score < self.logit_threshold:
            return GateDecision(False, "low_relevance", max_score)
        return GateDecision(True, "ok", max_score)
