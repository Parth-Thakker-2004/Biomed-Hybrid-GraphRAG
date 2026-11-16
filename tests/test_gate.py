import pytest

from core.gate import EvidenceGate, find_unknown_drug_terms
from core.sparse import BiomedicalBM25


@pytest.fixture
def gate(mini_corpus, linker) -> EvidenceGate:
    bm25 = BiomedicalBM25()
    bm25.index_chunks(mini_corpus)
    return EvidenceGate(logit_threshold=0.0, vocabulary=bm25.vocabulary, linker=linker)


def test_passes_when_evidence_is_relevant(gate):
    decision = gate.decide("Does clarithromycin raise simvastatin levels?", [2.5, -1.0])
    assert decision.passed and decision.reason == "ok" and decision.max_score == 2.5


def test_refuses_below_threshold(gate):
    decision = gate.decide("Does clarithromycin raise simvastatin levels?", [-0.4, -3.0])
    assert not decision.passed and decision.reason == "low_relevance"


def test_refuses_unknown_drug_even_with_a_high_score(gate):
    decision = gate.decide("Can Xanodrol be taken with warfarin?", [4.2])
    assert not decision.passed and decision.reason == "unknown_term"
    assert decision.unknown_terms == ["Xanodrol"]


def test_refuses_without_candidates(gate):
    assert gate.decide("anything about warfarin", []).reason == "no_candidates"


def test_unknown_term_heuristics(gate):
    find = lambda q: find_unknown_drug_terms(q, gate.vocabulary, gate.linker)  # noqa: E731
    assert find("is morvastatin safe") == ["morvastatin"]  # drug-like suffix, lower case
    assert find("Is Coumadin safe?") == []  # brand name resolved by the linker
    assert find("What about bleeding risk?") == []  # sentence-initial capital, ordinary words
    assert find("does it interact with spaghetti") == []  # unknown but not drug-like


def test_vocabulary_check_can_be_disabled(mini_corpus, linker):
    bm25 = BiomedicalBM25()
    bm25.index_chunks(mini_corpus)
    gate = EvidenceGate(0.0, bm25.vocabulary, linker, refuse_on_unknown_drug=False)
    assert gate.decide("Can Xanodrol be taken with warfarin?", [4.2]).passed
