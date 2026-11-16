import pytest

from core.chunking import ClinicalSectionChunker, count_tokens, split_sentences

DOC = {
    "doc_id": "warfarin",
    "title": "Warfarin (Coumadin)",
    "url": "https://example.test/warfarin",
    "sections": {
        "BOXED WARNING": "Warfarin can cause major or fatal bleeding. Monitor INR regularly.",
        "DRUG INTERACTIONS": " ".join(f"Sentence number {i} about an interacting drug." for i in range(60)),
    },
}


def test_chunks_never_cross_sections_and_keep_provenance():
    chunks = ClinicalSectionChunker(target_tokens=60, overlap_tokens=15).chunk_document(DOC)
    assert {c.section for c in chunks} == {"BOXED WARNING", "DRUG INTERACTIONS"}
    boxed = [c for c in chunks if c.section == "BOXED WARNING"]
    assert len(boxed) == 1 and "interacting drug" not in boxed[0].content
    assert all(c.doc_id == "warfarin" and c.url == DOC["url"] for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunks_respect_token_budget_and_overlap():
    chunker = ClinicalSectionChunker(target_tokens=60, overlap_tokens=15)
    windows = chunker.split_section(DOC["sections"]["DRUG INTERACTIONS"])
    assert len(windows) > 3
    assert all(count_tokens(w) <= 60 for w in windows)
    for prev, nxt in zip(windows, windows[1:]):
        last_sentence = split_sentences(prev)[-1]
        assert nxt.startswith(last_sentence), "consecutive windows should share a sentence"


def test_oversized_sentence_is_split():
    chunker = ClinicalSectionChunker(target_tokens=40, overlap_tokens=5)
    windows = chunker.split_section("word " * 500)
    assert len(windows) > 5
    assert all(count_tokens(w) <= 40 for w in windows)


def test_indexed_text_carries_drug_and_section():
    chunk = ClinicalSectionChunker().chunk_document(DOC)[0]
    assert chunk.indexed_text.startswith("Warfarin (Coumadin) | BOXED WARNING:")


def test_overlap_must_be_smaller_than_target():
    with pytest.raises(ValueError):
        ClinicalSectionChunker(target_tokens=50, overlap_tokens=50)
