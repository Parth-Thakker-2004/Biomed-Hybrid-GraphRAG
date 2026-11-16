from core.sparse import BiomedicalBM25, tokenize


def test_tokenizer_keeps_pharmacology_tokens_whole():
    tokens = tokenize("P-gp and HMG-CoA inhibitors; warfarin 5 mg with CYP3A4")
    assert "p-gp" in tokens and "hmg-coa" in tokens
    assert "5mg" in tokens  # dose joined with its unit
    assert "cyp3a4" in tokens
    assert "inhibitor" in tokens  # plural folded
    assert "and" not in tokens and "with" not in tokens  # stopwords dropped


def test_hyphenated_terms_are_also_indexed_by_part():
    assert {"beta-blocker", "beta", "blocker"} <= set(tokenize("beta-blockers"))


def test_search_ranks_the_matching_passage_first(mini_corpus):
    bm25 = BiomedicalBM25()
    bm25.index_chunks(mini_corpus)
    top, score = bm25.search("warfarin bleeding NSAIDs", top_k=3)[0]
    assert top.doc_id == "warfarin" and score > 0


def test_title_is_searchable(mini_corpus):
    # "clarithromycin" appears in the simvastatin passage, but the clarithromycin label
    # should win because the drug name is part of its indexed text.
    bm25 = BiomedicalBM25()
    bm25.index_chunks(mini_corpus)
    assert bm25.search("clarithromycin", top_k=1)[0][0].doc_id == "clarithromycin"


def test_unknown_terms_return_nothing(mini_corpus):
    bm25 = BiomedicalBM25()
    bm25.index_chunks(mini_corpus)
    assert bm25.search("xanodrol gloxipen") == []
    assert "xanodrol" not in bm25.vocabulary and "warfarin" in bm25.vocabulary
