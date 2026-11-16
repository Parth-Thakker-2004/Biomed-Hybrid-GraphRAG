import pytest

from core.fusion import ReciprocalRankFusion


def test_rrf_scores_and_order(mini_corpus):
    a, b, c = mini_corpus[:3]
    fused = ReciprocalRankFusion(k=60).fuse({"dense": [a, b], "sparse": [b, c]})
    assert [f.chunk.chunk_id for f in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].ranks == {"dense": 2, "sparse": 1}
    assert fused[1].rrf_score == pytest.approx(1 / 61)


def test_agreement_beats_a_single_top_rank(mini_corpus):
    a, b, c = mini_corpus[:3]
    fused = ReciprocalRankFusion(k=60).fuse({"dense": [a, b], "sparse": [c, b], "graph": [b]})
    assert fused[0].chunk.chunk_id == b.chunk_id


def test_weights_and_top_n(mini_corpus):
    a, b = mini_corpus[:2]
    fused = ReciprocalRankFusion(k=60, weights={"graph": 3.0}).fuse({"dense": [a], "graph": [b]}, top_n=1)
    assert len(fused) == 1 and fused[0].chunk.chunk_id == b.chunk_id


def test_empty_channels():
    assert ReciprocalRankFusion().fuse({"dense": [], "sparse": []}) == []
