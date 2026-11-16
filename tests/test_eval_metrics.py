import pytest

from eval.build_dataset import gold_chunk_ids
from eval.run_evaluation import auroc, calibrate_threshold, gate_outcomes, retrieval_metrics


def test_retrieval_metrics():
    m = retrieval_metrics(["a", "b", "c", "d"], gold={"c", "z"})
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert 0 < m["ndcg@10"] < 1
    assert retrieval_metrics(["a"], gold={"a"}) == {"hit@1": 1.0, "hit@3": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0}
    assert retrieval_metrics([], gold={"a"})["mrr@10"] == 0.0


def test_auroc():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1, 3], [2, 2]) == 0.5


ROWS = [
    {"answerable": True, "category": "direct", "max_logit": 4.0, "unknown_terms": []},
    {"answerable": True, "category": "direct", "max_logit": 1.0, "unknown_terms": []},
    {"answerable": False, "category": "neg_off_topic", "max_logit": -5.0, "unknown_terms": []},
    {"answerable": False, "category": "neg_fictional", "max_logit": 2.0, "unknown_terms": ["Xanodrol"]},
]


def test_calibration_and_gate_outcomes():
    tau = calibrate_threshold(ROWS, use_vocabulary=True)
    assert -5.0 < tau < 1.0
    combined = gate_outcomes(ROWS, lambda r: bool(r["unknown_terms"]) or r["max_logit"] < tau)
    assert combined["refusal_recall"] == 1.0 and combined["false_refusal_rate"] == 0.0
    relevance_only = gate_outcomes(ROWS, lambda r: r["max_logit"] < tau)
    assert relevance_only["refusal_recall_by_category"] == {"neg_off_topic": 1.0, "neg_fictional": 0.0}


def test_gold_rules(mini_corpus):
    assert gold_chunk_ids([("sildenafil", ["nitrate"])], mini_corpus) == ["sildenafil::contraindications::000"]
    assert gold_chunk_ids([("sildenafil", [""], "PHARMACOKINETICS")], mini_corpus) == ["sildenafil::pharmacokinetics::000"]
    assert gold_chunk_ids([("warfarin", ["nitrate"])], mini_corpus) == []
