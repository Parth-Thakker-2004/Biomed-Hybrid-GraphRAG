"""
Evaluation: retrieval ablation, evidence-gate calibration, end-to-end outcome, latency.

    python eval/run_evaluation.py            # writes eval/results.json and eval/RESULTS.md

Method
* Every configuration is run through Engine.retrieve, the code path that serves requests.
* Retrieval is scored against rule-defined gold passages (see eval/build_dataset.py).
* The gate threshold is chosen on the dev split only. All reported numbers come from the
  test split, which was not used for calibration or debugging.
* 95% confidence intervals are percentile bootstrap intervals over test queries.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gate import find_unknown_drug_terms  # noqa: E402
from core.pipeline import Engine  # noqa: E402
from eval.build_dataset import gold_chunk_ids  # noqa: E402

K = 10
TOP_K_ANSWER = 3

# name -> Engine.retrieve keyword arguments
CONFIGS: Dict[str, Dict[str, Any]] = {
    "BM25": dict(channels=("sparse",), rerank=False, normalize=False, entity_filter=False),
    "Dense (bge-small, HNSW)": dict(channels=("dense",), rerank=False, normalize=False, entity_filter=False),
    "Hybrid (BM25 + dense, RRF)": dict(channels=("sparse", "dense"), rerank=False, normalize=False, entity_filter=False),
    "Hybrid + cross-encoder": dict(channels=("sparse", "dense"), rerank=True, normalize=False, entity_filter=False),
    "Hybrid + entity linking + graph": dict(channels=("sparse", "dense", "graph"), rerank=False, normalize=True, entity_filter=True),
    "Full pipeline": dict(channels=("sparse", "dense", "graph"), rerank=True, normalize=True, entity_filter=True),
}
FULL = "Full pipeline"
BASELINE = "Hybrid + cross-encoder"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def retrieval_metrics(ranked_ids: Sequence[str], gold: set) -> Dict[str, float]:
    hits = [1.0 if cid in gold else 0.0 for cid in ranked_ids[:K]]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), K)))
    return {
        "hit@1": hits[0] if hits else 0.0,
        "hit@3": float(any(hits[:3])),
        "mrr@10": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg@10": dcg / ideal if ideal else 0.0,
    }


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> List[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability that a random answerable query scores above a random unanswerable one."""
    pos, neg = np.asarray(positives), np.asarray(negatives)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def gate_outcomes(rows: Sequence[Dict[str, Any]], refuse: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    refused_unans = [float(bool(refuse(r))) for r in unanswerable]
    refused_ans = [float(bool(refuse(r))) for r in answerable]
    by_category: Dict[str, List[float]] = {}
    for row, flag in zip(unanswerable, refused_unans):
        by_category.setdefault(row["category"], []).append(flag)
    return {
        "refusal_recall": float(np.mean(refused_unans)),
        "refusal_recall_ci": bootstrap_ci(refused_unans),
        "false_refusal_rate": float(np.mean(refused_ans)),
        "false_refusal_rate_ci": bootstrap_ci(refused_ans),
        "balanced_accuracy": float((np.mean(refused_unans) + 1 - np.mean(refused_ans)) / 2),
        "refusal_recall_by_category": {c: float(np.mean(v)) for c, v in by_category.items()},
    }


def calibrate_threshold(rows: Sequence[Dict[str, Any]], use_vocabulary: bool) -> float:
    """Threshold on the max reranker logit that maximises balanced accuracy on `rows`."""
    scores = sorted({r["max_logit"] for r in rows})
    candidates = [scores[0] - 1.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best_tau, best_score = candidates[0], -1.0
    for tau in candidates:
        outcome = gate_outcomes(
            rows,
            lambda r, tau=tau: (use_vocabulary and bool(r["unknown_terms"])) or r["max_logit"] < tau,
        )
        if outcome["balanced_accuracy"] > best_score:
            best_tau, best_score = tau, outcome["balanced_accuracy"]
    return float(best_tau)


# --------------------------------------------------------------------------- #
def run() -> Dict[str, Any]:
    engine = Engine()
    engine.warm_up()
    dataset = [
        json.loads(line)
        for line in (ROOT / engine.config["paths"]["eval_dataset"]).read_text(encoding="utf-8").splitlines()
    ]
    for row in dataset:
        rules = [(g["doc"], g["terms"], g["section"]) if "section" in g else (g["doc"], g["terms"]) for g in row["gold"]]
        row["gold_ids"] = set(gold_chunk_ids(rules, engine.chunks))

    # ---- run every configuration on every query ------------------------- #
    per_config: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CONFIGS}
    for name, kwargs in CONFIGS.items():
        for row in dataset:
            start = time.perf_counter()
            trace = engine.retrieve(row["query"], **kwargs)
            latency = (time.perf_counter() - start) * 1000.0
            ranked_ids = [chunk.chunk_id for chunk, _ in trace.ranked]
            record = {"id": row["id"], "latency_ms": latency, "ranked_ids": ranked_ids}
            if row["answerable"]:
                record.update(retrieval_metrics(ranked_ids, row["gold_ids"]))
            if name == FULL:
                row["max_logit"] = max((s for _, s in trace.ranked), default=-99.0)
                row["unknown_terms"] = find_unknown_drug_terms(row["query"], engine.bm25.vocabulary, engine.linker)
                row["ranked"] = [(chunk.chunk_id, score) for chunk, score in trace.ranked[:TOP_K_ANSWER]]
            per_config[name].append(record)
        logging.info("finished config: %s", name)

    test = [r for r in dataset if r["split"] == "test"]
    dev = [r for r in dataset if r["split"] == "dev"]
    test_ids = {r["id"] for r in test}
    test_answerable_ids = {r["id"] for r in test if r["answerable"]}
    category_of = {r["id"]: r["category"] for r in dataset}

    # ---- retrieval ablation (test split, answerable queries) ------------ #
    metric_names = ["hit@1", "hit@3", "mrr@10", "ndcg@10"]
    retrieval: Dict[str, Any] = {}
    for name, records in per_config.items():
        scored = [r for r in records if r["id"] in test_answerable_ids]
        latencies = [r["latency_ms"] for r in records if r["id"] in test_ids]
        entry = {m: float(np.mean([r[m] for r in scored])) for m in metric_names}
        entry["hit@3_ci"] = bootstrap_ci([r["hit@3"] for r in scored])
        entry["latency_p50_ms"] = float(np.percentile(latencies, 50))
        entry["latency_p95_ms"] = float(np.percentile(latencies, 95))
        by_cat: Dict[str, List[float]] = {}
        for r in scored:
            by_cat.setdefault(category_of[r["id"]], []).append(r["hit@3"])
        entry["hit@3_by_category"] = {c: float(np.mean(v)) for c, v in by_cat.items()}
        retrieval[name] = entry

    # Paired comparison of the full pipeline against the strongest standard baseline.
    full = {r["id"]: r["hit@3"] for r in per_config[FULL] if r["id"] in test_answerable_ids}
    base = {r["id"]: r["hit@3"] for r in per_config[BASELINE] if r["id"] in test_answerable_ids}
    paired = {
        "baseline": BASELINE,
        "full_only_hits": sum(1 for q in full if full[q] and not base[q]),
        "baseline_only_hits": sum(1 for q in full if base[q] and not full[q]),
        "hit@3_difference_ci": bootstrap_ci([full[q] - base[q] for q in full]),
    }

    # ---- HNSW vs exact search ------------------------------------------- #
    overlaps = []
    for row in test:
        approx = {c.chunk_id for c, _ in engine.dense.search(row["query"], top_k=K)}
        exact = {c.chunk_id for c, _ in engine.dense.search(row["query"], top_k=K, exact=True)}
        overlaps.append(len(approx & exact) / K)

    # ---- evidence gate --------------------------------------------------- #
    tau_combined = calibrate_threshold(dev, use_vocabulary=True)
    tau_relevance = calibrate_threshold(dev, use_vocabulary=False)
    gate = {
        "calibrated_on": f"dev split ({len(dev)} queries)",
        "threshold_combined": tau_combined,
        "threshold_relevance_only": tau_relevance,
        "configured_threshold": engine.config["gate"]["logit_threshold"],
        "auroc_max_logit_test": auroc(
            [r["max_logit"] for r in test if r["answerable"]],
            [r["max_logit"] for r in test if not r["answerable"]],
        ),
        "variants_test": {
            "No gate": gate_outcomes(test, lambda r: False),
            "Relevance threshold only": gate_outcomes(test, lambda r: r["max_logit"] < tau_relevance),
            "Vocabulary check only": gate_outcomes(test, lambda r: bool(r["unknown_terms"])),
            "Combined (shipped)": gate_outcomes(
                test, lambda r: bool(r["unknown_terms"]) or r["max_logit"] < tau_combined
            ),
        },
    }

    # ---- end-to-end outcome with the shipped gate ------------------------ #
    def refused(r: Dict[str, Any]) -> bool:
        return bool(r["unknown_terms"]) or r["max_logit"] < tau_combined

    # A passage is cited only if it individually clears the threshold (core.pipeline.select_passages).
    for r in test:
        cited = {cid for cid, score in r["ranked"] if score >= tau_combined}
        r["gold_in_top3"] = bool(r["gold_ids"] & cited)

    answerable = [r for r in test if r["answerable"]]
    unanswerable = [r for r in test if not r["answerable"]]
    grounded = [float(not refused(r) and r["gold_in_top3"]) for r in answerable]
    end_to_end = {
        "answerable_queries": len(answerable),
        "answered_with_gold_evidence": float(np.mean(grounded)),
        "answered_with_gold_evidence_ci": bootstrap_ci(grounded),
        "answered_without_gold_evidence": float(np.mean([not refused(r) and not r["gold_in_top3"] for r in answerable])),
        "wrongly_refused": float(np.mean([refused(r) for r in answerable])),
        "unanswerable_queries": len(unanswerable),
        "correctly_refused": float(np.mean([refused(r) for r in unanswerable])),
        "answered_despite_no_evidence": [r["query"] for r in unanswerable if not refused(r)],
        "wrongly_refused_queries": [r["query"] for r in answerable if refused(r)],
    }

    import torch

    build_meta = json.loads((ROOT / engine.config["paths"]["index_dir"] / "build_meta.json").read_text())
    return {
        "environment": {
            "device": engine.dense.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "embedding_model": engine.dense.model_name,
            "reranker_model": engine.reranker.model_name,
        },
        "corpus": build_meta,
        "dataset": {
            "total": len(dataset),
            "dev": len(dev),
            "test": len(test),
            "test_answerable": len(answerable),
            "test_unanswerable": len(unanswerable),
        },
        "retrieval_test": retrieval,
        "paired_full_vs_baseline": paired,
        "hnsw_recall@10_vs_exact": float(np.mean(overlaps)),
        "gate": gate,
        "end_to_end_test": end_to_end,
        "inhibitor_inducer_edges": sorted(
            f"{s} {r} {t} ({len(d['chunk_ids'])} passages)"
            for s, t, r, d in engine.kg.graph.edges(keys=True, data=True)
            if r in ("INHIBITS", "INDUCES")
        ),
    }


# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(pair: Sequence[float]) -> str:
    return f"{100 * pair[0]:.0f}–{100 * pair[1]:.0f}%"


def render_markdown(res: Dict[str, Any]) -> str:
    env, ds, corpus = res["environment"], res["dataset"], res["corpus"]
    lines = [
        "# Evaluation results",
        "",
        "Generated by `python eval/run_evaluation.py`. Do not edit by hand.",
        "",
        f"- Corpus: {corpus['labels']} FDA labels, {corpus['passages']:,} passages; knowledge graph with "
        f"{corpus['graph']['nodes']} nodes and {corpus['graph']['edges']:,} edges",
        f"- Queries: {ds['total']} total; {ds['dev']} dev (calibration and debugging), {ds['test']} test "
        f"({ds['test_answerable']} answerable, {ds['test_unanswerable']} unanswerable)",
        f"- Hardware: {env['gpu'] or 'CPU'} ({env['device']}), torch {env['torch']}",
        f"- Models: `{env['embedding_model']}`, `{env['reranker_model']}`",
        "",
        "All numbers below are on the test split. Intervals are 95% bootstrap intervals.",
        "",
        "## Retrieval ablation",
        "",
        f"Answerable test queries (n={ds['test_answerable']}). Hit@k: a gold passage is in the top k. "
        "Latency is retrieval only, per query, over all test queries.",
        "",
        "| Configuration | Hit@1 | Hit@3 | Hit@3 95% CI | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
        "| :-- | --: | --: | :-: | --: | --: | --: | --: |",
    ]
    for name, m in res["retrieval_test"].items():
        label = f"**{name}**" if name == FULL else name
        lines.append(
            f"| {label} | {pct(m['hit@1'])} | {pct(m['hit@3'])} | {ci(m['hit@3_ci'])} | {m['mrr@10']:.3f} | "
            f"{m['ndcg@10']:.3f} | {m['latency_p50_ms']:.0f} | {m['latency_p95_ms']:.0f} |"
        )

    categories = list(res["retrieval_test"][FULL]["hit@3_by_category"])
    lines += ["", "### Hit@3 by question type", "", "| Configuration | " + " | ".join(categories) + " |",
              "| :-- |" + " --: |" * len(categories)]
    for name, m in res["retrieval_test"].items():
        lines.append(f"| {name} | " + " | ".join(pct(m["hit@3_by_category"][c]) for c in categories) + " |")

    p = res["paired_full_vs_baseline"]
    lines += [
        "",
        f"Paired comparison, full pipeline vs. `{p['baseline']}` on Hit@3: the full pipeline answers "
        f"{p['full_only_hits']} queries the baseline misses and misses {p['baseline_only_hits']} the baseline "
        f"answers (difference 95% CI {100 * p['hit@3_difference_ci'][0]:+.0f} to "
        f"{100 * p['hit@3_difference_ci'][1]:+.0f} points).",
        "",
        f"HNSW returns {pct(res['hnsw_recall@10_vs_exact'])} of the exact top-10 neighbours.",
        "",
        "## Evidence gate",
        "",
        f"Threshold calibrated on the {res['gate']['calibrated_on']} by maximising balanced accuracy: "
        f"τ = {res['gate']['threshold_combined']:.2f} for the combined gate "
        f"(τ = {res['gate']['threshold_relevance_only']:.2f} for the relevance-only variant). "
        f"AUROC of the max reranker logit, answerable vs. unanswerable: {res['gate']['auroc_max_logit_test']:.3f}.",
        "",
        "| Gate | Unanswerable refused | 95% CI | Answerable wrongly refused | 95% CI | Balanced acc. |",
        "| :-- | --: | :-: | --: | :-: | --: |",
    ]
    for name, g in res["gate"]["variants_test"].items():
        label = f"**{name}**" if "shipped" in name else name
        lines.append(
            f"| {label} | {pct(g['refusal_recall'])} | {ci(g['refusal_recall_ci'])} | {pct(g['false_refusal_rate'])} | "
            f"{ci(g['false_refusal_rate_ci'])} | {pct(g['balanced_accuracy'])} |"
        )
    neg_categories = list(res["gate"]["variants_test"]["Combined (shipped)"]["refusal_recall_by_category"])
    lines += ["", "### Refusal rate by kind of unanswerable question", "",
              "| Gate | " + " | ".join(neg_categories) + " |", "| :-- |" + " --: |" * len(neg_categories)]
    for name, g in res["gate"]["variants_test"].items():
        if name == "No gate":
            continue
        lines.append(f"| {name} | " + " | ".join(pct(g["refusal_recall_by_category"][c]) for c in neg_categories) + " |")

    e = res["end_to_end_test"]
    lines += [
        "",
        "## End-to-end outcome",
        "",
        f"Answerable queries (n={e['answerable_queries']}):",
        "",
        f"- answered, with a gold passage among those cited (up to {TOP_K_ANSWER}): **{pct(e['answered_with_gold_evidence'])}** "
        f"({ci(e['answered_with_gold_evidence_ci'])})",
        f"- answered, but no gold passage among those cited: {pct(e['answered_without_gold_evidence'])}",
        f"- wrongly refused: {pct(e['wrongly_refused'])}",
        "",
        f"Unanswerable queries (n={e['unanswerable_queries']}): **{pct(e['correctly_refused'])}** refused.",
        "",
    ]
    if e["answered_despite_no_evidence"]:
        lines += ["Unanswerable queries that were answered anyway:", ""]
        lines += [f"- {q}" for q in e["answered_despite_no_evidence"]] + [""]
    if e["wrongly_refused_queries"]:
        lines += ["Answerable queries that were refused:", ""]
        lines += [f"- {q}" for q in e["wrongly_refused_queries"]] + [""]

    lines += [
        "## Knowledge graph: inhibitor and inducer edges",
        "",
        "Every INHIBITS / INDUCES edge extracted from the corpus, listed so the extraction can be "
        "checked against a pharmacology reference.",
        "",
    ]
    lines += [f"- {edge}" for edge in res["inhibitor_inducer_edges"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    results = run()
    (ROOT / "eval/results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT / "eval/RESULTS.md").write_text(render_markdown(results), encoding="utf-8")
    print(render_markdown(results))
