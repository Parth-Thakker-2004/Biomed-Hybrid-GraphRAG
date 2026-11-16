"""
End-to-end query pipeline.

    query -> entity linking -> [BM25 | dense | graph paths] -> RRF -> cross-encoder
          -> evidence gate -> grounded answer with citations

`retrieve` exposes every stage as a switch so the evaluation can run ablations through
exactly the code path that serves requests.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from core.chunking import ClinicalChunk
from core.config import load_config, resolve
from core.dense import DenseRetriever
from core.entities import DRUG, Entity, EntityLinker
from core.fusion import FusedCandidate, ReciprocalRankFusion
from core.gate import EvidenceGate, GateDecision
from core.generator import AnswerGenerator
from core.graph import GraphEvidence, KnowledgeGraph
from core.reranker import CrossEncoderReranker
from core.sparse import BiomedicalBM25

logger = logging.getLogger(__name__)

ALL_CHANNELS = ("dense", "sparse", "graph")


@dataclass
class RetrievalTrace:
    query: str
    entities: List[Entity]
    graph: GraphEvidence
    channels: Dict[str, List[ClinicalChunk]]
    fused: List[FusedCandidate]
    ranked: List[Tuple[ClinicalChunk, float]]  # final order; score is the logit if reranked
    reranked: bool
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: str
    answer: str
    refused: bool
    gate: GateDecision
    passages: List[Tuple[ClinicalChunk, float]]
    trace: RetrievalTrace
    answer_mode: str
    latency_ms: float


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_chunks(path: Path) -> List[ClinicalChunk]:
    with path.open("r", encoding="utf-8") as f:
        return [ClinicalChunk.from_dict(json.loads(line)) for line in f]


class Engine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        index_dir = resolve(self.config["paths"]["index_dir"])
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run: python scripts/build_indexes.py"
            )

        cfg = self.config
        self.chunks = load_chunks(index_dir / "chunks.jsonl")
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}

        self.linker = EntityLinker()
        self.kg = KnowledgeGraph.load(index_dir / "graph.json")
        self.bm25 = BiomedicalBM25(k1=cfg["sparse"]["k1"], b=cfg["sparse"]["b"])
        self.bm25.index_chunks(self.chunks)
        self.dense = DenseRetriever.load(index_dir, self.chunks, device=cfg["dense"]["device"])
        self.fusion = ReciprocalRankFusion(k=cfg["fusion"]["rrf_k"], weights=cfg["fusion"]["weights"])
        self.reranker = CrossEncoderReranker(
            model_name=cfg["reranker"]["model_name"],
            max_length=cfg["reranker"]["max_length"],
            batch_size=cfg["reranker"]["batch_size"],
            device=cfg["reranker"]["device"],
        )
        self.gate = EvidenceGate(
            logit_threshold=cfg["gate"]["logit_threshold"],
            vocabulary=self.bm25.vocabulary,
            linker=self.linker,
            refuse_on_unknown_drug=cfg["gate"]["refuse_on_unknown_drug"],
        )
        self.generator = AnswerGenerator(
            gemini_model=cfg["generator"]["gemini_model"],
            openai_model=cfg["generator"]["openai_model"],
            temperature=cfg["generator"]["temperature"],
            max_tokens=cfg["generator"]["max_tokens"],
        )
        logger.info(
            "Engine ready: %d passages, %d graph nodes, %d graph edges",
            len(self.chunks), self.kg.graph.number_of_nodes(), self.kg.graph.number_of_edges(),
        )

    def warm_up(self) -> None:
        """Loads both models so the first real request does not pay for it."""
        self.retrieve("warfarin interactions")

    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        channels: Sequence[str] = ALL_CHANNELS,
        rerank: bool = True,
        normalize: bool = True,
        entity_filter: bool = True,
        pool_size: int | None = None,
        exact_dense: bool = False,
    ) -> RetrievalTrace:
        """Runs retrieval with any subset of stages enabled.

        normalize      rewrite brand names / aliases to canonical names before searching
        entity_filter  keep only passages from, or mentioning, a drug named in the query
        """
        cfg = self.config
        per_channel = cfg["fusion"]["candidates_per_channel"]
        pool_size = pool_size or cfg["fusion"]["pool_size"]
        timings: Dict[str, float] = {}
        results: Dict[str, List[ClinicalChunk]] = {}

        t = time.perf_counter()
        entities = self.linker.link(query)
        original_query = query
        if normalize:
            query = self.linker.normalize_query(query)
        graph = GraphEvidence()
        if "graph" in channels:
            graph = self.kg.retrieve(
                [e.canonical for e in entities],
                max_chunks=cfg["graph"]["max_chunks"],
                chunks_per_edge=cfg["graph"]["chunks_per_edge"],
            )
            results["graph"] = [self.chunk_by_id[c] for c in graph.chunk_ids if c in self.chunk_by_id]
        timings["graph"] = _elapsed_ms(t)

        if "sparse" in channels:
            t = time.perf_counter()
            results["sparse"] = [c for c, _ in self.bm25.search(query, top_k=per_channel)]
            timings["sparse"] = _elapsed_ms(t)

        if "dense" in channels:
            t = time.perf_counter()
            results["dense"] = [
                c for c, _ in self.dense.search(query, top_k=per_channel, exact=exact_dense)
            ]
            timings["dense"] = _elapsed_ms(t)

        t = time.perf_counter()
        fused = self.fusion.fuse(results, top_n=3 * per_channel)
        if entity_filter:
            fused = self._filter_by_query_drugs(fused, entities)
        fused = fused[:pool_size]
        timings["fusion"] = _elapsed_ms(t)

        if rerank:
            t = time.perf_counter()
            ranked = self.reranker.rerank(query, [c.chunk for c in fused])
            timings["rerank"] = _elapsed_ms(t)
        else:
            ranked = [(c.chunk, c.rrf_score) for c in fused]

        return RetrievalTrace(original_query, entities, graph, results, fused, ranked, rerank, timings)

    @staticmethod
    def _filter_by_query_drugs(
        fused: List[FusedCandidate], entities: Sequence[Entity]
    ) -> List[FusedCandidate]:
        """Drops passages that are about a different drug than the ones asked about.

        A general-domain cross-encoder scores "MAOIs are contraindicated with
        nortriptyline" highly for a question about tramadol and MAOIs: right topic,
        wrong drug. A passage is kept only if it comes from the label of a drug named
        in the query or mentions one by name.
        """
        drugs = {e.canonical for e in entities if e.kind == DRUG}
        if not drugs:
            return fused
        doc_ids = {d.replace(" ", "_") for d in drugs}
        return [
            c for c in fused
            if c.chunk.doc_id in doc_ids or any(d in c.chunk.content.lower() for d in drugs)
        ]

    def answer(self, query: str) -> PipelineResult:
        start = time.perf_counter()
        trace = self.retrieve(query)
        decision = self.gate.decide(query, [score for _, score in trace.ranked])

        if decision.passed:
            passages = select_passages(
                trace.ranked, self.gate.logit_threshold, self.config["reranker"]["top_k"]
            )
            t = time.perf_counter()
            generated = self.generator.generate(query, passages, trace.graph.triples)
            trace.timings_ms["generate"] = _elapsed_ms(t)
            text, mode = generated.text, generated.mode
            # An LLM may itself decide the passages are insufficient.
            refused = text.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        else:
            passages, mode, refused = [], "none", True
            text = self.config["gate"]["refusal_message"].strip()
            if decision.unknown_terms:
                text += f" Not found in the corpus: {', '.join(decision.unknown_terms)}."

        return PipelineResult(
            query=query,
            answer=text,
            refused=refused,
            gate=decision,
            passages=[] if refused else passages,
            trace=trace,
            answer_mode=mode,
            latency_ms=_elapsed_ms(start),
        )


def select_passages(
    ranked: Sequence[Tuple[ClinicalChunk, float]], threshold: float, top_k: int
) -> List[Tuple[ClinicalChunk, float]]:
    """Passages handed to the generator: the top-k that individually clear the gate threshold.

    The gate guarantees the best passage clears it; the others are only cited if they do
    too, so a strong first passage cannot carry irrelevant ones into the answer.
    """
    return [(chunk, score) for chunk, score in ranked[:top_k] if score >= threshold]


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
