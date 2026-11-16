"""
HTTP API.

    GET  /            demo UI
    POST /v1/query    answer a question, with citations and the full retrieval trace
    GET  /health      readiness and corpus statistics

Run with:  uvicorn api.routes:app --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_engine, inference_lock
from api.schemas import (
    CandidateOut,
    Citation,
    EntityOut,
    GateOut,
    GraphPathOut,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from core.pipeline import Engine, PipelineResult

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load indexes and models before accepting traffic, so the first request is fast.
    if get_engine not in app.dependency_overrides:
        get_engine()
    yield


app = FastAPI(
    title="BioMed-GraphRAG",
    description="Evidence-grounded drug-interaction question answering over FDA labels.",
    version="1.0.0",
    lifespan=lifespan,
)


def to_response(result: PipelineResult, threshold: float) -> QueryResponse:
    trace = result.trace
    ranks = {c.chunk.chunk_id: c.ranks for c in trace.fused}
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        refused=result.refused,
        answer_mode=result.answer_mode,
        gate=GateOut(
            passed=result.gate.passed,
            reason=result.gate.reason,
            max_score=result.gate.max_score,
            threshold=threshold,
            unknown_terms=result.gate.unknown_terms,
        ),
        citations=[
            Citation(
                n=i,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section,
                text=chunk.content,
                score=round(score, 3),
                url=chunk.url,
                retrieved_by=ranks.get(chunk.chunk_id, {}),
            )
            for i, (chunk, score) in enumerate(result.passages, start=1)
        ],
        entities=[EntityOut(text=e.text, canonical=e.canonical, kind=e.kind) for e in trace.entities],
        graph_paths=[
            GraphPathOut(kind=p.kind, triples=[list(t) for t in p.triples]) for p in trace.graph.paths[:8]
        ],
        candidates=[
            CandidateOut(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section,
                ranks=ranks.get(chunk.chunk_id, {}),
                rerank_score=round(score, 3),
            )
            for chunk, score in trace.ranked[:10]
        ],
        timings_ms={k: round(v, 1) for k, v in trace.timings_ms.items()},
        latency_ms=round(result.latency_ms, 1),
    )


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(engine: Engine = Depends(get_engine)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=engine.dense.device,
        labels=len({c.doc_id for c in engine.chunks}),
        passages=len(engine.chunks),
        graph_nodes=engine.kg.graph.number_of_nodes(),
        graph_edges=engine.kg.graph.number_of_edges(),
        answer_mode=engine.generator.mode,
    )


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest, engine: Engine = Depends(get_engine)) -> QueryResponse:
    text = request.query.strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="Query is empty.")
    try:
        with inference_lock:
            result = engine.answer(text)
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal error while answering the query.")
    return to_response(result, engine.gate.logit_threshold)
