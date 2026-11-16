"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="A question about drug interactions, contraindications or metabolism.",
        examples=["Can I take Viagra with Nitrostat?"],
    )


class EntityOut(BaseModel):
    text: str
    canonical: str
    kind: str


class GateOut(BaseModel):
    passed: bool
    reason: str = Field(description="ok | low_relevance | unknown_term | no_candidates")
    max_score: float | None = Field(None, description="Best cross-encoder logit in the candidate pool")
    threshold: float
    unknown_terms: List[str] = []


class Citation(BaseModel):
    n: int = Field(description="Citation number used in the answer text, e.g. [1]")
    chunk_id: str
    title: str
    section: str
    text: str
    score: float = Field(description="Cross-encoder logit")
    url: str = Field(description="DailyMed page for the source label")
    retrieved_by: Dict[str, int] = Field(description="Channel -> rank at which it retrieved this passage")


class GraphPathOut(BaseModel):
    kind: str = Field(description="direct | class | enzyme | profile")
    triples: List[List[str]]


class CandidateOut(BaseModel):
    chunk_id: str
    title: str
    section: str
    ranks: Dict[str, int]
    rerank_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    answer_mode: str = Field(description="extractive | gemini | openai | none")
    gate: GateOut
    citations: List[Citation]
    entities: List[EntityOut]
    graph_paths: List[GraphPathOut]
    candidates: List[CandidateOut]
    timings_ms: Dict[str, float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    labels: int
    passages: int
    graph_nodes: int
    graph_edges: int
    answer_mode: str
