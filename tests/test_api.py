"""API contract tests against a stub engine (no models, no index needed)."""

from types import SimpleNamespace

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_engine
from api.routes import app
from core.entities import EntityLinker
from core.fusion import FusedCandidate
from core.gate import GateDecision
from core.graph import GraphEvidence, GraphPath
from core.pipeline import PipelineResult, RetrievalTrace
from tests.conftest import make_chunk


class StubEngine:
    def __init__(self):
        self.chunks = [make_chunk("sildenafil", "CONTRAINDICATIONS", "Do not use with nitrates.")]
        self.dense = SimpleNamespace(device="cpu")
        self.kg = SimpleNamespace(graph=nx.MultiDiGraph())
        self.generator = SimpleNamespace(mode="extractive")
        self.gate = SimpleNamespace(logit_threshold=0.23)
        self.linker = EntityLinker()

    def answer(self, query: str) -> PipelineResult:
        chunk = self.chunks[0]
        refused = "xanodrol" in query.lower()
        trace = RetrievalTrace(
            query=query,
            entities=self.linker.link(query),
            graph=GraphEvidence(
                paths=[GraphPath("class", [("sildenafil", "CONTRAINDICATED_WITH", "nitrates")], [chunk.chunk_id])],
                chunk_ids=[chunk.chunk_id],
            ),
            channels={},
            fused=[FusedCandidate(chunk, 0.03, {"graph": 1, "sparse": 2})],
            ranked=[(chunk, 5.1)],
            reranked=True,
            timings_ms={"rerank": 100.0},
        )
        gate = GateDecision(False, "unknown_term", 5.1, ["Xanodrol"]) if refused else GateDecision(True, "ok", 5.1)
        return PipelineResult(
            query=query,
            answer="INSUFFICIENT_EVIDENCE" if refused else "Do not use with nitrates. [1]",
            refused=refused,
            gate=gate,
            passages=[] if refused else [(chunk, 5.1)],
            trace=trace,
            answer_mode="none" if refused else "extractive",
            latency_ms=120.0,
        )


@pytest.fixture
def client():
    app.dependency_overrides[get_engine] = StubEngine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["passages"] == 1 and body["answer_mode"] == "extractive"


def test_demo_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "BioMed-GraphRAG" in res.text


def test_answered_query_has_citations_and_trace(client):
    body = client.post("/v1/query", json={"query": "Can I take Viagra with Nitrostat?"}).json()
    assert body["refused"] is False
    assert body["citations"][0]["n"] == 1
    assert body["citations"][0]["retrieved_by"] == {"graph": 1, "sparse": 2}
    assert [e["canonical"] for e in body["entities"]] == ["sildenafil", "nitroglycerin"]
    assert body["graph_paths"][0]["triples"] == [["sildenafil", "CONTRAINDICATED_WITH", "nitrates"]]
    assert body["gate"] == {
        "passed": True, "reason": "ok", "max_score": 5.1, "threshold": 0.23, "unknown_terms": [],
    }


def test_refused_query_has_no_citations(client):
    body = client.post("/v1/query", json={"query": "Is Xanodrol safe with warfarin?"}).json()
    assert body["refused"] is True and body["citations"] == []
    assert body["gate"]["reason"] == "unknown_term" and body["gate"]["unknown_terms"] == ["Xanodrol"]


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "  "}, {"query": "x" * 501}, {}])
def test_invalid_queries_are_rejected(client, payload):
    assert client.post("/v1/query", json=payload).status_code == 422
