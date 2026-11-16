"""
Drug-interaction knowledge graph and path-based evidence retrieval.

Nodes are drugs, drug classes and enzymes/transporters. Every extracted edge carries the
ids of the label passages it was extracted from, so a graph path is not an answer in
itself: it is a pointer to passages that the reranker then has to accept.

The graph earns its place on questions the text retrievers handle badly, where the two
drugs in the question are never named in the same passage:

    class-mediated    sildenafil -[CONTRAINDICATED_WITH]-> nitrates <-[MEMBER_OF]- nitroglycerin
    enzyme-mediated   clarithromycin -[INHIBITS]-> CYP3A4 <-[SUBSTRATE_OF]- simvastatin
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import networkx as nx

CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"
INTERACTS_WITH = "INTERACTS_WITH"
INHIBITS = "INHIBITS"
INDUCES = "INDUCES"
SUBSTRATE_OF = "SUBSTRATE_OF"
MEMBER_OF = "MEMBER_OF"

RELATIONS = {CONTRAINDICATED_WITH, INTERACTS_WITH, INHIBITS, INDUCES, SUBSTRATE_OF, MEMBER_OF}

Triple = Tuple[str, str, str]  # (source, relation, target)


@dataclass
class GraphPath:
    kind: str  # direct | class | enzyme | profile
    triples: List[Triple]
    chunk_ids: List[str]

    def render(self) -> str:
        return "  ;  ".join(f"({s}) -[{r}]-> ({t})" for s, r, t in self.triples)


@dataclass
class GraphEvidence:
    paths: List[GraphPath] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)  # ranked, de-duplicated

    @property
    def triples(self) -> List[str]:
        seen: Dict[str, None] = {}
        for path in self.paths:
            for s, r, t in path.triples:
                seen.setdefault(f"({s}) -[{r}]-> ({t})")
        return list(seen)


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def add_node(self, node_id: str, kind: str) -> None:
        if node_id not in self.graph:
            self.graph.add_node(node_id, kind=kind)

    def add_edge(
        self, source: str, relation: str, target: str, chunk_id: str | None = None
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation: {relation}")
        if source == target:
            return
        data = self.graph.get_edge_data(source, target, key=relation)
        if data is None:
            self.graph.add_edge(source, target, key=relation, chunk_ids=[])
            data = self.graph.get_edge_data(source, target, key=relation)
        if chunk_id and chunk_id not in data["chunk_ids"]:
            data["chunk_ids"].append(chunk_id)

    def edge_chunks(self, source: str, relation: str, target: str) -> List[str]:
        data = self.graph.get_edge_data(source, target, key=relation)
        return list(data["chunk_ids"]) if data else []

    def has_edge(self, source: str, relation: str, target: str) -> bool:
        return self.graph.has_edge(source, target, key=relation)

    def stats(self) -> Dict[str, Any]:
        by_relation: Dict[str, int] = {}
        for _, _, relation in self.graph.edges(keys=True):
            by_relation[relation] = by_relation.get(relation, 0) + 1
        by_kind: Dict[str, int] = {}
        for _, kind in self.graph.nodes(data="kind"):
            by_kind[kind] = by_kind.get(kind, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "nodes_by_kind": by_kind,
            "edges_by_relation": by_relation,
        }

    # ------------------------------------------------------------------ #
    # Query-time retrieval
    # ------------------------------------------------------------------ #
    def _direct_paths(self, a: str, b: str) -> List[GraphPath]:
        paths: List[GraphPath] = []
        for relation in (CONTRAINDICATED_WITH, INTERACTS_WITH, INHIBITS, INDUCES, SUBSTRATE_OF):
            for src, dst in ((a, b), (b, a)):
                if self.has_edge(src, relation, dst):
                    paths.append(
                        GraphPath("direct", [(src, relation, dst)], self.edge_chunks(src, relation, dst))
                    )
        return paths

    def _class_paths(self, a: str, b: str) -> List[GraphPath]:
        """b's label names a class that a belongs to (and the reverse)."""
        paths: List[GraphPath] = []
        for member, other in ((a, b), (b, a)):
            if member not in self.graph:
                continue
            for _, class_id, relation in self.graph.out_edges(member, keys=True):
                if relation != MEMBER_OF:
                    continue
                for rel in (CONTRAINDICATED_WITH, INTERACTS_WITH):
                    if self.has_edge(other, rel, class_id):
                        paths.append(
                            GraphPath(
                                "class",
                                [(other, rel, class_id), (member, MEMBER_OF, class_id)],
                                self.edge_chunks(other, rel, class_id),
                            )
                        )
        return paths

    def _enzyme_paths(self, a: str, b: str) -> List[GraphPath]:
        """One drug is a substrate of an enzyme/transporter that the other inhibits or induces."""
        paths: List[GraphPath] = []
        for victim, perpetrator in ((a, b), (b, a)):
            if victim not in self.graph or perpetrator not in self.graph:
                continue
            for _, enzyme, relation in self.graph.out_edges(victim, keys=True):
                if relation != SUBSTRATE_OF:
                    continue
                for effect in (INHIBITS, INDUCES):
                    if self.has_edge(perpetrator, effect, enzyme):
                        chunks = _interleave(
                            self.edge_chunks(victim, SUBSTRATE_OF, enzyme),
                            self.edge_chunks(perpetrator, effect, enzyme),
                        )
                        paths.append(
                            GraphPath(
                                "enzyme",
                                [(perpetrator, effect, enzyme), (victim, SUBSTRATE_OF, enzyme)],
                                chunks,
                            )
                        )
        return paths

    def _profile_paths(self, node: str) -> List[GraphPath]:
        """Pharmacokinetic profile of a single drug: its enzyme relations."""
        if node not in self.graph:
            return []
        paths = []
        for _, target, relation in self.graph.out_edges(node, keys=True):
            if relation in (SUBSTRATE_OF, INHIBITS, INDUCES):
                paths.append(
                    GraphPath("profile", [(node, relation, target)], self.edge_chunks(node, relation, target))
                )
        return paths

    def retrieve(
        self,
        entity_ids: Sequence[str],
        max_chunks: int = 12,
        chunks_per_edge: int = 3,
    ) -> GraphEvidence:
        """Finds paths of length <= 2 connecting the query entities.

        Paths are ranked direct > class-mediated > enzyme-mediated. Within a path,
        passages from the labels of the drugs named in the query come first.
        """
        nodes = [e for e in dict.fromkeys(entity_ids) if e in self.graph]
        paths: List[GraphPath] = []
        for a, b in combinations(nodes, 2):
            paths += self._direct_paths(a, b)
        for a, b in combinations(nodes, 2):
            paths += self._class_paths(a, b)
        for a, b in combinations(nodes, 2):
            paths += self._enzyme_paths(a, b)
        if not paths:
            for node in nodes:
                paths += self._profile_paths(node)

        query_docs = {n.replace(" ", "_") for n in nodes}
        ranked: Dict[str, None] = {}
        for path in paths:
            preferred = sorted(
                path.chunk_ids, key=lambda cid: cid.split("::")[0] not in query_docs
            )
            path.chunk_ids = preferred[:chunks_per_edge]
            for chunk_id in path.chunk_ids:
                ranked.setdefault(chunk_id)
        return GraphEvidence(paths=paths, chunk_ids=list(ranked)[:max_chunks])

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        payload = {
            "nodes": [{"id": n, "kind": k} for n, k in self.graph.nodes(data="kind")],
            "edges": [
                {"source": s, "relation": r, "target": t, "chunk_ids": d["chunk_ids"]}
                for s, t, r, d in self.graph.edges(keys=True, data=True)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        payload = json.loads(path.read_text(encoding="utf-8"))
        kg = cls()
        for node in payload["nodes"]:
            kg.add_node(node["id"], node["kind"])
        for edge in payload["edges"]:
            kg.graph.add_edge(
                edge["source"], edge["target"], key=edge["relation"], chunk_ids=edge["chunk_ids"]
            )
        return kg


def _interleave(*lists: Iterable[str]) -> List[str]:
    """Round-robin merge so each edge of a path contributes its best passage first."""
    out: Dict[str, None] = {}
    iterators = [iter(lst) for lst in lists]
    while iterators:
        for it in list(iterators):
            try:
                out.setdefault(next(it))
            except StopIteration:
                iterators.remove(it)
    return list(out)
