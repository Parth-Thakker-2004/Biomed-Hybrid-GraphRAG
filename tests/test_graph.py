import pytest

from core.graph import (
    CONTRAINDICATED_WITH,
    INHIBITS,
    INTERACTS_WITH,
    MEMBER_OF,
    SUBSTRATE_OF,
    KnowledgeGraph,
)


@pytest.fixture
def kg() -> KnowledgeGraph:
    g = KnowledgeGraph()
    for node, kind in [("sildenafil", "DRUG"), ("nitroglycerin", "DRUG"), ("simvastatin", "DRUG"),
                       ("clarithromycin", "DRUG"), ("warfarin", "DRUG"), ("fluconazole", "DRUG"),
                       ("nitrates", "CLASS"), ("CYP3A4", "ENZYME"), ("CYP2C9", "ENZYME")]:
        g.add_node(node, kind)
    g.add_edge("sildenafil", CONTRAINDICATED_WITH, "nitrates", "sildenafil::contraindications::000")
    g.add_edge("nitroglycerin", MEMBER_OF, "nitrates")
    g.add_edge("simvastatin", SUBSTRATE_OF, "CYP3A4", "simvastatin::pharmacokinetics::000")
    g.add_edge("clarithromycin", INHIBITS, "CYP3A4", "clarithromycin::drug_interactions::000")
    g.add_edge("warfarin", INTERACTS_WITH, "fluconazole", "warfarin::drug_interactions::003")
    g.add_edge("warfarin", SUBSTRATE_OF, "CYP2C9", "warfarin::pharmacokinetics::001")
    return g


def test_class_mediated_path(kg):
    evidence = kg.retrieve(["sildenafil", "nitroglycerin"])
    assert [p.kind for p in evidence.paths] == ["class"]
    assert evidence.paths[0].triples == [
        ("sildenafil", CONTRAINDICATED_WITH, "nitrates"), ("nitroglycerin", MEMBER_OF, "nitrates"),
    ]
    assert evidence.chunk_ids == ["sildenafil::contraindications::000"]


def test_enzyme_mediated_path(kg):
    evidence = kg.retrieve(["clarithromycin", "simvastatin"])
    assert [p.kind for p in evidence.paths] == ["enzyme"]
    assert set(evidence.chunk_ids) == {
        "simvastatin::pharmacokinetics::000", "clarithromycin::drug_interactions::000",
    }
    assert evidence.triples == [
        "(clarithromycin) -[INHIBITS]-> (CYP3A4)", "(simvastatin) -[SUBSTRATE_OF]-> (CYP3A4)",
    ]


def test_direct_path_and_argument_order_do_not_matter(kg):
    a = kg.retrieve(["warfarin", "fluconazole"])
    b = kg.retrieve(["fluconazole", "warfarin"])
    assert a.chunk_ids == b.chunk_ids == ["warfarin::drug_interactions::003"]
    assert a.paths[0].kind == "direct"


def test_single_drug_returns_its_enzyme_profile(kg):
    evidence = kg.retrieve(["warfarin"])
    assert [p.kind for p in evidence.paths] == ["profile"]
    assert evidence.chunk_ids == ["warfarin::pharmacokinetics::001"]


def test_unconnected_drugs_fall_back_to_enzyme_profiles(kg):
    # No path links sildenafil and warfarin, so only single-drug profiles are returned.
    evidence = kg.retrieve(["sildenafil", "warfarin"])
    assert {p.kind for p in evidence.paths} == {"profile"}
    assert evidence.chunk_ids == ["warfarin::pharmacokinetics::001"]


def test_unknown_entities_are_ignored(kg):
    assert kg.retrieve(["xanodrol"]).paths == []
    assert kg.retrieve([]).paths == []


def test_evidence_is_merged_and_capped(kg):
    for i in range(10):
        kg.add_edge("warfarin", INTERACTS_WITH, "fluconazole", f"warfarin::drug_interactions::{i:03d}")
    evidence = kg.retrieve(["warfarin", "fluconazole"], chunks_per_edge=3)
    assert len(evidence.chunk_ids) == 3


def test_save_and_load_round_trip(kg, tmp_path):
    kg.save(tmp_path / "graph.json")
    loaded = KnowledgeGraph.load(tmp_path / "graph.json")
    assert loaded.stats() == kg.stats()
    assert loaded.retrieve(["sildenafil", "nitroglycerin"]).chunk_ids == ["sildenafil::contraindications::000"]


def test_unknown_relation_is_rejected(kg):
    with pytest.raises(ValueError):
        kg.add_edge("warfarin", "CURES", "fluconazole")
