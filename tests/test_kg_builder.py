"""Relation-extraction rules, including the false positives found while building them."""

from core.graph import CONTRAINDICATED_WITH, INDUCES, INHIBITS, INTERACTS_WITH, MEMBER_OF, SUBSTRATE_OF
from core.kg_builder import build_graph, extract_triples
from tests.conftest import make_chunk


def triples(linker, label_drug, section, text):
    return set(extract_triples(make_chunk(label_drug, section, text), label_drug, linker))


def test_section_determines_relation(linker):
    text = "Do not use with organic nitrates or riociguat."
    assert triples(linker, "sildenafil", "CONTRAINDICATIONS", text) == {
        ("sildenafil", CONTRAINDICATED_WITH, "nitrates"),
        ("sildenafil", CONTRAINDICATED_WITH, "riociguat"),
    }
    assert ("sildenafil", INTERACTS_WITH, "riociguat") in triples(linker, "sildenafil", "DRUG INTERACTIONS", text)
    assert triples(linker, "sildenafil", "INDICATIONS AND USAGE", text) == set()


def test_negated_interactions_are_skipped(linker):
    text = "No clinically significant interaction was observed with digoxin or warfarin."
    assert triples(linker, "sildenafil", "DRUG INTERACTIONS", text) == set()


def test_own_class_is_not_an_interaction(linker):
    text = "Do not combine with other PDE5 inhibitors or with ritonavir."
    assert triples(linker, "sildenafil", "DRUG INTERACTIONS", text) == {
        ("sildenafil", INTERACTS_WITH, "ritonavir"),
    }


def test_inhibitor_enumeration(linker):
    text = "Strong CYP3A4 inhibitors (e.g., ketoconazole, itraconazole) and CYP3A4 inducers such as rifampin alter exposure."
    found = triples(linker, "apixaban", "PHARMACOKINETICS", text)
    assert {("ketoconazole", INHIBITS, "CYP3A4"), ("itraconazole", INHIBITS, "CYP3A4"),
            ("rifampin", INDUCES, "CYP3A4")} <= found
    assert ("rifampin", INHIBITS, "CYP3A4") not in found


def test_label_drug_is_victim_not_member_of_enumeration(linker):
    # Regression: tadalafil was extracted as a CYP3A4 inhibitor from this sentence.
    text = ("Other CYP3A4 inhibitors, such as erythromycin and itraconazole, "
            "would likely increase tadalafil exposure.")
    found = triples(linker, "tadalafil", "PHARMACOKINETICS", text)
    assert ("erythromycin", INHIBITS, "CYP3A4") in found
    assert ("tadalafil", INHIBITS, "CYP3A4") not in found


def test_appositive(linker):
    text = "Coadministration with ketoconazole, a strong CYP3A4 inhibitor, increased exposure."
    assert ("ketoconazole", INHIBITS, "CYP3A4") in triples(linker, "apixaban", "PHARMACOKINETICS", text)


def test_active_verbs_bind_to_the_right_enzyme(linker):
    # Regression: both enzymes were attributed to "inhibitor".
    text = "Topiramate is an inhibitor of CYP2C19 and an inducer of CYP3A4."
    found = triples(linker, "topiramate", "PHARMACOKINETICS", text)
    assert ("topiramate", INHIBITS, "CYP2C19") in found
    assert ("topiramate", INDUCES, "CYP3A4") in found
    assert ("topiramate", INHIBITS, "CYP3A4") not in found


def test_negated_or_weak_inhibition_is_skipped(linker):
    assert triples(linker, "apixaban", "PHARMACOKINETICS", "Apixaban does not inhibit CYP3A4.") == set()
    assert triples(linker, "sildenafil", "PHARMACOKINETICS", "Sildenafil is a weak inhibitor of CYP1A2.") == set()


def test_substrate_patterns(linker):
    assert ("sildenafil", SUBSTRATE_OF, "CYP3A4") in triples(
        linker, "sildenafil", "PHARMACOKINETICS",
        "Sildenafil is cleared predominantly by the CYP3A4 hepatic isoenzyme.")
    assert ("warfarin", SUBSTRATE_OF, "CYP2C9") in triples(
        linker, "warfarin", "PHARMACOKINETICS",
        "The metabolism of warfarin is mediated mainly by CYP2C9.")
    assert ("digoxin", SUBSTRATE_OF, "P-gp") in triples(
        linker, "clarithromycin", "DRUG INTERACTIONS",
        "Digoxin is a substrate for P-glycoprotein and clarithromycin is known to inhibit it.")


def test_substrate_subject_must_be_the_drug(linker):
    # Regression: fluconazole was extracted as a CYP2C9 substrate from this sentence.
    text = ("Patients treated with fluconazole who also receive drugs metabolized through "
            "CYP2C9 should be monitored.")
    assert ("fluconazole", SUBSTRATE_OF, "CYP2C9") not in triples(linker, "fluconazole", "DRUG INTERACTIONS", text)


def test_build_graph_requires_support_for_inhibitor_edges(linker):
    once = "Strong CYP3A4 inhibitors (e.g., ketoconazole) increase exposure."
    chunks = [
        make_chunk("apixaban", "PHARMACOKINETICS", once, 0),
        make_chunk("rivaroxaban", "PHARMACOKINETICS", once, 0),
        make_chunk("apixaban", "PHARMACOKINETICS", "CYP3A4 inhibitors (e.g., diltiazem) had a small effect.", 1),
    ]
    kg = build_graph(chunks, {"apixaban": "apixaban", "rivaroxaban": "rivaroxaban"}, linker)
    assert kg.has_edge("ketoconazole", INHIBITS, "CYP3A4")  # two passages
    assert not kg.has_edge("diltiazem", INHIBITS, "CYP3A4")  # one passage
    assert kg.has_edge("apixaban", MEMBER_OF, "anticoagulants")
