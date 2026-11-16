"""
Rule-based relation extraction from FDA label passages.

Label structure does most of the work. A drug named in the DRUG INTERACTIONS section of
warfarin's label interacts with warfarin; that needs no model. The rules below operate on
single sentences, skip negated statements ("no clinically significant interaction"), and
prefer missing an edge to inventing one. Extraction quality is measured, not assumed:
see the knowledge-graph audit in eval/RESULTS.md.

Rules
    R1  label drug D, section CONTRAINDICATIONS, sentence names X   -> D CONTRAINDICATED_WITH X
    R2  label drug D, section DRUG INTERACTIONS / BOXED WARNING     -> D INTERACTS_WITH X
    R3  "<enzyme> inhibitors (e.g., X, Y)" / "X, a strong <enzyme> inhibitor"
                                                                    -> X INHIBITS enzyme
        (same for inducers -> INDUCES and substrates -> SUBSTRATE_OF)
    R4  "X inhibits <enzyme>" / "X is an inducer of <enzyme>"       -> X INHIBITS / INDUCES enzyme
    R5  sentence names D and an enzyme with a metabolism verb       -> D SUBSTRATE_OF enzyme
    R6  lexicon class membership                                    -> X MEMBER_OF class
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from core.chunking import ClinicalChunk, split_sentences
from core.entities import CLASS, DRUG, ENZYME, Entity, EntityLinker
from core.graph import (
    CONTRAINDICATED_WITH,
    INDUCES,
    INHIBITS,
    INTERACTS_WITH,
    MEMBER_OF,
    SUBSTRATE_OF,
    KnowledgeGraph,
    Triple,
)

SECTION_RELATION = {
    "CONTRAINDICATIONS": CONTRAINDICATED_WITH,
    "DRUG INTERACTIONS": INTERACTS_WITH,
    "BOXED WARNING": INTERACTS_WITH,
}

# "no clinically significant interaction", "did not affect", "had no effect on" ...
_NO_INTERACTION_RE = re.compile(
    r"\b(?:no|not|without|neither|nor|lack of|unlikely)\b[^.;]{0,50}?"
    r"\b(?:interact|effect|affect|alter|chang|influenc|differen|adjust)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|n't|neither|nor|unlikely|weak|weakly|mild|mildly|minimal|minimally|negligible)\b",
    re.IGNORECASE,
)
# Where an unparenthesised "such as A, B and C ..." enumeration stops.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"\b(?:would|may|might|can|could|will|is|are|was|were|has|have|had|should|did|does|"
    r"increase[sd]?|decrease[sd]?|reduce[sd]?|results?|resulted|since|because|when|while|"
    r"in patients|with|on|the)\b",
    re.IGNORECASE,
)
# Words between a drug and a metabolism verb that mean the drug is not its subject:
# "fluconazole ... drugs metabolized through CYP2C9".
_SUBJECT_BLOCK_RE = re.compile(
    r"\b(?:drugs?|agents?|medications?|products?|substrates|inhibit\w*|induc\w*|other|with|and)\b",
    re.IGNORECASE,
)

_ROLE_AFTER_ENZYME_RE = re.compile(
    r"^\s*(?:\([^)]{0,25}\)\s*)?(?:enzyme\s+|isoenzyme\s+|isozyme\s+)?(inhibitor|inducer|substrate)s?\b",
    re.IGNORECASE,
)
_ENUMERATION_RE = re.compile(r"^\s*[,:]?\s*(?:\(|such as|including|like|e\.g\.)", re.IGNORECASE)
_APPOSITIVE_RE = re.compile(
    r",?\s+(?:an?|the)\s+(?:known\s+)?(?:strong|potent|moderate|selective|specific|combined)?\s*"
    r"(?:and\s+\S+\s+)?$",
    re.IGNORECASE,
)
_ACTIVE_VERB_RE = re.compile(
    r"\b(?:(inhibits|is an? (?:[\w-]+ ){0,3}inhibitor of|inhibitor of)"
    r"|(induces|is an? (?:[\w-]+ ){0,3}inducer of|inducer of))\b",
    re.IGNORECASE,
)
_METABOLISM_RE = re.compile(
    r"\b(?:metaboli[sz]ed|metabolism|substrate|biotransform\w*|cleared|clearance|"
    r"hydroxylat\w+|demethylat\w+|oxidi[sz]ed|oxidation)\b",
    re.IGNORECASE,
)

ROLE_RELATION = {"inhibitor": INHIBITS, "inducer": INDUCES, "substrate": SUBSTRATE_OF}
MAX_ENUMERATION_SPAN = 260  # characters after "<enzyme> inhibitors (" searched for drug names


def _enzyme_role_edges(
    sentence: str, entities: Sequence[Entity], label_drug: str
) -> List[Triple]:
    """R3: enumerations and appositives around '<enzyme> inhibitor(s)'.

    The label's own drug is never taken from an enumeration: in "CYP3A4 inhibitors such
    as ketoconazole increase tadalafil exposure", tadalafil is the victim, not a member
    of the list.
    """
    edges: List[Triple] = []
    anchors: List[Tuple[Entity, str, int]] = []  # (enzyme, relation, end of role word)
    for ent in entities:
        if ent.kind != ENZYME:
            continue
        match = _ROLE_AFTER_ENZYME_RE.match(sentence[ent.end :])
        if match:
            anchors.append((ent, ROLE_RELATION[match.group(1).lower()], ent.end + match.end()))

    for i, (enzyme, relation, role_end) in enumerate(anchors):
        # "..., a strong CYP3A4 inhibitor" directly after a drug name.
        before = sentence[: enzyme.start]
        appositive = _APPOSITIVE_RE.search(before)
        if appositive:
            preceding = [e for e in entities if e.kind == DRUG and e.end == appositive.start()]
            for drug in preceding:
                edges.append((drug.canonical, relation, enzyme.canonical))

        # "CYP3A4 inhibitors (e.g., ketoconazole, itraconazole)"
        if not _ENUMERATION_RE.match(sentence[role_end:]):
            continue
        next_anchor_start = anchors[i + 1][0].start if i + 1 < len(anchors) else len(sentence)
        limit = min(next_anchor_start, role_end + MAX_ENUMERATION_SPAN)
        closing = sentence.find(")", role_end)
        if sentence[role_end:].lstrip().startswith("(") and closing != -1:
            limit = min(limit, closing)
        else:
            clause = _CLAUSE_BOUNDARY_RE.search(sentence, role_end)
            if clause:
                limit = min(limit, clause.start())
        for drug in entities:
            if drug.kind == DRUG and drug.canonical != label_drug and role_end <= drug.start < limit:
                edges.append((drug.canonical, relation, enzyme.canonical))
    return edges


def _substrate_edges(sentence: str, entities: Sequence[Entity]) -> List[Triple]:
    """R5: '<drug> is metabolized by <enzyme>' / 'metabolism of <drug> ... <enzyme>'."""
    edges: List[Triple] = []
    drugs = [e for e in entities if e.kind == DRUG]
    enzymes = [e for e in entities if e.kind == ENZYME]
    for verb in _METABOLISM_RE.finditer(sentence):
        subject = None
        # "the metabolism of sildenafil is mediated by CYP3A4"
        of_drug = [d for d in drugs if 0 <= d.start - verb.end() <= 8 and "of" in sentence[verb.end() : d.start]]
        if of_drug:
            subject = of_drug[0]
        else:
            # "sildenafil is cleared predominantly by CYP3A4"
            before = [d for d in drugs if d.end <= verb.start() and verb.start() - d.end <= 80]
            if before and not _SUBJECT_BLOCK_RE.search(sentence[before[-1].end : verb.start()]):
                subject = before[-1]
        if subject is None:
            continue
        for enzyme in enzymes:
            if enzyme.start < verb.end() or enzyme.start - verb.end() > 140:
                continue
            between = sentence[verb.start() : enzyme.start]
            if _NEGATION_RE.search(sentence[max(0, verb.start() - 25) : enzyme.start]):
                continue
            if _SUBJECT_BLOCK_RE.search(between) or _ROLE_AFTER_ENZYME_RE.match(sentence[enzyme.end :]):
                continue
            edges.append((subject.canonical, SUBSTRATE_OF, enzyme.canonical))
    return edges


def _active_verb_edges(sentence: str, entities: Sequence[Entity]) -> List[Triple]:
    """R4: '<drug> inhibits <enzyme>' with drug, verb and enzyme in that order."""
    edges: List[Triple] = []
    for verb in _ACTIVE_VERB_RE.finditer(sentence):
        relation = INHIBITS if verb.group(1) else INDUCES
        subjects = [e for e in entities if e.kind == DRUG and e.end <= verb.start()]
        objects = [e for e in entities if e.kind == ENZYME and e.start >= verb.end()]
        if not subjects or not objects:
            continue
        subject = subjects[-1]
        if verb.start() - subject.end > 80:
            continue
        if _NEGATION_RE.search(sentence[subject.end : verb.end()]):
            continue
        next_verb = _ACTIVE_VERB_RE.search(sentence, verb.end())
        for enzyme in objects:
            if next_verb and enzyme.start > next_verb.start():
                break  # belongs to the next verb: "inhibitor of CYP2C19 and inducer of CYP3A4"
            gap = sentence[verb.end() : enzyme.start]
            if len(gap) > 90 or _NEGATION_RE.search(gap):
                break
            edges.append((subject.canonical, relation, enzyme.canonical))
    return edges


def extract_triples(chunk: ClinicalChunk, label_drug: str, linker: EntityLinker) -> List[Triple]:
    """Extracts relation triples from one passage of `label_drug`'s label."""
    triples: List[Triple] = []
    own_classes = set(linker.classes_of.get(label_drug, []))
    section_relation = SECTION_RELATION.get(chunk.section)

    for sentence in split_sentences(chunk.content):
        entities = linker.link(sentence)
        if not entities:
            continue
        enzymes = [e for e in entities if e.kind == ENZYME]

        # R1 / R2
        if section_relation and not _NO_INTERACTION_RE.search(sentence):
            for ent in entities:
                if ent.kind == ENZYME or ent.canonical == label_drug:
                    continue
                if ent.kind == CLASS and ent.canonical in own_classes:
                    continue
                triples.append((label_drug, section_relation, ent.canonical))

        if not enzymes:
            continue

        # R3 / R4 / R5
        triples += _enzyme_role_edges(sentence, entities, label_drug)
        triples += _active_verb_edges(sentence, entities)
        triples += _substrate_edges(sentence, entities)

    return list(dict.fromkeys(triples))


def build_graph(
    chunks: Iterable[ClinicalChunk],
    doc_generic: Dict[str, str],
    linker: EntityLinker,
    min_enzyme_support: int = 2,
) -> KnowledgeGraph:
    """Builds the graph from all passages. `doc_generic` maps doc_id -> generic name."""
    kg = KnowledgeGraph()
    kinds: Dict[str, str] = {}

    def node_kind(node: str) -> str:
        if node in linker.class_members:
            return CLASS
        return kinds.get(node, DRUG)

    extracted: List[Tuple[Triple, str]] = []
    enzyme_ids: Set[str] = set()
    for chunk in chunks:
        for triple in extract_triples(chunk, doc_generic[chunk.doc_id], linker):
            extracted.append((triple, chunk.chunk_id))
            if triple[1] in (INHIBITS, INDUCES, SUBSTRATE_OF):
                enzyme_ids.add(triple[2])
    kinds.update({enzyme: ENZYME for enzyme in enzyme_ids})

    # INHIBITS / INDUCES edges need independent support from several passages: a single
    # sentence is too often a quirk of phrasing, and a wrong perpetrator edge creates
    # wrong enzyme-mediated paths for every substrate of that enzyme.
    support: Dict[Triple, Set[str]] = {}
    for triple, chunk_id in extracted:
        support.setdefault(triple, set()).add(chunk_id)
    extracted = [
        (triple, chunk_id)
        for triple, chunk_id in extracted
        if triple[1] not in (INHIBITS, INDUCES) or len(support[triple]) >= min_enzyme_support
    ]

    for (source, relation, target), chunk_id in extracted:
        kg.add_node(source, node_kind(source))
        kg.add_node(target, node_kind(target))
        kg.add_edge(source, relation, target, chunk_id)

    # R6: class membership, restricted to drugs that actually appear in the graph.
    for class_id, members in linker.class_members.items():
        for member in members:
            if member in kg.graph or member in doc_generic.values():
                kg.add_node(member, DRUG)
                kg.add_node(class_id, CLASS)
                kg.add_edge(member, MEMBER_OF, class_id)
    return kg
