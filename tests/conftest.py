"""Shared fixtures. Nothing here loads a model or touches the network."""

from __future__ import annotations

import pytest

from core.chunking import ClinicalChunk
from core.entities import EntityLinker


def make_chunk(doc_id: str, section: str, content: str, n: int = 0) -> ClinicalChunk:
    return ClinicalChunk(
        chunk_id=f"{doc_id}::{section.lower().replace(' ', '_')}::{n:03d}",
        doc_id=doc_id,
        title=doc_id.replace("_", " ").title(),
        section=section,
        content=content,
        token_count=len(content.split()),
    )


@pytest.fixture(scope="session")
def linker() -> EntityLinker:
    return EntityLinker()


@pytest.fixture
def mini_corpus():
    return [
        make_chunk("sildenafil", "CONTRAINDICATIONS",
                   "Sildenafil is contraindicated in patients using organic nitrates in any form."),
        make_chunk("sildenafil", "PHARMACOKINETICS",
                   "Sildenafil is cleared predominantly by the CYP3A4 (major route) and CYP2C9 hepatic isoenzymes."),
        make_chunk("simvastatin", "DRUG INTERACTIONS",
                   "Strong CYP3A4 inhibitors (e.g., itraconazole, ketoconazole, clarithromycin) increase the risk of myopathy."),
        make_chunk("simvastatin", "PHARMACOKINETICS",
                   "Simvastatin is a substrate of CYP3A4."),
        make_chunk("clarithromycin", "DRUG INTERACTIONS",
                   "Clarithromycin inhibits CYP3A4. Strong CYP3A4 inhibitors such as clarithromycin raise statin levels."),
        make_chunk("warfarin", "DRUG INTERACTIONS",
                   "Concomitant use of warfarin 5 mg with NSAIDs increases the risk of bleeding."),
    ]
