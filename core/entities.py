"""
Dictionary-based entity linking for drugs, drug classes and metabolic enzymes.

A gazetteer is the right tool here: the entity inventory is closed (the drugs in the
corpus), brand names must normalise to generics ("Coumadin" -> warfarin), and linking
runs on every query, so it has to cost microseconds rather than a model forward pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from core import lexicon

DRUG, CLASS, ENZYME = "DRUG", "CLASS", "ENZYME"

_CYP_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:CYP\s?|cytochrome\s+P-?450\s+(?:isoenzymes?\s+|isozymes?\s+)?(?:CYP\s?)?"
    r"|P-?450\s+)(\d[A-Z]\d{0,2})(?:/\d+)?(?![A-Za-z])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entity:
    text: str  # surface form as written
    canonical: str  # graph node id: generic name, class id or enzyme id
    kind: str  # DRUG | CLASS | ENZYME
    start: int
    end: int


class EntityLinker:
    def __init__(
        self,
        drugs: Mapping[str, Sequence[str]] | None = None,
        classes: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        enzymes: Mapping[str, Sequence[str]] | None = None,
    ):
        drugs = lexicon.DRUGS if drugs is None else drugs
        classes = lexicon.DRUG_CLASSES if classes is None else classes
        enzymes = lexicon.ENZYMES if enzymes is None else enzymes

        self._surface: Dict[str, tuple] = {}
        for generic, brands in drugs.items():
            self._surface[generic.lower()] = (generic, DRUG)
            for brand in brands:
                self._surface[brand.lower()] = (generic, DRUG)
        for class_id, spec in classes.items():
            for alias in [class_id, *spec["aliases"]]:
                self._surface.setdefault(alias.lower(), (class_id, CLASS))
        self._cyp_alias: Dict[str, str] = {}
        for enzyme_id, aliases in enzymes.items():
            for alias in [enzyme_id, *aliases]:
                if alias.upper().startswith("CYP"):
                    self._cyp_alias[alias.upper().split("/")[0]] = enzyme_id
                else:
                    self._surface.setdefault(alias.lower(), (enzyme_id, ENZYME))

        self.class_members: Dict[str, List[str]] = {
            class_id: list(spec["members"]) for class_id, spec in classes.items()
        }
        self.classes_of: Dict[str, List[str]] = {}
        for class_id, members in self.class_members.items():
            for member in members:
                self.classes_of.setdefault(member, []).append(class_id)

        # Longest surface form first so "isosorbide mononitrate" wins over "nitrate".
        alternation = "|".join(
            re.escape(s) for s in sorted(self._surface, key=len, reverse=True)
        )
        self._surface_re = re.compile(
            rf"(?<![A-Za-z0-9])({alternation})(?![A-Za-z0-9])", re.IGNORECASE
        )

    def link(self, text: str) -> List[Entity]:
        """Returns non-overlapping entity mentions in order of appearance."""
        found: List[Entity] = []
        for match in _CYP_RE.finditer(text):
            isoform = "CYP" + match.group(1).upper()
            found.append(
                Entity(match.group(0), self._cyp_alias.get(isoform, isoform), ENZYME, *match.span())
            )
        taken = [(e.start, e.end) for e in found]
        for match in self._surface_re.finditer(text):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            canonical, kind = self._surface[match.group(1).lower()]
            found.append(Entity(match.group(1), canonical, kind, start, end))
        return sorted(found, key=lambda e: e.start)

    def normalize_query(self, text: str) -> str:
        """Writes the canonical name next to every alias: 'Coumadin' -> 'Coumadin (warfarin)'.

        Label text uses generic names almost exclusively, so a question phrased with brand
        names shares no vocabulary with the passages that answer it.
        """
        out: List[str] = []
        cursor = 0
        for ent in self.link(text):
            out.append(text[cursor : ent.end])
            if ent.text.lower() != ent.canonical.lower():
                out.append(f" ({ent.canonical})")
            cursor = ent.end
        out.append(text[cursor:])
        return "".join(out)

    def canonical_ids(self, text: str) -> List[str]:
        """Unique canonical ids mentioned in the text, in order of first appearance."""
        return list(dict.fromkeys(e.canonical for e in self.link(text)))
