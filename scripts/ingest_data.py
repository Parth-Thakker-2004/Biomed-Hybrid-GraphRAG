"""
Ingest FDA prescribing information from the openFDA drug-label API.

For every generic in ``core.lexicon.DRUGS`` this fetches candidate prescription labels,
keeps the most complete single-ingredient one, and stores the sections that matter for
interaction questions. The result is a compressed, versionable corpus snapshot
(``data/corpus.jsonl.gz``) so that index builds and evaluation are reproducible
without network access.

Usage:
    python scripts/ingest_data.py            # fetch anything not cached yet
    python scripts/ingest_data.py --refresh  # re-fetch everything

openFDA allows 240 requests/minute without a key; set OPENFDA_API_KEY to raise the
daily quota. Label text is a US Government work and is in the public domain.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from core.lexicon import DRUGS  # noqa: E402

logger = logging.getLogger("ingest")

# Section title -> openFDA fields to try, in order of preference. Newer labels use
# "warnings_and_cautions"; labels in the older format use "warnings" / "precautions".
SECTION_FIELDS: Dict[str, List[str]] = {
    "BOXED WARNING": ["boxed_warning"],
    "INDICATIONS AND USAGE": ["indications_and_usage"],
    "CONTRAINDICATIONS": ["contraindications"],
    "WARNINGS AND PRECAUTIONS": ["warnings_and_cautions", "warnings", "precautions"],
    "DRUG INTERACTIONS": ["drug_interactions"],
    "MECHANISM OF ACTION": ["mechanism_of_action"],
    "PHARMACOKINETICS": ["pharmacokinetics"],
}
# Used only when a label has neither of the two pharmacology subsections above.
FALLBACK_PHARMACOLOGY_FIELD = "clinical_pharmacology"
MAX_FALLBACK_CHARS = 15_000

_HEADER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)?\s+)?[A-Z][A-Z &/,\-]{6,}?(?=\s+(?:\d+\.\d+\s+)?[A-Z][a-z])")


def clean_section(text: str) -> str:
    """Collapses whitespace and strips the leading 'N SECTION TITLE' heading."""
    text = re.sub(r"\s+", " ", text).strip()
    return _HEADER_RE.sub("", text, count=1).strip()


def build_document(generic: str, label: Dict[str, Any]) -> Dict[str, Any] | None:
    """Converts one raw openFDA label into the corpus document schema."""
    sections: Dict[str, str] = {}
    for title, fields in SECTION_FIELDS.items():
        for field in fields:
            raw = " ".join(label.get(field) or [])
            if raw.strip():
                sections[title] = clean_section(raw)
                break

    if "MECHANISM OF ACTION" not in sections and "PHARMACOKINETICS" not in sections:
        raw = " ".join(label.get(FALLBACK_PHARMACOLOGY_FIELD) or [])
        if raw.strip():
            sections["CLINICAL PHARMACOLOGY"] = clean_section(raw)[:MAX_FALLBACK_CHARS]

    if not sections:
        return None

    meta = label.get("openfda", {})
    curated_brands = DRUGS.get(generic, [])
    label_brands = [
        b.title() for b in meta.get("brand_name", [])
        if generic.split()[0] not in b.lower()  # skip "Sildenafil Citrate"-style non-brands
    ]
    brands = list(dict.fromkeys(curated_brands + label_brands))
    set_id = (meta.get("spl_set_id") or [label.get("set_id", "")])[0]

    return {
        "doc_id": generic.replace(" ", "_"),
        "generic_name": generic,
        "brand_names": brands,
        "title": generic.title() + (f" ({', '.join(curated_brands)})" if curated_brands else ""),
        "source": "openFDA drug label",
        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else "",
        "set_id": set_id,
        "effective_time": label.get("effective_time", ""),
        "manufacturer": (meta.get("manufacturer_name") or [""])[0],
        "pharm_class": meta.get("pharm_class_epc", []),
        "sections": sections,
    }


def pick_best_label(generic: str, results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Chooses the most complete single-ingredient label for a generic."""
    first_word = generic.split()[0].upper()

    def is_single_ingredient(label: Dict[str, Any]) -> bool:
        substances = label.get("openfda", {}).get("substance_name", [])
        return len(substances) == 1 and first_word in substances[0].upper()

    def completeness(label: Dict[str, Any]) -> tuple:
        present = sum(
            1 for fields in SECTION_FIELDS.values() if any(label.get(f) for f in fields)
        )
        has_interactions = bool(label.get("drug_interactions"))
        return (has_interactions, present, label.get("effective_time", ""))

    candidates = [r for r in results if is_single_ingredient(r)]
    return max(candidates, key=completeness) if candidates else None


class OpenFDAClient:
    def __init__(self, base_url: str, requests_per_minute: int = 200):
        self.base_url = base_url
        self.min_interval = 60.0 / requests_per_minute
        self.api_key = os.getenv("OPENFDA_API_KEY")
        self.session = requests.Session()
        self._last_call = 0.0

    def search_labels(self, generic: str, limit: int = 15) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "search": (
                f'openfda.generic_name:"{generic}" AND '
                'openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
            ),
            "limit": limit,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                res = self.session.get(self.base_url, params=params, timeout=30)
            except requests.RequestException as exc:
                logger.warning("%s: network error (%s), retrying", generic, exc)
                time.sleep(2**attempt)
                continue
            if res.status_code == 200:
                return res.json().get("results", [])
            if res.status_code == 404:  # openFDA returns 404 for "no matches"
                return []
            if res.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            logger.warning("%s: HTTP %s", generic, res.status_code)
            time.sleep(2**attempt)
        return []


def run(refresh: bool = False, config_path: str = "configs/config.yaml") -> List[Dict[str, Any]]:
    config = load_config(config_path)
    labels_dir = ROOT / config["paths"]["labels_cache_dir"]
    corpus_file = ROOT / config["paths"]["corpus_file"]
    labels_dir.mkdir(parents=True, exist_ok=True)

    client = OpenFDAClient(
        base_url=config["ingestion"]["openfda"]["base_url"],
        requests_per_minute=config["ingestion"]["openfda"]["requests_per_minute"],
    )

    docs: List[Dict[str, Any]] = []
    missing: List[str] = []
    for i, generic in enumerate(DRUGS, start=1):
        cache_file = labels_dir / f"{generic.replace(' ', '_')}.json"
        if cache_file.exists() and not refresh:
            doc = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            label = pick_best_label(generic, client.search_labels(generic))
            doc = build_document(generic, label) if label else None
            if doc:
                cache_file.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        if doc:
            docs.append(doc)
        else:
            missing.append(generic)
        if i % 25 == 0:
            logger.info("%d/%d generics processed (%d labels)", i, len(DRUGS), len(docs))

    with gzip.open(corpus_file, "wt", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Wrote %d labels to %s", len(docs), corpus_file)
    if missing:
        logger.info("No usable single-ingredient label for: %s", ", ".join(missing))
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-fetch labels that are already cached")
    run(refresh=parser.parse_args().refresh)
