"""
Build every retrieval artefact from the corpus snapshot.

    data/corpus.jsonl.gz -> data/index/{chunks.jsonl, graph.json, dense.faiss, embeddings.npy}

The BM25 index is not stored: it is rebuilt from chunks.jsonl at start-up in well under a
second, which avoids shipping a pickle.

Usage:
    python scripts/build_indexes.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.chunking import ClinicalSectionChunker  # noqa: E402
from core.config import load_config, resolve  # noqa: E402
from core.dense import DenseRetriever  # noqa: E402
from core.entities import EntityLinker  # noqa: E402
from core.kg_builder import build_graph  # noqa: E402
from core.pipeline import load_corpus  # noqa: E402

logger = logging.getLogger("build")


def build(config_path: str = "configs/config.yaml", graph_only: bool = False) -> None:
    config = load_config(config_path)
    corpus_file = resolve(config["paths"]["corpus_file"])
    index_dir = resolve(config["paths"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_file.exists():
        raise SystemExit(f"{corpus_file} not found. Run: python scripts/ingest_data.py")

    docs = load_corpus(corpus_file)
    chunker = ClinicalSectionChunker(
        target_tokens=config["chunking"]["target_tokens"],
        overlap_tokens=config["chunking"]["overlap_tokens"],
    )
    chunks = chunker.chunk_corpus(docs)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Chunked %d labels into %d passages", len(docs), len(chunks))

    t = time.perf_counter()
    kg = build_graph(chunks, {d["doc_id"]: d["generic_name"] for d in docs}, EntityLinker())
    kg.save(index_dir / "graph.json")
    logger.info("Knowledge graph built in %.1fs: %s", time.perf_counter() - t, json.dumps(kg.stats()))

    meta = {"labels": len(docs), "passages": len(chunks), "graph": kg.stats()}
    (index_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if graph_only:
        return

    t = time.perf_counter()
    dense_cfg = config["dense"]
    dense = DenseRetriever(
        model_name=dense_cfg["model_name"],
        query_instruction=dense_cfg["query_instruction"],
        m=dense_cfg["hnsw"]["M"],
        ef_construction=dense_cfg["hnsw"]["ef_construction"],
        ef_search=dense_cfg["hnsw"]["ef_search"],
        batch_size=dense_cfg["batch_size"],
        device=dense_cfg["device"],
    )
    dense.index_chunks(chunks)
    dense.save(index_dir)
    logger.info("Dense index built in %.1fs on %s", time.perf_counter() - t, dense.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Build retrieval indexes from the corpus snapshot.")
    parser.add_argument(
        "--graph-only", action="store_true",
        help="rebuild chunks and the knowledge graph but keep the existing dense index",
    )
    build(graph_only=parser.parse_args().graph_only)
