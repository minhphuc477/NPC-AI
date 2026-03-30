#!/usr/bin/env python3
"""Build CAR dense index from retrieval corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.car_retriever import CAREncoder, CAREncoderConfig, CARIndex, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAR index from corpus.")
    parser.add_argument("--checkpoint", required=True, help="CAR checkpoint dir (best/epoch_xx).")
    parser.add_argument("--corpus", default="data/retrieval_corpus_wide.jsonl")
    parser.add_argument("--out-index", default="storage/artifacts/retrieval/car_index.npz")
    parser.add_argument("--out-meta", default="storage/artifacts/retrieval/car_index.meta.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    corpus_rows = read_jsonl(Path(args.corpus))
    doc_ids: List[str] = []
    texts: List[str] = []
    for row in corpus_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not doc_id or not text:
            continue
        doc_ids.append(doc_id)
        texts.append(text)

    encoder = CAREncoder(
        CAREncoderConfig(
            model_name_or_path=str(args.checkpoint),
            device=str(args.device),
            max_length=int(args.max_length),
        )
    )
    embeddings = encoder.encode(texts, batch_size=int(args.batch_size))
    index = CARIndex(doc_ids=doc_ids, embeddings=embeddings)
    out_index = Path(args.out_index)
    index.save(out_index)

    meta: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "doc_count": len(doc_ids),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "index_path": str(out_index),
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved_index={out_index}")
    print(f"saved_meta={out_meta}")


if __name__ == "__main__":
    main()
