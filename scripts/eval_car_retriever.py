#!/usr/bin/env python3
"""Evaluate CAR retriever against labeled retrieval set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.car_retriever import CARRetriever, read_jsonl


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(pred_rows: List[Dict[str, Any]], gold_rows: List[Dict[str, Any]], hit_k: int) -> Dict[str, Any]:
    by_query = {str(row.get("query_id", "")).strip(): row for row in gold_rows}
    per_query: List[Dict[str, Any]] = []
    hits = 0.0
    mrr = 0.0
    for row in pred_rows:
        query_id = str(row.get("query_id", "")).strip()
        gold = by_query.get(query_id, {})
        relevant = set(str(x) for x in gold.get("relevant_doc_ids", []) if str(x))
        ranked = [str(x) for x in row.get("predicted_doc_ids", [])[:hit_k]]

        hit = 1.0 if any(doc in relevant for doc in ranked) else 0.0
        rr = 0.0
        for idx, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                rr = 1.0 / float(idx)
                break

        hits += hit
        mrr += rr
        per_query.append({"query_id": query_id, f"hit@{hit_k}": hit, "mrr": rr})

    n = float(max(1, len(pred_rows)))
    return {
        "query_count": len(pred_rows),
        f"hit@{hit_k}": hits / n,
        "mrr": mrr / n,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAR retriever.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--gold", default="data/retrieval_gold_wide.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--predictions-out", default="storage/artifacts/retrieval/predictions_car.jsonl")
    parser.add_argument("--metrics-out", default="storage/artifacts/retrieval/metrics_car.json")
    args = parser.parse_args()

    gold_rows = read_jsonl(Path(args.gold))
    retriever = CARRetriever.from_checkpoint_and_index(
        checkpoint=str(args.checkpoint),
        index_path=str(args.index),
        device=str(args.device),
        max_length=int(args.max_length),
    )
    pred_rows: List[Dict[str, Any]] = []
    for row in gold_rows:
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        ranked = retriever.rank(query, top_k=int(args.top_k))
        pred_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "predicted_doc_ids": [doc_id for doc_id, _ in ranked],
                "scores": [float(score) for _, score in ranked],
            }
        )

    metrics = evaluate(pred_rows, gold_rows, hit_k=int(args.top_k))
    write_jsonl(Path(args.predictions_out), pred_rows)
    write_json(Path(args.metrics_out), metrics)
    print(f"saved_predictions={args.predictions_out}")
    print(f"saved_metrics={args.metrics_out}")


if __name__ == "__main__":
    main()
