#!/usr/bin/env python3
"""Build hard-negative retrieval training/eval sets from labeled retrieval artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

TOKEN_RE = re.compile(r"[a-z0-9']+")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> Set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    denom = len(a | b)
    if denom <= 0:
        return 0.0
    return float(len(a & b)) / float(denom)


def maybe_domain(row: Dict[str, Any]) -> str:
    return str(row.get("domain", "")).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-negative sets for retrieval reranking.")
    parser.add_argument("--retrieval-gold", default="data/retrieval_gold_wide.jsonl")
    parser.add_argument("--retrieval-corpus", default="data/retrieval_corpus_wide.jsonl")
    parser.add_argument("--hard-negatives-per-query", type=int, default=8)
    parser.add_argument("--cross-domain-negatives-per-query", type=int, default=3)
    parser.add_argument("--output-hard-negatives", default="data/retrieval_hard_negatives_wide.jsonl")
    parser.add_argument("--output-reranker-pairs", default="data/retrieval_reranker_pairs_wide.jsonl")
    parser.add_argument("--output-summary", default="data/retrieval_hard_negatives_wide.summary.json")
    args = parser.parse_args()

    gold_path = Path(args.retrieval_gold)
    corpus_path = Path(args.retrieval_corpus)
    if not gold_path.exists():
        raise FileNotFoundError(f"retrieval gold not found: {gold_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"retrieval corpus not found: {corpus_path}")

    gold_rows = read_jsonl(gold_path)
    corpus_rows = read_jsonl(corpus_path)
    by_doc_id = {
        str(row.get("doc_id", "")).strip(): row
        for row in corpus_rows
        if str(row.get("doc_id", "")).strip()
    }

    if not by_doc_id:
        raise RuntimeError("corpus is empty or missing doc_id fields")

    corpus_tokens: Dict[str, Set[str]] = {}
    for doc_id, row in by_doc_id.items():
        text = " ".join(
            [
                str(row.get("title", "")),
                str(row.get("text", "")),
                str(row.get("domain", "")),
            ]
        )
        corpus_tokens[doc_id] = tokenize(text)

    hard_rows: List[Dict[str, Any]] = []
    reranker_rows: List[Dict[str, Any]] = []
    total_pairs = 0
    total_queries = 0

    for row in gold_rows:
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not query_id or not query:
            continue
        rel_ids = [str(x).strip() for x in row.get("relevant_doc_ids", []) if str(x).strip()]
        rel_ids = [x for x in rel_ids if x in by_doc_id]
        if not rel_ids:
            continue

        q_tokens = tokenize(query)
        q_domain = maybe_domain(row)

        scored_same_domain: List[tuple[float, str]] = []
        scored_cross_domain: List[tuple[float, str]] = []
        rel_set = set(rel_ids)
        for doc_id, doc_row in by_doc_id.items():
            if doc_id in rel_set:
                continue
            score = jaccard(q_tokens, corpus_tokens.get(doc_id, set()))
            if math.isnan(score):
                continue
            d_domain = maybe_domain(doc_row)
            if q_domain and d_domain == q_domain:
                scored_same_domain.append((score, doc_id))
            else:
                scored_cross_domain.append((score, doc_id))

        scored_same_domain.sort(key=lambda x: (-x[0], x[1]))
        scored_cross_domain.sort(key=lambda x: (-x[0], x[1]))

        same_take = max(0, int(args.hard_negatives_per_query))
        cross_take = max(0, int(args.cross_domain_negatives_per_query))
        hard_neg_same = [doc_id for _, doc_id in scored_same_domain[:same_take]]
        hard_neg_cross = [doc_id for _, doc_id in scored_cross_domain[:cross_take]]
        hard_neg_ids = hard_neg_same + hard_neg_cross
        if not hard_neg_ids:
            continue

        hard_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "domain": q_domain,
                "relevant_doc_ids": rel_ids,
                "hard_negative_doc_ids": hard_neg_ids,
                "hard_negative_same_domain_doc_ids": hard_neg_same,
                "hard_negative_cross_domain_doc_ids": hard_neg_cross,
            }
        )

        positives_text = []
        for pos_id in rel_ids:
            pos = by_doc_id.get(pos_id, {})
            positives_text.append(
                {
                    "doc_id": pos_id,
                    "text": str(pos.get("text", "")),
                    "title": str(pos.get("title", "")),
                }
            )
        negatives_text = []
        for neg_id in hard_neg_ids:
            neg = by_doc_id.get(neg_id, {})
            negatives_text.append(
                {
                    "doc_id": neg_id,
                    "text": str(neg.get("text", "")),
                    "title": str(neg.get("title", "")),
                }
            )

        # Pairwise reranker rows (one positive paired with each hard negative).
        for pos in positives_text:
            for neg in negatives_text:
                reranker_rows.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "positive_doc_id": pos["doc_id"],
                        "negative_doc_id": neg["doc_id"],
                        "positive_text": pos["text"],
                        "negative_text": neg["text"],
                        "positive_title": pos["title"],
                        "negative_title": neg["title"],
                        "label": 1,
                    }
                )
                total_pairs += 1

        total_queries += 1

    out_hard = Path(args.output_hard_negatives)
    out_pairs = Path(args.output_reranker_pairs)
    out_summary = Path(args.output_summary)
    write_jsonl(out_hard, hard_rows)
    write_jsonl(out_pairs, reranker_rows)

    summary = {
        "retrieval_gold": str(gold_path),
        "retrieval_corpus": str(corpus_path),
        "query_count_with_hard_negatives": total_queries,
        "hard_negative_rows": len(hard_rows),
        "reranker_pair_rows": total_pairs,
        "hard_negatives_per_query": int(args.hard_negatives_per_query),
        "cross_domain_negatives_per_query": int(args.cross_domain_negatives_per_query),
        "output_hard_negatives": str(out_hard),
        "output_reranker_pairs": str(out_pairs),
    }
    write_json(out_summary, summary)

    print(f"Hard negatives: {out_hard}")
    print(f"Reranker pairs: {out_pairs}")
    print(f"Summary: {out_summary}")
    print(f"Queries with hard negatives: {total_queries}")
    print(f"Pair rows: {total_pairs}")


if __name__ == "__main__":
    main()

