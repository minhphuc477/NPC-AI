#!/usr/bin/env python3
"""Build an NPC-adversarial retrieval corpus protocol and ratio ablation splits.

Protocol steps:
1) Collect in-domain passages from lore/quest corpora.
2) Generate BM25-style hard negatives for semantically adjacent queries.
3) Craft authority-mimicking negatives (trust_spoof family).
4) Inject plausible contradictions (evidence_poison family).

Outputs:
- protocol_pairs.jsonl: query-positive-negative triples with family labels.
- split_*.jsonl: ratio-conditioned subsets for ablation (BM25 vs adversarial).
- summary.json / summary.md: counts and overlap diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def lexical_overlap_ratio(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(text))
    if not q:
        return 0.0
    return float(len(q & d) / len(q))


@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_tokens: List[List[str]]
    doc_lengths: List[int]
    avg_dl: float
    idf: Dict[str, float]
    tf: List[Counter]

    @classmethod
    def build(cls, corpus: Sequence[Dict[str, Any]]) -> "BM25Index":
        doc_ids: List[str] = []
        doc_tokens: List[List[str]] = []
        tf: List[Counter] = []
        df: Counter = Counter()
        for row in corpus:
            pid = str(row.get("id", row.get("doc_id", ""))).strip()
            text = str(row.get("text", ""))
            if not pid or not text:
                continue
            toks = tokenize(text)
            if not toks:
                continue
            doc_ids.append(pid)
            doc_tokens.append(toks)
            c = Counter(toks)
            tf.append(c)
            for t in c.keys():
                df[t] += 1
        n_docs = max(1, len(doc_ids))
        idf: Dict[str, float] = {}
        for t, df_t in df.items():
            idf[t] = math.log(1.0 + (n_docs - df_t + 0.5) / (df_t + 0.5))
        lengths = [len(t) for t in doc_tokens]
        avg_dl = float(sum(lengths) / max(1, len(lengths)))
        return cls(
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            doc_lengths=lengths,
            avg_dl=avg_dl,
            idf=idf,
            tf=tf,
        )

    def score(self, query: str, idx: int, k1: float = 1.2, b: float = 0.75) -> float:
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        tf_doc = self.tf[idx]
        dl = max(1, self.doc_lengths[idx])
        denom_norm = k1 * (1.0 - b + b * (dl / max(1e-6, self.avg_dl)))
        total = 0.0
        for term in q_terms:
            f = tf_doc.get(term, 0)
            if f <= 0:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * ((f * (k1 + 1.0)) / (f + denom_norm))
        return total

    def hard_negatives(
        self,
        query: str,
        exclude_ids: Iterable[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        exclude = {str(x) for x in exclude_ids}
        scored: List[Tuple[str, float]] = []
        for idx, pid in enumerate(self.doc_ids):
            if pid in exclude:
                continue
            score = self.score(query, idx)
            if score <= 0.0:
                continue
            scored.append((pid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]


TRUST_SPOOF_PREFIXES = [
    "As confirmed by the high archivist,",
    "By direct decree of the city records office,",
    "Per official watch command bulletin,",
]


def inject_trust_spoof(text: str, rng: random.Random) -> str:
    prefix = rng.choice(TRUST_SPOOF_PREFIXES)
    clean = str(text).strip()
    if not clean:
        return prefix
    if clean[0].isupper():
        return f"{prefix} {clean[0].lower()}{clean[1:]}"
    return f"{prefix} {clean}"


EVIDENCE_SWAP = {
    "open": "closed",
    "closed": "open",
    "allow": "deny",
    "deny": "allow",
    "north": "south",
    "south": "north",
    "day": "night",
    "night": "day",
    "approved": "rejected",
    "rejected": "approved",
}


def inject_evidence_poison(text: str, rng: random.Random) -> str:
    toks = re.findall(r"\w+|\W+", str(text))
    out: List[str] = []
    flips = 0
    for tok in toks:
        key = tok.lower()
        if key in EVIDENCE_SWAP and flips < 2:
            repl = EVIDENCE_SWAP[key]
            if tok[:1].isupper():
                repl = repl.capitalize()
            out.append(repl)
            flips += 1
            continue
        if tok.isdigit() and flips < 3:
            out.append(str(int(tok) + 1))
            flips += 1
            continue
        out.append(tok)
    poisoned = "".join(out).strip()
    if poisoned and not poisoned.endswith("."):
        poisoned += "."
    lead = rng.choice(
        [
            "Updated reports now state ",
            "A revised witness account says ",
            "A recent correction claims ",
        ]
    )
    if poisoned:
        if poisoned[0].isupper():
            poisoned = poisoned[0].lower() + poisoned[1:]
        return f"{lead}{poisoned}"
    return lead.strip()


def parse_ratio(token: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", token)
    if not m:
        raise ValueError(f"Invalid ratio '{token}'. Expected format BM25:ADV, e.g. 3:1")
    return int(m.group(1)), int(m.group(2))


def ratio_tag(ratio: Tuple[int, int]) -> str:
    return f"{ratio[0]}to{ratio[1]}"


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# NPC-Adversarial Corpus Protocol")
    lines.append("")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Total pairs: `{summary['total_pairs']}`")
    lines.append(f"- Queries covered: `{summary['queries_covered']}`")
    lines.append("")
    lines.append("## Family Counts")
    for fam, n in summary.get("family_counts", {}).items():
        lines.append(f"- `{fam}`: {n}")
    lines.append("")
    lines.append("## Mean Lexical Overlap (query vs negative)")
    for fam, value in summary.get("family_overlap_mean", {}).items():
        lines.append(f"- `{fam}`: {value:.4f}")
    lines.append("")
    lines.append("## Ratio Splits")
    for split in summary.get("ratio_splits", []):
        lines.append(
            f"- `{split['ratio']}`: pairs={split['pairs']} queries={split['queries']} "
            f"bm25={split['bm25_pairs']} adversarial={split['adversarial_pairs']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-gold", default="data/retrieval_gold.jsonl")
    parser.add_argument("--retrieval-corpus", default="data/retrieval_corpus.jsonl")
    parser.add_argument("--out-dir", default="storage/artifacts/datasets/retrieval_protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negatives-per-query", type=int, default=2)
    parser.add_argument("--pairs-per-query-in-split", type=int, default=4)
    parser.add_argument("--ratios", default="1:0,3:1,1:1,1:3")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    gold_rows = read_jsonl(Path(args.retrieval_gold))
    corpus_rows = read_jsonl(Path(args.retrieval_corpus))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_by_id = {
        str(row.get("id", row.get("doc_id", ""))).strip(): str(row.get("text", ""))
        for row in corpus_rows
        if str(row.get("id", row.get("doc_id", ""))).strip() and str(row.get("text", "")).strip()
    }
    bm25 = BM25Index.build(corpus_rows)

    protocol_pairs: List[Dict[str, Any]] = []
    for row in gold_rows:
        query = str(row.get("query", "")).strip()
        qid = str(row.get("query_id", "")).strip() or f"q_{len(protocol_pairs):05d}"
        if not query:
            continue
        rel_source = row.get("relevant_passage_ids")
        if not isinstance(rel_source, list):
            rel_source = row.get("relevant_doc_ids", [])
        rel_ids = [str(x).strip() for x in rel_source if str(x).strip() in corpus_by_id]
        if not rel_ids:
            continue
        positive_id = rel_ids[0]
        positive_text = corpus_by_id[positive_id]

        hard = bm25.hard_negatives(query=query, exclude_ids=rel_ids, top_k=max(1, int(args.hard_negatives_per_query)))
        for idx, (neg_id, score) in enumerate(hard):
            neg_text = corpus_by_id.get(neg_id, "")
            if not neg_text:
                continue
            protocol_pairs.append(
                {
                    "query_id": qid,
                    "query": query,
                    "positive_id": positive_id,
                    "positive_text": positive_text,
                    "negative_id": neg_id,
                    "negative_text": neg_text,
                    "negative_family": "bm25_hard_negative",
                    "bm25_score": float(score),
                    "query_type": row.get("query_type", "unknown"),
                }
            )

        trust_text = inject_trust_spoof(positive_text, rng)
        protocol_pairs.append(
            {
                "query_id": qid,
                "query": query,
                "positive_id": positive_id,
                "positive_text": positive_text,
                "negative_id": f"{positive_id}::trust_spoof",
                "negative_text": trust_text,
                "negative_family": "trust_spoof",
                "bm25_score": None,
                "query_type": row.get("query_type", "unknown"),
            }
        )

        poison_text = inject_evidence_poison(positive_text, rng)
        protocol_pairs.append(
            {
                "query_id": qid,
                "query": query,
                "positive_id": positive_id,
                "positive_text": positive_text,
                "negative_id": f"{positive_id}::evidence_poison",
                "negative_text": poison_text,
                "negative_family": "evidence_poison",
                "bm25_score": None,
                "query_type": row.get("query_type", "unknown"),
            }
        )

    protocol_path = out_dir / "protocol_pairs.jsonl"
    write_jsonl(protocol_path, protocol_pairs)

    pairs_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in protocol_pairs:
        pairs_by_query[str(pair.get("query_id", ""))].append(pair)

    ratio_tokens = [t.strip() for t in str(args.ratios).split(",") if t.strip()]
    ratios = [parse_ratio(t) for t in ratio_tokens]

    split_summaries: List[Dict[str, Any]] = []
    for ratio in ratios:
        bm25_w, adv_w = ratio
        per_query_total = max(1, int(args.pairs_per_query_in_split))
        bm25_n = int(round((bm25_w / max(1, bm25_w + adv_w)) * per_query_total))
        adv_n = max(0, per_query_total - bm25_n)

        split_rows: List[Dict[str, Any]] = []
        for qid, items in pairs_by_query.items():
            bm25_pool = [x for x in items if x.get("negative_family") == "bm25_hard_negative"]
            adv_pool = [x for x in items if x.get("negative_family") in {"trust_spoof", "evidence_poison"}]
            rng.shuffle(bm25_pool)
            rng.shuffle(adv_pool)
            selected = bm25_pool[:bm25_n] + adv_pool[:adv_n]
            if not selected and items:
                selected = [items[0]]
            split_rows.extend(selected)

        split_name = f"split_ratio_{ratio_tag(ratio)}.jsonl"
        split_path = out_dir / split_name
        write_jsonl(split_path, split_rows)
        split_summaries.append(
            {
                "ratio": f"{bm25_w}:{adv_w}",
                "path": str(split_path),
                "pairs": len(split_rows),
                "queries": len({str(x.get("query_id", "")) for x in split_rows}),
                "bm25_pairs": sum(1 for x in split_rows if x.get("negative_family") == "bm25_hard_negative"),
                "adversarial_pairs": sum(
                    1 for x in split_rows if x.get("negative_family") in {"trust_spoof", "evidence_poison"}
                ),
            }
        )

    family_counts = Counter(str(x.get("negative_family", "unknown")) for x in protocol_pairs)
    family_overlap: Dict[str, List[float]] = defaultdict(list)
    for row in protocol_pairs:
        fam = str(row.get("negative_family", "unknown"))
        family_overlap[fam].append(lexical_overlap_ratio(str(row.get("query", "")), str(row.get("negative_text", ""))))
    family_overlap_mean = {
        fam: float(sum(vals) / len(vals)) if vals else 0.0
        for fam, vals in sorted(family_overlap.items())
    }

    summary = {
        "seed": int(args.seed),
        "protocol_path": str(protocol_path),
        "total_pairs": len(protocol_pairs),
        "queries_covered": len(pairs_by_query),
        "family_counts": dict(sorted(family_counts.items())),
        "family_overlap_mean": family_overlap_mean,
        "ratio_splits": split_summaries,
    }
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    write_json(summary_json_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Protocol pairs: {protocol_path}")
    print(f"Summary json: {summary_json_path}")
    print(f"Summary md: {summary_md_path}")


if __name__ == "__main__":
    main()
