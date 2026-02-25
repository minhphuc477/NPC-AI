#!/usr/bin/env python3
"""Train and evaluate a lightweight retrieval reranker on hard-negative pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


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


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def bootstrap_ci(values: Sequence[float], seed: int, iters: int = 2000) -> Dict[str, float]:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean_val = sum(vals) / float(n)
    med_val = percentile(vals, 0.5)
    if n == 1:
        low = high = mean_val
    else:
        rng = random.Random(seed)
        samples: List[float] = []
        for _ in range(iters):
            draw = [vals[rng.randrange(n)] for __ in range(n)]
            samples.append(sum(draw) / float(n))
        samples.sort()
        low = percentile(samples, 0.025)
        high = percentile(samples, 0.975)
    return {
        "n": n,
        "mean": mean_val,
        "median": med_val,
        "ci95_low": low,
        "ci95_high": high,
    }


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


class BM25Index:
    def __init__(self, documents: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_dl = 0.0
        self._build()

    def _build(self) -> None:
        df: Dict[str, int] = {}
        total_len = 0
        for doc in self.documents:
            doc_id = str(doc["doc_id"])
            toks = tokenize(str(doc["text"]))
            total_len += len(toks)
            self.doc_len[doc_id] = len(toks)
            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs[doc_id] = tf
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
        self.avg_dl = (total_len / len(self.documents)) if self.documents else 0.0
        n = float(len(self.documents))
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))

    def rank(self, query: str, top_k: int) -> List[str]:
        q_toks = tokenize(query)
        scored: List[Tuple[str, float]] = []
        for doc in self.documents:
            doc_id = str(doc["doc_id"])
            score = self.score(query_tokens=q_toks, doc_id=doc_id)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    def score(self, query: str | None = None, doc_id: str | None = None, query_tokens: Sequence[str] | None = None) -> float:
        did = str(doc_id or "")
        if did not in self.term_freqs:
            return 0.0
        q_toks = list(query_tokens) if query_tokens is not None else tokenize(str(query or ""))
        tf = self.term_freqs.get(did, {})
        score = 0.0
        dl = self.doc_len.get(did, 0)
        for term in q_toks:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl if self.avg_dl > 0 else 0.0))
            score += idf * ((freq * (self.k1 + 1.0)) / (denom if denom > 0 else 1.0))
        return score


def pair_to_text(query: str, doc_text: str) -> str:
    return f"query: {query}\ndocument: {doc_text}"


def build_binary_examples(pair_rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    for row in pair_rows:
        query = str(row.get("query", ""))
        pos = str(row.get("positive_text", ""))
        neg = str(row.get("negative_text", ""))
        texts.append(pair_to_text(query, pos))
        labels.append(1)
        texts.append(pair_to_text(query, neg))
        labels.append(0)
    return texts, labels


def split_by_query(rows: Sequence[Dict[str, Any]], train_frac: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_qid: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        by_qid.setdefault(qid, []).append(row)

    qids = sorted(by_qid.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    cutoff = max(1, min(len(qids) - 1, int(round(len(qids) * train_frac)))) if len(qids) > 1 else len(qids)
    train_ids = set(qids[:cutoff])

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for qid, items in by_qid.items():
        if qid in train_ids:
            train_rows.extend(items)
        else:
            eval_rows.extend(items)
    if not eval_rows and train_rows:
        eval_rows.append(train_rows.pop())
    return train_rows, eval_rows


def predict_prob(model: LogisticRegression, vectorizer: TfidfVectorizer, query: str, doc_text: str) -> float:
    x = vectorizer.transform([pair_to_text(query, doc_text)])
    proba = model.predict_proba(x)[0][1]
    return float(proba)


def evaluate_pairwise(
    pair_rows: Sequence[Dict[str, Any]],
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    seed: int,
) -> Dict[str, Any]:
    pair_correct: List[float] = []
    pair_margins: List[float] = []
    all_texts, all_labels = build_binary_examples(pair_rows)
    x_all = vectorizer.transform(all_texts)
    y_pred = model.predict(x_all)
    y_prob = model.predict_proba(x_all)[:, 1]
    y_true = all_labels

    per_pair: List[Dict[str, Any]] = []
    for row in pair_rows:
        query = str(row.get("query", ""))
        pos_text = str(row.get("positive_text", ""))
        neg_text = str(row.get("negative_text", ""))
        pos_score = predict_prob(model, vectorizer, query, pos_text)
        neg_score = predict_prob(model, vectorizer, query, neg_text)
        margin = pos_score - neg_score
        correct = 1.0 if margin > 0.0 else 0.0
        pair_correct.append(correct)
        pair_margins.append(margin)
        per_pair.append(
            {
                "query_id": str(row.get("query_id", "")),
                "positive_doc_id": str(row.get("positive_doc_id", "")),
                "negative_doc_id": str(row.get("negative_doc_id", "")),
                "positive_score": pos_score,
                "negative_score": neg_score,
                "margin": margin,
                "pair_correct": correct,
            }
        )

    out: Dict[str, Any] = {
        "pair_count": len(pair_rows),
        "pair_accuracy": bootstrap_ci(pair_correct, seed=seed + 7),
        "pair_margin": bootstrap_ci(pair_margins, seed=seed + 17),
        "classification_accuracy": float(accuracy_score(y_true, y_pred)) if y_true else float("nan"),
        "classification_log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])) if y_true else float("nan"),
        "per_pair_preview": per_pair[:64],
    }
    if y_true and len(set(y_true)) > 1:
        out["classification_roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["classification_roc_auc"] = float("nan")
    return out


def rerank_queries(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    corpus_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    candidate_k: int,
    hit_k: int,
) -> List[Dict[str, Any]]:
    corpus = [
        {
            "doc_id": str(row.get("doc_id", "")),
            "text": str(row.get("text", "")),
        }
        for row in corpus_rows
        if str(row.get("doc_id", "")).strip()
    ]
    by_doc_id = {row["doc_id"]: row for row in corpus}
    bm25 = BM25Index(corpus)

    rows: List[Dict[str, Any]] = []
    for item in gold_rows:
        qid = str(item.get("query_id", "")).strip()
        query = str(item.get("query", "")).strip()
        if not qid or not query:
            continue
        candidates = bm25.rank(query, max(hit_k, candidate_k))
        q_tokens = tokenize(query)
        bm25_scores = [(doc_id, bm25.score(doc_id=doc_id, query_tokens=q_tokens)) for doc_id in candidates]
        bm25_max = max((score for _, score in bm25_scores), default=0.0)
        scored: List[Tuple[str, float]] = []
        for doc_id, bm25_score in bm25_scores:
            doc_text = str(by_doc_id.get(doc_id, {}).get("text", ""))
            rr_score = predict_prob(model, vectorizer, query, doc_text)
            bm25_norm = (bm25_score / bm25_max) if bm25_max > 0 else 0.0
            score = (0.85 * rr_score) + (0.15 * bm25_norm)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [doc_id for doc_id, _ in scored[:hit_k]]
        rows.append(
            {
                "query_id": qid,
                "query": query,
                "ranked_doc_ids": ranked_doc_ids,
                "candidate_pool_size": len(candidates),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate retrieval reranker and emit ranked predictions.")
    parser.add_argument("--pairs", default="data/retrieval_reranker_pairs_wide.jsonl")
    parser.add_argument("--retrieval-corpus", default="data/retrieval_corpus_wide.jsonl")
    parser.add_argument("--retrieval-gold", default="data/retrieval_gold_wide.jsonl")
    parser.add_argument("--output-dir", default="artifacts/reranker/latest")
    parser.add_argument("--predictions-out", default="artifacts/reranker/latest/predictions_reranker.jsonl")
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--bm25-candidate-k", type=int, default=24)
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    corpus_path = Path(args.retrieval_corpus)
    gold_path = Path(args.retrieval_gold)
    out_dir = Path(args.output_dir)
    pred_path = Path(args.predictions_out)

    for path in (pairs_path, corpus_path, gold_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    pair_rows = read_jsonl(pairs_path)
    corpus_rows = read_jsonl(corpus_path)
    gold_rows = read_jsonl(gold_path)
    if not pair_rows:
        raise RuntimeError(f"Pair dataset is empty: {pairs_path}")

    train_rows, eval_rows = split_by_query(pair_rows, train_frac=float(args.train_frac), seed=int(args.seed))
    if not train_rows or not eval_rows:
        raise RuntimeError("Reranker split failed: need both train and eval rows.")

    train_texts, train_labels = build_binary_examples(train_rows)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000)
    x_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(
        random_state=int(args.seed),
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
    )
    model.fit(x_train, train_labels)

    train_metrics = evaluate_pairwise(train_rows, model, vectorizer, seed=int(args.seed) + 101)
    eval_metrics = evaluate_pairwise(eval_rows, model, vectorizer, seed=int(args.seed) + 211)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, out_dir / "reranker_model.joblib")

    reranked_rows = rerank_queries(
        model=model,
        vectorizer=vectorizer,
        corpus_rows=corpus_rows,
        gold_rows=gold_rows,
        candidate_k=max(int(args.hit_k), int(args.bm25_candidate_k)),
        hit_k=int(args.hit_k),
    )
    write_jsonl(pred_path, reranked_rows)

    summary = {
        "pairs_path": str(pairs_path),
        "retrieval_corpus_path": str(corpus_path),
        "retrieval_gold_path": str(gold_path),
        "pair_rows_total": len(pair_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_query_count": len({str(r.get("query_id", "")) for r in train_rows}),
        "eval_query_count": len({str(r.get("query_id", "")) for r in eval_rows}),
        "train_frac": float(args.train_frac),
        "bm25_candidate_k": int(args.bm25_candidate_k),
        "hit_k": int(args.hit_k),
        "seed": int(args.seed),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "model_artifact": str(out_dir / "reranker_model.joblib"),
        "predictions_path": str(pred_path),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "metrics.json", {"train": train_metrics, "eval": eval_metrics})

    print(f"Reranker model: {out_dir / 'reranker_model.joblib'}")
    print(f"Reranker summary: {out_dir / 'summary.json'}")
    print(f"Reranker predictions: {pred_path}")
    print(f"Pairs: train={len(train_rows)} eval={len(eval_rows)} total={len(pair_rows)}")


if __name__ == "__main__":
    main()
