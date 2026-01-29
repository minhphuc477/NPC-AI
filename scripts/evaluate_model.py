"""Evaluation utilities: BERTScore, TF-IDF similarity fallback, and visualization.

Intended to be lightweight so unit tests can import and run without GPU. If
BERTScore or SBERT are not installed, falls back to TF-IDF cosine similarity.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[dict]:
    p = Path(path)
    out = []
    with p.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def compute_tfidf_similarity(hyps: List[str], refs: List[str]) -> Dict[str, Any]:
    vect = TfidfVectorizer().fit(hyps + refs)
    H = vect.transform(hyps)
    R = vect.transform(refs)
    sims = [float(cosine_similarity(H[i], R[i])[0, 0]) for i in range(len(hyps))]
    return {"mean_cosine": float(np.mean(sims)), "per_example": sims}


def compute_bertscore(hyps: List[str], refs: List[str]) -> Dict[str, Any]:
    try:
        from bert_score import score as bert_score
    except Exception:
        logger.warning("bert-score not available; skipping BERTScore")
        return {"bertscore_available": False}
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    return {"bertscore_available": True, "p": [float(x) for x in P], "r": [float(x) for x in R], "f1": [float(x) for x in F1], "f1_mean": float(float(np.mean(F1)))}


def run_evaluation(hyp_jsonl: str, ref_jsonl: str = None, out_dir: str = "outputs_eval") -> Dict[str, Any]:
    hyps = read_jsonl(hyp_jsonl)
    refs = read_jsonl(ref_jsonl) if ref_jsonl else None
    # align by index for simple evaluation
    hypo_texts = [h.get("output", h.get("text", "")) for h in hyps]
    if refs:
        ref_texts = [r.get("output", r.get("text", "")) for r in refs]
    else:
        # if no explicit references, use the original dataset's output field when available
        ref_texts = [h.get("output", "") for h in hyps]

    metrics = {}
    # TF-IDF baseline
    metrics["tfidf"] = compute_tfidf_similarity(hypo_texts, ref_texts)
    # BERTScore when available
    b = compute_bertscore(hypo_texts, ref_texts)
    metrics["bertscore"] = b

    # visualize similarity histogram
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    sims = metrics["tfidf"]["per_example"]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(sims, bins=20)
    ax.set_title("TF-IDF cosine similarity distribution")
    ax.set_xlabel("cosine")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(str(p / "similarity_hist.png"))
    plt.close(fig)

    # simple state distribution if present
    states = [h.get("npc_state") for h in hyps if h.get("npc_state")]
    if states:
        fig, ax = plt.subplots(figsize=(4, 3))
        from collections import Counter

        c = Counter(states)
        ax.bar(list(c.keys()), list(c.values()))
        ax.set_title("NPC state distribution (generated)")
        fig.tight_layout()
        fig.savefig(str(p / "state_distribution.png"))
        plt.close(fig)

    # create a small iaa-like report: sample cases and metrics
    report_lines = ["# Evaluation Summary", "", f"TF-IDF mean cosine: {metrics['tfidf']['mean_cosine']}"]
    if b.get("bertscore_available"):
        report_lines.append(f"BERTScore F1 mean: {b.get('f1_mean')}")
    else:
        report_lines.append("BERTScore: not available (install bert-score for advanced metrics)")

    report_lines.append("\n## Sample cases")
    for i in range(min(5, len(hypo_texts))):
        report_lines.append(f"### Case {i}")
        report_lines.append("- Hypothesis:\n\n```")
        report_lines.append(hypo_texts[i])
        report_lines.append("```")
        report_lines.append("- Reference:\n\n```")
        report_lines.append(ref_texts[i])
        report_lines.append("```")
        report_lines.append("")

    (p / "evaluation.md").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Evaluation written to %s", p)
    return {"metrics": metrics, "out_dir": str(p)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("hyp")
    p.add_argument("ref", nargs="?", default=None)
    p.add_argument("--out", default="outputs_eval")
    args = p.parse_args()
    run_evaluation(args.hyp, args.ref, args.out)
