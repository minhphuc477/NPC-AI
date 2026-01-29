"""Evaluation utilities: compute BERTScore (if available), TF-IDF cosine similarity, and save summaries/plots."""
from typing import List, Dict
import json
from pathlib import Path
import numpy as np

def compute_tfidf_cosine(refs: List[str], hyps: List[str]) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer().fit_transform(list(refs) + list(hyps))
    r = vec[:len(refs)]
    h = vec[len(refs):]
    sims = []
    n = min(r.shape[0], h.shape[0])
    for i in range(n):
        sims.append(float(cosine_similarity(r[i], h[i])[0,0]))
    return sims


def compute_metrics(refs: List[str], hyps: List[str]) -> Dict:
    """Compute metrics and return a dict. Uses BERTScore if available else fallback simple token-overlap."""
    out = {}
    # Try BERTScore
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(hyps, refs, lang='vi', rescale_with_baseline=True)
        out['bertscore_f1'] = float(F1.mean().item())
    except Exception:
        # fallback: token overlap
        def overlap(a,b):
            sa = set(a.split())
            sb = set(b.split())
            if not sa or not sb: return 0.0
            return len(sa & sb)/len(sa | sb)
        out['token_overlap'] = float(np.mean([overlap(r,h) for r,h in zip(refs, hyps)]))
    out['tfidf_cosine'] = float(np.mean(compute_tfidf_cosine(refs, hyps)))
    return out


def summarize_metrics(metrics: Dict, out_dir: str = "evaluation") -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    summary = p / "summary.json"
    with summary.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # small plot
    try:
        import matplotlib.pyplot as plt
        vals = [metrics.get('tfidf_cosine',0.0)]
        plt.figure()
        plt.hist(vals, bins=10)
        plt.title('TF-IDF cosine (sample)')
        plt.savefig(p / "plots_sim.png")
        plt.close()
    except Exception:
        pass
    return summary