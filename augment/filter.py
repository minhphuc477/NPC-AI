"""Filtering utilities for generated data.

- persona_check(record)
- semantic_dedupe(records)

This module attempts to import sentence-transformers for robust embeddings; otherwise falls back to sklearn TF-IDF cosine similarities.
"""
from typing import List, Dict


def persona_check(record: Dict) -> bool:
    """Simple persona check: ensure response isn't empty and not too long."""
    resp = record.get("response", "")
    if not resp.strip():
        return False
    if len(resp) > 500:
        return False
    return True


def semantic_dedupe(records: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """Dedupe records by semantic similarity. Falls back to simple TF-IDF cosine.

    Returns filtered list.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [r.get('response','') for r in records]
        emb = model.encode(texts, convert_to_numpy=True)
        keep = []
        for i, v in enumerate(emb):
            dup = False
            for j in keep:
                sim = float(np.dot(v, emb[j]) / (np.linalg.norm(v) * np.linalg.norm(emb[j]) + 1e-12))
                if sim > threshold:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        return [records[i] for i in keep]
    except Exception:
        # fallback to TF-IDF cosine
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        texts = [r.get('response','') for r in records]
        vec = TfidfVectorizer().fit_transform(texts)
        sim = cosine_similarity(vec)
        n = len(records)
        keep = []
        removed = set()
        for i in range(n):
            if i in removed:
                continue
            keep.append(i)
            for j in range(i+1, n):
                if j in removed:
                    continue
                if sim[i, j] > threshold:
                    removed.add(j)
        return [records[i] for i in keep]
