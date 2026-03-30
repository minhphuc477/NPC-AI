"""Contrastive Adversarial Retriever (CAR) utilities.

This module provides:
- Transformer encoder wrapper for dense retrieval
- Index build/search helpers
- Save/load support for reusable CAR indices
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@dataclass
class CAREncoderConfig:
    model_name_or_path: str
    device: str = "auto"
    max_length: int = 256


class CAREncoder:
    def __init__(self, config: CAREncoderConfig):
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model = AutoModel.from_pretrained(config.model_name_or_path).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            chunk = [str(text) for text in texts[start : start + int(batch_size)]]
            encoded = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=int(self.config.max_length),
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**encoded)
            pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype="float32")


@dataclass
class CARIndex:
    doc_ids: List[str]
    embeddings: np.ndarray

    def rank(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self.embeddings.size == 0:
            return []
        q = query_embedding.reshape(1, -1).astype("float32")
        scores = (self.embeddings @ q.T).reshape(-1)
        if scores.size == 0:
            return []
        k = max(1, min(int(top_k), scores.shape[0]))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, doc_ids=np.array(self.doc_ids, dtype=object), embeddings=self.embeddings)

    @staticmethod
    def load(path: Path) -> "CARIndex":
        payload = np.load(path, allow_pickle=True)
        doc_ids = [str(x) for x in payload["doc_ids"].tolist()]
        embeddings = payload["embeddings"].astype("float32")
        return CARIndex(doc_ids=doc_ids, embeddings=embeddings)


class CARRetriever:
    def __init__(self, encoder: CAREncoder, index: CARIndex):
        self.encoder = encoder
        self.index = index

    @classmethod
    def from_checkpoint_and_index(
        cls,
        *,
        checkpoint: str,
        index_path: str,
        device: str = "auto",
        max_length: int = 256,
    ) -> "CARRetriever":
        encoder = CAREncoder(
            CAREncoderConfig(
                model_name_or_path=str(checkpoint),
                device=str(device),
                max_length=int(max_length),
            )
        )
        index = CARIndex.load(Path(index_path))
        return cls(encoder=encoder, index=index)

    @classmethod
    def from_checkpoint_and_corpus(
        cls,
        *,
        checkpoint: str,
        corpus_rows: Sequence[Dict[str, Any]],
        device: str = "auto",
        max_length: int = 256,
        batch_size: int = 32,
    ) -> "CARRetriever":
        encoder = CAREncoder(
            CAREncoderConfig(
                model_name_or_path=str(checkpoint),
                device=str(device),
                max_length=int(max_length),
            )
        )
        doc_ids: List[str] = []
        texts: List[str] = []
        for row in corpus_rows:
            doc_id = str(row.get("doc_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not doc_id or not text:
                continue
            doc_ids.append(doc_id)
            texts.append(text)
        embeddings = encoder.encode(texts, batch_size=batch_size)
        index = CARIndex(doc_ids=doc_ids, embeddings=embeddings)
        return cls(encoder=encoder, index=index)

    def rank(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        query_vec = self.encoder.encode([str(query)], batch_size=1)
        if query_vec.shape[0] == 0:
            return []
        return self.index.rank(query_vec[0], top_k=top_k)

