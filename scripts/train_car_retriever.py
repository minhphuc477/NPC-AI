#!/usr/bin/env python3
"""Train Contrastive Adversarial Retriever (CAR) bi-encoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

SEED = 42


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class PairDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        return {
            "query": str(row.get("query", "")).strip(),
            "positive_text": str(row.get("positive_text", "")).strip(),
            "negative_text": str(row.get("negative_text", "")).strip(),
        }


def collate(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        "query": [item["query"] for item in batch],
        "positive_text": [item["positive_text"] for item in batch],
        "negative_text": [item["negative_text"] for item in batch],
    }


def encode_texts(model, tokenizer, device, texts: List[str], max_length: int) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**encoded)
    pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAR bi-encoder on query/positive/negative triples.")
    parser.add_argument("--pairs", default="data/retrieval_reranker_pairs_wide.jsonl")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out-dir", default="storage/outputs/checkpoints/car_retriever")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--train-frac", type=float, default=0.9)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows = read_jsonl(Path(args.pairs))
    rows = [r for r in rows if r.get("query") and r.get("positive_text") and r.get("negative_text")]
    if not rows:
        raise RuntimeError("No valid rows found in pairs file.")
    random.shuffle(rows)

    split = max(1, int(len(rows) * float(args.train_frac)))
    train_rows = rows[:split]
    eval_rows = rows[split:] or rows[: max(1, min(256, len(rows)))]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModel.from_pretrained(args.base_model).to(device)
    model.train()

    train_ds = PairDataset(train_rows)
    eval_ds = PairDataset(eval_rows)
    train_dl = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate)
    eval_dl = DataLoader(eval_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
    temp = max(1e-4, float(args.temperature))
    best_eval = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(args.epochs)):
        model.train()
        total_train_loss = 0.0
        for batch in train_dl:
            query_emb = encode_texts(model, tokenizer, device, batch["query"], int(args.max_length))
            pos_emb = encode_texts(model, tokenizer, device, batch["positive_text"], int(args.max_length))
            neg_emb = encode_texts(model, tokenizer, device, batch["negative_text"], int(args.max_length))

            pos_scores = (query_emb * pos_emb).sum(dim=1, keepdim=True) / temp
            neg_scores = (query_emb * neg_emb).sum(dim=1, keepdim=True) / temp
            logits = torch.cat([pos_scores, neg_scores], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += float(loss.item())

        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_dl:
                query_emb = encode_texts(model, tokenizer, device, batch["query"], int(args.max_length))
                pos_emb = encode_texts(model, tokenizer, device, batch["positive_text"], int(args.max_length))
                neg_emb = encode_texts(model, tokenizer, device, batch["negative_text"], int(args.max_length))
                pos_scores = (query_emb * pos_emb).sum(dim=1, keepdim=True) / temp
                neg_scores = (query_emb * neg_emb).sum(dim=1, keepdim=True) / temp
                logits = torch.cat([pos_scores, neg_scores], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                total_eval_loss += float(loss.item())

        train_loss = total_train_loss / float(max(1, len(train_dl)))
        eval_loss = total_eval_loss / float(max(1, len(eval_dl)))
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

        epoch_dir = out_dir / f"epoch_{epoch + 1:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        (epoch_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "base_model": str(args.base_model),
                    "train_rows": len(train_rows),
                    "eval_rows": len(eval_rows),
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "temperature": temp,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if eval_loss < best_eval:
            best_eval = eval_loss
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "base_model": str(args.base_model),
                        "train_rows": len(train_rows),
                        "eval_rows": len(eval_rows),
                        "best_eval_loss": best_eval,
                        "temperature": temp,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(f"training_complete best_eval_loss={best_eval:.4f}")
    print(f"checkpoint_root={out_dir}")


if __name__ == "__main__":
    main()

