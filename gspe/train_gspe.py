#!/usr/bin/env python3
"""Train GSPE with supervised next-token objective."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Dict, List
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gspe.gspe_model import GSPE, GSPEConfig, encode_game_state

SEED = 42


class GSPEDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        persona = str(row.get("persona", "NPC")).strip()
        player_input = str(row.get("player_input", "")).strip()
        response = str(row.get("response", "")).strip()
        passages = row.get("retrieved_passages", [])
        evidence = " | ".join(str(p).strip() for p in passages[:3] if str(p).strip())

        prompt = f"<|system|>\nYou are {persona}."
        if evidence:
            prompt += f"\nContext: {evidence}"
        prompt += f"\n<|user|>\n{player_input}\n<|assistant|>\n"
        full = prompt + response

        encoded = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        prompt_only = self.tokenizer(prompt, return_tensors="pt")
        n_prompt = int(prompt_only["input_ids"].shape[1])
        labels = input_ids.clone()
        labels[:n_prompt] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "game_state_ids": encode_game_state(dict(row.get("game_state", {}))),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(int(item["input_ids"].shape[0]) for item in batch)
    bsz = len(batch)

    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        n = int(item["input_ids"].shape[0])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        labels[i, :n] = item["labels"]

    game_state_ids: Dict[str, torch.Tensor] = {}
    for field in batch[0]["game_state_ids"].keys():
        game_state_ids[field] = torch.tensor(
            [int(item["game_state_ids"][field]) for item in batch],
            dtype=torch.long,
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "game_state_ids": game_state_ids,
    }


def forward_with_prefix(
    model: Any,
    gspe: GSPE,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    game_state_ids: Dict[str, torch.Tensor],
) -> torch.Tensor:
    device = input_ids.device
    gspe_input = {key: value.to(device) for key, value in game_state_ids.items()}
    virtual_tokens = gspe(gspe_input)

    embed_layer = model.get_input_embeddings()
    token_embeds = embed_layer(input_ids)
    inputs_embeds = torch.cat([virtual_tokens, token_embeds], dim=1)

    prefix_mask = torch.ones(
        (input_ids.shape[0], gspe.cfg.n_prefix_tokens),
        dtype=attention_mask.dtype,
        device=device,
    )
    extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    prefix_labels = torch.full(
        (input_ids.shape[0], gspe.cfg.n_prefix_tokens),
        -100,
        dtype=labels.dtype,
        device=device,
    )
    extended_labels = torch.cat([prefix_labels, labels], dim=1)

    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=extended_mask,
        labels=extended_labels,
    )
    return output.loss


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_checkpoint(out_dir: Path, gspe: GSPE, tokenizer, metadata: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gspe.state_dict(), out_dir / "gspe.pt")
    (out_dir / "gspe_config.json").write_text(
        json.dumps(dataclasses.asdict(gspe.cfg), indent=2),
        encoding="utf-8",
    )
    tokenizer.save_pretrained(out_dir / "tokenizer")
    (out_dir / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GSPE prefix encoder.")
    parser.add_argument("--data", required=True, help="Prepared GSPE JSONL.")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--out-dir", default="storage/outputs/checkpoints/gspe_v1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--freeze-lm", action="store_true", help="Freeze base LM weights (recommended).")
    parser.add_argument("--unfreeze-lm", action="store_true", help="Train base LM together with GSPE.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_path = Path(args.data)
    rows = read_jsonl(data_path)
    if not rows:
        raise RuntimeError(f"No rows in {data_path}")
    random.shuffle(rows)
    split_idx = max(1, int(len(rows) * float(args.train_frac)))
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:] or rows[: max(1, min(128, len(rows)))]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {}
    if bool(args.load_in_4bit):
        model_kwargs["load_in_4bit"] = True
    model: Any = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if "load_in_4bit" not in model_kwargs:
        model = model.to(device)

    freeze_lm = bool(args.freeze_lm) or not bool(args.unfreeze_lm)
    if freeze_lm:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    else:
        model.train()

    hidden_dim = int(getattr(model.config, "hidden_size", 3072))
    gspe_cfg = GSPEConfig(lm_hidden_dim=hidden_dim)
    gspe = GSPE(gspe_cfg).to(device)

    train_ds = GSPEDataset(train_rows, tokenizer, max_length=int(args.max_length))
    eval_ds = GSPEDataset(eval_rows, tokenizer, max_length=int(args.max_length))
    train_dl = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_batch)
    eval_dl = DataLoader(eval_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_batch)

    params = [p for p in gspe.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=0.01)

    best_eval = float("inf")
    step_count = 0
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(args.epochs)):
        gspe.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for idx, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = forward_with_prefix(
                model=model,
                gspe=gspe,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                game_state_ids=batch["game_state_ids"],
            )
            loss = loss / max(1, int(args.grad_accum))
            loss.backward()
            train_loss += float(loss.item()) * max(1, int(args.grad_accum))

            if idx % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

        gspe.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                loss = forward_with_prefix(
                    model=model,
                    gspe=gspe,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    game_state_ids=batch["game_state_ids"],
                )
                eval_loss += float(loss.item())
        eval_loss = eval_loss / float(max(1, len(eval_dl)))
        train_loss = train_loss / float(max(1, len(train_dl)))
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

        metadata = {
            "base_model": str(args.base_model),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "learning_rate": float(args.lr),
            "max_length": int(args.max_length),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "steps": step_count,
            "eval_loss": eval_loss,
            "gspe_parameters": gspe.num_parameters(),
            "freeze_lm": bool(freeze_lm),
        }

        epoch_dir = out_root / f"epoch_{epoch + 1:02d}"
        save_checkpoint(epoch_dir, gspe, tokenizer, metadata)
        if eval_loss < best_eval:
            best_eval = eval_loss
            save_checkpoint(out_root / "best", gspe, tokenizer, metadata)

    print(f"training_complete best_eval_loss={best_eval:.4f}")
    print(f"checkpoint_root={out_root}")


if __name__ == "__main__":
    main()
