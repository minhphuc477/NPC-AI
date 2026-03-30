#!/usr/bin/env python3
"""Inference runtime for GSPE + local HuggingFace causal LM."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gspe.gspe_model import GSPE, GSPEConfig
from gspe.state_codec import encode_game_state


@dataclasses.dataclass
class GenerationConfig:
    max_new_tokens: int = 80
    temperature: float = 0.2
    do_sample: bool = True
    repetition_penalty: float = 1.05


@dataclasses.dataclass
class PrefixCacheEntry:
    value: torch.Tensor
    created_ms: float


class PrefixCache:
    def __init__(self, max_size: int = 32, ttl_ms: float = 30_000.0):
        self.max_size = int(max_size)
        self.ttl_ms = float(ttl_ms)
        self._items: Dict[str, PrefixCacheEntry] = {}
        self._order: List[str] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(gs_ids: Dict[str, int]) -> str:
        raw = json.dumps(gs_ids, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, gs_ids: Dict[str, int]) -> Optional[torch.Tensor]:
        key = self.key(gs_ids)
        entry = self._items.get(key)
        if entry is None:
            self.misses += 1
            return None
        age = (time.time() * 1000.0) - float(entry.created_ms)
        if age > self.ttl_ms:
            self._items.pop(key, None)
            if key in self._order:
                self._order.remove(key)
            self.misses += 1
            return None
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self.hits += 1
        return entry.value

    def put(self, gs_ids: Dict[str, int], value: torch.Tensor) -> None:
        key = self.key(gs_ids)
        if key not in self._items and len(self._items) >= self.max_size:
            oldest = self._order.pop(0)
            self._items.pop(oldest, None)
        self._items[key] = PrefixCacheEntry(value=value, created_ms=time.time() * 1000.0)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def invalidate_key(self, key: str) -> bool:
        key_text = str(key or "").strip()
        if not key_text:
            return False
        removed = self._items.pop(key_text, None) is not None
        if key_text in self._order:
            self._order.remove(key_text)
        return removed

    def invalidate_ids(self, gs_ids: Dict[str, int]) -> bool:
        return self.invalidate_key(self.key(gs_ids))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits) / float(total) if total else 0.0


class GSPEInferenceEngine:
    def __init__(
        self,
        *,
        gspe: GSPE,
        model: Any,
        tokenizer,
        device: torch.device,
        generation_config: Optional[GenerationConfig] = None,
        invalidation_log_path: Optional[Path] = None,
    ):
        self.gspe = gspe.eval().to(device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = generation_config or GenerationConfig()
        self.cache = PrefixCache()
        self._invalidation_log_path = Path(invalidation_log_path) if invalidation_log_path else None
        self._invalidation_log_bytes = 0
        self._external_invalidations_applied = 0

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_dir: str,
        base_model: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        invalidation_log_path: str = "storage/artifacts/gspe/prefix_cache_invalidate_events.jsonl",
    ) -> "GSPEInferenceEngine":
        ckpt = Path(checkpoint_dir)
        cfg = GSPEConfig(**json.loads((ckpt / "gspe_config.json").read_text(encoding="utf-8")))
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        model_kwargs: Dict[str, Any] = {}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        model: Any = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if "load_in_4bit" not in model_kwargs:
            model = cast(Any, model).to(dev)

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        gspe = GSPE(cfg)
        gspe.load_state_dict(torch.load(ckpt / "gspe.pt", map_location=dev))
        return cls(
            gspe=gspe,
            model=model,
            tokenizer=tokenizer,
            device=dev,
            invalidation_log_path=Path(invalidation_log_path) if str(invalidation_log_path).strip() else None,
        )

    def _apply_external_invalidations(self) -> None:
        path = self._invalidation_log_path
        if path is None or not path.exists():
            return
        try:
            file_size = path.stat().st_size
        except Exception:
            return
        if file_size < self._invalidation_log_bytes:
            self._invalidation_log_bytes = 0

        try:
            with path.open("rb") as handle:
                handle.seek(self._invalidation_log_bytes)
                while True:
                    raw = handle.readline()
                    if not raw:
                        break
                    self._invalidation_log_bytes = handle.tell()
                    try:
                        row = json.loads(raw.decode("utf-8").strip())
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    state_hash = str(row.get("state_hash", "")).strip()
                    if state_hash and self.cache.invalidate_key(state_hash):
                        self._external_invalidations_applied += 1
        except Exception:
            # Best-effort side channel; generation path should not fail on log parse errors.
            return

    def _build_prompt(
        self,
        *,
        persona: str,
        player_input: str,
        retrieved_passages: Optional[List[str]] = None,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        retrieved = retrieved_passages or []
        history = dialogue_history or []
        evidence = " | ".join(str(item).strip() for item in retrieved[:3] if str(item).strip())
        system = f"You are {persona}."
        if evidence:
            system += f"\nContext: {evidence}"

        lines = [f"<|system|>\n{system}"]
        for turn in history[-4:]:
            role = str(turn.get("role", "player")).strip().lower()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            token = "<|assistant|>" if role == "assistant" else "<|user|>"
            lines.append(f"{token}\n{text}")
        lines.append(f"<|user|>\n{player_input}")
        lines.append("<|assistant|>\n")
        return "\n".join(lines)

    def _prefix_tokens(self, game_state: Dict[str, Any]) -> torch.Tensor:
        self._apply_external_invalidations()
        gs_ids = encode_game_state(game_state)
        cached = self.cache.get(gs_ids)
        if cached is not None:
            return cached
        field_ids = {
            field: torch.tensor([idx], dtype=torch.long, device=self.device)
            for field, idx in gs_ids.items()
        }
        with torch.no_grad():
            prefix = self.gspe(field_ids)
        self.cache.put(gs_ids, prefix)
        return prefix

    @torch.inference_mode()
    def generate(
        self,
        *,
        game_state: Dict[str, Any],
        persona: str,
        player_input: str,
        retrieved_passages: Optional[List[str]] = None,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        start_ns = time.perf_counter_ns()
        prompt = self._build_prompt(
            persona=persona,
            player_input=player_input,
            retrieved_passages=retrieved_passages,
            dialogue_history=dialogue_history,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        prefix = self._prefix_tokens(game_state)
        embeds = self.model.get_input_embeddings()(encoded["input_ids"])
        inputs_embeds = torch.cat([prefix, embeds], dim=1)

        prefix_mask = torch.ones(
            (1, prefix.shape[1]),
            dtype=encoded["attention_mask"].dtype,
            device=self.device,
        )
        attention_mask = torch.cat([prefix_mask, encoded["attention_mask"]], dim=1)
        prefill_ns = time.perf_counter_ns()

        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=int(self.generation_config.max_new_tokens),
            temperature=float(self.generation_config.temperature),
            do_sample=bool(self.generation_config.do_sample),
            repetition_penalty=float(self.generation_config.repetition_penalty),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        end_ns = time.perf_counter_ns()

        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return {
            "response": text,
            "ttft_ms": (prefill_ns - start_ns) / 1_000_000.0,
            "total_ms": (end_ns - start_ns) / 1_000_000.0,
            "prefix_cache_hit_rate": self.cache.hit_rate,
            "external_invalidations_applied": self._external_invalidations_applied,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="GSPE local inference smoke runner.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--invalidation-log-path",
        default="storage/artifacts/gspe/prefix_cache_invalidate_events.jsonl",
        help="Shared JSONL invalidation log produced by UE BT handler.",
    )
    parser.add_argument("--persona", default="Captain of the City Watch")
    parser.add_argument("--player-input", default="I need to pass through the gate right now.")
    parser.add_argument(
        "--game-state-json",
        default='{"alert_state":"investigating","location":"city_gate","behavior_state":"guarding"}',
    )
    args = parser.parse_args()

    engine = GSPEInferenceEngine.from_checkpoint(
        checkpoint_dir=str(args.checkpoint_dir),
        base_model=str(args.base_model),
        device=str(args.device),
        load_in_4bit=bool(args.load_in_4bit),
        invalidation_log_path=str(args.invalidation_log_path),
    )
    game_state = json.loads(str(args.game_state_json))
    result = engine.generate(
        game_state=game_state,
        persona=str(args.persona),
        player_input=str(args.player_input),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
