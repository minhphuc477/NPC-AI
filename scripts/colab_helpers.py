"""Colab helpers for QLoRA finetuning & model handling (NPC AI)

This file contains small utilities used by the Colab notebook. Designed to be
importable and unit-test friendly (no heavy GPU or network usage in unit tests).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import random
import shutil
import logging

logger = logging.getLogger(__name__)


def csv_to_jsonl(csv_path: str, jsonl_path: str) -> int:
    """Wrapper around the project converter.

    Uses `annotation_pipeline.convert_csv_to_jsonl.csv_to_jsonl` when available.
    Returns number of rows written.
    """
    try:
        from annotation_pipeline.convert_csv_to_jsonl import csv_to_jsonl as conv
    except Exception:
        raise
    return conv(csv_path, jsonl_path)


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


def write_jsonl(records: Iterable[dict], path: str) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("w", encoding="utf-8") as wf:
        for r in records:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
    return written


def train_val_split(jsonl_path: str, train_out: str, val_out: str, val_frac: float = 0.1, seed: int = 42) -> Tuple[int, int]:
    records = read_jsonl(jsonl_path)
    random.Random(seed).shuffle(records)
    n_val = max(1, int(len(records) * val_frac))
    val = records[:n_val]
    train = records[n_val:]
    w1 = write_jsonl(train, train_out)
    w2 = write_jsonl(val, val_out)
    logger.info("Split %s -> train=%d val=%d", jsonl_path, w1, w2)
    return w1, w2


def augment_via_templates(records: List[dict], target: int = 500, seed: int = 2025) -> List[dict]:
    """Lightweight augmentation using simple templates and value swaps.

    This is intentionally cheap and runs without external models. It creates
    paraphrase variants by re-ordering clauses and swapping polite forms. Use
    only for small-sample augmentation in Colab smoke runs; users should prefer
    LLM-based paraphrasing for production-quality augmentation.
    """
    rng = random.Random(seed)
    out = list(records)
    templates = [
        "{instruction}",
        "Please {instruction}",
        "A player says: '{instruction}'",
        "When prompted: {instruction}",
    ]
    i = 0
    while len(out) < target and i < target * 5:
        base = rng.choice(records)
        t = rng.choice(templates)
        instr = t.format(instruction=base.get("instruction", ""))
        new = dict(base)
        new["instruction"] = instr
        # keep output the same but prepend a short marker to indicate synthetic
        new["output"] = base.get("output", "")
        new["text"] = f"{instr}\n\nNPC: {new['output']}"
        out.append(new)
        i += 1
    return out


def make_modelfile_for_ollama(model_name: str, gguf_path: str, description: str = "NPC QLoRA adapter") -> str:
    """Create a minimal Modelfile text for Ollama that references a local GGUF path.

    Returns the Modelfile contents as a string for the user to save as `Modelfile`.
    """
    mf = f"""image: {model_name}
package:
  type: local
  model_path: {gguf_path}
  description: "{description}"
"""
    return mf


# small helper to show recommended conversion commands in one place
GGUF_CONVERSION_HELP = """
# Example: using `transformers-4` utilities to convert a model to GGUF (q4_k_m quant)
# 1) Save your QLoRA-resolved model to a HF folder (adapter merged if needed).
# 2) Use `python -m gguf.convert --model <hf-folder> --out <out.gguf> --quantization q4_k_m`
# Alternatively, see https://github.com/johnsmith/gguf for details (link your favourite converter)
"""


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("jsonl")
    p.add_argument("--augment", action="store_true")
    args = p.parse_args()
    csv_to_jsonl(args.csv, args.jsonl)
    print("Done")
