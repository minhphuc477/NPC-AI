"""Colab helpers for QLoRA finetuning & model handling (NPC AI)

This file contains small utilities used by the Colab notebook. Designed to be
importable and unit-test friendly (no heavy GPU or network usage in unit tests).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
import random
import shutil
import logging
import os
import time
import hashlib
import requests
from datetime import datetime

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


# Compatibility wrapper expected by tests and the notebook
def convert_csv_to_jsonl(csv_path: str, out_path: str, text_col: str = "text", label_cols=None) -> int:
    """Convenience wrapper around existing csv_to_jsonl or project converter."""
    # Try the local converter first; if it writes nothing, fall back to a simple converter
    try:
        n = csv_to_jsonl(csv_path, out_path)
        if n and n > 0:
            return n
    except Exception:
        pass
    # fallback: use annotation pipeline converter if present
    try:
        from annotation_pipeline.convert_csv_to_jsonl import csv_to_jsonl as conv
        n2 = conv(csv_path, out_path)
        if n2 and n2 > 0:
            return n2
    except Exception:
        pass
    # last-resort simple conversion
    import csv, json
    written = 0
    with open(csv_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
            rec = {"id": str(i), "text": row.get(text_col, "")}
            if label_cols:
                rec["labels"] = {c: row.get(c) for c in label_cols}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


# Simple production-ready prompt renderer used in notebook and tests
_PROMPT_TEMPLATE = """Bạn là một NPC trong một trò chơi phiêu lưu. Trạng thái môi trường: {state}
Người chơi: {player}
NPC trả lời: """


def render_prompt(state: str, player_utterance: str) -> str:
    return _PROMPT_TEMPLATE.format(state=state, player=player_utterance)


def call_groq_api(prompt: str, model_id: str = 'llama-3.1-8b', max_tokens: int = 128, temperature: float = 0.7, api_url: Optional[str] = None, api_key: Optional[str] = None, retries: int = 3) -> str:
    """Call Groq REST Responses API and return the generated text.

    The function respects environment variables `GROQ_API_URL` and `GROQ_API_KEY`
    when `api_url` or `api_key` are not provided. It implements basic retry
    with exponential backoff for HTTP 429 (rate limit) responses and logs
    progress. Returns the text content or raises on non-recoverable errors.
    """
    api_url = api_url or os.environ.get('GROQ_API_URL', 'https://api.groq.com/openai/v1/responses')
    api_key = api_key or os.environ.get('GROQ_API_KEY')

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    payload = {
        'model': model_id,
        'input': prompt,
        'max_output_tokens': max_tokens,
        'temperature': temperature,
    }

    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            logger.debug("Calling Groq API url=%s attempt=%d", api_url, attempt)
            resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Groq request exception (attempt %d/%d): %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 429:
            logger.warning("Groq rate limited (429). Backing off for %.1fs (attempt %d/%d)", backoff, attempt, retries)
            if attempt == retries:
                resp.raise_for_status()
            time.sleep(backoff)
            backoff *= 2
            continue

        if not resp.ok:
            logger.error("Groq API returned status=%s body=%s", resp.status_code, resp.text)
            resp.raise_for_status()

        # parse response robustly
        try:
            data = resp.json()
        except ValueError:
            logger.error("Groq response not JSON: %s", resp.text[:200])
            return resp.text

        # Possible shapes: {'output': 'text'}, {'output': [{'content': 'text'}]}, {'choices': [...]}
        text = None
        if isinstance(data, dict):
            if 'output' in data:
                out = data['output']
                if isinstance(out, str):
                    text = out
                elif isinstance(out, list) and out:
                    # try to pull content fields
                    first = out[0]
                    if isinstance(first, dict):
                        text = first.get('content') or first.get('text') or None
                    elif isinstance(first, str):
                        text = first
            elif 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                c = data['choices'][0]
                if isinstance(c, dict):
                    text = c.get('text') or c.get('message') or None
        if text is None:
            # fallback: try to stringify 'data' or specific keys
            text = str(data)
        return text


def batch_generate_with_groq(prompts: List[str], model_id: str, batch_size: int = 32) -> List[str]:
    """Batch-generate outputs from Groq with a simple local file cache.

    Cache files are stored under `.cache/groq/` with filenames of the form
    `<sha256>.json` containing keys: 'prompt','model','output','ts'. If a cached
    entry exists it will be returned instead of calling the network.
    """
    out = []
    cache_dir = Path('.cache') / 'groq'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # helper to get cache path
    def _cache_path(prompt_text: str) -> Path:
        h = hashlib.sha256((prompt_text + model_id).encode('utf-8')).hexdigest()
        return cache_dir / f"{h}.json"

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        for p in batch:
            cp = _cache_path(p)
            if cp.exists():
                try:
                    cdata = json.loads(cp.read_text(encoding='utf-8'))
                    out.append(cdata.get('output', ''))
                    continue
                except Exception:
                    logger.warning("Failed to read cache file %s, regenerating", cp)
            # call API and cache result
            try:
                txt = call_groq_api(p, model_id=model_id)
            except Exception as exc:
                logger.error("Groq generation failed for prompt: %s (%s)", p[:80], exc)
                txt = ''
            # write cache
            try:
                cp.write_text(json.dumps({'prompt': p, 'model': model_id, 'output': txt, 'ts': datetime.utcnow().isoformat()}), encoding='utf-8')
            except Exception:
                logger.warning("Failed to write cache file %s", cp)
            out.append(txt)
            # naive rate-limit delay
            time.sleep(0.05)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("jsonl")
    p.add_argument("--augment", action="store_true")
    args = p.parse_args()
    csv_to_jsonl(args.csv, args.jsonl)
    print("Done")
