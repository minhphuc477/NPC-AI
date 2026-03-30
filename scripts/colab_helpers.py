"""Kaggle/Colab helpers for QLoRA finetuning and model handling (NPC AI).

This file contains lightweight utilities used by notebook workflows.
It is importable and unit-test friendly (no heavy GPU/network work at import time).
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


def in_kaggle() -> bool:
    return Path("/kaggle").exists()


def in_colab() -> bool:
    try:
        import sys

        return "google.colab" in sys.modules
    except Exception:
        return False


def runtime_name() -> str:
    if in_kaggle():
        return "Kaggle"
    if in_colab():
        return "Colab"
    return "Local"


def runtime_working_dir() -> Path:
    """Return a writable working root for the current runtime."""
    if in_kaggle():
        return Path("/kaggle/working")
    return Path.cwd()


def resolve_writable_path(path: str | Path) -> Path:
    """Resolve a path to a writable location in notebook runtimes."""
    p = Path(path)
    if p.is_absolute():
        return p
    return runtime_working_dir() / p


def ensure_hf_cache(cache_dir: str | Path = "hf_cache") -> Path:
    """Create and return a HuggingFace cache path in writable storage."""
    p = resolve_writable_path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(p))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(p / "transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(p / "hub"))
    return p


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
    except Exception as exc:
        logger.warning("Primary csv_to_jsonl converter failed: %s", exc)
    # fallback: use annotation pipeline converter if present
    try:
        from annotation_pipeline.convert_csv_to_jsonl import csv_to_jsonl as conv
        n2 = conv(csv_path, out_path)
        if n2 and n2 > 0:
            return n2
    except Exception as exc:
        logger.warning("Annotation pipeline converter failed: %s", exc)
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
# BD-NSCA Prompt Formatter
# Structure: [Persona] + [Fixed Plot] + [Dynamic Context] + [Player Input]

class BDNSCAPromptFormatter:
    """Formatter for the Behavior-Driven Neuro-Symbolic Cognitive Architecture."""
    
    TEMPLATE = """<|system|>
{persona}

Cốt truyện:
{plot}

Ngữ cảnh hiện tại:
{context}

<|user|>
{player_input}
<|assistant|>
"""

    @staticmethod
    def format(persona: str, plot: str, context: dict, player_input: str) -> str:
        """Combine the 4 layers into a single prompt.
        
        Args:
            persona: Static character description.
            plot: Fixed narrative background/scenario.
            context: Dynamic telemetry from UE5 (Behavior Tree state, Location, etc.)
            player_input: The player's utterance.
        """
        # Format dynamic context into a readable block
        ctx_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
        
        return BDNSCAPromptFormatter.TEMPLATE.format(
            persona=persona.strip(),
            plot=plot.strip(),
            context=ctx_str,
            player_input=player_input.strip()
        )

# Legacy support / Simple wrappers
def render_prompt(state: str, player_utterance: str, language: str = "vi") -> str:
    """Legacy wrapper. Adapts simple state string to BD-NSCA format."""
    return BDNSCAPromptFormatter.format(
        persona="Bạn là một NPC trong trò chơi.",
        plot="Không có thông tin cốt truyện.",
        context={"Môi trường": state},
        player_input=player_utterance
    )


# Global API key rotation state
_GROQ_API_KEYS = []
_CURRENT_KEY_INDEX = 0

def _get_all_api_keys() -> list:
    """Get all available API keys from environment."""
    global _GROQ_API_KEYS
    if not _GROQ_API_KEYS:
        # Check for multiple keys in GROQ_API_KEYS (comma separated) or single GROQ_API_KEY
        keys_str = os.environ.get('GROQ_API_KEYS', '')
        if keys_str:
            _GROQ_API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]
        single_key = os.environ.get('GROQ_API_KEY', '')
        if single_key and single_key not in _GROQ_API_KEYS:
            _GROQ_API_KEYS.append(single_key)
    return _GROQ_API_KEYS

def _rotate_api_key() -> str:
    """Rotate to next available API key."""
    global _CURRENT_KEY_INDEX
    keys = _get_all_api_keys()
    if not keys:
        raise ValueError("No GROQ_API_KEY(s) set.")
    _CURRENT_KEY_INDEX = (_CURRENT_KEY_INDEX + 1) % len(keys)
    logger.info(f"Rotated to API key {_CURRENT_KEY_INDEX + 1}/{len(keys)}")
    return keys[_CURRENT_KEY_INDEX]

def _get_current_api_key() -> str:
    """Get current API key."""
    keys = _get_all_api_keys()
    if not keys:
        raise ValueError("No GROQ_API_KEY(s) set.")
    return keys[_CURRENT_KEY_INDEX % len(keys)]


def call_groq_api(prompt: str, model_id: str = 'llama-3.3-70b-versatile', max_tokens: int = 128, temperature: float = 0.7, api_url: Optional[str] = None, api_key: Optional[str] = None, retries: int = 3) -> str:
    """Call Groq Chat Completions API and return the generated text.

    Supports multiple API keys via GROQ_API_KEYS env var (comma-separated).
    Automatically rotates keys on rate limit (429).
    
    Supported production models (as of 2026):
    - llama-3.3-70b-versatile: Best quality, production-ready
    - llama-3.1-8b-instant: Fast, good for high-volume tasks
    """
    api_url = api_url or os.environ.get('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')
    
    # Use provided key or get from rotation pool
    current_key = api_key or _get_current_api_key()

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {current_key}'
    }

    # Use Chat Completions format
    payload = {
        'model': model_id,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }

    backoff = 1.0
    keys_tried = 0
    max_key_rotations = len(_get_all_api_keys()) if not api_key else 1
    
    attempt = 0
    backoff = 1.0
    keys_tried = 0
    max_key_rotations = len(_get_all_api_keys()) if not api_key else 1

    while True:
        attempt += 1
        try:
            logger.debug("Calling Groq API url=%s attempt=%d", api_url, attempt)
            resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Groq request exception (attempt %d/%d): %s", attempt, retries, exc)
            if attempt >= retries:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            wait_time = backoff
            if retry_after:
                try:
                    wait_time = float(retry_after) + 1.0
                except ValueError:
                    logger.debug("Invalid retry-after header value: %s", retry_after)
            
            logger.warning(f"Groq rate limited (429). Stalling for {wait_time:.1f}s.")
            
            # Key rotation
            if not api_key and keys_tried < max_key_rotations:
                current_key = _rotate_api_key()
                headers['Authorization'] = f'Bearer {current_key}'
                keys_tried += 1
                time.sleep(0.5)
                continue
            
            # Infinite retry for 429 (don't check attempt count)
            time.sleep(wait_time)
            if not retry_after:
                 backoff = min(backoff * 2, 60.0)
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

        # Chat Completions format: {'choices': [{'message': {'content': 'text'}}]}
        text = None
        if isinstance(data, dict):
            if 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                choice = data['choices'][0]
                if isinstance(choice, dict):
                    message = choice.get('message', {})
                    if isinstance(message, dict):
                        text = message.get('content', '')
                    elif isinstance(message, str):
                        text = message
                    else:
                        text = choice.get('text', '')
            # Fallback for other response formats
            elif 'output' in data:
                out = data['output']
                if isinstance(out, str):
                    text = out
                elif isinstance(out, list) and out:
                    first = out[0]
                    if isinstance(first, dict):
                        text = first.get('content') or first.get('text') or ''
                    elif isinstance(first, str):
                        text = first
        
        if text is None:
            logger.warning("Could not parse Groq response: %s", str(data)[:200])
            text = str(data)
        
        return text


def batch_generate_with_groq(prompts: List[str], model_id: str, batch_size: int = 32) -> List[str]:
    """Batch-generate outputs from Groq with a simple local file cache.

    Cache files are stored under `.cache/groq/` with filenames of the form
    `<sha256>.json` containing keys: 'prompt','model','output','ts'. If a cached
    entry exists it will be returned instead of calling the network.
    """
    out = []
    cache_dir = runtime_working_dir() / ".cache" / "groq"
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
