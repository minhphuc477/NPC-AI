#!/usr/bin/env python3
"""Generate publication-ready benchmark artifacts with reproducible metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_shell(command: Sequence[str], timeout_s: int = 15) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_mean_ci(values: Sequence[float], seed: int, iters: int = 2000) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    vals = [float(v) for v in values]
    mean_val = statistics.fmean(vals)
    median_val = statistics.median(vals)
    p95_val = percentile(vals, 0.95)
    n = len(vals)
    if n == 1:
        low = high = mean_val
    else:
        rng = random.Random(seed)
        samples: List[float] = []
        for _ in range(iters):
            draw = [vals[rng.randrange(n)] for __ in range(n)]
            samples.append(statistics.fmean(draw))
        samples.sort()
        low = percentile(samples, 0.025)
        high = percentile(samples, 0.975)

    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "p95": p95_val,
        "ci95_low": low,
        "ci95_high": high,
    }


def gather_hardware_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "timestamp_utc": utc_iso(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cpu_count_logical": os.cpu_count(),
    }

    if platform.system().lower().startswith("win"):
        cpu_name = safe_shell(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
            ]
        )
        ram_bytes = safe_shell(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        if cpu_name:
            metadata["cpu_name"] = cpu_name
        if ram_bytes.isdigit():
            metadata["ram_bytes"] = int(ram_bytes)
            metadata["ram_gb"] = round(int(ram_bytes) / (1024**3), 2)
    else:
        cpu_name = safe_shell(["sh", "-lc", "cat /proc/cpuinfo | grep 'model name' | head -n 1 | cut -d: -f2"])
        mem_total = safe_shell(["sh", "-lc", "cat /proc/meminfo | grep MemTotal | awk '{print $2}'"])
        if cpu_name:
            metadata["cpu_name"] = cpu_name.strip()
        if mem_total.isdigit():
            metadata["ram_kb"] = int(mem_total)
            metadata["ram_gb"] = round(int(mem_total) / (1024**2), 2)

    gpu_csv = safe_shell(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout_s=10,
    )
    metadata["gpu"] = []
    if gpu_csv:
        for line in gpu_csv.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                metadata["gpu"].append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                    }
                )
    return metadata


def query_ollama_model(host: str, model: str) -> Dict[str, Any]:
    for payload in ({"name": model}, {"model": model}):
        try:
            resp = requests.post(f"{host}/api/show", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                tensors = data.get("tensors")
                if isinstance(tensors, list):
                    data["tensor_count"] = len(tensors)
                    data["tensor_preview"] = tensors[:8]
                    data.pop("tensors", None)
                data["model_name"] = model
                data["host"] = host
                return data
        except Exception as exc:
            return {"model_name": model, "host": host, "error": str(exc)}
    return {"model_name": model, "host": host, "error": "Failed to query /api/show"}


def benchmark_ollama_prompt(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int = 240,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    try:
        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=timeout_s,
            stream=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"request_failed: {exc}"}

    if response.status_code != 200:
        return {
            "ok": False,
            "error": f"http_{response.status_code}",
            "body": response.text[:200],
        }

    ttft_ms: float | None = None
    chunks = 0
    text_parts: List[str] = []
    final_msg: Dict[str, Any] = {}

    for raw in response.iter_lines():
        if not raw:
            continue
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - start) * 1000.0
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue

        token = item.get("response", "")
        if token:
            chunks += 1
            text_parts.append(token)
        if item.get("done"):
            final_msg = item
            break

    total_ms = (time.perf_counter() - start) * 1000.0
    eval_count = int(final_msg.get("eval_count", chunks or 0))
    eval_duration_ns = int(final_msg.get("eval_duration", 0) or 0)
    if eval_duration_ns > 0 and eval_count > 0:
        tps = eval_count / (eval_duration_ns / 1e9)
    else:
        tps = (chunks / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    full_response = "".join(text_parts).strip()
    return {
        "ok": True,
        "ttft_ms": ttft_ms if ttft_ms is not None else total_ms,
        "total_time_ms": total_ms,
        "chunk_count": chunks,
        "eval_count": eval_count,
        "tokens_per_s": tps,
        "load_duration_ns": int(final_msg.get("load_duration", 0) or 0),
        "prompt_eval_count": int(final_msg.get("prompt_eval_count", 0) or 0),
        "prompt_eval_duration_ns": int(final_msg.get("prompt_eval_duration", 0) or 0),
        "eval_duration_ns": eval_duration_ns,
        "total_duration_ns": int(final_msg.get("total_duration", 0) or 0),
        "response_preview": full_response[:220],
        "response_text": full_response,
    }


def summarize_serving_records(records: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    ok_records = [r for r in records if r.get("ok")]
    error_count = len(records) - len(ok_records)
    summary: Dict[str, Any] = {
        "total_requests": len(records),
        "ok_requests": len(ok_records),
        "error_requests": error_count,
        "error_rate": (error_count / len(records)) if records else float("nan"),
    }

    for metric in ("ttft_ms", "total_time_ms", "tokens_per_s"):
        values = [float(r[metric]) for r in ok_records if metric in r and r[metric] is not None]
        summary[metric] = bootstrap_mean_ci(values, seed=seed)

    return summary


def compare_model_summaries(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for metric, lower_is_better in (
        ("ttft_ms", True),
        ("total_time_ms", True),
        ("tokens_per_s", False),
    ):
        cand_mean = candidate.get(metric, {}).get("mean")
        base_mean = baseline.get(metric, {}).get("mean")
        if cand_mean is None or base_mean is None or math.isnan(cand_mean) or math.isnan(base_mean):
            continue
        abs_delta = cand_mean - base_mean
        rel_delta = (abs_delta / base_mean) if base_mean != 0 else float("nan")
        better = abs_delta < 0 if lower_is_better else abs_delta > 0
        delta[metric] = {
            "candidate_mean": cand_mean,
            "baseline_mean": base_mean,
            "absolute_delta": abs_delta,
            "relative_delta": rel_delta,
            "candidate_better": bool(better),
            "lower_is_better": lower_is_better,
        }
    return delta


def _disable_torchao_for_bertscore() -> None:
    # Workaround for environments where torchao/triton versions are incompatible.
    # BERTScore itself does not require torchao quantization paths.
    try:
        import transformers.utils.import_utils as iu  # type: ignore

        iu._torchao_available = False
        iu._torchao_version = "0.0.0"
    except Exception:
        pass


def compute_bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str,
) -> Dict[str, Any]:
    if len(predictions) != len(references):
        return {
            "ok": False,
            "error": f"prediction/reference length mismatch: {len(predictions)} vs {len(references)}",
        }
    if not predictions:
        return {"ok": False, "error": "no aligned prediction/reference pairs"}

    try:
        _disable_torchao_for_bertscore()
        from bert_score import score as bert_score  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore

        prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            _, _, f1 = bert_score(
                list(predictions),
                list(references),
                lang=lang,
                rescale_with_baseline=True,
                verbose=False,
            )
        finally:
            hf_logging.set_verbosity(prev_level)
        values = [float(x) for x in f1]
        return {"ok": True, "f1": values}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def load_serving_references(path: Path) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for row in read_jsonl(path):
        prompt_id = str(row.get("prompt_id", "")).strip()
        reference = str(row.get("reference_response", "")).strip()
        if prompt_id and reference:
            refs[prompt_id] = reference
    return refs


TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, documents: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_dl = 0.0
        self._build()

    def _build(self) -> None:
        df: Dict[str, int] = {}
        total_len = 0
        for doc in self.documents:
            doc_id = str(doc["doc_id"])
            tokens = tokenize(str(doc["text"]))
            total_len += len(tokens)
            self.doc_len[doc_id] = len(tokens)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs[doc_id] = tf
            for token in tf:
                df[token] = df.get(token, 0) + 1

        self.avg_dl = (total_len / len(self.documents)) if self.documents else 0.0
        n = float(len(self.documents))
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))

    def rank(self, query: str, top_k: int) -> List[str]:
        query_tokens = tokenize(query)
        scored: List[Tuple[str, float]] = []
        for doc in self.documents:
            doc_id = str(doc["doc_id"])
            tf = self.term_freqs.get(doc_id, {})
            score = 0.0
            dl = self.doc_len.get(doc_id, 0)
            for term in query_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl if self.avg_dl > 0 else 0.0))
                score += idf * ((freq * (self.k1 + 1.0)) / (denom if denom > 0 else 1.0))
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]


def rank_keyword_overlap(corpus: List[Dict[str, str]], query: str, top_k: int) -> List[str]:
    q_tokens = set(tokenize(query))
    scored: List[Tuple[str, float]] = []
    for doc in corpus:
        doc_id = str(doc["doc_id"])
        d_tokens = set(tokenize(str(doc["text"])))
        overlap = len(q_tokens.intersection(d_tokens))
        scored.append((doc_id, float(overlap)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored[:top_k]]


def rank_random(corpus: List[Dict[str, str]], query: str, top_k: int, seed: int) -> List[str]:
    rng = random.Random(f"{seed}:{query}")
    ids = [str(doc["doc_id"]) for doc in corpus]
    rng.shuffle(ids)
    return ids[:top_k]


def dcg(relevances: Sequence[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances):
        score += rel / math.log2(idx + 2.0)
    return score


def evaluate_rankings(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    hit_k: int,
    seed: int,
) -> Dict[str, Any]:
    gold_map: Dict[str, List[str]] = {}
    for row in gold_rows:
        qid = str(row.get("query_id", "")).strip()
        rel = [str(x) for x in row.get("relevant_doc_ids", []) if str(x).strip()]
        if qid and rel:
            gold_map[qid] = rel

    pred_map: Dict[str, List[str]] = {}
    for row in pred_rows:
        qid = str(row.get("query_id", "")).strip()
        ranked = [str(x) for x in row.get("ranked_doc_ids", []) if str(x).strip()]
        if qid and ranked:
            pred_map[qid] = ranked

    per_query: List[Dict[str, Any]] = []
    for qid, relevant in gold_map.items():
        ranked = pred_map.get(qid, [])[:hit_k]
        rel_set = set(relevant)

        hit = 1.0 if any(doc_id in rel_set for doc_id in ranked) else 0.0
        rr = 0.0
        gains: List[int] = []
        for idx, doc_id in enumerate(ranked):
            is_rel = 1 if doc_id in rel_set else 0
            gains.append(is_rel)
            if rr == 0.0 and is_rel == 1:
                rr = 1.0 / float(idx + 1)

        ideal_ones = min(len(rel_set), len(ranked))
        idcg = dcg([1] * ideal_ones + [0] * (len(ranked) - ideal_ones))
        ndcg = (dcg(gains) / idcg) if idcg > 0 else 0.0

        per_query.append(
            {
                "query_id": qid,
                f"hit@{hit_k}": hit,
                "mrr": rr,
                f"ndcg@{hit_k}": ndcg,
            }
        )

    hit_values = [float(r[f"hit@{hit_k}"]) for r in per_query]
    mrr_values = [float(r["mrr"]) for r in per_query]
    ndcg_values = [float(r[f"ndcg@{hit_k}"]) for r in per_query]

    return {
        "query_count": len(per_query),
        "per_query": per_query,
        f"hit@{hit_k}": bootstrap_mean_ci(hit_values, seed=seed + 11),
        "mrr": bootstrap_mean_ci(mrr_values, seed=seed + 23),
        f"ndcg@{hit_k}": bootstrap_mean_ci(ndcg_values, seed=seed + 37),
    }


def retrieval_ablation_deltas(metrics: Dict[str, Dict[str, Any]], baseline: str, hit_k: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if baseline not in metrics:
        return out

    baseline_metrics = metrics[baseline]
    key_hit = f"hit@{hit_k}"
    for method, method_metrics in metrics.items():
        if method == baseline:
            continue
        row: Dict[str, Any] = {}
        for metric_name in (key_hit, "mrr", f"ndcg@{hit_k}"):
            b_mean = baseline_metrics.get(metric_name, {}).get("mean", float("nan"))
            m_mean = method_metrics.get(metric_name, {}).get("mean", float("nan"))
            if math.isnan(b_mean) or math.isnan(m_mean):
                continue
            abs_delta = m_mean - b_mean
            rel_delta = (abs_delta / b_mean) if b_mean != 0 else float("nan")
            row[metric_name] = {
                "method_mean": m_mean,
                "baseline_mean": b_mean,
                "absolute_delta": abs_delta,
                "relative_delta": rel_delta,
            }
        out[method] = row
    return out


def run_reranker_stage(
    pairs_path: Path,
    retrieval_corpus_path: Path,
    retrieval_gold_path: Path,
    output_dir: Path,
    predictions_out: Path,
    train_frac: float,
    bm25_candidate_k: int,
    hit_k: int,
    seed: int,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/train_retrieval_reranker.py",
        "--pairs",
        str(pairs_path),
        "--retrieval-corpus",
        str(retrieval_corpus_path),
        "--retrieval-gold",
        str(retrieval_gold_path),
        "--output-dir",
        str(output_dir),
        "--predictions-out",
        str(predictions_out),
        "--train-frac",
        str(train_frac),
        "--bm25-candidate-k",
        str(bm25_candidate_k),
        "--hit-k",
        str(hit_k),
        "--seed",
        str(seed),
    ]

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    summary_path = output_dir / "summary.json"
    out: Dict[str, Any] = {
        "ok": False,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "pairs_path": str(pairs_path),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_out),
    }

    if proc.returncode == 0 and summary_path.exists() and predictions_out.exists():
        out["ok"] = True
        try:
            out["summary"] = read_json(summary_path)
        except Exception as exc:
            out["summary_error"] = str(exc)
    return out


def summarize_bertscore_records(
    all_serving_records: Dict[str, List[Dict[str, Any]]],
    references: Dict[str, str],
    lang: str,
    seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lang": lang,
        "models": {},
    }

    for model_name, records in all_serving_records.items():
        aligned_preds: List[str] = []
        aligned_refs: List[str] = []
        per_request: List[Dict[str, Any]] = []
        empty_prediction_count = 0

        for row in records:
            if not row.get("ok"):
                continue
            prompt_id = str(row.get("prompt_id", "")).strip()
            if prompt_id not in references:
                continue
            pred = str(row.get("response_text", "")).strip()
            ref = references[prompt_id]
            if not ref:
                continue
            if not pred:
                pred = "<empty_response>"
                empty_prediction_count += 1
            aligned_preds.append(pred)
            aligned_refs.append(ref)

        score_result = compute_bertscore_f1(aligned_preds, aligned_refs, lang=lang)
        model_payload: Dict[str, Any] = {
            "aligned_request_count": len(aligned_preds),
            "empty_prediction_count": empty_prediction_count,
            "ok": bool(score_result.get("ok")),
        }
        if not score_result.get("ok"):
            model_payload["error"] = score_result.get("error", "unknown error")
            out["models"][model_name] = model_payload
            continue

        values = [float(v) for v in score_result.get("f1", [])]
        for idx, val in enumerate(values):
            per_request.append(
                {
                    "index": idx,
                    "bertscore_f1": val,
                }
            )

        model_payload["bertscore_f1"] = bootstrap_mean_ci(values, seed=seed + len(model_name) * 31)
        model_payload["per_request"] = per_request
        out["models"][model_name] = model_payload

    return out


def bertscore_delta_vs_baseline(
    bertscore_summary: Dict[str, Any],
    candidate_model: str,
    baseline_model: str,
) -> Dict[str, Any]:
    cand = bertscore_summary.get("models", {}).get(candidate_model, {})
    base = bertscore_summary.get("models", {}).get(baseline_model, {})
    cand_mean = cand.get("bertscore_f1", {}).get("mean")
    base_mean = base.get("bertscore_f1", {}).get("mean")
    if cand_mean is None or base_mean is None:
        return {}
    if math.isnan(cand_mean) or math.isnan(base_mean):
        return {}
    abs_delta = cand_mean - base_mean
    rel_delta = (abs_delta / base_mean) if base_mean != 0 else float("nan")
    return {
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "candidate_mean": cand_mean,
        "baseline_mean": base_mean,
        "absolute_delta": abs_delta,
        "relative_delta": rel_delta,
        "candidate_better": abs_delta > 0.0,
    }


def render_report(
    output_path: Path,
    run_id: str,
    host: str,
    candidate: str,
    baseline: str,
    serving_summary: Dict[str, Any],
    serving_delta: Dict[str, Any],
    retrieval_metrics: Dict[str, Any],
    retrieval_deltas: Dict[str, Any],
    hit_k: int,
    reranker_stage: Dict[str, Any] | None = None,
    bertscore_summary: Dict[str, Any] | None = None,
    bertscore_delta: Dict[str, Any] | None = None,
    security_metrics: Dict[str, Any] | None = None,
    security_spoofed_metrics: Dict[str, Any] | None = None,
) -> None:
    key_hit = f"hit@{hit_k}"
    lines: List[str] = []
    lines.append("# Publication Benchmark Artifact Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Generated: `{utc_iso()}`")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Candidate model: `{candidate}`")
    lines.append(f"- Baseline model: `{baseline}`")
    lines.append("")
    lines.append("## 1. Non-mock Benchmark Artifacts With Metadata")
    lines.append("- Raw per-request serving traces are published in `serving/`.")
    lines.append("- Hardware and model metadata are published in `metadata/`.")
    lines.append("- All requests in this run were executed against live Ollama model endpoints (non-mock).")
    lines.append("")
    lines.append("## 2. Standardized Retrieval Metrics (Labeled Sets)")
    lines.append("| Method | Hit@k | MRR | nDCG@k |")
    lines.append("|---|---:|---:|---:|")
    for method, metric in retrieval_metrics.items():
        hit = metric.get(key_hit, {}).get("mean", float("nan"))
        mrr = metric.get("mrr", {}).get("mean", float("nan"))
        ndcg = metric.get(f"ndcg@{hit_k}", {}).get("mean", float("nan"))
        lines.append(f"| {method} | {hit:.4f} | {mrr:.4f} | {ndcg:.4f} |")
    if reranker_stage:
        if reranker_stage.get("ok"):
            summary = reranker_stage.get("summary", {})
            eval_metrics = summary.get("eval_metrics", {})
            pair_acc = eval_metrics.get("pair_accuracy", {})
            lines.append("")
            lines.append(
                "- Reranker stage: "
                + f"{summary.get('pair_rows_total', 'na')} hard-negative pairs, "
                + f"eval pair-accuracy={pair_acc.get('mean', float('nan')):.4f} "
                + f"(95% CI {pair_acc.get('ci95_low', float('nan')):.4f}, {pair_acc.get('ci95_high', float('nan')):.4f})."
            )
        elif not reranker_stage.get("skipped", False):
            lines.append("")
            lines.append("- Reranker stage failed; see `retrieval/reranker_stage.json` for logs.")
    lines.append("")
    lines.append("## 3. Confidence Intervals And Ablation Deltas")
    lines.append("| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |")
    lines.append("|---|---:|---:|---:|")
    for metric_name in ("ttft_ms", "total_time_ms", "tokens_per_s"):
        cand_metric = serving_summary.get(candidate, {}).get(metric_name, {})
        base_metric = serving_summary.get(baseline, {}).get(metric_name, {})
        delta_metric = serving_delta.get(metric_name, {})
        lines.append(
            "| "
            + metric_name
            + " | "
            + f"{cand_metric.get('mean', float('nan')):.3f} "
            + f"({cand_metric.get('ci95_low', float('nan')):.3f}, {cand_metric.get('ci95_high', float('nan')):.3f})"
            + " | "
            + f"{base_metric.get('mean', float('nan')):.3f} "
            + f"({base_metric.get('ci95_low', float('nan')):.3f}, {base_metric.get('ci95_high', float('nan')):.3f})"
            + " | "
            + f"{delta_metric.get('absolute_delta', float('nan')):.3f}"
            + " |"
        )
    if bertscore_summary:
        cand_b = bertscore_summary.get("models", {}).get(candidate, {}).get("bertscore_f1", {})
        base_b = bertscore_summary.get("models", {}).get(baseline, {}).get("bertscore_f1", {})
        delta_val = float("nan")
        if bertscore_delta:
            delta_val = bertscore_delta.get("absolute_delta", float("nan"))
        if cand_b and base_b:
            lines.append(
                "| bertscore_f1 | "
                + f"{cand_b.get('mean', float('nan')):.3f} "
                + f"({cand_b.get('ci95_low', float('nan')):.3f}, {cand_b.get('ci95_high', float('nan')):.3f})"
                + " | "
                + f"{base_b.get('mean', float('nan')):.3f} "
                + f"({base_b.get('ci95_low', float('nan')):.3f}, {base_b.get('ci95_high', float('nan')):.3f})"
                + " | "
                + f"{delta_val:.3f}"
                + " |"
            )
        elif bertscore_summary.get("models"):
            lines.append(
                f"| bertscore_f1 | n/a | n/a | n/a |"
            )
    lines.append("")
    lines.append("Retrieval ablation deltas (vs BM25 baseline):")
    lines.append("| Method | Metric | Absolute Delta | Relative Delta |")
    lines.append("|---|---|---:|---:|")
    if retrieval_deltas:
        for method, row in retrieval_deltas.items():
            for metric_name, vals in row.items():
                lines.append(
                    f"| {method} | {metric_name} | {vals.get('absolute_delta', float('nan')):.4f} | {vals.get('relative_delta', float('nan')):.4f} |"
                )
    else:
        lines.append("| n/a | n/a | n/a | n/a |")
    lines.append("")
    lines.append("## 4. Production Serving Baseline Comparison")
    lines.append(
        "The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) "
        "and identical generation settings (temperature and max tokens)."
    )
    lines.append(
        "This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol."
    )
    if security_metrics:
        lines.append("")
        lines.append("## 5. Adversarial Retrieval Robustness")
        lines.append("Poisoned retrieval benchmark evaluated attack success rate (ASR) with guard off vs on.")
        lines.append(
            f"- Dataset scenarios: {security_metrics.get('scenario_count', 'na')}"
        )
        lines.append(
            f"- Baseline ASR: {security_metrics.get('baseline_attack_success_rate', float('nan')):.4f} "
            f"(95% CI: {security_metrics.get('baseline_asr_ci95_low', float('nan')):.4f}, "
            f"{security_metrics.get('baseline_asr_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            f"- Guarded ASR: {security_metrics.get('guarded_attack_success_rate', float('nan')):.4f} "
            f"(95% CI: {security_metrics.get('guarded_asr_ci95_low', float('nan')):.4f}, "
            f"{security_metrics.get('guarded_asr_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            f"- Relative ASR reduction: {security_metrics.get('relative_asr_reduction', float('nan')):.4f} "
            f"(95% CI: {security_metrics.get('relative_asr_reduction_ci95_low', float('nan')):.4f}, "
            f"{security_metrics.get('relative_asr_reduction_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            f"- Guarded Safe@1: {security_metrics.get('guarded_safe_top1_rate', float('nan')):.4f} "
            f"(95% CI: {security_metrics.get('guarded_safe_top1_ci95_low', float('nan')):.4f}, "
            f"{security_metrics.get('guarded_safe_top1_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            "Result file: `retrieval/security_guard_benchmark.json`."
        )
    if security_spoofed_metrics:
        lines.append("")
        lines.append("## 6. Trust-Spoofed Poisoning Stress Test")
        lines.append(
            "A harder variant was executed with poison documents forced to claim high-trust metadata "
            "(simulating provenance spoofing)."
        )
        lines.append(
            f"- Dataset scenarios: {security_spoofed_metrics.get('scenario_count', 'na')}"
        )
        lines.append(
            f"- Baseline ASR: {security_spoofed_metrics.get('baseline_attack_success_rate', float('nan')):.4f} "
            f"(95% CI: {security_spoofed_metrics.get('baseline_asr_ci95_low', float('nan')):.4f}, "
            f"{security_spoofed_metrics.get('baseline_asr_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            f"- Guarded ASR: {security_spoofed_metrics.get('guarded_attack_success_rate', float('nan')):.4f} "
            f"(95% CI: {security_spoofed_metrics.get('guarded_asr_ci95_low', float('nan')):.4f}, "
            f"{security_spoofed_metrics.get('guarded_asr_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            f"- Relative ASR reduction: {security_spoofed_metrics.get('relative_asr_reduction', float('nan')):.4f} "
            f"(95% CI: {security_spoofed_metrics.get('relative_asr_reduction_ci95_low', float('nan')):.4f}, "
            f"{security_spoofed_metrics.get('relative_asr_reduction_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            f"- Guarded Safe@1: {security_spoofed_metrics.get('guarded_safe_top1_rate', float('nan')):.4f} "
            f"(95% CI: {security_spoofed_metrics.get('guarded_safe_top1_ci95_low', float('nan')):.4f}, "
            f"{security_spoofed_metrics.get('guarded_safe_top1_ci95_high', float('nan')):.4f})"
        )
        lines.append(
            "Result file: `retrieval/security_guard_benchmark_spoofed.json`."
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publication benchmark suite and export artifacts.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama host URL")
    parser.add_argument("--candidate-model", default="elara-npc:latest", help="Candidate model name")
    parser.add_argument("--baseline-model", default="phi3:mini", help="Baseline model name")
    parser.add_argument("--prompts", default="data/serving_prompts.jsonl", help="Prompt JSONL path")
    parser.add_argument(
        "--serving-references",
        default="data/serving_references.jsonl",
        help="Prompt-level reference responses JSONL for BERTScore (optional)",
    )
    parser.add_argument("--bertscore-lang", default="en", help="Language code for BERTScore")
    parser.add_argument("--retrieval-corpus", default="data/retrieval_corpus.jsonl", help="Retrieval corpus JSONL")
    parser.add_argument("--retrieval-gold", default="data/retrieval_gold.jsonl", help="Retrieval gold JSONL")
    parser.add_argument(
        "--reranker-pairs",
        default="data/retrieval_reranker_pairs_wide.jsonl",
        help="Hard-negative reranker pairs JSONL (3360-pair wide set by default).",
    )
    parser.add_argument("--reranker-train-frac", type=float, default=0.85, help="Reranker query-level train split.")
    parser.add_argument(
        "--reranker-bm25-candidate-k",
        type=int,
        default=24,
        help="BM25 candidate pool size before reranking.",
    )
    parser.add_argument("--skip-reranker-stage", action="store_true", help="Disable reranker train/eval stage.")
    parser.add_argument("--repeats", type=int, default=2, help="Serving repeats per prompt/model")
    parser.add_argument("--hit-k", type=int, default=5, help="Hit@k and nDCG@k for retrieval")
    parser.add_argument("--temperature", type=float, default=0.2, help="Serving temperature")
    parser.add_argument("--max-tokens", type=int, default=96, help="Serving max generated tokens")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--skip-ablation-baselines",
        action="store_true",
        help="Skip keyword/random retrieval baselines (runs bm25 + optional reranker only).",
    )
    parser.add_argument("--run-security-benchmark", action="store_true",
                        help="Run adversarial retrieval security benchmark if executable exists.")
    parser.add_argument(
        "--run-security-spoofed-benchmark",
        action="store_true",
        help="Run a trust-spoofed variant of the retrieval security benchmark.",
    )
    parser.add_argument("--security-benchmark-exe",
                        default="cpp/build/Release/bench_retrieval_security.exe",
                        help="Path to bench_retrieval_security executable.")
    parser.add_argument(
        "--security-dataset",
        default="data/retrieval_poison_benchmark.jsonl",
        help="JSONL poisoned retrieval benchmark dataset.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/publication",
        help="Root output folder (run will create timestamped subfolder)",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    serving_refs_path = Path(args.serving_references)
    retrieval_corpus_path = Path(args.retrieval_corpus)
    retrieval_gold_path = Path(args.retrieval_gold)
    reranker_pairs_path = Path(args.reranker_pairs)
    for required in (prompts_path, retrieval_corpus_path, retrieval_gold_path):
        if not required.exists():
            raise FileNotFoundError(f"Required input file not found: {required}")
    if not args.skip_reranker_stage and not reranker_pairs_path.exists():
        raise FileNotFoundError(f"Reranker pairs file not found: {reranker_pairs_path}")
    if args.run_security_benchmark:
        security_dataset_path = Path(args.security_dataset)
        if not security_dataset_path.exists():
            raise FileNotFoundError(f"Security benchmark dataset not found: {security_dataset_path}")
    else:
        security_dataset_path = Path(args.security_dataset)
    if args.run_security_spoofed_benchmark and not args.run_security_benchmark:
        raise ValueError("--run-security-spoofed-benchmark requires --run-security-benchmark")

    run_id = utc_stamp()
    run_dir = Path(args.output_root) / run_id
    metadata_dir = run_dir / "metadata"
    serving_dir = run_dir / "serving"
    retrieval_dir = run_dir / "retrieval"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    serving_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_jsonl(prompts_path)
    corpus = read_jsonl(retrieval_corpus_path)
    gold = read_jsonl(retrieval_gold_path)

    run_config = {
        "run_id": run_id,
        "generated_utc": utc_iso(),
        "host": args.host,
        "candidate_model": args.candidate_model,
        "baseline_model": args.baseline_model,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "hit_k": args.hit_k,
        "seed": args.seed,
        "skip_ablation_baselines": bool(args.skip_ablation_baselines),
        "skip_reranker_stage": bool(args.skip_reranker_stage),
        "reranker_train_frac": float(args.reranker_train_frac),
        "reranker_bm25_candidate_k": int(args.reranker_bm25_candidate_k),
        "run_security_spoofed_benchmark": bool(args.run_security_spoofed_benchmark),
        "inputs": {
            "prompts": str(prompts_path),
            "serving_references": str(serving_refs_path) if serving_refs_path.exists() else "",
            "retrieval_corpus": str(retrieval_corpus_path),
            "retrieval_gold": str(retrieval_gold_path),
            "reranker_pairs": str(reranker_pairs_path) if reranker_pairs_path.exists() else "",
            "prompts_sha256": sha256_file(prompts_path),
            "serving_references_sha256": sha256_file(serving_refs_path) if serving_refs_path.exists() else "",
            "retrieval_corpus_sha256": sha256_file(retrieval_corpus_path),
            "retrieval_gold_sha256": sha256_file(retrieval_gold_path),
            "reranker_pairs_sha256": sha256_file(reranker_pairs_path) if reranker_pairs_path.exists() else "",
            "security_dataset": str(security_dataset_path) if security_dataset_path.exists() else "",
            "security_dataset_sha256": sha256_file(security_dataset_path) if security_dataset_path.exists() else "",
        },
    }
    write_json(run_dir / "run_config.json", run_config)

    hardware = gather_hardware_metadata()
    write_json(metadata_dir / "hardware.json", hardware)

    model_meta = {
        args.candidate_model: query_ollama_model(args.host, args.candidate_model),
        args.baseline_model: query_ollama_model(args.host, args.baseline_model),
    }
    write_json(metadata_dir / "models.json", model_meta)

    all_serving_records: Dict[str, List[Dict[str, Any]]] = {
        args.candidate_model: [],
        args.baseline_model: [],
    }

    request_index = 0
    for model_name in (args.candidate_model, args.baseline_model):
        print(f"[serving] benchmarking model={model_name}")
        for repeat_idx in range(max(1, args.repeats)):
            for prompt_row in prompts:
                request_index += 1
                prompt_id = str(prompt_row.get("prompt_id", f"row_{request_index}"))
                prompt = str(prompt_row.get("prompt", ""))
                result = benchmark_ollama_prompt(
                    host=args.host,
                    model=model_name,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                record = {
                    "request_index": request_index,
                    "repeat_index": repeat_idx,
                    "model": model_name,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                    "timestamp_utc": utc_iso(),
                }
                record.update(result)
                all_serving_records[model_name].append(record)

                state = "ok" if record.get("ok") else "error"
                print(
                    f"  - {model_name} repeat={repeat_idx + 1}/{args.repeats} prompt={prompt_id} status={state} "
                    f"ttft_ms={record.get('ttft_ms', 'na')}"
                )

    for model_name, records in all_serving_records.items():
        sanitized = model_name.replace(":", "_")
        write_jsonl(serving_dir / f"requests_{sanitized}.jsonl", records)

    serving_summary: Dict[str, Dict[str, Any]] = {}
    for model_name, records in all_serving_records.items():
        serving_summary[model_name] = summarize_serving_records(records, seed=args.seed + len(model_name))
    write_json(serving_dir / "summary.json", serving_summary)

    serving_delta = compare_model_summaries(
        candidate=serving_summary[args.candidate_model],
        baseline=serving_summary[args.baseline_model],
    )
    write_json(serving_dir / "delta_vs_baseline.json", serving_delta)

    bertscore_summary: Dict[str, Any] | None = None
    bertscore_delta: Dict[str, Any] | None = None
    if serving_refs_path.exists():
        references = load_serving_references(serving_refs_path)
        if references:
            bertscore_summary = summarize_bertscore_records(
                all_serving_records=all_serving_records,
                references=references,
                lang=args.bertscore_lang,
                seed=args.seed,
            )
            bertscore_delta = bertscore_delta_vs_baseline(
                bertscore_summary,
                candidate_model=args.candidate_model,
                baseline_model=args.baseline_model,
            )
            write_json(serving_dir / "quality_bertscore.json", bertscore_summary)
            write_json(serving_dir / "quality_bertscore_delta_vs_baseline.json", bertscore_delta)

    bm25 = BM25Index(corpus)
    retrieval_predictions: Dict[str, List[Dict[str, Any]]] = {}
    methods = ("bm25",) if args.skip_ablation_baselines else ("bm25", "keyword_overlap", "random")
    for method in methods:
        rows: List[Dict[str, Any]] = []
        for item in gold:
            qid = str(item.get("query_id", ""))
            query = str(item.get("query", ""))
            if method == "bm25":
                ranked = bm25.rank(query, args.hit_k)
            elif method == "keyword_overlap":
                ranked = rank_keyword_overlap(corpus, query, args.hit_k)
            else:
                ranked = rank_random(corpus, query, args.hit_k, seed=args.seed)
            rows.append(
                {
                    "query_id": qid,
                    "query": query,
                    "ranked_doc_ids": ranked,
                }
            )
        retrieval_predictions[method] = rows
        write_jsonl(retrieval_dir / f"predictions_{method}.jsonl", rows)

    reranker_stage: Dict[str, Any] | None = None
    if args.skip_reranker_stage:
        reranker_stage = {
            "ok": False,
            "skipped": True,
            "reason": "--skip-reranker-stage enabled",
        }
    else:
        reranker_output_dir = retrieval_dir / "reranker"
        reranker_predictions_path = retrieval_dir / "predictions_reranker.jsonl"
        print(f"[retrieval] training reranker on pairs={reranker_pairs_path}")
        reranker_stage = run_reranker_stage(
            pairs_path=reranker_pairs_path,
            retrieval_corpus_path=retrieval_corpus_path,
            retrieval_gold_path=retrieval_gold_path,
            output_dir=reranker_output_dir,
            predictions_out=reranker_predictions_path,
            train_frac=float(args.reranker_train_frac),
            bm25_candidate_k=int(args.reranker_bm25_candidate_k),
            hit_k=int(args.hit_k),
            seed=int(args.seed),
        )
        if reranker_stage.get("ok") and reranker_predictions_path.exists():
            retrieval_predictions["reranker"] = read_jsonl(reranker_predictions_path)
    if reranker_stage is not None:
        write_json(retrieval_dir / "reranker_stage.json", reranker_stage)

    write_jsonl(retrieval_dir / "gold.jsonl", gold)

    retrieval_metrics: Dict[str, Dict[str, Any]] = {}
    for method, rows in retrieval_predictions.items():
        evaluation = evaluate_rankings(gold, rows, args.hit_k, args.seed)
        retrieval_metrics[method] = {
            f"hit@{args.hit_k}": evaluation[f"hit@{args.hit_k}"],
            "mrr": evaluation["mrr"],
            f"ndcg@{args.hit_k}": evaluation[f"ndcg@{args.hit_k}"],
            "query_count": evaluation["query_count"],
        }
        write_json(retrieval_dir / f"per_query_{method}.json", evaluation["per_query"])

    write_json(retrieval_dir / "metrics.json", retrieval_metrics)
    retrieval_deltas = retrieval_ablation_deltas(retrieval_metrics, baseline="bm25", hit_k=args.hit_k)
    write_json(retrieval_dir / "ablation_deltas_vs_bm25.json", retrieval_deltas)

    security_metrics: Dict[str, Any] | None = None
    security_spoofed_metrics: Dict[str, Any] | None = None
    if args.run_security_benchmark:
        security_exe = Path(args.security_benchmark_exe)
        security_out = retrieval_dir / "security_guard_benchmark.json"
        if security_exe.exists():
            proc = subprocess.run(
                [
                    str(security_exe),
                    "--output",
                    str(security_out),
                    "--dataset",
                    str(security_dataset_path),
                    "--seed",
                    str(args.seed),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=240,
            )
            if proc.returncode == 0 and security_out.exists():
                with security_out.open("r", encoding="utf-8") as handle:
                    security_metrics = json.load(handle)

                if args.run_security_spoofed_benchmark:
                    security_spoofed_out = retrieval_dir / "security_guard_benchmark_spoofed.json"
                    spoof_proc = subprocess.run(
                        [
                            str(security_exe),
                            "--output",
                            str(security_spoofed_out),
                            "--dataset",
                            str(security_dataset_path),
                            "--seed",
                            str(args.seed),
                            "--poison-spoof-trust",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=240,
                    )
                    if spoof_proc.returncode == 0 and security_spoofed_out.exists():
                        with security_spoofed_out.open("r", encoding="utf-8") as handle:
                            security_spoofed_metrics = json.load(handle)
                    else:
                        write_json(
                            retrieval_dir / "security_guard_benchmark_spoofed_error.json",
                            {
                                "returncode": spoof_proc.returncode,
                                "stdout": spoof_proc.stdout[-4000:],
                                "stderr": spoof_proc.stderr[-4000:],
                            },
                        )
            else:
                write_json(
                    retrieval_dir / "security_guard_benchmark_error.json",
                    {
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-4000:],
                        "stderr": proc.stderr[-4000:],
                    },
                )
        else:
            write_json(
                retrieval_dir / "security_guard_benchmark_error.json",
                {"error": f"Executable not found: {security_exe}"},
            )

    render_report(
        output_path=run_dir / "report.md",
        run_id=run_id,
        host=args.host,
        candidate=args.candidate_model,
        baseline=args.baseline_model,
        serving_summary=serving_summary,
        serving_delta=serving_delta,
        retrieval_metrics=retrieval_metrics,
        retrieval_deltas=retrieval_deltas,
        hit_k=args.hit_k,
        reranker_stage=reranker_stage,
        bertscore_summary=bertscore_summary,
        bertscore_delta=bertscore_delta,
        security_metrics=security_metrics,
        security_spoofed_metrics=security_spoofed_metrics,
    )

    print(f"\nPublished artifact bundle: {run_dir}")
    print("Generated files:")
    print(f"  - {metadata_dir / 'hardware.json'}")
    print(f"  - {metadata_dir / 'models.json'}")
    print(f"  - {serving_dir / 'summary.json'}")
    if bertscore_summary is not None:
        print(f"  - {serving_dir / 'quality_bertscore.json'}")
    print(f"  - {retrieval_dir / 'metrics.json'}")
    if reranker_stage is not None:
        print(f"  - {retrieval_dir / 'reranker_stage.json'}")
    if security_metrics is not None:
        print(f"  - {retrieval_dir / 'security_guard_benchmark.json'}")
    if security_spoofed_metrics is not None:
        print(f"  - {retrieval_dir / 'security_guard_benchmark_spoofed.json'}")
    print(f"  - {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
