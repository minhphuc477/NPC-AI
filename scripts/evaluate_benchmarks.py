#!/usr/bin/env python3
"""Evaluate latency/quality/retrieval metrics without fabricated defaults."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


def run_performance_benchmark(
    prompt: str,
    target_model: str = "phi3:mini",
    host: str = "http://localhost:11434",
) -> Dict[str, object]:
    """Measure TTFT and generation throughput from Ollama streaming API."""
    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7},
    }
    headers = {"Content-Type": "application/json"}

    start = time.time()
    try:
        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            headers=headers,
            stream=True,
            timeout=180,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Connection failed: {exc}"}

    if response.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP {response.status_code}",
            "body": response.text[:300],
        }

    ttft = None
    token_chunks = 0
    full_response = []

    for raw in response.iter_lines():
        if not raw:
            continue
        if ttft is None:
            ttft = time.time() - start

        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue

        chunk = item.get("response", "")
        if chunk:
            token_chunks += 1
            full_response.append(chunk)

        if item.get("done"):
            break

    total = time.time() - start
    tps = token_chunks / total if total > 0 else 0.0

    return {
        "ok": True,
        "ttft_ms": (ttft or 0.0) * 1000.0,
        "tps_chunk": tps,
        "total_time_s": total,
        "chunk_count": token_chunks,
        "response": "".join(full_response).strip(),
    }


def run_llm_as_a_judge(
    npc_response: str,
    context: str,
    persona: str,
    judge_model: str = "llama3",
    host: str = "http://localhost:11434",
) -> Dict[str, object]:
    """Use an LLM judge and return parsed JSON scores."""
    judge_prompt = f"""
You are evaluating an NPC response.
Score each criterion from 1 to 5 and return JSON only:
{{
  "ContextAwareness": int,
  "PersonaConsistency": int,
  "Truthfulness": int,
  "NLI_Logic": int,
  "Reasoning": "short explanation"
}}

Context: {context}
Persona: {persona}
NPC Response: {npc_response}
"""
    payload = {
        "model": judge_model,
        "prompt": judge_prompt,
        "stream": False,
        "format": "json",
    }
    try:
        response = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        if response.status_code != 200:
            return {"ok": False, "error": f"Judge HTTP {response.status_code}"}
        data = response.json()
        parsed = json.loads(data.get("response", "{}"))
        parsed["ok"] = True
        return parsed
    except Exception as exc:
        return {"ok": False, "error": f"Judge failed: {exc}"}


def _load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_relevant_ids(record: Dict[str, object]) -> List[str]:
    if "relevant_doc_ids" in record and isinstance(record["relevant_doc_ids"], list):
        return [str(x) for x in record["relevant_doc_ids"]]
    if "relevant_doc_id" in record:
        return [str(record["relevant_doc_id"])]
    return []


def _dcg(relevances: List[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        denom = math.log2(i + 2)
        score += rel / denom
    return score


def evaluate_rag_metrics(
    gold_path: Optional[str],
    predictions_path: Optional[str],
    hit_k: int = 5,
) -> Dict[str, object]:
    """Compute Hit@k, MRR and nDCG@k from prediction/gold JSONL files."""
    if not gold_path or not predictions_path:
        return {"ok": False, "error": "RAG metrics skipped (missing --rag-gold or --rag-predictions)."}

    gold_file = Path(gold_path)
    pred_file = Path(predictions_path)
    if not gold_file.exists() or not pred_file.exists():
        return {
            "ok": False,
            "error": f"RAG metrics skipped (missing files: gold={gold_file.exists()} pred={pred_file.exists()}).",
        }

    gold = {}
    for row in _load_jsonl(gold_file):
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        relevant = _normalize_relevant_ids(row)
        if relevant:
            gold[qid] = set(relevant)

    if not gold:
        return {"ok": False, "error": "Gold file has no valid query_id/relevant_doc_ids entries."}

    hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0

    for row in _load_jsonl(pred_file):
        qid = str(row.get("query_id", "")).strip()
        ranked = [str(x) for x in row.get("ranked_doc_ids", []) if str(x).strip()]
        if not qid or qid not in gold or not ranked:
            continue

        evaluated += 1
        relevant = gold[qid]
        top_k = ranked[: max(1, hit_k)]

        if any(doc_id in relevant for doc_id in top_k):
            hits += 1

        rr = 0.0
        gains = []
        for idx, doc_id in enumerate(top_k):
            rel = 1 if doc_id in relevant else 0
            gains.append(rel)
            if rr == 0.0 and rel == 1:
                rr = 1.0 / float(idx + 1)
        mrr_sum += rr

        ideal_count = min(len(relevant), len(top_k))
        ideal_gains = [1] * ideal_count + [0] * (len(top_k) - ideal_count)
        idcg = _dcg(ideal_gains, len(top_k))
        ndcg = (_dcg(gains, len(top_k)) / idcg) if idcg > 0 else 0.0
        ndcg_sum += ndcg

    if evaluated == 0:
        return {"ok": False, "error": "No overlapping query_id entries between gold and predictions."}

    return {
        "ok": True,
        "query_count": evaluated,
        f"hit@{hit_k}": hits / evaluated,
        "mrr": mrr_sum / evaluated,
        f"ndcg@{hit_k}": ndcg_sum / evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NPC benchmark signals without hardcoded scores.")
    parser.add_argument("--model", default="phi3:mini")
    parser.add_argument("--judge-model", default="llama3")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--rag-gold", default="")
    parser.add_argument("--rag-predictions", default="")
    parser.add_argument("--hit-k", type=int, default=5)
    args = parser.parse_args()

    context = "The player has drawn a glowing sword inside the King's throne room."
    persona = "King Alaric, suspicious medieval ruler."
    player_input = "I am the new champion. Yield the throne."
    prompt = f"System: You are {persona}\nContext: {context}\nPlayer: {player_input}\nNPC:"

    print("=== Performance Benchmark ===")
    perf = run_performance_benchmark(prompt, target_model=args.model, host=args.host)
    if not perf.get("ok"):
        print(f"Performance benchmark failed: {perf.get('error')}")
    else:
        print(f"TTFT: {perf['ttft_ms']:.2f} ms")
        print(f"Throughput (chunk/s): {perf['tps_chunk']:.2f}")
        print(f"Total time: {perf['total_time_s']:.2f} s")
        print(f"Chunk count: {perf['chunk_count']}")
        print(f"Response sample: {str(perf.get('response', ''))[:180]}")

    if args.run_judge and perf.get("ok"):
        print("\n=== LLM Judge ===")
        judge = run_llm_as_a_judge(
            str(perf.get("response", "")),
            context=context,
            persona=persona,
            judge_model=args.judge_model,
            host=args.host,
        )
        if not judge.get("ok"):
            print(f"Judge skipped/failed: {judge.get('error')}")
        else:
            print(json.dumps(judge, ensure_ascii=False, indent=2))

    print("\n=== RAG Metrics ===")
    rag = evaluate_rag_metrics(
        gold_path=args.rag_gold,
        predictions_path=args.rag_predictions,
        hit_k=max(1, args.hit_k),
    )
    if not rag.get("ok"):
        print(rag.get("error"))
    else:
        print(json.dumps(rag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
