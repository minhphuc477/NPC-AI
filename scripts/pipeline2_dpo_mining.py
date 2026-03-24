#!/usr/bin/env python3
"""Mine conflict-state DPO pairs from existing runs and synthesize more pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

MODEL_DEFAULT = "gpt-4o-mini"
SEED_DEFAULT = 42

HIGH_RETRY_STATES = ["negotiating", "detained", "assisting"]
STATE_WEIGHTS = {
    "negotiating": 4,
    "detained": 4,
    "assisting": 3,
    "observing": 1,
    "patrolling": 1,
    "guarding": 1,
}
CONFLICT_TYPES = ["denial", "negotiation", "urgency", "deception", "memory_conflict"]
LOCATIONS = ["city gate", "shrine", "harbor", "market", "throne hall", "wilderness"]
ARCHETYPES = ["guard", "scholar", "merchant", "healer", "outlaw", "ritualist"]
FAILURE_MODES = [
    "persona_drift",
    "context_omission",
    "over_compliance",
    "generic_response",
    "role_violation",
]

CHOSEN_PROMPT = """\
Write one strong NPC reply for a fantasy RPG.

Character: {persona}
Location: {location}
Behavior state: {behavior_state}
Player says: "{player_input}"
Game state context: {game_state_summary}
Conflict type: {conflict_type}

Requirements:
- Stay in character
- Mention relevant context naturally
- Do not break role
- 2-4 sentences

Return only the NPC response.
"""

REJECTED_PROMPT = """\
Write one flawed NPC reply for preference training.
The reply must include failure mode: {failure_mode}

Character: {persona}
Location: {location}
Behavior state: {behavior_state}
Player says: "{player_input}"

Failure modes:
- persona_drift
- context_omission
- over_compliance
- generic_response
- role_violation

Return only the flawed NPC response.
"""

PLAYER_INPUT_PROMPT = """\
Generate one player utterance for an RPG NPC dialogue.

Persona: {persona}
Location: {location}
Behavior state: {behavior_state}
Conflict type: {conflict_type}

Return only the player utterance (1-2 sentences).
"""

GAME_STATE_PROMPT = """\
Generate compact game-state JSON for an RPG NPC scenario.
Persona: {persona}
Location: {location}
State: {behavior_state}

Return JSON with keys:
alert_level, nearby_entities, active_quest, recent_event, suspicion_level
"""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    cleaned = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return parsed
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        return json.loads(obj_match.group(0))
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        return json.loads(arr_match.group(0))
    raise ValueError(f"Failed to parse JSON: {raw[:180]}")


def build_client(api_key: str, api_base: str):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def call_model(client: Any, model: str, prompt: str, json_mode: bool = False) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.85,
        "max_tokens": 320,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            return str(resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def normalize_prompt_obj(prompt_obj: Dict[str, Any]) -> Dict[str, str]:
    return {
        "persona": str(prompt_obj.get("persona", "")).strip(),
        "location": str(prompt_obj.get("location", "")).strip(),
        "behavior_state": str(prompt_obj.get("behavior_state", "")).strip(),
        "player_input": str(prompt_obj.get("player_input", "")).strip(),
        "game_state": str(prompt_obj.get("game_state", "")).strip(),
    }


def mine_from_benchmark_traces(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for trace in traces:
        scenario_id = str(trace.get("scenario_id", "")).strip()
        candidates = trace.get("candidates", [])
        if not scenario_id or not isinstance(candidates, list):
            continue
        rejected = [c for c in candidates if not bool(c.get("accepted", False))]
        accepted = [c for c in candidates if bool(c.get("accepted", False))]
        if not rejected or not accepted:
            continue

        best_rejected = max(rejected, key=lambda c: float(c.get("score", 0.0) or 0.0))
        first_accepted = accepted[0]
        score_gap = float(first_accepted.get("score", 1.0) or 1.0) - float(best_rejected.get("score", 0.0) or 0.0)
        if score_gap < 0.05:
            continue

        prompt_ctx = {
            "persona": str(trace.get("persona", "")).strip(),
            "location": str(trace.get("location", "")).strip(),
            "behavior_state": str(trace.get("behavior_state", "")).strip(),
            "player_input": str(trace.get("player_input", "")).strip(),
            "game_state": json.dumps(trace.get("game_state", {}), ensure_ascii=False),
        }
        chosen = str(first_accepted.get("text", "")).strip()
        rejected_text = str(best_rejected.get("text", "")).strip()
        if not chosen or not rejected_text or chosen == rejected_text:
            continue

        pairs.append(
            {
                "id": f"mined_trace_{scenario_id}",
                "source": "mined_trace",
                "behavior_state": prompt_ctx["behavior_state"],
                "prompt": normalize_prompt_obj(prompt_ctx),
                "chosen": chosen,
                "rejected": rejected_text,
                "score_gap": round(score_gap, 4),
            }
        )
    return pairs


def _extract_player_input_from_prompt(prompt: str) -> str:
    m = re.search(r"Player says:\s*(.+?)\s*NPC reply:\s*$", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return " ".join(m.group(1).strip().split())


def mine_from_proposal_run(run_dir: Path, arm_id: str) -> List[Dict[str, Any]]:
    responses_path = run_dir / "responses" / f"{arm_id}.jsonl"
    scenarios_path = run_dir / "scenarios.jsonl"
    if not responses_path.exists():
        raise FileNotFoundError(f"Response file not found: {responses_path}")
    scenario_rows = read_jsonl(scenarios_path) if scenarios_path.exists() else []
    by_scenario = {
        str(row.get("scenario_id", "")).strip(): row
        for row in scenario_rows
        if str(row.get("scenario_id", "")).strip()
    }

    rows = read_jsonl(responses_path)
    pairs: List[Dict[str, Any]] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", "")).strip()
        if not scenario_id:
            continue

        rejected = str(row.get("raw_response", "")).strip()
        chosen = str(row.get("response", "")).strip()
        if not rejected or not chosen:
            continue
        if rejected == chosen:
            continue

        rewrite_attempted = bool(row.get("rewrite_attempted", False))
        rewrite_successful_attempts = int(row.get("rewrite_successful_attempts", 0) or 0)
        response_repaired = bool(row.get("response_repaired", False))
        control_source = str(row.get("response_control_source", "")).strip().lower()
        repaired_sources = {
            "rewritten",
            "rewrite_grounded",
            "structured_repair",
            "fallback",
            "near_pass",
        }
        first_pass_failed = (
            response_repaired
            or rewrite_successful_attempts > 0
            or (rewrite_attempted and control_source in repaired_sources)
        )
        if not first_pass_failed:
            continue

        scenario = by_scenario.get(scenario_id, {})
        behavior_state = str(row.get("behavior_state", "")).strip() or str(scenario.get("behavior_state", "")).strip()
        prompt_text = str(row.get("prompt", "")).strip()
        player_input = str(scenario.get("player_input", "")).strip() or _extract_player_input_from_prompt(prompt_text)

        prompt_ctx = {
            "persona": str(scenario.get("persona", "")).strip(),
            "location": str(scenario.get("location", "")).strip(),
            "behavior_state": behavior_state,
            "player_input": player_input,
            "game_state": json.dumps(
                {
                    "behavior_state": behavior_state,
                    "location": str(scenario.get("location", "")).strip(),
                    "scenario_tags": scenario.get("tags", []),
                },
                ensure_ascii=False,
            ),
            "prompt_text": prompt_text,
        }

        request_index = int(row.get("request_index", 0) or 0)
        repeat_index = int(row.get("repeat_index", 0) or 0)
        pairs.append(
            {
                "id": f"mined_run_{scenario_id}_{repeat_index}_{request_index}",
                "source": "mined_run",
                "behavior_state": behavior_state,
                "prompt": normalize_prompt_obj(prompt_ctx),
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "arm_id": str(row.get("arm_id", "")).strip(),
                    "response_control_source": control_source,
                    "rewrite_attempts": int(row.get("rewrite_attempts", 0) or 0),
                    "rewrite_successful_attempts": rewrite_successful_attempts,
                    "response_repaired": response_repaired,
                    "scenario_id": scenario_id,
                },
            }
        )
    return pairs


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(str(a).lower().split())
    tb = set(str(b).lower().split())
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb)) / float(len(ta | tb))


def dedupe_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt", {})
        key = {
            "prompt": prompt,
            "chosen": str(row.get("chosen", "")).strip(),
            "rejected": str(row.get("rejected", "")).strip(),
        }
        raw = json.dumps(key, ensure_ascii=False, sort_keys=True)
        if raw in seen:
            continue
        seen.add(raw)
        out.append(row)
    return out


def quality_filter(rows: List[Dict[str, Any]], max_jaccard: float = 0.85) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in rows:
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not chosen or not rejected:
            continue
        sim = jaccard_similarity(chosen, rejected)
        if sim > float(max_jaccard):
            continue
        row["jaccard_sim"] = round(sim, 4)
        kept.append(row)
    return kept


def weighted_state_choice() -> str:
    states = list(STATE_WEIGHTS.keys())
    weights = [STATE_WEIGHTS[s] for s in states]
    return random.choices(states, weights=weights, k=1)[0]


def synthesize_pair(client: Any, model: str, idx: int) -> Dict[str, Any] | None:
    behavior_state = weighted_state_choice()
    persona = random.choice(ARCHETYPES)
    location = random.choice(LOCATIONS)
    conflict_type = random.choice(CONFLICT_TYPES)
    failure_mode = random.choice(FAILURE_MODES)

    game_state_raw = call_model(
        client,
        model,
        GAME_STATE_PROMPT.format(persona=persona, location=location, behavior_state=behavior_state),
        json_mode=True,
    )
    try:
        game_state = extract_json(game_state_raw)
    except Exception:
        game_state = {}
    if not isinstance(game_state, dict):
        game_state = {}
    gs_summary = "; ".join(f"{k}={v}" for k, v in game_state.items()) or f"location={location}; state={behavior_state}"

    player_input = call_model(
        client,
        model,
        PLAYER_INPUT_PROMPT.format(
            persona=persona,
            location=location,
            behavior_state=behavior_state,
            conflict_type=conflict_type,
        ),
    )
    chosen = call_model(
        client,
        model,
        CHOSEN_PROMPT.format(
            persona=persona,
            location=location,
            behavior_state=behavior_state,
            player_input=player_input,
            game_state_summary=gs_summary,
            conflict_type=conflict_type,
        ),
    )
    rejected = call_model(
        client,
        model,
        REJECTED_PROMPT.format(
            persona=persona,
            location=location,
            behavior_state=behavior_state,
            player_input=player_input,
            failure_mode=failure_mode,
        ),
    )

    if not chosen.strip() or not rejected.strip():
        return None
    if chosen.strip() == rejected.strip():
        return None
    if len(chosen.strip()) < 20 or len(rejected.strip()) < 20:
        return None

    return {
        "id": f"synthetic_{idx:05d}",
        "source": "synthetic",
        "behavior_state": behavior_state,
        "conflict_type": conflict_type,
        "prompt": normalize_prompt_obj(
            {
                "persona": persona,
                "location": location,
                "behavior_state": behavior_state,
                "player_input": player_input.strip(),
                "game_state": json.dumps(game_state, ensure_ascii=False),
            }
        ),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "metadata": {"failure_mode": failure_mode},
    }


def synthesize_pairs(client: Any, model: str, n: int) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    failed = 0
    idx = 0
    pbar = tqdm(total=n, desc="Synthesize pairs")
    while len(pairs) < n:
        row = synthesize_pair(client, model, idx)
        idx += 1
        if row is None:
            failed += 1
            if failed > max(20, int(n * 0.35)):
                break
            continue
        pairs.append(row)
        pbar.update(1)
    pbar.close()
    return pairs


def behavior_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("behavior_state", "unknown")).strip() or "unknown"
        dist[key] = int(dist.get(key, 0)) + 1
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def find_latest_completed_proposal_run() -> Path:
    roots = [Path("storage/artifacts/proposal"), Path("artifacts/proposal")]
    for root in roots:
        if not root.exists():
            continue
        candidates = sorted([p for p in root.rglob("run_config.json") if p.is_file()])
        for cfg in reversed(candidates):
            run_dir = cfg.parent
            responses = run_dir / "responses"
            if responses.exists() and any(responses.glob("*.jsonl")):
                return run_dir
    raise FileNotFoundError("Could not find a completed proposal run with responses/*.jsonl")


def expand_proposal_run_targets(path: Path) -> List[Path]:
    responses = path / "responses"
    if responses.exists() and any(responses.glob("*.jsonl")):
        return [path]
    runs_dir = path / "runs"
    if runs_dir.exists():
        targets: List[Path] = []
        for child in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            child_responses = child / "responses"
            if child_responses.exists() and any(child_responses.glob("*.jsonl")):
                targets.append(child)
        if targets:
            return targets
    raise FileNotFoundError(f"No proposal run with responses/ found at: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 2: conflict-state DPO mining + synthesis")
    parser.add_argument("--traces", default="", help="Optional benchmark_traces JSONL with candidate lists")
    parser.add_argument("--proposal-run-dir", default="", help="Optional proposal run dir with responses/")
    parser.add_argument("--proposal-arm", default="proposed_contextual_controlled")
    parser.add_argument("--n-synthetic", type=int, default=400)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--max-jaccard", type=float, default=0.85)
    parser.add_argument("--out-dir", default="storage/artifacts/datasets/dpo_data")
    args = parser.parse_args()

    random.seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mined: List[Dict[str, Any]] = []
    mined_sources: List[str] = []

    traces_path = Path(args.traces) if str(args.traces).strip() else None
    if traces_path:
        if not traces_path.exists():
            raise FileNotFoundError(f"traces file not found: {traces_path}")
        trace_rows = read_jsonl(traces_path)
        mined.extend(mine_from_benchmark_traces(trace_rows))
        mined_sources.append(str(traces_path))

    if str(args.proposal_run_dir).strip():
        proposal_target = Path(args.proposal_run_dir)
    else:
        proposal_target = find_latest_completed_proposal_run()
    run_dirs = expand_proposal_run_targets(proposal_target)
    for run_dir in run_dirs:
        mined.extend(mine_from_proposal_run(run_dir=run_dir, arm_id=str(args.proposal_arm)))
        mined_sources.append(str(run_dir))

    mined = dedupe_pairs(quality_filter(mined, max_jaccard=float(args.max_jaccard)))

    synthetic: List[Dict[str, Any]] = []
    if int(args.n_synthetic) > 0:
        if not str(args.api_key).strip():
            raise ValueError("Missing API key for synthetic generation. Set --api-key or OPENAI_API_KEY.")
        client = build_client(api_key=str(args.api_key).strip(), api_base=str(args.api_base).strip())
        synthetic = synthesize_pairs(client, args.model, int(args.n_synthetic))
        synthetic = dedupe_pairs(quality_filter(synthetic, max_jaccard=float(args.max_jaccard)))

    combined = dedupe_pairs(mined + synthetic)
    random.shuffle(combined)

    out_mined = out_dir / "dpo_pairs_mined.jsonl"
    out_synth = out_dir / "dpo_pairs_synthetic.jsonl"
    out_combined = out_dir / "dpo_pairs_combined.jsonl"
    out_summary = out_dir / "dpo_pairs_summary.json"

    write_jsonl(out_mined, mined)
    write_jsonl(out_synth, synthetic)
    write_jsonl(out_combined, combined)
    write_json(
        out_summary,
        {
            "mined_sources": mined_sources,
            "proposal_run_dir": str(proposal_target),
            "proposal_run_dirs_expanded": [str(p) for p in run_dirs],
            "proposal_arm": str(args.proposal_arm),
            "counts": {
                "mined": len(mined),
                "synthetic": len(synthetic),
                "combined": len(combined),
            },
            "state_distribution": {
                "mined": behavior_distribution(mined),
                "synthetic": behavior_distribution(synthetic),
                "combined": behavior_distribution(combined),
            },
            "high_retry_states_targeted": HIGH_RETRY_STATES,
            "model": str(args.model),
            "max_jaccard": float(args.max_jaccard),
            "outputs": {
                "mined": str(out_mined),
                "synthetic": str(out_synth),
                "combined": str(out_combined),
                "summary": str(out_summary),
            },
        },
    )

    print(f"Mined pairs: {len(mined)} -> {out_mined}")
    print(f"Synthetic pairs: {len(synthetic)} -> {out_synth}")
    print(f"Combined pairs: {len(combined)} -> {out_combined}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
