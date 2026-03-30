#!/usr/bin/env python3
"""Run full proposal/publication pipeline on laptop hardware with stage checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests
try:
    from local_model_profiles import (
        baseline_profile_choices,
        fetch_ollama_models,
        format_model_csv,
        resolve_baseline_models,
        split_available_missing,
    )
except ModuleNotFoundError:
    from scripts.local_model_profiles import (
        baseline_profile_choices,
        fetch_ollama_models,
        format_model_csv,
        resolve_baseline_models,
        split_available_missing,
    )

STORAGE_ARTIFACT_ROOT = Path("storage/artifacts")
LEGACY_ARTIFACT_ROOT = Path("artifacts")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_subdir(root: Path) -> Path:
    dirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and p.name != "latest" and not p.name.endswith("_batch_tmp")
    ]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def resolve_run_path(raw: str, root: Path, fallback_root: Path | None = None) -> Path:
    token = str(raw or "").strip()
    if not token:
        raise ValueError("empty run token")
    if token.lower() == "latest":
        try:
            return latest_subdir(root)
        except FileNotFoundError:
            if fallback_root is not None:
                return latest_subdir(fallback_root)
            raise
    p = Path(token)
    if p.exists():
        return p
    candidate = root / token
    if candidate.exists():
        return candidate
    if fallback_root is not None:
        fallback = fallback_root / token
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Cannot resolve run path '{raw}' under {root}")


def ollama_ready(host: str, model_names: Sequence[str], timeout_s: int = 12) -> None:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Ollama host unavailable at {host}: {exc}") from exc
    models = {
        str(item.get("name", "")).strip()
        for item in payload.get("models", [])
        if str(item.get("name", "")).strip()
    }
    missing = [m for m in model_names if m and m not in models]
    if missing:
        raise RuntimeError(f"Missing Ollama models at {host}: {missing}")


def maybe_generate_inputs(dry_run: bool) -> None:
    def run(command: List[str]) -> None:
        if dry_run:
            print("[dry-run]", " ".join(command))
            return
        proc = subprocess.run(command, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}")

    if not Path("data/retrieval_gold_wide.jsonl").exists() or not Path("data/retrieval_corpus_wide.jsonl").exists():
        run(
            [
                sys.executable,
                "scripts/generate_wide_benchmark_sets.py",
                "--docs-per-domain",
                "24",
                "--queries-per-domain",
                "20",
                "--prompts-per-domain",
                "4",
            ]
        )
    if not Path("data/retrieval_poison_benchmark_large.jsonl").exists():
        run(
            [
                sys.executable,
                "scripts/generate_poison_benchmark_large.py",
                "--target-size",
                "400",
            ]
        )
    if not Path("data/proposal_eval_scenarios_large.jsonl").exists():
        run(
            [
                sys.executable,
                "scripts/generate_proposal_scenarios_large.py",
                "--variants-per-base",
                "14",
                "--output",
                "data/proposal_eval_scenarios_large.jsonl",
            ]
        )


def stage_enabled(stage: str, only: set[str]) -> bool:
    return "all" in only or stage in only


def update_stage_state(
    *,
    state: Dict[str, Any],
    state_path: Path,
    name: str,
    status: str,
    command: List[str] | None = None,
    log_path: Path | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    stages = state.setdefault("stages", {})
    payload: Dict[str, Any] = {
        "status": status,
        "updated_utc": utc_iso(),
    }
    if command is not None:
        payload["command"] = command
    if log_path is not None:
        payload["log_path"] = str(log_path)
    if extra:
        payload.update(extra)
    previous = stages.get(name, {})
    previous.update(payload)
    stages[name] = previous
    write_json(state_path, state)


def run_stage(
    *,
    name: str,
    command: List[str],
    state: Dict[str, Any],
    state_path: Path,
    logs_dir: Path,
    dry_run: bool,
    skip_if_completed: bool = True,
    timeout_s: int = 0,
    retries: int = 1,
    retry_delay_s: int = 3,
) -> None:
    stages = state.setdefault("stages", {})
    current = stages.get(name, {})
    if skip_if_completed and current.get("status") == "completed":
        print(f"[skip] {name} already completed")
        return

    if retries < 1:
        retries = 1

    stage_log_path = logs_dir / f"{name}.log"
    stages[name] = {
        "status": "running",
        "started_utc": utc_iso(),
        "command": command,
        "log_path": str(stage_log_path),
        "attempts": [],
    }
    write_json(state_path, state)

    printable = " ".join(command)
    print(f"[stage:{name}] {printable} (retries={retries}, timeout_s={timeout_s})")
    if dry_run:
        stages[name].update({"status": "completed", "finished_utc": utc_iso(), "returncode": 0, "dry_run": True})
        write_json(state_path, state)
        return

    last_rc = 1
    last_log = stage_log_path
    for attempt in range(1, retries + 1):
        attempt_log = logs_dir / f"{name}.attempt{attempt}.log"
        last_log = attempt_log
        attempt_started = utc_iso()
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=(timeout_s if timeout_s > 0 else None),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            last_rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout_raw = exc.stdout
            stderr_raw = exc.stderr
            if isinstance(stdout_raw, bytes):
                stdout = stdout_raw.decode("utf-8", errors="replace")
            elif isinstance(stdout_raw, str):
                stdout = stdout_raw
            else:
                stdout = ""
            if isinstance(stderr_raw, bytes):
                stderr = stderr_raw.decode("utf-8", errors="replace")
            elif isinstance(stderr_raw, str):
                stderr = stderr_raw
            else:
                stderr = ""
            stderr += f"\n[timeout] stage exceeded timeout_s={timeout_s}"
            last_rc = 124
            timed_out = True

        attempt_log.write_text(stdout + "\n\n[stderr]\n" + stderr, encoding="utf-8")
        stages[name].setdefault("attempts", []).append(
            {
                "attempt": attempt,
                "started_utc": attempt_started,
                "finished_utc": utc_iso(),
                "returncode": last_rc,
                "timed_out": timed_out,
                "log_path": str(attempt_log),
            }
        )
        write_json(state_path, state)

        if last_rc == 0:
            break

        if attempt < retries:
            print(f"[stage:{name}] attempt {attempt}/{retries} failed (rc={last_rc}), retrying in {retry_delay_s}s...")
            time.sleep(max(0, int(retry_delay_s)))

    stages[name].update(
        {
            "finished_utc": utc_iso(),
            "returncode": last_rc,
            "status": "completed" if last_rc == 0 else "failed",
            "log_path": str(last_log),
        }
    )
    write_json(state_path, state)
    if last_rc != 0:
        raise RuntimeError(f"Stage '{name}' failed ({last_rc}). Log: {last_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop-safe, resumable full-result runner.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:latest")
    parser.add_argument(
        "--baseline-profile",
        default="none",
        choices=baseline_profile_choices(),
        help="Optional baseline profile to expand baseline-models with laptop-safe packs.",
    )
    parser.add_argument("--serving-models", default="elara-npc:latest,phi3:mini,phi3:latest")
    parser.add_argument("--scenario-file", default="data/proposal_eval_scenarios_large.jsonl")
    parser.add_argument("--output-root", default="storage/artifacts/laptop_full")
    parser.add_argument("--run-id", default="", help="Optional fixed run id; defaults to UTC timestamp.")
    parser.add_argument("--resume-run", default="", help="Resume an existing laptop run directory.")
    parser.add_argument("--proposal-run", default="", help="Reuse proposal run id/path/latest.")
    parser.add_argument("--publication-run", default="", help="Reuse publication run id/path/latest.")
    parser.add_argument("--only-stages", default="all", help="Comma list or 'all'.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--proposal-max-tokens", type=int, default=80)
    parser.add_argument("--proposal-temperature", type=float, default=0.2)
    parser.add_argument(
        "--proposal-control-alt-profile",
        choices=[
            "none",
            "runtime_optimized",
            "quality_optimized",
            "risk_latency_aware",
            "hybrid_balanced",
            "intent_focus_adaptive",
            "blend_balanced",
            "custom",
        ],
        default="none",
        help="Optional alternate response-control profile to add as a second controlled arm.",
    )
    parser.add_argument(
        "--proposal-control-alt-arm-id",
        default="proposed_contextual_controlled_alt",
        help="Arm ID for alternate proposal control profile.",
    )
    parser.add_argument(
        "--proposal-min-arm-success-rate",
        type=float,
        default=0.90,
        help="Minimum required successful-request rate per arm in proposal eval.",
    )
    parser.add_argument("--proposal-enable-control-near-pass", action="store_true")
    parser.add_argument("--proposal-control-near-pass-max-context-gap", type=float, default=0.05)
    parser.add_argument("--proposal-control-near-pass-max-persona-gap", type=float, default=0.04)
    parser.add_argument("--proposal-control-near-pass-score-floor", type=float, default=0.34)
    parser.add_argument("--proposal-disable-control-near-pass-block-high-risk", action="store_true")
    parser.add_argument("--multirater-scenarios", type=int, default=36)
    parser.add_argument("--serving-max-tokens", type=int, default=56)
    parser.add_argument("--serving-temperature", type=float, default=0.2)
    parser.add_argument("--skip-ablation-baselines", action="store_true")
    parser.add_argument("--run-security-benchmark", action="store_true")
    parser.add_argument("--require-security-benchmark", action="store_true")
    parser.add_argument(
        "--allow-missing-ollama",
        action="store_true",
        help="If Ollama is unavailable, continue and skip Ollama-dependent stages where possible.",
    )
    parser.add_argument(
        "--allow-missing-baselines",
        action="store_true",
        help="Prune missing baseline-models; candidate-model and baseline-model remain required.",
    )
    parser.add_argument("--stage-timeout-s", type=int, default=0, help="Per-stage subprocess timeout in seconds; 0 disables timeout.")
    parser.add_argument("--stage-retries", type=int, default=2, help="Retries per stage on non-zero return codes/timeouts.")
    parser.add_argument("--stage-retry-delay-s", type=int, default=4, help="Delay between stage retries.")
    parser.add_argument("--run-dpo", action="store_true", help="Run DPO training stage.")
    parser.add_argument("--dpo-base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dpo-output", default="", help="DPO adapter output dir.")
    parser.add_argument("--dpo-epochs", type=int, default=2)
    parser.add_argument("--dpo-batch-size", type=int, default=1)
    parser.add_argument("--dpo-grad-acc", type=int, default=8)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=512)
    parser.add_argument("--dpo-max-length", type=int, default=768)
    parser.add_argument("--dpo-lr", type=float, default=5e-6)
    parser.add_argument("--register-dpo", action="store_true", help="Register trained DPO adapter into Ollama.")
    parser.add_argument("--dpo-model-tag", default="elara-npc-dpo-laptop:latest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resolved_baselines = resolve_baseline_models(str(args.baseline_models), str(args.baseline_profile))
    if not resolved_baselines:
        resolved_baselines = [str(args.baseline_model)]
    args.baseline_models = format_model_csv(resolved_baselines)

    proposal_root = STORAGE_ARTIFACT_ROOT / "proposal"
    proposal_fallback_root = LEGACY_ARTIFACT_ROOT / "proposal"
    publication_root = STORAGE_ARTIFACT_ROOT / "publication"
    publication_fallback_root = LEGACY_ARTIFACT_ROOT / "publication"
    serving_root = STORAGE_ARTIFACT_ROOT / "serving_efficiency"
    serving_fallback_root = LEGACY_ARTIFACT_ROOT / "serving_efficiency"
    profiles_root = STORAGE_ARTIFACT_ROOT / "publication_profiles"
    profiles_fallback_root = LEGACY_ARTIFACT_ROOT / "publication_profiles"

    if bool(args.require_security_benchmark) and not bool(args.run_security_benchmark):
        raise ValueError("--require-security-benchmark requires --run-security-benchmark")
    if args.register_dpo and not args.run_dpo:
        raise ValueError("--register-dpo requires --run-dpo")

    selected = {x.strip() for x in str(args.only_stages).split(",") if x.strip()}
    if not selected:
        selected = {"all"}

    if args.resume_run:
        run_dir = Path(args.resume_run)
        if not run_dir.exists():
            raise FileNotFoundError(f"--resume-run not found: {run_dir}")
    else:
        rid = str(args.run_id).strip() or utc_stamp()
        run_dir = Path(args.output_root) / rid
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "stage_status.json"
    if state_path.exists():
        state = read_json(state_path)
    else:
        state = {
            "run_dir": str(run_dir),
            "started_utc": utc_iso(),
            "host": str(args.host),
            "candidate_model": str(args.candidate_model),
            "baseline_model": str(args.baseline_model),
            "stages": {},
            "context": {},
        }
        write_json(state_path, state)

    context = state.setdefault("context", {})
    context.setdefault("ollama_available", True)

    if stage_enabled("check_ollama", selected):
        check_cmd = [sys.executable, "-c", "print('ollama readiness check')"]
        if args.dry_run:
            run_stage(
                name="check_ollama",
                command=check_cmd,
                state=state,
                state_path=state_path,
                logs_dir=logs_dir,
                dry_run=True,
                timeout_s=int(args.stage_timeout_s),
                retries=int(args.stage_retries),
                retry_delay_s=int(args.stage_retry_delay_s),
            )
        else:
            log_path = logs_dir / "check_ollama.log"
            update_stage_state(
                state=state,
                state_path=state_path,
                name="check_ollama",
                status="running",
                command=check_cmd,
                log_path=log_path,
                extra={"started_utc": utc_iso()},
            )
            required_models = [str(args.candidate_model), str(args.baseline_model)]
            baseline_tokens = [x.strip() for x in str(args.baseline_models).split(",") if x.strip()]
            model_checks = required_models + baseline_tokens
            try:
                ollama_ready(args.host, model_checks)
                log_path.write_text(f"Ollama OK at {args.host}\nmodels={model_checks}\n", encoding="utf-8")
                update_stage_state(
                    state=state,
                    state_path=state_path,
                    name="check_ollama",
                    status="completed",
                    command=check_cmd,
                    log_path=log_path,
                    extra={"finished_utc": utc_iso(), "returncode": 0},
                )
                context["ollama_available"] = True
            except Exception as exc:
                recovered = False
                if bool(args.allow_missing_baselines):
                    try:
                        installed = fetch_ollama_models(str(args.host), timeout_s=12)
                        required_avail, required_missing = split_available_missing(required_models, installed)
                        if required_missing:
                            raise RuntimeError(
                                f"Missing required models (cannot prune): {required_missing}"
                            ) from exc
                        baseline_avail, baseline_missing = split_available_missing(baseline_tokens, installed)
                        if not baseline_avail:
                            baseline_avail = [str(args.baseline_model)]
                        args.baseline_models = format_model_csv(baseline_avail)
                        model_checks = required_avail + baseline_avail
                        ollama_ready(args.host, model_checks)
                        log_path.write_text(
                            "Ollama OK after pruning missing baselines.\n"
                            f"pruned={baseline_missing}\n"
                            f"models={model_checks}\n",
                            encoding="utf-8",
                        )
                        update_stage_state(
                            state=state,
                            state_path=state_path,
                            name="check_ollama",
                            status="completed",
                            command=check_cmd,
                            log_path=log_path,
                            extra={
                                "finished_utc": utc_iso(),
                                "returncode": 0,
                                "pruned_missing_baselines": baseline_missing,
                                "effective_baseline_models": str(args.baseline_models),
                            },
                        )
                        context["ollama_available"] = True
                        context["effective_baseline_models"] = str(args.baseline_models)
                        recovered = True
                    except Exception as prune_exc:
                        exc = prune_exc
                if not recovered:
                    log_path.write_text(f"Ollama check failed at {args.host}\n{exc}\n", encoding="utf-8")
                    update_stage_state(
                        state=state,
                        state_path=state_path,
                        name="check_ollama",
                        status="failed",
                        command=check_cmd,
                        log_path=log_path,
                        extra={"finished_utc": utc_iso(), "returncode": 1, "error": str(exc)},
                    )
                    context["ollama_available"] = False
                    if not args.allow_missing_ollama:
                        raise

    if stage_enabled("generate_inputs", selected):
        run_stage(
            name="generate_inputs",
            command=[sys.executable, "-c", "print('generate_inputs delegated')"],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        maybe_generate_inputs(dry_run=args.dry_run)

    proposal_token = str(args.proposal_run).strip()
    if proposal_token:
        context["proposal_run"] = str(resolve_run_path(proposal_token, proposal_root, proposal_fallback_root))
    elif stage_enabled("proposal_eval", selected):
        if not context.get("ollama_available", True):
            raise RuntimeError("proposal_eval requires Ollama. Use --proposal-run to reuse an existing run.")
        proposal_eval_cmd = [
            sys.executable,
            "scripts/run_proposal_alignment_eval_batched.py",
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--baseline-models",
            str(args.baseline_models),
            "--scenarios",
            str(args.scenario_file),
            "--batch-size",
            str(args.batch_size),
            "--repeats",
            "1",
            "--max-tokens",
            str(args.proposal_max_tokens),
            "--temperature",
            str(args.proposal_temperature),
            "--min-arm-success-rate",
            str(args.proposal_min_arm_success_rate),
            "--bertscore-model-type",
            "roberta-large",
            "--bertscore-batch-size",
            "16",
            "--control-alt-profile",
            str(args.proposal_control_alt_profile),
            "--control-alt-arm-id",
            str(args.proposal_control_alt_arm_id),
            "--control-near-pass-max-context-gap",
            str(args.proposal_control_near_pass_max_context_gap),
            "--control-near-pass-max-persona-gap",
            str(args.proposal_control_near_pass_max_persona_gap),
            "--control-near-pass-score-floor",
            str(args.proposal_control_near_pass_score_floor),
        ]
        if bool(args.proposal_enable_control_near_pass):
            proposal_eval_cmd.append("--enable-control-near-pass")
        if bool(args.proposal_disable_control_near_pass_block_high_risk):
            proposal_eval_cmd.append("--disable-control-near-pass-block-high-risk")
        run_stage(
            name="proposal_eval",
            command=proposal_eval_cmd,
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        if not args.dry_run:
            if proposal_root.exists():
                context["proposal_run"] = str(latest_subdir(proposal_root))
            else:
                context["proposal_run"] = str(latest_subdir(proposal_fallback_root))
            write_json(state_path, state)
    proposal_run = Path(context["proposal_run"]) if "proposal_run" in context else None
    proposal_required_stages = {
        "multirater",
        "attach_human_eval",
        "lexical",
        "build_preference",
        "quality_gate",
    }
    needs_proposal_run = any(stage_enabled(stage, selected) for stage in proposal_required_stages) or (
        bool(args.run_dpo) and stage_enabled("dpo_train", selected)
    )
    if needs_proposal_run and proposal_run is None:
        raise RuntimeError(
            "Selected stages require a proposal run. Provide --proposal-run or enable proposal_eval stage."
        )

    human_csv = (
        (proposal_run / "human_eval_llm_multirater_consistent.csv")
        if proposal_run is not None
        else (run_dir / "human_eval_llm_multirater_consistent.csv")
    )
    if stage_enabled("multirater", selected):
        if not context.get("ollama_available", True):
            raise RuntimeError("multirater requires Ollama.")
        run_stage(
            name="multirater",
            command=[
                sys.executable,
                "scripts/run_llm_multirater_campaign.py",
                "--run-dir",
                str(proposal_run),
                "--host",
                str(args.host),
                "--annotators",
                "phi3:mini|balanced|0.00,phi3:mini|balanced|0.05,phi3:mini|balanced|0.10",
                "--arms",
                "proposed_contextual_controlled,baseline_no_context,baseline_no_context_phi3_latest",
                "--scenario-limit",
                str(args.multirater_scenarios),
                "--max-tokens",
                "160",
                "--timeout-s",
                "120",
                "--output-csv",
                str(human_csv),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )

    if stage_enabled("attach_human_eval", selected):
        run_stage(
            name="attach_human_eval",
            command=[
                sys.executable,
                "scripts/attach_human_eval_to_run.py",
                "--run-dir",
                str(proposal_run),
                "--human-eval-file",
                str(human_csv),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )

    if stage_enabled("lexical", selected):
        run_stage(
            name="lexical",
            command=[sys.executable, "scripts/run_lexical_diversity_benchmark.py", "--run-dir", str(proposal_run)],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )

    publication_token = str(args.publication_run).strip()
    if publication_token:
        context["publication_run"] = str(
            resolve_run_path(publication_token, publication_root, publication_fallback_root)
        )
    elif stage_enabled("publication", selected):
        if not context.get("ollama_available", True):
            raise RuntimeError("publication stage requires Ollama. Use --publication-run to reuse an existing run.")
        pub_cmd = [
            sys.executable,
            "scripts/run_publication_benchmark_suite.py",
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--repeats",
            "1",
            "--max-tokens",
            str(args.serving_max_tokens),
            "--temperature",
            str(args.serving_temperature),
            "--reranker-pairs",
            "data/retrieval_reranker_pairs_wide.jsonl",
            "--reranker-train-frac",
            "0.85",
            "--reranker-bm25-candidate-k",
            "24",
        ]
        if args.run_security_benchmark:
            pub_cmd.extend(["--run-security-benchmark", "--run-security-spoofed-benchmark"])
        if args.skip_ablation_baselines:
            pub_cmd.append("--skip-ablation-baselines")
        run_stage(
            name="publication",
            command=pub_cmd,
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        if not args.dry_run:
            if publication_root.exists():
                context["publication_run"] = str(latest_subdir(publication_root))
            else:
                context["publication_run"] = str(latest_subdir(publication_fallback_root))
            write_json(state_path, state)
    publication_run = Path(context["publication_run"]) if "publication_run" in context else None
    needs_publication_run = stage_enabled("quality_gate", selected)
    if needs_publication_run and publication_run is None:
        raise RuntimeError(
            "Selected stages require a publication run. Provide --publication-run or enable publication stage."
        )

    if stage_enabled("serving_matrix", selected):
        if not context.get("ollama_available", True):
            raise RuntimeError("serving_matrix requires Ollama.")
        run_stage(
            name="serving_matrix",
            command=[
                sys.executable,
                "scripts/run_serving_efficiency_matrix.py",
                "--host",
                str(args.host),
                "--models",
                str(args.serving_models),
                "--candidate-model",
                str(args.candidate_model),
                "--prompts",
                "data/serving_prompts_wide.jsonl",
                "--serving-references",
                "data/serving_references_wide.jsonl",
                "--max-tokens",
                str(args.serving_max_tokens),
                "--temperature",
                str(args.serving_temperature),
                "--repeats",
                "1",
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        if not args.dry_run:
            if serving_root.exists():
                context["serving_efficiency_run"] = str(latest_subdir(serving_root))
            else:
                context["serving_efficiency_run"] = str(latest_subdir(serving_fallback_root))
            write_json(state_path, state)

    if stage_enabled("external_profiles", selected):
        if not context.get("ollama_available", True):
            raise RuntimeError("external_profiles requires Ollama.")
        run_stage(
            name="external_profiles",
            command=[
                sys.executable,
                "scripts/run_external_profile_suite.py",
                "--profiles",
                "core,wide",
                "--host",
                str(args.host),
                "--candidate-model",
                str(args.candidate_model),
                "--baseline-model",
                str(args.baseline_model),
                "--repeats",
                "1",
                "--max-tokens",
                str(args.serving_max_tokens),
                "--temperature",
                str(args.serving_temperature),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        if not args.dry_run:
            if profiles_root.exists():
                context["publication_profile_suite"] = str(latest_subdir(profiles_root))
            else:
                context["publication_profile_suite"] = str(latest_subdir(profiles_fallback_root))
            write_json(state_path, state)

    if stage_enabled("build_preference", selected):
        run_stage(
            name="build_preference",
            command=[
                sys.executable,
                "scripts/build_preference_dataset_from_eval.py",
                "--run-dir",
                str(proposal_run),
                "--human-eval-file",
                str(human_csv),
                "--target-arm",
                "proposed_contextual_controlled",
                "--baseline-arms",
                "baseline_no_context,baseline_no_context_phi3_latest",
                "--metric",
                "overall_quality",
                "--min-raters",
                "2",
                "--min-margin",
                "0.25",
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )

    if stage_enabled("build_hard_negatives", selected):
        run_stage(
            name="build_hard_negatives",
            command=[
                sys.executable,
                "scripts/build_retrieval_hard_negative_set.py",
                "--retrieval-gold",
                "data/retrieval_gold_wide.jsonl",
                "--retrieval-corpus",
                "data/retrieval_corpus_wide.jsonl",
                "--hard-negatives-per-query",
                "10",
                "--cross-domain-negatives-per-query",
                "4",
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )

    dpo_output = Path(args.dpo_output) if str(args.dpo_output).strip() else Path("storage/outputs") / f"dpo_adapter_{run_dir.name}"
    if args.run_dpo and stage_enabled("dpo_train", selected):
        if proposal_run is None:
            raise RuntimeError("dpo_train requires a proposal run directory.")
        preference_dataset_path = proposal_run / "preference_dataset.jsonl"
        run_stage(
            name="dpo_train",
            command=[
                sys.executable,
                "scripts/train_dpo.py",
                "--dataset",
                str(preference_dataset_path),
                "--base-model",
                str(args.dpo_base_model),
                "--output-dir",
                str(dpo_output),
                "--batch-size",
                str(args.dpo_batch_size),
                "--grad-acc",
                str(args.dpo_grad_acc),
                "--epochs",
                str(args.dpo_epochs),
                "--max-prompt-length",
                str(args.dpo_max_prompt_length),
                "--max-length",
                str(args.dpo_max_length),
                "--lr",
                str(args.dpo_lr),
                "--save-steps",
                "20",
                "--logging-steps",
                "1",
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        context["dpo_adapter"] = str(dpo_output)
        write_json(state_path, state)

    if args.register_dpo and stage_enabled("dpo_register", selected):
        run_stage(
            name="dpo_register",
            command=[
                sys.executable,
                "scripts/register_dpo_candidate.py",
                "--adapter",
                str(dpo_output),
                "--model-tag",
                str(args.dpo_model_tag),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )
        context["dpo_model_tag"] = str(args.dpo_model_tag)
        write_json(state_path, state)

    if stage_enabled("quality_gate", selected):
        if proposal_run is None or publication_run is None:
            raise RuntimeError(
                "quality_gate requires both proposal and publication runs. "
                "Provide --proposal-run/--publication-run or enable proposal_eval/publication stages."
            )
        gate_cmd = [
            sys.executable,
            "scripts/proposal_quality_gate.py",
            "--proposal-run",
            str(proposal_run),
            "--publication-run",
            str(publication_run),
            "--require-human-eval",
            "--output-json",
            str(proposal_run / "quality_gate_report_final.json"),
            "--output-md",
            str(proposal_run / "quality_gate_report_final.md"),
        ]
        if args.require_security_benchmark:
            gate_cmd.append("--require-security-benchmark")
        run_stage(
            name="quality_gate",
            command=gate_cmd,
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
            timeout_s=int(args.stage_timeout_s),
            retries=int(args.stage_retries),
            retry_delay_s=int(args.stage_retry_delay_s),
        )

    manifest = {
        "run_dir": str(run_dir),
        "host": str(args.host),
        "candidate_model": str(args.candidate_model),
        "baseline_model": str(args.baseline_model),
        "proposal_run": str(proposal_run) if proposal_run is not None else "",
        "publication_run": str(publication_run) if publication_run is not None else "",
        "serving_efficiency_run": context.get("serving_efficiency_run", ""),
        "publication_profile_suite": context.get("publication_profile_suite", ""),
        "human_eval_csv": str(human_csv),
        "quality_gate_json": (
            str(proposal_run / "quality_gate_report_final.json") if proposal_run is not None else ""
        ),
        "quality_gate_md": (
            str(proposal_run / "quality_gate_report_final.md") if proposal_run is not None else ""
        ),
        "preference_dataset": str(proposal_run / "preference_dataset.jsonl") if proposal_run is not None else "",
        "retrieval_hard_negatives": "data/retrieval_hard_negatives_wide.jsonl",
        "retrieval_reranker_pairs": "data/retrieval_reranker_pairs_wide.jsonl",
        "skip_ablation_baselines": bool(args.skip_ablation_baselines),
        "run_dpo": bool(args.run_dpo),
        "dpo_adapter": context.get("dpo_adapter", ""),
        "dpo_model_tag": context.get("dpo_model_tag", ""),
        "stage_status": str(state_path),
        "dry_run": bool(args.dry_run),
        "generated_utc": utc_iso(),
    }
    write_json(run_dir / "manifest.json", manifest)
    state["finished_utc"] = utc_iso()
    write_json(state_path, state)

    print(f"Run dir: {run_dir}")
    print(f"Stage status: {state_path}")
    print(f"Manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
