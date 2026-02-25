#!/usr/bin/env python3
"""Run full proposal/publication pipeline on laptop hardware with stage checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests


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
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def resolve_run_path(raw: str, root: Path) -> Path:
    token = str(raw or "").strip()
    if not token:
        raise ValueError("empty run token")
    if token.lower() == "latest":
        return latest_subdir(root)
    p = Path(token)
    if p.exists():
        return p
    candidate = root / token
    if candidate.exists():
        return candidate
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


def run_stage(
    *,
    name: str,
    command: List[str],
    state: Dict[str, Any],
    state_path: Path,
    logs_dir: Path,
    dry_run: bool,
    skip_if_completed: bool = True,
) -> None:
    stages = state.setdefault("stages", {})
    current = stages.get(name, {})
    if skip_if_completed and current.get("status") == "completed":
        print(f"[skip] {name} already completed")
        return

    log_path = logs_dir / f"{name}.log"
    stages[name] = {
        "status": "running",
        "started_utc": utc_iso(),
        "command": command,
        "log_path": str(log_path),
    }
    write_json(state_path, state)

    printable = " ".join(command)
    print(f"[stage:{name}] {printable}")
    if dry_run:
        stages[name].update({"status": "completed", "finished_utc": utc_iso(), "returncode": 0, "dry_run": True})
        write_json(state_path, state)
        return

    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text((proc.stdout or "") + "\n\n[stderr]\n" + (proc.stderr or ""), encoding="utf-8")
    stages[name].update(
        {
            "finished_utc": utc_iso(),
            "returncode": proc.returncode,
            "status": "completed" if proc.returncode == 0 else "failed",
        }
    )
    write_json(state_path, state)
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed ({proc.returncode}). Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop-safe, resumable full-result runner.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:latest")
    parser.add_argument("--serving-models", default="elara-npc:latest,phi3:mini,phi3:latest")
    parser.add_argument("--scenario-file", default="data/proposal_eval_scenarios_large.jsonl")
    parser.add_argument("--output-root", default="artifacts/laptop_full")
    parser.add_argument("--run-id", default="", help="Optional fixed run id; defaults to UTC timestamp.")
    parser.add_argument("--resume-run", default="", help="Resume an existing laptop run directory.")
    parser.add_argument("--proposal-run", default="", help="Reuse proposal run id/path/latest.")
    parser.add_argument("--publication-run", default="", help="Reuse publication run id/path/latest.")
    parser.add_argument("--only-stages", default="all", help="Comma list or 'all'.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--proposal-max-tokens", type=int, default=80)
    parser.add_argument("--proposal-temperature", type=float, default=0.2)
    parser.add_argument("--multirater-scenarios", type=int, default=36)
    parser.add_argument("--serving-max-tokens", type=int, default=56)
    parser.add_argument("--serving-temperature", type=float, default=0.2)
    parser.add_argument("--skip-ablation-baselines", action="store_true")
    parser.add_argument("--run-security-benchmark", action="store_true")
    parser.add_argument("--require-security-benchmark", action="store_true")
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

    if stage_enabled("check_ollama", selected):
        run_stage(
            name="check_ollama",
            command=[sys.executable, "-c", "print('ollama check delegated')"],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            model_checks = [str(args.candidate_model), str(args.baseline_model)]
            model_checks.extend([x.strip() for x in str(args.baseline_models).split(",") if x.strip()])
            ollama_ready(args.host, model_checks)

    if stage_enabled("generate_inputs", selected):
        run_stage(
            name="generate_inputs",
            command=[sys.executable, "-c", "print('generate_inputs delegated')"],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
        )
        maybe_generate_inputs(dry_run=args.dry_run)

    proposal_token = str(args.proposal_run).strip()
    if proposal_token:
        context["proposal_run"] = str(resolve_run_path(proposal_token, Path("artifacts/proposal")))
    elif stage_enabled("proposal_eval", selected):
        run_stage(
            name="proposal_eval",
            command=[
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
                "--bertscore-model-type",
                "roberta-large",
                "--bertscore-batch-size",
                "16",
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            context["proposal_run"] = str(latest_subdir(Path("artifacts/proposal")))
            write_json(state_path, state)
    if "proposal_run" not in context:
        raise RuntimeError("No proposal run available. Provide --proposal-run or enable proposal_eval stage.")
    proposal_run = Path(context["proposal_run"])

    human_csv = proposal_run / "human_eval_llm_multirater_consistent.csv"
    if stage_enabled("multirater", selected):
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
        )

    if stage_enabled("lexical", selected):
        run_stage(
            name="lexical",
            command=[sys.executable, "scripts/run_lexical_diversity_benchmark.py", "--run-dir", str(proposal_run)],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            dry_run=args.dry_run,
        )

    publication_token = str(args.publication_run).strip()
    if publication_token:
        context["publication_run"] = str(resolve_run_path(publication_token, Path("artifacts/publication")))
    elif stage_enabled("publication", selected):
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
        )
        if not args.dry_run:
            context["publication_run"] = str(latest_subdir(Path("artifacts/publication")))
            write_json(state_path, state)
    if "publication_run" not in context:
        raise RuntimeError("No publication run available. Provide --publication-run or enable publication stage.")
    publication_run = Path(context["publication_run"])

    if stage_enabled("serving_matrix", selected):
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
        )
        if not args.dry_run:
            context["serving_efficiency_run"] = str(latest_subdir(Path("artifacts/serving_efficiency")))
            write_json(state_path, state)

    if stage_enabled("external_profiles", selected):
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
        )
        if not args.dry_run:
            context["publication_profile_suite"] = str(latest_subdir(Path("artifacts/publication_profiles")))
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
        )

    dpo_output = Path(args.dpo_output) if str(args.dpo_output).strip() else Path("outputs") / f"dpo_adapter_{run_dir.name}"
    if args.run_dpo and stage_enabled("dpo_train", selected):
        run_stage(
            name="dpo_train",
            command=[
                sys.executable,
                "scripts/train_dpo.py",
                "--dataset",
                str(proposal_run / "preference_dataset.jsonl"),
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
        )
        context["dpo_model_tag"] = str(args.dpo_model_tag)
        write_json(state_path, state)

    if stage_enabled("quality_gate", selected):
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
        )

    manifest = {
        "run_dir": str(run_dir),
        "host": str(args.host),
        "candidate_model": str(args.candidate_model),
        "baseline_model": str(args.baseline_model),
        "proposal_run": str(proposal_run),
        "publication_run": str(publication_run),
        "serving_efficiency_run": context.get("serving_efficiency_run", ""),
        "publication_profile_suite": context.get("publication_profile_suite", ""),
        "human_eval_csv": str(human_csv),
        "quality_gate_json": str(proposal_run / "quality_gate_report_final.json"),
        "quality_gate_md": str(proposal_run / "quality_gate_report_final.md"),
        "preference_dataset": str(proposal_run / "preference_dataset.jsonl"),
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

