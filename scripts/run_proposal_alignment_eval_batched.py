#!/usr/bin/env python3
"""Run proposal evaluation in scenario batches, then merge into one final artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Import sibling script as module.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_proposal_alignment_eval as evalmod  # type: ignore


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_subprocess(command: List[str]) -> None:
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Batch command failed ({proc.returncode}): {' '.join(command)}")


def split_batches(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    step = max(1, int(batch_size))
    for idx in range(0, len(rows), step):
        out.append(rows[idx : idx + step])
    return out


def latest_subdir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directory found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def maybe_append(cmd: List[str], flag: str, condition: bool) -> None:
    if condition:
        cmd.append(flag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run proposal eval in batches and merge outputs.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="")
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_large.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--bertscore-model-type", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-cache-dir", default="")
    parser.add_argument("--control-min-context-coverage", type=float, default=0.35)
    parser.add_argument("--control-min-persona-coverage", type=float, default=0.20)
    parser.add_argument("--control-rewrite-max-tokens", type=int, default=96)
    parser.add_argument("--control-rewrite-candidates", type=int, default=3)
    parser.add_argument("--control-rewrite-temperature-step", type=float, default=0.15)
    parser.add_argument("--control-rewrite-temperature", type=float, default=0.2)
    parser.add_argument("--disable-control-rewrite", action="store_true")
    parser.add_argument("--disable-control-best-effort-rewrite", action="store_true")
    parser.add_argument("--target-arm", default="proposed_contextual_controlled")
    parser.add_argument("--slice-keys", default="persona_archetype,conflict_type,location_type,behavior_state")
    parser.add_argument("--human-eval-file", default="")
    parser.add_argument(
        "--human-eval-metrics",
        default="context_relevance,persona_consistency,naturalness,overall_quality",
    )
    parser.add_argument("--human-eval-scale-max", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--output-root", default="artifacts/proposal")
    parser.add_argument(
        "--keep-batch-runs",
        action="store_true",
        help="Keep intermediate per-batch run directories under _batch_tmp.",
    )
    parser.add_argument(
        "--quality-gate",
        action="store_true",
        help="Run proposal/publication quality-gate validation after merging batches.",
    )
    parser.add_argument(
        "--quality-gate-strict",
        action="store_true",
        help="If set with --quality-gate, exit non-zero when any gate check fails.",
    )
    parser.add_argument(
        "--quality-gate-publication-run",
        default="latest",
        help="Publication run dir or run ID for gate checks; use 'none' to skip publication checks.",
    )
    parser.add_argument(
        "--quality-gate-require-human-eval",
        action="store_true",
        help="Require completed human-eval artifacts in quality gate.",
    )
    parser.add_argument(
        "--quality-gate-require-security-benchmark",
        action="store_true",
        help="Require security benchmark checks in quality gate.",
    )
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenarios_path}")
    all_scenarios = evalmod.read_jsonl(scenarios_path)
    if not all_scenarios:
        raise ValueError("No scenarios found in scenario file.")

    batches = split_batches(all_scenarios, batch_size=int(args.batch_size))
    parent_run_id = evalmod.utc_stamp()
    batch_root = Path(args.output_root) / f"{parent_run_id}_batch_tmp"
    batch_runs_root = batch_root / "runs"
    batch_scenarios_root = batch_root / "scenarios"
    batch_runs_root.mkdir(parents=True, exist_ok=True)
    batch_scenarios_root.mkdir(parents=True, exist_ok=True)

    batch_run_dirs: List[Path] = []
    for batch_idx, batch_rows in enumerate(batches):
        scenario_chunk_path = batch_scenarios_root / f"batch_{batch_idx:03d}.jsonl"
        evalmod.write_jsonl(scenario_chunk_path, batch_rows)
        print(f"[batch {batch_idx + 1}/{len(batches)}] scenarios={len(batch_rows)}")

        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_proposal_alignment_eval.py"),
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--baseline-models",
            str(args.baseline_models),
            "--scenarios",
            str(scenario_chunk_path),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--timeout-s",
            str(args.timeout_s),
            "--bertscore-lang",
            str(args.bertscore_lang),
            "--bertscore-model-type",
            str(args.bertscore_model_type),
            "--bertscore-batch-size",
            str(args.bertscore_batch_size),
            "--bertscore-cache-dir",
            str(args.bertscore_cache_dir),
            "--control-min-context-coverage",
            str(args.control_min_context_coverage),
            "--control-min-persona-coverage",
            str(args.control_min_persona_coverage),
            "--control-rewrite-max-tokens",
            str(args.control_rewrite_max_tokens),
            "--control-rewrite-candidates",
            str(args.control_rewrite_candidates),
            "--control-rewrite-temperature-step",
            str(args.control_rewrite_temperature_step),
            "--control-rewrite-temperature",
            str(args.control_rewrite_temperature),
            "--target-arm",
            str(args.target_arm),
            "--slice-keys",
            str(args.slice_keys),
            "--human-eval-file",
            str(args.human_eval_file),
            "--human-eval-metrics",
            str(args.human_eval_metrics),
            "--human-eval-scale-max",
            str(args.human_eval_scale_max),
            "--output-root",
            str(batch_runs_root),
        ]
        maybe_append(command, "--disable-control-rewrite", bool(args.disable_control_rewrite))
        maybe_append(
            command,
            "--disable-control-best-effort-rewrite",
            bool(args.disable_control_best_effort_rewrite),
        )
        run_subprocess(command)
        run_dir = latest_subdir(batch_runs_root)
        batch_run_dirs.append(run_dir)
        print(f"  -> batch run: {run_dir}")

    first_run = batch_run_dirs[0]
    first_config = json.loads((first_run / "run_config.json").read_text(encoding="utf-8"))
    first_hardware = json.loads((first_run / "metadata" / "hardware.json").read_text(encoding="utf-8"))
    first_models = json.loads((first_run / "metadata" / "models.json").read_text(encoding="utf-8"))
    comparison_plan = json.loads((first_run / "comparison_plan.json").read_text(encoding="utf-8"))

    final_run_id = evalmod.utc_stamp()
    final_run_dir = Path(args.output_root) / final_run_id
    metadata_dir = final_run_dir / "metadata"
    responses_dir = final_run_dir / "responses"
    scores_dir = final_run_dir / "scores"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Merge run config
    merged_config = dict(first_config)
    merged_config["run_id"] = final_run_id
    merged_config["generated_utc"] = evalmod.utc_iso()
    merged_config["scenario_path"] = str(scenarios_path)
    merged_config["scenario_sha256"] = evalmod.sha256_file(scenarios_path)
    # Ensure BERTScore reproducibility knobs are always preserved at merged-run level.
    merged_config["bertscore_lang"] = str(args.bertscore_lang)
    merged_config["bertscore_model_type"] = str(args.bertscore_model_type)
    merged_config["bertscore_batch_size"] = int(args.bertscore_batch_size)
    merged_config["bertscore_cache_dir"] = str(args.bertscore_cache_dir)
    merged_config["batch_mode"] = {
        "enabled": True,
        "batch_size": int(args.batch_size),
        "batch_count": len(batch_run_dirs),
        "batch_run_ids": [p.name for p in batch_run_dirs],
        "intermediate_root": str(batch_root),
    }
    write_json(final_run_dir / "run_config.json", merged_config)
    evalmod.write_jsonl(final_run_dir / "scenarios.jsonl", all_scenarios)
    write_json(metadata_dir / "hardware.json", first_hardware)
    write_json(metadata_dir / "models.json", first_models)

    arms = [
        evalmod.EvaluationArm(
            arm_id=str(item.get("arm_id", "")),
            model=str(item.get("model", "")),
            include_dynamic_context=bool(item.get("include_dynamic_context", False)),
            use_response_control=bool(item.get("use_response_control", False)),
        )
        for item in merged_config.get("arms", [])
    ]
    arm_ids = [a.arm_id for a in arms]
    all_scores: Dict[str, List[Dict[str, Any]]] = {arm_id: [] for arm_id in arm_ids}
    all_responses: Dict[str, List[Dict[str, Any]]] = {arm_id: [] for arm_id in arm_ids}

    for run_dir in batch_run_dirs:
        for arm_id in arm_ids:
            response_path = run_dir / "responses" / f"{arm_id}.jsonl"
            if response_path.exists():
                all_responses[arm_id].extend(evalmod.read_jsonl(response_path))
            score_path = run_dir / "scores" / f"{arm_id}.jsonl"
            if score_path.exists():
                all_scores[arm_id].extend(evalmod.read_jsonl(score_path))

    # Re-number request index per arm for cleanliness.
    for arm_id in arm_ids:
        rows = all_responses.get(arm_id, [])
        for idx, row in enumerate(rows):
            row["request_index"] = idx + 1
        evalmod.write_jsonl(responses_dir / f"{arm_id}.jsonl", rows)

    # Ensure overall_quality is present for all scored rows.
    for arm_id in arm_ids:
        for row in all_scores.get(arm_id, []):
            if "overall_quality" not in row:
                row["overall_quality"] = evalmod.row_overall_quality(row)
        evalmod.write_jsonl(scores_dir / f"{arm_id}.jsonl", all_scores.get(arm_id, []))

    include_bertscore = any(
        "bertscore_f1" in row for arm_id in arm_ids for row in all_scores.get(arm_id, [])
    )
    summary: Dict[str, Any] = {}
    for arm_idx, arm_id in enumerate(arm_ids):
        summary[arm_id] = evalmod.summarize_arm_scores(
            scored_rows=all_scores.get(arm_id, []),
            seed=int(args.seed) + 401 * (arm_idx + 1),
            include_bertscore=include_bertscore,
        )
    write_json(final_run_dir / "summary.json", summary)

    metric_list = comparison_plan.get("metrics", evalmod.metric_names(include_bertscore=include_bertscore))
    comparisons = comparison_plan.get("comparisons", [])
    deltas: Dict[str, Any] = {}
    paired_deltas: Dict[str, Any] = {}
    win_rates: Dict[str, Any] = {}
    for comp_idx, comp in enumerate(comparisons):
        comp_name = str(comp.get("comparison_id", ""))
        target_arm = str(comp.get("target_arm", ""))
        baseline_arm = str(comp.get("baseline_arm", ""))
        if not comp_name:
            continue
        deltas[comp_name] = evalmod.diff_against_reference(
            summary,
            target_arm,
            baseline_arm,
            metrics=metric_list,
        )
        paired_deltas[comp_name] = evalmod.paired_metric_deltas(
            all_scores,
            target_arm,
            baseline_arm,
            seed=int(args.seed) + 5001 + 997 * (comp_idx + 1),
            metrics=metric_list,
        )
        win_rates[comp_name] = evalmod.paired_metric_win_rates(
            all_scores,
            target_arm,
            baseline_arm,
            seed=int(args.seed) + 8001 + 991 * (comp_idx + 1),
            metrics=metric_list,
        )

    write_json(final_run_dir / "delta_vs_baselines.json", deltas)
    write_json(final_run_dir / "paired_delta_significance.json", paired_deltas)
    write_json(final_run_dir / "win_rates.json", win_rates)
    write_json(final_run_dir / "comparison_plan.json", comparison_plan)

    scenarios_by_id = {str(row.get("scenario_id", "")): row for row in all_scenarios}
    slice_summary = evalmod.summarize_scores_by_slice(
        scored_by_arm=all_scores,
        scenarios_by_id=scenarios_by_id,
        slice_keys=evalmod.parse_list_arg(str(args.slice_keys)),
        seed=int(args.seed) + 12001,
        include_bertscore=include_bertscore,
    )
    write_json(final_run_dir / "slice_summary.json", slice_summary)

    error_analysis: Dict[str, Any] = {}
    for arm_id in arm_ids:
        error_analysis[arm_id] = evalmod.analyze_errors(all_scores.get(arm_id, []))
    write_json(final_run_dir / "error_analysis.json", error_analysis)

    human_eval_summary = None
    human_eval_file = str(args.human_eval_file).strip()
    if human_eval_file:
        human_path = Path(human_eval_file)
        if not human_path.exists():
            raise FileNotFoundError(f"Human eval file not found: {human_path}")
        human_rows = evalmod.read_human_eval_rows(human_path)
        baseline_arm_ids = list(comparison_plan.get("baseline_arm_ids", []))
        target_arm = str(comparison_plan.get("target_arm_for_external", args.target_arm))
        human_eval_summary = evalmod.analyze_human_eval(
            rows=human_rows,
            metrics=evalmod.parse_list_arg(str(args.human_eval_metrics)),
            scale_max=float(args.human_eval_scale_max),
            target_arm=target_arm,
            baseline_arms=baseline_arm_ids,
            seed=int(args.seed) + 14001,
        )
        write_json(final_run_dir / "human_eval_summary.json", human_eval_summary)
        evalmod.render_human_eval_report(final_run_dir / "human_eval_report.md", human_eval_summary)

    batch_manifest = {
        "final_run_id": final_run_id,
        "source_scenario_file": str(scenarios_path),
        "scenario_count": len(all_scenarios),
        "batch_size": int(args.batch_size),
        "batch_count": len(batch_run_dirs),
        "batch_runs": [str(p) for p in batch_run_dirs],
    }
    write_json(final_run_dir / "batch_manifest.json", batch_manifest)

    bertscore_meta = (
        {"available": True, "lang": str(args.bertscore_lang)}
        if include_bertscore
        else {"available": False, "reason": "No BERTScore values found in merged scores."}
    )
    evalmod.render_report(
        output_path=final_run_dir / "report.md",
        run_id=final_run_id,
        scenarios_path=final_run_dir / "scenarios.jsonl",
        arms=arms,
        summary=summary,
        deltas=deltas,
        paired_deltas=paired_deltas,
        bertscore_meta=bertscore_meta,
        win_rates=win_rates,
        slice_summary=slice_summary,
        human_eval=human_eval_summary,
    )

    if args.quality_gate:
        gate_command = [
            sys.executable,
            str(SCRIPT_DIR / "proposal_quality_gate.py"),
            "--proposal-run",
            str(final_run_dir),
            "--proposal-root",
            str(Path(args.output_root)),
            "--publication-root",
            str(Path("artifacts/publication")),
        ]
        publication_token = str(args.quality_gate_publication_run).strip()
        if publication_token.lower() in {"", "none", "skip"}:
            gate_command.append("--skip-publication")
        else:
            gate_command.extend(["--publication-run", publication_token])
        maybe_append(
            gate_command,
            "--require-human-eval",
            bool(args.quality_gate_require_human_eval),
        )
        maybe_append(
            gate_command,
            "--require-security-benchmark",
            bool(args.quality_gate_require_security_benchmark),
        )
        maybe_append(gate_command, "--strict", bool(args.quality_gate_strict))
        run_subprocess(gate_command)

    if not args.keep_batch_runs and batch_root.exists():
        shutil.rmtree(batch_root, ignore_errors=True)

    print(f"\nPublished merged batched artifact bundle: {final_run_dir}")
    print("Generated files:")
    print(f"  - {final_run_dir / 'summary.json'}")
    print(f"  - {final_run_dir / 'delta_vs_baselines.json'}")
    print(f"  - {final_run_dir / 'paired_delta_significance.json'}")
    print(f"  - {final_run_dir / 'win_rates.json'}")
    print(f"  - {final_run_dir / 'slice_summary.json'}")
    print(f"  - {final_run_dir / 'comparison_plan.json'}")
    print(f"  - {final_run_dir / 'batch_manifest.json'}")
    if human_eval_summary is not None:
        print(f"  - {final_run_dir / 'human_eval_summary.json'}")
        print(f"  - {final_run_dir / 'human_eval_report.md'}")
    print(f"  - {final_run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
