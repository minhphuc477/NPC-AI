#!/usr/bin/env python3
"""Validate whether proposal/publication artifacts meet a strict quality bar."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def bool_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


@dataclass
class GateCheck:
    check_id: str
    passed: bool
    details: str
    section: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "check_id": self.check_id,
            "passed": bool(self.passed),
            "details": self.details,
        }


def add_file_existence_checks(section: str, root: Path, rel_paths: Sequence[str]) -> List[GateCheck]:
    out: List[GateCheck] = []
    for rel in rel_paths:
        path = root / rel
        out.append(
            GateCheck(
                check_id=f"file:{rel}",
                passed=path.exists(),
                details=str(path),
                section=section,
            )
        )
    return out


def check_bootstrap_payload(payload: Dict[str, Any]) -> bool:
    required = ("n", "mean", "ci95_low", "ci95_high")
    for key in required:
        if key not in payload:
            return False
    n = int(payload.get("n", 0) or 0)
    mean = maybe_float(payload.get("mean"))
    low = maybe_float(payload.get("ci95_low"))
    high = maybe_float(payload.get("ci95_high"))
    if n <= 0:
        return False
    if any(math.isnan(x) for x in (mean, low, high)):
        return False
    return True


def resolve_run_path(raw: str, default_root: Path) -> Path:
    token = str(raw or "").strip()
    if token.lower() == "latest":
        return latest_run_dir(default_root)
    path = Path(token)
    if path.exists():
        return path
    # If caller passed only a run ID, resolve under root.
    candidate = default_root / token
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve run path: {raw}")


def find_comparison_id(
    comparison_plan: Dict[str, Any],
    paired_deltas: Dict[str, Any],
    target_arm: str,
    baseline_arm: str,
) -> Optional[str]:
    comparisons = comparison_plan.get("comparisons", [])
    for row in comparisons:
        if str(row.get("target_arm", "")) == target_arm and str(row.get("baseline_arm", "")) == baseline_arm:
            cid = str(row.get("comparison_id", "")).strip()
            if cid and cid in paired_deltas:
                return cid

    fallback = f"{target_arm}_vs_{baseline_arm}"
    if fallback in paired_deltas:
        return fallback
    return None


def significant_positive_count(comp_payload: Dict[str, Any], alpha: float) -> int:
    count = 0
    for metric, vals in comp_payload.items():
        if not isinstance(vals, dict):
            continue
        mean_delta = maybe_float(vals.get("mean_delta"))
        p_val = maybe_float(vals.get("p_delta_le_0"))
        if mean_delta > 0.0 and not math.isnan(p_val) and p_val <= alpha:
            count += 1
    return count


def evaluate_proposal_run(
    run_dir: Path,
    require_human_eval: bool,
    min_scenarios: int,
    min_external_significant_wins: int,
    alpha: float,
    min_human_rows: int,
    min_human_kappa: float,
    min_human_soft_win_rate: float,
) -> List[GateCheck]:
    checks: List[GateCheck] = []
    section = "proposal"

    required_files = [
        "run_config.json",
        "comparison_plan.json",
        "summary.json",
        "paired_delta_significance.json",
        "win_rates.json",
        "slice_summary.json",
        "report.md",
        "scenarios.jsonl",
        "metadata/hardware.json",
        "metadata/models.json",
    ]
    checks.extend(add_file_existence_checks(section, run_dir, required_files))

    run_config = read_json(run_dir / "run_config.json")
    comparison_plan = read_json(run_dir / "comparison_plan.json")
    paired = read_json(run_dir / "paired_delta_significance.json")

    scenario_count = count_jsonl_rows(run_dir / "scenarios.jsonl")
    checks.append(
        GateCheck(
            check_id="scenario_coverage",
            passed=scenario_count >= int(min_scenarios),
            details=f"rows={scenario_count}, min_required={int(min_scenarios)}",
            section=section,
        )
    )

    # Reproducibility controls for BERTScore.
    bert_model = str(run_config.get("bertscore_model_type", "")).strip()
    bert_batch = int(run_config.get("bertscore_batch_size", 0) or 0)
    has_bertscore_config = bool(bert_model) and bert_batch > 0
    checks.append(
        GateCheck(
            check_id="bertscore_repro_config",
            passed=has_bertscore_config,
            details=f"model_type='{bert_model or 'missing'}', batch_size={bert_batch}",
            section=section,
        )
    )

    # Must beat raw contextual arm on key proposal metrics.
    target_arm = str(comparison_plan.get("target_arm_for_external", "proposed_contextual_controlled"))
    raw_arm = "proposed_contextual"
    ctrl_vs_raw_id = find_comparison_id(comparison_plan, paired, target_arm=target_arm, baseline_arm=raw_arm)
    required_metrics = ("context_relevance", "persona_consistency", "naturalness", "overall_quality")
    ctrl_payload = paired.get(ctrl_vs_raw_id, {}) if ctrl_vs_raw_id else {}
    for metric in required_metrics:
        m = ctrl_payload.get(metric, {}) if isinstance(ctrl_payload, dict) else {}
        mean_delta = maybe_float(m.get("mean_delta"))
        p_val = maybe_float(m.get("p_delta_le_0"))
        passed = mean_delta > 0.0 and (not math.isnan(p_val)) and p_val <= alpha
        checks.append(
            GateCheck(
                check_id=f"controlled_vs_raw:{metric}",
                passed=passed,
                details=(
                    f"comparison={ctrl_vs_raw_id or 'missing'}, mean_delta={mean_delta:.6f}, "
                    f"p_delta_le_0={p_val:.6f}, alpha={alpha}"
                ),
                section=section,
            )
        )

    # Must show broad external-baseline wins for the controlled arm.
    baseline_arm_ids = [str(x) for x in comparison_plan.get("baseline_arm_ids", [])]
    if not baseline_arm_ids:
        baseline_arm_ids = ["baseline_no_context"]
    for baseline_arm in baseline_arm_ids:
        comp_id = find_comparison_id(comparison_plan, paired, target_arm=target_arm, baseline_arm=baseline_arm)
        payload = paired.get(comp_id, {}) if comp_id else {}
        sig_pos = significant_positive_count(payload, alpha=alpha) if isinstance(payload, dict) else 0
        checks.append(
            GateCheck(
                check_id=f"external_wins:{baseline_arm}",
                passed=sig_pos >= int(min_external_significant_wins),
                details=(
                    f"comparison={comp_id or 'missing'}, significant_positive={sig_pos}, "
                    f"min_required={int(min_external_significant_wins)}"
                ),
                section=section,
            )
        )

        oq = payload.get("overall_quality", {}) if isinstance(payload, dict) else {}
        oq_delta = maybe_float(oq.get("mean_delta"))
        oq_p = maybe_float(oq.get("p_delta_le_0"))
        checks.append(
            GateCheck(
                check_id=f"external_overall_quality:{baseline_arm}",
                passed=oq_delta > 0.0 and (not math.isnan(oq_p)) and oq_p <= alpha,
                details=(
                    f"comparison={comp_id or 'missing'}, overall_quality_delta={oq_delta:.6f}, "
                    f"p_delta_le_0={oq_p:.6f}, alpha={alpha}"
                ),
                section=section,
            )
        )

    human_eval_path = run_dir / "human_eval_summary.json"
    if require_human_eval:
        checks.append(
            GateCheck(
                check_id="human_eval_present",
                passed=human_eval_path.exists(),
                details=str(human_eval_path),
                section=section,
            )
        )
        if human_eval_path.exists():
            human_eval = read_json(human_eval_path)
            row_count = int(human_eval.get("row_count", 0) or 0)
            checks.append(
                GateCheck(
                    check_id="human_eval_row_count",
                    passed=row_count >= int(min_human_rows),
                    details=f"rows={row_count}, min_required={int(min_human_rows)}",
                    section=section,
                )
            )

            metrics = [str(m) for m in human_eval.get("metrics", [])]
            agreement = human_eval.get("agreement", {})
            kappas: List[float] = []
            pair_counts: List[int] = []
            for metric in metrics:
                row = agreement.get(metric, {})
                pair_count = int(row.get("pair_count", 0) or 0)
                kappa = maybe_float(row.get("mean_pairwise_kappa"))
                pair_counts.append(pair_count)
                if not math.isnan(kappa):
                    kappas.append(kappa)

            min_pairs = min(pair_counts) if pair_counts else 0
            mean_kappa = statistics.fmean(kappas) if kappas else float("nan")
            checks.append(
                GateCheck(
                    check_id="human_eval_agreement",
                    passed=min_pairs >= 1 and (not math.isnan(mean_kappa)) and mean_kappa >= min_human_kappa,
                    details=(
                        f"min_pair_count={min_pairs}, mean_pairwise_kappa={mean_kappa:.6f}, "
                        f"min_kappa_required={min_human_kappa:.6f}"
                    ),
                    section=section,
                )
            )

            preferences = human_eval.get("preferences", {})
            for baseline_arm in baseline_arm_ids:
                comp_name = f"{target_arm}_vs_{baseline_arm}"
                pref = preferences.get(comp_name, {})
                soft = pref.get("soft_win_rate", {})
                soft_mean = maybe_float(soft.get("mean"))
                checks.append(
                    GateCheck(
                        check_id=f"human_pref_soft_win:{baseline_arm}",
                        passed=(not math.isnan(soft_mean)) and soft_mean >= min_human_soft_win_rate,
                        details=(
                            f"comparison={comp_name}, soft_win_rate={soft_mean:.6f}, "
                            f"min_required={min_human_soft_win_rate:.6f}"
                        ),
                        section=section,
                    )
                )

    return checks


def evaluate_publication_run(
    run_dir: Path,
    alpha: float,
    require_security_benchmark: bool,
    min_asr_reduction: float,
    max_guarded_asr: float,
) -> List[GateCheck]:
    checks: List[GateCheck] = []
    section = "publication"

    required_files = [
        "run_config.json",
        "metadata/hardware.json",
        "metadata/models.json",
        "serving/summary.json",
        "serving/delta_vs_baseline.json",
        "retrieval/metrics.json",
        "retrieval/ablation_deltas_vs_bm25.json",
        "report.md",
    ]
    checks.extend(add_file_existence_checks(section, run_dir, required_files))

    run_config = read_json(run_dir / "run_config.json")
    serving_summary = read_json(run_dir / "serving/summary.json")
    serving_delta = read_json(run_dir / "serving/delta_vs_baseline.json")
    retrieval_metrics = read_json(run_dir / "retrieval/metrics.json")
    retrieval_deltas = read_json(run_dir / "retrieval/ablation_deltas_vs_bm25.json")

    candidate_model = str(run_config.get("candidate_model", "")).strip()
    baseline_model = str(run_config.get("baseline_model", "")).strip()
    checks.append(
        GateCheck(
            check_id="serving_models_declared",
            passed=bool(candidate_model) and bool(baseline_model),
            details=f"candidate='{candidate_model or 'missing'}', baseline='{baseline_model or 'missing'}'",
            section=section,
        )
    )

    inputs = run_config.get("inputs", {})
    prompts_path = str(inputs.get("prompts", "")).strip()
    refs_path = str(inputs.get("serving_references", "")).strip()
    retrieval_gold = str(inputs.get("retrieval_gold", "")).strip()
    retrieval_corpus = str(inputs.get("retrieval_corpus", "")).strip()
    checks.append(
        GateCheck(
            check_id="identical_inputs_declared",
            passed=bool(prompts_path) and bool(retrieval_gold) and bool(retrieval_corpus),
            details=(
                f"prompts='{prompts_path or 'missing'}', references='{refs_path or 'missing'}', "
                f"retrieval_gold='{retrieval_gold or 'missing'}', retrieval_corpus='{retrieval_corpus or 'missing'}'"
            ),
            section=section,
        )
    )

    # Serving confidence intervals must exist for both models.
    serving_ok = True
    for model_name in (candidate_model, baseline_model):
        row = serving_summary.get(model_name, {})
        for metric in ("ttft_ms", "total_time_ms", "tokens_per_s"):
            payload = row.get(metric, {})
            serving_ok = serving_ok and isinstance(payload, dict) and check_bootstrap_payload(payload)
    checks.append(
        GateCheck(
            check_id="serving_confidence_intervals",
            passed=serving_ok,
            details="ttft_ms/total_time_ms/tokens_per_s each include mean + 95% CI for candidate and baseline",
            section=section,
        )
    )

    checks.append(
        GateCheck(
            check_id="serving_delta_present",
            passed=bool(serving_delta),
            details=f"delta_metrics={list(serving_delta.keys())}",
            section=section,
        )
    )

    # Retrieval standards check (Hit@k, MRR, nDCG@k on labeled set).
    bm25 = retrieval_metrics.get("bm25", {})
    bm25_hit_key = next((k for k in bm25.keys() if k.lower().startswith("hit@")), "")
    bm25_ndcg_key = next((k for k in bm25.keys() if k.lower().startswith("ndcg@")), "")
    hit_ok = check_bootstrap_payload(bm25.get(bm25_hit_key, {})) if bm25_hit_key else False
    mrr_ok = check_bootstrap_payload(bm25.get("mrr", {}))
    ndcg_ok = check_bootstrap_payload(bm25.get(bm25_ndcg_key, {})) if bm25_ndcg_key else False
    checks.append(
        GateCheck(
            check_id="retrieval_metrics_standardized",
            passed=hit_ok and mrr_ok and ndcg_ok,
            details=f"bm25_keys={list(bm25.keys())}",
            section=section,
        )
    )

    checks.append(
        GateCheck(
            check_id="retrieval_ablation_deltas_present",
            passed=isinstance(retrieval_deltas, dict) and len(retrieval_deltas) > 0,
            details=f"ablation_methods={list(retrieval_deltas.keys())}",
            section=section,
        )
    )

    # Verify both models were evaluated on identical prompt IDs.
    serving_dir = run_dir / "serving"
    req_files = sorted(serving_dir.glob("requests_*.jsonl"))
    req_ok = False
    if len(req_files) >= 2:
        prompt_sets: List[set[str]] = []
        for path in req_files:
            prompt_ids: set[str] = set()
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    prompt_ids.add(str(row.get("prompt_id", "")))
            prompt_sets.append(prompt_ids)
        req_ok = all(s == prompt_sets[0] for s in prompt_sets[1:]) and len(prompt_sets[0]) > 0
    checks.append(
        GateCheck(
            check_id="serving_prompt_parity",
            passed=req_ok,
            details=f"request_files={len(req_files)}",
            section=section,
        )
    )

    if require_security_benchmark:
        security_path = run_dir / "retrieval" / "security_guard_benchmark_spoofed.json"
        if not security_path.exists():
            security_path = run_dir / "retrieval" / "security_guard_benchmark.json"

        checks.append(
            GateCheck(
                check_id="security_benchmark_present",
                passed=security_path.exists(),
                details=str(security_path),
                section=section,
            )
        )
        if security_path.exists():
            security = read_json(security_path)
            baseline_asr = maybe_float(security.get("baseline_attack_success_rate"))
            guarded_asr = maybe_float(security.get("guarded_attack_success_rate"))
            asr_reduction = maybe_float(security.get("relative_asr_reduction"))
            checks.append(
                GateCheck(
                    check_id="security_asr_reduction",
                    passed=(
                        not any(math.isnan(x) for x in (baseline_asr, guarded_asr, asr_reduction))
                        and asr_reduction >= min_asr_reduction
                        and guarded_asr <= max_guarded_asr
                        and guarded_asr <= baseline_asr
                    ),
                    details=(
                        f"baseline_asr={baseline_asr:.6f}, guarded_asr={guarded_asr:.6f}, "
                        f"relative_reduction={asr_reduction:.6f}, "
                        f"min_reduction={min_asr_reduction:.6f}, max_guarded_asr={max_guarded_asr:.6f}, alpha={alpha}"
                    ),
                    section=section,
                )
            )

    return checks


def render_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Proposal Quality Gate Report")
    lines.append("")
    lines.append(f"- Generated UTC: `{payload.get('generated_utc', '')}`")
    lines.append(f"- Overall pass: `{payload.get('overall_pass', False)}`")
    lines.append(f"- Proposal run: `{payload.get('proposal_run', '')}`")
    lines.append(f"- Publication run: `{payload.get('publication_run', '')}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("| Section | Check | Result | Details |")
    lines.append("|---|---|---|---|")
    for row in payload.get("checks", []):
        result = "PASS" if bool(row.get("passed")) else "FAIL"
        lines.append(
            f"| {row.get('section', '')} | {row.get('check_id', '')} | {result} | "
            f"{str(row.get('details', '')).replace('|', '/')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_failures(checks: Iterable[GateCheck]) -> List[str]:
    return [f"[{c.section}] {c.check_id}: {c.details}" for c in checks if not c.passed]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proposal/publication artifacts against quality bar checks.")
    parser.add_argument("--proposal-run", default="latest", help="Proposal run dir or run ID, or 'latest'")
    parser.add_argument("--publication-run", default="latest", help="Publication run dir or run ID, or 'latest'")
    parser.add_argument("--proposal-root", default="artifacts/proposal", help="Proposal artifacts root")
    parser.add_argument("--publication-root", default="artifacts/publication", help="Publication artifacts root")
    parser.add_argument("--skip-publication", action="store_true", help="Skip publication artifact checks")
    parser.add_argument("--require-human-eval", action="store_true", help="Require completed multi-rater human eval")
    parser.add_argument("--require-security-benchmark", action="store_true", help="Require security benchmark checks")
    parser.add_argument("--min-scenarios", type=int, default=100, help="Minimum scenario rows in proposal run")
    parser.add_argument(
        "--min-external-significant-wins",
        type=int,
        default=10,
        help="Minimum count of significantly-positive metrics vs each external baseline",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--min-human-rows", type=int, default=300, help="Minimum normalized human-eval rows")
    parser.add_argument("--min-human-kappa", type=float, default=0.20, help="Minimum mean pairwise kappa")
    parser.add_argument(
        "--min-human-soft-win-rate",
        type=float,
        default=0.55,
        help="Minimum soft win rate vs each baseline in human preferences",
    )
    parser.add_argument(
        "--min-asr-reduction",
        type=float,
        default=0.80,
        help="Minimum required relative ASR reduction for security benchmark",
    )
    parser.add_argument(
        "--max-guarded-asr",
        type=float,
        default=0.05,
        help="Maximum allowed guarded ASR in security benchmark",
    )
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    parser.add_argument("--output-md", default="", help="Optional output markdown path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails")
    args = parser.parse_args()

    proposal_root = Path(args.proposal_root)
    publication_root = Path(args.publication_root)
    proposal_run = resolve_run_path(str(args.proposal_run), proposal_root)
    publication_run = None if args.skip_publication else resolve_run_path(str(args.publication_run), publication_root)

    proposal_checks = evaluate_proposal_run(
        run_dir=proposal_run,
        require_human_eval=bool(args.require_human_eval),
        min_scenarios=int(args.min_scenarios),
        min_external_significant_wins=int(args.min_external_significant_wins),
        alpha=float(args.alpha),
        min_human_rows=int(args.min_human_rows),
        min_human_kappa=float(args.min_human_kappa),
        min_human_soft_win_rate=float(args.min_human_soft_win_rate),
    )
    all_checks = list(proposal_checks)

    if publication_run is not None:
        publication_checks = evaluate_publication_run(
            run_dir=publication_run,
            alpha=float(args.alpha),
            require_security_benchmark=bool(args.require_security_benchmark),
            min_asr_reduction=float(args.min_asr_reduction),
            max_guarded_asr=float(args.max_guarded_asr),
        )
        all_checks.extend(publication_checks)

    overall_pass = all(c.passed for c in all_checks)
    payload = {
        "generated_utc": utc_iso(),
        "overall_pass": bool(overall_pass),
        "proposal_run": str(proposal_run),
        "publication_run": str(publication_run) if publication_run is not None else "",
        "thresholds": {
            "alpha": float(args.alpha),
            "min_scenarios": int(args.min_scenarios),
            "min_external_significant_wins": int(args.min_external_significant_wins),
            "require_human_eval": bool(args.require_human_eval),
            "min_human_rows": int(args.min_human_rows),
            "min_human_kappa": float(args.min_human_kappa),
            "min_human_soft_win_rate": float(args.min_human_soft_win_rate),
            "require_security_benchmark": bool(args.require_security_benchmark),
            "min_asr_reduction": float(args.min_asr_reduction),
            "max_guarded_asr": float(args.max_guarded_asr),
        },
        "checks": [c.as_dict() for c in all_checks],
        "failures": summarize_failures(all_checks),
    }

    output_json = Path(args.output_json) if str(args.output_json).strip() else proposal_run / "quality_gate_report.json"
    output_md = Path(args.output_md) if str(args.output_md).strip() else proposal_run / "quality_gate_report.md"
    write_json(output_json, payload)
    render_markdown(output_md, payload)

    print(f"Quality gate: {bool_text(overall_pass)}")
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {output_md}")
    if payload["failures"]:
        print("Failed checks:")
        for line in payload["failures"]:
            print(f"  - {line}")

    if args.strict and not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
