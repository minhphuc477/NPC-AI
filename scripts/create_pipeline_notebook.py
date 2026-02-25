#!/usr/bin/env python3
"""Synchronize the Kaggle notebook with current proposal/publication pipeline cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_NOTEBOOK_PATH = Path("notebooks/NPC_AI_Complete_Pipeline.ipynb")


def make_markdown_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def make_code_cell(code: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def create_empty_notebook() -> Dict[str, Any]:
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def load_notebook(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return create_empty_notebook()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "cells" not in payload or not isinstance(payload["cells"], list):
        raise ValueError(f"Notebook has invalid structure: {path}")
    return payload


def save_notebook(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def clear_notebook_outputs(payload: Dict[str, Any]) -> None:
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []


def notebook_has_token(payload: Dict[str, Any], token: str) -> bool:
    token = str(token)
    for cell in payload.get("cells", []):
        source = "".join(cell.get("source", []))
        if token in source:
            return True
    return False


def append_proposal_cells(payload: Dict[str, Any]) -> int:
    added = 0
    full_checkout_md = """
## Full Artifact Checkout (Recommended)
Run the complete proposal/publication pipeline and emit a single manifest with all output paths.
"""
    full_checkout_code = r"""
import subprocess
import sys

# Option: set True to skip keyword/random ablation baselines in publication retrieval metrics.
SKIP_ABLATION_BASELINES = False

cmd = [
    sys.executable,
    "scripts/run_kaggle_full_results.py",
    "--host",
    "http://127.0.0.1:11434",
]
if SKIP_ABLATION_BASELINES:
    cmd.append("--skip-ablation-baselines")

print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
"""

    if not notebook_has_token(payload, "run_kaggle_full_results.py"):
        payload["cells"].append(make_markdown_cell(full_checkout_md))
        payload["cells"].append(make_code_cell(full_checkout_code))
        added += 2
    else:
        for idx, cell in enumerate(payload.get("cells", [])):
            source = "".join(cell.get("source", []))
            if "scripts/run_kaggle_full_results.py" not in source:
                continue
            if cell.get("cell_type") == "code":
                payload["cells"][idx] = make_code_cell(full_checkout_code)
                # If the previous cell is markdown, keep it synchronized.
                if idx > 0 and payload["cells"][idx - 1].get("cell_type") == "markdown":
                    payload["cells"][idx - 1] = make_markdown_cell(full_checkout_md)
                break

    if not notebook_has_token(payload, "run_proposal_alignment_eval_batched.py"):
        payload["cells"].append(
            make_markdown_cell(
                """
## Proposal Evaluation (Batched)
Generate expanded scenarios and run proposal evaluation in batches for Kaggle stability.
"""
            )
        )
        payload["cells"].append(
            make_code_cell(
                r"""
import os
import subprocess
import sys

hf_cache = "/kaggle/working/hf_cache" if os.path.exists("/kaggle/working") else os.path.abspath("hf_cache")
os.makedirs(hf_cache, exist_ok=True)

subprocess.check_call(
    [
        sys.executable,
        "scripts/generate_proposal_scenarios_large.py",
        "--variants-per-base",
        "14",
        "--output",
        "data/proposal_eval_scenarios_large.jsonl",
    ]
)

cmd = [
    sys.executable,
    "scripts/run_proposal_alignment_eval_batched.py",
    "--scenarios",
    "data/proposal_eval_scenarios_large.jsonl",
    "--batch-size",
    "28",
    "--repeats",
    "1",
    "--max-tokens",
    "80",
    "--temperature",
    "0.2",
    "--baseline-models",
    "phi3:latest",
    "--bertscore-model-type",
    "roberta-large",
    "--bertscore-batch-size",
    "16",
    "--bertscore-cache-dir",
    hf_cache,
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
"""
            )
        )
        added += 2

    if not notebook_has_token(payload, "build_human_eval_pack.py"):
        payload["cells"].append(
            make_markdown_cell(
                """
## Human Evaluation Pack (Optional)
Build blind multi-rater annotation files from the latest proposal run.
"""
            )
        )
        payload["cells"].append(
            make_code_cell(
                r"""
import pathlib
import subprocess
import sys

proposal_root = pathlib.Path("artifacts/proposal")
run_dirs = sorted([p for p in proposal_root.iterdir() if p.is_dir()]) if proposal_root.exists() else []
if not run_dirs:
    raise RuntimeError("No proposal runs found under artifacts/proposal. Run proposal eval first.")
latest_run = run_dirs[-1]

subprocess.check_call(
    [
        sys.executable,
        "scripts/build_human_eval_pack.py",
        "--run-dir",
        str(latest_run),
        "--annotators",
        "annotator_1,annotator_2,annotator_3",
        "--shared-ratio",
        "0.35",
    ]
)
"""
            )
        )
        added += 2

    if not notebook_has_token(payload, "run_publication_benchmark_suite.py"):
        payload["cells"].append(
            make_markdown_cell(
                """
## Publication Benchmark Suite
Run non-mock benchmark suite with retrieval security checks.
"""
            )
        )
        payload["cells"].append(
            make_code_cell(
                r"""
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/run_publication_benchmark_suite.py",
    "--repeats",
    "1",
    "--max-tokens",
    "64",
    "--temperature",
    "0.2",
    "--run-security-benchmark",
    "--run-security-spoofed-benchmark",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
"""
            )
        )
        added += 2

    if not notebook_has_token(payload, "proposal_quality_gate.py"):
        payload["cells"].append(
            make_markdown_cell(
                """
## Proposal Quality Gate
Evaluate whether latest proposal/publication artifacts satisfy the quality bar.
"""
            )
        )
        payload["cells"].append(
            make_code_cell(
                r"""
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/proposal_quality_gate.py",
    "--proposal-run",
    "latest",
    "--publication-run",
    "latest",
    "--require-security-benchmark",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
"""
            )
        )
        added += 2

    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Kaggle notebook with current evaluation and gate cells.")
    parser.add_argument("--template", default=str(DEFAULT_NOTEBOOK_PATH), help="Notebook template path")
    parser.add_argument("--output", default=str(DEFAULT_NOTEBOOK_PATH), help="Output notebook path")
    parser.add_argument("--clear-outputs", action="store_true", help="Clear execution counts and outputs")
    parser.add_argument(
        "--no-proposal-cells",
        action="store_true",
        help="Skip appending proposal/publication/gate cells",
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.output)

    notebook = load_notebook(template_path)

    appended = 0
    if not args.no_proposal_cells:
        appended = append_proposal_cells(notebook)

    if args.clear_outputs:
        clear_notebook_outputs(notebook)

    save_notebook(output_path, notebook)

    print(f"Notebook updated: {output_path}")
    print(f"Cells appended: {appended}")
    if args.clear_outputs:
        print("Outputs cleared: yes")


if __name__ == "__main__":
    main()
