# NPC AI (BD-NSCA)

Behavior-Driven Neuro-Symbolic Cognitive Architecture for context-grounded NPC dialogue.

This repository is organized to support reproducible research and public release of code, benchmarks, and audit artifacts. The structure aligns with the four main components described in the project report:

1. **Runtime module** (inference + control): Python + C++ implementation of the full pipeline (UE5 state extraction, prompt compilation, guarded hybrid retrieval, local LLM inference, post-generation control).
2. **Training scripts & experiment configs**: Training pipelines for SFT/QLoRA and Direct Preference Optimization (DPO) plus environment configuration to reproduce experimental results.
3. **Benchmark suite**: A fixed set of benchmark scenarios (112+ cases) with run manifests, evaluation configuration, and scripts for computing quality metrics, bootstrap confidence intervals, and statistical tests.
4. **Audit artifact package**: Metadata and raw results from evaluation runs, plus step-by-step reproduction instructions to independently verify the published numeric results.

> ✅ **Minimum reproducibility standard:** A full independent verification run must reproduce the same scenario set, decoding configuration, and statistical analysis pipeline on equivalent hardware.

## Scope
- `cpp/`: low-latency runtime used by UE5/native serving.
- `core/` and `scripts/`: model iteration, evaluation, and benchmark pipelines.
- `docs/`: proposal alignment, benchmark evidence, architecture, and verification.

## Repository structure (high-level)

- `core/` — Python runtime pipeline, prompt builder, retrieval system, response controller, evaluation helpers.
- `core/car_retriever.py` — Contrastive Adversarial Retriever (CAR) dense retrieval utilities.
- `cpp/` — Low-latency runtime used by UE5/native serving (C++ inference engine, build scripts, tests).
- `scripts/` — Training, evaluation, benchmarking, and result aggregation scripts (SFT/QLoRA/DPO pipelines, benchmark runners, metric calculators).
- `data/` — Scenario definitions, prompt templates, and other dataset artifacts used for benchmarks and evaluation.
- `storage/artifacts/` — Output of benchmark/evaluation runs (audit artifact package).
- `gspe/` — Game-State Prefix Encoder module (data prep, training, inference, tests).
- `docs/` — Documentation, architecture diagrams, publications, and reproduction guides.

---

## Getting started (reproducibility)

1. **Read this README first** — it contains the recommended workflow for reproducing results.
2. **Install dependencies**:
   - Python: use the `annotation_pipeline/requirements.txt` (or your preferred environment manager) to install required Python packages.
   - C++: build the runtime from `cpp/` using CMake (see below).
3. **Run the benchmark pipeline**:
   - Use `scripts/run_kaggle_full_results.py` (or relevant `run_*` script) to run the unified benchmark suite and generate audit artifacts.
4. **Verify results**:
   - Confirm that the output matches the published artifacts in `storage/artifacts/` and the reported statistics in `docs/`.

### Reproducibility checklist

- [ ] Use the same **scenario set** (112 fixed benchmark scenarios) from `data/` and `scripts/benchmark_definitions.py`.
- [ ] Use the same **decoding config** (temperature, top-k, top-p, max tokens) used in the published runs (see `scripts/` configs and `docs/` descriptions).
- [ ] Run the **same statistical analysis pipeline** (bootstrap CI and significance tests) via `scripts/evaluate_benchmarks.py` (or the unified `run_kaggle_full_results.py` workflow).
- [ ] Validate against the published **audit artifacts** in `storage/artifacts/` and the results in `docs/`.

---

## Build (C++)
```powershell
cd cpp
cmake -B build
cmake --build build --config Release
```

---

## Quick run (full results checkout)
Run the unified pipeline (proposal + publication + quality gate + comparison artifacts):
```powershell
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```

Ablation option (skip keyword/random retrieval ablation baselines):
```powershell
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434 --skip-ablation-baselines
```

Scientific readiness audit (retrieval size + operational schema + novelty deltas):
```powershell
python scripts/run_scientific_readiness_audit.py
```

Scientific readiness audit with persisted operational backfill (for older runs missing p90/p95 fields):
```powershell
python scripts/run_scientific_readiness_audit.py --persist-operational-backfill
```

State-conditioned novelty benchmark:
```powershell
cpp/build/Release/bench_state_conditioned_novelty.exe --output storage/artifacts/benchmarks/state_conditioned_novelty.json
```

NPC-adversarial corpus protocol + ratio ablation split generation:
```powershell
python scripts/run_npc_adversarial_protocol.py --retrieval-gold data/retrieval_gold.jsonl --retrieval-corpus data/retrieval_corpus.jsonl
```

Harness-vs-UE5 parity verification protocol:
```powershell
python scripts/run_parity_verification_protocol.py --harness-jsonl <harness_responses.jsonl> --ue5-jsonl <ue5_responses.jsonl>
```

Enforce storage layout (`storage/artifacts/`, `outputs/`, `releases/`, `runs/` routed to `storage/*`):
```powershell
python scripts/enforce_storage_layout.py --apply --create-links
```

Local laptop LLM pack check (auto-suggest safe baseline set from installed Ollama models):
```powershell
python scripts/check_local_llm_pack.py --profile laptop_safe
```

Laptop full run with profile-expanded baselines and auto-pruning of missing baseline tags:
```powershell
python scripts/run_laptop_full_results.py --baseline-profile laptop_safe --allow-missing-baselines
```

Local inference model-matrix (same scenario set, multiple local models):
```powershell
python scripts/run_local_model_matrix.py --profile laptop_safe --max-scenarios 40
```

Adversarial retrieval ratio-split evaluation:
```powershell
python scripts/eval_retrieval_protocol_splits.py --protocol-summary storage/artifacts/datasets/retrieval_protocol/summary.json
```

Player study workflow:
```powershell
python scripts/run_player_study.py init --participants 20 --out-dir storage/artifacts/player_study
python scripts/run_player_study.py report --telemetry storage/artifacts/player_study/telemetry_sample.csv --questionnaire storage/artifacts/player_study/questionnaire_sample.csv --out-dir storage/artifacts/player_study
```

Parity campaign (multi-run parity gate):
```powershell
python scripts/run_parity_campaign.py --mapping-json <parity_mapping.json> --require-min-pass-rate 0.98
```

Preference replay update (weak pairs from scored runs + implicit feedback):
```powershell
python scripts/run_preference_replay_update.py --scores-glob "storage/artifacts/proposal/*/scores/*.jsonl"
```

SAGE BT setup helper + transition matrix build:
```powershell
python scripts/BT_SETUP_GUIDE.py --print-guide
python scripts/BT_SETUP_GUIDE.py --build-transition-matrix --input data/benchmark_scenarios.jsonl
```

Optional local SAGE BT HTTP bridge endpoints:
```powershell
python scripts/sage_bt_http_server.py --host 127.0.0.1 --port 8000
```

---

## Canonical documentation
- [Documentation Index](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Draft Paper](docs/DRAFT_PAPER.md)
- [Proposal Alignment](docs/PROPOSAL_ALIGNMENT.md)
- [Proposal Quality Bar](docs/PROPOSAL_QUALITY_BAR.md)
- [Industry Judgment](docs/INDUSTRY_PUBLICATION_JUDGMENT.md)
- [Final Benchmark Report](docs/FINAL_BENCHMARK_REPORT.md)
- [Kaggle Guide](docs/KAGGLE_GUIDE.md)
- [GSPE Module](gspe/README.md)
- [Scientific Novelty Roadmap (2026-03-24)](docs/SCIENTIFIC_NOVELTY_ROADMAP_20260324.md)

---

## Notes
- Publication/benchmark claims are grounded in `storage/artifacts/` runs referenced from `docs/`.
- Older planning notes and superseded deployment docs are kept under:
  - `docs/archive/legacy_20260225/`
  - `docs/archive/legacy_20260303/`


