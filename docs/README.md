# Documentation Index

## Unified Current Set
- [ARCHITECTURE.md](ARCHITECTURE.md): system design and runtime/training flow.
- [PROPOSAL_ALIGNMENT.md](PROPOSAL_ALIGNMENT.md): mapping to `proposal.txt`.
- [PROPOSAL_QUALITY_BAR.md](PROPOSAL_QUALITY_BAR.md): publication quality gate criteria.
- [INDUSTRY_PUBLICATION_JUDGMENT.md](INDUSTRY_PUBLICATION_JUDGMENT.md): competitive/publication judgment.
- [comparison_analysis.md](comparison_analysis.md): scientific-style external comparison.
- [FINAL_BENCHMARK_REPORT.md](FINAL_BENCHMARK_REPORT.md): latest benchmark summary.
- [HUMAN_EVAL_PIPELINE.md](HUMAN_EVAL_PIPELINE.md): human/LLM multi-rater process.
- [BENCHMARK_STANDARDS.md](BENCHMARK_STANDARDS.md): benchmark/reporting standards.
- [ABLATION_REPORT.md](ABLATION_REPORT.md): ablation evidence and interpretation.
- [VERIFICATION.md](VERIFICATION.md): executed verification and remaining gaps.
- [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md): Kaggle/local reproducible execution.
- [proposal.txt](proposal.txt): original proposal baseline.

## Reproduction Commands
- Full checkout: `python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434`
- Ablation option:
  `python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434 --skip-ablation-baselines`

## Archive
- Superseded docs were pruned from the top-level set and moved to:
  `docs/archive/legacy_20260225/`
