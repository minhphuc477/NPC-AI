# Documentation Index (Unified Current Set)

## Core Documents
- [ARCHITECTURE.md](ARCHITECTURE.md): architecture source of truth and system diagrams.
- [DRAFT_PAPER.md](DRAFT_PAPER.md): current publication-style paper draft (artifact-backed).
- [PROPOSAL_ALIGNMENT.md](PROPOSAL_ALIGNMENT.md): objective-by-objective proposal satisfaction matrix.
- [INDUSTRY_PUBLICATION_JUDGMENT.md](INDUSTRY_PUBLICATION_JUDGMENT.md): claim-safe industry comparison verdict.
- [comparison_analysis.md](comparison_analysis.md): scientific comparison against selected publications.
- [FINAL_BENCHMARK_REPORT.md](FINAL_BENCHMARK_REPORT.md): latest unified benchmark summary.
- [ABLATION_REPORT.md](ABLATION_REPORT.md): response-control/retrieval/serving ablation view.
- [VERIFICATION.md](VERIFICATION.md): strict verification checklist.
- [PROPOSAL_QUALITY_BAR.md](PROPOSAL_QUALITY_BAR.md): pass/fail gate definitions.
- [ACADEMIC_WRITING_STANDARD.md](ACADEMIC_WRITING_STANDARD.md): writing-standard checklist (IEEE/ACM/NeurIPS style).
- [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md): Kaggle execution and reproducibility flow.
- [proposal.txt](proposal.txt): original project proposal text.

## Repro Commands
- Full unified run:
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```
- Strict gate check:
```bash
python scripts/proposal_quality_gate.py --proposal-run latest --publication-run latest --require-human-eval --require-security-benchmark --strict
```

## Archive
Legacy/outdated docs are kept under:
- `docs/archive/legacy_20260225/`
- `docs/archive/legacy_20260303/`
