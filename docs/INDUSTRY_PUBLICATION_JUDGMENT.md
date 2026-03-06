# Industry and Publication Judgment (Current)

## Scope
Assess whether this project has defensible superiority over public baselines, and where it does not.

## Baselines Referenced
- Generative Agents (Park et al., 2023): https://arxiv.org/abs/2304.03442
- Speculative Decoding (Leviathan et al., 2022): https://arxiv.org/abs/2211.17192
- GraphRAG (Edge et al., 2024): https://arxiv.org/abs/2404.16130
- Presence with LLM NPCs (Christiansen et al., 2024): https://doi.org/10.1145/3641825.3687716
- LLM-Driven NPCs Cross-Platform (Song, 2025): https://arxiv.org/abs/2504.13928

## Evidence Runs
- Proposal: `artifacts/proposal/20260302T182844Z`
- Publication: `artifacts/publication/20260302T191131Z`
- Strict gate: `artifacts/proposal/20260302T182844Z/quality_gate_report.json`

## High-Confidence Strengths
1. **Context-grounded dialogue quality**
- Controlled generation significantly outperforms raw generation and no-context baselines on proposal-critical metrics.
- External baseline checks pass with 10/12 significant-positive metrics for each baseline.

2. **Adversarial retrieval robustness**
- Poisoned and trust-spoofed benchmarks show baseline ASR 1.0 to guarded ASR 0.0.
- Relative ASR reduction is 1.0 with published confidence intervals.

3. **Reproducibility maturity**
- Artifacts include metadata, prompt parity checks, confidence intervals, and strict pass/fail gate outputs.

## Where Superiority Is Not Yet Defensible
1. **Serving efficiency**
- Candidate is slower than `phi3:mini` under identical prompts/settings.

2. **Paper-to-paper replication depth**
- Comparisons are strong benchmark-aligned evaluations, but not full protocol recreations of each target paper.

## Direct Answer: "Better than other publications?"
- **Yes, on specific axes:** context-grounded control and retrieval-attack robustness.
- **No, not globally:** serving speed/throughput superiority is not supported.

## Recommended Claim Discipline
Use these claims:
1. "We show significant improvements in grounded NPC response quality under dynamic context and response control."
2. "We show strong robustness gains against poisoned and trust-spoofed retrieval attacks."
3. "We provide reproducible artifact-backed evaluation with strict quality gates."

Avoid these claims:
1. "State-of-the-art serving efficiency."
2. "Overall superior to all prior publications."
