# Scientific Comparative Analysis: NPC AI vs Prior Publications

## Abstract
This report compares the current project against representative public literature using only artifact-backed evidence from the latest runs. The project is superior on context-grounded response control and retrieval robustness, but not on serving efficiency.

## Compared Works
1. Park et al., *Generative Agents* (2023)
2. Leviathan et al., *Speculative Decoding* (2022)
3. Edge et al., *GraphRAG* (2024)
4. Christiansen et al., *Presence in Interactions with LLM-Driven NPCs* (2024)
5. Song, *LLM-Driven NPCs: Cross-Platform Dialogue System* (2025)

## Evidence Used
- Proposal run: `artifacts/proposal/20260302T182844Z`
- Publication run: `artifacts/publication/20260302T191131Z`
- Gate report: `artifacts/proposal/20260302T182844Z/quality_gate_report.json`
- Human eval: `artifacts/proposal/20260302T182844Z/human_eval_summary.json`
- Reranker stage: `artifacts/publication/20260302T191131Z/retrieval/reranker_stage.json`

## Results

### 1. Controlled Grounding Quality
Controlled vs raw (same model):
- Context relevance: +0.2131 (95% CI 0.1962, 0.2285)
- Persona consistency: +0.2150 (95% CI 0.1878, 0.2419)
- Naturalness: +0.1158 (95% CI 0.0964, 0.1337)
- Overall quality: +0.1808 (95% CI 0.1666, 0.1953)
All with `p(delta<=0)=0.0`.

### 2. External Baseline Wins
- Controlled vs `phi3:mini` (no context): 10/12 significant-positive metrics.
- Controlled vs `phi3:latest` (no context): 10/12 significant-positive metrics.
- Overall quality deltas: +0.1672 and +0.1685 (both significant).

### 3. Human Evaluation Strength
- 324 ratings, 3 raters, mean pairwise kappa 0.5329.
- Preference soft win rate:
  - 0.7315 vs `phi3:mini`
  - 0.6806 vs `phi3:latest`

### 4. Retrieval and Reranker Evidence
- Core retrieval benchmark publishes Hit@5/MRR/nDCG with CIs.
- Reranker training/eval stage uses 3360 hard-negative pairs.
- Eval pair accuracy: 0.9563 (95% CI 0.9365, 0.9742).

### 5. Security Robustness
- Poisoned and trust-spoofed stress tests:
  - baseline ASR: 1.0000
  - guarded ASR: 0.0000
  - relative reduction: 1.0000

### 6. Serving Efficiency
Under identical prompt and generation settings (`elara-npc:latest` vs `phi3:mini`):
- TTFT: 542.6 ms vs 273.9 ms
- Total time: 3601.5 ms vs 3138.4 ms
- Tokens/s: 18.45 vs 18.86

Conclusion: no serving-efficiency superiority.

## Publication-by-Publication Positioning
| Publication | What it is known for | Our standing |
|---|---|---|
| Generative Agents (2023) | Agentic memory/reflection behavior | Competitive on grounding pipeline rigor; stronger local-runtime reproducibility artifacts |
| Speculative Decoding (2022) | Speed/throughput gains | Not superior on speed metrics |
| GraphRAG (2024) | Retrieval-grounded generation | Strong on guard robustness and attack resistance; different retrieval objective |
| Presence with LLM NPCs (2024) | Human-subject presence protocol | Stronger artifactized evaluator pipeline; weaker on true user-study depth |
| LLM-Driven NPCs (2025) | Deployment-oriented NPC stack | Stronger adversarial retrieval evidence; serving tradeoff remains |

## Final Judgment
The project can credibly claim superiority in:
1. Context-grounded controlled dialogue quality.
2. Retrieval robustness under adversarial poisoning/spoofing.

It cannot credibly claim superiority in:
1. Serving latency/throughput.
2. Full paper-protocol replication breadth.
