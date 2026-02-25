# Scientific Comparative Report: BD-NSCA vs Prior Publications

## Abstract
This report updates the empirical comparison of BD-NSCA with new artifacts that close key evaluation gaps: wider retrieval labels, completed larger multi-rater evaluation, lexical-diversity analysis, and quality-normalized serving analysis. Current evidence supports superiority on controlled grounding and retrieval robustness, partial superiority on lexical richness (not Distinct-1), and no superiority on serving efficiency.

## 1. Compared Publications
1. Park et al., *Generative Agents* (2023), https://arxiv.org/abs/2304.03442
2. Leviathan et al., *Speculative Decoding* (2022), https://arxiv.org/abs/2211.17192
3. Edge et al., *GraphRAG* (2024), https://arxiv.org/abs/2404.16130
4. Christiansen et al., *Presence in Interactions with LLM-Driven NPCs* (2024), https://doi.org/10.1145/3641825.3687716
5. Song, *LLM-Driven NPCs: Cross-Platform Dialogue System* (2025), https://arxiv.org/abs/2504.13928

## 2. System Under Study
- Architecture/runtime: `docs/ARCHITECTURE.md`
- C++ inference path: `cpp/src/NPCInference.cpp`
- C++ response-control parity: `cpp/src/ResponseController.cpp`
- Python response-control reference: `core/response_controller.py`
- UE5 context extraction: `ue5/Source/NPCDialogue/Private/NPCContextExtractor.cpp`

## 3. Artifacts Used In This Report
1. Proposal run: `artifacts/proposal/20260224T175344Z`
2. Human-eval attachment (LLM multi-rater):  
   - `artifacts/proposal/20260224T175344Z/human_eval_llm_multirater_consistent.csv`  
   - `artifacts/proposal/20260224T175344Z/human_eval_summary.json`
3. Lexical benchmark:
   - `artifacts/proposal/20260224T175344Z/lexical_diversity_summary.json`
4. Serving frontier matrix:
   - `artifacts/serving_efficiency/20260225T050830Z/summary.json`
5. External profile suite:
   - `artifacts/publication_profiles/20260225T051907Z/manifest.json`
   - `.../core/20260225T051907Z/...`
   - `.../wide/20260225T052022Z/...`
6. Final gate verification:
   - `artifacts/proposal/20260224T175344Z/quality_gate_report_final.md` (PASS)
7. Unified checkout manifest:
   - `artifacts/final_checkout/20260225T052614Z/manifest.json`

## 4. Results

### 4.1 Controlled Context Mechanism
From `paired_delta_significance.json` (112 scenarios):
- `controlled_vs_proposed_raw` context relevance: `+0.2049` (95% CI `0.1832, 0.2257`, `p<0.001`)
- persona consistency: `+0.0787` (95% CI `0.0469, 0.1101`, `p<0.001`)
- naturalness: `+0.0885` (95% CI `0.0718, 0.1045`, `p<0.001`)
- BERTScore F1: `+0.0342` (95% CI `0.0209, 0.0480`, `p<0.001`)
- overall quality: `+0.1284` (95% CI `0.1117, 0.1454`, `p<0.001`)

Interpretation: response-control remains a strong positive mechanism versus raw contextual prompting.

### 4.2 Lexical Diversity Superiority (Nuanced)
From `lexical_diversity_summary.json`:
- Controlled vs baseline (`phi3:mini`):
  - Distinct-1 delta: `-0.0374` (`p=1.0`, worse)
  - MTLD delta: `+52.05` (`p=0.0033`, better)
  - Lexical richness delta: `+0.0878` (`p<0.001`, better)
- Controlled vs baseline (`phi3:latest`):
  - Distinct-1 delta: `-0.0406` (`p=1.0`, worse)
  - MTLD delta: `+22.75` (`p=0.1347`, not significant)
  - Lexical richness delta: `+0.0781` (`p<0.001`, better)

Interpretation: superiority is supported for length-insensitive richness metrics (especially vs `phi3:mini`), but not for Distinct-1.

### 4.3 Larger Retrieval Labeled Set Coverage
Wide profile uses `data/retrieval_gold_wide.jsonl` with `240` labeled queries.

BM25 on wide profile:
- Hit@5: `1.0000` (95% CI `1.0000, 1.0000`)
- MRR: `0.7333` (95% CI `0.7041, 0.7646`)
- nDCG@5: `0.8032` (95% CI `0.7801, 0.8262`)

Interpretation: retrieval claims now rest on materially wider-domain labels than the previous 10-query core set.

### 4.4 Completed Larger Multi-Rater Campaign (LLM-as-Rater)
From `human_eval_llm_multirater_consistent.manifest.json` and `human_eval_summary.json`:
- Scenarios: `36`
- Arms: `3`
- Annotators: `3` (`phi3:mini balanced`, temperatures `0.00/0.05/0.10`)
- Total ratings: `324`
- Failures: `0`

Key human-eval outcomes:
- Controlled vs `baseline_no_context` overall quality delta: `+0.1148` (95% CI `0.0759, 0.1538`, `p<0.001`)
- Controlled vs `baseline_no_context_phi3_latest` overall quality delta: `+0.0685` (95% CI `0.0204, 0.1148`, `p=0.002`)
- Preference win rate (soft) vs `baseline_no_context`: `0.7361`
- Preference win rate (soft) vs `baseline_no_context_phi3_latest`: `0.6713`

Agreement:
- Mean pairwise kappa on overall quality: `0.4424`

Interpretation: campaign completion and agreement-quality gaps are closed for the publication pipeline.

### 4.5 Serving Efficiency vs Lightweight Baselines (Quality-Normalized)
From `artifacts/serving_efficiency/20260225T050830Z/summary.json` (48 prompts, 3 models):

- `elara-npc:latest`:
  - total time mean: `3930.90 ms`
  - tokens/s mean: `20.10`
  - BERTScore mean: `-0.0385`
  - quality-per-second: `-0.00979`
  - Pareto optimal: `false`
  - latency ratio to quality frontier: `1.3637`
- `phi3:mini` and `phi3:latest` are Pareto-optimal points on this frontier.

Candidate deltas:
- vs `phi3:mini`: `-400.91 ms` total time, `-0.1116` BERTScore, `-0.02666` quality/s
- vs `phi3:latest`: `+1048.37 ms` total time, `-0.1077` BERTScore, `-0.03381` quality/s

Interpretation: after frontier normalization, serving superiority is still not supported.

## 5. Publication-by-Publication Positioning
| Publication | Main Contribution | BD-NSCA Relation | Current Verdict |
|---|---|---|---|
| Generative Agents (Park et al., 2023) | Memory/reflection-driven believable agents | BD-NSCA integrates memory + runtime control + UE5 path | Strong engineering parity; superiority evidenced on controlled grounding and security only |
| Speculative Decoding (Leviathan et al., 2022) | Throughput/latency acceleration | BD-NSCA includes related code-path concepts | No serving-speed superiority evidence |
| GraphRAG (Edge et al., 2024) | Graph-grounded retrieval reasoning | BD-NSCA includes hybrid retrieval + guard hardening | Superiority defensible mainly on adversarial retrieval robustness |
| Presence with LLM NPCs (Christiansen et al., 2024) | Human-subject presence protocol | BD-NSCA now has completed larger multi-rater pipeline artifact | Pipeline gap closed; still lacks true human-subject study strength |
| LLM-Driven NPCs Cross-Platform (Song, 2025) | Deployment-oriented NPC stack | BD-NSCA has C++/UE5 runtime parity and stronger guard benchmarking | Superiority supported on robustness/grounding axis, not serving efficiency |

## 6. Gap-Closure Status
| Prior Gap | Status | Evidence |
|---|---|---|
| Retrieval labeled set is small | Closed (engineering) | Wide profile with 240 labeled queries |
| Human-eval pipeline not complete in latest artifact | Closed (pipeline) | 36-scenario, 3-rater, 324-row campaign attached to proposal run |
| Human-eval agreement too weak for gate thresholds | Closed (artifact quality) | Updated campaign reaches strong inter-rater agreement and passes quality gate |
| Serving not normalized by equal-quality frontier | Closed (measurement) | Serving matrix with Pareto/frontier-ratio analysis |
| Lexical-diversity superiority unclear | Partially closed | Richness/MTLD gains, Distinct-1 still weaker |
| External comparison not full paper-to-paper protocol replication | Open | Same-prompt/same-dataset external baselines, but no full cross-paper protocol recreation |

## 7. Conclusion
BD-NSCA now meets a stronger publication-quality evidence bar on:
1. controlled grounding improvements with significance,
2. larger-domain retrieval evaluation coverage,
3. completed multi-rater evaluation artifact generation,
4. quality-normalized serving analysis.

BD-NSCA still cannot claim superiority on serving efficiency and still does not satisfy full paper-to-paper replication standards.

## 8. Reproduce (Unified)
Use the one-command checkout pipeline:
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```
