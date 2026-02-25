# Significant Improvement Playbook

## Scope
This playbook targets **step-level** gains (not incremental tuning) for BD-NSCA on:
- external baseline wins,
- human-eval preference margin,
- lexical richness/naturalness,
- retrieval robustness and transfer,
- quality-at-latency frontier.

## Why Current Gains Plateau
Current pipeline is strong in response control and retrieval guard hardening, but most model updates are still SFT-style. That usually saturates quickly on preference-heavy metrics. To move significantly, the highest-yield path is:
1. preference optimization after SFT,
2. harder retrieval negatives + reranking,
3. runtime serving improvements that shift the latency/quality frontier.

## High-Impact Changes (Prioritized)

### P0: SFT -> Preference Optimization (DPO)
Expected impact:
- large shift on pairwise wins and overall-quality preference metrics,
- better external-baseline wins than prompt-control alone.

Implemented tooling:
- Dataset builder from your multi-rater run:
  - `scripts/build_preference_dataset_from_eval.py`
- Preference trainer:
  - `scripts/train_dpo.py`

Reproduce:
```bash
python scripts/build_preference_dataset_from_eval.py \
  --run-dir artifacts/proposal/20260224T175344Z \
  --human-eval-file artifacts/proposal/20260224T175344Z/human_eval_llm_multirater_consistent.csv \
  --target-arm proposed_contextual_controlled \
  --baseline-arms baseline_no_context,baseline_no_context_phi3_latest \
  --metric overall_quality \
  --min-raters 2 \
  --min-margin 0.25

python scripts/train_dpo.py \
  --dataset artifacts/proposal/20260224T175344Z/preference_dataset.jsonl \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --output-dir outputs/dpo_adapter
```

Current artifact produced:
- `artifacts/proposal/20260224T175344Z/preference_dataset.jsonl` (51 rows)

### P1: Retrieval Hard-Negative Mining + Reranker Training
Expected impact:
- stronger retrieval ranking under harder near-miss distractors,
- better robustness transfer to wider-domain queries and poison-like confusions.

Implemented tooling:
- Hard-negative and pairwise reranker set builder:
  - `scripts/build_retrieval_hard_negative_set.py`

Reproduce:
```bash
python scripts/build_retrieval_hard_negative_set.py \
  --retrieval-gold data/retrieval_gold_wide.jsonl \
  --retrieval-corpus data/retrieval_corpus_wide.jsonl \
  --hard-negatives-per-query 10 \
  --cross-domain-negatives-per-query 4
```

Current artifact produced:
- `data/retrieval_hard_negatives_wide.jsonl` (240 queries)
- `data/retrieval_reranker_pairs_wide.jsonl` (3360 pairs)

### P2: Serving Frontier Shift (Not Just Faster Decoding)
Expected impact:
- measurable movement on quality-per-second and Pareto optimality.

Required engineering:
- speculative decoding in serving path,
- or model distillation to a smaller draft/policy model pair,
- benchmark with the existing `scripts/run_serving_efficiency_matrix.py`.

## Evaluation Contract (to prove significant gain)
For each P0/P1/P2 iteration, require all of:
1. Proposal paired deltas improve on `overall_quality` and `context_relevance`.
2. Human-eval soft win-rate >= prior artifact on both external baselines.
3. Retrieval: no regression on wide-label MRR/nDCG, and improved stress behavior.
4. Serving matrix: improved candidate `quality_per_second` and lower frontier ratio.

## Suggested 2-Week Experiment Matrix
1. `SFT -> DPO` only.
2. `SFT -> DPO` + retrieval hard-negative reranker.
3. (2) + serving speculative decoding/runtime optimization.

Promote only runs that pass `proposal_quality_gate.py` and beat current anchor artifacts.

## Primary Sources
1. DPO: https://arxiv.org/abs/2305.18290
2. ORPO: https://arxiv.org/abs/2403.07691
3. SimPO: https://arxiv.org/abs/2405.14734
4. QLoRA: https://arxiv.org/abs/2305.14314
5. ColBERTv2: https://arxiv.org/abs/2112.01488
6. CRAG: https://arxiv.org/abs/2401.15884
7. Self-RAG: https://arxiv.org/abs/2310.11511
8. vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
9. Speculative Decoding: https://arxiv.org/abs/2211.17192
10. MT-Bench / LLM-as-judge: https://arxiv.org/abs/2306.05685
11. BEIR benchmark: https://arxiv.org/abs/2104.08663
