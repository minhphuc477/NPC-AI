#!/usr/bin/env python3
"""BD-NSCA Evaluation Script using BERTScore."""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore between predictions and references.
    
    Requires: pip install bert-score
    """
    try:
        from bert_score import score
    except ImportError:
        logger.error("bert-score not installed. Run: pip install bert-score")
        return {"precision": 0, "recall": 0, "f1": 0}
    
    P, R, F1 = score(predictions, references, lang="vi", verbose=True)
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }


def evaluate_context_relevance(sample: Dict, response: str) -> float:
    """Evaluate how well the response relates to the provided context.
    
    Simple keyword-based scoring as a baseline.
    """
    context = sample.get("metadata", {}).get("context", {})
    if not context:
        return 0.5
    
    score = 0.0
    total_keys = len(context)
    
    response_lower = response.lower()
    for key, value in context.items():
        if isinstance(value, str) and value.lower() in response_lower:
            score += 1.0
    
    return score / total_keys if total_keys > 0 else 0.5


def evaluate_persona_consistency(sample: Dict, response: str) -> float:
    """Evaluate persona consistency based on expected traits.
    
    Simple heuristic scoring.
    """
    npc_id = sample.get("metadata", {}).get("npc_id", "")
    
    trait_keywords = {
        "gatekeeper": ["nghiêm", "cảnh giác", "ngươi", "ta"],
        "merchant": ["hàng", "bán", "tiền", "giá"],
        "healer": ["thuốc", "khỏe", "bệnh", "chữa"]
    }
    
    keywords = trait_keywords.get(npc_id, [])
    if not keywords:
        return 0.5
    
    response_lower = response.lower()
    matches = sum(1 for kw in keywords if kw in response_lower)
    return matches / len(keywords)


def run_evaluation(test_file: str, predictions_file: str = None) -> Dict:
    """Run full evaluation pipeline."""
    samples = load_jsonl(test_file)
    logger.info(f"Loaded {len(samples)} test samples")
    
    if predictions_file:
        predictions = load_jsonl(predictions_file)
        pred_responses = [p.get("prediction", p.get("response", "")) for p in predictions]
        
        # Slice references to match predictions length (handle partial evaluation)
        if len(predictions) < len(samples):
            logger.warning(f"Partial predictions found: {len(predictions)} vs {len(samples)} test samples. Truncating references.")
            samples = samples[:len(predictions)]
    else:
        pred_responses = [s.get("completion", "") for s in samples]
    
    ref_responses = [s.get("completion", "") for s in samples]
    
    results = {
        "num_samples": len(samples),
        "bertscore": calculate_bertscore(pred_responses, ref_responses),
        "context_relevance": 0.0,
        "persona_consistency": 0.0
    }
    
    ctx_scores = []
    persona_scores = []
    
    for i, sample in enumerate(samples):
        response = pred_responses[i] if i < len(pred_responses) else ""
        ctx_scores.append(evaluate_context_relevance(sample, response))
        persona_scores.append(evaluate_persona_consistency(sample, response))
    
    results["context_relevance"] = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0
    results["persona_consistency"] = sum(persona_scores) / len(persona_scores) if persona_scores else 0
    
    return results


def print_report(results: Dict):
    """Print evaluation report."""
    print("\n" + "=" * 50)
    print("BD-NSCA EVALUATION REPORT")
    print("=" * 50)
    print(f"Number of samples: {results['num_samples']}")
    print()
    print("BERTScore (Semantic Similarity):")
    print(f"  Precision: {results['bertscore']['precision']:.4f}")
    print(f"  Recall:    {results['bertscore']['recall']:.4f}")
    print(f"  F1:        {results['bertscore']['f1']:.4f}")
    print()
    print("Context Relevance Score: {:.4f}".format(results['context_relevance']))
    print("Persona Consistency Score: {:.4f}".format(results['persona_consistency']))
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="BD-NSCA Evaluation")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--predictions", help="Path to predictions JSONL file (optional)")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()
    
    results = run_evaluation(args.test, args.predictions)
    print_report(results)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
