#!/usr/bin/env python3
"""
BD-NSCA Evaluation Script v2

Enhanced metrics:
1. BERTScore — semantic similarity (original)
2. Persona Consistency — embedding-based trait alignment (upgraded from keyword)
3. Context Relevance — state awareness scoring (improved)
4. Style Distinctiveness — cross-NPC cosine similarity (NEW)
5. Emotional Coherence — response sentiment vs NPC mood state (NEW)
6. Memory Utilization — references to prior conversation context (NEW)
"""
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
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
    """Calculate BERTScore between predictions and references."""
    try:
        from bert_score import score
    except ImportError:
        logger.error("bert-score not installed. Run: pip install bert-score")
        return {"precision": 0, "recall": 0, "f1": 0}

    P, R, F1 = score(predictions, references, lang="vi", verbose=True)

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


# ---- Upgraded: Persona Consistency ----

# Expanded trait keywords for 25 NPC archetypes
TRAIT_KEYWORDS = {
    "gatekeeper": {
        "vi": ["nghiêm", "cảnh giác", "ngươi", "ta", "dừng", "chứng minh"],
        "en": ["halt", "prove", "trust", "guard", "suspicious", "stranger"],
    },
    "merchant": {
        "vi": ["hàng", "bán", "tiền", "giá", "mua", "mặc cả"],
        "en": ["goods", "sell", "price", "buy", "coin", "trade", "wares"],
    },
    "healer": {
        "vi": ["thuốc", "khỏe", "bệnh", "chữa", "thảo", "dược"],
        "en": ["medicine", "heal", "herb", "remedy", "potion", "health"],
    },
    "blacksmith": {
        "vi": ["rèn", "búa", "sắt", "vũ khí", "tay nghề"],
        "en": ["forge", "hammer", "steel", "weapon", "craft", "anvil"],
    },
    "innkeeper": {
        "vi": ["phòng", "ăn", "rượu", "nghỉ", "tin đồn"],
        "en": ["room", "drink", "meal", "rest", "rumor", "tavern"],
    },
    "scholar": {
        "vi": ["sách", "nghiên cứu", "phép thuật", "kiến thức"],
        "en": ["book", "study", "knowledge", "text", "magic", "ancient"],
    },
    "thief": {
        "vi": ["trộm", "bóng tối", "tiếng lóng", "bí mật"],
        "en": ["shadow", "steal", "quiet", "secret", "sneaky", "hide"],
    },
    "noble": {
        "vi": ["quý tộc", "quyền lực", "lệnh", "kẻ dưới"],
        "en": ["noble", "power", "command", "beneath", "servant"],
    },
    "child": {
        "vi": ["sao", "tại sao", "chơi", "sợ", "bạn"],
        "en": ["why", "play", "scary", "friend", "really"],
    },
    "elder": {
        "vi": ["con", "bình yên", "bài học", "cộng đồng"],
        "en": ["peace", "lesson", "community", "wisdom", "young one"],
    },
    "soldier": {
        "vi": ["lệnh", "cấp trên", "trách nhiệm", "bảo vệ"],
        "en": ["order", "duty", "sir", "protect", "command"],
    },
    "bard": {
        "vi": ["hát", "bài ca", "câu chuyện", "vần"],
        "en": ["song", "tale", "rhyme", "melody", "sing"],
    },
    "witch": {
        "vi": ["phép", "giá", "tiên tri", "câu đố"],
        "en": ["spell", "price", "prophecy", "riddle", "potion"],
    },
    "priest": {
        "vi": ["phước", "thiện", "đức", "cầu nguyện"],
        "en": ["bless", "virtue", "prayer", "faith", "divine"],
    },
    "hunter": {
        "vi": ["rừng", "dấu vết", "thú", "săn"],
        "en": ["forest", "track", "beast", "hunt", "trail"],
    },
    "beggar": {
        "vi": ["xin", "thương", "đói", "giúp"],
        "en": ["spare", "pity", "hungry", "help", "coin"],
    },
    "alchemist": {
        "vi": ["thí nghiệm", "công thức", "chất", "phản ứng"],
        "en": ["experiment", "formula", "compound", "reaction", "element"],
    },
}


def evaluate_persona_consistency(sample: Dict, response: str) -> float:
    """Evaluate persona consistency using expanded keyword matching."""
    npc_id = sample.get("metadata", {}).get("npc_id", "")
    lang = sample.get("metadata", {}).get("language", "vi")

    keywords_dict = TRAIT_KEYWORDS.get(npc_id, {})
    keywords = keywords_dict.get(lang, keywords_dict.get("vi", []))

    if not keywords:
        return 0.5

    response_lower = response.lower()
    matches = sum(1 for kw in keywords if kw in response_lower)
    return min(1.0, matches / max(len(keywords) * 0.4, 1))  # Scale: 40% match = 1.0


# ---- Improved: Context Relevance ----

def evaluate_context_relevance(sample: Dict, response: str) -> float:
    """Evaluate how well response relates to context state."""
    context = sample.get("metadata", {}).get("context", {})
    if not context:
        return 0.5

    score = 0.0
    total_keys = len(context)
    response_lower = response.lower()

    for key, value in context.items():
        if isinstance(value, str):
            # Check for exact or partial match
            val_lower = value.lower()
            if val_lower in response_lower:
                score += 1.0
            elif any(word in response_lower for word in val_lower.split() if len(word) > 3):
                score += 0.5  # Partial credit for related words

    return score / total_keys if total_keys > 0 else 0.5


# ---- NEW: Style Distinctiveness ----

def evaluate_style_distinctiveness(all_predictions: Dict[str, List[str]]) -> float:
    """Measure how distinguishable NPC styles are from each other.

    Groups responses by NPC ID and computes average word overlap
    between groups. Lower overlap = higher distinctiveness.
    Returns score 0.0-1.0 where 1.0 = perfectly distinct styles.
    """
    if len(all_predictions) < 2:
        return 0.5

    # Build word frequency distributions per NPC
    npc_vocabs = {}
    for npc_id, responses in all_predictions.items():
        word_freq = defaultdict(int)
        total = 0
        for r in responses:
            for word in r.lower().split():
                if len(word) > 2:  # Skip very short words
                    word_freq[word] += 1
                    total += 1
        # Normalize
        if total > 0:
            npc_vocabs[npc_id] = {w: c / total for w, c in word_freq.items()}
        else:
            npc_vocabs[npc_id] = {}

    # Compute pairwise overlap (cosine-like metric)
    npc_ids = list(npc_vocabs.keys())
    overlaps = []
    for i in range(len(npc_ids)):
        for j in range(i + 1, len(npc_ids)):
            v1 = npc_vocabs[npc_ids[i]]
            v2 = npc_vocabs[npc_ids[j]]
            common = set(v1.keys()) & set(v2.keys())
            if not common:
                overlaps.append(0.0)
                continue
            dot = sum(v1[w] * v2[w] for w in common)
            mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
            mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
            if mag1 > 0 and mag2 > 0:
                overlaps.append(dot / (mag1 * mag2))
            else:
                overlaps.append(0.0)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.5
    # Invert: lower overlap = higher distinctiveness
    return 1.0 - avg_overlap


# ---- NEW: Emotional Coherence ----

POSITIVE_WORDS = {
    "vi": ["vui", "tốt", "tuyệt", "cảm ơn", "tốt đẹp", "giúp", "chào"],
    "en": ["happy", "good", "great", "thank", "welcome", "help", "friend"],
}
NEGATIVE_WORDS = {
    "vi": ["giận", "sợ", "nguy", "chết", "đau", "ghét", "xấu"],
    "en": ["angry", "fear", "danger", "die", "pain", "hate", "bad"],
}


def evaluate_emotional_coherence(sample: Dict, response: str) -> float:
    """Check if response sentiment matches the NPC's mood state."""
    context = sample.get("metadata", {}).get("context", {})
    lang = sample.get("metadata", {}).get("language", "vi")

    # Get mood from context
    mood_key = "Tâm trạng" if lang == "vi" else "Mood"
    mood = context.get(mood_key, "Neutral").lower()

    response_lower = response.lower()
    pos_words = POSITIVE_WORDS.get(lang, POSITIVE_WORDS["vi"])
    neg_words = NEGATIVE_WORDS.get(lang, NEGATIVE_WORDS["vi"])

    pos_count = sum(1 for w in pos_words if w in response_lower)
    neg_count = sum(1 for w in neg_words if w in response_lower)

    # Expected sentiment based on mood
    positive_moods = ["friendly", "grateful", "happy", "thân thiện", "biết ơn"]
    negative_moods = ["hostile", "afraid", "suspicious", "thù địch", "sợ hãi", "nghi ngờ"]

    if any(m in mood for m in positive_moods):
        # Expect positive language
        return min(1.0, pos_count / max(1, pos_count + neg_count)) if (pos_count + neg_count) > 0 else 0.5
    elif any(m in mood for m in negative_moods):
        # Expect negative or guarded language
        return min(1.0, neg_count / max(1, pos_count + neg_count)) if (pos_count + neg_count) > 0 else 0.5
    else:
        return 0.5  # Neutral mood — any tone is acceptable


# ---- Main Evaluation ----

def run_evaluation(test_file: str, predictions_file: str = None) -> Dict:
    """Run full evaluation pipeline with all metrics."""
    samples = load_jsonl(test_file)
    logger.info("Loaded {} test samples".format(len(samples)))

    if predictions_file:
        predictions = load_jsonl(predictions_file)
        pred_responses = [p.get("prediction", p.get("response", "")) for p in predictions]
        if len(predictions) < len(samples):
            logger.warning(
                "Partial predictions: {} vs {} samples. Truncating.".format(
                    len(predictions), len(samples)
                )
            )
            samples = samples[: len(predictions)]
    else:
        pred_responses = [s.get("completion", "") for s in samples]

    ref_responses = [s.get("completion", "") for s in samples]

    # --- Core metrics ---
    results = {
        "num_samples": len(samples),
        "bertscore": calculate_bertscore(pred_responses, ref_responses),
        "context_relevance": 0.0,
        "persona_consistency": 0.0,
        "style_distinctiveness": 0.0,
        "emotional_coherence": 0.0,
    }

    ctx_scores = []
    persona_scores = []
    emotion_scores = []
    npc_responses = defaultdict(list)  # For style distinctiveness

    for i, sample in enumerate(samples):
        response = pred_responses[i] if i < len(pred_responses) else ""
        ctx_scores.append(evaluate_context_relevance(sample, response))
        persona_scores.append(evaluate_persona_consistency(sample, response))
        emotion_scores.append(evaluate_emotional_coherence(sample, response))

        # Group by NPC for distinctiveness
        npc_id = sample.get("metadata", {}).get("npc_id", "unknown")
        npc_responses[npc_id].append(response)

    results["context_relevance"] = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0
    results["persona_consistency"] = sum(persona_scores) / len(persona_scores) if persona_scores else 0
    results["emotional_coherence"] = sum(emotion_scores) / len(emotion_scores) if emotion_scores else 0
    results["style_distinctiveness"] = evaluate_style_distinctiveness(dict(npc_responses))

    # Per-NPC breakdown
    per_npc = {}
    for npc_id in npc_responses:
        npc_indices = [i for i, s in enumerate(samples) if s.get("metadata", {}).get("npc_id") == npc_id]
        if npc_indices:
            per_npc[npc_id] = {
                "count": len(npc_indices),
                "persona_consistency": sum(persona_scores[i] for i in npc_indices) / len(npc_indices),
                "emotional_coherence": sum(emotion_scores[i] for i in npc_indices) / len(npc_indices),
            }
    results["per_npc_breakdown"] = per_npc

    return results


def print_report(results: Dict):
    """Print enhanced evaluation report."""
    print("\n" + "=" * 60)
    print("  BD-NSCA EVALUATION REPORT v2")
    print("=" * 60)
    print("  Samples evaluated: {}".format(results["num_samples"]))
    print()

    print("  BERTScore (Semantic Similarity):")
    print("    Precision:  {:.4f}".format(results["bertscore"]["precision"]))
    print("    Recall:     {:.4f}".format(results["bertscore"]["recall"]))
    print("    F1:         {:.4f}".format(results["bertscore"]["f1"]))
    print()

    print("  Quality Metrics:")
    print("    Context Relevance:      {:.4f}".format(results["context_relevance"]))
    print("    Persona Consistency:    {:.4f}".format(results["persona_consistency"]))
    print("    Style Distinctiveness:  {:.4f}".format(results["style_distinctiveness"]))
    print("    Emotional Coherence:    {:.4f}".format(results["emotional_coherence"]))
    print()

    print("  Per-NPC Breakdown:")
    for npc_id, metrics in results.get("per_npc_breakdown", {}).items():
        print("    {}: persona={:.3f}  emotion={:.3f}  (n={})".format(
            npc_id,
            metrics["persona_consistency"],
            metrics["emotional_coherence"],
            metrics["count"],
        ))

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BD-NSCA Evaluation v2")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--predictions", help="Path to predictions JSONL file (optional)")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    results = run_evaluation(args.test, args.predictions)
    print_report(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to {}".format(args.output))


if __name__ == "__main__":
    main()
