#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) training data — v2.

Improved rejection strategies that create SUBTLE negatives,
forcing the model to learn fine-grained style differences rather
than just "don't be generic."

Rejection Types:
1. Style mismatch  — correct info, wrong personality
2. Tone violation  — right persona, wrong emotional tone for the state
3. Over-verbose    — correct content but unnaturally long
4. Info leak       — breaks character by referencing game mechanics
5. Wrong persona   — response from a completely different NPC (legacy)

Usage:
    python scripts/generate_dpo_data.py --input data/train_combined.jsonl --output data/train_dpo.jsonl
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# --- Rejection Strategy Implementations ---

def _reject_style_mismatch(sample: Dict, all_samples: List[Dict]) -> str:
    """Same archetype topic, totally wrong personality."""
    lang = sample.get('metadata', {}).get('language', 'en')
    chosen = sample['completion']

    # Make it polite if original is stern, or stern if original is friendly
    polite_prefixes_vi = [
        "Thưa quý khách, ",
        "Kính chào, xin phép hỏi, ",
        "Con xin lỗi, nhưng ",
    ]
    polite_prefixes_en = [
        "Dear esteemed traveler, ",
        "My most sincere apologies, but ",
        "If I may humbly suggest, ",
    ]
    stern_prefixes_vi = [
        "Nghe đây. ",
        "Im lặng. ",
        "Không có gì để nói thêm. ",
    ]
    stern_prefixes_en = [
        "Listen here. ",
        "Silence. ",
        "There is nothing more to discuss. ",
    ]

    if lang == "vi":
        prefix = random.choice(polite_prefixes_vi + stern_prefixes_vi)
    else:
        prefix = random.choice(polite_prefixes_en + stern_prefixes_en)

    # Truncate and re-prefix
    core = chosen.split(".", 1)[-1].strip() if "." in chosen else chosen
    return prefix + core[:80]


def _reject_tone_violation(sample: Dict, all_samples: List[Dict]) -> str:
    """Right persona cues but completely wrong emotional tone for the state."""
    lang = sample.get('metadata', {}).get('language', 'en')

    # Generate inappropriately cheerful response when context is tense
    happy_vi = [
        "Ha ha! Thật vui quá! Mọi chuyện đều ổn mà!",
        "Ồ, tuyệt vời! Không có gì phải lo đâu!",
        "Hôm nay trời đẹp quá nhỉ! Thật tuyệt!",
        "Vui quá! Hãy cùng ăn mừng nào!",
    ]
    happy_en = [
        "Ha ha! How wonderful! Everything is perfectly fine!",
        "Oh, splendid! Nothing to worry about at all!",
        "What a beautiful day! Simply marvelous!",
        "How exciting! Let's celebrate!",
    ]
    sad_vi = [
        "Ôi... mọi thứ thật vô vọng...",
        "Tôi không muốn nói gì nữa... cuộc sống thật đau khổ...",
        "Chẳng có gì tốt đẹp nữa đâu...",
    ]
    sad_en = [
        "Oh... everything is so hopeless...",
        "I don't want to talk anymore... life is suffering...",
        "Nothing good will ever happen again...",
    ]

    if lang == "vi":
        return random.choice(happy_vi + sad_vi)
    else:
        return random.choice(happy_en + sad_en)


def _reject_over_verbose(sample: Dict, all_samples: List[Dict]) -> str:
    """Correct content idea but painfully over-explained."""
    chosen = sample['completion']
    lang = sample.get('metadata', {}).get('language', 'en')

    # Pad with filler text
    fillers_vi = [
        " Bạn biết đấy, điều này rất quan trọng.",
        " Tôi muốn nói rằng, thật là, bạn hiểu không,",
        " Và nói thêm, như tôi đã từng nói với nhiều người,",
        " Nhưng mà, thực ra, xét cho cùng thì,",
        " Nếu bạn hỏi tôi, mà tôi chắc bạn đang hỏi,",
    ]
    fillers_en = [
        " You know, this is really quite important.",
        " I want to say that, well, you see,",
        " And furthermore, as I have told many others,",
        " But actually, when you think about it,",
        " If you ask me, and I believe you are asking,",
    ]

    fillers = fillers_vi if lang == "vi" else fillers_en
    padded = chosen
    for _ in range(random.randint(2, 4)):
        insert_pos = random.randint(0, len(padded))
        padded = padded[:insert_pos] + random.choice(fillers) + padded[insert_pos:]

    return padded[:300]  # Cap at 300 chars


def _reject_info_leak(sample: Dict, all_samples: List[Dict]) -> str:
    """Breaks immersion by referencing game mechanics or meta-knowledge."""
    lang = sample.get('metadata', {}).get('language', 'en')

    meta_vi = [
        "Đây là nhiệm vụ phụ, bạn cần hoàn thành trước khi tiếp tục cốt truyện chính.",
        "Bạn nên tăng cấp thêm 2 level nữa trước khi quay lại đây.",
        "Nếu chọn đáp án A, bạn sẽ nhận được 50 điểm kinh nghiệm.",
        "Hãy save game trước khi tiếp tục, vì đoạn này khó lắm.",
        "Trong bản cập nhật tiếp theo, khu vực này sẽ có thêm NPC.",
    ]
    meta_en = [
        "This is a side quest, you need to complete it before continuing the main story.",
        "You should gain 2 more levels before coming back here.",
        "If you choose option A, you'll receive 50 experience points.",
        "Make sure to save your game before proceeding, this part is hard.",
        "In the next update, this area will have more NPCs added.",
    ]

    if lang == "vi":
        return random.choice(meta_vi)
    else:
        return random.choice(meta_en)


def _reject_wrong_persona(sample: Dict, all_samples: List[Dict]) -> str:
    """Legacy: response from a completely different NPC (easy negative)."""
    npc_id = sample.get('metadata', {}).get('npc_id', '')
    other = [s for s in all_samples if s.get('metadata', {}).get('npc_id', '') != npc_id]
    if other:
        return random.choice(other)['completion']
    return sample['completion']


# Strategy weights — subtle rejections get higher weight
REJECTION_STRATEGIES = [
    (_reject_style_mismatch, 0.25),
    (_reject_tone_violation, 0.25),
    (_reject_over_verbose, 0.20),
    (_reject_info_leak, 0.15),
    (_reject_wrong_persona, 0.15),
]


def generate_rejected_response(sample: Dict, all_samples: List[Dict]) -> str:
    """Generate a rejected response using weighted random strategy selection."""
    strategies, weights = zip(*REJECTION_STRATEGIES)
    strategy = random.choices(strategies, weights=weights, k=1)[0]
    return strategy(sample, all_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_combined.jsonl")
    parser.add_argument("--output", default="data/train_dpo.jsonl")
    parser.add_argument("--multi-reject", type=int, default=1,
                        help="Number of rejection samples per chosen (1-3)")
    args = parser.parse_args()

    data = load_jsonl(args.input)
    dpo_data = []

    print("Generating DPO pairs from {} samples...".format(len(data)))
    print("Rejection strategies: style_mismatch, tone_violation, over_verbose, info_leak, wrong_persona")

    for sample in data:
        for _ in range(args.multi_reject):
            dpo_sample = {
                "prompt": sample['prompt'],
                "chosen": sample['completion'],
                "rejected": generate_rejected_response(sample, data),
                "metadata": sample.get('metadata', {})
            }
            dpo_data.append(dpo_sample)

    save_jsonl(dpo_data, args.output)
    print("Saved {} DPO pairs to {}".format(len(dpo_data), args.output))

    # Print strategy distribution for verification
    print("\nSample rejections:")
    for i, s in enumerate(dpo_data[:5]):
        print("  [{}] Chosen: {}...".format(i, s['chosen'][:50]))
        print("       Rejected: {}...".format(s['rejected'][:50]))


if __name__ == "__main__":
    main()
