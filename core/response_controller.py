"""Response control layer for proposal-aligned NPC dialogue quality.

This module improves inference-time quality by:
1. sanitizing meta/template artifacts,
2. attempting constrained self-rewrite when outputs drift,
3. falling back to a grounded deterministic reply using dynamic context.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Dict, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

BLOCKED_FRAGMENTS = (
    "your response here",
    "solution to instruction",
    "instruction with the same difficulty",
    "temporal memories",
    "ambient awareness",
    "behaviortreestate",
)

INTENT_STOPWORDS = {
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "us",
    "please",
    "can",
    "could",
    "would",
    "should",
    "let",
    "need",
    "want",
    "now",
    "to",
    "in",
    "into",
    "at",
    "the",
    "a",
    "an",
    "is",
    "are",
    "do",
    "does",
}


@dataclass(frozen=True)
class ControlConfig:
    min_context_coverage: float = 0.35
    min_persona_coverage: float = 0.20
    rewrite_temperature: float = 0.2
    rewrite_max_tokens: int = 96
    rewrite_candidates: int = 3
    rewrite_temperature_step: float = 0.15
    enable_rewrite: bool = True
    allow_best_effort_rewrite: bool = True


@dataclass(frozen=True)
class ControlResult:
    response: str
    source: str
    context_coverage: float
    persona_coverage: float
    repaired: bool
    repair_reason: str


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def keyword_coverage(text: str, keywords: Sequence[str]) -> float:
    kws = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not kws:
        return 0.0

    lowered = (text or "").lower()
    text_tokens = set(tokenize(text))
    hits = 0
    for kw in kws:
        if kw in lowered:
            hits += 1
            continue
        kw_tokens = tokenize(kw)
        if kw_tokens and all(token in text_tokens for token in kw_tokens):
            hits += 1
    return hits / len(kws)


def sanitize_response(text: str) -> str:
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()

        if low.startswith(("system persona:", "persona:", "rules:", "npc reply")):
            continue
        if low.startswith("[your response here]"):
            continue
        if low.startswith("**solution"):
            continue
        if low.startswith("as elara") and "respond" in low:
            continue
        if low.startswith("elara") and "=" in line:
            continue
        if any(fragment in low for fragment in BLOCKED_FRAGMENTS):
            continue
        cleaned_lines.append(line)

    merged = " ".join(cleaned_lines).strip()
    if not merged:
        return ""

    merged = re.sub(r"^[A-Za-z0-9_ ()-]{1,48}:\s*", "", merged).strip()
    merged = re.sub(r"^\[[^\]]+\]\s*", "", merged).strip()
    merged = merged.strip("\"' ")
    if any(fragment in merged.lower() for fragment in BLOCKED_FRAGMENTS):
        return ""

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(merged) if s.strip()]
    if not sentences:
        return ""
    return " ".join(sentences[:3]).strip()


def parse_dynamic_context(dynamic_context: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not dynamic_context:
        return out
    for part in dynamic_context.split(";"):
        chunk = part.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        elif ":" in chunk:
            key, value = chunk.split(":", 1)
        else:
            continue
        out[key.strip().lower()] = value.strip()
    return out


def _context_detail_phrases(dynamic_context: str) -> List[str]:
    kv = parse_dynamic_context(dynamic_context)
    details: List[str] = []

    if "location" in kv:
        details.append(f"at {kv['location']}")
    if "behaviortreestate" in kv or "behavior_state" in kv:
        state = kv.get("behaviortreestate", kv.get("behavior_state", "ready"))
        details.append(f"while on {state.lower()} duty")
    if "nearbyentity" in kv:
        details.append(f"with {kv['nearbyentity'].lower()} nearby")
    if "recentevent" in kv:
        details.append(f"after {kv['recentevent'].lower()}")

    if not details:
        for key, value in list(kv.items())[:2]:
            details.append(f"{key.replace('_', ' ')} {value}")

    return details


def _context_detail_sentences(dynamic_context: str) -> List[str]:
    phrases = _context_detail_phrases(dynamic_context)
    if not phrases:
        return []
    if len(phrases) == 1:
        return [f"We are {phrases[0]}."]
    return [f"We are {phrases[0]} and {phrases[1]}."] + [f"Current conditions remain {p}." for p in phrases[2:]]


def _persona_style(persona: str) -> str:
    p = (persona or "").lower()
    if any(x in p for x in ("strict", "guard", "captain", "procedural")):
        return "strict"
    if any(x in p for x in ("merchant", "talkative", "trader")):
        return "talkative"
    if any(x in p for x in ("healer", "calm", "caring", "medical")):
        return "calm"
    if any(x in p for x in ("witch", "mysterious", "indirect")):
        return "mysterious"
    if any(x in p for x in ("scholar", "formal", "precision")):
        return "formal"
    return "neutral"


def _intent_fragment(player_input: str) -> str:
    words = tokenize(player_input)
    if not words:
        return "your request"
    content = [word for word in words if word not in INTENT_STOPWORDS]
    if content:
        return " ".join(content[:4])
    return " ".join(words[:5])


def grounded_fallback_response(persona: str, dynamic_context: str, player_input: str) -> str:
    detail_phrases = _context_detail_phrases(dynamic_context)
    style = _persona_style(persona)
    intent = _intent_fragment(player_input)

    if detail_phrases:
        if len(detail_phrases) == 1:
            context_sentence = f"Here {detail_phrases[0]},"
        else:
            context_sentence = f"Here {detail_phrases[0]} and {detail_phrases[1]},"
    else:
        context_sentence = "Given the current situation,"

    if style == "strict":
        return (
            f"{context_sentence} I cannot approve {intent} until verification is complete. "
            "Follow protocol and I will move this forward."
        )
    if style == "talkative":
        return (
            f"{context_sentence} I can work with {intent}, but the terms must stay fair. "
            "Keep it honest and we can close this deal quickly."
        )
    if style == "calm":
        return (
            f"{context_sentence} we can handle {intent} safely, step by step. "
            "Stay steady and I will guide the next action."
        )
    if style == "mysterious":
        return (
            f"{context_sentence} your path around {intent} has a cost. "
            "Choose carefully, because timing matters as much as power."
        )
    if style == "formal":
        return (
            f"{context_sentence} the request regarding {intent} requires evidence before any conclusion. "
            "Provide verifiable details and I will continue."
        )
    return (
        f"{context_sentence} I can respond to {intent} within these conditions. "
        "Give one clear detail and I will proceed precisely."
    )


def build_rewrite_prompt(
    persona: str,
    dynamic_context: str,
    player_input: str,
    draft_response: str,
) -> str:
    lines: List[str] = []
    lines.append("Rewrite the NPC response to satisfy strict quality constraints.")
    lines.append(f"Persona: {persona}")
    lines.append("Dynamic game state:")
    for sentence in _context_detail_sentences(dynamic_context):
        lines.append(f"- {sentence}")
    lines.append(f"Player says: {player_input}")
    lines.append(f"Draft response: {draft_response}")
    lines.append("Constraints:")
    lines.append("- Keep 2-3 natural sentences.")
    lines.append("- Keep role-play tone consistent with persona.")
    lines.append("- Use at least two concrete dynamic game details.")
    lines.append("- Avoid repeated phrasing; prefer precise varied wording.")
    lines.append("- No labels, no metadata, no JSON, no analysis.")
    lines.append("Return only the rewritten NPC dialogue.")
    return "\n".join(lines)


def _needs_repair(
    response: str,
    context_cov: float,
    persona_cov: float,
    config: ControlConfig,
) -> Tuple[bool, str]:
    low = (response or "").lower()
    if not response:
        return True, "empty_response"
    if len(tokenize(response)) < 8:
        return True, "too_short"
    if any(fragment in low for fragment in BLOCKED_FRAGMENTS):
        return True, "meta_artifact"
    if context_cov < config.min_context_coverage:
        return True, "low_context_coverage"
    if persona_cov < config.min_persona_coverage:
        return True, "low_persona_coverage"
    return False, ""


def _candidate_length_score(response: str) -> float:
    wc = len(tokenize(response))
    if wc == 0:
        return 0.0
    target = 34.0
    spread = 28.0
    return max(0.0, 1.0 - abs(wc - target) / spread)


def _candidate_diversity_score(response: str) -> float:
    words = tokenize(response)
    wc = len(words)
    if wc == 0:
        return 0.0
    distinct_1 = len(set(words)) / float(wc)
    if wc < 3:
        repetition = 0.0
    else:
        grams = [tuple(words[i : i + 3]) for i in range(wc - 2)]
        if not grams:
            repetition = 0.0
        else:
            seen: Dict[tuple[str, ...], int] = {}
            for gram in grams:
                seen[gram] = seen.get(gram, 0) + 1
            repeats = sum(v - 1 for v in seen.values() if v > 1)
            repetition = repeats / float(len(grams))
    return max(0.0, min(1.0, 0.6 * distinct_1 + 0.4 * (1.0 - repetition)))


def _candidate_score(
    response: str,
    context_keywords: Sequence[str],
    persona_keywords: Sequence[str],
) -> float:
    if not response:
        return -1.0
    low = response.lower()
    if any(fragment in low for fragment in BLOCKED_FRAGMENTS):
        return -1.0
    context_cov = keyword_coverage(response, context_keywords)
    persona_cov = keyword_coverage(response, persona_keywords)
    length_score = _candidate_length_score(response)
    diversity_score = _candidate_diversity_score(response)
    return 0.50 * context_cov + 0.30 * persona_cov + 0.10 * length_score + 0.10 * diversity_score


def _rewrite_temperatures(config: ControlConfig) -> List[float]:
    n = max(1, int(config.rewrite_candidates))
    base = min(1.5, max(0.05, float(config.rewrite_temperature)))
    step = max(0.01, float(config.rewrite_temperature_step))
    temps: List[float] = [round(base, 4)]

    radius = 1
    while len(temps) < n:
        hi = round(min(1.5, base + radius * step), 4)
        if hi not in temps:
            temps.append(hi)
        if len(temps) >= n:
            break
        lo = round(max(0.05, base - radius * step), 4)
        if lo not in temps:
            temps.append(lo)
        radius += 1
    return temps[:n]


def control_response(
    raw_response: str,
    persona: str,
    dynamic_context: str,
    player_input: str,
    context_keywords: Sequence[str],
    persona_keywords: Sequence[str],
    rewrite_fn: Callable[[str, int, float], str] | None = None,
    config: ControlConfig = ControlConfig(),
) -> ControlResult:
    cleaned = sanitize_response(raw_response)
    context_cov = keyword_coverage(cleaned, context_keywords)
    persona_cov = keyword_coverage(cleaned, persona_keywords)
    repair_needed, reason = _needs_repair(cleaned, context_cov, persona_cov, config)
    if not repair_needed:
        return ControlResult(
            response=cleaned,
            source="raw",
            context_coverage=context_cov,
            persona_coverage=persona_cov,
            repaired=False,
            repair_reason="",
        )

    best_rewrite_passing: tuple[str, float, float, float] | None = None
    best_rewrite_candidate: tuple[str, float, float, float] | None = None

    if config.enable_rewrite and rewrite_fn is not None:
        rewrite_prompt = build_rewrite_prompt(persona, dynamic_context, player_input, cleaned or raw_response)
        for temp in _rewrite_temperatures(config):
            rewritten_raw = rewrite_fn(rewrite_prompt, config.rewrite_max_tokens, temp)
            rewritten = sanitize_response(rewritten_raw)
            if not rewritten:
                continue
            rewrite_context_cov = keyword_coverage(rewritten, context_keywords)
            rewrite_persona_cov = keyword_coverage(rewritten, persona_keywords)
            rewrite_score = _candidate_score(rewritten, context_keywords, persona_keywords)
            if best_rewrite_candidate is None or rewrite_score > best_rewrite_candidate[3]:
                best_rewrite_candidate = (
                    rewritten,
                    rewrite_context_cov,
                    rewrite_persona_cov,
                    rewrite_score,
                )
            rewrite_needed, _ = _needs_repair(rewritten, rewrite_context_cov, rewrite_persona_cov, config)
            if not rewrite_needed:
                if best_rewrite_passing is None or rewrite_score > best_rewrite_passing[3]:
                    best_rewrite_passing = (
                        rewritten,
                        rewrite_context_cov,
                        rewrite_persona_cov,
                        rewrite_score,
                    )

    if best_rewrite_passing is not None:
        return ControlResult(
            response=best_rewrite_passing[0],
            source="rewritten",
            context_coverage=best_rewrite_passing[1],
            persona_coverage=best_rewrite_passing[2],
            repaired=True,
            repair_reason=reason,
        )

    fallback = grounded_fallback_response(persona, dynamic_context, player_input)
    fallback = sanitize_response(fallback)
    fallback_context_cov = keyword_coverage(fallback, context_keywords)
    fallback_persona_cov = keyword_coverage(fallback, persona_keywords)

    if config.allow_best_effort_rewrite and best_rewrite_candidate is not None:
        fallback_score = _candidate_score(fallback, context_keywords, persona_keywords)
        if best_rewrite_candidate[3] >= (fallback_score + 0.05):
            return ControlResult(
                response=best_rewrite_candidate[0],
                source="rewritten_best_effort",
                context_coverage=best_rewrite_candidate[1],
                persona_coverage=best_rewrite_candidate[2],
                repaired=True,
                repair_reason=reason,
            )

    return ControlResult(
        response=fallback or "I need a clearer request before I can respond in character.",
        source="fallback",
        context_coverage=fallback_context_cov,
        persona_coverage=fallback_persona_cov,
        repaired=True,
        repair_reason=reason,
    )
