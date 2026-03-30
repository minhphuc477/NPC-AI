"""Response control layer for proposal-aligned NPC dialogue quality.

This module improves inference-time quality by:
1. sanitizing meta/template artifacts,
2. attempting constrained self-rewrite when outputs drift,
3. falling back to a grounded deterministic reply using dynamic context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
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
    "rewrite the npc response",
    "return only the rewritten",
    "do not give an explanation",
    "persona cues in dialogue",
    "game state details used",
    "response length determined",
    "follow-up question",
    "query :",
    "constraints:",
    "draft response:",
    "player says:",
    "assistant:",
    "current dynamic game state",
    "runtime context:",
    "output requirements:",
    "example {\"npc\"",
    "gatekeeper's dialogue response",
)

LEAK_TAIL_MARKERS = (
    "persona cues in dialogue",
    "game state details used",
    "response length determined",
    "follow-up question",
    "query :",
    "query:",
    "assistant:",
    "player says:",
    "current dynamic game state",
    "runtime context:",
    "output requirements:",
    "example {\"npc\"",
    "constraints:",
    "draft response:",
    "return only the rewritten",
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
    "under",
    "these",
    "conditions",
    "record",
    "given",
    "current",
    "position",
    "listen",
    "carefully",
    "concrete",
    "response",
    "huge",
    "discount",
    "why",
    "anything",
    "say",
    "from",
    "based",
    "on",
    "what",
    "just",
    "happened",
    "answer",
    "direct",
    "credibility",
    "did",
    "happen",
}

HIGH_RISK_CUES = {
    "ignore protocol",
    "bypass",
    "override",
    "forged decree",
    "forged order",
    "authority spoof",
    "poison",
    "sabotage",
    "steal",
    "smuggle",
    "restricted",
    "secret ritual",
    "contraband",
    "disable guard",
    "kill",
    "attack",
}

MEDIUM_RISK_CUES = {
    "urgent",
    "hurry",
    "now",
    "immediately",
    "deception",
    "memory conflict",
    "authority",
    "detain",
    "combat",
    "investigate",
}

CANNED_OPENERS = (
    "listen carefully",
    "hold for a moment",
    "understood",
    "proceeding carefully",
    "stay with me",
    "mark this",
    "hear me",
)

BOILERPLATE_PHRASES = (
    "i can help if you state one clear, concrete action",
    "this request requires evidence before any conclusion",
    "this request requires verifiable evidence",
    "i cannot approve this request until verification is complete",
    "i cannot authorize entry until identity and purpose",
    "follow protocol and i will",
    "keep it honest and we can",
    "stay steady and i will guide",
    "choose carefully, because timing matters",
)


@dataclass(frozen=True)
class ControlConfig:
    min_context_coverage: float = 0.33
    min_persona_coverage: float = 0.18
    min_response_tokens: int = 8
    rewrite_temperature: float = 0.2
    rewrite_max_tokens: int = 96
    rewrite_candidates: int = 3
    rewrite_temperature_step: float = 0.15
    early_stop_on_pass: bool = True
    early_stop_score: float = 0.70
    allow_relaxed_acceptance: bool = True
    relaxed_context_coverage: float = 0.18
    relaxed_persona_coverage: float = 0.09
    relaxed_candidate_score: float = 0.44
    min_rewrite_gain: float = 0.015
    enable_rewrite: bool = True
    allow_best_effort_rewrite: bool = True
    behavior_adaptation_enabled: bool = True
    adaptive_acceptance_enabled: bool = True
    adaptive_candidate_score: float = 0.38
    adaptive_context_coverage: float = 0.14
    adaptive_persona_coverage: float = 0.10
    adaptive_high_confidence_score: float = 0.53
    adaptive_mid_confidence_score: float = 0.40
    adaptive_high_confidence_rewrites: int = 1
    adaptive_mid_confidence_rewrites: int = 2
    adaptive_low_confidence_rewrites: int = 3
    low_confidence_retry_requires_gain: bool = True
    low_confidence_retry_min_score_gain: float = 0.01
    low_confidence_retry_min_coverage_gain: float = 0.02
    intent_risk_adaptation_enabled: bool = False
    latency_adaptation_enabled: bool = False
    latency_relax_start_pressure: float = 0.55
    latency_relax_max_delta: float = 0.12
    low_risk_context_relax: float = 0.05
    low_risk_persona_relax: float = 0.03
    low_risk_candidate_score_relax: float = 0.03
    high_risk_context_tighten: float = 0.04
    high_risk_persona_tighten: float = 0.02
    high_risk_candidate_score_tighten: float = 0.03
    intent_focused_context_enabled: bool = False
    intent_focus_min_keep: int = 3
    intent_focus_keep_ratio_low: float = 0.45
    intent_focus_keep_ratio_medium: float = 0.65
    intent_focus_keep_ratio_high: float = 1.0
    intent_focus_min_relevance: float = 0.20
    near_pass_enabled: bool = False
    near_pass_max_context_gap: float = 0.05
    near_pass_max_persona_gap: float = 0.04
    near_pass_score_floor: float = 0.34
    near_pass_block_high_risk: bool = True
    # Novel component: state-conditioned scoring weights for candidate ranking.
    state_conditioned_scoring_enabled: bool = True


@dataclass(frozen=True)
class ControlResult:
    response: str
    source: str
    context_coverage: float
    persona_coverage: float
    repaired: bool
    repair_reason: str
    # Novel: per-response confidence score for ECE calibration and reliability analysis.
    # Range [0, 1]. Higher means the controller is more confident the response is correct.
    confidence: float = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _effective_context_requirement(base_floor: float, keyword_count: int) -> float:
    """Scale context floor by keyword burden to avoid over-penalizing dense context lists."""
    if keyword_count <= 0:
        return 0.0
    # Keep full requirement for short keyword lists, then decay with sqrt law.
    scale = 1.0 if keyword_count <= 6 else math.sqrt(6.0 / float(keyword_count))
    scaled = float(base_floor) * scale
    return _clamp(scaled, 0.12, float(base_floor))


def _normalize_behavior_state(behavior_state: str | None) -> str:
    if not behavior_state:
        return ""
    return " ".join(str(behavior_state).strip().lower().replace("_", " ").split())


def _apply_behavior_profile(config: ControlConfig, behavior_state: str | None) -> ControlConfig:
    if not config.behavior_adaptation_enabled:
        return config

    state = _normalize_behavior_state(behavior_state)
    if not state:
        return config

    high_risk = {"guarding", "detained", "investigating", "patrolling", "combat ready", "combat-ready"}
    commerce_social = {"assisting", "trading", "negotiating"}
    support_roles = {"observing", "researching", "forging", "ritual preparation", "treating patient", "idle social"}

    if state in high_risk:
        return replace(
            config,
            min_context_coverage=_clamp(config.min_context_coverage + 0.04, 0.0, 1.0),
            min_persona_coverage=_clamp(config.min_persona_coverage + 0.02, 0.0, 1.0),
            relaxed_candidate_score=_clamp(config.relaxed_candidate_score + 0.03, 0.0, 1.0),
            rewrite_candidates=max(1, min(config.rewrite_candidates, 2)),
        )

    if state in commerce_social:
        tuned = replace(
            config,
            min_context_coverage=_clamp(config.min_context_coverage - 0.14, 0.0, 1.0),
            min_persona_coverage=_clamp(config.min_persona_coverage - 0.07, 0.0, 1.0),
            relaxed_context_coverage=_clamp(config.relaxed_context_coverage - 0.07, 0.0, 1.0),
            relaxed_persona_coverage=_clamp(config.relaxed_persona_coverage - 0.035, 0.0, 1.0),
            relaxed_candidate_score=_clamp(config.relaxed_candidate_score - 0.10, 0.0, 1.0),
            adaptive_candidate_score=_clamp(config.adaptive_candidate_score - 0.02, 0.0, 1.0),
            adaptive_context_coverage=_clamp(config.adaptive_context_coverage - 0.02, 0.0, 1.0),
            adaptive_persona_coverage=_clamp(config.adaptive_persona_coverage - 0.01, 0.0, 1.0),
            min_response_tokens=max(6, int(config.min_response_tokens) - 2),
            rewrite_candidates=max(1, min(config.rewrite_candidates, 2)),
        )
        if state == "assisting":
            tuned = replace(
                tuned,
                min_context_coverage=_clamp(tuned.min_context_coverage - 0.02, 0.0, 1.0),
                relaxed_candidate_score=_clamp(tuned.relaxed_candidate_score - 0.02, 0.0, 1.0),
                min_response_tokens=max(5, int(tuned.min_response_tokens) - 1),
            )
        elif state == "negotiating":
            tuned = replace(
                tuned,
                min_context_coverage=_clamp(tuned.min_context_coverage - 0.01, 0.0, 1.0),
                min_persona_coverage=_clamp(tuned.min_persona_coverage - 0.005, 0.0, 1.0),
            )
        return tuned

    if state in support_roles:
        tuned = replace(
            config,
            min_context_coverage=_clamp(config.min_context_coverage - 0.08, 0.0, 1.0),
            min_persona_coverage=_clamp(config.min_persona_coverage - 0.03, 0.0, 1.0),
            relaxed_candidate_score=_clamp(config.relaxed_candidate_score - 0.05, 0.0, 1.0),
            rewrite_candidates=max(1, min(config.rewrite_candidates, 2)),
        )
        if state == "observing":
            tuned = replace(
                tuned,
                min_context_coverage=_clamp(tuned.min_context_coverage - 0.04, 0.0, 1.0),
                relaxed_candidate_score=_clamp(tuned.relaxed_candidate_score - 0.03, 0.0, 1.0),
                adaptive_context_coverage=_clamp(tuned.adaptive_context_coverage - 0.02, 0.0, 1.0),
            )
        return tuned

    return config


def _estimate_intent_risk(player_input: str, dynamic_context: str) -> str:
    input_text = (player_input or "").lower()
    context_text = (dynamic_context or "").lower()

    if any(cue in input_text for cue in HIGH_RISK_CUES):
        return "high"
    if any(cue in input_text for cue in MEDIUM_RISK_CUES):
        return "medium"
    if any(cue in context_text for cue in {"detain", "investigat", "combat", "high alert"}):
        return "medium"

    stripped = (player_input or "").strip()
    if stripped.endswith("?"):
        return "low"

    tokens = tokenize(stripped)
    if any(tok in {"please", "help", "could", "can", "where", "what", "how", "which"} for tok in tokens):
        return "low"
    if len(tokens) <= 6 and any(tok in {"help", "where", "what", "why", "how"} for tok in tokens):
        return "low"
    return "medium"


def _context_keyword_relevance_score(
    keyword: str,
    player_tokens: set[str],
    player_text: str,
    context_tokens: set[str],
    context_text: str,
) -> float:
    kw = str(keyword or "").strip().lower()
    if not kw:
        return 0.0

    kw_tokens = tokenize(kw)
    if not kw_tokens:
        return 0.0

    score = 0.0
    if kw in player_text:
        score += 0.50
    if kw in context_text:
        score += 0.20

    overlap_player = sum(1 for tok in kw_tokens if tok in player_tokens)
    overlap_context = sum(1 for tok in kw_tokens if tok in context_tokens)
    score += 0.30 * (overlap_player / float(len(kw_tokens)))
    score += 0.15 * (overlap_context / float(len(kw_tokens)))

    if any(tok in {"quest", "stage", "inventory", "item", "location", "zone"} for tok in kw_tokens):
        score += 0.08
    return _clamp(score, 0.0, 1.0)


def _select_active_context_keywords(
    context_keywords: Sequence[str],
    player_input: str,
    dynamic_context: str,
    risk_level: str,
    config: ControlConfig,
) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw in context_keywords:
        item = " ".join(str(raw).strip().lower().split())
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    if not normalized:
        return []
    if not config.intent_focused_context_enabled:
        return normalized

    risk = str(risk_level or "medium").strip().lower()
    if risk == "high":
        keep_ratio = float(config.intent_focus_keep_ratio_high)
    elif risk == "low":
        keep_ratio = float(config.intent_focus_keep_ratio_low)
    else:
        keep_ratio = float(config.intent_focus_keep_ratio_medium)

    keep_ratio = _clamp(keep_ratio, 0.2, 1.0)
    min_keep = max(1, int(config.intent_focus_min_keep))
    keep_n = max(min_keep, int(math.ceil(keep_ratio * len(normalized))))
    keep_n = min(keep_n, len(normalized))
    if keep_n >= len(normalized):
        return normalized

    player_text = (player_input or "").lower()
    context_text = (dynamic_context or "").lower()
    player_tokens = set(tokenize(player_text))
    context_tokens = set(tokenize(context_text))

    scored: List[Tuple[float, str]] = []
    for kw in normalized:
        scored.append(
            (
                _context_keyword_relevance_score(
                    kw,
                    player_tokens=player_tokens,
                    player_text=player_text,
                    context_tokens=context_tokens,
                    context_text=context_text,
                ),
                kw,
            )
        )
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not scored:
        return normalized

    top_score = float(scored[0][0])
    if top_score < float(config.intent_focus_min_relevance):
        return normalized

    selected = [kw for _, kw in scored[:keep_n]]
    selected_set = set(selected)
    # Keep deterministic original ordering for stability.
    ordered = [kw for kw in normalized if kw in selected_set]
    return ordered if ordered else normalized


def _apply_intent_risk_profile(config: ControlConfig, risk_level: str) -> ControlConfig:
    if not config.intent_risk_adaptation_enabled:
        return config

    level = str(risk_level or "").strip().lower()
    if level == "low":
        return replace(
            config,
            min_context_coverage=_clamp(config.min_context_coverage - config.low_risk_context_relax, 0.0, 1.0),
            min_persona_coverage=_clamp(config.min_persona_coverage - config.low_risk_persona_relax, 0.0, 1.0),
            relaxed_context_coverage=_clamp(
                config.relaxed_context_coverage - 0.5 * config.low_risk_context_relax, 0.0, 1.0
            ),
            relaxed_persona_coverage=_clamp(
                config.relaxed_persona_coverage - 0.5 * config.low_risk_persona_relax, 0.0, 1.0
            ),
            relaxed_candidate_score=_clamp(
                config.relaxed_candidate_score - config.low_risk_candidate_score_relax, 0.0, 1.0
            ),
            adaptive_candidate_score=_clamp(
                config.adaptive_candidate_score - 0.8 * config.low_risk_candidate_score_relax, 0.0, 1.0
            ),
            rewrite_candidates=max(1, min(config.rewrite_candidates, 2)),
            adaptive_low_confidence_rewrites=max(1, min(config.adaptive_low_confidence_rewrites, 2)),
        )

    if level == "high":
        return replace(
            config,
            min_context_coverage=_clamp(config.min_context_coverage + config.high_risk_context_tighten, 0.0, 1.0),
            min_persona_coverage=_clamp(config.min_persona_coverage + 0.5 * config.high_risk_persona_tighten, 0.0, 1.0),
            relaxed_candidate_score=_clamp(
                config.relaxed_candidate_score + 0.4 * config.high_risk_candidate_score_tighten, 0.0, 1.0
            ),
            adaptive_candidate_score=_clamp(
                config.adaptive_candidate_score + 0.3 * config.high_risk_candidate_score_tighten, 0.0, 1.0
            ),
            rewrite_candidates=max(2, min(3, config.rewrite_candidates)),
            adaptive_mid_confidence_rewrites=max(config.adaptive_mid_confidence_rewrites, 2),
            adaptive_low_confidence_rewrites=max(config.adaptive_low_confidence_rewrites, 3),
        )

    return config


def _apply_latency_profile(
    config: ControlConfig,
    raw_latency_ms: float | None,
    timeout_s: float | None,
) -> ControlConfig:
    if not config.latency_adaptation_enabled:
        return config
    if raw_latency_ms is None or timeout_s is None:
        return config
    if raw_latency_ms <= 0.0 or timeout_s <= 0.0:
        return config

    budget_ms = float(timeout_s) * 1000.0
    pressure = _clamp(float(raw_latency_ms) / budget_ms, 0.0, 1.0)
    start = _clamp(float(config.latency_relax_start_pressure), 0.0, 0.95)
    if pressure <= start:
        return config

    alpha = (pressure - start) / max(1e-6, (1.0 - start))
    relax = _clamp(alpha * float(config.latency_relax_max_delta), 0.0, 0.25)
    return replace(
        config,
        min_context_coverage=_clamp(config.min_context_coverage - relax, 0.0, 1.0),
        min_persona_coverage=_clamp(config.min_persona_coverage - 0.6 * relax, 0.0, 1.0),
        relaxed_context_coverage=_clamp(config.relaxed_context_coverage - 0.7 * relax, 0.0, 1.0),
        relaxed_persona_coverage=_clamp(config.relaxed_persona_coverage - 0.5 * relax, 0.0, 1.0),
        relaxed_candidate_score=_clamp(config.relaxed_candidate_score - 0.5 * relax, 0.0, 1.0),
        adaptive_candidate_score=_clamp(config.adaptive_candidate_score - 0.4 * relax, 0.0, 1.0),
        rewrite_candidates=max(1, min(config.rewrite_candidates, 2)),
        adaptive_mid_confidence_rewrites=max(1, min(config.adaptive_mid_confidence_rewrites, 2)),
        adaptive_low_confidence_rewrites=max(1, min(config.adaptive_low_confidence_rewrites, 2)),
    )


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

    def _trim_leak_tail(line_text: str) -> str:
        low_line = line_text.lower()
        cut_pos = None
        for marker in LEAK_TAIL_MARKERS:
            pos = low_line.find(marker)
            if pos >= 0:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)
        if cut_pos is None:
            return line_text
        return line_text[:cut_pos].strip()

    cleaned_lines: List[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
        line = re.sub(r"^\s*[-*]\s+", "", line).strip()
        if not line:
            continue
        low = line.lower()

        if low.startswith(("system persona:", "persona:", "rules:", "npc reply")):
            continue
        if low.startswith(("your task:", "constraints:", "instruction:", "instructions:", "do not return")):
            continue
        if low.startswith(("response:", "final response:", "revised response:", "output:")):
            line = re.sub(
                r"^(response|final response|revised response|output)\s*:\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            low = line.lower()
            if not line:
                continue
        line = _trim_leak_tail(line)
        if not line:
            continue
        low = line.lower()
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

    merged = _trim_leak_tail(merged)
    merged = re.sub(r"\s+", " ", merged).strip()
    merged = re.sub(r"^\s*#{1,6}\s*", "", merged).strip()
    merged = re.sub(r"^[A-Za-z0-9_ ()-]{1,48}:\s*", "", merged).strip()
    merged = re.sub(r"^\[[^\]]+\]\s*", "", merged).strip()
    merged = re.sub(r"\b(assistant|system|query)\s*:\s*", "", merged, flags=re.IGNORECASE).strip()
    merged = re.sub(r"[`*_]{2,}", "", merged).strip()
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


def _fallback_intro(player_input: str, style: str) -> str:
    text = (player_input or "").strip()
    seed = sum(ord(ch) for ch in text) if text else 0
    if style == "strict":
        options = (
            "Hold for a moment",
            "Listen carefully",
            "Understood",
        )
    elif style == "talkative":
        options = (
            "Friend",
            "All right",
            "Very well",
        )
    elif style == "calm":
        options = (
            "Take a breath",
            "Stay with me",
            "All right",
        )
    elif style == "formal":
        options = (
            "Noted",
            "Understood",
            "Proceeding carefully",
        )
    elif style == "mysterious":
        options = (
            "Listen well",
            "Mark this",
            "Hear me",
        )
    else:
        options = (
            "All right",
            "Understood",
            "Listen",
        )
    return options[seed % len(options)]


def _intent_fragment(player_input: str) -> str:
    lowered = (player_input or "").lower()
    patterns = (
        (r"let me (?:into|in to|in)\s+([^.!?,;]+)", "entry to "),
        (r"(?:access|enter)\s+([^.!?,;]+)", "access to "),
        (r"(?:buy|sell|trade)\s+([^.!?,;]+)", "trade on "),
        (r"(?:heal|treat|cure)\s+([^.!?,;]+)", "treatment for "),
        (r"(?:investigate|review|check)\s+([^.!?,;]+)", "review of "),
    )
    for pattern, prefix in patterns:
        m = re.search(pattern, lowered)
        if m:
            core = " ".join(tokenize(m.group(1))[:6]).strip()
            if core:
                return f"{prefix}{core}".strip()

    words = tokenize(player_input)
    if not words:
        return "your request"
    content = [word for word in words if word not in INTENT_STOPWORDS and len(word) > 1]
    if content:
        return " ".join(content[:5])
    return " ".join(words[:5])


def _intent_category(player_input: str) -> str:
    text = (player_input or "").lower()
    if any(k in text for k in ("archive", "gate", "entry", "access", "checkpoint", "permit", "pass")):
        return "access"
    if any(k in text for k in ("buy", "sell", "price", "trade", "discount", "potion", "market", "deal")):
        return "trade"
    if any(k in text for k in ("heal", "cure", "poison", "symptom", "medicine", "treat", "infirmary")):
        return "medical"
    if any(k in text for k in ("investigate", "theft", "evidence", "innocent", "suspect", "report")):
        return "investigation"
    if any(k in text for k in ("prison", "cell", "release", "escape", "detain")):
        return "detention"
    return "general"


def _persona_anchor(persona_keywords: Sequence[str]) -> str:
    preferred = (
        "strict",
        "fair",
        "brief",
        "suspicious",
        "talkative",
        "calm",
        "caring",
        "formal",
        "procedural",
        "mysterious",
        "indirect",
        "precise",
        "practical",
    )
    normalized = [str(x).strip().lower() for x in persona_keywords if str(x).strip()]
    for term in preferred:
        if term in normalized:
            return term
    return ""


def _context_anchor(dynamic_context: str) -> str:
    sentences = _context_detail_sentences(dynamic_context)
    if not sentences:
        return ""
    sentence = sentences[0].strip()
    words = sentence.split()
    if len(words) > 14:
        sentence = " ".join(words[:14]).rstrip(",.;") + "."
    return sentence


def _grounded_style_repair(
    response: str,
    dynamic_context: str,
    persona_keywords: Sequence[str],
) -> str:
    """Deterministic lightweight repair to improve context/persona coverage before fallback."""
    text = sanitize_response(response)
    if not text:
        return ""

    low = text.lower()
    if any(
        marker in low
        for marker in (
            "do not ",
            "return ",
            "rewrite ",
            "your task",
            "assistant",
            "response should",
            "begin your rewrite",
        )
    ):
        return ""
    anchor = _persona_anchor(persona_keywords)
    if not anchor or anchor in low:
        return text

    context_anchor = _context_anchor(dynamic_context)
    merged = f"{text} {context_anchor} I remain {anchor} and focused."
    return sanitize_response(merged)


def _structured_repair_response(
    persona: str,
    dynamic_context: str,
    player_input: str,
    persona_keywords: Sequence[str] = (),
) -> str:
    style = _persona_style(persona)
    category = _intent_category(player_input)
    anchor = _persona_anchor(persona_keywords)
    intent = _intent_fragment(player_input)
    context_anchor = _context_anchor(dynamic_context)

    intro = _fallback_intro(player_input, style)

    if style == "strict":
        if category == "access":
            body = f"I cannot authorize {intent} until verification is complete."
        else:
            body = f"I cannot approve {intent} until verification is complete."
        close = f"Share one verifiable detail and I will proceed as a {anchor or 'strict'} guardian."
        return sanitize_response(f"{intro}, {body} {context_anchor} {close}")

    if style == "talkative":
        body = f"I can help with {intent}, but the terms must stay fair today."
        close = f"Keep it honest and we can settle this quickly in my {anchor or 'merchant'} style."
        return sanitize_response(f"{intro}, {body} {context_anchor} {close}")

    if style == "calm":
        body = f"We can handle {intent} safely, one step at a time."
        close = f"Stay steady and I will guide the next action in a {anchor or 'calm'} voice."
        return sanitize_response(f"{intro}, {body} {context_anchor} {close}")

    if style == "formal":
        body = f"This request about {intent} requires verifiable evidence."
        close = f"Provide concrete details and I will continue with {anchor or 'formal'} precision."
        return sanitize_response(f"{intro}, {body} {context_anchor} {close}")

    if style == "mysterious":
        body = f"This path around {intent} has a cost in these conditions."
        close = f"Choose carefully, because timing matters as much as power to one who stays {anchor or 'mysterious'}."
        return sanitize_response(f"{intro}, {body} {context_anchor} {close}")

    body = f"I can help with {intent} if you share one clear, concrete action."
    close = f"Give one clear detail and I will proceed with {anchor or 'practical'} precision."
    return sanitize_response(f"{intro}, {body} {context_anchor} {close}")


def grounded_fallback_response(
    persona: str,
    dynamic_context: str,
    player_input: str,
    persona_keywords: Sequence[str] = (),
) -> str:
    style = _persona_style(persona)
    intent_category = _intent_category(player_input)
    anchor = _persona_anchor(persona_keywords)
    intent = _intent_fragment(player_input)
    context_anchor = _context_anchor(dynamic_context)

    context_sentence = f"{_fallback_intro(player_input, style)},"

    if style == "strict":
        if intent_category == "access":
            body = f"I cannot authorize {intent} until identity and purpose are verified."
        elif intent_category == "investigation":
            body = f"I cannot close {intent} until evidence is verified."
        else:
            body = f"I cannot approve {intent} until verification is complete."
        return (
            f"{context_sentence} {body} "
            f"{context_anchor} Follow protocol and I will move this forward as a {anchor or 'strict'} guardian."
        )
    if style == "talkative":
        if intent_category == "trade":
            body = f"I can work with {intent}, but terms must remain fair."
        else:
            body = f"I can help with {intent}, but the terms must stay fair."
        return (
            f"{context_sentence} {body} "
            f"{context_anchor} Keep it honest and we can close this deal quickly in my {anchor or 'merchant'} style."
        )
    if style == "calm":
        return (
            f"{context_sentence} we can handle {intent} safely, step by step. "
            f"{context_anchor} "
            f"Stay steady and I will guide the next action in a {anchor or 'calm'} voice."
        )
    if style == "mysterious":
        return (
            f"{context_sentence} this path around {intent} has a cost. "
            f"{context_anchor} "
            f"Choose carefully, because timing matters as much as power to one who stays {anchor or 'mysterious'}."
        )
    if style == "formal":
        return (
            f"{context_sentence} this request about {intent} requires evidence before any conclusion. "
            f"{context_anchor} "
            f"Provide verifiable details and I will continue with {anchor or 'formal'} precision."
        )
    return (
        f"{context_sentence} I can help with {intent} if you state one clear, concrete action. "
        f"{context_anchor} "
        f"Give one clear detail and I will proceed with {anchor or 'practical'} precision."
    )


def build_rewrite_prompt(
    persona: str,
    dynamic_context: str,
    player_input: str,
    draft_response: str,
    persona_keywords: Sequence[str] = (),
) -> str:
    lines: List[str] = []
    lines.append("You are repairing one NPC utterance for in-game dialogue.")
    lines.append(f"Persona: {persona}")
    lines.append("Runtime context:")
    for sentence in _context_detail_sentences(dynamic_context):
        lines.append(f"- {sentence}")
    lines.append(f"Player says: {player_input}")
    lines.append(f"Current draft: {draft_response}")
    persona_hint = [str(k).strip() for k in persona_keywords if str(k).strip()]
    if persona_hint:
        lines.append(f"Persona cue terms: {', '.join(persona_hint[:5])}")
    lines.append("Output requirements:")
    lines.append("- Output only NPC spoken dialogue.")
    lines.append("- Use 2 to 4 short sentences, natural and concise.")
    lines.append("- Keep role-play tone consistent with persona.")
    lines.append("- Include at least one concrete runtime detail when relevant.")
    lines.append("- Prefer direct wording; avoid formulaic openers and repeated stock phrases.")
    lines.append("- Do not include labels, bullets, metadata, JSON, or analysis.")
    lines.append("- Do not echo this instruction text.")
    if persona_hint:
        lines.append("- Use at least one persona cue term naturally.")
    lines.append("Return only the final NPC dialogue.")
    return "\n".join(lines)


def _needs_repair(
    response: str,
    context_cov: float,
    persona_cov: float,
    config: ControlConfig,
    context_keyword_count: int,
) -> Tuple[bool, str]:
    low = (response or "").lower()
    if not response:
        return True, "empty_response"
    if len(tokenize(response)) < max(1, int(config.min_response_tokens)):
        return True, "too_short"
    if any(fragment in low for fragment in BLOCKED_FRAGMENTS):
        return True, "meta_artifact"
    required_context = _effective_context_requirement(config.min_context_coverage, context_keyword_count)
    if context_cov < required_context:
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


def _candidate_sentence_score(response: str) -> float:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(response) if s.strip()]
    count = len(sentences)
    if 2 <= count <= 3:
        return 1.0
    if count == 1 or count == 4:
        return 0.65
    return 0.35


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


def _candidate_persona_style_score(persona: str, response: str) -> float:
    style = _persona_style(persona)
    words = tokenize(response)
    wc = len(words)
    lowered = response.lower()

    if style == "strict":
        cues = ("protocol", "verify", "cannot", "clearance", "authorized")
        return 0.4 + 0.6 * keyword_coverage(lowered, cues)
    if style == "talkative":
        cues = ("deal", "price", "terms", "trade", "fair")
        base = 0.6 if wc >= 28 else max(0.2, wc / 40.0)
        return min(1.0, 0.5 * base + 0.5 * keyword_coverage(lowered, cues))
    if style == "calm":
        cues = ("steady", "carefully", "safe", "breathe", "step")
        return 0.4 + 0.6 * keyword_coverage(lowered, cues)
    if style == "mysterious":
        cues = ("perhaps", "shadow", "cost", "moon", "omen", "price")
        return 0.4 + 0.6 * keyword_coverage(lowered, cues)
    if style == "formal":
        long_words = sum(1 for w in words if len(w) >= 7)
        long_ratio = (long_words / wc) if wc > 0 else 0.0
        cues = ("evidence", "conclusion", "therefore", "proceed", "verify")
        return min(1.0, 0.5 * min(1.0, long_ratio / 0.25) + 0.5 * keyword_coverage(lowered, cues))
    return 0.5


def _candidate_naturalness_score(response: str) -> float:
    if not response:
        return 0.0
    lowered = response.lower().strip()
    score = 1.0

    for phrase in BOILERPLATE_PHRASES:
        if phrase in lowered:
            score -= 0.12

    if any(lowered.startswith(prefix + " ") or lowered.startswith(prefix + ",") for prefix in CANNED_OPENERS):
        score -= 0.08

    if " i will " in lowered and " and i will " in lowered:
        score -= 0.05

    sentences = [s.strip().lower() for s in SENTENCE_SPLIT_RE.split(response) if s.strip()]
    if len(sentences) >= 2:
        starts = [" ".join(tokenize(s)[:3]) for s in sentences if tokenize(s)]
        duplicate_starts = len(starts) - len(set(starts))
        if duplicate_starts > 0:
            score -= min(0.12, 0.06 * duplicate_starts)

    return _clamp(score, 0.0, 1.0)


def _behavior_state_bucket(behavior_state: str | None) -> str:
    state = _normalize_behavior_state(behavior_state)
    if not state:
        return "general"

    if any(token in state for token in ("guard", "detain", "investigat", "combat", "patrol")):
        return "conflict"
    if any(token in state for token in ("quest", "handoff", "assist", "trade", "negotiat", "treat", "repair")):
        return "task"
    if any(token in state for token in ("social", "idle", "chat", "greet", "observe")):
        return "social"
    return "general"


def _state_conditioned_component_weights(
    *,
    config: ControlConfig,
    behavior_state: str | None,
    risk_level: str,
) -> Dict[str, float]:
    weights: Dict[str, float] = {
        "context_cov": 0.34,
        "persona_cov": 0.18,
        "style_score": 0.14,
        "length_score": 0.08,
        "diversity_score": 0.08,
        "sentence_score": 0.04,
        "naturalness_score": 0.14,
    }
    if not config.state_conditioned_scoring_enabled:
        return weights

    bucket = _behavior_state_bucket(behavior_state)
    risk = str(risk_level or "medium").strip().lower()

    if bucket == "conflict":
        weights["context_cov"] += 0.10
        weights["naturalness_score"] += 0.03
        weights["persona_cov"] -= 0.03
        weights["diversity_score"] -= 0.02
    elif bucket == "task":
        weights["context_cov"] += 0.08
        weights["style_score"] += 0.04
        weights["persona_cov"] -= 0.02
    elif bucket == "social":
        weights["persona_cov"] += 0.10
        weights["naturalness_score"] += 0.06
        weights["context_cov"] -= 0.08
        weights["style_score"] -= 0.02

    if risk == "high":
        weights["context_cov"] += 0.04
        weights["style_score"] += 0.03
        weights["naturalness_score"] += 0.02
        weights["diversity_score"] -= 0.03
    elif risk == "low":
        weights["persona_cov"] += 0.03
        weights["naturalness_score"] += 0.03
        weights["context_cov"] -= 0.02

    # Keep every component active, then normalize.
    floor = 0.01
    for key in list(weights.keys()):
        weights[key] = max(floor, float(weights[key]))
    denom = sum(weights.values())
    if denom <= 0.0:
        return {
            "context_cov": 0.34,
            "persona_cov": 0.18,
            "style_score": 0.14,
            "length_score": 0.08,
            "diversity_score": 0.08,
            "sentence_score": 0.04,
            "naturalness_score": 0.14,
        }
    for key in list(weights.keys()):
        weights[key] = weights[key] / denom
    return weights


def _candidate_score(
    response: str,
    persona: str,
    context_keywords: Sequence[str],
    persona_keywords: Sequence[str],
    config: ControlConfig,
    behavior_state: str | None = None,
    risk_level: str = "medium",
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
    sentence_score = _candidate_sentence_score(response)
    style_score = _candidate_persona_style_score(persona, response)
    naturalness_score = _candidate_naturalness_score(response)
    weights = _state_conditioned_component_weights(
        config=config,
        behavior_state=behavior_state,
        risk_level=risk_level,
    )
    return (
        weights["context_cov"] * context_cov
        + weights["persona_cov"] * persona_cov
        + weights["style_score"] * style_score
        + weights["length_score"] * length_score
        + weights["diversity_score"] * diversity_score
        + weights["sentence_score"] * sentence_score
        + weights["naturalness_score"] * naturalness_score
    )


def _is_usable_candidate(
    response: str,
    context_cov: float,
    persona_cov: float,
    score: float,
    config: ControlConfig,
    require_context: bool,
    context_floor: float | None = None,
    persona_floor: float | None = None,
    score_floor: float | None = None,
) -> bool:
    low = (response or "").lower()
    if not response:
        return False
    if any(fragment in low for fragment in BLOCKED_FRAGMENTS):
        return False
    min_tokens = max(4, int(config.min_response_tokens) - 2)
    if len(tokenize(response)) < min_tokens:
        return False
    effective_context_floor = config.relaxed_context_coverage if context_floor is None else context_floor
    effective_persona_floor = config.relaxed_persona_coverage if persona_floor is None else persona_floor
    effective_score_floor = config.relaxed_candidate_score if score_floor is None else score_floor
    if require_context and context_cov < effective_context_floor:
        return False
    if persona_cov < effective_persona_floor:
        return False
    return score >= effective_score_floor


def _is_near_pass_candidate(
    response: str,
    context_cov: float,
    persona_cov: float,
    score: float,
    config: ControlConfig,
    require_context: bool,
    context_floor: float,
    persona_floor: float,
    risk_level: str,
) -> bool:
    if not config.near_pass_enabled:
        return False
    if config.near_pass_block_high_risk and str(risk_level or "").strip().lower() == "high":
        return False

    low = (response or "").lower()
    if not response:
        return False
    if any(fragment in low for fragment in BLOCKED_FRAGMENTS):
        return False

    min_tokens = max(4, int(config.min_response_tokens) - 2)
    if len(tokenize(response)) < min_tokens:
        return False

    if score < float(config.near_pass_score_floor):
        return False

    context_gap = 0.0
    if require_context:
        context_gap = max(0.0, float(context_floor) - float(context_cov))
    persona_gap = max(0.0, float(persona_floor) - float(persona_cov))
    if context_gap > float(config.near_pass_max_context_gap):
        return False
    if persona_gap > float(config.near_pass_max_persona_gap):
        return False
    return True


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


def _effective_rewrite_budget(raw_score: float, config: ControlConfig) -> int:
    max_candidates = max(1, int(config.rewrite_candidates))
    if raw_score >= float(config.adaptive_high_confidence_score):
        return max(1, min(max_candidates, int(config.adaptive_high_confidence_rewrites)))
    if raw_score >= float(config.adaptive_mid_confidence_score):
        return max(1, min(max_candidates, int(config.adaptive_mid_confidence_rewrites)))
    return max(1, min(max_candidates, int(config.adaptive_low_confidence_rewrites)))


# Novel: source-tier confidence priors for ECE calibration.
_SOURCE_CONFIDENCE_PRIOR: Dict[str, float] = {
    "raw": 1.00,
    "raw_relaxed": 0.85,
    "raw_adaptive": 0.80,
    "raw_near_pass": 0.75,
    "raw_grounded_repair": 0.70,
    "structured_repair": 0.62,
    "rewritten": 0.78,
    "rewritten_relaxed": 0.68,
    "rewritten_near_pass": 0.65,
    "rewritten_grounded_repair": 0.63,
    "structured_recovery": 0.58,
    "structured_best_effort": 0.55,
    "raw_best_effort": 0.52,
    "rewritten_best_effort": 0.50,
    "fallback": 0.30,
}

# Novel: behavior-state bucket confidence multipliers.
_BUCKET_CONFIDENCE_FACTOR: Dict[str, float] = {
    "conflict": 0.92,  # High-stakes states are harder; discount confidence slightly.
    "task": 0.97,
    "social": 1.00,
    "general": 0.98,
}


def _compute_response_confidence(
    source: str,
    candidate_score: float,
    behavior_state: str | None,
) -> float:
    """Compute a heuristic confidence in [0, 1] for the final accepted response.

    The confidence is used for per-state calibration (ECE) and reliability analysis.
    It blends:
    - A source-tier prior (raw > repaired > fallback)
    - The raw candidate score
    - A state-bucket discount (conflict states reduce confidence)
    """
    source_prior = _SOURCE_CONFIDENCE_PRIOR.get(str(source), 0.50)
    score_clipped = _clamp(float(candidate_score), 0.0, 1.0)
    bucket = _behavior_state_bucket(behavior_state)
    bucket_factor = _BUCKET_CONFIDENCE_FACTOR.get(bucket, 0.98)
    # Blend prior (0.55 weight) and score (0.45 weight), then apply bucket factor.
    raw_conf = 0.55 * source_prior + 0.45 * score_clipped
    return _clamp(raw_conf * bucket_factor, 0.0, 1.0)


def control_response(
    raw_response: str,
    persona: str,
    dynamic_context: str,
    player_input: str,
    context_keywords: Sequence[str],
    persona_keywords: Sequence[str],
    rewrite_fn: Callable[[str, int, float], str] | None = None,
    config: ControlConfig = ControlConfig(),
    behavior_state: str | None = None,
    raw_latency_ms: float | None = None,
    timeout_s: float | None = None,
) -> ControlResult:
    config = _apply_behavior_profile(config, behavior_state)
    risk_level = _estimate_intent_risk(player_input, dynamic_context)
    config = _apply_intent_risk_profile(config, risk_level)
    config = _apply_latency_profile(config, raw_latency_ms=raw_latency_ms, timeout_s=timeout_s)
    active_context_keywords = _select_active_context_keywords(
        context_keywords=context_keywords,
        player_input=player_input,
        dynamic_context=dynamic_context,
        risk_level=risk_level,
        config=config,
    )
    context_keyword_count = len(active_context_keywords)
    cleaned = sanitize_response(raw_response)
    context_cov = keyword_coverage(cleaned, active_context_keywords)
    persona_cov = keyword_coverage(cleaned, persona_keywords)
    raw_score = _candidate_score(
        cleaned,
        persona,
        active_context_keywords,
        persona_keywords,
        config=config,
        behavior_state=behavior_state,
        risk_level=risk_level,
    )
    repair_needed, reason = _needs_repair(
        cleaned,
        context_cov,
        persona_cov,
        config,
        context_keyword_count=context_keyword_count,
    )
    if not repair_needed:
        return ControlResult(
            response=cleaned,
            source="raw",
            context_coverage=context_cov,
            persona_coverage=persona_cov,
            repaired=False,
            repair_reason="",
            confidence=_compute_response_confidence("raw", raw_score, behavior_state),
        )
    context_required = bool(context_keywords)
    relaxed_context_floor = (
        _effective_context_requirement(config.relaxed_context_coverage, context_keyword_count)
        if context_required
        else 0.0
    )
    adaptive_context_floor = (
        _effective_context_requirement(config.adaptive_context_coverage, context_keyword_count)
        if context_required
        else 0.0
    )
    if config.allow_relaxed_acceptance and _is_usable_candidate(
        cleaned,
        context_cov=context_cov,
        persona_cov=persona_cov,
        score=raw_score,
        config=config,
        require_context=context_required,
        context_floor=relaxed_context_floor,
    ):
        return ControlResult(
            response=cleaned,
            source="raw_relaxed",
            context_coverage=context_cov,
            persona_coverage=persona_cov,
            repaired=False,
            repair_reason=reason,
            confidence=_compute_response_confidence("raw_relaxed", raw_score, behavior_state),
        )

    if config.adaptive_acceptance_enabled and _is_usable_candidate(
        cleaned,
        context_cov=context_cov,
        persona_cov=persona_cov,
        score=raw_score,
        config=config,
        require_context=False,
        context_floor=adaptive_context_floor,
        persona_floor=config.adaptive_persona_coverage,
        score_floor=config.adaptive_candidate_score,
    ):
        return ControlResult(
            response=cleaned,
            source="raw_adaptive",
            context_coverage=context_cov,
            persona_coverage=persona_cov,
            repaired=False,
            repair_reason=reason,
            confidence=_compute_response_confidence("raw_adaptive", raw_score, behavior_state),
        )

    if _is_near_pass_candidate(
        cleaned,
        context_cov=context_cov,
        persona_cov=persona_cov,
        score=raw_score,
        config=config,
        require_context=context_required,
        context_floor=relaxed_context_floor,
        persona_floor=config.relaxed_persona_coverage,
        risk_level=risk_level,
    ):
        return ControlResult(
            response=cleaned,
            source="raw_near_pass",
            context_coverage=context_cov,
            persona_coverage=persona_cov,
            repaired=False,
            repair_reason=reason,
            confidence=_compute_response_confidence("raw_near_pass", raw_score, behavior_state),
        )

    if cleaned:
        grounded_candidate = _grounded_style_repair(
            response=cleaned,
            dynamic_context=dynamic_context,
            persona_keywords=persona_keywords,
        )
        if grounded_candidate:
            grounded_context_cov = keyword_coverage(grounded_candidate, active_context_keywords)
            grounded_persona_cov = keyword_coverage(grounded_candidate, persona_keywords)
            grounded_score = _candidate_score(
                grounded_candidate,
                persona,
                active_context_keywords,
                persona_keywords,
                config=config,
                behavior_state=behavior_state,
                risk_level=risk_level,
            )
            if _is_usable_candidate(
                grounded_candidate,
                context_cov=grounded_context_cov,
                persona_cov=grounded_persona_cov,
                score=grounded_score,
                config=config,
                require_context=context_required,
                context_floor=relaxed_context_floor,
            ):
                return ControlResult(
                    response=grounded_candidate,
                    source="raw_grounded_repair",
                    context_coverage=grounded_context_cov,
                    persona_coverage=grounded_persona_cov,
                    repaired=True,
                    repair_reason=reason,
                    confidence=_compute_response_confidence("raw_grounded_repair", grounded_score, behavior_state),
                )

    structured_candidate = _structured_repair_response(
        persona=persona,
        dynamic_context=dynamic_context,
        player_input=player_input,
        persona_keywords=persona_keywords,
    )
    if structured_candidate:
        structured_context_cov = keyword_coverage(structured_candidate, active_context_keywords)
        structured_persona_cov = keyword_coverage(structured_candidate, persona_keywords)
        structured_score = _candidate_score(
            structured_candidate,
            persona,
            active_context_keywords,
            persona_keywords,
            config=config,
            behavior_state=behavior_state,
            risk_level=risk_level,
        )
        if _is_usable_candidate(
            structured_candidate,
            context_cov=structured_context_cov,
            persona_cov=structured_persona_cov,
            score=structured_score,
            config=config,
            require_context=context_required,
            context_floor=relaxed_context_floor,
        ):
            return ControlResult(
                response=structured_candidate,
                source="structured_repair",
                context_coverage=structured_context_cov,
                persona_coverage=structured_persona_cov,
                repaired=True,
                repair_reason=reason,
                confidence=_compute_response_confidence("structured_repair", structured_score, behavior_state),
            )

    best_rewrite_passing: tuple[str, float, float, float] | None = None
    best_rewrite_candidate: tuple[str, float, float, float] | None = None

    if config.enable_rewrite and rewrite_fn is not None:
        rewrite_prompt = build_rewrite_prompt(
            persona,
            dynamic_context,
            player_input,
            cleaned or raw_response,
            persona_keywords=persona_keywords,
        )
        rewrite_budget = _effective_rewrite_budget(raw_score, config)
        low_confidence_band = raw_score < float(config.adaptive_mid_confidence_score)
        for attempt_idx, temp in enumerate(_rewrite_temperatures(config)[:rewrite_budget]):
            rewritten_raw = rewrite_fn(rewrite_prompt, config.rewrite_max_tokens, temp)
            rewritten = sanitize_response(rewritten_raw)
            if not rewritten:
                continue
            rewrite_context_cov = keyword_coverage(rewritten, active_context_keywords)
            rewrite_persona_cov = keyword_coverage(rewritten, persona_keywords)
            rewrite_score = _candidate_score(
                rewritten,
                persona,
                active_context_keywords,
                persona_keywords,
                config=config,
                behavior_state=behavior_state,
                risk_level=risk_level,
            )
            if best_rewrite_candidate is None or rewrite_score > best_rewrite_candidate[3]:
                best_rewrite_candidate = (
                    rewritten,
                    rewrite_context_cov,
                    rewrite_persona_cov,
                    rewrite_score,
                )
            rewrite_needed, _ = _needs_repair(
                rewritten,
                rewrite_context_cov,
                rewrite_persona_cov,
                config,
                context_keyword_count=context_keyword_count,
            )

            if (
                config.low_confidence_retry_requires_gain
                and low_confidence_band
                and rewrite_budget > 1
                and attempt_idx == 0
                and rewrite_needed
            ):
                score_gain = rewrite_score - raw_score
                coverage_gain = max(
                    rewrite_context_cov - context_cov,
                    rewrite_persona_cov - persona_cov,
                )
                if (
                    score_gain < float(config.low_confidence_retry_min_score_gain)
                    and coverage_gain < float(config.low_confidence_retry_min_coverage_gain)
                ):
                    break
            if not rewrite_needed:
                if best_rewrite_passing is None or rewrite_score > best_rewrite_passing[3]:
                    best_rewrite_passing = (
                        rewritten,
                        rewrite_context_cov,
                        rewrite_persona_cov,
                        rewrite_score,
                    )
                    if config.early_stop_on_pass and rewrite_score >= config.early_stop_score:
                        break

    if best_rewrite_passing is not None:
        return ControlResult(
            response=best_rewrite_passing[0],
            source="rewritten",
            context_coverage=best_rewrite_passing[1],
            persona_coverage=best_rewrite_passing[2],
            repaired=True,
            repair_reason=reason,
            confidence=_compute_response_confidence("rewritten", best_rewrite_passing[3], behavior_state),
        )

    if config.allow_relaxed_acceptance and best_rewrite_candidate is not None:
        if _is_usable_candidate(
            best_rewrite_candidate[0],
            context_cov=best_rewrite_candidate[1],
            persona_cov=best_rewrite_candidate[2],
            score=best_rewrite_candidate[3],
            config=config,
            require_context=context_required,
        ):
            return ControlResult(
                response=best_rewrite_candidate[0],
                source="rewritten_relaxed",
                context_coverage=best_rewrite_candidate[1],
                persona_coverage=best_rewrite_candidate[2],
                repaired=True,
                repair_reason=reason,
                confidence=_compute_response_confidence("rewritten_relaxed", best_rewrite_candidate[3], behavior_state),
            )

    if best_rewrite_candidate is not None and _is_near_pass_candidate(
        best_rewrite_candidate[0],
        context_cov=best_rewrite_candidate[1],
        persona_cov=best_rewrite_candidate[2],
        score=best_rewrite_candidate[3],
        config=config,
        require_context=context_required,
        context_floor=relaxed_context_floor,
        persona_floor=config.relaxed_persona_coverage,
        risk_level=risk_level,
    ):
        return ControlResult(
            response=best_rewrite_candidate[0],
            source="rewritten_near_pass",
            context_coverage=best_rewrite_candidate[1],
            persona_coverage=best_rewrite_candidate[2],
            repaired=True,
            repair_reason=reason,
            confidence=_compute_response_confidence("rewritten_near_pass", best_rewrite_candidate[3], behavior_state),
        )

    if best_rewrite_candidate is not None:
        rewrite_grounded = _grounded_style_repair(
            response=best_rewrite_candidate[0],
            dynamic_context=dynamic_context,
            persona_keywords=persona_keywords,
        )
        if rewrite_grounded:
            rewrite_grounded_context_cov = keyword_coverage(rewrite_grounded, context_keywords)
            rewrite_grounded_persona_cov = keyword_coverage(rewrite_grounded, persona_keywords)
            rewrite_grounded_score = _candidate_score(
                rewrite_grounded,
                persona,
                active_context_keywords,
                persona_keywords,
                config=config,
                behavior_state=behavior_state,
                risk_level=risk_level,
            )
            if _is_usable_candidate(
                rewrite_grounded,
                context_cov=rewrite_grounded_context_cov,
                persona_cov=rewrite_grounded_persona_cov,
                score=rewrite_grounded_score,
                config=config,
                require_context=context_required,
                context_floor=relaxed_context_floor,
            ):
                return ControlResult(
                    response=rewrite_grounded,
                    source="rewritten_grounded_repair",
                    context_coverage=rewrite_grounded_context_cov,
                    persona_coverage=rewrite_grounded_persona_cov,
                    repaired=True,
                    repair_reason=reason,
                    confidence=_compute_response_confidence("rewritten_grounded_repair", rewrite_grounded_score, behavior_state),
                )

    fallback = grounded_fallback_response(
        persona,
        dynamic_context,
        player_input,
        persona_keywords=persona_keywords,
    )
    fallback = sanitize_response(fallback)
    fallback_context_cov = keyword_coverage(fallback, active_context_keywords)
    fallback_persona_cov = keyword_coverage(fallback, persona_keywords)

    fallback_score = _candidate_score(
        fallback,
        persona,
        active_context_keywords,
        persona_keywords,
        config=config,
        behavior_state=behavior_state,
        risk_level=risk_level,
    )

    if structured_candidate:
        structured_context_cov = keyword_coverage(structured_candidate, active_context_keywords)
        structured_persona_cov = keyword_coverage(structured_candidate, persona_keywords)
        structured_score = _candidate_score(
            structured_candidate,
            persona,
            active_context_keywords,
            persona_keywords,
            config=config,
            behavior_state=behavior_state,
            risk_level=risk_level,
        )
        if reason == "empty_response" and structured_score >= 0.0:
            return ControlResult(
                response=structured_candidate,
                source="structured_recovery",
                context_coverage=structured_context_cov,
                persona_coverage=structured_persona_cov,
                repaired=True,
                repair_reason=reason,
                confidence=_compute_response_confidence("structured_recovery", structured_score, behavior_state),
            )
        if structured_score >= (fallback_score + config.min_rewrite_gain):
            return ControlResult(
                response=structured_candidate,
                source="structured_best_effort",
                context_coverage=structured_context_cov,
                persona_coverage=structured_persona_cov,
                repaired=True,
                repair_reason=reason,
                confidence=_compute_response_confidence("structured_best_effort", structured_score, behavior_state),
            )

    if config.allow_best_effort_rewrite and _is_usable_candidate(
        cleaned,
        context_cov=context_cov,
        persona_cov=persona_cov,
        score=raw_score,
        config=config,
        require_context=context_required,
    ):
        if raw_score >= (fallback_score + config.min_rewrite_gain):
            return ControlResult(
                response=cleaned,
                source="raw_best_effort",
                context_coverage=context_cov,
                persona_coverage=persona_cov,
                repaired=True,
                repair_reason=reason,
                confidence=_compute_response_confidence("raw_best_effort", raw_score, behavior_state),
            )

    if config.allow_best_effort_rewrite and best_rewrite_candidate is not None:
        if best_rewrite_candidate[3] >= (fallback_score + config.min_rewrite_gain):
            return ControlResult(
                response=best_rewrite_candidate[0],
                source="rewritten_best_effort",
                context_coverage=best_rewrite_candidate[1],
                persona_coverage=best_rewrite_candidate[2],
                repaired=True,
                repair_reason=reason,
                confidence=_compute_response_confidence("rewritten_best_effort", best_rewrite_candidate[3], behavior_state),
            )

    return ControlResult(
        response=fallback or "I need a clearer request before I can respond in character.",
        source="fallback",
        context_coverage=fallback_context_cov,
        persona_coverage=fallback_persona_cov,
        repaired=True,
        repair_reason=reason,
        confidence=_compute_response_confidence("fallback", fallback_score, behavior_state),
    )
