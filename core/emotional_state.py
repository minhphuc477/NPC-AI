"""
BD-NSCA Emotional State Machine

Dynamic emotion tracker for NPCs with:
- 5 emotion axes: joy, anger, fear, trust, surprise (0.0-1.0)
- State transitions driven by player interaction sentiment
- Natural decay toward baseline emotional state
- Integration with behavior tree for emotion-driven decisions
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EmotionalState:
    """Represents a 5-axis emotional state for an NPC."""
    joy: float = 0.3
    anger: float = 0.1
    fear: float = 0.1
    trust: float = 0.4
    surprise: float = 0.1

    # Personality baseline — emotions decay toward these values
    _baseline_joy: float = 0.3
    _baseline_anger: float = 0.1
    _baseline_fear: float = 0.1
    _baseline_trust: float = 0.4
    _baseline_surprise: float = 0.1

    def clamp(self) -> None:
        """Clamp all axes to [0.0, 1.0]."""
        self.joy = max(0.0, min(1.0, self.joy))
        self.anger = max(0.0, min(1.0, self.anger))
        self.fear = max(0.0, min(1.0, self.fear))
        self.trust = max(0.0, min(1.0, self.trust))
        self.surprise = max(0.0, min(1.0, self.surprise))

    def to_dict(self) -> Dict[str, float]:
        """Return emotion axes as a dict (excludes baselines)."""
        return {
            "joy": round(self.joy, 3),
            "anger": round(self.anger, 3),
            "fear": round(self.fear, 3),
            "trust": round(self.trust, 3),
            "surprise": round(self.surprise, 3),
        }

    @property
    def dominant_emotion(self) -> str:
        """Return the strongest emotion axis."""
        d = self.to_dict()
        return max(d, key=d.get)

    @property
    def mood_label(self) -> str:
        """Map dominant emotion to a human-readable mood label."""
        dominant = self.dominant_emotion
        intensity = getattr(self, dominant)

        labels = {
            "joy": ["Content", "Happy", "Elated"],
            "anger": ["Irritated", "Angry", "Furious"],
            "fear": ["Uneasy", "Afraid", "Terrified"],
            "trust": ["Cautious", "Trusting", "Devoted"],
            "surprise": ["Curious", "Surprised", "Astonished"],
        }

        if intensity < 0.35:
            idx = 0
        elif intensity < 0.7:
            idx = 1
        else:
            idx = 2

        return labels.get(dominant, ["Neutral", "Neutral", "Neutral"])[idx]


# --- Personality presets ---

PERSONALITY_PRESETS: Dict[str, Dict[str, float]] = {
    "stern": {"joy": 0.1, "anger": 0.3, "fear": 0.05, "trust": 0.2, "surprise": 0.05},
    "friendly": {"joy": 0.5, "anger": 0.05, "fear": 0.05, "trust": 0.6, "surprise": 0.2},
    "suspicious": {"joy": 0.1, "anger": 0.2, "fear": 0.3, "trust": 0.1, "surprise": 0.15},
    "caring": {"joy": 0.4, "anger": 0.05, "fear": 0.1, "trust": 0.6, "surprise": 0.1},
    "shrewd": {"joy": 0.3, "anger": 0.1, "fear": 0.1, "trust": 0.3, "surprise": 0.2},
    "brave": {"joy": 0.3, "anger": 0.2, "fear": 0.02, "trust": 0.4, "surprise": 0.1},
    "cowardly": {"joy": 0.1, "anger": 0.05, "fear": 0.6, "trust": 0.15, "surprise": 0.3},
    "wise": {"joy": 0.3, "anger": 0.05, "fear": 0.05, "trust": 0.5, "surprise": 0.1},
    "aggressive": {"joy": 0.05, "anger": 0.6, "fear": 0.05, "trust": 0.1, "surprise": 0.1},
    "timid": {"joy": 0.2, "anger": 0.02, "fear": 0.4, "trust": 0.2, "surprise": 0.3},
    "neutral": {"joy": 0.2, "anger": 0.2, "fear": 0.2, "trust": 0.2, "surprise": 0.2},
}


class EmotionalStateMachine:
    """Manages emotional state transitions for an NPC.

    Usage:
        esm = EmotionalStateMachine.from_personality("stern")
        esm.process_interaction(sentiment=-0.5, threat_level=0.3)
        mood = esm.state.mood_label  # e.g. "Angry"
        esm.decay(steps=1)  # Emotions drift toward baseline
    """

    def __init__(self, state: EmotionalState):
        self.state = state
        self.decay_rate = 0.1  # How fast emotions return to baseline

    @classmethod
    def from_personality(cls, personality: str) -> EmotionalStateMachine:
        """Create an ESM from a named personality preset."""
        preset = PERSONALITY_PRESETS.get(personality, PERSONALITY_PRESETS["neutral"])
        state = EmotionalState(
            joy=preset["joy"],
            anger=preset["anger"],
            fear=preset["fear"],
            trust=preset["trust"],
            surprise=preset["surprise"],
            _baseline_joy=preset["joy"],
            _baseline_anger=preset["anger"],
            _baseline_fear=preset["fear"],
            _baseline_trust=preset["trust"],
            _baseline_surprise=preset["surprise"],
        )
        return cls(state)

    @classmethod
    def from_traits(cls, traits: list) -> EmotionalStateMachine:
        """Create an ESM by averaging personality presets from a list of traits."""
        if not traits:
            return cls.from_personality("neutral")

        # Average the matched presets
        sums = {"joy": 0, "anger": 0, "fear": 0, "trust": 0, "surprise": 0}
        count = 0
        for trait in traits:
            trait_lower = trait.lower()
            if trait_lower in PERSONALITY_PRESETS:
                for k, v in PERSONALITY_PRESETS[trait_lower].items():
                    sums[k] += v
                count += 1

        if count == 0:
            return cls.from_personality("neutral")

        avg = {k: v / count for k, v in sums.items()}
        state = EmotionalState(
            joy=avg["joy"],
            anger=avg["anger"],
            fear=avg["fear"],
            trust=avg["trust"],
            surprise=avg["surprise"],
            _baseline_joy=avg["joy"],
            _baseline_anger=avg["anger"],
            _baseline_fear=avg["fear"],
            _baseline_trust=avg["trust"],
            _baseline_surprise=avg["surprise"],
        )
        return cls(state)

    def process_interaction(
        self,
        sentiment: float = 0.0,
        threat_level: float = 0.0,
        surprise_factor: float = 0.0,
        trust_change: float = 0.0,
    ) -> None:
        """Update emotional state based on an interaction.

        Args:
            sentiment: -1.0 (hostile) to 1.0 (friendly)
            threat_level: 0.0 (safe) to 1.0 (dangerous)
            surprise_factor: 0.0 (expected) to 1.0 (completely unexpected)
            trust_change: direct trust adjustment (-1.0 to 1.0)
        """
        s = self.state

        # Joy: increases with positive sentiment, decreases with threats
        s.joy += sentiment * 0.2 - threat_level * 0.1

        # Anger: increases with negative sentiment and threats
        if sentiment < 0:
            s.anger += abs(sentiment) * 0.25
        s.anger += threat_level * 0.15

        # Fear: increases with threats, decreases with trust
        s.fear += threat_level * 0.3 - s.trust * 0.05

        # Trust: directly adjusted + sentiment influence
        s.trust += trust_change * 0.2 + sentiment * 0.1

        # Surprise: spikes with unexpected events, decays fast
        s.surprise += surprise_factor * 0.4

        s.clamp()

    def decay(self, steps: int = 1) -> None:
        """Decay emotions toward personality baseline over time.

        Call once per game tick or conversation idle period.
        """
        s = self.state
        rate = self.decay_rate * steps

        s.joy += (s._baseline_joy - s.joy) * rate
        s.anger += (s._baseline_anger - s.anger) * rate
        s.fear += (s._baseline_fear - s.fear) * rate
        s.trust += (s._baseline_trust - s.trust) * rate
        s.surprise += (s._baseline_surprise - s.surprise) * rate

        s.clamp()

    def get_behavior_modifiers(self) -> Dict[str, bool]:
        """Return flags for behavior tree integration.

        These can be injected into the BT blackboard.
        """
        s = self.state
        return {
            "is_mood_hostile": s.anger > 0.5,
            "is_mood_afraid": s.fear > 0.5,
            "is_mood_happy": s.joy > 0.5,
            "has_high_trust": s.trust > 0.6,
            "has_low_trust": s.trust < 0.25,
            "is_surprised": s.surprise > 0.4,
        }
