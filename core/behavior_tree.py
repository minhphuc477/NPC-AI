"""
BD-NSCA Behavior Tree v2

Enhanced with:
- Emotion-driven conditions (from EmotionalStateMachine)
- Dynamic trust-based branching
- Expanded social behaviors (greeting, trading, quest-giving)
- State-dependent dialogue modes
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Node(ABC):
    """Abstract base class for behavior tree nodes."""
    
    @abstractmethod
    def tick(self, blackboard: Dict[str, Any]) -> str:
        """Execute node logic and return status.
        
        Args:
            blackboard: Shared state dictionary for the behavior tree
            
        Returns:
            Status string: "SUCCESS", "FAILURE", or "RUNNING"
        """
        pass


class Selector(Node):
    def __init__(self, children):
        self.children = children

    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == "SUCCESS" or status == "RUNNING":
                return status
        return "FAILURE"


class Sequence(Node):
    def __init__(self, children):
        self.children = children

    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == "FAILURE" or status == "RUNNING":
                return status
        return "SUCCESS"


class Condition(Node):
    def __init__(self, condition_func):
        self.condition_func = condition_func

    def tick(self, blackboard):
        if self.condition_func(blackboard):
            return "SUCCESS"
        return "FAILURE"


class Action(Node):
    def __init__(self, action_func):
        self.action_func = action_func

    def tick(self, blackboard):
        return self.action_func(blackboard)


# --- Condition Functions ---

def is_hp_low(blackboard):
    return blackboard.get("hp", 100) < 30

def is_player_nearby(blackboard):
    return blackboard.get("is_player_nearby", False)

def is_player_talking(blackboard):
    return blackboard.get("is_player_talking", False)

# --- NEW: Emotion-driven conditions ---

def is_mood_hostile(blackboard):
    """Triggered when anger exceeds threshold (from EmotionalStateMachine)."""
    return blackboard.get("is_mood_hostile", False)

def is_mood_afraid(blackboard):
    """Triggered when fear exceeds threshold."""
    return blackboard.get("is_mood_afraid", False)

def has_high_trust(blackboard):
    """Player has built enough trust for special interactions."""
    return blackboard.get("has_high_trust", False)

def has_low_trust(blackboard):
    """Player is not trusted — restrict interactions."""
    return blackboard.get("has_low_trust", False)

def is_surprised(blackboard):
    """NPC is surprised by something unexpected."""
    return blackboard.get("is_surprised", False)

def is_mood_happy(blackboard):
    """NPC is in a good mood."""
    return blackboard.get("is_mood_happy", False)


# --- Action Functions ---

def action_flee(blackboard):
    blackboard["current_action"] = "Fleeing"
    blackboard["mood_state"] = "Afraid"
    logger.info("ACTION: Fleeing!")
    return "SUCCESS"

def action_attack(blackboard):
    blackboard["current_action"] = "Attacking"
    blackboard["mood_state"] = "Hostile"
    logger.info("ACTION: Attacking!")
    return "SUCCESS"

def action_dialogue(blackboard):
    blackboard["current_action"] = "Talking"
    return "SUCCESS"

def action_patrol(blackboard):
    blackboard["current_action"] = "Patrolling"
    blackboard["mood_state"] = "Neutral"
    logger.info("ACTION: Patrolling...")
    return "SUCCESS"

def action_idle(blackboard):
    blackboard["current_action"] = "Idle"
    return "SUCCESS"

# --- NEW: Emotion-driven actions ---

def action_threaten(blackboard):
    """NPC threatens the player when hostile but not in combat."""
    blackboard["current_action"] = "Threatening"
    blackboard["mood_state"] = "Hostile"
    logger.info("ACTION: Threatening player!")
    return "SUCCESS"

def action_cower(blackboard):
    """NPC cowers when afraid but can't flee."""
    blackboard["current_action"] = "Cowering"
    blackboard["mood_state"] = "Afraid"
    logger.info("ACTION: Cowering in fear!")
    return "SUCCESS"

def action_greet_warmly(blackboard):
    """NPC greets warmly when happy and trusting."""
    blackboard["current_action"] = "Greeting"
    blackboard["mood_state"] = "Friendly"
    logger.info("ACTION: Warm greeting!")
    return "SUCCESS"

def action_offer_quest(blackboard):
    """NPC offers quest when trust is high enough."""
    blackboard["current_action"] = "Offering Quest"
    logger.info("ACTION: Offering quest to trusted player!")
    return "SUCCESS"

def action_refuse_interaction(blackboard):
    """NPC refuses to interact when trust is too low."""
    blackboard["current_action"] = "Refusing"
    blackboard["mood_state"] = "Suspicious"
    logger.info("ACTION: Refusing interaction — low trust!")
    return "SUCCESS"

def action_react_surprised(blackboard):
    """NPC reacts to an unexpected event."""
    blackboard["current_action"] = "Reacting"
    blackboard["mood_state"] = "Surprised"
    logger.info("ACTION: Surprised reaction!")
    return "SUCCESS"


def create_npc_behavior_tree():
    """Create the enhanced NPC behavior tree with emotional branching.

    Tree structure:
        Root (Selector)
        +-- Surprise Branch: react when surprised
        +-- Combat Branch: flee or attack
        +-- Emotional Branch: threat/cower based on mood
        +-- Social Branch: greet/quest/dialogue/refuse based on trust
        +-- Idle
    """
    # Surprise Branch — interrupts everything
    surprise_seq = Sequence([
        Condition(is_surprised),
        Action(action_react_surprised),
    ])

    # Combat Branch
    combat_seq = Sequence([
        Condition(lambda bb: bb.get("is_combat", False)),
        Selector([
            Sequence([Condition(is_hp_low), Action(action_flee)]),
            Action(action_attack),
        ]),
    ])

    # Emotional Branch — mood-driven when not in combat
    emotional_seq = Selector([
        Sequence([
            Condition(is_mood_hostile),
            Condition(lambda bb: not bb.get("is_combat", False)),
            Action(action_threaten),
        ]),
        Sequence([
            Condition(is_mood_afraid),
            Selector([
                Sequence([Condition(is_hp_low), Action(action_flee)]),
                Action(action_cower),
            ]),
        ]),
    ])

    # Social Branch — trust-driven interactions
    social_seq = Sequence([
        Condition(is_player_nearby),
        Selector([
            # High trust: warm greeting then quest/dialogue
            Sequence([
                Condition(has_high_trust),
                Condition(is_player_talking),
                Selector([
                    Sequence([
                        Condition(lambda bb: bb.get("has_available_quest", False)),
                        Action(action_offer_quest),
                    ]),
                    Action(action_dialogue),
                ]),
            ]),
            # Low trust: refuse or minimal interaction
            Sequence([
                Condition(has_low_trust),
                Condition(is_player_talking),
                Action(action_refuse_interaction),
            ]),
            # Normal trust: standard dialogue
            Sequence([
                Condition(is_player_talking),
                Selector([
                    Sequence([Condition(is_mood_happy), Action(action_greet_warmly)]),
                    Action(action_dialogue),
                ]),
            ]),
            # Player nearby but not talking — patrol
            Action(action_patrol),
        ]),
    ])

    # Root Selector
    root = Selector([
        surprise_seq,
        combat_seq,
        emotional_seq,
        social_seq,
        Action(action_idle),
    ])

    return root
