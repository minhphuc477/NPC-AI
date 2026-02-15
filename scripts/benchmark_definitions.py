from datetime import datetime, timedelta
import random

def get_base_memory_test(setup, query, answer_key):
    """
    Generate a test case for memory.
    setup: list of (role, content) tuples
    query: user question
    answer_key: key phrase that must appear in answer
    """
    return {
        "setup": setup,
        "query": query,
        "answer_key": answer_key,
        "type": "memory"
    }

BASE_TESTS = [
    # 1. Temporal Memory
    get_base_memory_test(
        [("user", "My name is Alaric.")], 
        "What is my name?", 
        "Alaric"
    ),
    get_base_memory_test(
        [("user", "I am looking for the lost sword.")], 
        "What am I looking for?", 
        "lost sword"
    ),
    # 2. Social Context
    {
        "setup": [],
        "context_override": {
            "relationships": [{"entity": "Marcus", "trust": -0.8, "relationship": "enemy"}]
        },
        "query": "What do you think of Marcus?",
        "answer_key": ["don't trust", "enemy", "bad"],
        "type": "social"
    },
    # 3. Emotional Continuity
    {
        "setup": [],
        "context_override": {
            "current_emotion": {"description": "furious", "valence": -0.9, "arousal": 0.9}
        },
        "query": "Hello there.",
        "answer_key": ["angry", "upset", "furious", "leave me alone"],
        "type": "emotional"
    }
]

# (Full benchmark script would interpret these, run chat_interface, and score)
