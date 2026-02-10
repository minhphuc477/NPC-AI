
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.behavior_tree import create_npc_behavior_tree
from core.prompt_builder import PromptBuilder

def test_bt_logic():
    print("Testing Behavior Tree Logic...")
    bt = create_npc_behavior_tree()
    
    # Test 1: Combat - Low HP -> Flee
    print("\nTest 1: Combat (Low HP)")
    bb1 = {"is_combat": True, "hp": 20, "is_player_nearby": True}
    bt.tick(bb1)
    print(f"Action: {bb1.get('current_action')}")
    assert bb1.get("current_action") == "Fleeing"

    # Test 2: Combat - High HP -> Attack
    print("\nTest 2: Combat (High HP)")
    bb2 = {"is_combat": True, "hp": 100, "is_player_nearby": True}
    bt.tick(bb2)
    print(f"Action: {bb2.get('current_action')}")
    assert bb2.get("current_action") == "Attacking"

    # Test 3: Social - Player Talking -> Talk
    print("\nTest 3: Social (Player Talking)")
    bb3 = {"is_combat": False, "is_player_nearby": True, "is_player_talking": True}
    bt.tick(bb3)
    print(f"Action: {bb3.get('current_action')}")
    assert bb3.get("current_action") == "Talking"

    # Test 4: Social - Player Nearby (Not Talking) -> Patrol
    print("\nTest 4: Social (Player Nearby, Not Talking)")
    bb4 = {"is_combat": False, "is_player_nearby": True, "is_player_talking": False}
    bt.tick(bb4)
    print(f"Action: {bb4.get('current_action')}")
    assert bb4.get("current_action") == "Patrolling"

    # Test 5: Idle
    print("\nTest 5: Idle")
    bb5 = {"is_combat": False, "is_player_nearby": False}
    bt.tick(bb5)
    print(f"Action: {bb5.get('current_action')}")
    assert bb5.get("current_action") == "Idle"
    
    print("\n✓ All BT Logic Tests Passed")

def test_prompt_builder():
    print("\nTesting Prompt Builder...")
    builder = PromptBuilder(use_advanced_format=True)
    
    npc_data = {"name": "Guard", "persona": "A strict guard."}
    game_state = {
        "behavior_state": "Patrolling",
        "mood_state": "Suspicious",
        "health_state": "Healthy",
        "location": "Gate",
        "time_of_day": "Night",
        "nearby_entities": "Player",
        "trust_level": 10,
        "interaction_summary": "Player approached suspiciously.",
        "scenario_plot": "Protect the gate."
    }
    player_input = "Halt!"
    
    prompt = builder.build_prompt(npc_data, game_state, player_input)
    print("\nGenerated Prompt:")
    print(prompt)
    
    assert "<|system|>" in prompt
    assert "**Current State:**" in prompt
    assert "Suspicious" in prompt
    assert "Protect the gate." in prompt
    
    print("\n✓ Prompt Builder Tests Passed")

if __name__ == "__main__":
    test_bt_logic()
    test_prompt_builder()
