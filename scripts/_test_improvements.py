"""Quick integration test for conversation memory."""
from core.conversation_memory import ConversationMemory

m = ConversationMemory(window_size=3)

# Simulate 4 turns to test compression
m.add_turn("guard_01", "Hello!", "Halt! Who goes there?", 0.0)
m.add_turn("guard_01", "I am a friend.", "Prove it.", -0.2)
m.add_turn("guard_01", "Here is my letter of passage.", "Hmm, this looks genuine. You may pass.", 0.5)
m.add_turn("guard_01", "I come from the North kingdom.", "The North... I see. Enter carefully.", 0.3)

print("=== Memory Context ===")
print(m.get_memory_context("guard_01"))
print()
print("Trust:", m.get_trust_level("guard_01"))
print("Valence:", round(m.get_emotional_valence("guard_01"), 3))
print()

# Test prompt builder integration
from core.prompt_builder import PromptBuilder
pb = PromptBuilder()
npc_data = {"name": "Guard", "persona": "You are a stern gatekeeper."}
game_state = {"behavior_state": "Alert", "mood_state": "Suspicious", "trust_level": 60}

prompt = pb.build_prompt(
    npc_data=npc_data,
    game_state=game_state,
    player_input="Can I enter the city?",
    memory_context=m.get_memory_context("guard_01"),
    emotional_state={"joy": 0.1, "anger": 0.3, "fear": 0.1, "trust": 0.4, "surprise": 0.1},
)
print("=== Generated Prompt ===")
print(prompt[:500])
print("...")
print()
print("SUCCESS: All systems working!")
