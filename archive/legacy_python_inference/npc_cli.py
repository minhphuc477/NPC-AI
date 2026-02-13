import sys
import json
import traceback

# Ensure core module is importable
sys.path.append(".")

try:
    from core.inference import engine
    from core.behavior_tree import create_npc_behavior_tree
    from core.prompt_builder import PromptBuilder
except ImportError:
    # If running from root without core in path
    sys.path.append("..")
    from core.inference import engine
    from core.behavior_tree import create_npc_behavior_tree
    from core.prompt_builder import PromptBuilder

def main():
    # Load model immediately
    print("STATUS: Loading model...", file=sys.stderr)
    try:
        engine.load_model()
        print("STATUS: Model loaded successfully!", file=sys.stderr)
        print("READY", flush=True) # Signal to UE5 that we are ready
    except Exception as e:
        print(f"ERROR: Model load failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize Behavior Tree and Prompt Builder
    behavior_tree = create_npc_behavior_tree()
    prompt_builder = PromptBuilder(use_advanced_format=True)

    # Main Loop
    while True:
        try:
            # Read line from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
                continue

            # Check for exit command
            if data.get("command") == "exit":
                break
                
            # Extract context and input
            context = data.get("context", {})
            player_input = data.get("player_input", "")
            
            # Update Blackboard (Game State) for BT
            # Map input context to BT blackboard keys
            blackboard = {
                "hp": context.get("hp", 100),
                "is_player_nearby": True, # Assume true if communicating
                "is_player_talking": bool(player_input),
                "is_combat": context.get("is_combat", False),
                # Pass through other context for actions
                "context": context,
                "player_input": player_input
            }
            
            # Tick Behavior Tree
            # This decides the current action (e.g., "Talking", "Attacking", "Patrolling")
            status = behavior_tree.tick(blackboard)
            current_action = blackboard.get("current_action", "Idle")
            
            # If action is NOT talking, return action immediately
            if current_action != "Talking":
                output = {
                    "response": f"*[Action: {current_action}]*",
                    "npc_id": context.get("npc_id", "NPC"),
                    "action": current_action,
                    "success": True
                }
                print(json.dumps(output), flush=True)
                continue

            # If action IS talking, generate dialogue using LLM
            # Inject derived state into context for PromptBuilder
            context["behavior_state"] = current_action
            
            # Build Prompt
            prompt = prompt_builder.build_prompt(
                npc_data={"name": context.get("npc_id", "NPC"), "persona": context.get("persona", "")},
                game_state=context,
                player_input=player_input
            )
            
            # Generate
            response = engine.generate(prompt, npc_name=context.get("npc_id", "NPC"))
            
            # Output JSON
            output = {
                "response": response,
                "npc_id": context.get("npc_id", "NPC"),
                "action": current_action,
                "success": True
            }
            print(json.dumps(output), flush=True)
            
        except Exception as e:
            # Catch all errors to keep loop alive
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(json.dumps({"error": error_msg, "success": False}), flush=True)
            print(error_msg, file=sys.stderr)

if __name__ == "__main__":
    main()
