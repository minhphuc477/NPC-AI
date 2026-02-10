import requests
import json
import time

SERVER_URL = "http://localhost:8080/generate"

def test_chat(persona, scenario, player_input):
    payload = {
        "context": {
            "npc_id": "test_npc",
            "persona": persona,
            "scenario": scenario,
            "behavior_state": "idle"
        },
        "player_input": player_input
    }
    
    print(f"\nSending request to {SERVER_URL}...")
    print(f"Input: {player_input}")
    
    try:
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response ({end_time - start_time:.2f}s): {data.get('response')}")
            return True
        else:
            print(f"Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("-" * 50)
    print("Testing NPC Server Integration")
    print("-" * 50)
    
    # Test 1: Friendly Villager
    test_chat(
        "You are a friendly villager named Otto.",
        "The player just arrived in town.",
        "Hello there! What is this place?"
    )
    
    # Test 2: Grumpy Blacksmith
    test_chat(
        "You are a grumpy blacksmith named Goran.",
        "The player wants to repair a weapon.",
        "Can you fix my broken sword?"
    )
