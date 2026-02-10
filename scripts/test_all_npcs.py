import subprocess
import json
import sys
import time

CMD = [sys.executable, "npc_cli.py"]

TEST_CASES = [
    {
        "npc_id": "Guard_1",
        "persona": "You are a loyal gate guard, a former soldier who protects the village. You are stern but fair. You speak briefly and directly. You do not trust strangers easily.",
        "scenario": "Village gate",
        "player_input": "Hello there, I'm a traveler from afar. May I enter the village?"
    },
    {
        "npc_id": "Merchant_A",
        "persona": "You are Lão lái buôn, a cunning merchant who has traveled the kingdom. You always look for profit but never cheat. You talk a lot, tell stories, and love to bargain.",
        "scenario": "Marketplace",
        "player_input": "What's the most exotic item you have for sale today?"
    },
    {
        "npc_id": "Healer_B",
        "persona": "You are Thầy thuốc làng, a kind middle-aged woman who knows herbs. You care about everyone's health and are always ready to help. You speak gently.",
        "scenario": "Healer's hut",
        "player_input": "My head has been aching since I arrived. Do you have anything to help?"
    }
]

def run_test():
    print(f"Starting NPC CLI for comprehensive testing...")
    process = subprocess.Popen(
        CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Capture stderr for debugging
        text=True,
        bufsize=1
    )

    print("Waiting for model to load (max 10 minutes)...")
    start_time = time.time()
    timeout = 600 # 10 minutes
    
    while True:
        if time.time() - start_time > timeout:
            print(f"TIMEOUT: Model failed to load within {timeout} seconds.")
            process.kill()
            return

        # Try reading stdout
        try:
            line = process.stdout.readline()
            if line:
                print(f"STDOUT: {line.strip()}")
                if "READY" in line:
                    print("Model Loaded! Initializing tests...")
                    break
        except:
            pass

        # Try reading stderr for errors
        try:
            # We use non-blocking approach or just check if process is still alive
            if process.poll() is not None:
                err = process.stderr.read()
                print(f"CRASH DETECTED. STDERR:\n{err}")
                return
        except:
            pass
            
        time.sleep(1)

    for case in TEST_CASES:
        request = {
            "context": {
                "npc_id": case["npc_id"],
                "persona": case["persona"],
                "scenario": case["scenario"]
            },
            "player_input": case["player_input"]
        }
        
        print(f"\n--- Testing NPC: {case['npc_id']} ---")
        print(f"Input: {case['player_input']}")
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        try:
            resp_data = json.loads(response_line)
            print(f"Response: {resp_data.get('response')}")
        except:
            print(f"Error parsing response: {response_line}")

    # Exit
    process.stdin.write(json.dumps({"command": "exit"}) + "\n")
    process.stdin.flush()
    process.wait()
    print("\nAll tests completed.")

if __name__ == "__main__":
    run_test()
