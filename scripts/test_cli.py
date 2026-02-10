import subprocess
import json
import time
import sys

# Command to run
CMD = [sys.executable, "npc_cli.py"]

print(f"Starting NPC CLI process: {' '.join(CMD)}")

try:
    process = subprocess.Popen(
        CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # Pass stderr to console so we see errors
        text=True,
        bufsize=1 # Line buffered
    )
    
    # Wait for ready signal (optional, but good practice if implemented)
    # Our script prints "READY" to stdout? No, let's check npc_cli.py
    # Yes: print("READY", flush=True) inside try block after load.
    
    print("Waiting for model to load (look for READY)...")
    while True:
        line = process.stdout.readline()
        if "READY" in line:
            print("Model Loaded!")
            break
        if not line and process.poll() is not None:
            print("Process exited prematurely!")
            sys.exit(1)
            
    # Send request - persona must be descriptive like training data!
    request = {
        "context": {
            "npc_id": "Guard_1",
            "persona": "You are a loyal gate guard, a former soldier who protects the village. You are stern but fair. You speak briefly and directly. You do not trust strangers easily.",
            "scenario": "Village gate"
        },
        "player_input": "Hello there, I'm a traveler from afar. May I enter the village?"
    }
    
    print(f"\nSending: {json.dumps(request)}")
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    
    # Read response
    print("Waiting for response...")
    response_line = process.stdout.readline()
    print(f"Raw Output: {response_line.strip()}")
    
    try:
        response = json.loads(response_line)
        print(f"\nSuccess! Response: {response.get('response')}")
    except json.JSONDecodeError:
        print("Failed to decode JSON response")

    # Clean exit
    process.stdin.write(json.dumps({"command": "exit"}) + "\n")
    process.stdin.flush()
    process.wait(timeout=5)
    print("Process finished.")
    
except Exception as e:
    print(f"Test failed: {e}")
    if 'process' in locals():
        process.kill()
