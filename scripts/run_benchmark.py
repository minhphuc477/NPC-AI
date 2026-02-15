
import subprocess
import time
import json
import os
import sys

# Define test cases based on benchmark_definitions.py (simplified for this runner)
TEST_CASES = [
    {
        "name": "Temporal Memory - Short Term",
        "inputs": [
            "My name is Alaric.",
            "What is my name?"
        ],
        "expected_keywords": ["Alaric"]
    },
    {
        "name": "Emotional Reaction - Insult",
        "inputs": [
            "You are a stupid merchant!",
            "Do you want to trade?"
        ],
        "expected_keywords": ["angry", "upset", "rude", "leave"]
    },
    {
        "name": "Social Knowledge - Faction",
        "inputs": [
            "Who runs this town?",
            "Tell me about the Iron Guard."
        ],
        "expected_keywords": ["Iron Guard", "faction"]
    }
]

def run_benchmark():
    executable = r"f:\NPC AI\cpp\build\Release\chat_interface.exe"
    
    if not os.path.exists(executable):
        print(f"Error: Executable not found at {executable}")
        return

    print(f"Starting Benchmark using {executable}...")
    
    results = []
    
    # We run a separate process for each test case to ensure clean state? 
    # Or one session? 
    # The NPC has persistence. One session tests memory accumulation.
    # Let's run one continuous session.
    
    process = subprocess.Popen(
        [executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1, # Line buffered
        encoding='utf-8'
    )
    
    # Wait for startup
    time.sleep(5)
    
    # Clear initial output
    # (Simplified: just write and read)
    
    full_log = ""
    
    for test in TEST_CASES:
        print(f"Running test: {test['name']}")
        test_log = {"name": test["name"], "inputs": [], "responses": [], "passed": False}
        
        for user_input in test["inputs"]:
            # Write input
            try:
                if process.poll() is not None:
                    print("Process died!")
                    break
                    
                print(f"Sending: {user_input}")
                process.stdin.write(user_input + "\n")
                process.stdin.flush()
                
                # Read response (naive)
                # The chat_interface prints "Elara: ..." or similar.
                # We wait a bit or read line by line.
                response_buffer = []
                start_time = time.time()
                while time.time() - start_time < 5.0: # 5 sec timeout per response
                    line = process.stdout.readline()
                    if line:
                        print(f"Received: {line.strip()}")
                        response_buffer.append(line.strip())
                        if "Elara:" in line: # Assuming logic to detect end of turn
                            pass
                    else:
                        time.sleep(0.1)
                
                test_log["inputs"].append(user_input)
                test_log["responses"].append("\n".join(response_buffer))
                
            except Exception as e:
                print(f"Error: {e}")
        
        # simplified verification
        last_response = test_log["responses"][-1] if test_log["responses"] else ""
        passed = any(k.lower() in last_response.lower() for k in test["expected_keywords"])
        test_log["passed"] = passed
        results.append(test_log)
        full_log += f"\nTEST: {test['name']} - {'PASS' if passed else 'FAIL'}\n"

    process.terminate()
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Benchmark Complete.")
    print(full_log)

if __name__ == "__main__":
    run_benchmark()
