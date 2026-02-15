
import json
import os

def prepare_data():
    input_file = "data/npc_training.jsonl"
    output_file = "data/npc_training_llama.txt"
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    print(f"Formatting {len(data)} samples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            prompt = item['prompt']
            completion = item['completion']
            
            # Format: <s>[INST] {prompt} [/INST] {completion}</s>
            # Note: prompt already has [CONTEXT]...[PLAYER]...
            
            # Phi-3 Format is actually:
            # <|user|>\n{prompt}<|end|>\n<|assistant|>\n{completion}<|end|>
            # But llama-finetune usually expects a simpler format or handles special tokens if specified.
            # However, for pure text completion (which finetune usually does), we just concat.
            # But to make it chat-instruct compliant, we should add the tokens.
            # Since llama-finetune learns to predict the *whole sequence*, providing the chat structure works.
            
            text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{completion}<|end|>\n"
            f.write(text)
            
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    prepare_data()
