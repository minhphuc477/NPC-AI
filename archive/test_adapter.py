import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time

# Configuration
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "adapter_multiturn"  # Path to your downloaded adapter

def load_model():
    print(f"Loading base model: {BASE_MODEL}")
    
    # 4-bit quantization for 3050 Ti (4GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Load adapter
    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\nGenerating response...")
    start_time = time.time()
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=False,  # Fix for DynamicCache error with Phi-3 + PEFT
    )
    
    end_time = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generation took {end_time - start_time:.2f} seconds")
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    # Test prompts
    prompts = [
        """System: You are a friendly village baker named Otto. You love bread and gossip.
Question: What's the news in town today?
Answer:""",
        
        """System: You are a grumpy blacksmith named Goran. You hate wasted time.
Question: Can you fix my sword?
Answer:"""
    ]
    
    for p in prompts:
        print("-" * 50)
        print(f"PROMPT:\n{p}")
        response = generate_response(model, tokenizer, p)
        print(f"\nRESPONSE:\n{response}")
        print("-" * 50)
