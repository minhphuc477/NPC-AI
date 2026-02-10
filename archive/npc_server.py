from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging
import os
import time

# Configuration
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "adapter_multiturn"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    logger.info(f"Loading base model: {BASE_MODEL}...")
    import sys
    sys.stdout.flush()
    
    # 4-bit quantization config (Optimized for 4GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    try:
        logger.info("Initializing AutoModelForCausalLM...")
        sys.stdout.flush()
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,  # Explicitly set this
        )
        logger.info("Base model loaded. Loading tokenizer...")
        sys.stdout.flush()
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load adapter
        if os.path.exists(ADAPTER_PATH):
            logger.info(f"Loading adapter from {ADAPTER_PATH}...")
            sys.stdout.flush()
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        else:
            logger.warning(f"Adapter not found at {ADAPTER_PATH}. using base model only.")
            model = base_model
            
        logger.info("Model loaded successfully!")
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        # Keep the process alive to show the error
        raise e

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/generate", methods=["POST"])
def generate():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    context = data.get("context", {})
    player_input = data.get("player_input", "")
    
    # Extract context variables
    persona = context.get("persona", "You are a helpful NPC.")
    scenario = context.get("scenario", "")
    state = context.get("behavior_state", "idle")
    npc_name = context.get("npc_id", "NPC")
    
    # Construct prompt
    system_prompt = f"{persona}\nScenario: {scenario}\nCurrent State: {state}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": player_input}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    logger.info(f"Generating for {npc_name}: {player_input}")
    start_time = time.time()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=False,  # CRITICAL FIX for Phi-3 + PEFT compatibility
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove prompt)
        # Phi-3 chat template structure creates formatted text, we need to extract the last part
        response = generated_text.split("<|assistant|>")[-1].strip()
        
        end_time = time.time()
        logger.info(f"Generated in {end_time - start_time:.2f}s: {response}")
        
        return jsonify({
            "response": response,
            "npc_id": npc_name,
            "generation_time": end_time - start_time
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Load model on startup
    print("Starting NPC Server...")
    load_model()
    
    # Run single-threaded to avoid CUDA context issues on Windows
    print("Server ready! Listening on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False, threaded=False)
