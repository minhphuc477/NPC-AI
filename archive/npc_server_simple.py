import sys
# Disable tqdm globally before importing transformers
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import time

# Configuration
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "adapter_multiturn"
PORT = 8080

# Global model variables
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    print(f"Loading base model: {BASE_MODEL}...")
    
    # Clear CUDA cache first
    torch.cuda.empty_cache()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        if os.path.exists(ADAPTER_PATH):
            print(f"Loading adapter from {ADAPTER_PATH}...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        else:
            print(f"Adapter not found at {ADAPTER_PATH}. using base model only.")
            model = base_model
            
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                response_data = self.generate_response(data)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
            
        self.send_response(404)
        self.end_headers()

    def generate_response(self, data):
        global model, tokenizer
        context = data.get("context", {})
        player_input = data.get("player_input", "")
        npc_name = context.get("npc_id", "NPC")
        
        # Construct prompt
        persona = context.get("persona", "You are a helpful NPC.")
        scenario = context.get("scenario", "")
        state = context.get("behavior_state", "idle")
        
        system_prompt = f"{persona}\nScenario: {scenario}\nCurrent State: {state}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": player_input}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"Generating for {npc_name}: {player_input}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("<|assistant|>")[-1].strip()
        
        return {
            "response": response,
            "npc_id": npc_name,
            "success": True
        }
    
    def log_message(self, format, *args):
        # Override to use print (stderr)
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))

if __name__ == "__main__":
    load_model()
    server = HTTPServer(('0.0.0.0', PORT), NPCRequestHandler)
    print(f"NPC Server (Simple) running on http://localhost:{PORT}")
    server.serve_forever()
