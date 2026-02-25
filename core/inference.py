import sys
import os
import torch
import time
from typing import Dict, Any, Optional

# Disable tqdm globally to prevent console crashes on Windows
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from .config import BASE_MODEL, ADAPTER_PATH, QUANTIZATION_CONFIG, GENERATION_CONFIG

# Memory Optimization for Windows 1455 Error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class NPCInferenceEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NPCInferenceEngine, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self):
        """Loads the model and adapter. Safe to call multiple times (idempotent)."""
        if self.model is not None:
            return

        print(f"Loading base model: {BASE_MODEL}...", file=sys.stderr)
        torch.cuda.empty_cache()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "transformers import failed. Install/repair transformers + torch first."
            ) from e

        try:
            from peft import PeftModel
        except Exception:
            PeftModel = None

        try:
            bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
        except Exception as e:
            print(f"Quantization config unavailable, falling back to non-quantized load: {e}", file=sys.stderr)
            bnb_config = None

        try:
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "attn_implementation": "eager",
            }
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            else:
                load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
             
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
             
            if PeftModel is not None and os.path.exists(ADAPTER_PATH):
                print(f"Loading adapter from {ADAPTER_PATH}...", file=sys.stderr)
                self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            elif os.path.exists(ADAPTER_PATH):
                print("peft is unavailable; using base model without adapter.", file=sys.stderr)
                self.model = base_model
            else:
                print(f"Adapter not found at {ADAPTER_PATH}. Using base model.", file=sys.stderr)
                self.model = base_model
                
            print("Model loaded successfully!", file=sys.stderr)
            
        except Exception as e:
            print(f"Failed to load model: {e}", file=sys.stderr)
            raise e

    def generate(self, prompt: str, npc_name: str = "NPC") -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"Generating for {npc_name}...", file=sys.stderr)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                pad_token_id=self.tokenizer.eos_token_id,
                **GENERATION_CONFIG
            )
        
        
        # Safer extraction: slice the output tokens directly
        input_length = inputs["input_ids"].shape[1]
        output_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        return response

    def format_prompt(self, system: str, user: str) -> str:
        """Formats prompt using the model's chat template"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

# Global instance
engine = NPCInferenceEngine()
