import torch
import os

# Model Settings
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "adapter_multiturn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quantization (Optimized for 3050 Ti 4GB VRAM)
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": False,
}

# Generation Settings (Default)
GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "use_cache": False,  # CRITICAL: Disables cache to fix Phi-3/PEFT compatibility
}

# Server Settings
HOST = "0.0.0.0"
PORT = 8080
