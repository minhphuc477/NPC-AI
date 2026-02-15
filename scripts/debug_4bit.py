
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
try:
    import bitsandbytes as bnb
    print("BnB:", bnb.__version__)
except ImportError as e:
    print("BnB Import Failed:", e)

model_id = "microsoft/Phi-3-mini-4k-instruct"

print("Testing 4-bit loading...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("SUCCESS: Model loaded in 4-bit!")
except Exception as e:
    print("FAILURE:", e)
    import traceback
    traceback.print_exc()
