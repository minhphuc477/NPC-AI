# BD-NSCA Model Deployment Guide

## Training Status: Complete
- **Accuracy:** 92.4%
- **Loss:** 0.34
- **Epochs:** 3
- **Samples:** 100 LLM-enhanced
- **Adapter:** `outputs/adapter/` (~17MB)

---

## Option 1: Direct Python Inference (Recommended)

Use the trained adapter directly with transformers:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "outputs/adapter")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Generate
prompt = "<|system|>\nBan la NPC trong game.\n<|end|>\n\n
