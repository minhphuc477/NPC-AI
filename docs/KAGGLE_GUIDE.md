# Kaggle Training Guide for NPC AI

## Quick Start

### Step 1: Create Kaggle Dataset
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"+ New Dataset"**
3. Upload `data/train_multiturn.jsonl`
4. Name it: `npc-ai-training-data`
5. Set visibility to **Private**

### Step 2: Create Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Upload `notebooks/kaggle_train_multiturn.ipynb`

### Step 3: Configure Notebook
1. **Add Data**: Right sidebar → **"+ Add Data"** → Select your dataset
2. **Enable GPU**: Settings → Accelerator → **GPU P100** (or T4 x2)
3. **Enable Internet**: Settings → Internet → **On**

### Step 4: Update Data Path
In the CONFIG cell, update the path to match your dataset:
```python
"data_path": "/kaggle/input/YOUR-DATASET-NAME/train_multiturn.jsonl",
```

### Step 5: Run All Cells
Click **"Run All"** and wait ~20-30 minutes for training.

### Step 6: Download Results
After training, download `adapter_multiturn.zip` from the **Output** tab.

---

## Kaggle GPU Specs (Free Tier)
| GPU | VRAM | Weekly Quota |
|-----|------|--------------|
| P100 | 16 GB | 30 hours |
| T4 x2 | 16 GB each | 30 hours |

Both are significantly faster than your 3050 Ti (4GB)!

## Troubleshooting

**"Internet is not enabled"**: Settings → Internet → On

**"Out of memory"**: Reduce `batch_size` to 2 in CONFIG


## Using the Adapter Locally

1. **Download**: Get `adapter_multiturn.zip` from Kaggle Output tab
2. **Extract**: Unzip into your project folder. You should see an `adapter_multiturn` folder containing `adapter_model.safetensors` and `adapter_config.json`.
3. **Run Test Script**:
   ```powershell
   python scripts/test_adapter.py
   ```

### Troubleshooting
- **"DynamicCache object has no attribute 'seen_tokens'"**: 
  - This is a compatibility issue with Phi-3 and PEFT.
  - Fix: Add `use_cache=False` to `model.generate()`.
  
- **"Out of Memory" (CUDA OOM)**:
  - Reduce `max_new_tokens`
  - Ensure `load_in_4bit=True` is used
  - Close other apps using GPU

### Merging Adapter (Optional)
To merge the adapter into the base model for faster inference (requires ~8GB VRAM for merging):
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = PeftModel.from_pretrained(base_model, "adapter_multiturn")
model = model.merge_and_unload()
model.save_pretrained("models/phi3-npc-merged")
tokenizer.save_pretrained("models/phi3-merged")
```
