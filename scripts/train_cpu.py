
import torch
import os
import inspect
import importlib
from typing import Any, Dict

# CPU Training Configuration
output_dir = "models/phi3_npc_lora_cpu"
model_id = "microsoft/Phi-3-mini-4k-instruct"

# Check for bfloat16 support on CPU (usually available on modern CPUs with AVX512/AMX or just standard emulation)
# If not, use float32 (might consume ~16GB+ RAM)
dtype = torch.float32 # Safe default, but heavy
if torch.cuda.is_available():
    print("WARNING: CUDA detected but forcing CPU training due to library issues.")

device = "cpu"

def train():
    transformers_mod = importlib.import_module("transformers")
    AutoModelForCausalLM = getattr(transformers_mod, "AutoModelForCausalLM")
    AutoTokenizer = getattr(transformers_mod, "AutoTokenizer")
    TrainingArguments = getattr(transformers_mod, "TrainingArguments")
    peft_mod = importlib.import_module("peft")
    LoraConfig = getattr(peft_mod, "LoraConfig")
    get_peft_model = getattr(peft_mod, "get_peft_model")
    TaskType = getattr(peft_mod, "TaskType")
    datasets_mod = importlib.import_module("datasets")
    load_dataset = getattr(datasets_mod, "load_dataset")
    trl_mod = importlib.import_module("trl.trainer.sft_trainer")
    SFTTrainer = getattr(trl_mod, "SFTTrainer")

    print(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Fix padding

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    # Enable Gradient Checkpointing (saves memory)
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'] # Phi-3 modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data/npc_training.jsonl", split="train")

    # Format dataset
    def formatting_func(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        texts = []
        for p, c in zip(prompts, completions):
            texts.append(f"{p}{c}<|end|>")
        return { "text": texts }

    dataset = dataset.map(formatting_func, batched=True)

    print("Starting CPU training...")
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, # Reduced for CPU feasibility (1 epoch of 1000 samples is significant)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # Effective batch 8
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="epoch",
        use_cpu=True, # Explicitly tell Trainer to use CPU
        fp16=False,
        bf16=False, # bitsandbytes not needed for pure cpu
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "tokenizer" in init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in init_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in init_params:
        trainer_kwargs["max_seq_length"] = 1024  # Reduced context for RAM safety
    if "formatting_func" in init_params and "dataset_text_field" not in init_params:
        trainer_kwargs["formatting_func"] = lambda ex: str(ex["text"])

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    print("Done!")

if __name__ == "__main__":
    train()
