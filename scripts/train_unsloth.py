
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
import os

# Configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "unsloth/Phi-3-mini-4k-instruct" # optimized for Unsloth
output_dir = "models/phi3_npc_lora_unsloth"

def train():
    print(f"Loading Unsloth model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data/npc_training.jsonl", split="train")

    # Format dataset for SFTTrainer
    # The dataset has 'prompt' and 'completion' fields.
    # We need to format them into a text field.
    def formatting_prompts_func(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        texts = []
        for p, c in zip(prompts, completions):
            text = f"{p}{c}<|end|>" # Phi-3 format? Or just concat?
            # Actually, SFTTrainer expects a text field.
            # Our prompt already includes [CONTEXT]...[PLAYER]...
            # We just append completion.
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "none",
    ),
    )

    trainer.train()
    
    # Save model
    print(f"Saving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Merge and save GGUF ready model
    # model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit",)
    
    print("Done!")

if __name__ == "__main__":
    train()
