"""
QLoRA fine-tuning script with NEFTune and adjusted weight decay to prevent 
Catastrophic Forgetting and Overfitting when teaching LLMs semantic extraction.
"""
import argparse
from pathlib import Path


def _in_kaggle() -> bool:
    return Path("/kaggle").exists()


def _resolve_dataset_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if not _in_kaggle():
        return str(candidate)
    root = Path("/kaggle/input")
    if root.exists():
        matches = list(root.rglob(candidate.name))
        for match in matches:
            if match.is_file():
                return str(match)
    return str(candidate)


def _resolve_output_path(path: str) -> str:
    out = Path(path)
    if out.is_absolute():
        return str(out)
    if _in_kaggle():
        return str(Path("/kaggle/working") / out)
    return str(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="npc_lora_model")
    parser.add_argument("--neftune", type=float, default=5.0, help="NEFTune noise parameter (reduces overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay to maintain core knowledge")
    parser.add_argument("--kaggle-autopath", dest="kaggle_autopath", action="store_true")
    parser.add_argument("--no-kaggle-autopath", dest="kaggle_autopath", action="store_false")
    parser.set_defaults(kaggle_autopath=True)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependencies for qlora_anti_forgetting.py. "
            "Install unsloth, torch, trl, and datasets first."
        ) from exc

    dataset_path = str(args.dataset)
    output_path = str(args.output)
    if args.kaggle_autopath:
        dataset_path = _resolve_dataset_path(dataset_path)
        output_path = _resolve_output_path(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(Path(output_path) / "checkpoints")

    max_seq_length = 2048
    dtype = None # Auto
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Do NOT target all linear layers to prevent catastrophic forgetting.
    # We only target Attention + MLP layers to learn the extraction *skill*, 
    # not overwrite facts.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Dropout = 0 is optimized for Unsloth
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 60, # Small step count to prevent catastrophic forgetting
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay, # Phase 8 Fix: Increased weight decay
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = ckpt_dir,
            report_to = "none", # Disable wandb inside Kaggle
        ),
    )

    # Phase 8 Fix: Add NEFTune to the trainer config natively
    # Adds noise to embeddings to improve generalization and prevent memorizing format
    if args.neftune > 0:
        trainer.neftune_noise_alpha = args.neftune

    trainer.train()

    print(f"Saving LoRA adapters to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    main()
