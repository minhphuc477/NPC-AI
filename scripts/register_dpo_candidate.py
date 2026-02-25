#!/usr/bin/env python3
"""Merge a DPO LoRA adapter and register a new Ollama candidate model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _disable_torchao_if_needed() -> None:
    # Work around incompatible torchao/triton environments.
    try:
        import transformers.utils.import_utils as iu  # type: ignore

        iu._torchao_available = False
        iu._torchao_version = "0.0.0"
    except Exception:
        pass


def merge_adapter(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    _disable_torchao_if_needed()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(
        str(merged_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))


def convert_to_gguf(merged_dir: Path, gguf_path: Path, llama_cpp_dir: Path) -> None:
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"GGUF converter not found: {converter}")
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            "f16",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 or not gguf_path.exists():
        raise RuntimeError(
            "GGUF conversion failed.\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def build_modelfile(from_path: Path, output_path: Path, system_prompt: str) -> None:
    lines = [
        f"FROM {from_path.resolve()}",
        'TEMPLATE "{{ .System }}\\n\\n{{ .Prompt }}"',
        f'SYSTEM "{system_prompt}"',
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_ollama_model(model_tag: str, modelfile: Path) -> None:
    proc = subprocess.run(
        ["ollama", "create", model_tag, "-f", str(modelfile)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ollama create failed.\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Register merged DPO candidate model into Ollama.")
    parser.add_argument("--adapter", required=True, help="Path to DPO adapter directory.")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--merged-dir", default="outputs/merged_dpo_candidate")
    parser.add_argument("--model-tag", default="elara-npc-dpo:latest")
    parser.add_argument("--modelfile", default="outputs/merged_dpo_candidate/Modelfile")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--gguf-out", default="outputs/merged_dpo_candidate/elara_npc_dpo_f16.gguf")
    parser.add_argument(
        "--from-gguf",
        dest="from_gguf",
        action="store_true",
        help="Register model from GGUF.",
    )
    parser.add_argument(
        "--from-merged",
        dest="from_gguf",
        action="store_false",
        help="Register model directly from merged Safetensors (not recommended for Phi-3).",
    )
    parser.add_argument("--skip-merge", action="store_true", help="Reuse an existing merged checkpoint.")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are Elara, a merchant NPC in a fantasy world. "
            "Use dynamic context and retrieval evidence; stay concise and in-character."
        ),
    )
    parser.set_defaults(from_gguf=True)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    merged_dir = Path(args.merged_dir)
    modelfile = Path(args.modelfile)
    gguf_out = Path(args.gguf_out)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    if not args.skip_merge:
        print(f"[1/3] Merging adapter {adapter_dir} into base model {args.base_model}")
        merge_adapter(base_model=str(args.base_model), adapter_dir=adapter_dir, merged_dir=merged_dir)
    else:
        if not merged_dir.exists():
            raise FileNotFoundError(f"--skip-merge was set but merged directory does not exist: {merged_dir}")
        print(f"[1/3] Reusing merged directory: {merged_dir}")

    modelfile.parent.mkdir(parents=True, exist_ok=True)
    model_source = merged_dir
    if args.from_gguf:
        print(f"[2/4] Converting merged model to GGUF: {gguf_out}")
        convert_to_gguf(
            merged_dir=merged_dir,
            gguf_path=gguf_out,
            llama_cpp_dir=Path(args.llama_cpp_dir),
        )
        model_source = gguf_out
        print(f"[3/4] Writing Modelfile: {modelfile}")
    else:
        print(f"[2/3] Writing Modelfile: {modelfile}")
    build_modelfile(from_path=model_source, output_path=modelfile, system_prompt=str(args.system_prompt))

    if args.from_gguf:
        print(f"[4/4] Registering Ollama model: {args.model_tag}")
    else:
        print(f"[3/3] Registering Ollama model: {args.model_tag}")
    create_ollama_model(model_tag=str(args.model_tag), modelfile=modelfile)
    print(f"Registered model: {args.model_tag}")
    print(f"Merged checkpoint: {merged_dir}")
    if args.from_gguf:
        print(f"GGUF checkpoint: {gguf_out}")
    print(f"Modelfile: {modelfile}")


if __name__ == "__main__":
    main()
