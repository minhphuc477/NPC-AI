#!/usr/bin/env python3
"""
Export QLoRA adapter to GGUF format for Ollama.

Usage:
    python export_gguf.py --adapter outputs/adapter --output models/npc-phi3.gguf
    
Requires: llama.cpp with convert tools installed
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_adapter(base_model: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter with base model."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        logger.error("Install: pip install peft transformers torch")
        return False
    
    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    return True


def convert_to_gguf(model_path: str, output_gguf: str, quantization: str = "q4_k_m"):
    """Convert HuggingFace model to GGUF format.
    
    Requires llama.cpp convert tools.
    """
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        convert_script = Path.home() / "llama.cpp/convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        logger.error("llama.cpp not found. Clone: git clone https://github.com/ggerganov/llama.cpp")
        logger.info("Then run: python convert_hf_to_gguf.py <model_path> --outtype f16 --outfile <output.gguf>")
        return False
    
    logger.info(f"Converting to GGUF: {output_gguf}")
    cmd = [
        sys.executable, str(convert_script),
        model_path,
        "--outtype", "f16",
        "--outfile", output_gguf
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        return False
    
    logger.info(f"GGUF saved to: {output_gguf}")
    return True


def create_ollama_model(gguf_path: str, model_name: str = "npc-phi3"):
    """Create Ollama model from GGUF."""
    modelfile = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 256
SYSTEM "Ban la mot NPC trong tro choi. Hay tra loi ngan gon va phu hop voi tinh cach."
"""
    
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    
    logger.info(f"Created Modelfile at: {modelfile_path}")
    logger.info(f"Run: ollama create {model_name} -f {modelfile_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export QLoRA adapter to GGUF")
    parser.add_argument("--adapter", required=True, help="Path to adapter directory")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model")
    parser.add_argument("--output", default="models/npc-phi3.gguf", help="Output GGUF path")
    parser.add_argument("--merged-dir", default="outputs/merged", help="Merged model directory")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge, use existing merged model")
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_merge:
        if not merge_adapter(args.base_model, args.adapter, args.merged_dir):
            return 1
    
    if not convert_to_gguf(args.merged_dir, args.output):
        logger.warning("Auto-conversion failed. Manual steps:")
        logger.info("1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        logger.info("2. Run: python llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile models/npc.gguf")
        logger.info("3. Quantize: ./llama.cpp/llama-quantize models/npc.gguf models/npc-q4.gguf Q4_K_M")
    
    create_ollama_model(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
