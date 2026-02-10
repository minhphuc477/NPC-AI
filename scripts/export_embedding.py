"""
export_embedding.py - Export Sentence Transformer model to ONNX for C++ Vector Memory

This script exports the `all-MiniLM-L6-v2` model (or similar) to ONNX format.
It includes support for INT8 quantization.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path
import os
import shutil

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    quantize_dynamic = None

# Default model optimized for semantic search (Multilingual + SentencePiece)
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class EmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize embeddings
        embeddings = sum_embeddings / sum_mask
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

def export_embedding_model(model_name, output_path, quantize=False):
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    # Wrap model to include pooling and normalization in the ONNX graph
    model = EmbeddingModelWrapper(base_model)
    model.eval()
    
    # Dummy input
    dummy_text = ["This is a test sentence."]
    encoded_input = tokenizer(dummy_text, padding=True, truncation=True, return_tensors='pt')
    
    print(f"Exporting to ONNX: {output_path}")
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (encoded_input['input_ids'], encoded_input['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['embeddings'], # The output is the normalized vector
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'embeddings': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    # Export Tokenizer
    tokenizer_dir = str(Path(output_path).parent / "tokenizer_embedding")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Tokenizer saved to: {tokenizer_dir}")
    
    if quantize:
        if quantize_dynamic is None:
            print("! Warning: onnxruntime not installed. Skipping quantization.")
        else:
            print("Quantizing model (INT8)...")
            quantized_path = output_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(
                model_input=output_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8
            )
            print(f"✓ Quantized model saved to {quantized_path}")

    print("✓ Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--output", default="onnx_models/embedding.onnx", help="Output path")
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    
    args = parser.parse_args()
    export_embedding_model(args.model, args.output, args.quantize)
