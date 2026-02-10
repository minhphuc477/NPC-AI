import os
import shutil
from transformers import AutoTokenizer

def extract_tokenizer(model_name, output_dir):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check for tokenizer.model (SentencePiece)
    # Phi-3 / Llama models usually have this.
    try:
        # For Llama-based tokenizers, the spm model path is often in vocab_file attribute
        # or we can use save_vocabulary which dumps the model files
        
        print(f"Saving vocabulary to {output_dir}...")
        files = tokenizer.save_vocabulary(output_dir)
        print(f"Saved files: {files}")
        
        # Look for tokenizer.model in the saved files
        model_file = None
        for f in files:
            if f.endswith("tokenizer.model"):
                model_file = f
                break
        
        if model_file:
            print(f"Successfully extracted: {model_file}")
        else:
            print("WARNING: tokenizer.model not found in saved vocabulary.")
            # Some models might use fast tokenizer json only?
            # Phi-3 mini is Llama based, so it should have spm model.
            
    except Exception as e:
        print(f"Error extracting tokenizer: {e}")

if __name__ == "__main__":
    # Use the local adapter path if available, or download from HF
    # The user has "adapter_multiturn", let's try to get the base model first
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    output_path = "model_data" 
    
    extract_tokenizer(model_id, output_path)
