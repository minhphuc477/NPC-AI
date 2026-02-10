# Advanced BERT Integration - Requirements

## Core Dependencies

```bash
# Install PyTorch (with CUDA for GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and Sentence Transformers
pip install transformers sentence-transformers

# Install additional dependencies
pip install scikit-learn langdetect
```

## Optional Dependencies

```bash
# For advanced features
pip install accelerate  # GPU optimization
pip install bitsandbytes  # Quantization
pip install sentencepiece  # Tokenization
```

## GPU Requirements

- **NVIDIA GPU** with CUDA support (recommended)
- **CUDA 11.8+** or **CUDA 12.x**
- **8GB+ VRAM** for optimal performance
- **CPU fallback** available if no GPU

## Models Downloaded

The following models will be automatically downloaded on first use:

1. **Sentence Embeddings:**
   - `sentence-transformers/all-MiniLM-L6-v2` (English, 80MB)
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Multi-language, 420MB)

2. **Emotion Detection:**
   - `j-hartmann/emotion-english-distilroberta-base` (260MB)

3. **Translation (optional):**
   - `Helsinki-NLP/opus-mt-*` models (per language pair, ~300MB each)

## Installation Script

```bash
# Run this to install all dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers scikit-learn langdetect accelerate
```

## Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Usage

```python
from core.bert_evaluator import BERTSemanticEvaluator

# Create evaluator (auto-detects GPU)
evaluator = BERTSemanticEvaluator(use_gpu=True)

# Evaluate response
score = evaluator.evaluate_response(
    response="Greetings brave warrior!",
    context="Player approaches guard"
)

print(f"Overall: {score.overall:.3f}")
print(f"Emotion: {score.emotion} ({score.emotion_confidence:.3f})")
```
