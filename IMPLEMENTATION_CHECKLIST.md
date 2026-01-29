# Implementation checklist

- Open `notebooks/colab_qlora_finetune.ipynb`
- Run data prep cell to convert CSV
- Run minimal training cell to generate adapter (for full runs set higher steps)
- Export adapter to gguf following the model/peft docs
- Use Ollama or local inference server to test adapter
