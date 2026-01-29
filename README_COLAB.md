# Quick Colab steps

1. Open `notebooks/colab_qlora_finetune.ipynb` in Colab.
2. Install requirements (commented in notebook).
3. Upload `gatekeeper_dataset.csv` to /content or mount Google Drive.
4. Run the training cell (adjust `max_steps` for your compute). For smoke tests use small steps.

Resource guidance:
- Colab Free: small models only
- Colab Pro: medium models
- Colab Pro+: large models and longer runtimes
- Local 3050Ti: can use small-batch runs; watch VRAM.
