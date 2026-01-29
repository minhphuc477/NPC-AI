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

## Using Groq as a generation "teacher" (Optional)

Groq offers a REST Responses-style API that can be used to generate prompts and
responses for data augmentation. Set the following environment variables in
Colab before running generation:

```
import os
os.environ['GROQ_API_KEY'] = '<YOUR_GROQ_API_KEY>'
# optional: os.environ['GROQ_API_URL'] = 'https://api.groq.com/openai/v1/responses'
```

Example model IDs seen in the Groq UI include `llama-3.1-8b` and `gpt-oss-20b`.
Be mindful of rate limits (HTTP 429) and costs when generating many samples.
Our tools include a local cache (`.cache/groq`) and a batch generator that will
avoid repeated calls for identical prompts and write a small metadata file with
`model`, `timestamp` and a `cost_estimate` placeholder.
