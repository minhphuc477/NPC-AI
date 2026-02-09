from flask import Flask, request, jsonify
import os, logging

app = Flask(__name__)
TEST_MODE = os.environ.get("NPC_TEST_MODE", "true").lower() == "true" 
MODEL = TOKENIZER = None
logging.basicConfig(level=logging.INFO)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "test_mode": TEST_MODE})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    ctx = data.get("context", {})
    inp = data.get("player_input", "")
    npc_id = ctx.get("npc_id", "NPC")
    
    if TEST_MODE:
        resp = "[TEST] " + npc_id + ": Xin chao! Day la phan hoi test."
    else:
        resp = run_model(ctx, inp)
    
    return jsonify({"npc_id": npc_id, "response": resp, "success": True})


def run_model(ctx, inp):
    global MODEL, TOKENIZER
    import torch
    
    if MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            quantization_config=cfg,
            device_map="auto",
            trust_remote_code=True
        )
        MODEL = PeftModel.from_pretrained(base, "outputs/adapter")
        TOKENIZER = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True
        )
    
    persona = ctx.get("persona", "Ban la NPC trong game.")
    scenario = ctx.get("scenario", "")
    state = ctx.get("behavior_state", "idle")
    
    sys_start = chr(60) + "|system|" + chr(62)
    sys_end = chr(60) + "|end|" + chr(62)
    nl = chr(10)
    
    prompt = sys_start + nl + persona + nl
    prompt += "Canh: " + scenario + nl
    prompt += "Trang thai: " + state + nl
    prompt += sys_end + nl + nl
    prompt += chr(60) + "|user|" + chr(62) + nl + inp + nl + chr(60) + "|end|" + chr(62) + nl
    prompt += chr(60) + "|assistant|" + chr(62) + nl
    
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512).to(MODEL.device)
    
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id,
            use_cache=False
        )
    
    full = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    response = full[len(TOKENIZER.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return response.strip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    if not TEST_MODE:
        run_model({}, "")  # Pre-load model
    
    print(f"NPC Server running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
