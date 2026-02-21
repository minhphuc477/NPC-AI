import json

notebook_path = "notebooks/NPC_AI_Complete_Pipeline.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    text = f.read()

# The exact problematic string
old_str = '    subprocess.check_call([\\'accelerate\\', \\'launch\\', \\'--num_processes\\', \\'2\\', \\'scripts/train_unsloth.py\\', \\n'

if old_str in text:
    print("Found exact string!")
    text = text.replace(old_str, 'subprocess.check_call([\\'accelerate\\', \\'launch\\', \\'--num_processes\\', \\'2\\', \\'scripts/train_unsloth.py\\', \\n')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Replaced and saved!")
else:
    print("String not found. Trying another way...")
    import re
    # Match 4 spaces followed by subprocess.check_call(['accelerate'
    pattern = r'    subprocess\.check_call\(\[\'accelerate\''
    if re.search(pattern, text):
        text = re.sub(pattern, "subprocess.check_call(['accelerate'", text)
        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Regex replaced and saved!")
    else:
        print("Regex not found either.")
