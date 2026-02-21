import json

notebook_path = "notebooks/NPC_AI_Complete_Pipeline.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    updated = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if "subprocess.check_call(['accelerate'" in line:
                    source[i] = line.lstrip(' ')
                    print(f"Removed leading spaces from line {i}: {repr(source[i])}")
                    updated = True

    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("Notebook indentation fixed successfully.")
    else:
        print("Could not find the target line to fix.")

except Exception as e:
    print(f"Error updating notebook: {e}")
