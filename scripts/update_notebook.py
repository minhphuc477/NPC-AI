import json
import os

notebook_path = r"c:\Users\MPhuc\Desktop\NPC AI\notebooks\kaggle_complete_pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# New content for Section 3
section_3_source = [
    "# === SECTION 3: ADVANCED NEURO-SYMBOLIC INFERENCE ===\n",
    "# NOTE: We import the pipeline from the repository to ensure reproducibility\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# 1. Setup Repo Access\n",
    "if not os.path.exists(\"NPC-AI\"):\n",
    "    print(\"Cloning repository...\")\n",
    "    !git clone https://github.com/minhphuc477/NPC-AI.git\n",
    "else:\n",
    "    print(\"Repository already exists.\")\n",
    "\n",
    "# 2. Add to Path\n",
    "sys.path.append(os.path.abspath(\"NPC-AI\"))\n",
    "\n",
    "# 3. Import from Core\n",
    "try:\n",
    "    from core.neuro_symbolic_pipeline import NeuroSymbolicPipeline\n",
    "    print(\"Successfully imported NeuroSymbolicPipeline from repo!\")\n",
    "except ImportError as e:\n",
    "    print(f\"Error importing from repo: {e}\")\n",
    "    print(\"Ensure you are running this in a Kaggle environment where the repo is cloned or mounted.\")"
]

# New content for Section 4
section_4_source = [
    "# === SECTION 4: SCIENTIFIC BENCHMARK ===\n",
    "\n",
    "# Using the sophisticated engine\n",
    "print(\"Initializing Neuro-Symbolic Engine...\")\n",
    "# Use the trained model from previous step\n",
    "engine = NeuroSymbolicPipeline(trained_model_path)\n",
    "\n",
    "configs = {\n",
    "    \"Neuro-Symbolic (Full)\": {\"enable_rag\": True, \"enable_graph\": True},\n",
    "    \"RAG Only\": {\"enable_rag\": True, \"enable_graph\": False},\n",
    "    \"Graph Only\": {\"enable_rag\": False, \"enable_graph\": True}\n",
    "}\n",
    "\n",
    "prompts = [\n",
    "    \"Where can I find finding the Elder Stone?\", \n",
    "    \"Why did Duke Varen betray the King?\",\n",
    "    \"How do I make a healing potion?\"\n",
    "]\n",
    "\n",
    "results = []\n",
    "print(\"Running Experiments...\")\n",
    "\n",
    "for name, cfg in configs.items():\n",
    "    for p in prompts:\n",
    "        res = engine.generate(p, cfg)\n",
    "        results.append({\n",
    "            \"Config\": name,\n",
    "            \"Prompt\": p,\n",
    "            \"Latency\": res['latency_ms'],\n",
    "            \"Throughput\": res['tps'],\n",
    "            \"Response\": res['text'][:50] + \"...\"\n",
    "        })\n",
    "\n",
    "import pandas as pd\n",
    "df = pd.DataFrame(results)\n",
    "print(df.groupby(\"Config\")[[\"Latency\", \"Throughput\"]].mean().to_markdown())\n",
    "df.to_csv(\"final_results.csv\")"
]

# Find and replace cells
found_section_3 = False
found_section_4 = False

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "# === SECTION 3: ADVANCED NEURO-SYMBOLIC INFERENCE ===" in source:
            cell["source"] = section_3_source
            found_section_3 = True
            print("Updated Section 3")
        elif "# === SECTION 4: SCIENTIFIC BENCHMARK ===" in source:
            cell["source"] = section_4_source
            found_section_4 = True
            print("Updated Section 4")

if found_section_3 and found_section_4:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)
    print("Successfully updated notebook.")
else:
    print("Could not find sections to update.")
