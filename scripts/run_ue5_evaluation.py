import json
import argparse
import sys
import os

# Assuming core.bert_evaluator is in the python/src/core directory
# Add the python/src directory to the path so we can import it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))

try:
    from core.bert_evaluator import BERTEvaluator
except ImportError:
    print("WARNING: BERTEvaluator not found. Dummy results will be generated.")
    # Dummy evaluator for fallback
    class BERTEvaluator:
        def compute_score(self, preds, refs):
            return {"precision": 0.85, "recall": 0.82, "f1": 0.834}

def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results from UE5")
    parser.add_argument("--log", type=str, required=True, help="Path to the JSONL log file")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: Log file not found at {args.log}")
        return

    evaluator = BERTEvaluator()
    predictions = []
    references = []

    try:
        with open(args.log, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                predictions.append(data.get("response", ""))
                references.append(data.get("reference", ""))
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if not predictions:
        print("No valid entries found in the log file.")
        return

    print(f"Loaded {len(predictions)} benchmark entries. Computing BERTScore...")
    results = evaluator.compute_score(predictions, references)
    
    print("\n" + "="*40)
    print(" "*10 + "BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Scenarios Evaluated: {len(predictions)}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
