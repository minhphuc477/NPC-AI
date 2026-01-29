import tempfile
from pathlib import Path
import json
from scripts import colab_helpers
from scripts import evaluate_model


def test_convert_csv_to_jsonl_runs():
    src = Path(__file__).parent.parent / "annotation_pipeline" / "data" / "gatekeeper_dataset.csv"
    assert src.exists(), "Gatekeeper CSV must exist for test"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "train.jsonl"
        n = colab_helpers.csv_to_jsonl(str(src), str(out))
        assert n > 0
        assert out.exists()
        # small sanity: first line is JSON with instruction and output
        first = out.read_text(encoding="utf-8").splitlines()[0]
        assert "instruction" in first and "output" in first


def test_evaluate_model_runs_and_writes():
    # create a tiny hypothesis jsonl
    with tempfile.TemporaryDirectory() as td:
        hp = Path(td) / "hyp.jsonl"
        sample = [
            {"instruction": "Say hello", "output": "Hello, traveler."},
            {"instruction": "Refuse request", "output": "I cannot help with that."},
        ]
        with hp.open("w", encoding="utf-8") as wf:
            for s in sample:
                wf.write(json.dumps(s, ensure_ascii=False) + "\n")
        res = evaluate_model.run_evaluation(str(hp), None, out_dir=str(Path(td) / "out"))
        assert "metrics" in res
        assert Path(res["out_dir"]).exists()
        # check evaluation.md exists
        assert Path(res["out_dir"]) / "evaluation.md" .exists()