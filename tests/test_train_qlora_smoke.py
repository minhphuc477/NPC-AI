# NPC AI/tests/test_train_qlora_smoke.py
import tempfile
import json
from scripts import train_qlora


def test_train_qlora_dry_run(tmp_path):
    data = tmp_path / "small.jsonl"
    lines = [json.dumps({"id": str(i), "text": f"hello {i}"}, ensure_ascii=False) for i in range(3)]
    data.write_text("\n".join(lines), encoding='utf-8')

    # exercise dry-run
    import subprocess
    res = subprocess.run(["python", "-m", "scripts.train_qlora", "--data", str(data), "--output-dir", str(tmp_path / 'out'), "--dry-run"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Dry run: validating data" in res.stdout
