import json
from pathlib import Path
import sys
from augment import generate_scenarios


def test_generate_scenarios_groq_mock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # run CLI in mock mode
    monkeypatch.setattr(sys, 'argv', ['prog', '--teacher', 'groq', '--mock', '--n', '5', '--out', 'data/generated/groq_generated.jsonl'])
    generate_scenarios.main()
    out = tmp_path / 'data' / 'generated' / 'groq_generated.jsonl'
    meta = tmp_path / 'data' / 'generated' / 'groq_generated_meta.json'
    assert out.exists()
    lines = out.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 5
    data = [json.loads(l) for l in lines]
    assert 'prompt' in data[0]
    assert 'response' in data[0]
    assert meta.exists()
    m = json.loads(meta.read_text(encoding='utf-8'))
    assert m['n'] == 5
    assert 'model' in m
