import json
from pathlib import Path
from scripts import colab_helpers

def test_convert_csv_to_jsonl(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("text,label\nhello world,1\nhi there,0\n", encoding='utf-8')
    out = tmp_path / "out.jsonl"
    n = colab_helpers.convert_csv_to_jsonl(str(csv), str(out))
    assert n == 2
    lines = out.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 2
    data = [json.loads(l) for l in lines]
    assert data[0]['text'] == 'hello world'


def test_render_prompt():
    s = "A room with a chest"
    p = "Open the chest"
    prompt = colab_helpers.render_prompt(s, p)
    assert 'NPC' in prompt
    assert s in prompt
    assert p in prompt
