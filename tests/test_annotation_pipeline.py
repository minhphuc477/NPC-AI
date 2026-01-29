"""Unit tests for the annotation pipeline modules."""
import os
import tempfile
import json
from pathlib import Path
from annotation_pipeline.annotation_guidelines import AnnotationGuidelines
from annotation_pipeline.annotation_tool import create_app, create_html_template, create_sample_tasks
from annotation_pipeline.quality_control import AnnotationQualityController


def test_create_markdown_and_save(tmp_path):
    ag = AnnotationGuidelines()
    md = ag.create_markdown_guidelines()
    assert isinstance(md, str) and len(md) > 0
    out = ag.save_guidelines(output_dir=str(tmp_path / "guidelines"))
    assert (out / "guidelines.json").exists()
    assert (out / "guidelines.md").exists()
    assert (out / "quick_reference.md").exists()


def test_flask_app_login_task_submit_stats(tmp_path, monkeypatch):
    # Set secret to avoid RuntimeError
    monkeypatch.setenv("ANNOTATION_SECRET", "test-secret")
    app = create_app(db_path=":memory:")
    create_html_template()
    client = app.test_client()
    # login
    r = client.post("/login", json={"user": "alice", "pwd": "x"})
    assert r.status_code == 200
    # create sample tasks
    create_sample_tasks(app.annotation_db, count=2)
    # get task
    r = client.get("/task")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("task") is not None
    task = data["task"]
    # submit annotation
    ann = {"task_id": task["id"], "dialogue_acts": {"acts": ["refuse"]}}
    r = client.post("/submit", json=ann)
    assert r.status_code == 200
    # stats
    r = client.get("/stats")
    s = r.get_json()
    assert s["total_tasks"] >= 1
    assert s["total_annotations"] >= 1


def test_quality_control_and_report(tmp_path):
    qc = AnnotationQualityController()
    db_file = str(tmp_path / "sample_annotations.db")
    qc.create_sample_database(db_path=db_file)
    qc.db_path = db_file
    summary = qc.generate_iaa_report(output_path=str(tmp_path / "iaa_report.md"))
    assert "num_annotations" in summary
    assert (tmp_path / "iaa_report.md").exists()
    plots_dir = tmp_path / "plots"
    # generate_iaa_report writes plots to ./plots by default; move and check
    # We'll call create_visualizations explicitly with out_dir inside tmp_path
    qc.create_visualizations(summary, out_dir=str(plots_dir))
    assert plots_dir.exists()
    # at least one png file
    pngs = list(plots_dir.glob("*.png"))
    assert len(pngs) >= 1


def test_convert_csv_to_jsonl_stdout_and_file(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text('instruction,input,output\n"Hi",,"Hello"\n"Bye",,"Goodbye"\n', encoding="utf-8")
    import annotation_pipeline.convert_csv_to_jsonl as conv
    # write to stdout
    count = conv.csv_to_jsonl(str(csv_file), "-")
    assert count == 2
    # write to file
    out = tmp_path / "out.jsonl"
    count2 = conv.csv_to_jsonl(str(csv_file), str(out))
    assert count2 == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    import json
    rec = json.loads(lines[0])
    assert rec["instruction"] == "Hi"
    assert "NPC:" in rec["text"]


def test_load_tasks_from_csv_duplicate_and_npc_state(tmp_path):
    csv_file = tmp_path / "dataset.csv"
    csv_file.write_text('instruction,input,output,npc_state\n"Hi",,"Hello","calm"\n"Hi",,"Hello","calm"\n"Something",,"Resp","angry"\n', encoding="utf-8")
    from annotation_pipeline.annotation_tool import AnnotationDatabase, AnnotationTaskGenerator
    db = AnnotationDatabase(":memory:")
    db.init_db()
    gen = AnnotationTaskGenerator(db)
    added = gen.load_tasks_from_csv(str(csv_file), assign_state=True)
    # two unique rows expected (duplicate removed)
    assert added == 2
    # verify tasks stored include state
    conn = db._connect()
    cur = conn.cursor()
    cur.execute("SELECT text FROM tasks")
    texts = [row["text"] for row in cur.fetchall()]
    conn.close()
    assert any('[state: calm]' in t for t in texts)
    assert any('[state: angry]' in t for t in texts)
