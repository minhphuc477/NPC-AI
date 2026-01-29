from evaluate import evaluate_model
from pathlib import Path

def test_compute_metrics_and_summary(tmp_path):
    refs = ["Xin chào tôi là NPC"]
    hyps = ["Xin chào tôi là một NPC"]
    metrics = evaluate_model.compute_metrics(refs, hyps)
    assert 'tfidf_cosine' in metrics
    out = evaluate_model.summarize_metrics(metrics, out_dir=str(tmp_path / 'evaluation'))
    assert Path(out).exists()
