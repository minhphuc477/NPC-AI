from evaluate import evaluate_model
from pathlib import Path

def test_compute_metrics_and_summary(tmp_path):
    refs = ["Xin chào tôi là NPC"]
    hyps = ["Xin chào tôi là một NPC"]
    metrics_vi = evaluate_model.compute_metrics(refs, hyps, lang='vi')
    assert 'tfidf_cosine' in metrics_vi
    out = evaluate_model.summarize_metrics(metrics_vi, out_dir=str(tmp_path / 'evaluation'))
    assert Path(out).exists()

    # English case
    refs_en = ["Hello I am an NPC"]
    hyps_en = ["Hi, I am an NPC"]
    metrics_en = evaluate_model.compute_metrics(refs_en, hyps_en, lang='en')
    assert 'tfidf_cosine' in metrics_en
    out2 = evaluate_model.summarize_metrics(metrics_en, out_dir=str(tmp_path / 'evaluation_en'))
    assert Path(out2).exists()
