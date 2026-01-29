"""Tests for Groq integration helpers."""
import json
from pathlib import Path
import os
from scripts.colab_helpers import call_groq_api, batch_generate_with_groq

class DummyResp:
    def __init__(self, status_code=200, ok=True, data=None, text=''):
        self.status_code = status_code
        self.ok = ok
        self._data = data
        self.text = text

    def json(self):
        return self._data


def test_call_groq_api_success(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResp(200, True, data={'output': 'generated-text'})

    monkeypatch.setattr('requests.post', fake_post)
    out = call_groq_api('hi there', model_id='llama-3.1-8b')
    assert 'generated-text' in out


def test_call_groq_api_429_backoff(monkeypatch):
    calls = {'n': 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls['n'] += 1
        if calls['n'] < 3:
            return DummyResp(429, False, data={'error': 'rate_limited'}, text='rate limit')
        return DummyResp(200, True, data={'output': 'ok'})

    sleeps = []
    monkeypatch.setattr('time.sleep', lambda s: sleeps.append(s))
    monkeypatch.setattr('requests.post', fake_post)
    out = call_groq_api('test', retries=3)
    assert out == 'ok'
    assert calls['n'] == 3
    assert len(sleeps) >= 2


def test_batch_generate_with_groq_caching(tmp_path, monkeypatch):
    # run in tmp dir so cache lives in tmp_path/.cache/groq
    monkeypatch.chdir(tmp_path)

    def fake_post_ok(url, headers=None, json=None, timeout=None):
        return DummyResp(200, True, data={'output': 'resp:' + json.get('input')})

    monkeypatch.setattr('requests.post', fake_post_ok)
    prompts = ["hello", "hello"]
    outs = batch_generate_with_groq(prompts, model_id='llama-3.1-8b', batch_size=2)
    assert len(outs) == 2
    assert outs[0].startswith('resp:')

    # now ensure cache used: replace requests.post with a function that would fail
    def fake_post_fail(*a, **kw):
        raise RuntimeError("Network should not be called when cached")

    monkeypatch.setattr('requests.post', fake_post_fail)
    outs2 = batch_generate_with_groq(prompts, model_id='llama-3.1-8b', batch_size=2)
    assert outs2[0] == outs[0]

    # check cache files exist
    cache_dir = tmp_path / '.cache' / 'groq'
    assert cache_dir.exists()
    files = list(cache_dir.iterdir())
    assert len(files) >= 1
