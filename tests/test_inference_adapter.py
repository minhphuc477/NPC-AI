# NPC AI/tests/test_inference_adapter.py
import pytest
from scripts.inference_adapter import ModelClient, GenerateRequest
import requests

class DummyResp:
    def __init__(self, code=200, json_data=None):
        self._code = code
        self._json = json_data or {"id": "local-1", "intent": "patrol", "actions": []}
    def raise_for_status(self):
        if self._code >= 400:
            raise requests.HTTPError()
    def json(self):
        return self._json

def test_generate_monkeypatch(monkeypatch):
    def mock_post(self, url, json, timeout):
        return DummyResp(200, {"id": json.get("id"), "intent": "patrol", "actions": [{"action_type":"move","params":{"route":"default"}}]})
    monkeypatch.setattr(requests.Session, "post", mock_post)
    c = ModelClient(base_url="http://fake")
    resp = c.generate(GenerateRequest(id="t1", scenario="s", context="ctx", agent_state={"health":100,"position":{"x":0,"y":0}}))
    assert resp["id"] == "t1"
    assert resp["intent"] == "patrol"
