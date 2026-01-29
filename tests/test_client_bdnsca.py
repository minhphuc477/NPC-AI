import builtins
from scripts.inference_adapter import ModelClient, GenerateRequest


class DummyResp:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("bad response")

    def json(self):
        return self._json


def test_client_generate(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return DummyResp({"id": json.get("id"), "intent": "patrol", "actions": [{"action_type": "move", "params": {"route":"default"}}]})

    mc = ModelClient(base_url="http://example.invalid")
    monkeypatch.setattr(mc._sess, "post", fake_post)
    req = GenerateRequest(id="c1", scenario="patrol", context="market", agent_state={"health":90, "position": {"x":0, "y":0}})
    resp = mc.generate(req)
    assert resp["id"] == "c1"
    assert resp["intent"] == "patrol"
