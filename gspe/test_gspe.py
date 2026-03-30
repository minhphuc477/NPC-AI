#!/usr/bin/env python3
"""Lightweight GSPE unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gspe.gspe_model import FIELD_VOCABS, GSPE, GSPEConfig, encode_game_state


def test_vocab_integrity() -> None:
    for field, values in FIELD_VOCABS.items():
        assert values, f"{field}: empty vocab"
        assert len(values) == len(set(values)), f"{field}: duplicate vocab entries"


def test_encode_game_state() -> None:
    sample = {
        "alert_state": "investigating",
        "location": "harbor",
        "behavior_state": "negotiating",
        "suspicion_level": 3,
        "time_of_day": "dusk",
        "weather": "rain",
        "nearby_threat": True,
        "active_quest": False,
        "recent_event": "theft",
    }
    ids = encode_game_state(sample)
    assert set(ids.keys()) == set(FIELD_VOCABS.keys())
    for field, idx in ids.items():
        assert 0 <= int(idx) < len(FIELD_VOCABS[field]), f"{field}: out of range {idx}"

    unknown = encode_game_state({"location": "unmapped_place"})
    assert unknown["location"] == 0


def test_shape_and_determinism() -> None:
    cfg = GSPEConfig()
    model = GSPE(cfg).eval()
    batch_size = 2
    field_ids = {
        field: torch.zeros(batch_size, dtype=torch.long)
        for field in FIELD_VOCABS.keys()
    }
    out = model(field_ids)
    assert tuple(out.shape) == (batch_size, cfg.n_prefix_tokens, cfg.lm_hidden_dim)

    single = {field: torch.tensor([0], dtype=torch.long) for field in FIELD_VOCABS.keys()}
    out1 = model(single)
    out2 = model(single)
    assert torch.allclose(out1, out2)

    alt = dict(single)
    alt["behavior_state"] = torch.tensor([3], dtype=torch.long)
    out3 = model(alt)
    assert not torch.allclose(out1, out3)


def run_all() -> int:
    tests = [test_vocab_integrity, test_encode_game_state, test_shape_and_determinism]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"tests={len(tests)} failures={failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(run_all())
