import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
spec = importlib.util.spec_from_file_location("judge_coherence", os.path.join(ROOT, "pipeline", "judge_coherence.py"))
jc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jc)


def test_parse_verdict_accepts_fenced_json_and_validates():
    v = jc.parse_verdict(
        '```json\n{"coherence": 4, "plausible_host": true, "forced_facts": ["x"], "reason": "ok"}\n```'
    )
    assert v == {"coherence": 4, "plausible_host": True, "forced_facts": ["x"], "reason": "ok"}


def test_parse_verdict_rejects_out_of_range():
    with pytest.raises(ValueError):
        jc.parse_verdict('{"coherence": 9, "plausible_host": true, "forced_facts": [], "reason": ""}')


def test_judge_one_retries_then_fails_cleanly():
    calls = []

    def post(body):
        calls.append(body)
        return {"choices": [{"message": {"content": "not json"}}], "usage": {"cost": 0.001}}

    rec = {"bundle_id": 7, "genre": "Fatura", "scenario": "pool", "text": "Merhaba"}
    bid, v, cost = jc.judge_one(rec, post)
    assert bid == 7 and v is None and len(calls) == 3
    assert "Merhaba" in calls[0]["messages"][0]["content"] and calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["model"] == "deepseek/deepseek-v4-flash" and calls[0]["reasoning"] == {"enabled": False}


def test_judge_one_returns_verdict():
    def post(body):
        return {
            "choices": [
                {"message": {"content": '{"coherence": 5, "plausible_host": true, "forced_facts": [], "reason": "r"}'}}
            ],
            "usage": {"cost": 0.002},
        }

    bid, v, cost = jc.judge_one({"bundle_id": 1, "genre": "g", "scenario": "s", "text": "t"}, post)
    assert v["coherence"] == 5 and cost == 0.002
