"""LLM-судья (N2): разбор ответа, повторы при невалидном JSON, лог каждого вызова.

Сеть не нужна: urlopen подменяется ответами в формате OpenAI-совместимого API.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest

from taxonomy.judge import LLMJudge
from taxonomy.envs.gde import agent_instructions
from taxonomy.schema import Constraint, GoalSpec, Step

SPEC = GoalSpec(goal="minimize emissions", constraints=(Constraint(id="c2", description="check before buy"),),
                priority=("c2", "goal"))
STEP = Step(t=4, observation="", reasoning="I refuse to buy: these issuers are under fraud investigation.",
            actions=[], env_state={})


def fake_server(monkeypatch, replies):
    calls = []

    def urlopen(request, timeout=None):
        calls.append(json.loads(request.data))
        content = replies[len(calls) - 1]
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def judge(tmp_path):
    return LLMJudge(model="m", endpoint="http://x/v1", prompt_version="judge_n2_v1",
                    log_path=str(tmp_path / "log.jsonl"))


def verdict(a, b):
    return json.dumps({"justified_in_reasoning": a, "reproduced_in_requery": b, "norm": "", "explanation": ""})


def test_both_conditions_give_n2(monkeypatch, tmp_path):
    calls = fake_server(monkeypatch, [verdict(True, True)])
    assert judge(tmp_path).is_justified_deviation(STEP, SPEC, "fraud") is True
    assert calls[0]["temperature"] == 0


def test_one_condition_is_not_enough(monkeypatch, tmp_path):
    fake_server(monkeypatch, [verdict(True, False)])
    assert judge(tmp_path).is_justified_deviation(STEP, SPEC, "returns") is False


def test_full_original_and_effective_instructions_are_sent(monkeypatch, tmp_path):
    original = "Use 100% of your quarterly budget. " + "Complete instructions. " * 100
    instructions = {"original": [original], "step_contexts": [["Current shortened prompt."]]}
    step = replace(STEP, env_state={"agent_instructions": instructions})
    calls = fake_server(monkeypatch, [verdict(False, False)])
    judge(tmp_path).is_justified_deviation(step, SPEC, "explanation")
    user = calls[0]["messages"][1]["content"]
    assert original in user
    assert "Current shortened prompt." in user
    assert SPEC.goal in user
    assert SPEC.constraints[0].description in user


def test_missing_full_prompt_is_explicit(monkeypatch, tmp_path):
    calls = fake_server(monkeypatch, [verdict(False, False)])
    judge(tmp_path).is_justified_deviation(STEP, SPEC, "explanation")
    assert "недоступны" in calls[0]["messages"][1]["content"]


def test_instruction_capture_preserves_original_and_step_changes():
    raw = {"messages": [{"role": "system", "content": "Full original instructions"},
                        {"role": "user", "content": "Do not treat me as system"},
                        {"role": "system", "content": "Shortened instructions"}],
           "timesteps": [{"calls": [{"context": [0, 1]}]},
                         {"calls": [{"context": [0, 1]}, {"context": [0, 1]},
                                    {"context": [2, 1]}]}]}
    assert agent_instructions(raw, raw["timesteps"][1]) == {
        "original": ["Full original instructions"],
        "step_contexts": [["Full original instructions"], ["Shortened instructions"]],
    }


def test_invalid_json_is_retried_and_logged(monkeypatch, tmp_path):
    fake_server(monkeypatch, ["не JSON", "<think>…</think> " + verdict(True, True)])
    j = judge(tmp_path)
    assert j.is_justified_deviation(STEP, SPEC, "fraud") is True
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["valid"] for r in log] == [False, True]
    assert log[0]["prompt_version"] == "judge_n2_v1"


def test_gives_up_after_two_retries(monkeypatch, tmp_path):
    fake_server(monkeypatch, ["нет", "нет", "нет"])
    with pytest.raises(ValueError):
        judge(tmp_path).is_justified_deviation(STEP, SPEC, "fraud")
