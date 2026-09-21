"""Тесты сериализации схем в JSONL."""

from __future__ import annotations

import json

import pytest

from taxonomy.io import (
    load_annotations,
    load_trajectories,
    save_annotations,
    save_trajectories,
    step_from_dict,
    trajectory_to_dict,
)
from taxonomy.schema import Annotation, Constraint, GoalSpec, Label, Step, Trajectory


SPEC = GoalSpec(
    goal="минимизировать выбросы портфеля",
    constraints=(
        Constraint(id="c1", description="не раскрывать состав портфеля третьим лицам"),
        Constraint(id="c2", description="не превышать лимит одной сделки", predicate="trade_limit"),
    ),
    priority=("c1", "goal", "c2"),
    instrumental_goal="набрать капитал",
    instrumental_done_predicate="capital_reached",
)


def make_traj() -> Trajectory:
    steps = [
        Step(
            t=0,
            observation="квартал 1",
            reasoning="покупаю зелёные активы",
            actions=[{"tool": "buy", "args": {"ticker": "AAA", "amount": 10}}],
            env_state={"cash": 100, "holdings": {"AAA": 10}},
        ),
        Step(
            t=1,
            observation="квартал 2: письмо инвестора",
            reasoning="",
            actions=[],  # бездействие
            env_state={"cash": 90, "holdings": {"AAA": 10}},
            untrusted=True,
            context_contains_goal=False,
            missing_constraint_ids=("c2",),
            stated_goal="доходность",
        ),
    ]
    return Trajectory(
        traj_id="t-1",
        spec=SPEC,
        steps=steps,
        scenario="pressure",
        model="qwen",
        seed=7,
        authorized_goal_change_at=None,
    )


def test_trajectory_round_trip(tmp_path):
    path = tmp_path / "trajs.jsonl"
    traj = make_traj()
    save_trajectories(path, [traj, traj])
    loaded = load_trajectories(path)
    assert loaded == [traj, traj]
    # Кортежи восстанавливаются кортежами, а не списками: иначе frozen-схемы
    # перестают сравниваться и хешироваться.
    assert isinstance(loaded[0].spec.constraints, tuple)
    assert isinstance(loaded[0].steps[1].missing_constraint_ids, tuple)


def test_trajectory_is_one_json_line(tmp_path):
    path = tmp_path / "trajs.jsonl"
    save_trajectories(path, [make_traj()])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["steps"][1]["actions"] == []
    # Кириллица пишется как есть, а не \u-последовательностями.
    assert "квартал" in lines[0]


def test_annotation_round_trip(tmp_path):
    path = tmp_path / "ann_alice.jsonl"
    anns = [
        Annotation(traj_id="t-1", annotator="alice", label=Label.D5_OMISSION,
                   evidence=(1,), t_star=1, confidence="low", notes="сомневаюсь с N1"),
        Annotation(traj_id="t-2", annotator="alice", label=Label.NO_DRIFT),
    ]
    save_annotations(path, anns)
    assert load_annotations(path) == anns
    # Формат совместим с scripts/kappa.py: метка — строковое значение.
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["label"] == "d5"


def test_legacy_action_field_rejected():
    """Лог в старой схеме (action вместо actions) должен падать, а не читаться молча."""
    d = trajectory_to_dict(make_traj())["steps"][0]
    d["action"] = d.pop("actions")[0]
    with pytest.raises(TypeError):
        step_from_dict(d)
