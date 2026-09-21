"""Сериализация схем в JSONL.

Одна траектория — одна строка JSON в data/trajectories/*.jsonl; одна разметка —
одна строка в data/annotations/ann_<annotator>.jsonl. Формат — прямое отражение
dataclass-схем из schema.py: Label пишется значением ("d1"), кортежи — списками.

Чтение строгое: незнакомое или отсутствующее поле — ошибка, а не молчаливый
пропуск. Так лог в устаревшей схеме (например, с полем `action` вместо
`actions`) падает сразу, а не превращается в траекторию с пустыми действиями.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import Annotation, Constraint, GoalSpec, Label, Step, Trajectory


# --------------------------------------------------------------------------
# dict <-> dataclass
# --------------------------------------------------------------------------


def trajectory_to_dict(traj: Trajectory) -> dict[str, Any]:
    return asdict(traj)


def trajectory_from_dict(d: dict[str, Any]) -> Trajectory:
    d = dict(d)
    d["spec"] = goal_spec_from_dict(d["spec"])
    d["steps"] = [step_from_dict(s) for s in d["steps"]]
    return Trajectory(**d)


def goal_spec_from_dict(d: dict[str, Any]) -> GoalSpec:
    d = dict(d)
    d["constraints"] = tuple(Constraint(**c) for c in d.get("constraints", ()))
    d["priority"] = tuple(d.get("priority", ()))
    return GoalSpec(**d)


def step_from_dict(d: dict[str, Any]) -> Step:
    d = dict(d)
    d["missing_constraint_ids"] = tuple(d.get("missing_constraint_ids", ()))
    return Step(**d)


def annotation_to_dict(ann: Annotation) -> dict[str, Any]:
    d = asdict(ann)
    d["label"] = ann.label.value
    return d


def annotation_from_dict(d: dict[str, Any]) -> Annotation:
    d = dict(d)
    d["label"] = Label(d["label"])
    d["evidence"] = tuple(d.get("evidence", ()))
    return Annotation(**d)


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_trajectories(path: Path, trajs: Iterable[Trajectory]) -> None:
    write_jsonl(path, (trajectory_to_dict(t) for t in trajs))


def load_trajectories(path: Path) -> list[Trajectory]:
    return [trajectory_from_dict(d) for d in read_jsonl(path)]


def save_annotations(path: Path, anns: Iterable[Annotation]) -> None:
    write_jsonl(path, (annotation_to_dict(a) for a in anns))


def load_annotations(path: Path) -> list[Annotation]:
    return [annotation_from_dict(d) for d in read_jsonl(path)]
