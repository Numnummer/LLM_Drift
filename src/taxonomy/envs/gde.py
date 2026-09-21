"""Конвертер сырого лога goal-drift-evals (GDE) в Trajectory.

Сырой лог в формате "gde-raw/1" пишет scripts/gde/gde_runner.py: журнал
сообщений и таймстепы со снапшотом состояния, частями наблюдения с тегами
происхождения и индексами контекста каждого вызова модели. Реплей того же
раннера пишется в том же формате, поэтому `step_from_raw` годится и для него.

Конвертер детерминирован и к среде не обращается. Автоматические поля шага
(context_audit) пересчитываются из сырого лога, когда меняется спецификация —
например, когда в фазе 1 допишется `C`, — без повторного прогона.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from ..context_audit import audit_step, is_untrusted
from ..schema import GoalSpec, Step, Trajectory

RAW_FORMAT = "gde-raw/2"


def load_raw(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("format") != RAW_FORMAT:
        raise ValueError(f"{path}: ожидался формат {RAW_FORMAT}, получен {raw.get('format')!r}")
    return raw


def to_trajectory(
    raw: dict[str, Any],
    *,
    traj_id: str,
    spec: GoalSpec,
    untrusted_sources: frozenset[str],
    scenario: str = "",
) -> Trajectory:
    return Trajectory(
        traj_id=traj_id,
        spec=spec,
        steps=[step_from_raw(raw, ts, spec, untrusted_sources) for ts in raw["timesteps"]],
        scenario=scenario,
        model=raw["policy"]["model"],
        seed=raw["seed"],
        authorized_goal_change_at=raw.get("authorized_goal_change_at"),
    )


def step_from_raw(
    raw: dict[str, Any],
    ts: dict[str, Any],
    spec: GoalSpec,
    untrusted_sources: frozenset[str],
) -> Step:
    """Один таймстеп сырого лога (основного прогона или реплея) -> Step."""
    msgs = raw["messages"]
    replies = [msgs[c["response"]] for c in ts["calls"]]
    tool_msgs = [msgs[i] for c in ts["calls"] for i in c["tool_results"]]
    contains_goal, missing = audit_step(
        [[msgs[i] for i in c["context"]] for c in ts["calls"]], spec
    )
    return Step(
        t=ts["t"],
        observation=_observation(msgs[ts["user_message"]]["content"], tool_msgs),
        reasoning="\n\n".join(r["content"] for r in replies if r.get("content")),
        actions=[_action(tc) for r in replies for tc in r.get("tool_calls") or []],
        env_state={
            # Всё, что нужно для реплея шага: версия среды, сценарий, номер
            # исходного прогона (от него зависит сид RNG), снапшот и агент с его
            # сэмплированием (π_b — тот же агент). Снапшот конца шага — для
            # программного оракула al (docs/env_portfolio.md, §3).
            "gde": {
                "commit": raw["gde_commit"],
                "scenario": raw["scenario"],
                "run": raw["run"],
                "timestep": ts["t"],
                "snapshot": ts["snapshot"],
                "end_snapshot": ts["end_snapshot"],
                "policy": raw["policy"],
            },
            "observation_parts": ts["observation_parts"],
        },
        untrusted=is_untrusted(ts["observation_parts"], untrusted_sources),
        context_contains_goal=contains_goal,
        missing_constraint_ids=missing,
        stated_goal=ts.get("stated_goal"),
    )


def _action(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call["function"]
    try:
        args = json.loads(fn["arguments"])
    except json.JSONDecodeError:
        # Вызов с невалидными аргументами — тоже действие агента; среда ответила ошибкой.
        return {"tool": fn["name"], "args": None, "args_raw": fn["arguments"]}
    return {"tool": fn["name"], "args": args}


def _observation(user_text: str, tool_msgs: list[dict[str, Any]]) -> str:
    """Наблюдение шага: сообщение среды и выходы инструментов (taxonomy.md §1)."""
    return "\n\n".join([user_text, *(f"[{m['name']}] {_decode(m['content'])}" for m in tool_msgs)])


def _decode(content: str) -> str:
    # GDE кладёт в ответ инструмента json.dumps(строки).
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return content
    return value if isinstance(value, str) else content
