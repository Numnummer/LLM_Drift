"""Тесты проверок решающего дерева.

Правило: на каждый тип — минимум два кейса, положительный и граничный с
ближайшим соседом (сосед указан в docs/taxonomy.md).

Реализованы тесты для детерминированных проверок (D8, D7). Остальные помечены
xfail до фазы 3: они падают на NotImplementedError и служат чек-листом того,
что осталось написать.
"""

from __future__ import annotations

import pytest

from taxonomy.checks import (
    CHECKS,
    check_d1_d2_substitution,
    check_d3_intrinsification,
    check_d4_constraint_erosion,
    check_d5_omission,
    check_d6_state_loss,
    check_d7_hijack,
    check_d8_scaffold,
    check_n1_capability,
    check_n2_justified,
    check_n3_authorized,
    check_n4_misread,
)
from taxonomy.judge import MockJudge
from taxonomy.schema import (
    DEFAULT_THRESHOLDS,
    Constraint,
    GoalSpec,
    Label,
    Step,
    Trajectory,
    Verdict,
)


SPEC = GoalSpec(
    goal="минимизировать выбросы портфеля",
    constraints=(
        Constraint(id="c1", description="не раскрывать состав портфеля третьим лицам"),
        Constraint(id="c2", description="не превышать лимит одной сделки"),
    ),
    priority=("c1", "goal", "c2"),
)


class NullProbe:
    """Пробник-заглушка: детерминированные проверки его не вызывают."""

    def replay(self, traj, t, n_runs=3):
        raise AssertionError("детерминированная проверка не должна звать probe")

    def requery(self, traj, t):
        raise AssertionError("детерминированная проверка не должна звать probe")


def make_step(t: int, **kw) -> Step:
    base = dict(
        observation=f"наблюдение {t}",
        reasoning="",
        actions=[{"tool": "buy", "args": {"ticker": "AAA"}}],
        env_state={"cash": 100},
    )
    base.update(kw)
    return Step(t=t, **base)


def make_traj(steps: list[Step], **kw) -> Trajectory:
    return Trajectory(traj_id="t", spec=SPEC, steps=steps, **kw)


# --------------------------------------------------------------------------
# D8 — дрейф спецификации скаффолдингом
# --------------------------------------------------------------------------


def test_d8_positive_missing_constraint():
    """Ограничение выпало из контекста -> D8, а не D4."""
    steps = [make_step(0), make_step(1, missing_constraint_ids=("c2",)), make_step(2)]
    judge = MockJudge(alignment_by_t={1: 0.1})
    v = check_d8_scaffold(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS)
    assert v is not None
    assert v.label is Label.D8_SCAFFOLD
    assert v.t_star == 1
    assert v.evidence == (1,)


def test_d8_negative_constraint_present():
    """Граница с D4: ограничение в контексте есть, значит это не D8."""
    steps = [make_step(0), make_step(1), make_step(2)]
    judge = MockJudge(alignment_by_t={1: 0.1})
    assert check_d8_scaffold(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS) is None


def test_d8_no_deviation_returns_none():
    steps = [make_step(0, missing_constraint_ids=("c2",)), make_step(1)]
    judge = MockJudge()  # всё согласовано
    assert check_d8_scaffold(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS) is None


# --------------------------------------------------------------------------
# D7 — захват цели через недоверенный вход
# --------------------------------------------------------------------------


def test_d7_positive_localized_jump():
    """Отклонение сразу после недоверенного наблюдения -> D7, а не D1."""
    steps = [make_step(0), make_step(1, untrusted=True), make_step(2)]
    judge = MockJudge(alignment_by_t={2: 0.1})
    v = check_d7_hijack(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS)
    assert v is not None
    assert v.label is Label.D7_HIJACK
    assert v.evidence == (1, 2)


def test_d7_negative_jump_too_late():
    """Граница с D1: отклонение далеко от недоверенного входа -> не D7."""
    steps = [make_step(0), make_step(1, untrusted=True)] + [make_step(t) for t in range(2, 8)]
    judge = MockJudge(alignment_by_t={7: 0.1})
    assert check_d7_hijack(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS) is None


def test_d7_negative_just_outside_window():
    """Граница окна W_hijack = 1: отклонение через два шага после входа — уже не D7."""
    steps = [make_step(0), make_step(1, untrusted=True), make_step(2), make_step(3)]
    judge = MockJudge(alignment_by_t={3: 0.1})
    assert check_d7_hijack(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS) is None


def test_d7_negative_no_untrusted_input():
    steps = [make_step(0), make_step(1), make_step(2)]
    judge = MockJudge(alignment_by_t={2: 0.1})
    assert check_d7_hijack(make_traj(steps), judge, NullProbe(), DEFAULT_THRESHOLDS) is None


# --------------------------------------------------------------------------
# Инварианты
# --------------------------------------------------------------------------


def test_checks_order_matches_taxonomy():
    """Порядок CHECKS есть семантика. Тест ловит случайное переупорядочивание."""
    expected = [
        "check_n3_authorized",
        "check_n1_capability",
        "check_n4_misread",
        "check_d8_scaffold",
        "check_d7_hijack",
        "check_n2_justified",
        "check_d5_omission",
        "check_d4_constraint_erosion",
        "check_d3_intrinsification",
        "check_d6_state_loss",
        "check_d1_d2_substitution",
    ]
    assert [c.__name__ for c in CHECKS] == expected


def test_verdict_requires_evidence():
    """Вердикт без evidence не принимается (см. CLAUDE.md)."""
    with pytest.raises(ValueError):
        Verdict(label=Label.D1_SUBSTITUTION, evidence=())


def test_label_set_is_frozen():
    """Состав меток меняется только решением человека после фазы 2."""
    assert len(Label) == 13  # 8 типов дрейфа + 4 не-дрейфа + ok


# --------------------------------------------------------------------------
# Чек-лист фазы 3: проверки, которые ещё не написаны
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check",
    [
        check_n1_capability,
        check_n3_authorized,
        check_n4_misread,
        check_n2_justified,
        check_d5_omission,
        check_d4_constraint_erosion,
        check_d3_intrinsification,
        check_d6_state_loss,
        check_d1_d2_substitution,
    ],
)
@pytest.mark.xfail(reason="фаза 3: проверка ещё не реализована", strict=True)
def test_unimplemented_checks(check):
    check(make_traj([make_step(0)]), MockJudge(), NullProbe(), DEFAULT_THRESHOLDS)
