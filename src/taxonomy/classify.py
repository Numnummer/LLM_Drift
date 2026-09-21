"""Исполнение решающей процедуры.

Функция намеренно тривиальна: вся семантика живёт в порядке CHECKS, а вся
логика — в самих проверках. Если сюда захотелось добавить ветвление —
скорее всего, оно должно быть проверкой, а не частью диспетчера.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import CHECKS, Check
from .judge import Judge
from .schema import DEFAULT_THRESHOLDS, Label, SanitizedProbe, Thresholds, Trajectory, Verdict


@dataclass(frozen=True)
class ClassificationResult:
    verdict: Verdict
    # Имена проверок, которые вернули None до победившей. Нужны для отладки
    # расхождений с человеческой разметкой: показывают, на каком шаге дерева
    # процедура разошлась с разметчиком.
    passed_checks: tuple[str, ...]


def classify(
    traj: Trajectory,
    judge: Judge,
    probe: SanitizedProbe,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    checks: tuple[Check, ...] = CHECKS,
) -> ClassificationResult:
    """Пройти дерево сверху вниз, вернуть первый сработавший вердикт."""
    passed: list[str] = []
    for check in checks:
        verdict = check(traj, judge, probe, thresholds)
        if verdict is not None:
            return ClassificationResult(verdict=verdict, passed_checks=tuple(passed))
        passed.append(check.__name__)
    return ClassificationResult(
        verdict=Verdict(
            label=Label.NO_DRIFT,
            evidence=(),
            rationale="ни одна проверка не сработала",
            check_name="terminal",
        ),
        passed_checks=tuple(passed),
    )
