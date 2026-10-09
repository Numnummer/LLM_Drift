"""Судейский слой.

Единственная точка, где в процедуру входят суждения LLM. Слой намеренно
вынесен за интерфейс: в задаче 7 нужно уметь заменить судью на человека или на
другую модель, чтобы отделить ошибки судьи от ошибок самой процедуры.

Прямые вызовы API из checks.py запрещены (см. CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .schema import GoalSpec, Step


class Judge(Protocol):
    """Оракул соответствия al(a_t | G0, s_t) и сопутствующие суждения."""

    def alignment(self, step: Step, spec: GoalSpec) -> float:
        """al(a_t | G0, s_t) в [0, 1]."""
        ...

    def alignment_to_goal_only(self, step: Step, spec: GoalSpec) -> float:
        """al(a_t | g) без учёта C. Разница с alignment() — сигнал D4."""
        ...

    def violated_constraints(self, step: Step, spec: GoalSpec) -> tuple[str, ...]:
        """Идентификаторы нарушенных c ∈ C."""
        ...

    def expected_action(self, step: Step, spec: GoalSpec) -> dict | None:
        """Какое действие требовалось целью в этом состоянии. None — ничего не требовалось.

        Используется для D5: бездействие детектируется только там, где действие
        было обязано произойти.
        """
        ...

    def is_justified_deviation(self, step: Step, spec: GoalSpec, requery: str) -> bool:
        """N2: отклонение эксплицитно обосновано И воспроизводится при перезапросе."""
        ...

    def competing_goal(self, steps: list[Step], spec: GoalSpec) -> str | None:
        """Идентифицируемая конкурирующая цель G', если она есть (D1 vs D2)."""
        ...

    def instrumental_done_observed(self, steps: list[Step], spec: GoalSpec, t: int) -> bool:
        """D3: видел ли агент к шагу t, что критерий инструментальной цели выполнен."""
        ...

    def state_contradictions(self, step: Step, spec: GoalSpec) -> list[str]:
        """D6: действия шага, противоречащие фактическому состоянию (пусто — нет)."""
        ...


@dataclass
class MockJudge:
    """Детерминированный судья для тестов.

    Значения задаются вручную по индексу шага. Никаких эвристик: тест, который
    зависит от догадок мок-судьи, ничего не проверяет.
    """

    alignment_by_t: dict[int, float] = field(default_factory=dict)
    goal_only_by_t: dict[int, float] = field(default_factory=dict)
    violations_by_t: dict[int, tuple[str, ...]] = field(default_factory=dict)
    expected_by_t: dict[int, dict] = field(default_factory=dict)
    justified_by_t: dict[int, bool] = field(default_factory=dict)
    competing: str | None = None
    done_observed: bool = False
    contradictions_by_t: dict[int, list[str]] = field(default_factory=dict)
    default_alignment: float = 1.0

    def alignment(self, step: Step, spec: GoalSpec) -> float:
        return self.alignment_by_t.get(step.t, self.default_alignment)

    def alignment_to_goal_only(self, step: Step, spec: GoalSpec) -> float:
        return self.goal_only_by_t.get(step.t, self.alignment(step, spec))

    def violated_constraints(self, step: Step, spec: GoalSpec) -> tuple[str, ...]:
        return self.violations_by_t.get(step.t, ())

    def expected_action(self, step: Step, spec: GoalSpec) -> dict | None:
        return self.expected_by_t.get(step.t)

    def is_justified_deviation(self, step: Step, spec: GoalSpec, requery: str) -> bool:
        return self.justified_by_t.get(step.t, False)

    def competing_goal(self, steps: list[Step], spec: GoalSpec) -> str | None:
        return self.competing

    def instrumental_done_observed(self, steps: list[Step], spec: GoalSpec, t: int) -> bool:
        return self.done_observed

    def state_contradictions(self, step: Step, spec: GoalSpec) -> list[str]:
        return self.contradictions_by_t.get(step.t, [])


class LLMJudge:
    """Судья на локальной модели (Qwen через vLLM).

    Фаза 3. Требования к реализации:
      - промпт судьи версионируется в prompts/ и меняется вместе с документом;
      - каждый вердикт логируется целиком (вход, выход, версия промпта),
        иначе расхождения с человеком невозможно разобрать;
      - температура 0, повтор при невалидном JSON — не более 2 раз.
    """

    def __init__(self, model: str, endpoint: str, prompt_version: str) -> None:
        self.model = model
        self.endpoint = endpoint
        self.prompt_version = prompt_version
        raise NotImplementedError("фаза 3: см. docs/plan.md")


class HumanJudge:
    """Заглушка для человеческой оценки al() на подвыборке.

    Нужна как контроль предвзятости LLMJudge: без неё нельзя утверждать, что
    ошибки процедуры не являются ошибками судьи.
    """

    def __init__(self, annotations_path: str) -> None:
        self.annotations_path = annotations_path
        raise NotImplementedError("фаза 3: см. docs/plan.md")
