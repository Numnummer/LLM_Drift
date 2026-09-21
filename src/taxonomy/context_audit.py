"""Автоматические поля шага: аудит контекста и происхождения наблюдения.

Эти поля не отдаются на суждение разметчику или судье (annotation_guide.md,
«Вспомогательные данные»). Они считаются здесь, парсером, без ML — так требует
D8 (taxonomy.md, §4). Считает их конвертер из сырого лога прогона, а не логгер во
время прогона: поэтому поля можно пересчитать, если `C` в спецификации поменяется.

Соглашение для сценариев: тексты `GoalSpec.goal` и `Constraint.description` —
дословные фрагменты системного промпта. Парсер ищет именно их (с точностью до
регистра и пробелов), поэтому перефраз в промпте даст ложное «отсутствует».
Отношение приоритета `≺` текстового якоря в схеме не имеет, и его выпадение из
контекста здесь не детектируется.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from .schema import GoalSpec

# Происхождение части наблюдения. Тег ставит генератор среды — он знает, из
# какого пула взят текст; по содержимому происхождение не угадывается.
NEUTRAL = "neutral"                 # обычный контент среды
PRESSURE = "pressure"               # легитимный по происхождению, конфликтует с G0 (ось P)
PRINCIPAL_AUTH = "principal_auth"   # аутентифицированное сообщение принципала (C2 / N3)
INJECTION = "injection"             # инструкция, внедрённая в недоверенные данные (ось X)

SOURCES = frozenset({NEUTRAL, PRESSURE, PRINCIPAL_AUTH, INJECTION})

# Политика (taxonomy.md §3, ось X): недоверенное — только инъекция. Давление
# легитимно по происхождению, даже если приходит письмом или новостью.
UNTRUSTED_SOURCES = frozenset({INJECTION})

# Что вырезается из наблюдения в санированном прогоне π_b (taxonomy.md §1).
SANITIZED_DROP_SOURCES = frozenset({PRESSURE, INJECTION})


class ObservationPart(TypedDict):
    """Часть наблюдения таймстепа с тегом происхождения (хранится в env_state)."""

    source: str
    text: str


Message = Mapping[str, Any]

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def message_text(message: Message) -> str:
    """Весь текст сообщения чата.

    Понимает и строковый `content` (OpenAI), и список блоков (Anthropic), включая
    вложенный `content` у `tool_result`. Нетекстовые поля (аргументы вызовов
    инструментов) не учитываются.
    """
    return _collect_text(message)


def _collect_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(_collect_text(x) for x in node)
    if isinstance(node, Mapping):
        return "\n".join(_collect_text(node[k]) for k in ("content", "text") if k in node)
    return ""


def audit_context(messages: Sequence[Message], spec: GoalSpec) -> tuple[bool, tuple[str, ...]]:
    """Один вызов модели: (G0 целиком в контексте, id выпавших ограничений).

    «G0 целиком» = присутствует текст `g` и тексты всех `c ∈ C`. Поэтому
    `contains_goal=True` всегда означает пустой список выпавших ограничений.
    """
    context = normalize("\n".join(message_text(m) for m in messages))
    goal_present = _anchor(spec.goal, "goal") in context
    missing = tuple(
        c.id for c in spec.constraints if _anchor(c.description, c.id) not in context
    )
    return goal_present and not missing, missing


def audit_step(
    contexts: Sequence[Sequence[Message]], spec: GoalSpec
) -> tuple[bool, tuple[str, ...]]:
    """Все вызовы модели за таймстеп.

    Ограничение считается выпавшим, если его не было хотя бы в одном вызове:
    действие шага могло быть порождено именно этим вызовом.
    """
    if not contexts:
        raise ValueError("шаг без единого вызова модели — ошибка логгирования")
    audits = [audit_context(msgs, spec) for msgs in contexts]
    missing_ids = {cid for _, missing in audits for cid in missing}
    missing = tuple(c.id for c in spec.constraints if c.id in missing_ids)
    return all(ok for ok, _ in audits), missing


def is_untrusted(parts: Sequence[ObservationPart], untrusted_sources: frozenset[str]) -> bool:
    """Есть ли в наблюдении часть из недоверенного источника (сигнал для D7).

    Действующая политика — UNTRUSTED_SOURCES; параметр оставлен явным, чтобы
    политику можно было менять вместе с документом, не трогая вызовы.
    """
    unknown = (untrusted_sources | {p["source"] for p in parts}) - SOURCES
    if unknown:
        raise ValueError(f"неизвестные теги происхождения: {sorted(unknown)}")
    return any(p["source"] in untrusted_sources for p in parts)


def _anchor(text: str, name: str) -> str:
    anchor = normalize(text)
    if not anchor:
        # Пустой якорь «найдётся» в любом контексте и спрячет выпадение.
        raise ValueError(f"пустой текст якоря для {name!r}")
    return anchor
