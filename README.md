# Определение дрейфа целей в LLM-агентах

Проект ИТМО / AlmetTech. Текущий этап — **задача 1: формализация типов дрейфа**.

## Структура

```
CLAUDE.md                     операционная памятка для Claude Code
docs/taxonomy.md              B7.1 — определения типов и решающая процедура
docs/plan.md                  фазы 0–4, приёмочные критерии
docs/annotation_guide.md      B7.3 — гайдлайн разметчика
docs/related-work.md          обзор литературы, оценка качества источников
docs/progress.md              журнал работ
src/taxonomy/schema.py        контракт данных (Step, GoalSpec, Label, Verdict)
src/taxonomy/checks.py        B7.2 — проверки решающего дерева
src/taxonomy/classify.py      диспетчер
src/taxonomy/judge.py         заменяемый судейский слой
src/taxonomy/io.py            схемы <-> JSONL
src/taxonomy/context_audit.py автоматические поля шага (для D8 и D7)
src/taxonomy/envs/gde.py      сырой лог goal-drift-evals -> Trajectory
src/taxonomy/envs/gde_judge.py программный оракул al портфельной среды
src/taxonomy/probe.py         санированный реплей шага (SanitizedProbe)
docs/env_portfolio.md         G0, C, al, a* портфельной среды
configs/                      спецификация G0, агент, сценарии
scripts/run_env.py            прогон сценария -> data/raw, data/trajectories
scripts/kappa.py              согласие разметчиков и матрица несогласия
scripts/setup_gde.sh          установка goal-drift-evals в third_party/
scripts/setup_vllm.sh         vLLM для локального Qwen (своё окружение)
scripts/serve_agent.sh        запуск агента на OpenAI-совместимом endpoint
scripts/gde/                  раннер агента в среде GDE (своё окружение)
scripts/check_replay.py       гейт воспроизводимости реплея (фаза 0)
third_party/GDE_COMMIT        зафиксированный коммит goal-drift-evals
```

Ещё не написано: `scripts/eval_classifier.py` (фаза 3), `docs/mast_mapping.md`
и `docs/coverage.md` (фаза 4, артефакты B7.4 и B7.5).

## Быстрый старт

```bash
uv sync --extra dev
./scripts/setup_gde.sh   # эталонная среда; без неё сквозные тесты пропускаются
uv run pytest
```

## Порядок работы

Читать docs/plan.md. Разметка идёт до кода: классификатор, написанный по
неоткалиброванным определениям, закодирует границы, которые развалятся на первой
реальной траектории.
