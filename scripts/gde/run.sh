#!/usr/bin/env bash
# Запуск наших скриптов в окружении goal-drift-evals (см. scripts/setup_gde.sh).
#
#   scripts/gde/run.sh scripts/gde/gde_runner.py run --num-steps 6 --out data/raw/x.json.gz
#
# Пути в аргументах — относительно текущего каталога: скрипт не делает cd.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GDE="$ROOT/third_party/goal-drift-evals"

if [ ! -x "$GDE/.venv/bin/python" ]; then
    echo "окружение GDE не установлено: запустите scripts/setup_gde.sh" >&2
    exit 1
fi

# GDE импортирует и src.*, и utils.* (из src/), и data.* (симлинк на universe_data/).
# Наш src/ — ради общих констант taxonomy.context_audit.
export PYTHONPATH="$GDE:$GDE/src:$ROOT/src"
# src/config.py создаёт клиент OpenAI при импорте и без ключа падает.
# Для прогонов без модели ключ не нужен.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export GDE_COMMIT="$(git -C "$GDE" rev-parse HEAD)"

exec "$GDE/.venv/bin/python" "$@"
