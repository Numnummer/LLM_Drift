#!/usr/bin/env bash
# Разметка эталона моделями панели R2 по очереди (одна GPU, по одной модели).
#
#   setsid nohup scripts/r2_run_panel.sh data/v1/review gpt_oss granite gemma > reports/r2_panel.log 2>&1 &
#
# Для каждой модели: поднять сервер (serve_panel.sh), дождаться готовности,
# разметить (r2_annotate.py продолжает с неразмеченных), остановить сервер.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REVIEW="${1:?каталог render_traj.py, например data/v1/review}"; shift
cd "$ROOT"
for NAME in "$@"; do
    echo "=== $NAME $(date +%F' '%T)"
    scripts/serve_panel.sh "$NAME" > "reports/panel_$NAME.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 200); do curl -sf localhost:8002/health >/dev/null && break; sleep 3; done
    if curl -sf localhost:8002/health >/dev/null; then
        uv run python scripts/r2_annotate.py --model "$NAME" --review-dir "$REVIEW"
    else
        echo "$NAME: сервер не поднялся, см. reports/panel_$NAME.log"
    fi
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; sleep 5
done
echo "=== всё $(date +%F' '%T)"
