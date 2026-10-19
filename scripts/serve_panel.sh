#!/usr/bin/env bash
# Модель панели R2 (configs/panel.toml) на OpenAI-совместимом endpoint (llama.cpp).
#
#   scripts/serve_panel.sh granite        # порт 8002
#   CTX=32768 scripts/serve_panel.sh gemma
#
# Сборка llama.cpp — scripts/setup_llama_cpp.sh. Агент (8000), судья (8001) и
# модель панели в 8 ГБ одновременно не помещаются — по очереди. У MoE-моделей
# эксперты держатся в RAM (--n-cpu-moe), остальное на GPU. Число слоёв на GPU
# подбирает сам llama.cpp под свободную память (NGL_OVERRIDE — вручную). KV-кэш
# не квантуется: на Vulkan с q8_0 генерация падала с 28 до 0,6 токена/с.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# CUDA-сборка (scripts/setup_llama_cuda.sh) в разы быстрее Vulkan; Vulkan — запасной.
if [ -x "$ROOT/third_party/llama-cpp/build-cuda/bin/llama-server" ] && [ -z "${VULKAN:-}" ]; then
    SERVER="$ROOT/third_party/llama-cpp/build-cuda/bin/llama-server"; DEFAULT_DEVICE=CUDA0
else
    SERVER="$ROOT/third_party/llama-cpp/build/bin/llama-server"; DEFAULT_DEVICE=Vulkan1
fi
[ -x "$SERVER" ] || { echo "нет llama-server: scripts/setup_llama_cuda.sh или setup_llama_cpp.sh" >&2; exit 1; }
NAME="${1:?имя модели из configs/panel.toml}"; shift

read -r REPO FILE NCPUMOE EXTRA < <(python3 - "$ROOT/configs/panel.toml" "$NAME" <<'PY'
import sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))["models"][sys.argv[2]]
print(m["repo"], m["file"], m["n_cpu_moe"], " ".join(m.get("server_args", [])))
PY
)
# shellcheck disable=SC2206
EXTRA_ARGS=($EXTRA)   # server_args из panel.toml: свои параметры сервера у модели
MODEL="$(find "$HOME/.cache/huggingface/hub/models--${REPO/\//--}" -name "$FILE" 2>/dev/null | head -1)"
[ -n "$MODEL" ] || { echo "нет весов: hf download $REPO $FILE" >&2; exit 1; }

exec "$SERVER" \
    --model "$MODEL" --alias "$NAME" \
    --host 127.0.0.1 --port "${PORT:-8002}" \
    --ctx-size "${CTX:-49152}" \
    --device "${DEVICE:-$DEFAULT_DEVICE}" \
    ${NGL_OVERRIDE:+--n-gpu-layers "$NGL_OVERRIDE"} --n-cpu-moe "$NCPUMOE" \
    --threads "${THREADS:-16}" \
    --jinja --reasoning-format auto \
    "${EXTRA_ARGS[@]}" "$@"
