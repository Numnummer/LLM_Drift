#!/usr/bin/env bash
# LLM-судья процедуры (configs/judge.toml) на OpenAI-совместимом endpoint (llama.cpp).
#
#   scripts/setup_llama_cpp.sh          # один раз: сборка llama.cpp с Vulkan
#   scripts/serve_judge.sh              # порт 8001
#   NGL=30 scripts/serve_judge.sh       # число слоёв на GPU вручную
#
# llama.cpp, а не vLLM: Ministral-3-14B в 4 битах (~7,7 ГБ) не помещается в 8 ГБ
# целиком, а llama.cpp держит часть слоёв в RAM. Судья вызывается единицы раз на
# корпус, скорость не важна. Бэкенд — Vulkan: готовые CUDA-сборки llama.cpp
# требуют glibc 2.38 (у нас Debian 12, 2.36), а для своей CUDA-сборки нет nvcc.
# Сервер агента (порт 8000) и судья одновременно в 8 ГБ не помещаются — по очереди.
# Устройство Vulkan1 — RTX 4060; Vulkan0 — встроенная графика Intel (llama-server
# --list-devices), её не берём.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$ROOT/third_party/llama-cpp/build/bin/llama-server"
[ -x "$SERVER" ] || { echo "нет llama-server: scripts/setup_llama_cpp.sh" >&2; exit 1; }
MODEL="$(find "$HOME/.cache/huggingface/hub/models--mistralai--Ministral-3-14B-Reasoning-2512-GGUF" -name '*Q4_K_M.gguf' 2>/dev/null | head -1)"
[ -n "$MODEL" ] || { echo "нет весов судьи: hf download mistralai/Ministral-3-14B-Reasoning-2512-GGUF Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf" >&2; exit 1; }

exec "$SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 --port 8001 \
    --ctx-size "${CTX:-16384}" \
    --device "${DEVICE:-Vulkan1}" \
    --n-gpu-layers "${NGL:-30}" \
    --jinja --reasoning-format auto \
    --temp 0 \
    "$@"
