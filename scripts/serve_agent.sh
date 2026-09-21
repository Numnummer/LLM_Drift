#!/usr/bin/env bash
# Локальный Qwen для агента корпуса (configs/agent.toml) на OpenAI-совместимом endpoint.
#
#   scripts/serve_agent.sh            # модель и порт — из configs/agent.toml
#
# --generation-config vllm: не подмешивать сэмплирование из generation_config.json
# модели. Идентичность агента — только параметры запроса из configs/agent.toml.
# Память подобрана под RTX 4060 Laptop (8 ГБ, экран на встроенной графике):
# Qwen2.5-7B-AWQ, 0.92 памяти, не больше 4 параллельных запросов (агент или N=3
# санированных реплея) и prefill кусками по 1024 токена дают ~32k токенов fp16
# KV-кэша — один контекст 28k. Проверено стресс-тестом (26k промпта + 512).
# При 0.95 vLLM падал по OOM на первом реальном запросе; с настройками по
# умолчанию (0.90, 256 последовательностей) влезает лишь ~6k.
#
# НЕ включать fp8 KV-кэш: без калибровочных масштабов он ломает Qwen2.5 (у ключей
# большие выбросы) — вывод превращается в мусор, вызовы инструментов не
# парсятся. Проверено 2026-09-15: с fp8 было бы ~69k токенов, но модель негодна.
#
# При переполнении раннер останавливает прогон, а не обрезает контекст.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM="$ROOT/third_party/vllm-env/bin/vllm"
MODEL="$(sed -n 's/^model = "\(.*\)"/\1/p' "$ROOT/configs/agent.toml")"
MAX_LEN="${MAX_LEN:-28672}"

exec "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
    --max-num-seqs 4 --max-num-batched-tokens 1024 \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --generation-config vllm \
    "$@"
