#!/usr/bin/env bash
# Локальный агент корпуса на OpenAI-совместимом endpoint (vLLM).
#
#   scripts/serve_agent.sh                                     # configs/agent.toml
#   AGENT=configs/agent_qwen25_7b.toml scripts/serve_agent.sh  # другой агент
#
# Модель берётся из конфига агента, как и необязательная секция [serve]:
# окружение vLLM, парсер вызовов инструментов, потолок контекста и доп. флаги.
# Без секции — значения для Qwen2.5-7B-AWQ (vllm-env 0.11, hermes, 28672).
#
# --generation-config vllm: не подмешивать сэмплирование из generation_config.json
# модели. Идентичность агента — только параметры запроса из конфига агента.
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
AGENT="${AGENT:-$ROOT/configs/agent.toml}"

# Секция [serve] читается stdlib-парсером TOML (Python 3.11+) и превращается в
# присваивания shell; serve.env — переменные окружения для vLLM.
eval "$(python3 - "$AGENT" <<'EOF'
import shlex, sys, tomllib
cfg = tomllib.load(open(sys.argv[1], "rb"))
s = cfg.get("serve", {})
print(f"MODEL={shlex.quote(cfg['model'])}")
print(f"VLLM_ENV={shlex.quote(s.get('vllm_env', 'vllm-env'))}")
print(f"TOOL_PARSER={shlex.quote(s.get('tool_call_parser', 'hermes'))}")
print(f"CFG_MAX_LEN={int(s.get('max_model_len', 28672))}")
print(f"EXTRA={shlex.quote(' '.join(s.get('extra_args', [])))}")
for k, v in s.get("env", {}).items():
    print(f"export {k}={shlex.quote(str(v))}")
EOF
)"

VLLM="$ROOT/third_party/$VLLM_ENV/bin/vllm"
MAX_LEN="${MAX_LEN:-$CFG_MAX_LEN}"

# shellcheck disable=SC2086  # EXTRA — список флагов, разбиение на слова нужно
exec "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
    --max-num-seqs 4 --max-num-batched-tokens 1024 \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    --generation-config vllm \
    $EXTRA \
    "$@"
