#!/usr/bin/env bash
# Отдельное окружение vLLM для локального Qwen (агент и, позже, судья).
#
# Не входит в наш pyproject. Версия закреплена под драйвер NVIDIA 535 (CUDA 12.2):
# vllm 0.11.0 ставит torch 2.8.0 на CUDA 12.8, который работает на драйвере 535
# за счёт совместимости внутри CUDA 12.x. Свежие vLLM тянут более новый torch —
# на этом драйвере их не проверяли.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$ROOT/third_party/vllm-env"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

if [ ! -x "$ENV/bin/python" ]; then
    "$UV" venv --python 3.12 "$ENV"
fi
# transformers ограничен сверху: vllm 0.11.0 требует только >= 4.55.2, а с 5.x
# падает на старте (у токенизатора нет all_special_tokens_extended).
"$UV" pip install --python "$ENV/bin/python" "vllm==0.11.0" "transformers<5"
echo "vLLM готов: $ENV/bin/vllm"
