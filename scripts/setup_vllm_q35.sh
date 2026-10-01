#!/usr/bin/env bash
# Окружение vLLM для агента Qwen3.5-4B (configs/agent.toml, решение 2026-10-01).
#
# Отдельно от third_party/vllm-env (vLLM 0.11 для Qwen2.5-7B): Qwen3.5 требует
# vLLM ≥ 0.17. Колесо vLLM с PyPI собрано под CUDA 13, которую драйвер 535 не
# поддерживает (ImportError: libcudart.so.13), поэтому ставится колесо +cu129 из
# релиза на GitHub, а torch — с индекса cu129. torch 2.13+cu129 на драйвере 535
# работает (проверено 2026-09-29).
#
# Сэмплер flashinfer при старте компилирует ядра и требует nvcc, которого нет
# (в pip-пакете cu12 только ptxas), — он выключается в [serve].env конфига агента.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$ROOT/third_party/vllm-env-q35"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
VLLM_VERSION=0.30.0
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

if [ ! -x "$ENV/bin/python" ]; then
    "$UV" venv --python 3.12 "$ENV"
fi
"$UV" pip install --python "$ENV/bin/python" --torch-backend=cu129 "vllm @ $WHEEL"
"$ENV/bin/python" -c "import vllm, torch; print('vLLM', vllm.__version__, 'torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
echo "vLLM готов: $ENV/bin/vllm"
