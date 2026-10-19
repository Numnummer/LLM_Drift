#!/usr/bin/env bash
# llama.cpp с бэкендом CUDA — для моделей панели R2 и LLM-судьи.
#
# Почему не только Vulkan (scripts/setup_llama_cpp.sh): на RTX 4060 через Vulkan
# prompt обрабатывается ~200 ток/с, а KV-кэш q8_0 роняет генерацию до 0,6 ток/с.
# Входы панели — 25–35 тыс. токенов, разметка V1 одной моделью заняла бы сутки.
#
# nvcc системно не установлен, pip-пакет nvidia-cuda-nvcc содержит только ptxas.
# Тулкит ставится из conda-forge через micromamba в third_party (без sudo).
# Версия 12.2 — ровно та, что у драйвера 535: собранный тулкитом 12.9 код драйвер
# не загружает («device kernel image is invalid»), минорная совместимость CUDA 12
# здесь не работает. Код собирается под sm_89 (RTX 4060).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
SRC="$TP/llama-cpp"
TK="$TP/cuda-tk"
[ -d "$SRC/.git" ] || { echo "сначала scripts/setup_llama_cpp.sh (клон llama.cpp)" >&2; exit 1; }

if [ ! -x "$TP/micromamba/bin/micromamba" ]; then
    mkdir -p "$TP/micromamba"
    curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$TP/micromamba" bin/micromamba
fi
if [ ! -x "$TK/bin/nvcc" ]; then
    MAMBA_ROOT_PREFIX="$TP/micromamba/root" "$TP/micromamba/bin/micromamba" create -y -q -p "$TK" -c conda-forge \
        "cuda-nvcc=12.2" "cuda-cudart-dev=12.2" "libcublas-dev=12.2" "cuda-cccl=12.2" "cuda-version=12.2"
fi

export PATH="$TK/bin:$PATH"
cmake -S "$SRC" -B "$SRC/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER="$TK/bin/nvcc" -DCUDAToolkit_ROOT="$TK" \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build-cuda" --target llama-server -j "${JOBS:-12}"
echo "llama-server (CUDA) готов: $SRC/build-cuda/bin/llama-server"
