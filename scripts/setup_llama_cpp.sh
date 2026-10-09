#!/usr/bin/env bash
# llama.cpp с бэкендом Vulkan — для LLM-судьи (scripts/serve_judge.sh) и моделей
# панели R2, которые не помещаются в 8 ГБ целиком (часть слоёв в RAM).
#
# Почему сборка из исходников и почему Vulkan: готовые сборки llama.cpp для
# Linux требуют glibc 2.38 (у нас Debian 12, 2.36), а для CUDA-сборки нужен nvcc,
# которого нет. Vulkan работает на GPU через драйвер NVIDIA (nvidia-vulkan-icd).
# Нужны пакеты: sudo apt install libvulkan-dev glslc (плюс cmake, g++, git).
# SPIRV-Headers в Debian 12 без CMake-конфига, который ищет llama.cpp, — они
# ставятся из исходников в third_party (только заголовки, sudo не нужен). Путь к
# ним передаётся и флагом компилятора: CMake-скрипт ggml-vulkan находит пакет, но
# не добавляет его include-каталог к цели (b11429). Заголовки Vulkan в Debian 12
# (1.3.239) тоже старые — нет LayerSettingEXT; свежие Vulkan-Headers ставятся так
# же, из исходников. Системный загрузчик libvulkan остаётся прежним.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama-cpp"
TAG="${LLAMA_TAG:-b11429}"

SPIRV="$ROOT/third_party/spirv-headers"
if [ ! -f "$SPIRV/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]; then
    git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git "$SPIRV-src"
    cmake -S "$SPIRV-src" -B "$SPIRV-src/build" -DCMAKE_INSTALL_PREFIX="$SPIRV" -DSPIRV_HEADERS_ENABLE_TESTS=OFF
    cmake --install "$SPIRV-src/build"
fi

VKH="$ROOT/third_party/vulkan-headers"
if [ ! -f "$VKH/include/vulkan/vulkan.hpp" ]; then
    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git "$VKH"
fi

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
cmake -S "$SRC" -B "$SRC/build" -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$SPIRV" -DVulkan_INCLUDE_DIR="$VKH/include" \
    -DCMAKE_CXX_FLAGS="-I$VKH/include -I$SPIRV/include"
cmake --build "$SRC/build" --target llama-server -j "$(nproc)"
echo "llama-server готов: $SRC/build/bin/llama-server"
