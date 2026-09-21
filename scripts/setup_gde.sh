#!/usr/bin/env bash
# Установка эталонной среды goal-drift-evals (Arike et al., MIT) в third_party/.
#
# Клон не коммитится (.gitignore). Воспроизводимость держится на зафиксированном
# коммите third_party/GDE_COMMIT и нашем патче third_party/gde.patch (если есть).
# У GDE своё uv-окружение (Python >= 3.12): его зависимости (openai, anthropic, ...)
# не попадают в наш pyproject. Наши скрипты для GDE лежат в scripts/gde/ и
# запускаются через scripts/gde/run.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GDE="$ROOT/third_party/goal-drift-evals"
COMMIT="$(cat "$ROOT/third_party/GDE_COMMIT")"
PATCH="$ROOT/third_party/gde.patch"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

if [ ! -d "$GDE/.git" ]; then
    git clone https://github.com/RaunoArike/goal-drift-evals.git "$GDE"
fi
git -C "$GDE" checkout --quiet "$COMMIT"

# Код GDE импортирует data.*, а каталог в опубликованном репозитории
# называется universe_data/.
ln -sfn universe_data "$GDE/data"

if [ -f "$PATCH" ]; then
    if git -C "$GDE" apply --check "$PATCH" 2>/dev/null; then
        git -C "$GDE" apply "$PATCH"
    elif git -C "$GDE" apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "патч уже применён"
    else
        echo "патч $PATCH не применяется к коммиту $COMMIT" >&2
        exit 1
    fi
fi

# 3.12, а не новейший: у закреплённого в uv.lock tiktoken нет колёс под 3.14,
# и без Rust он не собирается.
(cd "$GDE" && "$UV" sync --python 3.12)
echo "GDE готов: $GDE @ $COMMIT"
