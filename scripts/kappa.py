"""Согласие разметчиков: κ и матрица несогласия.

Главный аналитический артефакт фазы 2. Матрица несогласия важнее самого κ:
пара типов, устойчиво попадающая в её топ, — кандидат на слияние.

Использование:
    uv run python scripts/kappa.py data/annotations/

Ожидается, что в каталоге лежат ann_<annotator>.jsonl с полями схемы Annotation.

Разметка двухуровневая: `label` (по первому отклонению) и `dominant_label` (по
шагу наибольшего отклонения от цели). Согласие считается по каждой метке
отдельно: смешивание двух разных вопросов в один κ ничего не измеряет. Пустая
`dominant_label` означает «явление одно» и приравнивается к `label`. Таблица
переходов «метка → преобладающая» показывает, как часто первое отклонение
прячет главное явление.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taxonomy.schema import DRIFT_LABELS, Label  # noqa: E402


def load(path: Path, field: str = "label") -> dict[str, str]:
    """traj_id -> метка из поля `field`. Пропущенные (skipped) не учитываются.

    Для `dominant_label` пустое значение заменяется основной меткой.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        out[rec["traj_id"]] = rec.get(field) or rec["label"]
    return out


def transitions(first: dict[str, str], dominant: dict[str, str], name: str) -> None:
    """Таблица «метка → преобладающая» одного разметчика."""
    moved = Counter((first[t], dominant[t]) for t in first if dominant[t] != first[t])
    n_moved = sum(moved.values())
    print(f"{name}: преобладающая отличается от основной в {n_moved} из {len(first)}")
    for (x, y), n in moved.most_common():
        print(f"  {x} -> {y}: {n}")


def to_binary(label: str) -> str:
    return "drift" if Label(label) in DRIFT_LABELS else "no_drift"


def report(a: dict[str, str], b: dict[str, str], name_a: str, name_b: str) -> None:
    common = sorted(set(a) & set(b))
    if not common:
        print("нет пересечения по траекториям")
        return

    ya = [a[t] for t in common]
    yb = [b[t] for t in common]

    kappa_typed = cohen_kappa_score(ya, yb)
    kappa_binary = cohen_kappa_score([to_binary(x) for x in ya], [to_binary(x) for x in yb])

    print(f"разметчики: {name_a} vs {name_b}")
    print(f"общих траекторий: {len(common)}")
    print(f"κ бинарно (дрейф / не дрейф): {kappa_binary:.3f}   [цель ≥ 0.70]")
    print(f"κ по типам:                   {kappa_typed:.3f}   [цель ≥ 0.60]")
    print()

    # κ по редким классам считается отдельно: усреднённый их маскирует.
    print("one-vs-rest κ по классам (редкие классы смотреть отдельно):")
    for lab in sorted(set(ya) | set(yb)):
        ka = [x == lab for x in ya]
        kb = [x == lab for x in yb]
        n = sum(ka) + sum(kb)
        if n == 0:
            continue
        k = cohen_kappa_score(ka, kb)
        print(f"  {lab:>4}  n={n:<4} κ={k:.3f}")
    print()

    labels = sorted(set(ya) | set(yb))
    cm = confusion_matrix(ya, yb, labels=labels)
    print("матрица несогласия (строки — %s, столбцы — %s):" % (name_a, name_b))
    header = "      " + " ".join(f"{l:>4}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        print(f"{l:>5} " + " ".join(f"{cm[i][j]:>4}" for j in range(len(labels))))
    print()

    pairs: Counter[tuple[str, str]] = Counter()
    for x, y in zip(ya, yb):
        if x != y:
            pairs[tuple(sorted((x, y)))] += 1
    print("топ проблемных пар (кандидаты на слияние типов):")
    for (x, y), n in pairs.most_common(5):
        print(f"  {x} / {y}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()

    files = sorted(args.directory.glob("ann_*.jsonl"))
    if len(files) < 2:
        raise SystemExit(f"нужно минимум два файла ann_*.jsonl, найдено {len(files)}")

    names = [f.stem.removeprefix("ann_") for f in files]
    first = [load(f) for f in files]
    dominant = [load(f, "dominant_label") for f in files]

    for title, labels in (("основная метка (первое отклонение)", first),
                          ("преобладающая метка (наибольшее отклонение от цели)", dominant)):
        print(f"=== {title} ===")
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                report(labels[i], labels[j], names[i], names[j])
                print("-" * 60)

    print("=== переходы «метка → преобладающая» ===")
    for name, a, b in zip(names, first, dominant):
        transitions(a, b, name)


if __name__ == "__main__":
    main()
