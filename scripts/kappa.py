"""Согласие разметчиков: κ и матрица несогласия.

Главный аналитический артефакт фазы 2. Матрица несогласия важнее самого κ:
пара типов, устойчиво попадающая в её топ, — кандидат на слияние.

Использование:
    uv run python scripts/kappa.py data/annotations/

Ожидается, что в каталоге лежат ann_<annotator>.jsonl с полями схемы Annotation.
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


def load(path: Path) -> dict[str, str]:
    """traj_id -> метка. Пропущенные (skipped) траектории не учитываются."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        out[rec["traj_id"]] = rec["label"]
    return out


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

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            report(
                load(files[i]),
                load(files[j]),
                files[i].stem.removeprefix("ann_"),
                files[j].stem.removeprefix("ann_"),
            )
            print("-" * 60)


if __name__ == "__main__":
    main()
