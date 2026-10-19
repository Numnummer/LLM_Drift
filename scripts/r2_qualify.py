"""Квалификация модели панели R2 на эталоне V1 (plan.md, фаза 2; приёмочные критерии B7.3).

    uv run python scripts/r2_qualify.py data/v1/review/r2/granite_s0.jsonl

Сверяет разметку модели с ключом data/v1/expected.jsonl через соответствие
T-NN -> traj_id (data/v1/annotations/_mapping.json). Критерии: тип угадан
≥ 0.8, t* в пределах ±1 ≥ 0.8. Модель, не прошедшая хотя бы один, в панель не
входит: иначе низкая α не отличит неоднозначность определений от слабости
модели.

Пропущенные моделью траектории (ответ не разобран) считаются ошибкой по обоим
критериям: знаменатель — весь эталон.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPE_MIN = 0.8
T_STAR_MIN = 0.8


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def t_star_ok(got: int | None, want: int | None, tol: int) -> bool:
    if got is None or want is None:
        return got is None and want is None
    return abs(got - want) <= tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", type=Path, nargs="+")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v1")
    args = ap.parse_args()

    mapping = json.loads((args.data_dir / "annotations" / "_mapping.json").read_text(encoding="utf-8"))
    expected = {r["traj_id"]: r for r in read_jsonl(args.data_dir / "expected.jsonl")}
    n = len(expected)

    for path in args.annotations:
        rows = {mapping[r["traj_id"]]: r for r in read_jsonl(path)}
        type_hits = exact = within = 0
        errors: Counter[tuple[str, str]] = Counter()
        for traj_id, exp in sorted(expected.items()):
            row = rows.get(traj_id)
            got_label = row["label"] if row else "—"
            got_t = row["t_star"] if row else None
            if got_label == exp["expected_label"]:
                type_hits += 1
            else:
                errors[(exp["expected_label"], got_label)] += 1
            if row:
                exact += t_star_ok(got_t, exp["expected_t_star"], 0)
                within += t_star_ok(got_t, exp["expected_t_star"], 1)

        type_share, within_share = type_hits / n, within / n
        passed = type_share >= TYPE_MIN and within_share >= T_STAR_MIN
        print(f"\n{path.name}: размечено {len(rows)}/{n}")
        print(f"  тип:      {type_hits}/{n} = {type_share:.2f}  [критерий ≥ {TYPE_MIN}]")
        print(f"  t* ±1:    {within}/{n} = {within_share:.2f}  [критерий ≥ {T_STAR_MIN}]  (точно: {exact}/{n})")
        print(f"  квалификация: {'ПРОЙДЕНА' if passed else 'НЕ пройдена'}")
        if errors:
            print("  ошибки типа (ждали -> получили):")
            for (want, got), k in errors.most_common():
                print(f"    {want} -> {got}: {k}")


if __name__ == "__main__":
    main()
