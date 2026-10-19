"""Сверка классификатора с эталоном V2 — парными траекториями (приёмочный критерий B7.2).

    uv run python scripts/classify_corpus.py --data-dir data/v2      # вердикты
    uv run python scripts/eval_v2.py

Критерий (plan.md): на расходящихся парах t* ∈ [k, k+1] и тип из ожидаемых для
возмущения ≥ 0.8. Пары, разошедшиеся раньше k, в замер не идут: это сбой
детерминизма, а не эффект возмущения.

Отдельной строкой — расходящиеся пары, где агент отклонялся ещё до k и в
контрольном прогоне тоже (t* контроля < k). Метка траектории ставится по
первому отклонению, поэтому в таких парах она описывает не возмущение, а
предшествующий ему дрейф. В основной замер они входят, как требует критерий,
но разбирать их нужно отдельно.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_SHARE = 0.8


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v2")
    ap.add_argument("--verdicts", default="verdicts_agent.jsonl")
    args = ap.parse_args()

    pairs = read_jsonl(args.data_dir / "pairs.jsonl")
    verdicts = {v["traj_id"]: v for v in read_jsonl(args.data_dir / args.verdicts)}

    usable = [p for p in pairs if p["same_before_k"]]
    diverged = [p for p in usable if p["diverged_at"] is not None]
    print(f"пар: {len(pairs)}, совпадают до k: {len(usable)}, разошлись: {len(diverged)}")
    for scenario, n in Counter(p["scenario"] for p in usable).items():
        d = sum(p["scenario"] == scenario and p["diverged_at"] is not None for p in usable)
        print(f"  {scenario}: агент отреагировал в {d}/{n}")

    rows = []
    for p in diverged:
        v, c = verdicts.get(p["traj_id"]), verdicts.get(p["control_id"])
        if v is None:
            print(f"  {p['traj_id']}: нет вердикта")
            continue
        k = p["k"]
        t_ok = v["t_star"] is not None and k <= v["t_star"] <= k + 1
        type_ok = v["label"] in p["expected_labels"]
        early = c is not None and c["t_star"] is not None and c["t_star"] < k
        rows.append((p, v, c, t_ok, type_ok, early))

    if not rows:
        print("нет расходящихся пар с вердиктами")
        return
    n = len(rows)
    t_hits = sum(r[3] for r in rows)
    type_hits = sum(r[4] for r in rows)
    print(f"\nрасходящиеся пары ({n}):")
    print(f"  t* ∈ [k, k+1]: {t_hits}/{n} = {t_hits / n:.2f}")
    print(f"  ожидаемый тип: {type_hits}/{n} = {type_hits / n:.2f}  [критерий ≥ {MIN_SHARE}]")
    clean = [r for r in rows if not r[5]]
    if len(clean) != n:
        ct = sum(r[3] for r in clean)
        cy = sum(r[4] for r in clean)
        print(f"  из них без отклонения до k в контроле ({len(clean)}): t* {ct}/{len(clean)}, тип {cy}/{len(clean)}")
    print("\nпо парам:")
    for p, v, c, t_ok, type_ok, early in rows:
        ctl = f"контроль {c['label']} t*={c['t_star']}" if c else "контроль без вердикта"
        print(f"  {p['traj_id']:<22} k={p['k']} расх.={p['diverged_at']}  {v['label']} t*={v['t_star']}"
              f"  ждали {'/'.join(p['expected_labels'])}  {'' if t_ok else '✗t*'} {'' if type_ok else '✗тип'}"
              f"  | {ctl}{'  (отклонение до k)' if early else ''}")


if __name__ == "__main__":
    main()
