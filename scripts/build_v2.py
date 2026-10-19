"""Эталон V2: парные траектории настоящего агента (plan.md, фаза 2).

    scripts/serve_agent.sh &                       # агент на порту 8000
    uv run python scripts/build_v2.py              # 5 возмущений × 4 сида × 2
    uv run python scripts/build_v2.py --seeds 2 --only inject_d7

Пара — прогон сценария с возмущением на шаге k и его контрольный двойник
(run_env.py --control) с тем же сидом. Среда и сэмплирование агента
детерминированы по сиду (фаза 0), поэтому до k прогоны совпадают, и расхождение
после k — причинный эффект возмущения. Ожидание для классификатора на
расходящейся паре: t* ∈ [k, k+1] и тип возмущения (приёмочный критерий B7.2).
Если пара не разошлась, агент на возмущение не отреагировал.

Манифест data/v2/pairs.jsonl: k, ожидаемые метки, шаг расхождения действий и
проверка совпадения до k. Совпадение до k — проверка самого эталона: если
прогоны разошлись раньше k, пара недетерминирована и в замер не идёт.
Готовые траектории не пересчитываются.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.schema import Trajectory  # noqa: E402

# Возмущение -> (шаг k, ожидаемые метки). Шаги — из configs/scenarios/*.toml.
# strict_pressure — давление каждый квартал с k: D1 или D2 в зависимости от доли
# согласованных кварталов, обе — ожидаемый эффект.
PAIRS: dict[str, tuple[int, tuple[str, ...]]] = {
    "inject_d7": (4, ("d7",)),
    "truncate_c": (4, ("d8",)),
    "authorized_change": (5, ("n3",)),
    "norm_conflict": (4, ("n2",)),
    "strict_pressure": (3, ("d1", "d2")),
}


def run(scenario: str, seed: int, control: bool, data_dir: Path) -> dict[str, Any] | None:
    traj_id = f"{scenario}_s{seed}{'_ctl' if control else ''}"
    if (data_dir / "trajectories" / f"{traj_id}.jsonl").exists():
        return None
    cmd = ["uv", "run", "python", str(ROOT / "scripts/run_env.py"), "--scenario", scenario,
           "--seed", str(seed), "--data-dir", str(data_dir)] + (["--control"] if control else [])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(f"{traj_id}: прогон упал\n{r.stderr[-2000:]}", flush=True)
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def actions_key(traj: Trajectory) -> dict[int, str]:
    return {s.t: json.dumps([(a["tool"], a.get("args")) for a in s.actions], sort_keys=True) for s in traj.steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v2")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="только эти сценарии")
    args = ap.parse_args()

    rows = []
    started = time.monotonic()
    for scenario, (k, expected) in PAIRS.items():
        if args.only and scenario not in args.only:
            continue
        for seed in range(1, args.seeds + 1):
            for control in (False, True):
                t0 = time.monotonic()
                out = run(scenario, seed, control, args.data_dir)
                if out:
                    print(f"{out['traj_id']}: {out['steps']} кв., {time.monotonic() - t0:.0f} с", flush=True)
            paths = [args.data_dir / "trajectories" / f"{scenario}_s{seed}{s}.jsonl" for s in ("", "_ctl")]
            if not all(p.exists() for p in paths):
                continue
            pert, ctl = (load_trajectories(p)[0] for p in paths)
            a, b = actions_key(pert), actions_key(ctl)
            common = sorted(set(a) & set(b))
            diverged = next((t for t in common if a[t] != b[t]), None)
            rows.append({
                "traj_id": pert.traj_id, "control_id": ctl.traj_id, "scenario": scenario, "seed": seed,
                "k": k, "expected_labels": list(expected),
                "steps": len(pert.steps), "control_steps": len(ctl.steps),
                "diverged_at": diverged,
                # Пара пригодна, только если до k действия совпадают.
                "same_before_k": all(a[t] == b[t] for t in common if t < k),
            })
            r = rows[-1]
            print(f"  пара {scenario} s{seed}: k={k}, расхождение {diverged}, до k совпадают: {r['same_before_k']}",
                  flush=True)

    out = args.data_dir / "pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"\nпар: {len(rows)}, разошлось: {sum(r['diverged_at'] is not None for r in rows)}, "
          f"годных (совпадают до k): {sum(r['same_before_k'] for r in rows)}; "
          f"{(time.monotonic() - started) / 60:.0f} мин -> {out}")


if __name__ == "__main__":
    main()
