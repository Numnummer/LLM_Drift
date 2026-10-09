"""Эталон V1: агенты по сценарию с заложенным поведением (plan.md, фаза 2).

    uv run python scripts/build_v1.py                 # все конфиги × 3 сида
    uv run python scripts/build_v1.py --seeds 5       # больше вариантов

Прогоняет configs/planted/*.toml через run_env.py --planted (без модели, секунды
на траекторию), затем сверяет t* оракула с заложенным. Это проверка самих
эталонов: если t* оракула не совпадает с заложенным, агент по сценарию повёл
себя не так, как задумано, и траектория не годится как эталон.

Манифест data/v1/expected.jsonl — ключ эталона: что заложено (intended), какую
метку и t* должна дать процедура по текущему определению (expected_label,
expected_t_star) и что нашёл оракул (oracle_t_star). Метку проверяет уже
классификатор (фаза 3) и модели-разметчики (квалификация R2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.envs.gde_judge import PortfolioJudge  # noqa: E402
from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.schema import DEFAULT_THRESHOLDS  # noqa: E402


def oracle_t_star(path: Path, judge: PortfolioJudge) -> int | None:
    traj = load_trajectories(path)[0]
    theta = DEFAULT_THRESHOLDS.aligned_threshold
    return next((s.t for s in traj.steps if judge.alignment(s, traj.spec) < theta), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v1")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    judge = PortfolioJudge()
    rows = []
    for cfg in sorted((ROOT / "configs/planted").glob("*.toml")):
        name = cfg.stem
        for seed in range(1, args.seeds + 1):
            traj_id = f"v1_{name}_s{seed}"
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_env.py"), "--planted", name,
                 "--seed", str(seed), "--data-dir", str(args.data_dir)],
                capture_output=True, text=True, cwd=ROOT,
            )
            if r.returncode != 0:
                print(f"СБОЙ {traj_id}: {r.stderr.strip()[-300:]}")
                continue
            out = json.loads(r.stdout.strip().splitlines()[-1])
            exp = out["expected"]
            found = oracle_t_star(args.data_dir / "trajectories" / f"{traj_id}.jsonl", judge)
            rows.append({"traj_id": traj_id, "planted": name, "seed": seed, "steps": out["steps"],
                         **exp, "oracle_t_star": found,
                         "t_star_ok": found == exp["expected_t_star"]})

    (args.data_dir / "expected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    print(f"{'траектория':<22} {'заложено':>8} {'ждём':>5} {'t* ждём':>8} {'t* оракул':>10}")
    for r in rows:
        mark = "" if r["t_star_ok"] else "  <- не совпало"
        print(f"{r['traj_id']:<22} {r['intended']:>8} {r['expected_label']:>5} "
              f"{str(r['expected_t_star']):>8} {str(r['oracle_t_star']):>10}{mark}")
    ok = sum(r["t_star_ok"] for r in rows)
    print(f"\nt* оракула совпал с заложенным: {ok} из {len(rows)}; манифест: {args.data_dir / 'expected.jsonl'}")


if __name__ == "__main__":
    main()
