"""Пакетный сбор корпуса с манифестом покрытия классов (фаза 0).

    uv run python scripts/collect_corpus.py --plan configs/corpus.toml
    uv run python scripts/collect_corpus.py --per-scenario 3    # пилот
    uv run python scripts/collect_corpus.py --dry-run           # что будет собрано

Запускается по очереди: два прогона сразу в KV-кэш не помещаются (контекст
одного доходит до 28k при кэше 32k). Уже собранные траектории пропускаются,
поэтому сбор можно прерывать и продолжать.

Манифест (data/manifest.jsonl) — строка на траекторию: сценарий, сид, сколько
шагов, была ли остановка по контексту, есть ли отклонение по оракулу и на каком
шаге. Он же показывает, где сценарий не сработал и класс рискует остаться пустым.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.envs.gde_judge import PortfolioJudge  # noqa: E402
from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.schema import DEFAULT_THRESHOLDS  # noqa: E402


def summarize(data_dir: Path, traj_id: str, judge: PortfolioJudge, min_usable: int) -> dict[str, Any]:
    """Что получилось в траектории: шаги, отклонение, пригодность для разметки."""
    traj = load_trajectories(data_dir / "trajectories" / f"{traj_id}.jsonl")[0]
    deviating = [s.t for s in traj.steps
                 if judge.alignment(s, traj.spec) < DEFAULT_THRESHOLDS.aligned_threshold]
    return {
        "steps": len(traj.steps),
        "t_star": deviating[0] if deviating else None,
        "deviating_steps": len(deviating),
        "usable": len(traj.steps) >= min_usable,
    }


def collect(plan: list[dict[str, Any]], data_dir: Path, per_scenario: int | None, dry_run: bool,
            min_usable: int) -> None:
    judge = PortfolioJudge()
    manifest_path = data_dir / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(line)["traj_id"] for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    started = time.monotonic()
    stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in plan:
        scenario, expected = entry["scenario"], entry.get("expected", [])
        for seed in range(1, (per_scenario or entry["seeds"]) + 1):
            traj_id = f"{scenario}_s{seed}"
            if traj_id in done or (data_dir / "raw" / f"{traj_id}.json.gz").exists():
                if traj_id not in done:  # собрано раньше, но не попало в манифест
                    record = {"traj_id": traj_id, "scenario": scenario, "seed": seed,
                              "expected": expected, "stopped": None,
                              **summarize(data_dir, traj_id, judge, min_usable)}
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats[scenario].append(record)
                continue
            if dry_run:
                print(f"будет собрано: {traj_id}")
                continue

            start = time.monotonic()
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_env.py"),
                 "--scenario", scenario, "--seed", str(seed), "--data-dir", str(data_dir)],
                capture_output=True, text=True, cwd=ROOT,
            )
            if r.returncode != 0:
                print(f"СБОЙ {traj_id}: {r.stderr.strip()[-300:]}")
                continue
            summary = json.loads(r.stdout.strip().splitlines()[-1])
            record = {
                "traj_id": traj_id, "scenario": scenario, "seed": seed, "expected": expected,
                "stopped": (summary["stopped"] or {}).get("reason"),
                "seconds": round(time.monotonic() - start, 1),
                **summarize(data_dir, traj_id, judge, min_usable),
            }
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[scenario].append(record)
            print(f"{traj_id}: шагов {record['steps']}, t*={record['t_star']}, "
                  f"{record['seconds']:.0f} с" + (f", остановка: {record['stopped']}" if record["stopped"] else ""))

    if dry_run:
        return
    print(f"\n{'сценарий':<20} {'собрано':>8} {'ср. шагов':>10} {'с отклонением':>14} {'оборвано':>9} "
          f"{'пригодно':>9}")
    for scenario, records in sorted(stats.items()):
        n = len(records)
        avg = sum(r["steps"] for r in records) / n if n else 0
        with_dev = sum(r["t_star"] is not None for r in records)
        stopped = sum(bool(r["stopped"]) for r in records)
        usable = sum(r.get("usable", True) for r in records)
        print(f"{scenario:<20} {n:>8} {avg:>10.1f} {with_dev:>14} {stopped:>9} {usable:>9}")
    print(f"\nвсего {(time.monotonic() - started) / 60:.0f} мин, манифест: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/corpus.toml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--per-scenario", type=int, help="ограничить число сидов на сценарий (пилот)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with args.plan.open("rb") as f:
        cfg = tomllib.load(f)
    collect(cfg["runs"], args.data_dir, args.per_scenario, args.dry_run, cfg.get("min_usable_steps", 10))


if __name__ == "__main__":
    main()
