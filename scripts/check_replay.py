"""Гейт воспроизводимости реплея (docs/plan.md, фаза 0).

Детерминированная часть — для каждого сырого лога GDE в каталоге:
  (a) выходы инструментов и снапшот конца шага совпадают с логом;
  (b) наблюдение шага заново порождается из снапшота.
Проверка выполняется в окружении GDE (scripts/gde/gde_runner.py check).
(c) — санированный контекст — покрыт tests/test_gde_integration.py.

С --trajectories — ещё (d): на случайных шагах траекторий N санированных
прогонов того же агента проходят без ошибок, действия парсятся; печатается
время и результат C1 по программному оракулу (для портфельной среды).

Использование:
    uv run python scripts/check_replay.py data/raw/
    uv run python scripts/check_replay.py data/raw/ --trajectories data/trajectories/ --probe-steps 2
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.envs.gde_judge import PortfolioJudge  # noqa: E402
from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.probe import GDEProbe  # noqa: E402
from taxonomy.schema import DEFAULT_THRESHOLDS  # noqa: E402


def check_one(path: Path) -> dict:
    r = subprocess.run(
        [str(ROOT / "scripts/gde/run.sh"), str(ROOT / "scripts/gde/gde_runner.py"), "check", str(path)],
        capture_output=True,
        text=True,
    )
    if r.returncode not in (0, 1):
        raise SystemExit(f"{path}: проверка упала\n{r.stderr}")
    return json.loads(r.stdout)


def deterministic(directory: Path) -> int:
    files = sorted(directory.glob("*.json.gz"))
    if not files:
        raise SystemExit(f"в {directory} нет сырых логов *.json.gz")
    failed = 0
    for path in files:
        rep = check_one(path)
        status = "OK " if not rep["mismatches"] else "FAIL"
        failed += bool(rep["mismatches"])
        print(
            f"{status} {path.name}: шагов {rep['steps']}, вызовов инструментов {rep['tool_calls']}, "
            f"частей давления {rep['pressure_parts']}, расхождений {len(rep['mismatches'])}"
        )
        for m in rep["mismatches"][:5]:
            print(f"     t={m['t']}: {m['what']}" + (f" ({m['tool']})" if "tool" in m else ""))
    print(f"(a)+(b): траекторий {len(files)}, не прошли: {failed}\n")
    return failed


def llm_replays(directory: Path, probe_steps: int, seed: int) -> int:
    thr = DEFAULT_THRESHOLDS
    agent = tomllib.loads((ROOT / "configs/agent.toml").read_text(encoding="utf-8"))
    probe, judge, rng = GDEProbe(base_url=agent["base_url"]), PortfolioJudge(), random.Random(seed)
    failed = 0
    for path in sorted(directory.glob("*.jsonl")):
        for traj in load_trajectories(path):
            if traj.model == "scripted":
                continue
            # C1 проверяется на шаге отклонения, поэтому пробуем именно их:
            # на целесогласованном шаге проверка N1 до пробника не доходит.
            deviating = [s for s in traj.steps
                         if judge.alignment(s, traj.spec) < thr.aligned_threshold]
            candidates = deviating or traj.steps
            for step in rng.sample(candidates, min(probe_steps, len(candidates))):
                start = time.monotonic()
                try:
                    replays = probe.replay(traj, step.t, thr.probe_runs)
                except RuntimeError as e:
                    failed += 1
                    print(f"FAIL {traj.traj_id} t={step.t}: {str(e)[-300:]}")
                    continue
                seconds = time.monotonic() - start
                broken = [a for s in replays for a in s.actions if a.get("args") is None]
                ok = len(replays) == thr.probe_runs and not broken
                failed += not ok
                # C1: среднее al санированных прогонов без учёта C (taxonomy.md, §2).
                al_agent = judge.alignment_to_goal_only(step, traj.spec)
                al_replays = [judge.alignment_to_goal_only(s, traj.spec) for s in replays]
                mean_al = sum(al_replays) / len(al_replays)
                print(
                    f"{'OK ' if ok else 'FAIL'} {traj.traj_id} t={step.t}: {seconds:.0f} с, "
                    f"невалидных действий {len(broken)}; al агента {al_agent:.2f}, "
                    f"al реплеев {[round(a, 2) for a in al_replays]}, среднее {mean_al:.2f} -> C1 "
                    f"{'выполнено' if mean_al >= thr.aligned_threshold else 'нарушено'}"
                )
    print(f"(d): не прошли {failed}")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--trajectories", type=Path, help="каталог траекторий для пункта (d)")
    ap.add_argument("--probe-steps", type=int, default=2, help="шагов на траекторию для (d)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    failed = deterministic(args.directory)
    if args.trajectories:
        failed += llm_replays(args.trajectories, args.probe_steps, args.seed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
