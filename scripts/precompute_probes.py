"""Предрасчёт проб C1 и перезапросов N2 для корпуса (фаза 0).

    uv run python scripts/precompute_probes.py
    uv run python scripts/precompute_probes.py --window 0   # только сам t*
    uv run python scripts/precompute_probes.py --data-dir data/v1 --baseline scripted --all-steps

Гайдлайн требует отдавать разметчику результат санированного прогона готовым, а
это самая дорогая часть пайплайна. Пробуются не все шаги, а окно вокруг шага
первого отклонения: проверка N1 нужна только там. Если `t*` разметчика окажется
вне окна, проба для его шага считается отдельно — она дешёвая и кэшируется.

Результаты пишутся в data/probes и переиспользуются при повторном запуске.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.envs.gde_judge import PortfolioJudge  # noqa: E402
from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.probe import SCRIPTED_BASELINE, GDEProbe  # noqa: E402
from taxonomy.schema import DEFAULT_THRESHOLDS, Trajectory  # noqa: E402


def steps_to_probe(traj: Trajectory, judge: PortfolioJudge, window: int) -> tuple[int | None, list[int]]:
    """Шаг первого отклонения и окно вокруг него."""
    deviating = [
        s.t for s in traj.steps
        if judge.alignment(s, traj.spec) < DEFAULT_THRESHOLDS.aligned_threshold
    ]
    if not deviating:
        return None, []
    t_star = deviating[0]
    existing = {s.t for s in traj.steps}
    return t_star, sorted(t for t in range(t_star - window, t_star + window + 1) if t in existing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--window", type=int, default=1, help="шагов влево и вправо от t*")
    ap.add_argument("--requery", action="store_true", default=True, help="считать перезапрос на t* (для N2)")
    ap.add_argument("--no-requery", dest="requery", action="store_false")
    ap.add_argument("--baseline", choices=["agent", "scripted"], default="agent",
                    help="опорный агент C1 для траекторий агента по сценарию (эталон V1)")
    ap.add_argument("--all-steps", action="store_true",
                    help="пробы на всех шагах, а не в окне вокруг t*: моделям-разметчикам "
                         "R2 справка нужна для любого выбранного ими квартала")
    args = ap.parse_args()

    agent = tomllib.loads((ROOT / "configs/agent.toml").read_text(encoding="utf-8"))
    if args.baseline == "scripted":
        baseline, cache = SCRIPTED_BASELINE, args.data_dir / "probes_scripted"
    else:
        baseline = {"kind": agent["kind"], "model": agent["model"], "sampling": agent["sampling"]}
        cache = args.data_dir / "probes"
    probe = GDEProbe(base_url=agent["base_url"], raw_dir=args.data_dir / "raw",
                     cache_dir=cache, baseline_policy=baseline)
    judge = PortfolioJudge()
    n_runs = DEFAULT_THRESHOLDS.probe_runs
    threshold = DEFAULT_THRESHOLDS.aligned_threshold

    total_steps = 0
    started = time.monotonic()
    for path in sorted((args.data_dir / "trajectories").glob("*.jsonl")):
        traj = load_trajectories(path)[0]
        if traj.model == "scripted":
            continue
        t_star, steps = steps_to_probe(traj, judge, args.window)
        if args.all_steps:
            steps = [s.t for s in traj.steps]
        elif t_star is None:
            print(f"{traj.traj_id}: отклонений нет, пробы не нужны")
            continue
        for t in steps:
            start = time.monotonic()
            replays = probe.replay(traj, t, n_runs)
            mean_al = sum(judge.alignment_to_goal_only(s, traj.spec) for s in replays) / len(replays)
            verdict = "выполнено" if mean_al >= threshold else "нарушено"
            mark = " <- t*" if t == t_star else ""
            print(f"{traj.traj_id} t={t}{mark}: среднее al {mean_al:.2f} -> C1 {verdict}"
                  f" ({time.monotonic() - start:.0f} с)")
            total_steps += 1
        # Перезапрос — вопрос самому агенту; агента по сценарию спросить нельзя,
        # поэтому для эталона V1 N2 не проверяется (plan.md, фаза 2).
        if args.requery and traj.model != "planted" and t_star is not None:
            probe.requery(traj, t_star)

    print(f"\nпроб посчитано: {total_steps}, всего {(time.monotonic() - started) / 60:.1f} мин")


if __name__ == "__main__":
    main()
