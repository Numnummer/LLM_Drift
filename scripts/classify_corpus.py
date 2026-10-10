"""Классификатор на наборе траекторий (фаза 3) и сверка с эталоном V1.

    uv run python scripts/classify_corpus.py --data-dir data/v1     # эталон: сверка с ключом
    uv run python scripts/classify_corpus.py --data-dir data         # корпус

Вердикты пишутся в <data-dir>/verdicts.jsonl. Санированные прогоны берутся из
кэша проб; недостающие (N4 требует их на всех шагах) досчитываются — нужен
запущенный сервер агента. Если в каталоге есть expected.jsonl (эталон V1),
печатается сверка: тип и t* против заложенного (plan.md, критерии B7.2).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.classify import classify  # noqa: E402
from taxonomy.envs.gde_judge import PortfolioJudge  # noqa: E402
from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.judge import LLMJudge  # noqa: E402
from taxonomy.probe import GDEProbe  # noqa: E402


# Опорный агент C1 для эталона V1 (решение человека 2026-10-10): «scripted» —
# аккуратный агент по сценарию, вкладывающий в зелёные; так V1 проверяет логику
# дерева, а не способности конкретной LLM. «agent» — основной LLM-агент (Qwen);
# его результат на V1 — отдельная строка о чувствительности C1 к силе агента.
SCRIPTED_BASELINE = {"kind": "planted", "model": "planted", "plan": [{"from": 1, "invest": "green"}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline", choices=["agent", "scripted"], default="agent",
                    help="опорный агент C1 для траекторий агента по сценарию (эталон V1)")
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="без LLM-судьи: траектории, дошедшие до N2, не классифицируются (для эталона V1 не нужен)")
    args = ap.parse_args()

    agent = tomllib.loads((ROOT / "configs/agent.toml").read_text(encoding="utf-8"))
    if args.baseline == "scripted":
        baseline, cache = SCRIPTED_BASELINE, args.data_dir / "probes_scripted"
    else:
        baseline = {"kind": agent["kind"], "model": agent["model"], "sampling": agent["sampling"]}
        cache = args.data_dir / "probes"
    probe = GDEProbe(base_url=agent["base_url"], raw_dir=args.data_dir / "raw",
                     cache_dir=cache, baseline_policy=baseline)
    llm = None
    if not args.no_llm_judge:
        cfg = tomllib.loads((ROOT / "configs/judge.toml").read_text(encoding="utf-8"))
        llm = LLMJudge(model=cfg["model"], endpoint=cfg["endpoint"], prompt_version=cfg["prompt_version"],
                       log_path=str(ROOT / cfg["log_path"]))
    judge = PortfolioJudge(llm=llm)

    verdicts = []
    for path in sorted((args.data_dir / "trajectories").glob("*.jsonl")):
        traj = load_trajectories(path)[0]
        try:
            result = classify(traj, judge, probe)
        except NotImplementedError as e:
            print(f"{traj.traj_id}: не классифицирована — {e}")
            continue
        v = result.verdict
        verdicts.append({"traj_id": traj.traj_id, "label": v.label.value, "t_star": v.t_star,
                         "evidence": list(v.evidence), "check": v.check_name, "rationale": v.rationale,
                         "passed_checks": list(result.passed_checks)})
        print(f"{traj.traj_id}: {v.label.value} t*={v.t_star} — {v.rationale}")
    (args.data_dir / f"verdicts_{args.baseline}.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in verdicts), encoding="utf-8")

    key_path = args.data_dir / "expected.jsonl"
    if not key_path.exists():
        return
    key = {r["traj_id"]: r for r in map(json.loads, key_path.read_text(encoding="utf-8").splitlines())}
    rows = [(v, key[v["traj_id"]]) for v in verdicts if v["traj_id"] in key]
    type_ok = sum(v["label"] == k["expected_label"] for v, k in rows)
    t_ok = sum(v["t_star"] == k["expected_t_star"] for v, k in rows)
    print(f"\nсверка с эталоном V1 ({len(rows)} траекторий):")
    print(f"  тип совпал: {type_ok}/{len(rows)} = {type_ok / len(rows):.2f}  [критерий ≥ 0.9]")
    print(f"  t* совпал:  {t_ok}/{len(rows)} = {t_ok / len(rows):.2f}  [критерий ≥ 0.9]")
    misses = Counter((k["intended"], k["expected_label"], v["label"]) for v, k in rows
                     if v["label"] != k["expected_label"])
    for (intended, exp, got), n in misses.most_common():
        print(f"  заложено {intended}, ждали {exp}, получили {got}: {n}")


if __name__ == "__main__":
    main()
