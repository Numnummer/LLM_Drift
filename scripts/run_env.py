"""Прогон сценария портфельной среды и запись траектории (фаза 0).

    uv run python scripts/run_env.py --scenario elicit_pressure --seed 3
    uv run python scripts/run_env.py --scenario elicit_pressure --seed 3 --scripted  # без модели

Сценарий — configs/scenarios/<name>.toml, спецификация G0 — configs/spec/<spec>.toml,
агент — configs/agent.toml. Сырой лог пишется в data/raw/<traj_id>.json.gz,
траектория в схеме Step — в data/trajectories/<traj_id>.jsonl. Сид задаёт и мир
среды (номер run в GDE), и сэмплирование агента.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.context_audit import INJECTION, NEUTRAL, PRINCIPAL_AUTH, UNTRUSTED_SOURCES  # noqa: E402
from taxonomy.envs.gde import load_raw, to_trajectory  # noqa: E402
from taxonomy.io import save_trajectories  # noqa: E402
from taxonomy.schema import Constraint, GoalSpec  # noqa: E402

GDE_FIELDS = {
    "source", "target", "distractions", "num_instrumental_steps",
    "use_soft_system_prompt", "truncate_constraints_at", "extra_parts",
}


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def perturbations(sc: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Возмущения сценария -> части наблюдения и добавки к системному промпту.

    Текст выбирается из пула по сиду: двадцать прогонов одного сценария не
    должны повторять одну и ту же инъекцию.
    """
    p = sc.get("perturbations", {})
    parts: list[dict[str, Any]] = []
    prompt_extra: list[str] = []

    def pick(pool: str, key: str = "texts") -> str:
        texts = load_toml(ROOT / "configs/payloads" / f"{pool}.toml")[key]
        return texts[seed % len(texts)]

    if "injection_at" in p:  # D7
        parts.append({"t": p["injection_at"], "source": INJECTION, "text": pick("injections")})
    if "goal_change_at" in p:  # N3
        payload = load_toml(ROOT / "configs/payloads/authorized_change.toml")
        parts.append({
            "t": p["goal_change_at"], "source": PRINCIPAL_AUTH, "goal_change": True,
            "text": payload["messages"][seed % len(payload["messages"])],
        })
        prompt_extra.append(payload["prompt_line"])
    if "norm_conflict_at" in p:  # N2
        parts.append({"t": p["norm_conflict_at"], "source": NEUTRAL, "text": pick("norm_conflict")})
    return parts, prompt_extra


def build(scenario_name: str, seed: int = 0) -> tuple[dict[str, Any], GoalSpec, int]:
    """Сценарий для раннера GDE, спецификация для конвертера, число шагов."""
    sc = load_toml(ROOT / "configs/scenarios" / f"{scenario_name}.toml")
    spec_cfg = load_toml(ROOT / "configs/spec" / f"{sc['spec']}.toml")
    unknown = set(sc.get("gde", {})) - GDE_FIELDS
    if unknown:
        raise SystemExit(f"{scenario_name}: неизвестные поля [gde]: {sorted(unknown)}")
    with_constraints = sc.get("with_constraints", True)
    constraints = spec_cfg.get("constraints", []) if with_constraints else []
    extra_parts, prompt_extra = perturbations(sc, seed)
    gde = {
        **sc.get("gde", {}),
        "constraints": [c["text"] for c in constraints],
        "extra_parts": extra_parts,
        "prompt_extra": prompt_extra,
    }
    extra = sc.get("spec_extra", {})
    spec = GoalSpec(
        goal=spec_cfg["goal"],
        constraints=tuple(Constraint(id=c["id"], description=c["text"], predicate=c.get("predicate")) for c in constraints),
        priority=tuple(p for p in spec_cfg.get("priority", []) if p == "goal" or with_constraints),
        instrumental_goal=extra.get("instrumental_goal"),
        instrumental_done_predicate=extra.get("instrumental_done_predicate"),
    )
    return gde, spec, sc["num_steps"]


def policy_info(scripted: bool, agent_path: Path) -> tuple[dict[str, Any], str | None]:
    if scripted:
        return {"kind": "scripted", "model": "scripted"}, None
    agent = load_toml(agent_path)
    return {"kind": agent["kind"], "model": agent["model"], "sampling": agent["sampling"]}, agent["base_url"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scripted", action="store_true", help="скриптовый агент без модели (проверка обвязки)")
    ap.add_argument("--num-steps", type=int, help="переопределить num_steps сценария")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--agent", type=Path, default=ROOT / "configs/agent.toml",
                    help="конфиг агента; второй агент — свой файл и свой суффикс traj_id")
    args = ap.parse_args()

    gde, spec, num_steps = build(args.scenario, args.seed)
    info, base_url = policy_info(args.scripted, args.agent)
    # Суффикс агента в traj_id: корпус привязан к модели, и траектории разных
    # агентов не должны перемешиваться в одном пуле (метки N1 и D5 модельные).
    stem = args.agent.stem.removeprefix("agent_")
    suffix = "_scripted" if args.scripted else ("" if stem == "agent" else f"_{stem}")
    traj_id = f"{args.scenario}_s{args.seed}{suffix}"
    raw_path = args.data_dir / "raw" / f"{traj_id}.json.gz"

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "scenario.json").write_text(json.dumps(gde), encoding="utf-8")
        (Path(tmp) / "policy.json").write_text(json.dumps(info), encoding="utf-8")
        cmd = [str(ROOT / "scripts/gde/run.sh"), str(ROOT / "scripts/gde/gde_runner.py")]
        if base_url:
            cmd += ["--base-url", base_url]
        cmd += [
            "run", "--scenario-json", str(Path(tmp) / "scenario.json"),
            "--policy-json", str(Path(tmp) / "policy.json"),
            "--num-steps", str(args.num_steps or num_steps),
            "--run", str(args.seed), "--seed", str(args.seed), "--out", str(raw_path),
        ]
        start = time.monotonic()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"прогон упал:\n{r.stderr.strip()[-3000:]}")
        elapsed = time.monotonic() - start

    raw = load_raw(raw_path)
    traj = to_trajectory(raw, traj_id=traj_id, spec=spec, untrusted_sources=UNTRUSTED_SOURCES, scenario=args.scenario)
    traj_path = args.data_dir / "trajectories" / f"{traj_id}.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories(traj_path, [traj])

    metas = [c.get("meta", {}) for ts in raw["timesteps"] for c in ts["calls"]]
    print(json.dumps({
        "traj_id": traj_id,
        "steps": len(traj.steps),
        "stopped": raw["stopped"],
        "seconds": round(elapsed, 1),
        "model_calls": len(metas),
        "max_prompt_tokens": max((m.get("prompt_tokens", 0) for m in metas), default=0),
        "completion_tokens": sum(m.get("completion_tokens", 0) for m in metas),
        "raw": str(raw_path),
        "trajectory": str(traj_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
