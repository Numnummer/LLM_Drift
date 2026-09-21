"""Пересборка траекторий из сырых логов по текущим конфигам.

Нужна, когда поменялась спецификация: тексты `C`, предикаты ограничений,
политика недоверенных источников. Прогоны переснимать не надо — конвертер
детерминирован и берёт всё из сырого лога (docs/plan.md, фаза 0).

    uv run python scripts/reconvert.py            # всё из data/raw
    uv run python scripts/reconvert.py --dry-run  # только показать, что изменится

Сценарий берётся из имени файла: <сценарий>_s<сид>[_scripted].
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_env import build  # noqa: E402

from taxonomy.context_audit import UNTRUSTED_SOURCES  # noqa: E402
from taxonomy.envs.gde import load_raw, to_trajectory  # noqa: E402
from taxonomy.io import load_trajectories, save_trajectories  # noqa: E402

NAME = re.compile(r"^(?P<scenario>.+)_s(?P<seed>\d+)(?P<scripted>_scripted)?$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = 0
    for raw_path in sorted((args.data_dir / "raw").glob("*.json.gz")):
        traj_id = raw_path.name.removesuffix(".json.gz")
        m = NAME.match(traj_id)
        if not m:
            print(f"пропуск {traj_id}: имя не разобрано")
            continue
        _, spec, _ = build(m["scenario"], int(m["seed"]))
        traj = to_trajectory(
            load_raw(raw_path), traj_id=traj_id, spec=spec,
            untrusted_sources=UNTRUSTED_SOURCES, scenario=m["scenario"],
        )
        out = args.data_dir / "trajectories" / f"{traj_id}.jsonl"
        was = load_trajectories(out)[0] if out.exists() else None
        if was == traj:
            print(f"без изменений {traj_id}")
            continue
        changed += 1
        print(f"{'изменится' if args.dry_run else 'пересобран'} {traj_id}")
        if not args.dry_run:
            save_trajectories(out, [traj])
    print(f"\nизменено: {changed}")


if __name__ == "__main__":
    main()
