"""Сторож квалификации R2: останавливает разметку модели, когда порог недостижим.

    setsid nohup scripts/r2_run_panel.sh data/v1/review gemma granite > reports/r2_panel.log 2>&1 &
    setsid nohup uv run python scripts/r2_guard.py gemma granite > reports/r2_guard.log 2>&1 &

Квалификация — тип ≥ 0.8 и t* ±1 ≥ 0.8 на всём V1 (r2_qualify.py). Когда ошибок
по одному из критериев больше, чем допускает порог, результат уже известен, и
дальнейшая разметка только тратит GPU. Сторож завершает r2_annotate.py этой
модели, очередь r2_run_panel.sh переходит к следующей. Модели обходятся в том же
порядке, что и в очереди. Ключ эталона модели не показывается: сторож читает
только её готовые ответы.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from r2_qualify import T_STAR_MIN, TYPE_MIN, read_jsonl, t_star_ok

ROOT = Path(__file__).resolve().parents[1]


def annotate_pids(model: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    return [int(line.split()[0]) for line in out.splitlines()
            if f"r2_annotate.py --model {model} " in line + " " and "r2_guard" not in line]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v1")
    ap.add_argument("--review-dir", type=Path, default=ROOT / "data" / "v1" / "review")
    args = ap.parse_args()

    mapping = json.loads((args.data_dir / "annotations" / "_mapping.json").read_text(encoding="utf-8"))
    expected = {r["traj_id"]: r for r in read_jsonl(args.data_dir / "expected.jsonl")}
    n = len(expected)
    max_type_err = n - math.ceil(TYPE_MIN * n)
    max_t_err = n - math.ceil(T_STAR_MIN * n)

    for model in args.models:
        path = args.review_dir / "r2" / f"{model}_s0.jsonl"
        seen_running = False
        while True:
            pids = annotate_pids(model)
            seen_running |= bool(pids)
            rows = read_jsonl(path) if path.exists() else []
            type_err = sum(r["label"] != expected[mapping[r["traj_id"]]]["expected_label"] for r in rows)
            t_err = sum(not t_star_ok(r["t_star"], expected[mapping[r["traj_id"]]]["expected_t_star"], 1)
                        for r in rows)
            if type_err > max_type_err or t_err > max_t_err:
                for pid in pids:
                    os.kill(pid, signal.SIGTERM)
                print(f"{model}: остановлен после {len(rows)}/{n}, ошибок типа {type_err} "
                      f"(допустимо {max_type_err}), t* {t_err} (допустимо {max_t_err})", flush=True)
                break
            if seen_running and not pids:
                print(f"{model}: разметка завершилась, {len(rows)}/{n}, ошибок типа {type_err}, t* {t_err}",
                      flush=True)
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
