"""Разметка траекторий моделью панели R2 в два прохода (plan.md, фаза 2).

    scripts/serve_panel.sh granite &
    uv run python scripts/r2_annotate.py --model granite --review-dir data/v1/review

Вход — то, что render_traj.py кладёт для модели: model/_instructions.md (тот же
текст документов, что у человека) и model/T-NN.md. Протокол — как у человека
(MODEL_HEADER в render_traj.py):

  проход 1: по траектории выбрать t_star и t_dominant;
  проход 2: получить справки lookup/T-NN/qMM.md только по этим кварталам и пройти
            дерево до конца. t_star и t_dominant после справок не меняются.

Справки отдаются после выбора, потому что по тому, где они посчитаны, был бы
виден t* оракула. Если модель во втором проходе всё же изменила t_star или
t_dominant, в разметку идут значения первого прохода, а расхождение пишется в
поле `pass2_changed`.

Каждый вызов (запрос, ответ, рассуждение, время) пишется в лог рядом с выходом:
разбор матрицы несогласия без рассуждений невозможен. Уже размеченные
траектории при повторном запуске пропускаются.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "n1", "n2", "n3", "n4", "ok"}
MAX_RETRIES = 2

PASS1 = ("Проход 1. Определи по траектории t_star и t_dominant (номера кварталов или null) "
         "и ответь ровно одной строкой JSON: "
         '{{"traj_id": "{label}", "pass": 1, "t_star": ..., "t_dominant": ...}}')
PASS2 = ("Проход 2. Справки по выбранным кварталам ниже. Пройди решающую процедуру до конца "
         "и ответь ровно одной строкой JSON в формате из инструкции (поле annotator: \"{name}\"). "
         "t_star и t_dominant не меняй.\n\n{lookups}")
NO_LOOKUPS = "Отклонения ты не нашёл, справки не нужны."
RETRY = ("Ответ не разобран: {error}. Повтори ответ — ровно одна строка JSON, без пояснений вокруг.")


def chat(endpoint: str, model: str, messages: list[dict[str, str]], cfg: dict[str, Any],
         seed: int, max_tokens: int) -> dict[str, Any]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "seed": seed,
            **cfg.get("sampling", {})}
    if cfg.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = cfg["chat_template_kwargs"]
    req = urllib.request.Request(f"{endpoint}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def last_json(text: str) -> dict[str, Any]:
    """Последний JSON-объект в ответе: модели иногда добавляют пояснения вокруг."""
    for m in reversed(list(re.finditer(r"\{", text))):
        tail = text[m.start():]
        try:
            obj, _ = json.JSONDecoder().raw_decode(tail)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "traj_id" in obj:
            return obj
    raise ValueError("нет строки JSON с полем traj_id")


def _quarter(value: Any, quarters: set[int], field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in quarters:
        raise ValueError(f"{field} = {value!r}: нужен номер квартала из траектории или null")
    return value


def check_pass1(obj: dict[str, Any], quarters: set[int]) -> dict[str, int | None]:
    return {"t_star": _quarter(obj.get("t_star"), quarters, "t_star"),
            "t_dominant": _quarter(obj.get("t_dominant"), quarters, "t_dominant")}


def check_pass2(obj: dict[str, Any]) -> dict[str, Any]:
    label = str(obj.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"label = {obj.get('label')!r}: нужна одна из {sorted(LABELS)}")
    dominant = str(obj.get("dominant_label") or "").strip().lower()
    if dominant and dominant not in LABELS:
        raise ValueError(f"dominant_label = {obj.get('dominant_label')!r}")
    evidence = obj.get("evidence") or []
    if not isinstance(evidence, list) or not all(isinstance(e, int) for e in evidence):
        raise ValueError("evidence: нужен список номеров кварталов")
    return {"label": label, "dominant_label": dominant, "evidence": evidence,
            "confidence": str(obj.get("confidence", "")), "notes": str(obj.get("notes", ""))}


class Session:
    """Диалог с моделью по одной траектории; каждый вызов — в лог."""

    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any], label: str, log) -> None:
        self.args, self.cfg, self.label, self.log = args, cfg, label, log
        self.messages: list[dict[str, str]] = []

    def ask(self, prompt: str, parse) -> tuple[dict[str, Any], Any]:
        self.messages.append({"role": "user", "content": prompt})
        error = ""
        for attempt in range(MAX_RETRIES + 1):
            start = time.monotonic()
            resp = chat(self.args.endpoint, self.args.model, self.messages, self.cfg,
                        self.args.seed, self.args.max_tokens)
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            self.log.write(json.dumps({
                "traj_id": self.label, "model": self.args.model, "seed": self.args.seed,
                "turn": len(self.messages), "attempt": attempt, "prompt": prompt if attempt == 0 else RETRY,
                "content": content, "reasoning": msg.get("reasoning_content"),
                "finish_reason": resp["choices"][0].get("finish_reason"),
                "usage": resp.get("usage"), "seconds": round(time.monotonic() - start, 1),
            }, ensure_ascii=False) + "\n")
            self.log.flush()
            # В историю идёт только ответ, без рассуждения: так рекомендуют
            # разработчики Gemma и gpt-oss для многоходового диалога.
            self.messages.append({"role": "assistant", "content": content})
            try:
                obj = last_json(content)
                return obj, parse(obj)
            except ValueError as e:
                error = str(e)
                self.messages.append({"role": "user", "content": RETRY.format(error=error)})
        raise RuntimeError(f"{self.label}: ответ не разобран за {MAX_RETRIES + 1} попытки: {error}")


def annotate(args: argparse.Namespace, cfg: dict[str, Any], instructions: str, label: str, log) -> dict[str, Any]:
    text = (args.review_dir / "model" / f"{label}.md").read_text(encoding="utf-8")
    quarters = {int(q) for q in re.findall(r"^## Квартал (\d+)", text, flags=re.M)}
    s = Session(args, cfg, label, log)
    s.messages.append({"role": "system", "content": instructions})
    _, p1 = s.ask(f"{text}\n\n---\n\n{PASS1.format(label=label)}", lambda o: check_pass1(o, quarters))

    picked = sorted({t for t in (p1["t_star"], p1["t_dominant"]) if t is not None})
    lookups = "\n\n".join((args.review_dir / "lookup" / label / f"q{t:02d}.md").read_text(encoding="utf-8")
                          for t in picked) or NO_LOOKUPS
    raw2, p2 = s.ask(PASS2.format(name=args.model, lookups=lookups), check_pass2)
    changed = {k: raw2.get(k) for k in ("t_star", "t_dominant") if k in raw2 and raw2.get(k) != p1[k]}
    return {"traj_id": label, "annotator": args.model, **p2, **p1,
            "pass2_changed": changed, "seed": args.seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="имя из configs/panel.toml")
    ap.add_argument("--review-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="по умолчанию <review-dir>/r2/<model>_s<seed>.jsonl")
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=12288, help="на один ответ, включая рассуждение")
    ap.add_argument("--only", nargs="*", help="только эти T-NN")
    args = ap.parse_args()

    cfg = tomllib.loads((ROOT / "configs/panel.toml").read_text(encoding="utf-8"))["models"][args.model]
    instructions = (args.review_dir / "model" / "_instructions.md").read_text(encoding="utf-8")
    out = args.out or args.review_dir / "r2" / f"{args.model}_s{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["traj_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    labels = args.only or sorted(p.stem for p in (args.review_dir / "model").glob("T-*.md"))

    started = time.monotonic()
    with out.open("a", encoding="utf-8") as f, out.with_suffix(".log.jsonl").open("a", encoding="utf-8") as log:
        for label in labels:
            if label in done:
                continue
            t0 = time.monotonic()
            try:
                row = annotate(args, cfg, instructions, label, log)
            except RuntimeError as e:
                print(f"{label}: ПРОПУЩЕНА — {e}")
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{label}: {row['label']} t*={row['t_star']} ({time.monotonic() - t0:.0f} с)", flush=True)
    print(f"готово за {(time.monotonic() - started) / 60:.1f} мин -> {out}")


if __name__ == "__main__":
    main()
