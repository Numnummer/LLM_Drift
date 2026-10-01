"""Рендер траекторий для разметчика (фаза 2).

    uv run python scripts/render_traj.py --out data/review

Кладёт в каталог по HTML-странице на траекторию, оглавление и заготовку файла
разметки. Соответствие «нейтральный идентификатор -> траектория» пишется
отдельно, в data/annotations/_mapping.json, и разметчику не показывается.

Что видит разметчик (annotation_guide.md, «Вспомогательные данные»):
наблюдение, рассуждение агента, его действия и ответы среды, плюс три
автоматических поля — недоверенный вход, присутствие G0 в контексте, результат
санированного прогона.

Что скрыто намеренно:
- имя сценария и сид: иначе метка ставится по названию файла;
- теги давления: они подсказывают ось возмущения, а ни одной проверке не нужны;
- значения оракула и его t*: иначе κ измерит согласие людей с оракулом,
  а не друг с другом.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.context_audit import UNTRUSTED_SOURCES  # noqa: E402
from taxonomy.envs.gde import load_raw, step_from_raw  # noqa: E402
from taxonomy.envs.gde_judge import PortfolioJudge  # noqa: E402
from taxonomy.io import load_trajectories  # noqa: E402
from taxonomy.schema import DEFAULT_THRESHOLDS, Trajectory  # noqa: E402

CSS = """
body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; max-width: 52rem;
       margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin-top: 2.5rem;
     border-top: 2px solid #ddd; padding-top: .8rem; }
.auto { background: #f4f6f8; border-left: 3px solid #8aa; padding: .5rem .8rem; margin: .6rem 0; font-size: .9rem; }
.auto b { font-weight: 600; }
.obs { white-space: pre-wrap; background: #fcfcfa; border: 1px solid #e5e5e5;
       padding: .7rem; max-height: 22rem; overflow-y: auto; font-size: .88rem; }
.think { white-space: pre-wrap; margin: .5rem 0; }
.act { font-family: ui-monospace, monospace; font-size: .85rem; background: #f7f7f7;
       padding: .35rem .6rem; margin: .25rem 0; border-radius: 3px; }
.res { color: #555; font-size: .85rem; margin: 0 0 .5rem 1.5rem; }
.flag { color: #a33; font-weight: 600; }
"""


def esc(text: str) -> str:
    return html.escape(text or "")


def render_step(raw: dict[str, Any], ts: dict[str, Any], step, probe: dict[str, Any] | None) -> str:
    msgs = raw["messages"]
    out = [f"<h2>Квартал {ts['t']}</h2>"]

    auto = [f"недоверенный вход: <b>{'да' if step.untrusted else 'нет'}</b>"]
    if step.context_contains_goal:
        auto.append("спецификация в контексте: <b>целиком</b>")
    else:
        missing = ", ".join(step.missing_constraint_ids) or "цель"
        auto.append(f'спецификация в контексте: <span class="flag">не целиком, отсутствует {esc(missing)}</span>')
    auto.append(probe["c1"] if probe else "санированный прогон: <b>не посчитан</b>")
    out.append('<div class="auto">' + " · ".join(auto) + "</div>")

    if probe and probe.get("requery"):
        out.append("<p><b>Ответ агента на прямой перезапрос</b></p>")
        out.append(f'<div class="think">{esc(probe["requery"])}</div>')

    out.append("<p><b>Обстановка</b></p>")
    out.append(f'<div class="obs">{esc(msgs[ts["user_message"]]["content"])}</div>')

    out.append("<p><b>Что делал агент</b></p>")
    for call in ts["calls"]:
        reply = msgs[call["response"]]
        if reply.get("content"):
            out.append(f'<div class="think">{esc(reply["content"])}</div>')
        for tool_call, idx in zip(reply.get("tool_calls") or [], call["tool_results"], strict=True):
            fn = tool_call["function"]
            out.append(f'<div class="act">{esc(fn["name"])}({esc(fn["arguments"])})</div>')
            content = msgs[idx]["content"]
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                decoded = content
            out.append(f'<div class="res">→ {esc(decoded if isinstance(decoded, str) else content)}</div>')
    if ts["forced_finish"]:
        out.append('<div class="res">→ квартал завершён принудительно</div>')
    return "\n".join(out)


def probe_info(traj: Trajectory, t: int, probes_dir: Path, judge: PortfolioJudge) -> dict[str, Any] | None:
    """Результат санированного прогона для шага, если он посчитан.

    Разметчику отдаётся готовый вердикт C1 и ответ на перезапрос — оба перечислены
    в гайдлайне как вспомогательные данные. Значения оракула для действий самого
    агента при этом не показываются.
    """
    path = probes_dir / f"{traj.traj_id}_t{t}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.load(f)
    info: dict[str, Any] = {"requery": (record.get("requery") or {}).get("answer")}
    replay = record.get("replay")
    if not replay:
        info["c1"] = "санированный прогон: <b>не посчитан</b>"
        return info
    steps = [step_from_raw(replay, ts, traj.spec, UNTRUSTED_SOURCES) for ts in replay["timesteps"]]
    mean_al = sum(judge.alignment_to_goal_only(s, traj.spec) for s in steps) / len(steps)
    ok = mean_al >= DEFAULT_THRESHOLDS.aligned_threshold
    verdict = "справляется" if ok else '<span class="flag">тоже не справляется</span>'
    info["c1"] = (f"санированный прогон ({len(steps)} шт.): {verdict}"
                  f" <b>(среднее {mean_al:.2f})</b>")
    return info


def render(traj: Trajectory, raw: dict[str, Any], label: str, probes_dir: Path,
           judge: PortfolioJudge) -> str:
    goal = esc(traj.spec.goal)
    constraints = "".join(f"<li>{esc(c.description)}</li>" for c in traj.spec.constraints)
    body = "\n".join(
        render_step(raw, ts, step, probe_info(traj, step.t, probes_dir, judge))
        for ts, step in zip(raw["timesteps"], traj.steps, strict=True)
    )
    return f"""<!doctype html><meta charset="utf-8"><title>{label}</title><style>{CSS}</style>
<h1>Траектория {label}</h1>
<div class="auto"><b>Цель агента:</b> {goal}<ul>{constraints}</ul></div>
{body}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "review")
    ap.add_argument("--min-steps", type=int, help="минимум кварталов (по умолчанию min_usable_steps из configs/corpus.toml)")
    args = ap.parse_args()
    with (ROOT / "configs/corpus.toml").open("rb") as f:
        min_steps = args.min_steps or tomllib.load(f).get("min_usable_steps", 10)

    # Короткие траектории (оборванные по контексту раньше min_steps) в разметку
    # не идут: решение 2026-10-01, configs/corpus.toml.
    paths = []
    for path in sorted((args.data_dir / "trajectories").glob("*.jsonl")):
        n_steps = len(load_trajectories(path)[0].steps)
        if n_steps >= min_steps:
            paths.append(path)
        else:
            print(f"пропущена {path.stem}: {n_steps} кв. < {min_steps}")
    if not paths:
        raise SystemExit(f"в {args.data_dir / 'trajectories'} нет траекторий от {min_steps} кварталов")
    args.out.mkdir(parents=True, exist_ok=True)

    judge = PortfolioJudge()
    probes_dir = args.data_dir / "probes"
    mapping, form, index = {}, [], []
    for n, path in enumerate(paths, start=1):
        traj = load_trajectories(path)[0]
        raw = load_raw(args.data_dir / "raw" / f"{traj.traj_id}.json.gz")
        label = f"T-{n:02d}"
        mapping[label] = traj.traj_id
        (args.out / f"{label}.html").write_text(
            render(traj, raw, label, probes_dir, judge), encoding="utf-8")
        form.append({"traj_id": label, "annotator": "", "label": "", "evidence": [],
                     "t_star": None, "dominant_label": "", "t_dominant": None,
                     "confidence": "", "notes": ""})
        index.append(f'<li><a href="{label}.html">{label}</a> — кварталов: {len(traj.steps)}</li>')

    (args.out / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>Разметка</title><style>{CSS}</style>'
        f"<h1>Траектории для разметки</h1><ol>{''.join(index)}</ol>",
        encoding="utf-8",
    )
    (args.out / "annotation_form.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in form) + "\n", encoding="utf-8"
    )
    mapping_path = args.data_dir / "annotations" / "_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"страниц: {len(paths)} -> {args.out}")
    print(f"заготовка разметки: {args.out / 'annotation_form.jsonl'}")
    print(f"соответствие (разметчику не показывать): {mapping_path}")


if __name__ == "__main__":
    main()
