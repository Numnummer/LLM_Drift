"""Offline event-based taxonomy v2 assessment, not the legacy V2 acceptance test.

    .venv/bin/python scripts/classify_corpus.py --data-dir data/v2 --taxonomy v2

Reads existing trajectories, raw context and optional caches. Never generates
probes or explanations and never calls the N2 judge. Legacy outputs stay intact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.classify_v2 import classify_v2  # noqa: E402
from taxonomy.envs.gde import agent_instructions, load_raw, step_from_raw  # noqa: E402
from taxonomy.envs.gde_v2 import GDEEventJudge  # noqa: E402
from taxonomy.context_audit import UNTRUSTED_SOURCES  # noqa: E402
from taxonomy.io import load_trajectories, write_jsonl  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def pair_assessment(pair: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    experimental, control = records[pair["traj_id"]], records[pair["control_id"]]
    a, b = ({s["t"]: s for s in r["steps"]} for r in (experimental, control))
    common = sorted(set(a) & set(b))
    k = pair["k"]
    comparisons = []
    for t in common:
        x, y = a[t], b[t]
        scores_differ = (x["alignment"] is not None and y["alignment"] is not None
                         and abs(x["alignment"] - y["alignment"]) > 1e-9)
        codes_x = sorted({f["code"] for f in x["manifestations"]})
        codes_y = sorted({f["code"] for f in y["manifestations"]})
        comparisons.append({"t": t, "after_intervention": t >= k,
                            "experimental": {"status": x["status"], "alignment": x["alignment"],
                                             "spec_version": x["spec_version"], "manifestations": codes_x},
                            "control": {"status": y["status"], "alignment": y["alignment"],
                                        "spec_version": y["spec_version"], "manifestations": codes_y},
                            "assessment_differs": scores_differ or x["status"] != y["status"] or codes_x != codes_y,
                            "same_goal": x["goal"] == y["goal"],
                            "experimental_authorization": x["authorization"],
                            "control_authorization": y["authorization"]})
    differences = [c["t"] for c in comparisons if c["after_intervention"] and c["assessment_differs"]]
    return {**pair, "taxonomy_version": "2", "usable_prefix": pair["same_before_k"],
            "experimental_verdict": {key: experimental[key] for key in
                                     ("label", "t_star", "manifestations", "first_manifestations",
                                      "t_dominant", "dominant_label", "dominant_labels")},
            "control_verdict": {key: control[key] for key in
                                ("label", "t_star", "manifestations", "first_manifestations",
                                 "t_dominant", "dominant_label", "dominant_labels")},
            "first_assessment_difference_after_k": differences[0] if differences else None,
            "first_specification_difference_after_k": next((c["t"] for c in comparisons
                                                            if c["after_intervention"] and not c["same_goal"]), None),
            "experimental_deviations_after_k": [t for t in a if t >= k and a[t]["status"] == "deviation"],
            "control_deviations_after_k": [t for t in b if t >= k and b[t]["status"] == "deviation"],
            "early_deviation": any(r["t_star"] is not None and r["t_star"] < k for r in (experimental, control)),
            "experimental_only_observed_steps": sorted(set(a) - set(b)),
            "control_only_observed_steps": sorted(set(b) - set(a)),
            "steps": comparisons,
            "note": "Expected labels are legacy hypotheses, not truth; differences do not establish causality. Each arm uses its own active goal."}


def run(data_dir: Path) -> None:
    output = data_dir / "taxonomy_v2"
    judge = GDEEventJudge()
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for path in sorted((data_dir / "trajectories").glob("*.jsonl")):
        traj = load_trajectories(path)[0]
        raw_path = data_dir / "raw" / f"{traj.traj_id}.json.gz"
        raw = load_raw(raw_path)
        timesteps = {ts["t"]: ts for ts in raw["timesteps"]}
        for s in traj.steps:
            s.env_state["agent_instructions"] = agent_instructions(raw, timesteps[s.t])
        result = classify_v2(traj, judge)
        prefix = []
        for step, assessment in zip(traj.steps, result.steps, strict=True):
            prefix.append(step)
            cache_path = data_dir / "probes" / f"{traj.traj_id}_t{step.t}.json.gz"
            assessment.diagnostics["sanitized_probe"] = {"status": "not_cached", "used_as_gate": False}
            if not cache_path.exists():
                continue
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("format") != "probe/1":
                raise ValueError(f"Unsupported probe cache: {cache_path}")
            source_hashes[str(cache_path)] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            if cache.get("requery"):
                assessment.explanation = {**cache["requery"], "source": str(cache_path),
                                          "authorization_status": "not_authorized_by_explanation",
                                          "used_as_gate": False}
            replay = cache.get("replay")
            if replay is None:
                continue
            scores, retained, removed = [], [], []
            for ts in replay["timesteps"][:5]:
                replica = step_from_raw(replay, ts, traj.spec, UNTRUSTED_SOURCES)
                scores.append(judge.probe_goal_alignment(replica, traj.spec, prefix))
                retained.append(replica.env_state["observation_parts"])
                kept = Counter((p["source"], p["text"]) for p in replica.env_state["observation_parts"])
                dropped = []
                for part in step.env_state["observation_parts"]:
                    key = (part["source"], part["text"])
                    if kept[key]:
                        kept[key] -= 1
                    else:
                        dropped.append(part)
                removed.append(dropped)
            assessment.diagnostics["sanitized_probe"] = {
                "status": "cached", "source": str(cache_path), "used_as_gate": False,
                "scores_goal_only": scores, "mean_goal_only": sum(scores) / len(scores) if scores else None,
                "seed": cache["seed"], "n_runs": len(scores), "policy": replay["policy"],
                "history_removed": True, "retained_observation_parts": retained,
                "removed_observation_parts": removed, "evaluation_spec_version": assessment.spec_version,
                "caveat": "Cached same-step counterfactual; no causal attribution and no suppression of deviation."}
        record = asdict(result)
        for assessment, serialized in zip(result.steps, record["steps"], strict=True):
            serialized["label"] = assessment.label
        records.append(record)
        for source in (path, raw_path):
            source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        print(f"{traj.traj_id}: {result.label} t*={result.t_star} manifestations={','.join(result.manifestations) or '—'} "
              f"events={len(result.evidence)} t_dominant={result.t_dominant}", flush=True)
    if not records:
        raise ValueError("No trajectories found")
    by_id = {r["traj_id"]: r for r in records}
    pairs_path = data_dir / "pairs.jsonl"
    pairs = [pair_assessment(p, by_id) for p in read_jsonl(pairs_path)] if pairs_path.exists() else []
    v1_path = data_dir / "verdicts_agent.jsonl"
    old = {v["traj_id"]: v for v in read_jsonl(v1_path)} if v1_path.exists() else {}
    for source in (pairs_path, v1_path):
        if source.exists():
            source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = {"taxonomy_version": "2", "trajectories": len(records), "pairs": len(pairs),
               "labels": dict(Counter(r["label"] for r in records)),
               "pairs_with_matching_prefix": sum(p["usable_prefix"] for p in pairs),
               "cached_probe_steps": sum(s["diagnostics"].get("sanitized_probe", {}).get("status") == "cached"
                                         for r in records for s in r["steps"]),
               "step_statuses": dict(Counter(s["status"] for r in records for s in r["steps"])),
               "manifestations": dict(Counter(f["code"] for r in records for s in r["steps"] for f in s["manifestations"])),
               "limitations": ["D1 semantic attribution requires review; opposing flows are candidates only.",
                               "D7/D8 are hypotheses, not established causes.",
                               "D2 has no validated new decision rule and is not assigned.",
                               "Only known GDE profit board updates supported; no generic environment monitor.",
                               "Literal G0 context audit is not semantic loss of active G_t.",
                               "No fresh probes, requery, LLM judge or authorization calls."],
               "legacy_comparison": [{"traj_id": r["traj_id"], "v1": old.get(r["traj_id"]),
                                      "v2_label": r["label"], "v2_t_star": r["t_star"],
                                      "v2_manifestations": r["manifestations"]} for r in records],
               "source_hashes": source_hashes}
    for source in (ROOT / "docs/taxonomy_v2.md", ROOT / "src/taxonomy/classify_v2.py",
                   ROOT / "src/taxonomy/envs/gde_v2.py", Path(__file__)):
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "verdicts.jsonl", records)
    write_jsonl(output / "pairs.jsonl", pairs)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} trajectories and {len(pairs)} pairs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/v2")
    run(parser.parse_args().data_dir)
