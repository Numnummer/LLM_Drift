"""Event-based taxonomy v2; deliberately separate from the legacy CHECKS tree.

Detection uses only the current prefix. Research summaries are retrospective.
No model calls, probes, authorization decisions or environment writes occur here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .schema import DEFAULT_THRESHOLDS, GoalSpec, Step, Trajectory


@dataclass(frozen=True)
class Finding:
    code: Literal["d1", "d4", "d5", "d6", "d7", "d8"]
    evidence: tuple[int, ...]
    rationale: str
    certainty: Literal["observed", "hypothesis", "requires_review"] = "observed"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("finding requires evidence")


@dataclass
class StepAssessment:
    t: int
    spec_version: str
    goal: str
    status: Literal["aligned", "deviation", "unknown"]
    alignment: float | None
    goal_alignment: float | None
    evidence: tuple[int, ...] = ()
    manifestations: list[Finding] = field(default_factory=list)
    mechanisms: list[Finding] = field(default_factory=list)
    review_candidates: list[Finding] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    authorization: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return {"aligned": "ok", "deviation": "deviation", "unknown": "unknown"}[self.status]

    def __post_init__(self) -> None:
        if self.status == "deviation" and not self.evidence:
            raise ValueError("deviation requires evidence")
        for value in (self.alignment, self.goal_alignment):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("alignment must be finite and in [0, 1]")


class EventJudge(Protocol):
    """Replaceable environment/judgment boundary for the v2 procedure."""

    def assess(self, step: Step, spec: GoalSpec, prefix: list[Step],
               threshold: float) -> StepAssessment: ...


@dataclass
class TrajectoryAssessment:
    traj_id: str
    taxonomy_version: str
    label: str
    t_star: int | None
    evidence: tuple[int, ...]
    manifestations: tuple[str, ...]
    first_manifestations: tuple[str, ...]
    t_dominant: int | None
    dominant_label: str | None
    dominant_labels: tuple[str, ...]
    steps: list[StepAssessment]
    research: dict[str, Any]


def classify_v2(traj: Trajectory, judge: EventJudge,
                threshold: float = DEFAULT_THRESHOLDS.aligned_threshold) -> TrajectoryAssessment:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if any(a.t >= b.t for a, b in zip(traj.steps, traj.steps[1:])):
        raise ValueError("step indices must be strictly increasing")
    prefix: list[Step] = []
    rows: list[StepAssessment] = []
    for step in traj.steps:
        prefix.append(step)
        rows.append(judge.assess(step, traj.spec, list(prefix), threshold))
    deviations = [r for r in rows if r.status == "deviation"]
    goal_devs = [r for r in rows if r.goal_alignment is not None and r.goal_alignment < threshold]
    # Earliest of equally bad goal deviations; never changes t_star or event labels.
    dominant = min(goal_devs, key=lambda r: (r.goal_alignment, r.t)) if goal_devs else None
    valid = [r for r in rows if r.status != "unknown"]
    return TrajectoryAssessment(
        traj_id=traj.traj_id, taxonomy_version="2", label="deviation" if deviations else
        ("unknown" if len(valid) != len(rows) or not rows else "ok"),
        t_star=deviations[0].t if deviations else None,
        evidence=tuple(r.t for r in deviations),
        manifestations=tuple(sorted({f.code for r in deviations for f in r.manifestations})),
        first_manifestations=tuple(sorted({f.code for f in deviations[0].manifestations})) if deviations else (),
        t_dominant=dominant.t if dominant else None,
        dominant_label="deviation" if dominant else None,
        dominant_labels=tuple(f.code for f in dominant.manifestations) if dominant else (),
        steps=rows,
        research={
            "d2": {"status": "not_assigned", "reason": "No validated semantic hybridization rule"},
            "deviation_steps": [r.t for r in deviations],
            "unknown_steps": [r.t for r in rows if r.status == "unknown"],
            "starts_at_first_step": bool(rows and rows[0].status == "deviation"),
            "aligned_share_observed": sum(r.status == "aligned" for r in valid) / len(valid) if valid else None,
            "spec_versions": list(dict.fromkeys(r.spec_version for r in rows)),
            "note": "Descriptive profile, not a D1/D2 decision or proof of causality.",
        },
    )
