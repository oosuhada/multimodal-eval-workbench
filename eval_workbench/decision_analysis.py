"""Multi-objective decision analysis for integrated experiment scorecards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_OBJECTIVES: dict[str, str] = {
    "clean_mean_r_at_1": "max",
    "ood_mean_r_at_1": "max",
    "ood_mean_retention": "max",
    "ood_worst_retention": "max",
    "ece": "min",
    "correctness_nll": "min",
    "trainable_parameter_pct": "min",
}


def _better_or_equal(a: float, b: float, direction: str, tol: float = 1e-12) -> bool:
    return a >= b - tol if direction == "max" else a <= b + tol


def _strictly_better(a: float, b: float, direction: str, tol: float = 1e-12) -> bool:
    return a > b + tol if direction == "max" else a < b - tol


def pareto_frontier(rows: list[dict[str, Any]], objectives: dict[str, str] | None = None) -> list[str]:
    objectives = objectives or DEFAULT_OBJECTIVES
    frontier: list[str] = []
    for candidate in rows:
        dominated = False
        for challenger in rows:
            if challenger is candidate:
                continue
            if not all(
                isinstance(candidate.get(metric), (int, float))
                and isinstance(challenger.get(metric), (int, float))
                for metric in objectives
            ):
                continue
            no_worse = all(
                _better_or_equal(float(challenger[metric]), float(candidate[metric]), direction)
                for metric, direction in objectives.items()
            )
            better_somewhere = any(
                _strictly_better(float(challenger[metric]), float(candidate[metric]), direction)
                for metric, direction in objectives.items()
            )
            if no_worse and better_somewhere:
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["variant"]))
    return frontier


def analyze_scorecard(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str = "base",
    objectives: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Scorecard is empty.")
    objectives = objectives or DEFAULT_OBJECTIVES
    by_variant = {str(row["variant"]): row for row in rows}
    if baseline_variant not in by_variant:
        raise ValueError(f"Baseline variant not found: {baseline_variant}")
    baseline = by_variant[baseline_variant]

    winners: dict[str, dict[str, Any]] = {}
    for metric, direction in objectives.items():
        available = [row for row in rows if isinstance(row.get(metric), (int, float))]
        if not available:
            continue
        values = [float(row[metric]) for row in available]
        best_value = max(values) if direction == "max" else min(values)
        winners[metric] = {
            "direction": direction,
            "value": best_value,
            "variants": [
                str(row["variant"])
                for row in available
                if abs(float(row[metric]) - best_value) <= 1e-12
            ],
        }

    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["variant"] == baseline_variant:
            continue
        entry: dict[str, Any] = {"variant": row["variant"]}
        for metric, direction in objectives.items():
            value, base_value = row.get(metric), baseline.get(metric)
            if not isinstance(value, (int, float)) or not isinstance(base_value, (int, float)):
                continue
            raw_delta = float(value) - float(base_value)
            entry[f"{metric}_delta"] = raw_delta
            entry[f"{metric}_directional_delta"] = raw_delta if direction == "max" else -raw_delta
        deltas.append(entry)

    return {
        "baseline": baseline_variant,
        "objectives": objectives,
        "pareto_frontier": pareto_frontier(rows, objectives),
        "objective_winners": winners,
        "deltas_vs_baseline": deltas,
    }


def analyze_scorecard_file(path: str | Path, *, baseline_variant: str = "base") -> dict[str, Any]:
    source = Path(path)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Scorecard JSON must contain a list of variant rows.")
    result = analyze_scorecard(rows, baseline_variant=baseline_variant)
    result["source"] = str(source)
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Canonical decision analysis",
        "",
        f"Baseline: `{result['baseline']}`",
        "",
        "## Objective winners",
        "",
        "| Objective | Direction | Winner | Value |",
        "|---|---|---|---:|",
    ]
    for metric, item in result["objective_winners"].items():
        lines.append(f"| {metric} | {item['direction']} | {', '.join(item['variants'])} | {item['value']:.6f} |")
    lines.extend([
        "",
        "## Pareto frontier",
        "",
        ", ".join(f"`{name}`" for name in result["pareto_frontier"]),
        "",
        "Multiple frontier variants mean there is no universal winner under the declared objectives.",
        "",
        "## Deltas vs base",
        "",
        "| Variant | Clean R@1 Δ | OOD R@1 Δ | Retention Δ | Worst retention Δ | ECE improvement | NLL improvement | Params % Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result["deltas_vs_baseline"]:
        lines.append(
            "| {variant} | {clean:+.5f} | {ood:+.5f} | {ret:+.5f} | {worst:+.5f} | {ece:+.5f} | {nll:+.5f} | {params:+.4f} |".format(
                variant=row["variant"],
                clean=row.get("clean_mean_r_at_1_directional_delta", 0.0),
                ood=row.get("ood_mean_r_at_1_directional_delta", 0.0),
                ret=row.get("ood_mean_retention_directional_delta", 0.0),
                worst=row.get("ood_worst_retention_directional_delta", 0.0),
                ece=row.get("ece_directional_delta", 0.0),
                nll=row.get("correctness_nll_directional_delta", 0.0),
                params=row.get("trainable_parameter_pct_delta", 0.0),
            )
        )
    return "\n".join(lines) + "\n"
