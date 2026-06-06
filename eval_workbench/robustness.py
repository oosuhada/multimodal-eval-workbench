"""OOD/corruption robustness profiles built from lmms-eval result summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .gate import load_policy
from .report import summarize_result_file


def _metric(summary: dict[str, Any], task: str, metric: str) -> float | None:
    value = summary.get("tasks", {}).get(task, {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _relative_regression(baseline: float, current: float, direction: str) -> float:
    scale = max(abs(baseline), 1e-12)
    regression = baseline - current if direction == "higher" else current - baseline
    return float(regression / scale)


def evaluate_robustness(
    baseline: dict[str, Any],
    conditions: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure clean-to-shift degradation for selected task metrics."""
    metrics: list[dict[str, Any]] = []
    condition_regressions: dict[str, list[float]] = {name: [] for name in conditions}

    for check in checks:
        task = check["task"]
        metric = check["metric"]
        direction = check["direction"]
        base_value = _metric(baseline, task, metric)
        row: dict[str, Any] = {
            "task": task,
            "metric": metric,
            "direction": direction,
            "baseline": base_value,
            "conditions": {},
        }
        regressions: list[tuple[str, float]] = []
        for name, summary in conditions.items():
            value = _metric(summary, task, metric)
            if base_value is None or value is None:
                row["conditions"][name] = {"value": value, "relative_regression": None}
                continue
            relative = _relative_regression(base_value, value, direction)
            absolute = base_value - value if direction == "higher" else value - base_value
            row["conditions"][name] = {
                "value": value,
                "absolute_regression": float(absolute),
                "relative_regression": relative,
                "retention": float(1.0 - relative),
            }
            regressions.append((name, relative))
            condition_regressions[name].append(relative)
        if regressions:
            worst_name, worst_relative = max(regressions, key=lambda item: item[1])
            values = np.asarray([value for _, value in regressions], dtype=np.float64)
            row["mean_relative_regression"] = float(values.mean())
            row["worst_condition"] = worst_name
            row["worst_relative_regression"] = float(worst_relative)
            row["worst_case_retention"] = float(1.0 - worst_relative)
        metrics.append(row)

    by_condition: dict[str, Any] = {}
    all_regressions: list[float] = []
    for name, values in condition_regressions.items():
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0:
            by_condition[name] = {"metric_count": 0}
            continue
        all_regressions.extend(array.tolist())
        by_condition[name] = {
            "metric_count": int(len(array)),
            "mean_relative_regression": float(array.mean()),
            "worst_relative_regression": float(array.max()),
            "mean_retention": float(1.0 - array.mean()),
        }

    overall = np.asarray(all_regressions, dtype=np.float64)
    return {
        "condition_count": len(conditions),
        "metric_count": len(checks),
        "overall_mean_relative_regression": float(overall.mean()) if len(overall) else None,
        "overall_worst_relative_regression": float(overall.max()) if len(overall) else None,
        "overall_mean_retention": float(1.0 - overall.mean()) if len(overall) else None,
        "by_condition": by_condition,
        "metrics": metrics,
    }


def robustness_from_files(
    baseline_path: str | Path,
    condition_paths: dict[str, str | Path],
    policy_path: str | Path,
) -> dict[str, Any]:
    baseline = summarize_result_file(baseline_path)
    conditions = {name: summarize_result_file(path) for name, path in condition_paths.items()}
    result = evaluate_robustness(baseline, conditions, load_policy(policy_path))
    result["baseline"] = str(baseline_path)
    result["conditions"] = {name: str(path) for name, path in condition_paths.items()}
    result["policy"] = str(policy_path)
    return result


def parse_condition_args(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --condition must use name=path syntax.")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("Each --condition must use non-empty name=path syntax.")
        if name in parsed:
            raise ValueError(f"Duplicate condition name: {name}")
        parsed[name] = path
    if not parsed:
        raise ValueError("At least one --condition is required.")
    return parsed
