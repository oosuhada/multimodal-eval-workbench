"""Regression gates for comparing lmms-eval result files against a baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .report import summarize_result_file


@dataclass(frozen=True)
class MetricCheck:
    task: str
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    max_regression: float
    direction: str
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_policy(path: str | Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Gate policy must contain a mapping/object.")
    checks = raw.get("checks") or []
    if not isinstance(checks, list) or not checks:
        raise ValueError("Gate policy field 'checks' must be a non-empty list.")
    normalized: list[dict[str, Any]] = []
    for item in checks:
        if not isinstance(item, dict):
            raise ValueError("Each gate check must be a mapping/object.")
        task = str(item.get("task", "")).strip()
        metric = str(item.get("metric", "")).strip()
        direction = str(item.get("direction", "higher")).strip().lower()
        if not task or not metric:
            raise ValueError("Each gate check requires 'task' and 'metric'.")
        if direction not in {"higher", "lower"}:
            raise ValueError("Gate check 'direction' must be 'higher' or 'lower'.")
        normalized.append(
            {
                "task": task,
                "metric": metric,
                "direction": direction,
                "max_regression": float(item.get("max_regression", 0.0)),
            }
        )
    return normalized


def _metric(summary: dict[str, Any], task: str, metric: str) -> float | None:
    value = summary.get("tasks", {}).get(task, {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def evaluate_gate(
    baseline: dict[str, Any],
    current: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    results: list[MetricCheck] = []
    for check in checks:
        task = check["task"]
        metric = check["metric"]
        direction = check["direction"]
        max_regression = float(check["max_regression"])
        before = _metric(baseline, task, metric)
        after = _metric(current, task, metric)

        if before is None or after is None:
            results.append(
                MetricCheck(
                    task=task,
                    metric=metric,
                    baseline=before,
                    current=after,
                    delta=None,
                    max_regression=max_regression,
                    direction=direction,
                    passed=False,
                    reason="metric missing from baseline or current result",
                )
            )
            continue

        delta = after - before
        regression = -delta if direction == "higher" else delta
        passed = regression <= max_regression
        results.append(
            MetricCheck(
                task=task,
                metric=metric,
                baseline=before,
                current=after,
                delta=delta,
                max_regression=max_regression,
                direction=direction,
                passed=passed,
                reason="within regression budget" if passed else "regression budget exceeded",
            )
        )

    return {
        "passed": all(result.passed for result in results),
        "check_count": len(results),
        "failed_count": sum(not result.passed for result in results),
        "checks": [result.as_dict() for result in results],
    }


def gate_result_files(baseline_path: str | Path, current_path: str | Path, policy_path: str | Path) -> dict[str, Any]:
    baseline = summarize_result_file(baseline_path)
    current = summarize_result_file(current_path)
    result = evaluate_gate(baseline, current, load_policy(policy_path))
    result["baseline"] = str(baseline_path)
    result["current"] = str(current_path)
    result["policy"] = str(policy_path)
    return result
