"""Joint clean/OOD/calibration comparison for multimodal model revisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .calibration import analyze_calibration_file
from .gate import load_policy
from .report import summarize_result_file
from .robustness import robustness_from_files


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    clean_result: Path
    calibration_predictions: Path | None
    conditions: dict[str, Path]

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationProfile":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Evaluation profile must contain a mapping/object.")
        name = str(raw.get("name") or source.stem)
        clean = raw.get("clean_result")
        if not clean:
            raise ValueError("Evaluation profile requires clean_result.")

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (source.parent / candidate).resolve()

        calibration = raw.get("calibration_predictions")
        conditions_raw = raw.get("conditions") or {}
        if not isinstance(conditions_raw, dict):
            raise ValueError("Evaluation profile conditions must be a mapping of name -> result path.")
        return cls(
            name=name,
            clean_result=resolve(str(clean)),
            calibration_predictions=resolve(str(calibration)) if calibration else None,
            conditions={str(key): resolve(str(value)) for key, value in conditions_raw.items()},
        )


def _metric(summary: dict[str, Any], task: str, metric: str) -> float | None:
    value = summary.get("tasks", {}).get(task, {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _clean_comparison(
    baseline: dict[str, Any], current: dict[str, Any], checks: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    directional_improvements: list[float] = []
    relative_improvements: list[float] = []
    for check in checks:
        before = _metric(baseline, check["task"], check["metric"])
        after = _metric(current, check["task"], check["metric"])
        if before is None or after is None:
            improvement = None
            delta = None
        else:
            delta = after - before
            improvement = delta if check["direction"] == "higher" else -delta
            directional_improvements.append(float(improvement))
            relative_improvement = improvement / max(abs(before), 1e-12)
            relative_improvements.append(float(relative_improvement))
        if before is None or after is None:
            relative_improvement = None
        rows.append(
            {
                "task": check["task"],
                "metric": check["metric"],
                "direction": check["direction"],
                "baseline": before,
                "current": after,
                "delta": delta,
                "directional_improvement": improvement,
                "relative_directional_improvement": relative_improvement,
            }
        )
    values = np.asarray(directional_improvements, dtype=np.float64)
    relative_values = np.asarray(relative_improvements, dtype=np.float64)
    return {
        "metric_count": len(rows),
        "mean_directional_improvement": float(values.mean()) if len(values) else None,
        "mean_relative_directional_improvement": float(relative_values.mean()) if len(relative_values) else None,
        "improved_metrics": int(sum(value > 0 for value in directional_improvements)),
        "regressed_metrics": int(sum(value < 0 for value in directional_improvements)),
        "metrics": rows,
    }


def _calibration_comparison(baseline_path: Path | None, current_path: Path | None, bins: int) -> dict[str, Any] | None:
    if baseline_path is None or current_path is None:
        return None
    baseline = analyze_calibration_file(baseline_path, bins=bins)
    current = analyze_calibration_file(current_path, bins=bins)
    definitions = {
        "accuracy": "higher",
        "ece": "lower",
        "adaptive_ece": "lower",
        "brier_score": "lower",
        "nll": "lower",
        "aurc": "lower",
        "mean_confidence_on_errors": "lower",
    }
    metrics: dict[str, Any] = {}
    improved = 0
    regressed = 0
    for key, direction in definitions.items():
        before = baseline.get(key)
        after = current.get(key)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        delta = float(after - before)
        directional = delta if direction == "higher" else -delta
        improved += int(directional > 0)
        regressed += int(directional < 0)
        metrics[key] = {
            "direction": direction,
            "baseline": float(before),
            "current": float(after),
            "delta": delta,
            "directional_improvement": directional,
        }
    return {"improved_metrics": improved, "regressed_metrics": regressed, "metrics": metrics}


def _robustness_comparison(
    baseline: EvaluationProfile,
    current: EvaluationProfile,
    policy_path: str | Path,
) -> dict[str, Any] | None:
    shared_conditions = sorted(set(baseline.conditions) & set(current.conditions))
    if not shared_conditions:
        return None
    before = robustness_from_files(
        baseline.clean_result,
        {name: baseline.conditions[name] for name in shared_conditions},
        policy_path,
    )
    after = robustness_from_files(
        current.clean_result,
        {name: current.conditions[name] for name in shared_conditions},
        policy_path,
    )
    before_retention = before.get("overall_mean_retention")
    after_retention = after.get("overall_mean_retention")
    conditions: dict[str, Any] = {}
    for name in shared_conditions:
        left = before.get("by_condition", {}).get(name, {}).get("mean_retention")
        right = after.get("by_condition", {}).get(name, {}).get("mean_retention")
        conditions[name] = {
            "baseline_retention": left,
            "current_retention": right,
            "delta": (right - left) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None,
        }
    return {
        "shared_conditions": shared_conditions,
        "baseline_mean_retention": before_retention,
        "current_mean_retention": after_retention,
        "retention_delta": (
            after_retention - before_retention
            if isinstance(before_retention, (int, float)) and isinstance(after_retention, (int, float))
            else None
        ),
        "conditions": conditions,
    }


def compare_evaluation_profiles(
    baseline_profile: str | Path,
    current_profile: str | Path,
    policy_path: str | Path,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    baseline = EvaluationProfile.load(baseline_profile)
    current = EvaluationProfile.load(current_profile)
    checks = load_policy(policy_path)
    clean = _clean_comparison(
        summarize_result_file(baseline.clean_result),
        summarize_result_file(current.clean_result),
        checks,
    )
    calibration = _calibration_comparison(
        baseline.calibration_predictions, current.calibration_predictions, bins
    )
    robustness = _robustness_comparison(baseline, current, policy_path)
    scorecard = {
        "clean_mean_relative_directional_improvement": clean["mean_relative_directional_improvement"],
        "ece_delta": calibration["metrics"].get("ece", {}).get("delta") if calibration else None,
        "nll_delta": calibration["metrics"].get("nll", {}).get("delta") if calibration else None,
        "robustness_retention_delta": robustness.get("retention_delta") if robustness else None,
    }
    return {
        "baseline": baseline.name,
        "current": current.name,
        "scorecard": scorecard,
        "clean": clean,
        "calibration": calibration,
        "robustness": robustness,
    }
