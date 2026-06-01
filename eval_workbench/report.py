"""Compact summaries for the standard lmms-eval result payload."""

from __future__ import annotations

import json
from numbers import Number
from pathlib import Path
from typing import Any


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return {"tasks": {}, "task_count": 0}

    task_summaries: dict[str, dict[str, int | float]] = {}
    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        scalar_metrics: dict[str, int | float] = {}
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, bool):
                continue
            if isinstance(metric_value, Number):
                scalar_metrics[str(metric_name)] = metric_value
        task_summaries[str(task_name)] = scalar_metrics

    return {
        "task_count": len(task_summaries),
        "tasks": task_summaries,
        "model": payload.get("model_name") or payload.get("model") or payload.get("model_configs"),
    }


def summarize_result_file(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lmms-eval result JSON must contain an object at the top level.")
    summary = summarize_payload(payload)
    summary["source"] = str(result_path)
    return summary
