"""Scalar post-hoc temperature scaling for multimodal classifier logits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import ConfidenceRecord, analyze_calibration


def load_logit_records(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("samples"))
        if not isinstance(payload, list):
            raise ValueError("Temperature input JSON must be a list or contain records/samples.")
        rows = payload
    else:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Temperature input is empty: {source}")

    logits: list[list[float]] = []
    targets: list[int] = []
    width: int | None = None
    for index, row in enumerate(rows):
        if "logits" not in row or "target" not in row:
            raise ValueError(f"Record {index} requires logits and target.")
        values = np.asarray(row["logits"], dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)):
            raise ValueError(f"Record {index} has invalid logits.")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("All logit vectors must use the same class dimension.")
        target = int(row["target"])
        if not 0 <= target < len(values):
            raise ValueError(f"Record {index} target is outside logit vector.")
        logits.append(values.tolist())
        targets.append(target)
    return np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=int)


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number.")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probabilities = softmax_with_temperature(logits, temperature)
    true_probability = probabilities[np.arange(len(targets)), targets]
    return -float(np.mean(np.log(np.maximum(true_probability, 1e-12))))


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    iterations: int = 80,
) -> float:
    """Minimize held-out NLL with a bounded golden-section search in log(T)."""
    if not 0 < min_temperature < max_temperature:
        raise ValueError("Expected 0 < min_temperature < max_temperature.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    left = math.log(min_temperature)
    right = math.log(max_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = multiclass_nll(logits, targets, math.exp(x1))
    f2 = multiclass_nll(logits, targets, math.exp(x2))
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = multiclass_nll(logits, targets, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = multiclass_nll(logits, targets, math.exp(x2))
    return math.exp((left + right) / 2.0)


def _confidence_records(probabilities: np.ndarray, targets: np.ndarray) -> list[ConfidenceRecord]:
    records: list[ConfidenceRecord] = []
    for probs, target in zip(probabilities, targets):
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        correct = prediction == int(target)
        one_hot = np.zeros_like(probs)
        one_hot[int(target)] = 1.0
        brier = float(np.sum((probs - one_hot) ** 2))
        nll = -math.log(max(float(probs[int(target)]), 1e-12))
        entropy = -float(np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0)))
        records.append(ConfidenceRecord(confidence, correct, brier, nll, entropy / math.log(len(probs))))
    return records


def temperature_report(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    iterations: int = 80,
) -> dict[str, Any]:
    temperature = fit_temperature(
        logits,
        targets,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        iterations=iterations,
    )
    baseline_probs = softmax_with_temperature(logits, 1.0)
    calibrated_probs = softmax_with_temperature(logits, temperature)
    baseline = analyze_calibration(_confidence_records(baseline_probs, targets), bins=bins)
    calibrated = analyze_calibration(_confidence_records(calibrated_probs, targets), bins=bins)
    return {
        "samples": int(len(targets)),
        "classes": int(logits.shape[1]),
        "temperature": float(temperature),
        "baseline": baseline,
        "calibrated": calibrated,
        "delta": {
            "nll": calibrated["nll"] - baseline["nll"],
            "ece": calibrated["ece"] - baseline["ece"],
            "brier_score": calibrated["brier_score"] - baseline["brier_score"],
        },
    }


def temperature_scale_file(
    path: str | Path,
    *,
    bins: int = 10,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    iterations: int = 80,
    write_probabilities: str | Path | None = None,
) -> dict[str, Any]:
    logits, targets = load_logit_records(path)
    report = temperature_report(
        logits,
        targets,
        bins=bins,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        iterations=iterations,
    )
    if write_probabilities is not None:
        destination = Path(write_probabilities)
        destination.parent.mkdir(parents=True, exist_ok=True)
        probabilities = softmax_with_temperature(logits, report["temperature"])
        with destination.open("w", encoding="utf-8") as handle:
            for probs, target in zip(probabilities, targets):
                handle.write(json.dumps({"probabilities": probs.tolist(), "target": int(target)}) + "\n")
        report["calibrated_predictions"] = str(destination)
    report["source"] = str(Path(path))
    return report
