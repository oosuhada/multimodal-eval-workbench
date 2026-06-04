"""Calibration and selective-risk analysis for multimodal predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConfidenceRecord:
    confidence: float
    correct: bool
    brier: float
    nll: float
    normalized_entropy: float | None = None


def _clip_probability(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Probability/confidence must be in [0, 1], got {value}.")
    return min(max(value, 1e-12), 1.0 - 1e-12)


def _record_from_mapping(raw: dict[str, Any]) -> ConfidenceRecord:
    if "probabilities" in raw:
        probs = np.asarray(raw["probabilities"], dtype=np.float64)
        if probs.ndim != 1 or len(probs) < 2 or np.any(~np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("probabilities must be a finite non-negative 1-D array with at least two classes.")
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value.")
        probs = probs / total
        target = int(raw["target"])
        if not 0 <= target < len(probs):
            raise ValueError(f"target index {target} is outside probability vector.")
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        correct = prediction == target
        one_hot = np.zeros_like(probs)
        one_hot[target] = 1.0
        brier = float(np.sum((probs - one_hot) ** 2))
        nll = -math.log(max(float(probs[target]), 1e-12))
        entropy = -float(np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0)))
        normalized_entropy = entropy / math.log(len(probs))
        return ConfidenceRecord(confidence, correct, brier, nll, normalized_entropy)

    if "confidence" not in raw or "correct" not in raw:
        raise ValueError("Each record needs either probabilities+target or confidence+correct.")
    confidence = _clip_probability(float(raw["confidence"]))
    correct = bool(raw["correct"])
    target = 1.0 if correct else 0.0
    brier = (confidence - target) ** 2
    nll = -(target * math.log(confidence) + (1.0 - target) * math.log(1.0 - confidence))
    binary_entropy = -(confidence * math.log(confidence) + (1.0 - confidence) * math.log(1.0 - confidence))
    return ConfidenceRecord(confidence, correct, brier, nll, binary_entropy / math.log(2.0))


def load_confidence_records(path: str | Path) -> list[ConfidenceRecord]:
    source = Path(path)
    records: list[dict[str, Any]]
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("samples"))
        if not isinstance(payload, list):
            raise ValueError("JSON calibration input must be a list or contain records/samples list.")
        records = payload
    else:
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Calibration input is empty: {source}")
    return [_record_from_mapping(record) for record in records]


def _fixed_bins(confidence: np.ndarray, correct: np.ndarray, bins: int) -> tuple[float, float, list[dict[str, Any]]]:
    if bins < 2:
        raise ValueError("bins must be >= 2.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    details: list[dict[str, Any]] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(avg_confidence - accuracy)
        ece += count / len(confidence) * gap
        mce = max(mce, gap)
        details.append({
            "lower": lower,
            "upper": upper,
            "count": count,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "gap": gap,
        })
    return float(ece), float(mce), details


def _adaptive_ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    order = np.argsort(confidence)
    groups = np.array_split(order, min(bins, len(order)))
    value = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        gap = abs(float(confidence[group].mean()) - float(correct[group].mean()))
        value += len(group) / len(confidence) * gap
    return float(value)


def _selective_risk(confidence: np.ndarray, correct: np.ndarray, coverages: tuple[float, ...]) -> dict[str, Any]:
    order = np.argsort(-confidence)
    errors = 1.0 - correct[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    selective: dict[str, float] = {}
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage values must be in (0, 1].")
        count = max(1, int(math.ceil(len(order) * coverage)))
        selective[f"{coverage:.2f}"] = float(1.0 - cumulative_risk[count - 1])
    return {
        "aurc": float(cumulative_risk.mean()),
        "selective_accuracy": selective,
    }


def analyze_calibration(
    records: list[ConfidenceRecord],
    *,
    bins: int = 10,
    coverages: tuple[float, ...] = (0.5, 0.8, 0.9, 1.0),
) -> dict[str, Any]:
    confidence = np.asarray([record.confidence for record in records], dtype=np.float64)
    correct = np.asarray([record.correct for record in records], dtype=np.float64)
    brier = np.asarray([record.brier for record in records], dtype=np.float64)
    nll = np.asarray([record.nll for record in records], dtype=np.float64)
    entropy = np.asarray([record.normalized_entropy for record in records if record.normalized_entropy is not None], dtype=np.float64)
    ece, mce, reliability = _fixed_bins(confidence, correct, bins)
    wrong = confidence[correct == 0]
    right = confidence[correct == 1]
    result: dict[str, Any] = {
        "samples": int(len(records)),
        "accuracy": float(correct.mean()),
        "avg_confidence": float(confidence.mean()),
        "confidence_accuracy_gap": float(confidence.mean() - correct.mean()),
        "ece": ece,
        "adaptive_ece": _adaptive_ece(confidence, correct, bins),
        "mce": mce,
        "brier_score": float(brier.mean()),
        "nll": float(nll.mean()),
        "mean_normalized_entropy": float(entropy.mean()) if len(entropy) else None,
        "mean_confidence_on_errors": float(wrong.mean()) if len(wrong) else None,
        "mean_confidence_on_correct": float(right.mean()) if len(right) else None,
        "reliability_bins": reliability,
    }
    result.update(_selective_risk(confidence, correct, coverages))
    return result


def analyze_calibration_file(path: str | Path, *, bins: int = 10) -> dict[str, Any]:
    result = analyze_calibration(load_confidence_records(path), bins=bins)
    result["source"] = str(Path(path))
    return result
