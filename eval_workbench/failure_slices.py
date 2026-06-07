"""Embedding-space clustering for automatic multimodal failure slice discovery."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_failure_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Failure JSONL rows must be objects.")
        if row.get("correct") is True:
            continue
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Each failure row requires a non-empty embedding list.")
        records.append(row)
    if len(records) < 2:
        raise ValueError("Failure slice discovery requires at least two failure records.")
    dimensions = {len(record["embedding"]) for record in records}
    if len(dimensions) != 1:
        raise ValueError("All failure embeddings must have the same dimension.")
    return records


def _normalized_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([record["embedding"] for record in records], dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _pca(values: np.ndarray, dimensions: int) -> tuple[np.ndarray, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:dimensions].T
    reduced = centered @ basis
    variances = singular_values**2
    explained = float(variances[:dimensions].sum() / max(float(variances.sum()), 1e-12))
    return reduced, explained


def _kmeans_plus_plus(values: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = [values[int(rng.integers(0, len(values)))]]
    while len(centroids) < k:
        distances = np.min(
            np.stack([np.square(values - centroid).sum(axis=1) for centroid in centroids], axis=1),
            axis=1,
        )
        total = float(distances.sum())
        if total <= 1e-12:
            remaining = [index for index in range(len(values)) if not any(np.array_equal(values[index], c) for c in centroids)]
            centroids.append(values[remaining[0] if remaining else int(rng.integers(0, len(values)))])
            continue
        centroids.append(values[int(rng.choice(len(values), p=distances / total))])
    return np.asarray(centroids, dtype=np.float64)


def _kmeans(values: np.ndarray, k: int, seed: int, iterations: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for restart in range(10):
        centroids = _kmeans_plus_plus(values, k, seed + restart)
        labels = np.zeros(len(values), dtype=int)
        for _ in range(iterations):
            distances = np.square(values[:, None, :] - centroids[None, :, :]).sum(axis=2)
            new_labels = np.argmin(distances, axis=1)
            new_centroids = centroids.copy()
            for cluster_id in range(k):
                members = values[new_labels == cluster_id]
                if len(members):
                    new_centroids[cluster_id] = members.mean(axis=0)
            if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
                labels = new_labels
                centroids = new_centroids
                break
            labels = new_labels
            centroids = new_centroids
        inertia = float(np.square(values - centroids[labels]).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), centroids.copy(), inertia)
    assert best is not None
    return best


def _silhouette(values: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2 or len(values) < 3:
        return None
    pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    scores: list[float] = []
    for index, cluster in enumerate(labels):
        same = np.flatnonzero(labels == cluster)
        same = same[same != index]
        a = float(pairwise[index, same].mean()) if len(same) else 0.0
        b_values = [
            float(pairwise[index, labels == other].mean())
            for other in unique
            if other != cluster and np.any(labels == other)
        ]
        b = min(b_values) if b_values else 0.0
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 1e-12 else 0.0)
    return float(np.mean(scores))


def _top_values(records: list[dict[str, Any]], key: str, limit: int = 3) -> list[dict[str, Any]]:
    counts = Counter(str(record[key]) for record in records if record.get(key) not in {None, ""})
    total = sum(counts.values())
    return [
        {"value": value, "count": count, "fraction": count / max(total, 1)}
        for value, count in counts.most_common(limit)
    ]


def _choose_k(values: np.ndarray, max_k: int, seed: int) -> tuple[int, dict[int, float]]:
    n = len(values)
    upper = min(max_k, n - 1)
    if upper < 2:
        return 1, {}
    scores: dict[int, float] = {}
    for k in range(2, upper + 1):
        labels, _, _ = _kmeans(values, k, seed)
        if len(set(labels.tolist())) < 2:
            continue
        score = _silhouette(values, labels)
        if score is not None:
            scores[k] = score
    if not scores:
        return 1, {}
    return max(scores, key=scores.get), scores


def discover_failure_slices(
    records: list[dict[str, Any]],
    *,
    clusters: int | None = None,
    max_clusters: int = 8,
    pca_dims: int | None = 32,
    representatives: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    if representatives < 1:
        raise ValueError("representatives must be >= 1.")
    values = _normalized_matrix(records)
    original_dimensions = int(values.shape[1])
    reduced_dimensions = original_dimensions
    explained_variance = None
    if pca_dims is not None and pca_dims > 0 and pca_dims < values.shape[1] and len(values) > 2:
        dims = min(int(pca_dims), len(values) - 1, values.shape[1])
        values, explained_variance = _pca(values, dims)
        reduced_dimensions = dims

    silhouette_by_k: dict[int, float] = {}
    if clusters is None:
        k, silhouette_by_k = _choose_k(values, max_clusters, seed)
    else:
        k = int(clusters)
        if k < 1 or k > len(records):
            raise ValueError("clusters must be between 1 and sample count.")

    if k == 1:
        labels = np.zeros(len(records), dtype=int)
        centroids = np.mean(values, axis=0, keepdims=True)
        inertia = float(np.square(values - centroids[0]).sum())
        silhouette = None
    else:
        labels, centroids, inertia = _kmeans(values, k, seed)
        silhouette = _silhouette(values, labels)

    slices: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for cluster_id in range(k):
        indices = np.flatnonzero(labels == cluster_id)
        cluster_records = [records[index] for index in indices]
        distances = np.linalg.norm(values[indices] - centroids[cluster_id], axis=1)
        order = indices[np.argsort(distances)[: min(representatives, len(indices))]]
        top_tasks = _top_values(cluster_records, "task")
        top_errors = _top_values(cluster_records, "error")
        label_bits = []
        if top_tasks:
            label_bits.append(top_tasks[0]["value"])
        if top_errors:
            label_bits.append(top_errors[0]["value"])
        slice_name = " / ".join(label_bits) if label_bits else f"failure-cluster-{cluster_id}"
        confidences = [float(record["confidence"]) for record in cluster_records if isinstance(record.get("confidence"), (int, float))]
        slices.append(
            {
                "cluster": cluster_id,
                "name": slice_name,
                "size": len(cluster_records),
                "fraction": len(cluster_records) / len(records),
                "mean_confidence": float(np.mean(confidences)) if confidences else None,
                "top_tasks": top_tasks,
                "top_errors": top_errors,
                "representatives": [
                    {
                        "id": str(records[index].get("id", index)),
                        "task": records[index].get("task"),
                        "error": records[index].get("error"),
                        "confidence": records[index].get("confidence"),
                    }
                    for index in order
                ],
            }
        )
        for index in indices:
            assignments.append(
                {
                    "id": str(records[index].get("id", index)),
                    "cluster": cluster_id,
                    "slice": slice_name,
                }
            )

    slices.sort(key=lambda item: (-item["size"], item["cluster"]))
    return {
        "failure_count": len(records),
        "embedding_dimensions": original_dimensions,
        "analysis_dimensions": reduced_dimensions,
        "pca_explained_variance": explained_variance,
        "cluster_count": k,
        "silhouette": silhouette,
        "silhouette_by_k": {str(key): value for key, value in sorted(silhouette_by_k.items())},
        "inertia": inertia,
        "slices": slices,
        "assignments": assignments,
    }


def discover_failure_slices_file(
    path: str | Path,
    *,
    clusters: int | None = None,
    max_clusters: int = 8,
    pca_dims: int | None = 32,
    representatives: int = 3,
    seed: int = 42,
    write_assignments: str | Path | None = None,
) -> dict[str, Any]:
    result = discover_failure_slices(
        load_failure_records(path),
        clusters=clusters,
        max_clusters=max_clusters,
        pca_dims=pca_dims,
        representatives=representatives,
        seed=seed,
    )
    if write_assignments:
        destination = Path(write_assignments)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in result["assignments"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["assignments_written_to"] = str(destination)
    return result
