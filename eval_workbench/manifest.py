"""Manifest parsing for reusable multimodal evaluation suites."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SuiteManifest:
    name: str
    model: str
    tasks: tuple[str, ...]
    model_args: dict[str, Any] = field(default_factory=dict)
    batch_size: str = "1"
    device: str | None = None
    limit: float | None = None
    output_path: str | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "SuiteManifest":
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Suite manifest does not exist: {manifest_path}")

        if manifest_path.suffix.lower() == ".json":
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise ValueError("Suite manifest must contain a mapping/object.")

        name = str(raw.get("name", "")).strip()
        model = str(raw.get("model", "")).strip()
        tasks = raw.get("tasks") or []
        model_args = raw.get("model_args") or {}

        if not name:
            raise ValueError("Manifest field 'name' is required.")
        if not model:
            raise ValueError("Manifest field 'model' is required.")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Manifest field 'tasks' must be a non-empty list.")
        if not isinstance(model_args, dict):
            raise ValueError("Manifest field 'model_args' must be a mapping/object.")

        limit = raw.get("limit")
        return cls(
            name=name,
            model=model,
            tasks=tuple(str(task) for task in tasks),
            model_args=dict(model_args),
            batch_size=str(raw.get("batch_size", "1")),
            device=str(raw["device"]) if raw.get("device") is not None else None,
            limit=float(limit) if limit is not None else None,
            output_path=str(raw["output_path"]) if raw.get("output_path") else None,
            tags=tuple(str(tag) for tag in (raw.get("tags") or [])),
            notes=str(raw.get("notes", "")),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "tasks": list(self.tasks),
            "model_args": self.model_args,
            "batch_size": self.batch_size,
            "device": self.device,
            "limit": self.limit,
            "output_path": self.output_path,
            "tags": list(self.tags),
            "notes": self.notes,
        }
