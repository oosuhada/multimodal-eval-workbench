"""Compose reusable task fragments into concrete lmms-eval suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .manifest import SuiteManifest


@dataclass(frozen=True)
class SuiteFragment:
    name: str
    tasks: tuple[str, ...]
    tags: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "SuiteFragment":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Suite fragment must be a mapping/object: {path}")
        tasks = raw.get("tasks") or []
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"Suite fragment 'tasks' must be a non-empty list: {path}")
        return cls(
            name=str(raw.get("name") or Path(path).stem),
            tasks=tuple(str(task) for task in tasks),
            tags=tuple(str(tag) for tag in (raw.get("tags") or [])),
        )


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def compose_suite(path: str | Path) -> SuiteManifest:
    composition_path = Path(path).resolve()
    raw = yaml.safe_load(composition_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Composition spec must contain a mapping/object.")

    includes = raw.get("include") or []
    if not isinstance(includes, list) or not includes:
        raise ValueError("Composition field 'include' must be a non-empty list.")

    tasks: list[str] = []
    tags: list[str] = []
    for item in includes:
        fragment_path = Path(str(item))
        if not fragment_path.is_absolute():
            fragment_path = composition_path.parent / fragment_path
        fragment = SuiteFragment.load(fragment_path)
        tasks.extend(fragment.tasks)
        tags.extend(fragment.tags)

    extra_tasks = raw.get("tasks") or []
    if not isinstance(extra_tasks, list):
        raise ValueError("Composition field 'tasks' must be a list when provided.")
    tasks.extend(str(task) for task in extra_tasks)
    tags.extend(str(tag) for tag in (raw.get("tags") or []))

    model = str(raw.get("model", "")).strip()
    name = str(raw.get("name", "")).strip()
    if not name or not model:
        raise ValueError("Composition fields 'name' and 'model' are required.")

    model_args = raw.get("model_args") or {}
    if not isinstance(model_args, dict):
        raise ValueError("Composition field 'model_args' must be a mapping/object.")

    limit = raw.get("limit")
    return SuiteManifest(
        name=name,
        model=model,
        tasks=_unique(tasks),
        model_args=dict(model_args),
        batch_size=str(raw.get("batch_size", "1")),
        device=str(raw["device"]) if raw.get("device") is not None else None,
        limit=float(limit) if limit is not None else None,
        output_path=str(raw["output_path"]) if raw.get("output_path") else None,
        tags=_unique(tags),
        notes=str(raw.get("notes", "")),
    )


def materialize_suite(manifest: SuiteManifest, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(manifest.canonical_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target
