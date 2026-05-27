"""Render suite manifests into exact lmms-eval commands."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys

from .manifest import SuiteManifest


@dataclass(frozen=True)
class SuitePlan:
    name: str
    command: tuple[str, ...]
    suite_hash: str

    @property
    def shell_command(self) -> str:
        return shlex.join(self.command)

    def as_dict(self) -> dict[str, str | list[str]]:
        return {
            "name": self.name,
            "command": list(self.command),
            "shell_command": self.shell_command,
            "suite_hash": self.suite_hash,
        }


def _suite_hash(manifest: SuiteManifest) -> str:
    canonical = json.dumps(
        manifest.canonical_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _render_model_args(model_args: dict) -> str:
    rendered: list[str] = []
    for key in sorted(model_args):
        value = model_args[key]
        if isinstance(value, bool):
            value = "true" if value else "false"
        rendered.append(f"{key}={value}")
    return ",".join(rendered)


def build_plan(manifest: SuiteManifest, repo_root: str | Path = ".") -> SuitePlan:
    root = Path(repo_root).resolve()
    if not (root / "lmms_eval").exists():
        raise FileNotFoundError(f"lmms_eval package not found under: {root}")

    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        manifest.model,
        "--tasks",
        ",".join(manifest.tasks),
        "--batch_size",
        manifest.batch_size,
    ]

    if manifest.model_args:
        command.extend(["--model_args", _render_model_args(manifest.model_args)])
    if manifest.device:
        command.extend(["--device", manifest.device])
    if manifest.limit is not None:
        command.extend(["--limit", str(manifest.limit)])
    if manifest.output_path:
        command.extend(["--output_path", manifest.output_path])

    return SuitePlan(name=manifest.name, command=tuple(command), suite_hash=_suite_hash(manifest))
