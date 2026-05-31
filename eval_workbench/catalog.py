"""Filesystem-only catalog for the bundled lmms-eval task definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re


_TASK_LINE = re.compile(r"^task:\s*['\"]?([^'\"#\s]+)", re.MULTILINE)
_GROUP_LINE = re.compile(r"^group:\s*['\"]?([^'\"#\s]+)", re.MULTILINE)


@dataclass(frozen=True)
class TaskEntry:
    name: str
    path: str
    group: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def build_task_catalog(repo_root: str | Path = ".") -> list[TaskEntry]:
    root = Path(repo_root).resolve()
    task_root = root / "lmms_eval" / "tasks"
    entries: list[TaskEntry] = []

    for path in sorted(task_root.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        task_match = _TASK_LINE.search(text)
        group_match = _GROUP_LINE.search(text)
        entries.append(
            TaskEntry(
                name=task_match.group(1) if task_match else path.stem,
                path=str(path.relative_to(root)),
                group=group_match.group(1) if group_match else None,
            )
        )
    return entries


def search_tasks(
    query: str = "",
    repo_root: str | Path = ".",
    limit: int = 50,
) -> list[TaskEntry]:
    query_lower = query.strip().lower()
    results: list[TaskEntry] = []
    for entry in build_task_catalog(repo_root):
        haystack = f"{entry.name} {entry.group or ''} {entry.path}".lower()
        if query_lower and query_lower not in haystack:
            continue
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def missing_tasks(task_names: tuple[str, ...], repo_root: str | Path = ".") -> list[str]:
    catalog = build_task_catalog(repo_root)
    known = {entry.name for entry in catalog}
    known.update(entry.group for entry in catalog if entry.group)
    known.update(Path(entry.path).stem for entry in catalog)
    return [task for task in task_names if task not in known]
