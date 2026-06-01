"""Append-only provenance ledger for lmms-eval suite executions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from .manifest import SuiteManifest
from .planner import SuitePlan


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EvalRunRecord:
    run_id: str
    suite: str
    model: str
    tasks: list[str]
    status: str
    started_at: str
    finished_at: str
    exit_code: int
    suite_hash: str
    output_path: str | None
    command: list[str]
    tags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvalRunLedger:
    def __init__(self, path: str | Path = "artifacts/run-ledger.jsonl") -> None:
        self.path = Path(path)

    def append(self, record: EvalRunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]


def execute_plan(
    manifest: SuiteManifest,
    plan: SuitePlan,
    ledger: EvalRunLedger,
    cwd: str | Path = ".",
) -> EvalRunRecord:
    started_at = _utc_now()
    run_id = f"{started_at.replace(':', '').replace('+00:00', 'Z')}-{plan.suite_hash[:8]}"
    result = subprocess.run(list(plan.command), cwd=Path(cwd), check=False)
    finished_at = _utc_now()
    record = EvalRunRecord(
        run_id=run_id,
        suite=manifest.name,
        model=manifest.model,
        tasks=list(manifest.tasks),
        status="succeeded" if result.returncode == 0 else "failed",
        started_at=started_at,
        finished_at=finished_at,
        exit_code=result.returncode,
        suite_hash=plan.suite_hash,
        output_path=manifest.output_path,
        command=list(plan.command),
        tags=list(manifest.tags),
    )
    ledger.append(record)
    return record
