"""Suite orchestration extensions for the bundled lmms-eval codebase."""

from .calibration import ConfidenceRecord, analyze_calibration, analyze_calibration_file
from .composition import SuiteFragment, compose_suite
from .gate import evaluate_gate, gate_result_files
from .manifest import SuiteManifest
from .planner import SuitePlan, build_plan
from .catalog import TaskEntry, missing_tasks, search_tasks

__all__ = [
    "ConfidenceRecord",
    "SuiteFragment",
    "SuiteManifest",
    "SuitePlan",
    "TaskEntry",
    "analyze_calibration",
    "analyze_calibration_file",
    "build_plan",
    "compose_suite",
    "evaluate_gate",
    "gate_result_files",
    "missing_tasks",
    "search_tasks",
]
