"""Command-line interface for multimodal benchmark suite planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import missing_tasks, search_tasks
from .manifest import SuiteManifest
from .planner import build_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mm-eval-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Render a suite into an exact lmms-eval command.")
    plan.add_argument("manifest")
    plan.add_argument("--repo-root", default=".")
    plan.add_argument("--json", action="store_true", dest="as_json")

    catalog = subparsers.add_parser("catalog", help="Search lmms-eval task definitions without importing models.")
    catalog.add_argument("query", nargs="?", default="")
    catalog.add_argument("--repo-root", default=".")
    catalog.add_argument("--limit", type=int, default=30)

    validate = subparsers.add_parser("validate", help="Check whether suite task names exist in the bundled task catalog.")
    validate.add_argument("manifest")
    validate.add_argument("--repo-root", default=".")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        manifest = SuiteManifest.load(Path(args.manifest))
        plan = build_plan(manifest, args.repo_root)
        if args.as_json:
            print(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False))
        else:
            print(plan.shell_command)
    elif args.command == "catalog":
        entries = search_tasks(args.query, args.repo_root, args.limit)
        print(json.dumps([entry.as_dict() for entry in entries], indent=2, ensure_ascii=False))
    elif args.command == "validate":
        manifest = SuiteManifest.load(Path(args.manifest))
        missing = missing_tasks(manifest.tasks, args.repo_root)
        result = {"suite": manifest.name, "valid": not missing, "missing_tasks": missing}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if missing:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
