"""Command-line interface for multimodal benchmark suite planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import SuiteManifest
from .planner import build_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mm-eval-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Render a suite into an exact lmms-eval command.")
    plan.add_argument("manifest")
    plan.add_argument("--repo-root", default=".")
    plan.add_argument("--json", action="store_true", dest="as_json")
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


if __name__ == "__main__":
    main()
