#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EEG_ANALYSIS_ROOT = REPO_ROOT / "eeg_analysis"
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.utils.mlflow_sync import (  # noqa: E402
    SyncError,
    format_sync_object,
    sync_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync one MLflow experiment between local and remote file stores.")
    parser.add_argument("--direction", choices=["both", "push", "pull"], default="both")
    parser.add_argument("--remote", default="abin@ssh.coalmine.ai", help="SSH target for the remote MLflow store.")
    parser.add_argument("--local-root", default="~/eeg-mlflow/mlruns", help="Local MLflow root.")
    parser.add_argument("--remote-root", default="~/eeg-mlflow/mlruns", help="Remote MLflow root.")
    parser.add_argument("--local-exp", default="134978379211635499", help="Local experiment ID.")
    parser.add_argument("--remote-exp", default="714557200892293258", help="Remote experiment ID.")
    parser.add_argument("--dry-run", action="store_true", help="Show the sync plan without copying anything.")
    parser.add_argument("--verbose", action="store_true", help="Print every push/pull/skip item.")
    parser.add_argument(
        "--exclude-run-id",
        action="append",
        default=[],
        help="Run ID to ignore during sync. Can be passed multiple times.",
    )
    return parser.parse_args()


def print_section(title: str, objects: list, verbose: bool) -> None:
    print(f"{title}: {len(objects)}")
    if verbose:
        for obj in objects:
            print(f"  - {format_sync_object(obj)}")


def print_conflicts(conflicts: list) -> None:
    print(f"Conflicts: {len(conflicts)}")
    for conflict in conflicts:
        print(
            f"  - {conflict.kind} {conflict.identifier}: "
            f"local={conflict.local_digest} remote={conflict.remote_digest}"
        )


def main() -> int:
    args = parse_args()
    try:
        result = sync_experiment(
            local_root=args.local_root,
            remote_root=args.remote_root,
            local_experiment_id=args.local_exp,
            remote_experiment_id=args.remote_exp,
            remote_host=args.remote,
            direction=args.direction,
            dry_run=args.dry_run,
            exclude_run_ids=args.exclude_run_id,
        )
    except SyncError as exc:
        print(f"Sync failed: {exc}", file=sys.stderr)
        return 1

    print(f"Local experiment: {result.local_root / result.local_experiment_id}")
    print(f"Remote experiment: {result.remote_host}:{result.remote_root}/{result.remote_experiment_id}")
    print(f"Direction: {args.direction}")
    print(f"Mode: {'dry-run' if args.dry_run else 'execute'}")

    print_section("Push", result.plan.push, args.verbose or args.dry_run)
    print_section("Pull", result.plan.pull, args.verbose or args.dry_run)
    print_section("Skip", result.plan.skip, args.verbose)

    if result.plan.conflicts:
        print_conflicts(result.plan.conflicts)
        return 2

    if args.dry_run:
        print("Dry run complete. No changes were made.")
        return 0

    print("Sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
