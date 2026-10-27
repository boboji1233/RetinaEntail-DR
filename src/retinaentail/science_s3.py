"""Command-line entry point for the bounded Science S3 HE/EX twin pilot."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from retinaentail.interventions.controlled_twin_pilot import (
    S3DataError,
    S3GateError,
    run_s3_pilot,
    run_s3_synthetic_fixture,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retinaentail.science_s3",
        description="Run the bounded Science S3 HE/EX controlled-twin pilot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser(
        "pilot", help="Run the authorized TRAIN/CALIBRATION-only real pilot."
    )
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--root", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)
    fixture = subparsers.add_parser(
        "fixture", help="Run the offline synthetic software fixture only."
    )
    fixture.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = _project_root()
    try:
        if args.command == "fixture":
            hashes = run_s3_synthetic_fixture(args.output_dir)
            print("status: PASS_SYNTHETIC_FIXTURE_ONLY")
            print(f"output: {args.output_dir}")
            print(f"artifact manifest hash: {hashes['artifact-manifest.json']}")
            return 0
        result = run_s3_pilot(
            project_root=project_root,
            config_path=args.config,
            dataset_root=args.root,
            output_dir=args.output_dir,
        )
    except (FileExistsError, S3DataError, S3GateError, OSError, ValueError) as error:
        print(f"Science S3 fail-closed: {error}", file=sys.stderr)
        return 2
    print(f"status: {result.status}")
    print(f"accepted twins: {result.accepted_count}")
    print(f"rejected attempts: {result.rejected_count}")
    print(f"accepted counts: {result.accepted_counts}")
    print(f"method counts: {result.method_counts}")
    print(f"locked-test root reads: {result.locked_test_root_reads}")
    print(f"artifact manifest hash: {result.artifact_hashes['artifact-manifest.json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
