"""Command-line entry point for the Science S2 lesion catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from retinaentail.data.lesion_catalog import (
    S2DataError,
    S2GateError,
    run_s2_catalog,
    run_s2_synthetic_fixture,
)


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("could not locate project root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m retinaentail.science_s2")
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="run the offline synthetic S2 fixture")
    fixture.add_argument("--output-dir", type=Path, required=True)
    catalog = commands.add_parser("catalog", help="run the authorized IDRiD TRAIN/CAL catalog")
    catalog.add_argument("--root", type=Path, required=True)
    catalog.add_argument(
        "--config",
        type=Path,
        default=Path("configs/science/s2_idrid_lesion_catalog.yaml"),
    )
    catalog.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated/science/s2_idrid_lesion_catalog"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = _project_root()
    try:
        if arguments.command == "fixture":
            result = run_s2_synthetic_fixture(arguments.output_dir.resolve(strict=False))
        else:
            config_path = arguments.config
            output_dir = arguments.output_dir
            if not config_path.is_absolute():
                config_path = project_root / config_path
            if not output_dir.is_absolute():
                output_dir = project_root / output_dir
            result = run_s2_catalog(
                project_root=project_root,
                config_path=config_path.resolve(strict=True),
                dataset_root=arguments.root.absolute(),
                output_dir=output_dir,
            )
    except S2GateError as error:
        print(f"S2 status: BLOCKED\nreason: {error}\nreal root accessed: false", file=sys.stderr)
        return 2
    except (OSError, S2DataError, TypeError, ValueError) as error:
        print(f"S2 status: FAIL_CLOSED\nreason: {error}", file=sys.stderr)
        return 1
    print(f"S2 status: {result.status}")
    print(f"development sources: {result.source_count}")
    print(f"development masks: {result.mask_count}")
    print(f"components: {result.component_count}")
    print(f"locked-test root reads: {result.locked_test_root_reads}")
    print(f"artifact manifest hash: {result.artifact_hashes['artifact-manifest.json']}")
    print(f"output: {result.output_dir}")
    print(json.dumps(result.component_counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
