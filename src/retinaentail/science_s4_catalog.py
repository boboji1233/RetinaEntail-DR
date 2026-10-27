"""Freeze and build the IDRiD MA/SE development catalog for Science S4."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from retinaentail.data.lesion_catalog import (
    S2DataError,
    S2GateError,
    run_s2_catalog,
)
from retinaentail.data.stage2_models import load_dataset_adapter_spec
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import canonical_json_bytes, sha256_bytes, write_bytes_fail_safe

_DEVELOPMENT_SPLITS = frozenset({"TRAIN", "CALIBRATION"})
_MA_DIRECTORY = "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms"
_SE_DIRECTORY = "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates"


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("could not locate project root")


def _inside_project(project_root: Path, path: Path, label: str) -> Path:
    resolved_root = project_root.resolve(strict=True)
    candidate = path if path.is_absolute() else resolved_root / path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise S2GateError(f"{label} must remain inside the project") from error
    return resolved


def freeze_idrid_ma_se_spec(
    *,
    project_root: Path,
    dataset_root: Path,
    upstream_spec_path: Path,
    output_spec_path: Path,
    output_summary_path: Path,
) -> dict[str, JSONValue]:
    """Freeze a complete-label MA/SE development spec without reading LOCKED_TEST paths."""

    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise S2DataError("IDRiD root is not a directory")
    upstream = load_dataset_adapter_spec(upstream_spec_path.resolve(strict=True))
    if upstream.dataset_id != "idrid":
        raise S2GateError("upstream spec must be IDRiD")
    output_spec = _inside_project(project_root, output_spec_path, "output spec")
    output_summary = _inside_project(project_root, output_summary_path, "output summary")
    expected_parent = (project_root / "data_registry" / "adapters").resolve(strict=True)
    if output_spec.parent != expected_parent or output_summary.parent != expected_parent:
        raise S2GateError("frozen adapter outputs must stay in data_registry/adapters")
    if output_spec.exists() or output_summary.exists():
        raise FileExistsError("refusing to overwrite an existing MA/SE freeze")

    entries: list[JSONValue] = []
    exclusions: list[JSONValue] = []
    split_counts: Counter[str] = Counter()
    development_entries = 0
    locked_entries = 0
    root_path_checks = 0
    for entry in upstream.entries:
        if entry.split not in _DEVELOPMENT_SPLITS:
            locked_entries += int(entry.split == "LOCKED_TEST")
            continue
        development_entries += 1
        image_name = entry.image_path.stem
        ma_relative = f"{_MA_DIRECTORY}/{image_name}_MA.tif"
        se_relative = f"{_SE_DIRECTORY}/{image_name}_SE.tif"
        ma_exists = (root / ma_relative).is_file()
        se_exists = (root / se_relative).is_file()
        root_path_checks += 2
        missing: list[JSONValue] = [
            semantic
            for semantic, exists in (
                ("MICROANEURYSM_MASK", ma_exists),
                ("SOFT_EXUDATE_MASK", se_exists),
            )
            if not exists
        ]
        if missing:
            exclusions.append(
                {
                    "source_image_id": entry.source_image_id,
                    "split": entry.split,
                    "reason": "MISSING_LABEL_UNKNOWN_EXCLUDED_NOT_NEGATIVE",
                    "missing_semantics": missing,
                }
            )
            continue
        split_counts[entry.split] += 1
        entries.append(
            {
                "schema_version": "2.0.0",
                "image_path": entry.image_path.as_posix(),
                "source_image_id": entry.source_image_id,
                "split": entry.split,
                "patient_id": entry.patient_id,
                "eye_id": entry.eye_id,
                "followup_id": entry.followup_id,
                "intervention_pair_id": entry.intervention_pair_id,
                "labels": [
                    {
                        "schema_version": "2.0.0",
                        "semantic": "MICROANEURYSM_MASK",
                        "relative_path": ma_relative,
                    },
                    {
                        "schema_version": "2.0.0",
                        "semantic": "SOFT_EXUDATE_MASK",
                        "relative_path": se_relative,
                    },
                ],
            }
        )
    if not entries:
        raise S2DataError("no complete-label MA/SE development records were found")
    spec: dict[str, JSONValue] = {
        "schema_version": "2.0.0",
        "dataset_id": "idrid",
        "label_semantics": ["MICROANEURYSM_MASK", "SOFT_EXUDATE_MASK"],
        "entries": entries,
    }
    spec_bytes = canonical_json_bytes(spec)
    summary: dict[str, JSONValue] = {
        "schema_version": "1.0.0",
        "freeze_id": "IDRID_MA_SE_S4_V1",
        "scope": "TRAIN_AND_CALIBRATION_COMPLETE_LABEL_COHORT_ONLY",
        "missing_label_policy": "UNKNOWN_EXCLUDED_NOT_NEGATIVE",
        "upstream_development_entries": development_entries,
        "selected_complete_label_entries": len(entries),
        "excluded_unknown_entries": len(exclusions),
        "locked_test_membership_entries": locked_entries,
        "locked_test_root_path_checks": 0,
        "development_label_path_checks": root_path_checks,
        "selected_split_counts": dict(sorted(split_counts.items())),
        "exclusions": exclusions,
        "adapter_spec_sha256": sha256_bytes(spec_bytes),
    }
    write_bytes_fail_safe(output_spec, spec_bytes)
    write_bytes_fail_safe(output_summary, canonical_json_bytes(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m retinaentail.science_s4_catalog")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-spec", help="freeze the IDRiD MA/SE dev adapter")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument(
        "--upstream-spec",
        type=Path,
        default=Path("data_registry/adapters/idrid-he-ex-s1-v1.json"),
    )
    freeze.add_argument(
        "--output-spec",
        type=Path,
        default=Path("data_registry/adapters/idrid-ma-se-s4-v1.json"),
    )
    freeze.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data_registry/adapters/idrid-ma-se-s4-v1.summary.json"),
    )
    catalog = commands.add_parser("catalog", help="build the frozen MA/SE catalog")
    catalog.add_argument("--root", type=Path, required=True)
    catalog.add_argument("--config", type=Path, required=True)
    catalog.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = _project_root()
    try:
        if arguments.command == "freeze-spec":
            summary = freeze_idrid_ma_se_spec(
                project_root=project_root,
                dataset_root=arguments.root,
                upstream_spec_path=_inside_project(
                    project_root, arguments.upstream_spec, "upstream spec"
                ),
                output_spec_path=arguments.output_spec,
                output_summary_path=arguments.output_summary,
            )
            print("S4 MA/SE spec status: PASS")
            print(f"selected records: {summary['selected_complete_label_entries']}")
            print(f"excluded unknown records: {summary['excluded_unknown_entries']}")
            print(f"locked-test root path checks: {summary['locked_test_root_path_checks']}")
            return 0
        config_path = _inside_project(project_root, arguments.config, "config")
        output_dir = _inside_project(project_root, arguments.output_dir, "output directory")
        result = run_s2_catalog(
            project_root=project_root,
            config_path=config_path.resolve(strict=True),
            dataset_root=arguments.root.absolute(),
            output_dir=output_dir,
        )
    except S2GateError as error:
        print(f"S4 MA/SE status: BLOCKED\nreason: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, OSError, S2DataError, TypeError, ValueError) as error:
        print(f"S4 MA/SE status: FAIL_CLOSED\nreason: {error}", file=sys.stderr)
        return 1
    print(f"S4 MA/SE status: {result.status}")
    print(f"development sources: {result.source_count}")
    print(f"development masks: {result.mask_count}")
    print(f"components: {result.component_count}")
    print(f"locked-test root reads: {result.locked_test_root_reads}")
    print(f"artifact manifest hash: {result.artifact_hashes['artifact-manifest.json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "freeze_idrid_ma_se_spec", "main"]
