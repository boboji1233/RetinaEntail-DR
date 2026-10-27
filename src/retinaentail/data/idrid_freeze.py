"""Freeze the published IDRiD HE/EX segmentation cohort into an explicit spec.

This is an acquisition-preparation utility, not an implicit dataset adapter.  It
accepts one explicit root, recognizes only the documented IDRiD segmentation
layout, and never infers patient, eye, follow-up, or intervention identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from retinaentail.data.stage2_models import (
    DatasetAdapterSpec,
    DatasetEntrySpec,
    LabelInputSpec,
    LabelSemantic,
)
from retinaentail.witnesses.replay import write_bytes_fail_safe

_PUBLISHED_KEY = re.compile(r"^IDRiD_(\d{2})$")
_TRAIN_IMAGE_DIR = Path("A. Segmentation/1. Original Images/a. Training Set")
_TEST_IMAGE_DIR = Path("A. Segmentation/1. Original Images/b. Testing Set")
_TRAIN_MASK_ROOT = Path("A. Segmentation/2. All Segmentation Groundtruths/a. Training Set")
_TEST_MASK_ROOT = Path("A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set")
_HE_DIR = Path("2. Haemorrhages")
_EX_DIR = Path("3. Hard Exudates")


@dataclass(frozen=True, slots=True)
class IDRiDFreezeSummary:
    """Root-free summary of one deterministic cohort freeze."""

    spec: DatasetAdapterSpec
    published_image_count: int
    included_count: int
    excluded_source_image_ids: tuple[str, ...]
    split_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "published_image_count": self.published_image_count,
            "included_count": self.included_count,
            "excluded_source_image_ids": list(self.excluded_source_image_ids),
            "split_counts": dict(self.split_counts),
            "missing_label_policy": "UNKNOWN_NEVER_NEGATIVE",
            "label_semantics": [
                LabelSemantic.HARD_EXUDATE_MASK.value,
                LabelSemantic.HEMORRHAGE_MASK.value,
            ],
        }


def _source_image_id(stem: str) -> str:
    match = _PUBLISHED_KEY.fullmatch(stem)
    if match is None:
        raise ValueError(f"unexpected IDRiD published image key: {stem!r}")
    return f"idrid-segmentation-{stem}"


def _partition_official_train(source_image_id: str) -> str:
    digest = hashlib.sha256(source_image_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 5
    return "CALIBRATION" if bucket == 0 else "TRAIN"


def _expected_images(root: Path, relative_dir: Path, start: int, stop: int) -> tuple[Path, ...]:
    directory = root / relative_dir
    if not directory.is_dir():
        raise FileNotFoundError(f"expected IDRiD directory is missing: {relative_dir.as_posix()}")
    observed = tuple(sorted(directory.glob("*.jpg"), key=lambda item: item.name))
    expected_names = {f"IDRiD_{index:02d}.jpg" for index in range(start, stop + 1)}
    observed_names = {path.name for path in observed}
    if observed_names != expected_names:
        raise ValueError(f"published IDRiD image inventory mismatch in {relative_dir.as_posix()}")
    return observed


def freeze_idrid_he_ex_spec(root: Path) -> IDRiDFreezeSummary:
    """Build the complete-label HE/EX task spec for the documented 81-image release."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(root)
    inventory = (
        (_expected_images(resolved_root, _TRAIN_IMAGE_DIR, 1, 54), _TRAIN_MASK_ROOT, False),
        (_expected_images(resolved_root, _TEST_IMAGE_DIR, 55, 81), _TEST_MASK_ROOT, True),
    )
    entries: list[DatasetEntrySpec] = []
    excluded: list[str] = []
    for images, mask_root, locked_test in inventory:
        for image_path in images:
            source_image_id = _source_image_id(image_path.stem)
            he_path = resolved_root / mask_root / _HE_DIR / f"{image_path.stem}_HE.tif"
            ex_path = resolved_root / mask_root / _EX_DIR / f"{image_path.stem}_EX.tif"
            if not he_path.is_file() or not ex_path.is_file():
                excluded.append(source_image_id)
                continue
            split = "LOCKED_TEST" if locked_test else _partition_official_train(source_image_id)
            entries.append(
                DatasetEntrySpec(
                    image_path=image_path.relative_to(resolved_root),
                    source_image_id=source_image_id,
                    split=split,
                    patient_id=None,
                    eye_id=None,
                    followup_id=None,
                    intervention_pair_id=None,
                    labels=(
                        LabelInputSpec(
                            semantic=LabelSemantic.HEMORRHAGE_MASK,
                            relative_path=he_path.relative_to(resolved_root),
                        ),
                        LabelInputSpec(
                            semantic=LabelSemantic.HARD_EXUDATE_MASK,
                            relative_path=ex_path.relative_to(resolved_root),
                        ),
                    ),
                )
            )
    spec = DatasetAdapterSpec(
        dataset_id="idrid",
        label_semantics=(
            LabelSemantic.HEMORRHAGE_MASK,
            LabelSemantic.HARD_EXUDATE_MASK,
        ),
        entries=tuple(entries),
    )
    counts: dict[str, int] = {}
    for entry in spec.entries:
        counts[entry.split] = counts.get(entry.split, 0) + 1
    return IDRiDFreezeSummary(
        spec=spec,
        published_image_count=81,
        included_count=len(spec.entries),
        excluded_source_image_ids=tuple(sorted(excluded)),
        split_counts=tuple(sorted(counts.items())),
    )


def write_idrid_he_ex_spec(summary: IDRiDFreezeSummary, output_path: Path) -> None:
    """Write a readable deterministic spec without overwriting a different file."""

    payload = (
        json.dumps(
            summary.spec.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_fail_safe(output_path, payload)


def write_idrid_freeze_summary(summary: IDRiDFreezeSummary, output_path: Path) -> None:
    """Write the root-free inclusion and split summary."""

    payload = (
        json.dumps(
            summary.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_fail_safe(output_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    arguments = parser.parse_args()
    summary = freeze_idrid_he_ex_spec(arguments.root)
    write_idrid_he_ex_spec(summary, arguments.output)
    if arguments.summary_output is not None:
        write_idrid_freeze_summary(summary, arguments.summary_output)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IDRiDFreezeSummary",
    "freeze_idrid_he_ex_spec",
    "write_idrid_freeze_summary",
    "write_idrid_he_ex_spec",
]
