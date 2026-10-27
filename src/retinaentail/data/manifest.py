"""Content-addressed image manifest indexing and split-isolation checks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from retinaentail.schemas import ImageRecord

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def _require_path(value: object, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    return value


def _require_image_record(value: object) -> ImageRecord:
    if not isinstance(value, ImageRecord):
        raise TypeError("records must contain only ImageRecord instances")
    return value


def sha256_file(path: Path, *, chunk_size: int = DEFAULT_HASH_CHUNK_SIZE) -> str:
    """Compute SHA-256 incrementally without loading the file into memory."""

    valid_path = _require_path(path, "path")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not valid_path.is_file():
        raise FileNotFoundError(f"not a regular file: {valid_path}")

    digest = hashlib.sha256()
    with valid_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _path_under_root(path: Path, root: Path, *, must_exist: bool) -> tuple[Path, str]:
    valid_path = _require_path(path, "path")
    valid_root = _require_path(root, "root")
    resolved_root = valid_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"manifest root is not a directory: {root}")
    candidate = valid_path if valid_path.is_absolute() else resolved_root / valid_path
    resolved_candidate = candidate.resolve(strict=must_exist)
    try:
        relative = resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path is outside manifest root: {path}") from error
    return resolved_candidate, relative.as_posix()


@dataclass(frozen=True, slots=True)
class ManifestIndexer:
    """Build strict ``ImageRecord`` rows relative to one local dataset root."""

    root: Path
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        resolved = _require_path(self.root, "root").resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"manifest root is not a directory: {self.root}")
        object.__setattr__(self, "root", resolved)
        if self.dataset_id is not None and not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string or None")

    def index(
        self,
        image_path: Path,
        *,
        source_image_id: str,
        patient_id: str | None = None,
        label_path: Path | None = None,
        split: str | None = None,
        intervention_pair_id: str | None = None,
    ) -> ImageRecord:
        """Inspect an image and return its reproducible, relative manifest row."""

        resolved_image, relative_image = _path_under_root(image_path, self.root, must_exist=True)
        if not resolved_image.is_file():
            raise FileNotFoundError(f"not a regular image file: {image_path}")
        with Image.open(resolved_image) as image:
            width, height = image.size
            channels = len(image.getbands())

        relative_label: str | None = None
        if label_path is not None:
            resolved_label, relative_label = _path_under_root(
                label_path, self.root, must_exist=True
            )
            if not resolved_label.is_file():
                raise FileNotFoundError(f"not a regular label file: {label_path}")

        return ImageRecord(
            dataset_id=self.dataset_id,
            relative_path=relative_image,
            sha256=sha256_file(resolved_image),
            width=width,
            height=height,
            channels=channels,
            source_image_id=source_image_id,
            patient_id=patient_id,
            label_path=relative_label,
            split=split,
            intervention_pair_id=intervention_pair_id,
        )


def index_image_record(
    image_path: Path,
    *,
    dataset_root: Path,
    source_image_id: str,
    patient_id: str | None = None,
    label_path: Path | None = None,
    split: str | None = None,
    intervention_pair_id: str | None = None,
    dataset_id: str | None = None,
) -> ImageRecord:
    """Convenience wrapper for indexing one image without retaining an indexer."""

    return ManifestIndexer(dataset_root, dataset_id).index(
        image_path,
        source_image_id=source_image_id,
        patient_id=patient_id,
        label_path=label_path,
        split=split,
        intervention_pair_id=intervention_pair_id,
    )


@dataclass(frozen=True, slots=True, order=True)
class SplitLeakage:
    """One isolation identifier observed in more than one data split."""

    identifier_type: str
    identifier: str
    splits: tuple[str, ...]


class SplitLeakageError(ValueError):
    """Raised when a manifest violates source, patient, or twin-pair isolation."""

    def __init__(self, leakages: tuple[SplitLeakage, ...]) -> None:
        self.leakages = leakages
        summary = "; ".join(
            f"{leak.identifier_type}={leak.identifier!r} across {','.join(leak.splits)}"
            for leak in leakages
        )
        super().__init__(f"split leakage detected: {summary}")


def detect_split_leakage(records: Iterable[ImageRecord]) -> tuple[SplitLeakage, ...]:
    """Return all source-image, patient, and intervention-pair split leaks."""

    assignments: dict[tuple[str, str], set[str]] = {}
    for candidate in records:
        record = _require_image_record(candidate)
        if record.split is None:
            continue
        identifiers = (
            ("source_image_id", record.source_image_id),
            ("patient_id", record.patient_id),
            ("intervention_pair_id", record.intervention_pair_id),
        )
        for identifier_type, identifier in identifiers:
            if identifier is not None:
                assignments.setdefault((identifier_type, identifier), set()).add(record.split)

    leakages = (
        SplitLeakage(identifier_type, identifier, tuple(sorted(splits)))
        for (identifier_type, identifier), splits in assignments.items()
        if len(splits) > 1
    )
    return tuple(sorted(leakages))


def check_split_leakage(records: Iterable[ImageRecord]) -> tuple[SplitLeakage, ...]:
    """Alias retaining a direct, report-friendly checker name."""

    return detect_split_leakage(records)


def assert_no_split_leakage(records: Iterable[ImageRecord]) -> None:
    """Raise ``SplitLeakageError`` when any protected identifier crosses splits."""

    leakages = detect_split_leakage(records)
    if leakages:
        raise SplitLeakageError(leakages)


__all__ = [
    "DEFAULT_HASH_CHUNK_SIZE",
    "ManifestIndexer",
    "SplitLeakage",
    "SplitLeakageError",
    "assert_no_split_leakage",
    "check_split_leakage",
    "detect_split_leakage",
    "index_image_record",
    "sha256_file",
]
