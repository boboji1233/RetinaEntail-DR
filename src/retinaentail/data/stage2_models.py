"""Versioned, read-only Stage 2 dataset specifications and manifest contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Final, TypeAlias

STAGE2_SCHEMA_VERSION: Final = "2.0.0"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class Stage2ContractError(ValueError):
    """Raised when an explicit Stage 2 data contract is incomplete or unsafe."""


class LabelAvailability(StrEnum):
    """Whether a label artifact was explicitly supplied and verified as a file."""

    AVAILABLE = "AVAILABLE"
    UNKNOWN = "UNKNOWN"


class LabelSemantic(StrEnum):
    """Controlled label meanings supported by the first Stage 2 adapters."""

    DR_GRADE = "DR_GRADE"
    DME_GRADE = "DME_GRADE"
    MICROANEURYSM_MASK = "MICROANEURYSM_MASK"
    HEMORRHAGE_MASK = "HEMORRHAGE_MASK"
    HARD_EXUDATE_MASK = "HARD_EXUDATE_MASK"
    SOFT_EXUDATE_MASK = "SOFT_EXUDATE_MASK"
    OPTIC_DISC_LOCALIZATION = "OPTIC_DISC_LOCALIZATION"
    FOVEA_LOCALIZATION = "FOVEA_LOCALIZATION"
    LESION_DETECTION = "LESION_DETECTION"
    IMAGE_QUALITY_GRADE = "IMAGE_QUALITY_GRADE"
    ARTIFACT_ATTRIBUTES = "ARTIFACT_ATTRIBUTES"
    CLARITY_ATTRIBUTE = "CLARITY_ATTRIBUTE"
    FIELD_DEFINITION_ATTRIBUTE = "FIELD_DEFINITION_ATTRIBUTE"
    PAIRED_VIEW_METADATA = "PAIRED_VIEW_METADATA"


DATASET_LABEL_SEMANTICS: Final[Mapping[str, frozenset[LabelSemantic]]] = MappingProxyType(
    {
        "idrid": frozenset(
            {
                LabelSemantic.DR_GRADE,
                LabelSemantic.DME_GRADE,
                LabelSemantic.MICROANEURYSM_MASK,
                LabelSemantic.HEMORRHAGE_MASK,
                LabelSemantic.HARD_EXUDATE_MASK,
                LabelSemantic.SOFT_EXUDATE_MASK,
                LabelSemantic.OPTIC_DISC_LOCALIZATION,
                LabelSemantic.FOVEA_LOCALIZATION,
            }
        ),
        "ddr": frozenset(
            {
                LabelSemantic.DR_GRADE,
                LabelSemantic.LESION_DETECTION,
                LabelSemantic.MICROANEURYSM_MASK,
                LabelSemantic.HEMORRHAGE_MASK,
                LabelSemantic.HARD_EXUDATE_MASK,
                LabelSemantic.SOFT_EXUDATE_MASK,
            }
        ),
        "deepdrid": frozenset(
            {
                LabelSemantic.DR_GRADE,
                LabelSemantic.IMAGE_QUALITY_GRADE,
                LabelSemantic.ARTIFACT_ATTRIBUTES,
                LabelSemantic.CLARITY_ATTRIBUTE,
                LabelSemantic.FIELD_DEFINITION_ATTRIBUTE,
                LabelSemantic.PAIRED_VIEW_METADATA,
            }
        ),
    }
)


def _mapping(
    value: object,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Stage2ContractError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise Stage2ContractError(f"{name} keys must be strings")
        result[key] = item
    allowed = required | (set() if optional is None else optional)
    missing = sorted(required.difference(result))
    unknown = sorted(set(result).difference(allowed))
    if missing:
        raise Stage2ContractError(f"{name} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise Stage2ContractError(f"{name} has unknown field(s): {', '.join(unknown)}")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage2ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_nonempty(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name)


def _schema_version(value: object) -> str:
    version = _nonempty(value, "schema_version")
    if version != STAGE2_SCHEMA_VERSION:
        raise Stage2ContractError(
            f"unsupported schema_version {version!r}; expected {STAGE2_SCHEMA_VERSION!r}"
        )
    return version


def _dataset_id(value: object) -> str:
    result = _nonempty(value, "dataset_id")
    if _DATASET_ID_PATTERN.fullmatch(result) is None:
        raise Stage2ContractError("dataset_id must match ^[a-z][a-z0-9_]*$")
    return result


def _dataset_version(value: object) -> str:
    result = _nonempty(value, "dataset_version")
    if _DATASET_VERSION_PATTERN.fullmatch(result) is None:
        raise Stage2ContractError("dataset_version must be a report-safe release identifier")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Stage2ContractError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise Stage2ContractError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _normalized_relative_string(value: object, name: str) -> str:
    if isinstance(value, Path):
        raw = value.as_posix()
    elif isinstance(value, str):
        raw = value
    else:
        raise Stage2ContractError(f"{name} must be a pathlib.Path or string")
    raw = _nonempty(raw, name)
    if "\\" in raw:
        raise Stage2ContractError(f"{name} must use normalized POSIX separators")
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(raw)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise Stage2ContractError(f"{name} must be relative")
    if (
        not posix_path.parts
        or raw != posix_path.as_posix()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise Stage2ContractError(f"{name} must be a normalized relative path")
    return posix_path.as_posix()


def _relative_path(value: object, name: str) -> Path:
    return Path(_normalized_relative_string(value, name))


def _label_semantic(value: object) -> LabelSemantic:
    if not isinstance(value, str):
        raise Stage2ContractError(f"unknown label semantic: {value!r}")
    try:
        return value if isinstance(value, LabelSemantic) else LabelSemantic(value)
    except (TypeError, ValueError) as error:
        raise Stage2ContractError(f"unknown label semantic: {value!r}") from error


def _availability(value: object) -> LabelAvailability:
    if not isinstance(value, str):
        raise Stage2ContractError(f"unknown label availability: {value!r}")
    try:
        return value if isinstance(value, LabelAvailability) else LabelAvailability(value)
    except (TypeError, ValueError) as error:
        raise Stage2ContractError(f"unknown label availability: {value!r}") from error


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelInputSpec:
    """Explicit label location; ``None`` means unknown, never a negative label."""

    semantic: LabelSemantic
    relative_path: Path | None = None
    schema_version: str = STAGE2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "semantic", _label_semantic(self.semantic))
        if self.relative_path is not None:
            object.__setattr__(
                self,
                "relative_path",
                _relative_path(self.relative_path, "label relative_path"),
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "semantic": self.semantic.value,
            "relative_path": (
                None if self.relative_path is None else self.relative_path.as_posix()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LabelInputSpec:
        data = _mapping(
            value,
            name="LabelInputSpec",
            required={"schema_version", "semantic", "relative_path"},
        )
        raw_path = data["relative_path"]
        return cls(
            schema_version=_schema_version(data["schema_version"]),
            semantic=_label_semantic(data["semantic"]),
            relative_path=(
                None if raw_path is None else _relative_path(raw_path, "label relative_path")
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetEntrySpec:
    """One explicit image entry; identity fields are never inferred from its path."""

    image_path: Path
    source_image_id: str
    split: str
    patient_id: str | None = None
    eye_id: str | None = None
    followup_id: str | None = None
    intervention_pair_id: str | None = None
    labels: tuple[LabelInputSpec, ...] = field(default_factory=tuple)
    schema_version: str = STAGE2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "image_path", _relative_path(self.image_path, "image_path"))
        object.__setattr__(
            self, "source_image_id", _nonempty(self.source_image_id, "source_image_id")
        )
        object.__setattr__(self, "split", _nonempty(self.split, "split"))
        for field_name in ("patient_id", "eye_id", "followup_id", "intervention_pair_id"):
            object.__setattr__(
                self,
                field_name,
                _optional_nonempty(getattr(self, field_name), field_name),
            )
        normalized_labels: list[LabelInputSpec] = []
        semantics: set[LabelSemantic] = set()
        for label in self.labels:
            if not isinstance(label, LabelInputSpec):
                raise Stage2ContractError("labels must contain LabelInputSpec values")
            if label.semantic in semantics:
                raise Stage2ContractError(f"duplicate label semantic: {label.semantic.value}")
            semantics.add(label.semantic)
            normalized_labels.append(label)
        object.__setattr__(self, "labels", tuple(normalized_labels))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "image_path": self.image_path.as_posix(),
            "source_image_id": self.source_image_id,
            "split": self.split,
            "patient_id": self.patient_id,
            "eye_id": self.eye_id,
            "followup_id": self.followup_id,
            "intervention_pair_id": self.intervention_pair_id,
            "labels": [label.to_dict() for label in self.labels],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetEntrySpec:
        data = _mapping(
            value,
            name="DatasetEntrySpec",
            required={
                "schema_version",
                "image_path",
                "source_image_id",
                "split",
                "patient_id",
                "eye_id",
                "followup_id",
                "intervention_pair_id",
                "labels",
            },
        )
        raw_labels = data["labels"]
        if isinstance(raw_labels, str | bytes) or not isinstance(raw_labels, Sequence):
            raise Stage2ContractError("labels must be a sequence")
        labels: list[LabelInputSpec] = []
        for index, raw_label in enumerate(raw_labels):
            label_data = _mapping(
                raw_label,
                name=f"labels[{index}]",
                required={"schema_version", "semantic", "relative_path"},
            )
            labels.append(LabelInputSpec.from_dict(label_data))
        return cls(
            schema_version=_schema_version(data["schema_version"]),
            image_path=_relative_path(data["image_path"], "image_path"),
            source_image_id=_nonempty(data["source_image_id"], "source_image_id"),
            split=_nonempty(data["split"], "split"),
            patient_id=_optional_nonempty(data["patient_id"], "patient_id"),
            eye_id=_optional_nonempty(data["eye_id"], "eye_id"),
            followup_id=_optional_nonempty(data["followup_id"], "followup_id"),
            intervention_pair_id=_optional_nonempty(
                data["intervention_pair_id"], "intervention_pair_id"
            ),
            labels=tuple(labels),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetAdapterSpec:
    """Explicit ordered entries for one dataset adapter invocation."""

    dataset_id: str
    entries: tuple[DatasetEntrySpec, ...]
    label_semantics: tuple[LabelSemantic, ...] | None = None
    schema_version: str = STAGE2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        dataset_id = _dataset_id(self.dataset_id)
        if dataset_id not in DATASET_LABEL_SEMANTICS:
            raise Stage2ContractError("dataset_id has no registered Stage 2 adapter")
        object.__setattr__(self, "dataset_id", dataset_id)
        if self.label_semantics is not None:
            if not isinstance(self.label_semantics, tuple) or not self.label_semantics:
                raise Stage2ContractError(
                    "label_semantics must be a non-empty tuple when explicitly scoped"
                )
            normalized_semantics = tuple(
                sorted(
                    {_label_semantic(value) for value in self.label_semantics},
                    key=lambda item: item.value,
                )
            )
            if len(normalized_semantics) != len(self.label_semantics):
                raise Stage2ContractError("label_semantics must not contain duplicates")
            unsupported = set(normalized_semantics).difference(DATASET_LABEL_SEMANTICS[dataset_id])
            if unsupported:
                names = ", ".join(sorted(item.value for item in unsupported))
                raise Stage2ContractError(
                    f"label_semantics contains unsupported values for {dataset_id}: {names}"
                )
            object.__setattr__(self, "label_semantics", normalized_semantics)
        normalized: list[DatasetEntrySpec] = []
        image_paths: set[Path] = set()
        for entry in self.entries:
            if not isinstance(entry, DatasetEntrySpec):
                raise Stage2ContractError("entries must contain DatasetEntrySpec values")
            if entry.image_path in image_paths:
                raise Stage2ContractError(f"duplicate image_path: {entry.image_path.as_posix()}")
            image_paths.add(entry.image_path)
            normalized.append(entry)
        if not normalized:
            raise Stage2ContractError("entries must contain at least one explicit entry")
        if self.label_semantics is not None:
            scoped = set(self.label_semantics)
            declared = {label.semantic for entry in normalized for label in entry.labels}
            outside_scope = declared.difference(scoped)
            if outside_scope:
                names = ", ".join(sorted(item.value for item in outside_scope))
                raise Stage2ContractError(
                    f"entry labels fall outside explicit label_semantics: {names}"
                )
        object.__setattr__(self, "entries", tuple(normalized))

    @property
    def effective_label_semantics(self) -> frozenset[LabelSemantic]:
        """Return the explicit task catalog or the adapter's legacy full catalog."""

        if self.label_semantics is None:
            return DATASET_LABEL_SEMANTICS[self.dataset_id]
        return frozenset(self.label_semantics)

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if self.label_semantics is not None:
            payload["label_semantics"] = [item.value for item in self.label_semantics]
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetAdapterSpec:
        data = _mapping(
            value,
            name="DatasetAdapterSpec",
            required={"schema_version", "dataset_id", "entries"},
            optional={"label_semantics"},
        )
        raw_entries = data["entries"]
        if isinstance(raw_entries, str | bytes) or not isinstance(raw_entries, Sequence):
            raise Stage2ContractError("entries must be a sequence")
        entries: list[DatasetEntrySpec] = []
        for index, raw_entry in enumerate(raw_entries):
            entry_data = _mapping(
                raw_entry,
                name=f"entries[{index}]",
                required={
                    "schema_version",
                    "image_path",
                    "source_image_id",
                    "split",
                    "patient_id",
                    "eye_id",
                    "followup_id",
                    "intervention_pair_id",
                    "labels",
                },
            )
            entries.append(DatasetEntrySpec.from_dict(entry_data))
        raw_semantics = data.get("label_semantics")
        label_semantics: tuple[LabelSemantic, ...] | None = None
        if raw_semantics is not None:
            if isinstance(raw_semantics, str | bytes) or not isinstance(raw_semantics, Sequence):
                raise Stage2ContractError("label_semantics must be a sequence")
            label_semantics = tuple(_label_semantic(value) for value in raw_semantics)
        return cls(
            schema_version=_schema_version(data["schema_version"]),
            dataset_id=_dataset_id(data["dataset_id"]),
            entries=tuple(entries),
            label_semantics=label_semantics,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelRecordV2:
    """A controlled label reference; missing labels are always ``UNKNOWN``."""

    semantic: LabelSemantic
    availability: LabelAvailability
    relative_path: str | None = None
    sha256: str | None = None
    byte_size: int | None = None
    schema_version: str = STAGE2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "semantic", _label_semantic(self.semantic))
        availability = _availability(self.availability)
        object.__setattr__(self, "availability", availability)
        relative_path = (
            None
            if self.relative_path is None
            else _normalized_relative_string(self.relative_path, "label relative_path")
        )
        object.__setattr__(self, "relative_path", relative_path)
        digest = _optional_sha256(self.sha256, "label sha256")
        object.__setattr__(self, "sha256", digest)
        if self.byte_size is not None:
            object.__setattr__(self, "byte_size", _positive_int(self.byte_size, "label byte_size"))

        if relative_path is None:
            if availability is not LabelAvailability.UNKNOWN:
                raise Stage2ContractError("a missing label path must have UNKNOWN availability")
            if digest is not None or self.byte_size is not None:
                raise Stage2ContractError("a missing label cannot have a hash or byte size")
        elif availability is LabelAvailability.AVAILABLE:
            if digest is None or self.byte_size is None:
                raise Stage2ContractError("an AVAILABLE label requires path, hash, and byte size")
        elif digest is not None or self.byte_size is not None:
            raise Stage2ContractError("an UNKNOWN label cannot claim a verified hash or byte size")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "semantic": self.semantic.value,
            "availability": self.availability.value,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LabelRecordV2:
        data = _mapping(
            value,
            name="LabelRecordV2",
            required={
                "schema_version",
                "semantic",
                "availability",
                "relative_path",
                "sha256",
                "byte_size",
            },
        )
        raw_size = data["byte_size"]
        return cls(
            schema_version=_schema_version(data["schema_version"]),
            semantic=_label_semantic(data["semantic"]),
            availability=_availability(data["availability"]),
            relative_path=(
                None
                if data["relative_path"] is None
                else _normalized_relative_string(data["relative_path"], "label relative_path")
            ),
            sha256=_optional_sha256(data["sha256"], "label sha256"),
            byte_size=(None if raw_size is None else _positive_int(raw_size, "label byte_size")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestRecordV2:
    """Content-addressed image metadata with explicit identities and multi-label state."""

    dataset_id: str
    relative_path: str
    sha256: str
    byte_size: int
    width: int
    height: int
    channels: int
    image_format: str
    source_image_id: str
    split: str
    labels: tuple[LabelRecordV2, ...]
    patient_id: str | None = None
    eye_id: str | None = None
    followup_id: str | None = None
    intervention_pair_id: str | None = None
    schema_version: str = STAGE2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "dataset_id", _dataset_id(self.dataset_id))
        object.__setattr__(
            self,
            "relative_path",
            _normalized_relative_string(self.relative_path, "relative_path"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        for field_name in ("byte_size", "width", "height", "channels"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self, "image_format", _nonempty(self.image_format, "image_format").upper()
        )
        object.__setattr__(
            self, "source_image_id", _nonempty(self.source_image_id, "source_image_id")
        )
        object.__setattr__(self, "split", _nonempty(self.split, "split"))
        for field_name in ("patient_id", "eye_id", "followup_id", "intervention_pair_id"):
            object.__setattr__(
                self,
                field_name,
                _optional_nonempty(getattr(self, field_name), field_name),
            )
        normalized_labels: list[LabelRecordV2] = []
        semantics: set[LabelSemantic] = set()
        for label in self.labels:
            if not isinstance(label, LabelRecordV2):
                raise Stage2ContractError("labels must contain LabelRecordV2 values")
            if label.semantic in semantics:
                raise Stage2ContractError(f"duplicate label semantic: {label.semantic.value}")
            semantics.add(label.semantic)
            normalized_labels.append(label)
        if not normalized_labels:
            raise Stage2ContractError("labels must declare at least one controlled semantic")
        object.__setattr__(self, "labels", tuple(normalized_labels))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "image_format": self.image_format,
            "source_image_id": self.source_image_id,
            "patient_id": self.patient_id,
            "eye_id": self.eye_id,
            "followup_id": self.followup_id,
            "intervention_pair_id": self.intervention_pair_id,
            "split": self.split,
            "labels": [label.to_dict() for label in self.labels],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ManifestRecordV2:
        required = {
            "schema_version",
            "dataset_id",
            "relative_path",
            "sha256",
            "byte_size",
            "width",
            "height",
            "channels",
            "image_format",
            "source_image_id",
            "patient_id",
            "eye_id",
            "followup_id",
            "intervention_pair_id",
            "split",
            "labels",
        }
        data = _mapping(value, name="ManifestRecordV2", required=required)
        raw_labels = data["labels"]
        if isinstance(raw_labels, str | bytes) or not isinstance(raw_labels, Sequence):
            raise Stage2ContractError("labels must be a sequence")
        labels: list[LabelRecordV2] = []
        for index, raw_label in enumerate(raw_labels):
            label_data = _mapping(
                raw_label,
                name=f"labels[{index}]",
                required={
                    "schema_version",
                    "semantic",
                    "availability",
                    "relative_path",
                    "sha256",
                    "byte_size",
                },
            )
            labels.append(LabelRecordV2.from_dict(label_data))
        return cls(
            schema_version=_schema_version(data["schema_version"]),
            dataset_id=_dataset_id(data["dataset_id"]),
            relative_path=_normalized_relative_string(data["relative_path"], "relative_path"),
            sha256=_sha256(data["sha256"], "sha256"),
            byte_size=_positive_int(data["byte_size"], "byte_size"),
            width=_positive_int(data["width"], "width"),
            height=_positive_int(data["height"], "height"),
            channels=_positive_int(data["channels"], "channels"),
            image_format=_nonempty(data["image_format"], "image_format"),
            source_image_id=_nonempty(data["source_image_id"], "source_image_id"),
            patient_id=_optional_nonempty(data["patient_id"], "patient_id"),
            eye_id=_optional_nonempty(data["eye_id"], "eye_id"),
            followup_id=_optional_nonempty(data["followup_id"], "followup_id"),
            intervention_pair_id=_optional_nonempty(
                data["intervention_pair_id"], "intervention_pair_id"
            ),
            split=_nonempty(data["split"], "split"),
            labels=tuple(labels),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetManifestV2:
    """Deterministic versioned envelope for records from one exact dataset release."""

    dataset_id: str
    dataset_version: str
    records: tuple[ManifestRecordV2, ...]
    schema_version: str = STAGE2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        dataset_id = _dataset_id(self.dataset_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "dataset_version", _dataset_version(self.dataset_version))
        normalized: list[ManifestRecordV2] = []
        relative_paths: set[str] = set()
        for record in self.records:
            if not isinstance(record, ManifestRecordV2):
                raise Stage2ContractError("records must contain ManifestRecordV2 values")
            if record.dataset_id != dataset_id:
                raise Stage2ContractError("every record dataset_id must match the manifest")
            if record.relative_path in relative_paths:
                raise Stage2ContractError(f"duplicate image relative_path: {record.relative_path}")
            relative_paths.add(record.relative_path)
            normalized.append(record)
        if not normalized:
            raise Stage2ContractError("records must contain at least one manifest record")
        object.__setattr__(self, "records", tuple(normalized))

    def __iter__(self) -> Iterator[ManifestRecordV2]:
        return iter(self.records)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetManifestV2:
        data = _mapping(
            value,
            name="DatasetManifestV2",
            required={"schema_version", "dataset_id", "dataset_version", "records"},
        )
        raw_records = data["records"]
        if isinstance(raw_records, str | bytes) or not isinstance(raw_records, Sequence):
            raise Stage2ContractError("records must be a sequence")
        records: list[ManifestRecordV2] = []
        for index, raw_record in enumerate(raw_records):
            record_data = _mapping(
                raw_record,
                name=f"records[{index}]",
                required={
                    "schema_version",
                    "dataset_id",
                    "relative_path",
                    "sha256",
                    "byte_size",
                    "width",
                    "height",
                    "channels",
                    "image_format",
                    "source_image_id",
                    "patient_id",
                    "eye_id",
                    "followup_id",
                    "intervention_pair_id",
                    "split",
                    "labels",
                },
            )
            records.append(ManifestRecordV2.from_dict(record_data))
        return cls(
            schema_version=_schema_version(data["schema_version"]),
            dataset_id=_dataset_id(data["dataset_id"]),
            dataset_version=_dataset_version(data["dataset_version"]),
            records=tuple(records),
        )


def load_dataset_adapter_spec(path: Path) -> DatasetAdapterSpec:
    """Load one strict JSON adapter spec; callers should assess licensing first."""

    if not isinstance(path, Path):
        raise TypeError("spec path must be a pathlib.Path")
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    data = _mapping(
        decoded,
        name="DatasetAdapterSpec",
        required={"schema_version", "dataset_id", "entries"},
        optional={"label_semantics"},
    )
    return DatasetAdapterSpec.from_dict(data)


__all__ = [
    "DATASET_LABEL_SEMANTICS",
    "STAGE2_SCHEMA_VERSION",
    "DatasetAdapterSpec",
    "DatasetEntrySpec",
    "DatasetManifestV2",
    "JSONValue",
    "LabelAvailability",
    "LabelInputSpec",
    "LabelRecordV2",
    "LabelSemantic",
    "ManifestRecordV2",
    "Stage2ContractError",
    "load_dataset_adapter_spec",
]
