"""Strict, path-free Stage 5 identity-isolation audit.

The audit consumes already-built manifest records and never opens a dataset root.  A
missing nullable identity remains unknown and therefore prevents an isolation proof.
Identity values are not serialized in findings; deterministic fingerprints and record
indices allow local investigation without copying patient identifiers into reports.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Final, TypeAlias, TypeVar

from retinaentail.data.stage2_models import ManifestRecordV2

STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION: Final = "5.0.0"

_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class Stage5IdentityContractError(ValueError):
    """Raised when a Stage 5 identity input or serialized audit is malformed."""


class Stage5Split(StrEnum):
    """The only splits admitted to a locked Stage 5 pilot."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class Stage5IdentityType(StrEnum):
    """The five protected identity namespaces."""

    SOURCE_IMAGE_ID = "source_image_id"
    PATIENT_ID = "patient_id"
    EYE_ID = "eye_id"
    FOLLOWUP_ID = "followup_id"
    INTERVENTION_PAIR_ID = "intervention_pair_id"


class Stage5IdentityAuditStatus(StrEnum):
    """Whether all five identity namespaces are proven split-isolated."""

    PASS = "PASS"
    FAIL_CLOSED = "FAIL_CLOSED"


class Stage5IdentityAuditReason(StrEnum):
    """Controlled reasons why split isolation could not be proven."""

    EMPTY_RECORD_SET = "EMPTY_RECORD_SET"
    INVALID_RECORD_TYPE = "INVALID_RECORD_TYPE"
    INVALID_RECORD_CONTRACT = "INVALID_RECORD_CONTRACT"
    MULTIPLE_DATASETS = "MULTIPLE_DATASETS"
    INVALID_SPLIT = "INVALID_SPLIT"
    REQUIRED_SPLIT_EMPTY = "REQUIRED_SPLIT_EMPTY"
    MISSING_PATIENT_ID = "MISSING_PATIENT_ID"
    MISSING_EYE_ID = "MISSING_EYE_ID"
    MISSING_FOLLOWUP_ID = "MISSING_FOLLOWUP_ID"
    MISSING_INTERVENTION_PAIR_ID = "MISSING_INTERVENTION_PAIR_ID"
    EYE_SCOPE_INCOMPLETE = "EYE_SCOPE_INCOMPLETE"
    FOLLOWUP_SCOPE_INCOMPLETE = "FOLLOWUP_SCOPE_INCOMPLETE"
    DUPLICATE_RELATIVE_PATH = "DUPLICATE_RELATIVE_PATH"
    CROSS_SPLIT_IDENTITY_OVERLAP = "CROSS_SPLIT_IDENTITY_OVERLAP"


_SPLIT_ORDER: Final = {split: index for index, split in enumerate(Stage5Split)}
_IDENTITY_ORDER: Final = {
    identity_type: index for index, identity_type in enumerate(Stage5IdentityType)
}
_REASON_ORDER: Final = {reason: index for index, reason in enumerate(Stage5IdentityAuditReason)}
_MISSING_REASONS: Final[Mapping[Stage5IdentityType, Stage5IdentityAuditReason]] = MappingProxyType(
    {
        Stage5IdentityType.PATIENT_ID: Stage5IdentityAuditReason.MISSING_PATIENT_ID,
        Stage5IdentityType.EYE_ID: Stage5IdentityAuditReason.MISSING_EYE_ID,
        Stage5IdentityType.FOLLOWUP_ID: Stage5IdentityAuditReason.MISSING_FOLLOWUP_ID,
        Stage5IdentityType.INTERVENTION_PAIR_ID: (
            Stage5IdentityAuditReason.MISSING_INTERVENTION_PAIR_ID
        ),
    }
)


def _strict_mapping(
    value: object,
    *,
    name: str,
    required: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Stage5IdentityContractError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise Stage5IdentityContractError(f"{name} keys must be strings")
        result[key] = item
    missing = sorted(required.difference(result))
    unknown = sorted(set(result).difference(required))
    if missing:
        raise Stage5IdentityContractError(f"{name} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise Stage5IdentityContractError(f"{name} has unknown field(s): {', '.join(unknown)}")
    return result


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise Stage5IdentityContractError(f"{name} must be an array")
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage5IdentityContractError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, name)


def _dataset_id(value: object) -> str:
    result = _nonempty_text(value, "dataset_id")
    if _DATASET_ID_PATTERN.fullmatch(result) is None:
        raise Stage5IdentityContractError("dataset_id must match ^[a-z][a-z0-9_]*$")
    return result


def _relative_path(value: object) -> str:
    raw = _nonempty_text(value, "relative_path")
    if "\\" in raw:
        raise Stage5IdentityContractError("relative_path must use normalized POSIX separators")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise Stage5IdentityContractError("relative_path must be relative")
    if (
        not posix.parts
        or raw != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise Stage5IdentityContractError("relative_path must be normalized")
    return raw


def _plain_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:  # noqa: E721
        raise Stage5IdentityContractError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Stage5IdentityContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _enum_value(value: object, enum_type: type[StrEnum], name: str) -> StrEnum:
    if not isinstance(value, str):
        raise Stage5IdentityContractError(f"{name} must be a controlled string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise Stage5IdentityContractError(f"unsupported {name}: {value!r}") from error


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5IdentityRecord:
    """Minimal, no-file-I/O projection accepted by the identity audit.

    The class is useful for synthetic fixtures.  Production callers may pass
    :class:`ManifestRecordV2` directly instead of constructing this projection.
    """

    dataset_id: str
    relative_path: str
    split: str
    source_image_id: str
    patient_id: str | None
    eye_id: str | None
    followup_id: str | None
    intervention_pair_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _dataset_id(self.dataset_id))
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(self, "split", _nonempty_text(self.split, "split"))
        object.__setattr__(
            self,
            "source_image_id",
            _nonempty_text(self.source_image_id, "source_image_id"),
        )
        for field_name in (
            "patient_id",
            "eye_id",
            "followup_id",
            "intervention_pair_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name),
            )

    @classmethod
    def from_manifest_record(cls, record: ManifestRecordV2) -> Stage5IdentityRecord:
        if not isinstance(record, ManifestRecordV2):
            raise Stage5IdentityContractError("record must be ManifestRecordV2")
        # Reparse the complete Stage 2 contract in memory.  This catches a forged or
        # subsequently mutated dataclass without touching any referenced file.
        validated = ManifestRecordV2.from_dict(record.to_dict())
        if validated.to_dict() != record.to_dict():
            raise Stage5IdentityContractError("manifest record is not canonically normalized")
        return cls(
            dataset_id=validated.dataset_id,
            relative_path=validated.relative_path,
            split=validated.split,
            source_image_id=validated.source_image_id,
            patient_id=validated.patient_id,
            eye_id=validated.eye_id,
            followup_id=validated.followup_id,
            intervention_pair_id=validated.intervention_pair_id,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "dataset_id": self.dataset_id,
            "relative_path": self.relative_path,
            "split": self.split,
            "source_image_id": self.source_image_id,
            "patient_id": self.patient_id,
            "eye_id": self.eye_id,
            "followup_id": self.followup_id,
            "intervention_pair_id": self.intervention_pair_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5IdentityIssue:
    """One controlled finding; raw identity values are intentionally omitted."""

    reason: Stage5IdentityAuditReason
    identity_type: Stage5IdentityType | None
    subject_fingerprint: str | None
    splits: tuple[Stage5Split, ...]
    record_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reason, Stage5IdentityAuditReason):
            raise Stage5IdentityContractError("issue reason must use Stage5IdentityAuditReason")
        if self.identity_type is not None and not isinstance(
            self.identity_type, Stage5IdentityType
        ):
            raise Stage5IdentityContractError(
                "issue identity_type must be null or Stage5IdentityType"
            )
        if self.subject_fingerprint is not None:
            _sha256(self.subject_fingerprint, "subject_fingerprint")
        if any(not isinstance(split, Stage5Split) for split in self.splits):
            raise Stage5IdentityContractError("issue splits must use Stage5Split")
        normalized_splits = tuple(sorted(set(self.splits), key=_SPLIT_ORDER.__getitem__))
        if self.splits != normalized_splits:
            raise Stage5IdentityContractError("issue splits must be unique and canonically sorted")
        if any(type(index) is not int or index < 0 for index in self.record_indices):  # noqa: E721
            raise Stage5IdentityContractError("issue record_indices must be non-negative integers")
        if self.record_indices != tuple(sorted(set(self.record_indices))):
            raise Stage5IdentityContractError("issue record_indices must be unique and sorted")
        self._validate_shape()

    def _validate_shape(self) -> None:
        missing_identity = {
            reason: identity_type for identity_type, reason in _MISSING_REASONS.items()
        }
        if self.reason in missing_identity:
            if (
                self.identity_type is not missing_identity[self.reason]
                or self.subject_fingerprint is not None
                or not self.record_indices
            ):
                raise Stage5IdentityContractError("missing-identity issue has invalid shape")
            return
        if self.reason is Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP:
            if (
                self.identity_type is None
                or self.subject_fingerprint is None
                or len(self.splits) < 2
                or len(self.record_indices) < 2
            ):
                raise Stage5IdentityContractError("overlap issue has invalid shape")
            return
        if self.reason is Stage5IdentityAuditReason.DUPLICATE_RELATIVE_PATH:
            if (
                self.identity_type is not None
                or self.subject_fingerprint is None
                or len(self.record_indices) < 2
            ):
                raise Stage5IdentityContractError("duplicate-path issue has invalid shape")
            return
        if self.reason is Stage5IdentityAuditReason.EYE_SCOPE_INCOMPLETE:
            if (
                self.identity_type is not Stage5IdentityType.EYE_ID
                or self.subject_fingerprint is not None
                or not self.record_indices
            ):
                raise Stage5IdentityContractError("eye-scope issue has invalid shape")
            return
        if self.reason is Stage5IdentityAuditReason.FOLLOWUP_SCOPE_INCOMPLETE:
            if (
                self.identity_type is not Stage5IdentityType.FOLLOWUP_ID
                or self.subject_fingerprint is not None
                or not self.record_indices
            ):
                raise Stage5IdentityContractError("followup-scope issue has invalid shape")
            return
        if self.reason is Stage5IdentityAuditReason.REQUIRED_SPLIT_EMPTY:
            if (
                self.identity_type is not None
                or self.subject_fingerprint is not None
                or len(self.splits) != 1
                or self.record_indices
            ):
                raise Stage5IdentityContractError("empty-split issue has invalid shape")
            return
        if self.reason is Stage5IdentityAuditReason.EMPTY_RECORD_SET:
            if (
                self.identity_type is not None
                or self.subject_fingerprint is not None
                or self.splits
                or self.record_indices
            ):
                raise Stage5IdentityContractError("empty-record issue has invalid shape")
            return
        if self.identity_type is not None or self.subject_fingerprint is not None:
            raise Stage5IdentityContractError("non-identity issue cannot carry identity data")
        if not self.record_indices:
            raise Stage5IdentityContractError("record-specific issue requires record indices")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "reason": self.reason.value,
            "identity_type": None if self.identity_type is None else self.identity_type.value,
            "subject_fingerprint": self.subject_fingerprint,
            "splits": [split.value for split in self.splits],
            "record_indices": list(self.record_indices),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage5IdentityIssue:
        data = _strict_mapping(
            value,
            name="Stage5IdentityIssue",
            required={
                "reason",
                "identity_type",
                "subject_fingerprint",
                "splits",
                "record_indices",
            },
        )
        raw_identity_type = data["identity_type"]
        raw_fingerprint = data["subject_fingerprint"]
        return cls(
            reason=Stage5IdentityAuditReason(
                _enum_value(data["reason"], Stage5IdentityAuditReason, "identity audit reason")
            ),
            identity_type=(
                None
                if raw_identity_type is None
                else Stage5IdentityType(
                    _enum_value(raw_identity_type, Stage5IdentityType, "identity_type")
                )
            ),
            subject_fingerprint=(
                None if raw_fingerprint is None else _sha256(raw_fingerprint, "subject_fingerprint")
            ),
            splits=tuple(
                Stage5Split(_enum_value(item, Stage5Split, "split"))
                for item in _sequence(data["splits"], "splits")
            ),
            record_indices=tuple(
                _plain_nonnegative_int(item, "record index")
                for item in _sequence(data["record_indices"], "record_indices")
            ),
        )


def _issue_sort_key(issue: Stage5IdentityIssue) -> tuple[object, ...]:
    identity_order = (
        len(_IDENTITY_ORDER)
        if issue.identity_type is None
        else _IDENTITY_ORDER[issue.identity_type]
    )
    return (
        _REASON_ORDER[issue.reason],
        identity_order,
        tuple(_SPLIT_ORDER[split] for split in issue.splits),
        issue.record_indices,
        issue.subject_fingerprint or "",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5IdentityAudit:
    """Versioned result of the strict five-identity isolation audit."""

    dataset_id: str | None
    status: Stage5IdentityAuditStatus
    isolation_proven: bool
    input_projection_sha256: str
    record_count: int
    audited_record_count: int
    rejected_record_count: int
    split_counts: Mapping[Stage5Split, int]
    unknown_counts: Mapping[Stage5IdentityType, int]
    reasons: tuple[Stage5IdentityAuditReason, ...]
    issues: tuple[Stage5IdentityIssue, ...]
    schema_version: str = STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION:
            raise Stage5IdentityContractError("unsupported Stage 5 identity audit schema_version")
        if self.dataset_id is not None:
            _dataset_id(self.dataset_id)
        if not isinstance(self.status, Stage5IdentityAuditStatus):
            raise Stage5IdentityContractError("status must use Stage5IdentityAuditStatus")
        if type(self.isolation_proven) is not bool:  # noqa: E721
            raise Stage5IdentityContractError("isolation_proven must be an exact bool")
        _sha256(self.input_projection_sha256, "input_projection_sha256")
        for field_name in (
            "record_count",
            "audited_record_count",
            "rejected_record_count",
        ):
            _plain_nonnegative_int(getattr(self, field_name), field_name)
        if self.record_count != self.audited_record_count + self.rejected_record_count:
            raise Stage5IdentityContractError(
                "record_count must equal audited plus rejected records"
            )
        split_counts = _exact_count_mapping(
            self.split_counts,
            enum_type=Stage5Split,
            name="split_counts",
        )
        unknown_counts = _exact_count_mapping(
            self.unknown_counts,
            enum_type=Stage5IdentityType,
            name="unknown_counts",
        )
        if sum(split_counts.values()) != self.audited_record_count:
            raise Stage5IdentityContractError("split_counts must sum to audited_record_count")
        if unknown_counts[Stage5IdentityType.SOURCE_IMAGE_ID] != 0:
            raise Stage5IdentityContractError("source_image_id cannot be unknown")
        object.__setattr__(self, "split_counts", MappingProxyType(split_counts))
        object.__setattr__(self, "unknown_counts", MappingProxyType(unknown_counts))
        if any(not isinstance(reason, Stage5IdentityAuditReason) for reason in self.reasons):
            raise Stage5IdentityContractError("reasons must use Stage5IdentityAuditReason")
        canonical_reasons = tuple(sorted(set(self.reasons), key=_REASON_ORDER.__getitem__))
        if self.reasons != canonical_reasons:
            raise Stage5IdentityContractError("reasons must be unique and canonically sorted")
        if any(not isinstance(issue, Stage5IdentityIssue) for issue in self.issues):
            raise Stage5IdentityContractError("issues must contain Stage5IdentityIssue values")
        canonical_issues = tuple(sorted(set(self.issues), key=_issue_sort_key))
        if self.issues != canonical_issues:
            raise Stage5IdentityContractError("issues must be unique and canonically sorted")
        if any(
            index >= self.record_count for issue in self.issues for index in issue.record_indices
        ):
            raise Stage5IdentityContractError("issue record index is outside the input")
        issue_reasons = tuple(
            sorted({issue.reason for issue in self.issues}, key=_REASON_ORDER.__getitem__)
        )
        if self.reasons != issue_reasons:
            raise Stage5IdentityContractError("reasons must exactly match issue reasons")
        self._validate_derived_invariants(split_counts, unknown_counts)

    def _validate_derived_invariants(
        self,
        split_counts: Mapping[Stage5Split, int],
        unknown_counts: Mapping[Stage5IdentityType, int],
    ) -> None:
        expected_pass = not self.reasons
        if self.isolation_proven is not expected_pass:
            raise Stage5IdentityContractError("isolation_proven must be false for every issue")
        expected_status = (
            Stage5IdentityAuditStatus.PASS
            if expected_pass
            else Stage5IdentityAuditStatus.FAIL_CLOSED
        )
        if self.status is not expected_status:
            raise Stage5IdentityContractError("status is inconsistent with the findings")
        if self.record_count == 0:
            if Stage5IdentityAuditReason.EMPTY_RECORD_SET not in self.reasons:
                raise Stage5IdentityContractError("empty input requires EMPTY_RECORD_SET")
        elif Stage5IdentityAuditReason.EMPTY_RECORD_SET in self.reasons:
            raise Stage5IdentityContractError("non-empty input cannot report EMPTY_RECORD_SET")
        empty_split_issues = {
            issue.splits[0]
            for issue in self.issues
            if issue.reason is Stage5IdentityAuditReason.REQUIRED_SPLIT_EMPTY
        }
        expected_empty_splits = {split for split, count in split_counts.items() if count == 0}
        if empty_split_issues != expected_empty_splits:
            raise Stage5IdentityContractError("empty-split issues must exactly match split counts")
        for identity_type, reason in _MISSING_REASONS.items():
            missing = unknown_counts[identity_type] > 0
            if (reason in self.reasons) is not missing:
                raise Stage5IdentityContractError(
                    f"{reason.value} must exactly match its unknown count"
                )
        if expected_pass and (self.dataset_id is None or any(unknown_counts.values())):
            raise Stage5IdentityContractError(
                "a passing identity audit requires one dataset and no unknown identities"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "isolation_proven": self.isolation_proven,
            "input_projection_sha256": self.input_projection_sha256,
            "record_count": self.record_count,
            "audited_record_count": self.audited_record_count,
            "rejected_record_count": self.rejected_record_count,
            "split_counts": {split.value: self.split_counts[split] for split in Stage5Split},
            "unknown_counts": {
                identity_type.value: self.unknown_counts[identity_type]
                for identity_type in Stage5IdentityType
            },
            "reasons": [reason.value for reason in self.reasons],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        """Return deterministic UTF-8-safe JSON without non-standard constants."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage5IdentityAudit:
        data = _strict_mapping(
            value,
            name="Stage5IdentityAudit",
            required={
                "schema_version",
                "dataset_id",
                "status",
                "isolation_proven",
                "input_projection_sha256",
                "record_count",
                "audited_record_count",
                "rejected_record_count",
                "split_counts",
                "unknown_counts",
                "reasons",
                "issues",
            },
        )
        if data["schema_version"] != STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION:
            raise Stage5IdentityContractError("unsupported Stage 5 identity audit schema_version")
        raw_dataset_id = data["dataset_id"]
        raw_isolation = data["isolation_proven"]
        if type(raw_isolation) is not bool:  # noqa: E721
            raise Stage5IdentityContractError("isolation_proven must be an exact bool")
        return cls(
            schema_version=STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION,
            dataset_id=None if raw_dataset_id is None else _dataset_id(raw_dataset_id),
            status=Stage5IdentityAuditStatus(
                _enum_value(data["status"], Stage5IdentityAuditStatus, "status")
            ),
            isolation_proven=raw_isolation,
            input_projection_sha256=_sha256(
                data["input_projection_sha256"], "input_projection_sha256"
            ),
            record_count=_plain_nonnegative_int(data["record_count"], "record_count"),
            audited_record_count=_plain_nonnegative_int(
                data["audited_record_count"], "audited_record_count"
            ),
            rejected_record_count=_plain_nonnegative_int(
                data["rejected_record_count"], "rejected_record_count"
            ),
            split_counts=_parse_count_mapping(
                data["split_counts"], enum_type=Stage5Split, name="split_counts"
            ),
            unknown_counts=_parse_count_mapping(
                data["unknown_counts"],
                enum_type=Stage5IdentityType,
                name="unknown_counts",
            ),
            reasons=tuple(
                Stage5IdentityAuditReason(
                    _enum_value(item, Stage5IdentityAuditReason, "identity audit reason")
                )
                for item in _sequence(data["reasons"], "reasons")
            ),
            issues=tuple(
                Stage5IdentityIssue.from_dict(
                    _strict_mapping(
                        item,
                        name=f"issues[{index}]",
                        required={
                            "reason",
                            "identity_type",
                            "subject_fingerprint",
                            "splits",
                            "record_indices",
                        },
                    )
                )
                for index, item in enumerate(_sequence(data["issues"], "issues"))
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> Stage5IdentityAudit:
        if not isinstance(value, str):
            raise Stage5IdentityContractError("identity audit JSON must be a string")

        def reject_constant(constant: str) -> None:
            raise Stage5IdentityContractError(f"non-standard JSON constant: {constant}")

        try:
            decoded = json.loads(value, parse_constant=reject_constant)
        except (json.JSONDecodeError, TypeError) as error:
            raise Stage5IdentityContractError("invalid identity audit JSON") from error
        data = _strict_mapping(
            decoded,
            name="Stage5IdentityAudit",
            required={
                "schema_version",
                "dataset_id",
                "status",
                "isolation_proven",
                "input_projection_sha256",
                "record_count",
                "audited_record_count",
                "rejected_record_count",
                "split_counts",
                "unknown_counts",
                "reasons",
                "issues",
            },
        )
        return cls.from_dict(data)


def _exact_count_mapping(
    value: Mapping[_EnumT, int],
    *,
    enum_type: type[_EnumT],
    name: str,
) -> dict[_EnumT, int]:
    expected = set(enum_type)
    if set(value) != expected:
        raise Stage5IdentityContractError(f"{name} must contain every controlled key exactly")
    return {key: _plain_nonnegative_int(value[key], f"{name}.{key.value}") for key in enum_type}


def _parse_count_mapping(
    value: object,
    *,
    enum_type: type[_EnumT],
    name: str,
) -> Mapping[_EnumT, int]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage5IdentityContractError(f"{name} must be an object with string keys")
    expected = {member.value for member in enum_type}
    if set(value) != expected:
        raise Stage5IdentityContractError(f"{name} must contain every controlled key exactly")
    return {
        member: _plain_nonnegative_int(value[member.value], f"{name}.{member.value}")
        for member in enum_type
    }


def _input_projection(records: tuple[object, ...]) -> str:
    projection: list[object] = []
    for index, record in enumerate(records):
        try:
            normalized = _normalize_record(record)
        except Stage5IdentityContractError:
            normalized = None
        if normalized is None:
            record_type = type(record)
            projection.append(
                {
                    "index": index,
                    "invalid_type": f"{record_type.__module__}.{record_type.__qualname__}",
                }
            )
        else:
            projection.append({"index": index, **normalized.to_dict()})
    return _canonical_sha256(
        {
            "schema_version": STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION,
            "records": projection,
        }
    )


def _normalize_record(record: object) -> Stage5IdentityRecord | None:
    if isinstance(record, Stage5IdentityRecord):
        normalized = Stage5IdentityRecord(
            dataset_id=record.dataset_id,
            relative_path=record.relative_path,
            split=record.split,
            source_image_id=record.source_image_id,
            patient_id=record.patient_id,
            eye_id=record.eye_id,
            followup_id=record.followup_id,
            intervention_pair_id=record.intervention_pair_id,
        )
        if normalized != record:
            raise Stage5IdentityContractError("identity record is not canonically normalized")
        return normalized
    if isinstance(record, ManifestRecordV2):
        return Stage5IdentityRecord.from_manifest_record(record)
    return None


def _subject_fingerprint(kind: str, *scope_parts: str) -> str:
    return _canonical_sha256(
        {
            "schema_version": STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION,
            "kind": kind,
            "scope_parts": list(scope_parts),
        }
    )


def _identity_subject(
    identity_type: Stage5IdentityType,
    record: Stage5IdentityRecord,
) -> tuple[str, ...] | None:
    """Return a dataset-scoped identity key or ``None`` when it is unknown.

    Eye tokens such as ``left`` are patient-local, while follow-up tokens such as
    ``baseline`` are patient-and-eye-local.  Source, patient, and intervention-pair
    identifiers are dataset-global.  No empty-string sentinel is ever introduced.
    """

    dataset_id = record.dataset_id
    if identity_type is Stage5IdentityType.SOURCE_IMAGE_ID:
        return (dataset_id, record.source_image_id)
    if identity_type is Stage5IdentityType.PATIENT_ID:
        return None if record.patient_id is None else (dataset_id, record.patient_id)
    if identity_type is Stage5IdentityType.EYE_ID:
        if record.patient_id is None or record.eye_id is None:
            return None
        return (dataset_id, record.patient_id, record.eye_id)
    if identity_type is Stage5IdentityType.FOLLOWUP_ID:
        if record.patient_id is None or record.eye_id is None or record.followup_id is None:
            return None
        return (dataset_id, record.patient_id, record.eye_id, record.followup_id)
    if identity_type is Stage5IdentityType.INTERVENTION_PAIR_ID:
        return (
            None
            if record.intervention_pair_id is None
            else (dataset_id, record.intervention_pair_id)
        )
    raise AssertionError("unreachable identity type")


def _make_issue(
    *,
    reason: Stage5IdentityAuditReason,
    identity_type: Stage5IdentityType | None = None,
    subject_fingerprint: str | None = None,
    splits: Iterable[Stage5Split] = (),
    record_indices: Iterable[int] = (),
) -> Stage5IdentityIssue:
    return Stage5IdentityIssue(
        reason=reason,
        identity_type=identity_type,
        subject_fingerprint=subject_fingerprint,
        splits=tuple(sorted(set(splits), key=_SPLIT_ORDER.__getitem__)),
        record_indices=tuple(sorted(set(record_indices))),
    )


def audit_five_identity_split_isolation(records: Iterable[object]) -> Stage5IdentityAudit:
    """Audit all five identities across train/calibration/test without file I/O.

    Repeated membership inside one split is valid (for example, two twin records may
    share a source or intervention-pair identifier).  Any protected identity observed
    in more than one split is leakage.  Missing nullable identifiers block the proof;
    they are never converted to empty strings or inferred from paths.
    """

    if isinstance(records, str | bytes) or not isinstance(records, Iterable):
        materialized: tuple[object, ...] = (records,)
    else:
        materialized = tuple(records)
    issues: list[Stage5IdentityIssue] = []
    normalized_by_index: dict[int, Stage5IdentityRecord] = {}
    audited_by_index: dict[int, tuple[Stage5IdentityRecord, Stage5Split]] = {}

    if not materialized:
        issues.append(_make_issue(reason=Stage5IdentityAuditReason.EMPTY_RECORD_SET))
    for index, record in enumerate(materialized):
        try:
            normalized = _normalize_record(record)
        except (AttributeError, Stage5IdentityContractError, TypeError, ValueError):
            normalized = None
            reason = Stage5IdentityAuditReason.INVALID_RECORD_CONTRACT
        else:
            reason = Stage5IdentityAuditReason.INVALID_RECORD_TYPE
        if normalized is None:
            issues.append(_make_issue(reason=reason, record_indices=(index,)))
            continue
        normalized_by_index[index] = normalized
        try:
            split = Stage5Split(normalized.split)
        except ValueError:
            issues.append(
                _make_issue(
                    reason=Stage5IdentityAuditReason.INVALID_SPLIT,
                    record_indices=(index,),
                )
            )
            continue
        audited_by_index[index] = (normalized, split)

    dataset_ids = {record.dataset_id for record in normalized_by_index.values()}
    dataset_id = next(iter(dataset_ids)) if len(dataset_ids) == 1 else None
    if len(dataset_ids) > 1:
        issues.append(
            _make_issue(
                reason=Stage5IdentityAuditReason.MULTIPLE_DATASETS,
                record_indices=normalized_by_index,
            )
        )

    split_counts = {split: 0 for split in Stage5Split}
    for _, split in audited_by_index.values():
        split_counts[split] += 1
    for split, count in split_counts.items():
        if count == 0:
            issues.append(
                _make_issue(
                    reason=Stage5IdentityAuditReason.REQUIRED_SPLIT_EMPTY,
                    splits=(split,),
                )
            )

    paths: dict[tuple[str, str], list[tuple[int, Stage5Split | None]]] = defaultdict(list)
    for index, record in normalized_by_index.items():
        audited = audited_by_index.get(index)
        paths[(record.dataset_id, record.relative_path)].append(
            (index, None if audited is None else audited[1])
        )
    for (path_dataset_id, relative_path), path_occurrences in paths.items():
        if len(path_occurrences) > 1:
            issues.append(
                _make_issue(
                    reason=Stage5IdentityAuditReason.DUPLICATE_RELATIVE_PATH,
                    subject_fingerprint=_subject_fingerprint(
                        "relative_path", path_dataset_id, relative_path
                    ),
                    splits=(split for _, split in path_occurrences if split is not None),
                    record_indices=(index for index, _ in path_occurrences),
                )
            )

    unknown_indices: dict[Stage5IdentityType, list[int]] = {
        identity_type: [] for identity_type in Stage5IdentityType
    }
    unknown_splits: dict[Stage5IdentityType, set[Stage5Split]] = {
        identity_type: set() for identity_type in Stage5IdentityType
    }
    eye_scope_incomplete: list[int] = []
    eye_scope_splits: set[Stage5Split] = set()
    followup_scope_incomplete: list[int] = []
    followup_scope_splits: set[Stage5Split] = set()
    assignments: dict[tuple[Stage5IdentityType, tuple[str, ...]], list[tuple[int, Stage5Split]]] = (
        defaultdict(list)
    )

    for index, (record, split) in audited_by_index.items():
        if record.eye_id is not None and record.patient_id is None:
            eye_scope_incomplete.append(index)
            eye_scope_splits.add(split)
        if record.followup_id is not None and (record.patient_id is None or record.eye_id is None):
            followup_scope_incomplete.append(index)
            followup_scope_splits.add(split)
        for identity_type in Stage5IdentityType:
            subject = _identity_subject(identity_type, record)
            raw_is_missing = {
                Stage5IdentityType.SOURCE_IMAGE_ID: False,
                Stage5IdentityType.PATIENT_ID: record.patient_id is None,
                Stage5IdentityType.EYE_ID: record.eye_id is None,
                Stage5IdentityType.FOLLOWUP_ID: record.followup_id is None,
                Stage5IdentityType.INTERVENTION_PAIR_ID: record.intervention_pair_id is None,
            }[identity_type]
            if raw_is_missing:
                unknown_indices[identity_type].append(index)
                unknown_splits[identity_type].add(split)
            if subject is not None:
                assignments[(identity_type, subject)].append((index, split))

    for identity_type, reason in _MISSING_REASONS.items():
        if unknown_indices[identity_type]:
            issues.append(
                _make_issue(
                    reason=reason,
                    identity_type=identity_type,
                    splits=unknown_splits[identity_type],
                    record_indices=unknown_indices[identity_type],
                )
            )
    if eye_scope_incomplete:
        issues.append(
            _make_issue(
                reason=Stage5IdentityAuditReason.EYE_SCOPE_INCOMPLETE,
                identity_type=Stage5IdentityType.EYE_ID,
                splits=eye_scope_splits,
                record_indices=eye_scope_incomplete,
            )
        )
    if followup_scope_incomplete:
        issues.append(
            _make_issue(
                reason=Stage5IdentityAuditReason.FOLLOWUP_SCOPE_INCOMPLETE,
                identity_type=Stage5IdentityType.FOLLOWUP_ID,
                splits=followup_scope_splits,
                record_indices=followup_scope_incomplete,
            )
        )

    for (identity_type, subject), identity_occurrences in assignments.items():
        splits = {split for _, split in identity_occurrences}
        if len(splits) > 1:
            issues.append(
                _make_issue(
                    reason=Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP,
                    identity_type=identity_type,
                    subject_fingerprint=_subject_fingerprint(identity_type.value, *subject),
                    splits=splits,
                    record_indices=(index for index, _ in identity_occurrences),
                )
            )

    canonical_issues = tuple(sorted(set(issues), key=_issue_sort_key))
    reasons = tuple(
        sorted({issue.reason for issue in canonical_issues}, key=_REASON_ORDER.__getitem__)
    )
    status = (
        Stage5IdentityAuditStatus.PASS if not reasons else Stage5IdentityAuditStatus.FAIL_CLOSED
    )
    return Stage5IdentityAudit(
        dataset_id=dataset_id,
        status=status,
        isolation_proven=status is Stage5IdentityAuditStatus.PASS,
        input_projection_sha256=_input_projection(materialized),
        record_count=len(materialized),
        audited_record_count=len(audited_by_index),
        rejected_record_count=len(materialized) - len(audited_by_index),
        split_counts=split_counts,
        unknown_counts={
            identity_type: len(indices) for identity_type, indices in unknown_indices.items()
        },
        reasons=reasons,
        issues=canonical_issues,
    )


def verify_five_identity_audit(
    audit: object,
    records: Iterable[object],
) -> bool:
    """Recompute the audit and require an exact canonical match."""

    if not isinstance(audit, Stage5IdentityAudit):
        return False
    try:
        expected = audit_five_identity_split_isolation(records)
        return audit.to_json() == expected.to_json()
    except (Stage5IdentityContractError, TypeError, ValueError):
        return False


__all__ = [
    "STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION",
    "Stage5IdentityAudit",
    "Stage5IdentityAuditReason",
    "Stage5IdentityAuditStatus",
    "Stage5IdentityContractError",
    "Stage5IdentityIssue",
    "Stage5IdentityRecord",
    "Stage5IdentityType",
    "Stage5Split",
    "audit_five_identity_split_isolation",
    "verify_five_identity_audit",
]
