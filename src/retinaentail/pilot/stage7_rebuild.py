"""Pure Stage 7 verification of a frozen and read-only-rebuilt Stage 2 view.

This module performs no filesystem I/O.  Callers must first consume the short-lived
Stage 6 root binding, rebuild the Stage 2 objects through the approved adapter, close
that root authority, and only then pass the two in-memory evidence pairs here.

The comparison is deliberately strict: record order and every record/label field must
match.  Missing protected identities, ``UNKNOWN`` labels, malformed audits, non-finite
values, or any verification exception produce a versioned ``FAIL_CLOSED`` result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias

from retinaentail.data.audit import AuditResult, AuditStatus
from retinaentail.data.stage2_models import (
    DatasetManifestV2,
    LabelAvailability,
)
from retinaentail.data.stage2_package import Stage2PackageEvidence
from retinaentail.pilot.identity import (
    Stage5IdentityAudit,
    Stage5IdentityAuditStatus,
    audit_five_identity_split_isolation,
)

STAGE7_REBUILD_SCHEMA_VERSION: Final = "7.0.0"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_FIELD_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_.\[\]-]*$")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class Stage7RebuildStatus(StrEnum):
    """Outcome of the in-memory frozen-versus-rebuilt verification."""

    PASS = "PASS"
    FAIL_CLOSED = "FAIL_CLOSED"


class Stage7RebuildReason(StrEnum):
    """Controlled reasons that prevent the rebuilt view from being authoritative."""

    INPUT_INVALID = "INPUT_INVALID"
    FROZEN_PACKAGE_EVIDENCE_INVALID = "FROZEN_PACKAGE_EVIDENCE_INVALID"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    DUPLICATE_RECORD = "DUPLICATE_RECORD"
    FROZEN_AUDIT_INVALID = "FROZEN_AUDIT_INVALID"
    REBUILT_AUDIT_INVALID = "REBUILT_AUDIT_INVALID"
    DATASET_ID_MISMATCH = "DATASET_ID_MISMATCH"
    DATASET_VERSION_MISMATCH = "DATASET_VERSION_MISMATCH"
    MANIFEST_HASH_MISMATCH = "MANIFEST_HASH_MISMATCH"
    AUDIT_EVIDENCE_MISMATCH = "AUDIT_EVIDENCE_MISMATCH"
    RECORD_COUNT_MISMATCH = "RECORD_COUNT_MISMATCH"
    RECORD_ORDER_MISMATCH = "RECORD_ORDER_MISMATCH"
    RECORD_FIELD_MISMATCH = "RECORD_FIELD_MISMATCH"
    LABEL_FIELD_MISMATCH = "LABEL_FIELD_MISMATCH"
    UNKNOWN_LABEL_STATE = "UNKNOWN_LABEL_STATE"
    FIVE_IDENTITY_ISOLATION_FAILED = "FIVE_IDENTITY_ISOLATION_FAILED"
    VERIFICATION_EXCEPTION = "VERIFICATION_EXCEPTION"


_REASON_ORDER: Final = {reason: index for index, reason in enumerate(Stage7RebuildReason)}


class Stage7DifferenceKind(StrEnum):
    """Location of a redacted comparison difference."""

    MANIFEST = "MANIFEST"
    RECORD = "RECORD"
    LABEL = "LABEL"
    AUDIT = "AUDIT"


_DIFFERENCE_ORDER: Final = {kind: index for index, kind in enumerate(Stage7DifferenceKind)}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _contains_nonfinite(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(
            _contains_nonfinite(key) or _contains_nonfinite(item) for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _value_digest(value: object) -> str:
    return _canonical_sha256({"value": value})


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7RebuildDifference:
    """One field-level difference without copying raw identity or label values."""

    kind: Stage7DifferenceKind
    field: str
    record_index: int | None
    label_index: int | None
    expected_sha256: str
    observed_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, Stage7DifferenceKind):
            raise ValueError("kind must use Stage7DifferenceKind")
        if not isinstance(self.field, str) or _FIELD_PATTERN.fullmatch(self.field) is None:
            raise ValueError("field must be a normalized comparison field")
        for name in ("record_index", "label_index"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):  # noqa: E721
                raise ValueError(f"{name} must be null or a non-negative integer")
        if self.kind in {Stage7DifferenceKind.MANIFEST, Stage7DifferenceKind.AUDIT} and (
            self.record_index is not None or self.label_index is not None
        ):
            raise ValueError("manifest/audit differences cannot carry record indices")
        if self.kind is Stage7DifferenceKind.RECORD and self.record_index is None:
            raise ValueError("record difference requires record_index")
        if self.kind is Stage7DifferenceKind.LABEL and (
            self.record_index is None or self.label_index is None
        ):
            raise ValueError("label difference requires both indices")
        for name in ("expected_sha256", "observed_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def sort_key(self) -> tuple[object, ...]:
        return (
            _DIFFERENCE_ORDER[self.kind],
            -1 if self.record_index is None else self.record_index,
            -1 if self.label_index is None else self.label_index,
            self.field,
            self.expected_sha256,
            self.observed_sha256,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kind": self.kind.value,
            "field": self.field,
            "record_index": self.record_index,
            "label_index": self.label_index,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7RebuildVerification:
    """Immutable, versioned evidence for the Stage 7 reconstruction comparison."""

    status: Stage7RebuildStatus
    dataset_id: str | None
    dataset_version: str | None
    frozen_manifest_sha256: str | None
    rebuilt_manifest_sha256: str | None
    frozen_audit_sha256: str | None
    rebuilt_audit_sha256: str | None
    frozen_record_count: int
    rebuilt_record_count: int
    manifests_identical: bool
    audits_valid: bool
    audits_identical: bool
    identity_isolation_proven: bool
    frozen_unknown_label_count: int
    rebuilt_unknown_label_count: int
    frozen_identity_audit: Stage5IdentityAudit | None
    rebuilt_identity_audit: Stage5IdentityAudit | None
    reasons: tuple[Stage7RebuildReason, ...]
    differences: tuple[Stage7RebuildDifference, ...]
    schema_version: str = STAGE7_REBUILD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE7_REBUILD_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 7 rebuild schema_version")
        if not isinstance(self.status, Stage7RebuildStatus):
            raise ValueError("status must use Stage7RebuildStatus")
        for name in (
            "frozen_record_count",
            "rebuilt_record_count",
            "frozen_unknown_label_count",
            "rebuilt_unknown_label_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:  # noqa: E721
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "manifests_identical",
            "audits_valid",
            "audits_identical",
            "identity_isolation_proven",
        ):
            if type(getattr(self, name)) is not bool:  # noqa: E721
                raise ValueError(f"{name} must be an exact bool")
        for name in (
            "frozen_manifest_sha256",
            "rebuilt_manifest_sha256",
            "frozen_audit_sha256",
            "rebuilt_audit_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"{name} must be null or a lowercase SHA-256 digest")
        canonical_reasons = tuple(sorted(set(self.reasons), key=_REASON_ORDER.__getitem__))
        if self.reasons != canonical_reasons:
            raise ValueError("reasons must be unique and canonically sorted")
        canonical_differences = tuple(
            sorted(set(self.differences), key=lambda item: item.sort_key())
        )
        if self.differences != canonical_differences:
            raise ValueError("differences must be unique and canonically sorted")
        for audit in (self.frozen_identity_audit, self.rebuilt_identity_audit):
            if audit is not None and not isinstance(audit, Stage5IdentityAudit):
                raise ValueError("identity audits must be Stage5IdentityAudit values or null")
        expected_pass = (
            not self.reasons
            and not self.differences
            and self.manifests_identical
            and self.audits_valid
            and self.audits_identical
            and self.identity_isolation_proven
            and self.frozen_unknown_label_count == 0
            and self.rebuilt_unknown_label_count == 0
            and self.frozen_manifest_sha256 is not None
            and self.frozen_manifest_sha256 == self.rebuilt_manifest_sha256
            and self.frozen_audit_sha256 is not None
            and self.frozen_audit_sha256 == self.rebuilt_audit_sha256
            and self.frozen_record_count == self.rebuilt_record_count > 0
            and self.frozen_identity_audit is not None
            and self.rebuilt_identity_audit is not None
        )
        expected_status = (
            Stage7RebuildStatus.PASS if expected_pass else Stage7RebuildStatus.FAIL_CLOSED
        )
        if self.status is not expected_status:
            raise ValueError("status is inconsistent with Stage 7 rebuild evidence")

    def _evidence_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "frozen_manifest_sha256": self.frozen_manifest_sha256,
            "rebuilt_manifest_sha256": self.rebuilt_manifest_sha256,
            "frozen_audit_sha256": self.frozen_audit_sha256,
            "rebuilt_audit_sha256": self.rebuilt_audit_sha256,
            "frozen_record_count": self.frozen_record_count,
            "rebuilt_record_count": self.rebuilt_record_count,
            "manifests_identical": self.manifests_identical,
            "audits_valid": self.audits_valid,
            "audits_identical": self.audits_identical,
            "identity_isolation_proven": self.identity_isolation_proven,
            "frozen_unknown_label_count": self.frozen_unknown_label_count,
            "rebuilt_unknown_label_count": self.rebuilt_unknown_label_count,
            "frozen_identity_audit": (
                None if self.frozen_identity_audit is None else self.frozen_identity_audit.to_dict()
            ),
            "rebuilt_identity_audit": (
                None
                if self.rebuilt_identity_audit is None
                else self.rebuilt_identity_audit.to_dict()
            ),
            "reasons": [reason.value for reason in self.reasons],
            "differences": [difference.to_dict() for difference in self.differences],
        }

    @property
    def verification_sha256(self) -> str:
        """Canonical digest of every serialized verification field except itself."""

        return _canonical_sha256(self._evidence_payload())

    @property
    def passed(self) -> bool:
        return self.status is Stage7RebuildStatus.PASS

    def to_dict(self) -> dict[str, JSONValue]:
        result = self._evidence_payload()
        result["verification_sha256"] = self.verification_sha256
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True, slots=True)
class _ManifestView:
    manifest: DatasetManifestV2
    payload: dict[str, JSONValue]
    sha256: str
    unknown_label_count: int


@dataclass(frozen=True, slots=True)
class _AuditView:
    audit: AuditResult
    payload: dict[str, object]
    sha256: str


def _manifest_view(
    value: object,
    reasons: set[Stage7RebuildReason],
) -> _ManifestView | None:
    if not isinstance(value, DatasetManifestV2):
        reasons.add(Stage7RebuildReason.INPUT_INVALID)
        return None
    try:
        paths = [record.relative_path for record in value.records]
        if len(paths) != len(set(paths)):
            reasons.add(Stage7RebuildReason.DUPLICATE_RECORD)
        payload = value.to_dict()
        if _contains_nonfinite(payload):
            reasons.add(Stage7RebuildReason.NONFINITE_VALUE)
            return None
        canonical = DatasetManifestV2.from_dict(payload)
        if canonical != value:
            reasons.add(Stage7RebuildReason.INPUT_INVALID)
            return None
        unknown = 0
        for record in value.records:
            for label in record.labels:
                if label.availability is LabelAvailability.UNKNOWN:
                    unknown += 1
        return _ManifestView(
            manifest=value,
            payload=payload,
            sha256=_canonical_sha256(payload),
            unknown_label_count=unknown,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        reasons.add(Stage7RebuildReason.INPUT_INVALID)
        return None


def _audit_view(value: object, reasons: set[Stage7RebuildReason]) -> _AuditView | None:
    if not isinstance(value, AuditResult):
        reasons.add(Stage7RebuildReason.INPUT_INVALID)
        return None
    try:
        payload = value.to_dict()
        if _contains_nonfinite(payload):
            reasons.add(Stage7RebuildReason.NONFINITE_VALUE)
            return None
        return _AuditView(audit=value, payload=payload, sha256=_canonical_sha256(payload))
    except (AttributeError, TypeError, ValueError, OverflowError):
        reasons.add(Stage7RebuildReason.INPUT_INVALID)
        return None


def _audit_matches_manifest(view: _AuditView, manifest: _ManifestView) -> bool:
    audit = view.audit
    summary = audit.summary
    records = manifest.manifest.records
    available_labels = sum(
        label.availability is LabelAvailability.AVAILABLE
        for record in records
        for label in record.labels
    )
    unknown_labels = sum(
        label.availability is LabelAvailability.UNKNOWN
        for record in records
        for label in record.labels
    )
    mask_labels = sum(
        label.availability is LabelAvailability.AVAILABLE and label.semantic.value.endswith("_MASK")
        for record in records
        for label in record.labels
    )
    expected_issue_counts = tuple(
        sorted(Counter(issue.severity.value for issue in audit.issues).items())
    )
    counts_are_plain = all(
        type(value) is int and value >= 0  # noqa: E721
        for value in (
            summary.record_count,
            summary.image_files_checked,
            summary.label_files_checked,
            summary.mask_files_decoded,
            summary.unknown_label_count,
        )
    )
    return (
        audit.status is AuditStatus.PASS
        and audit.dataset_id == manifest.manifest.dataset_id
        and audit.dataset_version == manifest.manifest.dataset_version
        and audit.manifest_sha256 == manifest.sha256
        and not audit.issues
        and counts_are_plain
        and summary.issue_counts == expected_issue_counts == ()
        and summary.record_count == len(records)
        and summary.image_files_checked == len(records)
        and summary.label_files_checked == available_labels
        and summary.mask_files_decoded == mask_labels
        and summary.unknown_label_count == unknown_labels
    )


def _difference(
    *,
    kind: Stage7DifferenceKind,
    field: str,
    expected: object,
    observed: object,
    record_index: int | None = None,
    label_index: int | None = None,
) -> Stage7RebuildDifference:
    return Stage7RebuildDifference(
        kind=kind,
        field=field,
        record_index=record_index,
        label_index=label_index,
        expected_sha256=_value_digest(expected),
        observed_sha256=_value_digest(observed),
    )


def _compare_manifest_fields(
    frozen: _ManifestView,
    rebuilt: _ManifestView,
    reasons: set[Stage7RebuildReason],
    differences: list[Stage7RebuildDifference],
) -> None:
    first = frozen.manifest
    second = rebuilt.manifest
    if first.dataset_id != second.dataset_id:
        reasons.add(Stage7RebuildReason.DATASET_ID_MISMATCH)
        differences.append(
            _difference(
                kind=Stage7DifferenceKind.MANIFEST,
                field="dataset_id",
                expected=first.dataset_id,
                observed=second.dataset_id,
            )
        )
    if first.dataset_version != second.dataset_version:
        reasons.add(Stage7RebuildReason.DATASET_VERSION_MISMATCH)
        differences.append(
            _difference(
                kind=Stage7DifferenceKind.MANIFEST,
                field="dataset_version",
                expected=first.dataset_version,
                observed=second.dataset_version,
            )
        )
    if len(first.records) != len(second.records):
        reasons.add(Stage7RebuildReason.RECORD_COUNT_MISMATCH)
        differences.append(
            _difference(
                kind=Stage7DifferenceKind.MANIFEST,
                field="record_count",
                expected=len(first.records),
                observed=len(second.records),
            )
        )

    frozen_paths = tuple(record.relative_path for record in first.records)
    rebuilt_paths = tuple(record.relative_path for record in second.records)
    pure_reorder = (
        frozen_paths != rebuilt_paths
        and len(frozen_paths) == len(rebuilt_paths)
        and sorted(frozen_paths) == sorted(rebuilt_paths)
    )
    if pure_reorder:
        reasons.add(Stage7RebuildReason.RECORD_ORDER_MISMATCH)
        differences.append(
            _difference(
                kind=Stage7DifferenceKind.MANIFEST,
                field="record_order",
                expected=frozen_paths,
                observed=rebuilt_paths,
            )
        )
        return

    record_fields = (
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
    )
    for record_index, (expected_record, observed_record) in enumerate(
        zip(first.records, second.records, strict=False)
    ):
        expected_payload = expected_record.to_dict()
        observed_payload = observed_record.to_dict()
        for field in record_fields:
            if expected_payload[field] != observed_payload[field]:
                reasons.add(Stage7RebuildReason.RECORD_FIELD_MISMATCH)
                differences.append(
                    _difference(
                        kind=Stage7DifferenceKind.RECORD,
                        field=field,
                        expected=expected_payload[field],
                        observed=observed_payload[field],
                        record_index=record_index,
                    )
                )
        expected_labels = expected_record.labels
        observed_labels = observed_record.labels
        if len(expected_labels) != len(observed_labels):
            reasons.add(Stage7RebuildReason.LABEL_FIELD_MISMATCH)
            differences.append(
                _difference(
                    kind=Stage7DifferenceKind.RECORD,
                    field="labels.count",
                    expected=len(expected_labels),
                    observed=len(observed_labels),
                    record_index=record_index,
                )
            )
        for label_index, (expected_label, observed_label) in enumerate(
            zip(expected_labels, observed_labels, strict=False)
        ):
            expected_label_payload = expected_label.to_dict()
            observed_label_payload = observed_label.to_dict()
            for field in (
                "schema_version",
                "semantic",
                "availability",
                "relative_path",
                "sha256",
                "byte_size",
            ):
                if expected_label_payload[field] != observed_label_payload[field]:
                    reasons.add(Stage7RebuildReason.LABEL_FIELD_MISMATCH)
                    differences.append(
                        _difference(
                            kind=Stage7DifferenceKind.LABEL,
                            field=field,
                            expected=expected_label_payload[field],
                            observed=observed_label_payload[field],
                            record_index=record_index,
                            label_index=label_index,
                        )
                    )


def _make_result(
    *,
    frozen_manifest: _ManifestView | None,
    rebuilt_manifest: _ManifestView | None,
    frozen_audit: _AuditView | None,
    rebuilt_audit: _AuditView | None,
    frozen_identity_audit: Stage5IdentityAudit | None,
    rebuilt_identity_audit: Stage5IdentityAudit | None,
    audits_valid: bool,
    reasons: set[Stage7RebuildReason],
    differences: list[Stage7RebuildDifference],
) -> Stage7RebuildVerification:
    manifests_identical = (
        frozen_manifest is not None
        and rebuilt_manifest is not None
        and frozen_manifest.payload == rebuilt_manifest.payload
    )
    audits_identical = (
        frozen_audit is not None
        and rebuilt_audit is not None
        and frozen_audit.payload == rebuilt_audit.payload
    )
    identity_proven = (
        frozen_identity_audit is not None
        and rebuilt_identity_audit is not None
        and frozen_identity_audit.status is Stage5IdentityAuditStatus.PASS
        and rebuilt_identity_audit.status is Stage5IdentityAuditStatus.PASS
        and frozen_identity_audit.isolation_proven
        and rebuilt_identity_audit.isolation_proven
    )
    canonical_reasons = tuple(sorted(reasons, key=_REASON_ORDER.__getitem__))
    canonical_differences = tuple(sorted(set(differences), key=lambda item: item.sort_key()))
    passed = (
        not canonical_reasons
        and not canonical_differences
        and manifests_identical
        and audits_valid
        and audits_identical
        and identity_proven
    )
    return Stage7RebuildVerification(
        status=Stage7RebuildStatus.PASS if passed else Stage7RebuildStatus.FAIL_CLOSED,
        dataset_id=(None if frozen_manifest is None else frozen_manifest.manifest.dataset_id),
        dataset_version=(
            None if frozen_manifest is None else frozen_manifest.manifest.dataset_version
        ),
        frozen_manifest_sha256=(None if frozen_manifest is None else frozen_manifest.sha256),
        rebuilt_manifest_sha256=(None if rebuilt_manifest is None else rebuilt_manifest.sha256),
        frozen_audit_sha256=None if frozen_audit is None else frozen_audit.sha256,
        rebuilt_audit_sha256=None if rebuilt_audit is None else rebuilt_audit.sha256,
        frozen_record_count=(
            0 if frozen_manifest is None else len(frozen_manifest.manifest.records)
        ),
        rebuilt_record_count=(
            0 if rebuilt_manifest is None else len(rebuilt_manifest.manifest.records)
        ),
        manifests_identical=manifests_identical,
        audits_valid=audits_valid,
        audits_identical=audits_identical,
        identity_isolation_proven=identity_proven,
        frozen_unknown_label_count=(
            0 if frozen_manifest is None else frozen_manifest.unknown_label_count
        ),
        rebuilt_unknown_label_count=(
            0 if rebuilt_manifest is None else rebuilt_manifest.unknown_label_count
        ),
        frozen_identity_audit=frozen_identity_audit,
        rebuilt_identity_audit=rebuilt_identity_audit,
        reasons=canonical_reasons,
        differences=canonical_differences,
    )


def verify_stage7_rebuild(
    *,
    frozen_manifest: object,
    frozen_audit: object,
    rebuilt_manifest: object,
    rebuilt_audit: object,
) -> Stage7RebuildVerification:
    """Fail-closed comparison of frozen and rebuilt Stage 2 evidence.

    The function is a pure object operation and never accepts or opens a filesystem
    path.  Record ordering is authoritative; callers may not normalize a changed order
    after the fact.  Every unexpected exception is reduced to a controlled failure.
    """

    reasons: set[Stage7RebuildReason] = set()
    differences: list[Stage7RebuildDifference] = []
    frozen_manifest_view: _ManifestView | None = None
    rebuilt_manifest_view: _ManifestView | None = None
    frozen_audit_view: _AuditView | None = None
    rebuilt_audit_view: _AuditView | None = None
    frozen_identity: Stage5IdentityAudit | None = None
    rebuilt_identity: Stage5IdentityAudit | None = None
    audits_valid = False

    try:
        frozen_manifest_view = _manifest_view(frozen_manifest, reasons)
        rebuilt_manifest_view = _manifest_view(rebuilt_manifest, reasons)
        frozen_audit_view = _audit_view(frozen_audit, reasons)
        rebuilt_audit_view = _audit_view(rebuilt_audit, reasons)

        if frozen_manifest_view is not None and frozen_manifest_view.unknown_label_count:
            reasons.add(Stage7RebuildReason.UNKNOWN_LABEL_STATE)
        if rebuilt_manifest_view is not None and rebuilt_manifest_view.unknown_label_count:
            reasons.add(Stage7RebuildReason.UNKNOWN_LABEL_STATE)

        frozen_audit_valid = (
            frozen_manifest_view is not None
            and frozen_audit_view is not None
            and _audit_matches_manifest(frozen_audit_view, frozen_manifest_view)
        )
        rebuilt_audit_valid = (
            rebuilt_manifest_view is not None
            and rebuilt_audit_view is not None
            and _audit_matches_manifest(rebuilt_audit_view, rebuilt_manifest_view)
        )
        if not frozen_audit_valid:
            reasons.add(Stage7RebuildReason.FROZEN_AUDIT_INVALID)
        if not rebuilt_audit_valid:
            reasons.add(Stage7RebuildReason.REBUILT_AUDIT_INVALID)
        audits_valid = frozen_audit_valid and rebuilt_audit_valid

        if frozen_manifest_view is not None and rebuilt_manifest_view is not None:
            _compare_manifest_fields(
                frozen_manifest_view,
                rebuilt_manifest_view,
                reasons,
                differences,
            )
            if frozen_manifest_view.sha256 != rebuilt_manifest_view.sha256:
                reasons.add(Stage7RebuildReason.MANIFEST_HASH_MISMATCH)
            frozen_identity = audit_five_identity_split_isolation(
                frozen_manifest_view.manifest.records
            )
            rebuilt_identity = audit_five_identity_split_isolation(
                rebuilt_manifest_view.manifest.records
            )
            if (
                frozen_identity.status is not Stage5IdentityAuditStatus.PASS
                or rebuilt_identity.status is not Stage5IdentityAuditStatus.PASS
            ):
                reasons.add(Stage7RebuildReason.FIVE_IDENTITY_ISOLATION_FAILED)

        if (
            frozen_audit_view is not None
            and rebuilt_audit_view is not None
            and frozen_audit_view.payload != rebuilt_audit_view.payload
        ):
            reasons.add(Stage7RebuildReason.AUDIT_EVIDENCE_MISMATCH)
            differences.append(
                _difference(
                    kind=Stage7DifferenceKind.AUDIT,
                    field="evidence",
                    expected=frozen_audit_view.payload,
                    observed=rebuilt_audit_view.payload,
                )
            )
    except Exception:  # fail closed without serializing filesystem or identity details
        reasons.add(Stage7RebuildReason.VERIFICATION_EXCEPTION)

    return _make_result(
        frozen_manifest=frozen_manifest_view,
        rebuilt_manifest=rebuilt_manifest_view,
        frozen_audit=frozen_audit_view,
        rebuilt_audit=rebuilt_audit_view,
        frozen_identity_audit=frozen_identity,
        rebuilt_identity_audit=rebuilt_identity,
        audits_valid=audits_valid,
        reasons=reasons,
        differences=differences,
    )


def verify_stage7_package_rebuild(
    *,
    frozen_evidence: object,
    rebuilt_manifest: object,
    rebuilt_audit: object,
) -> Stage7RebuildVerification:
    """Verify a replay-produced Stage 2 evidence object against a rebuilt pair.

    A detached dictionary, descriptor, or arbitrary look-alike is rejected.  The
    :class:`Stage2PackageEvidence` remains serializable evidence and does not acquire
    root-access authority by being accepted here.
    """

    if not isinstance(frozen_evidence, Stage2PackageEvidence):
        return _make_result(
            frozen_manifest=None,
            rebuilt_manifest=None,
            frozen_audit=None,
            rebuilt_audit=None,
            frozen_identity_audit=None,
            rebuilt_identity_audit=None,
            audits_valid=False,
            reasons={
                Stage7RebuildReason.INPUT_INVALID,
                Stage7RebuildReason.FROZEN_PACKAGE_EVIDENCE_INVALID,
            },
            differences=[],
        )
    try:
        validated = Stage2PackageEvidence(
            manifest=frozen_evidence.manifest,
            audit=frozen_evidence.audit,
            descriptor=frozen_evidence.descriptor,
        )
        if validated != frozen_evidence:
            raise ValueError("Stage 2 evidence is not canonically normalized")
    except (AttributeError, TypeError, ValueError):
        return _make_result(
            frozen_manifest=None,
            rebuilt_manifest=None,
            frozen_audit=None,
            rebuilt_audit=None,
            frozen_identity_audit=None,
            rebuilt_identity_audit=None,
            audits_valid=False,
            reasons={Stage7RebuildReason.FROZEN_PACKAGE_EVIDENCE_INVALID},
            differences=[],
        )
    return verify_stage7_rebuild(
        frozen_manifest=validated.manifest,
        frozen_audit=validated.audit,
        rebuilt_manifest=rebuilt_manifest,
        rebuilt_audit=rebuilt_audit,
    )


__all__ = [
    "STAGE7_REBUILD_SCHEMA_VERSION",
    "Stage7DifferenceKind",
    "Stage7RebuildDifference",
    "Stage7RebuildReason",
    "Stage7RebuildStatus",
    "Stage7RebuildVerification",
    "verify_stage7_package_rebuild",
    "verify_stage7_rebuild",
]
