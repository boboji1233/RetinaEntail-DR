"""Fail-closed Stage 5 evidence verification and short-lived authorization.

The zero-I/O preflight accepts no paths.  The verifier reads only explicitly supplied
evidence files, rejects links/reparse points, and never stats, resolves, lists, or opens
the dataset root.  A root may be consumed only by code that revalidates the process-
bound authorization immediately before access.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Never

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken

from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseReviewDecision,
    LicenseReviewRecord,
    permit_validation_reasons,
)
from retinaentail.data.stage2_models import DatasetAdapterSpec
from retinaentail.evaluation.gate import Stage4LockedEvidence

STAGE5_PREFLIGHT_SCHEMA_VERSION: Final = "5.0.0"
STAGE5_PREREQUISITE_SCHEMA_VERSION: Final = "5.0.0"
MAX_STAGE5_AUTHORIZATION_TTL_SECONDS: Final = 900
DEFAULT_STAGE5_AUTHORIZATION_TTL_SECONDS: Final = 300
MAX_STAGE5_ROOT_BINDING_TTL_SECONDS: Final = 5

REQUIRED_STAGE5_REVIEW_PURPOSES: Final = (
    DataUsePurpose.READ_ONLY_AUDIT,
    DataUsePurpose.LESION_CALIBRATION,
    DataUsePurpose.LOCKED_TEST_EVALUATION,
)

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_RFC3339_PATTERN: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_REPARSE_POINT_ATTRIBUTE: Final = 0x0400
_AUTHORIZATION_TOKEN: Final = object()
_BOUND_ROOT_TOKEN: Final = object()
_PROCESS_TOKEN: Final = object()
_PROCESS_ID: Final = os.getpid()

_ARTIFACT_LIMITS: Final[dict[str, int]] = {
    "license_review": 512 * 1024,
    "adapter_spec": 64 * 1024 * 1024,
    "stage2_audit": 32 * 1024 * 1024,
    "frozen_data_card": 2 * 1024 * 1024,
    "manual_confirmation": 2 * 1024 * 1024,
    "stage3_package_summary": 2 * 1024 * 1024,
    "stage4_locked_evidence": 2 * 1024 * 1024,
}
_ARTIFACT_NAMES: Final = tuple(_ARTIFACT_LIMITS)
_LESION_LABEL_SEMANTICS: Final = frozenset(
    {
        "MICROANEURYSM_MASK",
        "HEMORRHAGE_MASK",
        "HARD_EXUDATE_MASK",
        "SOFT_EXUDATE_MASK",
        "LESION_DETECTION",
    }
)


class Stage5VerificationContext(StrEnum):
    """Keep synthetic Gate tests visibly distinct from real-data authority."""

    REAL_PREREQUISITE = "REAL_PREREQUISITE"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class Stage5PreflightStatus(StrEnum):
    BLOCKED = "BLOCKED"
    READY_FOR_PREREQUISITE_VERIFICATION = "READY_FOR_PREREQUISITE_VERIFICATION"


class Stage5PreflightReason(StrEnum):
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    APPROVED_REVIEW_NOT_PROVIDED = "APPROVED_REVIEW_NOT_PROVIDED"
    READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED = "READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED"
    LESION_CALIBRATION_PURPOSE_NOT_APPROVED = "LESION_CALIBRATION_PURPOSE_NOT_APPROVED"
    LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED = "LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED"
    DATASET_ROOT_NOT_PROVIDED = "DATASET_ROOT_NOT_PROVIDED"
    ADAPTER_SPEC_NOT_PROVIDED = "ADAPTER_SPEC_NOT_PROVIDED"
    STAGE2_PASS_AUDIT_NOT_PROVIDED = "STAGE2_PASS_AUDIT_NOT_PROVIDED"
    FROZEN_DATA_CARD_NOT_PROVIDED = "FROZEN_DATA_CARD_NOT_PROVIDED"
    VALID_STAGE2_ACCESS_PERMIT_NOT_PROVIDED = "VALID_STAGE2_ACCESS_PERMIT_NOT_PROVIDED"
    MANUAL_CONFIRMATION_NOT_PROVIDED = "MANUAL_CONFIRMATION_NOT_PROVIDED"
    VERIFIED_STAGE3_PACKAGE_SUMMARY_NOT_PROVIDED = "VERIFIED_STAGE3_PACKAGE_SUMMARY_NOT_PROVIDED"
    STAGE4_LOCKED_EVIDENCE_NOT_PROVIDED = "STAGE4_LOCKED_EVIDENCE_NOT_PROVIDED"


class Stage5PrerequisiteStatus(StrEnum):
    BLOCKED = "BLOCKED"
    VERIFIED = "VERIFIED"


class Stage5PrerequisiteReason(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    TTL_INVALID = "TTL_INVALID"
    ROOT_INVALID = "ROOT_INVALID"
    EVIDENCE_PATH_COLLISION = "EVIDENCE_PATH_COLLISION"
    EVIDENCE_INSIDE_DATASET_ROOT = "EVIDENCE_INSIDE_DATASET_ROOT"
    EVIDENCE_PATH_NOT_ABSOLUTE = "EVIDENCE_PATH_NOT_ABSOLUTE"
    EVIDENCE_LINK_OR_REPARSE_POINT = "EVIDENCE_LINK_OR_REPARSE_POINT"
    EVIDENCE_NOT_REGULAR_FILE = "EVIDENCE_NOT_REGULAR_FILE"
    EVIDENCE_TOO_LARGE = "EVIDENCE_TOO_LARGE"
    EVIDENCE_CHANGED_DURING_READ = "EVIDENCE_CHANGED_DURING_READ"
    EVIDENCE_DECODE_FAILED = "EVIDENCE_DECODE_FAILED"
    EVIDENCE_CONTRACT_INVALID = "EVIDENCE_CONTRACT_INVALID"
    DATASET_MISMATCH = "DATASET_MISMATCH"
    DATASET_VERSION_MISMATCH = "DATASET_VERSION_MISMATCH"
    REVIEW_NOT_APPROVED = "REVIEW_NOT_APPROVED"
    REQUIRED_REVIEW_PURPOSE_MISSING = "REQUIRED_REVIEW_PURPOSE_MISSING"
    STAGE2_AUDIT_NOT_PASS = "STAGE2_AUDIT_NOT_PASS"
    FROZEN_DATA_CARD_NOT_APPROVED_REAL_DATA = "FROZEN_DATA_CARD_NOT_APPROVED_REAL_DATA"
    MANUAL_CONFIRMATION_INCOMPLETE = "MANUAL_CONFIRMATION_INCOMPLETE"
    STAGE3_PACKAGE_NOT_VERIFIED = "STAGE3_PACKAGE_NOT_VERIFIED"
    STAGE3_CALIBRATION_NOT_SUCCESS = "STAGE3_CALIBRATION_NOT_SUCCESS"
    LOCKED_EVIDENCE_INVALID = "LOCKED_EVIDENCE_INVALID"
    HASH_MISMATCH = "HASH_MISMATCH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    STAGE2_ACCESS_PERMIT_INVALID = "STAGE2_ACCESS_PERMIT_INVALID"
    REAL_STAGE2_AUDIT_CAPABILITY_UNAVAILABLE = "REAL_STAGE2_AUDIT_CAPABILITY_UNAVAILABLE"
    REAL_STAGE3_PACKAGE_CAPABILITY_UNAVAILABLE = "REAL_STAGE3_PACKAGE_CAPABILITY_UNAVAILABLE"


class Stage5AuthorizationError(PermissionError):
    """Raised when a Stage 5 capability is invalid at its point of use."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 5 dataset authorization rejected: " + "; ".join(reasons))


class Stage5RootBindingError(PermissionError):
    """Raised when post-authorization physical root binding fails closed."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 5 root binding rejected: " + "; ".join(reasons))


def _require_dataset_id(value: object) -> str:
    if not isinstance(value, str) or _DATASET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("dataset_id must be a normalized identifier")
    return value


def _require_dataset_version(value: object) -> str:
    if not isinstance(value, str) or _DATASET_VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError("dataset_version must be a report-safe exact version")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:  # noqa: E721
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if "\x00" in result or "\r" in result or "\n" in result:
        raise ValueError(f"{name} contains forbidden control characters")
    return result


def _require_rfc3339(value: object, name: str) -> str:
    result = _require_nonempty(value, name)
    if _RFC3339_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{name} must be an RFC 3339 timestamp")
    parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return result


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    return result


def _exact_fields(
    value: object,
    *,
    name: str,
    fields: set[str],
) -> dict[str, object]:
    result = _mapping(value, name=name)
    if set(result) != fields:
        raise ValueError(f"{name} has missing or unknown fields")
    return result


def _sequence(value: object, *, name: str) -> list[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return list(value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _expected_preflight_reasons(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    approved_review_provided: bool,
    approved_review_purposes: tuple[DataUsePurpose, ...],
    dataset_root_provided: bool,
    adapter_spec_provided: bool,
    stage2_pass_audit_provided: bool,
    frozen_data_card_provided: bool,
    stage2_access_permit_provided: bool,
    manual_confirmation_provided: bool,
    stage3_package_summary_provided: bool,
    stage4_locked_evidence_provided: bool,
) -> tuple[Stage5PreflightReason, ...]:
    reasons: list[Stage5PreflightReason] = []
    if dataset_id is None:
        reasons.append(Stage5PreflightReason.DATASET_NOT_EXPLICIT)
    if dataset_version is None:
        reasons.append(Stage5PreflightReason.DATASET_VERSION_NOT_EXPLICIT)
    if not approved_review_provided:
        reasons.append(Stage5PreflightReason.APPROVED_REVIEW_NOT_PROVIDED)
    purpose_reasons = {
        DataUsePurpose.READ_ONLY_AUDIT: (
            Stage5PreflightReason.READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED
        ),
        DataUsePurpose.LESION_CALIBRATION: (
            Stage5PreflightReason.LESION_CALIBRATION_PURPOSE_NOT_APPROVED
        ),
        DataUsePurpose.LOCKED_TEST_EVALUATION: (
            Stage5PreflightReason.LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED
        ),
    }
    purpose_set = set(approved_review_purposes)
    for purpose in REQUIRED_STAGE5_REVIEW_PURPOSES:
        if purpose not in purpose_set:
            reasons.append(purpose_reasons[purpose])
    if not dataset_root_provided:
        reasons.append(Stage5PreflightReason.DATASET_ROOT_NOT_PROVIDED)
    if not adapter_spec_provided:
        reasons.append(Stage5PreflightReason.ADAPTER_SPEC_NOT_PROVIDED)
    if not stage2_pass_audit_provided:
        reasons.append(Stage5PreflightReason.STAGE2_PASS_AUDIT_NOT_PROVIDED)
    if not frozen_data_card_provided:
        reasons.append(Stage5PreflightReason.FROZEN_DATA_CARD_NOT_PROVIDED)
    if not stage2_access_permit_provided:
        reasons.append(Stage5PreflightReason.VALID_STAGE2_ACCESS_PERMIT_NOT_PROVIDED)
    if not manual_confirmation_provided:
        reasons.append(Stage5PreflightReason.MANUAL_CONFIRMATION_NOT_PROVIDED)
    if not stage3_package_summary_provided:
        reasons.append(Stage5PreflightReason.VERIFIED_STAGE3_PACKAGE_SUMMARY_NOT_PROVIDED)
    if not stage4_locked_evidence_provided:
        reasons.append(Stage5PreflightReason.STAGE4_LOCKED_EVIDENCE_NOT_PROVIDED)
    return tuple(reasons)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5PreflightReport:
    """Path-free input presence report; it can never grant root access."""

    status: Stage5PreflightStatus
    verification_context: Stage5VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    approved_review_provided: bool
    approved_review_purposes: tuple[DataUsePurpose, ...]
    dataset_root_provided: bool
    adapter_spec_provided: bool
    stage2_pass_audit_provided: bool
    frozen_data_card_provided: bool
    stage2_access_permit_provided: bool
    manual_confirmation_provided: bool
    stage3_package_summary_provided: bool
    stage4_locked_evidence_provided: bool
    reasons: tuple[Stage5PreflightReason, ...]
    schema_version: str = STAGE5_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE5_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 5 preflight schema_version")
        if not isinstance(self.status, Stage5PreflightStatus):
            raise TypeError("status must be Stage5PreflightStatus")
        if not isinstance(self.verification_context, Stage5VerificationContext):
            raise TypeError("verification_context must be Stage5VerificationContext")
        if self.dataset_id is not None:
            _require_dataset_id(self.dataset_id)
        if self.dataset_version is not None:
            _require_dataset_version(self.dataset_version)
        flags = (
            self.approved_review_provided,
            self.dataset_root_provided,
            self.adapter_spec_provided,
            self.stage2_pass_audit_provided,
            self.frozen_data_card_provided,
            self.stage2_access_permit_provided,
            self.manual_confirmation_provided,
            self.stage3_package_summary_provided,
            self.stage4_locked_evidence_provided,
        )
        if any(type(value) is not bool for value in flags):  # noqa: E721
            raise TypeError("Stage 5 preflight presence fields must be exact bool values")
        if not isinstance(self.approved_review_purposes, tuple) or not all(
            isinstance(item, DataUsePurpose) for item in self.approved_review_purposes
        ):
            raise TypeError("approved_review_purposes must be a tuple of DataUsePurpose")
        if len(set(self.approved_review_purposes)) != len(self.approved_review_purposes):
            raise ValueError("approved_review_purposes must be unique")
        if not all(isinstance(item, Stage5PreflightReason) for item in self.reasons):
            raise TypeError("reasons must contain Stage5PreflightReason values")
        expected = _expected_preflight_reasons(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            approved_review_provided=self.approved_review_provided,
            approved_review_purposes=self.approved_review_purposes,
            dataset_root_provided=self.dataset_root_provided,
            adapter_spec_provided=self.adapter_spec_provided,
            stage2_pass_audit_provided=self.stage2_pass_audit_provided,
            frozen_data_card_provided=self.frozen_data_card_provided,
            stage2_access_permit_provided=self.stage2_access_permit_provided,
            manual_confirmation_provided=self.manual_confirmation_provided,
            stage3_package_summary_provided=self.stage3_package_summary_provided,
            stage4_locked_evidence_provided=self.stage4_locked_evidence_provided,
        )
        expected_status = (
            Stage5PreflightStatus.BLOCKED
            if expected
            else Stage5PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
        )
        if self.status is not expected_status or self.reasons != expected:
            raise ValueError("Stage 5 preflight status/reasons conflict with input presence")

    @property
    def authorizes_data_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "approved_review_provided": self.approved_review_provided,
            "approved_review_purposes": [
                purpose.value for purpose in self.approved_review_purposes
            ],
            "dataset_root_provided": self.dataset_root_provided,
            "adapter_spec_provided": self.adapter_spec_provided,
            "stage2_pass_audit_provided": self.stage2_pass_audit_provided,
            "frozen_data_card_provided": self.frozen_data_card_provided,
            "stage2_access_permit_provided": self.stage2_access_permit_provided,
            "manual_confirmation_provided": self.manual_confirmation_provided,
            "stage3_package_summary_provided": self.stage3_package_summary_provided,
            "stage4_locked_evidence_provided": self.stage4_locked_evidence_provided,
            "reasons": [reason.value for reason in self.reasons],
            "test_tuning_prohibited": True,
            "prerequisites_verified": False,
            "authorizes_data_access": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "sample_count": 0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage5PreflightReport:
        expected = {
            "schema_version",
            "status",
            "verification_context",
            "dataset_id",
            "dataset_version",
            "approved_review_provided",
            "approved_review_purposes",
            "dataset_root_provided",
            "adapter_spec_provided",
            "stage2_pass_audit_provided",
            "frozen_data_card_provided",
            "stage2_access_permit_provided",
            "manual_confirmation_provided",
            "stage3_package_summary_provided",
            "stage4_locked_evidence_provided",
            "reasons",
            "test_tuning_prohibited",
            "prerequisites_verified",
            "authorizes_data_access",
            "dataset_root_probed",
            "real_data_read",
            "sample_count",
        }
        data = _exact_fields(value, name="Stage 5 preflight", fields=expected)
        zero_io = (
            data["test_tuning_prohibited"] is True
            and data["prerequisites_verified"] is False
            and data["authorizes_data_access"] is False
            and data["dataset_root_probed"] is False
            and data["real_data_read"] is False
            and data["sample_count"] == 0
        )
        if not zero_io:
            raise ValueError("Stage 5 preflight must preserve zero-I/O constants")
        purposes = tuple(
            DataUsePurpose(str(item))
            for item in _sequence(data["approved_review_purposes"], name="review purposes")
        )
        reasons = tuple(
            Stage5PreflightReason(str(item))
            for item in _sequence(data["reasons"], name="preflight reasons")
        )
        flags = tuple(
            data[name]
            for name in (
                "approved_review_provided",
                "dataset_root_provided",
                "adapter_spec_provided",
                "stage2_pass_audit_provided",
                "frozen_data_card_provided",
                "stage2_access_permit_provided",
                "manual_confirmation_provided",
                "stage3_package_summary_provided",
                "stage4_locked_evidence_provided",
            )
        )
        if any(type(item) is not bool for item in flags):  # noqa: E721
            raise ValueError("preflight presence fields must be booleans")
        dataset_id = data["dataset_id"]
        dataset_version = data["dataset_version"]
        if dataset_id is not None and not isinstance(dataset_id, str):
            raise ValueError("dataset_id must be a string or null")
        if dataset_version is not None and not isinstance(dataset_version, str):
            raise ValueError("dataset_version must be a string or null")
        return cls(
            schema_version=str(data["schema_version"]),
            status=Stage5PreflightStatus(str(data["status"])),
            verification_context=Stage5VerificationContext(str(data["verification_context"])),
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            approved_review_provided=bool(flags[0]),
            approved_review_purposes=purposes,
            dataset_root_provided=bool(flags[1]),
            adapter_spec_provided=bool(flags[2]),
            stage2_pass_audit_provided=bool(flags[3]),
            frozen_data_card_provided=bool(flags[4]),
            stage2_access_permit_provided=bool(flags[5]),
            manual_confirmation_provided=bool(flags[6]),
            stage3_package_summary_provided=bool(flags[7]),
            stage4_locked_evidence_provided=bool(flags[8]),
            reasons=reasons,
        )


def assess_stage5_preflight(
    *,
    verification_context: Stage5VerificationContext,
    dataset_id: str | None,
    dataset_version: str | None,
    approved_review_provided: bool,
    approved_review_purposes: tuple[DataUsePurpose, ...],
    dataset_root_provided: bool,
    adapter_spec_provided: bool,
    stage2_pass_audit_provided: bool,
    frozen_data_card_provided: bool,
    stage2_access_permit_provided: bool,
    manual_confirmation_provided: bool,
    stage3_package_summary_provided: bool,
    stage4_locked_evidence_provided: bool,
) -> Stage5PreflightReport:
    """Assess only identifiers, purposes, and presence flags; accept no paths."""

    if not isinstance(verification_context, Stage5VerificationContext):
        raise TypeError("verification_context must be Stage5VerificationContext")
    flags = (
        approved_review_provided,
        dataset_root_provided,
        adapter_spec_provided,
        stage2_pass_audit_provided,
        frozen_data_card_provided,
        stage2_access_permit_provided,
        manual_confirmation_provided,
        stage3_package_summary_provided,
        stage4_locked_evidence_provided,
    )
    if any(type(value) is not bool for value in flags):  # noqa: E721
        raise TypeError("Stage 5 preflight presence fields must be exact bool values")
    if not isinstance(approved_review_purposes, tuple) or not all(
        isinstance(item, DataUsePurpose) for item in approved_review_purposes
    ):
        raise TypeError("approved_review_purposes must be a tuple of DataUsePurpose")
    reasons = _expected_preflight_reasons(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_provided=approved_review_provided,
        approved_review_purposes=approved_review_purposes,
        dataset_root_provided=dataset_root_provided,
        adapter_spec_provided=adapter_spec_provided,
        stage2_pass_audit_provided=stage2_pass_audit_provided,
        frozen_data_card_provided=frozen_data_card_provided,
        stage2_access_permit_provided=stage2_access_permit_provided,
        manual_confirmation_provided=manual_confirmation_provided,
        stage3_package_summary_provided=stage3_package_summary_provided,
        stage4_locked_evidence_provided=stage4_locked_evidence_provided,
    )
    return Stage5PreflightReport(
        status=(
            Stage5PreflightStatus.BLOCKED
            if reasons
            else Stage5PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_provided=approved_review_provided,
        approved_review_purposes=approved_review_purposes,
        dataset_root_provided=dataset_root_provided,
        adapter_spec_provided=adapter_spec_provided,
        stage2_pass_audit_provided=stage2_pass_audit_provided,
        frozen_data_card_provided=frozen_data_card_provided,
        stage2_access_permit_provided=stage2_access_permit_provided,
        manual_confirmation_provided=manual_confirmation_provided,
        stage3_package_summary_provided=stage3_package_summary_provided,
        stage4_locked_evidence_provided=stage4_locked_evidence_provided,
        reasons=reasons,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5EvidencePaths:
    """Seven explicit evidence files; no dataset path is inferred from them."""

    license_review: Path
    adapter_spec: Path
    stage2_audit: Path
    frozen_data_card: Path
    manual_confirmation: Path
    stage3_package_summary: Path
    stage4_locked_evidence: Path

    def __post_init__(self) -> None:
        values = self.as_dict()
        if not all(isinstance(value, Path) for value in values.values()):
            raise TypeError("Stage 5 evidence paths must be pathlib.Path values")

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in _ARTIFACT_NAMES}


class _EvidenceFailure(Exception):
    def __init__(self, reason: Stage5PrerequisiteReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _is_link_or_reparse(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_link_components(path: Path) -> None:
    for candidate in (*reversed(path.parents), path):
        try:
            metadata = os.lstat(candidate)
        except OSError as error:
            raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_NOT_REGULAR_FILE) from error
        if _is_link_or_reparse(metadata):
            raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_LINK_OR_REPARSE_POINT)


def _read_bounded_evidence(path: Path, *, limit: int) -> bytes:
    if not path.is_absolute():
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_PATH_NOT_ABSOLUTE)
    _reject_link_components(path)
    try:
        before = os.lstat(path)
    except OSError as error:
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_NOT_REGULAR_FILE) from error
    if not stat.S_ISREG(before.st_mode):
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_NOT_REGULAR_FILE)
    if before.st_size > limit:
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_TOO_LARGE)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_NOT_REGULAR_FILE) from error
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(before):
            raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_CHANGED_DURING_READ)
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_TOO_LARGE)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_CHANGED_DURING_READ) from error
    if _file_identity(before) != _file_identity(after_open) or _file_identity(
        before
    ) != _file_identity(after_path):
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_CHANGED_DURING_READ)
    return b"".join(chunks)


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("JSON object keys must be strings")
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _check_yaml_node(node: Node, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("YAML nesting is too deep")
    if isinstance(node, MappingNode):
        seen: set[tuple[str, str]] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode):
                raise ValueError("YAML mapping keys must be scalars")
            identity = (key_node.tag, key_node.value)
            if identity in seen:
                raise ValueError(f"duplicate YAML key: {key_node.value}")
            seen.add(identity)
            _check_yaml_node(value_node, depth=depth + 1)
    elif isinstance(node, SequenceNode):
        for child in node.value:
            _check_yaml_node(child, depth=depth + 1)


def _normalize_jsonlike(value: object, *, depth: int = 0) -> object:
    if depth > 64:
        raise ValueError("evidence nesting is too deep")
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite evidence values are forbidden")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or raw_key in result:
                raise ValueError("evidence object keys must be unique strings")
            result[raw_key] = _normalize_jsonlike(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_normalize_jsonlike(item, depth=depth + 1) for item in value]
    raise ValueError("evidence contains a non-JSON value")


def _decode_evidence(payload: bytes, *, allow_yaml: bool) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_DECODE_FAILED) from error
    try:
        decoded: object = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        return _normalize_jsonlike(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        if not allow_yaml:
            raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_DECODE_FAILED) from None
    try:
        tokens = yaml.scan(text)
        if any(isinstance(token, AliasToken | AnchorToken) for token in tokens):
            raise ValueError("YAML anchors and aliases are forbidden")
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        if node is None:
            raise ValueError("empty YAML evidence")
        _check_yaml_node(node)
        yaml_value: object = yaml.safe_load(text)
        return _normalize_jsonlike(yaml_value)
    except (TypeError, ValueError, yaml.YAMLError) as error:
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_DECODE_FAILED) from error


def _validated_audit(value: object) -> dict[str, object]:
    data = _exact_fields(
        value,
        name="Stage 2 audit",
        fields={
            "schema_version",
            "status",
            "dataset_id",
            "dataset_version",
            "manifest_sha256",
            "summary",
            "issues",
        },
    )
    if data["schema_version"] != "2.0.0":
        raise ValueError("unsupported Stage 2 audit schema")
    if data["status"] != "PASS":
        raise _EvidenceFailure(Stage5PrerequisiteReason.STAGE2_AUDIT_NOT_PASS)
    _require_dataset_id(data["dataset_id"])
    _require_dataset_version(data["dataset_version"])
    _require_sha256(data["manifest_sha256"], "manifest_sha256")
    summary = _exact_fields(
        data["summary"],
        name="audit summary",
        fields={
            "record_count",
            "image_files_checked",
            "label_files_checked",
            "mask_files_decoded",
            "unknown_label_count",
            "issue_counts",
        },
    )
    record_count = _require_positive_int(summary["record_count"], "record_count")
    if (
        _require_nonnegative_int(summary["image_files_checked"], "image_files_checked")
        != record_count
    ):
        raise ValueError("PASS audit must check every image")
    for name in ("label_files_checked", "mask_files_decoded", "unknown_label_count"):
        _require_nonnegative_int(summary[name], name)
    if _mapping(summary["issue_counts"], name="issue_counts"):
        raise ValueError("PASS audit cannot retain issue counts")
    if _sequence(data["issues"], name="audit issues"):
        raise ValueError("PASS audit cannot retain unresolved issues")
    return data


def _validated_card(value: object) -> dict[str, object]:
    fields = {
        "schema_version",
        "scope",
        "dataset_id",
        "dataset_version",
        "manifest_sha256",
        "registry_sha256",
        "adapter_spec_sha256",
        "audit_sha256",
        "audit_schema_version",
        "license_review_sha256",
        "license_review_schema_version",
        "license_terms_sha256",
        "license_evidence_sha256",
        "reviewer_fingerprint",
        "reviewed_at",
        "label_semantics",
        "label_semantics_evidence_sha256",
        "identity_confirmations",
        "identity_evidence_sha256",
        "split_counts",
        "sample_count",
        "config_sha256",
        "data_sha256",
        "model_sha256",
    }
    data = _exact_fields(value, name="frozen data card", fields=fields)
    if data["schema_version"] != "3.0.0":
        raise ValueError("unsupported frozen data card schema")
    if data["scope"] != "APPROVED_REAL_DATA":
        raise _EvidenceFailure(Stage5PrerequisiteReason.FROZEN_DATA_CARD_NOT_APPROVED_REAL_DATA)
    _require_dataset_id(data["dataset_id"])
    _require_dataset_version(data["dataset_version"])
    digest_fields = {
        "manifest_sha256",
        "registry_sha256",
        "adapter_spec_sha256",
        "audit_sha256",
        "license_review_sha256",
        "license_terms_sha256",
        "license_evidence_sha256",
        "reviewer_fingerprint",
        "label_semantics_evidence_sha256",
        "identity_evidence_sha256",
        "config_sha256",
        "data_sha256",
        "model_sha256",
    }
    for name in digest_fields:
        _require_sha256(data[name], name)
    if data["audit_schema_version"] != "2.0.0" or data["license_review_schema_version"] != (
        "1.0.0"
    ):
        raise ValueError("frozen data card references unsupported evidence schemas")
    _require_rfc3339(data["reviewed_at"], "reviewed_at")
    semantics = _sequence(data["label_semantics"], name="label_semantics")
    if not semantics or len(set(map(str, semantics))) != len(semantics):
        raise ValueError("label_semantics must be non-empty and unique")
    if not all(isinstance(item, str) and item in _LESION_LABEL_SEMANTICS for item in semantics):
        raise ValueError("frozen data card contains unsupported label semantics")
    confirmations = _exact_fields(
        data["identity_confirmations"],
        name="identity confirmations",
        fields={
            "source_image_id",
            "patient_id",
            "eye_id",
            "followup_id",
            "intervention_pair_id",
            "split_isolation",
        },
    )
    if not all(_require_bool(value, name) for name, value in confirmations.items()):
        raise ValueError("all frozen identity confirmations must be true")
    counts = _exact_fields(
        data["split_counts"],
        name="split counts",
        fields={"train", "calibration", "test"},
    )
    total = sum(_require_positive_int(counts[name], f"{name} count") for name in counts)
    if _require_positive_int(data["sample_count"], "sample_count") != total:
        raise ValueError("sample_count must equal the three split counts")
    if data["data_sha256"] != data["manifest_sha256"]:
        raise ValueError("card data_sha256 must equal manifest_sha256")
    return data


def _validated_manual_confirmation(value: object) -> dict[str, object]:
    data = _exact_fields(
        value,
        name="manual confirmation",
        fields={
            "schema_version",
            "record_type",
            "scope",
            "dataset_id",
            "dataset_version",
            "confirmed_by",
            "confirmed_at",
            "source_image_id_confirmed",
            "patient_id_confirmed",
            "eye_id_confirmed",
            "followup_id_confirmed",
            "intervention_pair_id_confirmed",
            "label_semantics_confirmed",
            "split_isolation_confirmed",
            "identity_evidence_sha256",
            "label_semantics_evidence_sha256",
        },
    )
    if (
        data["schema_version"] != "5.0.0"
        or data["record_type"] != "MANUAL_IDENTITY_LABEL_CONFIRMATION"
        or data["scope"] != "APPROVED_REAL_DATA"
    ):
        raise ValueError("manual confirmation header is invalid")
    _require_dataset_id(data["dataset_id"])
    _require_dataset_version(data["dataset_version"])
    _require_nonempty(data["confirmed_by"], "confirmed_by")
    _require_rfc3339(data["confirmed_at"], "confirmed_at")
    for name in (
        "source_image_id_confirmed",
        "patient_id_confirmed",
        "eye_id_confirmed",
        "followup_id_confirmed",
        "intervention_pair_id_confirmed",
        "label_semantics_confirmed",
        "split_isolation_confirmed",
    ):
        if not _require_bool(data[name], name):
            raise _EvidenceFailure(Stage5PrerequisiteReason.MANUAL_CONFIRMATION_INCOMPLETE)
    _require_sha256(data["identity_evidence_sha256"], "identity_evidence_sha256")
    _require_sha256(data["label_semantics_evidence_sha256"], "label_semantics_evidence_sha256")
    return data


def _validated_stage3_summary(value: object) -> dict[str, object]:
    data = _exact_fields(
        value,
        name="verified Stage 3 package summary",
        fields={
            "schema_version",
            "record_type",
            "scope",
            "dataset_id",
            "dataset_version",
            "verification_status",
            "package_sha256",
            "package_manifest_sha256",
            "package_verifier_sha256",
            "package_verifier_version",
            "independent_verifier_boundary",
            "verified_at",
            "verified_file_count",
            "verified_byte_count",
            "calibration_artifact_sha256",
            "calibration_status",
            "alpha",
            "positive_threshold",
            "negative_threshold",
            "positive_sample_count",
            "negative_sample_count",
            "total_sample_count",
            "config_hash",
            "data_hash",
            "model_hash",
            "positive_witness_replay_verified",
            "negative_witness_replay_verified",
            "thresholds_locked_before_test",
            "test_split_used_for_calibration",
            "test_tuning_prohibited",
        },
    )
    if (
        data["schema_version"] != "5.0.0"
        or data["record_type"] != "VERIFIED_STAGE3_CALIBRATION_PACKAGE"
        or data["scope"] != "APPROVED_REAL_DATA"
        or data["independent_verifier_boundary"] != "COMPLETE_VERIFIED_STAGE3_PACKAGE"
    ):
        raise ValueError("verified Stage 3 package summary header is invalid")
    _require_dataset_id(data["dataset_id"])
    _require_dataset_version(data["dataset_version"])
    if data["verification_status"] != "VERIFIED":
        raise _EvidenceFailure(Stage5PrerequisiteReason.STAGE3_PACKAGE_NOT_VERIFIED)
    if data["calibration_status"] != "SUCCESS":
        raise _EvidenceFailure(Stage5PrerequisiteReason.STAGE3_CALIBRATION_NOT_SUCCESS)
    for name in (
        "package_sha256",
        "package_manifest_sha256",
        "package_verifier_sha256",
        "calibration_artifact_sha256",
        "config_hash",
        "data_hash",
        "model_hash",
    ):
        _require_sha256(data[name], name)
    _require_nonempty(data["package_verifier_version"], "package_verifier_version")
    _require_rfc3339(data["verified_at"], "verified_at")
    _require_positive_int(data["verified_file_count"], "verified_file_count")
    _require_positive_int(data["verified_byte_count"], "verified_byte_count")
    alpha = _require_number(data["alpha"], "alpha", positive=True)
    if alpha >= 1:
        raise ValueError("alpha must be less than one")
    _require_number(data["positive_threshold"], "positive_threshold")
    _require_number(data["negative_threshold"], "negative_threshold")
    positive_count = _require_positive_int(data["positive_sample_count"], "positive count")
    negative_count = _require_positive_int(data["negative_sample_count"], "negative count")
    total_count = _require_positive_int(data["total_sample_count"], "total count")
    if total_count != positive_count + negative_count:
        raise ValueError("Stage 3 calibration counts do not add up")
    required_true = (
        "positive_witness_replay_verified",
        "negative_witness_replay_verified",
        "thresholds_locked_before_test",
        "test_tuning_prohibited",
    )
    if not all(_require_bool(data[name], name) for name in required_true):
        raise ValueError("Stage 3 summary has an unlocked or invalid provenance flag")
    if _require_bool(data["test_split_used_for_calibration"], "test_split_used_for_calibration"):
        raise ValueError("test split cannot be used for calibration")
    return data


def _dataset_version_matches(
    value: Mapping[str, object], dataset_id: str, dataset_version: str
) -> tuple[bool, bool]:
    return value.get("dataset_id") == dataset_id, value.get("dataset_version") == dataset_version


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5PrerequisiteReport:
    """Path-redacted verification result; this report is not a capability."""

    status: Stage5PrerequisiteStatus
    verification_context: Stage5VerificationContext
    dataset_id: str
    dataset_version: str
    purpose: DataUsePurpose
    reasons: tuple[Stage5PrerequisiteReason, ...]
    evidence_sha256: tuple[tuple[str, str], ...]
    evidence_files_read: int
    locked_evidence_binding_sha256: str | None
    stage3_package_sha256: str | None
    config_sha256: str | None
    data_sha256: str | None
    model_sha256: str | None
    root_binding_sha256: str | None
    authorization_ttl_seconds: int | None
    schema_version: str = STAGE5_PREREQUISITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE5_PREREQUISITE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 5 prerequisite schema_version")
        if not isinstance(self.status, Stage5PrerequisiteStatus):
            raise TypeError("status must be Stage5PrerequisiteStatus")
        if not isinstance(self.verification_context, Stage5VerificationContext):
            raise TypeError("verification_context must be Stage5VerificationContext")
        _require_dataset_id(self.dataset_id)
        _require_dataset_version(self.dataset_version)
        if self.purpose is not DataUsePurpose.LOCKED_TEST_EVALUATION:
            raise ValueError("Stage 5 authorization purpose must be locked_test_evaluation")
        if not all(isinstance(reason, Stage5PrerequisiteReason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage5PrerequisiteReason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("prerequisite reasons must be unique")
        normalized_hashes = tuple(sorted(self.evidence_sha256))
        if normalized_hashes != self.evidence_sha256:
            raise ValueError("evidence hashes must be sorted")
        names: set[str] = set()
        for name, digest in normalized_hashes:
            if name not in _ARTIFACT_NAMES or name in names:
                raise ValueError("evidence hash names must be known and unique")
            names.add(name)
            _require_sha256(digest, f"{name} digest")
        if not 0 <= self.evidence_files_read <= len(_ARTIFACT_NAMES):
            raise ValueError("evidence_files_read is outside its bounded range")
        for name in (
            "locked_evidence_binding_sha256",
            "stage3_package_sha256",
            "config_sha256",
            "data_sha256",
            "model_sha256",
            "root_binding_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        if self.status is Stage5PrerequisiteStatus.VERIFIED:
            if self.reasons or len(normalized_hashes) != len(_ARTIFACT_NAMES):
                raise ValueError("VERIFIED requires all evidence hashes and no reasons")
            required = (
                self.locked_evidence_binding_sha256,
                self.stage3_package_sha256,
                self.config_sha256,
                self.data_sha256,
                self.model_sha256,
                self.root_binding_sha256,
            )
            if any(value is None for value in required):
                raise ValueError("VERIFIED requires complete lock and provenance bindings")
            if self.evidence_files_read != len(_ARTIFACT_NAMES):
                raise ValueError("VERIFIED requires every evidence file to be read")
            if (
                type(self.authorization_ttl_seconds) is not int  # noqa: E721
                or not 1 <= self.authorization_ttl_seconds <= MAX_STAGE5_AUTHORIZATION_TTL_SECONDS
            ):
                raise ValueError("VERIFIED requires a bounded authorization TTL")
        else:
            if not self.reasons:
                raise ValueError("BLOCKED requires at least one reason")
            if any(
                value is not None
                for value in (
                    self.locked_evidence_binding_sha256,
                    self.stage3_package_sha256,
                    self.config_sha256,
                    self.data_sha256,
                    self.model_sha256,
                    self.root_binding_sha256,
                    self.authorization_ttl_seconds,
                )
            ):
                raise ValueError("BLOCKED cannot contain authorization bindings")

    @property
    def prerequisites_verified(self) -> bool:
        return self.status is Stage5PrerequisiteStatus.VERIFIED

    @property
    def authorizes_data_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        verified = self.prerequisites_verified
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "purpose": self.purpose.value,
            "reasons": [reason.value for reason in self.reasons],
            "evidence_sha256": dict(self.evidence_sha256),
            "evidence_files_read": self.evidence_files_read,
            "locked_evidence_binding_sha256": self.locked_evidence_binding_sha256,
            "stage3_package_sha256": self.stage3_package_sha256,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "model_sha256": self.model_sha256,
            "root_binding_sha256": self.root_binding_sha256,
            "authorization_ttl_seconds": self.authorization_ttl_seconds,
            "prerequisites_verified": verified,
            "authorization_issued": verified,
            "authorizes_data_access": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "sample_count": 0,
            "test_tuning_prohibited": True,
            "thresholds_reestimated": False,
            "test_split_used_for_calibration": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage5PrerequisiteReport:
        expected = {
            "schema_version",
            "status",
            "verification_context",
            "dataset_id",
            "dataset_version",
            "purpose",
            "reasons",
            "evidence_sha256",
            "evidence_files_read",
            "locked_evidence_binding_sha256",
            "stage3_package_sha256",
            "config_sha256",
            "data_sha256",
            "model_sha256",
            "root_binding_sha256",
            "authorization_ttl_seconds",
            "prerequisites_verified",
            "authorization_issued",
            "authorizes_data_access",
            "dataset_root_probed",
            "real_data_read",
            "sample_count",
            "test_tuning_prohibited",
            "thresholds_reestimated",
            "test_split_used_for_calibration",
        }
        data = _exact_fields(value, name="Stage 5 prerequisite report", fields=expected)
        status = Stage5PrerequisiteStatus(str(data["status"]))
        verified = status is Stage5PrerequisiteStatus.VERIFIED
        constants_ok = (
            data["prerequisites_verified"] is verified
            and data["authorization_issued"] is verified
            and data["authorizes_data_access"] is False
            and data["dataset_root_probed"] is False
            and data["real_data_read"] is False
            and data["sample_count"] == 0
            and data["test_tuning_prohibited"] is True
            and data["thresholds_reestimated"] is False
            and data["test_split_used_for_calibration"] is False
        )
        if not constants_ok:
            raise ValueError("Stage 5 prerequisite report constants conflict with status")
        raw_hashes = _mapping(data["evidence_sha256"], name="evidence_sha256")
        hashes = tuple(sorted((name, str(digest)) for name, digest in raw_hashes.items()))
        raw_reasons = _sequence(data["reasons"], name="prerequisite reasons")
        optional_hash_names = (
            "locked_evidence_binding_sha256",
            "stage3_package_sha256",
            "config_sha256",
            "data_sha256",
            "model_sha256",
            "root_binding_sha256",
        )
        optional_hashes: dict[str, str | None] = {}
        for name in optional_hash_names:
            item = data[name]
            if item is not None and not isinstance(item, str):
                raise ValueError(f"{name} must be a string or null")
            optional_hashes[name] = item
        count = data["evidence_files_read"]
        ttl = data["authorization_ttl_seconds"]
        if type(count) is not int or (ttl is not None and type(ttl) is not int):  # noqa: E721
            raise ValueError("report count and TTL fields must be integers or null")
        assert isinstance(count, int)
        assert ttl is None or isinstance(ttl, int)
        return cls(
            schema_version=str(data["schema_version"]),
            status=status,
            verification_context=Stage5VerificationContext(str(data["verification_context"])),
            dataset_id=str(data["dataset_id"]),
            dataset_version=str(data["dataset_version"]),
            purpose=DataUsePurpose(str(data["purpose"])),
            reasons=tuple(Stage5PrerequisiteReason(str(item)) for item in raw_reasons),
            evidence_sha256=hashes,
            evidence_files_read=count,
            locked_evidence_binding_sha256=optional_hashes["locked_evidence_binding_sha256"],
            stage3_package_sha256=optional_hashes["stage3_package_sha256"],
            config_sha256=optional_hashes["config_sha256"],
            data_sha256=optional_hashes["data_sha256"],
            model_sha256=optional_hashes["model_sha256"],
            root_binding_sha256=optional_hashes["root_binding_sha256"],
            authorization_ttl_seconds=ttl,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5DataAuthorization:
    """Prerequisite capability that still cannot authorize a filesystem open.

    A consumer must call :func:`bind_authorized_root` after validating this object.
    The primitive snapshot digests make mutation of nested permit/spec/lock objects
    detectable even if their own dataclass equality is compromised by in-place edits.
    """

    dataset_id: str
    dataset_version: str
    purpose: DataUsePurpose
    verification_context: Stage5VerificationContext
    _root: Path = field(repr=False, compare=False)
    adapter_spec: DatasetAdapterSpec
    locked_evidence: Stage4LockedEvidence
    stage2_access_permit: DatasetAccessPermit | None
    evidence_sha256: tuple[tuple[str, str], ...]
    evidence_binding_sha256: str
    registry_snapshot_sha256: str | None
    license_review_snapshot_sha256: str
    adapter_spec_snapshot_sha256: str
    locked_evidence_snapshot_sha256: str
    stage2_permit_snapshot_sha256: str | None
    root_binding_sha256: str
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _AUTHORIZATION_TOKEN or self._process_token is not _PROCESS_TOKEN:
            raise TypeError(
                "Stage5DataAuthorization can only be issued by verify_stage5_prerequisites"
            )

    @property
    def authorizes_file_open(self) -> bool:
        """Prerequisite authorization alone never crosses the root-I/O boundary."""

        return False


_ISSUED_AUTHORIZATIONS: dict[
    int,
    tuple[Stage5DataAuthorization, tuple[object, ...]],
] = {}


def _root_binding(
    *,
    dataset_id: str,
    dataset_version: str,
    root: Path,
    context: Stage5VerificationContext,
) -> str:
    return _canonical_sha256(
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "purpose": DataUsePurpose.LOCKED_TEST_EVALUATION.value,
            "verification_context": context.value,
            "root": root.as_posix(),
        }
    )


def _authorization_fingerprint(value: Stage5DataAuthorization) -> tuple[object, ...]:
    return (
        value.dataset_id,
        value.dataset_version,
        value.purpose,
        value.verification_context,
        value._root.as_posix(),
        _canonical_sha256(value.adapter_spec.to_dict()),
        value.locked_evidence.binding_sha256(),
        _live_permit_snapshot_sha256(value.stage2_access_permit),
        value.evidence_sha256,
        value.evidence_binding_sha256,
        value.registry_snapshot_sha256,
        value.license_review_snapshot_sha256,
        value.adapter_spec_snapshot_sha256,
        value.locked_evidence_snapshot_sha256,
        value.stage2_permit_snapshot_sha256,
        value.root_binding_sha256,
        value.issued_process_id,
        value.issued_monotonic_ns,
        value.expires_monotonic_ns,
        value.ttl_seconds,
        value._process_token,
        value._token,
    )


def authorization_validation_reasons(
    value: object,
    *,
    required_context: Stage5VerificationContext | None = None,
) -> tuple[str, ...]:
    """Revalidate primitive snapshots and expiry without touching the dataset root."""

    if not isinstance(value, Stage5DataAuthorization):
        return ("authorization has an invalid type",)
    issued = _ISSUED_AUTHORIZATIONS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("authorization was not issued by the Stage 5 Gate",)
    try:
        if issued[1] != _authorization_fingerprint(value):
            return ("authorization bindings changed after issuance",)
        reasons: list[str] = []
        if value._token is not _AUTHORIZATION_TOKEN or value._process_token is not _PROCESS_TOKEN:
            reasons.append("authorization issuance identity is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("authorization belongs to a different process")
        if value.purpose is not DataUsePurpose.LOCKED_TEST_EVALUATION:
            reasons.append("authorization purpose is not locked_test_evaluation")
        if required_context is not None and value.verification_context is not required_context:
            reasons.append(f"authorization context must be {required_context.value}")
        if value.verification_context is Stage5VerificationContext.REAL_PREREQUISITE:
            if value.stage2_access_permit is None:
                reasons.append("real authorization is missing its Stage 2 access permit")
            else:
                permit_reasons = permit_validation_reasons(value.stage2_access_permit)
                if permit_reasons:
                    reasons.append("Stage 2 access permit is no longer valid")
                elif (
                    value.stage2_access_permit.purpose is not DataUsePurpose.LOCKED_TEST_EVALUATION
                    or value.stage2_access_permit.root != value._root
                    or value.stage2_access_permit.registry.dataset_id != value.dataset_id
                    or value.stage2_access_permit.registry.dataset_version != value.dataset_version
                    or _canonical_sha256(value.stage2_access_permit.spec.to_dict())
                    != dict(value.evidence_sha256).get("adapter_spec")
                ):
                    reasons.append("Stage 2 access permit does not match Stage 5 bindings")
        elif value.stage2_access_permit is not None:
            reasons.append("synthetic authorization cannot carry a real-data permit")
        evidence = dict(value.evidence_sha256)
        if (
            len(evidence) != len(_ARTIFACT_NAMES)
            or set(evidence) != set(_ARTIFACT_NAMES)
            or len(evidence) != len(value.evidence_sha256)
            or any(_SHA256_PATTERN.fullmatch(digest) is None for digest in evidence.values())
        ):
            reasons.append("authorization evidence digest set is invalid")
        if value.evidence_binding_sha256 != _canonical_sha256(evidence):
            reasons.append("authorization evidence binding changed after issuance")
        if not value._root.is_absolute() or ".." in value._root.parts:
            reasons.append("authorization root binding is invalid")
        expected_root_binding = _root_binding(
            dataset_id=value.dataset_id,
            dataset_version=value.dataset_version,
            root=value._root,
            context=value.verification_context,
        )
        if value.root_binding_sha256 != expected_root_binding:
            reasons.append("authorization root binding does not match")
        if not 1 <= value.ttl_seconds <= MAX_STAGE5_AUTHORIZATION_TTL_SECONDS:
            reasons.append("authorization TTL is invalid")
        expected_duration = value.ttl_seconds * 1_000_000_000
        if value.expires_monotonic_ns - value.issued_monotonic_ns != expected_duration:
            reasons.append("authorization expiry binding does not match its TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("authorization has expired")
        live_spec_sha256 = _canonical_sha256(value.adapter_spec.to_dict())
        if (
            live_spec_sha256 != value.adapter_spec_snapshot_sha256
            or live_spec_sha256 != evidence.get("adapter_spec")
        ):
            reasons.append("authorization adapter spec does not match verified evidence")
        live_lock_sha256 = value.locked_evidence.binding_sha256()
        if (
            live_lock_sha256 != value.locked_evidence_snapshot_sha256
            or live_lock_sha256 != evidence.get("stage4_locked_evidence")
            or value.locked_evidence.dataset_id != value.dataset_id
            or value.locked_evidence.dataset_version != value.dataset_version
        ):
            reasons.append("authorization lock does not match its frozen snapshot")
        if value.license_review_snapshot_sha256 != evidence.get("license_review"):
            reasons.append("authorization review snapshot does not match verified evidence")
        if value.verification_context is Stage5VerificationContext.REAL_PREREQUISITE:
            permit = value.stage2_access_permit
            if permit is not None:
                live_registry_sha256 = _registry_snapshot_sha256(permit)
                live_review_sha256 = _canonical_sha256(permit.review.to_dict())
                live_permit_sha256 = _live_permit_snapshot_sha256(permit)
                if (
                    value.registry_snapshot_sha256 != live_registry_sha256
                    or value.license_review_snapshot_sha256 != live_review_sha256
                    or value.stage2_permit_snapshot_sha256 != live_permit_sha256
                ):
                    reasons.append("authorization Stage 2 snapshots changed after issuance")
        elif (
            value.registry_snapshot_sha256 is not None
            or value.stage2_permit_snapshot_sha256 is not None
        ):
            reasons.append("synthetic authorization contains real registry snapshots")
    except (AttributeError, TypeError, ValueError):
        return ("authorization fields are invalid",)
    return tuple(reasons)


def require_stage5_authorization(
    value: object,
    *,
    required_context: Stage5VerificationContext = Stage5VerificationContext.REAL_PREREQUISITE,
) -> Stage5DataAuthorization:
    """Validate prerequisites only; this still does not authorize any root open."""

    reasons = authorization_validation_reasons(value, required_context=required_context)
    if reasons:
        raise Stage5AuthorizationError(reasons)
    assert isinstance(value, Stage5DataAuthorization)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5VerificationOutcome:
    report: Stage5PrerequisiteReport
    authorization: Stage5DataAuthorization | None

    def __post_init__(self) -> None:
        if not isinstance(self.report, Stage5PrerequisiteReport):
            raise TypeError("report must be Stage5PrerequisiteReport")
        if self.report.status is Stage5PrerequisiteStatus.VERIFIED:
            if not isinstance(self.authorization, Stage5DataAuthorization):
                raise ValueError("VERIFIED outcome requires an authorization")
            if authorization_validation_reasons(self.authorization):
                raise ValueError("VERIFIED outcome authorization is invalid")
            authorization = self.authorization
            bindings_match = (
                self.report.verification_context is authorization.verification_context
                and self.report.dataset_id == authorization.dataset_id
                and self.report.dataset_version == authorization.dataset_version
                and self.report.purpose is authorization.purpose
                and self.report.evidence_sha256 == authorization.evidence_sha256
                and self.report.locked_evidence_binding_sha256
                == authorization.locked_evidence_snapshot_sha256
                and self.report.stage3_package_sha256
                == authorization.locked_evidence.stage3_calibration_package_sha256
                and self.report.config_sha256 == authorization.locked_evidence.config_sha256
                and self.report.data_sha256 == authorization.locked_evidence.data_sha256
                and self.report.model_sha256 == authorization.locked_evidence.model_sha256
                and self.report.root_binding_sha256 == authorization.root_binding_sha256
                and self.report.authorization_ttl_seconds == authorization.ttl_seconds
            )
            if not bindings_match:
                raise ValueError("VERIFIED report and authorization bindings do not match")
        elif self.authorization is not None:
            raise ValueError("BLOCKED outcome cannot contain an authorization")


def _blocked_outcome(
    *,
    context: Stage5VerificationContext,
    dataset_id: str,
    dataset_version: str,
    reason: Stage5PrerequisiteReason | None = None,
    reasons: tuple[Stage5PrerequisiteReason, ...] = (),
    evidence_hashes: Mapping[str, str] | None = None,
    evidence_files_read: int = 0,
) -> Stage5VerificationOutcome:
    normalized_reasons = reasons or (() if reason is None else (reason,))
    if not normalized_reasons:
        raise ValueError("blocked Stage 5 outcome requires at least one reason")
    report = Stage5PrerequisiteReport(
        status=Stage5PrerequisiteStatus.BLOCKED,
        verification_context=context,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
        reasons=normalized_reasons,
        evidence_sha256=tuple(sorted((evidence_hashes or {}).items())),
        evidence_files_read=evidence_files_read,
        locked_evidence_binding_sha256=None,
        stage3_package_sha256=None,
        config_sha256=None,
        data_sha256=None,
        model_sha256=None,
        root_binding_sha256=None,
        authorization_ttl_seconds=None,
    )
    return Stage5VerificationOutcome(report=report, authorization=None)


def _root_is_lexically_valid(root: Path) -> bool:
    return (
        root.is_absolute()
        and ".." not in root.parts
        and root != Path(root.anchor)
        and "\x00" not in str(root)
        and "\r" not in str(root)
        and "\n" not in str(root)
    )


def _strict_root_identity(root: Path) -> tuple[int, int, int, int, int, int]:
    """Physically bind a root only after prerequisite authorization succeeded."""

    if not _root_is_lexically_valid(root):
        raise Stage5RootBindingError(("root path is not a safe absolute dataset root",))
    final_metadata: os.stat_result | None = None
    for candidate in (*reversed(root.parents), root):
        try:
            metadata = os.lstat(candidate)
        except OSError as error:
            raise Stage5RootBindingError(("root or an ancestor does not exist",)) from error
        if _is_link_or_reparse(metadata):
            raise Stage5RootBindingError(("root path contains a link or reparse point",))
        if candidate == root:
            final_metadata = metadata
    if final_metadata is None or not stat.S_ISDIR(final_metadata.st_mode):
        raise Stage5RootBindingError(("authorized root is not a directory",))
    return _file_identity(final_metadata)


def _authorization_primitive_binding_sha256(value: Stage5DataAuthorization) -> str:
    return _canonical_sha256(
        {
            "dataset_id": value.dataset_id,
            "dataset_version": value.dataset_version,
            "purpose": value.purpose.value,
            "verification_context": value.verification_context.value,
            "root_binding_sha256": value.root_binding_sha256,
            "evidence_binding_sha256": value.evidence_binding_sha256,
            "registry_snapshot_sha256": value.registry_snapshot_sha256,
            "license_review_snapshot_sha256": value.license_review_snapshot_sha256,
            "adapter_spec_snapshot_sha256": value.adapter_spec_snapshot_sha256,
            "locked_evidence_snapshot_sha256": value.locked_evidence_snapshot_sha256,
            "stage2_permit_snapshot_sha256": value.stage2_permit_snapshot_sha256,
            "issued_process_id": value.issued_process_id,
            "issued_monotonic_ns": value.issued_monotonic_ns,
            "expires_monotonic_ns": value.expires_monotonic_ns,
        }
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5BoundRoot:
    """One-shot physical root binding issued only after authorization validation."""

    authorization: Stage5DataAuthorization
    root: Path
    authorization_binding_sha256: str
    logical_root_binding_sha256: str
    physical_identity: tuple[int, int, int, int, int, int]
    bound_process_id: int
    bound_monotonic_ns: int
    expires_monotonic_ns: int
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _BOUND_ROOT_TOKEN or self._process_token is not _PROCESS_TOKEN:
            raise TypeError("Stage5BoundRoot can only be issued by bind_authorized_root")

    @property
    def authorizes_file_open(self) -> bool:
        """The root must still be consumed through the immediate validator."""

        return False


_ISSUED_BOUND_ROOTS: dict[int, tuple[Stage5BoundRoot, tuple[object, ...]]] = {}
_ROOT_BOUND_AUTHORIZATION_IDS: set[int] = set()
_CONSUMED_BOUND_ROOT_IDS: set[int] = set()


def _bound_root_fingerprint(value: Stage5BoundRoot) -> tuple[object, ...]:
    return (
        id(value.authorization),
        _authorization_primitive_binding_sha256(value.authorization),
        value.root.as_posix(),
        value.authorization_binding_sha256,
        value.logical_root_binding_sha256,
        value.physical_identity,
        value.bound_process_id,
        value.bound_monotonic_ns,
        value.expires_monotonic_ns,
        value._process_token,
        value._token,
    )


def bind_authorized_root(
    value: object,
    *,
    required_context: Stage5VerificationContext = Stage5VerificationContext.REAL_PREREQUISITE,
) -> Stage5BoundRoot:
    """Perform the first permitted root lstat and issue a short one-shot binding."""

    authorization = require_stage5_authorization(value, required_context=required_context)
    if id(authorization) in _ROOT_BOUND_AUTHORIZATION_IDS:
        raise Stage5RootBindingError(("authorization has already been bound to a root",))
    physical_identity = _strict_root_identity(authorization._root)
    now = time.monotonic_ns()
    expires = min(
        authorization.expires_monotonic_ns,
        now + MAX_STAGE5_ROOT_BINDING_TTL_SECONDS * 1_000_000_000,
    )
    if expires <= now:
        raise Stage5RootBindingError(("authorization expired before root binding",))
    bound = Stage5BoundRoot(
        authorization=authorization,
        root=authorization._root,
        authorization_binding_sha256=_authorization_primitive_binding_sha256(authorization),
        logical_root_binding_sha256=authorization.root_binding_sha256,
        physical_identity=physical_identity,
        bound_process_id=_PROCESS_ID,
        bound_monotonic_ns=now,
        expires_monotonic_ns=expires,
        _process_token=_PROCESS_TOKEN,
        _token=_BOUND_ROOT_TOKEN,
    )
    _ISSUED_BOUND_ROOTS[id(bound)] = (bound, _bound_root_fingerprint(bound))
    _ROOT_BOUND_AUTHORIZATION_IDS.add(id(authorization))
    return bound


def bound_root_validation_reasons(
    value: object,
    *,
    required_context: Stage5VerificationContext = Stage5VerificationContext.REAL_PREREQUISITE,
) -> tuple[str, ...]:
    """Re-lstat a post-authorization root immediately before one-shot consumption."""

    if not isinstance(value, Stage5BoundRoot):
        return ("bound root has an invalid type",)
    issued = _ISSUED_BOUND_ROOTS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("bound root was not issued by the Stage 5 Gate",)
    if id(value) in _CONSUMED_BOUND_ROOT_IDS:
        return ("bound root has already been consumed",)
    try:
        if issued[1] != _bound_root_fingerprint(value):
            return ("bound root bindings changed after issuance",)
        reasons = list(
            authorization_validation_reasons(
                value.authorization,
                required_context=required_context,
            )
        )
        if value.bound_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("bound root belongs to a different process")
        if value._process_token is not _PROCESS_TOKEN or value._token is not _BOUND_ROOT_TOKEN:
            reasons.append("bound root issuance identity is invalid")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("bound root has expired")
        if value.root != value.authorization._root or (
            value.logical_root_binding_sha256 != value.authorization.root_binding_sha256
        ):
            reasons.append("bound root does not match its authorization")
        if value.authorization_binding_sha256 != _authorization_primitive_binding_sha256(
            value.authorization
        ):
            reasons.append("bound root authorization snapshot changed")
        try:
            current_identity = _strict_root_identity(value.root)
        except Stage5RootBindingError as error:
            reasons.extend(error.reasons)
        else:
            if current_identity != value.physical_identity:
                reasons.append("bound root physical identity changed")
    except (AttributeError, TypeError, ValueError):
        return ("bound root fields are invalid",)
    return tuple(dict.fromkeys(reasons))


def consume_bound_stage5_root(
    value: object,
    *,
    required_context: Stage5VerificationContext = Stage5VerificationContext.REAL_PREREQUISITE,
) -> Path:
    """Perform the immediate physical check and consume the binding exactly once.

    The returned path is for immediate, read-only adapter use.  The caller must not
    cache it as independent authority or use the prerequisite authorization directly.
    """

    reasons = bound_root_validation_reasons(value, required_context=required_context)
    if reasons:
        raise Stage5RootBindingError(reasons)
    assert isinstance(value, Stage5BoundRoot)
    _CONSUMED_BOUND_ROOT_IDS.add(id(value))
    return value.root


def _paths_are_distinct_and_outside_root(paths: Mapping[str, Path], root: Path) -> None:
    lexical = [os.path.normcase(os.path.normpath(str(path))) for path in paths.values()]
    if len(set(lexical)) != len(lexical):
        raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_PATH_COLLISION)
    for path in paths.values():
        if path == root or path.is_relative_to(root):
            raise _EvidenceFailure(Stage5PrerequisiteReason.EVIDENCE_INSIDE_DATASET_ROOT)


def _reviewer_fingerprint(reviewer_id: str) -> str:
    return hashlib.sha256(f"retinaentail-reviewer-v1:{reviewer_id}".encode()).hexdigest()


def _registry_snapshot_sha256(permit: DatasetAccessPermit) -> str:
    registry = permit.registry
    return _canonical_sha256(
        {
            "schema_version": registry.schema_version,
            "dataset_id": registry.dataset_id,
            "dataset_version": registry.dataset_version,
            "license_status": registry.license_status,
            "license_review_status": registry.license_review_status,
            "label_verification_status": registry.label_verification_status,
        }
    )


def _live_permit_snapshot_sha256(permit: DatasetAccessPermit | None) -> str | None:
    if permit is None:
        return None
    return _canonical_sha256(
        {
            "registry_sha256": _registry_snapshot_sha256(permit),
            "review": permit.review.to_dict(),
            "purpose": permit.purpose.value,
            "root": permit.root.as_posix(),
            "adapter_spec": permit.spec.to_dict(),
        }
    )


def _raise_mismatch(reason: Stage5PrerequisiteReason, condition: bool) -> None:
    if not condition:
        raise _EvidenceFailure(reason)


def _verify_cross_bindings(
    *,
    dataset_id: str,
    dataset_version: str,
    review: LicenseReviewRecord,
    review_sha256: str,
    spec: DatasetAdapterSpec,
    spec_sha256: str,
    audit: Mapping[str, object],
    audit_sha256: str,
    card: Mapping[str, object],
    card_sha256: str,
    manual: Mapping[str, object],
    manual_sha256: str,
    stage3: Mapping[str, object],
    locked: Stage4LockedEvidence,
    stage2_access_permit: DatasetAccessPermit | None,
) -> None:
    if review.dataset_id != dataset_id or spec.dataset_id != dataset_id:
        raise _EvidenceFailure(Stage5PrerequisiteReason.DATASET_MISMATCH)
    if review.dataset_version != dataset_version:
        raise _EvidenceFailure(Stage5PrerequisiteReason.DATASET_VERSION_MISMATCH)
    for value in (audit, card, manual, stage3):
        id_matches, version_matches = _dataset_version_matches(value, dataset_id, dataset_version)
        if not id_matches:
            raise _EvidenceFailure(Stage5PrerequisiteReason.DATASET_MISMATCH)
        if not version_matches:
            raise _EvidenceFailure(Stage5PrerequisiteReason.DATASET_VERSION_MISMATCH)
    if locked.dataset_id != dataset_id:
        raise _EvidenceFailure(Stage5PrerequisiteReason.DATASET_MISMATCH)
    if locked.dataset_version != dataset_version:
        raise _EvidenceFailure(Stage5PrerequisiteReason.DATASET_VERSION_MISMATCH)
    if review.decision is not LicenseReviewDecision.APPROVED:
        raise _EvidenceFailure(Stage5PrerequisiteReason.REVIEW_NOT_APPROVED)
    if not set(REQUIRED_STAGE5_REVIEW_PURPOSES) <= set(review.permitted_purposes):
        raise _EvidenceFailure(Stage5PrerequisiteReason.REQUIRED_REVIEW_PURPOSE_MISSING)
    split_counts = {"train": 0, "calibration": 0, "test": 0}
    for entry in spec.entries:
        if entry.split not in split_counts:
            raise _EvidenceFailure(Stage5PrerequisiteReason.PROVENANCE_MISMATCH)
        split_counts[entry.split] += 1
    card_split_counts = _mapping(card["split_counts"], name="card split counts")
    audit_summary = _mapping(audit["summary"], name="audit summary")
    spec_card_counts_match = (
        all(split_counts[name] > 0 for name in split_counts)
        and card_split_counts == split_counts
        and card["sample_count"] == len(spec.entries)
        and audit_summary["record_count"] == len(spec.entries)
    )
    _raise_mismatch(
        Stage5PrerequisiteReason.PROVENANCE_MISMATCH,
        spec_card_counts_match,
    )
    if stage2_access_permit is not None:
        permit_matches = (
            stage2_access_permit.registry.dataset_id == dataset_id
            and stage2_access_permit.registry.dataset_version == dataset_version
            and stage2_access_permit.review == review
            and _canonical_sha256(stage2_access_permit.spec.to_dict()) == spec_sha256
            and card["registry_sha256"] == _registry_snapshot_sha256(stage2_access_permit)
        )
        _raise_mismatch(Stage5PrerequisiteReason.PROVENANCE_MISMATCH, permit_matches)

    hashes_match = (
        locked.license_review_sha256 == review_sha256
        and locked.adapter_spec_sha256 == spec_sha256
        and locked.stage2_audit_sha256 == audit_sha256
        and locked.frozen_data_card_sha256 == card_sha256
        and locked.manual_confirmation_sha256 == manual_sha256
        and locked.stage3_calibration_package_sha256 == stage3["package_sha256"]
        and locked.calibration_artifact_sha256 == stage3["calibration_artifact_sha256"]
        and card["license_review_sha256"] == review_sha256
        and card["adapter_spec_sha256"] == spec_sha256
        and card["audit_sha256"] == audit_sha256
    )
    _raise_mismatch(Stage5PrerequisiteReason.HASH_MISMATCH, hashes_match)
    card_bindings_match = (
        card["manifest_sha256"] == audit["manifest_sha256"]
        and card["data_sha256"] == audit["manifest_sha256"]
        and card["license_terms_sha256"] == review.terms_sha256
        and card["license_evidence_sha256"] == review.evidence_sha256
        and card["reviewed_at"] == review.reviewed_at
        and card["reviewer_fingerprint"] == _reviewer_fingerprint(review.reviewer_id)
        and manual["identity_evidence_sha256"] == card["identity_evidence_sha256"]
        and manual["label_semantics_evidence_sha256"] == card["label_semantics_evidence_sha256"]
    )
    _raise_mismatch(Stage5PrerequisiteReason.PROVENANCE_MISMATCH, card_bindings_match)
    stage3_bindings_match = (
        locked.alpha == stage3["alpha"]
        and locked.positive_threshold == stage3["positive_threshold"]
        and locked.negative_threshold == stage3["negative_threshold"]
        and locked.positive_calibration_sample_count == stage3["positive_sample_count"]
        and locked.negative_calibration_sample_count == stage3["negative_sample_count"]
        and locked.total_calibration_sample_count == stage3["total_sample_count"]
        and locked.config_sha256 == stage3["config_hash"]
        and locked.data_sha256 == stage3["data_hash"]
        and locked.model_sha256 == stage3["model_hash"]
    )
    _raise_mismatch(Stage5PrerequisiteReason.PROVENANCE_MISMATCH, stage3_bindings_match)


def verify_stage5_prerequisites(
    *,
    verification_context: Stage5VerificationContext,
    dataset_id: str,
    dataset_version: str,
    dataset_root: Path,
    evidence_paths: Stage5EvidencePaths,
    stage2_access_permit: DatasetAccessPermit | None,
    authorization_ttl_seconds: int = DEFAULT_STAGE5_AUTHORIZATION_TTL_SECONDS,
) -> Stage5VerificationOutcome:
    """Verify evidence first and issue a process-bound capability without root I/O.

    Evidence is required to live outside ``dataset_root``.  The function deliberately
    performs no root existence, type, containment, symlink, directory, or data checks;
    those are permitted only after a consumer validates the returned capability.
    """

    safe_id = dataset_id if isinstance(dataset_id, str) else "invalid"
    safe_version = dataset_version if isinstance(dataset_version, str) else "invalid"
    if (
        not isinstance(verification_context, Stage5VerificationContext)
        or not isinstance(dataset_id, str)
        or _DATASET_ID_PATTERN.fullmatch(dataset_id) is None
        or not isinstance(dataset_version, str)
        or _DATASET_VERSION_PATTERN.fullmatch(dataset_version) is None
        or not isinstance(dataset_root, Path)
        or not isinstance(evidence_paths, Stage5EvidencePaths)
    ):
        return _blocked_outcome(
            context=(
                verification_context
                if isinstance(verification_context, Stage5VerificationContext)
                else Stage5VerificationContext.REAL_PREREQUISITE
            ),
            dataset_id=(safe_id if _DATASET_ID_PATTERN.fullmatch(safe_id) else "invalid"),
            dataset_version=(
                safe_version if _DATASET_VERSION_PATTERN.fullmatch(safe_version) else "invalid"
            ),
            reason=Stage5PrerequisiteReason.INPUT_INVALID,
        )
    if not _root_is_lexically_valid(dataset_root):
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reason=Stage5PrerequisiteReason.ROOT_INVALID,
        )
    if verification_context is Stage5VerificationContext.REAL_PREREQUISITE:
        if (
            not isinstance(stage2_access_permit, DatasetAccessPermit)
            or permit_validation_reasons(stage2_access_permit)
            or stage2_access_permit.purpose is not DataUsePurpose.LOCKED_TEST_EVALUATION
            or stage2_access_permit.root != dataset_root
            or stage2_access_permit.registry.dataset_id != dataset_id
            or stage2_access_permit.registry.dataset_version != dataset_version
        ):
            return _blocked_outcome(
                context=verification_context,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                reason=Stage5PrerequisiteReason.STAGE2_ACCESS_PERMIT_INVALID,
            )
    elif stage2_access_permit is not None:
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reason=Stage5PrerequisiteReason.INPUT_INVALID,
        )
    if (
        type(authorization_ttl_seconds) is not int  # noqa: E721
        or not 1 <= authorization_ttl_seconds <= MAX_STAGE5_AUTHORIZATION_TTL_SECONDS
    ):
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reason=Stage5PrerequisiteReason.TTL_INVALID,
        )

    paths = evidence_paths.as_dict()
    try:
        _paths_are_distinct_and_outside_root(paths, dataset_root)
    except _EvidenceFailure as error:
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reason=error.reason,
        )

    decoded: dict[str, object] = {}
    evidence_hashes: dict[str, str] = {}
    files_read = 0
    try:
        for name in _ARTIFACT_NAMES:
            payload = _read_bounded_evidence(paths[name], limit=_ARTIFACT_LIMITS[name])
            decoded[name] = _decode_evidence(
                payload,
                allow_yaml=name in {"license_review", "adapter_spec"},
            )
            files_read += 1

        review_mapping = _mapping(decoded["license_review"], name="license review")
        review = LicenseReviewRecord.from_mapping(review_mapping)
        review_sha256 = _canonical_sha256(review.to_dict())
        evidence_hashes["license_review"] = review_sha256

        spec_mapping = _mapping(decoded["adapter_spec"], name="adapter spec")
        spec = DatasetAdapterSpec.from_dict(spec_mapping)
        spec_sha256 = _canonical_sha256(spec.to_dict())
        evidence_hashes["adapter_spec"] = spec_sha256

        audit = _validated_audit(decoded["stage2_audit"])
        audit_sha256 = _canonical_sha256(audit)
        evidence_hashes["stage2_audit"] = audit_sha256

        card = _validated_card(decoded["frozen_data_card"])
        card_sha256 = _canonical_sha256(card)
        evidence_hashes["frozen_data_card"] = card_sha256

        manual = _validated_manual_confirmation(decoded["manual_confirmation"])
        manual_sha256 = _canonical_sha256(manual)
        evidence_hashes["manual_confirmation"] = manual_sha256

        stage3 = _validated_stage3_summary(decoded["stage3_package_summary"])
        evidence_hashes["stage3_package_summary"] = _canonical_sha256(stage3)

        locked_mapping = _mapping(decoded["stage4_locked_evidence"], name="Stage 4 locked evidence")
        try:
            locked = Stage4LockedEvidence.from_dict(locked_mapping)
        except (TypeError, ValueError) as error:
            raise _EvidenceFailure(Stage5PrerequisiteReason.LOCKED_EVIDENCE_INVALID) from error
        evidence_hashes["stage4_locked_evidence"] = locked.binding_sha256()

        _verify_cross_bindings(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            review=review,
            review_sha256=review_sha256,
            spec=spec,
            spec_sha256=spec_sha256,
            audit=audit,
            audit_sha256=audit_sha256,
            card=card,
            card_sha256=card_sha256,
            manual=manual,
            manual_sha256=manual_sha256,
            stage3=stage3,
            locked=locked,
            stage2_access_permit=stage2_access_permit,
        )
    except _EvidenceFailure as error:
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reason=error.reason,
            evidence_hashes=evidence_hashes,
            evidence_files_read=files_read,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        del error
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reason=Stage5PrerequisiteReason.EVIDENCE_CONTRACT_INVALID,
            evidence_hashes=evidence_hashes,
            evidence_files_read=files_read,
        )

    if verification_context is Stage5VerificationContext.REAL_PREREQUISITE:
        # Detached audit JSON and a detached Stage 3 verifier summary are evidence,
        # not independently issued capabilities.  Until the real manifest/audit and
        # complete Stage 3 package verifiers expose process-bound capabilities, no
        # combination of self-authored bytes may cross the real root-I/O boundary.
        return _blocked_outcome(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reasons=(
                Stage5PrerequisiteReason.REAL_STAGE2_AUDIT_CAPABILITY_UNAVAILABLE,
                Stage5PrerequisiteReason.REAL_STAGE3_PACKAGE_CAPABILITY_UNAVAILABLE,
            ),
            evidence_hashes=evidence_hashes,
            evidence_files_read=files_read,
        )

    root_binding = _root_binding(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        root=dataset_root,
        context=verification_context,
    )
    now = time.monotonic_ns()
    evidence_tuple = tuple(sorted(evidence_hashes.items()))
    evidence_binding_sha256 = _canonical_sha256(dict(evidence_tuple))
    spec_snapshot_sha256 = _canonical_sha256(spec.to_dict())
    locked_snapshot_sha256 = locked.binding_sha256()
    registry_snapshot_sha256 = (
        None if stage2_access_permit is None else _registry_snapshot_sha256(stage2_access_permit)
    )
    permit_snapshot_sha256 = _live_permit_snapshot_sha256(stage2_access_permit)
    authorization = Stage5DataAuthorization(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
        verification_context=verification_context,
        _root=dataset_root,
        adapter_spec=spec,
        locked_evidence=locked,
        stage2_access_permit=stage2_access_permit,
        evidence_sha256=evidence_tuple,
        evidence_binding_sha256=evidence_binding_sha256,
        registry_snapshot_sha256=registry_snapshot_sha256,
        license_review_snapshot_sha256=review_sha256,
        adapter_spec_snapshot_sha256=spec_snapshot_sha256,
        locked_evidence_snapshot_sha256=locked_snapshot_sha256,
        stage2_permit_snapshot_sha256=permit_snapshot_sha256,
        root_binding_sha256=root_binding,
        issued_process_id=_PROCESS_ID,
        issued_monotonic_ns=now,
        expires_monotonic_ns=now + authorization_ttl_seconds * 1_000_000_000,
        ttl_seconds=authorization_ttl_seconds,
        _process_token=_PROCESS_TOKEN,
        _token=_AUTHORIZATION_TOKEN,
    )
    _ISSUED_AUTHORIZATIONS[id(authorization)] = (
        authorization,
        _authorization_fingerprint(authorization),
    )
    report = Stage5PrerequisiteReport(
        status=Stage5PrerequisiteStatus.VERIFIED,
        verification_context=verification_context,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
        reasons=(),
        evidence_sha256=tuple(sorted(evidence_hashes.items())),
        evidence_files_read=len(_ARTIFACT_NAMES),
        locked_evidence_binding_sha256=locked.binding_sha256(),
        stage3_package_sha256=str(stage3["package_sha256"]),
        config_sha256=locked.config_sha256,
        data_sha256=locked.data_sha256,
        model_sha256=locked.model_sha256,
        root_binding_sha256=root_binding,
        authorization_ttl_seconds=authorization_ttl_seconds,
    )
    return Stage5VerificationOutcome(report=report, authorization=authorization)


__all__ = [
    "DEFAULT_STAGE5_AUTHORIZATION_TTL_SECONDS",
    "MAX_STAGE5_AUTHORIZATION_TTL_SECONDS",
    "MAX_STAGE5_ROOT_BINDING_TTL_SECONDS",
    "REQUIRED_STAGE5_REVIEW_PURPOSES",
    "STAGE5_PREFLIGHT_SCHEMA_VERSION",
    "STAGE5_PREREQUISITE_SCHEMA_VERSION",
    "Stage5AuthorizationError",
    "Stage5BoundRoot",
    "Stage5DataAuthorization",
    "Stage5EvidencePaths",
    "Stage5PreflightReason",
    "Stage5PreflightReport",
    "Stage5PreflightStatus",
    "Stage5PrerequisiteReason",
    "Stage5PrerequisiteReport",
    "Stage5PrerequisiteStatus",
    "Stage5RootBindingError",
    "Stage5VerificationContext",
    "Stage5VerificationOutcome",
    "assess_stage5_preflight",
    "authorization_validation_reasons",
    "bind_authorized_root",
    "bound_root_validation_reasons",
    "consume_bound_stage5_root",
    "require_stage5_authorization",
    "verify_stage5_prerequisites",
]
