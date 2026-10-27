"""Zero-I/O Stage 6 prerequisite gate and one-shot physical-root binding.

The preflight accepts presence flags, never paths.  The prerequisite verifier accepts
only in-memory, process-issued capabilities and evidence objects; it deliberately does
not stat, resolve, list, or open the dataset root.  Physical root inspection is a
separate, short-lived, one-shot operation after every prerequisite has been verified.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from retinaentail.calibration.data_card import (
    DatasetCardScope,
    FrozenDatasetCard,
    frozen_data_card_validation_reasons,
)
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    permit_validation_reasons,
)
from retinaentail.evaluation.gate import Stage4LockedEvidence

STAGE6_GATE_SCHEMA_VERSION: Final = "6.0.0"
MAX_STAGE6_AUTHORIZATION_TTL_SECONDS: Final = 5
DEFAULT_STAGE6_AUTHORIZATION_TTL_SECONDS: Final = 5
REQUIRED_STAGE6_REVIEW_PURPOSES: Final = (
    DataUsePurpose.READ_ONLY_AUDIT,
    DataUsePurpose.LESION_CALIBRATION,
    DataUsePurpose.LOCKED_TEST_EVALUATION,
)

_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_PATTERN: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_REPARSE_POINT_ATTRIBUTE: Final = 0x0400
_AUTHORIZATION_TOKEN: Final = object()
_BOUND_ROOT_TOKEN: Final = object()
_PROCESS_TOKEN: Final = object()
_PROCESS_ID: Final = os.getpid()
_BOUNDARY_LOCK: Final = threading.RLock()


class Stage6VerificationContext(StrEnum):
    """Keep synthetic verification incapable of satisfying a real-data consumer."""

    REAL_PREREQUISITE = "REAL_PREREQUISITE"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class Stage6PreflightStatus(StrEnum):
    BLOCKED = "BLOCKED"
    READY_FOR_PREREQUISITE_VERIFICATION = "READY_FOR_PREREQUISITE_VERIFICATION"


class Stage6PrerequisiteStatus(StrEnum):
    BLOCKED = "BLOCKED"
    VERIFIED = "VERIFIED"


class Stage6Reason(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED = "READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED"
    LESION_CALIBRATION_PURPOSE_NOT_APPROVED = "LESION_CALIBRATION_PURPOSE_NOT_APPROVED"
    LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED = "LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED"
    DATASET_ROOT_NOT_EXPLICIT = "DATASET_ROOT_NOT_EXPLICIT"
    STAGE2_ACCESS_PERMIT_NOT_PROVIDED = "STAGE2_ACCESS_PERMIT_NOT_PROVIDED"
    STAGE2_PACKAGE_CAPABILITY_NOT_PROVIDED = "STAGE2_PACKAGE_CAPABILITY_NOT_PROVIDED"
    STAGE3_PACKAGE_CAPABILITY_NOT_PROVIDED = "STAGE3_PACKAGE_CAPABILITY_NOT_PROVIDED"
    FROZEN_DATA_CARD_NOT_PROVIDED = "FROZEN_DATA_CARD_NOT_PROVIDED"
    MANUAL_CONFIRMATION_NOT_PROVIDED = "MANUAL_CONFIRMATION_NOT_PROVIDED"
    LOCKED_EVIDENCE_NOT_PROVIDED = "LOCKED_EVIDENCE_NOT_PROVIDED"
    ROOT_INVALID = "ROOT_INVALID"
    TTL_INVALID = "TTL_INVALID"
    STAGE2_ACCESS_PERMIT_INVALID = "STAGE2_ACCESS_PERMIT_INVALID"
    STAGE2_PACKAGE_CAPABILITY_API_UNAVAILABLE = "STAGE2_PACKAGE_CAPABILITY_API_UNAVAILABLE"
    STAGE2_PACKAGE_CAPABILITY_INVALID = "STAGE2_PACKAGE_CAPABILITY_INVALID"
    STAGE3_PACKAGE_CAPABILITY_API_UNAVAILABLE = "STAGE3_PACKAGE_CAPABILITY_API_UNAVAILABLE"
    STAGE3_PACKAGE_CAPABILITY_INVALID = "STAGE3_PACKAGE_CAPABILITY_INVALID"
    STAGE5_SYNTHETIC_PILOT_NOT_PROVIDED = "STAGE5_SYNTHETIC_PILOT_NOT_PROVIDED"
    STAGE5_SYNTHETIC_PILOT_INVALID = "STAGE5_SYNTHETIC_PILOT_INVALID"
    FROZEN_DATA_CARD_INVALID = "FROZEN_DATA_CARD_INVALID"
    MANUAL_CONFIRMATION_INVALID = "MANUAL_CONFIRMATION_INVALID"
    LOCKED_EVIDENCE_INVALID = "LOCKED_EVIDENCE_INVALID"
    DATASET_MISMATCH = "DATASET_MISMATCH"
    DATASET_VERSION_MISMATCH = "DATASET_VERSION_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    SPLIT_MISMATCH = "SPLIT_MISMATCH"
    HASH_MISMATCH = "HASH_MISMATCH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"


class Stage6AuthorizationError(PermissionError):
    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 6 authorization rejected: " + "; ".join(reasons))


class Stage6RootBindingError(PermissionError):
    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 6 physical root rejected: " + "; ".join(reasons))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_dataset_id(value: object) -> bool:
    return isinstance(value, str) and _DATASET_ID_PATTERN.fullmatch(value) is not None


def _valid_dataset_version(value: object) -> bool:
    return isinstance(value, str) and _DATASET_VERSION_PATTERN.fullmatch(value) is not None


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _enum_value(value: object, name: str) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} must be a non-empty enum value")
    return raw


def _root_is_lexically_valid(root: Path) -> bool:
    return (
        root.is_absolute()
        and ".." not in root.parts
        and root != Path(root.anchor)
        and "\x00" not in str(root)
        and "\r" not in str(root)
        and "\n" not in str(root)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6PreflightReport:
    status: Stage6PreflightStatus
    verification_context: Stage6VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage6Reason, ...]
    schema_version: str = STAGE6_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE6_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 6 gate schema")
        if not isinstance(self.status, Stage6PreflightStatus):
            raise TypeError("status must be Stage6PreflightStatus")
        if not isinstance(self.verification_context, Stage6VerificationContext):
            raise TypeError("verification_context must be Stage6VerificationContext")
        if not all(isinstance(reason, Stage6Reason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage6Reason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must be unique")
        ready = self.status is Stage6PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
        if ready == bool(self.reasons):
            raise ValueError("preflight status and reasons conflict")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "reasons": [reason.value for reason in self.reasons],
            "prerequisites_verified": False,
            "authorization_issued": False,
            "authorizes_file_open": False,
            "accepts_paths": False,
            "dataset_root_probed": False,
            "real_data_read": False,
        }


def assess_stage6_preflight(
    *,
    verification_context: Stage6VerificationContext,
    dataset_id: str | None,
    dataset_version: str | None,
    approved_review_purposes: Sequence[DataUsePurpose],
    dataset_root_provided: bool,
    stage2_access_permit_provided: bool,
    stage2_package_capability_provided: bool,
    stage3_package_capability_provided: bool,
    frozen_data_card_provided: bool,
    manual_confirmation_provided: bool,
    locked_evidence_provided: bool,
    stage5_pilot_report_provided: bool = False,
) -> Stage6PreflightReport:
    """Assess explicit in-memory presence only; this function accepts no path object."""

    if not isinstance(verification_context, Stage6VerificationContext) or any(
        type(value) is not bool  # noqa: E721
        for value in (
            dataset_root_provided,
            stage2_access_permit_provided,
            stage2_package_capability_provided,
            stage3_package_capability_provided,
            frozen_data_card_provided,
            manual_confirmation_provided,
            locked_evidence_provided,
            stage5_pilot_report_provided,
        )
    ):
        return Stage6PreflightReport(
            status=Stage6PreflightStatus.BLOCKED,
            verification_context=(
                verification_context
                if isinstance(verification_context, Stage6VerificationContext)
                else Stage6VerificationContext.REAL_PREREQUISITE
            ),
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage6Reason.INPUT_INVALID,),
        )
    try:
        purposes = tuple(approved_review_purposes)
    except TypeError:
        purposes = ()
    reasons: list[Stage6Reason] = []
    if not _valid_dataset_id(dataset_id):
        reasons.append(Stage6Reason.DATASET_NOT_EXPLICIT)
    if not _valid_dataset_version(dataset_version):
        reasons.append(Stage6Reason.DATASET_VERSION_NOT_EXPLICIT)
    if verification_context is Stage6VerificationContext.REAL_PREREQUISITE:
        purpose_set = {item for item in purposes if isinstance(item, DataUsePurpose)}
        if DataUsePurpose.READ_ONLY_AUDIT not in purpose_set:
            reasons.append(Stage6Reason.READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED)
        if DataUsePurpose.LESION_CALIBRATION not in purpose_set:
            reasons.append(Stage6Reason.LESION_CALIBRATION_PURPOSE_NOT_APPROVED)
        if DataUsePurpose.LOCKED_TEST_EVALUATION not in purpose_set:
            reasons.append(Stage6Reason.LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED)
    common_presence = (
        (dataset_root_provided, Stage6Reason.DATASET_ROOT_NOT_EXPLICIT),
        (
            stage2_package_capability_provided,
            Stage6Reason.STAGE2_PACKAGE_CAPABILITY_NOT_PROVIDED,
        ),
        (
            stage3_package_capability_provided,
            Stage6Reason.STAGE3_PACKAGE_CAPABILITY_NOT_PROVIDED,
        ),
    )
    real_presence = (
        (stage2_access_permit_provided, Stage6Reason.STAGE2_ACCESS_PERMIT_NOT_PROVIDED),
        (frozen_data_card_provided, Stage6Reason.FROZEN_DATA_CARD_NOT_PROVIDED),
        (manual_confirmation_provided, Stage6Reason.MANUAL_CONFIRMATION_NOT_PROVIDED),
        (locked_evidence_provided, Stage6Reason.LOCKED_EVIDENCE_NOT_PROVIDED),
    )
    synthetic_presence = (
        (stage5_pilot_report_provided, Stage6Reason.STAGE5_SYNTHETIC_PILOT_NOT_PROVIDED),
    )
    presence = common_presence + (
        real_presence
        if verification_context is Stage6VerificationContext.REAL_PREREQUISITE
        else synthetic_presence
    )
    reasons.extend(reason for provided, reason in presence if not provided)
    normalized = tuple(dict.fromkeys(reasons))
    return Stage6PreflightReport(
        status=(
            Stage6PreflightStatus.BLOCKED
            if normalized
            else Stage6PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id if _valid_dataset_id(dataset_id) else None,
        dataset_version=dataset_version if _valid_dataset_version(dataset_version) else None,
        reasons=normalized,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6ManualConfirmation:
    """Strict human confirmation record; authority comes from exact hash cross-binding."""

    scope: DatasetCardScope
    dataset_id: str
    dataset_version: str
    confirmed_by: str
    confirmed_at: str
    identity_evidence_sha256: str
    label_semantics_evidence_sha256: str
    source_image_id_confirmed: bool = True
    patient_id_confirmed: bool = True
    eye_id_confirmed: bool = True
    followup_id_confirmed: bool = True
    intervention_pair_id_confirmed: bool = True
    label_semantics_confirmed: bool = True
    split_isolation_confirmed: bool = True
    schema_version: str = STAGE6_GATE_SCHEMA_VERSION
    record_type: str = "MANUAL_IDENTITY_LABEL_CONFIRMATION"

    def __post_init__(self) -> None:
        if self.schema_version != STAGE6_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported manual-confirmation schema")
        if self.record_type != "MANUAL_IDENTITY_LABEL_CONFIRMATION":
            raise ValueError("manual-confirmation record_type is invalid")
        if not isinstance(self.scope, DatasetCardScope):
            raise TypeError("manual confirmation scope must be DatasetCardScope")
        if not _valid_dataset_id(self.dataset_id) or not _valid_dataset_version(
            self.dataset_version
        ):
            raise ValueError("manual confirmation dataset identity is invalid")
        if not isinstance(self.confirmed_by, str) or not self.confirmed_by.strip():
            raise ValueError("confirmed_by must be non-empty")
        if (
            not isinstance(self.confirmed_at, str)
            or _RFC3339_PATTERN.fullmatch(self.confirmed_at) is None
        ):
            raise ValueError("confirmed_at must be an RFC 3339 timestamp")
        parsed = datetime.fromisoformat(self.confirmed_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("confirmed_at must include a timezone")
        _require_sha256(self.identity_evidence_sha256, "identity_evidence_sha256")
        _require_sha256(
            self.label_semantics_evidence_sha256,
            "label_semantics_evidence_sha256",
        )
        confirmations = (
            self.source_image_id_confirmed,
            self.patient_id_confirmed,
            self.eye_id_confirmed,
            self.followup_id_confirmed,
            self.intervention_pair_id_confirmed,
            self.label_semantics_confirmed,
            self.split_isolation_confirmed,
        )
        if any(type(value) is not bool or not value for value in confirmations):  # noqa: E721
            raise ValueError("all manual confirmations must be exact true bool values")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "scope": self.scope.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at,
            "source_image_id_confirmed": self.source_image_id_confirmed,
            "patient_id_confirmed": self.patient_id_confirmed,
            "eye_id_confirmed": self.eye_id_confirmed,
            "followup_id_confirmed": self.followup_id_confirmed,
            "intervention_pair_id_confirmed": self.intervention_pair_id_confirmed,
            "label_semantics_confirmed": self.label_semantics_confirmed,
            "split_isolation_confirmed": self.split_isolation_confirmed,
            "identity_evidence_sha256": self.identity_evidence_sha256,
            "label_semantics_evidence_sha256": self.label_semantics_evidence_sha256,
        }

    def binding_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class _Stage2Binding:
    scope: str
    purpose: str
    dataset_id: str
    dataset_version: str
    record_count: int
    manifest_sha256: str
    audit_sha256: str
    registry_sha256: str | None
    license_review_sha256: str | None
    adapter_spec_sha256: str | None
    root_binding_sha256: str | None
    permit_binding_sha256: str | None
    package_sha256: str

    def digest(self) -> str:
        return _canonical_sha256(
            {
                "scope": self.scope,
                "purpose": self.purpose,
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "record_count": self.record_count,
                "manifest_sha256": self.manifest_sha256,
                "audit_sha256": self.audit_sha256,
                "registry_sha256": self.registry_sha256,
                "license_review_sha256": self.license_review_sha256,
                "adapter_spec_sha256": self.adapter_spec_sha256,
                "root_binding_sha256": self.root_binding_sha256,
                "permit_binding_sha256": self.permit_binding_sha256,
                "package_sha256": self.package_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class _Stage3Binding:
    scope: str
    dataset_id: str
    dataset_version: str
    protocol_scope: str
    calibration_artifact_sha256: str
    config_sha256: str
    data_sha256: str
    model_sha256: str
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    total_sample_count: int
    package_sha256: str
    frozen_data_card_sha256: str | None
    stage2_package_sha256: str | None
    fixture_only: bool

    def digest(self) -> str:
        return _canonical_sha256(
            {
                "scope": self.scope,
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "protocol_scope": self.protocol_scope,
                "calibration_artifact_sha256": self.calibration_artifact_sha256,
                "config_sha256": self.config_sha256,
                "data_sha256": self.data_sha256,
                "model_sha256": self.model_sha256,
                "alpha": self.alpha,
                "positive_threshold": self.positive_threshold,
                "negative_threshold": self.negative_threshold,
                "positive_sample_count": self.positive_sample_count,
                "negative_sample_count": self.negative_sample_count,
                "total_sample_count": self.total_sample_count,
                "package_sha256": self.package_sha256,
                "frozen_data_card_sha256": self.frozen_data_card_sha256,
                "stage2_package_sha256": self.stage2_package_sha256,
                "fixture_only": self.fixture_only,
            }
        )


class _CapabilityUnavailable(Exception):
    pass


class _CapabilityInvalid(Exception):
    pass


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name)


def _stage2_binding(value: object) -> _Stage2Binding:
    try:
        module = importlib.import_module("retinaentail.data.stage2_package")
        capability_type = module.VerifiedStage2PackageCapability
        validator = module.stage2_package_capability_validation_reasons
        snapshot = module.stage2_package_capability_snapshot
        descriptor_type = module.Stage2PackageDescriptor
    except (AttributeError, ImportError) as error:
        raise _CapabilityUnavailable from error
    if not isinstance(value, capability_type):
        raise _CapabilityInvalid
    try:
        if validator(value):
            raise _CapabilityInvalid
        descriptor = snapshot(value)
        if not isinstance(descriptor, descriptor_type):
            raise _CapabilityInvalid
        record_count = descriptor.record_count
        if type(record_count) is not int or record_count < 1:  # noqa: E721
            raise ValueError("Stage 2 record_count is invalid")
        return _Stage2Binding(
            scope=_enum_value(descriptor.scope, "Stage 2 scope"),
            purpose=_enum_value(descriptor.purpose, "Stage 2 purpose"),
            dataset_id=str(descriptor.dataset_id),
            dataset_version=str(descriptor.dataset_version),
            record_count=record_count,
            manifest_sha256=_require_sha256(descriptor.manifest_sha256, "Stage 2 manifest"),
            audit_sha256=_require_sha256(descriptor.audit_sha256, "Stage 2 audit"),
            registry_sha256=_optional_sha256(descriptor.registry_sha256, "Stage 2 registry"),
            license_review_sha256=_optional_sha256(
                descriptor.license_review_sha256,
                "Stage 2 license review",
            ),
            adapter_spec_sha256=_optional_sha256(
                descriptor.adapter_spec_sha256,
                "Stage 2 adapter spec",
            ),
            root_binding_sha256=_optional_sha256(
                descriptor.root_binding_sha256,
                "Stage 2 root binding",
            ),
            permit_binding_sha256=_optional_sha256(
                descriptor.permit_binding_sha256,
                "Stage 2 permit binding",
            ),
            package_sha256=_require_sha256(descriptor.package_sha256, "Stage 2 package"),
        )
    except _CapabilityInvalid:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise _CapabilityInvalid from error


def _stage3_binding(value: object) -> _Stage3Binding:
    """Load the strict Stage 3 capability API only when that module is installed."""

    try:
        module = importlib.import_module("retinaentail.calibration.stage3_package")
        capability_type = module.VerifiedStage3PackageCapability
        validator = module.stage3_package_capability_validation_reasons
        snapshot = module.stage3_package_capability_snapshot
        descriptor_type = module.Stage3PackageDescriptor
    except (AttributeError, ImportError) as error:
        raise _CapabilityUnavailable from error
    if not isinstance(value, capability_type):
        raise _CapabilityInvalid
    try:
        if validator(value):
            raise _CapabilityInvalid
        descriptor = snapshot(value)
        if not isinstance(descriptor, descriptor_type):
            raise _CapabilityInvalid
        frozen_data_card_sha256 = _optional_sha256(
            descriptor.frozen_data_card_sha256,
            "Stage 3 frozen data card",
        )
        stage2_package_sha256 = _optional_sha256(
            descriptor.stage2_package_sha256,
            "Stage 3 Stage 2 package",
        )
        fixture_only = descriptor.fixture_only
        if type(fixture_only) is not bool:  # noqa: E721
            raise ValueError("Stage 3 fixture_only is invalid")
        return _Stage3Binding(
            scope=_enum_value(descriptor.scope, "Stage 3 scope"),
            dataset_id=str(descriptor.dataset_id),
            dataset_version=str(descriptor.dataset_version),
            protocol_scope=str(descriptor.protocol_scope),
            calibration_artifact_sha256=_require_sha256(
                descriptor.calibration_artifact_hash,
                "Stage 3 calibration artifact",
            ),
            config_sha256=_require_sha256(descriptor.config_hash, "Stage 3 config"),
            data_sha256=_require_sha256(descriptor.data_hash, "Stage 3 data"),
            model_sha256=_require_sha256(descriptor.model_hash, "Stage 3 model"),
            alpha=float(descriptor.alpha),
            positive_threshold=float(descriptor.positive_threshold),
            negative_threshold=float(descriptor.negative_threshold),
            positive_sample_count=int(descriptor.positive_sample_count),
            negative_sample_count=int(descriptor.negative_sample_count),
            total_sample_count=int(descriptor.total_sample_count),
            package_sha256=_require_sha256(descriptor.package_sha256, "Stage 3 package"),
            frozen_data_card_sha256=frozen_data_card_sha256,
            stage2_package_sha256=stage2_package_sha256,
            fixture_only=fixture_only,
        )
    except _CapabilityInvalid:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise _CapabilityInvalid from error


def _stage5_report_binding(value: object) -> str:
    """Validate an issuer-bound synthetic Stage 5 pilot without treating it as real."""

    try:
        module = importlib.import_module("retinaentail.reporting.stage5")
        report_type = module.Stage5PilotReport
    except (AttributeError, ImportError) as error:
        raise _CapabilityUnavailable from error
    if not isinstance(value, report_type):
        raise _CapabilityInvalid
    try:
        payload = value.to_dict()
        if (
            payload.get("scope") != "SYNTHETIC_ONLY"
            or payload.get("split") != "test"
            or payload.get("real_data_used") is not False
            or payload.get("real_data_read_count") != 0
            or payload.get("test_tuning_prohibited") is not True
            or payload.get("thresholds_reestimated") is not False
            or payload.get("individual_report_authoritative") is not False
        ):
            raise _CapabilityInvalid
        return _canonical_sha256(payload)
    except _CapabilityInvalid:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise _CapabilityInvalid from error


def _replay_package_capabilities(stage2_value: object, stage3_value: object) -> None:
    """Perform both full replays without consuming either one-shot package capability."""

    try:
        stage2_module = importlib.import_module("retinaentail.data.stage2_package")
        stage3_module = importlib.import_module("retinaentail.calibration.stage3_package")
        stage2_replay = stage2_module.replay_stage2_package(stage2_value)
        if stage2_replay.valid is not True or stage2_replay.descriptor is None:
            raise _CapabilityInvalid
        stage2_live = _stage2_binding(stage2_value)
        if stage2_replay.descriptor.package_sha256 != stage2_live.package_sha256:
            raise _CapabilityInvalid
        stage3_replay = stage3_module.replay_stage3_package(stage3_value)
        stage3_live = _stage3_binding(stage3_value)
        if stage3_replay.package_sha256 != stage3_live.package_sha256:
            raise _CapabilityInvalid
    except _CapabilityInvalid:
        raise
    except (AttributeError, ImportError, OSError, PermissionError, TypeError, ValueError) as error:
        raise _CapabilityInvalid from error


def _permit_snapshot(permit: DatasetAccessPermit) -> dict[str, object]:
    registry = permit.registry
    return {
        "registry": {
            "schema_version": registry.schema_version,
            "dataset_id": registry.dataset_id,
            "dataset_version": registry.dataset_version,
            "license_status": registry.license_status,
            "license_review_status": registry.license_review_status,
            "label_verification_status": registry.label_verification_status,
        },
        "review": permit.review.to_dict(),
        "purpose": permit.purpose.value,
        "root": permit.root.as_posix(),
        "spec": permit.spec.to_dict(),
    }


def _registry_sha256(permit: DatasetAccessPermit) -> str:
    return _canonical_sha256(_permit_snapshot(permit)["registry"])


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6PrerequisiteReport:
    status: Stage6PrerequisiteStatus
    verification_context: Stage6VerificationContext
    dataset_id: str
    dataset_version: str
    reasons: tuple[Stage6Reason, ...]
    binding_sha256: tuple[tuple[str, str], ...] = ()
    root_binding_sha256: str | None = None
    authorization_ttl_seconds: int | None = None
    schema_version: str = STAGE6_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE6_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 6 report schema")
        if not _valid_dataset_id(self.dataset_id) or not _valid_dataset_version(
            self.dataset_version
        ):
            raise ValueError("report dataset identity is invalid")
        if self.status is Stage6PrerequisiteStatus.VERIFIED:
            if self.reasons or not self.binding_sha256 or self.root_binding_sha256 is None:
                raise ValueError("VERIFIED report requires complete bindings and no reasons")
            if (
                type(self.authorization_ttl_seconds) is not int  # noqa: E721
                or not 1 <= self.authorization_ttl_seconds <= MAX_STAGE6_AUTHORIZATION_TTL_SECONDS
            ):
                raise ValueError("VERIFIED report requires a bounded TTL")
        elif not self.reasons or self.binding_sha256 or self.root_binding_sha256 is not None:
            raise ValueError("BLOCKED report must contain only controlled reasons")
        for name, digest in self.binding_sha256:
            if not isinstance(name, str):
                raise TypeError("binding names must be strings")
            _require_sha256(digest, name)

    def to_dict(self) -> dict[str, object]:
        verified = self.status is Stage6PrerequisiteStatus.VERIFIED
        real_verified = (
            verified and self.verification_context is Stage6VerificationContext.REAL_PREREQUISITE
        )
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "reasons": [reason.value for reason in self.reasons],
            "binding_sha256": dict(self.binding_sha256),
            "root_binding_sha256": self.root_binding_sha256,
            "authorization_ttl_seconds": self.authorization_ttl_seconds,
            "prerequisites_verified": verified,
            "authorization_issued": verified,
            "real_authorization_issued": real_verified,
            "authorizes_file_open": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "same_dataset_provenance_verified": real_verified,
            "synthetic_components_independent": (
                verified
                and self.verification_context is Stage6VerificationContext.SYNTHETIC_FIXTURE
            ),
        }


def _blocked_report(
    *,
    context: Stage6VerificationContext,
    dataset_id: str,
    dataset_version: str,
    reason: Stage6Reason,
) -> Stage6PrerequisiteReport:
    safe_id = dataset_id if _valid_dataset_id(dataset_id) else "invalid"
    safe_version = dataset_version if _valid_dataset_version(dataset_version) else "invalid"
    return Stage6PrerequisiteReport(
        status=Stage6PrerequisiteStatus.BLOCKED,
        verification_context=context,
        dataset_id=safe_id,
        dataset_version=safe_version,
        reasons=(reason,),
    )


def _expected_scope(context: Stage6VerificationContext) -> DatasetCardScope:
    return (
        DatasetCardScope.APPROVED_REAL_DATA
        if context is Stage6VerificationContext.REAL_PREREQUISITE
        else DatasetCardScope.SYNTHETIC_ONLY
    )


def _root_binding_sha256(
    *,
    context: Stage6VerificationContext,
    dataset_id: str,
    dataset_version: str,
    root: Path,
) -> str:
    return _canonical_sha256(
        {
            "context": context.value,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "purpose": DataUsePurpose.LOCKED_TEST_EVALUATION.value,
            "root": root.as_posix(),
        }
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6DataAuthorization:
    """Short-lived prerequisite capability; it cannot itself open the dataset root."""

    dataset_id: str
    dataset_version: str
    verification_context: Stage6VerificationContext
    purpose: DataUsePurpose
    _root: Path = field(repr=False, compare=False)
    _stage2_access_permit: DatasetAccessPermit | None = field(repr=False, compare=False)
    _stage2_package_capability: object = field(repr=False, compare=False)
    _stage3_package_capability: object = field(repr=False, compare=False)
    _frozen_data_card: FrozenDatasetCard | None = field(repr=False, compare=False)
    _manual_confirmation: Stage6ManualConfirmation | None = field(repr=False, compare=False)
    _locked_evidence: Stage4LockedEvidence | None = field(repr=False, compare=False)
    _stage5_pilot_report: object = field(repr=False, compare=False)
    binding_sha256: tuple[tuple[str, str], ...]
    root_binding_sha256: str
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._process_token is not _PROCESS_TOKEN or self._token is not _AUTHORIZATION_TOKEN:
            raise TypeError("Stage6DataAuthorization can only be issued by the Stage 6 Gate")

    @property
    def authorizes_file_open(self) -> bool:
        return False


_ISSUED_AUTHORIZATIONS: dict[int, tuple[Stage6DataAuthorization, str]] = {}
_BINDING_AUTHORIZATION_IDS: set[int] = set()
_BOUND_AUTHORIZATION_IDS: set[int] = set()


def _authorization_fingerprint(value: Stage6DataAuthorization) -> str:
    try:
        stage2 = _stage2_binding(value._stage2_package_capability)
        stage3 = _stage3_binding(value._stage3_package_capability)
    except (_CapabilityInvalid, _CapabilityUnavailable) as error:
        raise ValueError("package capability is no longer valid") from error
    payload: dict[str, object] = {
        "dataset_id": value.dataset_id,
        "dataset_version": value.dataset_version,
        "verification_context": value.verification_context.value,
        "purpose": value.purpose.value,
        "root": value._root.as_posix(),
        "stage2": stage2.digest(),
        "stage3": stage3.digest(),
        "binding_sha256": dict(value.binding_sha256),
        "root_binding_sha256": value.root_binding_sha256,
        "issued_process_id": value.issued_process_id,
        "issued_monotonic_ns": value.issued_monotonic_ns,
        "expires_monotonic_ns": value.expires_monotonic_ns,
        "ttl_seconds": value.ttl_seconds,
    }
    if value.verification_context is Stage6VerificationContext.REAL_PREREQUISITE:
        if (
            value._stage2_access_permit is None
            or value._frozen_data_card is None
            or value._manual_confirmation is None
            or value._locked_evidence is None
            or value._stage5_pilot_report is not None
        ):
            raise ValueError("real authorization has incomplete or synthetic authorities")
        payload.update(
            {
                "permit": _permit_snapshot(value._stage2_access_permit),
                "card": _canonical_sha256(value._frozen_data_card.to_dict()),
                "manual": value._manual_confirmation.binding_sha256(),
                "locked": value._locked_evidence.binding_sha256(),
            }
        )
    else:
        if any(
            item is not None
            for item in (
                value._stage2_access_permit,
                value._frozen_data_card,
                value._manual_confirmation,
                value._locked_evidence,
            )
        ):
            raise ValueError("synthetic authorization carries real-like authorities")
        payload["stage5_pilot"] = _stage5_report_binding(value._stage5_pilot_report)
    return _canonical_sha256(payload)


def _authorization_validation_reasons(
    value: object,
    *,
    required_context: Stage6VerificationContext,
    allow_bound: bool,
) -> tuple[str, ...]:
    if not isinstance(value, Stage6DataAuthorization):
        return ("authorization has an invalid type",)
    issued = _ISSUED_AUTHORIZATIONS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("authorization was not issued by the Stage 6 Gate",)
    try:
        if issued[1] != _authorization_fingerprint(value):
            return ("authorization bindings changed after issuance",)
        reasons: list[str] = []
        if value._process_token is not _PROCESS_TOKEN or value._token is not _AUTHORIZATION_TOKEN:
            reasons.append("authorization issuance identity is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("authorization belongs to a different process")
        if value.verification_context is not required_context:
            reasons.append(f"authorization context must be {required_context.value}")
        if value.purpose is not DataUsePurpose.LOCKED_TEST_EVALUATION:
            reasons.append("authorization purpose is not locked_test_evaluation")
        if id(value) in _BOUND_AUTHORIZATION_IDS and not allow_bound:
            reasons.append("authorization has already been bound")
        if not 1 <= value.ttl_seconds <= MAX_STAGE6_AUTHORIZATION_TTL_SECONDS:
            reasons.append("authorization TTL is invalid")
        if value.expires_monotonic_ns - value.issued_monotonic_ns != (
            value.ttl_seconds * 1_000_000_000
        ):
            reasons.append("authorization expiry does not match TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("authorization has expired")
        if value.verification_context is Stage6VerificationContext.REAL_PREREQUISITE:
            if value._stage2_access_permit is None or permit_validation_reasons(
                value._stage2_access_permit
            ):
                reasons.append("Stage 2 access permit is no longer valid")
        else:
            try:
                _stage5_report_binding(value._stage5_pilot_report)
            except (_CapabilityInvalid, _CapabilityUnavailable):
                reasons.append("synthetic Stage 5 pilot is no longer valid")
        expected_root = _root_binding_sha256(
            context=value.verification_context,
            dataset_id=value.dataset_id,
            dataset_version=value.dataset_version,
            root=value._root,
        )
        if expected_root != value.root_binding_sha256:
            reasons.append("authorization logical root binding changed")
    except (AttributeError, TypeError, ValueError):
        return ("authorization fields are invalid",)
    return tuple(dict.fromkeys(reasons))


def authorization_validation_reasons(
    value: object,
    *,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
) -> tuple[str, ...]:
    return _authorization_validation_reasons(
        value,
        required_context=required_context,
        allow_bound=False,
    )


def require_stage6_authorization(
    value: object,
    *,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
) -> Stage6DataAuthorization:
    reasons = authorization_validation_reasons(value, required_context=required_context)
    if reasons:
        raise Stage6AuthorizationError(reasons)
    assert isinstance(value, Stage6DataAuthorization)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6VerificationOutcome:
    report: Stage6PrerequisiteReport
    authorization: Stage6DataAuthorization | None

    def __post_init__(self) -> None:
        verified = self.report.status is Stage6PrerequisiteStatus.VERIFIED
        if verified != isinstance(self.authorization, Stage6DataAuthorization):
            raise ValueError("Stage 6 report and authorization status conflict")


def verify_stage6_prerequisites(
    *,
    verification_context: Stage6VerificationContext,
    dataset_id: str,
    dataset_version: str,
    dataset_root: Path,
    stage2_access_permit: object = None,
    stage2_package_capability: object = None,
    stage3_package_capability: object = None,
    frozen_data_card: object = None,
    manual_confirmation: object = None,
    locked_evidence: object = None,
    stage5_pilot_report: object = None,
    authorization_ttl_seconds: int = DEFAULT_STAGE6_AUTHORIZATION_TTL_SECONDS,
) -> Stage6VerificationOutcome:
    """Verify complete in-memory capabilities without touching ``dataset_root``."""

    def blocked(reason: Stage6Reason) -> Stage6VerificationOutcome:
        return Stage6VerificationOutcome(
            report=_blocked_report(
                context=(
                    verification_context
                    if isinstance(verification_context, Stage6VerificationContext)
                    else Stage6VerificationContext.REAL_PREREQUISITE
                ),
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                reason=reason,
            ),
            authorization=None,
        )

    if (
        not isinstance(verification_context, Stage6VerificationContext)
        or not _valid_dataset_id(dataset_id)
        or not _valid_dataset_version(dataset_version)
        or not isinstance(dataset_root, Path)
    ):
        return blocked(Stage6Reason.INPUT_INVALID)
    if not _root_is_lexically_valid(dataset_root):
        return blocked(Stage6Reason.ROOT_INVALID)
    if (
        type(authorization_ttl_seconds) is not int  # noqa: E721
        or not 1 <= authorization_ttl_seconds <= MAX_STAGE6_AUTHORIZATION_TTL_SECONDS
    ):
        return blocked(Stage6Reason.TTL_INVALID)

    if verification_context is Stage6VerificationContext.SYNTHETIC_FIXTURE:
        if any(
            item is not None
            for item in (
                stage2_access_permit,
                frozen_data_card,
                manual_confirmation,
                locked_evidence,
            )
        ):
            return blocked(Stage6Reason.INPUT_INVALID)
        try:
            stage2 = _stage2_binding(stage2_package_capability)
        except _CapabilityUnavailable:
            return blocked(Stage6Reason.STAGE2_PACKAGE_CAPABILITY_API_UNAVAILABLE)
        except _CapabilityInvalid:
            return blocked(Stage6Reason.STAGE2_PACKAGE_CAPABILITY_INVALID)
        try:
            stage3 = _stage3_binding(stage3_package_capability)
        except _CapabilityUnavailable:
            return blocked(Stage6Reason.STAGE3_PACKAGE_CAPABILITY_API_UNAVAILABLE)
        except _CapabilityInvalid:
            return blocked(Stage6Reason.STAGE3_PACKAGE_CAPABILITY_INVALID)
        if (
            stage2.scope != "SYNTHETIC_FIXTURE"
            or stage2.purpose != DataUsePurpose.READ_ONLY_AUDIT.value
        ):
            return blocked(Stage6Reason.SCOPE_MISMATCH)
        if (
            stage3.scope != "SYNTHETIC_TOY"
            or stage3.protocol_scope != "TOY_PROTOCOL"
            or stage3.fixture_only is not True
            or stage3.stage2_package_sha256 is not None
            or stage3.frozen_data_card_sha256 is not None
        ):
            return blocked(Stage6Reason.SCOPE_MISMATCH)
        try:
            _replay_package_capabilities(
                stage2_package_capability,
                stage3_package_capability,
            )
        except _CapabilityInvalid:
            return blocked(Stage6Reason.PROVENANCE_MISMATCH)
        try:
            stage5_sha256 = _stage5_report_binding(stage5_pilot_report)
        except _CapabilityUnavailable:
            return blocked(Stage6Reason.STAGE5_SYNTHETIC_PILOT_INVALID)
        except _CapabilityInvalid:
            return blocked(Stage6Reason.STAGE5_SYNTHETIC_PILOT_INVALID)
        bindings = tuple(
            sorted(
                {
                    "independent_stage2_package": stage2.digest(),
                    "independent_stage3_package": stage3.digest(),
                    "synthetic_stage5_pilot": stage5_sha256,
                }.items()
            )
        )
        root_binding = _root_binding_sha256(
            context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            root=dataset_root,
        )
        now = time.monotonic_ns()
        authorization = Stage6DataAuthorization(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            verification_context=verification_context,
            purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
            _root=dataset_root,
            _stage2_access_permit=None,
            _stage2_package_capability=stage2_package_capability,
            _stage3_package_capability=stage3_package_capability,
            _frozen_data_card=None,
            _manual_confirmation=None,
            _locked_evidence=None,
            _stage5_pilot_report=stage5_pilot_report,
            binding_sha256=bindings,
            root_binding_sha256=root_binding,
            issued_process_id=_PROCESS_ID,
            issued_monotonic_ns=now,
            expires_monotonic_ns=now + authorization_ttl_seconds * 1_000_000_000,
            ttl_seconds=authorization_ttl_seconds,
            _process_token=_PROCESS_TOKEN,
            _token=_AUTHORIZATION_TOKEN,
        )
        with _BOUNDARY_LOCK:
            _ISSUED_AUTHORIZATIONS[id(authorization)] = (
                authorization,
                _authorization_fingerprint(authorization),
            )
        report = Stage6PrerequisiteReport(
            status=Stage6PrerequisiteStatus.VERIFIED,
            verification_context=verification_context,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reasons=(),
            binding_sha256=bindings,
            root_binding_sha256=root_binding,
            authorization_ttl_seconds=authorization_ttl_seconds,
        )
        return Stage6VerificationOutcome(report=report, authorization=authorization)

    if stage5_pilot_report is not None:
        return blocked(Stage6Reason.INPUT_INVALID)
    if not isinstance(stage2_access_permit, DatasetAccessPermit) or permit_validation_reasons(
        stage2_access_permit
    ):
        return blocked(Stage6Reason.STAGE2_ACCESS_PERMIT_INVALID)
    permit = stage2_access_permit
    if (
        permit.purpose is not DataUsePurpose.LOCKED_TEST_EVALUATION
        or permit.root != dataset_root
        or permit.registry.dataset_id != dataset_id
        or permit.registry.dataset_version != dataset_version
        or permit.spec.dataset_id != dataset_id
        or not set(REQUIRED_STAGE6_REVIEW_PURPOSES) <= set(permit.review.permitted_purposes)
    ):
        return blocked(Stage6Reason.STAGE2_ACCESS_PERMIT_INVALID)
    try:
        stage2 = _stage2_binding(stage2_package_capability)
    except _CapabilityUnavailable:
        return blocked(Stage6Reason.STAGE2_PACKAGE_CAPABILITY_API_UNAVAILABLE)
    except _CapabilityInvalid:
        return blocked(Stage6Reason.STAGE2_PACKAGE_CAPABILITY_INVALID)
    try:
        stage3 = _stage3_binding(stage3_package_capability)
    except _CapabilityUnavailable:
        return blocked(Stage6Reason.STAGE3_PACKAGE_CAPABILITY_API_UNAVAILABLE)
    except _CapabilityInvalid:
        return blocked(Stage6Reason.STAGE3_PACKAGE_CAPABILITY_INVALID)
    try:
        _replay_package_capabilities(stage2_package_capability, stage3_package_capability)
    except _CapabilityInvalid:
        return blocked(Stage6Reason.PROVENANCE_MISMATCH)
    if not isinstance(frozen_data_card, FrozenDatasetCard) or frozen_data_card_validation_reasons(
        frozen_data_card
    ):
        return blocked(Stage6Reason.FROZEN_DATA_CARD_INVALID)
    if not isinstance(manual_confirmation, Stage6ManualConfirmation):
        return blocked(Stage6Reason.MANUAL_CONFIRMATION_INVALID)
    if not isinstance(locked_evidence, Stage4LockedEvidence):
        return blocked(Stage6Reason.LOCKED_EVIDENCE_INVALID)
    card = frozen_data_card
    manual = manual_confirmation
    locked = locked_evidence
    identities = (
        stage2.dataset_id,
        stage3.dataset_id,
        card.dataset_id,
        manual.dataset_id,
        locked.dataset_id,
    )
    if any(value != dataset_id for value in identities):
        return blocked(Stage6Reason.DATASET_MISMATCH)
    versions = (
        stage2.dataset_version,
        stage3.dataset_version,
        card.dataset_version,
        manual.dataset_version,
        locked.dataset_version,
    )
    if any(value != dataset_version for value in versions):
        return blocked(Stage6Reason.DATASET_VERSION_MISMATCH)
    expected_scope = _expected_scope(verification_context)
    if (
        stage2.scope != expected_scope.value
        or stage3.scope != expected_scope.value
        or stage3.fixture_only is not False
        or card.scope is not expected_scope
        or manual.scope is not expected_scope
    ):
        return blocked(Stage6Reason.SCOPE_MISMATCH)
    if stage2.purpose != DataUsePurpose.READ_ONLY_AUDIT.value:
        return blocked(Stage6Reason.SPLIT_MISMATCH)

    permit_registry_sha256 = _registry_sha256(permit)
    permit_review_sha256 = _canonical_sha256(permit.review.to_dict())
    permit_spec_sha256 = _canonical_sha256(permit.spec.to_dict())
    expected_stage2_root_sha256 = _canonical_sha256(
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "purpose": DataUsePurpose.READ_ONLY_AUDIT.value,
            "root": dataset_root.as_posix(),
        }
    )
    expected_stage2_permit_sha256 = _canonical_sha256(
        {
            "registry_sha256": permit_registry_sha256,
            "license_review_sha256": permit_review_sha256,
            "adapter_spec_sha256": permit_spec_sha256,
            "root_binding_sha256": expected_stage2_root_sha256,
            "purpose": DataUsePurpose.READ_ONLY_AUDIT.value,
        }
    )
    card_sha256 = _canonical_sha256(card.to_dict())
    manual_sha256 = manual.binding_sha256()
    locked_sha256 = locked.binding_sha256()
    if (
        card.manifest_sha256 != stage2.manifest_sha256
        or card.audit_sha256 != stage2.audit_sha256
        or card.registry_sha256 != permit_registry_sha256
        or card.license_review_sha256 != permit_review_sha256
        or card.adapter_spec_sha256 != permit_spec_sha256
        or card.sample_count != stage2.record_count
        or manual.identity_evidence_sha256 != card.identity_evidence_sha256
        or manual.label_semantics_evidence_sha256 != card.label_semantics_evidence_sha256
    ):
        return blocked(Stage6Reason.PROVENANCE_MISMATCH)
    if verification_context is Stage6VerificationContext.REAL_PREREQUISITE and (
        stage2.registry_sha256 != permit_registry_sha256
        or stage2.license_review_sha256 != permit_review_sha256
        or stage2.adapter_spec_sha256 != permit_spec_sha256
        or stage2.root_binding_sha256 != expected_stage2_root_sha256
        or stage2.permit_binding_sha256 != expected_stage2_permit_sha256
        or stage3.frozen_data_card_sha256 != card_sha256
        or stage3.stage2_package_sha256 != stage2.package_sha256
    ):
        return blocked(Stage6Reason.PROVENANCE_MISMATCH)
    if (
        locked.license_review_sha256 != permit_review_sha256
        or locked.adapter_spec_sha256 != permit_spec_sha256
        or locked.stage2_audit_sha256 != stage2.audit_sha256
        or locked.frozen_data_card_sha256 != card_sha256
        or locked.stage3_calibration_package_sha256 != stage3.package_sha256
        or locked.manual_confirmation_sha256 != manual_sha256
        or locked.calibration_artifact_sha256 != stage3.calibration_artifact_sha256
        or locked.config_sha256 != stage3.config_sha256
        or locked.data_sha256 != stage3.data_sha256
        or locked.model_sha256 != stage3.model_sha256
        or locked.alpha != stage3.alpha
        or locked.positive_threshold != stage3.positive_threshold
        or locked.negative_threshold != stage3.negative_threshold
        or locked.positive_calibration_sample_count != stage3.positive_sample_count
        or locked.negative_calibration_sample_count != stage3.negative_sample_count
        or locked.total_calibration_sample_count != stage3.total_sample_count
    ):
        return blocked(Stage6Reason.HASH_MISMATCH)

    bindings = tuple(
        sorted(
            {
                "stage2_access_permit": _canonical_sha256(_permit_snapshot(permit)),
                "stage2_package": stage2.digest(),
                "stage3_package": stage3.digest(),
                "frozen_data_card": card_sha256,
                "manual_confirmation": manual_sha256,
                "locked_evidence": locked_sha256,
            }.items()
        )
    )
    root_binding = _root_binding_sha256(
        context=verification_context,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        root=dataset_root,
    )
    now = time.monotonic_ns()
    authorization = Stage6DataAuthorization(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        verification_context=verification_context,
        purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
        _root=dataset_root,
        _stage2_access_permit=permit,
        _stage2_package_capability=stage2_package_capability,
        _stage3_package_capability=stage3_package_capability,
        _frozen_data_card=card,
        _manual_confirmation=manual,
        _locked_evidence=locked,
        _stage5_pilot_report=None,
        binding_sha256=bindings,
        root_binding_sha256=root_binding,
        issued_process_id=_PROCESS_ID,
        issued_monotonic_ns=now,
        expires_monotonic_ns=now + authorization_ttl_seconds * 1_000_000_000,
        ttl_seconds=authorization_ttl_seconds,
        _process_token=_PROCESS_TOKEN,
        _token=_AUTHORIZATION_TOKEN,
    )
    with _BOUNDARY_LOCK:
        _ISSUED_AUTHORIZATIONS[id(authorization)] = (
            authorization,
            _authorization_fingerprint(authorization),
        )
    report = Stage6PrerequisiteReport(
        status=Stage6PrerequisiteStatus.VERIFIED,
        verification_context=verification_context,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        reasons=(),
        binding_sha256=bindings,
        root_binding_sha256=root_binding,
        authorization_ttl_seconds=authorization_ttl_seconds,
    )
    return Stage6VerificationOutcome(report=report, authorization=authorization)


def _is_link_or_reparse(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _strict_physical_root_identity(root: Path) -> tuple[int, int, int, int, int, int]:
    if not _root_is_lexically_valid(root):
        raise Stage6RootBindingError(("root path is not a safe absolute dataset root",))
    final_metadata: os.stat_result | None = None
    for candidate in (*reversed(root.parents), root):
        try:
            metadata = os.lstat(candidate)
        except OSError as error:
            raise Stage6RootBindingError(("root or an ancestor does not exist",)) from error
        if _is_link_or_reparse(metadata):
            raise Stage6RootBindingError(("root path contains a link or reparse point",))
        if candidate == root:
            final_metadata = metadata
    if final_metadata is None or not stat.S_ISDIR(final_metadata.st_mode):
        raise Stage6RootBindingError(("authorized root is not a directory",))
    return (
        final_metadata.st_dev,
        final_metadata.st_ino,
        final_metadata.st_mode,
        final_metadata.st_size,
        final_metadata.st_mtime_ns,
        final_metadata.st_ctime_ns,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6BoundRoot:
    """One-shot physical binding; only its consumer may return the root path."""

    authorization: Stage6DataAuthorization = field(repr=False, compare=False)
    _root: Path = field(repr=False, compare=False)
    authorization_snapshot_sha256: str
    physical_identity: tuple[int, int, int, int, int, int]
    bound_process_id: int
    bound_monotonic_ns: int
    expires_monotonic_ns: int
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._process_token is not _PROCESS_TOKEN or self._token is not _BOUND_ROOT_TOKEN:
            raise TypeError("Stage6BoundRoot can only be issued by bind_stage6_authorized_root")

    @property
    def authorizes_file_open(self) -> bool:
        return False


_ISSUED_BOUND_ROOTS: dict[int, tuple[Stage6BoundRoot, str]] = {}
_CONSUMED_BOUND_ROOT_IDS: set[int] = set()


def _bound_root_fingerprint(value: Stage6BoundRoot) -> str:
    return _canonical_sha256(
        {
            "authorization_id": id(value.authorization),
            "authorization_snapshot_sha256": value.authorization_snapshot_sha256,
            "root": value._root.as_posix(),
            "physical_identity": value.physical_identity,
            "bound_process_id": value.bound_process_id,
            "bound_monotonic_ns": value.bound_monotonic_ns,
            "expires_monotonic_ns": value.expires_monotonic_ns,
        }
    )


def bind_stage6_authorized_root(
    value: object,
    *,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
) -> Stage6BoundRoot:
    try:
        authorization = require_stage6_authorization(value, required_context=required_context)
    except Stage6AuthorizationError as error:
        raise Stage6RootBindingError(error.reasons) from error
    authorization_id = id(authorization)
    with _BOUNDARY_LOCK:
        if authorization_id in _BOUND_AUTHORIZATION_IDS or (
            authorization_id in _BINDING_AUTHORIZATION_IDS
        ):
            raise Stage6RootBindingError(("authorization has already been bound",))
        _BINDING_AUTHORIZATION_IDS.add(authorization_id)
    try:
        try:
            _replay_package_capabilities(
                authorization._stage2_package_capability,
                authorization._stage3_package_capability,
            )
        except _CapabilityInvalid as error:
            raise Stage6RootBindingError(("full package replay failed before binding",)) from error
        physical_identity = _strict_physical_root_identity(authorization._root)
        now = time.monotonic_ns()
        if now >= authorization.expires_monotonic_ns:
            raise Stage6RootBindingError(("authorization expired before physical binding",))
        bound = Stage6BoundRoot(
            authorization=authorization,
            _root=authorization._root,
            authorization_snapshot_sha256=_authorization_fingerprint(authorization),
            physical_identity=physical_identity,
            bound_process_id=_PROCESS_ID,
            bound_monotonic_ns=now,
            expires_monotonic_ns=authorization.expires_monotonic_ns,
            _process_token=_PROCESS_TOKEN,
            _token=_BOUND_ROOT_TOKEN,
        )
        with _BOUNDARY_LOCK:
            _ISSUED_BOUND_ROOTS[id(bound)] = (bound, _bound_root_fingerprint(bound))
            _BOUND_AUTHORIZATION_IDS.add(authorization_id)
        return bound
    finally:
        with _BOUNDARY_LOCK:
            _BINDING_AUTHORIZATION_IDS.discard(authorization_id)


def bound_root_validation_reasons(
    value: object,
    *,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
) -> tuple[str, ...]:
    if not isinstance(value, Stage6BoundRoot):
        return ("bound root has an invalid type",)
    issued = _ISSUED_BOUND_ROOTS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("bound root was not issued by the Stage 6 Gate",)
    if id(value) in _CONSUMED_BOUND_ROOT_IDS:
        return ("bound root has already been consumed",)
    try:
        if issued[1] != _bound_root_fingerprint(value):
            return ("bound root bindings changed after issuance",)
        reasons = list(
            _authorization_validation_reasons(
                value.authorization,
                required_context=required_context,
                allow_bound=True,
            )
        )
        if value.bound_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("bound root belongs to a different process")
        if value._process_token is not _PROCESS_TOKEN or value._token is not _BOUND_ROOT_TOKEN:
            reasons.append("bound root issuance identity is invalid")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("bound root has expired")
        if value.authorization_snapshot_sha256 != _authorization_fingerprint(value.authorization):
            reasons.append("bound root authorization snapshot changed")
        try:
            _replay_package_capabilities(
                value.authorization._stage2_package_capability,
                value.authorization._stage3_package_capability,
            )
        except _CapabilityInvalid:
            reasons.append("full package replay failed before root consumption")
        try:
            current_identity = _strict_physical_root_identity(value._root)
        except Stage6RootBindingError as error:
            reasons.extend(error.reasons)
        else:
            if current_identity != value.physical_identity:
                reasons.append("bound root physical identity changed")
    except (AttributeError, TypeError, ValueError):
        return ("bound root fields are invalid",)
    return tuple(dict.fromkeys(reasons))


def consume_bound_stage6_root(
    value: object,
    *,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
) -> Path:
    with _BOUNDARY_LOCK:
        reasons = bound_root_validation_reasons(value, required_context=required_context)
        if reasons:
            raise Stage6RootBindingError(reasons)
        assert isinstance(value, Stage6BoundRoot)
        _CONSUMED_BOUND_ROOT_IDS.add(id(value))
        return value._root


__all__ = [
    "DEFAULT_STAGE6_AUTHORIZATION_TTL_SECONDS",
    "MAX_STAGE6_AUTHORIZATION_TTL_SECONDS",
    "REQUIRED_STAGE6_REVIEW_PURPOSES",
    "STAGE6_GATE_SCHEMA_VERSION",
    "Stage6AuthorizationError",
    "Stage6BoundRoot",
    "Stage6DataAuthorization",
    "Stage6ManualConfirmation",
    "Stage6PreflightReport",
    "Stage6PreflightStatus",
    "Stage6PrerequisiteReport",
    "Stage6PrerequisiteStatus",
    "Stage6Reason",
    "Stage6RootBindingError",
    "Stage6VerificationContext",
    "Stage6VerificationOutcome",
    "assess_stage6_preflight",
    "authorization_validation_reasons",
    "bind_stage6_authorized_root",
    "bound_root_validation_reasons",
    "consume_bound_stage6_root",
    "require_stage6_authorization",
    "verify_stage6_prerequisites",
]
