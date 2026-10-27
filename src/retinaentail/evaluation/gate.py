"""Zero-I/O Stage 4 preflight and immutable lock-summary contracts.

This module deliberately accepts no filesystem paths.  A complete report only means
that an independent prerequisite verifier may begin comparing supplied artifacts; it
never grants dataset-root access and never claims that the artifacts are authentic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from retinaentail.data.license_gate import DataUsePurpose

STAGE4_LOCKED_EVIDENCE_SCHEMA_VERSION: Final = "4.0.0"
STAGE4_PREFLIGHT_SCHEMA_VERSION: Final = "4.0.0"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")


Stage4ReviewPurpose = DataUsePurpose


REQUIRED_STAGE4_REVIEW_PURPOSES: Final = (
    Stage4ReviewPurpose.READ_ONLY_AUDIT,
    Stage4ReviewPurpose.LESION_CALIBRATION,
    Stage4ReviewPurpose.LOCKED_TEST_EVALUATION,
)


class Stage4PreflightStatus(StrEnum):
    """Whether values are present for a later, independent evidence verification."""

    BLOCKED = "BLOCKED"
    READY_FOR_PREREQUISITE_VERIFICATION = "READY_FOR_PREREQUISITE_VERIFICATION"


class Stage4PreflightReason(StrEnum):
    """Controlled zero-I/O reasons that cannot expose a dataset path."""

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
    STAGE3_CALIBRATION_PACKAGE_NOT_PROVIDED = "STAGE3_CALIBRATION_PACKAGE_NOT_PROVIDED"
    MANUAL_CONFIRMATION_RECORD_NOT_PROVIDED = "MANUAL_CONFIRMATION_RECORD_NOT_PROVIDED"
    LOCKED_EVIDENCE_NOT_PROVIDED = "LOCKED_EVIDENCE_NOT_PROVIDED"
    LOCKED_EVIDENCE_DATASET_MISMATCH = "LOCKED_EVIDENCE_DATASET_MISMATCH"


def _require_dataset_id(value: object) -> str:
    if not isinstance(value, str) or _DATASET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("dataset_id must be a normalized identifier")
    return value


def _require_dataset_version(value: object) -> str:
    if not isinstance(value, str) or _DATASET_VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError("dataset_version must be a report-safe exact version")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_finite_number(
    value: object,
    field_name: str,
    *,
    strictly_positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (strictly_positive and result <= 0.0):
        raise ValueError(f"{field_name} must be a finite nonnegative number")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage4LockedEvidence:
    """Path-free summary of values that must later be verified against exact artifacts.

    The calibration artifact digest binds its serialized alpha, thresholds, sample
    counts, method, and provenance.  The repeated scalar fields make the intended lock
    explicit for reviewers.  This summary is not an access capability.
    """

    dataset_id: str
    dataset_version: str
    license_review_sha256: str
    adapter_spec_sha256: str
    stage2_audit_sha256: str
    frozen_data_card_sha256: str
    stage3_calibration_package_sha256: str
    manual_confirmation_sha256: str
    calibration_artifact_sha256: str
    config_sha256: str
    data_sha256: str
    model_sha256: str
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_calibration_sample_count: int
    negative_calibration_sample_count: int
    total_calibration_sample_count: int
    stage2_audit_status: str = "PASS"
    frozen_data_card_scope: str = "APPROVED_REAL_DATA"
    stage3_calibration_status: str = "SUCCESS"
    manual_confirmations_complete: bool = True
    thresholds_locked_before_test: bool = True
    test_split_used_for_calibration: bool = False
    test_tuning_prohibited: bool = True
    schema_version: str = STAGE4_LOCKED_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE4_LOCKED_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 4 locked-evidence schema_version")
        _require_dataset_id(self.dataset_id)
        _require_dataset_version(self.dataset_version)
        for field_name in (
            "license_review_sha256",
            "adapter_spec_sha256",
            "stage2_audit_sha256",
            "frozen_data_card_sha256",
            "stage3_calibration_package_sha256",
            "manual_confirmation_sha256",
            "calibration_artifact_sha256",
            "config_sha256",
            "data_sha256",
            "model_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        alpha = _require_finite_number(self.alpha, "alpha", strictly_positive=True)
        if alpha >= 1.0:
            raise ValueError("alpha must be less than one")
        _require_finite_number(
            self.positive_threshold,
            "positive_threshold",
            strictly_positive=False,
        )
        _require_finite_number(
            self.negative_threshold,
            "negative_threshold",
            strictly_positive=False,
        )
        positive_count = _require_positive_int(
            self.positive_calibration_sample_count,
            "positive_calibration_sample_count",
        )
        negative_count = _require_positive_int(
            self.negative_calibration_sample_count,
            "negative_calibration_sample_count",
        )
        total_count = _require_positive_int(
            self.total_calibration_sample_count,
            "total_calibration_sample_count",
        )
        if total_count != positive_count + negative_count:
            raise ValueError("total calibration count must equal positive plus negative counts")
        if self.stage2_audit_status != "PASS":
            raise ValueError("Stage 2 audit status must be PASS")
        if self.frozen_data_card_scope != "APPROVED_REAL_DATA":
            raise ValueError("frozen data card must have APPROVED_REAL_DATA scope")
        if self.stage3_calibration_status != "SUCCESS":
            raise ValueError("Stage 3 calibration status must be SUCCESS")
        if type(self.manual_confirmations_complete) is not bool:  # noqa: E721
            raise TypeError("manual_confirmations_complete must be an exact bool")
        if type(self.thresholds_locked_before_test) is not bool:  # noqa: E721
            raise TypeError("thresholds_locked_before_test must be an exact bool")
        if type(self.test_split_used_for_calibration) is not bool:  # noqa: E721
            raise TypeError("test_split_used_for_calibration must be an exact bool")
        if type(self.test_tuning_prohibited) is not bool:  # noqa: E721
            raise TypeError("test_tuning_prohibited must be an exact bool")
        if not self.manual_confirmations_complete:
            raise ValueError("manual confirmations must be complete before locking")
        if not self.thresholds_locked_before_test:
            raise ValueError("thresholds must be locked before any test evaluation")
        if self.test_split_used_for_calibration:
            raise ValueError("test split cannot be used for calibration")
        if not self.test_tuning_prohibited:
            raise ValueError("test tuning must remain prohibited")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "license_review_sha256": self.license_review_sha256,
            "adapter_spec_sha256": self.adapter_spec_sha256,
            "stage2_audit_sha256": self.stage2_audit_sha256,
            "frozen_data_card_sha256": self.frozen_data_card_sha256,
            "stage3_calibration_package_sha256": self.stage3_calibration_package_sha256,
            "manual_confirmation_sha256": self.manual_confirmation_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "model_sha256": self.model_sha256,
            "alpha": self.alpha,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "positive_calibration_sample_count": self.positive_calibration_sample_count,
            "negative_calibration_sample_count": self.negative_calibration_sample_count,
            "total_calibration_sample_count": self.total_calibration_sample_count,
            "stage2_audit_status": self.stage2_audit_status,
            "frozen_data_card_scope": self.frozen_data_card_scope,
            "stage3_calibration_status": self.stage3_calibration_status,
            "manual_confirmations_complete": self.manual_confirmations_complete,
            "thresholds_locked_before_test": self.thresholds_locked_before_test,
            "test_split_used_for_calibration": self.test_split_used_for_calibration,
            "test_tuning_prohibited": self.test_tuning_prohibited,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage4LockedEvidence:
        expected = {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "license_review_sha256",
            "adapter_spec_sha256",
            "stage2_audit_sha256",
            "frozen_data_card_sha256",
            "stage3_calibration_package_sha256",
            "manual_confirmation_sha256",
            "calibration_artifact_sha256",
            "config_sha256",
            "data_sha256",
            "model_sha256",
            "alpha",
            "positive_threshold",
            "negative_threshold",
            "positive_calibration_sample_count",
            "negative_calibration_sample_count",
            "total_calibration_sample_count",
            "stage2_audit_status",
            "frozen_data_card_scope",
            "stage3_calibration_status",
            "manual_confirmations_complete",
            "thresholds_locked_before_test",
            "test_split_used_for_calibration",
            "test_tuning_prohibited",
        }
        if set(value) != expected:
            raise ValueError("Stage 4 locked evidence has missing or unknown fields")
        return cls(**value)  # type: ignore[arg-type]

    def binding_sha256(self) -> str:
        """Hash the canonical summary without opening any referenced artifact."""

        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _expected_reasons(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    approved_review_provided: bool,
    approved_review_purposes: tuple[Stage4ReviewPurpose, ...],
    dataset_root_provided: bool,
    adapter_spec_provided: bool,
    stage2_pass_audit_provided: bool,
    frozen_data_card_provided: bool,
    stage3_calibration_package_provided: bool,
    manual_confirmation_record_provided: bool,
    locked_evidence_provided: bool,
    locked_evidence: Stage4LockedEvidence | None,
) -> tuple[Stage4PreflightReason, ...]:
    reasons: list[Stage4PreflightReason] = []
    if dataset_id is None:
        reasons.append(Stage4PreflightReason.DATASET_NOT_EXPLICIT)
    if dataset_version is None:
        reasons.append(Stage4PreflightReason.DATASET_VERSION_NOT_EXPLICIT)
    if not approved_review_provided:
        reasons.append(Stage4PreflightReason.APPROVED_REVIEW_NOT_PROVIDED)
    purpose_reasons = {
        Stage4ReviewPurpose.READ_ONLY_AUDIT: (
            Stage4PreflightReason.READ_ONLY_AUDIT_PURPOSE_NOT_APPROVED
        ),
        Stage4ReviewPurpose.LESION_CALIBRATION: (
            Stage4PreflightReason.LESION_CALIBRATION_PURPOSE_NOT_APPROVED
        ),
        Stage4ReviewPurpose.LOCKED_TEST_EVALUATION: (
            Stage4PreflightReason.LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED
        ),
    }
    purpose_set = set(approved_review_purposes)
    for purpose in REQUIRED_STAGE4_REVIEW_PURPOSES:
        if purpose not in purpose_set:
            reasons.append(purpose_reasons[purpose])
    if not dataset_root_provided:
        reasons.append(Stage4PreflightReason.DATASET_ROOT_NOT_PROVIDED)
    if not adapter_spec_provided:
        reasons.append(Stage4PreflightReason.ADAPTER_SPEC_NOT_PROVIDED)
    if not stage2_pass_audit_provided:
        reasons.append(Stage4PreflightReason.STAGE2_PASS_AUDIT_NOT_PROVIDED)
    if not frozen_data_card_provided:
        reasons.append(Stage4PreflightReason.FROZEN_DATA_CARD_NOT_PROVIDED)
    if not stage3_calibration_package_provided:
        reasons.append(Stage4PreflightReason.STAGE3_CALIBRATION_PACKAGE_NOT_PROVIDED)
    if not manual_confirmation_record_provided:
        reasons.append(Stage4PreflightReason.MANUAL_CONFIRMATION_RECORD_NOT_PROVIDED)
    if not locked_evidence_provided:
        reasons.append(Stage4PreflightReason.LOCKED_EVIDENCE_NOT_PROVIDED)
    if (
        locked_evidence is not None
        and dataset_id is not None
        and dataset_version is not None
        and (
            locked_evidence.dataset_id != dataset_id
            or locked_evidence.dataset_version != dataset_version
        )
    ):
        reasons.append(Stage4PreflightReason.LOCKED_EVIDENCE_DATASET_MISMATCH)
    return tuple(reasons)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage4PreflightReport:
    """A zero-I/O presence report that can never authorize dataset access."""

    status: Stage4PreflightStatus
    dataset_id: str | None
    dataset_version: str | None
    approved_review_provided: bool
    approved_review_purposes: tuple[Stage4ReviewPurpose, ...]
    dataset_root_provided: bool
    adapter_spec_provided: bool
    stage2_pass_audit_provided: bool
    frozen_data_card_provided: bool
    stage3_calibration_package_provided: bool
    manual_confirmation_record_provided: bool
    locked_evidence_provided: bool
    locked_evidence: Stage4LockedEvidence | None
    reasons: tuple[Stage4PreflightReason, ...]
    schema_version: str = STAGE4_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE4_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 4 preflight schema_version")
        if not isinstance(self.status, Stage4PreflightStatus):
            raise TypeError("status must be Stage4PreflightStatus")
        if self.dataset_id is not None:
            _require_dataset_id(self.dataset_id)
        if self.dataset_version is not None:
            _require_dataset_version(self.dataset_version)
        presence = (
            self.approved_review_provided,
            self.dataset_root_provided,
            self.adapter_spec_provided,
            self.stage2_pass_audit_provided,
            self.frozen_data_card_provided,
            self.stage3_calibration_package_provided,
            self.manual_confirmation_record_provided,
            self.locked_evidence_provided,
        )
        if any(type(value) is not bool for value in presence):  # noqa: E721
            raise TypeError("Stage 4 preflight presence fields must be exact bool values")
        if not all(
            isinstance(purpose, Stage4ReviewPurpose) for purpose in self.approved_review_purposes
        ):
            raise TypeError("approved_review_purposes must contain Stage4ReviewPurpose values")
        if len(set(self.approved_review_purposes)) != len(self.approved_review_purposes):
            raise ValueError("approved_review_purposes must be unique")
        if self.locked_evidence is not None and not isinstance(
            self.locked_evidence, Stage4LockedEvidence
        ):
            raise TypeError("locked_evidence must use Stage4LockedEvidence or be null")
        if not all(isinstance(reason, Stage4PreflightReason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage4PreflightReason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("preflight reasons must be unique")
        expected = _expected_reasons(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            approved_review_provided=self.approved_review_provided,
            approved_review_purposes=self.approved_review_purposes,
            dataset_root_provided=self.dataset_root_provided,
            adapter_spec_provided=self.adapter_spec_provided,
            stage2_pass_audit_provided=self.stage2_pass_audit_provided,
            frozen_data_card_provided=self.frozen_data_card_provided,
            stage3_calibration_package_provided=self.stage3_calibration_package_provided,
            manual_confirmation_record_provided=self.manual_confirmation_record_provided,
            locked_evidence_provided=self.locked_evidence_provided,
            locked_evidence=self.locked_evidence,
        )
        expected_status = (
            Stage4PreflightStatus.BLOCKED
            if expected
            else Stage4PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
        )
        if self.status is not expected_status or self.reasons != expected:
            raise ValueError("preflight status/reasons conflict with explicit input presence")

    @property
    def authorizes_data_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
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
            "stage3_calibration_package_provided": (self.stage3_calibration_package_provided),
            "manual_confirmation_record_provided": (self.manual_confirmation_record_provided),
            "locked_evidence_provided": self.locked_evidence_provided,
            "locked_evidence": (
                None if self.locked_evidence is None else self.locked_evidence.to_dict()
            ),
            "locked_evidence_binding_sha256": (
                None if self.locked_evidence is None else self.locked_evidence.binding_sha256()
            ),
            "reasons": [reason.value for reason in self.reasons],
            "test_tuning_prohibited": True,
            "prerequisites_verified": False,
            "authorizes_data_access": False,
            "real_data_read": False,
            "sample_count": 0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage4PreflightReport:
        required = {
            "schema_version",
            "status",
            "dataset_id",
            "dataset_version",
            "approved_review_provided",
            "approved_review_purposes",
            "dataset_root_provided",
            "adapter_spec_provided",
            "stage2_pass_audit_provided",
            "frozen_data_card_provided",
            "stage3_calibration_package_provided",
            "manual_confirmation_record_provided",
            "locked_evidence_provided",
            "locked_evidence",
            "locked_evidence_binding_sha256",
            "reasons",
            "test_tuning_prohibited",
            "prerequisites_verified",
            "authorizes_data_access",
            "real_data_read",
            "sample_count",
        }
        if set(value) != required:
            raise ValueError("Stage 4 preflight has missing or unknown fields")
        if value["test_tuning_prohibited"] is not True:
            raise ValueError("Stage 4 preflight must prohibit test tuning")
        if value["prerequisites_verified"] is not False:
            raise ValueError("preflight cannot claim prerequisite verification")
        if value["authorizes_data_access"] is not False:
            raise ValueError("preflight cannot authorize data access")
        if value["real_data_read"] is not False or value["sample_count"] != 0:
            raise ValueError("preflight must contain zero real-data activity")

        raw_purposes = value["approved_review_purposes"]
        raw_reasons = value["reasons"]
        if not isinstance(raw_purposes, list) or not all(
            isinstance(item, str) for item in raw_purposes
        ):
            raise ValueError("approved_review_purposes must be a list of strings")
        if not isinstance(raw_reasons, list) or not all(
            isinstance(item, str) for item in raw_reasons
        ):
            raise ValueError("preflight reasons must be a list of strings")
        raw_status = value["status"]
        if not isinstance(raw_status, str):
            raise ValueError("preflight status must be a string")
        try:
            purposes = tuple(Stage4ReviewPurpose(item) for item in raw_purposes)
            reasons = tuple(Stage4PreflightReason(item) for item in raw_reasons)
            status = Stage4PreflightStatus(raw_status)
        except (TypeError, ValueError) as error:
            raise ValueError("preflight status, purpose, or reason is invalid") from error

        raw_evidence = value["locked_evidence"]
        evidence: Stage4LockedEvidence | None
        if raw_evidence is None:
            evidence = None
        elif isinstance(raw_evidence, Mapping):
            evidence = Stage4LockedEvidence.from_dict(raw_evidence)
        else:
            raise ValueError("locked_evidence must be an object or null")
        expected_binding = None if evidence is None else evidence.binding_sha256()
        if value["locked_evidence_binding_sha256"] != expected_binding:
            raise ValueError("locked evidence binding digest does not match payload")

        schema_version = value["schema_version"]
        dataset_id = value["dataset_id"]
        dataset_version = value["dataset_version"]
        if not isinstance(schema_version, str):
            raise ValueError("schema_version must be a string")
        if dataset_id is not None and not isinstance(dataset_id, str):
            raise ValueError("dataset_id must be a string or null")
        if dataset_version is not None and not isinstance(dataset_version, str):
            raise ValueError("dataset_version must be a string or null")
        presence_names = (
            "approved_review_provided",
            "dataset_root_provided",
            "adapter_spec_provided",
            "stage2_pass_audit_provided",
            "frozen_data_card_provided",
            "stage3_calibration_package_provided",
            "manual_confirmation_record_provided",
            "locked_evidence_provided",
        )
        presence_values = tuple(value[name] for name in presence_names)
        if any(type(item) is not bool for item in presence_values):  # noqa: E721
            raise ValueError("preflight presence fields must be booleans")
        (
            review_provided,
            root_provided,
            spec_provided,
            audit_provided,
            card_provided,
            calibration_provided,
            confirmations_provided,
            lock_provided,
        ) = presence_values
        assert isinstance(review_provided, bool)
        assert isinstance(root_provided, bool)
        assert isinstance(spec_provided, bool)
        assert isinstance(audit_provided, bool)
        assert isinstance(card_provided, bool)
        assert isinstance(calibration_provided, bool)
        assert isinstance(confirmations_provided, bool)
        assert isinstance(lock_provided, bool)
        return cls(
            schema_version=schema_version,
            status=status,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            approved_review_provided=review_provided,
            approved_review_purposes=purposes,
            dataset_root_provided=root_provided,
            adapter_spec_provided=spec_provided,
            stage2_pass_audit_provided=audit_provided,
            frozen_data_card_provided=card_provided,
            stage3_calibration_package_provided=calibration_provided,
            manual_confirmation_record_provided=confirmations_provided,
            locked_evidence_provided=lock_provided,
            locked_evidence=evidence,
            reasons=reasons,
        )


def assess_stage4_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    approved_review_provided: bool,
    approved_review_purposes: tuple[Stage4ReviewPurpose, ...],
    dataset_root_provided: bool,
    adapter_spec_provided: bool,
    stage2_pass_audit_provided: bool,
    frozen_data_card_provided: bool,
    stage3_calibration_package_provided: bool,
    manual_confirmation_record_provided: bool,
    locked_evidence_provided: bool,
    locked_evidence: Stage4LockedEvidence | None,
) -> Stage4PreflightReport:
    """Assess in-memory values only; never resolve, stat, list, or open a path."""

    presence = (
        approved_review_provided,
        dataset_root_provided,
        adapter_spec_provided,
        stage2_pass_audit_provided,
        frozen_data_card_provided,
        stage3_calibration_package_provided,
        manual_confirmation_record_provided,
        locked_evidence_provided,
    )
    if any(type(value) is not bool for value in presence):  # noqa: E721
        raise TypeError("Stage 4 preflight presence fields must be exact bool values")
    if not isinstance(approved_review_purposes, tuple) or not all(
        isinstance(purpose, Stage4ReviewPurpose) for purpose in approved_review_purposes
    ):
        raise TypeError("approved_review_purposes must be a tuple of Stage4ReviewPurpose")
    if locked_evidence is not None and not isinstance(locked_evidence, Stage4LockedEvidence):
        raise TypeError("locked_evidence must use Stage4LockedEvidence or be null")
    reasons = _expected_reasons(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_provided=approved_review_provided,
        approved_review_purposes=approved_review_purposes,
        dataset_root_provided=dataset_root_provided,
        adapter_spec_provided=adapter_spec_provided,
        stage2_pass_audit_provided=stage2_pass_audit_provided,
        frozen_data_card_provided=frozen_data_card_provided,
        stage3_calibration_package_provided=stage3_calibration_package_provided,
        manual_confirmation_record_provided=manual_confirmation_record_provided,
        locked_evidence_provided=locked_evidence_provided,
        locked_evidence=locked_evidence,
    )
    return Stage4PreflightReport(
        status=(
            Stage4PreflightStatus.BLOCKED
            if reasons
            else Stage4PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_provided=approved_review_provided,
        approved_review_purposes=approved_review_purposes,
        dataset_root_provided=dataset_root_provided,
        adapter_spec_provided=adapter_spec_provided,
        stage2_pass_audit_provided=stage2_pass_audit_provided,
        frozen_data_card_provided=frozen_data_card_provided,
        stage3_calibration_package_provided=stage3_calibration_package_provided,
        manual_confirmation_record_provided=manual_confirmation_record_provided,
        locked_evidence_provided=locked_evidence_provided,
        locked_evidence=locked_evidence,
        reasons=reasons,
    )


__all__ = [
    "REQUIRED_STAGE4_REVIEW_PURPOSES",
    "STAGE4_LOCKED_EVIDENCE_SCHEMA_VERSION",
    "STAGE4_PREFLIGHT_SCHEMA_VERSION",
    "Stage4LockedEvidence",
    "Stage4PreflightReason",
    "Stage4PreflightReport",
    "Stage4PreflightStatus",
    "Stage4ReviewPurpose",
    "assess_stage4_preflight",
]
