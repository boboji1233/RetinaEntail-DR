"""Fail-closed Stage 3 dataset-card freezing contracts.

This module consumes only in-memory Stage 2 evidence.  It never resolves,
stats, scans, reads, or writes the dataset root carried by an access permit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from retinaentail.data.audit import AUDIT_SCHEMA_VERSION, AuditResult, AuditStatus
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    permit_validation_reasons,
)
from retinaentail.data.stage2_models import (
    DATASET_LABEL_SEMANTICS,
    DatasetManifestV2,
    LabelAvailability,
    LabelSemantic,
    ManifestRecordV2,
)

DATA_CARD_SCHEMA_VERSION: Final = "3.0.0"
_CARD_TOKEN: Final = object()
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")

_LESION_PROPOSITION_SEMANTICS: Final = frozenset(
    {
        LabelSemantic.MICROANEURYSM_MASK,
        LabelSemantic.HEMORRHAGE_MASK,
        LabelSemantic.HARD_EXUDATE_MASK,
        LabelSemantic.SOFT_EXUDATE_MASK,
        LabelSemantic.LESION_DETECTION,
    }
)


class DataCardFreezeStatus(StrEnum):
    """Outcome of a dataset-card freeze attempt."""

    FROZEN = "FROZEN"
    BLOCKED = "BLOCKED"
    FAIL_CLOSED = "FAIL_CLOSED"


class DatasetCardScope(StrEnum):
    """Whether a card represents approved real data or an explicit fixture."""

    APPROVED_REAL_DATA = "APPROVED_REAL_DATA"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"


class DatasetSplit(StrEnum):
    """The only split names accepted by the Stage 3 calibration protocol."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class DataCardReason(StrEnum):
    """Path-free failure codes safe for blocked reports."""

    NO_APPROVED_DATASET = "NO_APPROVED_DATASET"
    INPUT_TYPE_INVALID = "INPUT_TYPE_INVALID"
    ACCESS_PERMIT_INVALID = "ACCESS_PERMIT_INVALID"
    CARD_ISSUANCE_INVALID = "CARD_ISSUANCE_INVALID"
    CARD_BINDING_MISMATCH = "CARD_BINDING_MISMATCH"
    ACCESS_PURPOSE_INVALID = "ACCESS_PURPOSE_INVALID"
    CALIBRATION_PURPOSE_NOT_APPROVED = "CALIBRATION_PURPOSE_NOT_APPROVED"
    MANIFEST_PERMIT_MISMATCH = "MANIFEST_PERMIT_MISMATCH"
    AUDIT_BLOCKED = "AUDIT_BLOCKED"
    AUDIT_FAILED = "AUDIT_FAILED"
    AUDIT_BINDING_MISMATCH = "AUDIT_BINDING_MISMATCH"
    AUDIT_COUNTER_MISMATCH = "AUDIT_COUNTER_MISMATCH"
    LABEL_SEMANTICS_MISSING = "LABEL_SEMANTICS_MISSING"
    LABEL_SEMANTICS_INVALID = "LABEL_SEMANTICS_INVALID"
    LABEL_UNKNOWN_OR_MISSING = "LABEL_UNKNOWN_OR_MISSING"
    IDENTITY_CONFIRMATION_MISSING = "IDENTITY_CONFIRMATION_MISSING"
    SPLIT_CONFIRMATION_MISSING = "SPLIT_CONFIRMATION_MISSING"
    SPLIT_MISSING_OR_UNDECLARED = "SPLIT_MISSING_OR_UNDECLARED"
    SPLIT_COUNT_INVALID = "SPLIT_COUNT_INVALID"
    SPLIT_COUNT_MISMATCH = "SPLIT_COUNT_MISMATCH"
    SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
    HASH_INVALID = "HASH_INVALID"
    DATA_HASH_MISMATCH = "DATA_HASH_MISMATCH"


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityConfirmations:
    """Human confirmations; ``True`` may confirm an explicitly unavailable field."""

    source_image_id: bool
    patient_id: bool
    eye_id: bool
    followup_id: bool
    intervention_pair_id: bool
    split_isolation: bool

    def identities_complete(self) -> bool:
        values = (
            self.source_image_id,
            self.patient_id,
            self.eye_id,
            self.followup_id,
            self.intervention_pair_id,
        )
        return all(type(value) is bool and value for value in values)  # noqa: E721

    def split_complete(self) -> bool:
        return type(self.split_isolation) is bool and self.split_isolation  # noqa: E721

    def to_dict(self) -> dict[str, bool]:
        return {
            "source_image_id": self.source_image_id,
            "patient_id": self.patient_id,
            "eye_id": self.eye_id,
            "followup_id": self.followup_id,
            "intervention_pair_id": self.intervention_pair_id,
            "split_isolation": self.split_isolation,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitSampleCounts:
    """Manually declared counts that must exactly match the frozen manifest."""

    train: int
    calibration: int
    test: int

    def is_positive(self) -> bool:
        values = (self.train, self.calibration, self.test)
        return all(type(value) is int and value > 0 for value in values)  # noqa: E721

    @property
    def total(self) -> int:
        return self.train + self.calibration + self.test

    def to_dict(self) -> dict[str, int]:
        return {
            DatasetSplit.TRAIN.value: self.train,
            DatasetSplit.CALIBRATION.value: self.calibration,
            DatasetSplit.TEST.value: self.test,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenDatasetCard:
    """A process-issued, immutable binding of approved Stage 2 evidence."""

    scope: DatasetCardScope
    dataset_id: str
    dataset_version: str
    manifest_sha256: str
    registry_sha256: str
    adapter_spec_sha256: str
    audit_sha256: str
    audit_schema_version: str
    license_review_sha256: str
    license_review_schema_version: str
    license_terms_sha256: str
    license_evidence_sha256: str
    reviewer_fingerprint: str
    reviewed_at: str
    label_semantics: tuple[LabelSemantic, ...]
    label_semantics_evidence_sha256: str
    confirmations: IdentityConfirmations
    identity_evidence_sha256: str
    split_counts: SplitSampleCounts
    sample_count: int
    config_sha256: str
    data_sha256: str
    model_sha256: str
    schema_version: str = DATA_CARD_SCHEMA_VERSION
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _CARD_TOKEN:
            raise TypeError("FrozenDatasetCard can only be issued by freeze_dataset_card")
        if self.schema_version != DATA_CARD_SCHEMA_VERSION:
            raise ValueError("unsupported dataset-card schema_version")
        if not isinstance(self.scope, DatasetCardScope):
            raise TypeError("scope must be DatasetCardScope")
        if self.audit_schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError("audit schema version is not supported")
        for value in (
            self.manifest_sha256,
            self.registry_sha256,
            self.adapter_spec_sha256,
            self.audit_sha256,
            self.license_review_sha256,
            self.license_terms_sha256,
            self.license_evidence_sha256,
            self.reviewer_fingerprint,
            self.label_semantics_evidence_sha256,
            self.identity_evidence_sha256,
            self.config_sha256,
            self.data_sha256,
            self.model_sha256,
        ):
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError("dataset-card digests must be lowercase SHA-256 values")
        if self.data_sha256 != self.manifest_sha256:
            raise ValueError("data_sha256 must bind the exact Stage 2 manifest")
        if not self.label_semantics or len(set(self.label_semantics)) != len(self.label_semantics):
            raise ValueError("label_semantics must be non-empty and unique")
        if not set(self.label_semantics) <= _LESION_PROPOSITION_SEMANTICS:
            raise ValueError("label_semantics must describe lesion propositions")
        if not isinstance(self.confirmations, IdentityConfirmations):
            raise TypeError("confirmations must be IdentityConfirmations")
        if not self.confirmations.identities_complete() or not self.confirmations.split_complete():
            raise ValueError("all identity and split confirmations are required")
        if (
            not isinstance(self.split_counts, SplitSampleCounts)
            or not self.split_counts.is_positive()
        ):
            raise ValueError("split counts must be positive integers")
        if type(self.sample_count) is not int or self.sample_count != self.split_counts.total:  # noqa: E721
            raise ValueError("sample_count must equal the three split counts")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
            "registry_sha256": self.registry_sha256,
            "adapter_spec_sha256": self.adapter_spec_sha256,
            "audit_sha256": self.audit_sha256,
            "audit_schema_version": self.audit_schema_version,
            "license_review_sha256": self.license_review_sha256,
            "license_review_schema_version": self.license_review_schema_version,
            "license_terms_sha256": self.license_terms_sha256,
            "license_evidence_sha256": self.license_evidence_sha256,
            "reviewer_fingerprint": self.reviewer_fingerprint,
            "reviewed_at": self.reviewed_at,
            "label_semantics": [semantic.value for semantic in self.label_semantics],
            "label_semantics_evidence_sha256": self.label_semantics_evidence_sha256,
            "identity_confirmations": self.confirmations.to_dict(),
            "identity_evidence_sha256": self.identity_evidence_sha256,
            "split_counts": self.split_counts.to_dict(),
            "sample_count": self.sample_count,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "model_sha256": self.model_sha256,
        }


_ISSUED_CARDS: dict[int, tuple[FrozenDatasetCard, str]] = {}


@dataclass(frozen=True, slots=True, kw_only=True)
class DataCardFreezeResult:
    """Non-throwing freeze result; failures never contain a partially frozen card."""

    status: DataCardFreezeStatus
    card: FrozenDatasetCard | None
    reasons: tuple[DataCardReason, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, DataCardFreezeStatus):
            raise TypeError("status must be DataCardFreezeStatus")
        if not all(isinstance(reason, DataCardReason) for reason in self.reasons):
            raise TypeError("reasons must contain DataCardReason values")
        normalized = tuple(dict.fromkeys(self.reasons))
        object.__setattr__(self, "reasons", normalized)
        if self.status is DataCardFreezeStatus.FROZEN:
            if self.card is None or normalized:
                raise ValueError("FROZEN requires one card and no failure reasons")
        elif self.card is not None or not normalized:
            raise ValueError("blocked or fail-closed results require reasons and no card")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
            "card": None if self.card is None else self.card.to_dict(),
        }


def dataset_manifest_sha256(manifest: DatasetManifestV2) -> str:
    """Return the canonical digest used by the Stage 2 audit binding."""

    if not isinstance(manifest, DatasetManifestV2):
        raise TypeError("manifest must be DatasetManifestV2")
    payload = json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def blocked_dataset_card(
    *reasons: DataCardReason,
) -> DataCardFreezeResult:
    """Create a path-free blocked result without accepting or touching a root."""

    normalized = reasons or (DataCardReason.NO_APPROVED_DATASET,)
    if not all(isinstance(reason, DataCardReason) for reason in normalized):
        normalized = (DataCardReason.INPUT_TYPE_INVALID,)
    return DataCardFreezeResult(
        status=DataCardFreezeStatus.BLOCKED,
        card=None,
        reasons=tuple(normalized),
    )


def _failed(*reasons: DataCardReason) -> DataCardFreezeResult:
    return DataCardFreezeResult(
        status=DataCardFreezeStatus.FAIL_CLOSED,
        card=None,
        reasons=reasons,
    )


def _card_fingerprint(card: FrozenDatasetCard) -> str:
    """Snapshot every card binding without retaining nested object aliases."""

    return _canonical_sha256(card.to_dict())


def frozen_data_card_validation_reasons(value: object) -> tuple[DataCardReason, ...]:
    """Revalidate issuance identity and all immutable card bindings at consumption."""

    if not isinstance(value, FrozenDatasetCard):
        return (DataCardReason.INPUT_TYPE_INVALID,)
    issued = _ISSUED_CARDS.get(id(value))
    if issued is None or issued[0] is not value:
        return (DataCardReason.CARD_ISSUANCE_INVALID,)
    try:
        if issued[1] != _card_fingerprint(value):
            return (DataCardReason.CARD_BINDING_MISMATCH,)
    except (AttributeError, TypeError, ValueError):
        return (DataCardReason.INPUT_TYPE_INVALID,)
    return ()


def _reviewer_fingerprint(reviewer_id: str) -> str:
    material = f"retinaentail-reviewer-v1:{reviewer_id}".encode()
    return hashlib.sha256(material).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _registry_snapshot(permit: DatasetAccessPermit) -> dict[str, str]:
    registry = permit.registry
    return {
        "schema_version": registry.schema_version,
        "dataset_id": registry.dataset_id,
        "dataset_version": registry.dataset_version,
        "license_status": registry.license_status,
        "license_review_status": registry.license_review_status,
        "label_verification_status": registry.label_verification_status,
    }


def _digest_is_valid(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _manifest_matches_permit(
    manifest: DatasetManifestV2,
    permit: DatasetAccessPermit,
) -> bool:
    if (
        manifest.dataset_id != permit.registry.dataset_id
        or manifest.dataset_version != permit.registry.dataset_version
        or manifest.dataset_id != permit.review.dataset_id
        or manifest.dataset_version != permit.review.dataset_version
        or manifest.dataset_id != permit.spec.dataset_id
        or len(manifest.records) != len(permit.spec.entries)
    ):
        return False
    entries = {entry.image_path.as_posix(): entry for entry in permit.spec.entries}
    if len(entries) != len(permit.spec.entries):
        return False
    identity_fields = (
        "source_image_id",
        "split",
        "patient_id",
        "eye_id",
        "followup_id",
        "intervention_pair_id",
    )
    for record in manifest.records:
        entry = entries.get(record.relative_path)
        if entry is None or any(
            getattr(record, field_name) != getattr(entry, field_name)
            for field_name in identity_fields
        ):
            return False
        expected_labels = {label.semantic: label.relative_path for label in entry.labels}
        observed_labels = {label.semantic: label for label in record.labels}
        if len(observed_labels) != len(record.labels):
            return False
        for semantic, expected_path in expected_labels.items():
            observed = observed_labels.get(semantic)
            if observed is None:
                return False
            normalized_path = None if expected_path is None else expected_path.as_posix()
            if observed.relative_path != normalized_path:
                return False
            expected_availability = (
                LabelAvailability.UNKNOWN if expected_path is None else LabelAvailability.AVAILABLE
            )
            if observed.availability is not expected_availability:
                return False
        for semantic, observed in observed_labels.items():
            if semantic not in expected_labels and not (
                observed.availability is LabelAvailability.UNKNOWN
                and observed.relative_path is None
                and observed.sha256 is None
                and observed.byte_size is None
            ):
                return False
    return True


def _labels_are_complete(
    manifest: DatasetManifestV2,
    selected_semantics: tuple[LabelSemantic, ...],
) -> bool:
    supported = DATASET_LABEL_SEMANTICS.get(manifest.dataset_id)
    selected = set(selected_semantics)
    if supported is None or not selected or not selected <= supported:
        return False
    for record in manifest.records:
        labels = {label.semantic: label for label in record.labels}
        if not selected <= set(labels):
            return False
        if any(
            labels[semantic].availability is not LabelAvailability.AVAILABLE
            for semantic in selected_semantics
        ):
            return False
    return True


def _derived_split_counts(manifest: DatasetManifestV2) -> SplitSampleCounts | None:
    counts = {split: 0 for split in DatasetSplit}
    for record in manifest.records:
        try:
            split = DatasetSplit(record.split)
        except ValueError:
            return None
        counts[split] += 1
    return SplitSampleCounts(
        train=counts[DatasetSplit.TRAIN],
        calibration=counts[DatasetSplit.CALIBRATION],
        test=counts[DatasetSplit.TEST],
    )


def _has_split_leakage(records: tuple[ManifestRecordV2, ...]) -> bool:
    assignments: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for record in records:
        assignments[("source", record.source_image_id)].add(record.split)
        if record.patient_id is not None:
            assignments[("patient", record.patient_id)].add(record.split)
        if record.eye_id is not None:
            assignments[("eye", record.patient_id or "", record.eye_id)].add(record.split)
        if record.followup_id is not None:
            assignments[
                (
                    "followup",
                    record.patient_id or "",
                    record.eye_id or "",
                    record.followup_id,
                )
            ].add(record.split)
        if record.intervention_pair_id is not None:
            assignments[("intervention", record.intervention_pair_id)].add(record.split)
    return any(len(splits) > 1 for splits in assignments.values())


def _audit_binding_is_exact(
    audit: AuditResult,
    manifest: DatasetManifestV2,
    manifest_sha256: str,
) -> bool:
    return (
        audit.dataset_id == manifest.dataset_id
        and audit.dataset_version == manifest.dataset_version
        and audit.manifest_sha256 == manifest_sha256
    )


def _audit_counters_are_exact(audit: AuditResult, manifest: DatasetManifestV2) -> bool:
    available_labels = sum(
        label.availability is LabelAvailability.AVAILABLE
        for record in manifest.records
        for label in record.labels
    )
    return (
        audit.summary.record_count == len(manifest.records)
        and audit.summary.image_files_checked == len(manifest.records)
        and audit.summary.label_files_checked == available_labels
        and audit.summary.unknown_label_count == 0
    )


def _freeze_input_types_are_valid(
    manifest: object,
    audit: object,
    permit: object,
    scope: object,
    confirmations: object,
    declared_split_counts: object,
) -> bool:
    return (
        isinstance(manifest, DatasetManifestV2)
        and isinstance(audit, AuditResult)
        and isinstance(permit, DatasetAccessPermit)
        and isinstance(scope, DatasetCardScope)
        and isinstance(confirmations, IdentityConfirmations)
        and isinstance(declared_split_counts, SplitSampleCounts)
    )


def freeze_dataset_card(
    manifest: DatasetManifestV2,
    *,
    audit: AuditResult,
    permit: DatasetAccessPermit,
    scope: DatasetCardScope,
    label_semantics: tuple[LabelSemantic, ...],
    confirmations: IdentityConfirmations,
    declared_split_counts: SplitSampleCounts,
    identity_evidence_sha256: str,
    label_semantics_evidence_sha256: str,
    config_sha256: str,
    data_sha256: str,
    model_sha256: str,
) -> DataCardFreezeResult:
    """Freeze exact in-memory Stage 2 evidence without touching its dataset root."""

    if not _freeze_input_types_are_valid(
        manifest,
        audit,
        permit,
        scope,
        confirmations,
        declared_split_counts,
    ):
        return blocked_dataset_card(DataCardReason.INPUT_TYPE_INVALID)
    if permit_validation_reasons(permit):
        return blocked_dataset_card(DataCardReason.ACCESS_PERMIT_INVALID)
    if permit.purpose is not DataUsePurpose.READ_ONLY_AUDIT:
        return blocked_dataset_card(DataCardReason.ACCESS_PURPOSE_INVALID)
    if "lesion_calibration" not in {
        approved_purpose.value for approved_purpose in permit.review.permitted_purposes
    }:
        return blocked_dataset_card(DataCardReason.CALIBRATION_PURPOSE_NOT_APPROVED)
    if not _manifest_matches_permit(manifest, permit):
        return blocked_dataset_card(DataCardReason.MANIFEST_PERMIT_MISMATCH)

    if audit.status is AuditStatus.BLOCKED:
        return blocked_dataset_card(DataCardReason.AUDIT_BLOCKED)
    if audit.status is not AuditStatus.PASS:
        return _failed(DataCardReason.AUDIT_FAILED)

    manifest_digest = dataset_manifest_sha256(manifest)
    if not _audit_binding_is_exact(audit, manifest, manifest_digest):
        return _failed(DataCardReason.AUDIT_BINDING_MISMATCH)
    if not _audit_counters_are_exact(audit, manifest):
        return _failed(DataCardReason.AUDIT_COUNTER_MISMATCH)

    if not label_semantics:
        return blocked_dataset_card(DataCardReason.LABEL_SEMANTICS_MISSING)
    if not all(isinstance(semantic, LabelSemantic) for semantic in label_semantics):
        return blocked_dataset_card(DataCardReason.LABEL_SEMANTICS_INVALID)
    selected_semantics = tuple(sorted(set(label_semantics), key=lambda item: item.value))
    if (
        len(selected_semantics) != len(label_semantics)
        or not set(selected_semantics) <= _LESION_PROPOSITION_SEMANTICS
    ):
        return blocked_dataset_card(DataCardReason.LABEL_SEMANTICS_INVALID)
    if not _labels_are_complete(manifest, selected_semantics):
        return _failed(DataCardReason.LABEL_UNKNOWN_OR_MISSING)

    if not confirmations.identities_complete():
        return blocked_dataset_card(DataCardReason.IDENTITY_CONFIRMATION_MISSING)
    if not confirmations.split_complete():
        return blocked_dataset_card(DataCardReason.SPLIT_CONFIRMATION_MISSING)

    derived_counts = _derived_split_counts(manifest)
    if derived_counts is None or not derived_counts.is_positive():
        return blocked_dataset_card(DataCardReason.SPLIT_MISSING_OR_UNDECLARED)
    if not declared_split_counts.is_positive():
        return blocked_dataset_card(DataCardReason.SPLIT_COUNT_INVALID)
    if declared_split_counts != derived_counts:
        return _failed(DataCardReason.SPLIT_COUNT_MISMATCH)
    if _has_split_leakage(manifest.records):
        return _failed(DataCardReason.SPLIT_LEAKAGE)

    if not all(
        _digest_is_valid(value)
        for value in (
            identity_evidence_sha256,
            label_semantics_evidence_sha256,
            config_sha256,
            data_sha256,
            model_sha256,
        )
    ):
        return blocked_dataset_card(DataCardReason.HASH_INVALID)
    if data_sha256 != manifest_digest:
        return _failed(DataCardReason.DATA_HASH_MISMATCH)

    card = FrozenDatasetCard(
        scope=scope,
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        manifest_sha256=manifest_digest,
        registry_sha256=_canonical_sha256(_registry_snapshot(permit)),
        adapter_spec_sha256=_canonical_sha256(permit.spec.to_dict()),
        audit_sha256=_canonical_sha256(audit.to_dict()),
        audit_schema_version=audit.schema_version,
        license_review_sha256=_canonical_sha256(permit.review.to_dict()),
        license_review_schema_version=permit.review.schema_version,
        license_terms_sha256=permit.review.terms_sha256,
        license_evidence_sha256=permit.review.evidence_sha256,
        reviewer_fingerprint=_reviewer_fingerprint(permit.review.reviewer_id),
        reviewed_at=permit.review.reviewed_at,
        label_semantics=selected_semantics,
        label_semantics_evidence_sha256=label_semantics_evidence_sha256,
        confirmations=confirmations,
        identity_evidence_sha256=identity_evidence_sha256,
        split_counts=derived_counts,
        sample_count=derived_counts.total,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        model_sha256=model_sha256,
        _token=_CARD_TOKEN,
    )
    _ISSUED_CARDS[id(card)] = (card, _card_fingerprint(card))
    return DataCardFreezeResult(
        status=DataCardFreezeStatus.FROZEN,
        card=card,
    )


__all__ = [
    "DATA_CARD_SCHEMA_VERSION",
    "DataCardFreezeResult",
    "DataCardFreezeStatus",
    "DataCardReason",
    "DatasetCardScope",
    "DatasetSplit",
    "FrozenDatasetCard",
    "IdentityConfirmations",
    "SplitSampleCounts",
    "blocked_dataset_card",
    "dataset_manifest_sha256",
    "freeze_dataset_card",
    "frozen_data_card_validation_reasons",
]
