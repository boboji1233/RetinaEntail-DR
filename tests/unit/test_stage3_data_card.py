from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from retinaentail.calibration.data_card import (
    DataCardFreezeResult,
    DataCardFreezeStatus,
    DataCardReason,
    DatasetCardScope,
    IdentityConfirmations,
    SplitSampleCounts,
    blocked_dataset_card,
    dataset_manifest_sha256,
    freeze_dataset_card,
    frozen_data_card_validation_reasons,
)
from retinaentail.data.audit import AuditResult, AuditStatus, AuditSummary, blocked_audit_result
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseReviewDecision,
    LicenseReviewRecord,
    RegistryLicenseMetadata,
    authorize_dataset_access,
)
from retinaentail.data.stage2_models import (
    DATASET_LABEL_SEMANTICS,
    DatasetAdapterSpec,
    DatasetEntrySpec,
    DatasetManifestV2,
    LabelAvailability,
    LabelInputSpec,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)

_VERSION = "synthetic-stage3-v1"
_SELECTED = (LabelSemantic.MICROANEURYSM_MASK,)
_CONFIRMED = IdentityConfirmations(
    source_image_id=True,
    patient_id=True,
    eye_id=True,
    followup_id=True,
    intervention_pair_id=True,
    split_isolation=True,
)
_COUNTS = SplitSampleCounts(train=1, calibration=1, test=1)


def _records(
    *,
    unknown_semantic: LabelSemantic | None = None,
    leak_source: bool = False,
    test_split: str = "test",
) -> tuple[ManifestRecordV2, ...]:
    records: list[ManifestRecordV2] = []
    for index, split in enumerate(("train", "calibration", test_split)):
        labels = []
        for semantic_index, semantic in enumerate(sorted(DATASET_LABEL_SEMANTICS["idrid"])):
            if semantic is unknown_semantic:
                labels.append(
                    LabelRecordV2(
                        semantic=semantic,
                        availability=LabelAvailability.UNKNOWN,
                    )
                )
            else:
                labels.append(
                    LabelRecordV2(
                        semantic=semantic,
                        availability=LabelAvailability.AVAILABLE,
                        relative_path=f"labels/{index}-{semantic.value.casefold()}.png",
                        sha256=f"{semantic_index + 1:x}" * 64,
                        byte_size=semantic_index + 1,
                    )
                )
        source_id = "source-0" if leak_source and index == 2 else f"source-{index}"
        records.append(
            ManifestRecordV2(
                dataset_id="idrid",
                relative_path=f"images/{index}.png",
                sha256=f"{index + 9:x}" * 64,
                byte_size=100 + index,
                width=16,
                height=16,
                channels=3,
                image_format="PNG",
                source_image_id=source_id,
                patient_id=f"patient-{index}",
                eye_id="left",
                followup_id="baseline",
                intervention_pair_id=f"pair-{index}",
                split=split,
                labels=tuple(labels),
            )
        )
    return tuple(records)


def _context(
    tmp_path: Path,
    *,
    records: tuple[ManifestRecordV2, ...] | None = None,
    purpose: DataUsePurpose = DataUsePurpose.READ_ONLY_AUDIT,
    calibration_approved: bool = True,
) -> tuple[DatasetManifestV2, AuditResult, DatasetAccessPermit, Path]:
    manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version=_VERSION,
        records=_records() if records is None else records,
    )
    entries = []
    for record in manifest.records:
        labels = tuple(
            LabelInputSpec(
                semantic=label.semantic,
                relative_path=(None if label.relative_path is None else Path(label.relative_path)),
            )
            for label in record.labels
        )
        entries.append(
            DatasetEntrySpec(
                image_path=Path(record.relative_path),
                source_image_id=record.source_image_id,
                patient_id=record.patient_id,
                eye_id=record.eye_id,
                followup_id=record.followup_id,
                intervention_pair_id=record.intervention_pair_id,
                split=record.split,
                labels=labels,
            )
        )
    registry = RegistryLicenseMetadata(
        schema_version="1.0.0",
        dataset_id="idrid",
        dataset_version=_VERSION,
        license_status="verified",
        license_review_status="reviewed",
        label_verification_status="verified",
    )
    approved_purposes: tuple[DataUsePurpose, ...] = (purpose,)
    if calibration_approved and purpose is not DataUsePurpose.LESION_CALIBRATION:
        approved_purposes += (DataUsePurpose.LESION_CALIBRATION,)
    review = LicenseReviewRecord(
        dataset_id="idrid",
        dataset_version=_VERSION,
        decision=LicenseReviewDecision.APPROVED,
        permitted_purposes=approved_purposes,
        terms_sha256="a" * 64,
        evidence_sha256="b" * 64,
        reviewer_id="private-synthetic-reviewer-name",
        reviewed_at="2026-08-23T12:00:00+00:00",
    )
    untouched_root = tmp_path / "must-not-be-read"
    permit = authorize_dataset_access(
        registry,
        review=review,
        purpose=purpose,
        root=untouched_root,
        spec=DatasetAdapterSpec(dataset_id="idrid", entries=tuple(entries)),
    )
    digest = dataset_manifest_sha256(manifest)
    available_label_count = sum(
        label.availability is LabelAvailability.AVAILABLE
        for record in manifest.records
        for label in record.labels
    )
    audit = AuditResult(
        status=AuditStatus.PASS,
        dataset_id="idrid",
        dataset_version=_VERSION,
        manifest_sha256=digest,
        summary=AuditSummary(
            record_count=len(manifest.records),
            image_files_checked=len(manifest.records),
            label_files_checked=available_label_count,
            mask_files_decoded=0,
            unknown_label_count=0,
            issue_counts=(),
        ),
    )
    return manifest, audit, permit, untouched_root


def _freeze(
    manifest: DatasetManifestV2,
    audit: AuditResult,
    permit: DatasetAccessPermit,
    *,
    confirmations: IdentityConfirmations = _CONFIRMED,
    counts: SplitSampleCounts = _COUNTS,
    labels: tuple[LabelSemantic, ...] = _SELECTED,
    data_sha256: str | None = None,
) -> DataCardFreezeResult:
    return freeze_dataset_card(
        manifest,
        audit=audit,
        permit=permit,
        scope=DatasetCardScope.SYNTHETIC_ONLY,
        label_semantics=labels,
        confirmations=confirmations,
        declared_split_counts=counts,
        identity_evidence_sha256="1" * 64,
        label_semantics_evidence_sha256="2" * 64,
        config_sha256="c" * 64,
        data_sha256=dataset_manifest_sha256(manifest) if data_sha256 is None else data_sha256,
        model_sha256="d" * 64,
    )


def test_synthetic_authorized_manifest_freezes_without_root_io(tmp_path: Path) -> None:
    manifest, audit, permit, untouched_root = _context(tmp_path)

    result = _freeze(manifest, audit, permit)

    assert result.status is DataCardFreezeStatus.FROZEN
    assert result.card is not None
    assert result.card.scope is DatasetCardScope.SYNTHETIC_ONLY
    assert result.card.split_counts == _COUNTS
    assert result.card.sample_count == 3
    assert result.card.data_sha256 == result.card.manifest_sha256
    assert result.card.registry_sha256 != result.card.adapter_spec_sha256
    assert result.card.audit_sha256 != result.card.license_review_sha256
    assert result.card.license_terms_sha256 == "a" * 64
    assert result.card.license_evidence_sha256 == "b" * 64
    assert result.card.reviewer_fingerprint != permit.review.reviewer_id
    assert permit.review.reviewer_id not in str(result.to_dict())
    assert str(untouched_root) not in str(result.to_dict())
    assert frozen_data_card_validation_reasons(result.card) == ()
    assert not untouched_root.exists()


def test_copied_permit_and_card_fail_process_bound_revalidation(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)
    copied_permit = replace(permit)

    blocked = _freeze(manifest, audit, copied_permit)
    good = _freeze(manifest, audit, permit)

    assert blocked.status is DataCardFreezeStatus.BLOCKED
    assert blocked.card is None
    assert blocked.reasons == (DataCardReason.ACCESS_PERMIT_INVALID,)
    assert good.card is not None
    assert frozen_data_card_validation_reasons(replace(good.card)) == (
        DataCardReason.CARD_ISSUANCE_INVALID,
    )


def test_frozen_card_detects_nested_confirmation_mutation(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)
    result = _freeze(
        manifest,
        audit,
        permit,
        confirmations=replace(_CONFIRMED),
        counts=replace(_COUNTS),
    )
    assert result.card is not None

    object.__setattr__(result.card.confirmations, "patient_id", False)

    assert frozen_data_card_validation_reasons(result.card) == (
        DataCardReason.CARD_BINDING_MISMATCH,
    )


def test_frozen_card_detects_nested_split_count_mutation(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)
    result = _freeze(
        manifest,
        audit,
        permit,
        confirmations=replace(_CONFIRMED),
        counts=replace(_COUNTS),
    )
    assert result.card is not None

    object.__setattr__(result.card.split_counts, "test", 2)

    assert frozen_data_card_validation_reasons(result.card) == (
        DataCardReason.CARD_BINDING_MISMATCH,
    )


def test_non_audit_purpose_is_blocked(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(
        tmp_path,
        purpose=DataUsePurpose.METADATA_VALIDATION,
    )

    result = _freeze(manifest, audit, permit)

    assert result.status is DataCardFreezeStatus.BLOCKED
    assert result.reasons == (DataCardReason.ACCESS_PURPOSE_INVALID,)


def test_read_only_audit_permission_alone_cannot_authorize_calibration(tmp_path: Path) -> None:
    manifest, audit, permit, untouched_root = _context(
        tmp_path,
        calibration_approved=False,
    )

    result = _freeze(manifest, audit, permit)

    assert result.status is DataCardFreezeStatus.BLOCKED
    assert result.card is None
    assert result.reasons == (DataCardReason.CALIBRATION_PURPOSE_NOT_APPROVED,)
    assert not untouched_root.exists()


def test_audit_must_pass_and_bind_exact_manifest(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)

    failed = _freeze(manifest, replace(audit, status=AuditStatus.FAIL_CLOSED), permit)
    mismatched = _freeze(manifest, replace(audit, manifest_sha256="e" * 64), permit)
    blocked = _freeze(manifest, blocked_audit_result(("license review unavailable",)), permit)

    assert failed.status is DataCardFreezeStatus.FAIL_CLOSED
    assert failed.reasons == (DataCardReason.AUDIT_FAILED,)
    assert mismatched.status is DataCardFreezeStatus.FAIL_CLOSED
    assert mismatched.reasons == (DataCardReason.AUDIT_BINDING_MISMATCH,)
    assert blocked.status is DataCardFreezeStatus.BLOCKED
    assert blocked.reasons == (DataCardReason.AUDIT_BLOCKED,)


def test_unknown_selected_label_fails_closed_even_with_claimed_pass_audit(
    tmp_path: Path,
) -> None:
    manifest, audit, permit, _ = _context(
        tmp_path,
        records=_records(unknown_semantic=LabelSemantic.MICROANEURYSM_MASK),
    )
    claimed_pass = replace(
        audit,
        summary=replace(audit.summary, unknown_label_count=0),
    )

    result = _freeze(manifest, claimed_pass, permit)

    assert result.status is DataCardFreezeStatus.FAIL_CLOSED
    assert result.card is None
    assert result.reasons == (DataCardReason.LABEL_UNKNOWN_OR_MISSING,)


def test_label_semantics_must_be_explicit_unique_and_lesion_level(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)

    missing = _freeze(manifest, audit, permit, labels=())
    duplicate = _freeze(manifest, audit, permit, labels=(_SELECTED[0], _SELECTED[0]))
    grading = _freeze(manifest, audit, permit, labels=(LabelSemantic.DR_GRADE,))

    for result in (missing, duplicate, grading):
        assert result.status is DataCardFreezeStatus.BLOCKED
        assert result.card is None


def test_all_identity_and_split_confirmations_are_required(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)

    identity_missing = _freeze(
        manifest,
        audit,
        permit,
        confirmations=replace(_CONFIRMED, patient_id=False),
    )
    split_missing = _freeze(
        manifest,
        audit,
        permit,
        confirmations=replace(_CONFIRMED, split_isolation=False),
    )

    assert identity_missing.reasons == (DataCardReason.IDENTITY_CONFIRMATION_MISSING,)
    assert split_missing.reasons == (DataCardReason.SPLIT_CONFIRMATION_MISSING,)
    assert identity_missing.card is None
    assert split_missing.card is None


def test_three_positive_exact_split_counts_are_required(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)
    missing_manifest, missing_audit, missing_permit, _ = _context(
        tmp_path,
        records=_records(test_split="calibration"),
    )

    invalid = _freeze(
        manifest,
        audit,
        permit,
        counts=SplitSampleCounts(train=1, calibration=1, test=0),
    )
    mismatch = _freeze(
        manifest,
        audit,
        permit,
        counts=SplitSampleCounts(train=2, calibration=1, test=1),
    )
    missing = _freeze(missing_manifest, missing_audit, missing_permit)

    assert invalid.status is DataCardFreezeStatus.BLOCKED
    assert invalid.reasons == (DataCardReason.SPLIT_COUNT_INVALID,)
    assert mismatch.status is DataCardFreezeStatus.FAIL_CLOSED
    assert mismatch.reasons == (DataCardReason.SPLIT_COUNT_MISMATCH,)
    assert missing.status is DataCardFreezeStatus.BLOCKED
    assert missing.reasons == (DataCardReason.SPLIT_MISSING_OR_UNDECLARED,)


def test_protected_identity_leakage_fails_closed(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path, records=_records(leak_source=True))

    result = _freeze(manifest, audit, permit)

    assert result.status is DataCardFreezeStatus.FAIL_CLOSED
    assert result.reasons == (DataCardReason.SPLIT_LEAKAGE,)
    assert result.card is None


def test_hashes_are_required_and_data_hash_is_exact_manifest_binding(tmp_path: Path) -> None:
    manifest, audit, permit, _ = _context(tmp_path)

    malformed = freeze_dataset_card(
        manifest,
        audit=audit,
        permit=permit,
        scope=DatasetCardScope.SYNTHETIC_ONLY,
        label_semantics=_SELECTED,
        confirmations=_CONFIRMED,
        declared_split_counts=_COUNTS,
        identity_evidence_sha256="1" * 64,
        label_semantics_evidence_sha256="2" * 64,
        config_sha256="not-a-hash",
        data_sha256=dataset_manifest_sha256(manifest),
        model_sha256="d" * 64,
    )
    mismatched = _freeze(manifest, audit, permit, data_sha256="f" * 64)

    assert malformed.status is DataCardFreezeStatus.BLOCKED
    assert malformed.reasons == (DataCardReason.HASH_INVALID,)
    assert mismatched.status is DataCardFreezeStatus.FAIL_CLOSED
    assert mismatched.reasons == (DataCardReason.DATA_HASH_MISMATCH,)


def test_blocked_factory_is_path_free_and_has_no_partial_card(tmp_path: Path) -> None:
    untouched_root = tmp_path / "never-inspected"

    result = blocked_dataset_card()

    assert result.status is DataCardFreezeStatus.BLOCKED
    assert result.reasons == (DataCardReason.NO_APPROVED_DATASET,)
    assert result.card is None
    assert str(untouched_root) not in str(result.to_dict())
    assert not untouched_root.exists()
