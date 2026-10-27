from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from retinaentail.data.audit import AuditResult, AuditStatus, AuditSummary
from retinaentail.data.license_gate import DataUsePurpose
from retinaentail.data.stage2_models import (
    DatasetManifestV2,
    LabelAvailability,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)
from retinaentail.data.stage2_package import (
    STAGE2_PACKAGE_SCHEMA_VERSION,
    Stage2PackageDescriptor,
    Stage2PackageEvidence,
    Stage2PackageFile,
    Stage2PackageScope,
)
from retinaentail.pilot.identity import (
    Stage5IdentityAuditReason,
    Stage5IdentityType,
)
from retinaentail.pilot.stage7_rebuild import (
    STAGE7_REBUILD_SCHEMA_VERSION,
    Stage7DifferenceKind,
    Stage7RebuildReason,
    Stage7RebuildStatus,
    Stage7RebuildVerification,
    verify_stage7_package_rebuild,
    verify_stage7_rebuild,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _label(index: int) -> LabelRecordV2:
    return LabelRecordV2(
        semantic=LabelSemantic.MICROANEURYSM_MASK,
        availability=LabelAvailability.AVAILABLE,
        relative_path=f"labels/case-{index}.png",
        sha256=f"{index + 4:x}" * 64,
        byte_size=50 + index,
    )


def _record(index: int, split: str) -> ManifestRecordV2:
    return ManifestRecordV2(
        dataset_id="idrid",
        relative_path=f"images/case-{index}.png",
        sha256=f"{index + 1:x}" * 64,
        byte_size=100 + index,
        width=32,
        height=24,
        channels=3,
        image_format="PNG",
        source_image_id=f"source-{index}",
        patient_id=f"patient-{index}",
        eye_id=f"eye-{index}",
        followup_id=f"followup-{index}",
        intervention_pair_id=f"pair-{index}",
        split=split,
        labels=(_label(index),),
    )


def _manifest() -> DatasetManifestV2:
    return DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="stage7-fixture-v1",
        records=(
            _record(0, "train"),
            _record(1, "calibration"),
            _record(2, "test"),
        ),
    )


def _audit(manifest: DatasetManifestV2) -> AuditResult:
    available = sum(
        label.availability is LabelAvailability.AVAILABLE
        for record in manifest.records
        for label in record.labels
    )
    unknown = sum(
        label.availability is LabelAvailability.UNKNOWN
        for record in manifest.records
        for label in record.labels
    )
    masks = sum(
        label.availability is LabelAvailability.AVAILABLE
        and label.semantic is LabelSemantic.MICROANEURYSM_MASK
        for record in manifest.records
        for label in record.labels
    )
    return AuditResult(
        status=AuditStatus.PASS,
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        manifest_sha256=_canonical_sha256(manifest.to_dict()),
        summary=AuditSummary(
            record_count=len(manifest.records),
            image_files_checked=len(manifest.records),
            label_files_checked=available,
            mask_files_decoded=masks,
            unknown_label_count=unknown,
            issue_counts=(),
        ),
    )


def _clone(manifest: DatasetManifestV2) -> DatasetManifestV2:
    return DatasetManifestV2.from_dict(manifest.to_dict())


def _package_evidence() -> Stage2PackageEvidence:
    manifest = _manifest()
    audit = _audit(manifest)
    files = (
        Stage2PackageFile(relative_path="audit.json", sha256="a" * 64, byte_size=11),
        Stage2PackageFile(relative_path="manifest-v2.json", sha256="b" * 64, byte_size=22),
    )
    file_manifest_sha256 = _canonical_sha256(
        {
            "schema_version": STAGE2_PACKAGE_SCHEMA_VERSION,
            "files": [item.to_dict() for item in files],
        }
    )
    descriptor_payload = {
        "schema_version": STAGE2_PACKAGE_SCHEMA_VERSION,
        "scope": Stage2PackageScope.SYNTHETIC_FIXTURE.value,
        "purpose": DataUsePurpose.READ_ONLY_AUDIT.value,
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "record_count": len(manifest.records),
        "manifest_sha256": _canonical_sha256(manifest.to_dict()),
        "audit_sha256": _canonical_sha256(audit.to_dict()),
        "files": [item.to_dict() for item in files],
        "file_manifest_sha256": file_manifest_sha256,
        "registry_sha256": None,
        "license_review_sha256": None,
        "adapter_spec_sha256": None,
        "root_binding_sha256": None,
        "permit_binding_sha256": None,
    }
    descriptor = Stage2PackageDescriptor(
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        purpose=DataUsePurpose.READ_ONLY_AUDIT,
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        record_count=len(manifest.records),
        manifest_sha256=_canonical_sha256(manifest.to_dict()),
        audit_sha256=_canonical_sha256(audit.to_dict()),
        files=files,
        file_manifest_sha256=file_manifest_sha256,
        registry_sha256=None,
        license_review_sha256=None,
        adapter_spec_sha256=None,
        root_binding_sha256=None,
        permit_binding_sha256=None,
        package_sha256=_canonical_sha256(descriptor_payload),
    )
    return Stage2PackageEvidence(manifest=manifest, audit=audit, descriptor=descriptor)


def _verify(
    frozen: DatasetManifestV2,
    rebuilt: DatasetManifestV2,
    *,
    frozen_audit: AuditResult | None = None,
    rebuilt_audit: AuditResult | None = None,
) -> Stage7RebuildVerification:
    return verify_stage7_rebuild(
        frozen_manifest=frozen,
        frozen_audit=_audit(frozen) if frozen_audit is None else frozen_audit,
        rebuilt_manifest=rebuilt,
        rebuilt_audit=_audit(rebuilt) if rebuilt_audit is None else rebuilt_audit,
    )


def test_exact_rebuild_passes_and_produces_stable_redacted_evidence() -> None:
    frozen = _manifest()
    rebuilt = _clone(frozen)

    result = _verify(frozen, rebuilt)

    assert result.status is Stage7RebuildStatus.PASS
    assert result.passed
    assert result.schema_version == STAGE7_REBUILD_SCHEMA_VERSION
    assert result.manifests_identical
    assert result.audits_valid and result.audits_identical
    assert result.identity_isolation_proven
    assert result.reasons == ()
    assert result.differences == ()
    assert result.frozen_manifest_sha256 == result.rebuilt_manifest_sha256
    assert result.frozen_audit_sha256 == result.rebuilt_audit_sha256
    assert result.verification_sha256 == _canonical_sha256(result._evidence_payload())
    assert json.loads(result.to_json()) == result.to_dict()
    assert "patient-0" not in result.to_json()
    with pytest.raises(FrozenInstanceError):
        result.status = Stage7RebuildStatus.FAIL_CLOSED  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("relative_path", "images/renamed.png"),
        ("sha256", "d" * 64),
        ("byte_size", 999),
        ("width", 31),
        ("height", 23),
        ("channels", 1),
        ("image_format", "JPEG"),
        ("source_image_id", "different-source"),
        ("patient_id", "different-patient"),
        ("eye_id", "different-eye"),
        ("followup_id", "different-followup"),
        ("intervention_pair_id", "different-pair"),
        ("split", "calibration"),
    ],
)
def test_every_image_or_identity_field_drift_fails_closed(
    field: str,
    replacement: object,
) -> None:
    frozen = _manifest()
    records = list(_clone(frozen).records)
    records[0] = replace(
        records[0],
        **cast(dict[str, Any], {field: replacement}),
    )
    rebuilt = replace(frozen, records=tuple(records))

    result = _verify(frozen, rebuilt)

    assert result.status is Stage7RebuildStatus.FAIL_CLOSED
    assert Stage7RebuildReason.RECORD_FIELD_MISMATCH in result.reasons
    assert Stage7RebuildReason.MANIFEST_HASH_MISMATCH in result.reasons
    assert any(
        difference.kind is Stage7DifferenceKind.RECORD and difference.field == field
        for difference in result.differences
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("semantic", LabelSemantic.HEMORRHAGE_MASK),
        ("relative_path", "labels/renamed.png"),
        ("sha256", "e" * 64),
        ("byte_size", 999),
    ],
)
def test_every_label_field_drift_fails_closed(field: str, replacement: object) -> None:
    frozen = _manifest()
    records = list(_clone(frozen).records)
    labels = list(records[0].labels)
    labels[0] = replace(
        labels[0],
        **cast(dict[str, Any], {field: replacement}),
    )
    records[0] = replace(records[0], labels=tuple(labels))
    rebuilt = replace(frozen, records=tuple(records))

    result = _verify(frozen, rebuilt)

    assert Stage7RebuildReason.LABEL_FIELD_MISMATCH in result.reasons
    assert not result.passed
    assert any(
        difference.kind is Stage7DifferenceKind.LABEL and difference.field == field
        for difference in result.differences
    )


def test_dataset_id_and_exact_version_are_cross_bound() -> None:
    frozen = _manifest()
    wrong_version = replace(frozen, dataset_version="stage7-fixture-v2")
    wrong_dataset = DatasetManifestV2(
        dataset_id="ddr",
        dataset_version=frozen.dataset_version,
        records=tuple(replace(record, dataset_id="ddr") for record in frozen.records),
    )

    version_result = _verify(frozen, wrong_version)
    dataset_result = _verify(frozen, wrong_dataset)

    assert Stage7RebuildReason.DATASET_VERSION_MISMATCH in version_result.reasons
    assert Stage7RebuildReason.DATASET_ID_MISMATCH in dataset_result.reasons
    assert not version_result.passed and not dataset_result.passed


def test_record_addition_and_removal_are_detected() -> None:
    frozen = _manifest()
    added_record = replace(
        frozen.records[2],
        relative_path="images/case-3.png",
        sha256="9" * 64,
        source_image_id="source-3",
        patient_id="patient-3",
        eye_id="eye-3",
        followup_id="followup-3",
        intervention_pair_id="pair-3",
        labels=(replace(_label(2), relative_path="labels/case-3.png", sha256="8" * 64),),
    )
    added = replace(frozen, records=(*frozen.records, added_record))
    removed = replace(frozen, records=frozen.records[:2])

    for rebuilt in (added, removed):
        result = _verify(frozen, rebuilt)
        assert Stage7RebuildReason.RECORD_COUNT_MISMATCH in result.reasons
        assert not result.manifests_identical
        assert not result.passed


def test_record_reordering_is_not_silently_normalized() -> None:
    frozen = _manifest()
    rebuilt = replace(
        frozen,
        records=(frozen.records[1], frozen.records[0], frozen.records[2]),
    )

    result = _verify(frozen, rebuilt)

    assert Stage7RebuildReason.RECORD_ORDER_MISMATCH in result.reasons
    assert Stage7RebuildReason.RECORD_FIELD_MISMATCH not in result.reasons
    assert not result.passed


def test_malformed_duplicate_record_is_fail_closed() -> None:
    frozen = _manifest()
    rebuilt = _clone(frozen)
    object.__setattr__(
        rebuilt,
        "records",
        (rebuilt.records[0], rebuilt.records[0], *rebuilt.records[1:]),
    )

    result = _verify(frozen, rebuilt, rebuilt_audit=_audit(frozen))

    assert Stage7RebuildReason.DUPLICATE_RECORD in result.reasons
    assert Stage7RebuildReason.INPUT_INVALID in result.reasons
    assert not result.passed


@pytest.mark.parametrize("identity_type", list(Stage5IdentityType))
def test_all_five_identity_leakage_types_fail_closed(
    identity_type: Stage5IdentityType,
) -> None:
    frozen = _manifest()
    records = list(_clone(frozen).records)
    train = records[0]
    changes: dict[str, Any]
    if identity_type is Stage5IdentityType.SOURCE_IMAGE_ID:
        changes = {"source_image_id": train.source_image_id}
    elif identity_type is Stage5IdentityType.PATIENT_ID:
        changes = {"patient_id": train.patient_id}
    elif identity_type is Stage5IdentityType.EYE_ID:
        changes = {"patient_id": train.patient_id, "eye_id": train.eye_id}
    elif identity_type is Stage5IdentityType.FOLLOWUP_ID:
        changes = {
            "patient_id": train.patient_id,
            "eye_id": train.eye_id,
            "followup_id": train.followup_id,
        }
    else:
        changes = {"intervention_pair_id": train.intervention_pair_id}
    records[2] = replace(records[2], **changes)
    rebuilt = replace(frozen, records=tuple(records))

    result = _verify(frozen, rebuilt)

    assert Stage7RebuildReason.FIVE_IDENTITY_ISOLATION_FAILED in result.reasons
    assert result.rebuilt_identity_audit is not None
    assert any(
        issue.reason is Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP
        and issue.identity_type is identity_type
        for issue in result.rebuilt_identity_audit.issues
    )
    assert not result.passed


def test_missing_identity_is_unknown_and_never_inferred() -> None:
    frozen = _manifest()
    records = list(_clone(frozen).records)
    records[1] = replace(records[1], patient_id=None)
    rebuilt = replace(frozen, records=tuple(records))

    result = _verify(frozen, rebuilt)

    assert Stage7RebuildReason.FIVE_IDENTITY_ISOLATION_FAILED in result.reasons
    assert result.rebuilt_identity_audit is not None
    assert Stage5IdentityAuditReason.MISSING_PATIENT_ID in result.rebuilt_identity_audit.reasons


def test_unknown_label_availability_blocks_even_an_exact_rebuild() -> None:
    manifest = _manifest()
    records = list(manifest.records)
    records[0] = replace(
        records[0],
        labels=(
            LabelRecordV2(
                semantic=LabelSemantic.MICROANEURYSM_MASK,
                availability=LabelAvailability.UNKNOWN,
            ),
        ),
    )
    frozen = replace(manifest, records=tuple(records))
    rebuilt = _clone(frozen)

    result = _verify(frozen, rebuilt)

    assert Stage7RebuildReason.UNKNOWN_LABEL_STATE in result.reasons
    assert result.frozen_unknown_label_count == result.rebuilt_unknown_label_count == 1
    assert result.manifests_identical and result.audits_valid
    assert not result.passed


def test_undeclared_label_semantic_corruption_is_fail_closed() -> None:
    frozen = _manifest()
    rebuilt = _clone(frozen)
    object.__setattr__(rebuilt.records[0].labels[0], "semantic", "UNDECLARED")

    result = _verify(frozen, rebuilt, rebuilt_audit=_audit(frozen))

    assert Stage7RebuildReason.INPUT_INVALID in result.reasons
    assert Stage7RebuildReason.REBUILT_AUDIT_INVALID in result.reasons
    assert not result.passed


def test_audit_must_be_pass_and_cross_bound_to_its_manifest() -> None:
    frozen = _manifest()
    rebuilt = _clone(frozen)
    detached = replace(_audit(rebuilt), manifest_sha256="f" * 64)

    result = _verify(frozen, rebuilt, rebuilt_audit=detached)

    assert Stage7RebuildReason.REBUILT_AUDIT_INVALID in result.reasons
    assert Stage7RebuildReason.AUDIT_EVIDENCE_MISMATCH in result.reasons
    assert not result.audits_valid
    assert not result.passed


def test_audit_record_count_must_cover_the_complete_manifest() -> None:
    frozen = _manifest()
    rebuilt = _clone(frozen)
    audit = _audit(rebuilt)
    detached_summary = replace(
        audit.summary,
        record_count=4,
        image_files_checked=4,
    )
    detached = replace(audit, summary=detached_summary)

    result = _verify(frozen, rebuilt, rebuilt_audit=detached)

    assert Stage7RebuildReason.REBUILT_AUDIT_INVALID in result.reasons
    assert not result.audits_valid
    assert not result.passed


def test_nonfinite_or_wrong_input_is_reduced_to_controlled_failure() -> None:
    frozen = _manifest()
    frozen_audit = _audit(frozen)
    rebuilt_audit = _audit(frozen)
    object.__setattr__(rebuilt_audit.summary, "record_count", float("nan"))

    nonfinite = verify_stage7_rebuild(
        frozen_manifest=frozen,
        frozen_audit=frozen_audit,
        rebuilt_manifest=_clone(frozen),
        rebuilt_audit=rebuilt_audit,
    )
    wrong_type = verify_stage7_rebuild(
        frozen_manifest=None,
        frozen_audit=None,
        rebuilt_manifest=None,
        rebuilt_audit=None,
    )

    assert Stage7RebuildReason.NONFINITE_VALUE in nonfinite.reasons
    assert Stage7RebuildReason.INPUT_INVALID in wrong_type.reasons
    assert nonfinite.status is wrong_type.status is Stage7RebuildStatus.FAIL_CLOSED


def test_stage2_package_evidence_wrapper_accepts_only_the_typed_replay_snapshot() -> None:
    evidence = _package_evidence()
    rebuilt = _clone(evidence.manifest)
    corrupted_typed = _package_evidence()
    object.__setattr__(corrupted_typed, "descriptor", object())

    accepted = verify_stage7_package_rebuild(
        frozen_evidence=evidence,
        rebuilt_manifest=rebuilt,
        rebuilt_audit=_audit(rebuilt),
    )
    detached = verify_stage7_package_rebuild(
        frozen_evidence={
            "manifest": evidence.manifest.to_dict(),
            "audit": evidence.audit.to_dict(),
            "descriptor": evidence.descriptor.to_dict(),
        },
        rebuilt_manifest=rebuilt,
        rebuilt_audit=_audit(rebuilt),
    )
    wrong_type = verify_stage7_package_rebuild(
        frozen_evidence=evidence.descriptor,
        rebuilt_manifest=rebuilt,
        rebuilt_audit=_audit(rebuilt),
    )
    corrupted = verify_stage7_package_rebuild(
        frozen_evidence=corrupted_typed,
        rebuilt_manifest=rebuilt,
        rebuilt_audit=_audit(rebuilt),
    )

    assert accepted.passed
    for result in (detached, wrong_type, corrupted):
        assert Stage7RebuildReason.FROZEN_PACKAGE_EVIDENCE_INVALID in result.reasons
        assert not result.passed
