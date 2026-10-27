from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from retinaentail.data.stage2_models import (
    LabelAvailability,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)
from retinaentail.pilot.identity import (
    STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION,
    Stage5IdentityAudit,
    Stage5IdentityAuditReason,
    Stage5IdentityAuditStatus,
    Stage5IdentityContractError,
    Stage5IdentityRecord,
    Stage5IdentityType,
    Stage5Split,
    audit_five_identity_split_isolation,
    verify_five_identity_audit,
)

_HASH = "a" * 64


def _record(
    index: int,
    split: str,
    *,
    dataset_id: str = "idrid",
    relative_path: str | None = None,
    source_image_id: str | None = None,
    patient_id: str | None = None,
    eye_id: str | None = None,
    followup_id: str | None = None,
    intervention_pair_id: str | None = None,
) -> Stage5IdentityRecord:
    return Stage5IdentityRecord(
        dataset_id=dataset_id,
        relative_path=relative_path or f"images/image-{index}.png",
        split=split,
        source_image_id=source_image_id or f"source-{index}",
        patient_id=patient_id if patient_id is not None else f"patient-{index}",
        eye_id=eye_id if eye_id is not None else f"eye-{index}",
        followup_id=followup_id if followup_id is not None else f"visit-{index}",
        intervention_pair_id=(
            intervention_pair_id if intervention_pair_id is not None else f"pair-{index}"
        ),
    )


def _complete_records() -> tuple[Stage5IdentityRecord, ...]:
    return (
        _record(0, "train"),
        _record(1, "calibration"),
        _record(2, "test"),
    )


def _identity_issues(audit: Stage5IdentityAudit) -> set[Stage5IdentityType]:
    return {
        issue.identity_type
        for issue in audit.issues
        if issue.reason is Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP
        and issue.identity_type is not None
    }


def test_complete_isolated_records_pass_and_round_trip_exactly() -> None:
    records = _complete_records()
    audit = audit_five_identity_split_isolation(records)

    assert audit.status is Stage5IdentityAuditStatus.PASS
    assert audit.isolation_proven
    assert audit.dataset_id == "idrid"
    assert audit.reasons == ()
    assert audit.record_count == audit.audited_record_count == 3
    assert audit.rejected_record_count == 0
    assert dict(audit.split_counts) == {
        Stage5Split.TRAIN: 1,
        Stage5Split.CALIBRATION: 1,
        Stage5Split.TEST: 1,
    }
    assert all(count == 0 for count in audit.unknown_counts.values())
    assert Stage5IdentityAudit.from_dict(audit.to_dict()) == audit
    assert Stage5IdentityAudit.from_json(audit.to_json()) == audit
    assert verify_five_identity_audit(audit, records)


def test_repeated_twin_identities_inside_one_split_are_not_leakage() -> None:
    records = (
        *_complete_records(),
        _record(
            3,
            "train",
            source_image_id="source-0",
            patient_id="patient-0",
            eye_id="eye-0",
            followup_id="visit-0",
            intervention_pair_id="pair-0",
        ),
    )

    audit = audit_five_identity_split_isolation(records)

    assert audit.status is Stage5IdentityAuditStatus.PASS
    assert Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP not in audit.reasons


def test_all_five_identity_namespaces_detect_cross_split_overlap() -> None:
    train = _record(0, "train")
    calibration = _record(1, "calibration")
    test = _record(
        2,
        "test",
        source_image_id=train.source_image_id,
        patient_id=train.patient_id,
        eye_id=train.eye_id,
        followup_id=train.followup_id,
        intervention_pair_id=train.intervention_pair_id,
    )

    audit = audit_five_identity_split_isolation((train, calibration, test))

    assert audit.status is Stage5IdentityAuditStatus.FAIL_CLOSED
    assert not audit.isolation_proven
    assert _identity_issues(audit) == set(Stage5IdentityType)
    overlap_issues = [
        issue
        for issue in audit.issues
        if issue.reason is Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP
    ]
    assert all(issue.splits == (Stage5Split.TRAIN, Stage5Split.TEST) for issue in overlap_issues)
    assert all(issue.record_indices == (0, 2) for issue in overlap_issues)
    serialized = audit.to_json()
    for raw_identity in ("source-0", "patient-0", "eye-0", "visit-0", "pair-0"):
        assert raw_identity not in serialized


def test_eye_and_followup_are_composite_scoped_to_avoid_false_collisions() -> None:
    records = (
        _record(0, "train", eye_id="left", followup_id="baseline"),
        _record(1, "calibration", eye_id="left", followup_id="baseline"),
        _record(2, "test", eye_id="left", followup_id="baseline"),
    )

    audit = audit_five_identity_split_isolation(records)

    assert audit.status is Stage5IdentityAuditStatus.PASS
    assert _identity_issues(audit) == set()


def test_missing_nullable_identities_are_unknown_and_fail_closed() -> None:
    train, calibration, test = _complete_records()
    missing = Stage5IdentityRecord(
        dataset_id=test.dataset_id,
        relative_path=test.relative_path,
        split=test.split,
        source_image_id=test.source_image_id,
        patient_id=None,
        eye_id=None,
        followup_id=None,
        intervention_pair_id=None,
    )

    audit = audit_five_identity_split_isolation((train, calibration, missing))

    assert audit.status is Stage5IdentityAuditStatus.FAIL_CLOSED
    assert dict(audit.unknown_counts) == {
        Stage5IdentityType.SOURCE_IMAGE_ID: 0,
        Stage5IdentityType.PATIENT_ID: 1,
        Stage5IdentityType.EYE_ID: 1,
        Stage5IdentityType.FOLLOWUP_ID: 1,
        Stage5IdentityType.INTERVENTION_PAIR_ID: 1,
    }
    assert {
        Stage5IdentityAuditReason.MISSING_PATIENT_ID,
        Stage5IdentityAuditReason.MISSING_EYE_ID,
        Stage5IdentityAuditReason.MISSING_FOLLOWUP_ID,
        Stage5IdentityAuditReason.MISSING_INTERVENTION_PAIR_ID,
    } <= set(audit.reasons)
    assert Stage5IdentityAuditReason.CROSS_SPLIT_IDENTITY_OVERLAP not in audit.reasons


def test_known_child_without_parent_is_a_scope_error_not_an_inferred_identity() -> None:
    train, calibration, test = _complete_records()
    incomplete = Stage5IdentityRecord(
        dataset_id=test.dataset_id,
        relative_path=test.relative_path,
        split=test.split,
        source_image_id=test.source_image_id,
        patient_id=None,
        eye_id="left",
        followup_id="baseline",
        intervention_pair_id=test.intervention_pair_id,
    )

    audit = audit_five_identity_split_isolation((train, calibration, incomplete))

    assert Stage5IdentityAuditReason.MISSING_PATIENT_ID in audit.reasons
    assert Stage5IdentityAuditReason.EYE_SCOPE_INCOMPLETE in audit.reasons
    assert Stage5IdentityAuditReason.FOLLOWUP_SCOPE_INCOMPLETE in audit.reasons
    assert audit.unknown_counts[Stage5IdentityType.EYE_ID] == 0
    assert audit.unknown_counts[Stage5IdentityType.FOLLOWUP_ID] == 0


def test_invalid_split_is_rejected_and_cannot_supply_required_test_split() -> None:
    train, calibration, test = _complete_records()
    invalid = replace(test, split="holdout")

    audit = audit_five_identity_split_isolation((train, calibration, invalid))

    assert audit.status is Stage5IdentityAuditStatus.FAIL_CLOSED
    assert audit.audited_record_count == 2
    assert audit.rejected_record_count == 1
    assert Stage5IdentityAuditReason.INVALID_SPLIT in audit.reasons
    empty_test = [
        issue
        for issue in audit.issues
        if issue.reason is Stage5IdentityAuditReason.REQUIRED_SPLIT_EMPTY
    ]
    assert len(empty_test) == 1
    assert empty_test[0].splits == (Stage5Split.TEST,)


def test_duplicate_relative_path_and_multiple_datasets_fail_closed() -> None:
    train, calibration, test = _complete_records()
    duplicate = replace(test, relative_path=train.relative_path)
    duplicate_audit = audit_five_identity_split_isolation((train, calibration, duplicate))

    assert Stage5IdentityAuditReason.DUPLICATE_RELATIVE_PATH in duplicate_audit.reasons
    duplicate_issue = next(
        issue
        for issue in duplicate_audit.issues
        if issue.reason is Stage5IdentityAuditReason.DUPLICATE_RELATIVE_PATH
    )
    assert duplicate_issue.record_indices == (0, 2)
    assert duplicate_issue.subject_fingerprint is not None

    mixed = audit_five_identity_split_isolation(
        (train, calibration, replace(test, dataset_id="ddr"))
    )
    assert mixed.dataset_id is None
    assert Stage5IdentityAuditReason.MULTIPLE_DATASETS in mixed.reasons


def test_empty_and_invalid_records_fail_closed_without_throwing() -> None:
    empty = audit_five_identity_split_isolation(())
    assert empty.status is Stage5IdentityAuditStatus.FAIL_CLOSED
    assert Stage5IdentityAuditReason.EMPTY_RECORD_SET in empty.reasons
    assert {
        issue.splits[0]
        for issue in empty.issues
        if issue.reason is Stage5IdentityAuditReason.REQUIRED_SPLIT_EMPTY
    } == set(Stage5Split)

    invalid = audit_five_identity_split_isolation((*_complete_records(), object()))
    assert invalid.rejected_record_count == 1
    assert Stage5IdentityAuditReason.INVALID_RECORD_TYPE in invalid.reasons


def test_manifest_record_v2_is_accepted_without_reading_its_path() -> None:
    train, calibration, _ = _complete_records()
    manifest_record = ManifestRecordV2(
        dataset_id="idrid",
        relative_path="does/not/need/to/exist.png",
        sha256=_HASH,
        byte_size=1,
        width=1,
        height=1,
        channels=3,
        image_format="PNG",
        source_image_id="manifest-source",
        split="test",
        labels=(
            LabelRecordV2(
                semantic=LabelSemantic.DR_GRADE,
                availability=LabelAvailability.UNKNOWN,
            ),
        ),
        patient_id="manifest-patient",
        eye_id="right",
        followup_id="baseline",
        intervention_pair_id="manifest-pair",
    )

    audit = audit_five_identity_split_isolation((train, calibration, manifest_record))

    assert audit.status is Stage5IdentityAuditStatus.PASS


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value.pop("schema_version"),
        lambda value: value.update({"status": "FAIL_CLOSED"}),
        lambda value: value.update({"record_count": True}),
        lambda value: value["split_counts"].pop("test"),
        lambda value: value["unknown_counts"].update({"patient_id": 1}),
    ],
)
def test_parser_rejects_unknown_missing_or_inconsistent_fields(mutation: object) -> None:
    payload = audit_five_identity_split_isolation(_complete_records()).to_dict()
    assert callable(mutation)
    mutation(payload)

    with pytest.raises(Stage5IdentityContractError):
        Stage5IdentityAudit.from_dict(payload)


def test_json_parser_rejects_nonstandard_constants_and_verifier_detects_change() -> None:
    records = _complete_records()
    audit = audit_five_identity_split_isolation(records)

    with pytest.raises(Stage5IdentityContractError, match="non-standard JSON constant"):
        Stage5IdentityAudit.from_json('{"schema_version":NaN}')
    changed = (*records[:-1], replace(records[-1], source_image_id="changed-source"))
    assert not verify_five_identity_audit(audit, changed)
    assert not verify_five_identity_audit(object(), records)


def test_bypass_mutated_audit_and_record_cannot_be_trusted() -> None:
    records = _complete_records()
    audit = audit_five_identity_split_isolation(records)
    object.__setattr__(audit, "isolation_proven", False)

    assert not verify_five_identity_audit(audit, records)
    with pytest.raises(Stage5IdentityContractError, match="isolation_proven"):
        Stage5IdentityAudit.from_json(audit.to_json())

    mutated_record = records[0]
    object.__setattr__(mutated_record, "patient_id", " ")
    rejected = audit_five_identity_split_isolation(records)
    assert Stage5IdentityAuditReason.INVALID_RECORD_CONTRACT in rejected.reasons
    assert rejected.status is Stage5IdentityAuditStatus.FAIL_CLOSED


def test_schema_declares_exact_version_and_closed_top_level() -> None:
    schema_path = (
        Path(__file__).parents[2]
        / "data_registry"
        / "schemas"
        / "stage5-identity-audit-v5.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == (STAGE5_IDENTITY_AUDIT_SCHEMA_VERSION)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(audit_five_identity_split_isolation(()).to_dict())
