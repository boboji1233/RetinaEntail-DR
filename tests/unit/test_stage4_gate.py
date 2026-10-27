from __future__ import annotations

from pathlib import Path

import pytest

from retinaentail.evaluation.gate import (
    REQUIRED_STAGE4_REVIEW_PURPOSES,
    Stage4LockedEvidence,
    Stage4PreflightReason,
    Stage4PreflightReport,
    Stage4PreflightStatus,
    Stage4ReviewPurpose,
    assess_stage4_preflight,
)


def _locked_evidence(
    *,
    dataset_id: str = "idrid",
    dataset_version: str = "release-v1",
) -> Stage4LockedEvidence:
    return Stage4LockedEvidence(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        license_review_sha256="1" * 64,
        adapter_spec_sha256="2" * 64,
        stage2_audit_sha256="3" * 64,
        frozen_data_card_sha256="4" * 64,
        stage3_calibration_package_sha256="5" * 64,
        manual_confirmation_sha256="6" * 64,
        calibration_artifact_sha256="7" * 64,
        config_sha256="8" * 64,
        data_sha256="9" * 64,
        model_sha256="a" * 64,
        alpha=0.05,
        positive_threshold=0.125,
        negative_threshold=0.25,
        positive_calibration_sample_count=20,
        negative_calibration_sample_count=22,
        total_calibration_sample_count=42,
    )


def _assess_complete(
    *,
    purposes: tuple[Stage4ReviewPurpose, ...] = REQUIRED_STAGE4_REVIEW_PURPOSES,
    evidence: Stage4LockedEvidence | None = None,
    use_default_evidence: bool = True,
) -> Stage4PreflightReport:
    return assess_stage4_preflight(
        dataset_id="idrid",
        dataset_version="release-v1",
        approved_review_provided=True,
        approved_review_purposes=purposes,
        dataset_root_provided=True,
        adapter_spec_provided=True,
        stage2_pass_audit_provided=True,
        frozen_data_card_provided=True,
        stage3_calibration_package_provided=True,
        manual_confirmation_record_provided=True,
        locked_evidence_provided=True,
        locked_evidence=(
            _locked_evidence() if evidence is None and use_default_evidence else evidence
        ),
    )


def test_missing_inputs_block_without_real_data_activity_and_round_trip() -> None:
    report = assess_stage4_preflight(
        dataset_id=None,
        dataset_version=None,
        approved_review_provided=False,
        approved_review_purposes=(),
        dataset_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_audit_provided=False,
        frozen_data_card_provided=False,
        stage3_calibration_package_provided=False,
        manual_confirmation_record_provided=False,
        locked_evidence_provided=False,
        locked_evidence=None,
    )

    assert report.status is Stage4PreflightStatus.BLOCKED
    assert set(report.reasons) == set(Stage4PreflightReason) - {
        Stage4PreflightReason.LOCKED_EVIDENCE_DATASET_MISMATCH
    }
    payload = report.to_dict()
    assert payload["authorizes_data_access"] is False
    assert payload["prerequisites_verified"] is False
    assert payload["real_data_read"] is False
    assert payload["sample_count"] == 0
    assert payload["test_tuning_prohibited"] is True
    assert Stage4PreflightReport.from_dict(payload) == report


def test_complete_presence_only_allows_prerequisite_verification_without_root_io(
    tmp_path: Path,
) -> None:
    # No path is passed to the API.  The marker merely proves that a truthy
    # ``dataset_root_provided`` assertion cannot create or probe a filesystem entry.
    untouched_root = tmp_path / "must-not-be-probed"

    report = _assess_complete()

    assert report.status is Stage4PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
    assert report.reasons == ()
    assert report.authorizes_data_access is False
    assert report.to_dict()["prerequisites_verified"] is False
    assert report.to_dict()["real_data_read"] is False
    assert report.to_dict()["sample_count"] == 0
    assert not untouched_root.exists()
    assert Stage4PreflightReport.from_dict(report.to_dict()) == report


def test_provided_lock_file_can_be_ready_without_opening_or_parsing_it() -> None:
    report = _assess_complete(use_default_evidence=False)

    assert report.status is Stage4PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
    assert report.locked_evidence_provided is True
    assert report.locked_evidence is None
    assert report.to_dict()["locked_evidence_binding_sha256"] is None
    assert report.authorizes_data_access is False
    assert Stage4PreflightReport.from_dict(report.to_dict()) == report


def test_missing_lock_file_blocks_even_when_memory_summary_is_available() -> None:
    report = Stage4PreflightReport(
        status=Stage4PreflightStatus.BLOCKED,
        dataset_id="idrid",
        dataset_version="release-v1",
        approved_review_provided=True,
        approved_review_purposes=REQUIRED_STAGE4_REVIEW_PURPOSES,
        dataset_root_provided=True,
        adapter_spec_provided=True,
        stage2_pass_audit_provided=True,
        frozen_data_card_provided=True,
        stage3_calibration_package_provided=True,
        manual_confirmation_record_provided=True,
        locked_evidence_provided=False,
        locked_evidence=_locked_evidence(),
        reasons=(Stage4PreflightReason.LOCKED_EVIDENCE_NOT_PROVIDED,),
    )

    assert report.status is Stage4PreflightStatus.BLOCKED
    assert report.reasons == (Stage4PreflightReason.LOCKED_EVIDENCE_NOT_PROVIDED,)


def test_same_review_must_include_all_three_stage4_purposes() -> None:
    incomplete = (
        Stage4ReviewPurpose.READ_ONLY_AUDIT,
        Stage4ReviewPurpose.LESION_CALIBRATION,
    )

    report = _assess_complete(purposes=incomplete)

    assert report.status is Stage4PreflightStatus.BLOCKED
    assert report.reasons == (Stage4PreflightReason.LOCKED_TEST_EVALUATION_PURPOSE_NOT_APPROVED,)


@pytest.mark.parametrize("missing", REQUIRED_STAGE4_REVIEW_PURPOSES)
def test_each_required_review_purpose_is_independently_gated(
    missing: Stage4ReviewPurpose,
) -> None:
    purposes = tuple(item for item in REQUIRED_STAGE4_REVIEW_PURPOSES if item is not missing)

    report = _assess_complete(purposes=purposes)

    assert report.status is Stage4PreflightStatus.BLOCKED
    assert len(report.reasons) == 1
    assert "PURPOSE_NOT_APPROVED" in report.reasons[0].value


def test_lock_summary_binds_dataset_hashes_thresholds_counts_and_round_trips() -> None:
    evidence = _locked_evidence()
    payload = evidence.to_dict()

    assert Stage4LockedEvidence.from_dict(payload) == evidence
    assert len(evidence.binding_sha256()) == 64
    assert evidence.binding_sha256() == Stage4LockedEvidence.from_dict(payload).binding_sha256()
    assert payload["test_split_used_for_calibration"] is False
    assert payload["thresholds_locked_before_test"] is True
    assert payload["test_tuning_prohibited"] is True


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("alpha", float("nan"), "alpha"),
        ("positive_threshold", float("inf"), "positive_threshold"),
        ("negative_threshold", -0.1, "negative_threshold"),
        ("config_sha256", "not-a-hash", "config_sha256"),
        ("stage2_audit_status", "FAIL", "must be PASS"),
        ("frozen_data_card_scope", "SYNTHETIC_ONLY", "APPROVED_REAL_DATA"),
        ("stage3_calibration_status", "FAIL_CLOSED", "must be SUCCESS"),
        ("manual_confirmations_complete", False, "must be complete"),
        ("thresholds_locked_before_test", False, "locked before"),
        ("test_split_used_for_calibration", True, "cannot be used"),
        ("test_tuning_prohibited", False, "must remain prohibited"),
        ("total_calibration_sample_count", 41, "must equal"),
    ],
)
def test_invalid_or_unlocked_evidence_fails_closed_at_construction(
    field_name: str,
    bad_value: object,
    message: str,
) -> None:
    payload = _locked_evidence().to_dict()
    payload[field_name] = bad_value
    with pytest.raises((TypeError, ValueError), match=message):
        Stage4LockedEvidence.from_dict(payload)


def test_lock_summary_must_match_selected_single_dataset_and_exact_version() -> None:
    mismatched = _locked_evidence(dataset_version="release-v2")

    report = _assess_complete(evidence=mismatched)

    assert report.status is Stage4PreflightStatus.BLOCKED
    assert report.reasons == (Stage4PreflightReason.LOCKED_EVIDENCE_DATASET_MISMATCH,)


def test_serialized_binding_and_status_tampering_are_rejected() -> None:
    report = _assess_complete()
    payload = report.to_dict()
    payload["locked_evidence_binding_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding digest"):
        Stage4PreflightReport.from_dict(payload)

    payload = report.to_dict()
    payload["status"] = Stage4PreflightStatus.BLOCKED.value
    payload["reasons"] = [Stage4PreflightReason.LOCKED_EVIDENCE_NOT_PROVIDED.value]
    with pytest.raises(ValueError, match="conflict with explicit input presence"):
        Stage4PreflightReport.from_dict(payload)

    payload = report.to_dict()
    payload["locked_evidence_provided"] = 1
    with pytest.raises(ValueError, match="presence fields must be booleans"):
        Stage4PreflightReport.from_dict(payload)


def test_preflight_rejects_non_boolean_flags_duplicate_purposes_and_unsafe_version() -> None:
    with pytest.raises(TypeError, match="exact bool"):
        assess_stage4_preflight(
            dataset_id="idrid",
            dataset_version="release-v1",
            approved_review_provided=1,  # type: ignore[arg-type]
            approved_review_purposes=REQUIRED_STAGE4_REVIEW_PURPOSES,
            dataset_root_provided=True,
            adapter_spec_provided=True,
            stage2_pass_audit_provided=True,
            frozen_data_card_provided=True,
            stage3_calibration_package_provided=True,
            manual_confirmation_record_provided=True,
            locked_evidence_provided=True,
            locked_evidence=_locked_evidence(),
        )

    with pytest.raises(ValueError, match="unique"):
        _assess_complete(
            purposes=(
                *REQUIRED_STAGE4_REVIEW_PURPOSES,
                Stage4ReviewPurpose.READ_ONLY_AUDIT,
            )
        )

    with pytest.raises(ValueError, match="report-safe"):
        assess_stage4_preflight(
            dataset_id="idrid",
            dataset_version="release/v1",
            approved_review_provided=False,
            approved_review_purposes=(),
            dataset_root_provided=False,
            adapter_spec_provided=False,
            stage2_pass_audit_provided=False,
            frozen_data_card_provided=False,
            stage3_calibration_package_provided=False,
            manual_confirmation_record_provided=False,
            locked_evidence_provided=False,
            locked_evidence=None,
        )
