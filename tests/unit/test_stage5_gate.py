from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseReviewRecord,
    RegistryLicenseMetadata,
    authorize_dataset_access,
)
from retinaentail.data.stage2_models import DatasetAdapterSpec
from retinaentail.evaluation.gate import Stage4LockedEvidence
from retinaentail.pilot.gate import (
    REQUIRED_STAGE5_REVIEW_PURPOSES,
    Stage5AuthorizationError,
    Stage5EvidencePaths,
    Stage5PreflightReason,
    Stage5PreflightReport,
    Stage5PreflightStatus,
    Stage5PrerequisiteReason,
    Stage5PrerequisiteReport,
    Stage5PrerequisiteStatus,
    Stage5RootBindingError,
    Stage5VerificationContext,
    Stage5VerificationOutcome,
    assess_stage5_preflight,
    authorization_validation_reasons,
    bind_authorized_root,
    bound_root_validation_reasons,
    consume_bound_stage5_root,
    require_stage5_authorization,
    verify_stage5_prerequisites,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _adapter_spec() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for index, split in enumerate(("train", "calibration", "test")):
        entries.append(
            {
                "schema_version": "2.0.0",
                "image_path": f"images/fixture-{index}.png",
                "source_image_id": f"fixture-source-{index}",
                "split": split,
                "patient_id": f"fixture-patient-{index}",
                "eye_id": f"fixture-eye-{index}",
                "followup_id": f"fixture-followup-{index}",
                "intervention_pair_id": f"fixture-pair-{index}",
                "labels": [
                    {
                        "schema_version": "2.0.0",
                        "semantic": "LESION_DETECTION",
                        "relative_path": f"labels/fixture-{index}.json",
                    }
                ],
            }
        )
    return {"schema_version": "2.0.0", "dataset_id": "idrid", "entries": entries}


def _synthetic_evidence(tmp_path: Path) -> tuple[Path, Stage5EvidencePaths]:
    """Build contract-shaped bytes in temp storage; this is not real approval evidence."""

    evidence_dir = tmp_path / "synthetic-evidence"
    evidence_dir.mkdir(parents=True)
    dataset_root = tmp_path / "must-not-be-probed-real-root"
    manifest_sha256 = "1" * 64
    identity_sha256 = "2" * 64
    label_sha256 = "3" * 64
    config_sha256 = "4" * 64
    calibration_data_sha256 = "5" * 64
    model_sha256 = "6" * 64
    package_sha256 = "7" * 64
    artifact_sha256 = "8" * 64

    review: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset_id": "idrid",
        "dataset_version": "fixture-v1",
        "decision": "approved",
        "permitted_purposes": [purpose.value for purpose in REQUIRED_STAGE5_REVIEW_PURPOSES],
        "terms_sha256": "9" * 64,
        "evidence_sha256": "a" * 64,
        "reviewer_id": "synthetic-fixture-reviewer",
        "reviewed_at": "2026-08-24T00:00:00+08:00",
    }
    spec = _adapter_spec()
    audit: dict[str, object] = {
        "schema_version": "2.0.0",
        "status": "PASS",
        "dataset_id": "idrid",
        "dataset_version": "fixture-v1",
        "manifest_sha256": manifest_sha256,
        "summary": {
            "record_count": 3,
            "image_files_checked": 3,
            "label_files_checked": 3,
            "mask_files_decoded": 0,
            "unknown_label_count": 0,
            "issue_counts": {},
        },
        "issues": [],
    }
    review_sha256 = _canonical_sha256(review)
    spec_sha256 = _canonical_sha256(spec)
    audit_sha256 = _canonical_sha256(audit)
    card: dict[str, object] = {
        "schema_version": "3.0.0",
        "scope": "APPROVED_REAL_DATA",
        "dataset_id": "idrid",
        "dataset_version": "fixture-v1",
        "manifest_sha256": manifest_sha256,
        "registry_sha256": "b" * 64,
        "adapter_spec_sha256": spec_sha256,
        "audit_sha256": audit_sha256,
        "audit_schema_version": "2.0.0",
        "license_review_sha256": review_sha256,
        "license_review_schema_version": "1.0.0",
        "license_terms_sha256": review["terms_sha256"],
        "license_evidence_sha256": review["evidence_sha256"],
        "reviewer_fingerprint": hashlib.sha256(
            b"retinaentail-reviewer-v1:synthetic-fixture-reviewer"
        ).hexdigest(),
        "reviewed_at": review["reviewed_at"],
        "label_semantics": ["LESION_DETECTION"],
        "label_semantics_evidence_sha256": label_sha256,
        "identity_confirmations": {
            "source_image_id": True,
            "patient_id": True,
            "eye_id": True,
            "followup_id": True,
            "intervention_pair_id": True,
            "split_isolation": True,
        },
        "identity_evidence_sha256": identity_sha256,
        "split_counts": {"train": 1, "calibration": 1, "test": 1},
        "sample_count": 3,
        "config_sha256": "c" * 64,
        "data_sha256": manifest_sha256,
        "model_sha256": "d" * 64,
    }
    manual: dict[str, object] = {
        "schema_version": "5.0.0",
        "record_type": "MANUAL_IDENTITY_LABEL_CONFIRMATION",
        "scope": "APPROVED_REAL_DATA",
        "dataset_id": "idrid",
        "dataset_version": "fixture-v1",
        "confirmed_by": "synthetic-fixture-reviewer",
        "confirmed_at": "2026-08-24T00:01:00+08:00",
        "source_image_id_confirmed": True,
        "patient_id_confirmed": True,
        "eye_id_confirmed": True,
        "followup_id_confirmed": True,
        "intervention_pair_id_confirmed": True,
        "label_semantics_confirmed": True,
        "split_isolation_confirmed": True,
        "identity_evidence_sha256": identity_sha256,
        "label_semantics_evidence_sha256": label_sha256,
    }
    stage3: dict[str, object] = {
        "schema_version": "5.0.0",
        "record_type": "VERIFIED_STAGE3_CALIBRATION_PACKAGE",
        "scope": "APPROVED_REAL_DATA",
        "dataset_id": "idrid",
        "dataset_version": "fixture-v1",
        "verification_status": "VERIFIED",
        "package_sha256": package_sha256,
        "package_manifest_sha256": "e" * 64,
        "package_verifier_sha256": "f" * 64,
        "package_verifier_version": "fixture-verifier-v1",
        "independent_verifier_boundary": "COMPLETE_VERIFIED_STAGE3_PACKAGE",
        "verified_at": "2026-08-24T00:02:00+08:00",
        "verified_file_count": 12,
        "verified_byte_count": 4096,
        "calibration_artifact_sha256": artifact_sha256,
        "calibration_status": "SUCCESS",
        "alpha": 0.05,
        "positive_threshold": 0.1,
        "negative_threshold": 0.2,
        "positive_sample_count": 10,
        "negative_sample_count": 11,
        "total_sample_count": 21,
        "config_hash": config_sha256,
        "data_hash": calibration_data_sha256,
        "model_hash": model_sha256,
        "positive_witness_replay_verified": True,
        "negative_witness_replay_verified": True,
        "thresholds_locked_before_test": True,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
    }
    locked = Stage4LockedEvidence(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        license_review_sha256=review_sha256,
        adapter_spec_sha256=spec_sha256,
        stage2_audit_sha256=audit_sha256,
        frozen_data_card_sha256=_canonical_sha256(card),
        stage3_calibration_package_sha256=package_sha256,
        manual_confirmation_sha256=_canonical_sha256(manual),
        calibration_artifact_sha256=artifact_sha256,
        config_sha256=config_sha256,
        data_sha256=calibration_data_sha256,
        model_sha256=model_sha256,
        alpha=0.05,
        positive_threshold=0.1,
        negative_threshold=0.2,
        positive_calibration_sample_count=10,
        negative_calibration_sample_count=11,
        total_calibration_sample_count=21,
    )
    values = {
        "license_review": review,
        "adapter_spec": spec,
        "stage2_audit": audit,
        "frozen_data_card": card,
        "manual_confirmation": manual,
        "stage3_package_summary": stage3,
        "stage4_locked_evidence": locked.to_dict(),
    }
    paths: dict[str, Path] = {}
    for name, value in values.items():
        path = evidence_dir / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    return dataset_root, Stage5EvidencePaths(**paths)


def _verify_synthetic(tmp_path: Path) -> tuple[Path, Stage5VerificationOutcome]:
    root, paths = _synthetic_evidence(tmp_path)
    return root, verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
        authorization_ttl_seconds=60,
    )


def _simulated_process_permit(root: Path, paths: Stage5EvidencePaths) -> DatasetAccessPermit:
    """Issue a process permit from temp fixture bytes; it is not real-data approval."""

    review_value = json.loads(paths.license_review.read_text(encoding="utf-8"))
    spec_value = json.loads(paths.adapter_spec.read_text(encoding="utf-8"))
    assert isinstance(review_value, dict)
    assert isinstance(spec_value, dict)
    review = LicenseReviewRecord.from_mapping(review_value)
    spec = DatasetAdapterSpec.from_dict(spec_value)
    registry = RegistryLicenseMetadata(
        schema_version="1.0.0",
        dataset_id="idrid",
        dataset_version="fixture-v1",
        license_status="verified",
        license_review_status="reviewed",
        label_verification_status="verified",
    )
    registry_snapshot = {
        "schema_version": registry.schema_version,
        "dataset_id": registry.dataset_id,
        "dataset_version": registry.dataset_version,
        "license_status": registry.license_status,
        "license_review_status": registry.license_review_status,
        "label_verification_status": registry.label_verification_status,
    }
    card = json.loads(paths.frozen_data_card.read_text(encoding="utf-8"))
    locked = json.loads(paths.stage4_locked_evidence.read_text(encoding="utf-8"))
    assert isinstance(card, dict)
    assert isinstance(locked, dict)
    card["registry_sha256"] = _canonical_sha256(registry_snapshot)
    _write_json(paths.frozen_data_card, card)
    locked["frozen_data_card_sha256"] = _canonical_sha256(card)
    _write_json(paths.stage4_locked_evidence, locked)
    return authorize_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
        root=root,
        spec=spec,
    )


def test_stage5_preflight_missing_inputs_is_zero_io_and_round_trips() -> None:
    report = assess_stage5_preflight(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id=None,
        dataset_version=None,
        approved_review_provided=False,
        approved_review_purposes=(),
        dataset_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_audit_provided=False,
        frozen_data_card_provided=False,
        stage2_access_permit_provided=False,
        manual_confirmation_provided=False,
        stage3_package_summary_provided=False,
        stage4_locked_evidence_provided=False,
    )

    assert report.status is Stage5PreflightStatus.BLOCKED
    assert Stage5PreflightReason.VALID_STAGE2_ACCESS_PERMIT_NOT_PROVIDED in report.reasons
    assert report.to_dict()["dataset_root_probed"] is False
    assert report.to_dict()["real_data_read"] is False
    assert report.to_dict()["sample_count"] == 0
    assert Stage5PreflightReport.from_dict(report.to_dict()) == report


def test_stage5_preflight_complete_presence_still_does_not_authorize() -> None:
    report = assess_stage5_preflight(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id="idrid",
        dataset_version="release-v1",
        approved_review_provided=True,
        approved_review_purposes=REQUIRED_STAGE5_REVIEW_PURPOSES,
        dataset_root_provided=True,
        adapter_spec_provided=True,
        stage2_pass_audit_provided=True,
        frozen_data_card_provided=True,
        stage2_access_permit_provided=True,
        manual_confirmation_provided=True,
        stage3_package_summary_provided=True,
        stage4_locked_evidence_provided=True,
    )

    assert report.status is Stage5PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION
    assert report.authorizes_data_access is False
    assert report.reasons == ()


def test_synthetic_fixture_verifies_without_probing_root_and_report_round_trips(
    tmp_path: Path,
) -> None:
    root, outcome = _verify_synthetic(tmp_path)

    assert outcome.report.status is Stage5PrerequisiteStatus.VERIFIED
    assert outcome.report.verification_context is Stage5VerificationContext.SYNTHETIC_FIXTURE
    assert outcome.report.authorizes_data_access is False
    assert outcome.report.to_dict()["dataset_root_probed"] is False
    assert outcome.report.to_dict()["real_data_read"] is False
    assert outcome.report.evidence_files_read == 7
    assert outcome.authorization is not None
    assert not root.exists()
    assert Stage5PrerequisiteReport.from_dict(outcome.report.to_dict()) == outcome.report


def test_synthetic_capability_cannot_be_consumed_as_real_authorization(tmp_path: Path) -> None:
    _, outcome = _verify_synthetic(tmp_path)
    authorization = outcome.authorization
    assert authorization is not None

    with pytest.raises(Stage5AuthorizationError, match="REAL_PREREQUISITE"):
        require_stage5_authorization(authorization)
    assert (
        require_stage5_authorization(
            authorization,
            required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        )
        is authorization
    )


def test_verified_outcome_cannot_pair_real_report_with_synthetic_capability(
    tmp_path: Path,
) -> None:
    _, outcome = _verify_synthetic(tmp_path)
    real_claim = replace(
        outcome.report,
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
    )

    with pytest.raises(ValueError, match="bindings do not match"):
        Stage5VerificationOutcome(
            report=real_claim,
            authorization=outcome.authorization,
        )


def test_real_context_cannot_issue_from_synthetic_json_without_stage2_permit(
    tmp_path: Path,
) -> None:
    root, paths = _synthetic_evidence(tmp_path)

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
    )

    assert outcome.report.status is Stage5PrerequisiteStatus.BLOCKED
    assert outcome.report.reasons == (Stage5PrerequisiteReason.STAGE2_ACCESS_PERMIT_INVALID,)
    assert outcome.report.evidence_files_read == 0
    assert outcome.authorization is None
    assert not root.exists()


def test_detached_fixture_evidence_and_process_permit_still_cannot_issue_real_authority(
    tmp_path: Path,
) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    permit = _simulated_process_permit(root, paths)

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=permit,
    )

    assert outcome.report.status is Stage5PrerequisiteStatus.BLOCKED
    assert outcome.report.reasons == (
        Stage5PrerequisiteReason.REAL_STAGE2_AUDIT_CAPABILITY_UNAVAILABLE,
        Stage5PrerequisiteReason.REAL_STAGE3_PACKAGE_CAPABILITY_UNAVAILABLE,
    )
    assert outcome.report.evidence_files_read == 7
    assert outcome.report.to_dict()["prerequisites_verified"] is False
    assert outcome.report.to_dict()["authorization_issued"] is False
    assert outcome.authorization is None
    assert not root.exists()


def test_mutated_process_permit_blocks_before_evidence_or_root_io(tmp_path: Path) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    permit = _simulated_process_permit(root, paths)
    object.__setattr__(permit.spec, "dataset_id", "ddr")

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=permit,
    )

    assert outcome.report.reasons == (Stage5PrerequisiteReason.STAGE2_ACCESS_PERMIT_INVALID,)
    assert outcome.report.evidence_files_read == 0
    assert outcome.authorization is None
    assert not root.exists()


def test_authorization_is_not_forgeable_copyable_or_retargetable(tmp_path: Path) -> None:
    _, outcome = _verify_synthetic(tmp_path)
    authorization = outcome.authorization
    assert authorization is not None

    copied = copy.copy(authorization)
    retargeted = replace(authorization, _root=tmp_path / "different-root")

    assert authorization_validation_reasons(authorization) == ()
    assert "not issued" in authorization_validation_reasons(copied)[0]
    assert "not issued" in authorization_validation_reasons(retargeted)[0]

    object.__setattr__(authorization, "dataset_version", "tampered-v2")
    assert "bindings changed" in authorization_validation_reasons(authorization)[0]


def test_nested_spec_mutation_is_detected_by_primitive_snapshot(tmp_path: Path) -> None:
    _, outcome = _verify_synthetic(tmp_path)
    authorization = outcome.authorization
    assert authorization is not None

    object.__setattr__(authorization.adapter_spec, "dataset_id", "ddr")

    assert "bindings changed" in authorization_validation_reasons(authorization)[0]


def test_authorization_needs_one_shot_physical_root_binding_before_open(tmp_path: Path) -> None:
    root, outcome = _verify_synthetic(tmp_path)
    authorization = outcome.authorization
    assert authorization is not None
    assert authorization.authorizes_file_open is False
    root.mkdir()

    bound = bind_authorized_root(
        authorization,
        required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
    )
    assert bound.authorizes_file_open is False
    assert (
        bound_root_validation_reasons(
            bound,
            required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        )
        == ()
    )
    assert (
        consume_bound_stage5_root(
            bound,
            required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        )
        == root
    )
    assert (
        "consumed"
        in bound_root_validation_reasons(
            bound,
            required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        )[0]
    )
    with pytest.raises(Stage5RootBindingError, match="already been bound"):
        bind_authorized_root(
            authorization,
            required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_post_authorization_root_binding_rejects_symlink_when_supported(
    tmp_path: Path,
) -> None:
    root, outcome = _verify_synthetic(tmp_path)
    authorization = outcome.authorization
    assert authorization is not None
    target = tmp_path / "different-physical-root"
    target.mkdir()
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("process lacks directory symlink permission")

    with pytest.raises(Stage5RootBindingError, match="link or reparse"):
        bind_authorized_root(
            authorization,
            required_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_authorization_expires_by_monotonic_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, outcome = _verify_synthetic(tmp_path)
    authorization = outcome.authorization
    assert authorization is not None
    monkeypatch.setattr(
        "retinaentail.pilot.gate.time.monotonic_ns",
        lambda: authorization.expires_monotonic_ns,
    )

    assert "expired" in authorization_validation_reasons(authorization)[0]


def test_cross_hash_tampering_fails_closed(tmp_path: Path) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    stage3 = json.loads(paths.stage3_package_summary.read_text(encoding="utf-8"))
    assert isinstance(stage3, dict)
    stage3["package_sha256"] = "0" * 64
    _write_json(paths.stage3_package_summary, stage3)

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
    )

    assert outcome.report.status is Stage5PrerequisiteStatus.BLOCKED
    assert outcome.report.reasons == (Stage5PrerequisiteReason.HASH_MISMATCH,)
    assert outcome.authorization is None
    assert not root.exists()


def test_audit_record_count_must_match_spec_and_frozen_card(tmp_path: Path) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    audit = json.loads(paths.stage2_audit.read_text(encoding="utf-8"))
    assert isinstance(audit, dict)
    summary = audit["summary"]
    assert isinstance(summary, dict)
    summary["record_count"] = 4
    summary["image_files_checked"] = 4
    _write_json(paths.stage2_audit, audit)

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
    )

    assert outcome.report.reasons == (Stage5PrerequisiteReason.PROVENANCE_MISMATCH,)
    assert outcome.authorization is None


def test_evidence_inside_root_is_rejected_before_read(tmp_path: Path) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    root = paths.license_review.parent.parent

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
    )

    assert outcome.report.reasons == (Stage5PrerequisiteReason.EVIDENCE_INSIDE_DATASET_ROOT,)
    assert outcome.report.evidence_files_read == 0


def test_oversized_and_duplicate_key_evidence_fail_closed(tmp_path: Path) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    paths.license_review.write_bytes(b"x" * (512 * 1024 + 1))
    oversized = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
    )
    assert oversized.report.reasons == (Stage5PrerequisiteReason.EVIDENCE_TOO_LARGE,)

    root, paths = _synthetic_evidence(tmp_path / "second")
    paths.license_review.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}')
    duplicate = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=paths,
        stage2_access_permit=None,
    )
    assert duplicate.report.reasons == (Stage5PrerequisiteReason.EVIDENCE_DECODE_FAILED,)


def test_symlink_evidence_is_rejected_when_supported(tmp_path: Path) -> None:
    root, paths = _synthetic_evidence(tmp_path)
    link = paths.license_review.parent / "review-link.json"
    try:
        link.symlink_to(paths.license_review)
    except OSError:
        pytest.skip("process lacks symlink permission")
    linked = replace(paths, license_review=link)

    outcome = verify_stage5_prerequisites(
        verification_context=Stage5VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="idrid",
        dataset_version="fixture-v1",
        dataset_root=root,
        evidence_paths=linked,
        stage2_access_permit=None,
    )

    assert outcome.report.reasons == (Stage5PrerequisiteReason.EVIDENCE_LINK_OR_REPARSE_POINT,)
