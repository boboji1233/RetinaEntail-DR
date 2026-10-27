from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from retinaentail.calibration.stage3_package import (
    Stage3PackageScope,
    VerifiedStage3PackageCapability,
    verify_and_issue_stage3_package,
)
from retinaentail.calibration.toy_fixture import run_stage3_toy_fixture
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseReviewDecision,
    LicenseReviewRecord,
    RegistryLicenseMetadata,
    authorize_dataset_access,
)
from retinaentail.data.stage2_fixture import _run_stage2_fixture_at
from retinaentail.data.stage2_models import DatasetAdapterSpec, DatasetEntrySpec
from retinaentail.data.stage2_package import (
    Stage2PackageScope,
    VerifiedStage2PackageCapability,
    verify_and_issue_stage2_package,
)
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.pilot.identity import (
    Stage5IdentityRecord,
    audit_five_identity_split_isolation,
)
from retinaentail.pilot.stage6_gate import (
    REQUIRED_STAGE6_REVIEW_PURPOSES,
    Stage6AuthorizationError,
    Stage6PreflightStatus,
    Stage6PrerequisiteStatus,
    Stage6Reason,
    Stage6RootBindingError,
    Stage6VerificationContext,
    assess_stage6_preflight,
    authorization_validation_reasons,
    bind_stage6_authorized_root,
    bound_root_validation_reasons,
    consume_bound_stage6_root,
    require_stage6_authorization,
    verify_stage6_prerequisites,
)
from retinaentail.reporting.stage5 import Stage5PilotReport, issue_stage5_pilot_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE3_CONFIG = PROJECT_ROOT / "configs" / "stage3_toy_calibration.yaml"


def _locked_contract() -> dict[str, object]:
    return {
        "alpha": 0.25,
        "positive_threshold": 0.01,
        "negative_threshold": 0.02,
        "positive_sample_count": 4,
        "negative_sample_count": 4,
        "total_sample_count": 8,
        "calibration_artifact_hash": "4" * 64,
        "config_hash": "1" * 64,
        "data_hash": "2" * 64,
        "model_hash": "3" * 64,
    }


def _identity_records() -> tuple[Stage5IdentityRecord, ...]:
    splits = ("train", "calibration", "test", "test")
    return tuple(
        Stage5IdentityRecord(
            dataset_id="retinaentail_stage5_toy",
            relative_path=f"{split}/record-{index}.synthetic",
            split=split,
            source_image_id=f"source-{index}",
            patient_id=f"patient-{index}",
            eye_id="R",
            followup_id="baseline",
            intervention_pair_id=f"pair-{index}",
        )
        for index, split in enumerate(splits)
    )


def _case_index() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "pilot_index": index,
            "case_id": f"case-{index}",
            "reference_label": "UNKNOWN",
            "state": state.value,
            "status": "SUCCESS",
            "failure_reasons": [],
            "dual_witness_replay_passed": True,
            "positive_witness_replay_passed": True,
            "negative_witness_replay_passed": True,
            "evaluation_sha256": f"{index + 1:x}" * 64,
            "certificate_sha256": f"{index + 5:x}" * 64,
            "dual_replay_sha256": f"{index + 9:x}" * 64,
            "positive_witness_sha256": f"{index + 1:x}" * 64,
            "negative_witness_sha256": f"{index + 5:x}" * 64,
        }
        for index, state in enumerate(EntailmentState)
    )


def _evaluation() -> dict[str, object]:
    state_counts = {state.value: 1 for state in EntailmentState}
    state_proportions = {
        state.value: wilson_score_proportion(1, 4, confidence_level=0.95).to_dict()
        for state in EntailmentState
    }
    return {
        "schema_version": "4.0.0",
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "clinical_claim": False,
        "status": "SUCCESS",
        "locked_contract": _locked_contract(),
        "confidence_level": 0.95,
        "input_record_count": 4,
        "test_case_count": 4,
        "rejected_record_count": 0,
        "evaluable_count": 4,
        "unevaluable_count": 0,
        "fail_closed_count": 0,
        "out_of_distribution_count": 0,
        "both_witness_replays_passed_count": 4,
        "one_or_more_witness_replays_failed_count": 0,
        "labeled_reference_count": 0,
        "state_counts": state_counts,
        "reference_label_counts": {
            "LESION_PRESENT": 0,
            "LESION_ABSENT": 0,
            "UNKNOWN": 4,
        },
        "failure_reason_counts": {},
        "rejected_reason_counts": {},
        "aggregation_failure_reasons": [],
        "state_proportions": state_proportions,
        "evaluable_proportion": wilson_score_proportion(4, 4).to_dict(),
        "unevaluable_proportion": wilson_score_proportion(0, 4).to_dict(),
        "fail_closed_proportion": wilson_score_proportion(0, 4).to_dict(),
        "out_of_distribution_proportion": wilson_score_proportion(0, 4).to_dict(),
        "both_witness_replays_passed_proportion": wilson_score_proportion(4, 4).to_dict(),
        "one_or_more_witness_replays_failed_proportion": wilson_score_proportion(0, 4).to_dict(),
    }


def _stage5_report() -> Stage5PilotReport:
    records = _identity_records()
    return issue_stage5_pilot_report(
        pilot_id="stage6-independent-synthetic-orchestration",
        source_stage4_package_sha256="a" * 64,
        source_stage4_file_count=88,
        stage5_config_sha256="b" * 64,
        pilot_data_sha256="c" * 64,
        stage5_model_provenance_sha256="d" * 64,
        locked_contract=_locked_contract(),
        identity_isolation=cast(
            dict[str, object],
            audit_five_identity_split_isolation(records).to_dict(),
        ),
        identity_records=records,
        evaluation=_evaluation(),
        case_index=_case_index(),
    )


def _locked_test_permit(root: Path) -> DatasetAccessPermit:
    registry = RegistryLicenseMetadata(
        schema_version="1.0.0",
        dataset_id="idrid",
        dataset_version="release-v1",
        license_status="verified",
        license_review_status="reviewed",
        label_verification_status="verified",
    )
    review = LicenseReviewRecord(
        dataset_id="idrid",
        dataset_version="release-v1",
        decision=LicenseReviewDecision.APPROVED,
        permitted_purposes=REQUIRED_STAGE6_REVIEW_PURPOSES,
        terms_sha256="1" * 64,
        evidence_sha256="2" * 64,
        reviewer_id="stage6-test-reviewer",
        reviewed_at="2026-08-24T00:00:00+08:00",
    )
    spec = DatasetAdapterSpec(
        dataset_id="idrid",
        entries=(
            DatasetEntrySpec(
                image_path=Path("images/locked-test.png"),
                source_image_id="locked-test-source",
                patient_id="locked-test-patient",
                eye_id="R",
                followup_id="baseline",
                intervention_pair_id="locked-test-pair",
                split="test",
            ),
        ),
    )
    return authorize_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.LOCKED_TEST_EVALUATION,
        root=root,
        spec=spec,
    )


def _synthetic_capabilities(
    tmp_path: Path,
) -> tuple[VerifiedStage2PackageCapability, VerifiedStage3PackageCapability]:
    stage2_dir = _run_stage2_fixture_at(tmp_path / "stage2").package_dir
    stage2 = verify_and_issue_stage2_package(
        stage2_dir,
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        ttl_seconds=60,
    )
    stage3_dir = tmp_path / "stage3"
    run_stage3_toy_fixture(STAGE3_CONFIG, stage3_dir)
    stage3 = verify_and_issue_stage3_package(
        stage3_dir,
        scope=Stage3PackageScope.SYNTHETIC_TOY,
        ttl_seconds=60,
    )
    return stage2, stage3


def test_preflight_accepts_no_paths_and_missing_real_evidence_is_zero_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(assess_stage6_preflight)
    assert "dataset_root" not in signature.parameters
    assert all("path" not in name for name in signature.parameters)
    touched: list[Path] = []

    def trap_lstat(path: Path) -> object:
        touched.append(Path(path))
        raise AssertionError("zero-I/O preflight touched a path")

    monkeypatch.setattr("retinaentail.pilot.stage6_gate.os.lstat", trap_lstat)
    report = assess_stage6_preflight(
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        dataset_id=None,
        dataset_version=None,
        approved_review_purposes=(),
        dataset_root_provided=False,
        stage2_access_permit_provided=False,
        stage2_package_capability_provided=False,
        stage3_package_capability_provided=False,
        frozen_data_card_provided=False,
        manual_confirmation_provided=False,
        locked_evidence_provided=False,
    )

    assert report.status is Stage6PreflightStatus.BLOCKED
    assert Stage6Reason.STAGE2_PACKAGE_CAPABILITY_NOT_PROVIDED in report.reasons
    assert report.to_dict()["accepts_paths"] is False
    assert report.to_dict()["dataset_root_probed"] is False
    assert touched == []


def test_detached_pass_verified_summaries_cannot_cross_real_gate_or_probe_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "must-not-be-probed").absolute()
    permit = _locked_test_permit(root)
    touched: list[Path] = []

    def trap_lstat(path: Path) -> object:
        touched.append(Path(path))
        raise AssertionError("blocked REAL verification probed root")

    monkeypatch.setattr("retinaentail.pilot.stage6_gate.os.lstat", trap_lstat)
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        dataset_id="idrid",
        dataset_version="release-v1",
        dataset_root=root,
        stage2_access_permit=permit,
        stage2_package_capability={"status": "PASS"},
        stage3_package_capability={"verification_status": "VERIFIED"},
        frozen_data_card={"scope": "APPROVED_REAL_DATA"},
        manual_confirmation={"scope": "APPROVED_REAL_DATA"},
        locked_evidence={"thresholds_locked_before_test": True},
    )

    assert outcome.report.status is Stage6PrerequisiteStatus.BLOCKED
    assert outcome.report.reasons == (Stage6Reason.STAGE2_PACKAGE_CAPABILITY_INVALID,)
    assert outcome.authorization is None
    assert outcome.report.to_dict()["real_authorization_issued"] is False
    assert touched == []


def test_synthetic_orchestration_is_explicitly_independent_and_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage2, stage3 = _synthetic_capabilities(tmp_path)
    root = (tmp_path / "synthetic-consumer-root").absolute()
    root.mkdir()
    original_lstat = os.lstat
    verification_root_probes: list[Path] = []

    def tracking_lstat(path: Path) -> object:
        candidate = Path(path)
        if candidate == root:
            verification_root_probes.append(candidate)
        return original_lstat(path)

    monkeypatch.setattr("retinaentail.pilot.stage6_gate.os.lstat", tracking_lstat)
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="retinaentail_stage6_synthetic",
        dataset_version="stage6-synthetic-v1",
        dataset_root=root,
        stage2_package_capability=stage2,
        stage3_package_capability=stage3,
        stage5_pilot_report=_stage5_report(),
        authorization_ttl_seconds=5,
    )
    authorization = outcome.authorization

    assert outcome.report.status is Stage6PrerequisiteStatus.VERIFIED
    assert authorization is not None
    assert authorization.authorizes_file_open is False
    assert outcome.report.to_dict()["real_authorization_issued"] is False
    assert outcome.report.to_dict()["same_dataset_provenance_verified"] is False
    assert outcome.report.to_dict()["synthetic_components_independent"] is True
    assert verification_root_probes == []
    with pytest.raises(Stage6AuthorizationError, match="REAL_PREREQUISITE"):
        require_stage6_authorization(authorization)

    bound = bind_stage6_authorized_root(
        authorization,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
    )
    assert bound.authorizes_file_open is False
    with pytest.raises(Stage6RootBindingError, match="already been bound"):
        bind_stage6_authorized_root(
            authorization,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )
    assert (
        consume_bound_stage6_root(
            bound,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )
        == root
    )
    with pytest.raises(Stage6RootBindingError, match="already been consumed"):
        consume_bound_stage6_root(
            bound,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_stage6_authorization_is_process_issued_short_lived_and_not_copyable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage2, stage3 = _synthetic_capabilities(tmp_path)
    root = (tmp_path / "synthetic-root").absolute()
    root.mkdir()
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="retinaentail_stage6_synthetic",
        dataset_version="stage6-synthetic-v1",
        dataset_root=root,
        stage2_package_capability=stage2,
        stage3_package_capability=stage3,
        stage5_pilot_report=_stage5_report(),
        authorization_ttl_seconds=5,
    )
    authorization = outcome.authorization
    assert authorization is not None
    assert authorization.ttl_seconds <= 5
    assert (
        authorization_validation_reasons(
            authorization,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )
        == ()
    )
    copied = copy.copy(authorization)
    assert (
        "not issued"
        in authorization_validation_reasons(
            copied,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )[0]
    )
    monkeypatch.setattr(
        "retinaentail.pilot.stage6_gate.time.monotonic_ns",
        lambda: authorization.expires_monotonic_ns,
    )
    assert (
        "expired"
        in authorization_validation_reasons(
            authorization,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )[0]
    )


def test_root_link_or_reparse_is_rejected_after_synthetic_authorization(
    tmp_path: Path,
) -> None:
    stage2, stage3 = _synthetic_capabilities(tmp_path)
    physical = (tmp_path / "physical-root").absolute()
    physical.mkdir()
    linked = (tmp_path / "linked-root").absolute()
    try:
        linked.symlink_to(physical, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available to this Windows test process")
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="retinaentail_stage6_synthetic",
        dataset_version="stage6-synthetic-v1",
        dataset_root=linked,
        stage2_package_capability=stage2,
        stage3_package_capability=stage3,
        stage5_pilot_report=_stage5_report(),
    )
    assert outcome.authorization is not None

    with pytest.raises(Stage6RootBindingError, match="link or reparse"):
        bind_stage6_authorized_root(
            outcome.authorization,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_ancestor_reparse_attribute_is_rejected_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage2, stage3 = _synthetic_capabilities(tmp_path)
    container = (tmp_path / "authorized-container").absolute()
    root = container / "synthetic-root"
    root.mkdir(parents=True)
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="retinaentail_stage6_synthetic",
        dataset_version="stage6-synthetic-v1",
        dataset_root=root,
        stage2_package_capability=stage2,
        stage3_package_capability=stage3,
        stage5_pilot_report=_stage5_report(),
    )
    assert outcome.authorization is not None
    original_lstat = os.lstat

    def mark_container_as_reparse(path: Path) -> object:
        metadata = original_lstat(path)
        if Path(path) == container:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=0x0400,
            )
        return metadata

    monkeypatch.setattr("retinaentail.pilot.stage6_gate.os.lstat", mark_container_as_reparse)
    with pytest.raises(Stage6RootBindingError, match="link or reparse"):
        bind_stage6_authorized_root(
            outcome.authorization,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_package_mutation_blocks_binding_before_dataset_root_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage2, stage3 = _synthetic_capabilities(tmp_path)
    root = (tmp_path / "unprobed-after-package-mutation").absolute()
    root.mkdir()
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id="retinaentail_stage6_synthetic",
        dataset_version="stage6-synthetic-v1",
        dataset_root=root,
        stage2_package_capability=stage2,
        stage3_package_capability=stage3,
        stage5_pilot_report=_stage5_report(),
    )
    assert outcome.authorization is not None
    first_file = stage2.descriptor.files[0]
    package_file = stage2._package_dir / first_file.relative_path
    package_file.write_bytes(package_file.read_bytes() + b"\n")
    original_lstat = os.lstat
    root_probes: list[Path] = []

    def track_dataset_root(path: Path) -> object:
        candidate = Path(path)
        if candidate == root:
            root_probes.append(candidate)
        return original_lstat(path)

    monkeypatch.setattr("retinaentail.pilot.stage6_gate.os.lstat", track_dataset_root)
    with pytest.raises(Stage6RootBindingError, match="full package replay failed"):
        bind_stage6_authorized_root(
            outcome.authorization,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )
    assert root_probes == []


def test_bound_root_validation_rejects_unissued_value() -> None:
    assert "invalid type" in bound_root_validation_reasons(object())[0]
