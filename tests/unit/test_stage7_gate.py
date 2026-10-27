from __future__ import annotations

import copy
import inspect
import os
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import retinaentail.pilot.stage6_gate as stage6_gate
import retinaentail.pilot.stage7_gate as stage7_gate
from retinaentail.calibration.stage3_package import (
    Stage3PackageScope,
    verify_and_issue_stage3_package,
)
from retinaentail.calibration.toy_fixture import run_stage3_toy_fixture
from retinaentail.data.stage2_fixture import _run_stage2_fixture_at
from retinaentail.data.stage2_package import (
    Stage2PackageScope,
    VerifiedStage2PackageCapability,
    verify_and_issue_stage2_package,
)
from retinaentail.pilot.stage6_gate import (
    Stage6BoundRoot,
    Stage6VerificationContext,
    bind_stage6_authorized_root,
    verify_stage6_prerequisites,
)
from retinaentail.pilot.stage7_gate import (
    Stage7PreflightStatus,
    Stage7Reason,
    Stage7RootTransaction,
    Stage7RootTransactionError,
    assess_stage7_preflight,
    begin_stage7_root_transaction,
    consume_stage7_root_transaction,
    transaction_validation_reasons,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE3_CONFIG = PROJECT_ROOT / "configs" / "stage3_toy_calibration.yaml"
SYNTHETIC_DATASET_ID = "retinaentail_stage7_synthetic"
SYNTHETIC_DATASET_VERSION = "stage7-synthetic-v1"


@pytest.fixture
def synthetic_bound_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Stage6BoundRoot, Path]:
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
    monkeypatch.setattr(stage6_gate, "_stage5_report_binding", lambda value: "f" * 64)
    root = (tmp_path / "authorized-root").absolute()
    root.mkdir()
    outcome = verify_stage6_prerequisites(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id=SYNTHETIC_DATASET_ID,
        dataset_version=SYNTHETIC_DATASET_VERSION,
        dataset_root=root,
        stage2_package_capability=stage2,
        stage3_package_capability=stage3,
        stage5_pilot_report=object(),
        authorization_ttl_seconds=5,
    )
    assert outcome.authorization is not None
    bound = bind_stage6_authorized_root(
        outcome.authorization,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
    )
    return bound, root


def _begin_synthetic(bound: Stage6BoundRoot) -> Stage7RootTransaction:
    return begin_stage7_root_transaction(
        bound,
        expected_dataset_id=SYNTHETIC_DATASET_ID,
        expected_dataset_version=SYNTHETIC_DATASET_VERSION,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        transaction_ttl_seconds=60,
    )


def test_presence_preflight_accepts_no_path_and_missing_real_inputs_are_zero_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(assess_stage7_preflight)
    assert all(parameter.annotation is not Path for parameter in signature.parameters.values())
    touched: list[Path] = []

    def trap_lstat(path: Path) -> os.stat_result:
        touched.append(Path(path))
        raise AssertionError("presence-only preflight touched the filesystem")

    monkeypatch.setattr("retinaentail.pilot.stage7_gate.os.lstat", trap_lstat)
    report = assess_stage7_preflight(
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        dataset_id=None,
        dataset_version=None,
        three_purpose_review_provided=False,
        physical_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_package_provided=False,
        approved_real_data_card_provided=False,
        manual_confirmation_provided=False,
        stage3_calibration_package_provided=False,
        stage4_locked_evidence_provided=False,
        stage6_bound_root_provided=False,
    )

    assert report.status is Stage7PreflightStatus.BLOCKED
    assert Stage7Reason.THREE_PURPOSE_REVIEW_NOT_PROVIDED in report.reasons
    assert Stage7Reason.STAGE6_BOUND_ROOT_NOT_PROVIDED in report.reasons
    assert report.to_dict()["accepts_paths"] is False
    assert report.to_dict()["dataset_root_probed"] is False
    assert touched == []


def test_synthetic_preflight_is_independent_and_never_real_authority() -> None:
    report = assess_stage7_preflight(
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        dataset_id=SYNTHETIC_DATASET_ID,
        dataset_version=SYNTHETIC_DATASET_VERSION,
        three_purpose_review_provided=False,
        physical_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_package_provided=False,
        approved_real_data_card_provided=False,
        manual_confirmation_provided=False,
        stage3_calibration_package_provided=False,
        stage4_locked_evidence_provided=False,
        stage6_bound_root_provided=True,
    )

    assert report.status is Stage7PreflightStatus.READY_FOR_BOUND_ROOT_CONSUMPTION
    assert report.to_dict()["synthetic_authorizes_real"] is False
    assert report.to_dict()["authorizes_file_open"] is False


def test_detached_or_context_mismatched_input_is_rejected_before_root_probe(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, root = synthetic_bound_root
    original_lstat = os.lstat
    root_probes: list[Path] = []

    def tracking_lstat(path: Path) -> os.stat_result:
        if Path(path) == root:
            root_probes.append(Path(path))
        return original_lstat(path)

    monkeypatch.setattr("retinaentail.pilot.stage7_gate.os.lstat", tracking_lstat)
    with pytest.raises(Stage7RootTransactionError, match="invalid type"):
        begin_stage7_root_transaction(
            {"status": "VERIFIED", "root": str(root)},
            expected_dataset_id=SYNTHETIC_DATASET_ID,
            expected_dataset_version=SYNTHETIC_DATASET_VERSION,
        )
    with pytest.raises(Stage7RootTransactionError, match="REAL_PREREQUISITE"):
        begin_stage7_root_transaction(
            bound,
            expected_dataset_id=SYNTHETIC_DATASET_ID,
            expected_dataset_version=SYNTHETIC_DATASET_VERSION,
        )
    assert root_probes == []


def test_package_tamper_blocks_conversion_before_dataset_root_probe(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, root = synthetic_bound_root
    stage2_capability = cast(
        VerifiedStage2PackageCapability,
        bound.authorization._stage2_package_capability,
    )
    package_file = (
        stage2_capability._package_dir / stage2_capability.descriptor.files[0].relative_path
    )
    package_file.write_bytes(package_file.read_bytes() + b"\n")
    original_lstat = os.lstat
    root_probes: list[Path] = []

    def tracking_lstat(path: Path) -> os.stat_result:
        if Path(path) == root:
            root_probes.append(Path(path))
        return original_lstat(path)

    monkeypatch.setattr("retinaentail.pilot.stage7_gate.os.lstat", tracking_lstat)
    with pytest.raises(Stage7RootTransactionError, match="complete Stage 2/3 replay failed"):
        _begin_synthetic(bound)
    assert root_probes == []


def test_transaction_is_process_bound_nonserializable_and_callback_one_shot(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
) -> None:
    bound, root = synthetic_bound_root
    transaction = _begin_synthetic(bound)

    assert transaction.authorizes_file_open is False
    assert transaction.evidence.authorizes_dataset_root_access is False
    assert transaction.evidence.to_dict()["authorizes_dataset_root_access"] is False
    assert transaction.evidence.to_dict()["frozen_data_card_binding_sha256"] is None
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(transaction)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(transaction)
    with pytest.raises(TypeError):
        replace(transaction, ttl_seconds=1)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(transaction)
    assert "REAL_PREREQUISITE" in transaction_validation_reasons(transaction)[0]
    assert (
        transaction_validation_reasons(
            transaction,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )
        == ()
    )

    observed: dict[str, object] = {}

    def trusted_consumer(path: Path, evidence: object) -> dict[str, int]:
        observed["path"] = path
        observed["evidence"] = evidence
        return {
            "record_count": transaction.evidence.stage2_package_evidence.descriptor.record_count
        }

    result = consume_stage7_root_transaction(
        transaction,
        trusted_consumer,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
    )
    assert observed["path"] == root
    assert observed["evidence"] is transaction.evidence
    assert result["record_count"] > 0
    with pytest.raises(Stage7RootTransactionError, match="already been consumed"):
        consume_stage7_root_transaction(
            transaction,
            trusted_consumer,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_transaction_ttl_and_process_mismatch_fail_before_physical_probe(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, root = synthetic_bound_root
    transaction = _begin_synthetic(bound)
    touched: list[Path] = []

    def trap_lstat(path: Path) -> os.stat_result:
        touched.append(Path(path))
        raise AssertionError("invalid transaction reached physical-root validation")

    monkeypatch.setattr("retinaentail.pilot.stage7_gate.os.lstat", trap_lstat)
    monkeypatch.setattr(stage7_gate, "_PROCESS_ID", -1)
    reasons = transaction_validation_reasons(
        transaction,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
    )
    assert any("different process" in reason for reason in reasons)
    assert touched == []
    monkeypatch.setattr(stage7_gate, "_PROCESS_ID", os.getpid())
    monkeypatch.setattr(
        "retinaentail.pilot.stage7_gate.time.monotonic_ns",
        lambda: transaction.expires_monotonic_ns,
    )
    reasons = transaction_validation_reasons(
        transaction,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
    )
    assert any("expired" in reason for reason in reasons)
    assert touched == []
    assert root not in touched


def test_consumer_cannot_return_authorized_root(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
) -> None:
    bound, _ = synthetic_bound_root
    transaction = _begin_synthetic(bound)

    with pytest.raises(Stage7RootTransactionError, match="returned the authorized root"):
        consume_stage7_root_transaction(
            transaction,
            lambda path, evidence: path,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_consumer_failure_is_fail_closed_and_still_consumes_transaction(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
) -> None:
    bound, _ = synthetic_bound_root
    transaction = _begin_synthetic(bound)

    def failing_consumer(path: Path, evidence: object) -> None:
        del path, evidence
        raise RuntimeError("synthetic audit failure")

    with pytest.raises(Stage7RootTransactionError, match="trusted root consumer failed"):
        consume_stage7_root_transaction(
            transaction,
            failing_consumer,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )
    with pytest.raises(Stage7RootTransactionError, match="already been consumed"):
        consume_stage7_root_transaction(
            transaction,
            failing_consumer,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_post_callback_physical_identity_change_is_fail_closed(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _ = synthetic_bound_root
    transaction = _begin_synthetic(bound)
    original_identity = stage7_gate._strict_physical_root_identity
    calls = 0

    def changing_identity(root: Path) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        identity = original_identity(root)
        if calls >= 2:
            return (*identity[:-1], identity[-1] + 1)
        return identity

    monkeypatch.setattr(stage7_gate, "_strict_physical_root_identity", changing_identity)
    with pytest.raises(Stage7RootTransactionError, match="identity changed during callback"):
        consume_stage7_root_transaction(
            transaction,
            lambda path, evidence: {"status": "PASS"},
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )


def test_reparse_simulation_is_rejected_fail_closed(
    synthetic_bound_root: tuple[Stage6BoundRoot, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, root = synthetic_bound_root
    original_lstat = os.lstat

    def mark_root_as_reparse(path: Path) -> os.stat_result | SimpleNamespace:
        metadata = original_lstat(path)
        if Path(path) == root:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=0x0400,
            )
        return metadata

    monkeypatch.setattr(
        "retinaentail.pilot.stage7_gate.os.lstat",
        mark_root_as_reparse,
    )
    with pytest.raises(Stage7RootTransactionError, match="link or reparse"):
        _begin_synthetic(bound)


def test_unknown_transaction_is_fail_closed() -> None:
    assert "invalid type" in transaction_validation_reasons(object())[0]
