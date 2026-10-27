from __future__ import annotations

import copy
import inspect
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import retinaentail.pilot.stage8_gate as stage8_gate
from retinaentail.pilot.stage6_gate import Stage6VerificationContext
from retinaentail.pilot.stage7_gate import Stage7RootTransaction
from retinaentail.pilot.stage8_gate import (
    STAGE8_NO_PHYSICAL_ROOT_SHA256,
    Stage7PackageDescriptor,
    Stage8CapabilityError,
    Stage8FrozenBindings,
    Stage8PackageScope,
    Stage8PreflightStatus,
    Stage8Reason,
    Stage8RunCapability,
    Stage8SyntheticRunCapability,
    VerifiedStage7PackageCapability,
    assess_stage8_preflight,
    begin_stage8_run_capability,
    begin_stage8_synthetic_run_capability,
    consume_stage8_run_capability,
    consume_stage8_synthetic_run_capability,
    issue_synthetic_stage7_package_capability,
    package_capability_validation_reasons,
    run_capability_validation_reasons,
    synthetic_run_capability_snapshot,
    synthetic_run_capability_validation_reasons,
)

DATASET_ID = "synthetic_stage2"
DATASET_VERSION = "synthetic-fixture-v1"


def _frozen(*, suffix: str = "") -> Stage8FrozenBindings:
    offset = len(suffix)
    return Stage8FrozenBindings(
        fixed_seed=20260824,
        degradation_replay_set_sha256=f"{1 + offset:x}" * 64,
        inclusion_rules_sha256=f"{2 + offset:x}" * 64,
        stability_metrics_sha256=f"{3 + offset:x}" * 64,
        locked_contract_sha256=f"{4 + offset:x}" * 64,
        test_split_sha256=f"{5 + offset:x}" * 64,
        config_hash=f"{6 + offset:x}" * 64,
        data_hash=f"{7 + offset:x}" * 64,
        model_hash=f"{8 + offset:x}" * 64,
        provenance_sha256=f"{9 + offset:x}" * 64,
        preregistration_sha256=f"{10 + offset:x}" * 64,
    )


def _synthetic_descriptor(
    *,
    frozen: Stage8FrozenBindings | None = None,
) -> Stage7PackageDescriptor:
    return Stage7PackageDescriptor(
        scope=Stage8PackageScope.SYNTHETIC_FIXTURE,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        package_sha256="a" * 64,
        package_file_count=130,
        stage2_package_sha256="b" * 64,
        stage3_package_sha256="c" * 64,
        physical_root_identity_sha256=STAGE8_NO_PHYSICAL_ROOT_SHA256,
        frozen_bindings=frozen or _frozen(),
        fixture_only=True,
    )


def _synthetic_package_capability(
    descriptor: Stage7PackageDescriptor | None = None,
) -> tuple[VerifiedStage7PackageCapability, list[int]]:
    expected = descriptor or _synthetic_descriptor()
    replay_calls: list[int] = []

    def replay() -> Stage7PackageDescriptor:
        replay_calls.append(len(replay_calls) + 1)
        return expected

    return issue_synthetic_stage7_package_capability(replay), replay_calls


def _real_descriptor(*, frozen: Stage8FrozenBindings | None = None) -> Stage7PackageDescriptor:
    return Stage7PackageDescriptor(
        scope=Stage8PackageScope.APPROVED_REAL_DATA,
        dataset_id="idrid",
        dataset_version="release-v1",
        package_sha256="a" * 64,
        package_file_count=50,
        stage2_package_sha256="b" * 64,
        stage3_package_sha256="c" * 64,
        physical_root_identity_sha256="d" * 64,
        frozen_bindings=frozen or _frozen(),
        fixture_only=False,
    )


def _fake_real_transaction(frozen: Stage8FrozenBindings) -> Stage7RootTransaction:
    transaction = object.__new__(Stage7RootTransaction)
    stage3 = SimpleNamespace(
        package_sha256="c" * 64,
        config_hash=frozen.config_hash,
        data_hash=frozen.data_hash,
        model_hash=frozen.model_hash,
    )
    transaction_evidence = SimpleNamespace(
        dataset_id="idrid",
        dataset_version="release-v1",
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        stage2_package_evidence=SimpleNamespace(
            descriptor=SimpleNamespace(package_sha256="b" * 64)
        ),
        stage3_package_descriptor=stage3,
        physical_root_identity_sha256="d" * 64,
        binding_sha256=lambda: "e" * 64,
    )
    now = stage8_gate.time.monotonic_ns()
    for name, value in (
        ("dataset_id", "idrid"),
        ("dataset_version", "release-v1"),
        ("verification_context", Stage6VerificationContext.REAL_PREREQUISITE),
        ("evidence", transaction_evidence),
        ("expires_monotonic_ns", now + 5_000_000_000),
    ):
        object.__setattr__(transaction, name, value)
    return transaction


def _issue_fake_real_run(
    monkeypatch: pytest.MonkeyPatch,
    replay: object,
    *,
    descriptor: Stage7PackageDescriptor,
) -> Stage8RunCapability:
    assert callable(replay)
    package_capability = stage8_gate._issue_package_capability(
        descriptor,
        replay,
        ttl_seconds=5,
        issuer_token=stage8_gate._PACKAGE_TOKEN,
    )
    transaction = _fake_real_transaction(descriptor.frozen_bindings)
    monkeypatch.setattr(stage8_gate, "transaction_validation_reasons", lambda *a, **k: ())
    return begin_stage8_run_capability(
        package_capability,
        transaction,
        expected_dataset_id="idrid",
        expected_dataset_version="release-v1",
        frozen_bindings=descriptor.frozen_bindings,
    )


def test_real_preflight_is_zero_io_and_accepts_no_path_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(assess_stage8_preflight)
    assert all(parameter.annotation is not Path for parameter in signature.parameters.values())
    touched: list[object] = []

    def trap(*args: object, **kwargs: object) -> object:
        touched.append((args, kwargs))
        raise AssertionError("zero-I/O Stage 8 preflight touched external state")

    monkeypatch.setattr(stage8_gate.os, "lstat", trap)
    monkeypatch.setattr(stage8_gate.os, "getenv", trap)
    report = assess_stage8_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_stage7_package_capability_provided=False,
        fresh_root_run_capability_provided=False,
        fixed_seed_preregistered=False,
        degradation_replay_set_preregistered=False,
        inclusion_rules_preregistered=False,
        stability_metrics_preregistered=False,
        locked_contract_provided=False,
    )

    assert report.status is Stage8PreflightStatus.BLOCKED
    assert report.verification_context is Stage6VerificationContext.REAL_PREREQUISITE
    assert Stage8Reason.COMPLETE_REAL_STAGE7_PACKAGE_CAPABILITY_NOT_PROVIDED in report.reasons
    assert Stage8Reason.FRESH_ROOT_RUN_CAPABILITY_NOT_PROVIDED in report.reasons
    assert report.to_dict()["accepts_paths"] is False
    assert report.to_dict()["reads_environment"] is False
    assert report.to_dict()["dataset_root_probed"] is False
    assert touched == []


def test_presence_ready_report_is_not_authority() -> None:
    report = assess_stage8_preflight(
        dataset_id="idrid",
        dataset_version="release-v1",
        complete_stage7_package_capability_provided=True,
        fresh_root_run_capability_provided=True,
        fixed_seed_preregistered=True,
        degradation_replay_set_preregistered=True,
        inclusion_rules_preregistered=True,
        stability_metrics_preregistered=True,
        locked_contract_provided=True,
    )

    assert report.status is Stage8PreflightStatus.READY_FOR_PROCESS_CAPABILITY_VERIFICATION
    assert report.to_dict()["authorizes_file_open"] is False
    assert report.to_dict()["detached_receipt_authoritative"] is False


def test_synthetic_package_capability_is_process_bound_short_lived_and_unforgeable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, replay_calls = _synthetic_package_capability()

    assert replay_calls == [1, 2]
    assert capability.ttl_seconds <= 5
    assert capability.authorizes_file_open is False
    assert (
        package_capability_validation_reasons(
            capability,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
            expected_dataset_id=DATASET_ID,
            expected_dataset_version=DATASET_VERSION,
        )
        == ()
    )
    assert "APPROVED_REAL_DATA" in package_capability_validation_reasons(capability)[0]
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(capability)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(capability)
    with pytest.raises(TypeError):
        replace(capability, ttl_seconds=1)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(capability)

    forged = object.__new__(VerifiedStage7PackageCapability)
    assert "not issued" in package_capability_validation_reasons(forged)[0]
    assert "invalid type" in package_capability_validation_reasons({"status": "VERIFIED"})[0]

    monkeypatch.setattr(stage8_gate, "_PROCESS_ID", -1)
    assert (
        "different process"
        in package_capability_validation_reasons(
            capability,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )[0]
    )


def test_synthetic_issuer_cannot_issue_real_scope() -> None:
    real_descriptor = Stage7PackageDescriptor(
        scope=Stage8PackageScope.APPROVED_REAL_DATA,
        dataset_id="idrid",
        dataset_version="release-v1",
        package_sha256="a" * 64,
        package_file_count=20,
        stage2_package_sha256="b" * 64,
        stage3_package_sha256="c" * 64,
        physical_root_identity_sha256="d" * 64,
        frozen_bindings=_frozen(),
        fixture_only=False,
    )

    with pytest.raises(Stage8CapabilityError, match="SYNTHETIC_FIXTURE"):
        issue_synthetic_stage7_package_capability(lambda: real_descriptor)


def test_expired_package_capability_is_rejected_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, replay_calls = _synthetic_package_capability()
    monkeypatch.setattr(
        stage8_gate.time,
        "monotonic_ns",
        lambda: capability.expires_monotonic_ns,
    )

    assert (
        "expired"
        in package_capability_validation_reasons(
            capability,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )[0]
    )
    with pytest.raises(Stage8CapabilityError, match="expired"):
        begin_stage8_synthetic_run_capability(
            capability,
            expected_dataset_id=DATASET_ID,
            expected_dataset_version=DATASET_VERSION,
            frozen_bindings=_frozen(),
        )
    assert replay_calls == [1, 2]


def test_synthetic_run_replays_package_is_one_shot_and_never_has_root_authority() -> None:
    package_capability, replay_calls = _synthetic_package_capability()
    run = begin_stage8_synthetic_run_capability(
        package_capability,
        expected_dataset_id=DATASET_ID,
        expected_dataset_version=DATASET_VERSION,
        frozen_bindings=_frozen(),
    )

    assert replay_calls == [1, 2, 3]
    assert isinstance(run, Stage8SyntheticRunCapability)
    assert run.ttl_seconds <= 5
    assert run.authorizes_file_open is False
    assert run.authorizes_embedded_package_replay is True
    assert run.evidence.authorizes_dataset_root_access is False
    assert synthetic_run_capability_validation_reasons(run) == ()
    assert (
        "already been consumed"
        in package_capability_validation_reasons(
            package_capability,
            required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        )[0]
    )
    before = synthetic_run_capability_snapshot(run)
    assert before["consumed"] is False
    assert before["authorizes_real_data"] is False
    assert before["authorizes_physical_dataset_root"] is False
    assert before["detached_receipt_authoritative"] is False
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(run)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(run)

    observed: list[object] = []
    result = consume_stage8_synthetic_run_capability(
        run,
        lambda evidence: observed.append(evidence) or {"status": "PASS"},
    )
    assert result == {"status": "PASS"}
    assert observed == [run.evidence]
    assert replay_calls == [1, 2, 3, 4]
    after = synthetic_run_capability_snapshot(run)
    assert after["consumed"] is True
    assert after["complete_stage7_package_replayed"] is True

    with pytest.raises(Stage8CapabilityError, match="already been consumed"):
        consume_stage8_synthetic_run_capability(run, lambda evidence: None)
    assert synthetic_run_capability_snapshot(run)["second_consumption_rejected"] is True


def test_synthetic_run_mismatch_fails_before_replay_and_detached_input_is_rejected() -> None:
    package_capability, replay_calls = _synthetic_package_capability()
    with pytest.raises(Stage8CapabilityError, match="frozen bindings"):
        begin_stage8_synthetic_run_capability(
            package_capability,
            expected_dataset_id=DATASET_ID,
            expected_dataset_version=DATASET_VERSION,
            frozen_bindings=_frozen(suffix="x"),
        )
    assert replay_calls == [1, 2]

    with pytest.raises(Stage8CapabilityError, match="invalid type"):
        begin_stage8_synthetic_run_capability(
            {"status": "VERIFIED"},
            expected_dataset_id=DATASET_ID,
            expected_dataset_version=DATASET_VERSION,
            frozen_bindings=_frozen(),
        )


def test_real_binding_hash_or_root_mismatch_blocks_before_physical_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _frozen()
    descriptor = Stage7PackageDescriptor(
        scope=Stage8PackageScope.APPROVED_REAL_DATA,
        dataset_id="idrid",
        dataset_version="release-v1",
        package_sha256="a" * 64,
        package_file_count=50,
        stage2_package_sha256="b" * 64,
        stage3_package_sha256="c" * 64,
        physical_root_identity_sha256="d" * 64,
        frozen_bindings=frozen,
        fixture_only=False,
    )
    package_capability = stage8_gate._issue_package_capability(
        descriptor,
        lambda: descriptor,
        ttl_seconds=5,
        issuer_token=stage8_gate._PACKAGE_TOKEN,
    )
    transaction = object.__new__(Stage7RootTransaction)
    stage3 = SimpleNamespace(
        package_sha256="c" * 64,
        config_hash=frozen.config_hash,
        data_hash=frozen.data_hash,
        model_hash=frozen.model_hash,
    )
    transaction_evidence = SimpleNamespace(
        dataset_id="idrid",
        dataset_version="release-v1",
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        stage2_package_evidence=SimpleNamespace(
            descriptor=SimpleNamespace(package_sha256="b" * 64)
        ),
        stage3_package_descriptor=stage3,
        physical_root_identity_sha256="e" * 64,
    )
    for name, value in (
        ("dataset_id", "idrid"),
        ("dataset_version", "release-v1"),
        ("verification_context", Stage6VerificationContext.REAL_PREREQUISITE),
        ("evidence", transaction_evidence),
    ):
        object.__setattr__(transaction, name, value)
    physical_probes: list[object] = []

    def trap_transaction_validation(*args: object, **kwargs: object) -> tuple[str, ...]:
        physical_probes.append((args, kwargs))
        raise AssertionError("hash mismatch reached physical-root validation")

    monkeypatch.setattr(stage8_gate, "transaction_validation_reasons", trap_transaction_validation)
    with pytest.raises(Stage8CapabilityError, match="physical root identity"):
        begin_stage8_run_capability(
            package_capability,
            transaction,
            expected_dataset_id="idrid",
            expected_dataset_version="release-v1",
            frozen_bindings=frozen,
        )
    assert physical_probes == []


@pytest.mark.parametrize("failure_mode", ["changed", "raised"])
def test_real_consume_replays_package_before_root_and_failure_consumes_authority(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    descriptor = _real_descriptor()
    replay_calls = 0

    def replay() -> Stage7PackageDescriptor:
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls < 3:
            return descriptor
        if failure_mode == "raised":
            raise ValueError("package bytes changed")
        return replace(descriptor, package_sha256="f" * 64)

    run = _issue_fake_real_run(monkeypatch, replay, descriptor=descriptor)
    assert replay_calls == 2
    root_access: list[object] = []

    def trap_root_consume(*args: object, **kwargs: object) -> object:
        root_access.append((args, kwargs))
        raise AssertionError("root access occurred before final package replay")

    monkeypatch.setattr(stage8_gate, "consume_stage7_root_transaction", trap_root_consume)
    expected = "replay failed" if failure_mode == "raised" else "descriptor changed"
    with pytest.raises(Stage8CapabilityError, match=expected):
        consume_stage8_run_capability(run, lambda root, stage7, stage8: None)

    assert replay_calls == 3
    assert root_access == []
    assert "already been consumed" in run_capability_validation_reasons(run)[0]
    with pytest.raises(Stage8CapabilityError, match="already been consumed"):
        consume_stage8_run_capability(run, lambda root, stage7, stage8: None)
    assert replay_calls == 3


def test_real_consume_checks_stage8_ttl_after_stage7_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _real_descriptor()
    events: list[str] = []

    def replay() -> Stage7PackageDescriptor:
        events.append("package-replay")
        return descriptor

    run = _issue_fake_real_run(monkeypatch, replay, descriptor=descriptor)
    clock = [run.issued_monotonic_ns + 1]
    monkeypatch.setattr(stage8_gate.time, "monotonic_ns", lambda: clock[0])

    def stage7_consume(
        value: object,
        consumer: object,
        *,
        required_context: object,
    ) -> object:
        del value, required_context
        events.append("root-callback")
        assert callable(consumer)
        result = consumer(Path("C:/approved/root"), SimpleNamespace())
        clock[0] = run.expires_monotonic_ns
        return result

    monkeypatch.setattr(stage8_gate, "consume_stage7_root_transaction", stage7_consume)
    with pytest.raises(Stage8CapabilityError, match="expired after root callback"):
        consume_stage8_run_capability(
            run,
            lambda root, stage7, stage8: {"status": "PASS"},
        )

    assert events[-2:] == ["package-replay", "root-callback"]
    assert "already been consumed" in run_capability_validation_reasons(run)[0]


def test_expired_synthetic_run_fails_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_capability, replay_calls = _synthetic_package_capability()
    run = begin_stage8_synthetic_run_capability(
        package_capability,
        expected_dataset_id=DATASET_ID,
        expected_dataset_version=DATASET_VERSION,
        frozen_bindings=_frozen(),
    )
    monkeypatch.setattr(stage8_gate.time, "monotonic_ns", lambda: run.expires_monotonic_ns)

    assert "expired" in synthetic_run_capability_validation_reasons(run)[0]
    with pytest.raises(Stage8CapabilityError, match="expired"):
        consume_stage8_synthetic_run_capability(run, lambda evidence: None)
    assert replay_calls == [1, 2, 3]


def test_nested_root_disclosure_detector_is_fail_closed() -> None:
    root = Path("C:/approved/root")
    assert stage8_gate._result_discloses_root({"nested": [root]}, root)
    assert stage8_gate._result_discloses_root({"nested": root.as_posix()}, root)
    assert not stage8_gate._result_discloses_root({"status": "PASS"}, root)


def test_root_disclosure_detector_normalizes_windows_children_without_false_substrings() -> None:
    root = Path("C:/Approved/Root")

    assert stage8_gate._result_discloses_root("c:\\approved\\root", root)
    assert stage8_gate._result_discloses_root(
        "C:/APPROVED/ROOT/witnesses/positive.json",
        root,
    )
    assert stage8_gate._result_discloses_root(
        Path("c:/approved/root/cases/case-1.json"),
        root,
    )
    assert not stage8_gate._result_discloses_root(
        "C:/Approved/Rooted/witness.json",
        root,
    )
    assert not stage8_gate._result_discloses_root("prefix-C:/Approved/Root", root)
    assert not stage8_gate._result_discloses_root("ordinary status text", root)
    assert not stage8_gate._result_discloses_root(Path("C:/Approved/Other/file.json"), root)


def test_root_disclosure_detector_rejects_custom_or_unknown_return_objects() -> None:
    root = Path("C:/Approved/Root")

    class AttributeCarrier:
        def __init__(self, path: Path) -> None:
            self.path = path

    class BenignCustomResult:
        status = "PASS"

    assert stage8_gate._result_discloses_root(
        AttributeCarrier(root / "witness.json"),
        root,
    )
    assert stage8_gate._result_discloses_root(BenignCustomResult(), root)
    assert stage8_gate._result_discloses_root(b"opaque-binary-result", root)
    assert not stage8_gate._result_discloses_root(
        {"status": "PASS", "counts": [1, 2, None, True, 0.5]},
        root,
    )


def test_forged_synthetic_run_is_rejected() -> None:
    forged = cast(Stage8SyntheticRunCapability, object.__new__(Stage8SyntheticRunCapability))
    assert "not issued" in synthetic_run_capability_validation_reasons(forged)[0]
