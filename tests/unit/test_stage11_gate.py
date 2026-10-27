from __future__ import annotations

import copy
import inspect
import os
import pickle

import pytest

from retinaentail.pilot.stage11_gate import (
    Stage11CapabilityError,
    Stage11PreflightStatus,
    Stage11RunnerAuthority,
    assess_stage11_preflight,
    consume_stage11_host_pair,
    issue_synthetic_stage10_host_capability,
)


def _authority(runner_id: str, marker: str) -> Stage11RunnerAuthority:
    return Stage11RunnerAuthority(
        runner_id=runner_id,
        logical_host_id=f"{runner_id}-local-slot",
        producer_process_id=os.getpid(),
        session_nonce_sha256=marker * 64,
        dataset_id="synthetic_stage2",
        dataset_version="synthetic-fixture-v1",
        scope="SYNTHETIC_ONLY",
        split="test",
        source_stage10_package_sha256="1" * 64,
        stage10_environment_lock_sha256="2" * 64,
        runtime_lock_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        trust_anchor_sha256="5" * 64,
        current_key_id=("6" if runner_id == "runner-a" else "7") * 64,
        baseline_projection_sha256="8" * 64,
        signature_verified=True,
    )


def _preflight(**overrides: object):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "dataset_id": None,
        "dataset_version": None,
        "complete_real_stage10_package_a_capability_present": False,
        "complete_real_stage10_package_b_capability_present": False,
        "fresh_host_a_process_capability_present": False,
        "fresh_host_b_process_capability_present": False,
        "two_approved_physical_hosts_confirmed": False,
        "production_asymmetric_trust_anchor_verified": False,
        "unique_runner_leaf_keys_verified": False,
        "certificate_chains_verified": False,
        "revocation_status_current": False,
        "trusted_time_policy_verified": False,
        "complete_wheel_lock_hashes_verified": False,
        "host_environment_policy_preregistered": False,
        "case_seed_replay_preregistered": False,
        "locked_hashes_thresholds_preregistered": False,
    }
    values.update(overrides)
    return assess_stage11_preflight(**values)  # type: ignore[arg-type]


def test_real_preflight_is_zero_io_and_blocked_without_real_prerequisites() -> None:
    signature = inspect.signature(assess_stage11_preflight)
    assert all("path" not in name and "root" not in name for name in signature.parameters)
    report = _preflight()
    payload = report.to_dict()
    assert report.status is Stage11PreflightStatus.BLOCKED
    assert len(report.reasons) == 16
    assert payload["accepts_paths"] is False
    assert payload["reads_environment"] is False
    assert payload["authorizes_file_open"] is False
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False
    assert payload["production_private_key_accepted_as_input"] is False


def test_complete_presence_preflight_only_reaches_verification_boundary() -> None:
    report = _preflight(
        dataset_id="approved-dataset",
        dataset_version="exact-v1",
        complete_real_stage10_package_a_capability_present=True,
        complete_real_stage10_package_b_capability_present=True,
        fresh_host_a_process_capability_present=True,
        fresh_host_b_process_capability_present=True,
        two_approved_physical_hosts_confirmed=True,
        production_asymmetric_trust_anchor_verified=True,
        unique_runner_leaf_keys_verified=True,
        certificate_chains_verified=True,
        revocation_status_current=True,
        trusted_time_policy_verified=True,
        complete_wheel_lock_hashes_verified=True,
        host_environment_policy_preregistered=True,
        case_seed_replay_preregistered=True,
        locked_hashes_thresholds_preregistered=True,
    )
    assert report.status is Stage11PreflightStatus.READY_FOR_PRODUCTION_HOST_VERIFICATION
    assert report.reasons == ()
    assert report.to_dict()["two_physical_hosts_verified"] is False


def test_synthetic_host_capabilities_are_process_bound_nonserializable_and_one_shot() -> None:
    left = issue_synthetic_stage10_host_capability(lambda: _authority("runner-a", "a"))
    right = issue_synthetic_stage10_host_capability(lambda: _authority("runner-b", "b"))
    assert left.authorizes_file_open is False
    assert left.authorizes_real_data is False
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(left)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(left)
    observed = consume_stage11_host_pair(left, right, lambda a, b: (a.runner_id, b.runner_id))
    assert observed == ("runner-a", "runner-b")
    with pytest.raises(Stage11CapabilityError, match="already consumed"):
        consume_stage11_host_pair(left, right, lambda a, b: (a, b))


def test_shared_leaf_key_and_non_hex_authority_are_fail_closed() -> None:
    left_authority = _authority("runner-a", "a")
    right_authority = _authority("runner-b", "b")
    object.__setattr__(right_authority, "current_key_id", left_authority.current_key_id)
    left = issue_synthetic_stage10_host_capability(lambda: left_authority)
    right = issue_synthetic_stage10_host_capability(lambda: right_authority)
    with pytest.raises(Stage11CapabilityError, match="shared"):
        consume_stage11_host_pair(left, right, lambda a, b: (a, b))
    with pytest.raises(ValueError, match="digest"):
        Stage11RunnerAuthority(**{**left_authority.to_dict(), "session_nonce_sha256": "z" * 64})
