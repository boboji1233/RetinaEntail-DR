from __future__ import annotations

import copy
import inspect
import pickle

import pytest

from retinaentail.entailment.states import EntailmentState
from retinaentail.pilot.stage10_gate import (
    Stage10CapabilityError,
    Stage10PreflightStatus,
    assess_stage10_preflight,
    consume_stage10_runner_pair,
    issue_synthetic_stage9_runner_capability,
)
from retinaentail.reproducibility.stage9 import Stage9Observation
from retinaentail.reproducibility.stage10 import Stage10RunnerSnapshot


def _snapshot(runner_id: str, pid: int, nonce: str) -> Stage10RunnerSnapshot:
    observation = Stage9Observation(
        case_id="case",
        run_id="run",
        state=EntailmentState.OUT_OF_MODEL,
        positive_score=0.0,
        negative_score=0.0,
        fail_closed=True,
        out_of_distribution=True,
        positive_witness_sha256="1" * 64,
        negative_witness_sha256="2" * 64,
        positive_witness_replay_valid=False,
        negative_witness_replay_valid=False,
        canonical_sha256="3" * 64,
    )
    return Stage10RunnerSnapshot(
        runner_id=runner_id,
        producer_process_id=pid,
        session_nonce_sha256=nonce * 64,
        environment_lock_sha256="4" * 64,
        host_profile_sha256="5" * 64,
        dataset_id="toy",
        dataset_version="v1",
        scope="SYNTHETIC_ONLY",
        split="test",
        source_stage9_package_sha256="6" * 64,
        stage9_package_sha256=("7" if runner_id == "a" else "8") * 64,
        stage9_package_file_count=2,
        stage9_manifest_sha256=("9" if runner_id == "a" else "a") * 64,
        stage9_audit_sha256=("b" if runner_id == "a" else "c") * 64,
        stage9_preregistration_sha256="d" * 64,
        source_stage7_package_sha256="e" * 64,
        source_stage8_package_sha256="f" * 64,
        deterministic_projection_sha256="0" * 64,
        envelope_sha256=("1" if runner_id == "a" else "2") * 64,
        signature_verified=True,
        observations=(observation,),
    )


def test_real_preflight_is_path_free_and_blocked_by_default() -> None:
    signature = inspect.signature(assess_stage10_preflight)
    assert all("path" not in name and "root" not in name for name in signature.parameters)
    report = assess_stage10_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage9_package_a_capability_present=False,
        complete_real_stage9_package_b_capability_present=False,
        fresh_runner_a_capability_present=False,
        fresh_runner_b_capability_present=False,
        two_approved_runners_confirmed=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
        signer_trust_policy_preregistered=False,
    )
    payload = report.to_dict()
    assert report.status is Stage10PreflightStatus.BLOCKED
    assert payload["accepts_paths"] is False
    assert payload["reads_environment"] is False
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False


def test_capabilities_are_process_bound_nonserializable_and_one_shot() -> None:
    left = issue_synthetic_stage9_runner_capability(lambda: _snapshot("a", 101, "a"))
    right = issue_synthetic_stage9_runner_capability(lambda: _snapshot("b", 202, "b"))
    assert left.authorizes_file_open is False
    assert left.authorizes_real_data is False
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(left)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(left)
    observed = consume_stage10_runner_pair(left, right, lambda a, b: (a.runner_id, b.runner_id))
    assert observed == ("a", "b")
    with pytest.raises(Stage10CapabilityError, match="already consumed"):
        consume_stage10_runner_pair(left, right, lambda a, b: (a, b))


def test_detached_snapshot_and_same_runner_are_rejected() -> None:
    snapshot = _snapshot("a", 101, "a")
    left = issue_synthetic_stage9_runner_capability(lambda: snapshot)
    right = issue_synthetic_stage9_runner_capability(lambda: snapshot)
    with pytest.raises(Stage10CapabilityError, match="not independent"):
        consume_stage10_runner_pair(left, right, lambda a, b: (a, b))
    with pytest.raises(Stage10CapabilityError, match="not a live"):
        consume_stage10_runner_pair(snapshot, right, lambda a, b: (a, b))  # type: ignore[arg-type]
