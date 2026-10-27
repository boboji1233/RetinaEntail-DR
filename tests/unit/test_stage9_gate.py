from __future__ import annotations

import copy
import pickle

import pytest

from retinaentail.pilot.stage9_gate import (
    Stage9CapabilityError,
    Stage9PreflightStatus,
    assess_stage9_preflight,
    consume_stage9_session_pair,
    issue_synthetic_stage8_session_capability,
    session_capability_snapshot,
)
from retinaentail.reproducibility.stage9 import Stage9SessionSnapshot, parse_stage8_observations


def _snapshot(session_id: str, process_id: int, nonce: str) -> Stage9SessionSnapshot:
    base: dict[str, object] = {
        "schema_version": "8.0.0",
        "dataset_id": "synthetic_stage2",
        "dataset_version": "synthetic-fixture-v1",
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "case_id": "case-001",
        "run_id": "stage7_locked_baseline",
        "seed": 7,
        "degradation_replay_id": "frozen",
        "degradation_replay_set_sha256": "a" * 64,
        "case_degradation_trace_sha256": "b" * 64,
        "alpha": 0.1,
        "positive_score": 0.0,
        "negative_score": 0.2,
        "positive_threshold": 0.3,
        "negative_threshold": 0.3,
        "positive_sample_count": 4,
        "negative_sample_count": 4,
        "config_hash": "c" * 64,
        "data_hash": "d" * 64,
        "model_hash": "e" * 64,
        "provenance_sha256": "f" * 64,
        "stage7_package_sha256": "1" * 64,
        "state": "ENTAILED_PRESENT",
        "fail_closed": False,
        "out_of_distribution": False,
        "solver_succeeded": True,
        "degradation_known": True,
        "positive_witness_sha256": "2" * 64,
        "negative_witness_sha256": "3" * 64,
        "positive_witness_replay_valid": True,
        "negative_witness_replay_valid": True,
    }
    repeat = {**base, "run_id": "repeat-001", "seed": 8}
    observations = parse_stage8_observations(
        {
            "schema_version": "8.0.0",
            "scope": "SYNTHETIC_FIXTURE",
            "source_stage7_package_sha256": "1" * 64,
            "exact_case_set_enforced": True,
            "sample_replacement_performed": False,
            "baseline_records": [base],
            "repeat_records": [repeat],
            "replay_receipts": [],
        }
    )
    return Stage9SessionSnapshot(
        session_id=session_id,
        producer_process_id=process_id,
        session_nonce_sha256=nonce,
        environment_lock_sha256="4" * 64,
        dataset_id="synthetic_stage2",
        dataset_version="synthetic-fixture-v1",
        scope="SYNTHETIC_ONLY",
        split="test",
        package_sha256="5" * 64,
        package_file_count=140,
        package_manifest_sha256="6" * 64,
        stage8_audit_sha256="7" * 64,
        stage8_observations_sha256="8" * 64,
        stage8_preregistration_sha256="9" * 64,
        source_stage7_package_sha256="1" * 64,
        observations=observations,
    )


def test_real_preflight_is_blocked_and_path_free_by_default() -> None:
    report = assess_stage9_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_stage8_package_a_capability_provided=False,
        complete_stage8_package_b_capability_provided=False,
        fresh_session_a_capability_provided=False,
        fresh_session_b_capability_provided=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    )
    payload = report.to_dict()
    assert report.status is Stage9PreflightStatus.BLOCKED
    assert payload["accepts_paths"] is False
    assert payload["reads_environment"] is False
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False
    assert payload["detached_pass_json_authoritative"] is False


def test_pair_capabilities_are_process_bound_nonserializable_and_one_shot() -> None:
    left_snapshot = _snapshot("session_a", 101, "a" * 64)
    right_snapshot = _snapshot("session_b", 202, "b" * 64)
    left = issue_synthetic_stage8_session_capability(lambda: left_snapshot)
    right = issue_synthetic_stage8_session_capability(lambda: right_snapshot)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(left)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(left)
    observed = consume_stage9_session_pair(left, right, lambda first, second: (first, second))
    assert observed == (left_snapshot, right_snapshot)
    assert session_capability_snapshot(left)["consumed"] is True
    with pytest.raises(Stage9CapabilityError, match="already consumed"):
        consume_stage9_session_pair(left, right, lambda first, second: (first, second))


def test_same_producer_session_and_cross_pair_reuse_are_rejected() -> None:
    left_snapshot = _snapshot("session_a", 101, "a" * 64)
    right_snapshot = _snapshot("session_b", 101, "a" * 64)
    left = issue_synthetic_stage8_session_capability(lambda: left_snapshot)
    right = issue_synthetic_stage8_session_capability(lambda: right_snapshot)
    with pytest.raises(Stage9CapabilityError, match="not independent"):
        consume_stage9_session_pair(left, right, lambda first, second: (first, second))


def test_detached_snapshot_is_not_a_capability() -> None:
    snapshot = _snapshot("session_a", 101, "a" * 64)
    capability = issue_synthetic_stage8_session_capability(lambda: snapshot)
    with pytest.raises(Stage9CapabilityError, match="not a live"):
        consume_stage9_session_pair(snapshot, capability, lambda first, second: (first, second))
