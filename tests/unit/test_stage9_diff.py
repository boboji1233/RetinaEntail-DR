from __future__ import annotations

from dataclasses import replace

import pytest

from retinaentail.entailment.states import EntailmentState
from retinaentail.reproducibility.stage9 import (
    Stage9DifferenceKind,
    Stage9DiffStatus,
    Stage9SessionSnapshot,
    compare_stage8_sessions,
    parse_stage8_observations,
)

DIGEST = "a" * 64


def _observation(*, score: float = 0.0) -> dict[str, object]:
    return {
        "schema_version": "8.0.0",
        "dataset_id": "synthetic_stage2",
        "dataset_version": "synthetic-fixture-v1",
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "case_id": "case-001",
        "run_id": "stage7_locked_baseline",
        "seed": 7,
        "degradation_replay_id": "frozen",
        "degradation_replay_set_sha256": DIGEST,
        "case_degradation_trace_sha256": "b" * 64,
        "alpha": 0.1,
        "positive_score": score,
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


def _payload(*, score: float = 0.0) -> dict[str, object]:
    baseline = _observation(score=score)
    repeat = {**baseline, "run_id": "repeat-001", "seed": 8}
    return {
        "schema_version": "8.0.0",
        "scope": "SYNTHETIC_FIXTURE",
        "source_stage7_package_sha256": "1" * 64,
        "exact_case_set_enforced": True,
        "sample_replacement_performed": False,
        "baseline_records": [baseline],
        "repeat_records": [repeat],
        "replay_receipts": [],
    }


def _snapshot(
    session_id: str,
    process_id: int,
    nonce: str,
    *,
    score: float = 0.0,
) -> Stage9SessionSnapshot:
    observations = parse_stage8_observations(_payload(score=score))
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


def test_identical_independent_sessions_are_exactly_reproducible() -> None:
    report = compare_stage8_sessions(
        _snapshot("session_a", 101, "a" * 64),
        _snapshot("session_b", 202, "b" * 64),
    )
    assert report.status is Stage9DiffStatus.SUCCESS
    assert len(report.case_comparisons) == 2
    assert not report.global_differences
    assert all(item.deterministic_record_match for item in report.case_comparisons)
    assert all(
        item.effective_state is EntailmentState.ENTAILED_PRESENT for item in report.case_comparisons
    )
    intervals = dict(report.metric_intervals)
    assert intervals["FULL_DETERMINISTIC_RECORD_EXACT_MATCH"]["numerator"] == 2
    assert intervals["FULL_DETERMINISTIC_RECORD_EXACT_MATCH"]["denominator"] == 2


def test_score_or_package_drift_fails_closed_to_out_of_model() -> None:
    left = _snapshot("session_a", 101, "a" * 64)
    right = _snapshot("session_b", 202, "b" * 64, score=0.01)
    right = replace(right, package_sha256="0" * 64, stage8_observations_sha256="f" * 64)
    report = compare_stage8_sessions(left, right)
    assert report.status is Stage9DiffStatus.FAIL_CLOSED
    assert Stage9DifferenceKind.PACKAGE_HASH in report.global_differences
    assert Stage9DifferenceKind.OBSERVATION_SET in report.global_differences
    assert all(
        item.effective_state is EntailmentState.OUT_OF_MODEL for item in report.case_comparisons
    )
    assert any(Stage9DifferenceKind.SCORE in item.differences for item in report.case_comparisons)


def test_same_process_or_session_nonce_is_not_independent() -> None:
    left = _snapshot("session_a", 101, "a" * 64)
    right = _snapshot("session_b", 101, "a" * 64)
    report = compare_stage8_sessions(left, right)
    assert report.status is Stage9DiffStatus.FAIL_CLOSED
    assert Stage9DifferenceKind.INVALID_INPUT in report.global_differences


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_score_is_rejected(bad: object) -> None:
    payload = _payload()
    baseline = payload["baseline_records"]
    assert isinstance(baseline, list)
    assert isinstance(baseline[0], dict)
    baseline[0]["positive_score"] = bad
    with pytest.raises(ValueError, match="positive_score"):
        parse_stage8_observations(payload)


def test_unknown_observation_field_is_rejected() -> None:
    payload = _payload()
    baseline = payload["baseline_records"]
    assert isinstance(baseline, list)
    assert isinstance(baseline[0], dict)
    baseline[0]["test_tuned"] = False
    with pytest.raises(ValueError, match="incomplete or unknown"):
        parse_stage8_observations(payload)
