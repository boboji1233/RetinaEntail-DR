from __future__ import annotations

from dataclasses import replace

from retinaentail.entailment.states import EntailmentState
from retinaentail.reproducibility.stage9 import Stage9Observation
from retinaentail.reproducibility.stage10 import (
    Stage10DifferenceKind,
    Stage10DiffStatus,
    Stage10RunnerSnapshot,
    compare_stage10_runners,
)


def _observation(*, score: float = 0.0) -> Stage9Observation:
    return Stage9Observation(
        case_id="case-1",
        run_id="locked-run",
        state=EntailmentState.ENTAILED_PRESENT,
        positive_score=score,
        negative_score=0.5,
        fail_closed=False,
        out_of_distribution=False,
        positive_witness_sha256="1" * 64,
        negative_witness_sha256="2" * 64,
        positive_witness_replay_valid=True,
        negative_witness_replay_valid=True,
        canonical_sha256=("3" if score == 0.0 else "4") * 64,
    )


def _runner(runner_id: str, pid: int, nonce: str, *, score: float = 0.0) -> Stage10RunnerSnapshot:
    return Stage10RunnerSnapshot(
        runner_id=runner_id,
        producer_process_id=pid,
        session_nonce_sha256=nonce * 64,
        environment_lock_sha256="a" * 64,
        host_profile_sha256="b" * 64,
        dataset_id="dataset",
        dataset_version="v1",
        scope="SYNTHETIC_ONLY",
        split="test",
        source_stage9_package_sha256="c" * 64,
        stage9_package_sha256=("d" if runner_id == "runner-a" else "e") * 64,
        stage9_package_file_count=10,
        stage9_manifest_sha256=("5" if runner_id == "runner-a" else "6") * 64,
        stage9_audit_sha256=("7" if runner_id == "runner-a" else "8") * 64,
        stage9_preregistration_sha256="9" * 64,
        source_stage7_package_sha256="0" * 64,
        source_stage8_package_sha256="f" * 64,
        deterministic_projection_sha256="1" * 64,
        envelope_sha256=("2" if runner_id == "runner-a" else "3") * 64,
        signature_verified=True,
        observations=(_observation(score=score),),
    )


def test_exact_runner_comparison_allows_only_preregistered_provenance_differences() -> None:
    report = compare_stage10_runners(
        _runner("runner-a", 101, "a"),
        _runner("runner-b", 202, "b"),
        deterministic_file_mismatch_count=0,
        allowed_provenance_difference_count=5,
    )
    assert report.status is Stage10DiffStatus.SUCCESS
    assert report.global_differences == ()
    assert report.case_comparisons[0].effective_state is EntailmentState.ENTAILED_PRESENT
    assert report.allowed_provenance_difference_count == 5
    assert dict(report.metric_intervals)["score_exact_match"]["numerator"] == 1


def test_any_deterministic_file_or_score_drift_fails_closed() -> None:
    left = _runner("runner-a", 101, "a")
    right = _runner("runner-b", 202, "b", score=0.25)
    report = compare_stage10_runners(
        left,
        right,
        deterministic_file_mismatch_count=1,
        allowed_provenance_difference_count=4,
    )
    assert report.status is Stage10DiffStatus.FAIL_CLOSED
    assert Stage10DifferenceKind.DETERMINISTIC_PACKAGE_CONTENT in report.global_differences
    comparison = report.case_comparisons[0]
    assert comparison.effective_state is EntailmentState.OUT_OF_MODEL
    assert Stage10DifferenceKind.SCORE in comparison.differences


def test_runner_identity_or_environment_reuse_fails_closed() -> None:
    left = _runner("runner-a", 101, "a")
    right = replace(_runner("runner-b", 202, "b"), producer_process_id=101)
    report = compare_stage10_runners(
        left,
        right,
        deterministic_file_mismatch_count=0,
        allowed_provenance_difference_count=5,
    )
    assert report.status is Stage10DiffStatus.FAIL_CLOSED
    assert Stage10DifferenceKind.RUNNER_INDEPENDENCE in report.global_differences


def test_boolean_or_negative_file_counts_are_rejected() -> None:
    left = _runner("runner-a", 101, "a")
    right = _runner("runner-b", 202, "b")
    for bad in (True, -1):
        try:
            compare_stage10_runners(
                left,
                right,
                deterministic_file_mismatch_count=bad,
                allowed_provenance_difference_count=0,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid deterministic file count was accepted")
