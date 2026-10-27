from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest

from retinaentail.entailment.states import EntailmentState
from retinaentail.stability.stage8 import (
    STAGE8_BASELINE_RUN_ID,
    STAGE8_SCHEMA_VERSION,
    Stage8AuditStatus,
    Stage8CaseObservation,
    Stage8EventMetric,
    Stage8FailureReason,
    Stage8Preregistration,
    Stage8ReplaySpec,
    Stage8Scope,
    Stage8StabilityMetric,
    audit_stage8_repeatability,
)


def _hash(character: str) -> str:
    return character * 64


def _spec(run_id: str = "repeat-1", *, seed: int = 101) -> Stage8ReplaySpec:
    return Stage8ReplaySpec(
        run_id=run_id,
        seed=seed,
        degradation_replay_id="frozen_stage7_test_trace_set_v1",
        degradation_replay_set_sha256=_hash("a" if seed == 101 else "b"),
    )


def _preregistration(
    *,
    cases: tuple[str, ...] = ("case-a", "case-b"),
    specs: tuple[Stage8ReplaySpec, ...] | None = None,
) -> Stage8Preregistration:
    return Stage8Preregistration(
        dataset_id="toy-retina",
        dataset_version="1.0.0",
        scope=Stage8Scope.SYNTHETIC_ONLY,
        inclusion_case_ids=cases,
        replay_specs=specs or (_spec(), _spec("repeat-2", seed=202)),
        alpha=0.05,
        positive_threshold=0.4,
        negative_threshold=0.6,
        positive_sample_count=8,
        negative_sample_count=9,
        config_hash=_hash("1"),
        data_hash=_hash("2"),
        model_hash=_hash("3"),
        stage7_package_sha256=_hash("4"),
        provenance_sha256=_hash("5"),
    )


def _state_for(case_id: str) -> EntailmentState:
    return (
        EntailmentState.ENTAILED_PRESENT if case_id == "case-a" else EntailmentState.ENTAILED_ABSENT
    )


def _observation(
    case_id: str,
    *,
    spec: Stage8ReplaySpec | None,
    state: EntailmentState | str | object | None = None,
    **overrides: object,
) -> Stage8CaseObservation:
    chosen_state = _state_for(case_id) if state is None else state
    is_baseline = spec is None
    scores = {
        EntailmentState.ENTAILED_PRESENT: (0.2, 0.7),
        EntailmentState.ENTAILED_ABSENT: (0.7, 0.2),
        EntailmentState.RAW_UNIDENTIFIABLE: (0.2, 0.2),
        EntailmentState.OUT_OF_MODEL: (0.7, 0.7),
    }
    positive_score, negative_score = scores.get(chosen_state, (0.2, 0.7))
    values: dict[str, object] = {
        "case_id": case_id,
        "run_id": STAGE8_BASELINE_RUN_ID if is_baseline else spec.run_id,
        "dataset_id": "toy-retina",
        "dataset_version": "1.0.0",
        "split": "test",
        "scope": Stage8Scope.SYNTHETIC_ONLY,
        "seed": 0 if is_baseline else spec.seed,
        "degradation_replay_id": (
            "stage7_frozen_baseline" if is_baseline else spec.degradation_replay_id
        ),
        "degradation_replay_set_sha256": (
            _hash("c") if is_baseline else spec.degradation_replay_set_sha256
        ),
        "case_degradation_trace_sha256": (_hash("8") if case_id == "case-a" else _hash("9")),
        "state": chosen_state,
        "fail_closed": False,
        "out_of_distribution": False,
        "alpha": 0.05,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_threshold": 0.4,
        "negative_threshold": 0.6,
        "positive_sample_count": 8,
        "negative_sample_count": 9,
        "config_hash": _hash("1"),
        "data_hash": _hash("2"),
        "model_hash": _hash("3"),
        "stage7_package_sha256": _hash("4"),
        "provenance_sha256": _hash("5"),
        "degradation_known": True,
        "solver_succeeded": True,
        "positive_witness_sha256": (_hash("6") if case_id == "case-a" else _hash("a")),
        "negative_witness_sha256": (_hash("7") if case_id == "case-a" else _hash("b")),
        "positive_witness_replay_valid": True,
        "negative_witness_replay_valid": True,
    }
    values.update(overrides)
    return Stage8CaseObservation(**values)


def _valid_inputs(
    preregistration: Stage8Preregistration,
) -> tuple[list[Stage8CaseObservation], list[Stage8CaseObservation]]:
    baseline = [_observation(case_id, spec=None) for case_id in preregistration.inclusion_case_ids]
    repeats = [
        _observation(case_id, spec=spec)
        for case_id in preregistration.inclusion_case_ids
        for spec in preregistration.replay_specs
    ]
    return baseline, repeats


def test_exact_replays_cover_cartesian_set_and_all_sixteen_transitions() -> None:
    preregistration = _preregistration()
    baseline, repeats = _valid_inputs(preregistration)

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeats,
    )

    assert report.status is Stage8AuditStatus.SUCCESS
    assert report.expected_transition_count == 4
    assert len(report.transition_matrix) == 16
    assert sum(report.transition_matrix.values()) == 4
    assert report.transition_matrix["ENTAILED_PRESENT->ENTAILED_PRESENT"] == 2
    assert report.transition_matrix["ENTAILED_ABSENT->ENTAILED_ABSENT"] == 2
    for metric in Stage8StabilityMetric:
        interval = report.metric_intervals[metric]
        assert interval.numerator == interval.denominator == 4
        assert interval.estimate == 1.0
    assert report.event_intervals[Stage8EventMetric.REPEAT_FAIL_CLOSED].numerator == 0
    assert report.event_intervals[Stage8EventMetric.REPEAT_OUT_OF_DISTRIBUTION].numerator == 0
    assert report.event_intervals[Stage8EventMetric.REPEAT_OUT_OF_MODEL].numerator == 0
    assert (
        report.event_intervals[Stage8EventMetric.REPEAT_DUAL_WITNESS_REPLAY_PASSED].numerator == 4
    )
    payload = report.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    first_case = report.case_transitions[0]
    first_payload = first_case.to_dict()
    assert first_payload["baseline_positive_witness_sha256"] == _hash("6")
    assert first_payload["baseline_negative_witness_sha256"] == _hash("7")
    assert first_payload["repeat_positive_witness_sha256"] == _hash("6")
    assert first_payload["repeat_negative_witness_sha256"] == _hash("7")


def test_observed_state_and_witness_hash_instability_are_results_not_contract_drift() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline, repeats = _valid_inputs(preregistration)
    repeats[0] = _observation(
        "case-a",
        spec=preregistration.replay_specs[0],
        state=EntailmentState.ENTAILED_ABSENT,
        positive_witness_sha256=_hash("d"),
    )

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeats,
    )

    assert report.status is Stage8AuditStatus.SUCCESS
    transition = report.case_transitions[0]
    assert transition.transition == "ENTAILED_PRESENT->ENTAILED_ABSENT"
    assert not transition.state_stable
    assert not transition.positive_witness_hash_stable
    assert not transition.dual_witness_hash_stable
    assert transition.negative_witness_hash_stable
    assert not transition.failure_reasons


@pytest.mark.parametrize(
    ("field", "bad_value", "reason"),
    [
        ("dataset_id", "other", Stage8FailureReason.DATASET_ID_MISMATCH),
        ("dataset_version", "2", Stage8FailureReason.DATASET_VERSION_MISMATCH),
        ("split", "train", Stage8FailureReason.SPLIT_MISMATCH),
        ("scope", "APPROVED_REAL_DATA", Stage8FailureReason.SCOPE_MISMATCH),
        ("seed", 999, Stage8FailureReason.RUN_SEED_MISMATCH),
        (
            "degradation_replay_id",
            "other",
            Stage8FailureReason.DEGRADATION_REPLAY_ID_MISMATCH,
        ),
        (
            "degradation_replay_set_sha256",
            "f" * 64,
            Stage8FailureReason.DEGRADATION_REPLAY_SET_HASH_MISMATCH,
        ),
        ("alpha", 0.2, Stage8FailureReason.LOCKED_ALPHA_MISMATCH),
        (
            "positive_threshold",
            0.1,
            Stage8FailureReason.LOCKED_POSITIVE_THRESHOLD_MISMATCH,
        ),
        (
            "negative_threshold",
            0.1,
            Stage8FailureReason.LOCKED_NEGATIVE_THRESHOLD_MISMATCH,
        ),
        (
            "positive_sample_count",
            7,
            Stage8FailureReason.LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH,
        ),
        (
            "negative_sample_count",
            7,
            Stage8FailureReason.LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH,
        ),
        ("config_hash", "f" * 64, Stage8FailureReason.LOCKED_CONFIG_HASH_MISMATCH),
        ("data_hash", "f" * 64, Stage8FailureReason.LOCKED_DATA_HASH_MISMATCH),
        ("model_hash", "f" * 64, Stage8FailureReason.LOCKED_MODEL_HASH_MISMATCH),
        (
            "stage7_package_sha256",
            "f" * 64,
            Stage8FailureReason.STAGE7_PACKAGE_HASH_MISMATCH,
        ),
        ("provenance_sha256", "f" * 64, Stage8FailureReason.PROVENANCE_HASH_MISMATCH),
    ],
)
def test_locked_binding_drift_is_fail_closed_out_of_model(
    field: str,
    bad_value: object,
    reason: Stage8FailureReason,
) -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline, _ = _valid_inputs(preregistration)
    repeat = _observation(
        "case-a",
        spec=preregistration.replay_specs[0],
        **{field: bad_value},
    )

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=[repeat],
    )

    transition = report.case_transitions[0]
    assert report.status is Stage8AuditStatus.FAIL_CLOSED
    assert transition.repeat_state is EntailmentState.OUT_OF_MODEL
    assert transition.repeat_fail_closed
    assert reason in transition.failure_reasons
    assert all(interval.numerator == 0 for interval in report.metric_intervals.values())


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"positive_score": math.nan}, Stage8FailureReason.INVALID_NUMERIC_VALUE),
        ({"state": "UNKNOWN"}, Stage8FailureReason.INVALID_STATE),
        (
            {
                "state": EntailmentState.ENTAILED_ABSENT,
                "positive_score": 0.2,
                "negative_score": 0.7,
            },
            Stage8FailureReason.STATE_SCORE_MISMATCH,
        ),
        ({"fail_closed": 0}, Stage8FailureReason.INVALID_FLAG),
        ({"schema_version": "7.0.0"}, Stage8FailureReason.INVALID_SCHEMA_VERSION),
        (
            {"case_degradation_trace_sha256": "f" * 64},
            Stage8FailureReason.CASE_DEGRADATION_TRACE_MISMATCH,
        ),
    ],
)
def test_malformed_values_are_json_safe_and_never_retain_determinate_state(
    overrides: dict[str, object],
    expected_reason: Stage8FailureReason,
) -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline, _ = _valid_inputs(preregistration)
    repeat = _observation(
        "case-a",
        spec=preregistration.replay_specs[0],
        **overrides,
    )

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=[repeat],
    )

    assert report.case_transitions[0].repeat_state is EntailmentState.OUT_OF_MODEL
    assert expected_reason in report.case_transitions[0].failure_reasons
    json.dumps(report.to_dict(), allow_nan=False)


def test_unknown_degradation_solver_failure_and_invalid_witness_normalize_fail_closed() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline, _ = _valid_inputs(preregistration)
    repeat = _observation(
        "case-a",
        spec=preregistration.replay_specs[0],
        degradation_known=False,
        solver_succeeded=False,
        positive_witness_replay_valid=False,
    )

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=[repeat],
    )

    transition = report.case_transitions[0]
    assert transition.repeat_state is EntailmentState.OUT_OF_MODEL
    assert set(transition.failure_reasons) >= {
        Stage8FailureReason.UNKNOWN_DEGRADATION,
        Stage8FailureReason.SOLVER_FAILED,
        Stage8FailureReason.INVALID_WITNESS_REPLAY,
        Stage8FailureReason.FAIL_CLOSED_SEMANTICS_MISMATCH,
    }
    assert not transition.positive_witness_replay_stable


def test_declared_fail_closed_ood_is_stably_stratified_but_never_a_witness_pass() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    common: dict[str, object] = {
        "state": EntailmentState.OUT_OF_MODEL,
        "fail_closed": True,
        "out_of_distribution": True,
        "positive_witness_replay_valid": False,
    }
    baseline = [_observation("case-a", spec=None, **common)]
    repeats = [_observation("case-a", spec=preregistration.replay_specs[0], **common)]

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeats,
    )

    transition = report.case_transitions[0]
    assert transition.state_stable
    assert transition.fail_closed_status_stable
    assert transition.out_of_distribution_status_stable
    assert not transition.positive_witness_replay_stable
    assert transition.positive_witness_hash_stable
    assert transition.negative_witness_hash_stable
    assert transition.dual_witness_hash_stable
    assert report.metric_intervals[Stage8StabilityMetric.STATE_STABLE].estimate == 1.0
    assert report.status is Stage8AuditStatus.SUCCESS
    assert report.event_intervals[Stage8EventMetric.REPEAT_FAIL_CLOSED].estimate == 1.0
    assert report.event_intervals[Stage8EventMetric.REPEAT_OUT_OF_DISTRIBUTION].estimate == 1.0
    assert (
        report.event_intervals[Stage8EventMetric.REPEAT_DUAL_WITNESS_REPLAY_PASSED].estimate == 0.0
    )


def test_asymmetric_witness_replay_invalidates_hash_stability_even_when_hashes_match() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline = [_observation("case-a", spec=None)]
    repeat = [
        _observation(
            "case-a",
            spec=preregistration.replay_specs[0],
            state=EntailmentState.OUT_OF_MODEL,
            fail_closed=True,
            positive_witness_replay_valid=False,
        )
    ]

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeat,
    )

    transition = report.case_transitions[0]
    assert report.status is Stage8AuditStatus.SUCCESS
    assert not transition.positive_witness_hash_stable
    assert transition.negative_witness_hash_stable
    assert not transition.dual_witness_hash_stable
    assert not transition.positive_witness_replay_stable
    assert transition.negative_witness_replay_stable


def test_contract_drift_cannot_inflate_diagonal_out_of_model_stability() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline = [
        _observation(
            "case-a",
            spec=None,
            state=EntailmentState.OUT_OF_MODEL,
        )
    ]
    repeat = [
        _observation(
            "case-a",
            spec=preregistration.replay_specs[0],
            state=EntailmentState.OUT_OF_MODEL,
            provenance_sha256=_hash("f"),
        )
    ]

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeat,
    )

    transition = report.case_transitions[0]
    assert transition.transition == "OUT_OF_MODEL->OUT_OF_MODEL"
    assert not transition.state_stable
    assert not transition.fail_closed_status_stable
    assert not transition.out_of_distribution_status_stable
    assert all(interval.numerator == 0 for interval in report.metric_intervals.values())


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"seed": True}, Stage8FailureReason.INVALID_SEED),
        ({"seed": -1}, Stage8FailureReason.INVALID_SEED),
        ({"seed": 2**32}, Stage8FailureReason.INVALID_SEED),
        ({"positive_sample_count": True}, Stage8FailureReason.INVALID_SAMPLE_COUNT),
        ({"negative_sample_count": 0}, Stage8FailureReason.INVALID_SAMPLE_COUNT),
        ({"positive_sample_count": 2**31}, Stage8FailureReason.INVALID_SAMPLE_COUNT),
    ],
)
def test_untrusted_seed_and_sample_counts_require_exact_bounded_integers(
    overrides: dict[str, object],
    reason: Stage8FailureReason,
) -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline = [_observation("case-a", spec=None)]
    repeat = [
        _observation(
            "case-a",
            spec=preregistration.replay_specs[0],
            **overrides,
        )
    ]

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeat,
    )

    transition = report.case_transitions[0]
    assert transition.repeat_state is EntailmentState.OUT_OF_MODEL
    assert reason in transition.failure_reasons
    assert all(interval.numerator == 0 for interval in report.metric_intervals.values())


def test_missing_duplicate_and_replacement_records_cannot_change_the_frozen_denominator() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline = _observation("case-a", spec=None)
    replacement = _observation("replacement", spec=preregistration.replay_specs[0])

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=[baseline, baseline],
        repeat_records=[replacement],
    )

    assert report.expected_transition_count == 1
    assert report.status is Stage8AuditStatus.FAIL_CLOSED
    assert report.case_transitions[0].repeat_state is EntailmentState.OUT_OF_MODEL
    assert set(report.case_transitions[0].failure_reasons) >= {
        Stage8FailureReason.DUPLICATE_BASELINE_CASE,
        Stage8FailureReason.MISSING_REPEAT_OBSERVATION,
    }
    assert report.rejected_reason_counts[Stage8FailureReason.UNDECLARED_REPEAT_CASE] == 1
    assert all(interval.numerator == 0 for interval in report.metric_intervals.values())


def test_duplicate_repeat_run_is_not_silently_deduplicated() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline, repeats = _valid_inputs(preregistration)

    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=[repeats[0], repeats[0]],
    )

    assert report.case_transitions[0].repeat_state is EntailmentState.OUT_OF_MODEL
    assert (
        Stage8FailureReason.DUPLICATE_REPEAT_OBSERVATION
        in report.case_transitions[0].failure_reasons
    )


def test_preregistration_is_strict_versioned_and_immutable() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))

    with pytest.raises(FrozenInstanceError):
        preregistration.alpha = 0.5  # type: ignore[misc]
    with pytest.raises(ValueError, match="duplicates"):
        _preregistration(cases=("case-a", "case-a"), specs=(_spec(),))
    with pytest.raises(ValueError, match="complete, ordered"):
        Stage8Preregistration(
            dataset_id="toy-retina",
            dataset_version="1.0.0",
            scope=Stage8Scope.SYNTHETIC_ONLY,
            inclusion_case_ids=("case-a",),
            replay_specs=(_spec(),),
            alpha=0.05,
            positive_threshold=0.4,
            negative_threshold=0.6,
            positive_sample_count=8,
            negative_sample_count=9,
            config_hash=_hash("1"),
            data_hash=_hash("2"),
            model_hash=_hash("3"),
            stage7_package_sha256=_hash("4"),
            provenance_sha256=_hash("5"),
            stability_metrics=(Stage8StabilityMetric.STATE_STABLE,),
        )
    with pytest.raises(ValueError, match="schema version"):
        Stage8ReplaySpec(
            run_id="repeat",
            seed=1,
            degradation_replay_id="set",
            degradation_replay_set_sha256=_hash("a"),
            schema_version="7.0.0",
        )
    assert preregistration.schema_version == STAGE8_SCHEMA_VERSION


def test_report_mappings_are_read_only() -> None:
    preregistration = _preregistration(cases=("case-a",), specs=(_spec(),))
    baseline, repeats = _valid_inputs(preregistration)
    report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeats,
    )

    with pytest.raises(TypeError):
        report.transition_matrix["ENTAILED_PRESENT->ENTAILED_PRESENT"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        report.metric_intervals[Stage8StabilityMetric.STATE_STABLE] = report.metric_intervals[
            Stage8StabilityMetric.STATE_STABLE
        ]  # type: ignore[index]
