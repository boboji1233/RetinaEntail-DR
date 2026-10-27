from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from retinaentail.calibration.protocol import (
    CALIBRATION_METHOD_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    CalibrationArtifact,
    CalibrationFailureReason,
    CalibrationLabel,
    CalibrationRunStatus,
    CalibrationSample,
    EvaluationFailureReason,
    EvaluationStatus,
    HypothesisEvaluation,
    SampleSplit,
    calibrate_order_statistic,
    evaluate_hypotheses,
)
from retinaentail.entailment.states import EntailmentState

CONFIG_HASH = "1" * 64
DATA_HASH = "2" * 64
MODEL_HASH = "3" * 64


def _sample(
    sample_id: str,
    label: CalibrationLabel | str,
    positive_score: float,
    negative_score: float,
    *,
    split: SampleSplit | str = SampleSplit.CALIBRATION,
    source_image_id: str | None = None,
    patient_id: str | None = None,
    intervention_pair_id: str | None = None,
) -> CalibrationSample:
    return CalibrationSample(
        sample_id=sample_id,
        split=split,
        label=label,
        positive_score=positive_score,
        negative_score=negative_score,
        source_image_id=source_image_id,
        patient_id=patient_id,
        intervention_pair_id=intervention_pair_id,
    )


def _samples() -> tuple[CalibrationSample, ...]:
    return (
        _sample("p-1", CalibrationLabel.LESION_PRESENT, 0.10, 0.91),
        _sample("p-2", CalibrationLabel.LESION_PRESENT, 0.30, 0.80),
        _sample("p-3", CalibrationLabel.LESION_PRESENT, 0.20, 0.75),
        _sample("n-1", CalibrationLabel.LESION_ABSENT, 0.85, 0.40),
        _sample("n-2", CalibrationLabel.LESION_ABSENT, 0.90, 0.60),
        _sample("n-3", CalibrationLabel.LESION_ABSENT, 0.79, 0.50),
    )


def _calibrate(
    samples: tuple[CalibrationSample, ...] | None = None,
    *,
    alpha: float = 0.25,
    config_hash: str = CONFIG_HASH,
    data_hash: str = DATA_HASH,
    model_hash: str = MODEL_HASH,
) -> CalibrationArtifact:
    return calibrate_order_statistic(
        _samples() if samples is None else samples,
        alpha=alpha,
        config_hash=config_hash,
        data_hash=data_hash,
        model_hash=model_hash,
    )


def _evaluation(
    positive_score: float | None,
    negative_score: float | None,
    **overrides: object,
) -> HypothesisEvaluation:
    values: dict[str, object] = {
        "case_id": "synthetic-case-1",
        "lesion_class": "toy_dot",
        "positive_score": positive_score,
        "negative_score": negative_score,
        "config_hash": CONFIG_HASH,
        "data_hash": DATA_HASH,
        "model_hash": MODEL_HASH,
    }
    values.update(overrides)
    return HypothesisEvaluation(**values)  # type: ignore[arg-type]


def test_conservative_order_statistic_records_rank_counts_and_provenance() -> None:
    artifact = _calibrate()

    assert artifact.status is CalibrationRunStatus.SUCCESS
    assert artifact.failure_reasons == ()
    assert artifact.positive_threshold == pytest.approx(0.30)
    assert artifact.negative_threshold == pytest.approx(0.60)
    assert artifact.positive_order_index == 3
    assert artifact.negative_order_index == 3
    assert artifact.positive_sample_count == 3
    assert artifact.negative_sample_count == 3
    assert artifact.total_sample_count == 6
    assert artifact.alpha == pytest.approx(0.25)
    assert artifact.config_hash == CONFIG_HASH
    assert artifact.data_hash == DATA_HASH
    assert artifact.model_hash == MODEL_HASH
    assert artifact.schema_version == CALIBRATION_SCHEMA_VERSION
    assert artifact.method_version == CALIBRATION_METHOD_VERSION


def test_order_statistic_is_deterministic_and_input_order_independent() -> None:
    forward = _calibrate(_samples(), alpha=0.5)
    reverse = _calibrate(tuple(reversed(_samples())), alpha=0.5)

    assert forward == reverse
    assert forward.positive_threshold == pytest.approx(0.20)
    assert forward.negative_threshold == pytest.approx(0.50)
    assert forward.positive_order_index == 2
    assert forward.negative_order_index == 2


def test_too_few_samples_for_requested_alpha_fails_instead_of_clipping_rank() -> None:
    samples = (
        _sample("p", CalibrationLabel.LESION_PRESENT, 0.12, 0.8),
        _sample("n", CalibrationLabel.LESION_ABSENT, 0.9, 0.34),
    )

    artifact = _calibrate(samples, alpha=0.01)

    assert artifact.status is CalibrationRunStatus.FAIL_CLOSED
    assert artifact.positive_threshold is None
    assert artifact.negative_threshold is None
    assert CalibrationFailureReason.INSUFFICIENT_POSITIVE_SAMPLES in artifact.failure_reasons
    assert CalibrationFailureReason.INSUFFICIENT_NEGATIVE_SAMPLES in artifact.failure_reasons


@pytest.mark.parametrize(
    ("positive_score", "negative_score", "expected"),
    [
        (0.20, 0.70, EntailmentState.ENTAILED_PRESENT),
        (0.40, 0.50, EntailmentState.ENTAILED_ABSENT),
        (0.20, 0.50, EntailmentState.RAW_UNIDENTIFIABLE),
        (0.40, 0.70, EntailmentState.OUT_OF_MODEL),
    ],
)
def test_evaluation_covers_complete_four_state_truth_table(
    positive_score: float,
    negative_score: float,
    expected: EntailmentState,
) -> None:
    result = evaluate_hypotheses(_evaluation(positive_score, negative_score), _calibrate())

    assert result.state is expected
    assert result.status is EvaluationStatus.SUCCESS
    assert result.failure_reasons == ()
    assert result.positive_feasible is (positive_score <= 0.30)
    assert result.negative_feasible is (negative_score <= 0.60)


@pytest.mark.parametrize(
    ("samples", "expected_reason"),
    [
        ((), CalibrationFailureReason.EMPTY_CALIBRATION_SET),
        (
            (
                _sample("p", CalibrationLabel.UNKNOWN, 0.1, 0.9),
                _sample("n", CalibrationLabel.LESION_ABSENT, 0.9, 0.1),
            ),
            CalibrationFailureReason.UNKNOWN_LABEL,
        ),
        (
            (
                _sample("p", CalibrationLabel.LESION_PRESENT, 0.1, 0.9),
                _sample(
                    "n",
                    CalibrationLabel.LESION_ABSENT,
                    0.9,
                    0.1,
                    split=SampleSplit.TEST,
                ),
            ),
            CalibrationFailureReason.NON_CALIBRATION_SPLIT,
        ),
        (
            (
                _sample("p", CalibrationLabel.LESION_PRESENT, math.nan, 0.9),
                _sample("n", CalibrationLabel.LESION_ABSENT, 0.9, 0.1),
            ),
            CalibrationFailureReason.NONFINITE_SCORE,
        ),
        (
            (
                _sample("p", CalibrationLabel.LESION_PRESENT, 0.1, 0.9),
                _sample("p-2", CalibrationLabel.LESION_PRESENT, 0.2, 0.8),
            ),
            CalibrationFailureReason.MISSING_NEGATIVE_SAMPLES,
        ),
    ],
)
def test_invalid_calibration_batches_issue_no_partial_thresholds(
    samples: tuple[CalibrationSample, ...],
    expected_reason: CalibrationFailureReason,
) -> None:
    artifact = _calibrate(samples)

    assert artifact.status is CalibrationRunStatus.FAIL_CLOSED
    assert expected_reason in artifact.failure_reasons
    assert artifact.positive_threshold is None
    assert artifact.negative_threshold is None
    assert artifact.positive_order_index is None
    assert artifact.negative_order_index is None


def test_cross_split_identity_is_detected_before_threshold_issuance() -> None:
    samples = (
        _sample(
            "p",
            CalibrationLabel.LESION_PRESENT,
            0.1,
            0.9,
            source_image_id="source-1",
        ),
        _sample(
            "n",
            CalibrationLabel.LESION_ABSENT,
            0.9,
            0.1,
            split=SampleSplit.TEST,
            source_image_id="source-1",
        ),
    )

    artifact = _calibrate(samples)

    assert CalibrationFailureReason.IDENTITY_LEAKAGE in artifact.failure_reasons
    assert CalibrationFailureReason.NON_CALIBRATION_SPLIT in artifact.failure_reasons
    assert artifact.status is CalibrationRunStatus.FAIL_CLOSED


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, math.nan, math.inf])
def test_invalid_alpha_fails_closed(alpha: float) -> None:
    artifact = _calibrate(alpha=alpha)

    assert artifact.status is CalibrationRunStatus.FAIL_CLOSED
    assert CalibrationFailureReason.INVALID_ALPHA in artifact.failure_reasons
    assert artifact.positive_threshold is None
    assert artifact.negative_threshold is None


@pytest.mark.parametrize("bad_hash", ["", "A" * 64, "0" * 63, "g" * 64])
def test_invalid_calibration_hash_fails_closed(bad_hash: str) -> None:
    artifact = _calibrate(config_hash=bad_hash)

    assert artifact.status is CalibrationRunStatus.FAIL_CLOSED
    assert CalibrationFailureReason.INVALID_HASH in artifact.failure_reasons
    assert artifact.positive_threshold is None
    assert artifact.negative_threshold is None
    assert artifact.config_hash is None


def test_calibration_artifact_cannot_be_constructed_or_cloned_for_threshold_use() -> None:
    with pytest.raises(TypeError, match="can only be issued"):
        CalibrationArtifact(
            alpha=0.25,
            positive_threshold=999.0,
            negative_threshold=999.0,
            positive_sample_count=3,
            negative_sample_count=3,
            total_sample_count=6,
            positive_order_index=3,
            negative_order_index=3,
            config_hash=CONFIG_HASH,
            data_hash=DATA_HASH,
            model_hash=MODEL_HASH,
            status=CalibrationRunStatus.SUCCESS,
        )

    cloned = replace(_calibrate(), positive_threshold=999.0)
    result = evaluate_hypotheses(_evaluation(0.2, 0.7), cloned)

    assert result.status is EvaluationStatus.FAIL_CLOSED
    assert result.state is EntailmentState.OUT_OF_MODEL
    assert EvaluationFailureReason.INVALID_CALIBRATION in result.failure_reasons


@pytest.mark.parametrize(
    ("evaluation", "artifact_transform", "expected_reason"),
    [
        (_evaluation(None, 0.5), lambda value: value, EvaluationFailureReason.MISSING_SCORE),
        (
            _evaluation(math.nan, 0.5),
            lambda value: value,
            EvaluationFailureReason.NONFINITE_SCORE,
        ),
        (
            _evaluation(0.2, 0.5, degradation_known=False),
            lambda value: value,
            EvaluationFailureReason.UNKNOWN_DEGRADATION,
        ),
        (
            _evaluation(0.2, 0.5, negative_witness_replay_valid=False),
            lambda value: value,
            EvaluationFailureReason.INVALID_WITNESS,
        ),
        (
            _evaluation(0.2, 0.5, positive_solver_status="timeout"),
            lambda value: value,
            EvaluationFailureReason.SOLVER_NOT_SUCCESSFUL,
        ),
        (
            _evaluation(0.2, 0.5, out_of_distribution=True),
            lambda value: value,
            EvaluationFailureReason.OUT_OF_DISTRIBUTION,
        ),
        (
            _evaluation(0.2, 0.5, config_hash="not-a-hash"),
            lambda value: value,
            EvaluationFailureReason.INVALID_HASH,
        ),
        (
            _evaluation(0.2, 0.5, data_hash="4" * 64),
            lambda value: value,
            EvaluationFailureReason.PROVENANCE_MISMATCH,
        ),
        (
            _evaluation(0.2, 0.5),
            lambda value: replace(value, positive_threshold=None),
            EvaluationFailureReason.MISSING_THRESHOLD,
        ),
        (
            _evaluation(0.2, 0.5),
            lambda value: replace(value, negative_threshold=math.inf),
            EvaluationFailureReason.NONFINITE_THRESHOLD,
        ),
        (
            _evaluation(0.2, 0.5),
            lambda value: replace(value, method_version="unknown"),
            EvaluationFailureReason.INVALID_CALIBRATION,
        ),
    ],
)
def test_evaluation_prerequisite_failures_are_out_of_model_and_retained(
    evaluation: HypothesisEvaluation,
    artifact_transform: object,
    expected_reason: EvaluationFailureReason,
) -> None:
    transform = artifact_transform
    assert callable(transform)
    artifact = transform(_calibrate())
    assert isinstance(artifact, CalibrationArtifact)

    result = evaluate_hypotheses(evaluation, artifact)

    assert result.state is EntailmentState.OUT_OF_MODEL
    assert result.status is EvaluationStatus.FAIL_CLOSED
    assert expected_reason in result.failure_reasons
    assert result.positive_feasible is None
    assert result.negative_feasible is None


def test_evaluation_retains_multiple_simultaneous_failure_reasons() -> None:
    evaluation = _evaluation(
        math.nan,
        math.inf,
        degradation_known=False,
        positive_witness_replay_valid=False,
        solver_status="failed",
        out_of_distribution=True,
    )

    result = evaluate_hypotheses(evaluation, _calibrate())

    assert result.state is EntailmentState.OUT_OF_MODEL
    assert set(result.failure_reasons) >= {
        EvaluationFailureReason.NONFINITE_SCORE,
        EvaluationFailureReason.UNKNOWN_DEGRADATION,
        EvaluationFailureReason.INVALID_WITNESS,
        EvaluationFailureReason.SOLVER_NOT_SUCCESSFUL,
        EvaluationFailureReason.OUT_OF_DISTRIBUTION,
    }


def test_artifact_and_decision_dicts_are_traceable_and_json_compatible() -> None:
    artifact = _calibrate()
    decision = evaluate_hypotheses(_evaluation(0.2, 0.7), artifact)

    artifact_payload = artifact.to_dict()
    decision_payload = decision.to_dict()

    for payload in (artifact_payload, decision_payload):
        assert payload["schema_version"] == CALIBRATION_SCHEMA_VERSION
        assert payload["method_version"] == CALIBRATION_METHOD_VERSION
        assert payload["alpha"] == pytest.approx(0.25)
        assert payload["positive_sample_count"] == 3
        assert payload["negative_sample_count"] == 3
        assert payload["config_hash"] == CONFIG_HASH
        assert payload["data_hash"] == DATA_HASH
        assert payload["model_hash"] == MODEL_HASH

    assert decision_payload["degradation_known"] is True
    assert decision_payload["positive_witness_replay_valid"] is True
    assert decision_payload["negative_witness_replay_valid"] is True
    assert decision_payload["out_of_distribution"] is False
    assert decision_payload["solver_status"] == "success"
    assert decision_payload["input_config_hash"] == CONFIG_HASH
    assert decision_payload["input_data_hash"] == DATA_HASH
    assert decision_payload["input_model_hash"] == MODEL_HASH


def test_failed_evaluation_safely_retains_input_provenance() -> None:
    artifact = _calibrate()
    mismatched_data_hash = "4" * 64
    mismatch = evaluate_hypotheses(
        _evaluation(0.2, 0.5, data_hash=mismatched_data_hash),
        artifact,
    )
    invalid = evaluate_hypotheses(
        _evaluation(0.2, 0.5, config_hash="not-a-hash"),
        artifact,
    )

    assert mismatch.status is EvaluationStatus.FAIL_CLOSED
    assert mismatch.data_hash == DATA_HASH
    assert mismatch.input_data_hash == mismatched_data_hash
    assert EvaluationFailureReason.PROVENANCE_MISMATCH in mismatch.failure_reasons
    assert invalid.status is EvaluationStatus.FAIL_CLOSED
    assert invalid.config_hash == CONFIG_HASH
    assert invalid.input_config_hash is None
    assert EvaluationFailureReason.INVALID_HASH in invalid.failure_reasons


def test_fail_closed_evaluation_is_json_safe_for_malformed_runtime_values() -> None:
    evaluation = _evaluation(
        math.nan,
        -0.5,
        case_id=7,
        lesion_class=False,
        positive_solver_status=object(),
        degradation_known="yes",
        positive_witness_replay_valid=1,
        out_of_distribution="unknown",
    )
    artifact = replace(
        _calibrate(),
        alpha=math.inf,
        positive_threshold=math.inf,
        positive_sample_count=True,
    )

    result = evaluate_hypotheses(evaluation, artifact)
    payload = result.to_dict()

    assert result.status is EvaluationStatus.FAIL_CLOSED
    assert result.state is EntailmentState.OUT_OF_MODEL
    assert EvaluationFailureReason.INVALID_INPUT in result.failure_reasons
    assert payload["alpha"] is None
    assert payload["positive_score"] is None
    assert payload["negative_score"] is None
    assert payload["positive_threshold"] is None
    assert payload["positive_sample_count"] == 0
    assert payload["positive_solver_status"] is None
    assert payload["degradation_known"] is False
    assert payload["positive_witness_replay_valid"] is False
    assert payload["out_of_distribution"] is False
    json.dumps(payload, allow_nan=False)
