from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from retinaentail.calibration.certificate import (
    Stage3SyntheticCertificate,
    Stage3WitnessEvidence,
    WitnessSide,
)
from retinaentail.calibration.protocol import (
    CalibrationLabel,
    EvaluationFailureReason,
    EvaluationResult,
    EvaluationStatus,
    ProtocolScope,
    SampleSplit,
)
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import (
    BinomialProportion,
    LockedStage4Contract,
    Stage4AggregationFailureReason,
    Stage4CaseRecord,
    Stage4EvaluationScope,
    Stage4ReportStatus,
    aggregate_locked_test_evaluations,
    wilson_score_proportion,
)
from retinaentail.schemas import DegradationTrace, LesionRegion, WitnessRecord
from retinaentail.witnesses.replay import canonical_json_bytes, sha256_bytes
from retinaentail.witnesses.stage3_replay import (
    PairReplayFailureReason,
    ReplayFailureReason,
    Stage3DualReplayResult,
    Stage3ReplayResult,
)

CONFIG_HASH = "1" * 64
DATA_HASH = "2" * 64
MODEL_HASH = "3" * 64
ARTIFACT_HASH = "4" * 64


def _lock() -> LockedStage4Contract:
    return LockedStage4Contract(
        alpha=0.25,
        positive_threshold=0.5,
        negative_threshold=0.5,
        positive_sample_count=4,
        negative_sample_count=4,
        total_sample_count=8,
        calibration_artifact_hash=ARTIFACT_HASH,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        model_hash=MODEL_HASH,
    )


def _success_evaluation(case_id: str, state: EntailmentState) -> EvaluationResult:
    scores = {
        EntailmentState.ENTAILED_PRESENT: (0.1, 0.9, True, False),
        EntailmentState.ENTAILED_ABSENT: (0.9, 0.1, False, True),
        EntailmentState.RAW_UNIDENTIFIABLE: (0.1, 0.1, True, True),
        EntailmentState.OUT_OF_MODEL: (0.9, 0.9, False, False),
    }
    positive_score, negative_score, positive_feasible, negative_feasible = scores[state]
    return EvaluationResult(
        case_id=case_id,
        lesion_class="microaneurysm",
        alpha=0.25,
        positive_score=positive_score,
        negative_score=negative_score,
        positive_threshold=0.5,
        negative_threshold=0.5,
        positive_feasible=positive_feasible,
        negative_feasible=negative_feasible,
        positive_sample_count=4,
        negative_sample_count=4,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        model_hash=MODEL_HASH,
        input_config_hash=CONFIG_HASH,
        input_data_hash=DATA_HASH,
        input_model_hash=MODEL_HASH,
        positive_solver_status="success",
        negative_solver_status="success",
        solver_status="success",
        degradation_known=True,
        positive_witness_replay_valid=True,
        negative_witness_replay_valid=True,
        out_of_distribution=False,
        state=state,
        status=EvaluationStatus.SUCCESS,
        protocol_scope=ProtocolScope.TOY_PROTOCOL,
    )


def _fail_closed_evaluation(case_id: str) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        lesion_class="microaneurysm",
        alpha=0.25,
        positive_score=0.1,
        negative_score=0.1,
        positive_threshold=0.5,
        negative_threshold=0.5,
        positive_feasible=None,
        negative_feasible=None,
        positive_sample_count=4,
        negative_sample_count=4,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        model_hash=MODEL_HASH,
        input_config_hash=CONFIG_HASH,
        input_data_hash=DATA_HASH,
        input_model_hash=MODEL_HASH,
        positive_solver_status="success",
        negative_solver_status="success",
        solver_status="success",
        degradation_known=True,
        positive_witness_replay_valid=True,
        negative_witness_replay_valid=False,
        out_of_distribution=True,
        state=EntailmentState.OUT_OF_MODEL,
        status=EvaluationStatus.FAIL_CLOSED,
        failure_reasons=(
            EvaluationFailureReason.INVALID_WITNESS,
            EvaluationFailureReason.OUT_OF_DISTRIBUTION,
        ),
        protocol_scope=ProtocolScope.TOY_PROTOCOL,
    )


def _record(
    case_id: str,
    state: EntailmentState,
    label: CalibrationLabel,
) -> Stage4CaseRecord:
    evaluation = _success_evaluation(case_id, state)
    certificate, dual_replay = _evidence(evaluation)
    return Stage4CaseRecord(
        case_id=case_id,
        split=SampleSplit.TEST,
        reference_label=label,
        evaluation=evaluation,
        certificate=certificate,
        dual_replay=dual_replay,
    )


def _evidence(
    evaluation: EvaluationResult,
    *,
    artifact_hash: str = ARTIFACT_HASH,
) -> tuple[Stage3SyntheticCertificate, Stage3DualReplayResult]:
    region = LesionRegion(x=1, y=1, width=2, height=2, region_id="region")
    trace = DegradationTrace(
        operations=({"name": "identity", "severity": 0.0, "seed": 0, "parameters": {}},)
    )
    region_hash = sha256_bytes(canonical_json_bytes(region.to_dict()))
    trace_hash = sha256_bytes(canonical_json_bytes(trace.to_dict()))
    observation_hash = "5" * 64

    def side_evidence(
        side: str, replay_error: float, passed: bool
    ) -> tuple[WitnessRecord, Stage3ReplayResult]:
        latent_hash = "6" * 64 if side == "positive" else "7" * 64
        witness_id = f"{evaluation.case_id}:{side}"
        predicate_valid = passed
        witness = WitnessRecord(
            witness_id=witness_id,
            case_id=evaluation.case_id,
            lesion_class=evaluation.lesion_class,
            witness_kind=("lesion_present" if side == "positive" else "lesion_absent"),
            relative_path=f"{side}.png",
            observation_path="observation.png",
            sha256=latent_hash,
            degradation_trace=trace,
            replay_seed=0,
            replay_error=replay_error,
            predicate_valid=predicate_valid,
        )
        replay = Stage3ReplayResult(
            witness_id=witness_id,
            case_id=evaluation.case_id,
            lesion_class=evaluation.lesion_class,
            side=side,
            predicate_id="test_predicate",
            region_sha256=region_hash,
            expected_latent_sha256=latent_hash,
            actual_latent_sha256=latent_hash,
            expected_observation_sha256=observation_hash,
            actual_observation_sha256=observation_hash,
            expected_trace_sha256=trace_hash,
            actual_trace_sha256=trace_hash,
            replay_error=replay_error,
            predicate_valid=predicate_valid,
            passed=passed,
            failure_reasons=(() if passed else (ReplayFailureReason.PREDICATE_MISMATCH,)),
        )
        return witness, replay

    assert evaluation.positive_score is not None
    assert evaluation.negative_score is not None
    positive_witness, positive = side_evidence(
        "positive",
        evaluation.positive_score,
        evaluation.positive_witness_replay_valid,
    )
    negative_witness, negative = side_evidence(
        "negative",
        evaluation.negative_score,
        evaluation.negative_witness_replay_valid,
    )
    pair_reasons: list[PairReplayFailureReason] = []
    if not positive.passed:
        pair_reasons.append(PairReplayFailureReason.POSITIVE_REPLAY_FAILED)
    if not negative.passed:
        pair_reasons.append(PairReplayFailureReason.NEGATIVE_REPLAY_FAILED)
    dual = Stage3DualReplayResult(
        positive=positive,
        negative=negative,
        passed=positive.passed and negative.passed,
        failure_reasons=tuple(pair_reasons),
    )
    certificate = Stage3SyntheticCertificate(
        evaluation=evaluation,
        region=region,
        predicate_id="test_predicate",
        calibration_artifact_hash=artifact_hash,
        observation_sha256=observation_hash,
        degradation_trace_sha256=trace_hash,
        positive_evidence=Stage3WitnessEvidence(
            side=WitnessSide.POSITIVE,
            witness=positive_witness,
            replay=positive,
        ),
        negative_evidence=Stage3WitnessEvidence(
            side=WitnessSide.NEGATIVE,
            witness=negative_witness,
            replay=negative,
        ),
    )
    return certificate, dual


def _case_from_evaluation(
    evaluation: EvaluationResult,
    *,
    label: CalibrationLabel | str,
    split: SampleSplit | str = SampleSplit.TEST,
) -> Stage4CaseRecord:
    certificate, dual = _evidence(evaluation)
    return Stage4CaseRecord(
        case_id=evaluation.case_id,
        split=split,
        reference_label=label,
        evaluation=evaluation,
        certificate=certificate,
        dual_replay=dual,
    )


def test_four_states_are_counted_over_one_explicit_test_denominator() -> None:
    records = tuple(
        _record(f"case-{index}", state, label)
        for index, (state, label) in enumerate(
            zip(
                tuple(EntailmentState),
                (
                    CalibrationLabel.LESION_PRESENT,
                    CalibrationLabel.LESION_ABSENT,
                    CalibrationLabel.UNKNOWN,
                    CalibrationLabel.LESION_PRESENT,
                ),
                strict=True,
            ),
            start=1,
        )
    )
    report = aggregate_locked_test_evaluations(
        records,
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )

    assert report.status is Stage4ReportStatus.SUCCESS
    assert report.test_case_count == 4
    assert report.evaluable_count == 4
    assert report.fail_closed_count == 0
    assert report.labeled_reference_count == 3
    assert report.reference_label_counts[CalibrationLabel.UNKNOWN] == 1
    for state in EntailmentState:
        assert report.state_counts[state] == 1
        interval = report.state_proportions[state]
        assert interval.numerator == 1
        assert interval.denominator == 4
        assert interval.estimate == 0.25
        assert interval.lower is not None and interval.lower < 0.25
        assert interval.upper is not None and interval.upper > 0.25
    json.dumps(report.to_dict(), allow_nan=False)


def test_valid_fail_closed_case_retains_ood_and_witness_failure_strata() -> None:
    evaluation = _fail_closed_evaluation("ood-case")
    report = aggregate_locked_test_evaluations(
        (_case_from_evaluation(evaluation, label="UNKNOWN"),),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )

    # The report itself aggregated a valid fail-closed case; SUCCESS does not mean
    # the case was evaluable.
    assert report.status is Stage4ReportStatus.SUCCESS
    assert report.evaluable_count == 0
    assert report.unevaluable_count == report.fail_closed_count == 1
    assert report.out_of_distribution_count == 1
    assert report.one_or_more_witness_replays_failed_count == 1
    assert report.state_counts[EntailmentState.OUT_OF_MODEL] == 1
    assert report.failure_reason_counts == {
        "INVALID_WITNESS": 1,
        "OUT_OF_DISTRIBUTION": 1,
    }


def test_non_test_and_duplicate_records_never_enter_test_denominators() -> None:
    records = (
        _record(
            "duplicate",
            EntailmentState.ENTAILED_PRESENT,
            CalibrationLabel.LESION_PRESENT,
        ),
        _record(
            "duplicate",
            EntailmentState.ENTAILED_PRESENT,
            CalibrationLabel.LESION_PRESENT,
        ),
        _case_from_evaluation(
            _success_evaluation("train-case", EntailmentState.ENTAILED_ABSENT),
            split=SampleSplit.TRAIN,
            label=CalibrationLabel.LESION_ABSENT,
        ),
        _record(
            "valid",
            EntailmentState.RAW_UNIDENTIFIABLE,
            CalibrationLabel.UNKNOWN,
        ),
    )
    report = aggregate_locked_test_evaluations(
        records,
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )

    assert report.status is Stage4ReportStatus.FAIL_CLOSED
    assert report.input_record_count == 4
    assert report.test_case_count == 1
    assert report.rejected_record_count == 3
    assert report.rejected_reason_counts == {
        "DUPLICATE_CASE_ID": 2,
        "NON_TEST_SPLIT": 1,
    }
    assert report.state_counts[EntailmentState.RAW_UNIDENTIFIABLE] == 1


def test_unknown_reference_is_not_negative_and_invalid_label_fails_closed() -> None:
    unknown = _record(
        "unknown",
        EntailmentState.ENTAILED_PRESENT,
        CalibrationLabel.UNKNOWN,
    )
    invalid = _case_from_evaluation(
        _success_evaluation("invalid-label", EntailmentState.ENTAILED_ABSENT),
        label="",
    )
    report = aggregate_locked_test_evaluations(
        (unknown, invalid),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )

    assert report.reference_label_counts[CalibrationLabel.LESION_ABSENT] == 0
    assert report.reference_label_counts[CalibrationLabel.UNKNOWN] == 2
    assert report.labeled_reference_count == 0
    assert report.state_counts[EntailmentState.ENTAILED_PRESENT] == 1
    assert report.state_counts[EntailmentState.OUT_OF_MODEL] == 1
    assert report.failure_reason_counts["INVALID_REFERENCE_LABEL"] == 1


@pytest.mark.parametrize(
    ("field_name", "changed_value", "reason"),
    (
        (
            "positive_threshold",
            0.6,
            Stage4AggregationFailureReason.LOCKED_POSITIVE_THRESHOLD_MISMATCH,
        ),
        (
            "negative_threshold",
            0.6,
            Stage4AggregationFailureReason.LOCKED_NEGATIVE_THRESHOLD_MISMATCH,
        ),
        (
            "positive_sample_count",
            5,
            Stage4AggregationFailureReason.LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH,
        ),
        (
            "negative_sample_count",
            5,
            Stage4AggregationFailureReason.LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH,
        ),
    ),
)
def test_threshold_and_sample_count_drift_fail_closed(
    field_name: str,
    changed_value: float | int,
    reason: Stage4AggregationFailureReason,
) -> None:
    evaluation = _success_evaluation("drift", EntailmentState.RAW_UNIDENTIFIABLE)
    object.__setattr__(evaluation, field_name, changed_value)
    # Keep the Stage 3 record internally consistent where the changed field is a threshold.
    if field_name == "positive_threshold":
        object.__setattr__(evaluation, "positive_feasible", True)
    if field_name == "negative_threshold":
        object.__setattr__(evaluation, "negative_feasible", True)
    certificate, dual = _evidence(evaluation)
    report = aggregate_locked_test_evaluations(
        (
            Stage4CaseRecord(
                case_id="drift",
                split="test",
                reference_label="UNKNOWN",
                evaluation=evaluation,
                certificate=certificate,
                dual_replay=dual,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )

    assert report.status is Stage4ReportStatus.FAIL_CLOSED
    assert report.evaluable_count == 0
    assert report.state_counts[EntailmentState.OUT_OF_MODEL] == 1
    assert report.failure_reason_counts[reason.value] == 1


def test_nonfinite_tampering_is_json_safe_and_fail_closed() -> None:
    evaluation = _success_evaluation("nan-case", EntailmentState.ENTAILED_PRESENT)
    certificate, dual = _evidence(evaluation)
    object.__setattr__(evaluation, "positive_score", math.nan)
    report = aggregate_locked_test_evaluations(
        (
            Stage4CaseRecord(
                case_id="nan-case",
                split="test",
                reference_label="LESION_PRESENT",
                evaluation=evaluation,
                certificate=certificate,
                dual_replay=dual,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )

    assert report.status is Stage4ReportStatus.FAIL_CLOSED
    assert report.failure_reason_counts == {
        "INVALID_CERTIFICATE": 1,
        "INVALID_EVALUATION": 1,
    }
    assert "NaN" not in json.dumps(report.to_dict(), allow_nan=False)


def test_empty_test_set_has_null_intervals_and_explicit_failure() -> None:
    report = aggregate_locked_test_evaluations(
        (),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    assert report.status is Stage4ReportStatus.FAIL_CLOSED
    assert report.aggregation_failure_reasons == (Stage4AggregationFailureReason.EMPTY_TEST_SET,)
    assert report.evaluable_proportion.denominator == 0
    assert report.evaluable_proportion.estimate is None
    assert report.evaluable_proportion.lower is None
    assert report.evaluable_proportion.upper is None


def test_wilson_interval_is_deterministic_and_validates_inputs() -> None:
    first = wilson_score_proportion(5, 10, confidence_level=0.95)
    second = wilson_score_proportion(5, 10, confidence_level=0.95)
    assert first == second
    assert first.estimate == 0.5
    assert first.lower == pytest.approx(0.236593090512564)
    assert first.upper == pytest.approx(0.7634069094874361)
    with pytest.raises(ValueError):
        wilson_score_proportion(2, 1)
    with pytest.raises(ValueError):
        wilson_score_proportion(0, 0, confidence_level=math.nan)


def test_locked_contract_rejects_incomplete_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        LockedStage4Contract(
            alpha=0.25,
            positive_threshold=math.nan,
            negative_threshold=0.5,
            positive_sample_count=4,
            negative_sample_count=4,
            total_sample_count=8,
            calibration_artifact_hash=ARTIFACT_HASH,
            config_hash=CONFIG_HASH,
            data_hash=DATA_HASH,
            model_hash=MODEL_HASH,
        )


def test_report_authorization_cannot_be_inherited_by_dataclass_replace() -> None:
    report = aggregate_locked_test_evaluations(
        (
            _record(
                "issued",
                EntailmentState.ENTAILED_PRESENT,
                CalibrationLabel.LESION_PRESENT,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    with pytest.raises(TypeError, match="only be issued"):
        replace(report, evaluable_count=0)


def test_report_recomputes_wilson_intervals_instead_of_trusting_fields() -> None:
    report = aggregate_locked_test_evaluations(
        (
            _record(
                "wilson",
                EntailmentState.ENTAILED_PRESENT,
                CalibrationLabel.UNKNOWN,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    plausible_but_wrong = BinomialProportion(
        numerator=1,
        denominator=1,
        estimate=0.9,
        lower=0.8,
        upper=1.0,
        confidence_level=0.95,
    )
    object.__setattr__(report, "evaluable_proportion", plausible_but_wrong)
    with pytest.raises(ValueError, match="recomputed Wilson"):
        report.__post_init__()


def test_approved_real_data_scope_is_rejected_without_prerequisite_verifier() -> None:
    with pytest.raises(ValueError, match="prerequisite verifier"):
        aggregate_locked_test_evaluations(
            (),
            locked_contract=_lock(),
            scope=Stage4EvaluationScope.APPROVED_REAL_DATA,
        )


def test_missing_certificate_and_replay_normalize_to_out_of_model() -> None:
    evaluation = _success_evaluation("missing-evidence", EntailmentState.ENTAILED_PRESENT)
    report = aggregate_locked_test_evaluations(
        (
            Stage4CaseRecord(
                case_id="missing-evidence",
                split="test",
                reference_label="LESION_PRESENT",
                evaluation=evaluation,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    assert report.evaluable_count == 0
    assert report.state_counts[EntailmentState.OUT_OF_MODEL] == 1
    assert report.failure_reason_counts == {
        "INVALID_CERTIFICATE": 1,
        "INVALID_DUAL_REPLAY": 1,
    }


def test_certificate_artifact_and_evaluation_bindings_are_mandatory() -> None:
    evaluation = _success_evaluation("binding", EntailmentState.ENTAILED_PRESENT)
    wrong_evaluation = _success_evaluation("binding", EntailmentState.ENTAILED_ABSENT)
    certificate, dual = _evidence(wrong_evaluation, artifact_hash="f" * 64)
    report = aggregate_locked_test_evaluations(
        (
            Stage4CaseRecord(
                case_id="binding",
                split="test",
                reference_label="UNKNOWN",
                evaluation=evaluation,
                certificate=certificate,
                dual_replay=dual,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    assert report.evaluable_count == 0
    assert report.failure_reason_counts["CERTIFICATE_EVALUATION_MISMATCH"] == 1
    assert report.failure_reason_counts["CERTIFICATE_ARTIFACT_HASH_MISMATCH"] == 1


def test_certificate_replays_must_equal_supplied_dual_replay() -> None:
    evaluation = _success_evaluation("replay-binding", EntailmentState.RAW_UNIDENTIFIABLE)
    certificate, dual = _evidence(evaluation)
    different_positive = replace(dual.positive, witness_id="different-positive")
    different_dual = Stage3DualReplayResult(
        positive=different_positive,
        negative=dual.negative,
        passed=True,
    )
    report = aggregate_locked_test_evaluations(
        (
            Stage4CaseRecord(
                case_id="replay-binding",
                split="test",
                reference_label="UNKNOWN",
                evaluation=evaluation,
                certificate=certificate,
                dual_replay=different_dual,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    assert report.evaluable_count == 0
    assert report.failure_reason_counts == {"CERTIFICATE_REPLAY_MISMATCH": 1}


def test_mutated_lock_and_reason_maps_are_rejected() -> None:
    lock = _lock()
    object.__setattr__(lock, "positive_threshold", math.nan)
    with pytest.raises(ValueError, match="thresholds"):
        aggregate_locked_test_evaluations(
            (),
            locked_contract=lock,
            scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
        )

    report = aggregate_locked_test_evaluations(
        (
            _record(
                "reason-map",
                EntailmentState.ENTAILED_PRESENT,
                CalibrationLabel.UNKNOWN,
            ),
        ),
        locked_contract=_lock(),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )
    object.__setattr__(report, "failure_reason_counts", {"INVALID_CERTIFICATE": 1})
    with pytest.raises(ValueError, match="emptiness"):
        report.to_dict()
    with pytest.raises(ValueError, match="counts"):
        LockedStage4Contract(
            alpha=0.25,
            positive_threshold=0.5,
            negative_threshold=0.5,
            positive_sample_count=4,
            negative_sample_count=4,
            total_sample_count=7,
            calibration_artifact_hash=ARTIFACT_HASH,
            config_hash=CONFIG_HASH,
            data_hash=DATA_HASH,
            model_hash=MODEL_HASH,
        )
