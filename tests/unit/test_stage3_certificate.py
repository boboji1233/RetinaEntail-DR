import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from retinaentail.calibration.certificate import (
    Stage3CertificateValidationError,
    Stage3SyntheticCertificate,
    Stage3WitnessEvidence,
    WitnessSide,
    calibration_artifact_sha256,
)
from retinaentail.calibration.protocol import (
    CalibrationArtifact,
    CalibrationLabel,
    CalibrationRunStatus,
    CalibrationSample,
    EvaluationFailureReason,
    EvaluationResult,
    EvaluationStatus,
    HypothesisEvaluation,
    SampleSplit,
    calibrate_order_statistic,
    evaluate_hypotheses,
)
from retinaentail.degradations import DegradationStep, apply_pipeline, replay_trace
from retinaentail.entailment.states import EntailmentState
from retinaentail.entailment.toy_baseline import normalized_mse
from retinaentail.interventions.toy import generate_toy_fundus, toy_lesion_predicate
from retinaentail.schemas import DegradationTrace, JSONValue, LesionRegion, WitnessRecord
from retinaentail.witnesses.replay import canonical_json_bytes, rgb_png_bytes, sha256_bytes
from retinaentail.witnesses.stage3_replay import (
    ReplayFailureReason,
    Stage3DualReplayResult,
    Stage3ReplayResult,
    replay_stage3_witness_pair,
)


@dataclass(frozen=True)
class _SyntheticPackage:
    root: Path
    region: LesionRegion
    positive_witness: WitnessRecord
    negative_witness: WitnessRecord
    observation_sha256: str
    trace_sha256: str
    calibration: CalibrationArtifact
    evaluation: EvaluationResult


def _digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _build_package(tmp_path: Path) -> _SyntheticPackage:
    root = tmp_path / "package"
    root.mkdir()
    twin = generate_toy_fundus(size=64, lesion_radius=3, seed=17)
    degraded_positive, operation_trace = apply_pipeline(
        twin.lesion_present,
        (
            DegradationStep(
                name="gaussian_blur",
                severity=0.2,
                seed=71,
                parameters={},
            ),
        ),
    )
    degraded_negative = replay_trace(twin.lesion_absent, operation_trace)
    observation = degraded_positive.copy()
    positive_error = normalized_mse(degraded_positive, observation)
    negative_error = normalized_mse(degraded_negative, observation)
    schema_trace = DegradationTrace.from_dict(cast(Mapping[str, object], operation_trace.to_dict()))

    present_bytes = rgb_png_bytes(twin.lesion_present)
    absent_bytes = rgb_png_bytes(twin.lesion_absent)
    observation_bytes = rgb_png_bytes(observation)
    (root / "lesion_present.png").write_bytes(present_bytes)
    (root / "lesion_absent.png").write_bytes(absent_bytes)
    (root / "observation.png").write_bytes(observation_bytes)
    observation_hash = sha256_bytes(observation_bytes)
    trace_hash = sha256_bytes(canonical_json_bytes(schema_trace.to_dict()))

    positive = WitnessRecord(
        witness_id="synthetic-case:positive",
        case_id="synthetic-case",
        lesion_class="toy_dot",
        witness_kind="lesion_present",
        relative_path="lesion_present.png",
        observation_path="observation.png",
        sha256=sha256_bytes(present_bytes),
        degradation_trace=schema_trace,
        replay_seed=71,
        replay_error=positive_error,
        predicate_valid=True,
    )
    negative = WitnessRecord(
        witness_id="synthetic-case:negative",
        case_id="synthetic-case",
        lesion_class="toy_dot",
        witness_kind="lesion_absent",
        relative_path="lesion_absent.png",
        observation_path="observation.png",
        sha256=sha256_bytes(absent_bytes),
        degradation_trace=schema_trace,
        replay_seed=71,
        replay_error=negative_error,
        predicate_valid=True,
    )
    calibration_samples = tuple(
        CalibrationSample(
            sample_id=f"{label.value.casefold()}-{index}",
            split=SampleSplit.CALIBRATION,
            label=label,
            positive_score=1.0,
            negative_score=1.0,
            source_image_id=f"calibration-source-{label.value.casefold()}-{index}",
        )
        for label in (CalibrationLabel.LESION_PRESENT, CalibrationLabel.LESION_ABSENT)
        for index in range(4)
    )
    calibration = calibrate_order_statistic(
        calibration_samples,
        alpha=0.25,
        config_hash=_digest("config"),
        data_hash=observation_hash,
        model_hash=_digest("model"),
    )
    assert calibration.status is CalibrationRunStatus.SUCCESS
    evaluation = evaluate_hypotheses(
        HypothesisEvaluation(
            case_id="synthetic-case",
            lesion_class="toy_dot",
            positive_score=positive_error,
            negative_score=negative_error,
            config_hash=calibration.config_hash,
            data_hash=calibration.data_hash,
            model_hash=calibration.model_hash,
        ),
        calibration,
    )
    return _SyntheticPackage(
        root=root,
        region=twin.region,
        positive_witness=positive,
        negative_witness=negative,
        observation_sha256=observation_hash,
        trace_sha256=trace_hash,
        calibration=calibration,
        evaluation=evaluation,
    )


def _replay_pair(
    package: _SyntheticPackage,
    *,
    positive_witness: object | None = None,
    negative_witness: object | None = None,
    predicate: object = toy_lesion_predicate,
) -> Stage3DualReplayResult:
    return replay_stage3_witness_pair(
        package.root,
        positive_witness=(
            package.positive_witness if positive_witness is None else positive_witness
        ),
        negative_witness=(
            package.negative_witness if negative_witness is None else negative_witness
        ),
        region=package.region,
        predicate=predicate,
        predicate_id="toy_dark_red_v1",
        expected_case_id="synthetic-case",
        expected_lesion_class="toy_dot",
        observation_sha256=package.observation_sha256,
        trace_sha256=package.trace_sha256,
    )


def _certificate(
    package: _SyntheticPackage,
    pair: Stage3DualReplayResult,
    evaluation: EvaluationResult | None = None,
    *,
    positive_witness: WitnessRecord | None = None,
) -> Stage3SyntheticCertificate:
    return Stage3SyntheticCertificate(
        evaluation=package.evaluation if evaluation is None else evaluation,
        region=package.region,
        predicate_id="toy_dark_red_v1",
        calibration_artifact_hash=calibration_artifact_sha256(package.calibration),
        observation_sha256=package.observation_sha256,
        degradation_trace_sha256=package.trace_sha256,
        positive_evidence=Stage3WitnessEvidence(
            side=WitnessSide.POSITIVE,
            witness=(package.positive_witness if positive_witness is None else positive_witness),
            replay=pair.positive,
        ),
        negative_evidence=Stage3WitnessEvidence(
            side=WitnessSide.NEGATIVE,
            witness=package.negative_witness,
            replay=pair.negative,
        ),
    )


def test_dual_replay_and_certificate_round_trip_are_deterministic(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    calls: list[str] = []

    def counting_predicate(image: np.ndarray, region: LesionRegion) -> bool:
        calls.append(region.region_id or "none")
        return toy_lesion_predicate(image, region)

    pair = _replay_pair(package, predicate=counting_predicate)
    assert pair.passed
    assert len(calls) == 2
    assert pair.positive.replay_error == package.evaluation.positive_score
    assert pair.negative.replay_error == package.evaluation.negative_score

    pair_payload = json.loads(json.dumps(pair.to_dict(), allow_nan=False))
    assert Stage3DualReplayResult.from_dict(pair_payload) == pair
    certificate = _certificate(package, pair)
    payload = certificate.to_dict()
    assert payload == certificate.to_dict()
    assert Stage3SyntheticCertificate.from_dict(payload) == certificate
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert str(tmp_path.resolve()) not in serialized
    assert certificate.evaluation.state is EntailmentState.RAW_UNIDENTIFIABLE


def test_side_swap_and_missing_witness_fail_closed_without_short_circuit(
    tmp_path: Path,
) -> None:
    package = _build_package(tmp_path)
    swapped = _replay_pair(
        package,
        positive_witness=package.negative_witness,
        negative_witness=package.positive_witness,
    )
    assert not swapped.passed
    assert ReplayFailureReason.SIDE_MISMATCH in swapped.positive.failure_reasons
    assert ReplayFailureReason.SIDE_MISMATCH in swapped.negative.failure_reasons
    assert swapped.positive.replay_error is not None
    assert swapped.negative.replay_error is not None

    missing = replay_stage3_witness_pair(
        package.root,
        positive_witness=None,
        negative_witness=package.negative_witness,
        region=package.region,
        predicate=toy_lesion_predicate,
        predicate_id="toy_dark_red_v1",
        expected_case_id="synthetic-case",
        expected_lesion_class="toy_dot",
        observation_sha256=package.observation_sha256,
        trace_sha256=package.trace_sha256,
    )
    assert not missing.passed
    assert ReplayFailureReason.MISSING_WITNESS in missing.positive.failure_reasons
    assert missing.negative.passed
    assert Stage3DualReplayResult.from_dict(missing.to_dict()) == missing


def test_tamper_and_path_escape_return_controlled_path_free_reasons(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    tampered_image = generate_toy_fundus(size=64, lesion_radius=3, seed=17).lesion_present
    tampered_image = tampered_image.copy()
    tampered_image[0, 0, 0] ^= np.uint8(1)
    (package.root / "tampered.png").write_bytes(rgb_png_bytes(tampered_image))
    tampered_witness = replace(package.positive_witness, relative_path="tampered.png")
    tampered = _replay_pair(package, positive_witness=tampered_witness)
    assert not tampered.passed
    assert ReplayFailureReason.LATENT_HASH_MISMATCH in tampered.positive.failure_reasons
    assert str(tmp_path.resolve()) not in json.dumps(tampered.to_dict(), sort_keys=True)

    (tmp_path / "outside.png").write_bytes(rgb_png_bytes(tampered_image))
    escaped_witness = copy.deepcopy(package.positive_witness)
    object.__setattr__(escaped_witness, "relative_path", "../outside.png")
    escaped = _replay_pair(package, positive_witness=escaped_witness)
    assert not escaped.passed
    assert ReplayFailureReason.ARTIFACT_PATH_INVALID in escaped.positive.failure_reasons
    assert str(tmp_path.resolve()) not in json.dumps(escaped.to_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("predicate", "reason"),
    [
        (object(), ReplayFailureReason.INVALID_PREDICATE),
        (lambda _image, _region: np.bool_(True), ReplayFailureReason.INVALID_PREDICATE_RESULT),
    ],
)
def test_invalid_predicate_fails_closed(
    tmp_path: Path,
    predicate: object,
    reason: ReplayFailureReason,
) -> None:
    package = _build_package(tmp_path)
    pair = _replay_pair(package, predicate=predicate)
    assert not pair.passed
    assert reason in pair.positive.failure_reasons
    assert reason in pair.negative.failure_reasons


def test_predicate_exception_and_unknown_trace_fail_closed(tmp_path: Path) -> None:
    package = _build_package(tmp_path)

    def broken_predicate(_image: np.ndarray, _region: LesionRegion) -> bool:
        raise RuntimeError("synthetic predicate failure")

    broken = _replay_pair(package, predicate=broken_predicate)
    assert ReplayFailureReason.PREDICATE_EXCEPTION in broken.positive.failure_reasons
    assert ReplayFailureReason.PREDICATE_EXCEPTION in broken.negative.failure_reasons

    unknown_trace = DegradationTrace.from_dict(package.positive_witness.degradation_trace.to_dict())
    unknown_trace.operations[0]["name"] = "unknown_operator"
    unknown_witness = replace(package.positive_witness, degradation_trace=unknown_trace)
    unknown = _replay_pair(package, positive_witness=unknown_witness)
    assert not unknown.passed
    assert ReplayFailureReason.UNKNOWN_DEGRADATION in unknown.positive.failure_reasons
    assert ReplayFailureReason.TRACE_HASH_MISMATCH in unknown.positive.failure_reasons
    assert unknown.negative.passed


def test_out_of_bounds_region_fails_before_an_ignoring_predicate(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    outside = LesionRegion(x=63, y=63, width=3, height=3, region_id="outside")

    def ignores_supplied_region(image: np.ndarray, _region: LesionRegion) -> bool:
        return toy_lesion_predicate(image, package.region)

    pair = replay_stage3_witness_pair(
        package.root,
        positive_witness=package.positive_witness,
        negative_witness=package.negative_witness,
        region=outside,
        predicate=ignores_supplied_region,
        predicate_id="ignores_region_for_test",
        expected_case_id="synthetic-case",
        expected_lesion_class="toy_dot",
        observation_sha256=package.observation_sha256,
        trace_sha256=package.trace_sha256,
    )

    assert not pair.passed
    assert ReplayFailureReason.REGION_OUT_OF_BOUNDS in pair.positive.failure_reasons
    assert ReplayFailureReason.REGION_OUT_OF_BOUNDS in pair.negative.failure_reasons
    assert not pair.positive.predicate_valid
    assert not pair.negative.predicate_valid


def test_non_pixel_region_fails_before_predicate(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    normalized = replace(package.region, coordinate_space="normalized")
    calls = 0

    def ignoring_predicate(_image: np.ndarray, _region: LesionRegion) -> bool:
        nonlocal calls
        calls += 1
        return True

    pair = replay_stage3_witness_pair(
        package.root,
        positive_witness=package.positive_witness,
        negative_witness=package.negative_witness,
        region=normalized,
        predicate=ignoring_predicate,
        predicate_id="unsupported_coordinate_space_test",
        expected_case_id="synthetic-case",
        expected_lesion_class="toy_dot",
        observation_sha256=package.observation_sha256,
        trace_sha256=package.trace_sha256,
    )

    assert not pair.passed
    assert calls == 0
    assert ReplayFailureReason.UNSUPPORTED_COORDINATE_SPACE in pair.positive.failure_reasons
    assert ReplayFailureReason.UNSUPPORTED_COORDINATE_SPACE in pair.negative.failure_reasons


def test_certificate_rejects_non_pixel_region_even_with_rehashed_evidence(
    tmp_path: Path,
) -> None:
    package = _build_package(tmp_path)
    pair = _replay_pair(package)
    payload = _certificate(package, pair).to_dict()
    region = cast(dict[str, object], payload["region"])
    region["coordinate_space"] = "normalized"
    region_hash = sha256_bytes(canonical_json_bytes(cast(dict[str, JSONValue], region)))
    for evidence_name in ("positive_evidence", "negative_evidence"):
        evidence = cast(dict[str, object], payload[evidence_name])
        replay = cast(dict[str, object], evidence["replay"])
        replay["region_sha256"] = region_hash

    with pytest.raises(Stage3CertificateValidationError, match="pixel coordinates"):
        Stage3SyntheticCertificate.from_dict(payload)


def test_invalid_expected_hash_failure_evidence_round_trips(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    pair = replay_stage3_witness_pair(
        package.root,
        positive_witness=package.positive_witness,
        negative_witness=package.negative_witness,
        region=package.region,
        predicate=toy_lesion_predicate,
        predicate_id="invalid_expected_hash_test",
        expected_case_id="synthetic-case",
        expected_lesion_class="toy_dot",
        observation_sha256="not-a-digest",
        trace_sha256="also-not-a-digest",
    )

    assert not pair.passed
    assert pair.positive.expected_observation_sha256 is None
    assert pair.positive.expected_trace_sha256 is None
    assert ReplayFailureReason.INVALID_EXPECTED_HASH in pair.positive.failure_reasons
    payload = json.loads(json.dumps(pair.to_dict(), allow_nan=False))
    assert Stage3DualReplayResult.from_dict(payload) == pair


def test_passed_replay_requires_complete_hashes_and_shared_pair_context(
    tmp_path: Path,
) -> None:
    package = _build_package(tmp_path)
    pair = _replay_pair(package)
    missing_hashes = pair.positive.to_dict()
    missing_hashes["expected_observation_sha256"] = None
    missing_hashes["actual_observation_sha256"] = None
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        Stage3ReplayResult.from_dict(missing_hashes)

    mismatched_pair = pair.to_dict()
    negative = cast(dict[str, object], mismatched_pair["negative"])
    negative["case_id"] = "different-case"
    with pytest.raises(ValueError, match="declared context"):
        Stage3DualReplayResult.from_dict(mismatched_pair)


def test_failed_evidence_requires_fail_closed_out_of_model_evaluation(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    tampered_witness = replace(package.positive_witness, sha256="0" * 64)
    pair = _replay_pair(package, positive_witness=tampered_witness)
    assert pair.positive.replay_error is not None
    assert pair.negative.replay_error is not None
    failed_evaluation = evaluate_hypotheses(
        HypothesisEvaluation(
            case_id="synthetic-case",
            lesion_class="toy_dot",
            positive_score=pair.positive.replay_error,
            negative_score=pair.negative.replay_error,
            config_hash=package.calibration.config_hash,
            data_hash=package.calibration.data_hash,
            model_hash=package.calibration.model_hash,
            positive_witness_replay_valid=False,
            negative_witness_replay_valid=True,
        ),
        package.calibration,
    )
    certificate = _certificate(
        package,
        pair,
        failed_evaluation,
        positive_witness=tampered_witness,
    )
    assert certificate.evaluation.state is EntailmentState.OUT_OF_MODEL
    assert Stage3SyntheticCertificate.from_dict(certificate.to_dict()) == certificate

    with pytest.raises(ValueError, match="fail-closed flags"):
        replace(
            package.evaluation,
            positive_score=pair.positive.replay_error,
            positive_witness_replay_valid=False,
        )
    with pytest.raises(Stage3CertificateValidationError, match="scores must match"):
        _certificate(
            package,
            pair,
            replace(failed_evaluation, positive_score=pair.positive.replay_error + 0.1),
            positive_witness=tampered_witness,
        )


def test_certificate_requires_both_evidence_and_strict_payload_types(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    pair = _replay_pair(package)
    certificate = _certificate(package, pair)
    values = certificate.to_dict()

    with pytest.raises(Stage3CertificateValidationError, match="positive evidence"):
        replace(certificate, positive_evidence=None)  # type: ignore[arg-type]

    unknown = copy.deepcopy(values)
    unknown["unexpected"] = True
    with pytest.raises(Stage3CertificateValidationError, match="unknown fields"):
        Stage3SyntheticCertificate.from_dict(unknown)

    nonfinite = copy.deepcopy(values)
    evaluation = cast(dict[str, object], nonfinite["evaluation"])
    evaluation["positive_score"] = float("nan")
    with pytest.raises(Stage3CertificateValidationError, match="null or finite"):
        Stage3SyntheticCertificate.from_dict(nonfinite)

    coerced_bool = copy.deepcopy(values)
    positive_evidence = cast(dict[str, object], coerced_bool["positive_evidence"])
    replay = cast(dict[str, object], positive_evidence["replay"])
    replay["passed"] = 1
    with pytest.raises(Stage3CertificateValidationError, match="invalid witness replay evidence"):
        Stage3SyntheticCertificate.from_dict(coerced_bool)


def test_certificate_rejects_forged_witness_replay_bindings(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    pair = _replay_pair(package)
    payload = _certificate(package, pair).to_dict()

    replay_error = copy.deepcopy(payload)
    positive = cast(dict[str, object], replay_error["positive_evidence"])
    witness = cast(dict[str, object], positive["witness"])
    witness["replay_error"] = 99.0
    with pytest.raises(Stage3CertificateValidationError, match="replay_error"):
        Stage3SyntheticCertificate.from_dict(replay_error)

    predicate = copy.deepcopy(payload)
    positive = cast(dict[str, object], predicate["positive_evidence"])
    witness = cast(dict[str, object], positive["witness"])
    witness["predicate_valid"] = False
    with pytest.raises(Stage3CertificateValidationError, match="predicate_valid"):
        Stage3SyntheticCertificate.from_dict(predicate)

    seed = copy.deepcopy(payload)
    positive = cast(dict[str, object], seed["positive_evidence"])
    witness = cast(dict[str, object], positive["witness"])
    witness["replay_seed"] = 999_999
    with pytest.raises(Stage3CertificateValidationError, match="replay_seed"):
        Stage3SyntheticCertificate.from_dict(seed)

    parameter = copy.deepcopy(payload)
    positive = cast(dict[str, object], parameter["positive_evidence"])
    witness = cast(dict[str, object], positive["witness"])
    trace = cast(dict[str, object], witness["degradation_trace"])
    operations = cast(list[object], trace["operations"])
    operation = cast(dict[str, object], operations[0])
    parameters = cast(dict[str, object], operation["parameters"])
    parameters["typo"] = 1.0
    with pytest.raises(Stage3CertificateValidationError, match="degradation parameters"):
        Stage3SyntheticCertificate.from_dict(parameter)

    trace_tamper = copy.deepcopy(payload)
    positive = cast(dict[str, object], trace_tamper["positive_evidence"])
    witness = cast(dict[str, object], positive["witness"])
    trace = cast(dict[str, object], witness["degradation_trace"])
    operations = cast(list[object], trace["operations"])
    operation = cast(dict[str, object], operations[0])
    parameters = cast(dict[str, object], operation["parameters"])
    radius = parameters["radius"]
    assert isinstance(radius, float)
    parameters["radius"] = radius + 0.1
    with pytest.raises(Stage3CertificateValidationError, match="trace hash"):
        Stage3SyntheticCertificate.from_dict(trace_tamper)


def test_certificate_recomputes_feasibility_from_scores_and_thresholds(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    assert package.evaluation.negative_score is not None
    assert package.evaluation.negative_score > 0.0
    with pytest.raises(ValueError, match="feasibility conflicts"):
        replace(
            package.evaluation,
            negative_threshold=0.0,
            negative_feasible=True,
            state=EntailmentState.RAW_UNIDENTIFIABLE,
        )


def test_certificate_requires_traceable_expected_and_input_provenance(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    pair = _replay_pair(package)
    with pytest.raises(ValueError, match="matching complete provenance"):
        replace(
            package.evaluation,
            input_data_hash=_digest("different-data"),
        )

    mismatch = evaluate_hypotheses(
        HypothesisEvaluation(
            case_id="synthetic-case",
            lesion_class="toy_dot",
            positive_score=pair.positive.replay_error,
            negative_score=pair.negative.replay_error,
            config_hash=package.calibration.config_hash,
            data_hash=_digest("different-data"),
            model_hash=package.calibration.model_hash,
        ),
        package.calibration,
    )
    certificate = _certificate(package, pair, mismatch)
    assert certificate.evaluation.state is EntailmentState.OUT_OF_MODEL
    assert certificate.evaluation.input_data_hash == _digest("different-data")


def test_certificate_requires_ood_failure_reason(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    with pytest.raises(ValueError, match="required failure reason"):
        replace(
            package.evaluation,
            positive_feasible=None,
            negative_feasible=None,
            state=EntailmentState.OUT_OF_MODEL,
            status=EvaluationStatus.FAIL_CLOSED,
            failure_reasons=(EvaluationFailureReason.INVALID_INPUT,),
            out_of_distribution=True,
        )
