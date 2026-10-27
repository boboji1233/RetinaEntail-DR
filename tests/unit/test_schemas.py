from __future__ import annotations

import json
import math

import pytest

from retinaentail.entailment.states import EntailmentState
from retinaentail.schemas import (
    DegradationTrace,
    EntailmentCertificate,
    HypothesisScore,
    LesionRegion,
    SchemaValidationError,
    WitnessRecord,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _trace() -> DegradationTrace:
    return DegradationTrace(
        operations=(
            {
                "name": "gaussian_blur",
                "severity": 0.25,
                "seed": 7,
                "parameters": {"radius": 1.0},
            },
            {
                "name": "jpeg_compression",
                "severity": 0.4,
                "seed": 8,
                "parameters": {"quality": 80},
            },
        )
    )


def _witness(kind: str) -> WitnessRecord:
    return WitnessRecord(
        witness_id=f"witness-{kind}",
        case_id="toy-case",
        lesion_class="microaneurysm",
        witness_kind=kind,
        relative_path=f"witnesses/{kind}.json",
        observation_path="observations/raw.png",
        sha256=HASH_A if kind == "positive" else HASH_B,
        degradation_trace=_trace(),
        replay_seed=11,
        replay_error=0.0,
        predicate_valid=True,
    )


def _certificate(*, solver_status: str = "success") -> EntailmentCertificate:
    return EntailmentCertificate(
        case_id="toy-case",
        lesion_class="microaneurysm",
        region=LesionRegion(x=4, y=5, width=8, height=8, region_id="candidate-1"),
        alpha=0.05,
        positive=HypothesisScore(
            score=0.1,
            threshold=0.2,
            feasible=True,
            solver_status=solver_status,
        ),
        negative=HypothesisScore(
            score=0.8,
            threshold=0.2,
            feasible=False,
            solver_status="success",
        ),
        state=EntailmentState.ENTAILED_PRESENT,
        config_hash=HASH_A,
        data_hash=HASH_B,
        model_hashes={"toy_baseline": HASH_C},
        solver_status="success",
        witnesses=(_witness("positive"), _witness("negative")),
        degradation_trace=_trace(),
    )


def test_degradation_trace_json_round_trip_preserves_order() -> None:
    trace = _trace()
    payload = json.loads(json.dumps(trace.to_dict()))
    restored = DegradationTrace.from_dict(payload)
    assert restored == trace
    assert [step["name"] for step in restored.operations] == [
        "gaussian_blur",
        "jpeg_compression",
    ]


def test_unknown_degradation_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="unknown degradation"):
        DegradationTrace(
            operations=(
                {
                    "name": "unregistered_filter",
                    "severity": 0.2,
                    "seed": 1,
                    "parameters": {},
                },
            )
        )


@pytest.mark.parametrize("field", ["score", "threshold"])
def test_hypothesis_score_rejects_non_finite_values(field: str) -> None:
    values: dict[str, object] = {
        "score": 0.1,
        "threshold": 0.2,
        "feasible": True,
        "solver_status": "success",
    }
    values[field] = math.nan
    with pytest.raises(SchemaValidationError, match="finite"):
        HypothesisScore(**values)  # type: ignore[arg-type]


def test_certificate_is_json_safe_and_round_trips() -> None:
    certificate = _certificate()
    payload = json.loads(json.dumps(certificate.to_dict(), sort_keys=True))
    restored = EntailmentCertificate.from_dict(payload)
    assert restored == certificate
    assert restored.state is EntailmentState.ENTAILED_PRESENT
    assert {witness.hypothesis_side for witness in restored.witnesses} == {
        "positive",
        "negative",
    }


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "case_id", "region", "positive", "config_hash", "solver_status"],
)
def test_certificate_missing_critical_field_fails_validation(missing_field: str) -> None:
    payload = _certificate().to_dict()
    del payload[missing_field]
    with pytest.raises(SchemaValidationError, match="missing required"):
        EntailmentCertificate.from_dict(payload)


def test_failed_solver_status_forces_certificate_out_of_model() -> None:
    certificate = _certificate(solver_status="timeout")
    assert certificate.positive.solver_status == "timeout"
    assert certificate.state is EntailmentState.OUT_OF_MODEL


def test_invalid_witness_predicate_forces_certificate_out_of_model() -> None:
    invalid_witness = WitnessRecord(
        witness_id="witness-positive",
        case_id="toy-case",
        lesion_class="microaneurysm",
        witness_kind="positive",
        relative_path="witnesses/positive.json",
        observation_path="observations/raw.png",
        sha256=HASH_A,
        degradation_trace=_trace(),
        replay_seed=11,
        replay_error=0.0,
        predicate_valid=False,
    )
    certificate = EntailmentCertificate(
        case_id="toy-case",
        lesion_class="microaneurysm",
        region=LesionRegion(x=4, y=5, width=8, height=8),
        alpha=0.05,
        positive=HypothesisScore(score=0.1, threshold=0.2, feasible=True, solver_status="success"),
        negative=HypothesisScore(score=0.8, threshold=0.2, feasible=False, solver_status="success"),
        state=EntailmentState.ENTAILED_PRESENT,
        config_hash=HASH_A,
        data_hash=HASH_B,
        model_hashes={"toy_baseline": HASH_C},
        solver_status="success",
        witnesses=(invalid_witness,),
    )
    assert certificate.state is EntailmentState.OUT_OF_MODEL


def test_truth_table_mismatch_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="truth table"):
        EntailmentCertificate(
            case_id="toy-case",
            lesion_class="microaneurysm",
            region=LesionRegion(x=0, y=0, width=4, height=4),
            alpha=0.05,
            positive=HypothesisScore(
                score=0.1, threshold=0.2, feasible=True, solver_status="success"
            ),
            negative=HypothesisScore(
                score=0.8, threshold=0.2, feasible=False, solver_status="success"
            ),
            state=EntailmentState.ENTAILED_ABSENT,
            config_hash=HASH_A,
            data_hash=HASH_B,
            model_hashes={"toy_baseline": HASH_C},
            solver_status="success",
        )
