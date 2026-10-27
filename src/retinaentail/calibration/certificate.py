"""Strict synthetic Stage 3 certificate bound to calibration and dual replay evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from retinaentail.calibration.protocol import (
    CALIBRATION_METHOD_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    CalibrationArtifact,
    EvaluationFailureReason,
    EvaluationResult,
    EvaluationStatus,
    ProtocolScope,
    calibration_artifact_validation_reasons,
)
from retinaentail.degradations import validate_resolved_operator_parameters
from retinaentail.entailment.states import (
    EntailmentState,
    solver_status_succeeded,
    state_from_feasibility,
)
from retinaentail.schemas import JSONValue, LesionRegion, WitnessRecord
from retinaentail.witnesses.replay import canonical_json_bytes, sha256_bytes
from retinaentail.witnesses.stage3_replay import Stage3ReplayResult

STAGE3_CERTIFICATE_SCHEMA_VERSION: Final = "3.0.0"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SCORE_TOLERANCE: Final = 1e-12


class Stage3CertificateValidationError(ValueError):
    """Raised when evidence cannot satisfy the strict synthetic certificate contract."""


class Stage3CertificateScope(StrEnum):
    """The only scope currently eligible for a Stage 3 certificate."""

    SYNTHETIC_TOY = "SYNTHETIC_TOY"


class WitnessSide(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _require_sha256(value: object, name: str) -> str:
    if not _valid_sha256(value):
        raise Stage3CertificateValidationError(f"{name} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    return None if value is None else _require_sha256(value, name)


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage3CertificateValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_mapping(
    value: object,
    *,
    name: str,
    required: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Stage3CertificateValidationError(f"{name} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise Stage3CertificateValidationError(f"{name} keys must be strings")
        result[key] = item
    if required.difference(result):
        raise Stage3CertificateValidationError(f"{name} is missing required fields")
    if set(result).difference(required):
        raise Stage3CertificateValidationError(f"{name} contains unknown fields")
    return result


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage3CertificateValidationError(f"{name} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise Stage3CertificateValidationError(f"{name} keys must be strings")
        result[key] = item
    return result


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - strict serialized boundary
        raise Stage3CertificateValidationError(f"{name} must be a bool")
    assert isinstance(value, bool)
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    return None if value is None else _exact_bool(value, name)


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _require_identifier(value, name)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage3CertificateValidationError(f"{name} must be a non-negative integer")
    return value


def _optional_finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_bounds: bool = False,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise Stage3CertificateValidationError(f"{name} must be null or finite")
    clean = float(value)
    if minimum is not None and (clean <= minimum if exclusive_bounds else clean < minimum):
        raise Stage3CertificateValidationError(f"{name} is below its allowed range")
    if maximum is not None and (clean >= maximum if exclusive_bounds else clean > maximum):
        raise Stage3CertificateValidationError(f"{name} is above its allowed range")
    return clean


def _evaluation_from_dict(value: object) -> EvaluationResult:
    required = {
        "schema_version",
        "method_version",
        "protocol_scope",
        "case_id",
        "lesion_class",
        "alpha",
        "positive_score",
        "negative_score",
        "positive_threshold",
        "negative_threshold",
        "positive_feasible",
        "negative_feasible",
        "positive_sample_count",
        "negative_sample_count",
        "config_hash",
        "data_hash",
        "model_hash",
        "input_config_hash",
        "input_data_hash",
        "input_model_hash",
        "positive_solver_status",
        "negative_solver_status",
        "solver_status",
        "degradation_known",
        "positive_witness_replay_valid",
        "negative_witness_replay_valid",
        "out_of_distribution",
        "state",
        "status",
        "failure_reasons",
    }
    fields = _strict_mapping(value, name="evaluation", required=required)
    schema_version = _require_identifier(fields["schema_version"], "schema_version")
    method_version = _require_identifier(fields["method_version"], "method_version")
    try:
        protocol_scope = ProtocolScope(
            _require_identifier(fields["protocol_scope"], "protocol_scope")
        )
        state = EntailmentState(_require_identifier(fields["state"], "state"))
        status = EvaluationStatus(_require_identifier(fields["status"], "status"))
    except (TypeError, ValueError) as error:
        raise Stage3CertificateValidationError("evaluation contains an invalid enum") from error
    raw_reasons = fields["failure_reasons"]
    if isinstance(raw_reasons, str | bytes) or not isinstance(raw_reasons, Sequence):
        raise Stage3CertificateValidationError("failure_reasons must be an array")
    reasons: list[EvaluationFailureReason] = []
    for raw_reason in raw_reasons:
        if not isinstance(raw_reason, str):
            raise Stage3CertificateValidationError("failure reasons must be strings")
        try:
            reason = EvaluationFailureReason(raw_reason)
        except ValueError as error:
            raise Stage3CertificateValidationError("unknown evaluation failure reason") from error
        if reason in reasons:
            raise Stage3CertificateValidationError("duplicate evaluation failure reason")
        reasons.append(reason)
    return EvaluationResult(
        schema_version=schema_version,
        method_version=method_version,
        protocol_scope=protocol_scope,
        case_id=_require_identifier(fields["case_id"], "case_id"),
        lesion_class=_require_identifier(fields["lesion_class"], "lesion_class"),
        alpha=_optional_finite_number(
            fields["alpha"], "alpha", minimum=0.0, maximum=1.0, exclusive_bounds=True
        ),
        positive_score=_optional_finite_number(
            fields["positive_score"], "positive_score", minimum=0.0
        ),
        negative_score=_optional_finite_number(
            fields["negative_score"], "negative_score", minimum=0.0
        ),
        positive_threshold=_optional_finite_number(
            fields["positive_threshold"], "positive_threshold", minimum=0.0
        ),
        negative_threshold=_optional_finite_number(
            fields["negative_threshold"], "negative_threshold", minimum=0.0
        ),
        positive_feasible=_optional_bool(fields["positive_feasible"], "positive_feasible"),
        negative_feasible=_optional_bool(fields["negative_feasible"], "negative_feasible"),
        positive_sample_count=_nonnegative_int(
            fields["positive_sample_count"], "positive_sample_count"
        ),
        negative_sample_count=_nonnegative_int(
            fields["negative_sample_count"], "negative_sample_count"
        ),
        config_hash=_optional_sha256(fields["config_hash"], "config_hash"),
        data_hash=_optional_sha256(fields["data_hash"], "data_hash"),
        model_hash=_optional_sha256(fields["model_hash"], "model_hash"),
        input_config_hash=_optional_sha256(fields["input_config_hash"], "input_config_hash"),
        input_data_hash=_optional_sha256(fields["input_data_hash"], "input_data_hash"),
        input_model_hash=_optional_sha256(fields["input_model_hash"], "input_model_hash"),
        positive_solver_status=_optional_string(
            fields["positive_solver_status"], "positive_solver_status"
        ),
        negative_solver_status=_optional_string(
            fields["negative_solver_status"], "negative_solver_status"
        ),
        solver_status=_optional_string(fields["solver_status"], "solver_status"),
        degradation_known=_exact_bool(fields["degradation_known"], "degradation_known"),
        positive_witness_replay_valid=_exact_bool(
            fields["positive_witness_replay_valid"], "positive_witness_replay_valid"
        ),
        negative_witness_replay_valid=_exact_bool(
            fields["negative_witness_replay_valid"], "negative_witness_replay_valid"
        ),
        out_of_distribution=_exact_bool(fields["out_of_distribution"], "out_of_distribution"),
        state=state,
        status=status,
        failure_reasons=tuple(reasons),
    )


def _scores_match(score: object, replay_error: object) -> bool:
    if score is None or replay_error is None:
        return score is None and replay_error is None
    if (
        isinstance(score, bool)
        or isinstance(replay_error, bool)
        or not isinstance(score, int | float)
        or not isinstance(replay_error, int | float)
    ):
        return False
    clean_score = float(score)
    clean_error = float(replay_error)
    return (
        math.isfinite(clean_score)
        and math.isfinite(clean_error)
        and abs(clean_score - clean_error) <= _SCORE_TOLERANCE
    )


def calibration_artifact_sha256(artifact: CalibrationArtifact) -> str:
    """Hash the complete deterministic calibration artifact representation."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be a CalibrationArtifact")
    if calibration_artifact_validation_reasons(artifact):
        raise Stage3CertificateValidationError("calibration artifact is not process-issued")
    return sha256_bytes(canonical_json_bytes(cast(dict[str, JSONValue], artifact.to_dict())))


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3WitnessEvidence:
    """One original witness record paired with its independent replay result."""

    side: WitnessSide
    witness: WitnessRecord
    replay: Stage3ReplayResult
    schema_version: str = STAGE3_CERTIFICATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_CERTIFICATE_SCHEMA_VERSION:
            raise Stage3CertificateValidationError("unsupported evidence schema_version")
        if not isinstance(self.side, WitnessSide):
            raise Stage3CertificateValidationError("side must be a WitnessSide")
        if not isinstance(self.witness, WitnessRecord):
            raise Stage3CertificateValidationError("witness must be a WitnessRecord")
        if not isinstance(self.replay, Stage3ReplayResult):
            raise Stage3CertificateValidationError("replay must be a Stage3ReplayResult")
        if self.witness.hypothesis_side != self.side.value:
            raise Stage3CertificateValidationError("witness side does not match evidence side")
        if self.replay.side != self.side.value:
            raise Stage3CertificateValidationError("replay side does not match evidence side")
        if self.replay.witness_id != self.witness.witness_id:
            raise Stage3CertificateValidationError("replay witness_id does not match witness")
        if self.replay.expected_latent_sha256 != self.witness.sha256:
            raise Stage3CertificateValidationError("replay is not bound to witness latent hash")
        if self.replay.passed and not _scores_match(
            self.witness.replay_error, self.replay.replay_error
        ):
            raise Stage3CertificateValidationError("witness replay_error does not match replay")
        if self.replay.passed and self.witness.predicate_valid is not self.replay.predicate_valid:
            raise Stage3CertificateValidationError("witness predicate_valid does not match replay")
        operations = self.witness.degradation_trace.operations
        if not operations or operations[0].get("seed") != self.witness.replay_seed:
            raise Stage3CertificateValidationError(
                "witness replay_seed must match the first degradation operation"
            )
        try:
            for operation in operations:
                name = operation.get("name")
                parameters = operation.get("parameters")
                if not isinstance(name, str) or not isinstance(parameters, Mapping):
                    raise ValueError("degradation operation is incomplete")
                validate_resolved_operator_parameters(name, parameters)
        except (TypeError, ValueError) as error:
            raise Stage3CertificateValidationError(
                "witness degradation parameters are invalid"
            ) from error
        witness_trace_sha256 = sha256_bytes(
            canonical_json_bytes(self.witness.degradation_trace.to_dict())
        )
        if self.replay.actual_trace_sha256 != witness_trace_sha256:
            raise Stage3CertificateValidationError(
                "witness degradation trace hash does not match replay actual trace hash"
            )
        try:
            reparsed = Stage3ReplayResult.from_dict(self.replay.to_dict())
        except ValueError as error:
            raise Stage3CertificateValidationError("replay result is internally invalid") from error
        if reparsed != self.replay:
            raise Stage3CertificateValidationError("replay result is not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "side": self.side.value,
            "witness": self.witness.to_dict(),
            "replay": self.replay.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage3WitnessEvidence:
        fields = _strict_mapping(
            value,
            name="Stage3WitnessEvidence",
            required={"schema_version", "side", "witness", "replay"},
        )
        schema_version = _require_identifier(fields["schema_version"], "schema_version")
        try:
            side = WitnessSide(_require_identifier(fields["side"], "side"))
        except (TypeError, ValueError) as error:
            raise Stage3CertificateValidationError("invalid evidence side") from error
        try:
            witness = WitnessRecord.from_dict(_as_mapping(fields["witness"], "witness"))
            replay = Stage3ReplayResult.from_dict(_as_mapping(fields["replay"], "replay"))
        except (TypeError, ValueError) as error:
            raise Stage3CertificateValidationError("invalid witness replay evidence") from error
        return cls(
            schema_version=schema_version,
            side=side,
            witness=witness,
            replay=replay,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3SyntheticCertificate:
    """A provenance-complete certificate limited to the synthetic toy protocol.

    Construction enforces that both sides are present and independently replayed.
    Evidence failures may be retained only alongside a fail-closed
    ``OUT_OF_MODEL`` evaluation.
    """

    evaluation: EvaluationResult
    region: LesionRegion
    predicate_id: str
    calibration_artifact_hash: str
    observation_sha256: str
    degradation_trace_sha256: str
    positive_evidence: Stage3WitnessEvidence
    negative_evidence: Stage3WitnessEvidence
    scope: Stage3CertificateScope = Stage3CertificateScope.SYNTHETIC_TOY
    schema_version: str = STAGE3_CERTIFICATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_CERTIFICATE_SCHEMA_VERSION:
            raise Stage3CertificateValidationError("unsupported certificate schema_version")
        if self.scope is not Stage3CertificateScope.SYNTHETIC_TOY:
            raise Stage3CertificateValidationError("certificate scope must be SYNTHETIC_TOY")
        if not isinstance(self.evaluation, EvaluationResult):
            raise Stage3CertificateValidationError("evaluation must be an EvaluationResult")
        if not isinstance(self.region, LesionRegion):
            raise Stage3CertificateValidationError("region must be a LesionRegion")
        if self.region.coordinate_space != "pixel":
            raise Stage3CertificateValidationError(
                "Stage 3 synthetic certificates require pixel coordinates"
            )
        object.__setattr__(
            self, "predicate_id", _require_identifier(self.predicate_id, "predicate_id")
        )
        object.__setattr__(
            self,
            "calibration_artifact_hash",
            _require_sha256(self.calibration_artifact_hash, "calibration_artifact_hash"),
        )
        object.__setattr__(
            self,
            "observation_sha256",
            _require_sha256(self.observation_sha256, "observation_sha256"),
        )
        object.__setattr__(
            self,
            "degradation_trace_sha256",
            _require_sha256(self.degradation_trace_sha256, "degradation_trace_sha256"),
        )
        if not isinstance(self.positive_evidence, Stage3WitnessEvidence):
            raise Stage3CertificateValidationError("positive evidence is required")
        if not isinstance(self.negative_evidence, Stage3WitnessEvidence):
            raise Stage3CertificateValidationError("negative evidence is required")
        if self.positive_evidence.side is not WitnessSide.POSITIVE:
            raise Stage3CertificateValidationError("positive_evidence must be positive")
        if self.negative_evidence.side is not WitnessSide.NEGATIVE:
            raise Stage3CertificateValidationError("negative_evidence must be negative")
        if self.positive_evidence.witness.witness_id == self.negative_evidence.witness.witness_id:
            raise Stage3CertificateValidationError("positive and negative witnesses must differ")

        self._validate_evaluation_contract()
        self._validate_evidence_binding()
        self._validate_fail_closed_semantics()

    def _validate_evaluation_contract(self) -> None:
        evaluation = self.evaluation
        if _evaluation_from_dict(evaluation.to_dict()) != evaluation:
            raise Stage3CertificateValidationError("evaluation is not canonical")
        if (
            evaluation.schema_version != CALIBRATION_SCHEMA_VERSION
            or evaluation.method_version != CALIBRATION_METHOD_VERSION
            or evaluation.protocol_scope is not ProtocolScope.TOY_PROTOCOL
        ):
            raise Stage3CertificateValidationError(
                "evaluation is outside the supported toy calibration protocol"
            )
        if not isinstance(evaluation.state, EntailmentState) or not isinstance(
            evaluation.status, EvaluationStatus
        ):
            raise Stage3CertificateValidationError("evaluation state/status are invalid")
        _require_identifier(evaluation.case_id, "evaluation.case_id")
        _require_identifier(evaluation.lesion_class, "evaluation.lesion_class")
        for name, value in (
            ("evaluation.config_hash", evaluation.config_hash),
            ("evaluation.data_hash", evaluation.data_hash),
            ("evaluation.model_hash", evaluation.model_hash),
            ("evaluation.input_config_hash", evaluation.input_config_hash),
            ("evaluation.input_data_hash", evaluation.input_data_hash),
            ("evaluation.input_model_hash", evaluation.input_model_hash),
        ):
            _require_sha256(value, name)
        expected_hashes = (
            evaluation.config_hash,
            evaluation.data_hash,
            evaluation.model_hash,
        )
        input_hashes = (
            evaluation.input_config_hash,
            evaluation.input_data_hash,
            evaluation.input_model_hash,
        )
        if evaluation.status is EvaluationStatus.SUCCESS:
            if evaluation.failure_reasons:
                raise Stage3CertificateValidationError(
                    "successful evaluation cannot retain failure reasons"
                )
            if not isinstance(evaluation.positive_feasible, bool) or not isinstance(
                evaluation.negative_feasible, bool
            ):
                raise Stage3CertificateValidationError(
                    "successful evaluation requires both feasibility values"
                )
            numeric_evidence = (
                evaluation.alpha,
                evaluation.positive_score,
                evaluation.negative_score,
                evaluation.positive_threshold,
                evaluation.negative_threshold,
            )
            if any(value is None for value in numeric_evidence):
                raise Stage3CertificateValidationError(
                    "successful evaluation requires alpha, scores, and thresholds"
                )
            assert evaluation.positive_score is not None
            assert evaluation.negative_score is not None
            assert evaluation.positive_threshold is not None
            assert evaluation.negative_threshold is not None
            expected_positive = evaluation.positive_score <= evaluation.positive_threshold
            expected_negative = evaluation.negative_score <= evaluation.negative_threshold
            if (
                evaluation.positive_feasible is not expected_positive
                or evaluation.negative_feasible is not expected_negative
            ):
                raise Stage3CertificateValidationError(
                    "evaluation feasibility conflicts with calibrated score thresholds"
                )
            if evaluation.positive_sample_count < 1 or evaluation.negative_sample_count < 1:
                raise Stage3CertificateValidationError(
                    "successful evaluation requires calibration samples on both sides"
                )
            if input_hashes != expected_hashes:
                raise Stage3CertificateValidationError(
                    "successful evaluation input provenance must match calibration"
                )
            expected_state = state_from_feasibility(
                evaluation.positive_feasible,
                evaluation.negative_feasible,
            )
            if evaluation.state is not expected_state:
                raise Stage3CertificateValidationError(
                    "evaluation state conflicts with its feasibility truth table"
                )
        else:
            if evaluation.state is not EntailmentState.OUT_OF_MODEL:
                raise Stage3CertificateValidationError(
                    "fail-closed evaluation must be OUT_OF_MODEL"
                )
            if not evaluation.failure_reasons:
                raise Stage3CertificateValidationError(
                    "fail-closed evaluation must retain a failure reason"
                )
            if evaluation.positive_feasible is not None or evaluation.negative_feasible is not None:
                raise Stage3CertificateValidationError(
                    "fail-closed evaluation cannot assert feasibility"
                )
            provenance_mismatch = EvaluationFailureReason.PROVENANCE_MISMATCH in (
                evaluation.failure_reasons
            )
            if provenance_mismatch != (input_hashes != expected_hashes):
                raise Stage3CertificateValidationError(
                    "provenance mismatch reason conflicts with expected/input hashes"
                )

    def _validate_evidence_binding(self) -> None:
        evaluation = self.evaluation
        region_hash = sha256_bytes(canonical_json_bytes(self.region.to_dict()))
        for evidence in (self.positive_evidence, self.negative_evidence):
            witness = evidence.witness
            replay = evidence.replay
            if witness.case_id != evaluation.case_id or replay.case_id != evaluation.case_id:
                raise Stage3CertificateValidationError("evidence case_id mismatch")
            if (
                witness.lesion_class != evaluation.lesion_class
                or replay.lesion_class != evaluation.lesion_class
            ):
                raise Stage3CertificateValidationError("evidence lesion_class mismatch")
            if replay.predicate_id != self.predicate_id:
                raise Stage3CertificateValidationError("evidence predicate_id mismatch")
            if replay.region_sha256 != region_hash:
                raise Stage3CertificateValidationError("evidence region hash mismatch")
            if replay.expected_observation_sha256 != self.observation_sha256:
                raise Stage3CertificateValidationError("evidence observation hash mismatch")
            if replay.expected_trace_sha256 != self.degradation_trace_sha256:
                raise Stage3CertificateValidationError("evidence trace hash mismatch")
        if (
            self.positive_evidence.witness.observation_path
            != self.negative_evidence.witness.observation_path
        ):
            raise Stage3CertificateValidationError(
                "positive and negative witnesses must share one observation"
            )
        if not _scores_match(
            evaluation.positive_score,
            self.positive_evidence.replay.replay_error,
        ) or not _scores_match(
            evaluation.negative_score,
            self.negative_evidence.replay.replay_error,
        ):
            raise Stage3CertificateValidationError(
                "evaluation scores must match independent replay errors"
            )
        if (
            type(evaluation.positive_witness_replay_valid) is not bool  # noqa: E721
            or evaluation.positive_witness_replay_valid is not self.positive_evidence.replay.passed
            or type(evaluation.negative_witness_replay_valid) is not bool  # noqa: E721
            or evaluation.negative_witness_replay_valid is not self.negative_evidence.replay.passed
        ):
            raise Stage3CertificateValidationError(
                "evaluation witness flags must equal independent replay results"
            )

    def _validate_fail_closed_semantics(self) -> None:
        evaluation = self.evaluation
        evidence_failed = not (
            self.positive_evidence.replay.passed and self.negative_evidence.replay.passed
        )
        degradation_failed = (
            type(evaluation.degradation_known) is not bool  # noqa: E721
            or not evaluation.degradation_known
        )
        solver_failed = not all(
            solver_status_succeeded(status)
            for status in (
                evaluation.positive_solver_status,
                evaluation.negative_solver_status,
                evaluation.solver_status,
            )
        )
        ood_failed = (
            type(evaluation.out_of_distribution) is not bool  # noqa: E721
            or evaluation.out_of_distribution
        )
        actual_hash_failed = any(
            actual != expected
            for actual, expected in (
                (
                    self.positive_evidence.replay.actual_latent_sha256,
                    self.positive_evidence.replay.expected_latent_sha256,
                ),
                (
                    self.negative_evidence.replay.actual_latent_sha256,
                    self.negative_evidence.replay.expected_latent_sha256,
                ),
                (
                    self.positive_evidence.replay.actual_observation_sha256,
                    self.observation_sha256,
                ),
                (
                    self.negative_evidence.replay.actual_observation_sha256,
                    self.observation_sha256,
                ),
                (
                    self.positive_evidence.replay.actual_trace_sha256,
                    self.degradation_trace_sha256,
                ),
                (
                    self.negative_evidence.replay.actual_trace_sha256,
                    self.degradation_trace_sha256,
                ),
            )
        )
        must_fail_closed = any(
            (evidence_failed, degradation_failed, solver_failed, ood_failed, actual_hash_failed)
        )
        is_fail_closed = (
            evaluation.status is EvaluationStatus.FAIL_CLOSED
            and evaluation.state is EntailmentState.OUT_OF_MODEL
        )
        if must_fail_closed and not is_fail_closed:
            raise Stage3CertificateValidationError(
                "witness/degradation/hash/solver/OOD failure requires fail-closed OUT_OF_MODEL"
            )
        if evidence_failed and EvaluationFailureReason.INVALID_WITNESS not in (
            evaluation.failure_reasons
        ):
            raise Stage3CertificateValidationError(
                "failed replay must retain INVALID_WITNESS in evaluation"
            )
        if degradation_failed and EvaluationFailureReason.UNKNOWN_DEGRADATION not in (
            evaluation.failure_reasons
        ):
            raise Stage3CertificateValidationError(
                "unknown degradation must remain in evaluation failure reasons"
            )
        if solver_failed and EvaluationFailureReason.SOLVER_NOT_SUCCESSFUL not in (
            evaluation.failure_reasons
        ):
            raise Stage3CertificateValidationError(
                "solver failure must remain in evaluation failure reasons"
            )
        if ood_failed and EvaluationFailureReason.OUT_OF_DISTRIBUTION not in (
            evaluation.failure_reasons
        ):
            raise Stage3CertificateValidationError(
                "OOD failure must remain in evaluation failure reasons"
            )
        if evaluation.status is EvaluationStatus.SUCCESS and (
            evidence_failed
            or degradation_failed
            or solver_failed
            or ood_failed
            or actual_hash_failed
        ):
            raise Stage3CertificateValidationError(
                "successful certificate requires complete valid evidence"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the stable certificate representation without host paths."""

        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "evaluation": self.evaluation.to_dict(),
            "region": self.region.to_dict(),
            "predicate_id": self.predicate_id,
            "calibration_artifact_hash": self.calibration_artifact_hash,
            "observation_sha256": self.observation_sha256,
            "degradation_trace_sha256": self.degradation_trace_sha256,
            "positive_evidence": self.positive_evidence.to_dict(),
            "negative_evidence": self.negative_evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage3SyntheticCertificate:
        """Strictly reload and revalidate a serialized synthetic certificate."""

        fields = _strict_mapping(
            value,
            name="Stage3SyntheticCertificate",
            required={
                "schema_version",
                "scope",
                "evaluation",
                "region",
                "predicate_id",
                "calibration_artifact_hash",
                "observation_sha256",
                "degradation_trace_sha256",
                "positive_evidence",
                "negative_evidence",
            },
        )
        schema_version = _require_identifier(fields["schema_version"], "schema_version")
        try:
            scope = Stage3CertificateScope(_require_identifier(fields["scope"], "scope"))
            region = LesionRegion.from_dict(_as_mapping(fields["region"], "region"))
        except (TypeError, ValueError) as error:
            raise Stage3CertificateValidationError("invalid certificate scope or region") from error
        return cls(
            schema_version=schema_version,
            scope=scope,
            evaluation=_evaluation_from_dict(fields["evaluation"]),
            region=region,
            predicate_id=_require_identifier(fields["predicate_id"], "predicate_id"),
            calibration_artifact_hash=_require_sha256(
                fields["calibration_artifact_hash"], "calibration_artifact_hash"
            ),
            observation_sha256=_require_sha256(fields["observation_sha256"], "observation_sha256"),
            degradation_trace_sha256=_require_sha256(
                fields["degradation_trace_sha256"], "degradation_trace_sha256"
            ),
            positive_evidence=Stage3WitnessEvidence.from_dict(
                _as_mapping(fields["positive_evidence"], "positive_evidence")
            ),
            negative_evidence=Stage3WitnessEvidence.from_dict(
                _as_mapping(fields["negative_evidence"], "negative_evidence")
            ),
        )


__all__ = [
    "STAGE3_CERTIFICATE_SCHEMA_VERSION",
    "Stage3CertificateScope",
    "Stage3CertificateValidationError",
    "Stage3SyntheticCertificate",
    "Stage3WitnessEvidence",
    "WitnessSide",
    "calibration_artifact_sha256",
]
