"""Stage 3 lightweight calibration protocol and fail-closed score evaluation.

This module is deliberately limited to a deterministic toy/protocol baseline.  It
does not train a model, establish clinical performance, or replace the project's
positive/negative feasibility and replayable-witness contribution with conformal
prediction.  The order statistic below only freezes a small, auditable calibration
contract for later research stages.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, TypeVar

from retinaentail.entailment.states import (
    EntailmentState,
    solver_status_succeeded,
    state_from_feasibility,
)

CALIBRATION_SCHEMA_VERSION: Final = "3.0.0"
CALIBRATION_METHOD_VERSION: Final = "toy_conservative_order_statistic_v1"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_CALIBRATION_ARTIFACT_TOKEN: Final = object()
_T = TypeVar("_T")


class ProtocolScope(StrEnum):
    """Explicitly non-clinical scope of this calibration artifact."""

    TOY_PROTOCOL = "TOY_PROTOCOL"


class CalibrationLabel(StrEnum):
    """Known proposition label for a calibration sample."""

    LESION_PRESENT = "LESION_PRESENT"
    LESION_ABSENT = "LESION_ABSENT"
    UNKNOWN = "UNKNOWN"


class SampleSplit(StrEnum):
    """Splits accepted by the protocol boundary."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class CalibrationRunStatus(StrEnum):
    """Top-level status of threshold calibration."""

    SUCCESS = "SUCCESS"
    FAIL_CLOSED = "FAIL_CLOSED"


class CalibrationFailureReason(StrEnum):
    """Machine-readable reasons why no calibration threshold was issued."""

    EMPTY_CALIBRATION_SET = "EMPTY_CALIBRATION_SET"
    INVALID_ALPHA = "INVALID_ALPHA"
    INVALID_HASH = "INVALID_HASH"
    INVALID_SAMPLE = "INVALID_SAMPLE"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    DUPLICATE_SAMPLE_ID = "DUPLICATE_SAMPLE_ID"
    UNKNOWN_LABEL = "UNKNOWN_LABEL"
    NON_CALIBRATION_SPLIT = "NON_CALIBRATION_SPLIT"
    IDENTITY_LEAKAGE = "IDENTITY_LEAKAGE"
    NONFINITE_SCORE = "NONFINITE_SCORE"
    INVALID_SCORE_RANGE = "INVALID_SCORE_RANGE"
    MISSING_POSITIVE_SAMPLES = "MISSING_POSITIVE_SAMPLES"
    MISSING_NEGATIVE_SAMPLES = "MISSING_NEGATIVE_SAMPLES"
    INSUFFICIENT_POSITIVE_SAMPLES = "INSUFFICIENT_POSITIVE_SAMPLES"
    INSUFFICIENT_NEGATIVE_SAMPLES = "INSUFFICIENT_NEGATIVE_SAMPLES"


class EvaluationStatus(StrEnum):
    """Whether an evaluation passed every fail-closed prerequisite."""

    SUCCESS = "SUCCESS"
    FAIL_CLOSED = "FAIL_CLOSED"


class EvaluationFailureReason(StrEnum):
    """Retained internal causes for an externally OUT_OF_MODEL evaluation."""

    INVALID_INPUT = "INVALID_INPUT"
    INVALID_CALIBRATION = "INVALID_CALIBRATION"
    MISSING_THRESHOLD = "MISSING_THRESHOLD"
    NONFINITE_THRESHOLD = "NONFINITE_THRESHOLD"
    MISSING_SCORE = "MISSING_SCORE"
    NONFINITE_SCORE = "NONFINITE_SCORE"
    INVALID_SCORE_RANGE = "INVALID_SCORE_RANGE"
    UNKNOWN_DEGRADATION = "UNKNOWN_DEGRADATION"
    INVALID_WITNESS = "INVALID_WITNESS"
    SOLVER_NOT_SUCCESSFUL = "SOLVER_NOT_SUCCESSFUL"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    INVALID_HASH = "INVALID_HASH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationSample:
    """One explicit score pair used by the toy calibration protocol.

    Values are intentionally retained without constructor coercion.  The pure
    calibration function validates the complete batch and returns FAIL_CLOSED rather
    than allowing a partially usable calibration artifact.
    """

    sample_id: str
    split: SampleSplit | str
    label: CalibrationLabel | str
    positive_score: float | None
    negative_score: float | None
    source_image_id: str | None = None
    patient_id: str | None = None
    intervention_pair_id: str | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation without changing input values."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": _enum_or_value(self.split),
            "label": _enum_or_value(self.label),
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "source_image_id": self.source_image_id,
            "patient_id": self.patient_id,
            "intervention_pair_id": self.intervention_pair_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationArtifact:
    """Versioned, immutable thresholds and their complete calibration provenance."""

    alpha: float | None
    positive_threshold: float | None
    negative_threshold: float | None
    positive_sample_count: int
    negative_sample_count: int
    total_sample_count: int
    positive_order_index: int | None
    negative_order_index: int | None
    config_hash: str | None
    data_hash: str | None
    model_hash: str | None
    status: CalibrationRunStatus
    failure_reasons: tuple[CalibrationFailureReason, ...] = field(default_factory=tuple)
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    method_version: str = CALIBRATION_METHOD_VERSION
    protocol_scope: ProtocolScope = ProtocolScope.TOY_PROTOCOL
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CALIBRATION_ARTIFACT_TOKEN:
            raise TypeError("CalibrationArtifact can only be issued by calibrate_order_statistic")

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-facing artifact contract."""

        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "protocol_scope": self.protocol_scope.value,
            "alpha": self.alpha,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "positive_sample_count": self.positive_sample_count,
            "negative_sample_count": self.negative_sample_count,
            "total_sample_count": self.total_sample_count,
            "positive_order_index": self.positive_order_index,
            "negative_order_index": self.negative_order_index,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "status": self.status.value,
            "failure_reasons": [reason.value for reason in self.failure_reasons],
        }


_ISSUED_CALIBRATION_ARTIFACTS: dict[
    int,
    tuple[CalibrationArtifact, tuple[object, ...]],
] = {}


def _calibration_artifact_fingerprint(artifact: CalibrationArtifact) -> tuple[object, ...]:
    return (
        artifact.alpha,
        artifact.positive_threshold,
        artifact.negative_threshold,
        artifact.positive_sample_count,
        artifact.negative_sample_count,
        artifact.total_sample_count,
        artifact.positive_order_index,
        artifact.negative_order_index,
        artifact.config_hash,
        artifact.data_hash,
        artifact.model_hash,
        artifact.status,
        artifact.failure_reasons,
        artifact.schema_version,
        artifact.method_version,
        artifact.protocol_scope,
    )


def _issue_calibration_artifact(artifact: CalibrationArtifact) -> CalibrationArtifact:
    _ISSUED_CALIBRATION_ARTIFACTS[id(artifact)] = (
        artifact,
        _calibration_artifact_fingerprint(artifact),
    )
    return artifact


def calibration_artifact_validation_reasons(value: object) -> tuple[str, ...]:
    """Revalidate process issuance and immutable bindings before threshold use."""

    if not isinstance(value, CalibrationArtifact):
        return ("calibration artifact has an invalid type",)
    issued = _ISSUED_CALIBRATION_ARTIFACTS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("calibration artifact was not issued by calibrate_order_statistic",)
    try:
        if issued[1] != _calibration_artifact_fingerprint(value):
            return ("calibration artifact bindings changed after issuance",)
    except (AttributeError, TypeError, ValueError):
        return ("calibration artifact fields are invalid",)
    return ()


@dataclass(frozen=True, slots=True, kw_only=True)
class HypothesisEvaluation:
    """Untrusted inputs needed to evaluate both lesion propositions."""

    case_id: str
    lesion_class: str
    positive_score: float | None
    negative_score: float | None
    config_hash: str | None
    data_hash: str | None
    model_hash: str | None
    positive_solver_status: str | None = "success"
    negative_solver_status: str | None = "success"
    solver_status: str | None = "success"
    degradation_known: bool = True
    positive_witness_replay_valid: bool = True
    negative_witness_replay_valid: bool = True
    out_of_distribution: bool = False
    schema_version: str = CALIBRATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationResult:
    """Four-state decision plus retained internal fail-closed evidence."""

    case_id: str
    lesion_class: str
    alpha: float | None
    positive_score: float | None
    negative_score: float | None
    positive_threshold: float | None
    negative_threshold: float | None
    positive_feasible: bool | None
    negative_feasible: bool | None
    positive_sample_count: int
    negative_sample_count: int
    config_hash: str | None
    data_hash: str | None
    model_hash: str | None
    input_config_hash: str | None
    input_data_hash: str | None
    input_model_hash: str | None
    positive_solver_status: str | None
    negative_solver_status: str | None
    solver_status: str | None
    degradation_known: bool
    positive_witness_replay_valid: bool
    negative_witness_replay_valid: bool
    out_of_distribution: bool
    state: EntailmentState
    status: EvaluationStatus
    failure_reasons: tuple[EvaluationFailureReason, ...] = field(default_factory=tuple)
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    method_version: str = CALIBRATION_METHOD_VERSION
    protocol_scope: ProtocolScope = ProtocolScope.TOY_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.schema_version != CALIBRATION_SCHEMA_VERSION
            or self.method_version != CALIBRATION_METHOD_VERSION
            or self.protocol_scope is not ProtocolScope.TOY_PROTOCOL
        ):
            raise ValueError("evaluation result protocol metadata is invalid")
        if not isinstance(self.state, EntailmentState) or not isinstance(
            self.status, EvaluationStatus
        ):
            raise TypeError("evaluation result state/status are invalid")
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("evaluation result case_id must be non-empty")
        if not isinstance(self.lesion_class, str) or not self.lesion_class.strip():
            raise ValueError("evaluation result lesion_class must be non-empty")
        numeric_fields = (
            self.alpha,
            self.positive_score,
            self.negative_score,
            self.positive_threshold,
            self.negative_threshold,
        )
        if any(value is not None and not _finite_nonnegative(value) for value in numeric_fields):
            raise ValueError("evaluation result numeric fields must be finite and non-negative")
        if self.alpha is not None and not 0.0 < self.alpha < 1.0:
            raise ValueError("evaluation result alpha must lie strictly between zero and one")
        if not _plain_nonnegative_int(self.positive_sample_count) or not _plain_nonnegative_int(
            self.negative_sample_count
        ):
            raise ValueError("evaluation result sample counts must be non-negative integers")
        hashes = (
            self.config_hash,
            self.data_hash,
            self.model_hash,
            self.input_config_hash,
            self.input_data_hash,
            self.input_model_hash,
        )
        if any(value is not None and not _valid_hash(value) for value in hashes):
            raise ValueError("evaluation result hashes must be lowercase SHA-256 values or null")
        statuses = (
            self.positive_solver_status,
            self.negative_solver_status,
            self.solver_status,
        )
        if any(
            value is not None
            and (not isinstance(value, str) or not value.strip() or value != value.strip().lower())
            for value in statuses
        ):
            raise ValueError(
                "evaluation result solver statuses must be canonical lowercase strings"
            )
        flags = (
            self.degradation_known,
            self.positive_witness_replay_valid,
            self.negative_witness_replay_valid,
            self.out_of_distribution,
        )
        if any(type(value) is not bool for value in flags):  # noqa: E721
            raise TypeError("evaluation result flags must be exact booleans")
        feasibility = (self.positive_feasible, self.negative_feasible)
        if any(value is not None and type(value) is not bool for value in feasibility):  # noqa: E721
            raise TypeError("evaluation result feasibility values must be booleans or null")
        if not all(isinstance(reason, EvaluationFailureReason) for reason in self.failure_reasons):
            raise TypeError("evaluation result failure reasons are invalid")
        if len(set(self.failure_reasons)) != len(self.failure_reasons):
            raise ValueError("evaluation result failure reasons must be unique")

        expected_hashes = (self.config_hash, self.data_hash, self.model_hash)
        input_hashes = (self.input_config_hash, self.input_data_hash, self.input_model_hash)
        if self.status is EvaluationStatus.SUCCESS:
            if self.failure_reasons:
                raise ValueError("successful evaluation cannot retain failure reasons")
            if any(value is None for value in numeric_fields):
                raise ValueError("successful evaluation requires alpha, scores, and thresholds")
            if not all(_valid_hash(value) for value in hashes) or input_hashes != expected_hashes:
                raise ValueError("successful evaluation requires matching complete provenance")
            if self.positive_sample_count < 1 or self.negative_sample_count < 1:
                raise ValueError("successful evaluation requires calibration samples on both sides")
            if not all(solver_status_succeeded(value) for value in statuses):
                raise ValueError("successful evaluation requires recognized solver statuses")
            if (
                not self.degradation_known
                or not self.positive_witness_replay_valid
                or not self.negative_witness_replay_valid
                or self.out_of_distribution
            ):
                raise ValueError("successful evaluation requires all fail-closed flags to pass")
            if not all(isinstance(value, bool) for value in feasibility):
                raise ValueError("successful evaluation requires both feasibility values")
            assert self.positive_score is not None
            assert self.negative_score is not None
            assert self.positive_threshold is not None
            assert self.negative_threshold is not None
            expected_positive = self.positive_score <= self.positive_threshold
            expected_negative = self.negative_score <= self.negative_threshold
            if feasibility != (expected_positive, expected_negative):
                raise ValueError("evaluation feasibility conflicts with scores and thresholds")
            if self.state is not state_from_feasibility(expected_positive, expected_negative):
                raise ValueError("evaluation state conflicts with the feasibility truth table")
        else:
            if self.state is not EntailmentState.OUT_OF_MODEL or not self.failure_reasons:
                raise ValueError("fail-closed evaluation requires OUT_OF_MODEL and failure reasons")
            if feasibility != (None, None):
                raise ValueError("fail-closed evaluation cannot assert feasibility")
            required_reasons = (
                (
                    not self.degradation_known,
                    EvaluationFailureReason.UNKNOWN_DEGRADATION,
                ),
                (
                    not self.positive_witness_replay_valid
                    or not self.negative_witness_replay_valid,
                    EvaluationFailureReason.INVALID_WITNESS,
                ),
                (
                    not all(solver_status_succeeded(value) for value in statuses),
                    EvaluationFailureReason.SOLVER_NOT_SUCCESSFUL,
                ),
                (
                    self.out_of_distribution,
                    EvaluationFailureReason.OUT_OF_DISTRIBUTION,
                ),
            )
            if any(
                required and reason not in self.failure_reasons
                for required, reason in required_reasons
            ):
                raise ValueError("fail-closed evaluation is missing a required failure reason")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, provenance-complete decision record."""

        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "protocol_scope": self.protocol_scope.value,
            "case_id": self.case_id,
            "lesion_class": self.lesion_class,
            "alpha": self.alpha,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "positive_feasible": self.positive_feasible,
            "negative_feasible": self.negative_feasible,
            "positive_sample_count": self.positive_sample_count,
            "negative_sample_count": self.negative_sample_count,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "input_config_hash": self.input_config_hash,
            "input_data_hash": self.input_data_hash,
            "input_model_hash": self.input_model_hash,
            "positive_solver_status": self.positive_solver_status,
            "negative_solver_status": self.negative_solver_status,
            "solver_status": self.solver_status,
            "degradation_known": self.degradation_known,
            "positive_witness_replay_valid": self.positive_witness_replay_valid,
            "negative_witness_replay_valid": self.negative_witness_replay_valid,
            "out_of_distribution": self.out_of_distribution,
            "state": self.state.value,
            "status": self.status.value,
            "failure_reasons": [reason.value for reason in self.failure_reasons],
        }


def _enum_or_value(value: StrEnum | str) -> str:
    return value.value if isinstance(value, StrEnum) else value


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _plain_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0  # noqa: E721 - reject bool explicitly


def _plain_positive_int(value: object) -> bool:
    return type(value) is int and value >= 1  # noqa: E721 - reject bool explicitly


def _safe_nonnegative_float(value: object) -> float | None:
    if not _finite_nonnegative(value):
        return None
    assert isinstance(value, int | float) and not isinstance(value, bool)
    return float(value)


def _safe_alpha(value: object) -> float | None:
    result = _safe_nonnegative_float(value)
    return result if result is not None and 0.0 < result < 1.0 else None


def _safe_nonnegative_int(value: object) -> int:
    if not _plain_nonnegative_int(value):
        return 0
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _valid_order_index(value: object, sample_count: object) -> bool:
    if not _plain_positive_int(value) or not _plain_positive_int(sample_count):
        return False
    assert isinstance(value, int) and not isinstance(value, bool)
    assert isinstance(sample_count, int) and not isinstance(sample_count, bool)
    return value <= sample_count


def _safe_identifier(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and value.strip() else fallback


def _safe_status(value: object) -> str | None:
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _safe_bool(value: object, *, fallback: bool = False) -> bool:
    return value if type(value) is bool else fallback  # noqa: E721 - exact bool boundary


def _append_once(items: list[_T], value: _T) -> None:
    if value not in items:
        items.append(value)


def _as_label(value: object) -> CalibrationLabel | None:
    if not isinstance(value, str):
        return None
    try:
        return CalibrationLabel(value)
    except ValueError:
        return None


def _as_split(value: object) -> SampleSplit | None:
    if not isinstance(value, str):
        return None
    try:
        return SampleSplit(value)
    except ValueError:
        return None


def _identity_leaks(samples: Sequence[CalibrationSample]) -> bool:
    for field_name in ("source_image_id", "patient_id", "intervention_pair_id"):
        observed: dict[str, set[SampleSplit]] = {}
        for sample in samples:
            identifier = getattr(sample, field_name)
            split = _as_split(sample.split)
            if identifier is None or split is None or not isinstance(identifier, str):
                continue
            observed.setdefault(identifier, set()).add(split)
        if any(len(splits) > 1 for splits in observed.values()):
            return True
    return False


def _requested_order_index(sample_count: int, alpha: float) -> int:
    return max(1, math.ceil((sample_count + 1) * (1.0 - alpha)))


def _conservative_order_statistic(values: Sequence[float], rank: int) -> float:
    """Return an upper empirical order statistic and its one-based index.

    The caller computes ``ceil((n + 1) * (1 - alpha))``.  A rank beyond the
    observed sample count is not clipped: calibration fails closed instead of
    replacing the mathematically conservative ``+infinity`` order statistic with an
    unsupported finite threshold.  This remains a protocol baseline, not a coverage
    or clinical guarantee.
    """

    ordered = sorted(values)
    return ordered[rank - 1]


def calibrate_order_statistic(
    samples: Sequence[CalibrationSample],
    *,
    alpha: float,
    config_hash: str,
    data_hash: str,
    model_hash: str,
) -> CalibrationArtifact:
    """Calibrate both hypothesis thresholds or return a threshold-free failure.

    Only explicitly labelled ``calibration`` samples are eligible.  Unknown labels,
    cross-split identities, invalid scores, invalid provenance, and a missing
    proposition side invalidate the entire artifact rather than issuing one usable
    threshold.
    """

    try:
        sample_tuple = tuple(samples)
    except TypeError:
        sample_tuple = ()
        sequence_invalid = True
    else:
        sequence_invalid = False

    reasons: list[CalibrationFailureReason] = []
    if sequence_invalid:
        _append_once(reasons, CalibrationFailureReason.INVALID_SAMPLE)
    if not sample_tuple:
        _append_once(reasons, CalibrationFailureReason.EMPTY_CALIBRATION_SET)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, int | float)
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        _append_once(reasons, CalibrationFailureReason.INVALID_ALPHA)
    if not all(_valid_hash(value) for value in (config_hash, data_hash, model_hash)):
        _append_once(reasons, CalibrationFailureReason.INVALID_HASH)

    typed_samples = tuple(
        sample for sample in sample_tuple if isinstance(sample, CalibrationSample)
    )
    if len(typed_samples) != len(sample_tuple):
        _append_once(reasons, CalibrationFailureReason.INVALID_SAMPLE)
    if any(sample.schema_version != CALIBRATION_SCHEMA_VERSION for sample in typed_samples):
        _append_once(reasons, CalibrationFailureReason.UNSUPPORTED_SCHEMA_VERSION)

    sample_ids = [sample.sample_id for sample in typed_samples if isinstance(sample.sample_id, str)]
    if any(
        not isinstance(sample.sample_id, str) or not sample.sample_id.strip()
        for sample in typed_samples
    ):
        _append_once(reasons, CalibrationFailureReason.INVALID_SAMPLE)
    if len(sample_ids) != len(set(sample_ids)):
        _append_once(reasons, CalibrationFailureReason.DUPLICATE_SAMPLE_ID)

    for sample in typed_samples:
        identities = (
            sample.source_image_id,
            sample.patient_id,
            sample.intervention_pair_id,
        )
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in identities
        ):
            _append_once(reasons, CalibrationFailureReason.INVALID_SAMPLE)

    if _identity_leaks(typed_samples):
        _append_once(reasons, CalibrationFailureReason.IDENTITY_LEAKAGE)
    splits = tuple(_as_split(sample.split) for sample in typed_samples)
    if any(split is not SampleSplit.CALIBRATION for split in splits):
        _append_once(reasons, CalibrationFailureReason.NON_CALIBRATION_SPLIT)

    labels = tuple(_as_label(sample.label) for sample in typed_samples)
    if any(label is None or label is CalibrationLabel.UNKNOWN for label in labels):
        _append_once(reasons, CalibrationFailureReason.UNKNOWN_LABEL)

    scores = tuple(
        score
        for sample in typed_samples
        for score in (sample.positive_score, sample.negative_score)
    )
    if any(
        score is None
        or isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(float(score))
        for score in scores
    ):
        _append_once(reasons, CalibrationFailureReason.NONFINITE_SCORE)
    elif any(float(score) < 0.0 for score in scores if score is not None):
        _append_once(reasons, CalibrationFailureReason.INVALID_SCORE_RANGE)

    positive_values: list[float] = []
    negative_values: list[float] = []
    for sample, label in zip(typed_samples, labels, strict=True):
        if label is CalibrationLabel.LESION_PRESENT and _finite_nonnegative(sample.positive_score):
            assert sample.positive_score is not None
            positive_values.append(float(sample.positive_score))
        if label is CalibrationLabel.LESION_ABSENT and _finite_nonnegative(sample.negative_score):
            assert sample.negative_score is not None
            negative_values.append(float(sample.negative_score))
    if not positive_values:
        _append_once(reasons, CalibrationFailureReason.MISSING_POSITIVE_SAMPLES)
    if not negative_values:
        _append_once(reasons, CalibrationFailureReason.MISSING_NEGATIVE_SAMPLES)

    alpha_valid = (
        not isinstance(alpha, bool)
        and isinstance(alpha, int | float)
        and math.isfinite(float(alpha))
        and 0.0 < float(alpha) < 1.0
    )
    positive_rank = (
        _requested_order_index(len(positive_values), float(alpha))
        if positive_values and alpha_valid
        else None
    )
    negative_rank = (
        _requested_order_index(len(negative_values), float(alpha))
        if negative_values and alpha_valid
        else None
    )
    if positive_rank is not None and positive_rank > len(positive_values):
        _append_once(reasons, CalibrationFailureReason.INSUFFICIENT_POSITIVE_SAMPLES)
    if negative_rank is not None and negative_rank > len(negative_values):
        _append_once(reasons, CalibrationFailureReason.INSUFFICIENT_NEGATIVE_SAMPLES)

    if reasons:
        return _issue_calibration_artifact(
            CalibrationArtifact(
                alpha=float(alpha) if _finite_nonnegative(alpha) else None,
                positive_threshold=None,
                negative_threshold=None,
                positive_sample_count=len(positive_values),
                negative_sample_count=len(negative_values),
                total_sample_count=len(sample_tuple),
                positive_order_index=None,
                negative_order_index=None,
                config_hash=config_hash if _valid_hash(config_hash) else None,
                data_hash=data_hash if _valid_hash(data_hash) else None,
                model_hash=model_hash if _valid_hash(model_hash) else None,
                status=CalibrationRunStatus.FAIL_CLOSED,
                failure_reasons=tuple(reasons),
                _token=_CALIBRATION_ARTIFACT_TOKEN,
            )
        )

    clean_alpha = float(alpha)
    assert positive_rank is not None
    assert negative_rank is not None
    positive_threshold = _conservative_order_statistic(positive_values, positive_rank)
    negative_threshold = _conservative_order_statistic(negative_values, negative_rank)
    return _issue_calibration_artifact(
        CalibrationArtifact(
            alpha=clean_alpha,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            positive_sample_count=len(positive_values),
            negative_sample_count=len(negative_values),
            total_sample_count=len(sample_tuple),
            positive_order_index=positive_rank,
            negative_order_index=negative_rank,
            config_hash=config_hash,
            data_hash=data_hash,
            model_hash=model_hash,
            status=CalibrationRunStatus.SUCCESS,
            _token=_CALIBRATION_ARTIFACT_TOKEN,
        )
    )


def evaluate_hypotheses(
    evaluation: HypothesisEvaluation,
    calibration: CalibrationArtifact,
) -> EvaluationResult:
    """Apply calibrated thresholds and preserve every fail-closed reason.

    A score is feasible when it is no greater than its side's replay-error
    threshold.  Any failed prerequisite forces external ``OUT_OF_MODEL`` and leaves
    both feasibility flags unset.  A valid ``False, False`` truth-table result is
    also ``OUT_OF_MODEL`` but retains ``SUCCESS`` status and no failure reasons.
    """

    reasons: list[EvaluationFailureReason] = []
    if not isinstance(evaluation, HypothesisEvaluation) or not isinstance(
        calibration, CalibrationArtifact
    ):
        raise TypeError("evaluation and calibration must use the Stage 3 dataclass contracts")

    if (
        calibration_artifact_validation_reasons(calibration)
        or evaluation.schema_version != CALIBRATION_SCHEMA_VERSION
        or calibration.schema_version != CALIBRATION_SCHEMA_VERSION
        or calibration.method_version != CALIBRATION_METHOD_VERSION
        or calibration.protocol_scope is not ProtocolScope.TOY_PROTOCOL
        or calibration.status is not CalibrationRunStatus.SUCCESS
        or calibration.failure_reasons
    ):
        _append_once(reasons, EvaluationFailureReason.INVALID_CALIBRATION)
    counts_valid = (
        _plain_positive_int(calibration.positive_sample_count)
        and _plain_positive_int(calibration.negative_sample_count)
        and _plain_nonnegative_int(calibration.total_sample_count)
    )
    order_indices_valid = (
        counts_valid
        and _valid_order_index(
            calibration.positive_order_index,
            calibration.positive_sample_count,
        )
        and _valid_order_index(
            calibration.negative_order_index,
            calibration.negative_sample_count,
        )
    )
    if (
        calibration.alpha is None
        or isinstance(calibration.alpha, bool)
        or not isinstance(calibration.alpha, int | float)
        or not math.isfinite(float(calibration.alpha))
        or not 0.0 < float(calibration.alpha) < 1.0
        or not counts_valid
        or calibration.total_sample_count
        < calibration.positive_sample_count + calibration.negative_sample_count
        or not order_indices_valid
    ):
        _append_once(reasons, EvaluationFailureReason.INVALID_CALIBRATION)
    if not isinstance(evaluation.case_id, str) or not evaluation.case_id.strip():
        _append_once(reasons, EvaluationFailureReason.INVALID_INPUT)
    if not isinstance(evaluation.lesion_class, str) or not evaluation.lesion_class.strip():
        _append_once(reasons, EvaluationFailureReason.INVALID_INPUT)

    thresholds = (calibration.positive_threshold, calibration.negative_threshold)
    if any(value is None for value in thresholds):
        _append_once(reasons, EvaluationFailureReason.MISSING_THRESHOLD)
    elif any(not _finite_nonnegative(value) for value in thresholds):
        _append_once(reasons, EvaluationFailureReason.NONFINITE_THRESHOLD)

    scores = (evaluation.positive_score, evaluation.negative_score)
    if any(value is None for value in scores):
        _append_once(reasons, EvaluationFailureReason.MISSING_SCORE)
    elif any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in scores
    ):
        _append_once(reasons, EvaluationFailureReason.NONFINITE_SCORE)
    elif any(value is not None and float(value) < 0.0 for value in scores):
        _append_once(reasons, EvaluationFailureReason.INVALID_SCORE_RANGE)

    calibration_hashes = (
        calibration.config_hash,
        calibration.data_hash,
        calibration.model_hash,
    )
    evaluation_hashes = (evaluation.config_hash, evaluation.data_hash, evaluation.model_hash)
    if not all(_valid_hash(value) for value in calibration_hashes + evaluation_hashes):
        _append_once(reasons, EvaluationFailureReason.INVALID_HASH)
    elif calibration_hashes != evaluation_hashes:
        _append_once(reasons, EvaluationFailureReason.PROVENANCE_MISMATCH)

    if type(evaluation.degradation_known) is not bool or not evaluation.degradation_known:  # noqa: E721
        _append_once(reasons, EvaluationFailureReason.UNKNOWN_DEGRADATION)
    witness_flags = (
        evaluation.positive_witness_replay_valid,
        evaluation.negative_witness_replay_valid,
    )
    if any(type(value) is not bool or not value for value in witness_flags):  # noqa: E721
        _append_once(reasons, EvaluationFailureReason.INVALID_WITNESS)
    if not all(
        solver_status_succeeded(status)
        for status in (
            evaluation.positive_solver_status,
            evaluation.negative_solver_status,
            evaluation.solver_status,
        )
    ):
        _append_once(reasons, EvaluationFailureReason.SOLVER_NOT_SUCCESSFUL)
    out_of_distribution: object = evaluation.out_of_distribution
    if type(out_of_distribution) is not bool:  # noqa: E721
        _append_once(reasons, EvaluationFailureReason.INVALID_INPUT)
    elif out_of_distribution:
        _append_once(reasons, EvaluationFailureReason.OUT_OF_DISTRIBUTION)

    if reasons:
        return _evaluation_result(
            evaluation=evaluation,
            calibration=calibration,
            state=EntailmentState.OUT_OF_MODEL,
            positive_feasible=None,
            negative_feasible=None,
            status=EvaluationStatus.FAIL_CLOSED,
            failure_reasons=tuple(reasons),
        )

    positive_score = float(evaluation.positive_score)  # type: ignore[arg-type]
    negative_score = float(evaluation.negative_score)  # type: ignore[arg-type]
    positive_threshold = float(calibration.positive_threshold)  # type: ignore[arg-type]
    negative_threshold = float(calibration.negative_threshold)  # type: ignore[arg-type]
    positive_feasible = positive_score <= positive_threshold
    negative_feasible = negative_score <= negative_threshold
    return _evaluation_result(
        evaluation=evaluation,
        calibration=calibration,
        state=state_from_feasibility(positive_feasible, negative_feasible),
        positive_feasible=positive_feasible,
        negative_feasible=negative_feasible,
        status=EvaluationStatus.SUCCESS,
        failure_reasons=(),
    )


def _evaluation_result(
    *,
    evaluation: HypothesisEvaluation,
    calibration: CalibrationArtifact,
    state: EntailmentState,
    positive_feasible: bool | None,
    negative_feasible: bool | None,
    status: EvaluationStatus,
    failure_reasons: tuple[EvaluationFailureReason, ...],
) -> EvaluationResult:
    return EvaluationResult(
        case_id=_safe_identifier(evaluation.case_id, "<invalid-case-id>"),
        lesion_class=_safe_identifier(evaluation.lesion_class, "<invalid-lesion-class>"),
        alpha=_safe_alpha(calibration.alpha),
        positive_score=_safe_nonnegative_float(evaluation.positive_score),
        negative_score=_safe_nonnegative_float(evaluation.negative_score),
        positive_threshold=_safe_nonnegative_float(calibration.positive_threshold),
        negative_threshold=_safe_nonnegative_float(calibration.negative_threshold),
        positive_feasible=positive_feasible,
        negative_feasible=negative_feasible,
        positive_sample_count=_safe_nonnegative_int(calibration.positive_sample_count),
        negative_sample_count=_safe_nonnegative_int(calibration.negative_sample_count),
        config_hash=calibration.config_hash if _valid_hash(calibration.config_hash) else None,
        data_hash=calibration.data_hash if _valid_hash(calibration.data_hash) else None,
        model_hash=calibration.model_hash if _valid_hash(calibration.model_hash) else None,
        input_config_hash=(evaluation.config_hash if _valid_hash(evaluation.config_hash) else None),
        input_data_hash=evaluation.data_hash if _valid_hash(evaluation.data_hash) else None,
        input_model_hash=evaluation.model_hash if _valid_hash(evaluation.model_hash) else None,
        positive_solver_status=_safe_status(evaluation.positive_solver_status),
        negative_solver_status=_safe_status(evaluation.negative_solver_status),
        solver_status=_safe_status(evaluation.solver_status),
        degradation_known=_safe_bool(evaluation.degradation_known),
        positive_witness_replay_valid=_safe_bool(evaluation.positive_witness_replay_valid),
        negative_witness_replay_valid=_safe_bool(evaluation.negative_witness_replay_valid),
        out_of_distribution=_safe_bool(evaluation.out_of_distribution),
        state=state,
        status=status,
        failure_reasons=failure_reasons,
    )


__all__ = [
    "CALIBRATION_METHOD_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationArtifact",
    "CalibrationFailureReason",
    "CalibrationLabel",
    "CalibrationRunStatus",
    "CalibrationSample",
    "EvaluationFailureReason",
    "EvaluationResult",
    "EvaluationStatus",
    "HypothesisEvaluation",
    "ProtocolScope",
    "SampleSplit",
    "calibrate_order_statistic",
    "calibration_artifact_validation_reasons",
    "evaluate_hypotheses",
]
