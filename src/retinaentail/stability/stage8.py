"""Fail-closed Stage 8 repeatability and dual-witness stability audit.

This module is deliberately filesystem-free.  A Stage 8 authorization gate must
establish the authority to produce the observations; this module then checks the
frozen Stage 7 baseline against the exact preregistered case/run Cartesian product.
Contract drift is never ignored: an affected transition is normalized to
``OUT_OF_MODEL`` and the aggregate status becomes ``FAIL_CLOSED``.  A valid but
unstable rerun remains an observed audit result rather than being rewritten.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from retinaentail.entailment.states import EntailmentState, state_from_feasibility
from retinaentail.evaluation.metrics import BinomialProportion, wilson_score_proportion

STAGE8_SCHEMA_VERSION: Final = "8.0.0"
STAGE8_BASELINE_RUN_ID: Final = "stage7_locked_baseline"
STAGE8_EXPECTED_SPLIT: Final = "test"
STAGE8_INTERVAL_METHOD: Final = "wilson_score"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_SEED: Final = (2**32) - 1
_MAX_SAMPLE_COUNT: Final = (2**31) - 1


class Stage8Scope(StrEnum):
    """Explicit provenance scope; synthetic evidence never authorizes real data."""

    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    APPROVED_REAL_DATA = "APPROVED_REAL_DATA"


class Stage8AuditStatus(StrEnum):
    """Whether every observation honored the frozen preregistration contract."""

    SUCCESS = "SUCCESS"
    FAIL_CLOSED = "FAIL_CLOSED"


class Stage8StabilityMetric(StrEnum):
    """Fixed, preregistered descriptive metrics; none is a clinical claim."""

    STATE_STABLE = "STATE_STABLE"
    FAIL_CLOSED_STATUS_STABLE = "FAIL_CLOSED_STATUS_STABLE"
    OUT_OF_DISTRIBUTION_STATUS_STABLE = "OUT_OF_DISTRIBUTION_STATUS_STABLE"
    POSITIVE_WITNESS_REPLAY_STABLE = "POSITIVE_WITNESS_REPLAY_STABLE"
    NEGATIVE_WITNESS_REPLAY_STABLE = "NEGATIVE_WITNESS_REPLAY_STABLE"
    DUAL_WITNESS_REPLAY_STABLE = "DUAL_WITNESS_REPLAY_STABLE"
    POSITIVE_WITNESS_HASH_STABLE = "POSITIVE_WITNESS_HASH_STABLE"
    NEGATIVE_WITNESS_HASH_STABLE = "NEGATIVE_WITNESS_HASH_STABLE"
    DUAL_WITNESS_HASH_STABLE = "DUAL_WITNESS_HASH_STABLE"


class Stage8EventMetric(StrEnum):
    """Repeat-run event strata reported separately from stability agreement."""

    REPEAT_FAIL_CLOSED = "REPEAT_FAIL_CLOSED"
    REPEAT_OUT_OF_DISTRIBUTION = "REPEAT_OUT_OF_DISTRIBUTION"
    REPEAT_OUT_OF_MODEL = "REPEAT_OUT_OF_MODEL"
    REPEAT_DUAL_WITNESS_REPLAY_PASSED = "REPEAT_DUAL_WITNESS_REPLAY_PASSED"


class Stage8FailureReason(StrEnum):
    """Controlled causes for blocking or normalizing an observation."""

    EMPTY_BASELINE = "EMPTY_BASELINE"
    EMPTY_REPEAT_SET = "EMPTY_REPEAT_SET"
    INVALID_RECORD = "INVALID_RECORD"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    INVALID_SCHEMA_VERSION = "INVALID_SCHEMA_VERSION"
    DUPLICATE_BASELINE_CASE = "DUPLICATE_BASELINE_CASE"
    DUPLICATE_REPEAT_OBSERVATION = "DUPLICATE_REPEAT_OBSERVATION"
    MISSING_BASELINE_CASE = "MISSING_BASELINE_CASE"
    MISSING_REPEAT_OBSERVATION = "MISSING_REPEAT_OBSERVATION"
    UNDECLARED_BASELINE_CASE = "UNDECLARED_BASELINE_CASE"
    UNDECLARED_REPEAT_CASE = "UNDECLARED_REPEAT_CASE"
    UNDECLARED_RUN = "UNDECLARED_RUN"
    DATASET_ID_MISMATCH = "DATASET_ID_MISMATCH"
    DATASET_VERSION_MISMATCH = "DATASET_VERSION_MISMATCH"
    SPLIT_MISMATCH = "SPLIT_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    RUN_SEED_MISMATCH = "RUN_SEED_MISMATCH"
    DEGRADATION_REPLAY_ID_MISMATCH = "DEGRADATION_REPLAY_ID_MISMATCH"
    DEGRADATION_REPLAY_SET_HASH_MISMATCH = "DEGRADATION_REPLAY_SET_HASH_MISMATCH"
    CASE_DEGRADATION_TRACE_MISMATCH = "CASE_DEGRADATION_TRACE_MISMATCH"
    INVALID_STATE = "INVALID_STATE"
    INVALID_FLAG = "INVALID_FLAG"
    INVALID_NUMERIC_VALUE = "INVALID_NUMERIC_VALUE"
    INVALID_SEED = "INVALID_SEED"
    INVALID_SAMPLE_COUNT = "INVALID_SAMPLE_COUNT"
    STATE_SCORE_MISMATCH = "STATE_SCORE_MISMATCH"
    LOCKED_ALPHA_MISMATCH = "LOCKED_ALPHA_MISMATCH"
    LOCKED_POSITIVE_THRESHOLD_MISMATCH = "LOCKED_POSITIVE_THRESHOLD_MISMATCH"
    LOCKED_NEGATIVE_THRESHOLD_MISMATCH = "LOCKED_NEGATIVE_THRESHOLD_MISMATCH"
    LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH = "LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH"
    LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH = "LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH"
    LOCKED_CONFIG_HASH_MISMATCH = "LOCKED_CONFIG_HASH_MISMATCH"
    LOCKED_DATA_HASH_MISMATCH = "LOCKED_DATA_HASH_MISMATCH"
    LOCKED_MODEL_HASH_MISMATCH = "LOCKED_MODEL_HASH_MISMATCH"
    STAGE7_PACKAGE_HASH_MISMATCH = "STAGE7_PACKAGE_HASH_MISMATCH"
    PROVENANCE_HASH_MISMATCH = "PROVENANCE_HASH_MISMATCH"
    INVALID_WITNESS_HASH = "INVALID_WITNESS_HASH"
    FAIL_CLOSED_SEMANTICS_MISMATCH = "FAIL_CLOSED_SEMANTICS_MISMATCH"
    UNKNOWN_DEGRADATION = "UNKNOWN_DEGRADATION"
    SOLVER_FAILED = "SOLVER_FAILED"
    INVALID_WITNESS_REPLAY = "INVALID_WITNESS_REPLAY"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"


_RUNTIME_FAIL_CLOSED_REASONS: Final = frozenset(
    {
        Stage8FailureReason.UNKNOWN_DEGRADATION,
        Stage8FailureReason.SOLVER_FAILED,
        Stage8FailureReason.INVALID_WITNESS_REPLAY,
        Stage8FailureReason.OUT_OF_DISTRIBUTION,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8ReplaySpec:
    """One preregistered rerun and its frozen ordered degradation-set digest."""

    run_id: str
    seed: int
    degradation_replay_id: str
    degradation_replay_set_sha256: str
    schema_version: str = STAGE8_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 replay-spec schema version")
        if not all(_valid_identifier(value) for value in (self.run_id, self.degradation_replay_id)):
            raise ValueError("replay identifiers must be non-empty canonical strings")
        if self.run_id == STAGE8_BASELINE_RUN_ID:
            raise ValueError("a repeat run cannot use the reserved Stage 7 baseline id")
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:  # noqa: E721
            raise ValueError("replay seed must be a uint32 integer")
        if not _valid_sha256(self.degradation_replay_set_sha256):
            raise ValueError("degradation replay-set hash must be lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seed": self.seed,
            "degradation_replay_id": self.degradation_replay_id,
            "degradation_replay_set_sha256": self.degradation_replay_set_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8Preregistration:
    """Frozen case set, run set, locks, and metrics declared before rerunning."""

    dataset_id: str
    dataset_version: str
    scope: Stage8Scope
    inclusion_case_ids: tuple[str, ...]
    replay_specs: tuple[Stage8ReplaySpec, ...]
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    config_hash: str
    data_hash: str
    model_hash: str
    stage7_package_sha256: str
    provenance_sha256: str
    confidence_level: float = 0.95
    stability_metrics: tuple[Stage8StabilityMetric, ...] = tuple(Stage8StabilityMetric)
    split: str = STAGE8_EXPECTED_SPLIT
    inclusion_rule: str = "EXACT_FROZEN_STAGE7_TEST_CASE_SET"
    schema_version: str = STAGE8_SCHEMA_VERSION
    clinical_claim: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 preregistration schema version")
        if not all(_valid_identifier(value) for value in (self.dataset_id, self.dataset_version)):
            raise ValueError("dataset identity must contain canonical non-empty strings")
        if not isinstance(self.scope, Stage8Scope):
            raise TypeError("scope must use Stage8Scope")
        if self.split != STAGE8_EXPECTED_SPLIT:
            raise ValueError("Stage 8 preregistration is locked to the test split")
        if self.inclusion_rule != "EXACT_FROZEN_STAGE7_TEST_CASE_SET":
            raise ValueError("unsupported Stage 8 inclusion rule")
        if self.clinical_claim is not False:
            raise ValueError("Stage 8 repeatability reports cannot make a clinical claim")
        cases = tuple(self.inclusion_case_ids)
        if not cases or any(not _valid_identifier(value) for value in cases):
            raise ValueError("the frozen inclusion case set must be non-empty and valid")
        if len(set(cases)) != len(cases):
            raise ValueError("the frozen inclusion case set cannot contain duplicates")
        specs = tuple(self.replay_specs)
        if not specs or any(not isinstance(value, Stage8ReplaySpec) for value in specs):
            raise ValueError("at least one versioned replay specification is required")
        for value in specs:
            value.__post_init__()
        if len({value.run_id for value in specs}) != len(specs):
            raise ValueError("replay run ids must be unique")
        if not _finite_between(self.alpha, lower=0.0, upper=1.0, strict=True):
            raise ValueError("alpha must be finite and strictly between zero and one")
        if not all(
            _finite_between(value, lower=0.0, upper=None, strict=False)
            for value in (self.positive_threshold, self.negative_threshold)
        ):
            raise ValueError("locked thresholds must be finite and non-negative")
        if not all(
            type(value) is int and 0 < value <= _MAX_SAMPLE_COUNT  # noqa: E721
            for value in (self.positive_sample_count, self.negative_sample_count)
        ):
            raise ValueError("locked calibration sample counts must be positive int32 integers")
        hashes = (
            self.config_hash,
            self.data_hash,
            self.model_hash,
            self.stage7_package_sha256,
            self.provenance_sha256,
        )
        if not all(_valid_sha256(value) for value in hashes):
            raise ValueError("all Stage 8 preregistration hashes must be lowercase SHA-256")
        if not _finite_between(self.confidence_level, lower=0.0, upper=1.0, strict=True):
            raise ValueError("confidence level must be finite and strictly between zero and one")
        metrics = tuple(self.stability_metrics)
        if metrics != tuple(Stage8StabilityMetric):
            raise ValueError("Stage 8 requires the complete, ordered stability metric set")
        object.__setattr__(self, "inclusion_case_ids", cases)
        object.__setattr__(self, "replay_specs", specs)
        object.__setattr__(self, "stability_metrics", metrics)

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "scope": self.scope.value,
            "split": self.split,
            "inclusion_rule": self.inclusion_rule,
            "inclusion_case_ids": list(self.inclusion_case_ids),
            "replay_specs": [value.to_dict() for value in self.replay_specs],
            "alpha": float(self.alpha),
            "positive_threshold": float(self.positive_threshold),
            "negative_threshold": float(self.negative_threshold),
            "positive_sample_count": self.positive_sample_count,
            "negative_sample_count": self.negative_sample_count,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "stage7_package_sha256": self.stage7_package_sha256,
            "provenance_sha256": self.provenance_sha256,
            "confidence_level": float(self.confidence_level),
            "stability_metrics": [value.value for value in self.stability_metrics],
            "clinical_claim": self.clinical_claim,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8CaseObservation:
    """Untrusted, immutable wrapper for a baseline or one preregistered rerun.

    Runtime validation intentionally occurs in :func:`audit_stage8_repeatability` so
    malformed values (including NaN) produce a JSON-safe fail-closed transition.
    """

    case_id: object
    run_id: object
    dataset_id: object
    dataset_version: object
    split: object
    scope: object
    seed: object
    degradation_replay_id: object
    degradation_replay_set_sha256: object
    case_degradation_trace_sha256: object
    state: object
    fail_closed: object
    out_of_distribution: object
    alpha: object
    positive_score: object
    negative_score: object
    positive_threshold: object
    negative_threshold: object
    positive_sample_count: object
    negative_sample_count: object
    config_hash: object
    data_hash: object
    model_hash: object
    stage7_package_sha256: object
    provenance_sha256: object
    degradation_known: object
    solver_succeeded: object
    positive_witness_sha256: object
    negative_witness_sha256: object
    positive_witness_replay_valid: object
    negative_witness_replay_valid: object
    schema_version: object = STAGE8_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8CaseStability:
    """One baseline-to-repeat transition after fail-closed normalization."""

    case_id: str
    run_id: str
    baseline_state: EntailmentState
    repeat_state: EntailmentState
    baseline_fail_closed: bool
    repeat_fail_closed: bool
    baseline_out_of_distribution: bool
    repeat_out_of_distribution: bool
    state_stable: bool
    fail_closed_status_stable: bool
    out_of_distribution_status_stable: bool
    positive_witness_replay_stable: bool
    negative_witness_replay_stable: bool
    dual_witness_replay_stable: bool
    baseline_positive_witness_replay_valid: bool
    baseline_negative_witness_replay_valid: bool
    repeat_positive_witness_replay_valid: bool
    repeat_negative_witness_replay_valid: bool
    baseline_positive_witness_sha256: str | None
    baseline_negative_witness_sha256: str | None
    repeat_positive_witness_sha256: str | None
    repeat_negative_witness_sha256: str | None
    positive_witness_hash_stable: bool
    negative_witness_hash_stable: bool
    dual_witness_hash_stable: bool
    failure_reasons: tuple[Stage8FailureReason, ...] = field(default_factory=tuple)
    schema_version: str = STAGE8_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 case-stability schema version")
        if not all(_valid_identifier(value) for value in (self.case_id, self.run_id)):
            raise ValueError("case-stability identifiers must be canonical and non-empty")
        if not isinstance(self.baseline_state, EntailmentState) or not isinstance(
            self.repeat_state, EntailmentState
        ):
            raise TypeError("case-stability states must use EntailmentState")
        if any(not isinstance(reason, Stage8FailureReason) for reason in self.failure_reasons):
            raise TypeError("case-stability failure reasons must use Stage8FailureReason")
        if len(set(self.failure_reasons)) != len(self.failure_reasons):
            raise ValueError("case-stability failure reasons must be unique")
        contract_valid = not any(
            reason not in _RUNTIME_FAIL_CLOSED_REASONS for reason in self.failure_reasons
        )
        flags = (
            self.baseline_fail_closed,
            self.repeat_fail_closed,
            self.baseline_out_of_distribution,
            self.repeat_out_of_distribution,
            self.state_stable,
            self.fail_closed_status_stable,
            self.out_of_distribution_status_stable,
            self.positive_witness_replay_stable,
            self.negative_witness_replay_stable,
            self.dual_witness_replay_stable,
            self.baseline_positive_witness_replay_valid,
            self.baseline_negative_witness_replay_valid,
            self.repeat_positive_witness_replay_valid,
            self.repeat_negative_witness_replay_valid,
            self.positive_witness_hash_stable,
            self.negative_witness_hash_stable,
            self.dual_witness_hash_stable,
        )
        if any(type(value) is not bool for value in flags):  # noqa: E721
            raise TypeError("case-stability flags must be exact booleans")
        witness_hashes = (
            self.baseline_positive_witness_sha256,
            self.baseline_negative_witness_sha256,
            self.repeat_positive_witness_sha256,
            self.repeat_negative_witness_sha256,
        )
        if any(value is not None and not _valid_sha256(value) for value in witness_hashes):
            raise ValueError("case-stability witness hashes must be lowercase SHA-256 or null")
        if self.baseline_fail_closed and self.baseline_state is not EntailmentState.OUT_OF_MODEL:
            raise ValueError("a fail-closed baseline must expose OUT_OF_MODEL")
        if self.repeat_fail_closed and self.repeat_state is not EntailmentState.OUT_OF_MODEL:
            raise ValueError("a fail-closed repeat must expose OUT_OF_MODEL")
        if self.baseline_out_of_distribution and not self.baseline_fail_closed:
            raise ValueError("an OOD baseline must be fail closed")
        if self.repeat_out_of_distribution and not self.repeat_fail_closed:
            raise ValueError("an OOD repeat must be fail closed")
        if self.state_stable is not (contract_valid and self.baseline_state is self.repeat_state):
            raise ValueError("state-stability flag conflicts with the transition")
        if self.fail_closed_status_stable is not (
            contract_valid and self.baseline_fail_closed is self.repeat_fail_closed
        ):
            raise ValueError("fail-closed stability flag conflicts with case statuses")
        if self.out_of_distribution_status_stable is not (
            contract_valid and self.baseline_out_of_distribution is self.repeat_out_of_distribution
        ):
            raise ValueError("OOD stability flag conflicts with case statuses")
        positive_replay_stable = (
            contract_valid
            and self.baseline_positive_witness_replay_valid
            and self.repeat_positive_witness_replay_valid
        )
        negative_replay_stable = (
            contract_valid
            and self.baseline_negative_witness_replay_valid
            and self.repeat_negative_witness_replay_valid
        )
        if self.positive_witness_replay_stable is not positive_replay_stable:
            raise ValueError("positive witness replay stability is inconsistent")
        if self.negative_witness_replay_stable is not negative_replay_stable:
            raise ValueError("negative witness replay stability is inconsistent")
        if self.dual_witness_replay_stable is not (
            positive_replay_stable and negative_replay_stable
        ):
            raise ValueError("dual witness replay stability is inconsistent")
        positive_hash_stable = (
            contract_valid
            and self.baseline_positive_witness_sha256 is not None
            and self.baseline_positive_witness_sha256 == self.repeat_positive_witness_sha256
            and self.baseline_positive_witness_replay_valid
            is self.repeat_positive_witness_replay_valid
        )
        negative_hash_stable = (
            contract_valid
            and self.baseline_negative_witness_sha256 is not None
            and self.baseline_negative_witness_sha256 == self.repeat_negative_witness_sha256
            and self.baseline_negative_witness_replay_valid
            is self.repeat_negative_witness_replay_valid
        )
        if self.positive_witness_hash_stable is not positive_hash_stable:
            raise ValueError("positive witness hash stability is inconsistent")
        if self.negative_witness_hash_stable is not negative_hash_stable:
            raise ValueError("negative witness hash stability is inconsistent")
        if self.dual_witness_hash_stable is not (positive_hash_stable and negative_hash_stable):
            raise ValueError("dual witness hash stability is inconsistent")
        if not contract_valid and (
            self.repeat_state is not EntailmentState.OUT_OF_MODEL or not self.repeat_fail_closed
        ):
            raise ValueError("contract failure must normalize the repeat OUT_OF_MODEL")

    @property
    def transition(self) -> str:
        return f"{self.baseline_state.value}->{self.repeat_state.value}"

    @property
    def status(self) -> Stage8AuditStatus:
        return (
            Stage8AuditStatus.FAIL_CLOSED
            if self.failure_reasons or self.repeat_fail_closed
            else Stage8AuditStatus.SUCCESS
        )

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "baseline_state": self.baseline_state.value,
            "repeat_state": self.repeat_state.value,
            "transition": self.transition,
            "baseline_fail_closed": self.baseline_fail_closed,
            "repeat_fail_closed": self.repeat_fail_closed,
            "baseline_out_of_distribution": self.baseline_out_of_distribution,
            "repeat_out_of_distribution": self.repeat_out_of_distribution,
            "state_stable": self.state_stable,
            "fail_closed_status_stable": self.fail_closed_status_stable,
            "out_of_distribution_status_stable": self.out_of_distribution_status_stable,
            "positive_witness_replay_stable": self.positive_witness_replay_stable,
            "negative_witness_replay_stable": self.negative_witness_replay_stable,
            "dual_witness_replay_stable": self.dual_witness_replay_stable,
            "baseline_positive_witness_replay_valid": (self.baseline_positive_witness_replay_valid),
            "baseline_negative_witness_replay_valid": (self.baseline_negative_witness_replay_valid),
            "repeat_positive_witness_replay_valid": (self.repeat_positive_witness_replay_valid),
            "repeat_negative_witness_replay_valid": (self.repeat_negative_witness_replay_valid),
            "baseline_positive_witness_sha256": self.baseline_positive_witness_sha256,
            "baseline_negative_witness_sha256": self.baseline_negative_witness_sha256,
            "repeat_positive_witness_sha256": self.repeat_positive_witness_sha256,
            "repeat_negative_witness_sha256": self.repeat_negative_witness_sha256,
            "positive_witness_hash_stable": self.positive_witness_hash_stable,
            "negative_witness_hash_stable": self.negative_witness_hash_stable,
            "dual_witness_hash_stable": self.dual_witness_hash_stable,
            "failure_reasons": [value.value for value in self.failure_reasons],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8StabilityReport:
    """Complete path-free Stage 8 repeatability result with Wilson intervals."""

    preregistration: Stage8Preregistration
    status: Stage8AuditStatus
    input_baseline_count: int
    input_repeat_count: int
    expected_transition_count: int
    rejected_record_count: int
    case_transitions: tuple[Stage8CaseStability, ...]
    transition_matrix: Mapping[str, int]
    metric_intervals: Mapping[Stage8StabilityMetric, BinomialProportion]
    event_intervals: Mapping[Stage8EventMetric, BinomialProportion]
    failure_reason_counts: Mapping[Stage8FailureReason, int]
    rejected_reason_counts: Mapping[Stage8FailureReason, int]
    schema_version: str = STAGE8_SCHEMA_VERSION
    interval_method: str = STAGE8_INTERVAL_METHOD
    clinical_claim: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 report schema version")
        if not isinstance(self.preregistration, Stage8Preregistration):
            raise TypeError("report requires a Stage8Preregistration")
        self.preregistration.__post_init__()
        if not isinstance(self.status, Stage8AuditStatus):
            raise TypeError("report status must use Stage8AuditStatus")
        if self.interval_method != STAGE8_INTERVAL_METHOD or self.clinical_claim is not False:
            raise ValueError("Stage 8 report metadata is invalid")
        counts = (
            self.input_baseline_count,
            self.input_repeat_count,
            self.expected_transition_count,
            self.rejected_record_count,
        )
        if not all(type(value) is int and value >= 0 for value in counts):  # noqa: E721
            raise ValueError("report counts must be non-negative integers")
        expected = len(self.preregistration.inclusion_case_ids) * len(
            self.preregistration.replay_specs
        )
        if self.expected_transition_count != expected or len(self.case_transitions) != expected:
            raise ValueError("case transitions must cover the exact preregistered Cartesian set")
        matrix_keys = _transition_keys()
        if set(self.transition_matrix) != set(matrix_keys):
            raise ValueError("transition matrix must contain all sixteen four-state cells")
        if any(type(value) is not int or value < 0 for value in self.transition_matrix.values()):
            raise ValueError("transition matrix values must be non-negative integers")
        if sum(self.transition_matrix.values()) != expected:
            raise ValueError("transition matrix must partition all expected transitions")
        recomputed_matrix = Counter(item.transition for item in self.case_transitions)
        for item in self.case_transitions:
            item.__post_init__()
        if any(self.transition_matrix[key] != recomputed_matrix[key] for key in matrix_keys):
            raise ValueError("transition matrix conflicts with per-case transitions")
        if set(self.metric_intervals) != set(Stage8StabilityMetric):
            raise ValueError("metric intervals must contain every preregistered metric")
        metric_counts = _metric_counts(self.case_transitions)
        for metric in Stage8StabilityMetric:
            expected_interval = wilson_score_proportion(
                metric_counts[metric],
                expected,
                confidence_level=self.preregistration.confidence_level,
            )
            if self.metric_intervals[metric] != expected_interval:
                raise ValueError("metric interval conflicts with recomputed Wilson interval")
        if set(self.event_intervals) != set(Stage8EventMetric):
            raise ValueError("event intervals must contain every controlled repeat event")
        event_counts = _event_counts(self.case_transitions)
        for event in Stage8EventMetric:
            expected_interval = wilson_score_proportion(
                event_counts[event],
                expected,
                confidence_level=self.preregistration.confidence_level,
            )
            if self.event_intervals[event] != expected_interval:
                raise ValueError("event interval conflicts with recomputed Wilson interval")
        _validate_reason_mapping(self.failure_reason_counts)
        _validate_reason_mapping(self.rejected_reason_counts)
        recomputed_failures: Counter[Stage8FailureReason] = Counter()
        for item in self.case_transitions:
            recomputed_failures.update(set(item.failure_reasons))
        if dict(recomputed_failures) != dict(self.failure_reason_counts):
            raise ValueError("failure reason counts conflict with per-case transitions")
        blocked = bool(self.rejected_reason_counts) or any(
            reason not in _RUNTIME_FAIL_CLOSED_REASONS for reason in self.failure_reason_counts
        )
        if self.status is not (
            Stage8AuditStatus.FAIL_CLOSED if blocked else Stage8AuditStatus.SUCCESS
        ):
            raise ValueError("report status conflicts with contract failure reasons")
        object.__setattr__(self, "case_transitions", tuple(self.case_transitions))
        object.__setattr__(
            self,
            "transition_matrix",
            MappingProxyType({key: self.transition_matrix[key] for key in matrix_keys}),
        )
        object.__setattr__(
            self,
            "metric_intervals",
            MappingProxyType(
                {metric: self.metric_intervals[metric] for metric in Stage8StabilityMetric}
            ),
        )
        object.__setattr__(
            self,
            "event_intervals",
            MappingProxyType({event: self.event_intervals[event] for event in Stage8EventMetric}),
        )
        object.__setattr__(
            self,
            "failure_reason_counts",
            MappingProxyType(
                dict(sorted(self.failure_reason_counts.items(), key=lambda x: x[0].value))
            ),
        )
        object.__setattr__(
            self,
            "rejected_reason_counts",
            MappingProxyType(
                dict(sorted(self.rejected_reason_counts.items(), key=lambda x: x[0].value))
            ),
        )

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "interval_method": self.interval_method,
            "clinical_claim": self.clinical_claim,
            "preregistration": self.preregistration.to_dict(),
            "input_baseline_count": self.input_baseline_count,
            "input_repeat_count": self.input_repeat_count,
            "expected_transition_count": self.expected_transition_count,
            "rejected_record_count": self.rejected_record_count,
            "case_transitions": [value.to_dict() for value in self.case_transitions],
            "transition_matrix": dict(self.transition_matrix),
            "metric_intervals": {
                metric.value: self.metric_intervals[metric].to_dict()
                for metric in Stage8StabilityMetric
            },
            "event_intervals": {
                event.value: self.event_intervals[event].to_dict() for event in Stage8EventMetric
            },
            "failure_reason_counts": {
                reason.value: count for reason, count in self.failure_reason_counts.items()
            },
            "rejected_reason_counts": {
                reason.value: count for reason, count in self.rejected_reason_counts.items()
            },
        }
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
        return payload


def audit_stage8_repeatability(
    *,
    preregistration: Stage8Preregistration,
    baseline_records: Sequence[object],
    repeat_records: Sequence[object],
) -> Stage8StabilityReport:
    """Audit the exact frozen case/run set, normalizing every bad item fail closed."""

    if not isinstance(preregistration, Stage8Preregistration):
        raise TypeError("preregistration must be a Stage8Preregistration")
    preregistration.__post_init__()
    if isinstance(baseline_records, str | bytes) or not isinstance(baseline_records, Sequence):
        raise TypeError("baseline_records must be a non-string sequence")
    if isinstance(repeat_records, str | bytes) or not isinstance(repeat_records, Sequence):
        raise TypeError("repeat_records must be a non-string sequence")

    expected_cases = set(preregistration.inclusion_case_ids)
    specs = {value.run_id: value for value in preregistration.replay_specs}
    baseline_buckets: dict[str, list[Stage8CaseObservation]] = {}
    repeat_buckets: dict[tuple[str, str], list[Stage8CaseObservation]] = {}
    rejected: Counter[Stage8FailureReason] = Counter()

    for raw in baseline_records:
        if not isinstance(raw, Stage8CaseObservation):
            rejected[Stage8FailureReason.INVALID_RECORD] += 1
            continue
        if not _valid_identifier(raw.case_id):
            rejected[Stage8FailureReason.INVALID_IDENTIFIER] += 1
            continue
        case_id = str(raw.case_id)
        if case_id not in expected_cases:
            rejected[Stage8FailureReason.UNDECLARED_BASELINE_CASE] += 1
            continue
        baseline_buckets.setdefault(case_id, []).append(raw)

    for raw in repeat_records:
        if not isinstance(raw, Stage8CaseObservation):
            rejected[Stage8FailureReason.INVALID_RECORD] += 1
            continue
        if not _valid_identifier(raw.case_id) or not _valid_identifier(raw.run_id):
            rejected[Stage8FailureReason.INVALID_IDENTIFIER] += 1
            continue
        case_id = str(raw.case_id)
        run_id = str(raw.run_id)
        if case_id not in expected_cases:
            rejected[Stage8FailureReason.UNDECLARED_REPEAT_CASE] += 1
            continue
        if run_id not in specs:
            rejected[Stage8FailureReason.UNDECLARED_RUN] += 1
            continue
        repeat_buckets.setdefault((case_id, run_id), []).append(raw)

    transitions: list[Stage8CaseStability] = []
    for case_id in preregistration.inclusion_case_ids:
        baseline_values = baseline_buckets.get(case_id, [])
        for spec in preregistration.replay_specs:
            repeat_values = repeat_buckets.get((case_id, spec.run_id), [])
            reasons: list[Stage8FailureReason] = []
            if len(baseline_values) == 0:
                _append_once(reasons, Stage8FailureReason.MISSING_BASELINE_CASE)
                baseline = None
            elif len(baseline_values) > 1:
                _append_once(reasons, Stage8FailureReason.DUPLICATE_BASELINE_CASE)
                baseline = None
            else:
                baseline = baseline_values[0]
            if len(repeat_values) == 0:
                _append_once(reasons, Stage8FailureReason.MISSING_REPEAT_OBSERVATION)
                repeat = None
            elif len(repeat_values) > 1:
                _append_once(reasons, Stage8FailureReason.DUPLICATE_REPEAT_OBSERVATION)
                repeat = None
            else:
                repeat = repeat_values[0]

            baseline_view = _validated_observation(
                baseline,
                preregistration=preregistration,
                spec=None,
                expected_case_id=case_id,
            )
            repeat_view = _validated_observation(
                repeat,
                preregistration=preregistration,
                spec=spec,
                expected_case_id=case_id,
            )
            for reason in (*baseline_view.reasons, *repeat_view.reasons):
                _append_once(reasons, reason)
            if (
                baseline is not None
                and repeat is not None
                and _valid_sha256(baseline.case_degradation_trace_sha256)
                and repeat.case_degradation_trace_sha256 != baseline.case_degradation_trace_sha256
            ):
                _append_once(reasons, Stage8FailureReason.CASE_DEGRADATION_TRACE_MISMATCH)

            contract_reasons = tuple(
                reason for reason in reasons if reason not in _RUNTIME_FAIL_CLOSED_REASONS
            )
            contract_valid = not contract_reasons
            repeat_state = EntailmentState.OUT_OF_MODEL if contract_reasons else repeat_view.state
            repeat_fail_closed = bool(contract_reasons) or repeat_view.fail_closed
            transitions.append(
                Stage8CaseStability(
                    case_id=case_id,
                    run_id=spec.run_id,
                    baseline_state=baseline_view.state,
                    repeat_state=repeat_state,
                    baseline_fail_closed=baseline_view.fail_closed,
                    repeat_fail_closed=repeat_fail_closed,
                    baseline_out_of_distribution=baseline_view.out_of_distribution,
                    repeat_out_of_distribution=repeat_view.out_of_distribution,
                    state_stable=(contract_valid and baseline_view.state is repeat_state),
                    fail_closed_status_stable=(
                        contract_valid and baseline_view.fail_closed is repeat_fail_closed
                    ),
                    out_of_distribution_status_stable=(
                        contract_valid
                        and baseline_view.out_of_distribution is repeat_view.out_of_distribution
                    ),
                    positive_witness_replay_stable=(
                        baseline_view.positive_replay_valid
                        and repeat_view.positive_replay_valid
                        and contract_valid
                    ),
                    negative_witness_replay_stable=(
                        baseline_view.negative_replay_valid
                        and repeat_view.negative_replay_valid
                        and contract_valid
                    ),
                    dual_witness_replay_stable=(
                        baseline_view.positive_replay_valid
                        and baseline_view.negative_replay_valid
                        and repeat_view.positive_replay_valid
                        and repeat_view.negative_replay_valid
                        and contract_valid
                    ),
                    baseline_positive_witness_replay_valid=(baseline_view.positive_replay_valid),
                    baseline_negative_witness_replay_valid=(baseline_view.negative_replay_valid),
                    repeat_positive_witness_replay_valid=(repeat_view.positive_replay_valid),
                    repeat_negative_witness_replay_valid=(repeat_view.negative_replay_valid),
                    baseline_positive_witness_sha256=(baseline_view.positive_witness_sha256),
                    baseline_negative_witness_sha256=(baseline_view.negative_witness_sha256),
                    repeat_positive_witness_sha256=(repeat_view.positive_witness_sha256),
                    repeat_negative_witness_sha256=(repeat_view.negative_witness_sha256),
                    positive_witness_hash_stable=(
                        contract_valid
                        and baseline_view.positive_witness_sha256 is not None
                        and baseline_view.positive_witness_sha256
                        == repeat_view.positive_witness_sha256
                        and baseline_view.positive_replay_valid is repeat_view.positive_replay_valid
                    ),
                    negative_witness_hash_stable=(
                        contract_valid
                        and baseline_view.negative_witness_sha256 is not None
                        and baseline_view.negative_witness_sha256
                        == repeat_view.negative_witness_sha256
                        and baseline_view.negative_replay_valid is repeat_view.negative_replay_valid
                    ),
                    dual_witness_hash_stable=(
                        contract_valid
                        and baseline_view.positive_witness_sha256 is not None
                        and baseline_view.negative_witness_sha256 is not None
                        and baseline_view.positive_witness_sha256
                        == repeat_view.positive_witness_sha256
                        and baseline_view.negative_witness_sha256
                        == repeat_view.negative_witness_sha256
                        and baseline_view.positive_replay_valid is repeat_view.positive_replay_valid
                        and baseline_view.negative_replay_valid is repeat_view.negative_replay_valid
                    ),
                    failure_reasons=tuple(reasons),
                )
            )

    if not baseline_records:
        rejected[Stage8FailureReason.EMPTY_BASELINE] += 1
    if not repeat_records:
        rejected[Stage8FailureReason.EMPTY_REPEAT_SET] += 1
    matrix = Counter(value.transition for value in transitions)
    transition_matrix = {key: matrix[key] for key in _transition_keys()}
    metric_counts = _metric_counts(transitions)
    expected_count = len(transitions)
    intervals = {
        metric: wilson_score_proportion(
            metric_counts[metric],
            expected_count,
            confidence_level=preregistration.confidence_level,
        )
        for metric in Stage8StabilityMetric
    }
    event_counts = _event_counts(transitions)
    event_intervals = {
        event: wilson_score_proportion(
            event_counts[event],
            expected_count,
            confidence_level=preregistration.confidence_level,
        )
        for event in Stage8EventMetric
    }
    failures: Counter[Stage8FailureReason] = Counter()
    for transition in transitions:
        failures.update(set(transition.failure_reasons))
    blocking_failure = any(reason not in _RUNTIME_FAIL_CLOSED_REASONS for reason in failures)
    status = (
        Stage8AuditStatus.FAIL_CLOSED if blocking_failure or rejected else Stage8AuditStatus.SUCCESS
    )
    report = Stage8StabilityReport(
        preregistration=preregistration,
        status=status,
        input_baseline_count=len(baseline_records),
        input_repeat_count=len(repeat_records),
        expected_transition_count=expected_count,
        rejected_record_count=sum(rejected.values()),
        case_transitions=tuple(transitions),
        transition_matrix=transition_matrix,
        metric_intervals=intervals,
        event_intervals=event_intervals,
        failure_reason_counts=dict(failures),
        rejected_reason_counts=dict(rejected),
    )
    json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)
    return report


@dataclass(frozen=True, slots=True)
class _ObservationView:
    state: EntailmentState
    fail_closed: bool
    out_of_distribution: bool
    positive_replay_valid: bool
    negative_replay_valid: bool
    positive_witness_sha256: str | None
    negative_witness_sha256: str | None
    reasons: tuple[Stage8FailureReason, ...]


def _validated_observation(
    value: Stage8CaseObservation | None,
    *,
    preregistration: Stage8Preregistration,
    spec: Stage8ReplaySpec | None,
    expected_case_id: str,
) -> _ObservationView:
    if value is None:
        return _ObservationView(
            state=EntailmentState.OUT_OF_MODEL,
            fail_closed=True,
            out_of_distribution=False,
            positive_replay_valid=False,
            negative_replay_valid=False,
            positive_witness_sha256=None,
            negative_witness_sha256=None,
            reasons=(),
        )
    reasons: list[Stage8FailureReason] = []
    if value.schema_version != STAGE8_SCHEMA_VERSION:
        _append_once(reasons, Stage8FailureReason.INVALID_SCHEMA_VERSION)
    if value.case_id != expected_case_id:
        _append_once(reasons, Stage8FailureReason.INVALID_IDENTIFIER)
    _compare(
        value.dataset_id,
        preregistration.dataset_id,
        Stage8FailureReason.DATASET_ID_MISMATCH,
        reasons,
    )
    _compare(
        value.dataset_version,
        preregistration.dataset_version,
        Stage8FailureReason.DATASET_VERSION_MISMATCH,
        reasons,
    )
    _compare(value.split, preregistration.split, Stage8FailureReason.SPLIT_MISMATCH, reasons)
    scope_value = value.scope.value if isinstance(value.scope, Stage8Scope) else value.scope
    _compare(scope_value, preregistration.scope.value, Stage8FailureReason.SCOPE_MISMATCH, reasons)
    seed_valid = type(value.seed) is int and 0 <= value.seed <= _MAX_SEED  # noqa: E721
    if not seed_valid:
        _append_once(reasons, Stage8FailureReason.INVALID_SEED)
    if spec is None:
        _compare(value.run_id, STAGE8_BASELINE_RUN_ID, Stage8FailureReason.UNDECLARED_RUN, reasons)
    else:
        _compare(value.run_id, spec.run_id, Stage8FailureReason.UNDECLARED_RUN, reasons)
        if not seed_valid or value.seed != spec.seed:
            _append_once(reasons, Stage8FailureReason.RUN_SEED_MISMATCH)
        _compare(
            value.degradation_replay_id,
            spec.degradation_replay_id,
            Stage8FailureReason.DEGRADATION_REPLAY_ID_MISMATCH,
            reasons,
        )
        _compare(
            value.degradation_replay_set_sha256,
            spec.degradation_replay_set_sha256,
            Stage8FailureReason.DEGRADATION_REPLAY_SET_HASH_MISMATCH,
            reasons,
        )
    if not _valid_sha256(value.case_degradation_trace_sha256):
        _append_once(reasons, Stage8FailureReason.CASE_DEGRADATION_TRACE_MISMATCH)

    state = _state(value.state)
    if state is None:
        _append_once(reasons, Stage8FailureReason.INVALID_STATE)
        state = EntailmentState.OUT_OF_MODEL
    flags = (
        value.fail_closed,
        value.out_of_distribution,
        value.degradation_known,
        value.solver_succeeded,
        value.positive_witness_replay_valid,
        value.negative_witness_replay_valid,
    )
    if any(type(flag) is not bool for flag in flags):  # noqa: E721
        _append_once(reasons, Stage8FailureReason.INVALID_FLAG)
    fail_closed = value.fail_closed if type(value.fail_closed) is bool else True  # noqa: E721
    ood = (
        value.out_of_distribution
        if type(value.out_of_distribution) is bool  # noqa: E721
        else False
    )
    degradation_known = (
        value.degradation_known if type(value.degradation_known) is bool else False  # noqa: E721
    )
    solver_succeeded = (
        value.solver_succeeded if type(value.solver_succeeded) is bool else False  # noqa: E721
    )
    positive_replay = (
        value.positive_witness_replay_valid
        if type(value.positive_witness_replay_valid) is bool  # noqa: E721
        else False
    )
    negative_replay = (
        value.negative_witness_replay_valid
        if type(value.negative_witness_replay_valid) is bool  # noqa: E721
        else False
    )
    numeric_values = (
        value.alpha,
        value.positive_score,
        value.negative_score,
        value.positive_threshold,
        value.negative_threshold,
    )
    numeric_valid = all(
        _finite_between(item, lower=0.0, upper=None, strict=False) for item in numeric_values
    )
    if not numeric_valid:
        _append_once(reasons, Stage8FailureReason.INVALID_NUMERIC_VALUE)
    _compare_numeric(
        value.alpha, preregistration.alpha, Stage8FailureReason.LOCKED_ALPHA_MISMATCH, reasons
    )
    _compare_numeric(
        value.positive_threshold,
        preregistration.positive_threshold,
        Stage8FailureReason.LOCKED_POSITIVE_THRESHOLD_MISMATCH,
        reasons,
    )
    _compare_numeric(
        value.negative_threshold,
        preregistration.negative_threshold,
        Stage8FailureReason.LOCKED_NEGATIVE_THRESHOLD_MISMATCH,
        reasons,
    )
    sample_counts = (
        (
            value.positive_sample_count,
            preregistration.positive_sample_count,
            Stage8FailureReason.LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH,
        ),
        (
            value.negative_sample_count,
            preregistration.negative_sample_count,
            Stage8FailureReason.LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH,
        ),
    )
    for sample_actual, sample_expected, mismatch_reason in sample_counts:
        valid_count = (
            type(sample_actual) is int  # noqa: E721
            and 0 < sample_actual <= _MAX_SAMPLE_COUNT
        )
        if not valid_count:
            _append_once(reasons, Stage8FailureReason.INVALID_SAMPLE_COUNT)
        if not valid_count or sample_actual != sample_expected:
            _append_once(reasons, mismatch_reason)
    bindings = (
        (
            value.config_hash,
            preregistration.config_hash,
            Stage8FailureReason.LOCKED_CONFIG_HASH_MISMATCH,
        ),
        (value.data_hash, preregistration.data_hash, Stage8FailureReason.LOCKED_DATA_HASH_MISMATCH),
        (
            value.model_hash,
            preregistration.model_hash,
            Stage8FailureReason.LOCKED_MODEL_HASH_MISMATCH,
        ),
        (
            value.stage7_package_sha256,
            preregistration.stage7_package_sha256,
            Stage8FailureReason.STAGE7_PACKAGE_HASH_MISMATCH,
        ),
        (
            value.provenance_sha256,
            preregistration.provenance_sha256,
            Stage8FailureReason.PROVENANCE_HASH_MISMATCH,
        ),
    )
    for binding_actual, binding_expected, reason in bindings:
        _compare(binding_actual, binding_expected, reason, reasons)
    positive_hash = (
        str(value.positive_witness_sha256) if _valid_sha256(value.positive_witness_sha256) else None
    )
    negative_hash = (
        str(value.negative_witness_sha256) if _valid_sha256(value.negative_witness_sha256) else None
    )
    if positive_replay and positive_hash is None:
        _append_once(reasons, Stage8FailureReason.INVALID_WITNESS_HASH)
    if negative_replay and negative_hash is None:
        _append_once(reasons, Stage8FailureReason.INVALID_WITNESS_HASH)

    runtime_reasons: list[Stage8FailureReason] = []
    if not degradation_known:
        runtime_reasons.append(Stage8FailureReason.UNKNOWN_DEGRADATION)
    if not solver_succeeded:
        runtime_reasons.append(Stage8FailureReason.SOLVER_FAILED)
    if not positive_replay or not negative_replay:
        runtime_reasons.append(Stage8FailureReason.INVALID_WITNESS_REPLAY)
    if ood:
        runtime_reasons.append(Stage8FailureReason.OUT_OF_DISTRIBUTION)
    for reason in runtime_reasons:
        _append_once(reasons, reason)
    if numeric_valid and not runtime_reasons and not fail_closed:
        assert isinstance(value.positive_score, int | float)
        assert isinstance(value.negative_score, int | float)
        assert isinstance(value.positive_threshold, int | float)
        assert isinstance(value.negative_threshold, int | float)
        score_state = state_from_feasibility(
            float(value.positive_score) <= float(value.positive_threshold),
            float(value.negative_score) <= float(value.negative_threshold),
        )
        if state is not score_state:
            _append_once(reasons, Stage8FailureReason.STATE_SCORE_MISMATCH)
    if runtime_reasons:
        if not fail_closed or state is not EntailmentState.OUT_OF_MODEL:
            _append_once(reasons, Stage8FailureReason.FAIL_CLOSED_SEMANTICS_MISMATCH)
        fail_closed = True
        state = EntailmentState.OUT_OF_MODEL
    elif fail_closed:
        _append_once(reasons, Stage8FailureReason.FAIL_CLOSED_SEMANTICS_MISMATCH)
        state = EntailmentState.OUT_OF_MODEL
    nonruntime_reasons = tuple(
        reason for reason in reasons if reason not in _RUNTIME_FAIL_CLOSED_REASONS
    )
    if nonruntime_reasons:
        state = EntailmentState.OUT_OF_MODEL
        fail_closed = True
    return _ObservationView(
        state=state,
        fail_closed=fail_closed,
        out_of_distribution=ood,
        positive_replay_valid=positive_replay,
        negative_replay_valid=negative_replay,
        positive_witness_sha256=positive_hash,
        negative_witness_sha256=negative_hash,
        reasons=tuple(reasons),
    )


def _metric_counts(
    transitions: Sequence[Stage8CaseStability],
) -> Counter[Stage8StabilityMetric]:
    counts: Counter[Stage8StabilityMetric] = Counter()
    fields = {
        Stage8StabilityMetric.STATE_STABLE: "state_stable",
        Stage8StabilityMetric.FAIL_CLOSED_STATUS_STABLE: "fail_closed_status_stable",
        Stage8StabilityMetric.OUT_OF_DISTRIBUTION_STATUS_STABLE: (
            "out_of_distribution_status_stable"
        ),
        Stage8StabilityMetric.POSITIVE_WITNESS_REPLAY_STABLE: ("positive_witness_replay_stable"),
        Stage8StabilityMetric.NEGATIVE_WITNESS_REPLAY_STABLE: ("negative_witness_replay_stable"),
        Stage8StabilityMetric.DUAL_WITNESS_REPLAY_STABLE: "dual_witness_replay_stable",
        Stage8StabilityMetric.POSITIVE_WITNESS_HASH_STABLE: "positive_witness_hash_stable",
        Stage8StabilityMetric.NEGATIVE_WITNESS_HASH_STABLE: "negative_witness_hash_stable",
        Stage8StabilityMetric.DUAL_WITNESS_HASH_STABLE: "dual_witness_hash_stable",
    }
    for metric, attribute in fields.items():
        counts[metric] = sum(bool(getattr(value, attribute)) for value in transitions)
    return counts


def _event_counts(
    transitions: Sequence[Stage8CaseStability],
) -> Counter[Stage8EventMetric]:
    counts: Counter[Stage8EventMetric] = Counter()
    counts[Stage8EventMetric.REPEAT_FAIL_CLOSED] = sum(
        value.repeat_fail_closed for value in transitions
    )
    counts[Stage8EventMetric.REPEAT_OUT_OF_DISTRIBUTION] = sum(
        value.repeat_out_of_distribution for value in transitions
    )
    counts[Stage8EventMetric.REPEAT_OUT_OF_MODEL] = sum(
        value.repeat_state is EntailmentState.OUT_OF_MODEL for value in transitions
    )
    counts[Stage8EventMetric.REPEAT_DUAL_WITNESS_REPLAY_PASSED] = sum(
        value.repeat_positive_witness_replay_valid
        and value.repeat_negative_witness_replay_valid
        and _case_contract_valid(value)
        for value in transitions
    )
    return counts


def _case_contract_valid(value: Stage8CaseStability) -> bool:
    return not any(reason not in _RUNTIME_FAIL_CLOSED_REASONS for reason in value.failure_reasons)


def _transition_keys() -> tuple[str, ...]:
    return tuple(
        f"{before.value}->{after.value}" for before in EntailmentState for after in EntailmentState
    )


def _state(value: object) -> EntailmentState | None:
    if isinstance(value, EntailmentState):
        return value
    if isinstance(value, str):
        try:
            return EntailmentState(value)
        except ValueError:
            return None
    return None


def _compare(
    actual: object,
    expected: object,
    reason: Stage8FailureReason,
    reasons: list[Stage8FailureReason],
) -> None:
    if actual != expected:
        _append_once(reasons, reason)


def _compare_numeric(
    actual: object,
    expected: float,
    reason: Stage8FailureReason,
    reasons: list[Stage8FailureReason],
) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, int | float)
        or not math.isfinite(float(actual))
        or float(actual) != float(expected)
    ):
        _append_once(reasons, reason)


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _finite_between(
    value: object,
    *,
    lower: float,
    upper: float | None,
    strict: bool,
) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        return False
    number = float(value)
    lower_ok = number > lower if strict else number >= lower
    upper_ok = True if upper is None else (number < upper if strict else number <= upper)
    return lower_ok and upper_ok


def _append_once(values: list[Stage8FailureReason], value: Stage8FailureReason) -> None:
    if value not in values:
        values.append(value)


def _validate_reason_mapping(mapping: Mapping[Stage8FailureReason, int]) -> None:
    if not isinstance(mapping, Mapping):
        raise TypeError("reason counts must be a mapping")
    if any(not isinstance(key, Stage8FailureReason) for key in mapping):
        raise TypeError("reason count keys must use Stage8FailureReason")
    if any(type(value) is not int or value <= 0 for value in mapping.values()):  # noqa: E721
        raise ValueError("reported reason counts must be positive integers")


__all__ = [
    "STAGE8_BASELINE_RUN_ID",
    "STAGE8_SCHEMA_VERSION",
    "Stage8AuditStatus",
    "Stage8CaseObservation",
    "Stage8CaseStability",
    "Stage8EventMetric",
    "Stage8FailureReason",
    "Stage8Preregistration",
    "Stage8ReplaySpec",
    "Stage8Scope",
    "Stage8StabilityMetric",
    "Stage8StabilityReport",
    "audit_stage8_repeatability",
]
