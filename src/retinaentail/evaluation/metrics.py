"""Locked, test-only Stage 4 aggregation with deterministic Wilson intervals.

The metrics in this module are descriptive audit summaries, not clinical performance
claims.  Input evaluations are revalidated against one immutable Stage 3 lock before
they can contribute a determinate state.  Contract failures are normalized to
``OUT_OF_MODEL`` and remain visible in the reason counts.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from enum import StrEnum
from statistics import NormalDist
from types import MappingProxyType
from typing import Final, TypeVar

from retinaentail.calibration.certificate import Stage3SyntheticCertificate
from retinaentail.calibration.protocol import (
    CalibrationLabel,
    EvaluationFailureReason,
    EvaluationResult,
    EvaluationStatus,
    SampleSplit,
)
from retinaentail.entailment.states import EntailmentState
from retinaentail.witnesses.stage3_replay import Stage3DualReplayResult

STAGE4_EVALUATION_REPORT_SCHEMA_VERSION: Final = "4.0.0"
STAGE4_INTERVAL_METHOD: Final = "wilson_score"
STAGE4_EXPECTED_SPLIT: Final = SampleSplit.TEST
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TOKEN: Final = object()
_ISSUED_REPORTS: dict[int, tuple[object, str]] = {}
_K = TypeVar("_K")


class Stage4EvaluationScope(StrEnum):
    """Permitted, explicit provenance scopes for a Stage 4 report."""

    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    APPROVED_REAL_DATA = "APPROVED_REAL_DATA"


class Stage4ReportStatus(StrEnum):
    """Whether the aggregation inputs satisfied the locked test contract."""

    SUCCESS = "SUCCESS"
    FAIL_CLOSED = "FAIL_CLOSED"


class Stage4AggregationFailureReason(StrEnum):
    """Machine-readable Stage 4 input and lock failures."""

    EMPTY_TEST_SET = "EMPTY_TEST_SET"
    INVALID_CASE_RECORD = "INVALID_CASE_RECORD"
    INVALID_CASE_ID = "INVALID_CASE_ID"
    DUPLICATE_CASE_ID = "DUPLICATE_CASE_ID"
    NON_TEST_SPLIT = "NON_TEST_SPLIT"
    INVALID_REFERENCE_LABEL = "INVALID_REFERENCE_LABEL"
    INVALID_EVALUATION = "INVALID_EVALUATION"
    CASE_ID_MISMATCH = "CASE_ID_MISMATCH"
    LOCKED_ALPHA_MISMATCH = "LOCKED_ALPHA_MISMATCH"
    LOCKED_POSITIVE_THRESHOLD_MISMATCH = "LOCKED_POSITIVE_THRESHOLD_MISMATCH"
    LOCKED_NEGATIVE_THRESHOLD_MISMATCH = "LOCKED_NEGATIVE_THRESHOLD_MISMATCH"
    LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH = "LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH"
    LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH = "LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH"
    LOCKED_CONFIG_HASH_MISMATCH = "LOCKED_CONFIG_HASH_MISMATCH"
    LOCKED_DATA_HASH_MISMATCH = "LOCKED_DATA_HASH_MISMATCH"
    LOCKED_MODEL_HASH_MISMATCH = "LOCKED_MODEL_HASH_MISMATCH"
    INVALID_CERTIFICATE = "INVALID_CERTIFICATE"
    INVALID_DUAL_REPLAY = "INVALID_DUAL_REPLAY"
    CERTIFICATE_EVALUATION_MISMATCH = "CERTIFICATE_EVALUATION_MISMATCH"
    CERTIFICATE_ARTIFACT_HASH_MISMATCH = "CERTIFICATE_ARTIFACT_HASH_MISMATCH"
    CERTIFICATE_REPLAY_MISMATCH = "CERTIFICATE_REPLAY_MISMATCH"


@dataclass(frozen=True, slots=True, kw_only=True)
class LockedStage4Contract:
    """The exact Stage 3 alpha and provenance hashes permitted at test time."""

    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    total_sample_count: int
    calibration_artifact_hash: str
    config_hash: str
    data_hash: str
    model_hash: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, int | float)
            or not math.isfinite(float(self.alpha))
            or not 0.0 < float(self.alpha) < 1.0
        ):
            raise ValueError("locked alpha must be finite and strictly between zero and one")
        thresholds = (self.positive_threshold, self.negative_threshold)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in thresholds
        ):
            raise ValueError("locked thresholds must be finite and non-negative")
        counts = (
            self.positive_sample_count,
            self.negative_sample_count,
            self.total_sample_count,
        )
        if (
            not all(_plain_nonnegative_int(value) for value in counts)
            or self.positive_sample_count < 1
            or self.negative_sample_count < 1
            or self.total_sample_count != self.positive_sample_count + self.negative_sample_count
        ):
            raise ValueError("locked calibration counts are inconsistent")
        if not all(
            _valid_sha256(value)
            for value in (
                self.calibration_artifact_hash,
                self.config_hash,
                self.data_hash,
                self.model_hash,
            )
        ):
            raise ValueError("locked provenance hashes must be lowercase SHA-256 values")

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": float(self.alpha),
            "positive_threshold": float(self.positive_threshold),
            "negative_threshold": float(self.negative_threshold),
            "positive_sample_count": self.positive_sample_count,
            "negative_sample_count": self.negative_sample_count,
            "total_sample_count": self.total_sample_count,
            "calibration_artifact_hash": self.calibration_artifact_hash,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage4CaseRecord:
    """Untrusted wrapper around one purported test evaluation.

    Validation intentionally happens in :func:`aggregate_locked_test_evaluations` so
    malformed records can be reported and fail closed instead of aborting the audit.
    ``UNKNOWN`` is a first-class reference label and is never coerced to lesion-absent.
    """

    case_id: str
    split: SampleSplit | str
    reference_label: CalibrationLabel | str
    evaluation: object
    certificate: object = None
    dual_replay: object = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BinomialProportion:
    """One count, denominator, and deterministic two-sided Wilson interval."""

    numerator: int
    denominator: int
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    method: str = STAGE4_INTERVAL_METHOD

    def __post_init__(self) -> None:
        if not _plain_nonnegative_int(self.numerator) or not _plain_nonnegative_int(
            self.denominator
        ):
            raise ValueError("proportion counts must be non-negative integers")
        if self.numerator > self.denominator:
            raise ValueError("proportion numerator cannot exceed denominator")
        _validate_confidence(self.confidence_level)
        if self.method != STAGE4_INTERVAL_METHOD:
            raise ValueError("unsupported interval method")
        values = (self.estimate, self.lower, self.upper)
        if self.denominator == 0:
            if values != (None, None, None):
                raise ValueError("zero-denominator proportions must contain null estimates")
            return
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in values
        ):
            raise ValueError("non-empty proportion estimates must be finite in [0, 1]")
        assert self.estimate is not None
        assert self.lower is not None
        assert self.upper is not None
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("proportion estimate must lie inside its interval")

    def to_dict(self) -> dict[str, object]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage4EvaluationReport:
    """Versioned aggregate report issued only by the locked aggregation function."""

    scope: Stage4EvaluationScope
    status: Stage4ReportStatus
    locked_contract: LockedStage4Contract
    confidence_level: float
    input_record_count: int
    test_case_count: int
    rejected_record_count: int
    evaluable_count: int
    unevaluable_count: int
    fail_closed_count: int
    out_of_distribution_count: int
    both_witness_replays_passed_count: int
    one_or_more_witness_replays_failed_count: int
    labeled_reference_count: int
    state_counts: Mapping[EntailmentState, int]
    reference_label_counts: Mapping[CalibrationLabel, int]
    failure_reason_counts: Mapping[str, int]
    rejected_reason_counts: Mapping[str, int]
    aggregation_failure_reasons: tuple[Stage4AggregationFailureReason, ...]
    state_proportions: Mapping[EntailmentState, BinomialProportion]
    evaluable_proportion: BinomialProportion
    unevaluable_proportion: BinomialProportion
    fail_closed_proportion: BinomialProportion
    out_of_distribution_proportion: BinomialProportion
    both_witness_replays_passed_proportion: BinomialProportion
    one_or_more_witness_replays_failed_proportion: BinomialProportion
    schema_version: str = STAGE4_EVALUATION_REPORT_SCHEMA_VERSION
    split: str = STAGE4_EXPECTED_SPLIT.value
    clinical_claim: bool = False
    interval_method: str = STAGE4_INTERVAL_METHOD
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        issued = _ISSUED_REPORTS.get(id(self))
        new_issue = _token is _REPORT_TOKEN
        if not new_issue and (issued is None or issued[0] is not self):
            raise TypeError(
                "Stage4EvaluationReport can only be issued by aggregate_locked_test_evaluations"
            )
        if self.schema_version != STAGE4_EVALUATION_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 4 report schema version")
        if not isinstance(self.scope, Stage4EvaluationScope) or not isinstance(
            self.status, Stage4ReportStatus
        ):
            raise TypeError("Stage 4 report scope/status use controlled enums")
        if not isinstance(self.locked_contract, LockedStage4Contract):
            raise TypeError("Stage 4 report requires a LockedStage4Contract")
        self.locked_contract.__post_init__()
        if self.split != STAGE4_EXPECTED_SPLIT.value:
            raise ValueError("Stage 4 evaluation reports are test-only")
        if self.clinical_claim is not False:
            raise ValueError("Stage 4 audit reports cannot make a clinical claim")
        if self.interval_method != STAGE4_INTERVAL_METHOD:
            raise ValueError("unsupported interval method")
        _validate_confidence(self.confidence_level)
        counts = (
            self.input_record_count,
            self.test_case_count,
            self.rejected_record_count,
            self.evaluable_count,
            self.unevaluable_count,
            self.fail_closed_count,
            self.out_of_distribution_count,
            self.both_witness_replays_passed_count,
            self.one_or_more_witness_replays_failed_count,
            self.labeled_reference_count,
        )
        if not all(_plain_nonnegative_int(value) for value in counts):
            raise ValueError("Stage 4 report counts must be non-negative integers")
        if self.input_record_count != self.test_case_count + self.rejected_record_count:
            raise ValueError("input count must equal test plus rejected records")
        if self.test_case_count != self.evaluable_count + self.unevaluable_count:
            raise ValueError("test count must equal evaluable plus unevaluable cases")
        if self.fail_closed_count != self.unevaluable_count:
            raise ValueError("fail-closed and unevaluable counts must agree")
        if (
            self.test_case_count
            != self.both_witness_replays_passed_count
            + self.one_or_more_witness_replays_failed_count
        ):
            raise ValueError("witness pass/fail counts must partition test cases")
        _require_count_mapping(self.state_counts, tuple(EntailmentState), "state_counts")
        _require_count_mapping(
            self.reference_label_counts,
            tuple(CalibrationLabel),
            "reference_label_counts",
        )
        if sum(self.state_counts.values()) != self.test_case_count:
            raise ValueError("four-state counts must partition test cases")
        if sum(self.reference_label_counts.values()) != self.test_case_count:
            raise ValueError("reference-label counts must partition test cases")
        if self.labeled_reference_count != (
            self.reference_label_counts[CalibrationLabel.LESION_PRESENT]
            + self.reference_label_counts[CalibrationLabel.LESION_ABSENT]
        ):
            raise ValueError("labeled denominator must exclude UNKNOWN references")
        _require_exact_keys(
            self.state_proportions,
            tuple(EntailmentState),
            "state_proportions",
        )
        for state in EntailmentState:
            interval = self.state_proportions[state]
            if not isinstance(interval, BinomialProportion):
                raise TypeError("state proportions must use BinomialProportion values")
            expected_interval = wilson_score_proportion(
                self.state_counts[state],
                self.test_case_count,
                confidence_level=self.confidence_level,
            )
            if interval != expected_interval:
                raise ValueError("state interval counts conflict with state counts")
        expected_intervals = (
            (self.evaluable_proportion, self.evaluable_count),
            (self.unevaluable_proportion, self.unevaluable_count),
            (self.fail_closed_proportion, self.fail_closed_count),
            (self.out_of_distribution_proportion, self.out_of_distribution_count),
            (
                self.both_witness_replays_passed_proportion,
                self.both_witness_replays_passed_count,
            ),
            (
                self.one_or_more_witness_replays_failed_proportion,
                self.one_or_more_witness_replays_failed_count,
            ),
        )
        for interval, count in expected_intervals:
            if not isinstance(interval, BinomialProportion) or interval != (
                wilson_score_proportion(
                    count,
                    self.test_case_count,
                    confidence_level=self.confidence_level,
                )
            ):
                raise ValueError("summary interval conflicts with recomputed Wilson interval")
        if any(
            not isinstance(reason, Stage4AggregationFailureReason)
            for reason in self.aggregation_failure_reasons
        ):
            raise TypeError("aggregation failure reasons must use the Stage 4 enum")
        if len(set(self.aggregation_failure_reasons)) != len(self.aggregation_failure_reasons):
            raise ValueError("aggregation failure reasons must be unique")
        expected_status = (
            Stage4ReportStatus.FAIL_CLOSED
            if self.aggregation_failure_reasons
            else Stage4ReportStatus.SUCCESS
        )
        if self.status is not expected_status:
            raise ValueError("report status conflicts with aggregation failure reasons")
        allowed_case_reasons = {
            *(reason.value for reason in EvaluationFailureReason),
            *(reason.value for reason in Stage4AggregationFailureReason),
        }
        allowed_rejected_reasons = {reason.value for reason in Stage4AggregationFailureReason}
        _validate_reason_counts(
            self.failure_reason_counts,
            allowed=allowed_case_reasons,
        )
        _validate_reason_counts(
            self.rejected_reason_counts,
            allowed=allowed_rejected_reasons,
        )
        if (self.fail_closed_count == 0) != (not self.failure_reason_counts):
            raise ValueError("fail-closed count and failure-reason map emptiness must agree")
        if self.fail_closed_count and sum(self.failure_reason_counts.values()) < (
            self.fail_closed_count
        ):
            raise ValueError("failure-reason counts cannot cover every fail-closed case")
        if (self.rejected_record_count == 0) != (not self.rejected_reason_counts):
            raise ValueError("rejected count and rejected-reason map emptiness must agree")
        if self.rejected_record_count and sum(self.rejected_reason_counts.values()) < (
            self.rejected_record_count
        ):
            raise ValueError("rejected-reason counts cannot cover every rejected record")
        stage4_reason_names = {reason.value for reason in Stage4AggregationFailureReason}
        expected_aggregation_names = (
            set(self.failure_reason_counts) | set(self.rejected_reason_counts)
        ) & stage4_reason_names
        if self.test_case_count == 0:
            expected_aggregation_names.add(Stage4AggregationFailureReason.EMPTY_TEST_SET.value)
        if {reason.value for reason in self.aggregation_failure_reasons} != (
            expected_aggregation_names
        ):
            raise ValueError("aggregation reasons do not exactly match Stage 4 reason maps")

        object.__setattr__(self, "state_counts", MappingProxyType(dict(self.state_counts)))
        object.__setattr__(
            self,
            "reference_label_counts",
            MappingProxyType(dict(self.reference_label_counts)),
        )
        object.__setattr__(
            self,
            "failure_reason_counts",
            MappingProxyType(dict(sorted(self.failure_reason_counts.items()))),
        )
        object.__setattr__(
            self,
            "rejected_reason_counts",
            MappingProxyType(dict(sorted(self.rejected_reason_counts.items()))),
        )
        object.__setattr__(
            self,
            "state_proportions",
            MappingProxyType(dict(self.state_proportions)),
        )
        fingerprint = json.dumps(
            self._payload_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if new_issue:
            _ISSUED_REPORTS[id(self)] = (self, fingerprint)
        elif issued is None or issued[1] != fingerprint:
            raise ValueError("issued Stage 4 report bindings changed after issuance")

    def _payload_dict(self) -> dict[str, object]:
        """Build the payload after validation without recursively revalidating."""

        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "split": self.split,
            "clinical_claim": self.clinical_claim,
            "status": self.status.value,
            "locked_contract": self.locked_contract.to_dict(),
            "confidence_level": self.confidence_level,
            "interval_method": self.interval_method,
            "input_record_count": self.input_record_count,
            "test_case_count": self.test_case_count,
            "rejected_record_count": self.rejected_record_count,
            "evaluable_count": self.evaluable_count,
            "unevaluable_count": self.unevaluable_count,
            "fail_closed_count": self.fail_closed_count,
            "out_of_distribution_count": self.out_of_distribution_count,
            "both_witness_replays_passed_count": self.both_witness_replays_passed_count,
            "one_or_more_witness_replays_failed_count": (
                self.one_or_more_witness_replays_failed_count
            ),
            "labeled_reference_count": self.labeled_reference_count,
            "state_counts": {state.value: self.state_counts[state] for state in EntailmentState},
            "reference_label_counts": {
                label.value: self.reference_label_counts[label] for label in CalibrationLabel
            },
            "failure_reason_counts": dict(self.failure_reason_counts),
            "rejected_reason_counts": dict(self.rejected_reason_counts),
            "aggregation_failure_reasons": [
                reason.value for reason in self.aggregation_failure_reasons
            ],
            "state_proportions": {
                state.value: self.state_proportions[state].to_dict() for state in EntailmentState
            },
            "evaluable_proportion": self.evaluable_proportion.to_dict(),
            "unevaluable_proportion": self.unevaluable_proportion.to_dict(),
            "fail_closed_proportion": self.fail_closed_proportion.to_dict(),
            "out_of_distribution_proportion": (self.out_of_distribution_proportion.to_dict()),
            "both_witness_replays_passed_proportion": (
                self.both_witness_replays_passed_proportion.to_dict()
            ),
            "one_or_more_witness_replays_failed_proportion": (
                self.one_or_more_witness_replays_failed_proportion.to_dict()
            ),
        }

    def to_dict(self) -> dict[str, object]:
        """Revalidate issuance/bindings, then return deterministic JSON-safe data."""

        self.__post_init__()
        return self._payload_dict()


def wilson_score_proportion(
    numerator: int,
    denominator: int,
    *,
    confidence_level: float = 0.95,
) -> BinomialProportion:
    """Return a two-sided Wilson score interval using only the standard library."""

    if not _plain_nonnegative_int(numerator) or not _plain_nonnegative_int(denominator):
        raise ValueError("Wilson counts must be non-negative integers")
    if numerator > denominator:
        raise ValueError("Wilson numerator cannot exceed denominator")
    confidence = _validate_confidence(confidence_level)
    if denominator == 0:
        return BinomialProportion(
            numerator=numerator,
            denominator=denominator,
            estimate=None,
            lower=None,
            upper=None,
            confidence_level=confidence,
        )
    estimate = numerator / denominator
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    scale = 1.0 + z_squared / denominator
    center = (estimate + z_squared / (2.0 * denominator)) / scale
    margin = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / denominator
            + z_squared / (4.0 * denominator * denominator)
        )
        / scale
    )
    return BinomialProportion(
        numerator=numerator,
        denominator=denominator,
        estimate=estimate,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence_level=confidence,
    )


def aggregate_locked_test_evaluations(
    records: Sequence[object],
    *,
    locked_contract: LockedStage4Contract,
    scope: Stage4EvaluationScope,
    confidence_level: float = 0.95,
) -> Stage4EvaluationReport:
    """Aggregate only unique test records under one exact Stage 3 lock.

    Non-test, invalid-ID, and duplicate-ID records are rejected before any test
    denominator is formed.  Within the accepted test set, malformed evaluations,
    metadata conflicts, and lock drift are counted as fail-closed ``OUT_OF_MODEL``.
    Valid Stage 3 fail-closed records retain their original reasons.  Reference
    ``UNKNOWN`` contributes only to its own count and is excluded from the explicitly
    named ``labeled_reference_count`` denominator.
    """

    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("records must be a non-string sequence")
    if not isinstance(locked_contract, LockedStage4Contract):
        raise TypeError("locked_contract must be a LockedStage4Contract")
    locked_contract.__post_init__()
    if not isinstance(scope, Stage4EvaluationScope):
        raise TypeError("scope must be a Stage4EvaluationScope")
    if scope is not Stage4EvaluationScope.SYNTHETIC_ONLY:
        raise ValueError(
            "APPROVED_REAL_DATA aggregation requires an independent prerequisite verifier"
        )
    confidence = _validate_confidence(confidence_level)

    duplicate_ids = _duplicate_valid_ids(records)
    state_counts: Counter[EntailmentState] = Counter()
    label_counts: Counter[CalibrationLabel] = Counter()
    failure_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    aggregation_failures: list[Stage4AggregationFailureReason] = []
    test_count = 0
    rejected_count = 0
    evaluable_count = 0
    ood_count = 0
    witness_pass_count = 0

    for raw_record in records:
        rejection_reasons = _record_rejection_reasons(raw_record, duplicate_ids)
        if rejection_reasons:
            rejected_count += 1
            for reason in rejection_reasons:
                rejected_counts[reason.value] += 1
                _append_once(aggregation_failures, reason)
            continue
        assert isinstance(raw_record, Stage4CaseRecord)
        test_count += 1
        case_reasons: list[str] = []
        label = _reference_label(raw_record.reference_label)
        if label is None:
            label = CalibrationLabel.UNKNOWN
            reason = Stage4AggregationFailureReason.INVALID_REFERENCE_LABEL
            case_reasons.append(reason.value)
            _append_once(aggregation_failures, reason)
        label_counts[label] += 1

        evaluation = raw_record.evaluation
        evaluation_valid = _evaluation_is_valid(evaluation)
        evidence_reasons = _evidence_validation_reasons(
            raw_record,
            evaluation if evaluation_valid and isinstance(evaluation, EvaluationResult) else None,
            locked_contract,
        )
        for evidence_reason in evidence_reasons:
            case_reasons.append(evidence_reason.value)
            _append_once(aggregation_failures, evidence_reason)
        if evaluation_valid:
            assert isinstance(evaluation, EvaluationResult)
            case_reasons.extend(reason.value for reason in evaluation.failure_reasons)
            if evaluation.case_id != raw_record.case_id:
                reason = Stage4AggregationFailureReason.CASE_ID_MISMATCH
                case_reasons.append(reason.value)
                _append_once(aggregation_failures, reason)
            for reason in _locked_contract_mismatches(evaluation, locked_contract):
                case_reasons.append(reason.value)
                _append_once(aggregation_failures, reason)
            if evaluation.out_of_distribution:
                ood_count += 1
            if (
                not evidence_reasons
                and isinstance(raw_record.dual_replay, Stage3DualReplayResult)
                and raw_record.dual_replay.passed
            ):
                witness_pass_count += 1
        else:
            reason = Stage4AggregationFailureReason.INVALID_EVALUATION
            case_reasons.append(reason.value)
            _append_once(aggregation_failures, reason)

        stage4_contract_failure = any(
            reason in {item.value for item in Stage4AggregationFailureReason}
            for reason in case_reasons
        )
        if (
            evaluation_valid
            and not stage4_contract_failure
            and isinstance(evaluation, EvaluationResult)
            and evaluation.status is EvaluationStatus.SUCCESS
        ):
            state = evaluation.state
            evaluable_count += 1
        else:
            state = EntailmentState.OUT_OF_MODEL
        state_counts[state] += 1
        for reason_name in dict.fromkeys(case_reasons):
            failure_counts[reason_name] += 1

    if test_count == 0:
        _append_once(
            aggregation_failures,
            Stage4AggregationFailureReason.EMPTY_TEST_SET,
        )
    unevaluable_count = test_count - evaluable_count
    witness_failed_count = test_count - witness_pass_count
    for state in EntailmentState:
        state_counts.setdefault(state, 0)
    for label in CalibrationLabel:
        label_counts.setdefault(label, 0)
    state_proportions = {
        state: wilson_score_proportion(
            state_counts[state],
            test_count,
            confidence_level=confidence,
        )
        for state in EntailmentState
    }
    report = Stage4EvaluationReport(
        scope=scope,
        status=(
            Stage4ReportStatus.FAIL_CLOSED if aggregation_failures else Stage4ReportStatus.SUCCESS
        ),
        locked_contract=locked_contract,
        confidence_level=confidence,
        input_record_count=len(records),
        test_case_count=test_count,
        rejected_record_count=rejected_count,
        evaluable_count=evaluable_count,
        unevaluable_count=unevaluable_count,
        fail_closed_count=unevaluable_count,
        out_of_distribution_count=ood_count,
        both_witness_replays_passed_count=witness_pass_count,
        one_or_more_witness_replays_failed_count=witness_failed_count,
        labeled_reference_count=(
            label_counts[CalibrationLabel.LESION_PRESENT]
            + label_counts[CalibrationLabel.LESION_ABSENT]
        ),
        state_counts=dict(state_counts),
        reference_label_counts=dict(label_counts),
        failure_reason_counts=dict(failure_counts),
        rejected_reason_counts=dict(rejected_counts),
        aggregation_failure_reasons=tuple(aggregation_failures),
        state_proportions=state_proportions,
        evaluable_proportion=wilson_score_proportion(
            evaluable_count, test_count, confidence_level=confidence
        ),
        unevaluable_proportion=wilson_score_proportion(
            unevaluable_count, test_count, confidence_level=confidence
        ),
        fail_closed_proportion=wilson_score_proportion(
            unevaluable_count, test_count, confidence_level=confidence
        ),
        out_of_distribution_proportion=wilson_score_proportion(
            ood_count, test_count, confidence_level=confidence
        ),
        both_witness_replays_passed_proportion=wilson_score_proportion(
            witness_pass_count, test_count, confidence_level=confidence
        ),
        one_or_more_witness_replays_failed_proportion=wilson_score_proportion(
            witness_failed_count, test_count, confidence_level=confidence
        ),
        _token=_REPORT_TOKEN,
    )
    # Prove the public payload cannot emit NaN/Infinity before returning it.
    json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)
    return report


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _plain_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0  # noqa: E721


def _validate_confidence(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    return float(value)


def _require_exact_keys(mapping: Mapping[_K, object], keys: tuple[_K, ...], name: str) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != set(keys):
        raise ValueError(f"{name} must contain every controlled key exactly once")


def _require_count_mapping(mapping: Mapping[_K, object], keys: tuple[_K, ...], name: str) -> None:
    _require_exact_keys(mapping, keys, name)
    if not all(_plain_nonnegative_int(value) for value in mapping.values()):
        raise ValueError(f"{name} values must be non-negative integers")


def _validate_reason_counts(mapping: Mapping[str, int], *, allowed: set[str]) -> None:
    if not isinstance(mapping, Mapping):
        raise TypeError("reason counts must be a mapping")
    if any(not isinstance(key, str) or not key for key in mapping):
        raise ValueError("reason count keys must be non-empty strings")
    if any(key not in allowed for key in mapping):
        raise ValueError("reason counts contain an uncontrolled reason")
    if any(not _plain_nonnegative_int(value) or value == 0 for value in mapping.values()):
        raise ValueError("reported reason counts must be positive integers")


def _duplicate_valid_ids(records: Sequence[object]) -> set[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if isinstance(record, Stage4CaseRecord) and _valid_identifier(record.case_id):
            counts[record.case_id] += 1
    return {case_id for case_id, count in counts.items() if count > 1}


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _record_rejection_reasons(
    value: object,
    duplicate_ids: set[str],
) -> tuple[Stage4AggregationFailureReason, ...]:
    if not isinstance(value, Stage4CaseRecord):
        return (Stage4AggregationFailureReason.INVALID_CASE_RECORD,)
    reasons: list[Stage4AggregationFailureReason] = []
    if not _valid_identifier(value.case_id):
        reasons.append(Stage4AggregationFailureReason.INVALID_CASE_ID)
    elif value.case_id in duplicate_ids:
        reasons.append(Stage4AggregationFailureReason.DUPLICATE_CASE_ID)
    split = value.split.value if isinstance(value.split, SampleSplit) else value.split
    if split != STAGE4_EXPECTED_SPLIT.value:
        reasons.append(Stage4AggregationFailureReason.NON_TEST_SPLIT)
    return tuple(reasons)


def _reference_label(value: object) -> CalibrationLabel | None:
    if isinstance(value, CalibrationLabel):
        return value
    if not isinstance(value, str):
        return None
    try:
        return CalibrationLabel(value)
    except (TypeError, ValueError):
        return None


def _evaluation_is_valid(value: object) -> bool:
    if not isinstance(value, EvaluationResult):
        return False
    try:
        value.__post_init__()
        json.dumps(value.to_dict(), allow_nan=False, sort_keys=True)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _certificate_is_valid(value: object) -> bool:
    if not isinstance(value, Stage3SyntheticCertificate):
        return False
    try:
        value.__post_init__()
        json.dumps(value.to_dict(), allow_nan=False, sort_keys=True)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _dual_replay_is_valid(value: object) -> bool:
    if not isinstance(value, Stage3DualReplayResult):
        return False
    try:
        value.positive.__post_init__()
        value.negative.__post_init__()
        value.__post_init__()
        json.dumps(value.to_dict(), allow_nan=False, sort_keys=True)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _evidence_validation_reasons(
    record: Stage4CaseRecord,
    evaluation: EvaluationResult | None,
    locked: LockedStage4Contract,
) -> tuple[Stage4AggregationFailureReason, ...]:
    reasons: list[Stage4AggregationFailureReason] = []
    certificate_valid = _certificate_is_valid(record.certificate)
    dual_replay_valid = _dual_replay_is_valid(record.dual_replay)
    if not certificate_valid:
        reasons.append(Stage4AggregationFailureReason.INVALID_CERTIFICATE)
    if not dual_replay_valid:
        reasons.append(Stage4AggregationFailureReason.INVALID_DUAL_REPLAY)
    if not certificate_valid or not isinstance(record.certificate, Stage3SyntheticCertificate):
        return tuple(reasons)
    certificate = record.certificate
    if evaluation is None or certificate.evaluation != evaluation:
        reasons.append(Stage4AggregationFailureReason.CERTIFICATE_EVALUATION_MISMATCH)
    if certificate.calibration_artifact_hash != locked.calibration_artifact_hash:
        reasons.append(Stage4AggregationFailureReason.CERTIFICATE_ARTIFACT_HASH_MISMATCH)
    if dual_replay_valid and isinstance(record.dual_replay, Stage3DualReplayResult):
        if (
            certificate.positive_evidence.replay != record.dual_replay.positive
            or certificate.negative_evidence.replay != record.dual_replay.negative
        ):
            reasons.append(Stage4AggregationFailureReason.CERTIFICATE_REPLAY_MISMATCH)
    return tuple(reasons)


def _locked_contract_mismatches(
    evaluation: EvaluationResult,
    locked: LockedStage4Contract,
) -> tuple[Stage4AggregationFailureReason, ...]:
    reasons: list[Stage4AggregationFailureReason] = []
    if evaluation.alpha != locked.alpha:
        reasons.append(Stage4AggregationFailureReason.LOCKED_ALPHA_MISMATCH)
    if evaluation.positive_threshold != locked.positive_threshold:
        reasons.append(Stage4AggregationFailureReason.LOCKED_POSITIVE_THRESHOLD_MISMATCH)
    if evaluation.negative_threshold != locked.negative_threshold:
        reasons.append(Stage4AggregationFailureReason.LOCKED_NEGATIVE_THRESHOLD_MISMATCH)
    if evaluation.positive_sample_count != locked.positive_sample_count:
        reasons.append(Stage4AggregationFailureReason.LOCKED_POSITIVE_SAMPLE_COUNT_MISMATCH)
    if evaluation.negative_sample_count != locked.negative_sample_count:
        reasons.append(Stage4AggregationFailureReason.LOCKED_NEGATIVE_SAMPLE_COUNT_MISMATCH)
    if evaluation.config_hash != locked.config_hash:
        reasons.append(Stage4AggregationFailureReason.LOCKED_CONFIG_HASH_MISMATCH)
    if evaluation.data_hash != locked.data_hash:
        reasons.append(Stage4AggregationFailureReason.LOCKED_DATA_HASH_MISMATCH)
    if evaluation.model_hash != locked.model_hash:
        reasons.append(Stage4AggregationFailureReason.LOCKED_MODEL_HASH_MISMATCH)
    return tuple(reasons)


def _append_once(
    target: list[Stage4AggregationFailureReason],
    value: Stage4AggregationFailureReason,
) -> None:
    if value not in target:
        target.append(value)


__all__ = [
    "STAGE4_EVALUATION_REPORT_SCHEMA_VERSION",
    "STAGE4_EXPECTED_SPLIT",
    "STAGE4_INTERVAL_METHOD",
    "BinomialProportion",
    "LockedStage4Contract",
    "Stage4AggregationFailureReason",
    "Stage4CaseRecord",
    "Stage4EvaluationReport",
    "Stage4EvaluationScope",
    "Stage4ReportStatus",
    "aggregate_locked_test_evaluations",
    "wilson_score_proportion",
]
