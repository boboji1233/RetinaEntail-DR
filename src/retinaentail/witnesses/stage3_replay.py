"""Generic, fail-closed Stage 3 replay for synthetic dual witnesses.

The predicate is deliberately supplied by the caller.  This module knows only the
positive/negative proposition semantics; it does not hard-code the toy predicate or
claim that the replay contract is a clinical lesion detector.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray

from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import replay_trace
from retinaentail.entailment.toy_baseline import normalized_mse
from retinaentail.schemas import (
    KNOWN_DEGRADATIONS,
    DegradationTrace,
    LesionRegion,
    WitnessRecord,
)
from retinaentail.witnesses.replay import canonical_json_bytes, load_rgb, sha256_bytes

STAGE3_REPLAY_SCHEMA_VERSION: Final = "3.0.0"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SIDES: Final = frozenset({"positive", "negative"})

WitnessPredicate = Callable[[NDArray[np.uint8], LesionRegion], bool]
_E = TypeVar("_E", bound=StrEnum)


class ReplayFailureReason(StrEnum):
    """Controlled reasons for rejecting one witness without leaking host paths."""

    INVALID_INPUT = "INVALID_INPUT"
    INVALID_EXPECTED_HASH = "INVALID_EXPECTED_HASH"
    INVALID_PREDICATE = "INVALID_PREDICATE"
    INVALID_ROOT = "INVALID_ROOT"
    CASE_MISMATCH = "CASE_MISMATCH"
    LESION_CLASS_MISMATCH = "LESION_CLASS_MISMATCH"
    SIDE_MISMATCH = "SIDE_MISMATCH"
    ARTIFACT_PATH_INVALID = "ARTIFACT_PATH_INVALID"
    ARTIFACT_PATH_ESCAPE = "ARTIFACT_PATH_ESCAPE"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_NOT_FILE = "ARTIFACT_NOT_FILE"
    LATENT_HASH_MISMATCH = "LATENT_HASH_MISMATCH"
    OBSERVATION_HASH_MISMATCH = "OBSERVATION_HASH_MISMATCH"
    TRACE_HASH_MISMATCH = "TRACE_HASH_MISMATCH"
    TRACE_SERIALIZATION_FAILED = "TRACE_SERIALIZATION_FAILED"
    UNKNOWN_DEGRADATION = "UNKNOWN_DEGRADATION"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    UNSUPPORTED_COORDINATE_SPACE = "UNSUPPORTED_COORDINATE_SPACE"
    REGION_OUT_OF_BOUNDS = "REGION_OUT_OF_BOUNDS"
    REPLAY_FAILED = "REPLAY_FAILED"
    NONFINITE_REPLAY_ERROR = "NONFINITE_REPLAY_ERROR"
    REPLAY_ERROR_MISMATCH = "REPLAY_ERROR_MISMATCH"
    PREDICATE_EXCEPTION = "PREDICATE_EXCEPTION"
    INVALID_PREDICATE_RESULT = "INVALID_PREDICATE_RESULT"
    PREDICATE_MISMATCH = "PREDICATE_MISMATCH"
    RECORDED_PREDICATE_INVALID = "RECORDED_PREDICATE_INVALID"
    MISSING_WITNESS = "MISSING_WITNESS"


class PairReplayFailureReason(StrEnum):
    """Pair-level failures in addition to each side's detailed reasons."""

    POSITIVE_REPLAY_FAILED = "POSITIVE_REPLAY_FAILED"
    NEGATIVE_REPLAY_FAILED = "NEGATIVE_REPLAY_FAILED"
    DUPLICATE_WITNESS_ID = "DUPLICATE_WITNESS_ID"


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3ReplayResult:
    """Path-free replay evidence for one explicitly expected hypothesis side."""

    witness_id: str | None
    case_id: str
    lesion_class: str
    side: str
    predicate_id: str
    region_sha256: str | None
    expected_latent_sha256: str | None
    actual_latent_sha256: str | None
    expected_observation_sha256: str | None
    actual_observation_sha256: str | None
    expected_trace_sha256: str | None
    actual_trace_sha256: str | None
    replay_error: float | None
    predicate_valid: bool
    passed: bool
    failure_reasons: tuple[ReplayFailureReason, ...] = field(default_factory=tuple)
    schema_version: str = STAGE3_REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage3ReplayResult schema_version")
        if self.side not in _SIDES:
            raise ValueError("replay side must be positive or negative")
        if not all(
            _valid_identifier(value)
            for value in (self.case_id, self.lesion_class, self.predicate_id)
        ):
            raise ValueError("replay identifiers must be non-empty strings")
        if self.witness_id is not None and not _valid_identifier(self.witness_id):
            raise ValueError("replay witness_id must be non-empty or null")
        hashes = (
            self.region_sha256,
            self.expected_latent_sha256,
            self.actual_latent_sha256,
            self.expected_observation_sha256,
            self.actual_observation_sha256,
            self.expected_trace_sha256,
            self.actual_trace_sha256,
        )
        if any(value is not None and not _valid_sha256(value) for value in hashes):
            raise ValueError("replay hashes must be lowercase SHA-256 values or null")
        if self.replay_error is not None and (
            isinstance(self.replay_error, bool)
            or not isinstance(self.replay_error, int | float)
            or not math.isfinite(float(self.replay_error))
            or float(self.replay_error) < 0.0
        ):
            raise ValueError("replay_error must be finite, non-negative, or null")
        if type(self.predicate_valid) is not bool or type(self.passed) is not bool:  # noqa: E721
            raise TypeError("replay flags must be exact booleans")
        if not all(isinstance(reason, ReplayFailureReason) for reason in self.failure_reasons):
            raise TypeError("replay failure reasons are invalid")
        if len(set(self.failure_reasons)) != len(self.failure_reasons):
            raise ValueError("replay failure reasons must be unique")
        if self.passed != (not self.failure_reasons):
            raise ValueError("passed must be true exactly when failure_reasons is empty")
        if self.passed and (
            self.witness_id is None
            or self.region_sha256 is None
            or self.expected_latent_sha256 is None
            or self.actual_latent_sha256 != self.expected_latent_sha256
            or self.expected_observation_sha256 is None
            or self.actual_observation_sha256 != self.expected_observation_sha256
            or self.expected_trace_sha256 is None
            or self.actual_trace_sha256 != self.expected_trace_sha256
            or self.replay_error is None
            or not self.predicate_valid
        ):
            raise ValueError("passed replay evidence is incomplete or inconsistent")

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-facing evidence with no filesystem paths."""

        return {
            "schema_version": self.schema_version,
            "witness_id": self.witness_id,
            "case_id": self.case_id,
            "lesion_class": self.lesion_class,
            "side": self.side,
            "predicate_id": self.predicate_id,
            "region_sha256": self.region_sha256,
            "expected_latent_sha256": self.expected_latent_sha256,
            "actual_latent_sha256": self.actual_latent_sha256,
            "expected_observation_sha256": self.expected_observation_sha256,
            "actual_observation_sha256": self.actual_observation_sha256,
            "expected_trace_sha256": self.expected_trace_sha256,
            "actual_trace_sha256": self.actual_trace_sha256,
            "replay_error": self.replay_error,
            "predicate_valid": self.predicate_valid,
            "passed": self.passed,
            "failure_reasons": [reason.value for reason in self.failure_reasons],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage3ReplayResult:
        """Strictly parse path-free replay evidence without type coercion."""

        fields = _strict_mapping(
            value,
            name="Stage3ReplayResult",
            required={
                "schema_version",
                "witness_id",
                "case_id",
                "lesion_class",
                "side",
                "predicate_id",
                "region_sha256",
                "expected_latent_sha256",
                "actual_latent_sha256",
                "expected_observation_sha256",
                "actual_observation_sha256",
                "expected_trace_sha256",
                "actual_trace_sha256",
                "replay_error",
                "predicate_valid",
                "passed",
                "failure_reasons",
            },
        )
        schema_version = _required_string(fields["schema_version"], "schema_version")
        if schema_version != STAGE3_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage3ReplayResult schema_version")
        side = _required_string(fields["side"], "side")
        if side not in _SIDES:
            raise ValueError("side must be positive or negative")
        witness_id = _optional_string(fields["witness_id"], "witness_id")
        expected_latent = _optional_sha256(
            fields["expected_latent_sha256"], "expected_latent_sha256"
        )
        actual_latent = _optional_sha256(fields["actual_latent_sha256"], "actual_latent_sha256")
        expected_observation = _optional_sha256(
            fields["expected_observation_sha256"], "expected_observation_sha256"
        )
        actual_observation = _optional_sha256(
            fields["actual_observation_sha256"], "actual_observation_sha256"
        )
        expected_trace = _optional_sha256(fields["expected_trace_sha256"], "expected_trace_sha256")
        actual_trace = _optional_sha256(fields["actual_trace_sha256"], "actual_trace_sha256")
        replay_error = _optional_finite_nonnegative(fields["replay_error"], "replay_error")
        predicate_valid = _exact_bool(fields["predicate_valid"], "predicate_valid")
        passed = _exact_bool(fields["passed"], "passed")
        reasons = _enum_tuple(fields["failure_reasons"], ReplayFailureReason, "failure_reasons")
        if passed != (not reasons):
            raise ValueError("passed must be true exactly when failure_reasons is empty")
        if passed and (
            witness_id is None
            or _optional_sha256(fields["region_sha256"], "region_sha256") is None
            or replay_error is None
            or not predicate_valid
            or expected_latent is None
            or actual_latent != expected_latent
            or expected_observation is None
            or actual_observation != expected_observation
            or expected_trace is None
            or actual_trace != expected_trace
        ):
            raise ValueError("passed replay evidence has incomplete or inconsistent hashes")
        return cls(
            schema_version=schema_version,
            witness_id=witness_id,
            case_id=_required_string(fields["case_id"], "case_id"),
            lesion_class=_required_string(fields["lesion_class"], "lesion_class"),
            side=side,
            predicate_id=_required_string(fields["predicate_id"], "predicate_id"),
            region_sha256=_optional_sha256(fields["region_sha256"], "region_sha256"),
            expected_latent_sha256=expected_latent,
            actual_latent_sha256=actual_latent,
            expected_observation_sha256=expected_observation,
            actual_observation_sha256=actual_observation,
            expected_trace_sha256=expected_trace,
            actual_trace_sha256=actual_trace,
            replay_error=replay_error,
            predicate_valid=predicate_valid,
            passed=passed,
            failure_reasons=reasons,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3DualReplayResult:
    """Independent positive and negative replays plus pair-level consistency."""

    positive: Stage3ReplayResult
    negative: Stage3ReplayResult
    passed: bool
    failure_reasons: tuple[PairReplayFailureReason, ...] = field(default_factory=tuple)
    schema_version: str = STAGE3_REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage3DualReplayResult schema_version")
        if not isinstance(self.positive, Stage3ReplayResult) or not isinstance(
            self.negative, Stage3ReplayResult
        ):
            raise TypeError("dual replay requires two Stage3ReplayResult values")
        if self.positive.side != "positive" or self.negative.side != "negative":
            raise ValueError("dual replay must contain positive then negative evidence")
        shared_context = (
            "case_id",
            "lesion_class",
            "predicate_id",
            "region_sha256",
            "expected_observation_sha256",
            "expected_trace_sha256",
        )
        if any(
            getattr(self.positive, name) != getattr(self.negative, name) for name in shared_context
        ):
            raise ValueError("dual replay sides do not share one declared context")
        if type(self.passed) is not bool:  # noqa: E721
            raise TypeError("dual replay passed must be an exact bool")
        if not all(isinstance(reason, PairReplayFailureReason) for reason in self.failure_reasons):
            raise TypeError("dual replay failure reasons are invalid")
        expected_reasons: list[PairReplayFailureReason] = []
        if not self.positive.passed:
            expected_reasons.append(PairReplayFailureReason.POSITIVE_REPLAY_FAILED)
        if not self.negative.passed:
            expected_reasons.append(PairReplayFailureReason.NEGATIVE_REPLAY_FAILED)
        if (
            self.positive.witness_id is not None
            and self.positive.witness_id == self.negative.witness_id
        ):
            expected_reasons.append(PairReplayFailureReason.DUPLICATE_WITNESS_ID)
        if self.failure_reasons != tuple(expected_reasons):
            raise ValueError("dual replay failure reasons are inconsistent")
        if self.passed is not (
            self.positive.passed and self.negative.passed and not self.failure_reasons
        ):
            raise ValueError("dual replay passed flag is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
            "passed": self.passed,
            "failure_reasons": [reason.value for reason in self.failure_reasons],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage3DualReplayResult:
        """Strictly parse a complete positive/negative replay pair."""

        fields = _strict_mapping(
            value,
            name="Stage3DualReplayResult",
            required={
                "schema_version",
                "positive",
                "negative",
                "passed",
                "failure_reasons",
            },
        )
        schema_version = _required_string(fields["schema_version"], "schema_version")
        if schema_version != STAGE3_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage3DualReplayResult schema_version")
        positive = Stage3ReplayResult.from_dict(_as_mapping(fields["positive"], "positive"))
        negative = Stage3ReplayResult.from_dict(_as_mapping(fields["negative"], "negative"))
        if positive.side != "positive" or negative.side != "negative":
            raise ValueError("dual replay must contain positive then negative evidence")
        passed = _exact_bool(fields["passed"], "passed")
        reasons = _enum_tuple(fields["failure_reasons"], PairReplayFailureReason, "failure_reasons")
        expected_reasons: list[PairReplayFailureReason] = []
        if not positive.passed:
            expected_reasons.append(PairReplayFailureReason.POSITIVE_REPLAY_FAILED)
        if not negative.passed:
            expected_reasons.append(PairReplayFailureReason.NEGATIVE_REPLAY_FAILED)
        if positive.witness_id is not None and positive.witness_id == negative.witness_id:
            expected_reasons.append(PairReplayFailureReason.DUPLICATE_WITNESS_ID)
        if reasons != tuple(expected_reasons):
            raise ValueError("dual replay failure reasons are inconsistent")
        expected_passed = positive.passed and negative.passed and not reasons
        if passed is not expected_passed:
            raise ValueError("dual replay passed flag is inconsistent")
        return cls(
            schema_version=schema_version,
            positive=positive,
            negative=negative,
            passed=passed,
            failure_reasons=reasons,
        )


class _ArtifactFailure(StrEnum):
    INVALID = "INVALID"
    ESCAPE = "ESCAPE"
    MISSING = "MISSING"
    NOT_FILE = "NOT_FILE"


def _strict_mapping(
    value: object,
    *,
    name: str,
    required: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    missing = required.difference(result)
    unknown = set(result).difference(required)
    if missing:
        raise ValueError(f"{name} is missing required fields")
    if unknown:
        raise ValueError(f"{name} contains unknown fields")
    return result


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _required_string(value, name)


def _required_sha256(value: object, name: str) -> str:
    if not _valid_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    return None if value is None else _required_sha256(value, name)


def _optional_finite_nonnegative(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be null or a finite non-negative number")
    return float(value)


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - strict artifact boundary
        raise ValueError(f"{name} must be a bool")
    assert isinstance(value, bool)
    return value


def _enum_tuple(
    value: object,
    enum_type: type[_E],
    name: str,
) -> tuple[_E, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    parsed: list[_E] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} entries must be strings")
        try:
            reason = enum_type(item)
        except ValueError as error:
            raise ValueError(f"{name} contains an unknown reason") from error
        if reason in parsed:
            raise ValueError(f"{name} cannot contain duplicate reasons")
        parsed.append(reason)
    return tuple(parsed)


def _append_once(items: list[ReplayFailureReason], value: ReplayFailureReason) -> None:
    if value not in items:
        items.append(value)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _region_sha256(region: LesionRegion) -> str:
    return sha256_bytes(canonical_json_bytes(region.to_dict()))


def _resolve_artifact(
    resolved_root: Path,
    relative_path: object,
) -> tuple[Path | None, _ArtifactFailure | None]:
    if not isinstance(relative_path, str) or not relative_path:
        return None, _ArtifactFailure.INVALID
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        "\\" in relative_path
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or not posix.parts
        or posix.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        return None, _ArtifactFailure.INVALID
    candidate = resolved_root.joinpath(*posix.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None, _ArtifactFailure.MISSING
    except (OSError, RuntimeError):
        return None, _ArtifactFailure.INVALID
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None, _ArtifactFailure.ESCAPE
    if not resolved.is_file():
        return None, _ArtifactFailure.NOT_FILE
    return resolved, None


def _artifact_reason(failure: _ArtifactFailure) -> ReplayFailureReason:
    return {
        _ArtifactFailure.INVALID: ReplayFailureReason.ARTIFACT_PATH_INVALID,
        _ArtifactFailure.ESCAPE: ReplayFailureReason.ARTIFACT_PATH_ESCAPE,
        _ArtifactFailure.MISSING: ReplayFailureReason.ARTIFACT_MISSING,
        _ArtifactFailure.NOT_FILE: ReplayFailureReason.ARTIFACT_NOT_FILE,
    }[failure]


def _trace_has_unknown_degradation(trace: DegradationTrace) -> bool:
    try:
        for operation in trace.operations:
            name = operation.get("name")
            if not isinstance(name, str) or name not in KNOWN_DEGRADATIONS:
                return True
    except (AttributeError, TypeError):
        return True
    return False


def replay_stage3_witness(
    root: object,
    witness: object,
    region: object,
    *,
    predicate: object,
    predicate_id: str,
    expected_case_id: str,
    expected_lesion_class: str,
    expected_side: str,
    observation_sha256: str,
    trace_sha256: str,
    tolerance: object = 1e-12,
) -> Stage3ReplayResult:
    """Replay one witness and return controlled, path-free validation evidence.

    Every expected failure is retained in ``failure_reasons``.  Positive and
    negative semantics require the injected predicate to return exactly ``True``
    and ``False`` respectively.
    """

    reasons: list[ReplayFailureReason] = []
    clean_tolerance = 0.0
    tolerance_valid = False
    if not isinstance(tolerance, bool) and isinstance(tolerance, int | float):
        clean_tolerance = float(tolerance)
        tolerance_valid = math.isfinite(clean_tolerance) and clean_tolerance >= 0.0
    safe_case_id = expected_case_id if _valid_identifier(expected_case_id) else "<invalid>"
    safe_lesion_class = (
        expected_lesion_class if _valid_identifier(expected_lesion_class) else "<invalid>"
    )
    safe_side = expected_side if expected_side in _SIDES else "positive"
    safe_predicate_id = predicate_id if _valid_identifier(predicate_id) else "<invalid>"
    safe_observation_hash = observation_sha256 if _valid_sha256(observation_sha256) else None
    safe_trace_hash = trace_sha256 if _valid_sha256(trace_sha256) else None
    region_digest: str | None = None
    if isinstance(region, LesionRegion):
        try:
            region_digest = _region_sha256(region)
        except (TypeError, ValueError):
            _append_once(reasons, ReplayFailureReason.INVALID_INPUT)

    if not isinstance(root, Path) or not isinstance(region, LesionRegion):
        _append_once(reasons, ReplayFailureReason.INVALID_INPUT)
    elif region.coordinate_space != "pixel":
        _append_once(reasons, ReplayFailureReason.UNSUPPORTED_COORDINATE_SPACE)
    if (
        not all(
            _valid_identifier(value)
            for value in (expected_case_id, expected_lesion_class, predicate_id)
        )
        or expected_side not in _SIDES
    ):
        _append_once(reasons, ReplayFailureReason.INVALID_INPUT)
    if not _valid_sha256(observation_sha256) or not _valid_sha256(trace_sha256):
        _append_once(reasons, ReplayFailureReason.INVALID_EXPECTED_HASH)
    if not callable(predicate):
        _append_once(reasons, ReplayFailureReason.INVALID_PREDICATE)
    if not tolerance_valid:
        _append_once(reasons, ReplayFailureReason.INVALID_INPUT)
    if witness is None:
        _append_once(reasons, ReplayFailureReason.MISSING_WITNESS)
    elif not isinstance(witness, WitnessRecord):
        _append_once(reasons, ReplayFailureReason.INVALID_INPUT)

    witness_id = (
        witness.witness_id
        if isinstance(witness, WitnessRecord) and _valid_identifier(witness.witness_id)
        else None
    )
    expected_latent_hash = (
        witness.sha256
        if isinstance(witness, WitnessRecord) and _valid_sha256(witness.sha256)
        else None
    )
    actual_latent_hash: str | None = None
    actual_observation_hash: str | None = None
    actual_trace_hash: str | None = None
    replay_error: float | None = None
    predicate_valid = False

    resolved_root: Path | None = None
    if isinstance(root, Path):
        try:
            resolved_root = root.resolve(strict=True)
            if not resolved_root.is_dir():
                resolved_root = None
        except (OSError, RuntimeError, ValueError):
            resolved_root = None
        if resolved_root is None:
            _append_once(reasons, ReplayFailureReason.INVALID_ROOT)

    latent_path: Path | None = None
    observation_path: Path | None = None
    if isinstance(witness, WitnessRecord):
        if witness.case_id != expected_case_id:
            _append_once(reasons, ReplayFailureReason.CASE_MISMATCH)
        if witness.lesion_class != expected_lesion_class:
            _append_once(reasons, ReplayFailureReason.LESION_CLASS_MISMATCH)
        if witness.hypothesis_side != expected_side:
            _append_once(reasons, ReplayFailureReason.SIDE_MISMATCH)
        if not _valid_sha256(witness.sha256):
            _append_once(reasons, ReplayFailureReason.INVALID_EXPECTED_HASH)
        if type(witness.predicate_valid) is not bool or not witness.predicate_valid:  # noqa: E721
            _append_once(reasons, ReplayFailureReason.RECORDED_PREDICATE_INVALID)

        if resolved_root is not None:
            latent_path, latent_failure = _resolve_artifact(resolved_root, witness.relative_path)
            observation_path, observation_failure = _resolve_artifact(
                resolved_root, witness.observation_path
            )
            for failure in (latent_failure, observation_failure):
                if failure is not None:
                    _append_once(reasons, _artifact_reason(failure))

        try:
            actual_trace_hash = sha256_bytes(
                canonical_json_bytes(witness.degradation_trace.to_dict())
            )
        except Exception:  # noqa: BLE001 - untrusted trace serialization boundary
            _append_once(reasons, ReplayFailureReason.TRACE_SERIALIZATION_FAILED)
        else:
            if actual_trace_hash != trace_sha256:
                _append_once(reasons, ReplayFailureReason.TRACE_HASH_MISMATCH)
        if not isinstance(witness.degradation_trace, DegradationTrace) or (
            _trace_has_unknown_degradation(witness.degradation_trace)
        ):
            _append_once(reasons, ReplayFailureReason.UNKNOWN_DEGRADATION)

    if latent_path is not None:
        try:
            actual_latent_hash = sha256_file(latent_path)
        except OSError:
            _append_once(reasons, ReplayFailureReason.ARTIFACT_NOT_FILE)
        else:
            if actual_latent_hash != expected_latent_hash:
                _append_once(reasons, ReplayFailureReason.LATENT_HASH_MISMATCH)
    if observation_path is not None:
        try:
            actual_observation_hash = sha256_file(observation_path)
        except OSError:
            _append_once(reasons, ReplayFailureReason.ARTIFACT_NOT_FILE)
        else:
            if actual_observation_hash != observation_sha256:
                _append_once(reasons, ReplayFailureReason.OBSERVATION_HASH_MISMATCH)

    latent: NDArray[np.uint8] | None = None
    observation: NDArray[np.uint8] | None = None
    if latent_path is not None:
        try:
            latent = load_rgb(latent_path)
        except Exception:  # noqa: BLE001 - untrusted image decoding boundary
            _append_once(reasons, ReplayFailureReason.IMAGE_DECODE_FAILED)
    if observation_path is not None:
        try:
            observation = load_rgb(observation_path)
        except Exception:  # noqa: BLE001 - untrusted image decoding boundary
            _append_once(reasons, ReplayFailureReason.IMAGE_DECODE_FAILED)

    if (
        latent is not None
        and observation is not None
        and isinstance(witness, WitnessRecord)
        and ReplayFailureReason.UNKNOWN_DEGRADATION not in reasons
    ):
        try:
            replayed = replay_trace(latent, witness.degradation_trace.to_dict())
            replay_error = normalized_mse(replayed, observation)
        except Exception:  # noqa: BLE001 - degradation replay is a fail-closed boundary
            _append_once(reasons, ReplayFailureReason.REPLAY_FAILED)
        else:
            if not math.isfinite(replay_error):
                replay_error = None
                _append_once(reasons, ReplayFailureReason.NONFINITE_REPLAY_ERROR)
            elif (
                isinstance(witness.replay_error, bool)
                or not isinstance(witness.replay_error, int | float)
                or not math.isfinite(float(witness.replay_error))
                or abs(replay_error - float(witness.replay_error)) > clean_tolerance
            ):
                _append_once(reasons, ReplayFailureReason.REPLAY_ERROR_MISMATCH)

    region_contained = False
    if (
        latent is not None
        and observation is not None
        and isinstance(region, LesionRegion)
        and region.coordinate_space == "pixel"
    ):
        x1 = region.x + region.width
        y1 = region.y + region.height
        region_contained = (
            region.x >= 0
            and region.y >= 0
            and x1 <= latent.shape[1]
            and y1 <= latent.shape[0]
            and x1 <= observation.shape[1]
            and y1 <= observation.shape[0]
        )
        if not region_contained:
            _append_once(reasons, ReplayFailureReason.REGION_OUT_OF_BOUNDS)

    if latent is not None and region_contained and callable(predicate):
        try:
            predicate_result = predicate(latent.copy(), region)
        except Exception:  # noqa: BLE001 - untrusted research predicate is a fail-closed boundary
            _append_once(reasons, ReplayFailureReason.PREDICATE_EXCEPTION)
        else:
            if type(predicate_result) is not bool:  # noqa: E721 - reject numpy.bool_ coercion
                _append_once(reasons, ReplayFailureReason.INVALID_PREDICATE_RESULT)
            else:
                predicate_valid = predicate_result is (expected_side == "positive")
                if not predicate_valid:
                    _append_once(reasons, ReplayFailureReason.PREDICATE_MISMATCH)

    return Stage3ReplayResult(
        witness_id=witness_id,
        case_id=safe_case_id,
        lesion_class=safe_lesion_class,
        side=safe_side,
        predicate_id=safe_predicate_id,
        region_sha256=region_digest,
        expected_latent_sha256=expected_latent_hash,
        actual_latent_sha256=actual_latent_hash,
        expected_observation_sha256=safe_observation_hash,
        actual_observation_sha256=actual_observation_hash,
        expected_trace_sha256=safe_trace_hash,
        actual_trace_sha256=actual_trace_hash,
        replay_error=replay_error,
        predicate_valid=predicate_valid,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def replay_stage3_witness_pair(
    root: object,
    *,
    positive_witness: object,
    negative_witness: object,
    region: object,
    predicate: object,
    predicate_id: str,
    expected_case_id: str,
    expected_lesion_class: str,
    observation_sha256: str,
    trace_sha256: str,
    tolerance: object = 1e-12,
) -> Stage3DualReplayResult:
    """Replay both sides independently, even when the first side fails."""

    positive = replay_stage3_witness(
        root=root,
        witness=positive_witness,
        region=region,
        predicate=predicate,
        predicate_id=predicate_id,
        expected_case_id=expected_case_id,
        expected_lesion_class=expected_lesion_class,
        expected_side="positive",
        observation_sha256=observation_sha256,
        trace_sha256=trace_sha256,
        tolerance=tolerance,
    )
    negative = replay_stage3_witness(
        root=root,
        witness=negative_witness,
        region=region,
        predicate=predicate,
        predicate_id=predicate_id,
        expected_case_id=expected_case_id,
        expected_lesion_class=expected_lesion_class,
        expected_side="negative",
        observation_sha256=observation_sha256,
        trace_sha256=trace_sha256,
        tolerance=tolerance,
    )
    pair_reasons: list[PairReplayFailureReason] = []
    if not positive.passed:
        pair_reasons.append(PairReplayFailureReason.POSITIVE_REPLAY_FAILED)
    if not negative.passed:
        pair_reasons.append(PairReplayFailureReason.NEGATIVE_REPLAY_FAILED)
    if positive.witness_id is not None and positive.witness_id == negative.witness_id:
        pair_reasons.append(PairReplayFailureReason.DUPLICATE_WITNESS_ID)
    return Stage3DualReplayResult(
        positive=positive,
        negative=negative,
        passed=not pair_reasons,
        failure_reasons=tuple(pair_reasons),
    )


__all__ = [
    "PairReplayFailureReason",
    "ReplayFailureReason",
    "STAGE3_REPLAY_SCHEMA_VERSION",
    "Stage3DualReplayResult",
    "Stage3ReplayResult",
    "WitnessPredicate",
    "replay_stage3_witness",
    "replay_stage3_witness_pair",
]
