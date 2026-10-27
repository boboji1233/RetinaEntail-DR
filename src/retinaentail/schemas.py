"""Strict, versioned standard-library schemas for RetinaEntail-DR artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final, TypeAlias

from retinaentail.entailment.states import (
    EntailmentState,
    fail_closed_state,
    solver_status_succeeded,
)

SCHEMA_VERSION: Final = "1.0.0"
_MAX_SEED: Final = 2**64 - 1
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_ENV_PATTERN: Final = re.compile(r"^[A-Z_][A-Z0-9_]*$")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

KNOWN_DEGRADATIONS: Final[frozenset[str]] = frozenset(
    {
        "identity",
        "gaussian_blur",
        "defocus_blur",
        "motion_blur",
        "downsample_restore",
        "jpeg_compression",
        "poisson_gaussian_noise",
        "gamma_exposure",
        "vignetting",
        "nonuniform_illumination",
    }
)


class SchemaValidationError(ValueError):
    """Raised when an artifact does not satisfy its declared schema version."""


def _strict_mapping(
    value: object,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SchemaValidationError(f"{name} keys must be strings")
        result[key] = item
    allowed = required | (set() if optional is None else optional)
    missing = sorted(required.difference(result))
    unknown = sorted(set(result).difference(allowed))
    if missing:
        raise SchemaValidationError(f"{name} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise SchemaValidationError(f"{name} has unknown field(s): {', '.join(unknown)}")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _schema_version(value: object) -> str:
    version = _nonempty_string(value, "schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    return version


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - exact bool is a schema boundary
        raise SchemaValidationError(f"{name} must be a bool")
    return value


def _bounded_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "unbounded" if maximum is None else str(maximum)
        raise SchemaValidationError(f"{name} must be in [{minimum}, {upper}]")
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{name} must be finite")
    if minimum is not None:
        too_small = result < minimum if minimum_inclusive else result <= minimum
        if too_small:
            operator = ">=" if minimum_inclusive else ">"
            raise SchemaValidationError(f"{name} must be {operator} {minimum}")
    if maximum is not None:
        too_large = result > maximum if maximum_inclusive else result >= maximum
        if too_large:
            operator = "<=" if maximum_inclusive else "<"
            raise SchemaValidationError(f"{name} must be {operator} {maximum}")
    return result


def _sha256(value: object, name: str) -> str:
    digest = _nonempty_string(value, name)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise SchemaValidationError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return digest


def _relative_path(value: object, name: str) -> str:
    raw = _nonempty_string(value, name)
    normalized = raw.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(raw)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise SchemaValidationError(f"{name} must be relative")
    if (
        not posix_path.parts
        or normalized != posix_path.as_posix()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise SchemaValidationError(f"{name} must be a normalized relative path")
    return posix_path.as_posix()


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{name} must be a sequence of strings")
    return tuple(_nonempty_string(item, f"{name} item") for item in value)


def _json_value(value: object, name: str) -> JSONValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{name} mapping keys must be strings")
            result[key] = _json_value(item, f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_value(item, f"{name} item") for item in value]
    raise SchemaValidationError(f"{name} contains a non-JSON value")


def _hash_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        clean_key = _nonempty_string(key, f"{name} key")
        result[clean_key] = _sha256(digest, f"{name}[{clean_key!r}]")
    if not result:
        raise SchemaValidationError(f"{name} must contain at least one model hash")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetRecord:
    """Dataset registry entry containing provenance and access constraints."""

    dataset_id: str
    source_url: str
    access_mode: str
    license_status: str
    local_root_env: str
    expected_labels: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "dataset_id", _nonempty_string(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "source_url", _nonempty_string(self.source_url, "source_url"))
        object.__setattr__(self, "access_mode", _nonempty_string(self.access_mode, "access_mode"))
        object.__setattr__(
            self, "license_status", _nonempty_string(self.license_status, "license_status")
        )
        root_env = _nonempty_string(self.local_root_env, "local_root_env")
        if _ENV_PATTERN.fullmatch(root_env) is None:
            raise SchemaValidationError("local_root_env must be an uppercase environment variable")
        object.__setattr__(self, "local_root_env", root_env)
        object.__setattr__(
            self, "expected_labels", _string_tuple(self.expected_labels, "expected_labels")
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "access_mode": self.access_mode,
            "license_status": self.license_status,
            "local_root_env": self.local_root_env,
            "expected_labels": list(self.expected_labels),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DatasetRecord:
        values = _strict_mapping(
            data,
            name="DatasetRecord",
            required={
                "schema_version",
                "dataset_id",
                "source_url",
                "access_mode",
                "license_status",
                "local_root_env",
                "expected_labels",
            },
        )
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            dataset_id=_nonempty_string(values["dataset_id"], "dataset_id"),
            source_url=_nonempty_string(values["source_url"], "source_url"),
            access_mode=_nonempty_string(values["access_mode"], "access_mode"),
            license_status=_nonempty_string(values["license_status"], "license_status"),
            local_root_env=_nonempty_string(values["local_root_env"], "local_root_env"),
            expected_labels=_string_tuple(values["expected_labels"], "expected_labels"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRecord:
    """One image row in a content-addressed manifest."""

    relative_path: str
    sha256: str
    width: int
    height: int
    channels: int
    source_image_id: str
    patient_id: str | None = None
    label_path: str | None = None
    split: str | None = None
    intervention_pair_id: str | None = None
    dataset_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "relative_path", _relative_path(self.relative_path, "relative_path")
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(self, "width", _bounded_int(self.width, "width", minimum=1))
        object.__setattr__(self, "height", _bounded_int(self.height, "height", minimum=1))
        object.__setattr__(self, "channels", _bounded_int(self.channels, "channels", minimum=1))
        object.__setattr__(
            self, "source_image_id", _nonempty_string(self.source_image_id, "source_image_id")
        )
        object.__setattr__(self, "patient_id", _optional_string(self.patient_id, "patient_id"))
        if self.label_path is not None:
            object.__setattr__(self, "label_path", _relative_path(self.label_path, "label_path"))
        object.__setattr__(self, "split", _optional_string(self.split, "split"))
        object.__setattr__(
            self,
            "intervention_pair_id",
            _optional_string(self.intervention_pair_id, "intervention_pair_id"),
        )
        object.__setattr__(self, "dataset_id", _optional_string(self.dataset_id, "dataset_id"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "source_image_id": self.source_image_id,
            "patient_id": self.patient_id,
            "label_path": self.label_path,
            "split": self.split,
            "intervention_pair_id": self.intervention_pair_id,
            "dataset_id": self.dataset_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ImageRecord:
        values = _strict_mapping(
            data,
            name="ImageRecord",
            required={
                "schema_version",
                "relative_path",
                "sha256",
                "width",
                "height",
                "channels",
                "source_image_id",
            },
            optional={"patient_id", "label_path", "split", "intervention_pair_id", "dataset_id"},
        )
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            relative_path=_relative_path(values["relative_path"], "relative_path"),
            sha256=_sha256(values["sha256"], "sha256"),
            width=_bounded_int(values["width"], "width", minimum=1),
            height=_bounded_int(values["height"], "height", minimum=1),
            channels=_bounded_int(values["channels"], "channels", minimum=1),
            source_image_id=_nonempty_string(values["source_image_id"], "source_image_id"),
            patient_id=_optional_string(values.get("patient_id"), "patient_id"),
            label_path=(
                None
                if values.get("label_path") is None
                else _relative_path(values["label_path"], "label_path")
            ),
            split=_optional_string(values.get("split"), "split"),
            intervention_pair_id=_optional_string(
                values.get("intervention_pair_id"), "intervention_pair_id"
            ),
            dataset_id=_optional_string(values.get("dataset_id"), "dataset_id"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LesionRegion:
    """Axis-aligned candidate lesion region in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    region_id: str | None = None
    coordinate_space: str = "pixel"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "x", _bounded_int(self.x, "x", minimum=0))
        object.__setattr__(self, "y", _bounded_int(self.y, "y", minimum=0))
        object.__setattr__(self, "width", _bounded_int(self.width, "width", minimum=1))
        object.__setattr__(self, "height", _bounded_int(self.height, "height", minimum=1))
        object.__setattr__(self, "region_id", _optional_string(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "coordinate_space",
            _nonempty_string(self.coordinate_space, "coordinate_space"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "region_id": self.region_id,
            "coordinate_space": self.coordinate_space,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> LesionRegion:
        values = _strict_mapping(
            data,
            name="LesionRegion",
            required={"schema_version", "x", "y", "width", "height"},
            optional={"region_id", "coordinate_space"},
        )
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            x=_bounded_int(values["x"], "x", minimum=0),
            y=_bounded_int(values["y"], "y", minimum=0),
            width=_bounded_int(values["width"], "width", minimum=1),
            height=_bounded_int(values["height"], "height", minimum=1),
            region_id=_optional_string(values.get("region_id"), "region_id"),
            coordinate_space=_nonempty_string(
                values.get("coordinate_space", "pixel"), "coordinate_space"
            ),
        )


def _degradation_operation(value: object, index: int) -> dict[str, JSONValue]:
    name = f"operations[{index}]"
    values = _strict_mapping(
        value,
        name=name,
        required={"name", "severity", "seed", "parameters"},
    )
    operator_name = _nonempty_string(values["name"], f"{name}.name")
    if operator_name not in KNOWN_DEGRADATIONS:
        raise SchemaValidationError(f"unknown degradation operator: {operator_name!r}")
    severity = _finite_float(values["severity"], f"{name}.severity", minimum=0.0, maximum=1.0)
    seed = _bounded_int(values["seed"], f"{name}.seed", minimum=0, maximum=_MAX_SEED)
    parameters = _json_value(values["parameters"], f"{name}.parameters")
    if not isinstance(parameters, dict):
        raise SchemaValidationError(f"{name}.parameters must be a mapping")
    return {
        "name": operator_name,
        "severity": severity,
        "seed": seed,
        "parameters": parameters,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class DegradationTrace:
    """Ordered, JSON-safe record of deterministic degradation operations."""

    operations: tuple[dict[str, JSONValue], ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        normalized = tuple(
            _degradation_operation(operation, index)
            for index, operation in enumerate(self.operations)
        )
        object.__setattr__(self, "operations", normalized)

    def to_dict(self) -> dict[str, JSONValue]:
        operations: list[JSONValue] = []
        for operation in self.operations:
            copied = _json_value(operation, "operation")
            operations.append(copied)
        return {"schema_version": self.schema_version, "operations": operations}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DegradationTrace:
        values = _strict_mapping(
            data,
            name="DegradationTrace",
            required={"schema_version", "operations"},
        )
        operations = values["operations"]
        if isinstance(operations, str | bytes) or not isinstance(operations, Sequence):
            raise SchemaValidationError("operations must be a sequence")
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            operations=tuple(
                _degradation_operation(operation, index)
                for index, operation in enumerate(operations)
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HypothesisScore:
    """Score, calibrated threshold, feasibility, and retained solver status."""

    score: float
    threshold: float
    feasible: bool
    solver_status: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "score", _finite_float(self.score, "score"))
        object.__setattr__(self, "threshold", _finite_float(self.threshold, "threshold"))
        object.__setattr__(self, "feasible", _exact_bool(self.feasible, "feasible"))
        object.__setattr__(
            self, "solver_status", _nonempty_string(self.solver_status, "solver_status")
        )

    @property
    def decision_ready(self) -> bool:
        return solver_status_succeeded(self.solver_status)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "score": self.score,
            "threshold": self.threshold,
            "feasible": self.feasible,
            "solver_status": self.solver_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HypothesisScore:
        values = _strict_mapping(
            data,
            name="HypothesisScore",
            required={"schema_version", "score", "threshold", "feasible", "solver_status"},
        )
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            score=_finite_float(values["score"], "score"),
            threshold=_finite_float(values["threshold"], "threshold"),
            feasible=_exact_bool(values["feasible"], "feasible"),
            solver_status=_nonempty_string(values["solver_status"], "solver_status"),
        )


_POSITIVE_WITNESS_KINDS: Final = frozenset({"positive", "lesion_present"})
_NEGATIVE_WITNESS_KINDS: Final = frozenset({"negative", "lesion_absent"})
_WITNESS_KINDS: Final = _POSITIVE_WITNESS_KINDS | _NEGATIVE_WITNESS_KINDS


@dataclass(frozen=True, slots=True, kw_only=True)
class WitnessRecord:
    """Content-addressed replay artifact for one side of the lesion proposition."""

    witness_id: str
    case_id: str
    lesion_class: str
    witness_kind: str
    relative_path: str
    observation_path: str
    sha256: str
    degradation_trace: DegradationTrace
    replay_seed: int
    replay_error: float
    predicate_valid: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "witness_id", _nonempty_string(self.witness_id, "witness_id"))
        object.__setattr__(self, "case_id", _nonempty_string(self.case_id, "case_id"))
        object.__setattr__(
            self, "lesion_class", _nonempty_string(self.lesion_class, "lesion_class")
        )
        witness_kind = _nonempty_string(self.witness_kind, "witness_kind").lower()
        if witness_kind not in _WITNESS_KINDS:
            raise SchemaValidationError(
                "witness_kind must be positive/negative or lesion_present/lesion_absent"
            )
        object.__setattr__(self, "witness_kind", witness_kind)
        object.__setattr__(
            self, "relative_path", _relative_path(self.relative_path, "relative_path")
        )
        object.__setattr__(
            self, "observation_path", _relative_path(self.observation_path, "observation_path")
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        if not isinstance(self.degradation_trace, DegradationTrace):
            raise SchemaValidationError("degradation_trace must be a DegradationTrace")
        object.__setattr__(
            self,
            "replay_seed",
            _bounded_int(self.replay_seed, "replay_seed", minimum=0, maximum=_MAX_SEED),
        )
        object.__setattr__(
            self,
            "replay_error",
            _finite_float(self.replay_error, "replay_error", minimum=0.0),
        )
        object.__setattr__(
            self, "predicate_valid", _exact_bool(self.predicate_valid, "predicate_valid")
        )

    @property
    def hypothesis_side(self) -> str:
        return "positive" if self.witness_kind in _POSITIVE_WITNESS_KINDS else "negative"

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "witness_id": self.witness_id,
            "case_id": self.case_id,
            "lesion_class": self.lesion_class,
            "witness_kind": self.witness_kind,
            "relative_path": self.relative_path,
            "observation_path": self.observation_path,
            "sha256": self.sha256,
            "degradation_trace": self.degradation_trace.to_dict(),
            "replay_seed": self.replay_seed,
            "replay_error": self.replay_error,
            "predicate_valid": self.predicate_valid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> WitnessRecord:
        values = _strict_mapping(
            data,
            name="WitnessRecord",
            required={
                "schema_version",
                "witness_id",
                "case_id",
                "lesion_class",
                "witness_kind",
                "relative_path",
                "observation_path",
                "sha256",
                "degradation_trace",
                "replay_seed",
                "replay_error",
                "predicate_valid",
            },
        )
        trace_data = _strict_mapping(
            values["degradation_trace"],
            name="degradation_trace",
            required={"schema_version", "operations"},
        )
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            witness_id=_nonempty_string(values["witness_id"], "witness_id"),
            case_id=_nonempty_string(values["case_id"], "case_id"),
            lesion_class=_nonempty_string(values["lesion_class"], "lesion_class"),
            witness_kind=_nonempty_string(values["witness_kind"], "witness_kind"),
            relative_path=_relative_path(values["relative_path"], "relative_path"),
            observation_path=_relative_path(values["observation_path"], "observation_path"),
            sha256=_sha256(values["sha256"], "sha256"),
            degradation_trace=DegradationTrace.from_dict(trace_data),
            replay_seed=_bounded_int(
                values["replay_seed"], "replay_seed", minimum=0, maximum=_MAX_SEED
            ),
            replay_error=_finite_float(values["replay_error"], "replay_error", minimum=0.0),
            predicate_valid=_exact_bool(values["predicate_valid"], "predicate_valid"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EntailmentCertificate:
    """Auditable result binding scores, thresholds, provenance, state, and witnesses."""

    case_id: str
    lesion_class: str
    region: LesionRegion
    alpha: float
    positive: HypothesisScore
    negative: HypothesisScore
    state: EntailmentState
    config_hash: str
    data_hash: str
    model_hashes: dict[str, str]
    solver_status: str
    witnesses: tuple[WitnessRecord, ...] = field(default_factory=tuple)
    degradation_trace: DegradationTrace | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "case_id", _nonempty_string(self.case_id, "case_id"))
        object.__setattr__(
            self, "lesion_class", _nonempty_string(self.lesion_class, "lesion_class")
        )
        if not isinstance(self.region, LesionRegion):
            raise SchemaValidationError("region must be a LesionRegion")
        object.__setattr__(
            self,
            "alpha",
            _finite_float(
                self.alpha,
                "alpha",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
                maximum_inclusive=False,
            ),
        )
        if not isinstance(self.positive, HypothesisScore):
            raise SchemaValidationError("positive must be a HypothesisScore")
        if not isinstance(self.negative, HypothesisScore):
            raise SchemaValidationError("negative must be a HypothesisScore")
        try:
            state = (
                self.state
                if isinstance(self.state, EntailmentState)
                else EntailmentState(self.state)
            )
        except (TypeError, ValueError) as error:
            raise SchemaValidationError("state must be a valid EntailmentState") from error
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "config_hash", _sha256(self.config_hash, "config_hash"))
        object.__setattr__(self, "data_hash", _sha256(self.data_hash, "data_hash"))
        object.__setattr__(self, "model_hashes", _hash_mapping(self.model_hashes, "model_hashes"))
        object.__setattr__(
            self, "solver_status", _nonempty_string(self.solver_status, "solver_status")
        )

        normalized_witnesses: list[WitnessRecord] = []
        sides: set[str] = set()
        witnesses_valid = True
        for witness in self.witnesses:
            if not isinstance(witness, WitnessRecord):
                raise SchemaValidationError("each witness must be a WitnessRecord")
            if witness.case_id != self.case_id or witness.lesion_class != self.lesion_class:
                raise SchemaValidationError(
                    "witness case_id and lesion_class must match certificate"
                )
            if witness.hypothesis_side in sides:
                raise SchemaValidationError("certificate cannot contain duplicate witness sides")
            sides.add(witness.hypothesis_side)
            witnesses_valid = witnesses_valid and witness.predicate_valid
            normalized_witnesses.append(witness)
        object.__setattr__(self, "witnesses", tuple(normalized_witnesses))

        if self.degradation_trace is not None and not isinstance(
            self.degradation_trace, DegradationTrace
        ):
            raise SchemaValidationError("degradation_trace must be a DegradationTrace or None")

        expected_state = fail_closed_state(
            positive_feasible=self.positive.feasible,
            negative_feasible=self.negative.feasible,
            positive_score=self.positive.score,
            negative_score=self.negative.score,
            positive_threshold=self.positive.threshold,
            negative_threshold=self.negative.threshold,
            positive_solver_status=self.positive.solver_status,
            negative_solver_status=self.negative.solver_status,
            solver_status=self.solver_status,
            degradation_known=True,
        )
        if not witnesses_valid:
            expected_state = EntailmentState.OUT_OF_MODEL

        if state is EntailmentState.OUT_OF_MODEL:
            return
        if expected_state is EntailmentState.OUT_OF_MODEL:
            object.__setattr__(self, "state", EntailmentState.OUT_OF_MODEL)
        elif state is not expected_state:
            raise SchemaValidationError(
                "state "
                f"{state.value} conflicts with feasibility truth table ({expected_state.value})"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "lesion_class": self.lesion_class,
            "region": self.region.to_dict(),
            "alpha": self.alpha,
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
            "state": self.state.value,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hashes": dict(self.model_hashes),
            "solver_status": self.solver_status,
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "degradation_trace": (
                None if self.degradation_trace is None else self.degradation_trace.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EntailmentCertificate:
        values = _strict_mapping(
            data,
            name="EntailmentCertificate",
            required={
                "schema_version",
                "case_id",
                "lesion_class",
                "region",
                "alpha",
                "positive",
                "negative",
                "state",
                "config_hash",
                "data_hash",
                "model_hashes",
                "solver_status",
            },
            optional={"witnesses", "degradation_trace"},
        )
        region_data = _strict_mapping(
            values["region"],
            name="region",
            required={"schema_version", "x", "y", "width", "height"},
            optional={"region_id", "coordinate_space"},
        )
        positive_data = _strict_mapping(
            values["positive"],
            name="positive",
            required={"schema_version", "score", "threshold", "feasible", "solver_status"},
        )
        negative_data = _strict_mapping(
            values["negative"],
            name="negative",
            required={"schema_version", "score", "threshold", "feasible", "solver_status"},
        )

        raw_witnesses = values.get("witnesses", ())
        if isinstance(raw_witnesses, str | bytes) or not isinstance(raw_witnesses, Sequence):
            raise SchemaValidationError("witnesses must be a sequence")
        witnesses: list[WitnessRecord] = []
        for index, raw_witness in enumerate(raw_witnesses):
            witness_data = _strict_mapping(
                raw_witness,
                name=f"witnesses[{index}]",
                required={
                    "schema_version",
                    "witness_id",
                    "case_id",
                    "lesion_class",
                    "witness_kind",
                    "relative_path",
                    "observation_path",
                    "sha256",
                    "degradation_trace",
                    "replay_seed",
                    "replay_error",
                    "predicate_valid",
                },
            )
            witnesses.append(WitnessRecord.from_dict(witness_data))

        raw_trace = values.get("degradation_trace")
        trace: DegradationTrace | None = None
        if raw_trace is not None:
            trace_data = _strict_mapping(
                raw_trace,
                name="degradation_trace",
                required={"schema_version", "operations"},
            )
            trace = DegradationTrace.from_dict(trace_data)

        try:
            state = EntailmentState(_nonempty_string(values["state"], "state"))
        except ValueError as error:
            raise SchemaValidationError("state must be a valid EntailmentState") from error
        return cls(
            schema_version=_schema_version(values["schema_version"]),
            case_id=_nonempty_string(values["case_id"], "case_id"),
            lesion_class=_nonempty_string(values["lesion_class"], "lesion_class"),
            region=LesionRegion.from_dict(region_data),
            alpha=_finite_float(
                values["alpha"],
                "alpha",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
                maximum_inclusive=False,
            ),
            positive=HypothesisScore.from_dict(positive_data),
            negative=HypothesisScore.from_dict(negative_data),
            state=state,
            config_hash=_sha256(values["config_hash"], "config_hash"),
            data_hash=_sha256(values["data_hash"], "data_hash"),
            model_hashes=_hash_mapping(values["model_hashes"], "model_hashes"),
            solver_status=_nonempty_string(values["solver_status"], "solver_status"),
            witnesses=tuple(witnesses),
            degradation_trace=trace,
        )


__all__ = [
    "SCHEMA_VERSION",
    "DatasetRecord",
    "DegradationTrace",
    "EntailmentCertificate",
    "HypothesisScore",
    "ImageRecord",
    "JSONValue",
    "KNOWN_DEGRADATIONS",
    "LesionRegion",
    "SchemaValidationError",
    "WitnessRecord",
]
