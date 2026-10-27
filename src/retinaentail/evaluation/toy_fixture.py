"""Deterministic Stage 4 locked-test fixture for the synthetic toy protocol.

This module never opens a medical dataset root, uses no network or GPU, and makes
no clinical claim.  It accepts thresholds only from a successfully verified Stage
3 toy package and never invokes a calibration routine.  The exact Stage 3 lock and
the ordered Stage 4 test plan are code- and configuration-pinned for this fixture
version so that modifying alpha, thresholds, provenance, case membership, or case
order fails before any test evaluation is accepted.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from retinaentail.calibration.certificate import (
    Stage3SyntheticCertificate,
    Stage3WitnessEvidence,
    WitnessSide,
)
from retinaentail.calibration.protocol import (
    CALIBRATION_METHOD_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    CalibrationLabel,
    EvaluationFailureReason,
    EvaluationResult,
    EvaluationStatus,
    ProtocolScope,
    SampleSplit,
)
from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.degradations import (
    DegradationStep,
    ParameterValue,
    apply_pipeline,
    replay_trace,
    validate_operator_parameters,
)
from retinaentail.entailment.states import EntailmentState, state_from_feasibility
from retinaentail.entailment.toy_baseline import normalized_mse
from retinaentail.evaluation.metrics import (
    LockedStage4Contract,
    Stage4CaseRecord,
    Stage4EvaluationReport,
    Stage4EvaluationScope,
    aggregate_locked_test_evaluations,
)
from retinaentail.interventions.toy import generate_toy_fundus, toy_lesion_predicate
from retinaentail.schemas import KNOWN_DEGRADATIONS, DegradationTrace, LesionRegion, WitnessRecord
from retinaentail.witnesses.replay import load_witness, rgb_png_bytes, sha256_bytes
from retinaentail.witnesses.stage3_replay import (
    PairReplayFailureReason,
    ReplayFailureReason,
    Stage3DualReplayResult,
    Stage3ReplayResult,
    replay_stage3_witness_pair,
)

STAGE4_TOY_CONFIG_VERSION: Final = "4.0"
STAGE4_TOY_PACKAGE_VERSION: Final = "4.0.0"
STAGE4_TOY_PREDICATE_ID: Final = "toy_dark_red_round_lesion_v1"
STAGE4_LOCK_METHOD: Final = "verified_stage3_frozen_artifact_v1"
STAGE4_TEST_METHOD: Final = "locked_toy_pixel_replay_v1"
_MAX_STAGE4_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS: Final = ("source_image_id", "patient_id", "intervention_pair_id")

_STAGE3_PACKAGE_FILES: Final = frozenset(
    {
        "calibration_artifact.json",
        "calibration_samples.json",
        "certificate.json",
        "config_snapshot.json",
        "dual_replay.json",
        "evaluation.json",
        "fail_closed/missing_threshold.json",
        "fail_closed/unknown_degradation.json",
        "images/lesion_absent.png",
        "images/lesion_present.png",
        "images/observation.png",
        "model_provenance.json",
        "summary.json",
        "witnesses/negative.json",
        "witnesses/positive.json",
    }
)


class ObservationMode(StrEnum):
    LESION_PRESENT = "LESION_PRESENT"
    LESION_ABSENT = "LESION_ABSENT"
    MIDPOINT = "MIDPOINT"
    SHIFTED_MIDPOINT = "SHIFTED_MIDPOINT"


class FailureMode(StrEnum):
    NONE = "NONE"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    UNKNOWN_DEGRADATION = "UNKNOWN_DEGRADATION"
    INVALID_WITNESS = "INVALID_WITNESS"
    MISSING_THRESHOLD = "MISSING_THRESHOLD"


_LOCKED_STAGE3_VALUES: Final[dict[str, object]] = {
    "schema_version": "3.0.0",
    "method_version": "toy_conservative_order_statistic_v1",
    "protocol_scope": "TOY_PROTOCOL",
    "artifact_sha256": "c84762f2d44d345cd39732e1ce0f5a44003e0a635e482f28d3f8abcd6395e78e",
    "alpha": 0.25,
    "positive_threshold": 1.2299400515471258e-05,
    "negative_threshold": 1.2299400515471254e-05,
    "positive_sample_count": 4,
    "negative_sample_count": 4,
    "total_sample_count": 8,
    "positive_order_index": 4,
    "negative_order_index": 4,
    "config_hash": "0ea540363d0a768bc1964c007f6a37a393f1b29b3c9a312ed92bd59a76bbab52",
    "data_hash": "ab4fa68cb4228df6a1db23caaeacb8d1c35cca6ec7da220e600a8bdcbc067ecd",
    "model_hash": "ca2fa473841f6b79d8b327525fb14de0e099ac8d342538a5031d9a54783b81c1",
}

_LOCKED_RUN_ID: Final = "stage4-toy-locked-test-001"
_LOCKED_SEED: Final = 20260824
_LOCKED_IMAGE_SIZE: Final = 96
_LOCKED_LESION_CLASS: Final = "MA"
_LOCKED_LESION_RADIUS: Final = 2
_LOCKED_DEGRADATIONS: Final = (
    ("gaussian_blur", 0.55, ()),
    ("downsample_restore", 0.35, ()),
    ("jpeg_compression", 0.25, ()),
)

_LOCKED_TEST_PLAN: Final = (
    (
        "stage4-toy-entailed-present",
        ObservationMode.LESION_PRESENT,
        FailureMode.NONE,
        EntailmentState.ENTAILED_PRESENT,
    ),
    (
        "stage4-toy-entailed-absent",
        ObservationMode.LESION_ABSENT,
        FailureMode.NONE,
        EntailmentState.ENTAILED_ABSENT,
    ),
    (
        "stage4-toy-raw-unidentifiable",
        ObservationMode.MIDPOINT,
        FailureMode.NONE,
        EntailmentState.RAW_UNIDENTIFIABLE,
    ),
    (
        "stage4-toy-infeasible-out-of-model",
        ObservationMode.SHIFTED_MIDPOINT,
        FailureMode.NONE,
        EntailmentState.OUT_OF_MODEL,
    ),
    (
        "stage4-toy-domain-out-of-model",
        ObservationMode.LESION_PRESENT,
        FailureMode.OUT_OF_DISTRIBUTION,
        EntailmentState.OUT_OF_MODEL,
    ),
    (
        "stage4-toy-unknown-degradation",
        ObservationMode.LESION_PRESENT,
        FailureMode.UNKNOWN_DEGRADATION,
        EntailmentState.OUT_OF_MODEL,
    ),
    (
        "stage4-toy-invalid-witness",
        ObservationMode.LESION_PRESENT,
        FailureMode.INVALID_WITNESS,
        EntailmentState.OUT_OF_MODEL,
    ),
)


class Stage4ToyConfigurationError(ValueError):
    """Raised when the frozen synthetic test declaration is not exact."""


@dataclass(frozen=True, slots=True)
class LockedCalibrationSnapshot:
    """Strict immutable view of the already-issued and verified Stage 3 artifact."""

    schema_version: str
    method_version: str
    protocol_scope: str
    artifact_sha256: str
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    total_sample_count: int
    positive_order_index: int
    negative_order_index: int
    config_hash: str
    data_hash: str
    model_hash: str

    def __post_init__(self) -> None:
        if self.to_dict() != _LOCKED_STAGE3_VALUES:
            raise Stage4ToyConfigurationError(
                "Stage 3 alpha/threshold/count/provenance lock does not match fixture version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "protocol_scope": self.protocol_scope,
            "artifact_sha256": self.artifact_sha256,
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
        }

    def artifact_dict(self) -> dict[str, object]:
        """Return the exact canonical Stage 3 artifact representation."""

        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "protocol_scope": self.protocol_scope,
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
            "status": "SUCCESS",
            "failure_reasons": [],
        }


@dataclass(frozen=True, slots=True)
class Stage4ToyDegradation:
    name: str
    severity: float
    parameters: dict[str, ParameterValue]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "severity": self.severity}
        if self.parameters:
            result["parameters"] = dict(self.parameters)
        return result


@dataclass(frozen=True, slots=True)
class Stage4ToyCaseSpec:
    case_id: str
    observation_mode: ObservationMode
    failure_mode: FailureMode
    expected_state: EntailmentState

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "observation_mode": self.observation_mode.value,
            "failure_mode": self.failure_mode.value,
            "expected_state": self.expected_state.value,
        }


@dataclass(frozen=True, slots=True)
class Stage4ToyConfig:
    schema_version: str
    run_id: str
    seed: int
    image_size: int
    lesion_class: str
    lesion_radius: int
    stage3_lock: LockedCalibrationSnapshot
    degradations: tuple[Stage4ToyDegradation, ...]
    test_cases: tuple[Stage4ToyCaseSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seed": self.seed,
            "image_size": self.image_size,
            "lesion_class": self.lesion_class,
            "lesion_radius": self.lesion_radius,
            "stage3_lock": self.stage3_lock.to_dict(),
            "degradations": [item.to_dict() for item in self.degradations],
            "test_cases": [item.to_dict() for item in self.test_cases],
        }


@dataclass(frozen=True, slots=True)
class Stage4ToyRunResult:
    output_dir: Path
    case_count: int
    states: tuple[EntailmentState, ...]
    locked_calibration_artifact_hash: str
    test_data_hash: str
    verified: bool


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    spec: Stage4ToyCaseSpec
    region: LesionRegion
    observation_sha256: str
    trace_sha256: str
    positive_witness: WitnessRecord
    negative_witness: WitnessRecord
    replay: Stage3DualReplayResult
    evaluation: EvaluationResult
    certificate: Stage3SyntheticCertificate
    source_image_id: str
    patient_id: str
    intervention_pair_id: str
    artifacts: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    config: Stage4ToyConfig
    config_hash: str
    test_plan_hash: str
    test_data_hash: str
    model_provenance_hash: str
    cases: tuple[_PreparedCase, ...]
    artifacts: dict[str, bytes]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_mapping(value: object, *, name: str, required: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Stage4ToyConfigurationError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise Stage4ToyConfigurationError(f"{name} keys must be strings")
        result[key] = item
    missing = required.difference(result)
    unknown = set(result).difference(required)
    if missing or unknown:
        raise Stage4ToyConfigurationError(
            f"{name} missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage4ToyConfigurationError(f"{name} must be a non-empty string")
    result = value.strip()
    if result != value or any(character in result for character in "\r\n\x00"):
        raise Stage4ToyConfigurationError(f"{name} is not canonical")
    return result


def _plain_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise Stage4ToyConfigurationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite_float(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage4ToyConfigurationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise Stage4ToyConfigurationError(f"{name} is outside its finite range")
    return result


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Stage4ToyConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parameters(value: object, name: str) -> dict[str, ParameterValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Stage4ToyConfigurationError(f"{name} must be a mapping")
    result: dict[str, ParameterValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise Stage4ToyConfigurationError(f"{name} keys must be non-empty strings")
        if isinstance(item, str | bool | int):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise Stage4ToyConfigurationError(f"{name}.{key} must be a finite JSON scalar")
    return result


def _parse_stage3_lock(value: object) -> LockedCalibrationSnapshot:
    data = _strict_mapping(
        value,
        name="stage3_lock",
        required=set(_LOCKED_STAGE3_VALUES),
    )
    snapshot = LockedCalibrationSnapshot(
        schema_version=_nonempty_string(data["schema_version"], "stage3_lock.schema_version"),
        method_version=_nonempty_string(data["method_version"], "stage3_lock.method_version"),
        protocol_scope=_nonempty_string(data["protocol_scope"], "stage3_lock.protocol_scope"),
        artifact_sha256=_require_sha256(data["artifact_sha256"], "stage3_lock.artifact_sha256"),
        alpha=_finite_float(data["alpha"], "stage3_lock.alpha", minimum=0.0, maximum=1.0),
        positive_threshold=_finite_float(
            data["positive_threshold"],
            "stage3_lock.positive_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        negative_threshold=_finite_float(
            data["negative_threshold"],
            "stage3_lock.negative_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        positive_sample_count=_plain_int(
            data["positive_sample_count"],
            "stage3_lock.positive_sample_count",
            minimum=1,
            maximum=1_000_000,
        ),
        negative_sample_count=_plain_int(
            data["negative_sample_count"],
            "stage3_lock.negative_sample_count",
            minimum=1,
            maximum=1_000_000,
        ),
        total_sample_count=_plain_int(
            data["total_sample_count"],
            "stage3_lock.total_sample_count",
            minimum=2,
            maximum=2_000_000,
        ),
        positive_order_index=_plain_int(
            data["positive_order_index"],
            "stage3_lock.positive_order_index",
            minimum=1,
            maximum=1_000_000,
        ),
        negative_order_index=_plain_int(
            data["negative_order_index"],
            "stage3_lock.negative_order_index",
            minimum=1,
            maximum=1_000_000,
        ),
        config_hash=_require_sha256(data["config_hash"], "stage3_lock.config_hash"),
        data_hash=_require_sha256(data["data_hash"], "stage3_lock.data_hash"),
        model_hash=_require_sha256(data["model_hash"], "stage3_lock.model_hash"),
    )
    if not 0.0 < snapshot.alpha < 1.0:
        raise Stage4ToyConfigurationError("stage3_lock.alpha must lie between zero and one")
    if snapshot.total_sample_count != (
        snapshot.positive_sample_count + snapshot.negative_sample_count
    ):
        raise Stage4ToyConfigurationError("Stage 3 calibration sample counts are inconsistent")
    if snapshot.positive_order_index > snapshot.positive_sample_count or (
        snapshot.negative_order_index > snapshot.negative_sample_count
    ):
        raise Stage4ToyConfigurationError("Stage 3 order indices exceed sample counts")
    return snapshot


def parse_stage4_toy_config(value: object) -> Stage4ToyConfig:
    """Parse and pin an exact Stage 4 synthetic locked-test declaration."""

    data = _strict_mapping(
        value,
        name="Stage 4 toy config",
        required={
            "schema_version",
            "run_id",
            "seed",
            "image_size",
            "lesion_class",
            "lesion_radius",
            "stage3_lock",
            "degradations",
            "test_cases",
        },
    )
    if data["schema_version"] != STAGE4_TOY_CONFIG_VERSION:
        raise Stage4ToyConfigurationError("unsupported Stage 4 toy config schema_version")

    raw_degradations = data["degradations"]
    if isinstance(raw_degradations, str | bytes) or not isinstance(raw_degradations, Sequence):
        raise Stage4ToyConfigurationError("degradations must be a list")
    degradations: list[Stage4ToyDegradation] = []
    for index, raw in enumerate(raw_degradations):
        required = (
            {"name", "severity", "parameters"}
            if isinstance(raw, Mapping) and "parameters" in raw
            else {"name", "severity"}
        )
        operation = _strict_mapping(
            raw,
            name=f"degradations[{index}]",
            required=required,
        )
        name = _nonempty_string(operation["name"], f"degradations[{index}].name")
        if name not in KNOWN_DEGRADATIONS:
            raise Stage4ToyConfigurationError(f"unknown degradation operator: {name}")
        severity = _finite_float(
            operation["severity"],
            f"degradations[{index}].severity",
            minimum=0.0,
            maximum=1.0,
        )
        parameters = _parameters(operation.get("parameters"), f"degradations[{index}].parameters")
        try:
            validate_operator_parameters(name, parameters)
        except (TypeError, ValueError) as error:
            raise Stage4ToyConfigurationError(
                f"invalid degradations[{index}] parameters: {error}"
            ) from error
        degradations.append(Stage4ToyDegradation(name, severity, parameters))
    if not degradations:
        raise Stage4ToyConfigurationError("at least one known degradation is required")
    actual_degradations = tuple(
        (item.name, item.severity, tuple(sorted(item.parameters.items()))) for item in degradations
    )
    if actual_degradations != _LOCKED_DEGRADATIONS:
        raise Stage4ToyConfigurationError(
            "degradation membership, order, severities, and parameters are locked for version 4.0"
        )

    raw_cases = data["test_cases"]
    if isinstance(raw_cases, str | bytes) or not isinstance(raw_cases, Sequence):
        raise Stage4ToyConfigurationError("test_cases must be a list")
    cases: list[Stage4ToyCaseSpec] = []
    for index, raw in enumerate(raw_cases):
        case = _strict_mapping(
            raw,
            name=f"test_cases[{index}]",
            required={"case_id", "observation_mode", "failure_mode", "expected_state"},
        )
        try:
            observation_mode = ObservationMode(str(case["observation_mode"]))
            failure_mode = FailureMode(str(case["failure_mode"]))
            expected_state = EntailmentState(str(case["expected_state"]))
        except (TypeError, ValueError) as error:
            raise Stage4ToyConfigurationError(
                f"test_cases[{index}] contains an unsupported enum value"
            ) from error
        cases.append(
            Stage4ToyCaseSpec(
                case_id=_nonempty_string(case["case_id"], f"test_cases[{index}].case_id"),
                observation_mode=observation_mode,
                failure_mode=failure_mode,
                expected_state=expected_state,
            )
        )
    actual_plan = tuple(
        (item.case_id, item.observation_mode, item.failure_mode, item.expected_state)
        for item in cases
    )
    if actual_plan != _LOCKED_TEST_PLAN:
        raise Stage4ToyConfigurationError(
            "test case membership, order, modes, and expected states are locked for version 4.0"
        )

    config = Stage4ToyConfig(
        schema_version=STAGE4_TOY_CONFIG_VERSION,
        run_id=_nonempty_string(data["run_id"], "run_id"),
        seed=_plain_int(data["seed"], "seed", minimum=0, maximum=2_000_000_000),
        image_size=_plain_int(data["image_size"], "image_size", minimum=48, maximum=512),
        lesion_class=_nonempty_string(data["lesion_class"], "lesion_class"),
        lesion_radius=_plain_int(data["lesion_radius"], "lesion_radius", minimum=2, maximum=64),
        stage3_lock=_parse_stage3_lock(data["stage3_lock"]),
        degradations=tuple(degradations),
        test_cases=tuple(cases),
    )
    if config.lesion_radius * 6 >= config.image_size:
        raise Stage4ToyConfigurationError("lesion_radius is too large for image_size")
    run_values = (
        config.run_id,
        config.seed,
        config.image_size,
        config.lesion_class,
        config.lesion_radius,
    )
    if run_values != (
        _LOCKED_RUN_ID,
        _LOCKED_SEED,
        _LOCKED_IMAGE_SIZE,
        _LOCKED_LESION_CLASS,
        _LOCKED_LESION_RADIUS,
    ):
        raise Stage4ToyConfigurationError(
            "run identity, seed, image geometry, and lesion declaration are version-locked"
        )
    return config


def load_stage4_toy_config(path: Path) -> Stage4ToyConfig:
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_stage4_toy_config(decoded)


def _decode_json(payload: bytes, name: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error


def _verified_locked_snapshot(
    stage3_package: Path,
    expected: LockedCalibrationSnapshot,
) -> tuple[LockedCalibrationSnapshot, bytes, bytes]:
    """Verify Stage 3 first, then bind its canonical artifact without recalibration here."""

    # Deliberately lazy: calibration.toy_fixture imports the reporting package, whose
    # Stage 4 exports import this module.  Keeping the verifier out of module import
    # time prevents that package-level compatibility surface from becoming a cycle.
    from retinaentail.calibration.toy_fixture import verify_stage3_toy_package

    verified = verify_stage3_toy_package(stage3_package)
    if not verified.verified:
        raise ValueError("Stage 3 source package did not verify")
    artifact_bytes = _read_bounded_artifact(stage3_package / "calibration_artifact.json")
    summary_bytes = _read_bounded_artifact(stage3_package / "summary.json")
    artifact_value = _decode_json(artifact_bytes, "Stage 3 calibration artifact")
    summary_value = _decode_json(summary_bytes, "Stage 3 summary")
    if not isinstance(artifact_value, Mapping) or not isinstance(summary_value, Mapping):
        raise ValueError("Stage 3 lock inputs must be JSON objects")
    artifact = dict(artifact_value)
    summary = dict(summary_value)
    if artifact != expected.artifact_dict():
        raise ValueError("verified Stage 3 calibration artifact does not match the frozen lock")
    artifact_hash = sha256_bytes(_canonical_json_bytes(artifact))
    if artifact_hash != expected.artifact_sha256 or (
        verified.calibration_artifact_hash != expected.artifact_sha256
    ):
        raise ValueError("verified Stage 3 calibration artifact hash does not match the lock")
    summary_bindings = {
        "alpha": expected.alpha,
        "positive_threshold": expected.positive_threshold,
        "negative_threshold": expected.negative_threshold,
        "positive_sample_count": expected.positive_sample_count,
        "negative_sample_count": expected.negative_sample_count,
        "config_hash": expected.config_hash,
        "calibration_data_hash": expected.data_hash,
        "model_hash": expected.model_hash,
        "calibration_artifact_hash": expected.artifact_sha256,
    }
    if any(summary.get(name) != value for name, value in summary_bindings.items()):
        raise ValueError("verified Stage 3 summary does not match the frozen lock")
    return expected, artifact_bytes, summary_bytes


def _operation_steps(
    config: Stage4ToyConfig,
    *,
    trace_seed: int,
) -> tuple[DegradationStep, ...]:
    return tuple(
        DegradationStep(
            name=item.name,
            severity=item.severity,
            seed=trace_seed + index,
            parameters=item.parameters,
        )
        for index, item in enumerate(config.degradations)
    )


def _schema_trace(value: object) -> DegradationTrace:
    if not isinstance(value, Mapping):
        raise TypeError("degradation trace must be a mapping")
    return DegradationTrace.from_dict(cast(Mapping[str, object], value))


def _observation(
    mode: ObservationMode,
    positive: NDArray[np.uint8],
    negative: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if mode is ObservationMode.LESION_PRESENT:
        return positive.copy()
    if mode is ObservationMode.LESION_ABSENT:
        return negative.copy()
    midpoint = ((positive.astype(np.uint16) + negative.astype(np.uint16)) // 2).astype(np.uint8)
    if mode is ObservationMode.MIDPOINT:
        return midpoint
    if mode is ObservationMode.SHIFTED_MIDPOINT:
        return np.clip(midpoint.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported observation mode: {mode}")


def _predicted_replay(
    *,
    case_id: str,
    lesion_class: str,
    region: LesionRegion,
    observation_sha256: str,
    trace_sha256: str,
    positive_witness: WitnessRecord,
    negative_witness: WitnessRecord,
    invalidate_positive: bool,
) -> Stage3DualReplayResult:
    region_sha256 = sha256_bytes(_canonical_json_bytes(region.to_dict()))

    def one(witness: WitnessRecord, side: str, invalid: bool) -> Stage3ReplayResult:
        reasons = (ReplayFailureReason.RECORDED_PREDICATE_INVALID,) if invalid else ()
        return Stage3ReplayResult(
            witness_id=witness.witness_id,
            case_id=case_id,
            lesion_class=lesion_class,
            side=side,
            predicate_id=STAGE4_TOY_PREDICATE_ID,
            region_sha256=region_sha256,
            expected_latent_sha256=witness.sha256,
            actual_latent_sha256=witness.sha256,
            expected_observation_sha256=observation_sha256,
            actual_observation_sha256=observation_sha256,
            expected_trace_sha256=trace_sha256,
            actual_trace_sha256=trace_sha256,
            replay_error=witness.replay_error,
            predicate_valid=True,
            passed=not reasons,
            failure_reasons=reasons,
        )

    positive = one(positive_witness, "positive", invalidate_positive)
    negative = one(negative_witness, "negative", False)
    pair_reasons = (PairReplayFailureReason.POSITIVE_REPLAY_FAILED,) if invalidate_positive else ()
    return Stage3DualReplayResult(
        positive=positive,
        negative=negative,
        passed=not pair_reasons,
        failure_reasons=pair_reasons,
    )


def evaluate_locked_replay(
    *,
    case_id: str,
    lesion_class: str,
    lock: LockedCalibrationSnapshot,
    replay: Stage3DualReplayResult,
    failure_mode: FailureMode,
) -> EvaluationResult:
    """Apply frozen thresholds exactly once; this function has no calibration path."""

    if not isinstance(lock, LockedCalibrationSnapshot):
        raise TypeError("lock must be a LockedCalibrationSnapshot")
    if not isinstance(replay, Stage3DualReplayResult):
        raise TypeError("replay must be a Stage3DualReplayResult")
    if not isinstance(failure_mode, FailureMode):
        raise TypeError("failure_mode must be a FailureMode")
    _nonempty_string(case_id, "case_id")
    _nonempty_string(lesion_class, "lesion_class")
    lock.__post_init__()
    replay.__post_init__()
    positive_score = replay.positive.replay_error
    negative_score = replay.negative.replay_error
    positive_threshold: float | None = lock.positive_threshold
    negative_threshold: float | None = lock.negative_threshold
    degradation_known = failure_mode is not FailureMode.UNKNOWN_DEGRADATION
    out_of_distribution = failure_mode is FailureMode.OUT_OF_DISTRIBUTION
    if failure_mode is FailureMode.MISSING_THRESHOLD:
        positive_threshold = None

    reasons: list[EvaluationFailureReason] = []
    if positive_threshold is None or negative_threshold is None:
        reasons.append(EvaluationFailureReason.MISSING_THRESHOLD)
    scores = (positive_score, negative_score)
    if any(value is None for value in scores):
        reasons.append(EvaluationFailureReason.MISSING_SCORE)
    elif any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in scores
    ):
        reasons.append(EvaluationFailureReason.NONFINITE_SCORE)
    if not degradation_known:
        reasons.append(EvaluationFailureReason.UNKNOWN_DEGRADATION)
    if not replay.positive.passed or not replay.negative.passed:
        reasons.append(EvaluationFailureReason.INVALID_WITNESS)
    if out_of_distribution:
        reasons.append(EvaluationFailureReason.OUT_OF_DISTRIBUTION)

    if reasons:
        state = EntailmentState.OUT_OF_MODEL
        status = EvaluationStatus.FAIL_CLOSED
        positive_feasible: bool | None = None
        negative_feasible: bool | None = None
    else:
        assert positive_score is not None
        assert negative_score is not None
        assert positive_threshold is not None
        assert negative_threshold is not None
        positive_feasible = float(positive_score) <= positive_threshold
        negative_feasible = float(negative_score) <= negative_threshold
        state = state_from_feasibility(positive_feasible, negative_feasible)
        status = EvaluationStatus.SUCCESS

    return EvaluationResult(
        case_id=case_id,
        lesion_class=lesion_class,
        alpha=lock.alpha,
        positive_score=(float(positive_score) if positive_score is not None else None),
        negative_score=(float(negative_score) if negative_score is not None else None),
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        positive_feasible=positive_feasible,
        negative_feasible=negative_feasible,
        positive_sample_count=lock.positive_sample_count,
        negative_sample_count=lock.negative_sample_count,
        config_hash=lock.config_hash,
        data_hash=lock.data_hash,
        model_hash=lock.model_hash,
        input_config_hash=lock.config_hash,
        input_data_hash=lock.data_hash,
        input_model_hash=lock.model_hash,
        positive_solver_status="toy_baseline",
        negative_solver_status="toy_baseline",
        solver_status="toy_baseline",
        degradation_known=degradation_known,
        positive_witness_replay_valid=replay.positive.passed,
        negative_witness_replay_valid=replay.negative.passed,
        out_of_distribution=out_of_distribution,
        state=state,
        status=status,
        failure_reasons=tuple(reasons),
        schema_version=CALIBRATION_SCHEMA_VERSION,
        method_version=CALIBRATION_METHOD_VERSION,
        protocol_scope=ProtocolScope.TOY_PROTOCOL,
    )


def _case_completion(
    *,
    spec: Stage4ToyCaseSpec,
    region: LesionRegion,
    observation_sha256: str,
    trace_sha256: str,
    positive_witness: WitnessRecord,
    negative_witness: WitnessRecord,
    replay: Stage3DualReplayResult,
    lock: LockedCalibrationSnapshot,
) -> tuple[EvaluationResult, Stage3SyntheticCertificate]:
    evaluation = evaluate_locked_replay(
        case_id=spec.case_id,
        lesion_class=positive_witness.lesion_class,
        lock=lock,
        replay=replay,
        failure_mode=spec.failure_mode,
    )
    if evaluation.state is not spec.expected_state:
        raise RuntimeError(
            f"locked test case {spec.case_id} produced {evaluation.state.value}, "
            f"expected {spec.expected_state.value}"
        )
    certificate = Stage3SyntheticCertificate(
        evaluation=evaluation,
        region=region,
        predicate_id=STAGE4_TOY_PREDICATE_ID,
        calibration_artifact_hash=lock.artifact_sha256,
        observation_sha256=observation_sha256,
        degradation_trace_sha256=trace_sha256,
        positive_evidence=Stage3WitnessEvidence(
            side=WitnessSide.POSITIVE,
            witness=positive_witness,
            replay=replay.positive,
        ),
        negative_evidence=Stage3WitnessEvidence(
            side=WitnessSide.NEGATIVE,
            witness=negative_witness,
            replay=replay.negative,
        ),
    )
    return evaluation, certificate


def _prepare_case(
    config: Stage4ToyConfig,
    spec: Stage4ToyCaseSpec,
    index: int,
) -> _PreparedCase:
    image_seed = config.seed + index
    trace_seed = config.seed + 100_000 + index * 100
    twin = generate_toy_fundus(
        size=config.image_size,
        lesion_radius=config.lesion_radius,
        seed=image_seed,
    )
    positive_replay, operation_trace = apply_pipeline(
        twin.lesion_present,
        _operation_steps(config, trace_seed=trace_seed),
    )
    negative_replay = replay_trace(twin.lesion_absent, operation_trace)
    observation = _observation(spec.observation_mode, positive_replay, negative_replay)
    if not toy_lesion_predicate(twin.lesion_present, twin.region) or toy_lesion_predicate(
        twin.lesion_absent, twin.region
    ):
        raise RuntimeError("procedural Stage 4 twins violate the declared toy predicate")

    trace = _schema_trace(operation_trace.to_dict())
    trace_sha256 = sha256_bytes(_canonical_json_bytes(trace.to_dict()))
    observation_bytes = rgb_png_bytes(observation)
    positive_bytes = rgb_png_bytes(twin.lesion_present)
    negative_bytes = rgb_png_bytes(twin.lesion_absent)
    observation_sha256 = sha256_bytes(observation_bytes)
    invalidate_positive = spec.failure_mode is FailureMode.INVALID_WITNESS
    positive_witness = WitnessRecord(
        witness_id=f"{spec.case_id}:positive",
        case_id=spec.case_id,
        lesion_class=config.lesion_class,
        witness_kind="lesion_present",
        relative_path="images/lesion_present.png",
        observation_path="images/observation.png",
        sha256=sha256_bytes(positive_bytes),
        degradation_trace=trace,
        replay_seed=trace_seed,
        replay_error=normalized_mse(positive_replay, observation),
        predicate_valid=not invalidate_positive,
    )
    negative_witness = WitnessRecord(
        witness_id=f"{spec.case_id}:negative",
        case_id=spec.case_id,
        lesion_class=config.lesion_class,
        witness_kind="lesion_absent",
        relative_path="images/lesion_absent.png",
        observation_path="images/observation.png",
        sha256=sha256_bytes(negative_bytes),
        degradation_trace=trace,
        replay_seed=trace_seed,
        replay_error=normalized_mse(negative_replay, observation),
        predicate_valid=True,
    )
    replay = _predicted_replay(
        case_id=spec.case_id,
        lesion_class=config.lesion_class,
        region=twin.region,
        observation_sha256=observation_sha256,
        trace_sha256=trace_sha256,
        positive_witness=positive_witness,
        negative_witness=negative_witness,
        invalidate_positive=invalidate_positive,
    )
    evaluation, certificate = _case_completion(
        spec=spec,
        region=twin.region,
        observation_sha256=observation_sha256,
        trace_sha256=trace_sha256,
        positive_witness=positive_witness,
        negative_witness=negative_witness,
        replay=replay,
        lock=config.stage3_lock,
    )
    source_image_id = f"stage4-toy-source-{index:03d}"
    patient_id = f"stage4-toy-patient-{index:03d}"
    intervention_pair_id = f"stage4-toy-pair-{index:03d}"
    case_payload: dict[str, object] = {
        "schema_version": STAGE4_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_TOY",
        "split": "test",
        "preregistered_index": index,
        "case_spec": spec.to_dict(),
        "image_seed": image_seed,
        "trace_seed": trace_seed,
        "source_image_id": source_image_id,
        "patient_id": patient_id,
        "intervention_pair_id": intervention_pair_id,
        "observation_sha256": observation_sha256,
        "degradation_trace_sha256": trace_sha256,
        "positive_latent_sha256": positive_witness.sha256,
        "negative_latent_sha256": negative_witness.sha256,
        "state": evaluation.state.value,
        "evaluation_status": evaluation.status.value,
        "failure_reasons": [reason.value for reason in evaluation.failure_reasons],
        "dual_witness_replay_passed": replay.passed,
        "synthetic": True,
        "clinical_claim": False,
    }
    artifacts = {
        "images/observation.png": observation_bytes,
        "images/lesion_present.png": positive_bytes,
        "images/lesion_absent.png": negative_bytes,
        "witnesses/positive.json": _canonical_json_bytes(positive_witness.to_dict()),
        "witnesses/negative.json": _canonical_json_bytes(negative_witness.to_dict()),
        "dual_replay.json": _canonical_json_bytes(replay.to_dict()),
        "evaluation.json": _canonical_json_bytes(evaluation.to_dict()),
        "certificate.json": _canonical_json_bytes(certificate.to_dict()),
        "case.json": _canonical_json_bytes(case_payload),
    }
    return _PreparedCase(
        spec=spec,
        region=twin.region,
        observation_sha256=observation_sha256,
        trace_sha256=trace_sha256,
        positive_witness=positive_witness,
        negative_witness=negative_witness,
        replay=replay,
        evaluation=evaluation,
        certificate=certificate,
        source_image_id=source_image_id,
        patient_id=patient_id,
        intervention_pair_id=intervention_pair_id,
        artifacts=artifacts,
    )


def _model_provenance() -> tuple[str, dict[str, object]]:
    package_root = Path(__file__).resolve().parents[1]
    components = {
        "calibration_certificate": package_root / "calibration" / "certificate.py",
        "calibration_protocol": package_root / "calibration" / "protocol.py",
        "entailment_states": package_root / "entailment" / "states.py",
        "stage3_replay": package_root / "witnesses" / "stage3_replay.py",
        "stage4_fixture": package_root / "evaluation" / "toy_fixture.py",
        "stage4_metrics": package_root / "evaluation" / "metrics.py",
        "stage4_reporting": package_root / "reporting" / "stage4.py",
        "toy_baseline": package_root / "entailment" / "toy_baseline.py",
        "toy_twins": package_root / "interventions" / "toy.py",
    }
    payload: dict[str, object] = {
        "schema_version": STAGE4_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_TOY",
        "method": STAGE4_TEST_METHOD,
        "components": [
            {
                "component": name,
                "sha256": sha256_bytes(
                    path.read_text(encoding="utf-8")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .encode("utf-8")
                ),
            }
            for name, path in sorted(components.items())
        ],
    }
    return sha256_bytes(_canonical_json_bytes(payload)), payload


def _test_data_payload(cases: tuple[_PreparedCase, ...]) -> dict[str, object]:
    return {
        "schema_version": STAGE4_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "label_source": "procedural_twin_control",
        "missing_label_policy": "UNKNOWN_NEVER_NEGATIVE",
        "test_tuning_prohibited": True,
        "calibration_test_identity_overlap": False,
        "records": [
            {
                "preregistered_index": index,
                "case_id": case.spec.case_id,
                "source_image_id": case.source_image_id,
                "patient_id": case.patient_id,
                "intervention_pair_id": case.intervention_pair_id,
                "observation_sha256": case.observation_sha256,
                "degradation_trace_sha256": case.trace_sha256,
                "observation_mode": case.spec.observation_mode.value,
                "failure_mode": case.spec.failure_mode.value,
                "expected_state": case.spec.expected_state.value,
            }
            for index, case in enumerate(cases)
        ],
    }


def _stage3_calibration_identities(stage3_root: Path) -> dict[str, set[str]]:
    value = _decode_json(
        _read_bounded_artifact(stage3_root / "calibration_samples.json"),
        "Stage 3 calibration samples",
    )
    if not isinstance(value, Mapping):
        raise ValueError("Stage 3 calibration samples must be a JSON object")
    records = value.get("records")
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("Stage 3 calibration samples records must be a list")
    identities: dict[str, set[str]] = {field_name: set() for field_name in _IDENTITY_FIELDS}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("sample"), Mapping):
            raise ValueError(f"Stage 3 calibration record {index} is malformed")
        sample = cast(Mapping[str, object], record["sample"])
        for field_name in _IDENTITY_FIELDS:
            identifier = sample.get(field_name)
            if identifier is None:
                continue
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"Stage 3 calibration {field_name} is malformed")
            identities[field_name].add(identifier)
    return identities


def _identity_isolation_payload(
    stage3_root: Path,
    cases: tuple[_PreparedCase, ...],
) -> dict[str, object]:
    calibration = _stage3_calibration_identities(stage3_root)
    test = {
        "source_image_id": {case.source_image_id for case in cases},
        "patient_id": {case.patient_id for case in cases},
        "intervention_pair_id": {case.intervention_pair_id for case in cases},
    }
    overlaps = {
        field_name: sorted(calibration[field_name].intersection(test[field_name]))
        for field_name in _IDENTITY_FIELDS
    }
    if any(overlaps.values()):
        raise ValueError("Stage 3 calibration identities overlap the locked Stage 4 test set")
    return {
        "schema_version": STAGE4_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "calibration_split": "calibration",
        "evaluation_split": "test",
        "identity_fields": list(_IDENTITY_FIELDS),
        "calibration_unique_identity_counts": {
            field_name: len(calibration[field_name]) for field_name in _IDENTITY_FIELDS
        },
        "test_unique_identity_counts": {
            field_name: len(test[field_name]) for field_name in _IDENTITY_FIELDS
        },
        "overlaps": overlaps,
        "passed": True,
    }


def _locked_metrics_contract(lock: LockedCalibrationSnapshot) -> LockedStage4Contract:
    return LockedStage4Contract(
        alpha=lock.alpha,
        positive_threshold=lock.positive_threshold,
        negative_threshold=lock.negative_threshold,
        positive_sample_count=lock.positive_sample_count,
        negative_sample_count=lock.negative_sample_count,
        total_sample_count=lock.total_sample_count,
        calibration_artifact_hash=lock.artifact_sha256,
        config_hash=lock.config_hash,
        data_hash=lock.data_hash,
        model_hash=lock.model_hash,
    )


def _reference_label(case: _PreparedCase) -> CalibrationLabel:
    if case.spec.observation_mode is ObservationMode.LESION_PRESENT:
        return CalibrationLabel.LESION_PRESENT
    if case.spec.observation_mode is ObservationMode.LESION_ABSENT:
        return CalibrationLabel.LESION_ABSENT
    return CalibrationLabel.UNKNOWN


def _aggregate_cases(
    cases: tuple[_PreparedCase, ...],
    lock: LockedCalibrationSnapshot,
) -> Stage4EvaluationReport:
    records = tuple(
        Stage4CaseRecord(
            case_id=case.spec.case_id,
            split=SampleSplit.TEST,
            reference_label=_reference_label(case),
            evaluation=case.evaluation,
            certificate=case.certificate,
            dual_replay=case.replay,
        )
        for case in cases
    )
    return aggregate_locked_test_evaluations(
        records,
        locked_contract=_locked_metrics_contract(lock),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
        confidence_level=0.95,
    )


def _render_report_payloads(report: Stage4EvaluationReport) -> tuple[bytes, bytes]:
    # See the lazy-import rationale in _verified_locked_snapshot.
    from retinaentail.reporting.stage4 import (
        render_stage4_evaluation_html,
        render_stage4_evaluation_json,
    )

    return (
        render_stage4_evaluation_json(report).encode("utf-8"),
        render_stage4_evaluation_html(report).encode("utf-8"),
    )


def _summary_payload(
    *,
    prepared: _PreparedRun,
    lock: LockedCalibrationSnapshot,
) -> dict[str, object]:
    state_counts = {state.value: 0 for state in EntailmentState}
    status_counts = {status.value: 0 for status in EvaluationStatus}
    failure_counts: dict[str, int] = {}
    valid_dual_replays = 0
    determinate_dual_replays = 0
    for case in prepared.cases:
        state_counts[case.evaluation.state.value] += 1
        status_counts[case.evaluation.status.value] += 1
        if case.replay.passed:
            valid_dual_replays += 1
        if case.evaluation.status is EvaluationStatus.SUCCESS and case.replay.passed:
            determinate_dual_replays += 1
        for reason in case.evaluation.failure_reasons:
            failure_counts[reason.value] = failure_counts.get(reason.value, 0) + 1
    return {
        "schema_version": STAGE4_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_TOY",
        "run_id": prepared.config.run_id,
        "method": STAGE4_TEST_METHOD,
        "lock_method": STAGE4_LOCK_METHOD,
        "case_count": len(prepared.cases),
        "state_counts": state_counts,
        "evaluation_status_counts": status_counts,
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "valid_dual_witness_replay_count": valid_dual_replays,
        "successful_truth_table_case_count": determinate_dual_replays,
        "alpha": lock.alpha,
        "positive_threshold": lock.positive_threshold,
        "negative_threshold": lock.negative_threshold,
        "positive_calibration_sample_count": lock.positive_sample_count,
        "negative_calibration_sample_count": lock.negative_sample_count,
        "calibration_artifact_sha256": lock.artifact_sha256,
        "calibration_config_sha256": lock.config_hash,
        "calibration_data_sha256": lock.data_hash,
        "calibration_model_sha256": lock.model_hash,
        "stage4_config_sha256": prepared.config_hash,
        "test_plan_sha256": prepared.test_plan_hash,
        "test_data_sha256": prepared.test_data_hash,
        "stage4_model_provenance_sha256": prepared.model_provenance_hash,
        "thresholds_locked_before_test": True,
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "cpu_only": True,
        "gpu_used": False,
        "network_used": False,
        "real_data_used": False,
        "real_data_read_count": 0,
        "real_data_gate": "BLOCKED_NO_EXPLICIT_APPROVED_DATASET",
        "individual_certificate_authoritative": False,
        "authority_boundary": "COMPLETE_VERIFIED_STAGE4_PACKAGE",
        "clinical_claim": False,
    }


def _prepare_run(config: Stage4ToyConfig) -> _PreparedRun:
    config_hash = sha256_bytes(_canonical_json_bytes(config.to_dict()))
    test_plan_hash = sha256_bytes(
        _canonical_json_bytes([case.to_dict() for case in config.test_cases])
    )
    cases = tuple(
        _prepare_case(config, spec, index) for index, spec in enumerate(config.test_cases)
    )
    test_data = _test_data_payload(cases)
    test_data_hash = sha256_bytes(_canonical_json_bytes(test_data))
    model_provenance_hash, model_provenance = _model_provenance()
    base_artifacts: dict[str, bytes] = {
        "config_snapshot.json": _canonical_json_bytes(config.to_dict()),
        "locked_calibration.json": _canonical_json_bytes(config.stage3_lock.to_dict()),
        "test_data.json": _canonical_json_bytes(test_data),
        "model_provenance.json": _canonical_json_bytes(model_provenance),
    }
    for case in cases:
        for relative_name, payload in case.artifacts.items():
            base_artifacts[f"cases/{case.spec.case_id}/{relative_name}"] = payload
    prepared = _PreparedRun(
        config=config,
        config_hash=config_hash,
        test_plan_hash=test_plan_hash,
        test_data_hash=test_data_hash,
        model_provenance_hash=model_provenance_hash,
        cases=cases,
        artifacts=base_artifacts,
    )
    prepared.artifacts["summary.json"] = _canonical_json_bytes(
        _summary_payload(prepared=prepared, lock=config.stage3_lock)
    )
    report = _aggregate_cases(prepared.cases, config.stage3_lock)
    audit_json, audit_html = _render_report_payloads(report)
    prepared.artifacts["audit.json"] = audit_json
    prepared.artifacts["audit.html"] = audit_html
    return prepared


def _read_bounded_artifact(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("Stage 4 package artifacts cannot be symbolic links")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError("Stage 4 package artifact cannot be inspected") from error
    if size > _MAX_STAGE4_ARTIFACT_BYTES:
        raise ValueError(f"Stage 4 package artifact exceeds size limit: {path.name}")
    try:
        with path.open("rb") as stream:
            payload = stream.read(_MAX_STAGE4_ARTIFACT_BYTES + 1)
    except OSError as error:
        raise ValueError("Stage 4 package artifact cannot be read") from error
    if len(payload) > _MAX_STAGE4_ARTIFACT_BYTES:
        raise ValueError(f"Stage 4 package artifact exceeds size limit: {path.name}")
    return payload


def _case_file_names(case_id: str) -> set[str]:
    prefix = f"cases/{case_id}/"
    return {
        prefix + "case.json",
        prefix + "certificate.json",
        prefix + "dual_replay.json",
        prefix + "evaluation.json",
        prefix + "images/lesion_absent.png",
        prefix + "images/lesion_present.png",
        prefix + "images/observation.png",
        prefix + "witnesses/negative.json",
        prefix + "witnesses/positive.json",
    }


def _expected_package_files() -> frozenset[str]:
    names = {
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "identity_isolation.json",
        "locked_calibration.json",
        "locked_calibration_artifact.json",
        "locked_calibration_summary.json",
        "model_provenance.json",
        "summary.json",
        "test_data.json",
    }
    for case_id, _, _, _ in _LOCKED_TEST_PLAN:
        names.update(_case_file_names(case_id))
    names.update(f"stage3_lock/package/{name}" for name in _STAGE3_PACKAGE_FILES)
    return frozenset(names)


_STAGE4_PACKAGE_FILES: Final = _expected_package_files()


def _file_mapping(root: Path) -> dict[str, bytes]:
    if root.is_symlink():
        raise ValueError("Stage 4 package root cannot be a symbolic link")
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Stage 4 package cannot contain symbolic links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Stage 4 package contains an unsupported entry: {relative}")
        if relative not in _STAGE4_PACKAGE_FILES:
            raise ValueError(f"Stage 4 package contains an unknown file: {relative}")
        discovered[relative] = path
    missing = sorted(_STAGE4_PACKAGE_FILES.difference(discovered))
    if missing:
        raise ValueError(f"Stage 4 package files missing={missing!r}")
    return {
        name: _read_bounded_artifact(discovered[name]) for name in sorted(_STAGE4_PACKAGE_FILES)
    }


def _serialized_replay(
    output_dir: Path,
    prepared: _PreparedCase,
) -> Stage3DualReplayResult:
    case_root = output_dir / "cases" / prepared.spec.case_id
    return replay_stage3_witness_pair(
        case_root,
        positive_witness=load_witness(case_root / "witnesses" / "positive.json"),
        negative_witness=load_witness(case_root / "witnesses" / "negative.json"),
        region=prepared.region,
        predicate=toy_lesion_predicate,
        predicate_id=STAGE4_TOY_PREDICATE_ID,
        expected_case_id=prepared.spec.case_id,
        expected_lesion_class=prepared.positive_witness.lesion_class,
        observation_sha256=prepared.observation_sha256,
        trace_sha256=prepared.trace_sha256,
    )


def verify_stage4_toy_package(output_dir: Path) -> Stage4ToyRunResult:
    """Verify source lock, exact file set, byte regeneration, and every witness replay."""

    if not isinstance(output_dir, Path) or not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    actual = _file_mapping(output_dir)
    config = parse_stage4_toy_config(
        _decode_json(actual["config_snapshot.json"], "Stage 4 config snapshot")
    )
    stage3_root = output_dir / "stage3_lock" / "package"
    lock, artifact_bytes, summary_bytes = _verified_locked_snapshot(
        stage3_root,
        config.stage3_lock,
    )
    if actual["locked_calibration_artifact.json"] != artifact_bytes:
        raise ValueError("copied Stage 3 calibration artifact does not match verified source")
    if actual["locked_calibration_summary.json"] != summary_bytes:
        raise ValueError("copied Stage 3 summary does not match verified source")
    prepared = _prepare_run(config)
    expected = dict(prepared.artifacts)
    expected["identity_isolation.json"] = _canonical_json_bytes(
        _identity_isolation_payload(stage3_root, prepared.cases)
    )
    expected["locked_calibration_artifact.json"] = artifact_bytes
    expected["locked_calibration_summary.json"] = summary_bytes
    for name in _STAGE3_PACKAGE_FILES:
        expected[f"stage3_lock/package/{name}"] = actual[f"stage3_lock/package/{name}"]

    replayed_cases: list[_PreparedCase] = []
    for case in prepared.cases:
        replay = _serialized_replay(output_dir, case)
        evaluation, certificate = _case_completion(
            spec=case.spec,
            region=case.region,
            observation_sha256=case.observation_sha256,
            trace_sha256=case.trace_sha256,
            positive_witness=case.positive_witness,
            negative_witness=case.negative_witness,
            replay=replay,
            lock=lock,
        )
        prefix = f"cases/{case.spec.case_id}/"
        expected[prefix + "dual_replay.json"] = _canonical_json_bytes(replay.to_dict())
        expected[prefix + "evaluation.json"] = _canonical_json_bytes(evaluation.to_dict())
        expected[prefix + "certificate.json"] = _canonical_json_bytes(certificate.to_dict())
        serialized_certificate = _decode_json(
            actual[prefix + "certificate.json"],
            f"{case.spec.case_id} certificate",
        )
        if not isinstance(serialized_certificate, Mapping):
            raise ValueError("Stage 4 case certificate must be a JSON object")
        if Stage3SyntheticCertificate.from_dict(serialized_certificate) != certificate:
            raise ValueError(f"Stage 4 certificate mismatch: {case.spec.case_id}")
        replayed_cases.append(
            _PreparedCase(
                spec=case.spec,
                region=case.region,
                observation_sha256=case.observation_sha256,
                trace_sha256=case.trace_sha256,
                positive_witness=case.positive_witness,
                negative_witness=case.negative_witness,
                replay=replay,
                evaluation=evaluation,
                certificate=certificate,
                source_image_id=case.source_image_id,
                patient_id=case.patient_id,
                intervention_pair_id=case.intervention_pair_id,
                artifacts=case.artifacts,
            )
        )
    if len(replayed_cases) != len(_LOCKED_TEST_PLAN):
        raise ValueError("Stage 4 test case count does not match the locked plan")
    if set(expected) != _STAGE4_PACKAGE_FILES:
        raise RuntimeError("internal Stage 4 package file contract is inconsistent")
    for name, expected_payload in expected.items():
        if actual[name] != expected_payload:
            raise ValueError(f"Stage 4 package artifact mismatch: {name}")
    actual_prepared = _PreparedRun(
        config=prepared.config,
        config_hash=prepared.config_hash,
        test_plan_hash=prepared.test_plan_hash,
        test_data_hash=prepared.test_data_hash,
        model_provenance_hash=prepared.model_provenance_hash,
        cases=tuple(replayed_cases),
        artifacts={},
    )
    expected["summary.json"] = _canonical_json_bytes(
        _summary_payload(prepared=actual_prepared, lock=lock)
    )
    actual_report = _aggregate_cases(tuple(replayed_cases), lock)
    audit_json, audit_html = _render_report_payloads(actual_report)
    expected["audit.json"] = audit_json
    expected["audit.html"] = audit_html
    successful = [
        case for case in replayed_cases if case.evaluation.status is EvaluationStatus.SUCCESS
    ]
    if any(not case.replay.passed for case in successful):
        raise ValueError("every successful truth-table case requires a valid dual witness replay")
    states = tuple(case.evaluation.state for case in replayed_cases)
    if set(states) != set(EntailmentState):
        raise ValueError("the Stage 4 toy package must exercise all four entailment states")
    return Stage4ToyRunResult(
        output_dir=output_dir,
        case_count=len(replayed_cases),
        states=states,
        locked_calibration_artifact_hash=lock.artifact_sha256,
        test_data_hash=prepared.test_data_hash,
        verified=True,
    )


def run_stage4_toy_fixture(
    config_path: Path,
    stage3_package: Path,
    output_dir: Path,
) -> Stage4ToyRunResult:
    """Generate the CPU/offline package from a verified frozen Stage 3 source package."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(stage3_package, Path):
        raise TypeError("stage3_package must be a pathlib.Path")
    try:
        source_root = stage3_package.resolve(strict=True)
        target_root = output_dir.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Stage 3 source and Stage 4 output paths must be resolvable") from error
    if (
        source_root == target_root
        or source_root in target_root.parents
        or target_root in source_root.parents
    ):
        raise ValueError("Stage 4 output must not overlap the read-only Stage 3 source package")
    config = load_stage4_toy_config(config_path)
    _, artifact_bytes, summary_bytes = _verified_locked_snapshot(
        stage3_package,
        config.stage3_lock,
    )
    prepared = _prepare_run(config)
    artifacts = dict(prepared.artifacts)
    artifacts["identity_isolation.json"] = _canonical_json_bytes(
        _identity_isolation_payload(stage3_package, prepared.cases)
    )
    artifacts["locked_calibration_artifact.json"] = artifact_bytes
    artifacts["locked_calibration_summary.json"] = summary_bytes
    for name in _STAGE3_PACKAGE_FILES:
        artifacts[f"stage3_lock/package/{name}"] = _read_bounded_artifact(stage3_package / name)
    if set(artifacts) != _STAGE4_PACKAGE_FILES:
        raise RuntimeError("internal Stage 4 package file contract is inconsistent")
    from retinaentail.reporting.stage3 import write_stage3_bundle

    write_stage3_bundle(
        output_dir,
        artifacts,
        boundary=AuditWriteBoundary(allowed_root=output_dir),
    )
    return verify_stage4_toy_package(output_dir)


__all__ = [
    "FailureMode",
    "LockedCalibrationSnapshot",
    "ObservationMode",
    "STAGE4_LOCK_METHOD",
    "STAGE4_TEST_METHOD",
    "STAGE4_TOY_CONFIG_VERSION",
    "STAGE4_TOY_PACKAGE_VERSION",
    "STAGE4_TOY_PREDICATE_ID",
    "Stage4ToyCaseSpec",
    "Stage4ToyConfig",
    "Stage4ToyConfigurationError",
    "Stage4ToyDegradation",
    "Stage4ToyRunResult",
    "evaluate_locked_replay",
    "load_stage4_toy_config",
    "parse_stage4_toy_config",
    "run_stage4_toy_fixture",
    "verify_stage4_toy_package",
]
