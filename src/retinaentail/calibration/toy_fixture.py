"""Deterministic, CPU-only Stage 3 calibration and dual-witness fixture.

This module is a protocol exerciser.  It uses only procedural toy fundus twins,
does not read a dataset root, and does not claim clinical or statistical validity.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from retinaentail.calibration.certificate import (
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
    EvaluationResult,
    HypothesisEvaluation,
    SampleSplit,
    calibrate_order_statistic,
    evaluate_hypotheses,
)
from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.degradations import (
    DegradationStep,
    ParameterValue,
    apply_pipeline,
    replay_trace,
    validate_operator_parameters,
)
from retinaentail.entailment.states import EntailmentState
from retinaentail.entailment.toy_baseline import normalized_mse
from retinaentail.interventions.toy import generate_toy_fundus, toy_lesion_predicate
from retinaentail.reporting.stage3 import write_stage3_bundle
from retinaentail.schemas import KNOWN_DEGRADATIONS, DegradationTrace, LesionRegion, WitnessRecord
from retinaentail.witnesses.replay import (
    load_witness,
    rgb_png_bytes,
    sha256_bytes,
)
from retinaentail.witnesses.stage3_replay import (
    Stage3DualReplayResult,
    Stage3ReplayResult,
    replay_stage3_witness_pair,
)

STAGE3_TOY_CONFIG_VERSION: Final = "3.0"
STAGE3_TOY_PACKAGE_VERSION: Final = "3.0.0"
STAGE3_TOY_PREDICATE_ID: Final = "toy_dark_red_round_lesion_v1"
STAGE3_TOY_RESIDUAL_RULE: Final = "seeded_sparse_bounded_rgb_delta_v1"
_MAX_STAGE3_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
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


class Stage3ToyConfigurationError(ValueError):
    """Raised when the synthetic calibration configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class Stage3ToyDegradation:
    name: str
    severity: float
    parameters: dict[str, ParameterValue]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "severity": self.severity,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class Stage3ToyConfig:
    schema_version: str
    case_id: str
    seed: int
    image_size: int
    lesion_class: str
    lesion_radius: int
    alpha: float
    calibration_residual_levels: tuple[int, ...]
    degradations: tuple[Stage3ToyDegradation, ...]

    @property
    def per_side_sample_count(self) -> int:
        return len(self.calibration_residual_levels)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "seed": self.seed,
            "image_size": self.image_size,
            "lesion_class": self.lesion_class,
            "lesion_radius": self.lesion_radius,
            "alpha": self.alpha,
            "calibration_residual_levels": list(self.calibration_residual_levels),
            "degradations": [item.to_dict() for item in self.degradations],
        }


@dataclass(frozen=True, slots=True)
class Stage3ToyRunResult:
    output_dir: Path
    certificate_path: Path
    state: EntailmentState
    calibration_artifact_hash: str
    replay: Stage3DualReplayResult
    verified: bool


@dataclass(frozen=True, slots=True)
class _CalibrationPreparation:
    config: Stage3ToyConfig
    config_hash: str
    data_hash: str
    model_hash: str
    model_provenance: dict[str, object]
    samples: tuple[CalibrationSample, ...]
    samples_payload: dict[str, object]
    artifact: CalibrationArtifact


@dataclass(frozen=True, slots=True)
class _FixturePreparation:
    calibration: _CalibrationPreparation
    region: LesionRegion
    observation_sha256: str
    trace_sha256: str
    positive_witness: WitnessRecord
    negative_witness: WitnessRecord
    base_artifacts: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class _FixtureCompletion:
    replay: Stage3DualReplayResult
    evaluation: EvaluationResult
    certificate: Stage3SyntheticCertificate
    final_artifacts: dict[str, bytes]


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
        raise Stage3ToyConfigurationError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise Stage3ToyConfigurationError(f"{name} keys must be strings")
        result[key] = item
    missing = required.difference(result)
    unknown = set(result).difference(required)
    if missing or unknown:
        raise Stage3ToyConfigurationError(
            f"{name} missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage3ToyConfigurationError(f"{name} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in "\r\n\x00"):
        raise Stage3ToyConfigurationError(f"{name} contains a forbidden control character")
    return result


def _plain_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage3ToyConfigurationError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise Stage3ToyConfigurationError(f"{name} must be <= {maximum}")
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage3ToyConfigurationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise Stage3ToyConfigurationError(f"{name} is outside its finite range")
    return result


def _parameters(value: object, name: str) -> dict[str, ParameterValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Stage3ToyConfigurationError(f"{name} must be a mapping")
    result: dict[str, ParameterValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise Stage3ToyConfigurationError(f"{name} keys must be non-empty strings")
        if isinstance(item, str | bool | int):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise Stage3ToyConfigurationError(f"{name}.{key} must be a finite JSON scalar")
    return result


def parse_stage3_toy_config(value: object) -> Stage3ToyConfig:
    """Parse an exact, versioned synthetic configuration without coercing unsafe values."""

    required = {
        "schema_version",
        "case_id",
        "seed",
        "image_size",
        "lesion_class",
        "lesion_radius",
        "alpha",
        "calibration_residual_levels",
        "degradations",
    }
    data = _strict_mapping(value, name="Stage 3 toy config", required=required)
    if data["schema_version"] != STAGE3_TOY_CONFIG_VERSION:
        raise Stage3ToyConfigurationError("unsupported Stage 3 toy config schema_version")

    raw_levels = data["calibration_residual_levels"]
    if isinstance(raw_levels, str | bytes) or not isinstance(raw_levels, Sequence):
        raise Stage3ToyConfigurationError("calibration_residual_levels must be a list")
    levels = tuple(
        _plain_int(item, f"calibration_residual_levels[{index}]", minimum=1, maximum=32)
        for index, item in enumerate(raw_levels)
    )
    if len(levels) < 3 or len(set(levels)) != len(levels):
        raise Stage3ToyConfigurationError(
            "calibration_residual_levels must contain at least three unique values"
        )

    raw_degradations = data["degradations"]
    if isinstance(raw_degradations, str | bytes) or not isinstance(raw_degradations, Sequence):
        raise Stage3ToyConfigurationError("degradations must be a list")
    degradations: list[Stage3ToyDegradation] = []
    for index, raw in enumerate(raw_degradations):
        operation = _strict_mapping(
            raw,
            name=f"degradations[{index}]",
            required={"name", "severity", "parameters"}
            if isinstance(raw, Mapping) and "parameters" in raw
            else {"name", "severity"},
        )
        name = _nonempty_string(operation["name"], f"degradations[{index}].name")
        if name not in KNOWN_DEGRADATIONS:
            raise Stage3ToyConfigurationError(f"unknown degradation operator: {name}")
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
            raise Stage3ToyConfigurationError(
                f"invalid degradations[{index}] parameters: {error}"
            ) from error
        degradations.append(
            Stage3ToyDegradation(
                name=name,
                severity=severity,
                parameters=parameters,
            )
        )
    if not degradations:
        raise Stage3ToyConfigurationError("at least one known degradation is required")

    alpha = _finite_float(data["alpha"], "alpha", minimum=0.0, maximum=1.0)
    if alpha in {0.0, 1.0}:
        raise Stage3ToyConfigurationError("alpha must lie strictly between zero and one")
    rank = math.ceil((len(levels) + 1) * (1.0 - alpha))
    if rank > len(levels):
        raise Stage3ToyConfigurationError(
            "alpha and sample count cannot issue a finite conservative threshold"
        )

    config = Stage3ToyConfig(
        schema_version=STAGE3_TOY_CONFIG_VERSION,
        case_id=_nonempty_string(data["case_id"], "case_id"),
        seed=_plain_int(data["seed"], "seed", minimum=0, maximum=2_000_000_000),
        image_size=_plain_int(data["image_size"], "image_size", minimum=48, maximum=512),
        lesion_class=_nonempty_string(data["lesion_class"], "lesion_class"),
        lesion_radius=_plain_int(data["lesion_radius"], "lesion_radius", minimum=2, maximum=64),
        alpha=alpha,
        calibration_residual_levels=levels,
        degradations=tuple(degradations),
    )
    if config.lesion_radius * 6 >= config.image_size:
        raise Stage3ToyConfigurationError("lesion_radius is too large for image_size")
    return config


def load_stage3_toy_config(path: Path) -> Stage3ToyConfig:
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_stage3_toy_config(decoded)


def _operation_steps(
    config: Stage3ToyConfig,
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
        raise TypeError("degradation trace serialization must be a mapping")
    return DegradationTrace.from_dict(cast(Mapping[str, object], value))


def _add_sparse_residual(
    image: NDArray[np.uint8],
    *,
    delta: int,
    seed: int,
) -> NDArray[np.uint8]:
    """Add a deterministic bounded residual used only to make toy thresholds non-zero."""

    output = image.copy()
    flat = output.reshape(-1)
    generator = np.random.default_rng(seed)
    count = max(1, flat.size // 20)
    indices = generator.choice(flat.size, size=count, replace=False)
    values = flat[indices].astype(np.int16)
    direction = np.where(values <= 255 - delta, delta, -delta)
    flat[indices] = (values + direction).astype(np.uint8)
    return output


def _model_provenance() -> tuple[str, dict[str, object]]:
    package_root = Path(__file__).resolve().parents[1]
    components = {
        "certificate_contract": package_root / "calibration" / "certificate.py",
        "calibration_protocol": package_root / "calibration" / "protocol.py",
        "degradation_operators": package_root / "degradations" / "operators.py",
        "degradation_pipeline": package_root / "degradations" / "pipeline.py",
        "entailment_states": package_root / "entailment" / "states.py",
        "shared_schemas": package_root / "schemas.py",
        "stage3_fixture": package_root / "calibration" / "toy_fixture.py",
        "toy_pixel_replay_baseline": package_root / "entailment" / "toy_baseline.py",
        "toy_twin_and_predicate": package_root / "interventions" / "toy.py",
        "witness_serialization": package_root / "witnesses" / "replay.py",
        "witness_replay": package_root / "witnesses" / "stage3_replay.py",
    }
    payload: dict[str, object] = {
        "schema_version": STAGE3_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_TOY",
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


def _prepare_calibration(config: Stage3ToyConfig) -> _CalibrationPreparation:
    config_hash = sha256_bytes(_canonical_json_bytes(config.to_dict()))
    model_hash, model_provenance = _model_provenance()
    samples: list[CalibrationSample] = []
    records: list[dict[str, object]] = []
    for index, level in enumerate(config.calibration_residual_levels):
        image_seed = config.seed + index
        trace_seed = config.seed + 10_000 + index * 100
        twin = generate_toy_fundus(
            size=config.image_size,
            lesion_radius=config.lesion_radius,
            seed=image_seed,
        )
        positive_replay, trace = apply_pipeline(
            twin.lesion_present,
            _operation_steps(config, trace_seed=trace_seed),
        )
        negative_replay = replay_trace(twin.lesion_absent, trace)
        positive_observation = _add_sparse_residual(
            positive_replay,
            delta=level,
            seed=trace_seed + 50,
        )
        negative_observation = _add_sparse_residual(
            negative_replay,
            delta=level,
            seed=trace_seed + 51,
        )
        source_id = f"toy-source-{index:03d}"
        patient_id = f"toy-patient-{index:03d}"
        pair_id = f"toy-intervention-pair-{index:03d}"
        pair_samples = (
            CalibrationSample(
                sample_id=f"toy-calibration-{index:03d}-present",
                split=SampleSplit.CALIBRATION,
                label=CalibrationLabel.LESION_PRESENT,
                positive_score=normalized_mse(positive_replay, positive_observation),
                negative_score=normalized_mse(negative_replay, positive_observation),
                source_image_id=source_id,
                patient_id=patient_id,
                intervention_pair_id=pair_id,
            ),
            CalibrationSample(
                sample_id=f"toy-calibration-{index:03d}-absent",
                split=SampleSplit.CALIBRATION,
                label=CalibrationLabel.LESION_ABSENT,
                positive_score=normalized_mse(positive_replay, negative_observation),
                negative_score=normalized_mse(negative_replay, negative_observation),
                source_image_id=source_id,
                patient_id=patient_id,
                intervention_pair_id=pair_id,
            ),
        )
        trace_payload = trace.to_dict()
        trace_hash = sha256_bytes(_canonical_json_bytes(trace_payload))
        observations = (positive_observation, negative_observation)
        for sample, observation in zip(pair_samples, observations, strict=True):
            samples.append(sample)
            records.append(
                {
                    "sample": sample.to_dict(),
                    "synthetic": True,
                    "image_seed": image_seed,
                    "trace_seed": trace_seed,
                    "residual_delta": level,
                    "residual_rule": STAGE3_TOY_RESIDUAL_RULE,
                    "observation_sha256": sha256_bytes(rgb_png_bytes(observation)),
                    "degradation_trace_sha256": trace_hash,
                    "degradation_trace": trace_payload,
                }
            )
    samples_payload: dict[str, object] = {
        "schema_version": STAGE3_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "label_source": "procedural_twin_control",
        "missing_label_policy": "UNKNOWN_NEVER_NEGATIVE",
        "records": records,
    }
    data_hash = sha256_bytes(_canonical_json_bytes(samples_payload))
    sample_tuple = tuple(samples)
    artifact = calibrate_order_statistic(
        sample_tuple,
        alpha=config.alpha,
        config_hash=config_hash,
        data_hash=data_hash,
        model_hash=model_hash,
    )
    if artifact.status is not CalibrationRunStatus.SUCCESS:
        reasons = ", ".join(reason.value for reason in artifact.failure_reasons)
        raise RuntimeError(f"synthetic calibration failed closed: {reasons}")
    return _CalibrationPreparation(
        config=config,
        config_hash=config_hash,
        data_hash=data_hash,
        model_hash=model_hash,
        model_provenance=model_provenance,
        samples=sample_tuple,
        samples_payload=samples_payload,
        artifact=artifact,
    )


def _prepare_fixture(config: Stage3ToyConfig) -> _FixturePreparation:
    calibration = _prepare_calibration(config)
    image_seed = config.seed + 100_000
    trace_seed = config.seed + 110_000
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
    observation = positive_replay.copy()
    if not toy_lesion_predicate(twin.lesion_present, twin.region) or toy_lesion_predicate(
        twin.lesion_absent, twin.region
    ):
        raise RuntimeError("procedural Stage 3 twins violate the declared toy predicate")
    trace = _schema_trace(operation_trace.to_dict())
    trace_sha256 = sha256_bytes(_canonical_json_bytes(trace.to_dict()))
    observation_bytes = rgb_png_bytes(observation)
    present_bytes = rgb_png_bytes(twin.lesion_present)
    absent_bytes = rgb_png_bytes(twin.lesion_absent)
    observation_sha256 = sha256_bytes(observation_bytes)
    positive_witness = WitnessRecord(
        witness_id=f"{config.case_id}:positive",
        case_id=config.case_id,
        lesion_class=config.lesion_class,
        witness_kind="lesion_present",
        relative_path="images/lesion_present.png",
        observation_path="images/observation.png",
        sha256=sha256_bytes(present_bytes),
        degradation_trace=trace,
        replay_seed=trace_seed,
        replay_error=normalized_mse(positive_replay, observation),
        predicate_valid=True,
    )
    negative_witness = WitnessRecord(
        witness_id=f"{config.case_id}:negative",
        case_id=config.case_id,
        lesion_class=config.lesion_class,
        witness_kind="lesion_absent",
        relative_path="images/lesion_absent.png",
        observation_path="images/observation.png",
        sha256=sha256_bytes(absent_bytes),
        degradation_trace=trace,
        replay_seed=trace_seed,
        replay_error=normalized_mse(negative_replay, observation),
        predicate_valid=True,
    )
    base_artifacts = {
        "config_snapshot.json": _canonical_json_bytes(config.to_dict()),
        "calibration_samples.json": _canonical_json_bytes(calibration.samples_payload),
        "calibration_artifact.json": _canonical_json_bytes(calibration.artifact.to_dict()),
        "model_provenance.json": _canonical_json_bytes(calibration.model_provenance),
        "images/observation.png": observation_bytes,
        "images/lesion_present.png": present_bytes,
        "images/lesion_absent.png": absent_bytes,
        "witnesses/positive.json": _canonical_json_bytes(positive_witness.to_dict()),
        "witnesses/negative.json": _canonical_json_bytes(negative_witness.to_dict()),
    }
    return _FixturePreparation(
        calibration=calibration,
        region=twin.region,
        observation_sha256=observation_sha256,
        trace_sha256=trace_sha256,
        positive_witness=positive_witness,
        negative_witness=negative_witness,
        base_artifacts=base_artifacts,
    )


def _predicted_replay(prepared: _FixturePreparation) -> Stage3DualReplayResult:
    """Build pre-write evidence from already computed deterministic in-memory artifacts."""

    calibration = prepared.calibration
    region_sha256 = sha256_bytes(_canonical_json_bytes(prepared.region.to_dict()))

    def one(witness: WitnessRecord, side: str) -> Stage3ReplayResult:
        result = Stage3ReplayResult(
            witness_id=witness.witness_id,
            case_id=calibration.config.case_id,
            lesion_class=calibration.config.lesion_class,
            side=side,
            predicate_id=STAGE3_TOY_PREDICATE_ID,
            region_sha256=region_sha256,
            expected_latent_sha256=witness.sha256,
            actual_latent_sha256=witness.sha256,
            expected_observation_sha256=prepared.observation_sha256,
            actual_observation_sha256=prepared.observation_sha256,
            expected_trace_sha256=prepared.trace_sha256,
            actual_trace_sha256=prepared.trace_sha256,
            replay_error=witness.replay_error,
            predicate_valid=True,
            passed=True,
        )
        return Stage3ReplayResult.from_dict(result.to_dict())

    result = Stage3DualReplayResult(
        positive=one(prepared.positive_witness, "positive"),
        negative=one(prepared.negative_witness, "negative"),
        passed=True,
    )
    return Stage3DualReplayResult.from_dict(result.to_dict())


def _replay_serialized_package(
    prepared: _FixturePreparation, package_root: Path
) -> Stage3DualReplayResult:
    calibration = prepared.calibration
    return replay_stage3_witness_pair(
        package_root,
        positive_witness=load_witness(package_root / "witnesses" / "positive.json"),
        negative_witness=load_witness(package_root / "witnesses" / "negative.json"),
        region=prepared.region,
        predicate=toy_lesion_predicate,
        predicate_id=STAGE3_TOY_PREDICATE_ID,
        expected_case_id=calibration.config.case_id,
        expected_lesion_class=calibration.config.lesion_class,
        observation_sha256=prepared.observation_sha256,
        trace_sha256=prepared.trace_sha256,
    )


def _complete_fixture(
    prepared: _FixturePreparation, replay: Stage3DualReplayResult
) -> _FixtureCompletion:
    calibration = prepared.calibration
    evaluation_input = HypothesisEvaluation(
        case_id=calibration.config.case_id,
        lesion_class=calibration.config.lesion_class,
        positive_score=replay.positive.replay_error,
        negative_score=replay.negative.replay_error,
        config_hash=calibration.config_hash,
        data_hash=calibration.data_hash,
        model_hash=calibration.model_hash,
        positive_solver_status="toy_baseline",
        negative_solver_status="toy_baseline",
        solver_status="toy_baseline",
        degradation_known=True,
        positive_witness_replay_valid=replay.positive.passed,
        negative_witness_replay_valid=replay.negative.passed,
        out_of_distribution=False,
    )
    evaluation = evaluate_hypotheses(evaluation_input, calibration.artifact)
    certificate = Stage3SyntheticCertificate(
        evaluation=evaluation,
        region=prepared.region,
        predicate_id=STAGE3_TOY_PREDICATE_ID,
        calibration_artifact_hash=calibration_artifact_sha256(calibration.artifact),
        observation_sha256=prepared.observation_sha256,
        degradation_trace_sha256=prepared.trace_sha256,
        positive_evidence=Stage3WitnessEvidence(
            side=WitnessSide.POSITIVE,
            witness=prepared.positive_witness,
            replay=replay.positive,
        ),
        negative_evidence=Stage3WitnessEvidence(
            side=WitnessSide.NEGATIVE,
            witness=prepared.negative_witness,
            replay=replay.negative,
        ),
    )
    missing_threshold = evaluate_hypotheses(
        evaluation_input,
        replace(calibration.artifact, positive_threshold=None),
    )
    unknown_degradation = evaluate_hypotheses(
        replace(evaluation_input, degradation_known=False),
        calibration.artifact,
    )
    summary: dict[str, object] = {
        "schema_version": STAGE3_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_TOY",
        "case_id": evaluation.case_id,
        "state": evaluation.state.value,
        "evaluation_status": evaluation.status.value,
        "alpha": evaluation.alpha,
        "positive_threshold": evaluation.positive_threshold,
        "negative_threshold": evaluation.negative_threshold,
        "positive_sample_count": evaluation.positive_sample_count,
        "negative_sample_count": evaluation.negative_sample_count,
        "config_hash": calibration.config_hash,
        "calibration_data_hash": calibration.data_hash,
        "model_hash": calibration.model_hash,
        "observation_sha256": prepared.observation_sha256,
        "degradation_trace_sha256": prepared.trace_sha256,
        "calibration_artifact_hash": calibration_artifact_sha256(calibration.artifact),
        "positive_witness_replay_passed": replay.positive.passed,
        "negative_witness_replay_passed": replay.negative.passed,
        "cpu_only": True,
        "gpu_used": False,
        "network_used": False,
        "real_data_used": False,
        "real_data_gate": "BLOCKED_NO_EXPLICIT_APPROVED_DATASET",
        "clinical_claim": False,
    }
    final_artifacts = {
        "dual_replay.json": _canonical_json_bytes(replay.to_dict()),
        "evaluation.json": _canonical_json_bytes(evaluation.to_dict()),
        "certificate.json": _canonical_json_bytes(certificate.to_dict()),
        "fail_closed/missing_threshold.json": _canonical_json_bytes(missing_threshold.to_dict()),
        "fail_closed/unknown_degradation.json": _canonical_json_bytes(
            unknown_degradation.to_dict()
        ),
        "summary.json": _canonical_json_bytes(summary),
    }
    return _FixtureCompletion(
        replay=replay,
        evaluation=evaluation,
        certificate=certificate,
        final_artifacts=final_artifacts,
    )


def _read_bounded_artifact(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError("Stage 3 package artifact cannot be inspected") from error
    if size > _MAX_STAGE3_ARTIFACT_BYTES:
        raise ValueError(f"Stage 3 package artifact exceeds size limit: {path.name}")
    try:
        with path.open("rb") as stream:
            payload = stream.read(_MAX_STAGE3_ARTIFACT_BYTES + 1)
    except OSError as error:
        raise ValueError("Stage 3 package artifact cannot be read") from error
    if len(payload) > _MAX_STAGE3_ARTIFACT_BYTES:
        raise ValueError(f"Stage 3 package artifact exceeds size limit: {path.name}")
    return payload


def _file_mapping(root: Path) -> dict[str, bytes]:
    if root.is_symlink():
        raise ValueError("Stage 3 package root cannot be a symbolic link")
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Stage 3 package cannot contain symbolic links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Stage 3 package contains an unsupported entry: {relative}")
        if relative not in _STAGE3_PACKAGE_FILES:
            raise ValueError(f"Stage 3 package contains an unknown file: {relative}")
        discovered[relative] = path
    missing = sorted(_STAGE3_PACKAGE_FILES.difference(discovered))
    if missing:
        raise ValueError(f"Stage 3 package files missing={missing!r}")
    return {
        name: _read_bounded_artifact(discovered[name]) for name in sorted(_STAGE3_PACKAGE_FILES)
    }


def verify_stage3_toy_package(output_dir: Path) -> Stage3ToyRunResult:
    """Regenerate the synthetic contract and independently replay the serialized package."""

    if not isinstance(output_dir, Path) or not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    actual = _file_mapping(output_dir)
    try:
        decoded: object = json.loads(actual["config_snapshot.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 3 config snapshot is not valid UTF-8 JSON") from error
    config = parse_stage3_toy_config(decoded)
    prepared = _prepare_fixture(config)
    for name, expected_payload in prepared.base_artifacts.items():
        if actual.get(name) != expected_payload:
            raise ValueError(f"Stage 3 package base artifact mismatch: {name}")
    completion = _complete_fixture(prepared, _replay_serialized_package(prepared, output_dir))
    expected = dict(prepared.base_artifacts)
    expected.update(completion.final_artifacts)
    if set(expected) != _STAGE3_PACKAGE_FILES:
        raise RuntimeError("internal Stage 3 package file contract is inconsistent")
    for name, expected_payload in expected.items():
        if actual[name] != expected_payload:
            raise ValueError(f"Stage 3 package artifact mismatch: {name}")
    if completion.evaluation.state is not EntailmentState.ENTAILED_PRESENT:
        raise ValueError("the deterministic Stage 3 fixture must entail the present twin")
    if not completion.replay.passed:
        raise ValueError("the deterministic Stage 3 fixture must replay both witnesses")
    return Stage3ToyRunResult(
        output_dir=output_dir,
        certificate_path=output_dir / "certificate.json",
        state=completion.evaluation.state,
        calibration_artifact_hash=calibration_artifact_sha256(prepared.calibration.artifact),
        replay=completion.replay,
        verified=True,
    )


def run_stage3_toy_fixture(config_path: Path, output_dir: Path) -> Stage3ToyRunResult:
    """Generate, calibrate, write, replay, and verify the deterministic toy package."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    config = load_stage3_toy_config(config_path)
    prepared = _prepare_fixture(config)
    completion = _complete_fixture(prepared, _predicted_replay(prepared))
    artifacts = dict(prepared.base_artifacts)
    artifacts.update(completion.final_artifacts)
    boundary = AuditWriteBoundary(allowed_root=output_dir)
    write_stage3_bundle(
        output_dir,
        artifacts,
        boundary=boundary,
    )
    return verify_stage3_toy_package(output_dir)


__all__ = [
    "STAGE3_TOY_CONFIG_VERSION",
    "STAGE3_TOY_PACKAGE_VERSION",
    "STAGE3_TOY_PREDICATE_ID",
    "STAGE3_TOY_RESIDUAL_RULE",
    "Stage3ToyConfig",
    "Stage3ToyConfigurationError",
    "Stage3ToyDegradation",
    "Stage3ToyRunResult",
    "load_stage3_toy_config",
    "parse_stage3_toy_config",
    "run_stage3_toy_fixture",
    "verify_stage3_toy_package",
]
