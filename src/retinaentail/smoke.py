"""Offline end-to-end toy pipeline for Stage 0/1 reproducibility checks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from retinaentail.degradations import (
    DegradationStep,
    ParameterValue,
    apply_pipeline,
    replay_trace,
)
from retinaentail.entailment.states import EntailmentState, fail_closed_state
from retinaentail.entailment.toy_baseline import normalized_mse, score_toy_hypotheses
from retinaentail.interventions.toy import generate_toy_fundus, toy_lesion_predicate
from retinaentail.schemas import (
    DegradationTrace,
    EntailmentCertificate,
    JSONValue,
    WitnessRecord,
)
from retinaentail.witnesses.replay import (
    ReplayResult,
    canonical_json_bytes,
    load_witness,
    rgb_png_bytes,
    sha256_bytes,
    verify_witness,
    write_bytes_fail_safe,
)


class SmokeConfigurationError(ValueError):
    """Raised when the toy smoke configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class SmokeDegradation:
    name: str
    severity: float
    parameters: dict[str, ParameterValue]


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    schema_version: str
    case_id: str
    seed: int
    image_size: int
    lesion_class: str
    lesion_radius: int
    alpha: float
    score_threshold: float
    degradations: tuple[SmokeDegradation, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "seed": self.seed,
            "image_size": self.image_size,
            "lesion_class": self.lesion_class,
            "lesion_radius": self.lesion_radius,
            "alpha": self.alpha,
            "score_threshold": self.score_threshold,
            "degradations": [
                {
                    "name": item.name,
                    "severity": item.severity,
                    "parameters": dict(item.parameters),
                }
                for item in self.degradations
            ],
        }


def _plain_int(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SmokeConfigurationError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_float(
    value: object, field: str, *, minimum: float, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeConfigurationError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise SmokeConfigurationError(f"{field} is outside its valid finite range")
    return result


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _parameter_mapping(value: object, field: str) -> dict[str, ParameterValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SmokeConfigurationError(f"{field} must be a mapping")
    result: dict[str, ParameterValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise SmokeConfigurationError(f"{field} keys must be non-empty strings")
        if isinstance(item, (str, bool, int)):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise SmokeConfigurationError(f"{field}.{key} must be a finite JSON scalar")
    return result


def parse_smoke_config(value: object) -> SmokeConfig:
    if not isinstance(value, Mapping):
        raise SmokeConfigurationError("smoke config must be a mapping")
    data: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SmokeConfigurationError("smoke config keys must be strings")
        data[key] = item
    required = {
        "schema_version",
        "case_id",
        "seed",
        "image_size",
        "lesion_class",
        "lesion_radius",
        "alpha",
        "score_threshold",
        "degradations",
    }
    missing = required.difference(data)
    unknown = set(data).difference(required)
    if missing or unknown:
        raise SmokeConfigurationError(
            f"smoke config missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    if data["schema_version"] != "1.0":
        raise SmokeConfigurationError("unsupported smoke config schema_version")
    raw_degradations = data["degradations"]
    if isinstance(raw_degradations, (str, bytes)) or not isinstance(raw_degradations, Sequence):
        raise SmokeConfigurationError("degradations must be a list")
    degradations: list[SmokeDegradation] = []
    for index, raw in enumerate(raw_degradations):
        if not isinstance(raw, Mapping):
            raise SmokeConfigurationError(f"degradations[{index}] must be a mapping")
        operation = dict(raw)
        unknown_operation = set(operation).difference({"name", "severity", "parameters"})
        if "name" not in operation or "severity" not in operation or unknown_operation:
            raise SmokeConfigurationError(f"degradations[{index}] has missing/unknown fields")
        degradations.append(
            SmokeDegradation(
                name=_nonempty(operation["name"], f"degradations[{index}].name"),
                severity=_finite_float(
                    operation["severity"],
                    f"degradations[{index}].severity",
                    minimum=0.0,
                    maximum=1.0,
                ),
                parameters=_parameter_mapping(
                    operation.get("parameters"), f"degradations[{index}].parameters"
                ),
            )
        )
    if not degradations:
        raise SmokeConfigurationError("at least one degradation is required")
    alpha = _finite_float(data["alpha"], "alpha", minimum=0.0, maximum=1.0)
    if alpha in {0.0, 1.0}:
        raise SmokeConfigurationError("alpha must lie strictly between zero and one")
    return SmokeConfig(
        schema_version="1.0",
        case_id=_nonempty(data["case_id"], "case_id"),
        seed=_plain_int(data["seed"], "seed", minimum=0),
        image_size=_plain_int(data["image_size"], "image_size", minimum=48),
        lesion_class=_nonempty(data["lesion_class"], "lesion_class"),
        lesion_radius=_plain_int(data["lesion_radius"], "lesion_radius", minimum=2),
        alpha=alpha,
        score_threshold=_finite_float(data["score_threshold"], "score_threshold", minimum=0.0),
        degradations=tuple(degradations),
    )


def load_smoke_config(path: Path) -> SmokeConfig:
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_smoke_config(decoded)


@dataclass(frozen=True, slots=True)
class SmokeRunResult:
    output_dir: Path
    certificate_path: Path
    state: EntailmentState
    replay_results: tuple[ReplayResult, ...]


def _certificate_from_path(path: Path) -> EntailmentCertificate:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("certificate JSON must contain an object")
    return EntailmentCertificate.from_dict(decoded)


def verify_smoke_package(
    output_dir: Path,
) -> tuple[EntailmentCertificate, tuple[ReplayResult, ...]]:
    """Reload the serialized package and independently replay both witness hypotheses."""

    certificate = _certificate_from_path(output_dir / "certificate.json")
    expected_paths = {
        "positive": output_dir / "positive_witness.json",
        "negative": output_dir / "negative_witness.json",
    }
    loaded: list[WitnessRecord] = []
    for side in ("positive", "negative"):
        witness = load_witness(expected_paths[side])
        if witness.hypothesis_side != side:
            raise ValueError(f"{side} witness file contains the wrong hypothesis side")
        loaded.append(witness)
    embedded = {witness.hypothesis_side: witness.to_dict() for witness in certificate.witnesses}
    if {witness.hypothesis_side: witness.to_dict() for witness in loaded} != embedded:
        raise ValueError("standalone witness files do not match the certificate")
    results = tuple(verify_witness(output_dir, witness, certificate.region) for witness in loaded)
    if not all(result.passed for result in results):
        raise ValueError("one or more witness replays failed")
    return certificate, results


def run_smoke(config_path: Path, output_dir: Path) -> SmokeRunResult:
    """Run the deterministic CPU toy pipeline and write a replayable artifact package."""

    config = load_smoke_config(config_path)
    twin = generate_toy_fundus(
        size=config.image_size,
        lesion_radius=config.lesion_radius,
        seed=config.seed,
    )
    positive_predicate = toy_lesion_predicate(twin.lesion_present, twin.region)
    negative_predicate = toy_lesion_predicate(twin.lesion_absent, twin.region)
    if not positive_predicate or negative_predicate:
        raise RuntimeError("procedural intervention failed its explicit toy predicate")

    requested_steps = tuple(
        DegradationStep(
            name=item.name,
            severity=item.severity,
            seed=config.seed + index,
            parameters=item.parameters,
        )
        for index, item in enumerate(config.degradations)
    )
    degraded_positive, operation_trace = apply_pipeline(twin.lesion_present, requested_steps)
    degraded_negative = replay_trace(twin.lesion_absent, operation_trace)
    observation = degraded_positive.copy()
    positive, negative = score_toy_hypotheses(
        observation,
        degraded_positive,
        degraded_negative,
        threshold=config.score_threshold,
    )
    state = fail_closed_state(
        positive_feasible=positive.feasible,
        negative_feasible=negative.feasible,
        positive_score=positive.score,
        negative_score=negative.score,
        positive_threshold=positive.threshold,
        negative_threshold=negative.threshold,
        positive_solver_status=positive.solver_status,
        negative_solver_status=negative.solver_status,
        solver_status="toy_baseline",
        degradation_known=True,
    )
    if state is not EntailmentState.RAW_UNIDENTIFIABLE:
        raise RuntimeError(
            "toy smoke configuration must make both explicit hypotheses feasible; "
            f"got {state.value}"
        )

    trace_mapping = operation_trace.to_dict()
    schema_trace = DegradationTrace.from_dict(cast(Mapping[str, object], trace_mapping))
    present_bytes = rgb_png_bytes(twin.lesion_present)
    absent_bytes = rgb_png_bytes(twin.lesion_absent)
    observation_bytes = rgb_png_bytes(observation)
    positive_error = normalized_mse(degraded_positive, observation)
    negative_error = normalized_mse(degraded_negative, observation)
    positive_witness = WitnessRecord(
        witness_id=f"{config.case_id}:positive",
        case_id=config.case_id,
        lesion_class=config.lesion_class,
        witness_kind="lesion_present",
        relative_path="lesion_present.png",
        observation_path="observation.png",
        sha256=sha256_bytes(present_bytes),
        degradation_trace=schema_trace,
        replay_seed=config.seed,
        replay_error=positive_error,
        predicate_valid=True,
    )
    negative_witness = WitnessRecord(
        witness_id=f"{config.case_id}:negative",
        case_id=config.case_id,
        lesion_class=config.lesion_class,
        witness_kind="lesion_absent",
        relative_path="lesion_absent.png",
        observation_path="observation.png",
        sha256=sha256_bytes(absent_bytes),
        degradation_trace=schema_trace,
        replay_seed=config.seed,
        replay_error=negative_error,
        predicate_valid=True,
    )
    certificate = EntailmentCertificate(
        case_id=config.case_id,
        lesion_class=config.lesion_class,
        region=twin.region,
        alpha=config.alpha,
        positive=positive,
        negative=negative,
        state=state,
        config_hash=sha256_bytes(canonical_json_bytes(config.to_dict())),
        data_hash=sha256_bytes(observation_bytes),
        model_hashes={
            "toy_pixel_replay_baseline": sha256_bytes(
                b"retinaentail-dr:toy-pixel-replay-baseline:v1"
            )
        },
        solver_status="toy_baseline",
        witnesses=(positive_witness, negative_witness),
        degradation_trace=schema_trace,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_bytes_fail_safe(output_dir / "observation.png", observation_bytes)
    write_bytes_fail_safe(output_dir / "lesion_present.png", present_bytes)
    write_bytes_fail_safe(output_dir / "lesion_absent.png", absent_bytes)
    write_bytes_fail_safe(
        output_dir / "positive_witness.json",
        canonical_json_bytes(positive_witness.to_dict()),
    )
    write_bytes_fail_safe(
        output_dir / "negative_witness.json",
        canonical_json_bytes(negative_witness.to_dict()),
    )
    certificate_path = output_dir / "certificate.json"
    write_bytes_fail_safe(certificate_path, canonical_json_bytes(certificate.to_dict()))

    reloaded_certificate, replay_results = verify_smoke_package(output_dir)
    summary: dict[str, JSONValue] = {
        "schema_version": "1.0.0",
        "case_id": reloaded_certificate.case_id,
        "state": reloaded_certificate.state.value,
        "cpu_only": True,
        "network_used": False,
        "real_data_used": False,
        "witnesses": [
            {
                "witness_id": result.witness_id,
                "replay_error": result.replay_error,
                "predicate_valid": result.predicate_valid,
                "digest_valid": result.digest_valid,
                "passed": result.passed,
            }
            for result in replay_results
        ],
    }
    write_bytes_fail_safe(output_dir / "summary.json", canonical_json_bytes(summary))
    return SmokeRunResult(
        output_dir=output_dir,
        certificate_path=certificate_path,
        state=reloaded_certificate.state,
        replay_results=replay_results,
    )


__all__ = [
    "SmokeConfig",
    "SmokeConfigurationError",
    "SmokeDegradation",
    "SmokeRunResult",
    "load_smoke_config",
    "parse_smoke_config",
    "run_smoke",
    "verify_smoke_package",
]
