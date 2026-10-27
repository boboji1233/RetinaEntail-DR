"""Ordered, JSON-serializable degradation traces and replay support."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray

from .operators import (
    OPERATORS,
    ImageArray,
    Parameters,
    ParameterValue,
    apply_operator,
    validate_image,
    validate_operator_parameter_names,
    validate_operator_parameters,
    validate_seed,
    validate_severity,
)

TRACE_SCHEMA_VERSION = "1.0.0"


def _normalize_parameter_value(name: str, value: object) -> ParameterValue:
    if isinstance(value, str | bool):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"parameter {name!r} must be finite")
        return result
    raise TypeError(f"parameter {name!r} is not JSON-compatible")


def _normalize_parameters(parameters: Mapping[str, object]) -> Parameters:
    result: Parameters = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or not name:
            raise TypeError("parameter names must be non-empty strings")
        result[name] = _normalize_parameter_value(name, value)
    return result


@dataclass(frozen=True, slots=True)
class DegradationStep:
    """One validated operation in an ordered degradation trace."""

    name: str
    severity: float
    seed: int
    parameters: Parameters = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in OPERATORS:
            raise ValueError(f"unknown degradation operator: {self.name!r}")
        object.__setattr__(self, "severity", validate_severity(self.severity))
        object.__setattr__(self, "seed", validate_seed(self.seed))
        normalized_parameters = _normalize_parameters(self.parameters)
        validate_operator_parameters(self.name, normalized_parameters)
        object.__setattr__(self, "parameters", normalized_parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "severity": self.severity,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        required = {"name", "severity", "seed", "parameters"}
        missing = required.difference(value)
        unknown = set(value).difference(required)
        if missing:
            raise ValueError(f"degradation step missing field(s): {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"degradation step has unknown field(s): {', '.join(sorted(unknown))}")
        name = value["name"]
        severity = value["severity"]
        seed = value["seed"]
        parameters = value["parameters"]
        if not isinstance(name, str):
            raise TypeError("degradation step name must be a string")
        if isinstance(severity, bool) or not isinstance(severity, int | float):
            raise TypeError("degradation step severity must be a real number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("degradation step seed must be an integer")
        if not isinstance(parameters, Mapping):
            raise TypeError("degradation step parameters must be an object")
        string_parameters = cast(Mapping[str, object], parameters)
        return cls(
            name=name,
            severity=float(severity),
            seed=seed,
            parameters=_normalize_parameters(string_parameters),
        )


@dataclass(frozen=True, slots=True)
class OperationTrace:
    """Versioned, ordered trace used by the CPU degradation MVP."""

    operations: tuple[DegradationStep, ...] = ()
    schema_version: str = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported degradation trace schema_version: {self.schema_version!r}"
            )
        operations = tuple(self.operations)
        if not all(isinstance(step, DegradationStep) for step in operations):
            raise TypeError("operations must contain only DegradationStep values")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operations": [step.to_dict() for step in self.operations],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        required = {"schema_version", "operations"}
        missing = required.difference(value)
        unknown = set(value).difference(required)
        if missing:
            raise ValueError(f"degradation trace missing field(s): {', '.join(sorted(missing))}")
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"degradation trace has unknown field(s): {names}")
        schema_version = value["schema_version"]
        operations = value["operations"]
        if not isinstance(schema_version, str):
            raise TypeError("schema_version must be a string")
        if not isinstance(operations, list):
            raise TypeError("operations must be a JSON array")
        parsed: list[DegradationStep] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise TypeError("each degradation operation must be an object")
            parsed.append(DegradationStep.from_dict(cast(Mapping[str, object], operation)))
        return cls(operations=tuple(parsed), schema_version=schema_version)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("degradation trace JSON must contain an object")
        return cls.from_dict(cast(Mapping[str, object], decoded))


def coerce_trace(trace: object) -> OperationTrace:
    """Accept the local trace, a mapping, or a compatible shared schema."""

    if isinstance(trace, OperationTrace):
        return trace
    if isinstance(trace, Mapping):
        return OperationTrace.from_dict(cast(Mapping[str, object], trace))
    to_dict = getattr(trace, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if not isinstance(dumped, Mapping):
            raise TypeError("trace to_dict() must return a mapping")
        return OperationTrace.from_dict(cast(Mapping[str, object], dumped))
    model_dump = getattr(trace, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if not isinstance(dumped, Mapping):
            raise TypeError("trace model_dump() must return a mapping")
        return OperationTrace.from_dict(cast(Mapping[str, object], dumped))
    raise TypeError("trace must be an OperationTrace, mapping, or compatible schema model")


def _resolve_parameters(
    name: str,
    severity: float,
    seed: int,
    supplied: Mapping[str, ParameterValue],
) -> Parameters:
    """Materialize seeded defaults so a trace records every effective value."""

    validate_operator_parameter_names(name, supplied)
    amount = validate_severity(severity)
    valid_seed = validate_seed(seed)
    generator = np.random.default_rng(valid_seed)
    if name == "identity":
        return dict(supplied)
    if name == "gaussian_blur":
        return {"radius": supplied.get("radius", 4.0 * amount)}
    if name == "defocus_blur":
        return {"radius": supplied.get("radius", max(1, int(math.ceil(5.0 * amount))))}
    if name == "motion_blur":
        return {
            "length": supplied.get("length", 2 * int(math.ceil(5.0 * amount)) + 1),
            "angle_degrees": supplied.get("angle_degrees", float(generator.uniform(0.0, 180.0))),
        }
    if name == "downsample_restore":
        return {"scale": supplied.get("scale", 1.0 - 0.8 * amount)}
    if name == "jpeg_compression":
        return {"quality": supplied.get("quality", int(round(100.0 - 95.0 * amount)))}
    if name == "poisson_gaussian_noise":
        return {
            "peak_photons": supplied.get("peak_photons", 80.0 / max(amount, 0.02)),
            "gaussian_sigma": supplied.get("gaussian_sigma", 0.035 * amount),
        }
    if name == "gamma_exposure":
        direction = -1.0 if int(generator.integers(0, 2)) == 0 else 1.0
        return {
            "gamma": supplied.get("gamma", 2.0 ** (direction * amount)),
            "gain": supplied.get("gain", 2.0 ** (-direction * 0.35 * amount)),
        }
    if name == "vignetting":
        center_x = supplied.get("center_x")
        if center_x is None:
            center_x = 0.5 + float(generator.uniform(-0.05, 0.05)) * amount
        center_y = supplied.get("center_y")
        if center_y is None:
            center_y = 0.5 + float(generator.uniform(-0.05, 0.05)) * amount
        return {
            "strength": supplied.get("strength", 0.75 * amount),
            "center_x": center_x,
            "center_y": center_y,
            "exponent": supplied.get("exponent", 2.0),
        }
    if name == "nonuniform_illumination":
        angle = supplied.get("angle_degrees")
        if angle is None:
            angle = float(generator.uniform(0.0, 180.0))
        phase = supplied.get("phase")
        if phase is None:
            phase = float(generator.uniform(0.0, 2.0 * math.pi))
        frequency = supplied.get("frequency")
        if frequency is None:
            frequency = float(generator.uniform(0.5, 1.25))
        return {
            "strength": supplied.get("strength", 0.45 * amount),
            "angle_degrees": angle,
            "phase": phase,
            "frequency": frequency,
        }
    if name == "localized_shadow":
        return {
            "strength": supplied.get("strength", 0.95 * amount),
            "center_x": supplied.get("center_x", float(generator.uniform(0.25, 0.75))),
            "center_y": supplied.get("center_y", float(generator.uniform(0.25, 0.75))),
            "radius_x": supplied.get("radius_x", 0.10 + 0.35 * amount),
            "radius_y": supplied.get("radius_y", 0.08 + 0.27 * amount),
            "angle_degrees": supplied.get("angle_degrees", float(generator.uniform(0.0, 180.0))),
            "softness": supplied.get("softness", 0.2),
        }
    if name == "glare_reflection":
        return {
            "intensity": supplied.get("intensity", min(1.0, 1.05 * amount)),
            "center_x": supplied.get("center_x", float(generator.uniform(0.2, 0.8))),
            "center_y": supplied.get("center_y", float(generator.uniform(0.2, 0.8))),
            "radius_x": supplied.get("radius_x", 0.08 + 0.42 * amount),
            "radius_y": supplied.get("radius_y", 0.06 + 0.34 * amount),
            "angle_degrees": supplied.get("angle_degrees", float(generator.uniform(0.0, 180.0))),
            "softness": supplied.get("softness", 0.15),
            "veil_r": supplied.get("veil_r", 255),
            "veil_g": supplied.get("veil_g", 252),
            "veil_b": supplied.get("veil_b", 238),
        }
    if name == "haze_veil":
        return {
            "strength": supplied.get("strength", 0.85 * amount),
            "veil_r": supplied.get("veil_r", 224),
            "veil_g": supplied.get("veil_g", 220),
            "veil_b": supplied.get("veil_b", 206),
            "radial_strength": supplied.get("radial_strength", 0.15),
        }
    if name == "color_cast":
        direction = -1.0 if int(generator.integers(0, 2)) == 0 else 1.0
        return {
            "red_gain": supplied.get("red_gain", 2.0 ** (0.75 * direction * amount)),
            "green_gain": supplied.get("green_gain", 2.0 ** (-0.15 * direction * amount)),
            "blue_gain": supplied.get("blue_gain", 2.0 ** (-0.75 * direction * amount)),
            "normalize_luminance": supplied.get("normalize_luminance", True),
        }
    raise ValueError(f"unknown degradation operator: {name!r}")


def apply_and_record(
    image: NDArray[Any],
    name: str,
    *,
    severity: float,
    seed: int,
    parameters: Mapping[str, ParameterValue] | None = None,
) -> tuple[ImageArray, DegradationStep]:
    """Apply one operation and return its fully resolved trace entry."""

    valid = validate_image(image)
    supplied: Parameters = {} if parameters is None else _normalize_parameters(parameters)
    resolved = _resolve_parameters(name, severity, seed, supplied)
    output = apply_operator(valid, name, severity=severity, seed=seed, parameters=resolved)
    return output, DegradationStep(name, severity, seed, resolved)


def apply_pipeline(
    image: NDArray[Any], operations: Iterable[object]
) -> tuple[ImageArray, OperationTrace]:
    """Apply operations in order and return a trace with resolved parameters."""

    output = validate_image(image).copy()
    recorded: list[DegradationStep] = []
    for operation in operations:
        if not isinstance(operation, DegradationStep):
            raise TypeError("operations must contain only DegradationStep values")
        output, resolved = apply_and_record(
            output,
            operation.name,
            severity=operation.severity,
            seed=operation.seed,
            parameters=operation.parameters,
        )
        recorded.append(resolved)
    return output, OperationTrace(tuple(recorded))


def replay_trace(image: NDArray[Any], trace: object) -> ImageArray:
    """Replay a validated trace in its recorded order."""

    output = validate_image(image).copy()
    parsed = coerce_trace(trace)
    for operation in parsed.operations:
        output = apply_operator(
            output,
            operation.name,
            severity=operation.severity,
            seed=operation.seed,
            parameters=operation.parameters,
        )
    return output
