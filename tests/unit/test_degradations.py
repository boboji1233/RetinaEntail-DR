from __future__ import annotations

import json

import numpy as np
import pytest

from retinaentail.degradations import (
    OPERATORS,
    DegradationStep,
    OperationTrace,
    ParameterValue,
    apply_and_record,
    apply_operator,
    apply_pipeline,
    replay_trace,
)


def _sample_image(height: int = 31, width: int = 37) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    red = (7 * xx + 3 * yy) % 256
    green = (2 * xx + 11 * yy) % 256
    blue = ((xx - width // 2) ** 2 + (yy - height // 2) ** 2) % 256
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_operator_contract_and_seed_determinism(name: str) -> None:
    image = _sample_image()
    first = apply_operator(image, name, severity=0.7, seed=1234)
    second = apply_operator(image, name, severity=0.7, seed=1234)

    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert first is not image


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_severity_zero_is_exact_identity(name: str) -> None:
    image = _sample_image()
    output = apply_operator(image, name, severity=0.0, seed=91)

    assert np.array_equal(output, image)
    assert output is not image


@pytest.mark.parametrize("severity", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_severity_fails_closed(severity: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_operator(_sample_image(), "gaussian_blur", severity=severity, seed=0)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 1), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((0, 8, 3), dtype=np.uint8),
    ],
)
def test_invalid_image_contract_is_rejected(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_operator(image, "identity", severity=0.0, seed=0)


def test_unknown_operator_and_parameters_are_rejected() -> None:
    image = _sample_image()
    with pytest.raises(ValueError, match="unknown degradation operator"):
        apply_operator(image, "unknown", severity=0.5, seed=0)
    with pytest.raises(ValueError, match="unknown operator parameter"):
        apply_operator(
            image,
            "gaussian_blur",
            severity=0.5,
            seed=0,
            parameters={"typo": 3},
        )

    with pytest.raises(ValueError, match="unknown operator parameter"):
        apply_pipeline(
            image,
            (
                DegradationStep(
                    "gaussian_blur",
                    severity=0.5,
                    seed=0,
                    parameters={"typo": 3},
                ),
            ),
        )


def test_non_finite_operator_parameter_is_rejected_even_at_zero_severity() -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_operator(
            _sample_image(),
            "gaussian_blur",
            severity=0.0,
            seed=0,
            parameters={"radius": float("nan")},
        )


def test_trace_json_round_trip_and_replay_are_exact() -> None:
    image = _sample_image()
    requested = (
        DegradationStep("gaussian_blur", severity=0.3, seed=10),
        DegradationStep("motion_blur", severity=0.4, seed=11),
        DegradationStep("poisson_gaussian_noise", severity=0.2, seed=12),
        DegradationStep("nonuniform_illumination", severity=0.3, seed=13),
    )

    output, trace = apply_pipeline(image, requested)
    restored = OperationTrace.from_json(trace.to_json())
    replayed = replay_trace(image, restored)

    assert restored == trace
    assert [step.name for step in trace.operations] == [step.name for step in requested]
    assert np.array_equal(replayed, output)
    assert all(step.parameters for step in trace.operations)
    assert json.loads(trace.to_json())["schema_version"] == "1.0.0"


def test_apply_and_record_materializes_seeded_defaults() -> None:
    image = _sample_image()
    output, step = apply_and_record(image, "motion_blur", severity=0.6, seed=42)

    assert output.shape == image.shape
    assert step.parameters["length"] == 7
    assert isinstance(step.parameters["angle_degrees"], float)
    assert np.array_equal(
        output,
        apply_operator(
            image,
            step.name,
            severity=step.severity,
            seed=step.seed,
            parameters=step.parameters,
        ),
    )


def test_new_science_s5_operators_materialize_complete_replayable_parameters() -> None:
    image = _sample_image(64, 64)
    for name in ("localized_shadow", "glare_reflection", "haze_veil", "color_cast"):
        output, step = apply_and_record(image, name, severity=0.8, seed=2026082606)
        replayed = replay_trace(image, OperationTrace((step,)))
        assert step.parameters
        assert np.array_equal(output, replayed)
        assert not np.array_equal(output, image)


def test_full_field_glare_is_an_exact_content_independent_observation() -> None:
    left = _sample_image(32, 32)
    right = np.flip(left, axis=1).copy()
    parameters: dict[str, ParameterValue] = {
        "intensity": 1.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "radius_x": 2.0,
        "radius_y": 2.0,
        "angle_degrees": 0.0,
        "softness": 0.001,
        "veil_r": 255,
        "veil_g": 252,
        "veil_b": 238,
    }
    left_observation = apply_operator(
        left, "glare_reflection", severity=1.0, seed=7, parameters=parameters
    )
    right_observation = apply_operator(
        right, "glare_reflection", severity=1.0, seed=7, parameters=parameters
    )
    assert np.array_equal(left_observation, right_observation)


def test_degradation_order_is_recorded_and_noncommutative() -> None:
    image = _sample_image(64, 64)
    blur_then_noise, first_trace = apply_pipeline(
        image,
        (
            DegradationStep("gaussian_blur", 0.7, 1),
            DegradationStep("poisson_gaussian_noise", 0.5, 2),
        ),
    )
    noise_then_blur, second_trace = apply_pipeline(
        image,
        (
            DegradationStep("poisson_gaussian_noise", 0.5, 2),
            DegradationStep("gaussian_blur", 0.7, 1),
        ),
    )
    assert [step.name for step in first_trace.operations] == [
        "gaussian_blur",
        "poisson_gaussian_noise",
    ]
    assert [step.name for step in second_trace.operations] == [
        "poisson_gaussian_noise",
        "gaussian_blur",
    ]
    assert not np.array_equal(blur_then_noise, noise_then_blur)


def test_replay_accepts_mapping_and_preserves_operation_order() -> None:
    image = _sample_image()
    trace = OperationTrace(
        (
            DegradationStep("gamma_exposure", 0.5, 7),
            DegradationStep("jpeg_compression", 0.7, 8),
        )
    )
    expected, recorded = apply_pipeline(image, trace.operations)

    assert np.array_equal(replay_trace(image, recorded.to_dict()), expected)


def test_replay_accepts_shared_schema_degradation_trace() -> None:
    from retinaentail.schemas import DegradationTrace

    image = _sample_image()
    shared_trace = DegradationTrace(
        operations=(
            {
                "name": "gaussian_blur",
                "severity": 0.25,
                "seed": 3,
                "parameters": {"radius": 1.0},
            },
        )
    )

    expected = apply_operator(image, "gaussian_blur", severity=0.25, seed=3)
    replayed = replay_trace(image, shared_trace)

    assert np.array_equal(replayed, expected)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "1.0.0"},
        {"schema_version": "999", "operations": []},
        {"schema_version": "1.0.0", "operations": [], "unexpected": True},
        {
            "schema_version": "1.0.0",
            "operations": [
                {
                    "name": "gaussian_blur",
                    "severity": float("nan"),
                    "seed": 0,
                    "parameters": {},
                }
            ],
        },
    ],
)
def test_invalid_trace_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        OperationTrace.from_dict(payload)


def test_trace_json_rejects_non_finite_parameter() -> None:
    with pytest.raises(ValueError, match="finite"):
        DegradationStep("gaussian_blur", severity=0.2, seed=0, parameters={"radius": float("nan")})


@pytest.mark.parametrize("parameters", [{"typo": 1.0}, {"radius": 65.0}])
def test_degradation_step_rejects_unknown_or_out_of_range_parameters(
    parameters: dict[str, ParameterValue],
) -> None:
    with pytest.raises(ValueError):
        DegradationStep(
            "gaussian_blur",
            severity=0.2,
            seed=0,
            parameters=parameters,
        )
