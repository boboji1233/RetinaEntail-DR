from __future__ import annotations

import numpy as np
import pytest

from retinaentail.degradations import OPERATORS, apply_operator

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies


@settings(max_examples=40, deadline=None)  # type: ignore[untyped-decorator]
@given(  # type: ignore[untyped-decorator]
    name=st.sampled_from(sorted(OPERATORS)),
    height=st.integers(min_value=1, max_value=40),
    width=st.integers(min_value=1, max_value=40),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    image_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_operator_preserves_shape_range_and_is_seed_deterministic(
    name: str,
    height: int,
    width: int,
    severity: float,
    seed: int,
    image_seed: int,
) -> None:
    image = np.random.default_rng(image_seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )

    first = apply_operator(image, name, severity=severity, seed=seed)
    second = apply_operator(image, name, severity=severity, seed=seed)

    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert int(first.min()) >= 0
    assert int(first.max()) <= 255
    assert np.array_equal(first, second)


@settings(max_examples=30, deadline=None)  # type: ignore[untyped-decorator]
@given(  # type: ignore[untyped-decorator]
    name=st.sampled_from(sorted(OPERATORS)),
    height=st.integers(min_value=1, max_value=48),
    width=st.integers(min_value=1, max_value=48),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_zero_severity_is_exact_identity_property(
    name: str, height: int, width: int, seed: int
) -> None:
    image = np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    output = apply_operator(image, name, severity=0.0, seed=seed)

    assert np.array_equal(output, image)
    assert output is not image
