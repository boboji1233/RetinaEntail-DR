from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from retinaentail.data.lesion_catalog import (
    apply_transform,
    extract_fixed_patch,
    patch_geometry,
)


@given(
    width=st.integers(min_value=8, max_value=1024),
    height=st.integers(min_value=8, max_value=1024),
    x=st.floats(min_value=0, max_value=7, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0, max_value=7, allow_nan=False, allow_infinity=False),
)
def test_patch_transform_round_trip_is_exact_and_size_is_fixed(
    width: int, height: int, x: float, y: float
) -> None:
    bounded_x = min(x, width - 1)
    bounded_y = min(y, height - 1)
    geometry = patch_geometry(
        image_size=(width, height), centroid=(bounded_x, bounded_y), patch_size=(32, 32)
    )
    point = apply_transform(geometry.original_to_patch, bounded_x, bounded_y)
    restored = apply_transform(geometry.patch_to_original, *point)
    patch = extract_fixed_patch(np.zeros((height, width), dtype=np.bool_), geometry)

    assert max(abs(restored[0] - bounded_x), abs(restored[1] - bounded_y)) <= 1.0
    assert patch.shape == (32, 32)
