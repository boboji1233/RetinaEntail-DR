"""Procedural lesion-present/lesion-absent twins for offline smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from retinaentail.schemas import LesionRegion

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ToyFundusTwin:
    """Synthetic fundus-like patch with a controlled round lesion intervention."""

    lesion_present: UInt8Image
    lesion_absent: UInt8Image
    region: LesionRegion
    lesion_mask: NDArray[np.bool_]


def _require_plain_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def generate_toy_fundus(
    *, size: int = 96, lesion_radius: int = 4, seed: int = 20260823
) -> ToyFundusTwin:
    """Generate deterministic RGB uint8 twins without medical data or network access."""

    image_size = _require_plain_int(size, "size", minimum=48)
    radius = _require_plain_int(lesion_radius, "lesion_radius", minimum=2)
    valid_seed = _require_plain_int(seed, "seed", minimum=0)
    if radius * 6 >= image_size:
        raise ValueError("lesion_radius is too large for the requested toy image")

    generator = np.random.default_rng(valid_seed)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    normalized_x = (xx - (image_size - 1) / 2.0) / (image_size * 0.48)
    normalized_y = (yy - (image_size - 1) / 2.0) / (image_size * 0.48)
    radial = np.sqrt(normalized_x**2 + normalized_y**2)
    inside = radial <= 1.0

    texture = generator.normal(0.0, 1.5, size=(image_size, image_size))
    red = 135.0 + 35.0 * (1.0 - np.clip(radial, 0.0, 1.0)) + texture
    green = 48.0 + 24.0 * (1.0 - np.clip(radial, 0.0, 1.0)) + 0.45 * texture
    blue = 34.0 + 14.0 * (1.0 - np.clip(radial, 0.0, 1.0)) + 0.30 * texture
    background = np.stack((red, green, blue), axis=-1)
    background[~inside] = np.array([5.0, 4.0, 6.0])
    absent = np.clip(np.rint(background), 0.0, 255.0).astype(np.uint8)

    # The vessels are deliberately stylized. They exercise shared anatomical context,
    # not medical realism and not a learned retinal prior.
    canvas = Image.fromarray(absent, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    origin = (int(image_size * 0.28), int(image_size * 0.52))
    vessel_color = (78, 24, 27)
    vessel_paths = (
        (origin, (int(image_size * 0.48), int(image_size * 0.30)), (image_size - 8, 10)),
        (origin, (int(image_size * 0.52), int(image_size * 0.44)), (image_size - 5, 34)),
        (origin, (int(image_size * 0.50), int(image_size * 0.64)), (image_size - 7, 78)),
        (origin, (int(image_size * 0.43), int(image_size * 0.75)), (60, image_size - 5)),
    )
    for path in vessel_paths:
        draw.line(path, fill=vessel_color, width=max(1, image_size // 48), joint="curve")
    draw.ellipse(
        (
            origin[0] - image_size // 16,
            origin[1] - image_size // 16,
            origin[0] + image_size // 16,
            origin[1] + image_size // 16,
        ),
        fill=(183, 116, 67),
    )
    absent = np.asarray(canvas, dtype=np.uint8).copy()

    center_x = int(round(image_size * 0.68))
    center_y = int(round(image_size * 0.52))
    lesion_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    if not bool(np.all(inside[lesion_mask])):
        raise ValueError("toy lesion falls outside the synthetic field of view")
    present = absent.copy()
    present[lesion_mask] = np.array([62, 13, 19], dtype=np.uint8)
    region = LesionRegion(
        x=center_x - radius,
        y=center_y - radius,
        width=2 * radius + 1,
        height=2 * radius + 1,
        region_id="toy-lesion-1",
    )
    return ToyFundusTwin(present, absent, region, lesion_mask)


def toy_lesion_predicate(image: NDArray[np.uint8], region: LesionRegion) -> bool:
    """Return the explicit toy predicate used only to verify smoke witnesses."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("toy predicate expects an RGB uint8 HxWx3 image")
    if not isinstance(region, LesionRegion):
        raise TypeError("region must be a LesionRegion")
    x0, y0 = region.x, region.y
    x1, y1 = x0 + region.width, y0 + region.height
    if x1 > image.shape[1] or y1 > image.shape[0]:
        raise ValueError("region falls outside the image")
    patch = image[y0:y1, x0:x1]
    toy_dark_red = (patch[..., 0] <= 85) & (patch[..., 1] <= 30) & (patch[..., 2] <= 40)
    return float(np.mean(toy_dark_red)) >= 0.30


__all__ = ["ToyFundusTwin", "generate_toy_fundus", "toy_lesion_predicate"]
