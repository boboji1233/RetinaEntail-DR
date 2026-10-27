"""Deterministic, CPU-only image degradation operators.

The public operators deliberately accept and return only ``uint8`` RGB NumPy
arrays.  They never resize the returned image.  Every stochastic choice is
derived from the explicit seed so that a recorded operation can be replayed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from io import BytesIO
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

ImageArray: TypeAlias = NDArray[np.uint8]
ParameterValue: TypeAlias = str | int | float | bool
Parameters: TypeAlias = dict[str, ParameterValue]

_MAX_SEED = 2**64 - 1


def validate_image(image: object) -> ImageArray:
    """Validate the degradation image contract without silently coercing it."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.dtype != np.uint8:
        raise TypeError("image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image height and width must be positive")
    return cast(ImageArray, image)


def validate_severity(severity: object) -> float:
    """Return a finite severity in the closed interval [0, 1]."""

    if isinstance(severity, bool | np.bool_) or not isinstance(
        severity, int | float | np.integer | np.floating
    ):
        raise TypeError("severity must be a real number")
    value = float(severity)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("severity must be finite and in [0, 1]")
    return value


def validate_seed(seed: object) -> int:
    """Return a portable NumPy seed, rejecting booleans and negative values."""

    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError("seed must be an integer")
    value = int(seed)
    if not 0 <= value <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")
    return value


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(validate_seed(seed))


def _as_pillow(image: ImageArray) -> Image.Image:
    return Image.fromarray(image)


def _from_pillow(image: Image.Image) -> ImageArray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _finite_float(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _positive_int(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return result


def _byte_channel(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= 255:
        raise ValueError(f"{name} must be in [0, 255]")
    return result


def _require_odd(name: str, value: int) -> int:
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd")
    return value


def _check_parameter_names(parameters: Mapping[str, object], allowed: set[str]) -> None:
    unknown = sorted(set(parameters).difference(allowed))
    if unknown:
        raise ValueError(f"unknown operator parameter(s): {', '.join(unknown)}")


def identity(image: NDArray[Any], *, severity: float = 0.0, seed: int = 0) -> ImageArray:
    """Return an independent copy of the input image."""

    valid = validate_image(image)
    validate_severity(severity)
    validate_seed(seed)
    return valid.copy()


def gaussian_blur(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    radius: float | None = None,
) -> ImageArray:
    """Apply an isotropic Gaussian blur using Pillow's CPU implementation."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    validate_seed(seed)
    resolved_radius = 4.0 * amount if radius is None else radius
    resolved_radius = _finite_float("radius", resolved_radius, minimum=0.0, maximum=64.0)
    if amount == 0.0 or resolved_radius == 0.0:
        return valid.copy()
    return _from_pillow(_as_pillow(valid).filter(ImageFilter.GaussianBlur(resolved_radius)))


def _convolve(image: ImageArray, kernel: NDArray[np.float64]) -> ImageArray:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("convolution kernel must be square")
    size = kernel.shape[0]
    if size < 1 or size % 2 == 0:
        raise ValueError("convolution kernel size must be a positive odd number")
    total = float(kernel.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("convolution kernel must have a positive finite sum")
    normalized = kernel / total
    radius = size // 2
    height, width, _ = image.shape
    padded = np.pad(
        image.astype(np.float32), ((radius, radius), (radius, radius), (0, 0)), mode="edge"
    )
    output = np.zeros((height, width, 3), dtype=np.float32)
    for row, column in np.argwhere(normalized != 0.0):
        output += padded[row : row + height, column : column + width] * np.float32(
            normalized[row, column]
        )
    return np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8)


def _disk_kernel(radius: int) -> NDArray[np.float64]:
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx * xx + yy * yy) <= (radius + 0.25) ** 2).astype(np.float64)


def defocus_blur(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    radius: int | None = None,
) -> ImageArray:
    """Approximate circular defocus with a normalized disk point-spread function."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    validate_seed(seed)
    resolved_radius = max(1, int(math.ceil(5.0 * amount))) if radius is None else radius
    resolved_radius = _positive_int("radius", resolved_radius, maximum=32)
    if amount == 0.0:
        return valid.copy()
    return _convolve(valid, _disk_kernel(resolved_radius))


def _motion_kernel(length: int, angle_degrees: float) -> NDArray[np.float64]:
    radius = length // 2
    kernel = np.zeros((length, length), dtype=np.float64)
    angle = math.radians(angle_degrees)
    samples = max(length * 4, 1)
    offsets = np.linspace(-radius, radius, samples)
    rows = np.rint(radius + offsets * math.sin(angle)).astype(int)
    columns = np.rint(radius + offsets * math.cos(angle)).astype(int)
    np.add.at(kernel, (rows, columns), 1.0)
    return kernel


def motion_blur(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    length: int | None = None,
    angle_degrees: float | None = None,
) -> ImageArray:
    """Apply a seeded linear motion point-spread function."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    resolved_length = 2 * int(math.ceil(5.0 * amount)) + 1 if length is None else length
    resolved_length = _require_odd("length", _positive_int("length", resolved_length, maximum=65))
    resolved_angle = (
        float(generator.uniform(0.0, 180.0)) if angle_degrees is None else angle_degrees
    )
    resolved_angle = _finite_float("angle_degrees", resolved_angle, minimum=-360.0, maximum=360.0)
    if amount == 0.0:
        return valid.copy()
    return _convolve(valid, _motion_kernel(resolved_length, resolved_angle))


def downsample_restore(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    scale: float | None = None,
) -> ImageArray:
    """Downsample then restore the original dimensions with fixed resamplers."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    validate_seed(seed)
    resolved_scale = 1.0 - 0.8 * amount if scale is None else scale
    resolved_scale = _finite_float("scale", resolved_scale, minimum=0.01, maximum=1.0)
    if amount == 0.0:
        return valid.copy()
    height, width, _ = valid.shape
    down_width = max(1, min(width, int(round(width * resolved_scale))))
    down_height = max(1, min(height, int(round(height * resolved_scale))))
    if (down_width, down_height) == (width, height):
        return valid.copy()
    source = _as_pillow(valid)
    reduced = source.resize((down_width, down_height), resample=Image.Resampling.BILINEAR)
    restored = reduced.resize((width, height), resample=Image.Resampling.BICUBIC)
    return _from_pillow(restored)


def jpeg_compression(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    quality: int | None = None,
) -> ImageArray:
    """Round-trip an image through an in-memory JPEG stream."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    validate_seed(seed)
    resolved_quality = int(round(100.0 - 95.0 * amount)) if quality is None else quality
    resolved_quality = _positive_int("quality", resolved_quality, maximum=100)
    if amount == 0.0:
        return valid.copy()
    stream = BytesIO()
    _as_pillow(valid).save(
        stream,
        format="JPEG",
        quality=resolved_quality,
        optimize=False,
        progressive=False,
        subsampling=0,
    )
    stream.seek(0)
    with Image.open(stream) as decoded:
        return _from_pillow(decoded)


def poisson_gaussian_noise(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    peak_photons: float | None = None,
    gaussian_sigma: float | None = None,
) -> ImageArray:
    """Apply independent seeded Poisson shot noise and additive Gaussian noise."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    resolved_peak = 80.0 / max(amount, 0.02) if peak_photons is None else peak_photons
    resolved_sigma = 0.035 * amount if gaussian_sigma is None else gaussian_sigma
    resolved_peak = _finite_float("peak_photons", resolved_peak, minimum=0.001, maximum=1_000_000.0)
    resolved_sigma = _finite_float("gaussian_sigma", resolved_sigma, minimum=0.0, maximum=1.0)
    if amount == 0.0:
        return valid.copy()
    normalized = valid.astype(np.float64) / 255.0
    noisy = generator.poisson(normalized * resolved_peak) / resolved_peak
    if resolved_sigma > 0.0:
        noisy = noisy + generator.normal(0.0, resolved_sigma, size=normalized.shape)
    return np.clip(np.rint(noisy * 255.0), 0.0, 255.0).astype(np.uint8)


def gamma_exposure(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    gamma: float | None = None,
    gain: float | None = None,
) -> ImageArray:
    """Apply a seeded brightening or darkening gamma/exposure transform."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    direction = -1.0 if int(generator.integers(0, 2)) == 0 else 1.0
    resolved_gamma = 2.0 ** (direction * amount) if gamma is None else gamma
    resolved_gain = 2.0 ** (-direction * 0.35 * amount) if gain is None else gain
    resolved_gamma = _finite_float("gamma", resolved_gamma, minimum=0.05, maximum=20.0)
    resolved_gain = _finite_float("gain", resolved_gain, minimum=0.01, maximum=20.0)
    if amount == 0.0:
        return valid.copy()
    normalized = valid.astype(np.float64) / 255.0
    transformed = resolved_gain * np.power(normalized, resolved_gamma)
    return np.clip(np.rint(transformed * 255.0), 0.0, 255.0).astype(np.uint8)


def vignetting(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    strength: float | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    exponent: float = 2.0,
) -> ImageArray:
    """Darken the periphery using a seeded, slightly off-centre radial field."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    resolved_strength = 0.75 * amount if strength is None else strength
    resolved_center_x = (
        0.5 + float(generator.uniform(-0.05, 0.05)) * amount if center_x is None else center_x
    )
    resolved_center_y = (
        0.5 + float(generator.uniform(-0.05, 0.05)) * amount if center_y is None else center_y
    )
    resolved_strength = _finite_float("strength", resolved_strength, minimum=0.0, maximum=1.0)
    resolved_center_x = _finite_float("center_x", resolved_center_x, minimum=0.0, maximum=1.0)
    resolved_center_y = _finite_float("center_y", resolved_center_y, minimum=0.0, maximum=1.0)
    resolved_exponent = _finite_float("exponent", exponent, minimum=0.1, maximum=10.0)
    if amount == 0.0:
        return valid.copy()
    height, width, _ = valid.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    distance = np.sqrt((x - resolved_center_x) ** 2 + (y - resolved_center_y) ** 2)
    max_distance = math.sqrt(
        max(resolved_center_x, 1.0 - resolved_center_x) ** 2
        + max(resolved_center_y, 1.0 - resolved_center_y) ** 2
    )
    field = 1.0 - resolved_strength * np.power(distance / max_distance, resolved_exponent)
    output = valid.astype(np.float64) * field[..., None]
    return cast(ImageArray, np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8))


def nonuniform_illumination(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    strength: float | None = None,
    angle_degrees: float | None = None,
    phase: float | None = None,
    frequency: float | None = None,
) -> ImageArray:
    """Modulate brightness with a seeded, low-frequency illumination field."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    resolved_strength = 0.45 * amount if strength is None else strength
    resolved_angle = (
        float(generator.uniform(0.0, 180.0)) if angle_degrees is None else angle_degrees
    )
    resolved_phase = float(generator.uniform(0.0, 2.0 * math.pi)) if phase is None else phase
    resolved_frequency = float(generator.uniform(0.5, 1.25)) if frequency is None else frequency
    resolved_strength = _finite_float("strength", resolved_strength, minimum=0.0, maximum=1.0)
    resolved_angle = _finite_float("angle_degrees", resolved_angle, minimum=-360.0, maximum=360.0)
    resolved_phase = _finite_float(
        "phase", resolved_phase, minimum=-100.0 * math.pi, maximum=100.0 * math.pi
    )
    resolved_frequency = _finite_float("frequency", resolved_frequency, minimum=0.01, maximum=10.0)
    if amount == 0.0:
        return valid.copy()
    height, width, _ = valid.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1) - 0.5
    y = yy / max(height - 1, 1) - 0.5
    angle = math.radians(resolved_angle)
    coordinate = x * math.cos(angle) + y * math.sin(angle)
    field = 1.0 + resolved_strength * np.sin(
        2.0 * math.pi * resolved_frequency * coordinate + resolved_phase
    )
    output = valid.astype(np.float64) * field[..., None]
    return cast(ImageArray, np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8))


def _elliptical_field(
    height: int,
    width: int,
    *,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    angle_degrees: float,
    softness: float,
) -> NDArray[np.float64]:
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1) - center_x
    y = yy / max(height - 1, 1) - center_y
    angle = math.radians(angle_degrees)
    rotated_x = x * math.cos(angle) + y * math.sin(angle)
    rotated_y = -x * math.sin(angle) + y * math.cos(angle)
    distance = np.sqrt((rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2)
    transition = np.clip((1.0 + softness - distance) / (2.0 * softness), 0.0, 1.0)
    return cast(NDArray[np.float64], transition * transition * (3.0 - 2.0 * transition))


def localized_shadow(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    strength: float | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    radius_x: float | None = None,
    radius_y: float | None = None,
    angle_degrees: float | None = None,
    softness: float = 0.2,
) -> ImageArray:
    """Apply a soft, seeded elliptical illumination shadow."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    resolved_strength = 0.95 * amount if strength is None else strength
    resolved_center_x = float(generator.uniform(0.25, 0.75)) if center_x is None else center_x
    resolved_center_y = float(generator.uniform(0.25, 0.75)) if center_y is None else center_y
    resolved_radius_x = 0.10 + 0.35 * amount if radius_x is None else radius_x
    resolved_radius_y = 0.08 + 0.27 * amount if radius_y is None else radius_y
    resolved_angle = (
        float(generator.uniform(0.0, 180.0)) if angle_degrees is None else angle_degrees
    )
    resolved_strength = _finite_float("strength", resolved_strength, minimum=0.0, maximum=1.0)
    resolved_center_x = _finite_float("center_x", resolved_center_x, minimum=0.0, maximum=1.0)
    resolved_center_y = _finite_float("center_y", resolved_center_y, minimum=0.0, maximum=1.0)
    resolved_radius_x = _finite_float("radius_x", resolved_radius_x, minimum=0.01, maximum=2.0)
    resolved_radius_y = _finite_float("radius_y", resolved_radius_y, minimum=0.01, maximum=2.0)
    resolved_angle = _finite_float("angle_degrees", resolved_angle, minimum=-360.0, maximum=360.0)
    resolved_softness = _finite_float("softness", softness, minimum=0.001, maximum=1.0)
    if amount == 0.0 or resolved_strength == 0.0:
        return valid.copy()
    field = _elliptical_field(
        valid.shape[0],
        valid.shape[1],
        center_x=resolved_center_x,
        center_y=resolved_center_y,
        radius_x=resolved_radius_x,
        radius_y=resolved_radius_y,
        angle_degrees=resolved_angle,
        softness=resolved_softness,
    )
    output = valid.astype(np.float64) * (1.0 - resolved_strength * field[..., None])
    return cast(ImageArray, np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8))


def glare_reflection(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    intensity: float | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    radius_x: float | None = None,
    radius_y: float | None = None,
    angle_degrees: float | None = None,
    softness: float = 0.15,
    veil_r: int = 255,
    veil_g: int = 252,
    veil_b: int = 238,
) -> ImageArray:
    """Blend a soft elliptical specular reflection toward an explicit veil color."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    resolved_intensity = min(1.0, 1.05 * amount) if intensity is None else intensity
    resolved_center_x = float(generator.uniform(0.2, 0.8)) if center_x is None else center_x
    resolved_center_y = float(generator.uniform(0.2, 0.8)) if center_y is None else center_y
    resolved_radius_x = 0.08 + 0.42 * amount if radius_x is None else radius_x
    resolved_radius_y = 0.06 + 0.34 * amount if radius_y is None else radius_y
    resolved_angle = (
        float(generator.uniform(0.0, 180.0)) if angle_degrees is None else angle_degrees
    )
    resolved_intensity = _finite_float("intensity", resolved_intensity, minimum=0.0, maximum=1.0)
    resolved_center_x = _finite_float("center_x", resolved_center_x, minimum=0.0, maximum=1.0)
    resolved_center_y = _finite_float("center_y", resolved_center_y, minimum=0.0, maximum=1.0)
    resolved_radius_x = _finite_float("radius_x", resolved_radius_x, minimum=0.01, maximum=2.0)
    resolved_radius_y = _finite_float("radius_y", resolved_radius_y, minimum=0.01, maximum=2.0)
    resolved_angle = _finite_float("angle_degrees", resolved_angle, minimum=-360.0, maximum=360.0)
    resolved_softness = _finite_float("softness", softness, minimum=0.001, maximum=1.0)
    veil = np.asarray(
        [
            _byte_channel("veil_r", veil_r),
            _byte_channel("veil_g", veil_g),
            _byte_channel("veil_b", veil_b),
        ],
        dtype=np.float64,
    )
    if amount == 0.0 or resolved_intensity == 0.0:
        return valid.copy()
    field = _elliptical_field(
        valid.shape[0],
        valid.shape[1],
        center_x=resolved_center_x,
        center_y=resolved_center_y,
        radius_x=resolved_radius_x,
        radius_y=resolved_radius_y,
        angle_degrees=resolved_angle,
        softness=resolved_softness,
    )
    alpha = resolved_intensity * field[..., None]
    output = valid.astype(np.float64) * (1.0 - alpha) + veil * alpha
    return cast(ImageArray, np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8))


def haze_veil(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    strength: float | None = None,
    veil_r: int = 224,
    veil_g: int = 220,
    veil_b: int = 206,
    radial_strength: float = 0.15,
) -> ImageArray:
    """Apply a cataract-like veiling luminance with a weak radial field."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    validate_seed(seed)
    resolved_strength = 0.85 * amount if strength is None else strength
    resolved_strength = _finite_float("strength", resolved_strength, minimum=0.0, maximum=1.0)
    resolved_radial = _finite_float("radial_strength", radial_strength, minimum=0.0, maximum=1.0)
    veil = np.asarray(
        [
            _byte_channel("veil_r", veil_r),
            _byte_channel("veil_g", veil_g),
            _byte_channel("veil_b", veil_b),
        ],
        dtype=np.float64,
    )
    if amount == 0.0 or resolved_strength == 0.0:
        return valid.copy()
    height, width, _ = valid.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1) - 0.5
    y = yy / max(height - 1, 1) - 0.5
    radial = np.clip(1.0 + resolved_radial * (1.0 - 2.0 * np.sqrt(x * x + y * y)), 0.0, 2.0)
    alpha = np.clip(resolved_strength * radial, 0.0, 1.0)[..., None]
    output = valid.astype(np.float64) * (1.0 - alpha) + veil * alpha
    return cast(ImageArray, np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8))


def color_cast(
    image: NDArray[Any],
    *,
    severity: float,
    seed: int,
    red_gain: float | None = None,
    green_gain: float | None = None,
    blue_gain: float | None = None,
    normalize_luminance: bool = True,
) -> ImageArray:
    """Apply explicit channel gains with optional mean-luminance preservation."""

    valid = validate_image(image)
    amount = validate_severity(severity)
    generator = _rng(seed)
    direction = -1.0 if int(generator.integers(0, 2)) == 0 else 1.0
    resolved_red = 2.0 ** (0.75 * direction * amount) if red_gain is None else red_gain
    resolved_green = 2.0 ** (-0.15 * direction * amount) if green_gain is None else green_gain
    resolved_blue = 2.0 ** (-0.75 * direction * amount) if blue_gain is None else blue_gain
    resolved_red = _finite_float("red_gain", resolved_red, minimum=0.1, maximum=4.0)
    resolved_green = _finite_float("green_gain", resolved_green, minimum=0.1, maximum=4.0)
    resolved_blue = _finite_float("blue_gain", resolved_blue, minimum=0.1, maximum=4.0)
    if not isinstance(normalize_luminance, bool):
        raise TypeError("normalize_luminance must be a bool")
    if amount == 0.0:
        return valid.copy()
    gains = np.asarray([resolved_red, resolved_green, resolved_blue], dtype=np.float64)
    output = valid.astype(np.float64) * gains
    if normalize_luminance:
        weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        before = float(np.mean(valid.astype(np.float64) @ weights))
        after = float(np.mean(output @ weights))
        if after > 0.0:
            output *= before / after
    return cast(ImageArray, np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8))


Operator: TypeAlias = Callable[..., ImageArray]

OPERATORS: dict[str, Operator] = {
    "identity": identity,
    "gaussian_blur": gaussian_blur,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "downsample_restore": downsample_restore,
    "jpeg_compression": jpeg_compression,
    "poisson_gaussian_noise": poisson_gaussian_noise,
    "gamma_exposure": gamma_exposure,
    "vignetting": vignetting,
    "nonuniform_illumination": nonuniform_illumination,
    "localized_shadow": localized_shadow,
    "glare_reflection": glare_reflection,
    "haze_veil": haze_veil,
    "color_cast": color_cast,
}


_ALLOWED_PARAMETERS: dict[str, set[str]] = {
    "identity": set(),
    "gaussian_blur": {"radius"},
    "defocus_blur": {"radius"},
    "motion_blur": {"length", "angle_degrees"},
    "downsample_restore": {"scale"},
    "jpeg_compression": {"quality"},
    "poisson_gaussian_noise": {"peak_photons", "gaussian_sigma"},
    "gamma_exposure": {"gamma", "gain"},
    "vignetting": {"strength", "center_x", "center_y", "exponent"},
    "nonuniform_illumination": {"strength", "angle_degrees", "phase", "frequency"},
    "localized_shadow": {
        "strength",
        "center_x",
        "center_y",
        "radius_x",
        "radius_y",
        "angle_degrees",
        "softness",
    },
    "glare_reflection": {
        "intensity",
        "center_x",
        "center_y",
        "radius_x",
        "radius_y",
        "angle_degrees",
        "softness",
        "veil_r",
        "veil_g",
        "veil_b",
    },
    "haze_veil": {"strength", "veil_r", "veil_g", "veil_b", "radial_strength"},
    "color_cast": {"red_gain", "green_gain", "blue_gain", "normalize_luminance"},
}


def validate_operator_parameter_names(
    name: str,
    parameters: Mapping[str, object],
) -> None:
    """Reject unknown operators and parameter names before defaults are resolved."""

    if name not in OPERATORS:
        raise ValueError(f"unknown degradation operator: {name!r}")
    _check_parameter_names(parameters, _ALLOWED_PARAMETERS[name])


def validate_operator_parameters(name: str, parameters: Mapping[str, object]) -> None:
    """Validate an operator's complete explicit parameter mapping without degradation."""

    supplied = dict(parameters)
    validate_operator_parameter_names(name, supplied)
    validation_image = np.zeros((1, 1, 3), dtype=np.uint8)
    OPERATORS[name](validation_image, severity=0.0, seed=0, **supplied)


def validate_resolved_operator_parameters(name: str, parameters: Mapping[str, object]) -> None:
    """Require every effective parameter to be present, then validate its value."""

    validate_operator_parameter_names(name, parameters)
    missing = sorted(_ALLOWED_PARAMETERS[name].difference(parameters))
    if missing:
        raise ValueError(f"missing resolved operator parameter(s): {', '.join(missing)}")
    validate_operator_parameters(name, parameters)


def apply_operator(
    image: NDArray[Any],
    name: str,
    *,
    severity: float,
    seed: int,
    parameters: Mapping[str, ParameterValue] | None = None,
) -> ImageArray:
    """Apply one registered operator and reject unknown names or parameters."""

    supplied = {} if parameters is None else dict(parameters)
    validate_operator_parameter_names(name, supplied)
    return OPERATORS[name](image, severity=severity, seed=seed, **supplied)
