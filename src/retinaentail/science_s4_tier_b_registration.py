"""Science S4 Tier-B Paired Retina registration and overlap audit v4.

Only the 96 frozen CALIBRATION primary subject-eye pairs are decoded.  The
module has no DDR, lesion-label, or locked-test input and cannot start twins or
any downstream experiment.
"""

from __future__ import annotations

import argparse
import importlib
import io
import math
import platform
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageOps

import retinaentail.science_s4_recovery_contract as contract_v3
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "4.0.0"
EXPERIMENT_ID: Final = "S4_TIER_B_REGISTRATION_V4"
READY_STATUS: Final = "READY_FOR_AI_VISUAL_QC"
BLOCKED_STATUS: Final = "BLOCKED_PREREQUISITES"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
EXPECTED_BINDINGS: Final = {
    "registration_prompt",
    "contract_config",
    "contract_report",
    "contract_audit",
    "contract_artifact_manifest",
    "image_manifest",
    "primary_pair_contract",
}


class S4TierBRegistrationError(RuntimeError):
    """Raised when the frozen registration audit cannot be replayed exactly."""


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """Registration audit disposition and immutable generated artifact hashes."""

    status: str
    output_dir: Path
    accepted_count: int
    rejected_count: int
    artifact_hashes: dict[str, str]


@dataclass(slots=True)
class _PairOutcome:
    record: dict[str, JSONValue]
    panels: tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]


def _cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise S4TierBRegistrationError(
            "frozen opencv-python-headless runtime is unavailable"
        ) from error


def _mapping(value: object, name: str) -> dict[str, object]:
    try:
        return contract_v3._mapping(value, name)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _sequence(value: object, name: str) -> tuple[object, ...]:
    try:
        return contract_v3._sequence(value, name)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _string(value: object, name: str) -> str:
    try:
        return contract_v3._string(value, name)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    try:
        return contract_v3._integer(value, name, minimum=minimum)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    try:
        return contract_v3._number(value, name, minimum=minimum)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _signed_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S4TierBRegistrationError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise S4TierBRegistrationError(f"{name} must be finite")
    return rendered


def _sha256_file(path: Path) -> str:
    return contract_v3._sha256_file(path)  # noqa: SLF001


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        return contract_v3._load_json(path, name)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _load_config(path: Path) -> tuple[Path, dict[str, object]]:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4TierBRegistrationError("config path must remain in the project root") from error
    try:
        config = contract_v3._load_config(resolved)  # noqa: SLF001
        contract_v3._verify_baseline(config.get("expected_baseline_commit"))  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("status") != "EXECUTABLE_REGISTRATION_AUDIT"
    ):
        raise S4TierBRegistrationError("unsupported S4 Tier-B registration config")
    return resolved, config


def _verify_binding(binding: object, name: str) -> tuple[Path, str]:
    try:
        return contract_v3._verify_binding(binding, name)  # noqa: SLF001
    except contract_v3.S4RecoveryContractError as error:
        raise S4TierBRegistrationError(str(error)) from error


def _verify_upstream(config: Mapping[str, object]) -> dict[str, tuple[Path, str]]:
    bindings = _mapping(config.get("bindings"), "bindings")
    if set(bindings) != EXPECTED_BINDINGS:
        raise S4TierBRegistrationError("v4 binding inventory changed")
    verified = {
        name: _verify_binding(binding, f"bindings.{name}") for name, binding in bindings.items()
    }
    audit = _load_json(verified["contract_audit"][0], "contract v3 audit")
    if (
        audit.get("status") != contract_v3.READY_STATUS
        or audit.get("contract_frozen") is not True
        or _mapping(audit.get("downstream_authorization"), "v3 authorization").get(
            "s4_recovery_image_audit"
        )
        is not True
    ):
        raise S4TierBRegistrationError("contract v3 does not authorize the image audit")
    manifest = _load_json(verified["contract_artifact_manifest"][0], "v3 artifact manifest")
    entries = _sequence(manifest.get("files"), "v3 artifact files")
    manifest_map = {
        _string(_mapping(item, "v3 artifact").get("relative_path"), "v3 artifact path"): _mapping(
            item, "v3 artifact"
        )
        for item in entries
    }
    for key, relative in (
        ("contract_audit", "audit.json"),
        ("image_manifest", "paired-retina-image-manifest.json"),
        ("primary_pair_contract", "tier-b-primary-pair-contract.json"),
    ):
        entry = manifest_map.get(relative)
        if entry is None or entry.get("sha256") != verified[key][1]:
            raise S4TierBRegistrationError(f"v3 artifact manifest binding drift: {relative}")
    return verified


def _verify_runtime(config: Mapping[str, object]) -> dict[str, JSONValue]:
    cv2 = _cv2()
    runtime = _mapping(config.get("runtime"), "runtime")
    import PIL

    actual = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "opencv_python_headless": cv2.__version__,
    }
    expected_cv = _string(runtime.get("opencv_python_headless"), "OpenCV runtime")
    expected_cv_import = expected_cv.rsplit(".", 1)[0]
    if (
        actual["python"] != _string(runtime.get("python"), "Python runtime")
        or actual["pillow"] != _string(runtime.get("pillow"), "Pillow runtime")
        or actual["numpy"] != _string(runtime.get("numpy"), "NumPy runtime")
        or actual["opencv_python_headless"] != expected_cv_import
    ):
        raise S4TierBRegistrationError(f"runtime drift: {actual}")
    cv2.setNumThreads(_integer(runtime.get("opencv_threads"), "OpenCV threads", minimum=1))
    cv2.ocl.setUseOpenCL(bool(runtime.get("opencl")))
    cv2.setRNGSeed(_integer(runtime.get("deterministic_seed"), "deterministic seed"))
    if cv2.getNumThreads() != 1 or cv2.ocl.useOpenCL():
        raise S4TierBRegistrationError("OpenCV deterministic runtime settings were not applied")
    return {
        **actual,
        "opencv_wheel_version": expected_cv,
        "opencv_threads": cv2.getNumThreads(),
        "opencl": bool(cv2.ocl.useOpenCL()),
        "platform": platform.platform(),
    }


def _snapshot_root(config: Mapping[str, object], contract_root: Path) -> Path:
    input_config = _mapping(config.get("input"), "input")
    expected_name = _string(input_config.get("external_root_name"), "external root name")
    if not contract_root.is_absolute():
        raise S4TierBRegistrationError("contract root must be absolute")
    root = contract_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or root.name != expected_name:
        raise S4TierBRegistrationError("contract root is not the frozen v3 external directory")
    relative = Path(_string(input_config.get("snapshot_relative_path"), "snapshot path"))
    snapshot = (root / relative).resolve(strict=True)
    try:
        snapshot.relative_to(root)
    except ValueError as error:
        raise S4TierBRegistrationError("snapshot escapes the contract root") from error
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise S4TierBRegistrationError("frozen Paired Retina snapshot is unavailable")
    return snapshot


def _primary_inputs(
    config: Mapping[str, object],
    verified: Mapping[str, tuple[Path, str]],
    snapshot: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], int]:
    image_manifest = _load_json(verified["image_manifest"][0], "image manifest")
    pair_contract = _load_json(verified["primary_pair_contract"][0], "primary pair contract")
    scope = _mapping(config.get("scope"), "scope")
    input_config = _mapping(config.get("input"), "input")
    if (
        image_manifest.get("repository_revision") != scope.get("input_revision")
        or image_manifest.get("image_count") != input_config.get("expected_image_count")
        or image_manifest.get("total_bytes_hashed") != input_config.get("expected_image_bytes")
        or pair_contract.get("split") != "CALIBRATION_ONLY"
        or pair_contract.get("primary_unit_count") != scope.get("primary_pair_count")
        or pair_contract.get("supplemental_pair_count")
        != input_config.get("expected_supplemental_pair_count")
    ):
        raise S4TierBRegistrationError("v3 image or primary-pair contract drift")
    if (
        scope.get("supplemental_pair_decodes_allowed") is not False
        or scope.get("lesion_label_reads_allowed") is not False
        or scope.get("ddr_reads_allowed") is not False
        or scope.get("locked_test_root_reads_allowed") is not False
    ):
        raise S4TierBRegistrationError("v4 input boundary changed")
    files: dict[str, dict[str, object]] = {}
    for raw in _sequence(image_manifest.get("files"), "image files"):
        entry = _mapping(raw, "image entry")
        relative = _string(entry.get("relative_path"), "image path")
        if relative in files:
            raise S4TierBRegistrationError("duplicate image-manifest path")
        files[relative] = entry
    pairs = [
        _mapping(raw, "primary pair")
        for raw in _sequence(pair_contract.get("primary_pairs"), "primary pairs")
    ]
    expected_pairs = _integer(scope.get("primary_pair_count"), "primary pair count")
    if len(pairs) != expected_pairs:
        raise S4TierBRegistrationError("primary-pair count drift")
    image_uses: Counter[str] = Counter()
    bytes_hashed = 0
    for pair in pairs:
        for field, expected_device in (
            ("visucam_image_path", "visucam"),
            ("eyer_image_path", "eyer"),
        ):
            relative = _string(pair.get(field), field)
            found = files.get(relative)
            if found is None or found.get("device") != expected_device:
                raise S4TierBRegistrationError("primary pair references an invalid device image")
            path = snapshot.joinpath(*PurePosixPath(relative).parts)
            expected_size = _integer(found.get("byte_size"), f"{relative} size", minimum=1)
            expected_hash = _string(found.get("sha256"), f"{relative} SHA-256")
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != expected_size
                or _sha256_file(path) != expected_hash
            ):
                raise S4TierBRegistrationError(f"primary image hash drift: {relative}")
            image_uses[relative] += 1
            bytes_hashed += expected_size
    if any(count != 1 for count in image_uses.values()) or len(image_uses) != expected_pairs * 2:
        raise S4TierBRegistrationError("duplicate image reuse in the frozen primary set")
    return pairs, files, bytes_hashed


def _decode_canvas(path: Path, side: int) -> NDArray[np.uint8]:
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise S4TierBRegistrationError("decoded image has invalid dimensions")
        scale = side / max(width, height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(resized, ((side - resized_width) // 2, (side - resized_height) // 2))
    return np.asarray(canvas, dtype=np.uint8).copy()


def _fov_mask(image: NDArray[np.uint8], threshold: int, kernel_size: int) -> NDArray[np.bool_]:
    cv2 = _cv2()
    raw = (np.max(image, axis=2) > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return np.zeros(raw.shape, dtype=np.bool_)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return np.asarray(filled, dtype=np.bool_)


def normalized_mutual_information(
    fixed: NDArray[np.floating[Any]],
    moving: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
    *,
    bins: int,
) -> float:
    """Return frozen 2*MI/(H(X)+H(Y)) over the supplied overlap mask."""

    x = np.asarray(fixed[mask], dtype=np.float64)
    y = np.asarray(moving[mask], dtype=np.float64)
    if x.size == 0 or y.size != x.size:
        raise S4TierBRegistrationError("mutual-information overlap is empty")
    joint, _, _ = np.histogram2d(x, y, bins=bins, range=((0.0, 1.0), (0.0, 1.0)))
    probability = joint / float(joint.sum())
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    nz = probability > 0.0
    product = px[:, None] * py[None, :]
    mi = float(np.sum(probability[nz] * np.log(probability[nz] / product[nz])))
    hx = float(-np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    hy = float(-np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    denominator = hx + hy
    if denominator <= 0.0:
        raise S4TierBRegistrationError("mutual-information entropy is degenerate")
    return 2.0 * mi / denominator


def photometric_quantile_affine(
    fixed: NDArray[np.floating[Any]],
    moving: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
    *,
    lower: float,
    upper: float,
    scale_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
) -> tuple[NDArray[np.float64], tuple[float, ...], tuple[float, ...]]:
    """Fit the frozen per-channel overlap quantile affine mapping."""

    mapped = np.asarray(moving, dtype=np.float64).copy()
    scales: list[float] = []
    offsets: list[float] = []
    for channel in range(3):
        fixed_q = np.quantile(fixed[..., channel][mask], (lower, upper))
        moving_q = np.quantile(moving[..., channel][mask], (lower, upper))
        moving_range = float(moving_q[1] - moving_q[0])
        if moving_range <= 1e-12:
            raise S4TierBRegistrationError("photometric quantile range is degenerate")
        scale = float((fixed_q[1] - fixed_q[0]) / moving_range)
        offset = float(fixed_q[0] - scale * moving_q[0])
        scale = min(max(scale, scale_bounds[0]), scale_bounds[1])
        offset = min(max(offset, offset_bounds[0]), offset_bounds[1])
        mapped[..., channel] = np.clip(scale * mapped[..., channel] + offset, 0.0, 1.0)
        scales.append(scale)
        offsets.append(offset)
    return mapped, tuple(scales), tuple(offsets)


def transform_statistics(matrix: NDArray[np.floating[Any]], side: int) -> dict[str, float]:
    """Return deterministic affine bounds diagnostics."""

    affine = np.asarray(matrix, dtype=np.float64)
    linear = affine[:2, :2]
    determinant = float(np.linalg.det(linear))
    u, singular, vt = np.linalg.svd(linear)
    polar_rotation = u @ vt
    angle = math.degrees(math.atan2(polar_rotation[1, 0], polar_rotation[0, 0]))
    minimum_scale = float(np.min(singular))
    maximum_scale = float(np.max(singular))
    anisotropy = math.inf if minimum_scale <= 0.0 else maximum_scale / minimum_scale
    translation = float(np.linalg.norm(affine[:2, 2]) / math.hypot(side, side))
    return {
        "determinant": determinant,
        "minimum_scale": minimum_scale,
        "maximum_scale": maximum_scale,
        "anisotropy": anisotropy,
        "rotation_degrees": float(angle),
        "translation_fraction_of_diagonal": translation,
    }


def _transform_reasons(stats: Mapping[str, float], bounds: Mapping[str, object]) -> list[str]:
    if stats["determinant"] <= 0.0:
        return ["REFLECTION_DETECTED"]
    outside = (
        stats["minimum_scale"] < _number(bounds.get("minimum_scale"), "minimum scale")
        or stats["maximum_scale"] > _number(bounds.get("maximum_scale"), "maximum scale")
        or stats["anisotropy"] > _number(bounds.get("maximum_anisotropy"), "anisotropy")
        or abs(stats["rotation_degrees"])
        > _number(bounds.get("maximum_absolute_rotation_degrees"), "rotation bound")
        or stats["translation_fraction_of_diagonal"]
        > _number(bounds.get("maximum_translation_fraction_of_diagonal"), "translation bound")
    )
    return ["TRANSFORM_OUT_OF_BOUNDS"] if outside else []


def _linear_to_srgb(linear: NDArray[np.floating[Any]]) -> NDArray[np.uint8]:
    values = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055)
    return np.asarray(np.rint(np.clip(encoded, 0.0, 1.0) * 255.0), dtype=np.uint8)


def _visual_panels(
    fixed_rgb: NDArray[np.uint8],
    moving_rgb: NDArray[np.uint8],
    fixed_linear: NDArray[np.float64],
    registered_linear: NDArray[np.float64] | None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    registered = (
        np.asarray(moving_rgb, dtype=np.float64) / 255.0
        if registered_linear is None
        else registered_linear
    )
    registered_rgb = _linear_to_srgb(registered)
    overlay = np.asarray(
        np.rint(0.5 * fixed_rgb.astype(np.float64) + 0.5 * registered_rgb.astype(np.float64)),
        dtype=np.uint8,
    )
    difference = _linear_to_srgb(np.clip(np.abs(fixed_linear - registered) * 4.0, 0.0, 1.0))
    return fixed_rgb, moving_rgb, overlay, difference


def _match_similarity(
    fixed_rgb: NDArray[np.uint8],
    moving_rgb: NDArray[np.uint8],
    fixed_mask: NDArray[np.bool_],
    moving_mask: NDArray[np.bool_],
    config: Mapping[str, object],
) -> tuple[NDArray[np.float64] | None, dict[str, JSONValue], list[str]]:
    cv2 = _cv2()
    features = _mapping(config.get("features"), "features")
    ransac = _mapping(config.get("ransac"), "RANSAC")
    sift = cv2.SIFT_create(nfeatures=_integer(features.get("maximum_features"), "features"))
    fixed_green = fixed_rgb[..., 1]
    moving_green = moving_rgb[..., 1]
    fixed_keypoints, fixed_descriptors = sift.detectAndCompute(
        fixed_green, fixed_mask.astype(np.uint8) * 255
    )
    moving_keypoints, moving_descriptors = sift.detectAndCompute(
        moving_green, moving_mask.astype(np.uint8) * 255
    )
    diagnostics: dict[str, JSONValue] = {
        "fixed_keypoints": len(fixed_keypoints),
        "moving_keypoints": len(moving_keypoints),
        "mutual_good_matches": 0,
        "ransac_inliers": 0,
        "ransac_inlier_fraction": None,
    }
    if fixed_descriptors is None or moving_descriptors is None:
        return None, diagnostics, ["INSUFFICIENT_FEATURE_MATCHES"]
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward_raw = matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2)
    reverse_raw = matcher.knnMatch(fixed_descriptors, moving_descriptors, k=2)
    ratio = _number(features.get("lowe_ratio"), "Lowe ratio")
    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse_raw
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    mutual = [
        pair[0]
        for pair in forward_raw
        if len(pair) == 2
        and pair[0].distance < ratio * pair[1].distance
        and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    diagnostics["mutual_good_matches"] = len(mutual)
    if len(mutual) < _integer(features.get("minimum_good_matches"), "minimum matches"):
        return None, diagnostics, ["INSUFFICIENT_FEATURE_MATCHES"]
    moving_coordinates = cast(
        list[tuple[float, float]], [moving_keypoints[item.queryIdx].pt for item in mutual]
    )
    fixed_coordinates = cast(
        list[tuple[float, float]], [fixed_keypoints[item.trainIdx].pt for item in mutual]
    )
    moving_points = np.asarray(moving_coordinates, dtype=np.float32)
    fixed_points = np.asarray(fixed_coordinates, dtype=np.float32)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        moving_points,
        fixed_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=_number(ransac.get("reprojection_threshold_pixels"), "RANSAC"),
        maxIters=_integer(ransac.get("max_iterations"), "RANSAC iterations"),
        confidence=_number(ransac.get("confidence"), "RANSAC confidence"),
        refineIters=_integer(ransac.get("refine_iterations"), "RANSAC refine"),
    )
    if matrix is None or inlier_mask is None:
        return None, diagnostics, ["INSUFFICIENT_RANSAC_INLIERS"]
    inliers = int(np.count_nonzero(inlier_mask))
    fraction = inliers / len(mutual)
    diagnostics["ransac_inliers"] = inliers
    diagnostics["ransac_inlier_fraction"] = fraction
    if inliers < _integer(ransac.get("minimum_inliers"), "minimum inliers") or fraction < _number(
        ransac.get("minimum_inlier_fraction"), "inlier fraction"
    ):
        return None, diagnostics, ["INSUFFICIENT_RANSAC_INLIERS"]
    return np.asarray(matrix, dtype=np.float64), diagnostics, []


def _warp(
    image: NDArray[np.generic], matrix: NDArray[np.floating[Any]], side: int, *, mask: bool
) -> NDArray[Any]:
    cv2 = _cv2()
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LANCZOS4
    result = cv2.warpAffine(
        image,
        np.asarray(matrix, dtype=np.float64),
        (side, side),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.asarray(result)


def _register_pair(
    *,
    pair: Mapping[str, object],
    files: Mapping[str, Mapping[str, object]],
    snapshot: Path,
    config: Mapping[str, object],
) -> _PairOutcome:
    registration = _mapping(config.get("registration"), "registration")
    side = _integer(registration.get("grid_long_side"), "registration side", minimum=1)
    fixed_relative = _string(pair.get("visucam_image_path"), "Visucam path")
    moving_relative = _string(pair.get("eyer_image_path"), "Eyer path")
    fixed_path = snapshot.joinpath(*PurePosixPath(fixed_relative).parts)
    moving_path = snapshot.joinpath(*PurePosixPath(moving_relative).parts)
    fixed_rgb = _decode_canvas(fixed_path, side)
    moving_rgb = _decode_canvas(moving_path, side)
    fixed_linear = np.asarray(
        contract_v3.srgb_to_linear(fixed_rgb.astype(np.float64) / 255.0), dtype=np.float64
    )
    moving_linear = np.asarray(
        contract_v3.srgb_to_linear(moving_rgb.astype(np.float64) / 255.0), dtype=np.float64
    )
    fov = _mapping(registration.get("fov"), "FOV")
    threshold = _integer(fov.get("rgb_max_threshold_8bit"), "FOV threshold")
    kernel = _integer(fov.get("close_kernel_diameter"), "FOV kernel", minimum=1)
    fixed_mask = _fov_mask(fixed_rgb, threshold, kernel)
    moving_mask = _fov_mask(moving_rgb, threshold, kernel)
    fixed_fraction = float(np.mean(fixed_mask))
    moving_fraction = float(np.mean(moving_mask))
    minimum_fov = _number(fov.get("minimum_area_fraction"), "minimum FOV")
    maximum_fov = _number(fov.get("maximum_area_fraction"), "maximum FOV")
    reasons: list[str] = []
    if not (
        minimum_fov <= fixed_fraction <= maximum_fov
        and minimum_fov <= moving_fraction <= maximum_fov
    ):
        reasons.append("INVALID_FOV_FRACTION")
    diagnostics: dict[str, JSONValue] = {
        "fixed_keypoints": None,
        "moving_keypoints": None,
        "mutual_good_matches": None,
        "ransac_inliers": None,
        "ransac_inlier_fraction": None,
    }
    initial: NDArray[np.float64] | None = None
    final: NDArray[np.float64] | None = None
    registered_linear: NDArray[np.float64] | None = None
    metrics: dict[str, JSONValue] = {
        "fov_overlap_fraction": None,
        "overlap_pixels": None,
        "normalized_mutual_information": None,
        "linear_green_nrmse": None,
        "ecc_correlation": None,
    }
    photometric: dict[str, JSONValue] = {"scales": None, "offsets": None}
    residual: JSONValue = None
    stats: dict[str, JSONValue] | None = None
    if not reasons:
        initial, diagnostics, match_reasons = _match_similarity(
            fixed_rgb, moving_rgb, fixed_mask, moving_mask, registration
        )
        reasons.extend(match_reasons)
    if initial is not None and not reasons:
        bounds = _mapping(registration.get("transform_bounds"), "transform bounds")
        initial_stats = transform_statistics(initial, side)
        reasons.extend(_transform_reasons(initial_stats, bounds))
    if initial is not None and not reasons:
        cv2 = _cv2()
        ecc = _mapping(registration.get("ecc"), "ECC")
        initial_linear = np.asarray(
            _warp(moving_linear, initial, side, mask=False), dtype=np.float64
        )
        initial_mask = np.asarray(
            _warp(moving_mask.astype(np.uint8), initial, side, mask=True), dtype=bool
        )
        ecc_mask = (fixed_mask & initial_mask).astype(np.uint8) * 255
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            _integer(ecc.get("max_iterations"), "ECC iterations"),
            _number(ecc.get("epsilon"), "ECC epsilon"),
        )
        try:
            correlation, returned = cv2.findTransformECC(
                np.asarray(fixed_linear[..., 1], dtype=np.float32),
                np.asarray(initial_linear[..., 1], dtype=np.float32),
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=ecc_mask,
                gaussFiltSize=_integer(ecc.get("gaussian_filter_size"), "ECC Gaussian"),
            )
            returned3 = np.eye(3, dtype=np.float64)
            returned3[:2] = returned
            initial3 = np.eye(3, dtype=np.float64)
            initial3[:2] = initial
            final = np.asarray((np.linalg.inv(returned3) @ initial3)[:2], dtype=np.float64)
            metrics["ecc_correlation"] = float(correlation)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError, cv2.error):
            reasons.append("ECC_REFINEMENT_FAILED")
    if final is not None and not reasons:
        bounds = _mapping(registration.get("transform_bounds"), "transform bounds")
        final_stats = transform_statistics(final, side)
        stats = cast(dict[str, JSONValue], final_stats)
        reasons.extend(_transform_reasons(final_stats, bounds))
    if final is not None and not reasons:
        quality = _mapping(registration.get("quality"), "quality")
        warped_mask = np.asarray(
            _warp(moving_mask.astype(np.uint8), final, side, mask=True), dtype=bool
        )
        overlap = fixed_mask & warped_mask
        overlap_pixels = int(np.count_nonzero(overlap))
        denominator = min(int(np.count_nonzero(fixed_mask)), int(np.count_nonzero(warped_mask)))
        overlap_fraction = overlap_pixels / denominator if denominator else 0.0
        metrics["overlap_pixels"] = overlap_pixels
        metrics["fov_overlap_fraction"] = overlap_fraction
        if overlap_pixels < _integer(
            quality.get("minimum_overlap_pixels"), "overlap pixels"
        ) or overlap_fraction < _number(
            quality.get("minimum_fov_overlap_fraction"), "overlap fraction"
        ):
            reasons.append("FOV_OVERLAP_TOO_SMALL")
        else:
            warped_linear = np.asarray(
                _warp(moving_linear, final, side, mask=False), dtype=np.float64
            )
            nmi = normalized_mutual_information(
                fixed_linear[..., 1],
                warped_linear[..., 1],
                overlap,
                bins=_integer(quality.get("normalized_mutual_information_bins"), "NMI bins"),
            )
            metrics["normalized_mutual_information"] = nmi
            photo = _mapping(config.get("photometric_alignment"), "photometric alignment")
            registered_linear, scales, offsets = photometric_quantile_affine(
                fixed_linear,
                warped_linear,
                overlap,
                lower=_number(photo.get("lower_quantile"), "lower quantile"),
                upper=_number(photo.get("upper_quantile"), "upper quantile"),
                scale_bounds=(
                    _number(photo.get("minimum_scale"), "photo minimum scale"),
                    _number(photo.get("maximum_scale"), "photo maximum scale"),
                ),
                offset_bounds=(
                    _signed_number(photo.get("minimum_offset"), "photo minimum offset"),
                    _number(photo.get("maximum_offset"), "photo maximum offset"),
                ),
            )
            photometric = {"scales": list(scales), "offsets": list(offsets)}
            fixed_green = fixed_linear[..., 1][overlap]
            mapped_green = registered_linear[..., 1][overlap]
            q10, q90 = np.quantile(fixed_green, (0.10, 0.90))
            green_range = float(q90 - q10)
            nrmse = (
                math.inf
                if green_range <= 1e-12
                else float(np.sqrt(np.mean((fixed_green - mapped_green) ** 2)) / green_range)
            )
            if not math.isfinite(nrmse):
                reasons.append("LINEAR_GREEN_NRMSE_TOO_HIGH")
            else:
                metrics["linear_green_nrmse"] = nrmse
                if nmi < _number(
                    quality.get("minimum_normalized_mutual_information"), "minimum NMI"
                ):
                    reasons.append("MUTUAL_INFORMATION_TOO_LOW")
                if nrmse > _number(quality.get("maximum_linear_green_nrmse"), "maximum NRMSE"):
                    reasons.append("LINEAR_GREEN_NRMSE_TOO_HIGH")
            if not reasons:
                witness = _mapping(config.get("overlap_witness"), "overlap witness")
                quantile = _number(witness.get("unit_residual_quantile"), "residual quantile")
                residual = [
                    float(
                        np.quantile(
                            np.abs(fixed_linear[..., channel] - registered_linear[..., channel])[
                                overlap
                            ],
                            quantile,
                        )
                    )
                    for channel in range(3)
                ]
    ordered_reasons = list(dict.fromkeys(reasons))
    status = "ACCEPTED" if not ordered_reasons and residual is not None else "REJECTED"
    if registered_linear is None and final is not None:
        registered_linear = np.asarray(
            _warp(moving_linear, final, side, mask=False), dtype=np.float64
        )
    record_base: dict[str, JSONValue] = {
        "pair_id": _string(pair.get("pair_id"), "pair ID"),
        "subject_id": _string(pair.get("subject_id"), "subject ID"),
        "eye_laterality": _string(pair.get("eye_laterality"), "eye laterality"),
        "center": _string(pair.get("center"), "center"),
        "split": "CALIBRATION_ONLY",
        "selection_sha256": _string(pair.get("selection_sha256"), "selection SHA-256"),
        "fixed_image": {
            "relative_path": fixed_relative,
            "sha256": _string(files[fixed_relative].get("sha256"), "fixed SHA-256"),
        },
        "moving_image": {
            "relative_path": moving_relative,
            "sha256": _string(files[moving_relative].get("sha256"), "moving SHA-256"),
        },
        "decode_count": 2,
        "fov": {"fixed_fraction": fixed_fraction, "moving_fraction": moving_fraction},
        "feature_diagnostics": diagnostics,
        "initial_similarity_matrix": None if initial is None else initial.tolist(),
        "final_affine_matrix": None if final is None else final.tolist(),
        "transform_statistics": stats,
        "metrics": metrics,
        "photometric_alignment": photometric,
        "residual_q995_linear_rgb_absolute": residual,
        "status": status,
        "rejection_reasons": cast(list[JSONValue], ordered_reasons),
    }
    record_hash = sha256_bytes(canonical_json_bytes(record_base))
    record_base["record_sha256"] = record_hash
    return _PairOutcome(
        record=record_base,
        panels=_visual_panels(fixed_rgb, moving_rgb, fixed_linear, registered_linear),
    )


def _contact_sheet(outcomes: Sequence[_PairOutcome], *, title: str, tile_width: int) -> bytes:
    panel_names = ("fixed Visucam", "moving Eyer", "registered overlay", "linear RGB diff x4")
    image_height = tile_width
    label_height = 42
    header_height = 54
    width = tile_width * 4
    height = header_height + max(1, len(outcomes)) * (image_height + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill="black")
    for column, name in enumerate(panel_names):
        draw.text((column * tile_width + 8, 30), name, fill="black")
    if not outcomes:
        draw.text((10, header_height + 10), "No pairs in this stratum.", fill="black")
    for row, outcome in enumerate(outcomes):
        y = header_height + row * (image_height + label_height)
        for column, panel in enumerate(outcome.panels):
            rendered = Image.fromarray(panel, mode="RGB").resize(
                (tile_width, image_height), Image.Resampling.LANCZOS
            )
            sheet.paste(rendered, (column * tile_width, y))
        reasons = cast(list[str], outcome.record["rejection_reasons"])
        label = f"{outcome.record['pair_id']} | {outcome.record['status']}" + (
            f" | {','.join(reasons)}" if reasons else ""
        )
        draw.text((8, y + image_height + 5), label[:190], fill="black")
    stream = io.BytesIO()
    sheet.save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _csv_bytes(outcomes: Sequence[_PairOutcome]) -> bytes:
    lines = [
        "pair_id,subject_id,eye_laterality,center,status,primary_rejection_reason,"
        "all_rejection_reasons,fov_overlap_fraction,overlap_pixels,nmi,linear_green_nrmse,"
        "record_sha256"
    ]
    for outcome in outcomes:
        record = outcome.record
        metrics = cast(dict[str, JSONValue], record["metrics"])
        reasons = cast(list[str], record["rejection_reasons"])
        fields = [
            record["pair_id"],
            record["subject_id"],
            record["eye_laterality"],
            record["center"],
            record["status"],
            reasons[0] if reasons else "",
            "|".join(reasons),
            metrics["fov_overlap_fraction"],
            metrics["overlap_pixels"],
            metrics["normalized_mutual_information"],
            metrics["linear_green_nrmse"],
            record["record_sha256"],
        ]
        rendered = ["" if value is None else str(value) for value in fields]
        lines.append(",".join(f'"{value.replace(chr(34), chr(34) * 2)}"' for value in rendered))
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_registration_audit(
    *, config_path: Path, contract_root: Path, output_dir: Path
) -> RegistrationResult:
    """Run the frozen 96-pair real-image registration audit once."""

    resolved_config, config = _load_config(config_path)
    frozen_output = (PROJECT_ROOT / _string(config.get("artifact_root"), "artifact root")).resolve()
    if output_dir.resolve() != frozen_output:
        raise S4TierBRegistrationError("output directory changed")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite S4 Tier-B output: {output_dir}")
    verified = _verify_upstream(config)
    runtime = _verify_runtime(config)
    snapshot = _snapshot_root(config, contract_root)
    pairs, files, bytes_hashed = _primary_inputs(config, verified, snapshot)
    outcomes = [
        _register_pair(pair=pair, files=files, snapshot=snapshot, config=config) for pair in pairs
    ]
    accepted = [outcome for outcome in outcomes if outcome.record["status"] == "ACCEPTED"]
    rejected = [outcome for outcome in outcomes if outcome.record["status"] == "REJECTED"]
    witness_config = _mapping(config.get("overlap_witness"), "overlap witness")
    minimum_units = _integer(
        witness_config.get("minimum_accepted_primary_units"), "minimum accepted units"
    )
    alpha = _number(witness_config.get("calibration_alpha"), "calibration alpha")
    radius: tuple[float, ...] | None = None
    if len(accepted) >= minimum_units:
        residuals = [
            cast(list[float], outcome.record["residual_q995_linear_rgb_absolute"])
            for outcome in accepted
        ]
        radius = contract_v3.conformal_residual_radius(
            residuals, alpha=alpha, minimum_units=minimum_units
        )
    status = READY_STATUS if radius is not None else BLOCKED_STATUS
    rejection_counts = Counter(
        reason
        for outcome in rejected
        for reason in cast(list[str], outcome.record["rejection_reasons"])
    )
    records_payload: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "records": [outcome.record for outcome in outcomes],
    }
    witness_payload: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "FIT" if radius is not None else "NOT_FIT",
        "accepted_primary_units": len(accepted),
        "minimum_accepted_primary_units": minimum_units,
        "alpha": alpha,
        "rank": math.ceil((len(accepted) + 1) * (1.0 - alpha)) if accepted else None,
        "unit_residual_quantile": _number(
            witness_config.get("unit_residual_quantile"), "unit residual quantile"
        ),
        "channel_radius": None if radius is None else list(radius),
        "set_overlap_rule": _string(witness_config.get("set_overlap_rule"), "set overlap rule"),
        "accepted_record_hashes": cast(
            list[JSONValue], [outcome.record["record_sha256"] for outcome in accepted]
        ),
    }
    contact = _mapping(config.get("contact_sheets"), "contact sheets")
    accepted_sample = sorted(accepted, key=lambda item: str(item.record["selection_sha256"]))[
        : _integer(contact.get("accepted_sample_count"), "accepted sample count")
    ]
    rejected_by_reason: dict[str, list[_PairOutcome]] = defaultdict(list)
    for outcome in rejected:
        reasons = cast(list[str], outcome.record["rejection_reasons"])
        rejected_by_reason[reasons[0]].append(outcome)
    rejected_sample: list[_PairOutcome] = []
    maximum_rejected = _integer(
        contact.get("rejected_sample_count_maximum"), "rejected sample maximum"
    )
    for reason in sorted(rejected_by_reason):
        rejected_sample.extend(
            sorted(
                rejected_by_reason[reason], key=lambda item: str(item.record["selection_sha256"])
            )[:1]
        )
    remaining = sorted(
        [item for item in rejected if item not in rejected_sample],
        key=lambda item: str(item.record["selection_sha256"]),
    )
    rejected_sample.extend(remaining[: max(0, maximum_rejected - len(rejected_sample))])
    tile_width = _integer(contact.get("tile_width"), "contact-sheet tile", minimum=1)
    accepted_sheet = _contact_sheet(
        accepted_sample,
        title="Science S4 Tier-B accepted registration QC (calibration evidence only)",
        tile_width=tile_width,
    )
    rejected_sheet = _contact_sheet(
        rejected_sample[:maximum_rejected],
        title="Science S4 Tier-B rejected registration QC (calibration evidence only)",
        tile_width=tile_width,
    )
    audit: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "config_sha256": _sha256_file(resolved_config),
        "runtime": runtime,
        "upstream_replay": {
            name: {"path": str(path), "sha256": digest} for name, (path, digest) in verified.items()
        },
        "scope": {
            "split": "CALIBRATION_ONLY",
            "primary_pair_count": len(pairs),
            "supplemental_pair_decodes": 0,
            "lesion_label_reads": 0,
            "ddr_reads": 0,
            "locked_test_root_reads": 0,
            "locked_test_image_decodes": 0,
        },
        "registration": {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "contact_sheet_accepted_count": len(accepted_sample),
            "contact_sheet_rejected_count": min(len(rejected_sample), maximum_rejected),
        },
        "overlap_witness": witness_payload,
        "io_audit": {
            "primary_file_hash_reads": len(pairs) * 2,
            "primary_bytes_hashed": bytes_hashed,
            "primary_image_decodes": len(pairs) * 2,
            "supplemental_image_decodes": 0,
            "ddr_reads": 0,
            "locked_test_root_reads": 0,
        },
        "downstream_authorization": {
            "ai_visual_qc": status == READY_STATUS,
            "s4_recovery_twin_execution": False,
            "s5_degradation_calibration": False,
            "s6_verify": False,
            "s7_lite": False,
            "enhancer_training": False,
            "locked_test_evaluation": False,
            "fcr_estimation": False,
            "paper_performance_experiment": False,
        },
        "limitations": ["PROVIDER_ASSERTION_WITHOUT_IMAGE_TIMESTAMP"],
        "evidence_boundary": [
            "This is calibration-pair registration evidence, not FCR or clinical performance.",
            "Visual QC is required before the S4 recovery twin can be authorized.",
            "No supplemental pair, DDR file, lesion label, or locked-test input was read.",
        ],
    }
    payloads: dict[str, bytes] = {
        "registration-records.json": canonical_json_bytes(records_payload),
        "accepted-rejected.csv": _csv_bytes(outcomes),
        "tier-b-overlap-witness.json": canonical_json_bytes(witness_payload),
        "qc-accepted-registration-contact-sheet.png": accepted_sheet,
        "qc-rejected-registration-contact-sheet.png": rejected_sheet,
        "audit.json": canonical_json_bytes(audit),
    }
    artifact_manifest: dict[str, JSONValue] = {
        "files": [
            {
                "relative_path": name,
                "byte_size": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, payload in sorted(payloads.items())
        ]
    }
    manifest_bytes = canonical_json_bytes(artifact_manifest)
    for name, payload in payloads.items():
        write_bytes_fail_safe(output_dir / name, payload)
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", manifest_bytes)
    hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    hashes["artifact-manifest.json"] = sha256_bytes(manifest_bytes)
    return RegistrationResult(
        status=status,
        output_dir=output_dir,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        artifact_hashes=hashes,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded Tier-B registration CLI."""

    parser = argparse.ArgumentParser(description="Science S4 Tier-B registration audit v4.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded registration audit."""

    args = build_parser().parse_args(argv)
    try:
        result = run_registration_audit(
            config_path=cast(Path, args.config),
            contract_root=cast(Path, args.contract_root),
            output_dir=cast(Path, args.output_dir),
        )
    except (
        FileExistsError,
        OSError,
        S4TierBRegistrationError,
        contract_v3.S4RecoveryContractError,
    ) as error:
        print(f"Science S4 Tier-B registration fail-closed: {error}", file=sys.stderr)
        return 1
    print(f"Science S4 Tier-B registration status: {result.status}")
    print(f"accepted={result.accepted_count} rejected={result.rejected_count}")
    for name, digest in result.artifact_hashes.items():
        print(f"{name}: {digest}")
    return 0 if result.status == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_STATUS",
    "READY_STATUS",
    "RegistrationResult",
    "S4TierBRegistrationError",
    "build_parser",
    "main",
    "normalized_mutual_information",
    "photometric_quantile_affine",
    "run_registration_audit",
    "transform_statistics",
]
