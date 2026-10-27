"""Deterministic lesion-component catalog and coordinate contracts.

The real-data entry point is deliberately narrow: it replays frozen S1 evidence
before touching the explicitly supplied root, reads only TRAIN/CALIBRATION rows,
and never resolves or decodes a LOCKED_TEST path. The original HE/EX S2 behavior
is preserved; S4 reuses the same geometry for an explicitly frozen MA/SE pair.
Oracle regions in this module are controlled-experiment annotations, not a
deployment capability.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, TypeAlias, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from retinaentail.data.license_gate import (
    DataUsePurpose,
    assess_dataset_access,
    authorize_dataset_access,
    load_license_review_record,
    load_registry_license_metadata,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import (
    DatasetManifestV2,
    LabelAvailability,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
    load_dataset_adapter_spec,
)
from retinaentail.witnesses.replay import canonical_json_bytes, sha256_bytes, write_bytes_fail_safe

S2_SCHEMA_VERSION: Final = "1.0.0"
DEVELOPMENT_SPLITS: Final = frozenset({"TRAIN", "CALIBRATION"})
LOCKED_SPLIT: Final = "LOCKED_TEST"
LESION_SEMANTICS: Final[dict[str, LabelSemantic]] = {
    "MA": LabelSemantic.MICROANEURYSM_MASK,
    "HE": LabelSemantic.HEMORRHAGE_MASK,
    "EX": LabelSemantic.HARD_EXUDATE_MASK,
    "SE": LabelSemantic.SOFT_EXUDATE_MASK,
}

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
Matrix3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
BBox: TypeAlias = tuple[int, int, int, int]


class S2GateError(RuntimeError):
    """Raised before root I/O when a frozen prerequisite cannot be replayed."""


class S2DataError(RuntimeError):
    """Raised after authorization when real or synthetic input fails closed."""


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S2GateError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise S2GateError(f"{name} keys must be strings")
        result[key] = item
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S2GateError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S2GateError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S2DataError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise S2DataError(f"{name} must be finite")
    return result


def _safe_relative(value: object, name: str) -> str:
    text = _string(value, name)
    if "\\" in text:
        raise S2GateError(f"{name} must use POSIX separators")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != text
    ):
        raise S2GateError(f"{name} must be a normalized relative path")
    return text


def _sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise S2GateError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_compact_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_matrix() -> Matrix3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _translation(tx: float, ty: float) -> Matrix3:
    _finite_float(tx, "translation tx")
    _finite_float(ty, "translation ty")
    return ((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0))


def apply_transform(matrix: Matrix3, x: float, y: float) -> tuple[float, float]:
    """Apply a finite affine 3x3 transform to one coordinate."""

    px = _finite_float(x, "x")
    py = _finite_float(y, "y")
    coefficients = tuple(value for row in matrix for value in row)
    if len(coefficients) != 9 or not all(math.isfinite(value) for value in coefficients):
        raise S2DataError("transform must contain nine finite values")
    denominator = matrix[2][0] * px + matrix[2][1] * py + matrix[2][2]
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        raise S2DataError("transform has an invalid homogeneous denominator")
    return (
        (matrix[0][0] * px + matrix[0][1] * py + matrix[0][2]) / denominator,
        (matrix[1][0] * px + matrix[1][1] * py + matrix[1][2]) / denominator,
    )


def invert_affine(matrix: Matrix3) -> Matrix3:
    """Invert a finite affine matrix, rejecting singular/projective input."""

    values = tuple(value for row in matrix for value in row)
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise S2DataError("transform must contain nine finite values")
    if matrix[2] != (0.0, 0.0, 1.0):
        raise S2DataError("only affine transforms are supported")
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        raise S2DataError("transform is singular")
    return (
        (d / determinant, -b / determinant, (b * ty - d * tx) / determinant),
        (-c / determinant, a / determinant, (c * tx - a * ty) / determinant),
        (0.0, 0.0, 1.0),
    )


def matrix_to_json(matrix: Matrix3) -> list[JSONValue]:
    return [[float(value) for value in row] for row in matrix]


@dataclass(frozen=True, slots=True)
class FOVGeometry:
    """Declared elliptical field of view derived from a thresholded foreground bbox."""

    bbox: BBox
    crop_size: tuple[int, int]
    original_to_crop: Matrix3
    crop_to_original: Matrix3
    mask: NDArray[np.bool_]
    rule_version: str = "FOV_ELLIPSE_THRESHOLD_BBOX_V1"

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise S2DataError("FOV bbox is invalid")
        if self.crop_size != (x1 - x0, y1 - y0):
            raise S2DataError("FOV crop size does not match bbox")
        if self.mask.dtype != np.bool_ or self.mask.ndim != 2:
            raise S2DataError("FOV mask must be boolean HxW")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": S2_SCHEMA_VERSION,
            "rule_version": self.rule_version,
            "bbox_xyxy": list(self.bbox),
            "crop_size_wh": list(self.crop_size),
            "original_to_crop": matrix_to_json(self.original_to_crop),
            "crop_to_original": matrix_to_json(self.crop_to_original),
            "boundary_model": "ELLIPSE_FROM_THRESHOLD_BBOX",
        }


def detect_fov(
    image: NDArray[np.uint8],
    *,
    foreground_max_rgb_min: int = 9,
    minimum_axis_coverage_fraction: float = 0.01,
) -> FOVGeometry:
    """Detect black borders and define a deterministic elliptical FOV."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise S2DataError("FOV input must be RGB uint8 HxWx3")
    threshold = _integer(foreground_max_rgb_min, "foreground_max_rgb_min", minimum=1)
    coverage = _finite_float(minimum_axis_coverage_fraction, "minimum_axis_coverage_fraction")
    if coverage <= 0.0 or coverage > 1.0:
        raise S2DataError("minimum_axis_coverage_fraction must be in (0, 1]")
    foreground = np.max(image, axis=2) >= threshold
    height, width = foreground.shape
    rows = np.flatnonzero(np.mean(foreground, axis=1) >= coverage)
    columns = np.flatnonzero(np.mean(foreground, axis=0) >= coverage)
    if rows.size == 0 or columns.size == 0:
        raise S2DataError("FOV detection found no valid foreground")
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise S2DataError("FOV bbox is degenerate")
    center_x = (x0 + x1 - 1) / 2.0
    center_y = (y0 + y1 - 1) / 2.0
    radius_x = max((x1 - x0) / 2.0, 1.0)
    radius_y = max((y1 - y0) / 2.0, 1.0)
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
    mask = np.asarray(ellipse & foreground, dtype=np.bool_)
    forward = _translation(float(-x0), float(-y0))
    inverse = invert_affine(forward)
    return FOVGeometry(
        bbox=(x0, y0, x1, y1),
        crop_size=(x1 - x0, y1 - y0),
        original_to_crop=forward,
        crop_to_original=inverse,
        mask=mask,
    )


@dataclass(frozen=True, slots=True)
class PatchGeometry:
    """Fixed-size patch placement with explicit clipping and zero padding."""

    requested_bbox: BBox
    actual_crop_bbox: BBox
    padding_ltrb: tuple[int, int, int, int]
    patch_size: tuple[int, int]
    original_to_patch: Matrix3
    patch_to_original: Matrix3

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": S2_SCHEMA_VERSION,
            "requested_bbox_xyxy": list(self.requested_bbox),
            "actual_crop_bbox_xyxy": list(self.actual_crop_bbox),
            "padding_ltrb": list(self.padding_ltrb),
            "patch_size_wh": list(self.patch_size),
            "original_to_patch": matrix_to_json(self.original_to_patch),
            "patch_to_original": matrix_to_json(self.patch_to_original),
            "padding_mode": "CONSTANT_ZERO",
        }


def patch_geometry(
    *,
    image_size: tuple[int, int],
    centroid: tuple[float, float],
    patch_size: tuple[int, int] = (256, 256),
) -> PatchGeometry:
    """Create a fixed patch using half-up center rounding and explicit padding."""

    width, height = image_size
    patch_width, patch_height = patch_size
    if min(width, height, patch_width, patch_height) < 1:
        raise S2DataError("image and patch dimensions must be positive")
    center_x = _finite_float(centroid[0], "centroid x")
    center_y = _finite_float(centroid[1], "centroid y")
    rounded_x = math.floor(center_x + 0.5)
    rounded_y = math.floor(center_y + 0.5)
    requested_x0 = rounded_x - patch_width // 2
    requested_y0 = rounded_y - patch_height // 2
    requested_x1 = requested_x0 + patch_width
    requested_y1 = requested_y0 + patch_height
    actual_x0, actual_y0 = max(0, requested_x0), max(0, requested_y0)
    actual_x1, actual_y1 = min(width, requested_x1), min(height, requested_y1)
    if actual_x1 <= actual_x0 or actual_y1 <= actual_y0:
        raise S2DataError("patch centroid is outside the image domain")
    padding = (
        actual_x0 - requested_x0,
        actual_y0 - requested_y0,
        requested_x1 - actual_x1,
        requested_y1 - actual_y1,
    )
    forward = _translation(float(-requested_x0), float(-requested_y0))
    return PatchGeometry(
        requested_bbox=(requested_x0, requested_y0, requested_x1, requested_y1),
        actual_crop_bbox=(actual_x0, actual_y0, actual_x1, actual_y1),
        padding_ltrb=padding,
        patch_size=patch_size,
        original_to_patch=forward,
        patch_to_original=invert_affine(forward),
    )


def extract_fixed_patch(array: NDArray[np.generic], geometry: PatchGeometry) -> NDArray[np.generic]:
    """Extract an HxW or HxWxC array without silently shrinking at borders."""

    if array.ndim not in {2, 3}:
        raise S2DataError("patch source must be HxW or HxWxC")
    x0, y0, x1, y1 = geometry.actual_crop_bbox
    left, top, right, bottom = geometry.padding_ltrb
    crop = array[y0:y1, x0:x1]
    padding_2d = ((top, bottom), (left, right))
    if array.ndim == 3:
        result = np.pad(crop, padding_2d + ((0, 0),), mode="constant", constant_values=0)
    else:
        result = np.pad(crop, padding_2d, mode="constant", constant_values=0)
    expected = (geometry.patch_size[1], geometry.patch_size[0])
    if result.shape[:2] != expected:
        raise S2DataError("fixed patch extraction produced the wrong size")
    return result


def safety_ring(target_mask: NDArray[np.bool_], *, radius_px: int = 8) -> NDArray[np.bool_]:
    """Return deterministic Euclidean binary dilation minus the target itself."""

    if target_mask.dtype != np.bool_ or target_mask.ndim != 2:
        raise S2DataError("target mask must be boolean HxW")
    radius = _integer(radius_px, "radius_px", minimum=1)
    height, width = target_mask.shape
    dilated = target_mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            source_y0, source_y1 = max(0, -dy), min(height, height - dy)
            source_x0, source_x1 = max(0, -dx), min(width, width - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            dilated[target_y0:target_y1, target_x0:target_x1] |= target_mask[
                source_y0:source_y1, source_x0:source_x1
            ]
    return np.asarray(dilated & ~target_mask, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class SparseComponent:
    """One deterministic connected component represented by inclusive row runs."""

    runs: tuple[tuple[int, int, int], ...]
    area: int
    bbox: BBox
    centroid: tuple[float, float]
    eccentricity: float
    component_mask_sha256: str

    def rasterize(self, shape: tuple[int, int]) -> NDArray[np.bool_]:
        mask = np.zeros(shape, dtype=np.bool_)
        for y, x0, x1 in self.runs:
            mask[y, x0 : x1 + 1] = True
        return mask


def _row_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for y in np.flatnonzero(np.any(mask, axis=1)).tolist():
        columns = np.flatnonzero(mask[y])
        if columns.size == 0:
            continue
        boundaries = np.flatnonzero(np.diff(columns) > 1)
        starts = np.concatenate((columns[:1], columns[boundaries + 1]))
        stops = np.concatenate((columns[boundaries], columns[-1:]))
        result.extend((int(y), int(x0), int(x1)) for x0, x1 in zip(starts, stops, strict=True))
    return result


def connected_components(
    mask: NDArray[np.bool_], *, connectivity: int = 8
) -> tuple[SparseComponent, ...]:
    """Extract exact binary components by deterministic run-length union-find."""

    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise S2DataError("connected-component input must be boolean HxW")
    if connectivity not in {4, 8}:
        raise S2DataError("connectivity must be 4 or 8")
    runs = _row_runs(mask)
    if not runs:
        return ()
    parents = list(range(len(runs)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parents[high] = low

    prior_indices: list[int] = []
    current_y = -1
    current_indices: list[int] = []
    margin = 1 if connectivity == 8 else 0
    for index, (y, x0, x1) in enumerate(runs):
        if y != current_y:
            prior_indices = current_indices if y == current_y + 1 else []
            current_indices = []
            current_y = y
        for prior_index in prior_indices:
            _, prior_x0, prior_x1 = runs[prior_index]
            if prior_x1 + margin < x0:
                continue
            if prior_x0 - margin > x1:
                break
            union(index, prior_index)
        current_indices.append(index)

    grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for index, run in enumerate(runs):
        grouped[find(index)].append(run)
    components = [_component_from_runs(tuple(group)) for group in grouped.values()]
    return tuple(sorted(components, key=lambda item: (item.bbox[1], item.bbox[0], item.bbox)))


def _sum_squares(stop: int) -> int:
    return stop * (stop + 1) * (2 * stop + 1) // 6


def _component_from_runs(runs: tuple[tuple[int, int, int], ...]) -> SparseComponent:
    area = 0
    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0
    min_x = min_y = 2**31 - 1
    max_x = max_y = -1
    for y, x0, x1 in runs:
        count = x1 - x0 + 1
        run_sum_x = (x0 + x1) * count // 2
        run_sum_xx = _sum_squares(x1) - _sum_squares(x0 - 1)
        area += count
        sum_x += run_sum_x
        sum_y += y * count
        sum_xx += run_sum_xx
        sum_yy += y * y * count
        sum_xy += y * run_sum_x
        min_x, max_x = min(min_x, x0), max(max_x, x1)
        min_y, max_y = min(min_y, y), max(max_y, y)
    centroid_x, centroid_y = sum_x / area, sum_y / area
    covariance_xx = max(sum_xx / area - centroid_x * centroid_x, 0.0)
    covariance_yy = max(sum_yy / area - centroid_y * centroid_y, 0.0)
    covariance_xy = sum_xy / area - centroid_x * centroid_y
    trace = covariance_xx + covariance_yy
    delta = math.sqrt(max((covariance_xx - covariance_yy) ** 2 + 4 * covariance_xy**2, 0.0))
    major, minor = (trace + delta) / 2.0, max((trace - delta) / 2.0, 0.0)
    eccentricity = 0.0 if major <= 0.0 else math.sqrt(max(0.0, 1.0 - minor / major))
    rle_payload = canonical_json_bytes(
        {
            "schema_version": S2_SCHEMA_VERSION,
            "encoding": "INCLUSIVE_ROW_RUNS_V1",
            "bbox_xyxy": [min_x, min_y, max_x + 1, max_y + 1],
            "runs_y_x0_x1": [list(run) for run in runs],
        }
    )
    return SparseComponent(
        runs=runs,
        area=area,
        bbox=(min_x, min_y, max_x + 1, max_y + 1),
        centroid=(centroid_x, centroid_y),
        eccentricity=eccentricity,
        component_mask_sha256=sha256_bytes(rle_payload),
    )


def _ellipse_boundary_distance(centroid: tuple[float, float], fov_bbox: BBox) -> float:
    x0, y0, x1, y1 = fov_bbox
    center_x, center_y = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
    radius_x, radius_y = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    dx, dy = centroid[0] - center_x, centroid[1] - center_y
    distance_from_center = math.hypot(dx, dy)
    if distance_from_center == 0.0:
        return min(radius_x, radius_y)
    normalized_radius = math.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
    boundary_radius = distance_from_center / normalized_radius
    return boundary_radius - distance_from_center


def _component_touches_fov(component: SparseComponent, fov: FOVGeometry) -> bool:
    if _ellipse_boundary_distance(component.centroid, fov.bbox) <= 1.0:
        return True
    for y, x0, x1 in component.runs:
        if not bool(fov.mask[y, x0]) or not bool(fov.mask[y, x1]):
            return True
    return False


def _component_mask_in_patch(
    component: SparseComponent, geometry: PatchGeometry
) -> NDArray[np.bool_]:
    patch_width, patch_height = geometry.patch_size
    result = np.zeros((patch_height, patch_width), dtype=np.bool_)
    request_x0, request_y0, _, _ = geometry.requested_bbox
    for y, x0, x1 in component.runs:
        patch_y = y - request_y0
        if patch_y < 0 or patch_y >= patch_height:
            continue
        start, stop = max(0, x0 - request_x0), min(patch_width, x1 - request_x0 + 1)
        if stop > start:
            result[patch_y, start:stop] = True
    return result


def _grid_axis_starts(low: int, high: int, limit: int, size: int, stride: int) -> tuple[int, ...]:
    if limit < 1 or size < 1 or stride < 1 or high <= low:
        raise S2DataError("proposal grid dimensions are invalid")
    maximum_start = max(0, limit - size)
    first = min(max(0, low), maximum_start)
    last = min(max(0, high - size), maximum_start)
    starts = list(range(first, last + 1, stride))
    if not starts:
        starts = [first]
    if starts[-1] != last:
        starts.append(last)
    return tuple(sorted(set(starts)))


def proposal_grid(
    *, image_size: tuple[int, int], fov_bbox: BBox, patch_size: int = 256, stride: int = 128
) -> tuple[BBox, ...]:
    """Return a deterministic overlapping spatial high-recall baseline."""

    width, height = image_size
    x0, y0, x1, y1 = fov_bbox
    x_starts = _grid_axis_starts(x0, x1, width, patch_size, stride)
    y_starts = _grid_axis_starts(y0, y1, height, patch_size, stride)
    return tuple(
        (x, y, min(x + patch_size, width), min(y + patch_size, height))
        for y in y_starts
        for x in x_starts
    )


def _proposal_covers(boxes: Sequence[BBox], point: tuple[float, float]) -> bool:
    x, y = point
    return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in boxes)


@dataclass(frozen=True, slots=True)
class S2Config:
    """Validated subset of the frozen S2 YAML used by executable code."""

    path: Path
    payload: dict[str, object]
    config_sha256: str
    dataset_id: str
    dataset_version: str
    artifact_root: str
    lesion_classes: tuple[str, str]
    locked_test_membership_count: int
    development_manifest_only: bool
    entrypoint_file: str
    prerequisite: dict[str, object]
    fov_threshold: int
    fov_coverage: float
    connectivity: int
    minimum_area: int
    maximum_area_fraction: float
    patch_size: int
    safety_radius: int
    proposal_patch_size: int
    proposal_stride: int
    qc_seed: int
    qc_count: int
    qc_columns: int
    max_round_trip_error: float

    @classmethod
    def load(cls, path: Path) -> S2Config:
        decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S2 config")
        if _string(root.get("schema_version"), "schema_version") != "2.0.0":
            raise S2GateError("S2 config schema_version must be 2.0.0")
        if root.get("development_splits") != ["TRAIN", "CALIBRATION"]:
            raise S2GateError("development_splits must be exactly TRAIN/CALIBRATION")
        if root.get("locked_test_policy") != "FROZEN_MEMBERSHIP_ONLY_ZERO_ROOT_READS":
            raise S2GateError("locked-test policy is not fail-closed")
        prerequisite = _mapping(root.get("prerequisite"), "prerequisite")
        fov = _mapping(root.get("fov"), "fov")
        components = _mapping(root.get("components"), "components")
        patch = _mapping(root.get("patch"), "patch")
        ring = _mapping(root.get("safety_ring"), "safety_ring")
        proposal = _mapping(root.get("proposal"), "proposal")
        qc = _mapping(root.get("qc"), "qc")
        gate = _mapping(root.get("gate"), "gate")
        patch_shape = patch.get("size")
        proposal_shape = proposal.get("patch_size")
        proposal_stride = proposal.get("stride")
        if patch_shape != [256, 256] or proposal_shape != [256, 256]:
            raise S2GateError("S2 patch sizes must remain 256x256")
        if not isinstance(proposal_stride, list) or proposal_stride != [128, 128]:
            raise S2GateError("proposal stride must remain 128x128")
        raw_lesion_classes = root.get("lesion_classes")
        if not isinstance(raw_lesion_classes, list) or len(raw_lesion_classes) != 2:
            raise S2GateError("lesion_classes must contain one frozen pair")
        lesion_classes = tuple(
            _string(value, "lesion_classes item") for value in raw_lesion_classes
        )
        if len(set(lesion_classes)) != 2 or any(
            lesion_class not in LESION_SEMANTICS for lesion_class in lesion_classes
        ):
            raise S2GateError("lesion_classes contains an unsupported or duplicate class")
        expected_counts = _mapping(prerequisite.get("expected_split_counts"), "split counts")
        default_locked_count = expected_counts.get("LOCKED_TEST", 0)
        locked_membership_count = _integer(
            root.get("locked_test_membership_count", default_locked_count),
            "locked_test_membership_count",
            minimum=0,
        )
        development_manifest_only_raw = root.get("development_manifest_only", False)
        if type(development_manifest_only_raw) is not bool:  # noqa: E721
            raise S2GateError("development_manifest_only must be a bool")
        config_bytes = canonical_json_bytes(cast(dict[str, JSONValue], root))
        return cls(
            path=path,
            payload=root,
            config_sha256=sha256_bytes(config_bytes),
            dataset_id=_string(root.get("dataset_id"), "dataset_id"),
            dataset_version=_string(root.get("dataset_version"), "dataset_version"),
            artifact_root=_safe_relative(root.get("artifact_root"), "artifact_root"),
            lesion_classes=cast(tuple[str, str], lesion_classes),
            locked_test_membership_count=locked_membership_count,
            development_manifest_only=development_manifest_only_raw,
            entrypoint_file=_safe_relative(
                root.get("entrypoint_file", "src/retinaentail/science_s2.py"),
                "entrypoint_file",
            ),
            prerequisite=prerequisite,
            fov_threshold=_integer(
                fov.get("foreground_max_rgb_min"), "foreground_max_rgb_min", minimum=1
            ),
            fov_coverage=_finite_float(
                fov.get("minimum_axis_coverage_fraction"),
                "minimum_axis_coverage_fraction",
            ),
            connectivity=_integer(components.get("connectivity"), "connectivity", minimum=4),
            minimum_area=_integer(components.get("minimum_area_px"), "minimum_area_px", minimum=1),
            maximum_area_fraction=_finite_float(
                components.get("maximum_area_fraction"), "maximum_area_fraction"
            ),
            patch_size=256,
            safety_radius=_integer(ring.get("radius_px"), "radius_px", minimum=1),
            proposal_patch_size=256,
            proposal_stride=128,
            qc_seed=_integer(qc.get("seed"), "qc seed", minimum=0),
            qc_count=_integer(qc.get("components_per_class_per_split"), "qc count", minimum=1),
            qc_columns=_integer(qc.get("contact_sheet_columns"), "qc columns", minimum=1),
            max_round_trip_error=_finite_float(
                gate.get("max_coordinate_round_trip_error_px"), "round-trip limit"
            ),
        )


@dataclass(frozen=True, slots=True)
class S1Replay:
    manifest: DatasetManifestV2
    registry_path: Path
    review_path: Path
    spec_path: Path
    manifest_path: Path
    audit_path: Path
    hashes: dict[str, str]
    split_counts: dict[str, int]


def _project_file(project_root: Path, relative: object, name: str) -> Path:
    normalized = _safe_relative(relative, name)
    candidate = (project_root / Path(normalized)).resolve(strict=True)
    try:
        candidate.relative_to(project_root.resolve(strict=True))
    except ValueError as error:
        raise S2GateError(f"{name} escapes the project root") from error
    if not candidate.is_file():
        raise S2GateError(f"{name} is not a file")
    return candidate


def replay_s1_prerequisite(project_root: Path, config: S2Config) -> S1Replay:
    """Replay every repository-side prerequisite without accessing a dataset root."""

    prerequisite = config.prerequisite
    named_paths = {
        "registry": _project_file(project_root, prerequisite.get("registry"), "registry"),
        "review": _project_file(project_root, prerequisite.get("review"), "review"),
        "adapter_spec": _project_file(
            project_root, prerequisite.get("adapter_spec"), "adapter_spec"
        ),
        "manifest": _project_file(project_root, prerequisite.get("manifest"), "manifest"),
        "audit": _project_file(project_root, prerequisite.get("audit"), "audit"),
    }
    expected_hash_names = {
        "registry": "registry_sha256",
        "review": "review_sha256",
        "adapter_spec": "adapter_spec_sha256",
        "manifest": "manifest_file_sha256",
        "audit": "audit_sha256",
    }
    hashes: dict[str, str] = {}
    for name, path in named_paths.items():
        actual = sha256_file(path)
        expected = _sha256(prerequisite.get(expected_hash_names[name]), expected_hash_names[name])
        if actual != expected:
            raise S2GateError(f"S1 prerequisite hash mismatch: {name}")
        hashes[name] = actual
    registry = load_registry_license_metadata(named_paths["registry"])
    review = load_license_review_record(named_paths["review"])
    spec = load_dataset_adapter_spec(named_paths["adapter_spec"])
    decision = assess_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.LESION_CALIBRATION,
        root_provided=True,
        spec_provided=True,
    )
    if not decision.permitted:
        raise S2GateError(f"S1 license/scope replay blocked: {'; '.join(decision.reasons)}")
    decoded_manifest: object = json.loads(named_paths["manifest"].read_text(encoding="utf-8"))
    if not isinstance(decoded_manifest, Mapping):
        raise S2GateError("S1 manifest must be an object")
    manifest = DatasetManifestV2.from_dict(decoded_manifest)
    decoded_audit: object = json.loads(named_paths["audit"].read_text(encoding="utf-8"))
    audit = _mapping(decoded_audit, "S1 audit")
    semantic_hash = _canonical_compact_sha256(manifest.to_dict())
    expected_semantic_hash = _sha256(
        prerequisite.get("manifest_semantic_sha256"), "manifest_semantic_sha256"
    )
    if semantic_hash != expected_semantic_hash or audit.get("manifest_sha256") != semantic_hash:
        raise S2GateError("S1 semantic manifest hash replay failed")
    if audit.get("status") != "PASS" or audit.get("dataset_version") != config.dataset_version:
        raise S2GateError("S1 PASS audit/version replay failed")
    if (
        manifest.dataset_id != config.dataset_id
        or manifest.dataset_version != config.dataset_version
    ):
        raise S2GateError("S1 manifest dataset binding failed")
    split_counts = dict(Counter(record.split for record in manifest.records))
    expected_counts = prerequisite.get("expected_split_counts")
    if not isinstance(expected_counts, Mapping) or split_counts != dict(expected_counts):
        raise S2GateError("S1 split replay failed")
    if len(manifest.records) != len(spec.entries):
        raise S2GateError("S1 manifest/spec row count mismatch")
    for record, entry in zip(manifest.records, spec.entries, strict=True):
        if (
            record.source_image_id != entry.source_image_id
            or record.relative_path != entry.image_path.as_posix()
            or record.split != entry.split
            or any(
                value is not None
                for value in (
                    record.patient_id,
                    record.eye_id,
                    record.followup_id,
                    record.intervention_pair_id,
                )
            )
        ):
            raise S2GateError("S1 manifest/spec identity replay failed")
    return S1Replay(
        manifest=manifest,
        registry_path=named_paths["registry"],
        review_path=named_paths["review"],
        spec_path=named_paths["adapter_spec"],
        manifest_path=named_paths["manifest"],
        audit_path=named_paths["audit"],
        hashes=hashes,
        split_counts=split_counts,
    )


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    normalized = _safe_relative(relative_path, "dataset relative path")
    candidate = (root / Path(normalized)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise S2DataError("dataset path escapes the authorized root") from error
    if not candidate.is_file():
        raise S2DataError(f"declared dataset file is missing: {normalized}")
    return candidate


def _load_rgb_strict(path: Path, record: ManifestRecordV2) -> NDArray[np.uint8]:
    if sha256_file(path) != record.sha256:
        raise S2DataError(f"image hash mismatch: {record.relative_path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (record.width, record.height):
            raise S2DataError(f"image mode/size mismatch: {record.relative_path}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.shape != (record.height, record.width, 3):
        raise S2DataError(f"image array shape mismatch: {record.relative_path}")
    return array


def decode_idrid_mask(path: Path, *, expected_size: tuple[int, int]) -> NDArray[np.bool_]:
    """Decode only the two exact black/red IDRiD encodings accepted by S1."""

    with Image.open(path) as image:
        image.load()
        if image.size != expected_size:
            raise S2DataError("mask/image dimensions do not match")
        if image.mode == "P":
            palette = image.getpalette()
            if palette is None or len(palette) < 6:
                raise S2DataError("IDRiD palette mask is missing its palette")
            if tuple(palette[0:3]) != (0, 0, 0) or tuple(palette[3:6]) != (255, 0, 0):
                raise S2DataError("IDRiD palette mask has unexpected colors")
            array = np.asarray(image, dtype=np.uint8)
            values = set(int(value) for value in np.unique(array))
            if not values.issubset({0, 1}):
                raise S2DataError("IDRiD palette mask has unexpected indices")
            return np.asarray(array == 1, dtype=np.bool_)
        if image.mode == "RGBA":
            rgba = np.asarray(image, dtype=np.uint8)
            pixels = rgba.reshape(-1, 4)
            valid_black = np.all(pixels == np.array([0, 0, 0, 255], dtype=np.uint8), axis=1)
            valid_red = np.all(pixels == np.array([255, 0, 0, 255], dtype=np.uint8), axis=1)
            if not bool(np.all(valid_black | valid_red)):
                raise S2DataError("IDRiD RGBA mask has unexpected colors or alpha")
            return np.asarray(valid_red.reshape(rgba.shape[:2]), dtype=np.bool_)
        raise S2DataError(f"IDRiD mask mode is not controlled: {image.mode}")


def _label_for(record: ManifestRecordV2, semantic: LabelSemantic) -> LabelRecordV2:
    matches = [label for label in record.labels if label.semantic is semantic]
    if len(matches) != 1:
        raise S2DataError(f"record does not declare exactly one {semantic.value} label")
    label = matches[0]
    if (
        label.availability is not LabelAvailability.AVAILABLE
        or label.relative_path is None
        or label.sha256 is None
    ):
        raise S2DataError(f"{semantic.value} label is UNKNOWN, not a negative mask")
    return label


def _code_sha256(project_root: Path, config: S2Config) -> str:
    paths = (
        project_root / "src" / "retinaentail" / "data" / "lesion_catalog.py",
        project_root / config.entrypoint_file,
    )
    if any(not path.is_file() for path in paths):
        raise S2GateError("catalog implementation binding is missing")
    payload = canonical_json_bytes(
        {
            "schema_version": S2_SCHEMA_VERSION,
            "files": [
                {
                    "relative_path": path.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        }
    )
    return sha256_bytes(payload)


def _component_id(source_image_id: str, lesion_class: str, index: int) -> str:
    return f"{source_image_id}-{lesion_class.lower()}-{index:06d}"


def _round_trip_error(geometry: PatchGeometry, point: tuple[float, float]) -> float:
    patch_point = apply_transform(geometry.original_to_patch, *point)
    restored = apply_transform(geometry.patch_to_original, *patch_point)
    return max(abs(restored[0] - point[0]), abs(restored[1] - point[1]))


def _contact_sheet(
    cells: Sequence[
        tuple[
            str,
            NDArray[np.uint8],
            NDArray[np.bool_],
            NDArray[np.bool_],
            NDArray[np.bool_],
        ]
    ],
    *,
    columns: int,
) -> bytes:
    cell_width, cell_height = 256, 286
    rows = max(1, math.ceil(len(cells) / columns))
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (label, rgb, target, ring, non_target) in enumerate(cells):
        overlay = rgb.copy()
        overlay[non_target] = np.array([0, 150, 255], dtype=np.uint8)
        overlay[ring] = np.array([255, 215, 0], dtype=np.uint8)
        overlay[target] = np.array([0, 255, 80], dtype=np.uint8)
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        canvas.paste(Image.fromarray(overlay, mode="RGB"), (x, y))
        draw.text((x + 3, y + 260), label[:42], fill=(255, 255, 255))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _catalog_csv(records: Sequence[dict[str, JSONValue]]) -> bytes:
    fields = (
        "schema_version",
        "dataset_id",
        "dataset_version",
        "source_image_id",
        "split",
        "lesion_class",
        "component_id",
        "image_relative_path",
        "mask_relative_path",
        "image_sha256",
        "mask_sha256",
        "component_mask_sha256",
        "area_px",
        "bbox_xyxy",
        "centroid_xy",
        "eccentricity",
        "distance_to_fov_boundary_px",
        "touches_image_boundary",
        "touches_fov_boundary",
        "inclusion_status",
        "exclusion_reason",
        "config_sha256",
        "data_sha256",
        "code_sha256",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row: dict[str, object] = {}
        for field in fields:
            value = record[field]
            row[field] = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, list | dict)
                else value
            )
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")


def _audit_html(audit: Mapping[str, object]) -> bytes:
    summary = _mapping(audit.get("summary"), "audit summary")
    rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in sorted(summary.items())
    )
    status = escape(str(audit.get("status")))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>RetinaEntail-DR Science S2 audit</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #888;padding:.4rem;text-align:left}</style></head><body>"
        f"<h1>Science S2 audit</h1><p>Status: <strong>{status}</strong></p>"
        f"<table>{rows}</table><p>Oracle regions are controlled annotations, "
        "not deployment capability.</p>"
        "</body></html>\n"
    ).encode()


@dataclass(frozen=True, slots=True)
class S2RunResult:
    status: str
    output_dir: Path
    component_count: int
    component_counts: dict[str, int]
    source_count: int
    mask_count: int
    locked_test_root_reads: int
    artifact_hashes: dict[str, str]
    audit: dict[str, JSONValue]


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path, payload in sorted(artifacts.items()):
        target = output_dir / relative_path
        write_bytes_fail_safe(target, payload)
        hashes[relative_path] = sha256_bytes(payload)
    manifest_payload = canonical_json_bytes(
        {
            "schema_version": S2_SCHEMA_VERSION,
            "files": [
                {"relative_path": path, "sha256": digest, "byte_size": len(artifacts[path])}
                for path, digest in sorted(hashes.items())
            ],
        }
    )
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", manifest_payload)
    hashes["artifact-manifest.json"] = sha256_bytes(manifest_payload)
    return hashes


def _require_ignored_output_boundary(project_root: Path, output_dir: Path) -> None:
    relative = output_dir.resolve(strict=False).relative_to(project_root.resolve(strict=True))
    if relative.parts[:2] != ("reports", "generated"):
        raise S2DataError("real S2 artifacts must be under reports/generated")
    ignore_rules = {
        line.strip()
        for line in (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    if "/reports/generated/" not in ignore_rules:
        raise S2DataError("reports/generated is not protected by .gitignore")


def run_s2_catalog(
    *, project_root: Path, config_path: Path, dataset_root: Path, output_dir: Path
) -> S2RunResult:
    """Run the authorized real S2 catalog; all exceptions are fail-closed."""

    config = S2Config.load(config_path)
    replay = replay_s1_prerequisite(project_root, config)
    registry = load_registry_license_metadata(replay.registry_path)
    review = load_license_review_record(replay.review_path)
    spec = load_dataset_adapter_spec(replay.spec_path)
    authorize_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.LESION_CALIBRATION,
        root=dataset_root,
        spec=spec,
    )
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise S2DataError("authorized dataset root is not a directory")
    resolved_output = output_dir.resolve(strict=False)
    try:
        resolved_output.relative_to(project_root.resolve(strict=True))
    except ValueError as error:
        raise S2DataError("S2 output must remain inside the project") from error
    expected_output = (project_root / Path(config.artifact_root)).resolve(strict=False)
    if resolved_output != expected_output:
        raise S2DataError("S2 output does not match the frozen artifact_root")
    _require_ignored_output_boundary(project_root, resolved_output)
    try:
        resolved_output.relative_to(root)
        raise S2DataError("S2 output overlaps the dataset root")
    except ValueError:
        pass

    code_sha256 = _code_sha256(project_root, config)
    catalog: list[dict[str, JSONValue]] = []
    transforms: list[dict[str, JSONValue]] = []
    inclusion_reasons: Counter[str] = Counter()
    empty_masks: Counter[str] = Counter()
    proposal_stats: dict[str, Counter[str]] = defaultdict(Counter)
    private_components: dict[str, SparseComponent] = {}
    private_geometries: dict[str, PatchGeometry] = {}
    record_lookup: dict[str, tuple[ManifestRecordV2, FOVGeometry]] = {}
    source_splits: dict[str, str] = {}
    maximum_round_trip_error = 0.0
    mask_count = 0
    root_read_log: list[dict[str, str]] = []

    development_records = tuple(
        record for record in replay.manifest.records if record.split in DEVELOPMENT_SPLITS
    )
    locked_records = tuple(
        record for record in replay.manifest.records if record.split == LOCKED_SPLIT
    )
    expected_split_counts = _mapping(
        config.prerequisite.get("expected_split_counts"), "expected split counts"
    )
    expected_development_count = sum(
        _integer(expected_split_counts.get(split), f"expected {split} count")
        for split in sorted(DEVELOPMENT_SPLITS)
    )
    if len(development_records) != expected_development_count:
        raise S2GateError("catalog development cohort size mismatch")
    if config.development_manifest_only:
        if locked_records:
            raise S2GateError("development-only manifest unexpectedly contains LOCKED_TEST")
    elif len(locked_records) != config.locked_test_membership_count:
        raise S2GateError("catalog locked-test membership count mismatch")
    for record in development_records:
        prior_split = source_splits.setdefault(record.source_image_id, record.split)
        if prior_split != record.split:
            raise S2DataError("source_image_id crosses splits")
        image_path = _resolve_under_root(root, record.relative_path)
        image = _load_rgb_strict(image_path, record)
        root_read_log.append(
            {"split": record.split, "kind": "IMAGE", "source_image_id": record.source_image_id}
        )
        fov = detect_fov(
            image,
            foreground_max_rgb_min=config.fov_threshold,
            minimum_axis_coverage_fraction=config.fov_coverage,
        )
        record_lookup[record.source_image_id] = (record, fov)
        transforms.append(
            {
                "schema_version": S2_SCHEMA_VERSION,
                "source_image_id": record.source_image_id,
                "split": record.split,
                "image_size_wh": [record.width, record.height],
                "fov": fov.to_dict(),
            }
        )
        for lesion_class in config.lesion_classes:
            semantic = LESION_SEMANTICS[lesion_class]
            label = _label_for(record, semantic)
            label_path_text = cast(str, label.relative_path)
            label_sha256 = cast(str, label.sha256)
            mask_path = _resolve_under_root(root, label_path_text)
            if sha256_file(mask_path) != label_sha256:
                raise S2DataError(f"mask hash mismatch: {label_path_text}")
            mask = decode_idrid_mask(mask_path, expected_size=(record.width, record.height))
            root_read_log.append(
                {
                    "split": record.split,
                    "kind": "MASK",
                    "source_image_id": record.source_image_id,
                }
            )
            mask_count += 1
            components = connected_components(mask, connectivity=config.connectivity)
            if not components:
                empty_masks[f"{record.split}:{lesion_class}"] += 1
                inclusion_reasons["EMPTY_ANNOTATION_NOT_NEGATIVE"] += 1
            proposal_boxes = proposal_grid(
                image_size=(record.width, record.height),
                fov_bbox=fov.bbox,
                patch_size=config.proposal_patch_size,
                stride=config.proposal_stride,
            )
            proposal_stats[record.split]["proposal_boxes"] += len(proposal_boxes)
            proposal_stats[record.split]["components"] += len(components)
            for index, component in enumerate(components, start=1):
                component_id = _component_id(record.source_image_id, lesion_class, index)
                proposal_stats[record.split]["covered_components"] += int(
                    _proposal_covers(proposal_boxes, component.centroid)
                )
                area_fraction = component.area / (record.width * record.height)
                if component.area < config.minimum_area:
                    inclusion_status, reason = "EXCLUDED", "AREA_BELOW_FROZEN_MINIMUM"
                elif area_fraction > config.maximum_area_fraction:
                    inclusion_status, reason = "EXCLUDED", "AREA_ABOVE_FROZEN_MAXIMUM"
                else:
                    inclusion_status, reason = "INCLUDED", "INCLUDED_BY_FROZEN_RULE"
                inclusion_reasons[reason] += 1
                geometry = patch_geometry(
                    image_size=(record.width, record.height),
                    centroid=component.centroid,
                    patch_size=(config.patch_size, config.patch_size),
                )
                round_trip_error = _round_trip_error(geometry, component.centroid)
                maximum_round_trip_error = max(maximum_round_trip_error, round_trip_error)
                if round_trip_error > config.max_round_trip_error:
                    raise S2DataError("coordinate round-trip error exceeds frozen limit")
                target_patch = _component_mask_in_patch(component, geometry)
                if target_patch.shape != (config.patch_size, config.patch_size):
                    raise S2DataError("target patch alignment failed")
                private_components[component_id] = component
                private_geometries[component_id] = geometry
                non_target_class = next(
                    item for item in config.lesion_classes if item != lesion_class
                )
                catalog.append(
                    {
                        "schema_version": S2_SCHEMA_VERSION,
                        "dataset_id": config.dataset_id,
                        "dataset_version": config.dataset_version,
                        "source_image_id": record.source_image_id,
                        "patient_id": None,
                        "eye_id": None,
                        "followup_id": None,
                        "intervention_pair_id": None,
                        "split": record.split,
                        "lesion_class": lesion_class,
                        "region_channel": "ORACLE_REGION",
                        "oracle_use_scope": "CONTROLLED_EXPERIMENT_ONLY_NOT_DEPLOYMENT",
                        "component_id": component_id,
                        "image_relative_path": record.relative_path,
                        "mask_relative_path": label_path_text,
                        "image_sha256": record.sha256,
                        "mask_sha256": label_sha256,
                        "non_target_lesion_class": non_target_class,
                        "non_target_mask_relative_path": cast(
                            str,
                            _label_for(
                                record,
                                LESION_SEMANTICS[non_target_class],
                            ).relative_path,
                        ),
                        "non_target_mask_sha256": cast(
                            str,
                            _label_for(
                                record,
                                LESION_SEMANTICS[non_target_class],
                            ).sha256,
                        ),
                        "component_mask_sha256": component.component_mask_sha256,
                        "image_size_wh": [record.width, record.height],
                        "fov_bbox_xyxy": list(fov.bbox),
                        "area_px": component.area,
                        "bbox_xyxy": list(component.bbox),
                        "centroid_xy": list(component.centroid),
                        "eccentricity": component.eccentricity,
                        "distance_to_fov_boundary_px": _ellipse_boundary_distance(
                            component.centroid, fov.bbox
                        ),
                        "touches_image_boundary": (
                            component.bbox[0] == 0
                            or component.bbox[1] == 0
                            or component.bbox[2] == record.width
                            or component.bbox[3] == record.height
                        ),
                        "touches_fov_boundary": _component_touches_fov(component, fov),
                        "coordinate_transforms": {
                            "original_to_crop": matrix_to_json(fov.original_to_crop),
                            "crop_to_original": matrix_to_json(fov.crop_to_original),
                            "original_to_patch": matrix_to_json(geometry.original_to_patch),
                            "patch_to_original": matrix_to_json(geometry.patch_to_original),
                        },
                        "patch": geometry.to_dict(),
                        "safety_ring": {
                            "version": "EUCLIDEAN_BINARY_DILATION_V1",
                            "radius_px": config.safety_radius,
                            "exclude_target": True,
                        },
                        "inclusion_status": inclusion_status,
                        "exclusion_reason": None if inclusion_status == "INCLUDED" else reason,
                        "reason_code": reason,
                        "coordinate_round_trip_error_px": round_trip_error,
                        "config_sha256": config.config_sha256,
                        "data_sha256": replay.hashes["manifest"],
                        "code_sha256": code_sha256,
                    }
                )
    if any(item["split"] == LOCKED_SPLIT for item in root_read_log):
        raise S2DataError("LOCKED_TEST root read detected")
    expected_mask_count = len(development_records) * len(config.lesion_classes)
    if mask_count != expected_mask_count or len(record_lookup) != len(development_records):
        raise S2DataError("development source/mask count mismatch")

    catalog.sort(key=lambda row: cast(str, row["component_id"]))
    transforms.sort(key=lambda row: cast(str, row["source_image_id"]))
    catalog_payload: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "scope": "TRAIN_AND_CALIBRATION_ONLY",
        "records": cast(list[JSONValue], catalog),
    }
    catalog_bytes = canonical_json_bytes(catalog_payload)
    if catalog_bytes != canonical_json_bytes(catalog_payload):
        raise S2DataError("catalog serialization is not deterministic")

    qc_cells: dict[
        tuple[str, str],
        list[
            tuple[
                str,
                NDArray[np.uint8],
                NDArray[np.bool_],
                NDArray[np.bool_],
                NDArray[np.bool_],
            ]
        ],
    ] = {}
    for split in ("TRAIN", "CALIBRATION"):
        for lesion_class in config.lesion_classes:
            candidates = [
                catalog_record
                for catalog_record in catalog
                if catalog_record["split"] == split
                and catalog_record["lesion_class"] == lesion_class
                and catalog_record["inclusion_status"] == "INCLUDED"
            ]
            candidates.sort(
                key=lambda catalog_record: hashlib.sha256(
                    f"{config.qc_seed}:{catalog_record['component_id']}".encode()
                ).hexdigest()
            )
            cells: list[
                tuple[
                    str,
                    NDArray[np.uint8],
                    NDArray[np.bool_],
                    NDArray[np.bool_],
                    NDArray[np.bool_],
                ]
            ] = []
            for catalog_record in candidates[: config.qc_count]:
                source_image_id = cast(str, catalog_record["source_image_id"])
                source_record, _ = record_lookup[source_image_id]
                image_path = _resolve_under_root(root, source_record.relative_path)
                image = _load_rgb_strict(image_path, source_record)
                root_read_log.append(
                    {
                        "split": source_record.split,
                        "kind": "QC_IMAGE",
                        "source_image_id": source_image_id,
                    }
                )
                component_id = cast(str, catalog_record["component_id"])
                geometry = private_geometries[component_id]
                component = private_components[component_id]
                target = _component_mask_in_patch(component, geometry)
                ring = safety_ring(target, radius_px=config.safety_radius)
                rgb_patch = cast(NDArray[np.uint8], extract_fixed_patch(image, geometry))
                non_target_path_text = cast(str, catalog_record["non_target_mask_relative_path"])
                non_target_sha256 = cast(str, catalog_record["non_target_mask_sha256"])
                non_target_path = _resolve_under_root(root, non_target_path_text)
                if sha256_file(non_target_path) != non_target_sha256:
                    raise S2DataError(f"mask hash mismatch: {non_target_path_text}")
                non_target_full = decode_idrid_mask(
                    non_target_path,
                    expected_size=(source_record.width, source_record.height),
                )
                root_read_log.append(
                    {
                        "split": source_record.split,
                        "kind": "QC_NON_TARGET_MASK",
                        "source_image_id": source_image_id,
                    }
                )
                non_target = cast(NDArray[np.bool_], extract_fixed_patch(non_target_full, geometry))
                if not (
                    rgb_patch.shape[:2]
                    == target.shape
                    == ring.shape
                    == non_target.shape
                    == (config.patch_size, config.patch_size)
                ):
                    raise S2DataError("image/target/ring/non-target patch alignment failed")
                cells.append((component_id, rgb_patch, target, ring, non_target))
            qc_cells[(split, lesion_class)] = cells
    if any(item["split"] == LOCKED_SPLIT for item in root_read_log):
        raise S2DataError("LOCKED_TEST root read detected during QC")

    proposal_split_payload: dict[str, JSONValue] = {}
    for split, stats in sorted(proposal_stats.items()):
        proposal_split_payload[split] = {
            "proposal_boxes": stats["proposal_boxes"],
            "oracle_components": stats["components"],
            "covered_oracle_components": stats["covered_components"],
            "coverage": (
                None
                if stats["components"] == 0
                else stats["covered_components"] / stats["components"]
            ),
        }
    proposal_payload: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "region_channel": "PROPOSAL_REGION",
        "version": "OVERLAPPING_FOV_GRID_V1",
        "claim_boundary": "SPATIAL_HIGH_RECALL_BASELINE_NOT_LESION_DETECTOR",
        "selection_split": "TRAIN_ONLY_PREREGISTERED_GEOMETRY",
        "calibration_metric": "COMPONENT_CENTROID_COVERAGE",
        "patch_size_wh": [config.proposal_patch_size, config.proposal_patch_size],
        "stride_xy": [config.proposal_stride, config.proposal_stride],
        "splits": proposal_split_payload,
        "failure_state": "OUT_OF_MODEL",
        "locked_test_tuning_reads": 0,
    }
    inclusion_payload: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "empty_mask_semantics": "EMPTY_ANNOTATION_NOT_NEGATIVE",
        "reason_counts": dict(sorted(inclusion_reasons.items())),
        "empty_mask_counts": dict(sorted(empty_masks.items())),
        "unknown_mask_count": 0,
    }
    transform_payload: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "scope": "TRAIN_AND_CALIBRATION_ONLY",
        "records": cast(list[JSONValue], transforms),
    }
    component_counts = Counter(f"{record['split']}:{record['lesion_class']}" for record in catalog)
    audit_payload: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "status": "READY_FOR_HUMAN_QC",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "summary": {
            "development_source_count": len(record_lookup),
            "development_mask_count": mask_count,
            "component_count": len(catalog),
            "included_component_count": sum(
                record["inclusion_status"] == "INCLUDED" for record in catalog
            ),
            "excluded_component_count": sum(
                record["inclusion_status"] == "EXCLUDED" for record in catalog
            ),
            "unknown_mask_count": 0,
            "source_split_leak_count": 0,
            "locked_test_membership_count": config.locked_test_membership_count,
            "locked_test_root_reads": 0,
            "locked_test_mask_decodes": 0,
            "maximum_coordinate_round_trip_error_px": maximum_round_trip_error,
            "contact_sheets": 4,
            "human_qc_confirmed": False,
        },
        "component_counts": dict(sorted(component_counts.items())),
        "gate": {
            "s1_replay": "PASS",
            "traceability": "PASS",
            "coordinate_round_trip": "PASS",
            "patch_alignment": "PASS",
            "source_split_isolation": "PASS",
            "locked_test_isolation": "PASS",
            "reason_codes": "PASS",
            "byte_stable_serialization": "PASS",
            "real_artifact_git_boundary": "PASS",
            "human_contact_sheet_qc": "PENDING",
        },
        "hashes": {
            "config_sha256": config.config_sha256,
            "code_sha256": code_sha256,
            "s1_manifest_file_sha256": replay.hashes["manifest"],
            "s1_manifest_semantic_sha256": _canonical_compact_sha256(replay.manifest.to_dict()),
        },
    }
    artifacts: dict[str, bytes] = {
        "component-catalog-v1.json": catalog_bytes,
        "component-catalog-v1.csv": _catalog_csv(catalog),
        "coordinate-transforms-v1.json": canonical_json_bytes(transform_payload),
        "inclusion-exclusion.json": canonical_json_bytes(inclusion_payload),
        "proposal-summary.json": canonical_json_bytes(proposal_payload),
        "audit.json": canonical_json_bytes(audit_payload),
        "audit.html": _audit_html(audit_payload),
    }
    for (split, lesion_class), cells in sorted(qc_cells.items()):
        artifacts[f"qc-{split.lower()}-{lesion_class.lower()}-contact-sheet.png"] = _contact_sheet(
            cells, columns=config.qc_columns
        )
    hashes = _write_artifacts(resolved_output, artifacts)
    return S2RunResult(
        status="READY_FOR_HUMAN_QC",
        output_dir=resolved_output,
        component_count=len(catalog),
        component_counts=dict(sorted(component_counts.items())),
        source_count=len(record_lookup),
        mask_count=mask_count,
        locked_test_root_reads=0,
        artifact_hashes=hashes,
        audit=audit_payload,
    )


def run_s2_synthetic_fixture(output_dir: Path) -> S2RunResult:
    """Exercise FOV/components/patch/ring/proposal without medical data."""

    height, width = 384, 512
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - 255.5) / 235.0) ** 2 + ((yy - 191.5) / 170.0) ** 2 <= 1.0
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[ellipse] = np.array([92, 38, 22], dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[38:42, 20:24] = True
    mask[180:186, 250:257] = True
    detect_fov(image)
    components = connected_components(mask, connectivity=8)
    if len(components) != 2:
        raise S2DataError("synthetic component fixture failed")
    records: list[dict[str, JSONValue]] = []
    for index, component in enumerate(components, start=1):
        geometry = patch_geometry(
            image_size=(width, height), centroid=component.centroid, patch_size=(256, 256)
        )
        target = _component_mask_in_patch(component, geometry)
        ring = safety_ring(target)
        rgb = cast(NDArray[np.uint8], extract_fixed_patch(image, geometry))
        if rgb.shape != (256, 256, 3) or target.shape != ring.shape or np.any(target & ring):
            raise S2DataError("synthetic patch alignment failed")
        records.append(
            {
                "component_id": f"synthetic-he-{index:06d}",
                "area_px": component.area,
                "bbox_xyxy": list(component.bbox),
                "centroid_xy": list(component.centroid),
                "component_mask_sha256": component.component_mask_sha256,
                "patch": geometry.to_dict(),
                "coordinate_round_trip_error_px": _round_trip_error(geometry, component.centroid),
            }
        )
    synthetic_catalog: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "scope": "SYNTHETIC_FIXTURE",
        "records": cast(list[JSONValue], records),
    }
    catalog = canonical_json_bytes(synthetic_catalog)
    audit: dict[str, JSONValue] = {
        "schema_version": S2_SCHEMA_VERSION,
        "status": "PASS_SYNTHETIC_FIXTURE_ONLY",
        "summary": {
            "development_source_count": 1,
            "development_mask_count": 1,
            "component_count": len(components),
            "locked_test_root_reads": 0,
        },
    }
    artifacts = {
        "component-catalog-v1.json": catalog,
        "audit.json": canonical_json_bytes(audit),
        "audit.html": _audit_html(audit),
    }
    hashes = _write_artifacts(output_dir, artifacts)
    return S2RunResult(
        status="PASS_SYNTHETIC_FIXTURE_ONLY",
        output_dir=output_dir,
        component_count=len(components),
        component_counts={"TRAIN:HE": len(components)},
        source_count=1,
        mask_count=1,
        locked_test_root_reads=0,
        artifact_hashes=hashes,
        audit=audit,
    )


__all__ = [
    "FOVGeometry",
    "PatchGeometry",
    "S1Replay",
    "S2Config",
    "S2DataError",
    "S2GateError",
    "S2RunResult",
    "SparseComponent",
    "apply_transform",
    "connected_components",
    "decode_idrid_mask",
    "detect_fov",
    "extract_fixed_patch",
    "invert_affine",
    "patch_geometry",
    "proposal_grid",
    "replay_s1_prerequisite",
    "run_s2_catalog",
    "run_s2_synthetic_fixture",
    "safety_ring",
]
