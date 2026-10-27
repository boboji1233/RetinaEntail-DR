"""FGADR archive sealing, safe extraction, and deterministic target audit.

This Science S7-P1 utility is deliberately upstream of controlled-twin
construction.  It can inspect and extract one explicitly hash-bound ZIP,
validate source/mask identity, and freeze component membership.  It never
creates edited images, runs a model, opens a locked-test root, or computes a
performance endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

import numpy as np
import yaml
from PIL import Image

from retinaentail.data.lesion_catalog import (
    FOVGeometry,
    SparseComponent,
    connected_components,
    detect_fov,
    patch_geometry,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S7_P1_FGADR_IDENTITY_AND_TARGET_MANIFEST_V1"
CLASS_ORDER: Final = ("HE", "EX", "MA", "SE")
ALL_MASK_CLASSES: Final = ("HE", "EX", "MA", "SE", "IRMA", "NV")
READY_STATUS: Final = "READY_FOR_OFFICIAL_ARCHIVE_NOT_RUN"
PASS_STATUS: Final = "PASS_IDENTITY_AND_TARGET_MANIFEST"


class S7FGADRError(RuntimeError):
    """Raised when archive, identity, or coordinate evidence is ambiguous."""


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S7FGADRError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise S7FGADRError(f"{name} keys must be strings")
        result[key] = item
    return result


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise S7FGADRError(f"{name} must be a sequence")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S7FGADRError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7FGADRError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7FGADRError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < minimum:
        raise S7FGADRError(f"{name} must be finite and >= {minimum}")
    return rendered


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise S7FGADRError(f"{name} must be boolean")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise S7FGADRError(f"{name} must be a lowercase SHA-256")
    return digest


def _safe_relative(value: object, name: str) -> str:
    text = _string(value, name)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != text
    ):
        raise S7FGADRError(f"{name} must be a normalized relative POSIX path")
    return text.rstrip("/")


def _load_config(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise S7FGADRError(f"config is not a regular file: {path}")
    raw = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    config = _mapping(raw, "config")
    if _string(config.get("experiment_id"), "experiment ID") != EXPERIMENT_ID:
        raise S7FGADRError("unexpected experiment ID")
    if _string(config.get("schema_version"), "schema version") != SCHEMA_VERSION:
        raise S7FGADRError("unexpected schema version")
    return config


def _member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise S7FGADRError("unsafe ZIP member spelling")
    normalized = _safe_relative(name.rstrip("/"), "ZIP member path")
    return f"{normalized}/" if info.is_dir() else normalized


def _member_kind(info: zipfile.ZipInfo) -> str:
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        return "SYMLINK"
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        return "SPECIAL"
    return "DIRECTORY" if info.is_dir() else "REGULAR_FILE"


def inspect_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_byte_size: int,
    maximum_compression_ratio: float,
    maximum_declared_bytes: int,
    allowed_compression_methods: frozenset[int],
) -> dict[str, object]:
    """Inspect ZIP central-directory metadata without opening member content."""

    expected_digest = _sha256(expected_sha256, "archive SHA-256")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise S7FGADRError("archive must be one regular non-symlink file")
    actual_size = archive_path.stat().st_size
    actual_hash = sha256_file(archive_path)
    if actual_size != expected_byte_size or actual_hash != expected_digest:
        raise S7FGADRError("archive byte size or SHA-256 drift")
    seen: set[str] = set()
    records: list[dict[str, object]] = []
    declared_bytes = 0
    file_count = 0
    with zipfile.ZipFile(archive_path) as bundle:
        for info in bundle.infolist():
            name = _member_name(info)
            collision_key = name.rstrip("/").casefold()
            if collision_key in seen:
                raise S7FGADRError(f"duplicate normalized ZIP path: {name}")
            seen.add(collision_key)
            if info.flag_bits & 1:
                raise S7FGADRError(f"encrypted ZIP member is forbidden: {name}")
            kind = _member_kind(info)
            if kind not in {"DIRECTORY", "REGULAR_FILE"}:
                raise S7FGADRError(f"link or special ZIP member is forbidden: {name}")
            if info.compress_type not in allowed_compression_methods:
                raise S7FGADRError(f"unsupported ZIP compression method: {name}")
            if not info.is_dir():
                file_count += 1
                declared_bytes += info.file_size
                ratio = info.file_size / max(1, info.compress_size)
                if ratio > maximum_compression_ratio:
                    raise S7FGADRError(f"ZIP compression ratio exceeds contract: {name}")
            records.append(
                {
                    "relative_path": name,
                    "kind": kind,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "compression_method": info.compress_type,
                }
            )
    if declared_bytes > maximum_declared_bytes:
        raise S7FGADRError("ZIP declared uncompressed bytes exceed contract")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ARCHIVE_STRUCTURE",
        "archive": {
            "filename": archive_path.name,
            "byte_size": actual_size,
            "sha256": actual_hash,
        },
        "summary": {
            "member_count": len(records),
            "file_count": file_count,
            "declared_uncompressed_bytes": declared_bytes,
            "duplicate_conflict_count": 0,
            "encrypted_member_count": 0,
            "link_or_special_member_count": 0,
            "path_safety": "PASS",
        },
        "members": records,
        "io_audit": {
            "archive_hash_reads": 1,
            "central_directory_reads": 1,
            "member_content_reads": 0,
            "image_decodes": 0,
            "mask_decodes": 0,
            "twin_writes": 0,
            "model_executions": 0,
            "locked_test_reads": 0,
        },
    }


def _output_root(path: Path, name: str) -> Path:
    if path.exists():
        raise S7FGADRError(f"refusing existing {name}: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve(strict=True)


def _extract_archive(
    archive_path: Path, extraction_root: Path, inspection: Mapping[str, object]
) -> list[dict[str, object]]:
    root = _output_root(extraction_root, "extraction root")
    inspected_members = {
        _string(record.get("relative_path"), "inspected member path"): record
        for item in _sequence(inspection.get("members"), "inspected members")
        for record in (_mapping(item, "inspected member"),)
    }
    extracted: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as bundle:
        for info in bundle.infolist():
            name = _member_name(info)
            expected = inspected_members.get(name)
            if expected is None:
                raise S7FGADRError("central-directory membership changed before extraction")
            relative = PurePosixPath(name.rstrip("/"))
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = target.parent.resolve(strict=True)
            try:
                resolved_parent.relative_to(root)
            except ValueError as error:
                raise S7FGADRError("extraction target escaped extraction root") from error
            if info.is_dir():
                target.mkdir(exist_ok=True)
                continue
            if target.exists():
                raise S7FGADRError(f"refusing existing extraction target: {target}")
            digest = hashlib.sha256()
            written = 0
            with bundle.open(info, "r") as source, target.open("xb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            if written != info.file_size:
                raise S7FGADRError(f"extracted size mismatch: {name}")
            extracted.append(
                {
                    "source_relative_path": name,
                    "byte_size": written,
                    "sha256": digest.hexdigest(),
                    "crc32": _string(expected.get("crc32"), "member CRC32"),
                }
            )
    expected_file_count = sum(
        _string(record.get("kind"), "inspected member kind") == "REGULAR_FILE"
        for record in inspected_members.values()
    )
    if len(extracted) != expected_file_count:
        raise S7FGADRError("extraction file count mismatch")
    return sorted(extracted, key=lambda record: cast(str, record["source_relative_path"]))


def _under_prefix(relative: str, prefix: str) -> str | None:
    normalized = f"{prefix.rstrip('/')}/"
    return relative[len(normalized) :] if relative.startswith(normalized) else None


def _extension_set(value: object, name: str) -> frozenset[str]:
    result = frozenset(_string(item, name).lower() for item in _sequence(value, name))
    if not result or any(not suffix.startswith(".") for suffix in result):
        raise S7FGADRError(f"{name} must contain file suffixes")
    return result


def _inventory_layout(
    records: Sequence[Mapping[str, object]], layout: Mapping[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, dict[str, object]]]]:
    image_prefix = _safe_relative(layout.get("image_prefix"), "image prefix")
    image_extensions = _extension_set(layout.get("image_extensions"), "image extensions")
    mask_extensions = _extension_set(layout.get("mask_extensions"), "mask extensions")
    prefix_map = _mapping(layout.get("mask_prefixes"), "mask prefixes")
    if set(prefix_map) != set(ALL_MASK_CLASSES):
        raise S7FGADRError("mask prefixes must bind exactly HE, EX, MA, SE, IRMA, and NV")
    mask_prefixes = {
        lesion: _safe_relative(prefix_map[lesion], f"{lesion} mask prefix")
        for lesion in ALL_MASK_CLASSES
    }
    images: dict[str, dict[str, object]] = {}
    masks: dict[str, dict[str, dict[str, object]]] = {lesion: {} for lesion in ALL_MASK_CLASSES}
    assigned: set[str] = set()
    for item in records:
        record = dict(item)
        relative = _string(record.get("source_relative_path"), "extracted path")
        image_tail = _under_prefix(relative, image_prefix)
        if image_tail is not None and Path(image_tail).suffix.lower() in image_extensions:
            source_id = Path(image_tail).stem
            if not source_id or source_id in images:
                raise S7FGADRError("official source ID is empty or duplicated")
            images[source_id] = record
            assigned.add(relative)
            continue
        for lesion, prefix in mask_prefixes.items():
            mask_tail = _under_prefix(relative, prefix)
            if mask_tail is None or Path(mask_tail).suffix.lower() not in mask_extensions:
                continue
            source_id = Path(mask_tail).stem
            if not source_id or source_id in masks[lesion]:
                raise S7FGADRError(f"{lesion} mask source ID is empty or duplicated")
            masks[lesion][source_id] = record
            assigned.add(relative)
            break
    unassigned = [
        _string(record.get("source_relative_path"), "extracted path")
        for record in records
        if _string(record.get("source_relative_path"), "extracted path") not in assigned
    ]
    if unassigned and _boolean(layout.get("reject_unassigned_files"), "reject unassigned files"):
        raise S7FGADRError(f"layout contains {len(unassigned)} unassigned files")
    return images, masks


def _resolved_file(root: Path, relative: object) -> Path:
    text = _safe_relative(relative, "extracted relative path")
    candidate = (root / Path(text)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise S7FGADRError("decoded file escaped extraction root") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise S7FGADRError("decoded path is not one regular file")
    return candidate


def _decode_image(path: Path) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise S7FGADRError(f"source image mode is not frozen RGB: {path.name}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise S7FGADRError("source image is not RGB HxWx3")
    return array


def _decode_mask(
    path: Path, *, expected_shape: tuple[int, int], allowed_values: frozenset[int]
) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
    with Image.open(path) as image:
        if image.mode != "L":
            raise S7FGADRError(f"mask mode is not frozen 8-bit grayscale: {path.name}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.shape != expected_shape:
        raise S7FGADRError("image/mask dimension mismatch")
    values = frozenset(int(value) for value in np.unique(array).tolist())
    if not values.issubset(allowed_values):
        raise S7FGADRError(f"mask contains undeclared values: {sorted(values)}")
    return np.asarray(array != 0, dtype=np.bool_)


def _collect_source_ids(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_image_id", "source_id"} and isinstance(item, str):
                result.add(item)
            result.update(_collect_source_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            result.update(_collect_source_ids(item))
    return result


def _historical_ids(config: Mapping[str, object], project_root: Path) -> set[str]:
    result: set[str] = set()
    for item in _sequence(config.get("historical_source_artifacts"), "historical artifacts"):
        binding = _mapping(item, "historical artifact")
        relative = _safe_relative(binding.get("path"), "historical artifact path")
        path = (project_root / Path(relative)).resolve(strict=True)
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise S7FGADRError("historical artifact escaped project root") from error
        if path.is_symlink() or not path.is_file():
            raise S7FGADRError("historical artifact is not one regular file")
        if sha256_file(path) != _sha256(binding.get("sha256"), "historical artifact SHA-256"):
            raise S7FGADRError(f"historical artifact hash drift: {relative}")
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
        result.update(_collect_source_ids(raw))
    return result


def _ellipse_boundary_distance(
    centroid: tuple[float, float], fov_bbox: tuple[int, int, int, int]
) -> float:
    x0, y0, x1, y1 = fov_bbox
    center_x, center_y = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
    radius_x, radius_y = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    dx, dy = centroid[0] - center_x, centroid[1] - center_y
    distance_from_center = math.hypot(dx, dy)
    if distance_from_center == 0.0:
        return min(radius_x, radius_y)
    normalized_radius = math.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
    return distance_from_center / normalized_radius - distance_from_center


def _touches_fov(component: SparseComponent, fov: FOVGeometry) -> bool:
    if _ellipse_boundary_distance(component.centroid, fov.bbox) <= 1.0:
        return True
    return any(
        not bool(fov.mask[y, x0]) or not bool(fov.mask[y, x1]) for y, x0, x1 in component.runs
    )


def _component_order(source_id: str, lesion: str, component: SparseComponent) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "class": lesion,
                "source_image_id": source_id,
                "area_px": component.area,
                "bbox_xyxy": list(component.bbox),
                "centroid_xy": list(component.centroid),
                "component_mask_sha256": component.component_mask_sha256,
            }
        )
    )


def _rejection_reasons(
    component: SparseComponent,
    *,
    image_size: tuple[int, int],
    fov: FOVGeometry,
    selection: Mapping[str, object],
) -> tuple[str, ...]:
    constraints = _mapping(selection.get("target_constraints"), "target constraints")
    reasons: list[str] = []
    if component.area < _integer(constraints.get("minimum_area_px"), "minimum area", minimum=1):
        reasons.append("AREA_BELOW_FROZEN_MINIMUM")
    if component.area > _integer(constraints.get("maximum_area_px"), "maximum area", minimum=1):
        reasons.append("AREA_ABOVE_FROZEN_MAXIMUM")
    x0, y0, x1, y1 = component.bbox
    if x1 - x0 > _integer(
        constraints.get("maximum_bbox_width_px"), "maximum bbox width", minimum=1
    ) or y1 - y0 > _integer(
        constraints.get("maximum_bbox_height_px"), "maximum bbox height", minimum=1
    ):
        reasons.append("BBOX_ABOVE_FROZEN_MAXIMUM")
    width, height = image_size
    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        reasons.append("IMAGE_BOUNDARY_TOUCH")
    if _touches_fov(component, fov):
        reasons.append("FOV_BOUNDARY_TOUCH")
    if _ellipse_boundary_distance(component.centroid, fov.bbox) < _number(
        constraints.get("minimum_fov_margin_px"), "minimum FOV margin"
    ):
        reasons.append("FOV_MARGIN_BELOW_FROZEN_MINIMUM")
    patch_size_items = _sequence(selection.get("patch_size_wh"), "patch size")
    if len(patch_size_items) != 2:
        raise S7FGADRError("patch size must contain width and height")
    geometry = patch_geometry(
        image_size=image_size,
        centroid=component.centroid,
        patch_size=(
            _integer(patch_size_items[0], "patch width", minimum=1),
            _integer(patch_size_items[1], "patch height", minimum=1),
        ),
    )
    if geometry.padding_ltrb != (0, 0, 0, 0):
        reasons.append("PATCH_PADDING_PRESENT")
    request_x0, request_y0, request_x1, request_y1 = geometry.requested_bbox
    if x0 < request_x0 or y0 < request_y0 or x1 > request_x1 or y1 > request_y1:
        reasons.append("TARGET_NOT_FULLY_CONTAINED")
    return tuple(reasons)


def _build_identity_and_targets(
    *,
    config: Mapping[str, object],
    extraction_root: Path,
    extraction_records: Sequence[Mapping[str, object]],
    project_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    identity = _mapping(config.get("identity"), "identity")
    labels = _mapping(config.get("labels"), "labels")
    selection = _mapping(config.get("selection"), "selection")
    images, masks = _inventory_layout(extraction_records, _mapping(config.get("layout"), "layout"))
    expected_sources = _integer(
        identity.get("expected_source_images"), "expected sources", minimum=1
    )
    if len(images) != expected_sources:
        raise S7FGADRError(f"expected {expected_sources} source images, observed {len(images)}")
    source_ids = set(images)
    for lesion in CLASS_ORDER:
        if set(masks[lesion]) != source_ids:
            raise S7FGADRError(f"required {lesion} labels are incomplete; missing is unknown")
    allowed_values = frozenset(
        _integer(item, "mask value")
        for item in _sequence(labels.get("allowed_binary_values"), "allowed mask values")
    )
    historical_ids = _historical_ids(config, project_root)
    historical_intersection = sorted(source_ids & historical_ids)
    if historical_intersection:
        raise S7FGADRError("FGADR source IDs intersect historical S7 roles/results/witnesses")
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for source_id, record in images.items():
        duplicate_groups[_sha256(record.get("sha256"), "image SHA-256")].append(source_id)
    representatives = {
        min(group, key=lambda item: hashlib.sha256(item.encode("utf-8")).hexdigest())
        for group in duplicate_groups.values()
    }
    duplicate_exclusions = sorted(source_ids - representatives)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    decode_counts: Counter[str] = Counter()
    ordered_sources = sorted(
        representatives, key=lambda item: hashlib.sha256(item.encode()).hexdigest()
    )
    for source_id in ordered_sources:
        image_record = images[source_id]
        image_path = _resolved_file(extraction_root, image_record["source_relative_path"])
        image = _decode_image(image_path)
        decode_counts["images"] += 1
        height, width = image.shape[:2]
        fov = detect_fov(
            image,
            foreground_max_rgb_min=_integer(
                selection.get("fov_foreground_max_rgb_min"), "FOV threshold", minimum=1
            ),
        )
        for lesion in CLASS_ORDER:
            mask_record = masks[lesion][source_id]
            mask_path = _resolved_file(extraction_root, mask_record["source_relative_path"])
            mask = _decode_mask(
                mask_path, expected_shape=(height, width), allowed_values=allowed_values
            )
            decode_counts["masks"] += 1
            components = sorted(
                connected_components(mask, connectivity=8),
                key=lambda component: _component_order(source_id, lesion, component),
            )
            selected = False
            if not components:
                rejected.append(
                    {
                        "source_image_id": source_id,
                        "lesion_class": lesion,
                        "component_order_key": None,
                        "reasons": ["NO_POSITIVE_COMPONENT"],
                    }
                )
            for component in components:
                order_key = _component_order(source_id, lesion, component)
                reasons = list(
                    _rejection_reasons(
                        component,
                        image_size=(width, height),
                        fov=fov,
                        selection=selection,
                    )
                )
                if not reasons and selected:
                    reasons.append("NOT_SELECTED_AFTER_SOURCE_CLASS_LIMIT")
                base = {
                    "source_image_id": source_id,
                    "lesion_class": lesion,
                    "component_order_key": order_key,
                    "area_px": component.area,
                    "bbox_xyxy": list(component.bbox),
                    "centroid_xy": list(component.centroid),
                    "component_mask_sha256": component.component_mask_sha256,
                    "source_relative_path": image_record["source_relative_path"],
                    "source_sha256": image_record["sha256"],
                    "mask_relative_path": mask_record["source_relative_path"],
                    "mask_sha256": mask_record["sha256"],
                    "image_size_wh": [width, height],
                }
                if reasons:
                    rejected.append({**base, "reasons": reasons})
                else:
                    selected = True
                    target_id = sha256_bytes(canonical_json_bytes(cast(dict[str, JSONValue], base)))
                    accepted.append({**base, "target_id": target_id})
    sparse_label_counts = {lesion: len(masks[lesion]) for lesion in ("IRMA", "NV")}
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": PASS_STATUS,
        "source_count": len(images),
        "unique_source_count_after_exact_duplicate_collapse": len(representatives),
        "exact_duplicate_group_count": sum(len(group) > 1 for group in duplicate_groups.values()),
        "exact_duplicate_excluded_source_ids": duplicate_exclusions,
        "required_mask_counts": {lesion: len(masks[lesion]) for lesion in CLASS_ORDER},
        "sparse_other_mask_counts": sparse_label_counts,
        "historical_source_intersection_count": 0,
        "historical_source_intersections": [],
        "coordinate_audit": {
            "image_mask_dimension_mismatches": 0,
            "mask_value_semantic_failures": 0,
            "connectivity": 8,
            "coordinate_relation": "IDENTICAL_PIXEL_GRID",
        },
        "io_audit": {
            "image_decodes": decode_counts["images"],
            "mask_decodes": decode_counts["masks"],
            "twin_writes": 0,
            "model_executions": 0,
            "locked_test_reads": 0,
        },
    }
    target_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_DETERMINISTIC_TARGET_MEMBERSHIP",
        "source_order": "SHA256_OFFICIAL_SOURCE_ID_ASC",
        "class_order": list(CLASS_ORDER),
        "component_connectivity": 8,
        "component_order": "SHA256_CLASS_SOURCE_COMPONENT_GEOMETRY_ASC",
        "maximum_targets_per_source_class": 1,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "downstream_authorization": {
            "controlled_twin_generation": False,
            "model_execution": False,
            "fcr_estimation": False,
            "science_s8": False,
            "final_locked_test": False,
        },
    }
    return audit, target_manifest


def _write_artifact(output_dir: Path, name: str, value: object) -> tuple[str, int]:
    payload = canonical_json_bytes(cast(JSONValue, value))
    write_bytes_fail_safe(output_dir / name, payload)
    return sha256_bytes(payload), len(payload)


def run_identity_audit(
    *, config_path: Path, archive_path: Path, extraction_root: Path, output_dir: Path
) -> dict[str, object]:
    """Run the bound archive-to-target workflow without constructing twins."""

    config = _load_config(config_path)
    if not _boolean(config.get("execution_enabled"), "execution enabled"):
        raise S7FGADRError("real execution remains disabled until archive/layout hashes are bound")
    archive = _mapping(config.get("archive"), "archive")
    limits = _mapping(config.get("archive_limits"), "archive limits")
    methods = frozenset(
        _integer(item, "compression method")
        for item in _sequence(limits.get("allowed_compression_methods"), "compression methods")
    )
    inspection = inspect_archive(
        archive_path,
        expected_sha256=_sha256(archive.get("sha256"), "archive SHA-256"),
        expected_byte_size=_integer(archive.get("byte_size"), "archive byte size", minimum=1),
        maximum_compression_ratio=_number(
            limits.get("maximum_compression_ratio"), "maximum compression ratio", minimum=1.0
        ),
        maximum_declared_bytes=_integer(
            limits.get("maximum_declared_uncompressed_bytes"),
            "maximum declared bytes",
            minimum=1,
        ),
        allowed_compression_methods=methods,
    )
    output = _output_root(output_dir, "audit output directory")
    records = _extract_archive(archive_path, extraction_root, inspection)
    extraction_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_SAFE_EXTRACTION",
        "archive_sha256": _mapping(inspection["archive"], "archive")["sha256"],
        "file_count": len(records),
        "files": records,
        "path_safety": "PASS",
        "io_audit": {"locked_test_reads": 0, "twin_writes": 0, "model_executions": 0},
    }
    identity_audit, target_manifest = _build_identity_and_targets(
        config=config,
        extraction_root=extraction_root.resolve(strict=True),
        extraction_records=records,
        project_root=config_path.resolve(strict=True).parents[2],
    )
    artifacts: dict[str, dict[str, object]] = {}
    for name, value in (
        ("archive-structure.json", inspection),
        ("extraction-manifest.json", extraction_manifest),
        ("identity-audit.json", identity_audit),
        ("target-manifest.json", target_manifest),
    ):
        digest, byte_size = _write_artifact(output, name, value)
        artifacts[name] = {"sha256": digest, "byte_size": byte_size}
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": PASS_STATUS,
        "config_sha256": sha256_file(config_path),
        "archive_sha256": _mapping(inspection["archive"], "archive")["sha256"],
        "artifacts": artifacts,
        "prohibited_outputs_observed": [],
    }
    digest, byte_size = _write_artifact(output, "artifact-manifest.json", artifact_manifest)
    artifacts["artifact-manifest.json"] = {"sha256": digest, "byte_size": byte_size}
    return {
        "status": PASS_STATUS,
        "output_dir": str(output),
        "extraction_root": str(extraction_root.resolve(strict=True)),
        "artifact_hashes": {
            name: _string(record["sha256"], "artifact SHA-256")
            for name, record in artifacts.items()
        },
    }


def preflight(config_path: Path) -> dict[str, object]:
    """Validate the committed non-executable template without data access."""

    config = _load_config(config_path)
    execution_enabled = _boolean(config.get("execution_enabled"), "execution enabled")
    archive = _mapping(config.get("archive"), "archive")
    layout = _mapping(config.get("layout"), "layout")
    bound = all(archive.get(key) is not None for key in ("sha256", "byte_size")) and all(
        layout.get(key) is not None for key in ("image_prefix", "mask_prefixes")
    )
    if execution_enabled or bound:
        raise S7FGADRError("committed preflight template must remain unbound and non-executable")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS,
        "execution_enabled": False,
        "archive_bound": False,
        "layout_bound": False,
        "data_reads": 0,
        "image_decodes": 0,
        "mask_decodes": 0,
        "twin_writes": 0,
        "model_executions": 0,
        "locked_test_reads": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--archive", type=Path, required=True)
    run_parser.add_argument("--extraction-root", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.config)
        else:
            result = run_identity_audit(
                config_path=args.config,
                archive_path=args.archive,
                extraction_root=args.extraction_root,
                output_dir=args.output_dir,
            )
    except (OSError, S7FGADRError, zipfile.BadZipFile) as error:
        print(f"Science S7-P1 blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
