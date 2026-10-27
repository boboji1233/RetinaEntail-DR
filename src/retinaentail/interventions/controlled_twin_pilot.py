"""Science S3 deterministic HE/EX controlled-twin pilot.

This module consumes the immutable Science S2 component catalog, reads only
TRAIN/CALIBRATION source material from an explicitly authorized IDRiD root, and
constructs paired sham/removed patches for responsible-person QC.  It does not
run RetinaEntail, estimate FCR, access LOCKED_TEST pixels, or create a formal
ambiguity benchmark.
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
from pathlib import Path
from typing import Final, TypeAlias, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from retinaentail.data.lesion_catalog import (
    JSONValue,
    S1Replay,
    S2Config,
    S2DataError,
    S2GateError,
    SparseComponent,
    _component_mask_in_patch,
    _finite_float,
    _integer,
    _load_rgb_strict,
    _mapping,
    _project_file,
    _resolve_under_root,
    _safe_relative,
    _sha256,
    _string,
    connected_components,
    decode_idrid_mask,
    detect_fov,
    extract_fixed_patch,
    patch_geometry,
    replay_s1_prerequisite,
)
from retinaentail.data.license_gate import (
    DataUsePurpose,
    authorize_dataset_access,
    load_license_review_record,
    load_registry_license_metadata,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import ManifestRecordV2, load_dataset_adapter_spec
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    rgb_png_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

S3_SCHEMA_VERSION: Final = "1.0.0"
S3_EXPERIMENT_ID: Final = "S3_HE_EX_TWIN_PILOT"
DEVELOPMENT_SPLITS: Final = ("TRAIN", "CALIBRATION")
LESION_CLASSES: Final = ("HE", "EX")
METHODS: Final = ("TELEA", "NAVIER_STOKES", "DONOR_POISSON")
LOCKED_SPLIT: Final = "LOCKED_TEST"

Offset: TypeAlias = tuple[int, int]


class S3GateError(RuntimeError):
    """Raised before dataset-root I/O when an S2/S3 contract cannot be replayed."""


class S3DataError(RuntimeError):
    """Raised after authorization when S3 input or output fails closed."""


class TwinRejected(RuntimeError):
    """A frozen constructibility rejection that remains in the pilot ledger."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _offsets(value: object, name: str) -> tuple[Offset, ...]:
    if not isinstance(value, list) or not value:
        raise S3GateError(f"{name} must be a non-empty list")
    result: list[Offset] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(number, bool) or not isinstance(number, int) for number in item)
        ):
            raise S3GateError(f"{name}[{index}] must contain two integers")
        dx, dy = cast(list[int], item)
        if dx == 0 and dy == 0:
            raise S3GateError(f"{name}[{index}] cannot be zero")
        result.append((dx, dy))
    if len(set(result)) != len(result):
        raise S3GateError(f"{name} contains duplicate offsets")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class S3Config:
    """Validated executable subset of the frozen Science S3 YAML."""

    path: Path
    payload: dict[str, object]
    config_sha256: str
    dataset_id: str
    dataset_version: str
    artifact_root: str
    upstream: dict[str, object]
    seed: int
    class_maximum: int
    split_quotas: dict[str, int]
    minimum_area: int
    maximum_area: int
    maximum_bbox_width: int
    maximum_bbox_height: int
    minimum_fov_margin: float
    methods: tuple[str, ...]
    method_settings: dict[str, dict[str, object]]
    target_dilation: int
    lesion_exclusion_dilation: int
    sham_offsets: tuple[Offset, ...]
    donor_offsets: tuple[Offset, ...]
    qc_columns: int
    qc_panel_size: int
    rejection_taxonomy: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> S3Config:
        decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S3 config")
        if _string(root.get("schema_version"), "schema_version") != "2.0.0":
            raise S3GateError("S3 config schema_version must be 2.0.0")
        if _string(root.get("experiment_id"), "experiment_id") != S3_EXPERIMENT_ID:
            raise S3GateError("S3 experiment_id mismatch")
        if root.get("status") != "IMPLEMENTING" or root.get("execution_enabled") is not True:
            raise S3GateError("only the frozen IMPLEMENTING run config is executable")
        if root.get("scope") != "TRAIN_AND_CALIBRATION_ONLY":
            raise S3GateError("S3 scope must be TRAIN_AND_CALIBRATION_ONLY")
        if root.get("splits") != list(DEVELOPMENT_SPLITS):
            raise S3GateError("S3 splits must be exactly TRAIN/CALIBRATION")
        if root.get("locked_test_allowed") is not False:
            raise S3GateError("LOCKED_TEST must remain prohibited")
        if root.get("lesion_classes") != list(LESION_CLASSES):
            raise S3GateError("S3 lesion classes must be exactly HE/EX")
        if root.get("paired_outputs") != ["x_plus_sham", "x_minus_removed"]:
            raise S3GateError("S3 paired output contract mismatch")

        upstream = _mapping(root.get("upstream"), "upstream")
        selection = _mapping(root.get("selection"), "selection")
        assignment = _mapping(root.get("method_assignment"), "method_assignment")
        edit = _mapping(root.get("edit"), "edit")
        qc = _mapping(root.get("qc"), "qc")
        if selection.get("order") != "SHA256_SEED_CLASS_SPLIT_COMPONENT_ID_ASC":
            raise S3GateError("selection order is not frozen")
        if (
            selection.get("replacement_policy")
            != "NEXT_ELIGIBLE_AFTER_AUTOMATED_CONSTRUCTIBILITY_REJECTION"
        ):
            raise S3GateError("replacement policy is not frozen")
        if assignment.get("order") != "SHA256_SEED_COMPONENT_ID_MOD_METHOD_COUNT":
            raise S3GateError("method assignment is not frozen")
        raw_methods = assignment.get("methods")
        if raw_methods != list(METHODS):
            raise S3GateError("S3 method set/order mismatch")
        method_settings = {
            method: _mapping(assignment.get(method), f"method {method}") for method in METHODS
        }
        implementation_ids = {
            "TELEA": "TELEA_STYLE_BOUNDARY_WEIGHTED_NUMPY_V1",
            "NAVIER_STOKES": "NAVIER_STOKES_STYLE_ISOPHOTE_DIFFUSION_NUMPY_V1",
            "DONOR_POISSON": "DONOR_TEXTURE_POISSON_JACOBI_NUMPY_V1",
        }
        for method, expected in implementation_ids.items():
            if method_settings[method].get("implementation_id") != expected:
                raise S3GateError(f"{method} implementation_id mismatch")
        raw_quotas = _mapping(selection.get("accepted_quota_per_class_split"), "accepted quotas")
        split_quotas = {
            split: _integer(raw_quotas.get(split), f"quota {split}", minimum=0)
            for split in DEVELOPMENT_SPLITS
        }
        class_maximum = _integer(
            selection.get("accepted_target_per_class_max"), "class maximum", minimum=1
        )
        if sum(split_quotas.values()) > class_maximum or class_maximum > 20:
            raise S3GateError("S3 per-class quotas exceed the frozen maximum")
        if edit.get("patch_size_wh") != [256, 256]:
            raise S3GateError("S3 patch size must remain 256x256")
        taxonomy = root.get("rejection_reason_taxonomy")
        if not isinstance(taxonomy, list) or not all(
            isinstance(item, str) and item for item in taxonomy
        ):
            raise S3GateError("rejection taxonomy must be a string list")
        config_bytes = canonical_json_bytes(cast(dict[str, JSONValue], root))
        return cls(
            path=path,
            payload=root,
            config_sha256=sha256_bytes(config_bytes),
            dataset_id=_string(root.get("dataset_id"), "dataset_id"),
            dataset_version=_string(root.get("dataset_version"), "dataset_version"),
            artifact_root=_safe_relative(root.get("artifact_root"), "artifact_root"),
            upstream=upstream,
            seed=_integer(selection.get("seed"), "selection seed", minimum=0),
            class_maximum=class_maximum,
            split_quotas=split_quotas,
            minimum_area=_integer(selection.get("minimum_area_px"), "minimum area", minimum=1),
            maximum_area=_integer(selection.get("maximum_area_px"), "maximum area", minimum=1),
            maximum_bbox_width=_integer(
                selection.get("maximum_bbox_width_px"), "maximum bbox width", minimum=1
            ),
            maximum_bbox_height=_integer(
                selection.get("maximum_bbox_height_px"), "maximum bbox height", minimum=1
            ),
            minimum_fov_margin=_finite_float(
                selection.get("minimum_fov_margin_px"), "minimum FOV margin"
            ),
            methods=METHODS,
            method_settings=method_settings,
            target_dilation=_integer(
                edit.get("target_mask_dilation_px"), "target dilation", minimum=1
            ),
            lesion_exclusion_dilation=_integer(
                edit.get("lesion_exclusion_dilation_px"),
                "lesion exclusion dilation",
                minimum=1,
            ),
            sham_offsets=_offsets(edit.get("sham_offset_candidates_xy"), "sham offset candidates"),
            donor_offsets=_offsets(
                edit.get("donor_offset_candidates_xy"), "donor offset candidates"
            ),
            qc_columns=_integer(qc.get("contact_sheet_columns"), "QC columns", minimum=1),
            qc_panel_size=_integer(qc.get("panel_size_px"), "QC panel size", minimum=64),
            rejection_taxonomy=frozenset(cast(list[str], taxonomy)),
        )

    def implementation_id(self, method: str) -> str:
        return _string(
            self.method_settings[method].get("implementation_id"),
            f"{method} implementation_id",
        )


@dataclass(frozen=True, slots=True)
class S3Upstream:
    s1: S1Replay
    s2_config: S2Config
    catalog_records: tuple[dict[str, object], ...]
    hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class QCCell:
    twin_id: str
    component_id: str
    split: str
    lesion_class: str
    method: str
    source: NDArray[np.uint8]
    target_component: NDArray[np.bool_]
    target_edit: NDArray[np.bool_]
    sham_edit: NDArray[np.bool_]
    x_plus_sham: NDArray[np.uint8]
    x_minus_removed: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TwinBuild:
    record: dict[str, JSONValue]
    artifacts: dict[str, bytes]
    qc_cell: QCCell


@dataclass(frozen=True, slots=True)
class S3RunResult:
    status: str
    output_dir: Path
    accepted_count: int
    rejected_count: int
    accepted_counts: dict[str, int]
    method_counts: dict[str, int]
    locked_test_root_reads: int
    artifact_hashes: dict[str, str]
    audit: dict[str, JSONValue]


def _load_json_mapping(path: Path, name: str) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    try:
        return _mapping(decoded, name)
    except S2GateError as error:
        raise S3GateError(str(error)) from error


def _replay_upstream(project_root: Path, config: S3Config) -> S3Upstream:
    path_fields = {
        "s2_config": "s2_config_sha256",
        "s2_catalog": "s2_catalog_sha256",
        "s2_generated_audit": "s2_generated_audit_sha256",
        "s2_artifact_manifest": "s2_artifact_manifest_sha256",
        "s2_qc_record": "s2_qc_record_sha256",
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for path_name, hash_name in path_fields.items():
        try:
            path = _project_file(project_root, config.upstream.get(path_name), path_name)
            expected = _sha256(config.upstream.get(hash_name), hash_name)
        except S2GateError as error:
            raise S3GateError(str(error)) from error
        actual = sha256_file(path)
        if actual != expected:
            raise S3GateError(f"upstream hash mismatch: {path_name}")
        paths[path_name] = path
        hashes[path_name] = actual

    s2_state: object = yaml.safe_load(paths["s2_config"].read_text(encoding="utf-8"))
    s2_state_mapping = _mapping(s2_state, "S2 state config")
    s2_result = _mapping(s2_state_mapping.get("result"), "S2 result")
    if (
        s2_state_mapping.get("status") != "PASS"
        or s2_result.get("human_qc_confirmed") is not True
        or s2_result.get("qc_review_mode") != "AI_ASSISTED_PLUS_RESPONSIBLE_PERSON_CONFIRMATION"
    ):
        raise S3GateError("S2 committed PASS/QC state replay failed")

    catalog = _load_json_mapping(paths["s2_catalog"], "S2 catalog")
    if (
        catalog.get("dataset_id") != config.dataset_id
        or catalog.get("dataset_version") != config.dataset_version
        or catalog.get("scope") != "TRAIN_AND_CALIBRATION_ONLY"
    ):
        raise S3GateError("S2 catalog dataset/scope binding failed")
    raw_records = catalog.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != _integer(
        config.upstream.get("expected_component_count"), "expected component count", minimum=1
    ):
        raise S3GateError("S2 catalog component count mismatch")
    records: list[dict[str, object]] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise S3GateError(f"S2 catalog record {index} is not an object")
        record = {str(key): value for key, value in item.items()}
        if record.get("split") not in DEVELOPMENT_SPLITS:
            raise S3GateError("S2 catalog contains a non-development split")
        records.append(record)

    generated_audit = _load_json_mapping(paths["s2_generated_audit"], "S2 generated audit")
    audit_summary = _mapping(generated_audit.get("summary"), "S2 audit summary")
    if (
        generated_audit.get("status") != "READY_FOR_HUMAN_QC"
        or audit_summary.get("component_count") != len(records)
        or audit_summary.get("locked_test_root_reads")
        != _integer(
            config.upstream.get("expected_locked_test_root_reads"),
            "expected locked-test reads",
            minimum=0,
        )
    ):
        raise S3GateError("S2 generated audit replay failed")

    artifact_manifest = _load_json_mapping(paths["s2_artifact_manifest"], "S2 artifact manifest")
    manifest_files = artifact_manifest.get("files")
    if not isinstance(manifest_files, list):
        raise S3GateError("S2 artifact manifest file list is invalid")
    manifest_hashes = {
        item.get("relative_path"): item.get("sha256")
        for item in manifest_files
        if isinstance(item, Mapping)
    }
    if (
        manifest_hashes.get("component-catalog-v1.json") != hashes["s2_catalog"]
        or manifest_hashes.get("audit.json") != hashes["s2_generated_audit"]
    ):
        raise S3GateError("S2 artifact manifest does not bind catalog/audit")

    qc_record = _load_json_mapping(paths["s2_qc_record"], "S2 QC record")
    qc_evidence = _mapping(qc_record.get("automated_evidence"), "S2 QC evidence")
    if (
        qc_record.get("decision") != "PASS"
        or qc_record.get("decision_scope") != "SCIENCE_S2_CATALOG_GATE_ONLY"
        or qc_evidence.get("artifact_manifest_sha256") != hashes["s2_artifact_manifest"]
    ):
        raise S3GateError("S2 responsible confirmation replay failed")

    try:
        s2_config = S2Config.load(paths["s2_config"])
        s1 = replay_s1_prerequisite(project_root, s2_config)
    except (S2GateError, S2DataError) as error:
        raise S3GateError(f"S1/S2 prerequisite replay failed: {error}") from error
    return S3Upstream(
        s1=s1,
        s2_config=s2_config,
        catalog_records=tuple(records),
        hashes=hashes,
    )


def _eligibility_reason(record: Mapping[str, object], config: S3Config) -> str:
    if record.get("inclusion_status") != "INCLUDED":
        return "S2_NOT_INCLUDED"
    area = _integer(record.get("area_px"), "component area", minimum=1)
    if area < config.minimum_area:
        return "AREA_BELOW_FROZEN_MINIMUM"
    if area > config.maximum_area:
        return "AREA_ABOVE_FROZEN_MAXIMUM"
    bbox = record.get("bbox_xyxy")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
    ):
        raise S3GateError("catalog bbox is invalid")
    x0, y0, x1, y1 = cast(list[int], bbox)
    if x1 - x0 > config.maximum_bbox_width or y1 - y0 > config.maximum_bbox_height:
        return "BBOX_ABOVE_FROZEN_MAXIMUM"
    if record.get("touches_image_boundary") is True:
        return "IMAGE_BOUNDARY_TOUCH"
    if record.get("touches_fov_boundary") is True:
        return "FOV_BOUNDARY_TOUCH"
    margin = _finite_float(record.get("distance_to_fov_boundary_px"), "FOV margin")
    if margin < config.minimum_fov_margin:
        return "FOV_MARGIN_BELOW_FROZEN_MINIMUM"
    patch = _mapping(record.get("patch"), "catalog patch")
    if patch.get("padding_ltrb") != [0, 0, 0, 0]:
        return "PATCH_PADDING_PRESENT"
    return "ELIGIBLE"


def _selection_key(config: S3Config, record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        (
            f"{config.seed}:{_string(record.get('lesion_class'), 'lesion class')}:"
            f"{_string(record.get('split'), 'split')}:"
            f"{_string(record.get('component_id'), 'component id')}"
        ).encode()
    ).hexdigest()


def _assigned_method(config: S3Config, component_id: str) -> str:
    digest = hashlib.sha256(f"{config.seed}:{component_id}".encode()).digest()
    return config.methods[int.from_bytes(digest[:8], "big") % len(config.methods)]


def _binary_dilate(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    if mask.dtype != np.bool_ or mask.ndim != 2 or radius < 0:
        raise S3DataError("binary dilation contract failed")
    result = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = source_y0 + dy
            target_y1 = source_y1 + dy
            target_x0 = source_x0 + dx
            target_x1 = source_x1 + dx
            result[target_y0:target_y1, target_x0:target_x1] |= mask[
                source_y0:source_y1, source_x0:source_x1
            ]
    return np.asarray(result, dtype=np.bool_)


def _translate_mask(mask: NDArray[np.bool_], offset: Offset) -> NDArray[np.bool_] | None:
    dx, dy = offset
    height, width = mask.shape
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        return None
    translated = np.zeros_like(mask)
    translated[
        source_y0 + dy : source_y1 + dy,
        source_x0 + dx : source_x1 + dx,
    ] = mask[source_y0:source_y1, source_x0:source_x1]
    if int(np.count_nonzero(translated)) != int(np.count_nonzero(mask)):
        return None
    return translated


def _ordered_offsets(config: S3Config, component_id: str, purpose: str) -> tuple[Offset, ...]:
    offsets = config.sham_offsets if purpose == "sham" else config.donor_offsets
    return tuple(
        sorted(
            offsets,
            key=lambda offset: hashlib.sha256(
                f"{config.seed}:{component_id}:{purpose}:{offset[0]}:{offset[1]}".encode()
            ).hexdigest(),
        )
    )


def _find_offset(
    *,
    config: S3Config,
    component_id: str,
    purpose: str,
    edit_mask: NDArray[np.bool_],
    forbidden: NDArray[np.bool_],
    fov: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], Offset]:
    for offset in _ordered_offsets(config, component_id, purpose):
        translated = _translate_mask(edit_mask, offset)
        if translated is None:
            continue
        if np.any(translated & forbidden) or not np.all(fov[translated]):
            continue
        return translated, offset
    reason = "SHAM_LOCATION_UNAVAILABLE" if purpose == "sham" else "DONOR_LOCATION_UNAVAILABLE"
    raise TwinRejected(reason)


def _mask_sha256(mask: NDArray[np.bool_]) -> str:
    header = f"bool:{mask.shape[0]}:{mask.shape[1]}:".encode()
    return sha256_bytes(header + np.packbits(mask, bitorder="little").tobytes())


def _frontier_fill(
    image: NDArray[np.uint8], mask: NDArray[np.bool_], *, radius: int
) -> NDArray[np.float64]:
    result = np.asarray(image, dtype=np.float64).copy()
    unknown = mask.copy()
    height, width = mask.shape
    while np.any(unknown):
        frontier = unknown & _binary_dilate(~unknown, 1)
        coordinates = np.argwhere(frontier)
        if coordinates.size == 0:
            raise S3DataError("inpainting mask has no reachable boundary")
        values: list[NDArray[np.float64]] = []
        for coordinate in coordinates:
            y, x = int(coordinate[0]), int(coordinate[1])
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            known = ~unknown[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            distance_squared = (yy - y) ** 2 + (xx - x) ** 2
            valid = known & (distance_squared > 0) & (distance_squared <= radius * radius)
            if not np.any(valid):
                valid = known & (distance_squared > 0)
            if not np.any(valid):
                raise S3DataError("inpainting frontier has no known neighbor")
            weights = 1.0 / np.asarray(distance_squared[valid], dtype=np.float64)
            pixels = result[y0:y1, x0:x1][valid]
            values.append(np.sum(pixels * weights[:, None], axis=0) / np.sum(weights))
        for coordinate, value in zip(coordinates, values, strict=True):
            result[int(coordinate[0]), int(coordinate[1])] = value
        unknown[frontier] = False
    return result


def _laplacian(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    padded = np.pad(channel, 1, mode="edge")
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * channel
    )


def _inpaint_telea_style(
    image: NDArray[np.uint8], mask: NDArray[np.bool_], *, radius: int
) -> NDArray[np.uint8]:
    filled = _frontier_fill(image, mask, radius=radius)
    filled[~mask] = image[~mask]
    if not np.all(np.isfinite(filled)):
        raise TwinRejected("NONFINITE_OUTPUT")
    return np.asarray(np.rint(np.clip(filled, 0.0, 255.0)), dtype=np.uint8)


def _inpaint_navier_stokes_style(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    *,
    iterations: int,
    transport_step: float,
    diffusion_step: float,
) -> NDArray[np.uint8]:
    original = np.asarray(image, dtype=np.float64)
    result = _frontier_fill(image, mask, radius=3)
    for _ in range(iterations):
        updated = result.copy()
        for channel_index in range(3):
            channel = result[:, :, channel_index]
            laplacian = _laplacian(channel)
            padded_channel = np.pad(channel, 1, mode="edge")
            gradient_x = (padded_channel[1:-1, 2:] - padded_channel[1:-1, :-2]) / 2.0
            gradient_y = (padded_channel[2:, 1:-1] - padded_channel[:-2, 1:-1]) / 2.0
            padded_laplacian = np.pad(laplacian, 1, mode="edge")
            laplacian_x = (padded_laplacian[1:-1, 2:] - padded_laplacian[1:-1, :-2]) / 2.0
            laplacian_y = (padded_laplacian[2:, 1:-1] - padded_laplacian[:-2, 1:-1]) / 2.0
            transport = -gradient_y * laplacian_x + gradient_x * laplacian_y
            normalized_transport = transport / (np.abs(transport) + 1.0)
            candidate = channel + transport_step * normalized_transport + diffusion_step * laplacian
            updated_channel = updated[:, :, channel_index]
            updated_channel[mask] = candidate[mask]
            updated[:, :, channel_index] = updated_channel
        updated[~mask] = original[~mask]
        result = np.clip(updated, 0.0, 255.0)
    if not np.all(np.isfinite(result)):
        raise TwinRejected("NONFINITE_OUTPUT")
    return np.asarray(np.rint(result), dtype=np.uint8)


def _inpaint_donor_poisson(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    *,
    donor_offset: Offset,
    iterations: int,
) -> NDArray[np.uint8]:
    dx, dy = donor_offset
    original = np.asarray(image, dtype=np.float64)
    donor = np.roll(original, shift=(-dy, -dx), axis=(0, 1))
    result = original.copy()
    result[mask] = donor[mask]
    donor_laplacian = np.stack([_laplacian(donor[:, :, channel]) for channel in range(3)], axis=2)
    for _ in range(iterations):
        updated = result.copy()
        for channel_index in range(3):
            channel = result[:, :, channel_index]
            padded = np.pad(channel, 1, mode="edge")
            neighbor_sum = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            candidate = (neighbor_sum + donor_laplacian[:, :, channel_index]) / 4.0
            updated_channel = updated[:, :, channel_index]
            updated_channel[mask] = candidate[mask]
            updated[:, :, channel_index] = updated_channel
        updated[~mask] = original[~mask]
        result = np.clip(updated, 0.0, 255.0)
    if not np.all(np.isfinite(result)):
        raise TwinRejected("NONFINITE_OUTPUT")
    return np.asarray(np.rint(result), dtype=np.uint8)


def inpaint_candidate(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    *,
    method: str,
    donor_offset: Offset | None = None,
    telea_radius: int = 3,
    navier_iterations: int = 80,
    navier_transport_step: float = 0.015,
    navier_diffusion_step: float = 0.12,
    poisson_iterations: int = 160,
) -> NDArray[np.uint8]:
    """Apply one frozen S3 candidate implementation to a uint8 RGB patch."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise S3DataError("inpainting image must be RGB uint8")
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2] or not np.any(mask):
        raise S3DataError("inpainting mask must be a non-empty aligned boolean array")
    if method == "TELEA":
        output = _inpaint_telea_style(image, mask, radius=telea_radius)
    elif method == "NAVIER_STOKES":
        output = _inpaint_navier_stokes_style(
            image,
            mask,
            iterations=navier_iterations,
            transport_step=navier_transport_step,
            diffusion_step=navier_diffusion_step,
        )
    elif method == "DONOR_POISSON":
        if donor_offset is None:
            raise S3DataError("DONOR_POISSON requires a frozen donor offset")
        output = _inpaint_donor_poisson(
            image,
            mask,
            donor_offset=donor_offset,
            iterations=poisson_iterations,
        )
    else:
        raise S3DataError(f"unsupported S3 method: {method}")
    if output.shape != image.shape or output.dtype != np.uint8:
        raise S3DataError("inpainting output contract failed")
    if not np.array_equal(output[~mask], image[~mask]):
        raise S3DataError("inpainting changed pixels outside its declared edit mask")
    return output


def _method_output(
    *,
    image: NDArray[np.uint8],
    edit_mask: NDArray[np.bool_],
    method: str,
    config: S3Config,
    component_id: str,
    lesion_forbidden: NDArray[np.bool_],
    fov: NDArray[np.bool_],
    purpose: str,
) -> tuple[NDArray[np.uint8], Offset | None]:
    donor_offset: Offset | None = None
    if method == "DONOR_POISSON":
        _, donor_offset = _find_offset(
            config=config,
            component_id=component_id,
            purpose=f"donor-{purpose}",
            edit_mask=edit_mask,
            forbidden=lesion_forbidden | edit_mask,
            fov=fov,
        )
    telea = config.method_settings["TELEA"]
    navier = config.method_settings["NAVIER_STOKES"]
    poisson = config.method_settings["DONOR_POISSON"]
    output = inpaint_candidate(
        image,
        edit_mask,
        method=method,
        donor_offset=donor_offset,
        telea_radius=_integer(telea.get("neighborhood_radius_px"), "Telea radius", minimum=1),
        navier_iterations=_integer(navier.get("iterations"), "Navier iterations", minimum=1),
        navier_transport_step=_finite_float(navier.get("transport_step"), "Navier transport step"),
        navier_diffusion_step=_finite_float(navier.get("diffusion_step"), "Navier diffusion step"),
        poisson_iterations=_integer(poisson.get("iterations"), "Poisson iterations", minimum=1),
    )
    return output, donor_offset


def _component_for_record(
    mask: NDArray[np.bool_], record: Mapping[str, object]
) -> tuple[NDArray[np.bool_], SparseComponent]:
    component_id = _string(record.get("component_id"), "component id")
    try:
        component_index = int(component_id.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise S3DataError("component id does not contain a numeric S2 index") from error
    components = connected_components(mask, connectivity=8)
    if component_index < 1 or component_index > len(components):
        raise S3DataError("component index is outside the decoded S2 mask")
    component = components[component_index - 1]
    expected_hash = _sha256(record.get("component_mask_sha256"), "component mask hash")
    if component.component_mask_sha256 != expected_hash:
        raise S3DataError("component reconstruction hash mismatch")
    expected_bbox = record.get("bbox_xyxy")
    expected_centroid = record.get("centroid_xy")
    if list(component.bbox) != expected_bbox or component.area != record.get("area_px"):
        raise S3DataError("component reconstruction geometry mismatch")
    if not isinstance(expected_centroid, list) or len(expected_centroid) != 2:
        raise S3DataError("catalog centroid is invalid")
    if (
        max(
            abs(component.centroid[0] - float(expected_centroid[0])),
            abs(component.centroid[1] - float(expected_centroid[1])),
        )
        > 1e-12
    ):
        raise S3DataError("component reconstruction centroid mismatch")
    return mask, component


def _code_sha256(project_root: Path) -> str:
    names = (
        "src/retinaentail/interventions/controlled_twin_pilot.py",
        "src/retinaentail/science_s3.py",
        "data_registry/schemas/controlled-twin-pilot-v1.schema.json",
        "data_registry/schemas/science-s3-audit-v1.schema.json",
    )
    entries: list[dict[str, JSONValue]] = []
    for name in names:
        path = project_root / name
        if not path.is_file():
            raise S3GateError(f"S3 code bundle file is missing: {name}")
        entries.append({"relative_path": name, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(cast(list[JSONValue], entries)))


def _build_one(
    *,
    project_root: Path,
    dataset_root: Path,
    config: S3Config,
    upstream: S3Upstream,
    record: Mapping[str, object],
    source_record: ManifestRecordV2,
    root_read_log: list[dict[str, str]],
) -> TwinBuild:
    component_id = _string(record.get("component_id"), "component id")
    split = _string(record.get("split"), "split")
    lesion_class = _string(record.get("lesion_class"), "lesion class")
    if split not in DEVELOPMENT_SPLITS or source_record.split != split:
        raise S3DataError("source split binding failed")
    if source_record.source_image_id != record.get("source_image_id"):
        raise S3DataError("source identity binding failed")
    if source_record.relative_path != record.get(
        "image_relative_path"
    ) or source_record.sha256 != record.get("image_sha256"):
        raise S3DataError("source image provenance binding failed")

    root = dataset_root.resolve(strict=True)
    image_path = _resolve_under_root(root, source_record.relative_path)
    image = _load_rgb_strict(image_path, source_record)
    root_read_log.append(
        {"split": split, "kind": "IMAGE", "source_image_id": source_record.source_image_id}
    )

    mask_relative_path = _safe_relative(record.get("mask_relative_path"), "mask relative path")
    mask_path = _resolve_under_root(root, mask_relative_path)
    if sha256_file(mask_path) != _sha256(record.get("mask_sha256"), "mask hash"):
        raise S3DataError("target mask hash mismatch")
    target_full = decode_idrid_mask(
        mask_path, expected_size=(source_record.width, source_record.height)
    )
    root_read_log.append(
        {
            "split": split,
            "kind": "TARGET_MASK",
            "source_image_id": source_record.source_image_id,
        }
    )
    _, component_object = _component_for_record(target_full, record)
    component = component_object

    non_target_relative_path = _safe_relative(
        record.get("non_target_mask_relative_path"), "non-target mask relative path"
    )
    non_target_path = _resolve_under_root(root, non_target_relative_path)
    if sha256_file(non_target_path) != _sha256(
        record.get("non_target_mask_sha256"), "non-target mask hash"
    ):
        raise S3DataError("non-target mask hash mismatch")
    non_target_full = decode_idrid_mask(
        non_target_path, expected_size=(source_record.width, source_record.height)
    )
    root_read_log.append(
        {
            "split": split,
            "kind": "NON_TARGET_MASK",
            "source_image_id": source_record.source_image_id,
        }
    )

    centroid = record.get("centroid_xy")
    if not isinstance(centroid, list) or len(centroid) != 2:
        raise S3DataError("catalog centroid is invalid")
    geometry = patch_geometry(
        image_size=(source_record.width, source_record.height),
        centroid=(float(centroid[0]), float(centroid[1])),
        patch_size=(256, 256),
    )
    if geometry.to_dict() != record.get("patch"):
        raise S3DataError("S2 patch geometry replay failed")
    source_patch = cast(NDArray[np.uint8], extract_fixed_patch(image, geometry))
    target_component = _component_mask_in_patch(component, geometry)
    target_all_patch = cast(NDArray[np.bool_], extract_fixed_patch(target_full, geometry))
    non_target_patch = cast(NDArray[np.bool_], extract_fixed_patch(non_target_full, geometry))
    combined_lesions = target_all_patch | non_target_patch
    other_lesions = combined_lesions & ~target_component
    target_edit = _binary_dilate(target_component, config.target_dilation)
    if np.any(target_edit & other_lesions):
        raise TwinRejected("TARGET_EDIT_OVERLAPS_OTHER_LESION")
    fov = detect_fov(
        image,
        foreground_max_rgb_min=upstream.s2_config.fov_threshold,
        minimum_axis_coverage_fraction=upstream.s2_config.fov_coverage,
    )
    fov_patch = cast(NDArray[np.bool_], extract_fixed_patch(fov.mask, geometry))
    if not np.all(fov_patch[target_edit]):
        raise S3DataError("frozen eligible target edit leaves the FOV")
    lesion_forbidden = _binary_dilate(combined_lesions, config.lesion_exclusion_dilation)
    sham_edit, sham_offset = _find_offset(
        config=config,
        component_id=component_id,
        purpose="sham",
        edit_mask=target_edit,
        forbidden=lesion_forbidden | target_edit,
        fov=fov_patch,
    )

    method = _assigned_method(config, component_id)
    x_plus, sham_donor = _method_output(
        image=source_patch,
        edit_mask=sham_edit,
        method=method,
        config=config,
        component_id=component_id,
        lesion_forbidden=lesion_forbidden,
        fov=fov_patch,
        purpose="sham",
    )
    x_minus, target_donor = _method_output(
        image=source_patch,
        edit_mask=target_edit,
        method=method,
        config=config,
        component_id=component_id,
        lesion_forbidden=lesion_forbidden & ~target_component,
        fov=fov_patch,
        purpose="target",
    )
    if not np.array_equal(x_plus[target_component], source_patch[target_component]):
        raise S3DataError("sham edit changed the target component")
    plus_changed = int(np.count_nonzero(np.any(x_plus != source_patch, axis=2)))
    minus_changed = int(np.count_nonzero(np.any(x_minus != source_patch, axis=2)))
    if plus_changed == 0 or minus_changed == 0:
        raise TwinRejected("EMPTY_EDIT_EFFECT")

    twin_digest = hashlib.sha256(component_id.encode()).hexdigest()[:12]
    twin_id = f"s3-{lesion_class.lower()}-{split.lower()}-{twin_digest}-{method.lower()}"
    plus_path = f"twins/{twin_id}/x_plus_sham.png"
    minus_path = f"twins/{twin_id}/x_minus_removed.png"
    plus_bytes = rgb_png_bytes(x_plus)
    minus_bytes = rgb_png_bytes(x_minus)
    source_patch_bytes = rgb_png_bytes(source_patch)
    record_payload: dict[str, JSONValue] = {
        "schema_version": S3_SCHEMA_VERSION,
        "twin_id": twin_id,
        "component_id": component_id,
        "source_image_id": source_record.source_image_id,
        "split": split,
        "lesion_class": lesion_class,
        "method": method,
        "method_implementation_id": config.implementation_id(method),
        "seed": config.seed,
        "source": {
            "image_relative_path": source_record.relative_path,
            "image_sha256": source_record.sha256,
            "mask_relative_path": mask_relative_path,
            "mask_sha256": _sha256(record.get("mask_sha256"), "mask hash"),
            "non_target_mask_relative_path": non_target_relative_path,
            "non_target_mask_sha256": _sha256(
                record.get("non_target_mask_sha256"), "non-target mask hash"
            ),
            "source_patch_sha256": sha256_bytes(source_patch_bytes),
            "patch_geometry": geometry.to_dict(),
        },
        "target": {
            "component_mask_sha256": _sha256(
                record.get("component_mask_sha256"), "component mask hash"
            ),
            "component_area_px": _integer(record.get("area_px"), "component area", minimum=1),
            "bbox_xyxy": cast(list[JSONValue], record.get("bbox_xyxy")),
            "centroid_xy": cast(list[JSONValue], record.get("centroid_xy")),
            "edit_mask_sha256": _mask_sha256(target_edit),
            "edit_mask_area_px": int(np.count_nonzero(target_edit)),
            "donor_offset_xy": None if target_donor is None else list(target_donor),
        },
        "sham": {
            "offset_xy": list(sham_offset),
            "edit_mask_sha256": _mask_sha256(sham_edit),
            "edit_mask_area_px": int(np.count_nonzero(sham_edit)),
            "donor_offset_xy": None if sham_donor is None else list(sham_donor),
        },
        "outputs": {
            "x_plus_sham": {
                "relative_path": plus_path,
                "sha256": sha256_bytes(plus_bytes),
                "byte_size": len(plus_bytes),
            },
            "x_minus_removed": {
                "relative_path": minus_path,
                "sha256": sha256_bytes(minus_bytes),
                "byte_size": len(minus_bytes),
            },
        },
        "automated_gate": {
            "complete_pair": True,
            "outside_sham_edit_exact_preservation": bool(
                np.array_equal(x_plus[~sham_edit], source_patch[~sham_edit])
            ),
            "outside_target_edit_exact_preservation": bool(
                np.array_equal(x_minus[~target_edit], source_patch[~target_edit])
            ),
            "target_exact_preservation_in_sham": bool(
                np.array_equal(x_plus[target_component], source_patch[target_component])
            ),
            "other_lesion_overlap_px": int(np.count_nonzero(target_edit & other_lesions)),
            "x_plus_changed_pixel_count": plus_changed,
            "x_minus_changed_pixel_count": minus_changed,
            "finite_uint8_outputs": True,
        },
    }
    return TwinBuild(
        record=record_payload,
        artifacts={plus_path: plus_bytes, minus_path: minus_bytes},
        qc_cell=QCCell(
            twin_id=twin_id,
            component_id=component_id,
            split=split,
            lesion_class=lesion_class,
            method=method,
            source=source_patch,
            target_component=target_component,
            target_edit=target_edit,
            sham_edit=sham_edit,
            x_plus_sham=x_plus,
            x_minus_removed=x_minus,
        ),
    )


def _outline(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return _binary_dilate(mask, 2) & ~mask


def _overlay(
    image: NDArray[np.uint8], target: NDArray[np.bool_], sham: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    result = image.copy()
    result[_outline(target)] = np.array([0, 255, 0], dtype=np.uint8)
    result[_outline(sham)] = np.array([255, 220, 0], dtype=np.uint8)
    return result


def _contact_sheet(cells: Sequence[QCCell], *, columns: int, panel_size: int) -> bytes:
    if not cells:
        raise S3DataError("S3 contact sheet cannot be empty")
    cell_width = panel_size * 3 + 18
    cell_height = panel_size + 42
    rows = math.ceil(len(cells) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, cell in enumerate(cells):
        origin_x = (index % columns) * cell_width
        origin_y = (index // columns) * cell_height
        draw.text(
            (origin_x + 4, origin_y + 2),
            f"{cell.split} {cell.component_id} {cell.method}",
            fill="black",
        )
        images = (
            _overlay(cell.source, cell.target_component, cell.sham_edit),
            _overlay(cell.x_plus_sham, cell.target_component, cell.sham_edit),
            _overlay(cell.x_minus_removed, cell.target_component, cell.sham_edit),
        )
        labels = ("source", "x+ sham", "x- removed")
        for panel_index, (array, label) in enumerate(zip(images, labels, strict=True)):
            panel = Image.fromarray(array, mode="RGB").resize(
                (panel_size, panel_size), Image.Resampling.BILINEAR
            )
            panel_x = origin_x + 4 + panel_index * panel_size
            panel_y = origin_y + 20
            sheet.paste(panel, (panel_x, panel_y))
            draw.text((panel_x + 2, panel_y + panel_size + 2), label, fill="black")
    stream = io.BytesIO()
    sheet.save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _selection_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = (
        "component_id",
        "source_image_id",
        "split",
        "lesion_class",
        "selection_key",
        "frozen_rank",
        "eligibility_reason",
        "attempted",
        "pilot_outcome",
        "method",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _accepted_rejected_csv(
    accepted: Sequence[Mapping[str, object]], rejected: Sequence[Mapping[str, object]]
) -> bytes:
    fields = (
        "status",
        "twin_id",
        "component_id",
        "source_image_id",
        "split",
        "lesion_class",
        "method",
        "reason_code",
        "x_plus_sham_path",
        "x_plus_sham_sha256",
        "x_minus_removed_path",
        "x_minus_removed_sha256",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in accepted:
        outputs = cast(Mapping[str, object], record["outputs"])
        plus = cast(Mapping[str, object], outputs["x_plus_sham"])
        minus = cast(Mapping[str, object], outputs["x_minus_removed"])
        writer.writerow(
            {
                "status": "ACCEPTED",
                "twin_id": record.get("twin_id"),
                "component_id": record.get("component_id"),
                "source_image_id": record.get("source_image_id"),
                "split": record.get("split"),
                "lesion_class": record.get("lesion_class"),
                "method": record.get("method"),
                "reason_code": "",
                "x_plus_sham_path": plus.get("relative_path"),
                "x_plus_sham_sha256": plus.get("sha256"),
                "x_minus_removed_path": minus.get("relative_path"),
                "x_minus_removed_sha256": minus.get("sha256"),
            }
        )
    for record in rejected:
        writer.writerow(
            {
                "status": "REJECTED",
                "twin_id": "",
                "component_id": record.get("component_id"),
                "source_image_id": record.get("source_image_id"),
                "split": record.get("split"),
                "lesion_class": record.get("lesion_class"),
                "method": record.get("method"),
                "reason_code": record.get("reason_code"),
                "x_plus_sham_path": "",
                "x_plus_sham_sha256": "",
                "x_minus_removed_path": "",
                "x_minus_removed_sha256": "",
            }
        )
    return stream.getvalue().encode("utf-8")


def _audit_html(audit: Mapping[str, object]) -> bytes:
    summary = cast(Mapping[str, object], audit["summary"])
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Science S3 audit</title>"
        "</head><body><h1>Science S3 controlled-twin pilot audit</h1>"
        f"<p>Status: <strong>{escape(str(audit['status']))}</strong></p>"
        f"<table>{rows}</table>"
        "<p>This is a TRAIN/CALIBRATION pilot artifact, not FCR, clinical, medical, or "
        "paper-performance evidence.</p></body></html>"
    ).encode()


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path, payload in sorted(artifacts.items()):
        write_bytes_fail_safe(output_dir / relative_path, payload)
        hashes[relative_path] = sha256_bytes(payload)
    manifest_payload = canonical_json_bytes(
        {
            "schema_version": S3_SCHEMA_VERSION,
            "files": [
                {
                    "relative_path": path,
                    "sha256": digest,
                    "byte_size": len(artifacts[path]),
                }
                for path, digest in sorted(hashes.items())
            ],
        }
    )
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", manifest_payload)
    hashes["artifact-manifest.json"] = sha256_bytes(manifest_payload)
    return hashes


def _require_new_ignored_output(
    project_root: Path, dataset_root: Path, config: S3Config, output_dir: Path
) -> Path:
    resolved_project = project_root.resolve(strict=True)
    resolved_output = output_dir.resolve(strict=False)
    expected = (resolved_project / Path(config.artifact_root)).resolve(strict=False)
    if resolved_output != expected:
        raise S3DataError("S3 output does not match the frozen artifact_root")
    try:
        relative = resolved_output.relative_to(resolved_project)
    except ValueError as error:
        raise S3DataError("S3 output must remain inside the project") from error
    if relative.parts[:2] != ("reports", "generated"):
        raise S3DataError("real S3 artifacts must be under reports/generated")
    ignore_rules = {
        line.strip()
        for line in (resolved_project / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    if "/reports/generated/" not in ignore_rules:
        raise S3DataError("reports/generated is not protected by .gitignore")
    try:
        resolved_output.relative_to(dataset_root.resolve(strict=True))
        raise S3DataError("S3 output overlaps the dataset root")
    except ValueError:
        pass
    if resolved_output.exists():
        raise FileExistsError(f"S3 output directory must be new: {resolved_output}")
    return resolved_output


def run_s3_pilot(
    *, project_root: Path, config_path: Path, dataset_root: Path, output_dir: Path
) -> S3RunResult:
    """Execute the authorized real S3 pilot and stop at responsible confirmation."""

    config = S3Config.load(config_path)
    upstream = _replay_upstream(project_root, config)
    registry = load_registry_license_metadata(upstream.s1.registry_path)
    review = load_license_review_record(upstream.s1.review_path)
    spec = load_dataset_adapter_spec(upstream.s1.spec_path)
    authorize_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.LESION_CALIBRATION,
        root=dataset_root,
        spec=spec,
    )
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise S3DataError("authorized dataset root is not a directory")
    resolved_output = _require_new_ignored_output(project_root, root, config, output_dir)
    source_lookup = {record.source_image_id: record for record in upstream.s1.manifest.records}
    if any(record.split == LOCKED_SPLIT for record in source_lookup.values()) is False:
        raise S3GateError("S1 replay unexpectedly lacks frozen LOCKED_TEST membership")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    selection_rows: list[dict[str, object]] = []
    selection_row_lookup: dict[str, dict[str, object]] = {}
    for record in upstream.catalog_records:
        split = _string(record.get("split"), "split")
        lesion_class = _string(record.get("lesion_class"), "lesion class")
        component_id = _string(record.get("component_id"), "component id")
        reason = _eligibility_reason(record, config)
        row: dict[str, object] = {
            "component_id": component_id,
            "source_image_id": _string(record.get("source_image_id"), "source image id"),
            "split": split,
            "lesion_class": lesion_class,
            "selection_key": _selection_key(config, record),
            "frozen_rank": "",
            "eligibility_reason": reason,
            "attempted": False,
            "pilot_outcome": reason,
            "method": "",
        }
        selection_rows.append(row)
        selection_row_lookup[component_id] = row
        if reason == "ELIGIBLE":
            grouped[(split, lesion_class)].append(record)

    for candidates in grouped.values():
        candidates.sort(key=lambda record: _selection_key(config, record))
        for rank, record in enumerate(candidates, start=1):
            selection_row_lookup[_string(record.get("component_id"), "component id")][
                "frozen_rank"
            ] = rank

    accepted: list[dict[str, JSONValue]] = []
    rejected: list[dict[str, JSONValue]] = []
    qc_cells: dict[str, list[QCCell]] = defaultdict(list)
    artifacts: dict[str, bytes] = {}
    root_read_log: list[dict[str, str]] = []
    source_splits: dict[str, str] = {}
    for lesion_class in LESION_CLASSES:
        for split in DEVELOPMENT_SPLITS:
            accepted_for_group = 0
            quota = config.split_quotas[split]
            for record in grouped[(split, lesion_class)]:
                component_id = _string(record.get("component_id"), "component id")
                row = selection_row_lookup[component_id]
                if accepted_for_group >= quota:
                    row["pilot_outcome"] = "NOT_REACHED_AFTER_QUOTA_FILLED"
                    continue
                method = _assigned_method(config, component_id)
                row["attempted"] = True
                row["method"] = method
                source_image_id = _string(record.get("source_image_id"), "source image id")
                source_record = source_lookup.get(source_image_id)
                if source_record is None:
                    raise S3DataError("selected component source is absent from S1 manifest")
                prior_split = source_splits.setdefault(source_image_id, split)
                if prior_split != split:
                    raise S3DataError("source image crosses S3 splits")
                try:
                    built = _build_one(
                        project_root=project_root,
                        dataset_root=root,
                        config=config,
                        upstream=upstream,
                        record=record,
                        source_record=source_record,
                        root_read_log=root_read_log,
                    )
                except TwinRejected as error:
                    if error.reason_code not in config.rejection_taxonomy:
                        raise S3DataError("unregistered S3 rejection reason") from error
                    row["pilot_outcome"] = error.reason_code
                    rejected.append(
                        {
                            "schema_version": S3_SCHEMA_VERSION,
                            "component_id": component_id,
                            "source_image_id": source_image_id,
                            "split": split,
                            "lesion_class": lesion_class,
                            "method": method,
                            "reason_code": error.reason_code,
                        }
                    )
                    continue
                row["pilot_outcome"] = "ACCEPTED"
                accepted.append(built.record)
                artifacts.update(built.artifacts)
                qc_cells[lesion_class].append(built.qc_cell)
                accepted_for_group += 1

    if any(item["split"] == LOCKED_SPLIT for item in root_read_log):
        raise S3DataError("LOCKED_TEST root read detected")
    accepted_counts = Counter(f"{record['split']}:{record['lesion_class']}" for record in accepted)
    method_counts = Counter(str(record["method"]) for record in accepted)
    for lesion_class in LESION_CLASSES:
        if (
            sum(count for key, count in accepted_counts.items() if key.endswith(f":{lesion_class}"))
            > config.class_maximum
        ):
            raise S3DataError("accepted class count exceeds frozen maximum")
        if not qc_cells[lesion_class]:
            raise S3DataError(f"no accepted {lesion_class} twins")

    reason_counts = Counter(str(row["pilot_outcome"]) for row in selection_rows)
    manifest_payload: dict[str, JSONValue] = {
        "schema_version": S3_SCHEMA_VERSION,
        "experiment_id": S3_EXPERIMENT_ID,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "scope": "TRAIN_AND_CALIBRATION_ONLY",
        "config_sha256": config.config_sha256,
        "upstream": {
            "s2_commit": _string(config.upstream.get("s2_commit"), "S2 commit"),
            "s2_config_sha256": upstream.hashes["s2_config"],
            "s2_catalog_sha256": upstream.hashes["s2_catalog"],
            "s2_generated_audit_sha256": upstream.hashes["s2_generated_audit"],
            "s2_artifact_manifest_sha256": upstream.hashes["s2_artifact_manifest"],
            "s2_qc_record_sha256": upstream.hashes["s2_qc_record"],
        },
        "accepted": cast(list[JSONValue], accepted),
        "rejected": cast(list[JSONValue], rejected),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
    }
    artifacts["pilot-manifest-v1.json"] = canonical_json_bytes(manifest_payload)
    artifacts["accepted-rejected-v1.csv"] = _accepted_rejected_csv(accepted, rejected)
    selection_rows.sort(
        key=lambda row: (
            str(row["lesion_class"]),
            str(row["split"]),
            str(row["selection_key"]),
        )
    )
    artifacts["selection-audit-v1.csv"] = _selection_csv(selection_rows)
    for lesion_class in LESION_CLASSES:
        cells = sorted(qc_cells[lesion_class], key=lambda cell: (cell.split, cell.twin_id))
        artifacts[f"qc-{lesion_class.lower()}-controlled-twins-contact-sheet.png"] = _contact_sheet(
            cells, columns=config.qc_columns, panel_size=config.qc_panel_size
        )

    audit_payload: dict[str, JSONValue] = {
        "schema_version": S3_SCHEMA_VERSION,
        "experiment_id": S3_EXPERIMENT_ID,
        "status": "READY_FOR_RESPONSIBLE_CONFIRMATION",
        "summary": {
            "accepted_twin_count": len(accepted),
            "rejected_twin_count": len(rejected),
            "paired_output_count": len(accepted) * 2,
            "accepted_counts": dict(sorted(accepted_counts.items())),
            "method_counts": dict(sorted(method_counts.items())),
            "selection_record_count": len(selection_rows),
            "root_read_event_count": len(root_read_log),
            "locked_test_membership_count": sum(
                record.split == LOCKED_SPLIT for record in upstream.s1.manifest.records
            ),
            "locked_test_root_reads": 0,
            "locked_test_mask_decodes": 0,
            "source_split_leak_count": 0,
            "responsible_confirmation": "PENDING",
        },
        "gate": {
            "s1_s2_upstream_replay": "PASS",
            "s2_qc_responsible_confirmation": "PASS",
            "frozen_selection": "PASS",
            "provenance_hashes": "PASS",
            "source_split_isolation": "PASS",
            "locked_test_isolation": "PASS",
            "complete_pairs": "PASS",
            "outside_edit_exact_preservation": "PASS",
            "target_preservation_in_sham": "PASS",
            "artifact_git_boundary": "PASS",
            "ai_contact_sheet_review": "PENDING",
            "responsible_confirmation": "PENDING",
        },
        "hashes": {
            "config_sha256": config.config_sha256,
            "code_sha256": _code_sha256(project_root),
            "s2_catalog_sha256": upstream.hashes["s2_catalog"],
            "s2_artifact_manifest_sha256": upstream.hashes["s2_artifact_manifest"],
            "s2_qc_record_sha256": upstream.hashes["s2_qc_record"],
            "pilot_manifest_sha256": sha256_bytes(artifacts["pilot-manifest-v1.json"]),
            "selection_audit_sha256": sha256_bytes(artifacts["selection-audit-v1.csv"]),
            "accepted_rejected_sha256": sha256_bytes(artifacts["accepted-rejected-v1.csv"]),
        },
    }
    artifacts["audit.json"] = canonical_json_bytes(audit_payload)
    artifacts["audit.html"] = _audit_html(audit_payload)
    artifact_hashes = _write_artifacts(resolved_output, artifacts)
    return S3RunResult(
        status="READY_FOR_RESPONSIBLE_CONFIRMATION",
        output_dir=resolved_output,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_counts=dict(sorted(accepted_counts.items())),
        method_counts=dict(sorted(method_counts.items())),
        locked_test_root_reads=0,
        artifact_hashes=artifact_hashes,
        audit=audit_payload,
    )


def run_s3_synthetic_fixture(output_dir: Path) -> dict[str, str]:
    """Exercise S3 candidate code on a toy RGB patch; never a medical result."""

    if output_dir.exists():
        raise FileExistsError(f"synthetic S3 output directory must be new: {output_dir}")
    yy, xx = np.mgrid[0:128, 0:128]
    image = np.stack(
        (
            45 + xx,
            35 + yy,
            30 + ((xx + yy) // 3),
        ),
        axis=2,
    ).astype(np.uint8)
    target = (xx - 64) ** 2 + (yy - 64) ** 2 <= 7**2
    image[target] = np.array([150, 40, 35], dtype=np.uint8)
    edit = _binary_dilate(np.asarray(target, dtype=np.bool_), 3)
    sham = _translate_mask(edit, (36, 0))
    if sham is None:
        raise S3DataError("synthetic sham translation failed")
    artifacts: dict[str, bytes] = {}
    for method in METHODS:
        donor = (0, 32) if method == "DONOR_POISSON" else None
        plus = inpaint_candidate(
            image,
            sham,
            method=method,
            donor_offset=donor,
            navier_iterations=8,
            poisson_iterations=16,
        )
        minus = inpaint_candidate(
            image,
            edit,
            method=method,
            donor_offset=donor,
            navier_iterations=8,
            poisson_iterations=16,
        )
        artifacts[f"{method.lower()}-x-plus.png"] = rgb_png_bytes(plus)
        artifacts[f"{method.lower()}-x-minus.png"] = rgb_png_bytes(minus)
    artifacts["audit.json"] = canonical_json_bytes(
        {
            "schema_version": S3_SCHEMA_VERSION,
            "experiment_id": S3_EXPERIMENT_ID,
            "status": "PASS_SYNTHETIC_FIXTURE_ONLY",
            "summary": {
                "accepted_twin_count": 3,
                "rejected_twin_count": 0,
                "paired_output_count": 6,
                "locked_test_root_reads": 0,
                "source_split_leak_count": 0,
            },
            "gate": {"software_fixture": "PASS"},
            "hashes": {},
        }
    )
    return _write_artifacts(output_dir, artifacts)


__all__ = [
    "S3Config",
    "S3DataError",
    "S3GateError",
    "S3RunResult",
    "inpaint_candidate",
    "run_s3_pilot",
    "run_s3_synthetic_fixture",
]
