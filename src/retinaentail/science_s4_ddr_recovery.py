"""Science S4 DDR TRAIN/VALID controlled-twin recovery v5.

The runner uses only the hash-bound DDR development adapter.  It constructs
balanced sham/removed twins, but deliberately defers support-truth assignment
until S5 freezes a degradation/quantization observation domain.  DDR TEST is
not an accepted input to this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageOps

import retinaentail.science_s4_twins as s4_twins
from retinaentail.data.lesion_catalog import (
    JSONValue,
    SparseComponent,
    _component_touches_fov,
    _ellipse_boundary_distance,
    _safe_relative,
    connected_components,
    detect_fov,
    patch_geometry,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import ManifestRecordV2
from retinaentail.interventions.controlled_twin_pilot import (
    METHODS,
    QCCell,
    TwinRejected,
    _assigned_method,
    _contact_sheet,
    _selection_key,
)
from retinaentail.interventions.oracle import (
    ControlledTwinOracleRecord,
    DomainTruth,
    SourceTruth,
    SupportTruth,
)
from retinaentail.schemas import JSONValue as SchemaJSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "5.0.0"
EXPERIMENT_ID: Final = "S4_DDR_TIER_A_RECOVERY_V5"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_SAMPLE_OR_CONSTRUCTION_QC"
FINAL_READY_STATUS: Final = "READY_FOR_S5_DEGRADATION_CALIBRATION__S4_ORACLE_PENDING"
LESION_CLASSES: Final = ("HE", "EX", "MA", "SE")
CANONICAL_SPLITS: Final = ("TRAIN", "CALIBRATION")
DEFERRED_ORACLE_ID: Final = "DEFERRED_S5_NORMALIZED_OBSERVATION_ORACLE_V2"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class S4DDRRecoveryError(RuntimeError):
    """Raised when the bounded DDR recovery cannot be replayed exactly."""


@dataclass(frozen=True, slots=True)
class DDRSourceRecord:
    """Minimal source identity required by the frozen S4 twin helpers."""

    source_image_id: str
    split: str
    raw_split: str
    relative_path: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DDRRecoveryConfig:
    """Validated v5 contract plus a compatible frozen twin helper config."""

    path: Path
    payload: dict[str, object]
    config_sha256: str
    twin: s4_twins.S4TwinConfig
    raw_split_mapping: dict[str, str]
    minimum_sources_per_split: dict[str, dict[str, int]]
    expected_source_count: int
    expected_file_count: int
    expected_total_bytes: int
    expected_split_counts: dict[str, int]
    expected_label_counts: dict[str, int]
    external_root_name: str
    development_leaf_name: str
    fov_threshold: int
    fov_coverage: float
    mask_modes: tuple[str, ...]
    mask_l_values: frozenset[int]
    qc_samples_per_class: int
    qc_columns: int
    qc_panel_size: int

    @classmethod
    def load(cls, path: Path) -> DDRRecoveryConfig:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "DDR recovery config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("status") != "EXECUTABLE_TWIN_RECOVERY"
        ):
            raise S4DDRRecoveryError("unsupported DDR recovery config")
        _verify_baseline(root.get("expected_baseline_commit"))
        dataset = _mapping(root.get("dataset"), "dataset")
        if (
            dataset.get("dataset_id") != "ddr"
            or dataset.get("source_splits") != ["TRAIN", "VALID"]
            or dataset.get("test_content_reads_allowed") is not False
            or dataset.get("test_image_decodes_allowed") is not False
            or dataset.get("missing_label_policy") != "UNKNOWN"
        ):
            raise S4DDRRecoveryError("DDR development/test boundary changed")
        raw_mapping = {
            str(key): _string(value, f"split mapping {key}")
            for key, value in _mapping(
                dataset.get("canonical_split_mapping"), "split mapping"
            ).items()
        }
        if raw_mapping != {"TRAIN": "TRAIN", "VALID": "CALIBRATION"}:
            raise S4DDRRecoveryError("DDR canonical split mapping changed")
        selection = _mapping(root.get("selection"), "selection")
        if (
            selection.get("class_order") != list(LESION_CLASSES)
            or selection.get("source_coverage_order") != "SHA256_SEED_CLASS_SPLIT_SOURCE_ID_ASC"
            or selection.get("within_source_order") != "SHA256_SEED_CLASS_SPLIT_COMPONENT_ID_ASC"
            or selection.get("fill_order") != "SHA256_SEED_CLASS_SPLIT_COMPONENT_ID_ASC"
            or selection.get("replacement_policy") != "NEXT_CONSTRUCTIBLE_IN_FROZEN_ORDER"
        ):
            raise S4DDRRecoveryError("DDR recovery selection contract changed")
        targets_raw = _mapping(selection.get("accepted_target_per_class"), "targets")
        targets = {
            lesion_class: _integer(targets_raw.get(lesion_class), f"{lesion_class} target", 1)
            for lesion_class in LESION_CLASSES
        }
        if targets != {"HE": 264, "EX": 54, "MA": 54, "SE": 249}:
            raise S4DDRRecoveryError("DDR recovery targets changed")
        quotas_raw = _mapping(selection.get("accepted_quota_per_class_split"), "split quotas")
        minimum_raw = _mapping(
            selection.get("minimum_distinct_sources_per_class_split"), "source minima"
        )
        quotas: dict[str, dict[str, int]] = {}
        minima: dict[str, dict[str, int]] = {}
        for lesion_class in LESION_CLASSES:
            class_quotas = _mapping(quotas_raw.get(lesion_class), f"{lesion_class} quotas")
            class_minima = _mapping(minimum_raw.get(lesion_class), f"{lesion_class} minima")
            quotas[lesion_class] = {
                split: _integer(class_quotas.get(split), f"{lesion_class}:{split} quota", 0)
                for split in CANONICAL_SPLITS
            }
            minima[lesion_class] = {
                split: _integer(class_minima.get(split), f"{lesion_class}:{split} minimum", 0)
                for split in CANONICAL_SPLITS
            }
            if sum(quotas[lesion_class].values()) != targets[lesion_class] or any(
                minima[lesion_class][split] > quotas[lesion_class][split]
                for split in CANONICAL_SPLITS
            ):
                raise S4DDRRecoveryError("DDR quota/source-minimum contract is inconsistent")
        assignment = _mapping(root.get("method_assignment"), "method assignment")
        edit = _mapping(root.get("edit"), "edit")
        qc = _mapping(root.get("qc"), "QC")
        support = _mapping(root.get("support_oracle"), "support oracle")
        if (
            assignment.get("methods") != list(METHODS)
            or assignment.get("order") != "SHA256_SEED_COMPONENT_ID_MOD_METHOD_COUNT"
            or edit.get("patch_size_wh") != [256, 256]
            or support.get("status") != "DEFERRED_TO_S5_DEGRADATION_OBSERVATION"
            or support.get("raw_edit_array_equality_is_confirmatory") is not False
            or support.get("support_truth_before_s5") != "ORACLE_UNRESOLVED"
            or support.get("domain_truth_before_s5") != "OUT_OF_DOMAIN"
            or support.get("tier_b_status") != "NOT_AVAILABLE"
        ):
            raise S4DDRRecoveryError("method/edit/deferred-Oracle contract changed")
        method_settings = {
            method: _mapping(assignment.get(method), f"method {method}") for method in METHODS
        }
        implementations = {
            "TELEA": "TELEA_STYLE_BOUNDARY_WEIGHTED_NUMPY_V1",
            "NAVIER_STOKES": "NAVIER_STOKES_STYLE_ISOPHOTE_DIFFUSION_NUMPY_V1",
            "DONOR_POISSON": "DONOR_TEXTURE_POISSON_JACOBI_NUMPY_V1",
        }
        if any(
            method_settings[method].get("implementation_id") != implementation
            for method, implementation in implementations.items()
        ):
            raise S4DDRRecoveryError("method implementation binding changed")
        taxonomy = root.get("rejection_reason_taxonomy")
        if not isinstance(taxonomy, list) or not all(
            isinstance(item, str) and item for item in taxonomy
        ):
            raise S4DDRRecoveryError("rejection taxonomy is invalid")
        gate = _mapping(root.get("gate"), "gate")
        if not gate or any(value is not True for value in gate.values()):
            raise S4DDRRecoveryError("one or more v5 construction gates are disabled")
        twin = s4_twins.S4TwinConfig(
            path=resolved,
            payload=root,
            config_sha256=sha256_file(resolved),
            dataset_id="ddr",
            dataset_version=_string(dataset.get("dataset_version"), "dataset version"),
            artifact_root=_safe_relative(root.get("artifact_root"), "artifact root"),
            bindings=_mapping(root.get("bindings"), "bindings"),
            seed=_integer(selection.get("seed"), "seed", 0),
            class_targets=targets,
            split_quotas=quotas,
            minimum_sources={
                lesion_class: sum(minima[lesion_class].values()) for lesion_class in LESION_CLASSES
            },
            minimum_area=_integer(selection.get("minimum_area_px"), "minimum area", 1),
            maximum_area=_integer(selection.get("maximum_area_px"), "maximum area", 1),
            maximum_bbox_width=_integer(
                selection.get("maximum_bbox_width_px"), "maximum bbox width", 1
            ),
            maximum_bbox_height=_integer(
                selection.get("maximum_bbox_height_px"), "maximum bbox height", 1
            ),
            minimum_fov_margin=_number(
                selection.get("minimum_fov_margin_px"), "minimum FOV margin", 0.0
            ),
            methods=METHODS,
            method_settings=method_settings,
            target_dilation=_integer(edit.get("target_mask_dilation_px"), "target dilation", 1),
            lesion_exclusion_dilation=_integer(
                edit.get("lesion_exclusion_dilation_px"), "lesion exclusion", 1
            ),
            sham_offsets=_offsets(edit.get("sham_offset_candidates_xy"), "sham offsets"),
            donor_offsets=_offsets(edit.get("donor_offset_candidates_xy"), "donor offsets"),
            qc_samples_per_class=_integer(
                qc.get("deterministic_samples_per_class"), "QC samples", 1
            ),
            qc_columns=_integer(qc.get("contact_sheet_columns"), "QC columns", 1),
            qc_panel_size=_integer(qc.get("panel_size_px"), "QC panel size", 64),
            minimum_collisions=1,
            rejection_taxonomy=frozenset(cast(list[str], taxonomy)),
        )
        decode = _mapping(root.get("decode"), "decode")
        fov = _mapping(root.get("fov"), "FOV")
        modes_raw = decode.get("accepted_mask_modes")
        values_raw = decode.get("accepted_l_values")
        if not isinstance(modes_raw, list) or not isinstance(values_raw, list):
            raise S4DDRRecoveryError("mask decoding contract is invalid")
        return cls(
            path=resolved,
            payload=root,
            config_sha256=twin.config_sha256,
            twin=twin,
            raw_split_mapping=raw_mapping,
            minimum_sources_per_split=minima,
            expected_source_count=_integer(dataset.get("expected_source_count"), "source count", 1),
            expected_file_count=_integer(dataset.get("expected_file_count"), "file count", 1),
            expected_total_bytes=_integer(dataset.get("expected_total_bytes"), "total bytes", 1),
            expected_split_counts={
                str(key): _integer(value, f"split count {key}", 0)
                for key, value in _mapping(
                    dataset.get("expected_split_source_counts"), "split counts"
                ).items()
            },
            expected_label_counts={
                str(key): _integer(value, f"label count {key}", 0)
                for key, value in _mapping(
                    dataset.get("expected_label_availability"), "label counts"
                ).items()
            },
            external_root_name=_string(dataset.get("external_root_name"), "external root name"),
            development_leaf_name=_string(
                dataset.get("development_leaf_name"), "development leaf name"
            ),
            fov_threshold=_integer(fov.get("foreground_max_rgb_min"), "FOV threshold", 0),
            fov_coverage=_number(fov.get("minimum_axis_coverage_fraction"), "FOV coverage", 0.0),
            mask_modes=tuple(cast(list[str], modes_raw)),
            mask_l_values=frozenset(cast(list[int], values_raw)),
            qc_samples_per_class=twin.qc_samples_per_class,
            qc_columns=twin.qc_columns,
            qc_panel_size=twin.qc_panel_size,
        )


@dataclass(frozen=True, slots=True)
class DDRRecoveryResult:
    """v5 disposition and immutable generated hashes."""

    status: str
    output_dir: Path
    accepted_count: int
    rejected_count: int
    artifact_hashes: dict[str, str]
    audit: dict[str, JSONValue]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S4DDRRecoveryError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise S4DDRRecoveryError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S4DDRRecoveryError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S4DDRRecoveryError(f"{name} must be numeric")
    rendered = float(value)
    if not np.isfinite(rendered) or rendered < minimum:
        raise S4DDRRecoveryError(f"{name} is outside the frozen range")
    return rendered


def _offsets(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise S4DDRRecoveryError(f"{name} must be a non-empty list")
    result: list[tuple[int, int]] = []
    for raw in value:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
        ):
            raise S4DDRRecoveryError(f"{name} contains an invalid offset")
        x, y = cast(list[int], raw)
        if x == 0 and y == 0:
            raise S4DDRRecoveryError(f"{name} contains the zero offset")
        result.append((x, y))
    if len(result) != len(set(result)):
        raise S4DDRRecoveryError(f"{name} contains duplicate offsets")
    return tuple(result)


def _verify_baseline(value: object) -> None:
    expected = _string(value, "expected baseline commit")
    try:
        import subprocess

        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise S4DDRRecoveryError("cannot verify Git baseline") from error
    if actual != expected:
        raise S4DDRRecoveryError(f"baseline drift: expected {expected}, got {actual}")


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S4DDRRecoveryError(f"cannot load {name}: {error}") from error
    return _mapping(decoded, name)


def _verify_binding(value: object, name: str) -> tuple[Path, str]:
    binding = _mapping(value, name)
    relative = _safe_relative(binding.get("path"), f"{name} path")
    expected = _string(binding.get("sha256"), f"{name} SHA-256")
    path = (PROJECT_ROOT / relative).resolve(strict=True)
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4DDRRecoveryError(f"{name} escapes the project") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise S4DDRRecoveryError(f"binding hash mismatch: {name}")
    return path, expected


def _replay_upstream(config: DDRRecoveryConfig) -> tuple[dict[str, str], dict[str, object]]:
    bindings = _mapping(config.payload.get("bindings"), "bindings")
    required = {
        "recovery_prompt",
        "s4_preregistration_v1",
        "idrid_run_config",
        "idrid_audit",
        "idrid_artifact_manifest",
        "idrid_qc",
        "idrid_report",
        "contract_v3_config",
        "contract_v3_audit",
        "ddr_development_adapter",
        "contract_v3_artifact_manifest",
        "tier_b_v4_config",
        "tier_b_v4_audit",
        "tier_b_v4_artifact_manifest",
        "tier_b_v4_ai_qc",
        "tier_b_v4_report",
    }
    if set(bindings) != required:
        raise S4DDRRecoveryError("v5 binding inventory changed")
    verified = {name: _verify_binding(item, name) for name, item in bindings.items()}
    idrid = _load_json(verified["idrid_audit"][0], "IDRiD S4 audit")
    idrid_summary = _mapping(idrid.get("summary"), "IDRiD summary")
    if (
        idrid.get("status") != "BLOCKED_SAMPLE_OR_ORACLE_SUFFICIENCY"
        or idrid_summary.get("accepted_twin_count") != 1237
        or idrid_summary.get("accepted_distinct_sources") != {"EX": 26, "HE": 24, "MA": 26, "SE": 1}
    ):
        raise S4DDRRecoveryError("IDRiD S4 result drift")
    idrid_qc = _load_json(verified["idrid_qc"][0], "IDRiD S4 QC")
    if idrid_qc.get("decision") != "PASS_WITH_LIMITATIONS":
        raise S4DDRRecoveryError("IDRiD S4 QC drift")
    v3 = _load_json(verified["contract_v3_audit"][0], "contract v3 audit")
    if (
        v3.get("status") != "READY_FOR_S4_RECOVERY_IMAGE_AUDIT"
        or _mapping(v3.get("ddr"), "v3 DDR").get("development_source_count") != 532
        or _mapping(v3.get("ddr"), "v3 DDR").get("test_content_reads") != 0
    ):
        raise S4DDRRecoveryError("contract v3 DDR gate drift")
    v4 = _load_json(verified["tier_b_v4_audit"][0], "Tier-B v4 audit")
    v4_qc = _load_json(verified["tier_b_v4_ai_qc"][0], "Tier-B v4 QC")
    if (
        v4.get("status") != "BLOCKED_PREREQUISITES"
        or _mapping(v4.get("registration"), "v4 registration").get("accepted_count") != 1
        or v4_qc.get("decision") != "REJECT_TIER_B_AVAILABILITY"
    ):
        raise S4DDRRecoveryError("Tier-B v4 negative disposition drift")
    for manifest_name, expected_entries in (
        ("idrid_artifact_manifest", {"audit.json"}),
        (
            "contract_v3_artifact_manifest",
            {"audit.json", "ddr-development-adapter-manifest.json"},
        ),
        ("tier_b_v4_artifact_manifest", {"audit.json"}),
    ):
        manifest = _load_json(verified[manifest_name][0], manifest_name)
        entries = {
            _string(
                _mapping(item, "artifact entry").get("relative_path"), "artifact path"
            ): _string(_mapping(item, "artifact entry").get("sha256"), "artifact hash")
            for item in _sequence(manifest.get("files"), "artifact files")
        }
        if not expected_entries.issubset(entries):
            raise S4DDRRecoveryError(f"{manifest_name} is incomplete")
    adapter = _load_json(verified["ddr_development_adapter"][0], "DDR adapter")
    return {name: digest for name, (_, digest) in verified.items()}, adapter


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise S4DDRRecoveryError(f"{name} must be a list")
    return tuple(value)


def _development_root(config: DDRRecoveryConfig, contract_root: Path) -> Path:
    if not contract_root.is_absolute():
        raise S4DDRRecoveryError("contract root must be absolute")
    root = contract_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or root.name != config.external_root_name:
        raise S4DDRRecoveryError("contract root is not the frozen v3 external directory")
    development = (root / config.development_leaf_name).resolve(strict=True)
    if not development.is_dir() or development.is_symlink():
        raise S4DDRRecoveryError("DDR development leaf is unavailable")
    return development


def _output_boundary(config: DDRRecoveryConfig, development: Path, output_dir: Path) -> Path:
    output = output_dir.resolve(strict=False)
    expected = (PROJECT_ROOT / config.twin.artifact_root).resolve(strict=False)
    if output != expected:
        raise S4DDRRecoveryError("v5 output does not match the frozen artifact root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DDR recovery output: {output}")
    try:
        relative = output.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4DDRRecoveryError("v5 output escapes the project") from error
    if relative.parts[:2] != ("reports", "generated"):
        raise S4DDRRecoveryError("v5 output must remain under reports/generated")
    try:
        output.relative_to(development)
        raise S4DDRRecoveryError("v5 output overlaps DDR development input")
    except ValueError:
        pass
    return output


def decode_ddr_mask(
    path: Path, *, expected_size: tuple[int, int], config: DDRRecoveryConfig
) -> NDArray[np.bool_]:
    """Decode only the mask modes/values frozen before first DDR mask read."""

    with Image.open(path) as source:
        source.load()
        if source.size != expected_size or source.mode not in config.mask_modes:
            raise S4DDRRecoveryError("DDR mask mode/size violates the frozen contract")
        if source.mode == "1":
            return np.asarray(source, dtype=np.bool_).copy()
        array = np.asarray(source, dtype=np.uint8)
    values = frozenset(int(value) for value in np.unique(array))
    if not values.issubset(config.mask_l_values):
        raise S4DDRRecoveryError(f"DDR L mask has unexpected values: {sorted(values)}")
    return np.asarray(array != 0, dtype=np.bool_)


def _resolve_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(_safe_relative(relative, "adapter path")).parts).resolve(
        strict=True
    )
    try:
        path.relative_to(root)
    except ValueError as error:
        raise S4DDRRecoveryError("adapter path escapes DDR development root") from error
    if not path.is_file() or path.is_symlink():
        raise S4DDRRecoveryError("adapter file is missing or a symlink")
    return path


def _adapter_records(
    config: DDRRecoveryConfig, adapter: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    if (
        adapter.get("dataset_id") != "ddr"
        or adapter.get("dataset_version") != config.twin.dataset_version
        or adapter.get("source_count") != config.expected_source_count
        or adapter.get("file_count") != config.expected_file_count
        or adapter.get("total_bytes_hashed") != config.expected_total_bytes
        or adapter.get("source_split_leaks") != 0
        or adapter.get("split_source_counts") != config.expected_split_counts
        or adapter.get("label_availability_counts") != config.expected_label_counts
        or adapter.get("missing_label_policy") != "UNKNOWN"
    ):
        raise S4DDRRecoveryError("DDR development adapter contract drift")
    records = tuple(
        _mapping(item, "DDR adapter record")
        for item in _sequence(adapter.get("records"), "DDR adapter records")
    )
    if len(records) != config.expected_source_count:
        raise S4DDRRecoveryError("DDR adapter record count drift")
    seen: dict[str, str] = {}
    for record in records:
        source_id = _string(record.get("source_image_id"), "source ID")
        raw_split = _string(record.get("split"), "raw split")
        if raw_split not in config.raw_split_mapping:
            raise S4DDRRecoveryError("DDR adapter contains a non-development split")
        prior = seen.setdefault(source_id, raw_split)
        if prior != raw_split:
            raise S4DDRRecoveryError("DDR source crosses splits")
        labels = _mapping(record.get("labels"), "labels")
        if set(labels) != set(LESION_CLASSES):
            raise S4DDRRecoveryError("DDR record does not declare four labels")
        for lesion_class in LESION_CLASSES:
            if _mapping(labels[lesion_class], "label").get("availability") != "AVAILABLE":
                raise S4DDRRecoveryError("DDR missing label cannot become a negative label")
    return records


def _load_context(
    *,
    config: DDRRecoveryConfig,
    development: Path,
    adapter_record: Mapping[str, object],
    io_counts: Counter[str],
) -> s4_twins.SourceContext:
    source_id = _string(adapter_record.get("source_image_id"), "source ID")
    raw_split = _string(adapter_record.get("split"), "raw split")
    split = config.raw_split_mapping[raw_split]
    image_entry = _mapping(adapter_record.get("image"), "image entry")
    image_relative = _string(image_entry.get("relative_path"), "image path")
    image_path = _resolve_file(development, image_relative)
    expected_image_size = _integer(image_entry.get("byte_size"), "image byte size", 1)
    expected_image_hash = _string(image_entry.get("sha256"), "image SHA-256")
    if (
        image_path.stat().st_size != expected_image_size
        or sha256_file(image_path) != expected_image_hash
    ):
        raise S4DDRRecoveryError(f"DDR image hash drift: {image_relative}")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        image_array = np.asarray(image, dtype=np.uint8).copy()
    io_counts["image_hash_reads"] += 1
    io_counts["image_decodes"] += 1
    io_counts["bytes_hashed"] += expected_image_size
    labels = _mapping(adapter_record.get("labels"), "labels")
    masks: dict[str, NDArray[np.bool_]] = {}
    components: dict[str, tuple[SparseComponent, ...]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for lesion_class in LESION_CLASSES:
        label = _mapping(labels.get(lesion_class), f"{lesion_class} label")
        relative = _string(label.get("relative_path"), f"{lesion_class} path")
        path = _resolve_file(development, relative)
        expected_size = _integer(label.get("byte_size"), f"{lesion_class} bytes", 1)
        expected_hash = _string(label.get("sha256"), f"{lesion_class} SHA-256")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise S4DDRRecoveryError(f"DDR mask hash drift: {relative}")
        mask = decode_ddr_mask(path, expected_size=(width, height), config=config)
        masks[lesion_class] = mask
        components[lesion_class] = connected_components(mask, connectivity=8)
        provenance[lesion_class] = {"relative_path": relative, "sha256": expected_hash}
        io_counts["mask_hash_reads"] += 1
        io_counts["mask_decodes"] += 1
        io_counts["bytes_hashed"] += expected_size
    fov = detect_fov(
        image_array,
        foreground_max_rgb_min=config.fov_threshold,
        minimum_axis_coverage_fraction=config.fov_coverage,
    )
    source_record = DDRSourceRecord(
        source_image_id=source_id,
        split=split,
        raw_split=raw_split,
        relative_path=image_relative,
        sha256=expected_image_hash,
        width=width,
        height=height,
    )
    return s4_twins.SourceContext(
        source_record=cast(ManifestRecordV2, source_record),
        image=image_array,
        masks=masks,
        components=components,
        fov=fov.mask,
        label_provenance=provenance,
    )


def _component_records(
    context: s4_twins.SourceContext, config: DDRRecoveryConfig
) -> list[dict[str, object]]:
    source = cast(DDRSourceRecord, context.source_record)
    fov = detect_fov(
        context.image,
        foreground_max_rgb_min=config.fov_threshold,
        minimum_axis_coverage_fraction=config.fov_coverage,
    )
    records: list[dict[str, object]] = []
    for lesion_class in LESION_CLASSES:
        provenance = context.label_provenance[lesion_class]
        for index, component in enumerate(context.components[lesion_class], start=1):
            component_id = f"ddr-{source.source_image_id}-{lesion_class.lower()}-{index:06d}"
            geometry = patch_geometry(
                image_size=(source.width, source.height),
                centroid=component.centroid,
                patch_size=(256, 256),
            )
            x0, y0, x1, y1 = component.bbox
            records.append(
                {
                    "component_id": component_id,
                    "source_image_id": source.source_image_id,
                    "split": source.split,
                    "raw_split": source.raw_split,
                    "lesion_class": lesion_class,
                    "inclusion_status": "INCLUDED",
                    "area_px": component.area,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "centroid_xy": list(component.centroid),
                    "component_mask_sha256": component.component_mask_sha256,
                    "image_relative_path": source.relative_path,
                    "image_sha256": source.sha256,
                    "mask_relative_path": provenance["relative_path"],
                    "mask_sha256": provenance["sha256"],
                    "touches_image_boundary": x0 == 0
                    or y0 == 0
                    or x1 == source.width
                    or y1 == source.height,
                    "touches_fov_boundary": _component_touches_fov(component, fov),
                    "distance_to_fov_boundary_px": _ellipse_boundary_distance(
                        component.centroid, fov.bbox
                    ),
                    "patch": geometry.to_dict(),
                }
            )
    return records


def _source_order(config: DDRRecoveryConfig, lesion_class: str, split: str, source: str) -> str:
    return hashlib.sha256(
        f"{config.twin.seed}:{lesion_class}:{split}:{source}".encode()
    ).hexdigest()


def _deferred_oracle(record: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    outputs = cast(dict[str, JSONValue], record["outputs"])
    plus = cast(dict[str, JSONValue], outputs["x_plus_sham"])
    minus = cast(dict[str, JSONValue], outputs["x_minus_removed"])
    raw_equal = plus["raw_array_sha256"] == minus["raw_array_sha256"]
    evidence = {
        "method_id": DEFERRED_ORACLE_ID,
        "x_plus_raw_sha256": plus["raw_array_sha256"],
        "x_minus_raw_sha256": minus["raw_array_sha256"],
        "raw_edit_array_equal_diagnostic": raw_equal,
        "required_future_method": "NORMALIZED_DEGRADED_OBSERVATION_EXACT_COLLISION",
    }
    evidence_hash = sha256_bytes(canonical_json_bytes(evidence))
    truth_records = [
        ControlledTwinOracleRecord(
            source_truth=source_truth,
            support_truth=SupportTruth.ORACLE_UNRESOLVED,
            domain_truth=DomainTruth.OUT_OF_DOMAIN,
            oracle_tier=None,
            collision_mode=None,
            oracle_method_id=DEFERRED_ORACLE_ID,
            evidence_sha256=evidence_hash,
            retinaentail_independent=True,
        ).to_dict()
        for source_truth in (SourceTruth.PRESENT, SourceTruth.ABSENT)
    ]
    return {
        "method_id": DEFERRED_ORACLE_ID,
        "retinaentail_independent": True,
        "forbidden_inputs_observed": [],
        "status": "DEFERRED_TO_S5_DEGRADATION_OBSERVATION",
        "raw_array_exact_equal": raw_equal,
        "raw_edit_array_equality_is_confirmatory": False,
        "evidence_sha256": evidence_hash,
        "truth_records": cast(list[JSONValue], truth_records),
    }


def _build_one(
    context: s4_twins.SourceContext,
    record: Mapping[str, object],
    config: DDRRecoveryConfig,
) -> s4_twins.BuiltTwin:
    built = s4_twins._build_one(context=context, record=record, config=config.twin)  # noqa: SLF001
    built.record["experiment_id"] = EXPERIMENT_ID
    built.record["dataset_id"] = "ddr"
    built.record["dataset_version"] = config.twin.dataset_version
    built.record["oracle"] = _deferred_oracle(built.record)
    return s4_twins.BuiltTwin(
        record=built.record,
        artifacts=built.artifacts,
        qc_cell=built.qc_cell,
        exact_collision=False,
    )


def _write_artifacts(output: Path, artifacts: Mapping[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, payload in sorted(artifacts.items()):
        write_bytes_fail_safe(output / relative, payload)
        hashes[relative] = sha256_bytes(payload)
    manifest: dict[str, SchemaJSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "relative_path": relative,
                "byte_size": len(artifacts[relative]),
                "sha256": digest,
            }
            for relative, digest in sorted(hashes.items())
        ],
    }
    payload = canonical_json_bytes(manifest)
    write_bytes_fail_safe(output / "artifact-manifest.json", payload)
    hashes["artifact-manifest.json"] = sha256_bytes(payload)
    return hashes


def _code_sha256() -> str:
    files = (
        "src/retinaentail/science_s4_ddr_recovery.py",
        "src/retinaentail/science_s4_twins.py",
        "src/retinaentail/interventions/controlled_twin_pilot.py",
        "src/retinaentail/interventions/oracle.py",
    )
    payload: dict[str, SchemaJSONValue] = {
        "files": [
            {"relative_path": relative, "sha256": sha256_file(PROJECT_ROOT / relative)}
            for relative in files
        ]
    }
    return sha256_bytes(canonical_json_bytes(payload))


def run_ddr_recovery(
    *, config_path: Path, contract_root: Path, output_dir: Path
) -> DDRRecoveryResult:
    """Build the frozen DDR development twins and stop before S5."""

    config = DDRRecoveryConfig.load(config_path)
    upstream_hashes, adapter = _replay_upstream(config)
    development = _development_root(config, contract_root)
    output = _output_boundary(config, development, output_dir)
    adapter_records = _adapter_records(config, adapter)
    adapter_lookup = {
        _string(record.get("source_image_id"), "source ID"): record for record in adapter_records
    }
    io_counts: Counter[str] = Counter()
    component_records: list[dict[str, object]] = []
    for adapter_record in sorted(
        adapter_records, key=lambda item: _string(item.get("source_image_id"), "source ID")
    ):
        context = _load_context(
            config=config,
            development=development,
            adapter_record=adapter_record,
            io_counts=io_counts,
        )
        component_records.extend(_component_records(context, config))

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_source: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    selection_rows: list[dict[str, object]] = []
    selection_lookup: dict[str, dict[str, object]] = {}
    for record in component_records:
        lesion_class = _string(record.get("lesion_class"), "lesion class")
        split = _string(record.get("split"), "split")
        source_id = _string(record.get("source_image_id"), "source ID")
        component_id = _string(record.get("component_id"), "component ID")
        reason = s4_twins._catalog_eligibility_reason(record, config.twin)  # noqa: SLF001
        row: dict[str, object] = {
            "component_id": component_id,
            "source_image_id": source_id,
            "split": split,
            "lesion_class": lesion_class,
            "selection_key": _selection_key(config.twin.as_s3(), record),
            "frozen_rank": "",
            "eligibility_reason": reason,
            "constructibility_reason": "NOT_EVALUATED",
            "attempted": False,
            "outcome": reason,
            "method": "",
        }
        selection_rows.append(row)
        selection_lookup[component_id] = row
        if reason == "ELIGIBLE":
            grouped[(split, lesion_class)].append(record)
            by_source[(split, lesion_class, source_id)].append(record)
    for candidates in grouped.values():
        candidates.sort(key=lambda item: _selection_key(config.twin.as_s3(), item))
        for rank, record in enumerate(candidates, start=1):
            selection_lookup[_string(record.get("component_id"), "component ID")]["frozen_rank"] = (
                rank
            )
    for candidates in by_source.values():
        candidates.sort(key=lambda item: _selection_key(config.twin.as_s3(), item))

    cache: OrderedDict[str, s4_twins.SourceContext] = OrderedDict()

    def context_for(source_id: str) -> s4_twins.SourceContext:
        found = cache.pop(source_id, None)
        if found is None:
            found = _load_context(
                config=config,
                development=development,
                adapter_record=adapter_lookup[source_id],
                io_counts=io_counts,
            )
        cache[source_id] = found
        while len(cache) > 8:
            cache.popitem(last=False)
        return found

    accepted: list[dict[str, JSONValue]] = []
    rejected: list[dict[str, JSONValue]] = []
    artifacts: dict[str, bytes] = {}
    qc_cells: dict[str, list[QCCell]] = defaultdict(list)
    accepted_group: Counter[str] = Counter()
    accepted_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    attempted: set[str] = set()

    def attempt(record: dict[str, object]) -> bool:
        component_id = _string(record.get("component_id"), "component ID")
        source_id = _string(record.get("source_image_id"), "source ID")
        split = _string(record.get("split"), "split")
        lesion_class = _string(record.get("lesion_class"), "lesion class")
        row = selection_lookup[component_id]
        attempted.add(component_id)
        row["attempted"] = True
        row["method"] = _assigned_method(config.twin.as_s3(), component_id)
        try:
            built = _build_one(context_for(source_id), record, config)
        except TwinRejected as error:
            if error.reason_code not in config.twin.rejection_taxonomy:
                raise S4DDRRecoveryError("unregistered DDR construction rejection") from error
            row["constructibility_reason"] = error.reason_code
            row["outcome"] = error.reason_code
            rejected.append(
                s4_twins._rejected_record(record, config.twin, error.reason_code)  # noqa: SLF001
            )
            return False
        row["constructibility_reason"] = "CONSTRUCTIBLE"
        row["outcome"] = "ACCEPTED"
        accepted.append(built.record)
        artifacts.update(built.artifacts)
        accepted_group[f"{split}:{lesion_class}"] += 1
        accepted_sources[(split, lesion_class)].add(source_id)
        qc_cells[lesion_class].append(built.qc_cell)
        qc_cells[lesion_class].sort(
            key=lambda cell: hashlib.sha256(
                f"{config.twin.seed}:qc:{cell.twin_id}".encode()
            ).hexdigest()
        )
        del qc_cells[lesion_class][config.qc_samples_per_class :]
        return True

    for lesion_class in LESION_CLASSES:
        for split in CANONICAL_SPLITS:
            group = (split, lesion_class)
            minimum_sources = config.minimum_sources_per_split[lesion_class][split]
            sources = sorted(
                {key[2] for key in by_source if key[:2] == group},
                key=lambda source: _source_order(config, lesion_class, split, source),
            )
            for source_id in sources:
                if len(accepted_sources[group]) >= minimum_sources:
                    break
                for record in by_source[(split, lesion_class, source_id)]:
                    if attempt(record):
                        break
            quota = config.twin.split_quotas[lesion_class][split]
            for record in grouped[group]:
                if accepted_group[f"{split}:{lesion_class}"] >= quota:
                    break
                component_id = _string(record.get("component_id"), "component ID")
                if component_id not in attempted:
                    attempt(record)

    for record in component_records:
        component_id = _string(record.get("component_id"), "component ID")
        row = selection_lookup[component_id]
        if row["eligibility_reason"] == "ELIGIBLE" and component_id not in attempted:
            row["constructibility_reason"] = "NOT_EVALUATED_AFTER_STOP"
            row["outcome"] = "NOT_REACHED_AFTER_QUOTA_FILLED"

    selection_rows.sort(
        key=lambda row: (str(row["lesion_class"]), str(row["split"]), str(row["selection_key"]))
    )
    accepted_counts = {
        f"{split}:{lesion_class}": accepted_group[f"{split}:{lesion_class}"]
        for lesion_class in LESION_CLASSES
        for split in CANONICAL_SPLITS
    }
    distinct_sources = {
        f"{split}:{lesion_class}": len(accepted_sources[(split, lesion_class)])
        for lesion_class in LESION_CLASSES
        for split in CANONICAL_SPLITS
    }
    target_sufficient = {
        lesion_class: all(
            accepted_group[f"{split}:{lesion_class}"]
            >= config.twin.split_quotas[lesion_class][split]
            for split in CANONICAL_SPLITS
        )
        for lesion_class in LESION_CLASSES
    }
    source_sufficient = {
        lesion_class: all(
            len(accepted_sources[(split, lesion_class)])
            >= config.minimum_sources_per_split[lesion_class][split]
            for split in CANONICAL_SPLITS
        )
        for lesion_class in LESION_CLASSES
    }
    all_sufficient = all(target_sufficient.values()) and all(source_sufficient.values())
    status = READY_STATUS if all_sufficient else BLOCKED_STATUS
    raw_equal_diagnostic = Counter(
        str(record["lesion_class"])
        for record in accepted
        if cast(dict[str, JSONValue], record["oracle"])["raw_array_exact_equal"] is True
    )
    method_counts = Counter(str(record["method"]) for record in accepted)
    component_counts = Counter(str(record["lesion_class"]) for record in component_records)
    manifest: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": "ddr",
        "dataset_version": config.twin.dataset_version,
        "scope": "DDR_TRAIN_VALID_TIER_A_CONSTRUCTION_RECOVERY",
        "config_sha256": config.config_sha256,
        "bindings": cast(dict[str, JSONValue], upstream_hashes),
        "accepted": cast(list[JSONValue], accepted),
        "rejected": cast(list[JSONValue], rejected),
        "support_oracle_status": "DEFERRED_TO_S5_DEGRADATION_OBSERVATION",
    }
    inventory: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "component_count": len(component_records),
        "component_counts": cast(dict[str, JSONValue], dict(sorted(component_counts.items()))),
        "records": cast(list[JSONValue], component_records),
    }
    artifacts["component-inventory-v5.json"] = canonical_json_bytes(inventory)
    artifacts["twin-manifest-v5.json"] = canonical_json_bytes(manifest)
    artifacts["selection-audit-v5.csv"] = s4_twins._selection_csv(selection_rows)  # noqa: SLF001
    artifacts["accepted-rejected-v5.csv"] = s4_twins._accepted_rejected_csv(  # noqa: SLF001
        accepted, rejected
    )
    for lesion_class in LESION_CLASSES:
        artifacts[f"qc-{lesion_class.lower()}-ddr-recovery-contact-sheet.png"] = _contact_sheet(
            qc_cells[lesion_class],
            columns=config.qc_columns,
            panel_size=config.qc_panel_size,
        )
    audit: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "config_sha256": config.config_sha256,
        "summary": {
            "adapter_source_count": len(adapter_records),
            "component_count": len(component_records),
            "component_counts": cast(dict[str, JSONValue], dict(sorted(component_counts.items()))),
            "accepted_twin_count": len(accepted),
            "rejected_attempt_count": len(rejected),
            "paired_output_count": len(accepted) * 2,
            "accepted_counts": cast(dict[str, JSONValue], accepted_counts),
            "accepted_distinct_sources": cast(dict[str, JSONValue], distinct_sources),
            "method_counts": cast(dict[str, JSONValue], dict(sorted(method_counts.items()))),
            "raw_edit_array_equal_diagnostic": cast(
                dict[str, JSONValue],
                {
                    lesion_class: raw_equal_diagnostic[lesion_class]
                    for lesion_class in LESION_CLASSES
                },
            ),
            "selection_record_count": len(selection_rows),
        },
        "gate": {
            "upstream_hash_replay": "PASS",
            "adapter_hash_replay": "PASS",
            "target_count_sufficiency": cast(dict[str, JSONValue], target_sufficient),
            "distinct_source_sufficiency": cast(dict[str, JSONValue], source_sufficient),
            "complete_pairs": "PASS",
            "outside_edit_exact_preservation": "PASS",
            "target_preservation_in_sham": "PASS",
            "support_oracle": "DEFERRED_TO_S5_DEGRADATION_OBSERVATION",
            "tier_b": "NOT_AVAILABLE",
            "ai_contact_sheet_review": "PENDING",
            "s5_degradation_calibration_authorized": False,
            "s6_or_later_authorized": False,
        },
        "io_audit": {
            "development_image_hash_reads": io_counts["image_hash_reads"],
            "development_mask_hash_reads": io_counts["mask_hash_reads"],
            "development_image_decodes": io_counts["image_decodes"],
            "development_mask_decodes": io_counts["mask_decodes"],
            "development_bytes_hashed": io_counts["bytes_hashed"],
            "ddr_test_content_reads": 0,
            "ddr_test_image_decodes": 0,
            "locked_test_root_reads": 0,
            "source_split_leaks": 0,
        },
        "hashes": {
            "code_sha256": _code_sha256(),
            "component_inventory_sha256": sha256_bytes(artifacts["component-inventory-v5.json"]),
            "twin_manifest_sha256": sha256_bytes(artifacts["twin-manifest-v5.json"]),
            "selection_audit_sha256": sha256_bytes(artifacts["selection-audit-v5.csv"]),
            "accepted_rejected_sha256": sha256_bytes(artifacts["accepted-rejected-v5.csv"]),
        },
        "evidence_boundary": [
            "These are controlled development twins, not FCR, medical, clinical, or paper results.",
            "Raw edited-array equality is diagnostic only and is not a Tier-A "
            "observation collision.",
            "All support truth remains ORACLE_UNRESOLVED until S5 freezes the observation domain.",
            "No DDR TEST, locked test, RetinaEntail score, S5, S6, S7, or enhancer input was read.",
        ],
    }
    artifacts["audit.json"] = canonical_json_bytes(audit)
    hashes = _write_artifacts(output, artifacts)
    return DDRRecoveryResult(
        status=status,
        output_dir=output,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        artifact_hashes=hashes,
        audit=audit,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded DDR development-only CLI."""

    parser = argparse.ArgumentParser(description="Science S4 DDR recovery twin v5.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run v5 and stop before S5."""

    args = build_parser().parse_args(argv)
    try:
        result = run_ddr_recovery(
            config_path=cast(Path, args.config),
            contract_root=cast(Path, args.contract_root),
            output_dir=cast(Path, args.output_dir),
        )
    except (
        FileExistsError,
        OSError,
        S4DDRRecoveryError,
        s4_twins.S4TwinDataError,
        s4_twins.S4TwinGateError,
    ) as error:
        print(f"Science S4 DDR recovery fail-closed: {error}", file=sys.stderr)
        return 1
    print(f"Science S4 DDR recovery status: {result.status}")
    print(f"accepted={result.accepted_count} rejected={result.rejected_count}")
    for name, digest in result.artifact_hashes.items():
        print(f"{name}: {digest}")
    return 0 if result.status == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_STATUS",
    "DDRRecoveryConfig",
    "DDRRecoveryResult",
    "FINAL_READY_STATUS",
    "READY_STATUS",
    "S4DDRRecoveryError",
    "build_parser",
    "decode_ddr_mask",
    "main",
    "run_ddr_recovery",
]
