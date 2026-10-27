"""Science S5 real-quality calibration and regional Tier-A Oracle.

The implementation is deliberately independent of RetinaEntail scores.  It
uses only hash-bound DeepDRiD quality labels/features to freeze a degradation
domain, then compares normalized observations from the accepted S4 twins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps

from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import (
    OperationTrace,
    apply_and_record,
    replay_trace,
)
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S5_DEGRADATION_DOMAIN_CALIBRATION_V1"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
LESION_CLASSES: Final = ("HE", "EX", "MA", "SE")
CANONICAL_SPLITS: Final = ("TRAIN", "CALIBRATION")
QUALITY_FEATURES: Final = (
    "fov_fraction",
    "luminance_median",
    "luminance_iqr",
    "robust_contrast",
    "gradient_sharpness",
    "hf_residual_mad",
    "periphery_center_ratio",
    "block_illumination_spread",
    "jpeg_blockiness",
    "glare_fraction",
    "color_cast_magnitude",
)


class S5CalibrationError(RuntimeError):
    """Raised when the frozen S5 contract cannot be replayed exactly."""


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(cast(JSONValue, value))


@dataclass(frozen=True, slots=True)
class DeepDRiDRecord:
    """Quality-only record from one frozen regular-fundus CSV row."""

    split: str
    patient_id: str
    image_id: str
    relative_path: str
    overall_quality: int
    clarity: int
    field_definition: int
    artifact: int
    git_blob_sha1: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class FamilyRule:
    """One preregistered family signature and direction."""

    name: str
    feature: str
    direction: str


@dataclass(frozen=True, slots=True)
class S5Config:
    """Validated executable S5 contract."""

    path: Path
    payload: dict[str, object]
    config_sha256: str
    archive_name: str
    archive_sha256: str
    archive_byte_size: int
    zip_prefix: str
    commit: str
    tree_sha: str
    tree_json_sha256: str
    image_counts: dict[str, int]
    patient_counts: dict[str, int]
    quality_counts: dict[str, dict[int, int]]
    selected_image_bytes: int
    csv_paths: dict[str, str]
    csv_blob_sha1: dict[str, str]
    readme_paths: dict[str, str]
    readme_blob_sha1: dict[str, str]
    resize_wh: tuple[int, int]
    fov_threshold: int
    block_grid: int
    seed: int
    calibration_source_count: int
    severities: tuple[float, ...]
    train_quantiles: tuple[float, float]
    minimum_interval_overlap: float
    minimum_direction_fraction: float
    minimum_valid_overlap: float
    family_rules: tuple[FamilyRule, ...]
    context_margin: int
    observation_wh: tuple[int, int]
    minimum_collisions: int
    minimum_sources: int
    qc_samples: int

    @classmethod
    def load(cls, path: Path) -> S5Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S5 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("status") != "EXECUTABLE_CALIBRATION"
        ):
            raise S5CalibrationError("unsupported S5 config")
        _verify_baseline(_string(root.get("expected_baseline_commit"), "baseline"))
        dataset = _mapping(root.get("dataset"), "dataset")
        archive = _mapping(dataset.get("archive"), "archive")
        if (
            dataset.get("dataset_id") != "deepdrid"
            or dataset.get("source_splits") != ["TRAIN", "VALID"]
            or dataset.get("canonical_split_mapping") != {"TRAIN": "TRAIN", "VALID": "CALIBRATION"}
            or dataset.get("online_evaluation_reads_allowed") is not False
            or dataset.get("uwf_reads_allowed") is not False
            or dataset.get("locked_test_root_reads_allowed") is not False
            or dataset.get("dr_grade_use_allowed") is not False
        ):
            raise S5CalibrationError("DeepDRiD boundary changed")
        decode = _mapping(root.get("decode"), "decode")
        if (
            decode.get("backend") != "Pillow"
            or decode.get("mode") != "RGB"
            or decode.get("exif_orientation") != "APPLY_TRANSPOSE_THEN_DROP"
            or decode.get("resize_resample") != "BOX"
            or decode.get("bad_file_policy") != "FAIL_CLOSED"
        ):
            raise S5CalibrationError("decode contract changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        features = _sequence(calibration.get("quality_features"), "quality features")
        if tuple(features) != QUALITY_FEATURES:
            raise S5CalibrationError("quality feature contract changed")
        severities = tuple(
            _number(item, "severity", 0.0, 1.0)
            for item in _sequence(calibration.get("severity_grid"), "severity grid")
        )
        if severities != (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            raise S5CalibrationError("severity grid changed")
        families = _mapping(calibration.get("families"), "families")
        rules: list[FamilyRule] = []
        for name, raw in families.items():
            rule = _mapping(raw, f"family {name}")
            feature = _string(rule.get("signature_feature"), f"{name} feature")
            direction = _string(rule.get("direction"), f"{name} direction")
            if feature not in QUALITY_FEATURES and feature != "luminance_deviation":
                raise S5CalibrationError(f"unknown signature feature for {name}")
            if direction not in {"INCREASE", "DECREASE"}:
                raise S5CalibrationError(f"unknown direction for {name}")
            rules.append(FamilyRule(name=name, feature=feature, direction=direction))
        required = {
            "gaussian_blur",
            "defocus_blur",
            "motion_blur",
            "downsample_restore",
            "jpeg_compression",
            "poisson_gaussian_noise",
            "gamma_exposure",
            "vignetting",
            "nonuniform_illumination",
            "localized_shadow",
            "glare_reflection",
            "haze_veil",
            "color_cast",
        }
        if set(families) != required:
            raise S5CalibrationError("required degradation families changed")
        if calibration.get("id_composition_orders") != [
            ["gaussian_blur", "poisson_gaussian_noise"],
            ["nonuniform_illumination", "jpeg_compression"],
            ["haze_veil", "color_cast"],
        ]:
            raise S5CalibrationError("ID composition-order contract changed")
        if calibration.get("far_ood_families") != ["dust_occlusion", "UNREGISTERED_FAMILY"]:
            raise S5CalibrationError("far-OOD family contract changed")
        if calibration.get("gamma_parameter_policy") != {
            "gamma": "1_PLUS_SEVERITY",
            "gain": "1_MINUS_0P35_TIMES_SEVERITY",
        }:
            raise S5CalibrationError("gamma calibration parameter policy changed")
        oracle = _mapping(root.get("regional_oracle"), "regional oracle")
        if (
            oracle.get("normalization") != "PILLOW_BOX_RGB_UINT8"
            or oracle.get("additional_bit_quantization") is not False
            or oracle.get("collision_modes") != ["SAME_THETA", "TWO_THETA"]
            or oracle.get("mode_priority") != "SAME_THETA_BEFORE_TWO_THETA"
            or oracle.get("search_order") != "SHA256_SEED_CLASS_SPLIT_TWIN_ID_MODE_TRACE_ID_ASC"
            or oracle.get("glare_center_policy")
            != "TARGET_CONTEXT_CENTER_IF_WITHIN_CALIBRATED_0P2_0P8_RANGE"
            or oracle.get("tier_b_status") != "NOT_AVAILABLE"
            or oracle.get("retinaentail_independent") is not True
        ):
            raise S5CalibrationError("regional Oracle contract changed")
        gate = _mapping(root.get("gate"), "gate")
        if not gate or any(value is not True for value in gate.values()):
            raise S5CalibrationError("one or more S5 gates are disabled")
        bindings = _mapping(root.get("bindings"), "bindings")
        for name, binding in bindings.items():
            _verify_binding(_mapping(binding, f"binding {name}"), name)
        counts_raw = _mapping(dataset.get("expected_image_counts"), "image counts")
        patients_raw = _mapping(dataset.get("expected_patient_counts"), "patient counts")
        quality_raw = _mapping(dataset.get("expected_quality_counts"), "quality counts")
        quality_counts: dict[str, dict[int, int]] = {}
        for split in ("TRAIN", "VALID"):
            split_counts = _mapping(quality_raw.get(split), f"{split} quality counts")
            quality_counts[split] = {
                0: _integer(split_counts.get("0"), f"{split} poor", 0),
                1: _integer(split_counts.get("1"), f"{split} good", 0),
            }
        csv_config = _mapping(dataset.get("csv"), "CSV")
        csv_paths: dict[str, str] = {}
        csv_shas: dict[str, str] = {}
        readme_paths: dict[str, str] = {}
        readme_shas: dict[str, str] = {}
        readme_config = _mapping(dataset.get("readme"), "Readme")
        for split in ("TRAIN", "VALID"):
            item = _mapping(csv_config.get(split), f"{split} CSV")
            csv_paths[split] = _safe_relative_string(item.get("path"), f"{split} CSV path")
            csv_shas[split] = _sha1(item.get("git_blob_sha1"), f"{split} CSV blob")
            readme_item = _mapping(readme_config.get(split), f"{split} Readme")
            readme_paths[split] = _safe_relative_string(
                readme_item.get("path"), f"{split} Readme path"
            )
            readme_shas[split] = _sha1(readme_item.get("git_blob_sha1"), f"{split} Readme blob")
        semantics = _mapping(dataset.get("quality_label_semantics"), "quality semantics")
        if semantics != {
            "overall_quality_0": "NOT_GOOD_ENOUGH_FOR_DIAGNOSIS",
            "overall_quality_1": "GOOD_ENOUGH_FOR_DIAGNOSIS",
            "clarity_direction": "HIGHER_IS_CLEARER",
            "field_definition_direction": "HIGHER_IS_BETTER",
            "artifact_direction": "HIGHER_IS_MORE_SEVERE",
        }:
            raise S5CalibrationError("DeepDRiD quality-label semantics changed")
        resize = _integer_pair(decode.get("resize_wh"), "resize", minimum=32)
        quantiles = _number_pair(calibration.get("train_envelope_quantiles"), "quantiles")
        if not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
            raise S5CalibrationError("invalid train envelope quantiles")
        return cls(
            path=resolved,
            payload=root,
            config_sha256=sha256_file(resolved),
            archive_name=_string(archive.get("filename"), "archive filename"),
            archive_sha256=_sha256(archive.get("sha256"), "archive SHA-256"),
            archive_byte_size=_integer(archive.get("byte_size"), "archive bytes", 1),
            zip_prefix=_safe_relative_string(archive.get("zip_prefix"), "zip prefix"),
            commit=_hex(archive.get("commit"), "commit", 40),
            tree_sha=_hex(archive.get("tree_sha"), "tree SHA", 40),
            tree_json_sha256=_sha256(archive.get("tree_json_sha256"), "tree JSON SHA-256"),
            image_counts={
                split: _integer(counts_raw.get(split), f"{split} count", 1)
                for split in ("TRAIN", "VALID")
            },
            patient_counts={
                split: _integer(patients_raw.get(split), f"{split} patients", 1)
                for split in ("TRAIN", "VALID")
            },
            quality_counts=quality_counts,
            selected_image_bytes=_integer(
                dataset.get("expected_selected_image_bytes"), "selected image bytes", 1
            ),
            csv_paths=csv_paths,
            csv_blob_sha1=csv_shas,
            readme_paths=readme_paths,
            readme_blob_sha1=readme_shas,
            resize_wh=resize,
            fov_threshold=_integer(decode.get("fov_foreground_max_rgb_min"), "FOV threshold", 0),
            block_grid=_integer(calibration.get("block_grid"), "block grid", 2),
            seed=_integer(calibration.get("seed"), "seed", 0),
            calibration_source_count=_integer(
                calibration.get("synthetic_source_count"), "synthetic sources", 1
            ),
            severities=severities,
            train_quantiles=quantiles,
            minimum_interval_overlap=_number(
                calibration.get("minimum_train_interval_overlap"), "train interval overlap", 0, 1
            ),
            minimum_direction_fraction=_number(
                calibration.get("minimum_direction_fraction"), "direction fraction", 0, 1
            ),
            minimum_valid_overlap=_number(
                calibration.get("minimum_valid_interval_overlap"), "valid interval overlap", 0, 1
            ),
            family_rules=tuple(rules),
            context_margin=_integer(oracle.get("context_margin_px"), "context margin", 0),
            observation_wh=_integer_pair(oracle.get("observation_wh"), "observation", minimum=1),
            minimum_collisions=_integer(
                oracle.get("minimum_collisions_per_class"), "collision minimum", 1
            ),
            minimum_sources=_integer(
                oracle.get("minimum_distinct_sources_per_class"), "source minimum", 1
            ),
            qc_samples=_integer(
                _mapping(root.get("qc"), "QC").get("samples_per_group"), "QC samples", 1
            ),
        )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S5CalibrationError(f"{name} must be a mapping")
    return {str(key): cast(object, item) for key, item in value.items()}


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise S5CalibrationError(f"{name} must be a sequence")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise S5CalibrationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S5CalibrationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S5CalibrationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise S5CalibrationError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _number_pair(value: object, name: str) -> tuple[float, float]:
    items = _sequence(value, name)
    if len(items) != 2:
        raise S5CalibrationError(f"{name} must have two values")
    return (_number(items[0], name, 0.0, 1.0), _number(items[1], name, 0.0, 1.0))


def _integer_pair(value: object, name: str, *, minimum: int) -> tuple[int, int]:
    items = _sequence(value, name)
    if len(items) != 2:
        raise S5CalibrationError(f"{name} must have two values")
    return (_integer(items[0], name, minimum), _integer(items[1], name, minimum))


def _hex(value: object, name: str, length: int) -> str:
    result = _string(value, name).lower()
    if len(result) != length or any(ch not in "0123456789abcdef" for ch in result):
        raise S5CalibrationError(f"{name} must be {length} lowercase hexadecimal characters")
    return result


def _sha256(value: object, name: str) -> str:
    return _hex(value, name, 64)


def _sha1(value: object, name: str) -> str:
    return _hex(value, name, 40)


def _safe_relative_string(value: object, name: str) -> str:
    text = _string(value, name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise S5CalibrationError(f"{name} is not a safe relative path")
    return path.as_posix()


def _verify_baseline(expected: str) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != expected:
        raise S5CalibrationError("HEAD does not match the frozen S5 baseline")


def _verify_binding(binding: Mapping[str, object], name: str) -> None:
    path = PROJECT_ROOT / _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    if not path.is_file() or sha256_file(path) != expected:
        raise S5CalibrationError(f"binding mismatch: {name}")


def git_blob_sha1_bytes(payload: bytes) -> str:
    """Return the Git blob object ID for exact file bytes."""

    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object identity


def _load_tree(path: Path, config: S5Config) -> dict[str, dict[str, object]]:
    resolved = path.resolve(strict=True)
    if sha256_file(resolved) != config.tree_json_sha256:
        raise S5CalibrationError("official tree JSON hash mismatch")
    decoded: object = json.loads(resolved.read_text(encoding="utf-8"))
    root = _mapping(decoded, "tree JSON")
    if root.get("sha") != config.tree_sha or root.get("truncated") is not False:
        raise S5CalibrationError("official tree JSON identity/truncation mismatch")
    entries: dict[str, dict[str, object]] = {}
    for raw in _sequence(root.get("tree"), "tree entries"):
        item = _mapping(raw, "tree entry")
        path_text = _safe_relative_string(item.get("path"), "tree path")
        if path_text in entries:
            raise S5CalibrationError("duplicate path in official tree")
        entries[path_text] = item
    return entries


def _selected_tree_images(
    entries: Mapping[str, Mapping[str, object]], config: S5Config
) -> dict[str, tuple[str, int]]:
    prefixes = {
        "TRAIN": "regular_fundus_images/regular-fundus-training/Images/",
        "VALID": "regular_fundus_images/regular-fundus-validation/Images/",
    }
    selected: dict[str, tuple[str, int]] = {}
    counts: Counter[str] = Counter()
    total = 0
    for path, item in entries.items():
        split = next((key for key, prefix in prefixes.items() if path.startswith(prefix)), None)
        if split is None or item.get("type") != "blob" or not path.lower().endswith(".jpg"):
            continue
        sha = _sha1(item.get("sha"), f"tree blob {path}")
        size = _integer(item.get("size"), f"tree size {path}", 1)
        selected[path] = (sha, size)
        counts[split] += 1
        total += size
    if dict(counts) != config.image_counts or total != config.selected_image_bytes:
        raise S5CalibrationError("official selected image count/bytes mismatch")
    return selected


def audit_and_extract_archive(
    archive_path: Path,
    tree_json_path: Path,
    extract_root: Path,
    config: S5Config,
) -> dict[str, object]:
    """Verify the immutable archive and extract only frozen regular TRAIN/VALID files."""

    archive = archive_path.resolve(strict=True)
    if archive.name != config.archive_name:
        raise S5CalibrationError("archive filename mismatch")
    if (
        archive.stat().st_size != config.archive_byte_size
        or sha256_file(archive) != config.archive_sha256
    ):
        raise S5CalibrationError("archive byte size/SHA-256 mismatch")
    root = extract_root.resolve()
    if root.exists():
        raise S5CalibrationError(
            "extraction directory already exists; use a new versioned directory"
        )
    entries = _load_tree(tree_json_path, config)
    selected_images = _selected_tree_images(entries, config)
    selected_paths = (
        set(selected_images) | set(config.csv_paths.values()) | set(config.readme_paths.values())
    )
    expected_blobs = {
        **dict(zip(config.csv_paths.values(), config.csv_blob_sha1.values(), strict=True)),
        **dict(zip(config.readme_paths.values(), config.readme_blob_sha1.values(), strict=True)),
    }
    for path, expected_sha in expected_blobs.items():
        item = entries.get(path)
        if (
            item is None
            or item.get("type") != "blob"
            or _sha1(item.get("sha"), f"supporting blob {path}") != expected_sha
        ):
            raise S5CalibrationError(f"supporting file missing from official tree: {path}")
    zip_names = {f"{config.zip_prefix}/{path}": path for path in selected_paths}
    root.mkdir(parents=True, exist_ok=False)
    hashes: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as handle:
        info_by_name = {info.filename.rstrip("/"): info for info in handle.infolist()}
        if set(zip_names).difference(info_by_name):
            raise S5CalibrationError("one or more frozen files are absent from archive")
        for zip_name, relative in sorted(zip_names.items(), key=lambda item: item[1]):
            info = info_by_name[zip_name]
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if info.flag_bits & 0x1 or stat.S_ISLNK(unix_mode) or info.is_dir():
                raise S5CalibrationError(f"unsafe ZIP entry: {zip_name}")
            payload = handle.read(info)
            tree_item = entries[relative]
            expected_size = _integer(tree_item.get("size"), f"tree size {relative}", 1)
            expected_blob = _sha1(tree_item.get("sha"), f"tree blob {relative}")
            actual_blob = git_blob_sha1_bytes(payload)
            if len(payload) != expected_size or actual_blob != expected_blob:
                raise S5CalibrationError(f"Git blob replay mismatch: {relative}")
            target = root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise S5CalibrationError(f"refusing to overwrite extracted file: {relative}")
            write_bytes_fail_safe(target, payload)
            hashes.append(
                {
                    "relative_path": relative,
                    "byte_size": len(payload),
                    "git_blob_sha1": actual_blob,
                    "sha256": sha256_bytes(payload),
                }
            )
    return {
        "archive_sha256": config.archive_sha256,
        "archive_byte_size": config.archive_byte_size,
        "selected_file_count": len(hashes),
        "selected_image_count": len(selected_images),
        "selected_image_bytes": sum(size for _, size in selected_images.values()),
        "files": hashes,
    }


def _csv_image_relative(split: str, value: str) -> str:
    normalized = value.strip().replace("\\", "/").lstrip("/")
    raw_name = "regular-fundus-training" if split == "TRAIN" else "regular-fundus-validation"
    prefix = f"{raw_name}/"
    if not normalized.startswith(prefix):
        raise S5CalibrationError("CSV image path escapes its frozen split")
    tail = normalized[len(prefix) :]
    return f"regular_fundus_images/{raw_name}/Images/{tail}"


def load_deepdrid_records(
    extracted_root: Path,
    selected_manifest: Mapping[str, object],
    config: S5Config,
) -> tuple[DeepDRiDRecord, ...]:
    """Load only quality fields and bind every row to the selected-file manifest."""

    file_rows = _sequence(selected_manifest.get("files"), "selected manifest files")
    files = {
        _safe_relative_string(
            _mapping(item, "selected file").get("relative_path"), "path"
        ): _mapping(item, "selected file")
        for item in file_rows
    }
    records: list[DeepDRiDRecord] = []
    for raw_split in ("TRAIN", "VALID"):
        csv_relative = config.csv_paths[raw_split]
        csv_path = extracted_root.joinpath(*PurePosixPath(csv_relative).parts)
        payload = csv_path.read_bytes()
        if git_blob_sha1_bytes(payload) != config.csv_blob_sha1[raw_split]:
            raise S5CalibrationError(f"{raw_split} CSV Git blob mismatch")
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        required = {
            "patient_id",
            "image_id",
            "image_path",
            "Overall quality",
            "Clarity",
            "Field definition",
            "Artifact",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise S5CalibrationError("DeepDRiD CSV quality fields are incomplete")
        for row in reader:
            relative = _csv_image_relative(raw_split, row["image_path"])
            file_item = files.get(relative)
            if file_item is None:
                raise S5CalibrationError(f"CSV image absent from selected manifest: {relative}")
            records.append(
                DeepDRiDRecord(
                    split="TRAIN" if raw_split == "TRAIN" else "CALIBRATION",
                    patient_id=_string(row["patient_id"], "patient ID"),
                    image_id=_string(row["image_id"], "image ID"),
                    relative_path=relative,
                    overall_quality=_csv_code(row["Overall quality"], "overall quality", {0, 1}),
                    clarity=_csv_code(row["Clarity"], "clarity", {1, 4, 6, 8, 10}),
                    field_definition=_csv_code(
                        row["Field definition"], "field definition", {1, 4, 6, 8, 10}
                    ),
                    artifact=_csv_code(row["Artifact"], "artifact", {0, 1, 4, 6, 8, 10}),
                    git_blob_sha1=_sha1(file_item.get("git_blob_sha1"), "image Git blob"),
                    byte_size=_integer(file_item.get("byte_size"), "image bytes", 1),
                )
            )
    _validate_records(records, config)
    return tuple(records)


def _csv_code(value: str, name: str, allowed: set[int]) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise S5CalibrationError(f"{name} is not an integer") from exc
    if result not in allowed:
        raise S5CalibrationError(f"{name} has an unknown code")
    return result


def _validate_records(records: Sequence[DeepDRiDRecord], config: S5Config) -> None:
    if len({record.relative_path for record in records}) != len(records):
        raise S5CalibrationError("duplicate DeepDRiD image row")
    split_counts = Counter("TRAIN" if r.split == "TRAIN" else "VALID" for r in records)
    patient_sets = {
        raw: {r.patient_id for r in records if r.split == canonical}
        for raw, canonical in (("TRAIN", "TRAIN"), ("VALID", "CALIBRATION"))
    }
    if dict(split_counts) != config.image_counts:
        raise S5CalibrationError("DeepDRiD CSV split counts mismatch")
    if {key: len(value) for key, value in patient_sets.items()} != config.patient_counts:
        raise S5CalibrationError("DeepDRiD patient counts mismatch")
    if patient_sets["TRAIN"].intersection(patient_sets["VALID"]):
        raise S5CalibrationError("DeepDRiD patient split leak")
    quality = {
        raw: dict(Counter(r.overall_quality for r in records if r.split == canonical))
        for raw, canonical in (("TRAIN", "TRAIN"), ("VALID", "CALIBRATION"))
    }
    if quality != config.quality_counts:
        raise S5CalibrationError("DeepDRiD quality-label counts mismatch")


def decode_quality_image(path: Path, config: S5Config) -> NDArray[np.uint8]:
    """Decode one hash-verified quality image under the frozen Pillow rules."""

    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgb = oriented.convert("RGB")
            resized = rgb.resize(config.resize_wh, resample=Image.Resampling.BOX)
            array = np.asarray(resized, dtype=np.uint8).copy()
    except Exception as exc:
        raise S5CalibrationError(f"bad DeepDRiD image: {path.name}") from exc
    expected_shape = (config.resize_wh[1], config.resize_wh[0], 3)
    if array.shape != expected_shape or array.dtype != np.uint8:
        raise S5CalibrationError("decoded image violates RGB uint8 contract")
    return array


def quality_features(
    image: NDArray[np.uint8], *, fov_threshold: int, block_grid: int
) -> dict[str, float]:
    """Compute the frozen deterministic quality-feature vector inside the FOV."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise S5CalibrationError("quality feature input must be RGB uint8")
    rgb = image.astype(np.float64) / 255.0
    mask = image.max(axis=2) >= fov_threshold
    if int(mask.sum()) < max(16, image.shape[0] * image.shape[1] // 100):
        raise S5CalibrationError("FOV mask is too small")
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    values = luminance[mask]
    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    gx = np.abs(np.diff(luminance, axis=1))
    gy = np.abs(np.diff(luminance, axis=0))
    gx_mask = mask[:, 1:] & mask[:, :-1]
    gy_mask = mask[1:, :] & mask[:-1, :]
    gradients = np.concatenate((gx[gx_mask], gy[gy_mask]))
    padded = np.pad(luminance, 1, mode="edge")
    local_mean = (
        sum(
            padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
            for dy in range(3)
            for dx in range(3)
        )
        / 9.0
    )
    residual = np.abs(luminance - local_mean)[mask]
    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - (width - 1) / 2.0) / max(width / 2.0, 1.0)
    ny = (yy - (height - 1) / 2.0) / max(height / 2.0, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    center = mask & (radius <= 0.45)
    periphery = mask & (radius >= 0.72) & (radius <= 1.0)
    if not center.any() or not periphery.any():
        raise S5CalibrationError("FOV lacks center/periphery support")
    block_medians: list[float] = []
    for block_rows in np.array_split(np.arange(height), block_grid):
        for block_columns in np.array_split(np.arange(width), block_grid):
            block_mask = mask[np.ix_(block_rows, block_columns)]
            if block_mask.mean() >= 0.25:
                block_medians.append(
                    float(np.median(luminance[np.ix_(block_rows, block_columns)][block_mask]))
                )
    if len(block_medians) < block_grid:
        raise S5CalibrationError("insufficient FOV blocks")
    vertical_boundaries = [column for column in range(8, width, 8)]
    horizontal_boundaries = [row for row in range(8, height, 8)]
    boundary_terms: list[float] = []
    for boundary_column in vertical_boundaries:
        valid = mask[:, boundary_column] & mask[:, boundary_column - 1]
        if valid.any():
            boundary_terms.append(
                float(
                    np.mean(
                        np.abs(luminance[:, boundary_column] - luminance[:, boundary_column - 1])[
                            valid
                        ]
                    )
                )
            )
    for boundary_row in horizontal_boundaries:
        valid = mask[boundary_row, :] & mask[boundary_row - 1, :]
        if valid.any():
            boundary_terms.append(
                float(
                    np.mean(
                        np.abs(luminance[boundary_row, :] - luminance[boundary_row - 1, :])[valid]
                    )
                )
            )
    all_neighbor = float(np.mean(gradients)) if gradients.size else 0.0
    blockiness = max(0.0, float(np.mean(boundary_terms)) - all_neighbor) if boundary_terms else 0.0
    channel_sum = rgb.sum(axis=2)
    valid_sum = channel_sum[mask]
    chroma = rgb[mask] / np.maximum(valid_sum[:, None], 1e-8)
    median_chroma = np.median(chroma, axis=0)
    return {
        "fov_fraction": float(mask.mean()),
        "luminance_median": float(q50),
        "luminance_iqr": float(q75 - q25),
        "robust_contrast": float(q95 - q05),
        "gradient_sharpness": float(np.median(gradients)) if gradients.size else 0.0,
        "hf_residual_mad": float(np.median(residual)),
        "periphery_center_ratio": float(
            np.median(luminance[periphery]) / max(float(np.median(luminance[center])), 1e-8)
        ),
        "block_illumination_spread": float(
            np.quantile(block_medians, 0.9) - np.quantile(block_medians, 0.1)
        ),
        "jpeg_blockiness": blockiness,
        "glare_fraction": float(
            (
                (image.max(axis=2) >= 250) & ((image.max(axis=2) - image.min(axis=2)) <= 12) & mask
            ).sum()
            / mask.sum()
        ),
        "color_cast_magnitude": float(np.linalg.norm(median_chroma - (1.0 / 3.0))),
    }


def _record_path(root: Path, record: DeepDRiDRecord) -> Path:
    return root.joinpath(*PurePosixPath(record.relative_path).parts)


def _file_manifest_map(selected_manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw in _sequence(selected_manifest.get("files"), "selected manifest files"):
        item = _mapping(raw, "selected file")
        relative = _safe_relative_string(item.get("relative_path"), "selected path")
        result[relative] = item
    return result


def compute_real_feature_records(
    records: Sequence[DeepDRiDRecord],
    extracted_root: Path,
    selected_manifest: Mapping[str, object],
    config: S5Config,
) -> tuple[list[dict[str, object]], dict[str, NDArray[np.uint8]]]:
    """Hash, decode, and featurize every frozen TRAIN/VALID image exactly once."""

    files = _file_manifest_map(selected_manifest)
    rows: list[dict[str, object]] = []
    qc_images: dict[str, NDArray[np.uint8]] = {}
    qc_candidates: dict[str, list[DeepDRiDRecord]] = {"good": [], "poor": []}
    for record in records:
        path = _record_path(extracted_root, record)
        item = files.get(record.relative_path)
        if item is None or not path.is_file():
            raise S5CalibrationError("selected image is absent at feature decode")
        if path.stat().st_size != record.byte_size:
            raise S5CalibrationError("selected image byte size changed")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != _sha256(item.get("sha256"), "selected file SHA-256"):
            raise S5CalibrationError("selected image SHA-256 changed")
        image = decode_quality_image(path, config)
        features = quality_features(
            image,
            fov_threshold=config.fov_threshold,
            block_grid=config.block_grid,
        )
        rows.append(
            {
                "split": record.split,
                "patient_id": record.patient_id,
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "image_sha256": actual_sha256,
                "overall_quality": record.overall_quality,
                "clarity": record.clarity,
                "field_definition": record.field_definition,
                "artifact": record.artifact,
                "features": features,
            }
        )
        if record.split == "TRAIN":
            group = "good" if record.overall_quality == 1 else "poor"
            qc_candidates[group].append(record)
    for group, candidates in qc_candidates.items():
        ordered = sorted(
            candidates,
            key=lambda record: _selection_digest(config.seed, "QC", group, record.image_id),
        )[: config.qc_samples]
        for record in ordered:
            qc_images[f"{group}:{record.image_id}"] = decode_quality_image(
                _record_path(extracted_root, record), config
            )
    return rows, qc_images


def _selection_digest(seed: int, *parts: object) -> str:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).hexdigest()


def select_synthetic_sources(
    records: Sequence[DeepDRiDRecord], config: S5Config
) -> tuple[DeepDRiDRecord, ...]:
    """Select one good TRAIN image per distinct patient in frozen hash order."""

    candidates = [
        record for record in records if record.split == "TRAIN" and record.overall_quality == 1
    ]
    ordered = sorted(
        candidates,
        key=lambda record: _selection_digest(config.seed, record.patient_id, record.image_id),
    )
    selected: list[DeepDRiDRecord] = []
    seen: set[str] = set()
    for record in ordered:
        if record.patient_id in seen:
            continue
        seen.add(record.patient_id)
        selected.append(record)
        if len(selected) == config.calibration_source_count:
            break
    if len(selected) != config.calibration_source_count:
        raise S5CalibrationError("insufficient distinct good TRAIN calibration patients")
    return tuple(selected)


def _feature_value(
    features: Mapping[str, float], feature: str, good_luminance_median: float
) -> float:
    if feature == "luminance_deviation":
        return abs(features["luminance_median"] - good_luminance_median)
    return features[feature]


def _central_interval(
    values: Sequence[float], quantiles: tuple[float, float]
) -> tuple[float, float]:
    if not values:
        raise S5CalibrationError("cannot form an interval from no values")
    result = np.quantile(np.asarray(values, dtype=np.float64), quantiles)
    return (float(result[0]), float(result[1]))


def interval_overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return intersection length divided by the smaller nonzero interval width."""

    lower = max(first[0], second[0])
    upper = min(first[1], second[1])
    intersection = max(0.0, upper - lower)
    widths = (max(0.0, first[1] - first[0]), max(0.0, second[1] - second[0]))
    denominator = min(width for width in widths if width > 0.0) if any(widths) else 0.0
    if denominator == 0.0:
        return 1.0 if math.isclose(first[0], second[0], abs_tol=1e-12) else 0.0
    return min(1.0, intersection / denominator)


def _operator_seed(config: S5Config, family: str, image_id: str) -> int:
    digest = _selection_digest(config.seed, "SYNTH", family, image_id)
    return int(digest[:16], 16)


def _frozen_family_parameters(family: str, severity: float) -> dict[str, float] | None:
    if family == "gamma_exposure":
        return {"gamma": 1.0 + severity, "gain": 1.0 - 0.35 * severity}
    return None


def fit_degradation_domain(
    real_rows: Sequence[Mapping[str, object]],
    sources: Sequence[DeepDRiDRecord],
    extracted_root: Path,
    config: S5Config,
) -> tuple[dict[str, object], dict[str, list[NDArray[np.uint8]]]]:
    """Fit family ID limits on TRAIN only, then apply the frozen VALID gate."""

    train_good_luminance = [
        cast(Mapping[str, float], row["features"])["luminance_median"]
        for row in real_rows
        if row["split"] == "TRAIN" and row["overall_quality"] == 1
    ]
    good_luminance_median = float(np.median(train_good_luminance))
    source_images = {
        source.image_id: decode_quality_image(_record_path(extracted_root, source), config)
        for source in sources
    }
    family_results: dict[str, object] = {}
    ladder_images: dict[str, list[NDArray[np.uint8]]] = {}
    replay_checks = 0
    replay_mismatches = 0
    for rule in config.family_rules:
        poor_train = [
            _feature_value(
                cast(Mapping[str, float], row["features"]),
                rule.feature,
                good_luminance_median,
            )
            for row in real_rows
            if row["split"] == "TRAIN" and row["overall_quality"] == 0
        ]
        poor_valid = [
            _feature_value(
                cast(Mapping[str, float], row["features"]),
                rule.feature,
                good_luminance_median,
            )
            for row in real_rows
            if row["split"] == "CALIBRATION" and row["overall_quality"] == 0
        ]
        train_interval = _central_interval(poor_train, config.train_quantiles)
        valid_interval = _central_interval(poor_valid, config.train_quantiles)
        by_severity: dict[float, list[float]] = defaultdict(list)
        severity_records: list[dict[str, object]] = []
        representative = source_images[sources[0].image_id]
        ladder: list[NDArray[np.uint8]] = []
        per_source: dict[str, list[float]] = defaultdict(list)
        for severity in config.severities:
            traces: list[str] = []
            for source in sources:
                image = source_images[source.image_id]
                seed = _operator_seed(config, rule.name, source.image_id)
                output, step = apply_and_record(
                    image,
                    rule.name,
                    severity=severity,
                    seed=seed,
                    parameters=_frozen_family_parameters(rule.name, severity),
                )
                replay = replay_trace(image, OperationTrace((step,)))
                replay_checks += 1
                if not np.array_equal(output, replay):
                    replay_mismatches += 1
                features = quality_features(
                    output,
                    fov_threshold=config.fov_threshold,
                    block_grid=config.block_grid,
                )
                value = _feature_value(features, rule.feature, good_luminance_median)
                by_severity[severity].append(value)
                per_source[source.image_id].append(value)
                traces.append(sha256_bytes(_canonical(OperationTrace((step,)).to_dict())))
            rep_seed = _operator_seed(config, rule.name, sources[0].image_id)
            rep_output, _ = apply_and_record(
                representative,
                rule.name,
                severity=severity,
                seed=rep_seed,
                parameters=_frozen_family_parameters(rule.name, severity),
            )
            ladder.append(rep_output)
            interval = _central_interval(by_severity[severity], config.train_quantiles)
            severity_records.append(
                {
                    "severity": severity,
                    "median": float(np.median(by_severity[severity])),
                    "interval": list(interval),
                    "train_poor_interval_overlap": interval_overlap(interval, train_interval),
                    "trace_set_sha256": sha256_bytes(_canonical(sorted(traces))),
                }
            )
        ladder_images[rule.name] = ladder
        baseline_median = float(np.median(by_severity[0.0]))
        eligible: list[float] = []
        direction_by_severity: dict[float, float] = {}
        for index, severity in enumerate(config.severities[1:], start=1):
            direction_flags: list[bool] = []
            for source in sources:
                values = per_source[source.image_id]
                changes = np.diff(values[: index + 1])
                if rule.direction == "INCREASE":
                    direction_flags.append(bool(np.all(changes >= -1e-12)))
                else:
                    direction_flags.append(bool(np.all(changes <= 1e-12)))
            fraction = sum(direction_flags) / len(direction_flags)
            direction_by_severity[severity] = fraction
            median = float(np.median(by_severity[severity]))
            moved = (
                median > baseline_median
                if rule.direction == "INCREASE"
                else median < baseline_median
            )
            overlap = cast(float, severity_records[index]["train_poor_interval_overlap"])
            if (
                moved
                and fraction >= config.minimum_direction_fraction
                and overlap >= config.minimum_interval_overlap
            ):
                eligible.append(severity)
        id_max = max(eligible, default=0.0)
        id_values = [
            value
            for severity, values in by_severity.items()
            if severity <= id_max
            for value in values
        ]
        synthetic_id_interval = (
            _central_interval(id_values, config.train_quantiles)
            if id_values
            else _central_interval(by_severity[0.0], config.train_quantiles)
        )
        valid_overlap = interval_overlap(synthetic_id_interval, valid_interval)
        next_levels = [severity for severity in config.severities if severity > id_max]
        family_results[rule.name] = {
            "signature_feature": rule.feature,
            "direction": rule.direction,
            "train_poor_interval": list(train_interval),
            "valid_poor_interval": list(valid_interval),
            "severity_records": severity_records,
            "direction_fraction_by_severity": {
                f"{severity:.1f}": fraction for severity, fraction in direction_by_severity.items()
            },
            "id_severity_max": id_max,
            "id_severity_grid": [severity for severity in config.severities if severity <= id_max],
            "near_ood_severity": next_levels[0] if next_levels else None,
            "synthetic_id_interval": list(synthetic_id_interval),
            "valid_interval_overlap": valid_overlap,
            "nonzero_id_gate": id_max > 0.0,
            "valid_coverage_gate": valid_overlap >= config.minimum_valid_overlap,
        }
    required_pass = all(
        cast(Mapping[str, object], result)["nonzero_id_gate"] is True
        and cast(Mapping[str, object], result)["valid_coverage_gate"] is True
        for result in family_results.values()
    )
    return (
        {
            "good_train_luminance_median": good_luminance_median,
            "synthetic_source_count": len(sources),
            "synthetic_distinct_patient_count": len({source.patient_id for source in sources}),
            "source_selection_sha256": sha256_bytes(
                _canonical(
                    [
                        {
                            "patient_id": source.patient_id,
                            "image_id": source.image_id,
                            "git_blob_sha1": source.git_blob_sha1,
                        }
                        for source in sources
                    ]
                )
            ),
            "families": family_results,
            "trace_replay_checks": replay_checks,
            "trace_replay_mismatches": replay_mismatches,
            "required_family_gate": required_pass,
        },
        ladder_images,
    )


def audit_noncommutative_orders(
    source: NDArray[np.uint8], calibration: Mapping[str, object], config: S5Config
) -> tuple[list[dict[str, object]], dict[str, list[NDArray[np.uint8]]]]:
    """Replay the three frozen ordered pairs and require their outputs to differ."""

    family_results = _mapping(calibration.get("families"), "calibration families")
    pairs = (
        ("gaussian_blur", "poisson_gaussian_noise", "blur_noise"),
        ("nonuniform_illumination", "jpeg_compression", "illumination_jpeg"),
        ("haze_veil", "color_cast", "haze_color_cast"),
    )
    records: list[dict[str, object]] = []
    images: dict[str, list[NDArray[np.uint8]]] = {}
    for first, second, label in pairs:
        first_result = _mapping(family_results.get(first), first)
        second_result = _mapping(family_results.get(second), second)
        severity = min(
            _number(first_result.get("id_severity_max"), "first ID maximum", 0.0, 1.0),
            _number(second_result.get("id_severity_max"), "second ID maximum", 0.0, 1.0),
        )
        if severity <= 0.0:
            records.append(
                {
                    "pair": [first, second],
                    "severity": severity,
                    "noncommutative": False,
                    "reason": "NO_SHARED_NONZERO_ID_SEVERITY",
                }
            )
            images[label] = [source.copy(), source.copy()]
            continue
        seed_first = int(_selection_digest(config.seed, "ORDER", label, first)[:16], 16)
        seed_second = int(_selection_digest(config.seed, "ORDER", label, second)[:16], 16)
        first_output, first_step = apply_and_record(
            source, first, severity=severity, seed=seed_first
        )
        forward, second_step = apply_and_record(
            first_output, second, severity=severity, seed=seed_second
        )
        second_output, reverse_second = apply_and_record(
            source, second, severity=severity, seed=seed_second
        )
        reverse, reverse_first = apply_and_record(
            second_output, first, severity=severity, seed=seed_first
        )
        forward_trace = OperationTrace((first_step, second_step))
        reverse_trace = OperationTrace((reverse_second, reverse_first))
        replay_ok = np.array_equal(forward, replay_trace(source, forward_trace)) and np.array_equal(
            reverse, replay_trace(source, reverse_trace)
        )
        records.append(
            {
                "pair": [first, second],
                "severity": severity,
                "noncommutative": not np.array_equal(forward, reverse),
                "differing_values": int(np.count_nonzero(forward != reverse)),
                "forward_trace": forward_trace.to_dict(),
                "reverse_trace": reverse_trace.to_dict(),
                "trace_replay_exact": replay_ok,
            }
        )
        images[label] = [forward, reverse]
    return records, images


def _resize_panel(image: NDArray[np.uint8], size: int) -> Image.Image:
    return Image.fromarray(image).resize((size, size), resample=Image.Resampling.BOX)


def make_labeled_contact_sheet(
    groups: Sequence[tuple[str, NDArray[np.uint8]]],
    *,
    columns: int,
    panel_size: int = 160,
) -> bytes:
    """Render a deterministic PNG contact sheet with a short label per panel."""

    label_height = 28
    rows = max(1, math.ceil(len(groups) / columns))
    canvas = Image.new("RGB", (columns * panel_size, rows * (panel_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, array) in enumerate(groups):
        row_index, column_index = divmod(index, columns)
        x = column_index * panel_size
        y = row_index * (panel_size + label_height)
        canvas.paste(_resize_panel(array, panel_size), (x, y))
        draw.text((x + 3, y + panel_size + 3), label[:40], fill="black", font=font)
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def regional_context_bbox(
    twin: Mapping[str, object], *, margin: int, patch_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Map the original target bbox to a square, margin-expanded patch context."""

    source = _mapping(twin.get("source"), "twin source")
    geometry = _mapping(source.get("patch_geometry"), "patch geometry")
    crop = _sequence(geometry.get("actual_crop_bbox_xyxy"), "crop bbox")
    target = _mapping(twin.get("target"), "twin target")
    bbox = _sequence(target.get("bbox_xyxy"), "target bbox")
    if len(crop) != 4 or len(bbox) != 4:
        raise S5CalibrationError("target/crop bbox must have four coordinates")
    crop_values = [_integer(value, "crop coordinate", 0) for value in crop]
    target_values = [_integer(value, "target coordinate", 0) for value in bbox]
    left = target_values[0] - crop_values[0] - margin
    top = target_values[1] - crop_values[1] - margin
    right = target_values[2] - crop_values[0] + margin
    bottom = target_values[3] - crop_values[1] + margin
    side = max(right - left, bottom - top)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    square_left = math.floor(center_x - side / 2.0)
    square_top = math.floor(center_y - side / 2.0)
    square_right = square_left + side
    square_bottom = square_top + side
    width, height = patch_wh
    if square_left < 0:
        square_right -= square_left
        square_left = 0
    if square_top < 0:
        square_bottom -= square_top
        square_top = 0
    if square_right > width:
        square_left -= square_right - width
        square_right = width
    if square_bottom > height:
        square_top -= square_bottom - height
        square_bottom = height
    if (
        square_left < 0
        or square_top < 0
        or square_right > width
        or square_bottom > height
        or square_right - square_left != square_bottom - square_top
        or square_right <= square_left
    ):
        raise S5CalibrationError("regional context cannot be kept inside patch")
    return (square_left, square_top, square_right, square_bottom)


def normalize_regional_observation(
    degraded: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
    observation_wh: tuple[int, int],
) -> NDArray[np.uint8]:
    """Crop and BOX-normalize one full-patch RGB uint8 observation."""

    if degraded.dtype != np.uint8 or degraded.ndim != 3 or degraded.shape[2] != 3:
        raise S5CalibrationError("regional Oracle input must be RGB uint8")
    left, top, right, bottom = bbox
    if left < 0 or top < 0 or right > degraded.shape[1] or bottom > degraded.shape[0]:
        raise S5CalibrationError("regional context is outside degraded patch")
    crop = Image.fromarray(degraded[top:bottom, left:right])
    normalized = crop.resize(observation_wh, resample=Image.Resampling.BOX)
    return np.asarray(normalized, dtype=np.uint8).copy()


def _decode_twin_png(path: Path, expected_sha256: str) -> NDArray[np.uint8]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise S5CalibrationError("S4 twin PNG hash mismatch")
    try:
        with Image.open(path) as source:
            source.load()
            if source.size != (256, 256):
                raise S5CalibrationError("S4 twin patch size changed")
            array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    except S5CalibrationError:
        raise
    except Exception as exc:
        raise S5CalibrationError("S4 twin PNG decode failed") from exc
    return array


def _twin_output(twin: Mapping[str, object], name: str, s4_root: Path) -> tuple[Path, str]:
    outputs = _mapping(twin.get("outputs"), "twin outputs")
    item = _mapping(outputs.get(name), name)
    relative = _safe_relative_string(item.get("relative_path"), f"{name} path")
    expected_sha = _sha256(item.get("sha256"), f"{name} SHA-256")
    return s4_root.joinpath(*PurePosixPath(relative).parts), expected_sha


def _trace_identifier(trace: OperationTrace) -> str:
    return sha256_bytes(_canonical(trace.to_dict()))


def _oracle_trace_candidates(
    twin: Mapping[str, object],
    family_results: Mapping[str, object],
    bbox: tuple[int, int, int, int],
    config: S5Config,
    source: NDArray[np.uint8],
) -> list[tuple[str, OperationTrace]]:
    candidates: list[tuple[str, OperationTrace]] = []
    twin_id = _string(twin.get("twin_id"), "twin ID")
    center_x = ((bbox[0] + bbox[2]) / 2.0) / source.shape[1]
    center_y = ((bbox[1] + bbox[3]) / 2.0) / source.shape[0]
    for family, raw in family_results.items():
        result = _mapping(raw, f"family {family}")
        levels = _sequence(result.get("id_severity_grid"), f"{family} ID grid")
        for raw_severity in levels:
            severity = _number(raw_severity, "ID severity", 0.0, 1.0)
            if severity <= 0.0:
                continue
            seed = int(
                _selection_digest(config.seed, "ORACLE", twin_id, family, f"{severity:.1f}")[:16],
                16,
            )
            parameters = _frozen_family_parameters(family, severity)
            if family == "glare_reflection":
                if not 0.2 <= center_x <= 0.8 or not 0.2 <= center_y <= 0.8:
                    continue
                parameters = {"center_x": center_x, "center_y": center_y}
            _, step = apply_and_record(
                source,
                family,
                severity=severity,
                seed=seed,
                parameters=parameters,
            )
            trace = OperationTrace((step,))
            candidates.append((_trace_identifier(trace), trace))
    return candidates


def _observation_pair(
    plus: NDArray[np.uint8],
    minus: NDArray[np.uint8],
    plus_trace: OperationTrace,
    minus_trace: OperationTrace,
    bbox: tuple[int, int, int, int],
    config: S5Config,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    degraded_plus = replay_trace(plus, plus_trace)
    degraded_minus = replay_trace(minus, minus_trace)
    return (
        normalize_regional_observation(degraded_plus, bbox, config.observation_wh),
        normalize_regional_observation(degraded_minus, bbox, config.observation_wh),
    )


def run_regional_tier_a_oracle(
    s4_root: Path,
    calibration: Mapping[str, object],
    config: S5Config,
) -> tuple[dict[str, object], dict[str, bytes], dict[str, list[tuple[str, NDArray[np.uint8]]]]]:
    """Search the frozen SAME_THETA then TWO_THETA regional exact collisions."""

    manifest_path = s4_root / "twin-manifest-v5.json"
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted = _sequence(_mapping(manifest, "S4 twin manifest").get("accepted"), "accepted twins")
    family_results = _mapping(calibration.get("families"), "calibration families")
    domain_ready = calibration.get("domain_gate") is True
    records: list[dict[str, object]] = []
    artifacts: dict[str, bytes] = {}
    qc: dict[str, list[tuple[str, NDArray[np.uint8]]]] = {
        lesion_class: [] for lesion_class in LESION_CLASSES
    }
    collisions: Counter[str] = Counter()
    collision_sources: dict[str, set[str]] = {
        lesion_class: set() for lesion_class in LESION_CLASSES
    }
    tested: Counter[str] = Counter()
    ordered_twins = sorted(
        accepted,
        key=lambda raw: _selection_digest(
            config.seed,
            "TWIN",
            _mapping(raw, "accepted twin").get("lesion_class"),
            _mapping(raw, "accepted twin").get("split"),
            _mapping(raw, "accepted twin").get("twin_id"),
        ),
    )
    for raw in ordered_twins:
        twin = _mapping(raw, "accepted twin")
        lesion_class = _string(twin.get("lesion_class"), "lesion class")
        if lesion_class not in LESION_CLASSES:
            raise S5CalibrationError("unexpected S4 lesion class")
        twin_id = _string(twin.get("twin_id"), "twin ID")
        source_id = _string(twin.get("source_image_id"), "source image ID")
        split = _string(twin.get("split"), "split")
        if split not in CANONICAL_SPLITS:
            raise S5CalibrationError("unexpected S4 twin split")
        if not domain_ready:
            records.append(
                {
                    "twin_id": twin_id,
                    "lesion_class": lesion_class,
                    "split": split,
                    "source_image_id": source_id,
                    "support_truth": "ORACLE_UNRESOLVED",
                    "domain_truth": "OUT_OF_DOMAIN",
                    "oracle_tier": None,
                    "collision_mode": None,
                    "reason": "S5_DEGRADATION_DOMAIN_GATE_FAILED",
                }
            )
            continue
        plus_path, plus_sha = _twin_output(twin, "x_plus_sham", s4_root)
        minus_path, minus_sha = _twin_output(twin, "x_minus_removed", s4_root)
        plus = _decode_twin_png(plus_path, plus_sha)
        minus = _decode_twin_png(minus_path, minus_sha)
        bbox = regional_context_bbox(twin, margin=config.context_margin, patch_wh=(256, 256))
        identity_trace = OperationTrace()
        baseline_observations = _observation_pair(
            plus, minus, identity_trace, identity_trace, bbox, config
        )
        if not np.array_equal(*baseline_observations) and len(qc[lesion_class]) < max(
            1, config.qc_samples // 2
        ):
            baseline_combined = np.concatenate(baseline_observations, axis=1)
            qc[lesion_class].append((f"BASELINE NONCOLLISION {twin_id[-12:]}", baseline_combined))
        candidates = _oracle_trace_candidates(twin, family_results, bbox, config, plus)
        attempts: list[tuple[str, str, OperationTrace, OperationTrace]] = []
        for trace_id, trace in candidates:
            order = _selection_digest(
                config.seed,
                lesion_class,
                split,
                twin_id,
                "SAME_THETA",
                trace_id,
            )
            attempts.append((order, "SAME_THETA", trace, trace))
        by_family: dict[str, list[tuple[str, OperationTrace]]] = defaultdict(list)
        for trace_id, trace in candidates:
            by_family[trace.operations[0].name].append((trace_id, trace))
        for family_candidates in by_family.values():
            for plus_id, plus_trace in family_candidates:
                for minus_id, minus_trace in family_candidates:
                    if plus_id == minus_id:
                        continue
                    trace_id = sha256_bytes(f"{plus_id}|{minus_id}".encode())
                    order = _selection_digest(
                        config.seed,
                        lesion_class,
                        split,
                        twin_id,
                        "TWO_THETA",
                        trace_id,
                    )
                    attempts.append((order, "TWO_THETA", plus_trace, minus_trace))
        attempts.sort(key=lambda item: (item[1] != "SAME_THETA", item[0]))
        collision: dict[str, object] | None = None
        last_observations: tuple[NDArray[np.uint8], NDArray[np.uint8]] | None = None
        observation_cache: dict[tuple[str, str], NDArray[np.uint8]] = {}
        for _, mode, plus_trace, minus_trace in attempts:
            plus_trace_id = _trace_identifier(plus_trace)
            minus_trace_id = _trace_identifier(minus_trace)
            plus_key = ("x_plus", plus_trace_id)
            minus_key = ("x_minus", minus_trace_id)
            if plus_key not in observation_cache:
                observation_cache[plus_key] = normalize_regional_observation(
                    replay_trace(plus, plus_trace), bbox, config.observation_wh
                )
            if minus_key not in observation_cache:
                observation_cache[minus_key] = normalize_regional_observation(
                    replay_trace(minus, minus_trace), bbox, config.observation_wh
                )
            observations = (
                observation_cache[plus_key],
                observation_cache[minus_key],
            )
            last_observations = observations
            tested[lesion_class] += 1
            if not np.array_equal(*observations):
                continue
            plus_observation, minus_observation = observations
            plus_array_sha = sha256_bytes(plus_observation.tobytes(order="C"))
            minus_array_sha = sha256_bytes(minus_observation.tobytes(order="C"))
            evidence = {
                "schema_version": SCHEMA_VERSION,
                "twin_id": twin_id,
                "component_id": _string(twin.get("component_id"), "component ID"),
                "source_image_id": source_id,
                "lesion_class": lesion_class,
                "split": split,
                "source_truth_pair": ["PRESENT", "ABSENT"],
                "support_truth": "OBSERVATION_AMBIGUOUS",
                "domain_truth": "IN_DOMAIN",
                "oracle_tier": "A",
                "collision_mode": mode,
                "oracle_method_id": "S5_REGIONAL_BOX32_EXACT_COLLISION_V1",
                "retinaentail_independent": True,
                "regional_bbox_xyxy": list(bbox),
                "observation_wh": list(config.observation_wh),
                "x_plus_input_sha256": plus_sha,
                "x_minus_input_sha256": minus_sha,
                "x_plus_trace": plus_trace.to_dict(),
                "x_minus_trace": minus_trace.to_dict(),
                "x_plus_observation_array_sha256": plus_array_sha,
                "x_minus_observation_array_sha256": minus_array_sha,
            }
            evidence_sha = sha256_bytes(_canonical(evidence))
            collision = {**evidence, "evidence_sha256": evidence_sha}
            collisions[lesion_class] += 1
            collision_sources[lesion_class].add(source_id)
            witness_root = f"oracle-witnesses/{lesion_class.lower()}/{twin_id}"
            for side, observation in (
                ("x_plus", plus_observation),
                ("x_minus", minus_observation),
            ):
                stream = io.BytesIO()
                Image.fromarray(observation).save(stream, format="PNG", optimize=False)
                artifacts[f"{witness_root}/{side}.png"] = stream.getvalue()
            artifacts[f"{witness_root}/evidence.json"] = _canonical(collision)
            combined = np.concatenate((plus_observation, minus_observation), axis=1)
            if len(qc[lesion_class]) < config.qc_samples:
                qc[lesion_class].append((f"COLLISION {twin_id[-12:]}", combined))
            break
        if collision is None:
            record: dict[str, object] = {
                "twin_id": twin_id,
                "lesion_class": lesion_class,
                "split": split,
                "source_image_id": source_id,
                "support_truth": "ORACLE_UNRESOLVED",
                "domain_truth": "IN_DOMAIN",
                "oracle_tier": None,
                "collision_mode": None,
                "reason": "NO_EXACT_COLLISION_IN_FROZEN_ID_SEARCH",
                "regional_bbox_xyxy": list(bbox),
                "attempt_count": len(attempts),
            }
            if last_observations is not None and len(qc[lesion_class]) < config.qc_samples:
                combined = np.concatenate(last_observations, axis=1)
                qc[lesion_class].append((f"UNRESOLVED {twin_id[-12:]}", combined))
            records.append(record)
        else:
            records.append(collision)
    collision_gate = {
        lesion_class: (
            collisions[lesion_class] >= config.minimum_collisions
            and len(collision_sources[lesion_class]) >= config.minimum_sources
        )
        for lesion_class in LESION_CLASSES
    }
    return (
        {
            "method_id": "S5_REGIONAL_BOX32_EXACT_COLLISION_V1",
            "domain_ready": domain_ready,
            "twin_record_count": len(records),
            "collision_counts": {
                lesion_class: collisions[lesion_class] for lesion_class in LESION_CLASSES
            },
            "collision_distinct_sources": {
                key: len(value) for key, value in collision_sources.items()
            },
            "search_attempt_counts": {
                lesion_class: tested[lesion_class] for lesion_class in LESION_CLASSES
            },
            "collision_gate": collision_gate,
            "all_classes_gate": all(collision_gate.values()),
            "records": records,
        },
        artifacts,
        qc,
    )


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _csv_feature_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "split",
        "patient_id",
        "image_id",
        "relative_path",
        "image_sha256",
        "overall_quality",
        "clarity",
        "field_definition",
        "artifact",
        *QUALITY_FEATURES,
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        features = cast(Mapping[str, float], row["features"])
        writer.writerow(
            {
                **{key: row[key] for key in fields[:9]},
                **{key: f"{features[key]:.17g}" for key in QUALITY_FEATURES},
            }
        )
    return stream.getvalue().encode()


def _write_artifacts(output: Path, artifacts: Mapping[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    manifest_rows: list[dict[str, object]] = []
    for relative, payload in sorted(artifacts.items()):
        safe = _safe_relative_string(relative, "artifact path")
        target = output.joinpath(*PurePosixPath(safe).parts)
        write_bytes_fail_safe(target, payload)
        digest = sha256_bytes(payload)
        hashes[safe] = digest
        manifest_rows.append({"relative_path": safe, "byte_size": len(payload), "sha256": digest})
    manifest_payload = _canonical({"files": manifest_rows})
    write_bytes_fail_safe(output / "artifact-manifest.json", manifest_payload)
    hashes["artifact-manifest.json"] = sha256_bytes(manifest_payload)
    return hashes


def run_s5(
    config: S5Config,
    *,
    archive_path: Path,
    tree_json_path: Path,
    external_run_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Execute the frozen real calibration and, when allowed, S4 Oracle search."""

    external = external_run_root.resolve()
    output = output_dir.resolve()
    if external.exists() or output.exists():
        raise S5CalibrationError("run directories already exist; use new versioned paths")
    external.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    extracted = external / "selected-regular-fundus"
    selected_manifest = audit_and_extract_archive(archive_path, tree_json_path, extracted, config)
    write_bytes_fail_safe(external / "selected-file-manifest.json", _canonical(selected_manifest))
    records = load_deepdrid_records(extracted, selected_manifest, config)
    real_rows, quality_qc = compute_real_feature_records(
        records, extracted, selected_manifest, config
    )
    feature_payload = _csv_feature_bytes(real_rows)
    write_bytes_fail_safe(external / "real-quality-features.csv", feature_payload)
    sources = select_synthetic_sources(records, config)
    calibration, ladders = fit_degradation_domain(real_rows, sources, extracted, config)
    representative = decode_quality_image(_record_path(extracted, sources[0]), config)
    order_records, order_images = audit_noncommutative_orders(representative, calibration, config)
    order_gate = all(
        record.get("noncommutative") is True and record.get("trace_replay_exact") is True
        for record in order_records
    )
    calibration["ordered_combinations"] = order_records
    calibration["ordered_combination_gate"] = order_gate
    calibration["ood_disposition"] = {
        "id": "REGISTERED_SINGLE_FAMILY_WITHIN_FITTED_ID_GRID_OR_FROZEN_FORWARD_ORDER",
        "near_ood": "SAME_FAMILY_NEXT_GRID_LEVEL_ABOVE_ID_MAXIMUM",
        "far_ood": ["dust_occlusion", "UNREGISTERED_FAMILY"],
        "compositional_ood": [
            ["poisson_gaussian_noise", "gaussian_blur"],
            ["jpeg_compression", "nonuniform_illumination"],
            ["color_cast", "haze_veil"],
            "ANY_UNFROZEN_ORDER",
        ],
    }
    calibration["domain_gate"] = (
        calibration["required_family_gate"] is True
        and calibration["trace_replay_mismatches"] == 0
        and order_gate
    )
    calibration["real_feature_table_sha256"] = sha256_bytes(feature_payload)
    write_bytes_fail_safe(external / "degradation-calibration-v1.json", _canonical(calibration))
    bindings = _mapping(config.payload.get("bindings"), "bindings")
    s4_binding = _mapping(bindings.get("s4_v5_twin_manifest"), "S4 twin manifest binding")
    s4_manifest_path = PROJECT_ROOT / _safe_relative_string(
        s4_binding.get("path"), "S4 twin manifest path"
    )
    s4_root = s4_manifest_path.parent
    oracle, oracle_artifacts, oracle_qc = run_regional_tier_a_oracle(s4_root, calibration, config)
    calibration_gate = calibration["domain_gate"] is True
    automated_gate = calibration_gate and oracle["all_classes_gate"] is True
    artifacts: dict[str, bytes] = {
        "selected-file-manifest.json": _canonical(selected_manifest),
        "degradation-domain-v1.json": _canonical(calibration),
        "regional-tier-a-oracle-v1.json": _canonical(oracle),
        **oracle_artifacts,
    }
    quality_groups = [(label, image) for label, image in sorted(quality_qc.items())]
    artifacts["qc-deepdrid-good-poor-contact-sheet.png"] = make_labeled_contact_sheet(
        quality_groups, columns=min(8, config.qc_samples)
    )
    for family in ("localized_shadow", "glare_reflection", "haze_veil", "color_cast"):
        artifacts[f"qc-{family.replace('_', '-')}-severity-ladder.png"] = (
            make_labeled_contact_sheet(
                [
                    (f"severity={severity:.1f}", image)
                    for severity, image in zip(config.severities, ladders[family], strict=True)
                ],
                columns=len(config.severities),
            )
        )
    for label, images in order_images.items():
        artifacts[f"qc-order-{label.replace('_', '-')}.png"] = make_labeled_contact_sheet(
            [("forward", images[0]), ("reverse", images[1])], columns=2
        )
    for lesion_class in LESION_CLASSES:
        cells = oracle_qc[lesion_class]
        if not cells:
            cells = [("NO IN-DOMAIN OBSERVATION", np.zeros((32, 64, 3), dtype=np.uint8))]
        artifacts[f"qc-{lesion_class.lower()}-regional-oracle-contact-sheet.png"] = (
            make_labeled_contact_sheet(cells, columns=4)
        )
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_FOR_AI_QC" if automated_gate else "BLOCKED_AUTOMATED_GATE",
        "config_sha256": config.config_sha256,
        "dataset": {
            "commit": config.commit,
            "tree_sha": config.tree_sha,
            "archive_sha256": config.archive_sha256,
            "archive_byte_size": config.archive_byte_size,
            "image_counts": config.image_counts,
            "patient_counts": config.patient_counts,
            "quality_counts": config.quality_counts,
            "selected_image_bytes": config.selected_image_bytes,
        },
        "calibration": {
            "synthetic_source_count": calibration["synthetic_source_count"],
            "synthetic_distinct_patient_count": calibration["synthetic_distinct_patient_count"],
            "trace_replay_checks": calibration["trace_replay_checks"],
            "trace_replay_mismatches": calibration["trace_replay_mismatches"],
            "required_family_gate": calibration["required_family_gate"],
            "ordered_combination_gate": order_gate,
        },
        "oracle": {
            key: oracle[key]
            for key in (
                "domain_ready",
                "twin_record_count",
                "collision_counts",
                "collision_distinct_sources",
                "search_attempt_counts",
                "collision_gate",
                "all_classes_gate",
            )
        },
        "io_audit": {
            "real_feature_image_decodes": sum(config.image_counts.values()),
            "synthetic_source_decode_operations": len(sources) + 1,
            "quality_qc_decode_operations": len(quality_qc),
            "online_evaluation_reads": 0,
            "uwf_reads": 0,
            "locked_test_root_reads": 0,
            "dr_grade_values_used": 0,
            "source_split_leaks": 0,
        },
        "gate": {
            "archive_and_git_blob_replay": "PASS",
            "real_feature_calibration": "PASS" if calibration_gate else "FAIL",
            "regional_tier_a_oracle": "PASS" if oracle["all_classes_gate"] else "FAIL",
            "ai_visual_qc": "PENDING",
            "s6_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "official_tree_json_sha256": config.tree_json_sha256,
            "selected_file_manifest_sha256": sha256_bytes(artifacts["selected-file-manifest.json"]),
            "degradation_domain_sha256": sha256_bytes(artifacts["degradation-domain-v1.json"]),
            "regional_oracle_sha256": sha256_bytes(artifacts["regional-tier-a-oracle-v1.json"]),
        },
        "evidence_boundary": [
            "Quality calibration uses DeepDRiD regular TRAIN/VALID only.",
            "The regional Oracle is independent of RetinaEntail scores and four-valued outputs.",
            "Twin, contact-sheet, and collision counts are not FCR, medical, clinical, "
            "or paper performance results.",
            "No online evaluation, UWF, locked-test, S6, S7, enhancer, or paper "
            "experiment was run.",
        ],
    }
    artifacts["audit.json"] = _canonical(audit)
    _write_artifacts(output, artifacts)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tree-json", type=Path, required=True)
    parser.add_argument("--external-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = S5Config.load(args.config)
        audit = run_s5(
            config,
            archive_path=args.archive,
            tree_json_path=args.tree_json,
            external_run_root=args.external_run_root,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, S5CalibrationError, zipfile.BadZipFile) as exc:
        print(f"Science S5 failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "READY_FOR_AI_QC" else 3


if __name__ == "__main__":
    raise SystemExit(main())
