"""Science S5 v2 amendment: paired-effect calibration and bounded Tier-A Oracle.

Version 2 preserves the failed v1 run.  It separates real-distribution coverage
from paired operator-effect monotonicity, uses a fixed source FOV for synthetic
measurements, and applies a preregistered target-centred localized-glare Oracle.
No RetinaEntail score or four-valued output is an input to this module.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

import retinaentail.science_s5_degradation_calibration as v1
from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import OperationTrace, apply_and_record, replay_trace
from retinaentail.witnesses.replay import sha256_bytes, write_bytes_fail_safe

SCHEMA_VERSION: Final = "2.0.0"
EXPERIMENT_ID: Final = "S5_DEGRADATION_DOMAIN_CALIBRATION_V2"
ORACLE_METHOD_ID: Final = "S5_REGIONAL_BOX32_TARGETED_GLARE_EXACT_COLLISION_V2"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> v1.S5Config:
    """Load the v2 amendment and inherit immutable data/decode fields from v1."""

    resolved = path.resolve(strict=True)
    decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = v1._mapping(decoded, "S5 v2 config")
    if (
        root.get("schema_version") != SCHEMA_VERSION
        or root.get("experiment_id") != EXPERIMENT_ID
        or root.get("status") != "EXECUTABLE_AMENDMENT"
    ):
        raise v1.S5CalibrationError("unsupported S5 v2 config")
    v1._verify_baseline(v1._string(root.get("expected_baseline_commit"), "baseline"))
    base_binding = v1._mapping(root.get("base_contract"), "base contract")
    v1._verify_binding(base_binding, "base_contract")
    base_path = PROJECT_ROOT / v1._safe_relative_string(base_binding.get("path"), "base path")
    base = v1.S5Config.load(base_path)
    amendment = v1._mapping(root.get("amendment"), "amendment")
    if amendment.get("development_disclosure") != "RUN1_AND_S4_DEVELOPMENT_TWINS_OBSERVED":
        raise v1.S5CalibrationError("v2 development disclosure changed")
    if amendment.get("valid_disposition") != "REUSED_CONFIRMATION_NOT_FRESH_BLIND_HOLDOUT":
        raise v1.S5CalibrationError("v2 VALID disposition changed")
    if amendment.get("synthetic_fov_policy") != "FIXED_FROM_UNDEGRADED_SOURCE":
        raise v1.S5CalibrationError("v2 synthetic FOV policy changed")
    if amendment.get("direction_gate") != "PAIRED_MEAN_ABSOLUTE_RGB_EFFECT_MONOTONICITY":
        raise v1.S5CalibrationError("v2 direction gate changed")
    if amendment.get("real_coverage_gate") != "RAW_SIGNATURE_TRAIN_POOR_INTERVAL_OVERLAP":
        raise v1.S5CalibrationError("v2 coverage gate changed")
    policies = v1._mapping(amendment.get("operator_parameter_policies"), "parameter policies")
    if policies != {
        "poisson_gaussian_noise": {
            "peak_photons": "20000_DIV_MAX_SEVERITY_0P02",
            "gaussian_sigma": "0P001_TIMES_SEVERITY",
        },
        "gamma_exposure": {
            "gamma": "1_PLUS_SEVERITY",
            "gain": "1_MINUS_0P35_TIMES_SEVERITY",
        },
        "glare_reflection": {
            "intensity": "MIN_1_1P25_TIMES_SEVERITY",
            "radius_x": "0P03_PLUS_0P03_TIMES_SEVERITY",
            "radius_y": "0P02_PLUS_0P02_TIMES_SEVERITY",
            "softness": 0.15,
        },
        "haze_veil": {"radial_strength": 0.0},
    }:
        raise v1.S5CalibrationError("v2 parameter policy changed")
    glare = v1._mapping(amendment.get("glare_feature"), "glare feature")
    if glare != {
        "luminance_min": 0.72,
        "saturation_max": 0.22,
        "denominator": "FIXED_FOV_PIXELS",
    }:
        raise v1.S5CalibrationError("v2 glare feature changed")
    oracle = v1._mapping(root.get("regional_oracle_amendment"), "regional Oracle amendment")
    if oracle != {
        "method_id": ORACLE_METHOD_ID,
        "search_policy": "SAME_THETA_GLARE_SEVERITY_1_TARGET_CENTERED_ONLY",
        "unsearched_disposition": "ORACLE_UNRESOLVED",
        "two_theta_disposition": "DEFERRED_NOT_REQUIRED_AFTER_CLASS_MINIMA",
        "minimum_collisions_per_class": 20,
        "minimum_distinct_sources_per_class": 10,
        "retinaentail_independent": True,
    }:
        raise v1.S5CalibrationError("v2 Oracle amendment changed")
    bindings = v1._mapping(root.get("bindings"), "bindings")
    for name, binding in bindings.items():
        v1._verify_binding(v1._mapping(binding, f"binding {name}"), name)
    seed = v1._integer(amendment.get("seed"), "v2 seed", 0)
    return replace(
        base,
        path=resolved,
        payload=root,
        config_sha256=sha256_file(resolved),
        seed=seed,
    )


def quality_features_v2(
    image: NDArray[np.uint8],
    *,
    fov_threshold: int,
    block_grid: int,
    fixed_fov_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Compute v1-compatible features with a fixed-FOV option and corrected glare score."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise v1.S5CalibrationError("quality feature input must be RGB uint8")
    if fixed_fov_mask is None:
        mask = image.max(axis=2) >= fov_threshold
    else:
        if fixed_fov_mask.shape != image.shape[:2] or fixed_fov_mask.dtype != np.bool_:
            raise v1.S5CalibrationError("fixed FOV mask shape/dtype mismatch")
        mask = fixed_fov_mask
    if int(mask.sum()) < max(16, image.shape[0] * image.shape[1] // 100):
        raise v1.S5CalibrationError("FOV mask is too small")
    rgb = image.astype(np.float64) / 255.0
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
        raise v1.S5CalibrationError("FOV lacks center/periphery support")
    block_medians: list[float] = []
    for block_rows in np.array_split(np.arange(height), block_grid):
        for block_columns in np.array_split(np.arange(width), block_grid):
            block_mask = mask[np.ix_(block_rows, block_columns)]
            if block_mask.mean() >= 0.25:
                block_medians.append(
                    float(np.median(luminance[np.ix_(block_rows, block_columns)][block_mask]))
                )
    if len(block_medians) < block_grid:
        raise v1.S5CalibrationError("insufficient FOV blocks")
    boundary_terms: list[float] = []
    for boundary_column in range(8, width, 8):
        valid = mask[:, boundary_column] & mask[:, boundary_column - 1]
        if valid.any():
            delta = np.abs(luminance[:, boundary_column] - luminance[:, boundary_column - 1])
            boundary_terms.append(float(np.mean(delta[valid])))
    for boundary_row in range(8, height, 8):
        valid = mask[boundary_row, :] & mask[boundary_row - 1, :]
        if valid.any():
            delta = np.abs(luminance[boundary_row, :] - luminance[boundary_row - 1, :])
            boundary_terms.append(float(np.mean(delta[valid])))
    all_neighbor = float(np.mean(gradients)) if gradients.size else 0.0
    blockiness = max(0.0, float(np.mean(boundary_terms)) - all_neighbor) if boundary_terms else 0.0
    channel_sum = rgb.sum(axis=2)
    chroma = rgb[mask] / np.maximum(channel_sum[mask, None], 1e-8)
    median_chroma = np.median(chroma, axis=0)
    maximum = rgb.max(axis=2)
    saturation = (maximum - rgb.min(axis=2)) / np.maximum(maximum, 1e-8)
    glare = (luminance >= 0.72) & (saturation <= 0.22) & mask
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
        "glare_fraction": float(glare.sum() / mask.sum()),
        "color_cast_magnitude": float(np.linalg.norm(median_chroma - (1.0 / 3.0))),
    }


def compute_real_feature_records_v2(
    records: Sequence[v1.DeepDRiDRecord],
    extracted_root: Path,
    selected_manifest: Mapping[str, object],
    config: v1.S5Config,
) -> tuple[list[dict[str, object]], dict[str, NDArray[np.uint8]]]:
    """Hash, decode and featurize each exact DeepDRiD image once."""

    files = v1._file_manifest_map(selected_manifest)
    rows: list[dict[str, object]] = []
    candidates: dict[str, list[v1.DeepDRiDRecord]] = {"good": [], "poor": []}
    for record in records:
        path = v1._record_path(extracted_root, record)
        item = files.get(record.relative_path)
        if item is None or not path.is_file() or path.stat().st_size != record.byte_size:
            raise v1.S5CalibrationError("selected image changed before v2 feature decode")
        actual_sha = sha256_file(path)
        if actual_sha != v1._sha256(item.get("sha256"), "selected file SHA-256"):
            raise v1.S5CalibrationError("selected image SHA-256 changed")
        image = v1.decode_quality_image(path, config)
        rows.append(
            {
                "split": record.split,
                "patient_id": record.patient_id,
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "image_sha256": actual_sha,
                "overall_quality": record.overall_quality,
                "clarity": record.clarity,
                "field_definition": record.field_definition,
                "artifact": record.artifact,
                "features": quality_features_v2(
                    image,
                    fov_threshold=config.fov_threshold,
                    block_grid=config.block_grid,
                ),
            }
        )
        if record.split == "TRAIN":
            candidates["good" if record.overall_quality == 1 else "poor"].append(record)
    qc: dict[str, NDArray[np.uint8]] = {}
    for group, group_records in candidates.items():
        ordered = sorted(
            group_records,
            key=lambda record: v1._selection_digest(config.seed, "QC", group, record.image_id),
        )[: config.qc_samples]
        for record in ordered:
            qc[f"{group}:{record.image_id}"] = v1.decode_quality_image(
                v1._record_path(extracted_root, record), config
            )
    return rows, qc


def frozen_family_parameters(
    family: str, severity: float
) -> dict[str, str | int | float | bool] | None:
    """Return the run1-informed parameters frozen before the formal v2 run."""

    if family == "poisson_gaussian_noise":
        return {
            "peak_photons": 20000.0 / max(severity, 0.02),
            "gaussian_sigma": 0.001 * severity,
        }
    if family == "gamma_exposure":
        return {"gamma": 1.0 + severity, "gain": 1.0 - 0.35 * severity}
    if family == "glare_reflection":
        return {
            "intensity": min(1.0, 1.25 * severity),
            "radius_x": 0.03 + 0.03 * severity,
            "radius_y": 0.02 + 0.02 * severity,
            "softness": 0.15,
        }
    if family == "haze_veil":
        return {"radial_strength": 0.0}
    return None


def _effect_score(
    source: NDArray[np.uint8], output: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> float:
    difference = np.abs(output.astype(np.int16) - source.astype(np.int16))
    return float(np.mean(difference[mask]))


def fit_degradation_domain_v2(
    real_rows: Sequence[Mapping[str, object]],
    sources: Sequence[v1.DeepDRiDRecord],
    extracted_root: Path,
    config: v1.S5Config,
) -> tuple[
    dict[str, object],
    dict[str, list[NDArray[np.uint8]]],
    NDArray[np.uint8],
]:
    """Fit raw support and paired-effect gates without conflating the two."""

    train_good_luminance = [
        cast(Mapping[str, float], row["features"])["luminance_median"]
        for row in real_rows
        if row["split"] == "TRAIN" and row["overall_quality"] == 1
    ]
    good_luminance_median = float(np.median(train_good_luminance))
    source_images = {
        source.image_id: v1.decode_quality_image(v1._record_path(extracted_root, source), config)
        for source in sources
    }
    family_results: dict[str, object] = {}
    ladder_images: dict[str, list[NDArray[np.uint8]]] = {}
    replay_checks = 0
    replay_mismatches = 0
    for rule in config.family_rules:
        poor_train = [
            v1._feature_value(
                cast(Mapping[str, float], row["features"]),
                rule.feature,
                good_luminance_median,
            )
            for row in real_rows
            if row["split"] == "TRAIN" and row["overall_quality"] == 0
        ]
        poor_valid = [
            v1._feature_value(
                cast(Mapping[str, float], row["features"]),
                rule.feature,
                good_luminance_median,
            )
            for row in real_rows
            if row["split"] == "CALIBRATION" and row["overall_quality"] == 0
        ]
        train_interval = v1._central_interval(poor_train, config.train_quantiles)
        valid_interval = v1._central_interval(poor_valid, config.train_quantiles)
        by_severity: dict[float, list[float]] = {severity: [] for severity in config.severities}
        effects_by_source: dict[str, list[float]] = {}
        severity_records: list[dict[str, object]] = []
        trace_sets: dict[float, list[str]] = {severity: [] for severity in config.severities}
        ladder: list[NDArray[np.uint8]] = []
        for source in sources:
            image = source_images[source.image_id]
            fixed_mask = image.max(axis=2) >= config.fov_threshold
            effects: list[float] = []
            for severity in config.severities:
                seed = v1._operator_seed(config, rule.name, source.image_id)
                output, step = apply_and_record(
                    image,
                    rule.name,
                    severity=severity,
                    seed=seed,
                    parameters=frozen_family_parameters(rule.name, severity),
                )
                replay = replay_trace(image, OperationTrace((step,)))
                replay_checks += 1
                if not np.array_equal(output, replay):
                    replay_mismatches += 1
                features = quality_features_v2(
                    output,
                    fov_threshold=config.fov_threshold,
                    block_grid=config.block_grid,
                    fixed_fov_mask=fixed_mask,
                )
                by_severity[severity].append(
                    v1._feature_value(features, rule.feature, good_luminance_median)
                )
                effects.append(_effect_score(image, output, fixed_mask))
                trace_sets[severity].append(
                    sha256_bytes(v1._canonical(OperationTrace((step,)).to_dict()))
                )
            effects_by_source[source.image_id] = effects
        representative = source_images[sources[0].image_id]
        for severity in config.severities:
            seed = v1._operator_seed(config, rule.name, sources[0].image_id)
            output, _ = apply_and_record(
                representative,
                rule.name,
                severity=severity,
                seed=seed,
                parameters=frozen_family_parameters(rule.name, severity),
            )
            ladder.append(output)
            interval = v1._central_interval(by_severity[severity], config.train_quantiles)
            severity_records.append(
                {
                    "severity": severity,
                    "raw_signature_median": float(np.median(by_severity[severity])),
                    "raw_signature_interval": list(interval),
                    "train_poor_interval_overlap": v1.interval_overlap(interval, train_interval),
                    "paired_effect_median": float(
                        np.median(
                            [
                                effects_by_source[source.image_id][
                                    config.severities.index(severity)
                                ]
                                for source in sources
                            ]
                        )
                    ),
                    "trace_set_sha256": sha256_bytes(v1._canonical(sorted(trace_sets[severity]))),
                }
            )
        ladder_images[rule.name] = ladder
        prefix: list[float] = []
        fractions: dict[float, float] = {}
        for index, severity in enumerate(config.severities[1:], start=1):
            flags = [
                bool(np.all(np.diff(effects_by_source[source.image_id][: index + 1]) >= -1e-12))
                for source in sources
            ]
            fraction = sum(flags) / len(flags)
            fractions[severity] = fraction
            record = severity_records[index]
            eligible = (
                cast(float, record["paired_effect_median"]) > 0.0
                and fraction >= config.minimum_direction_fraction
                and cast(float, record["train_poor_interval_overlap"])
                >= config.minimum_interval_overlap
            )
            if not eligible:
                break
            prefix.append(severity)
        id_max = max(prefix, default=0.0)
        id_grid = [0.0, *prefix]
        id_values = [
            value
            for severity, values in by_severity.items()
            if severity in id_grid
            for value in values
        ]
        synthetic_id_interval = v1._central_interval(id_values, config.train_quantiles)
        valid_overlap = v1.interval_overlap(synthetic_id_interval, valid_interval)
        next_levels = [severity for severity in config.severities if severity > id_max]
        family_results[rule.name] = {
            "signature_feature": rule.feature,
            "raw_signature_direction": rule.direction,
            "direction_gate": "PAIRED_MEAN_ABSOLUTE_RGB_EFFECT_MONOTONICITY",
            "train_poor_interval": list(train_interval),
            "valid_poor_interval": list(valid_interval),
            "severity_records": severity_records,
            "effect_monotonic_fraction_by_severity": {
                f"{severity:.1f}": fraction for severity, fraction in fractions.items()
            },
            "id_severity_max": id_max,
            "id_severity_grid": id_grid,
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
    calibration = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "development_disclosure": "RUN1_AND_S4_DEVELOPMENT_TWINS_OBSERVED",
        "valid_disposition": "REUSED_CONFIRMATION_NOT_FRESH_BLIND_HOLDOUT",
        "good_train_luminance_median": good_luminance_median,
        "synthetic_source_count": len(sources),
        "synthetic_distinct_patient_count": len({source.patient_id for source in sources}),
        "source_selection_sha256": sha256_bytes(
            v1._canonical(
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
        "synthetic_fov_policy": "FIXED_FROM_UNDEGRADED_SOURCE",
        "families": family_results,
        "trace_replay_checks": replay_checks,
        "trace_replay_mismatches": replay_mismatches,
        "required_family_gate": required_pass,
    }
    return calibration, ladder_images, source_images[sources[0].image_id]


def audit_noncommutative_orders_v2(
    source: NDArray[np.uint8], calibration: Mapping[str, object], config: v1.S5Config
) -> tuple[list[dict[str, object]], dict[str, list[NDArray[np.uint8]]]]:
    """Audit the three frozen forward/reverse order pairs at shared ID severity."""

    families = v1._mapping(calibration.get("families"), "families")
    pairs = (
        ("gaussian_blur", "poisson_gaussian_noise", "blur_noise"),
        ("nonuniform_illumination", "jpeg_compression", "illumination_jpeg"),
        ("haze_veil", "color_cast", "haze_color_cast"),
    )
    records: list[dict[str, object]] = []
    images: dict[str, list[NDArray[np.uint8]]] = {}
    for first, second, label in pairs:
        first_result = v1._mapping(families.get(first), first)
        second_result = v1._mapping(families.get(second), second)
        severity = min(
            v1._number(first_result.get("id_severity_max"), "first ID maximum", 0.0, 1.0),
            v1._number(second_result.get("id_severity_max"), "second ID maximum", 0.0, 1.0),
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
        first_seed = int(v1._selection_digest(config.seed, "ORDER", label, first)[:16], 16)
        second_seed = int(v1._selection_digest(config.seed, "ORDER", label, second)[:16], 16)
        first_output, first_step = apply_and_record(
            source,
            first,
            severity=severity,
            seed=first_seed,
            parameters=frozen_family_parameters(first, severity),
        )
        forward, second_step = apply_and_record(
            first_output,
            second,
            severity=severity,
            seed=second_seed,
            parameters=frozen_family_parameters(second, severity),
        )
        second_output, reverse_second = apply_and_record(
            source,
            second,
            severity=severity,
            seed=second_seed,
            parameters=frozen_family_parameters(second, severity),
        )
        reverse, reverse_first = apply_and_record(
            second_output,
            first,
            severity=severity,
            seed=first_seed,
            parameters=frozen_family_parameters(first, severity),
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


def run_bounded_regional_oracle_v2(
    s4_root: Path,
    calibration: Mapping[str, object],
    config: v1.S5Config,
) -> tuple[dict[str, object], dict[str, bytes], dict[str, list[tuple[str, NDArray[np.uint8]]]]]:
    """Run the frozen target-centred severity-1 localized-glare SAME_THETA Oracle."""

    manifest: object = json.loads((s4_root / "twin-manifest-v5.json").read_text(encoding="utf-8"))
    accepted = v1._sequence(v1._mapping(manifest, "S4 twin manifest").get("accepted"), "twins")
    families = v1._mapping(calibration.get("families"), "families")
    glare = v1._mapping(families.get("glare_reflection"), "glare family")
    glare_grid = {
        v1._number(value, "glare ID severity", 0.0, 1.0)
        for value in v1._sequence(glare.get("id_severity_grid"), "glare grid")
    }
    domain_ready = calibration.get("domain_gate") is True and 1.0 in glare_grid
    records: list[dict[str, object]] = []
    artifacts: dict[str, bytes] = {}
    qc: dict[str, list[tuple[str, NDArray[np.uint8]]]] = {
        lesion_class: [] for lesion_class in v1.LESION_CLASSES
    }
    collisions: Counter[str] = Counter()
    sources: dict[str, set[str]] = {lesion_class: set() for lesion_class in v1.LESION_CLASSES}
    attempts: Counter[str] = Counter()
    ordered = sorted(
        accepted,
        key=lambda raw: v1._selection_digest(
            config.seed,
            "V2_TWIN",
            v1._mapping(raw, "twin").get("lesion_class"),
            v1._mapping(raw, "twin").get("split"),
            v1._mapping(raw, "twin").get("twin_id"),
        ),
    )
    for raw in ordered:
        twin = v1._mapping(raw, "accepted twin")
        lesion_class = v1._string(twin.get("lesion_class"), "lesion class")
        twin_id = v1._string(twin.get("twin_id"), "twin ID")
        source_id = v1._string(twin.get("source_image_id"), "source ID")
        split = v1._string(twin.get("split"), "split")
        if lesion_class not in v1.LESION_CLASSES or split not in v1.CANONICAL_SPLITS:
            raise v1.S5CalibrationError("S4 twin class/split changed")
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
                    "reason": "S5_V2_DEGRADATION_DOMAIN_GATE_FAILED",
                }
            )
            continue
        plus_path, plus_sha = v1._twin_output(twin, "x_plus_sham", s4_root)
        minus_path, minus_sha = v1._twin_output(twin, "x_minus_removed", s4_root)
        plus = v1._decode_twin_png(plus_path, plus_sha)
        minus = v1._decode_twin_png(minus_path, minus_sha)
        bbox = v1.regional_context_bbox(twin, margin=config.context_margin, patch_wh=(256, 256))
        center_x = ((bbox[0] + bbox[2]) / 2.0) / plus.shape[1]
        center_y = ((bbox[1] + bbox[3]) / 2.0) / plus.shape[0]
        if not 0.2 <= center_x <= 0.8 or not 0.2 <= center_y <= 0.8:
            records.append(
                {
                    "twin_id": twin_id,
                    "lesion_class": lesion_class,
                    "split": split,
                    "source_image_id": source_id,
                    "support_truth": "ORACLE_UNRESOLVED",
                    "domain_truth": "IN_DOMAIN",
                    "oracle_tier": None,
                    "collision_mode": None,
                    "reason": "TARGET_CENTER_OUTSIDE_FROZEN_GLARE_CALIBRATION_RANGE",
                    "attempt_count": 0,
                }
            )
            continue
        parameters = {
            **cast(
                dict[str, str | int | float | bool],
                frozen_family_parameters("glare_reflection", 1.0),
            ),
            "center_x": center_x,
            "center_y": center_y,
        }
        seed = int(
            v1._selection_digest(config.seed, "V2_ORACLE", twin_id, "glare_reflection", "1.0")[:16],
            16,
        )
        plus_output, plus_step = apply_and_record(
            plus,
            "glare_reflection",
            severity=1.0,
            seed=seed,
            parameters=parameters,
        )
        minus_output, minus_step = apply_and_record(
            minus,
            "glare_reflection",
            severity=1.0,
            seed=seed,
            parameters=parameters,
        )
        if plus_step != minus_step:
            raise v1.S5CalibrationError("SAME_THETA trace mismatch")
        trace = OperationTrace((plus_step,))
        if not np.array_equal(plus_output, replay_trace(plus, trace)) or not np.array_equal(
            minus_output, replay_trace(minus, trace)
        ):
            raise v1.S5CalibrationError("v2 Oracle trace replay mismatch")
        plus_observation = v1.normalize_regional_observation(
            plus_output, bbox, config.observation_wh
        )
        minus_observation = v1.normalize_regional_observation(
            minus_output, bbox, config.observation_wh
        )
        attempts[lesion_class] += 1
        combined = np.concatenate((plus_observation, minus_observation), axis=1)
        if np.array_equal(plus_observation, minus_observation):
            evidence = {
                "schema_version": SCHEMA_VERSION,
                "twin_id": twin_id,
                "component_id": v1._string(twin.get("component_id"), "component ID"),
                "source_image_id": source_id,
                "lesion_class": lesion_class,
                "split": split,
                "source_truth_pair": ["PRESENT", "ABSENT"],
                "support_truth": "OBSERVATION_AMBIGUOUS",
                "domain_truth": "IN_DOMAIN",
                "oracle_tier": "A",
                "collision_mode": "SAME_THETA",
                "oracle_method_id": ORACLE_METHOD_ID,
                "retinaentail_independent": True,
                "regional_bbox_xyxy": list(bbox),
                "observation_wh": list(config.observation_wh),
                "x_plus_input_sha256": plus_sha,
                "x_minus_input_sha256": minus_sha,
                "x_plus_trace": trace.to_dict(),
                "x_minus_trace": trace.to_dict(),
                "x_plus_observation_array_sha256": sha256_bytes(plus_observation.tobytes()),
                "x_minus_observation_array_sha256": sha256_bytes(minus_observation.tobytes()),
            }
            collision = {
                **evidence,
                "evidence_sha256": sha256_bytes(v1._canonical(evidence)),
            }
            records.append(collision)
            collisions[lesion_class] += 1
            sources[lesion_class].add(source_id)
            witness_root = f"oracle-witnesses/{lesion_class.lower()}/{twin_id}"
            for side, observation in (
                ("x_plus", plus_observation),
                ("x_minus", minus_observation),
            ):
                stream = io.BytesIO()
                Image.fromarray(observation).save(stream, format="PNG", optimize=False)
                artifacts[f"{witness_root}/{side}.png"] = stream.getvalue()
            artifacts[f"{witness_root}/evidence.json"] = v1._canonical(collision)
            if len(qc[lesion_class]) < config.qc_samples:
                qc[lesion_class].append((f"COLLISION {twin_id[-12:]}", combined))
        else:
            records.append(
                {
                    "twin_id": twin_id,
                    "lesion_class": lesion_class,
                    "split": split,
                    "source_image_id": source_id,
                    "support_truth": "ORACLE_UNRESOLVED",
                    "domain_truth": "IN_DOMAIN",
                    "oracle_tier": None,
                    "collision_mode": None,
                    "reason": "NO_EXACT_COLLISION_IN_FROZEN_BOUNDED_SEARCH",
                    "regional_bbox_xyxy": list(bbox),
                    "attempt_count": 1,
                }
            )
            if len(qc[lesion_class]) < config.qc_samples:
                qc[lesion_class].append((f"UNRESOLVED {twin_id[-12:]}", combined))
    collision_gate = {
        lesion_class: collisions[lesion_class] >= config.minimum_collisions
        and len(sources[lesion_class]) >= config.minimum_sources
        for lesion_class in v1.LESION_CLASSES
    }
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "method_id": ORACLE_METHOD_ID,
            "search_policy": "SAME_THETA_GLARE_SEVERITY_1_TARGET_CENTERED_ONLY",
            "unsearched_disposition": "ORACLE_UNRESOLVED",
            "two_theta_disposition": "DEFERRED_NOT_REQUIRED_AFTER_CLASS_MINIMA",
            "domain_ready": domain_ready,
            "twin_record_count": len(records),
            "collision_counts": {
                lesion_class: collisions[lesion_class] for lesion_class in v1.LESION_CLASSES
            },
            "collision_distinct_sources": {
                lesion_class: len(sources[lesion_class]) for lesion_class in v1.LESION_CLASSES
            },
            "search_attempt_counts": {
                lesion_class: attempts[lesion_class] for lesion_class in v1.LESION_CLASSES
            },
            "collision_gate": collision_gate,
            "all_classes_gate": all(collision_gate.values()),
            "records": records,
        },
        artifacts,
        qc,
    )


def _csv_feature_bytes_v2(rows: Sequence[Mapping[str, object]]) -> bytes:
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
        *v1.QUALITY_FEATURES,
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        features = cast(Mapping[str, float], row["features"])
        writer.writerow(
            {
                **{key: row[key] for key in fields[:9]},
                **{key: f"{features[key]:.17g}" for key in v1.QUALITY_FEATURES},
            }
        )
    return stream.getvalue().encode()


def run_s5_v2(
    config: v1.S5Config,
    *,
    archive_path: Path,
    tree_json_path: Path,
    external_run_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Execute the frozen v2 calibration and bounded regional Oracle."""

    external = external_run_root.resolve()
    output = output_dir.resolve()
    if external.exists() or output.exists():
        raise v1.S5CalibrationError("run directories already exist; use new versioned paths")
    external.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    extracted = external / "selected-regular-fundus"
    selected_manifest = v1.audit_and_extract_archive(
        archive_path, tree_json_path, extracted, config
    )
    write_bytes_fail_safe(
        external / "selected-file-manifest.json", v1._canonical(selected_manifest)
    )
    records = v1.load_deepdrid_records(extracted, selected_manifest, config)
    real_rows, quality_qc = compute_real_feature_records_v2(
        records, extracted, selected_manifest, config
    )
    feature_payload = _csv_feature_bytes_v2(real_rows)
    write_bytes_fail_safe(external / "real-quality-features-v2.csv", feature_payload)
    sources = v1.select_synthetic_sources(records, config)
    calibration, ladders, representative = fit_degradation_domain_v2(
        real_rows, sources, extracted, config
    )
    order_records, order_images = audit_noncommutative_orders_v2(
        representative, calibration, config
    )
    order_gate = all(
        record.get("noncommutative") is True and record.get("trace_replay_exact") is True
        for record in order_records
    )
    calibration["ordered_combinations"] = order_records
    calibration["ordered_combination_gate"] = order_gate
    calibration["ood_disposition"] = {
        "id": "REGISTERED_SINGLE_FAMILY_WITHIN_FITTED_CONTIGUOUS_ID_GRID_OR_FORWARD_ORDER",
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
    write_bytes_fail_safe(external / "degradation-calibration-v2.json", v1._canonical(calibration))
    bindings = v1._mapping(config.payload.get("bindings"), "bindings")
    s4_binding = v1._mapping(bindings.get("s4_v5_twin_manifest"), "S4 twin binding")
    s4_manifest_path = PROJECT_ROOT / v1._safe_relative_string(
        s4_binding.get("path"), "S4 twin path"
    )
    oracle, oracle_artifacts, oracle_qc = run_bounded_regional_oracle_v2(
        s4_manifest_path.parent, calibration, config
    )
    calibration_gate = calibration["domain_gate"] is True
    automated_gate = calibration_gate and oracle["all_classes_gate"] is True
    artifacts: dict[str, bytes] = {
        "selected-file-manifest.json": v1._canonical(selected_manifest),
        "degradation-domain-v2.json": v1._canonical(calibration),
        "regional-tier-a-oracle-v2.json": v1._canonical(oracle),
        **oracle_artifacts,
    }
    artifacts["qc-deepdrid-good-poor-contact-sheet.png"] = v1.make_labeled_contact_sheet(
        [(label, image) for label, image in sorted(quality_qc.items())],
        columns=min(8, config.qc_samples),
    )
    for family in ("localized_shadow", "glare_reflection", "haze_veil", "color_cast"):
        artifacts[f"qc-{family.replace('_', '-')}-severity-ladder.png"] = (
            v1.make_labeled_contact_sheet(
                [
                    (f"severity={severity:.1f}", image)
                    for severity, image in zip(config.severities, ladders[family], strict=True)
                ],
                columns=len(config.severities),
            )
        )
    for label, images in order_images.items():
        artifacts[f"qc-order-{label.replace('_', '-')}.png"] = v1.make_labeled_contact_sheet(
            [("forward", images[0]), ("reverse", images[1])], columns=2
        )
    for lesion_class in v1.LESION_CLASSES:
        cells = oracle_qc[lesion_class] or [
            ("NO IN-DOMAIN OBSERVATION", np.zeros((32, 64, 3), dtype=np.uint8))
        ]
        artifacts[f"qc-{lesion_class.lower()}-regional-oracle-contact-sheet.png"] = (
            v1.make_labeled_contact_sheet(cells, columns=4)
        )
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_FOR_AI_QC" if automated_gate else "BLOCKED_AUTOMATED_GATE",
        "config_sha256": config.config_sha256,
        "development_disclosure": "RUN1_AND_S4_DEVELOPMENT_TWINS_OBSERVED",
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
            "synthetic_source_decode_operations": len(sources),
            "quality_qc_decode_operations": len(quality_qc),
            "s4_twin_pair_decodes": 2 * 621 if calibration_gate else 0,
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
            "degradation_domain_sha256": sha256_bytes(artifacts["degradation-domain-v2.json"]),
            "regional_oracle_sha256": sha256_bytes(artifacts["regional-tier-a-oracle-v2.json"]),
        },
        "evidence_boundary": [
            "V2 is a run1-informed development amendment, not an independent replication.",
            "DeepDRiD VALID is reused confirmation and not a fresh blind holdout.",
            "The regional Oracle is independent of RetinaEntail scores and four-valued outputs.",
            "Noncollision records remain ORACLE_UNRESOLVED because the bounded search "
            "is not exhaustive.",
            "Twin, contact-sheet and collision counts are not FCR, medical, clinical or "
            "paper performance results.",
            "No online evaluation, UWF, locked-test, S6, S7, enhancer or paper experiment was run.",
        ],
    }
    artifacts["audit.json"] = v1._canonical(audit)
    v1._write_artifacts(output, artifacts)
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
        config = load_config(args.config)
        audit = run_s5_v2(
            config,
            archive_path=args.archive,
            tree_json_path=args.tree_json,
            external_run_root=args.external_run_root,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, v1.S5CalibrationError, zipfile.BadZipFile) as exc:
        print(f"Science S5 v2 failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "READY_FOR_AI_QC" else 3


if __name__ == "__main__":
    raise SystemExit(main())
