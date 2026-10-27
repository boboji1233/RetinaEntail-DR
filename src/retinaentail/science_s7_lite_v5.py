"""Run-informed S7 v5 target-aligned scoring of immutable v1 witnesses."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
    _write_artifacts,
    regional_context_bbox,
)
from retinaentail.science_s6_verify import HYPOTHESES, LESION_CLASSES, TwinRecord, load_inputs
from retinaentail.science_s7_lite import (
    CALIBRATION_HIERARCHIES,
    DETERMINATE_STATES,
    S7Config,
    ThresholdResolution,
    _apply_multi_region,
    _cluster_bootstrap_fcr,
    _load_s6_contract,
    _p_value,
    _policy_metrics,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest, _v1_probe_hash_audit
from retinaentail.science_s7_lite_v4 import _build_calibration_rows, _state, _threshold
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "5.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V5"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class S7LiteV5Error(ValueError):
    """Raised when v5 target-aligned witness scoring cannot be replayed."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV5Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV5Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV5Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV5Error(f"{name} is not SHA-256")
    return text


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV5Error(f"binding mismatch: {name}")
    return path


def _verify_baseline(expected: object) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != _string(expected, "expected baseline commit"):
        raise S7LiteV5Error("v5 baseline drift")


@dataclass(frozen=True, slots=True)
class V5Config:
    path: Path
    config_sha256: str
    seed: int
    crop_scale: float
    minimum_crop_side: int
    output_side: int
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V5Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v5 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_ALIGNED_WITNESS_AMENDMENT"
        ):
            raise S7LiteV5Error("unsupported S7 v5 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_v2_v3_v4_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        }:
            raise S7LiteV5Error("v5 disclosure changed")
        method = _mapping(root.get("method"), "method")
        if method.get("search_policy") != "REUSE_V1_IMMUTABLE_BEST_WITNESSES":
            raise S7LiteV5Error("v5 search policy changed")
        if method.get("geometry") != "QUERY_AND_ANCHOR_TARGET_BBOX_INDEPENDENT_ALIGNMENT":
            raise S7LiteV5Error("v5 geometry policy changed")
        if method.get("feature") != "RGB_GREEN_GRADIENT_LOCAL_RESIDUAL_V1":
            raise S7LiteV5Error("v5 feature changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        if calibration.get("source_roles") != "REUSE_V4_EXACT_MAP":
            raise S7LiteV5Error("v5 source roles changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required = {
            "v5_prompt",
            "v5_code",
            "v5_audit_schema",
            "v1_config",
            "v1_audit",
            "v1_artifact_manifest",
            "v1_results",
            "v4_config",
            "v4_audit",
            "v4_artifact_manifest",
            "v4_results",
            "v4_source_split",
        }
        if set(bindings) != required:
            raise S7LiteV5Error("v5 binding set changed")
        gate = _mapping(root.get("gate"), "gate")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            seed=_integer(root.get("seed"), "seed"),
            crop_scale=_number(method.get("crop_scale"), "crop scale", 1.0),
            minimum_crop_side=_integer(method.get("minimum_crop_side_px"), "crop side", 1),
            output_side=_integer(method.get("output_side_px"), "output side", 1),
            bootstrap_replicates=_integer(
                _mapping(root.get("statistics"), "statistics").get("bootstrap_replicates"),
                "bootstrap replicates",
                1,
            ),
            maximum_fcr=_number(gate.get("maximum_validation_fcr"), "FCR gate"),
            minimum_moderate_coverage=_number(
                gate.get("minimum_validation_moderate_coverage_per_class"), "coverage gate"
            ),
            minimum_verify_agreement=_number(
                gate.get("minimum_validation_lite_verify_agreement"), "agreement gate"
            ),
            bindings=bindings,
        )


def target_bbox_in_observation(
    record: TwinRecord, *, dx: int = 0, dy: int = 0
) -> tuple[float, float, float, float]:
    """Map a target bbox from original coordinates into the S6 32x32 observation."""

    source = _mapping(record.raw.get("source"), "source")
    geometry = _mapping(source.get("patch_geometry"), "patch geometry")
    crop = _sequence(geometry.get("actual_crop_bbox_xyxy"), "crop bbox")
    target = _mapping(record.raw.get("target"), "target")
    bbox = _sequence(target.get("bbox_xyxy"), "target bbox")
    if len(crop) != 4 or len(bbox) != 4:
        raise S7LiteV5Error("target/crop bbox is invalid")
    crop_values = [_integer(value, "crop coordinate") for value in crop]
    target_values = [_integer(value, "target coordinate") for value in bbox]
    context = regional_context_bbox(record.raw, margin=8, patch_wh=(256, 256))
    target_patch = (
        target_values[0] - crop_values[0],
        target_values[1] - crop_values[1],
        target_values[2] - crop_values[0],
        target_values[3] - crop_values[1],
    )
    scale_x = 32.0 / (context[2] - context[0])
    scale_y = 32.0 / (context[3] - context[1])
    mapped = (
        (target_patch[0] - context[0]) * scale_x + dx,
        (target_patch[1] - context[1]) * scale_y + dy,
        (target_patch[2] - context[0]) * scale_x + dx,
        (target_patch[3] - context[1]) * scale_y + dy,
    )
    if not all(math.isfinite(value) for value in mapped):
        raise S7LiteV5Error("mapped target bbox is non-finite")
    return mapped


def aligned_crop_feature(
    image: NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    *,
    crop_scale: float,
    minimum_side: int,
    output_side: int,
) -> NDArray[np.float64]:
    """Extract the frozen square target-aligned RGB/gradient/residual feature."""

    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
        raise S7LiteV5Error("aligned crop requires 32x32 RGB uint8")
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    side = max((bbox[2] - bbox[0]) * crop_scale, (bbox[3] - bbox[1]) * crop_scale, minimum_side)
    left = max(0, int(math.floor(center_x - side / 2.0)))
    top = max(0, int(math.floor(center_y - side / 2.0)))
    right = min(32, int(math.ceil(center_x + side / 2.0)))
    bottom = min(32, int(math.ceil(center_y + side / 2.0)))
    if right - left < 2 or bottom - top < 2:
        raise S7LiteV5Error("aligned target crop is empty")
    resized = np.asarray(
        Image.fromarray(image[top:bottom, left:right]).resize(
            (output_side, output_side), resample=Image.Resampling.BOX
        ),
        dtype=np.uint8,
    ).copy()
    rgb = resized.astype(np.float64) / 255.0
    green = rgb[:, :, 1]
    padded = np.pad(green, 1, mode="edge")
    gradient = np.hypot(
        padded[1:-1, 2:] - padded[1:-1, :-2],
        padded[2:, 1:-1] - padded[:-2, 1:-1],
    ) / math.sqrt(2.0)
    local = np.zeros_like(green)
    for dy_shift in range(3):
        for dx_shift in range(3):
            local += padded[dy_shift : dy_shift + output_side, dx_shift : dx_shift + output_side]
    residual = np.abs(green - local / 9.0)
    feature = np.concatenate([rgb.reshape(-1), gradient.reshape(-1), residual.reshape(-1)])
    expected = output_side * output_side * 5
    if feature.shape != (expected,) or not np.isfinite(feature).all():
        raise S7LiteV5Error("aligned crop feature is invalid")
    return feature


def _decode_bound_png(path: Path, expected_sha256: object) -> NDArray[np.uint8]:
    if not path.is_file() or sha256_file(path) != _sha256(expected_sha256, "PNG SHA-256"):
        raise S7LiteV5Error("bound v1 PNG changed")
    with Image.open(path) as source:
        source.load()
        image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if image.shape != (32, 32, 3):
        raise S7LiteV5Error("bound v1 PNG shape changed")
    return image


def run_v5(config: V5Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV5Error("S7 v5 output exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    v1_config = S7Config.load(config.bindings["v1_config"])
    s6_config = _load_s6_contract(v1_config)
    records, _collisions, _domain = load_inputs(s6_config)
    record_lookup = {record.twin_id: record for record in records}
    split_decoded: object = json.loads(
        config.bindings["v4_source_split"].read_text(encoding="utf-8")
    )
    roles_raw = _mapping(_mapping(split_decoded, "v4 source split").get("roles"), "roles")
    roles = {key: _string(value, "source role") for key, value in roles_raw.items()}
    v4_decoded: object = json.loads(config.bindings["v4_results"].read_text(encoding="utf-8"))
    v4_rows = [
        _mapping(raw, "v4 row")
        for raw in _sequence(_mapping(v4_decoded, "v4 results").get("records"), "records")
    ]
    v1_root = config.bindings["v1_audit"].parent
    result_rows: list[dict[str, object]] = []
    png_decodes = 0
    candidate_hash_checks = 0
    for raw in v4_rows:
        row = dict(raw)
        row["v4_record_sha256"] = sha256_bytes(_canonical(raw))
        if raw.get("expected_truth") == "OOD":
            for hierarchy in CALIBRATION_HIERARCHIES:
                row[f"state_{hierarchy.lower()}"] = EntailmentState.OUT_OF_MODEL.value
            row["state_region_primary"] = EntailmentState.OUT_OF_MODEL.value
            row["state_source_max"] = EntailmentState.OUT_OF_MODEL.value
            result_rows.append(row)
            continue
        query_record = record_lookup[_string(raw.get("twin_id"), "query twin ID")]
        observation_relative = _safe_relative_string(
            raw.get("observation_png_path"), "observation path"
        )
        observation = _decode_bound_png(
            v1_root.joinpath(*PurePosixPath(observation_relative).parts),
            raw.get("observation_png_sha256"),
        )
        png_decodes += 1
        query_feature = aligned_crop_feature(
            observation,
            target_bbox_in_observation(query_record),
            crop_scale=config.crop_scale,
            minimum_side=config.minimum_crop_side,
            output_side=config.output_side,
        )
        scores: dict[str, float] = {}
        for hypothesis in HYPOTHESES:
            search = _mapping(raw.get(f"{hypothesis.lower()}_search"), "v1 search")
            anchor_raw = _mapping(search.get("anchor"), "anchor")
            anchor = record_lookup[_string(anchor_raw.get("twin_id"), "anchor twin ID")]
            geometry = _mapping(search.get("geometry"), "search geometry")
            dx = _integer(geometry.get("dx"), "dx", -1)
            dy = _integer(geometry.get("dy"), "dy", -1)
            candidate_relative = _safe_relative_string(
                search.get("candidate_png_path"), "candidate path"
            )
            candidate = _decode_bound_png(
                v1_root.joinpath(*PurePosixPath(candidate_relative).parts),
                search.get("candidate_png_sha256"),
            )
            png_decodes += 1
            candidate_hash_checks += 1
            candidate_feature = aligned_crop_feature(
                candidate,
                target_bbox_in_observation(anchor, dx=dx, dy=dy),
                crop_scale=config.crop_scale,
                minimum_side=config.minimum_crop_side,
                output_side=config.output_side,
            )
            score = float(np.mean(np.abs(query_feature - candidate_feature)))
            if not math.isfinite(score):
                raise S7LiteV5Error("aligned witness score is non-finite")
            scores[hypothesis] = score
        row["positive_score"] = scores["PRESENT"]
        row["negative_score"] = scores["ABSENT"]
        row["aligned_feature_dimensions"] = config.output_side * config.output_side * 5
        result_rows.append(row)
    calibration_rows = _build_calibration_rows(result_rows, roles)
    for row in result_rows:
        if row.get("expected_truth") == "OOD":
            continue
        lesion_class = _string(row.get("lesion_class"), "lesion class")
        degradation = _string(row.get("observation_condition"), "condition")
        quality = _string(row.get("quality_bin"), "quality")
        present_score = cast(float, row["positive_score"])
        absent_score = cast(float, row["negative_score"])
        resolutions: dict[str, dict[str, ThresholdResolution]] = {
            hypothesis: {} for hypothesis in HYPOTHESES
        }
        for hypothesis in HYPOTHESES:
            for hierarchy in CALIBRATION_HIERARCHIES:
                resolutions[hypothesis][hierarchy] = _threshold(
                    calibration_rows,
                    v1_config,
                    hierarchy,
                    lesion_class=lesion_class,
                    hypothesis=hypothesis,
                    degradation=degradation,
                    quality=quality,
                    source_max=False,
                )
        for hierarchy in CALIBRATION_HIERARCHIES:
            present = resolutions["PRESENT"][hierarchy]
            absent = resolutions["ABSENT"][hierarchy]
            row[f"state_{hierarchy.lower()}"] = _state(
                present_score, absent_score, present.threshold, absent.threshold
            )
        primary_present = resolutions["PRESENT"]["LESION_DEGRADATION_QUALITY"]
        primary_absent = resolutions["ABSENT"]["LESION_DEGRADATION_QUALITY"]
        group_present = _threshold(
            calibration_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="PRESENT",
            degradation=degradation,
            quality=quality,
            source_max=True,
        )
        group_absent = _threshold(
            calibration_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="ABSENT",
            degradation=degradation,
            quality=quality,
            source_max=True,
        )
        row.update(
            {
                "positive_threshold": primary_present.threshold,
                "negative_threshold": primary_absent.threshold,
                "positive_threshold_hierarchy": primary_present.used_hierarchy,
                "negative_threshold_hierarchy": primary_absent.used_hierarchy,
                "positive_p_value": _p_value(present_score, primary_present.values),
                "negative_p_value": _p_value(absent_score, primary_absent.values),
                "state_region_primary": row["state_lesion_degradation_quality"],
                "state_source_max": _state(
                    present_score,
                    absent_score,
                    group_present.threshold,
                    group_absent.threshold,
                ),
            }
        )
    _apply_multi_region(result_rows, v1_config)
    validation_rows = [
        row
        for row in result_rows
        if roles[_string(row.get("source_key"), "source key")] == "DEVELOPMENT_VALIDATION"
        and row.get("expected_truth") != "OOD"
    ]
    verify_decoded: object = json.loads(
        v1_config.bindings["s6_v2_results"].read_text(encoding="utf-8")
    )
    verify_states = {
        _string(_mapping(raw, "S6 row").get("case_id"), "case ID"): _string(
            _mapping(raw, "S6 row").get("state"), "state"
        )
        for raw in _sequence(_mapping(verify_decoded, "S6 results").get("records"), "S6 rows")
    }
    shared = [
        row
        for row in validation_rows
        if row.get("case_id") in verify_states
        and row.get("stratum") in {"PRESENT_ID", "ABSENT_ID", "S5_TIER_A_AMBIGUOUS_ID"}
    ]
    agreements = sum(
        row["state_region_primary"] == verify_states[cast(str, row["case_id"])] for row in shared
    )
    agreement_rate = agreements / len(shared) if shared else 0.0
    moderate_coverage: dict[str, float] = {}
    for lesion_class in LESION_CLASSES:
        moderate = [
            row
            for row in validation_rows
            if row.get("lesion_class") == lesion_class
            and row.get("stratum") in {"PRESENT_GLARE_MODERATE_ID", "ABSENT_GLARE_MODERATE_ID"}
        ]
        moderate_coverage[lesion_class] = (
            sum(row["state_region_primary"] in DETERMINATE_STATES for row in moderate)
            / len(moderate)
            if moderate
            else 0.0
        )
    state_keys = {
        "UNADJUSTED_REGION": "state_region_primary",
        "BONFERRONI": "state_bonferroni",
        "HOLM": "state_holm",
        "SOURCE_MAX": "state_source_max",
    }
    policy_metrics = {
        policy: _policy_metrics(validation_rows, state_key)
        for policy, state_key in state_keys.items()
    }
    hierarchy_metrics = {
        hierarchy: _policy_metrics(validation_rows, f"state_{hierarchy.lower()}")
        for hierarchy in CALIBRATION_HIERARCHIES
    }
    fcr_raw = policy_metrics["UNADJUSTED_REGION"]["development_fcr"]
    validation_fcr = float(fcr_raw) if isinstance(fcr_raw, int | float) else math.inf
    ood = [row for row in result_rows if row.get("expected_truth") == "OOD"]
    ood_pass = sum(row["state_region_primary"] == EntailmentState.OUT_OF_MODEL.value for row in ood)
    manifests = {
        "v1": _audit_manifest(v1_root, config.bindings["v1_artifact_manifest"]),
        "v4": _audit_manifest(
            config.bindings["v4_audit"].parent, config.bindings["v4_artifact_manifest"]
        ),
    }
    v1_hashes = _v1_probe_hash_audit(v4_rows, v1_root)
    upstream_pass = (
        all(
            audit["mismatches"] == 0 and audit["unlisted"] == 0 and audit["missing"] == 0
            for audit in manifests.values()
        )
        and v1_hashes["mismatches"] == 0
    )
    gate_pass = (
        upstream_pass
        and png_decodes == 1200
        and candidate_hash_checks == 800
        and len(result_rows) == 560
        and ood_pass == len(ood) == 160
        and validation_fcr <= config.maximum_fcr
        and all(value >= config.minimum_moderate_coverage for value in moderate_coverage.values())
        and agreement_rate >= config.minimum_verify_agreement
    )
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_roles_sha256": sha256_file(config.bindings["v4_source_split"]),
        "ambiguous_policy": "CONTRIBUTE_TO_BOTH_LABEL_SETS",
        "rows": [
            {
                "lesion_class": row.lesion_class,
                "hypothesis": row.hypothesis,
                "degradation": row.degradation,
                "quality": row.quality,
                "source_key": row.source_key,
                "twin_id": row.twin_id,
                "score": row.score,
            }
            for row in calibration_rows
        ],
    }
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "probe_searches_rerun": 0,
        "records": result_rows,
    }
    artifacts: dict[str, bytes] = {
        "aligned-witness-calibration-v5.json": _canonical(calibration_payload),
        "lite-results-v5.json": _canonical(results_payload),
    }
    for lesion_class in LESION_CLASSES:
        source = v1_root / f"qc-{lesion_class.lower()}-lite-contact-sheet.png"
        artifacts[f"qc-{lesion_class.lower()}-lite-v5-contact-sheet.png"] = source.read_bytes()
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_v2_v3_v4_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        },
        "method": {
            "search_policy": "REUSE_V1_IMMUTABLE_BEST_WITNESSES",
            "geometry": "QUERY_AND_ANCHOR_TARGET_BBOX_INDEPENDENT_ALIGNMENT",
            "crop_scale": config.crop_scale,
            "minimum_crop_side_px": config.minimum_crop_side,
            "output_side_px": config.output_side,
            "feature": "RGB_GREEN_GRADIENT_LOCAL_RESIDUAL_V1",
            "feature_dimensions": config.output_side * config.output_side * 5,
        },
        "preservation": {
            "manifests": manifests,
            "v1_probe_hashes": v1_hashes,
            "candidate_hash_checks": candidate_hash_checks,
            "probe_searches_rerun": 0,
        },
        "calibration": {
            "source_roles": "REUSE_V4_EXACT_MAP",
            "calibration_rows": len(calibration_rows),
            "validation_hierarchy_metrics": hierarchy_metrics,
        },
        "multi_region": {"validation_policy_metrics": policy_metrics},
        "development_gate_metrics": {
            "validation_fcr": None if not math.isfinite(validation_fcr) else validation_fcr,
            "validation_cluster_bootstrap": _cluster_bootstrap_fcr(
                validation_rows, "state_region_primary", config.seed, config.bootstrap_replicates
            ),
            "validation_moderate_glare_determinate_coverage_per_class": moderate_coverage,
            "validation_lite_verify_shared_in_domain_records": len(shared),
            "validation_lite_verify_agreement_count": agreements,
            "validation_lite_verify_agreement_rate": agreement_rate,
            "all_ood_records": len(ood),
            "all_ood_out_of_model": ood_pass,
        },
        "io_audit": {
            "bound_png_decodes": png_decodes,
            "probe_searches_rerun": 0,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "source_split_leaks": 0,
        },
        "gate": {
            "maximum_validation_fcr": config.maximum_fcr,
            "minimum_validation_moderate_coverage_per_class": config.minimum_moderate_coverage,
            "minimum_validation_lite_verify_agreement": config.minimum_verify_agreement,
            "automated": "PASS" if gate_pass else "FAIL",
            "ai_visual_qc": "PENDING",
            "s8_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "calibration_sha256": sha256_bytes(artifacts["aligned-witness-calibration-v5.json"]),
            "results_sha256": sha256_bytes(artifacts["lite-results-v5.json"]),
        },
        "evidence_boundary": [
            "V5 is run-informed development work and not an independent replication.",
            "Only immutable v1 best witnesses are rescored; no search is rerun.",
            "Target geometry is Oracle-region metadata, not a proposal-region result.",
            "No final test, clinical inference, enhancer training, or paper experiment ran.",
        ],
    }
    artifacts["audit.json"] = _canonical(audit)
    _write_artifacts(output, artifacts)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = V5Config.load(args.config)
        audit = run_v5(config, output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"S7 Lite v5 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
