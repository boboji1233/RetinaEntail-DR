"""Run-informed Science S7 Lite v3 lesion-core prototype amendment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState, fail_closed_state
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
    _write_artifacts,
)
from retinaentail.science_s6_verify import (
    HYPOTHESES,
    LESION_CLASSES,
    IOAudit,
    load_inputs,
    select_bank_and_cases,
)
from retinaentail.science_s7_lite import (
    CALIBRATION_HIERARCHIES,
    DETERMINATE_STATES,
    CalibrationRow,
    S7Config,
    ThresholdResolution,
    _apply_multi_region,
    _cluster_bootstrap_fcr,
    _load_s6_contract,
    _p_value,
    _policy_metrics,
    _quality_bin,
    build_lite_bank,
    resolve_threshold,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest, _v1_probe_hash_audit
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "3.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V3"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
CONDITIONS: Final = ("IDENTITY", "GLARE_MODERATE", "GLARE_SEVERE")


class S7LiteV3Error(ValueError):
    """Raised when the frozen v3 amendment cannot execute safely."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV3Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV3Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV3Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV3Error(f"{name} is not SHA-256")
    return text


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV3Error(f"binding mismatch: {name}")
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
        raise S7LiteV3Error("v3 baseline drift")


@dataclass(frozen=True, slots=True)
class V3Config:
    path: Path
    config_sha256: str
    seed: int
    alpha: float
    variance_regularizer: float
    minimum_region_units: int
    minimum_group_units: int
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V3Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v3 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_LESION_CORE_AMENDMENT"
        ):
            raise S7LiteV3Error("unsupported S7 v3 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_observed": True,
            "v2_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        }:
            raise S7LiteV3Error("v3 disclosure changed")
        method = _mapping(root.get("method"), "method")
        if method.get("encoder") != "FROZEN_LESION_CORE_MULTISCALE_NUMPY_V2":
            raise S7LiteV3Error("v3 encoder changed")
        if method.get("head") != "PAIRED_DIAGONAL_LDA_PROTOTYPE":
            raise S7LiteV3Error("v3 prototype head changed")
        if method.get("calibration_fit") != "LEAVE_ONE_SOURCE_OUT":
            raise S7LiteV3Error("v3 calibration fit changed")
        if method.get("probe_search_policy") != "REUSE_V1_WITHOUT_RERUN":
            raise S7LiteV3Error("v3 probe policy changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        if calibration.get("hierarchies") != list(CALIBRATION_HIERARCHIES):
            raise S7LiteV3Error("v3 calibration hierarchy changed")
        if calibration.get("rank_rule") != "CEIL_N_PLUS_1_TIMES_ONE_MINUS_ALPHA":
            raise S7LiteV3Error("v3 rank rule changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required = {
            "v3_prompt",
            "v3_code",
            "v3_audit_schema",
            "v1_config",
            "v1_audit",
            "v1_artifact_manifest",
            "v1_results",
            "v2_config",
            "v2_audit",
            "v2_artifact_manifest",
            "v2_results",
        }
        if set(bindings) != required:
            raise S7LiteV3Error("v3 binding set changed")
        gate = _mapping(root.get("gate"), "gate")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            seed=_integer(root.get("seed"), "seed"),
            alpha=_number(calibration.get("alpha"), "alpha"),
            variance_regularizer=_number(
                method.get("variance_regularizer"), "variance regularizer"
            ),
            minimum_region_units=_integer(
                calibration.get("minimum_region_units"), "minimum region units", 1
            ),
            minimum_group_units=_integer(
                calibration.get("minimum_group_units"), "minimum group units", 1
            ),
            bootstrap_replicates=_integer(
                _mapping(root.get("statistics"), "statistics").get("bootstrap_replicates"),
                "bootstrap replicates",
                1,
            ),
            maximum_fcr=_number(gate.get("maximum_development_fcr"), "FCR gate"),
            minimum_moderate_coverage=_number(
                gate.get("minimum_moderate_coverage_per_class"), "coverage gate"
            ),
            minimum_verify_agreement=_number(
                gate.get("minimum_lite_verify_in_domain_agreement"), "agreement gate"
            ),
            bindings=bindings,
        )


def lesion_core_encoder(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Encode the registered lesion core without trainable spatial filters."""

    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
        raise S7LiteV3Error("v3 encoder requires 32x32 RGB uint8")
    rgb = image.astype(np.float64) / 255.0
    core = rgb[4:28, 4:28]
    rgb_blocks = core.reshape(8, 3, 8, 3, 3).mean(axis=(1, 3)).reshape(-1)
    green = rgb[:, :, 1]
    padded = np.pad(green, 1, mode="edge")
    gradient = np.hypot(
        padded[1:-1, 2:] - padded[1:-1, :-2],
        padded[2:, 1:-1] - padded[:-2, 1:-1],
    ) / math.sqrt(2.0)
    gradient_blocks = gradient[4:28, 4:28].reshape(8, 3, 8, 3).mean(axis=(1, 3)).reshape(-1)
    local_sum = np.zeros_like(green)
    padded_green = np.pad(green, 2, mode="reflect")
    for dy in range(5):
        for dx in range(5):
            local_sum += padded_green[dy : dy + 32, dx : dx + 32]
    residual = np.abs(green - local_sum / 25.0)
    residual_blocks = residual[4:28, 4:28].reshape(8, 3, 8, 3).mean(axis=(1, 3)).reshape(-1)
    center_mean = rgb[8:24, 8:24].mean(axis=(0, 1))
    context_mask = np.ones((32, 32), dtype=bool)
    context_mask[8:24, 8:24] = False
    context_mean = rgb[context_mask].mean(axis=0)
    feature = np.concatenate(
        [rgb_blocks, gradient_blocks, residual_blocks, center_mean - context_mean]
    )
    if feature.shape != (323,) or not np.isfinite(feature).all():
        raise S7LiteV3Error("v3 encoder emitted invalid features")
    return feature


@dataclass(frozen=True, slots=True)
class PrototypeHead:
    midpoint: NDArray[np.float64]
    weight: NDArray[np.float64]
    scale: float

    def logit(self, feature: NDArray[np.float64]) -> float:
        value = float(np.dot(feature - self.midpoint, self.weight) / self.scale)
        if not math.isfinite(value):
            raise S7LiteV3Error("prototype head emitted a non-finite logit")
        return value


def fit_prototype_head(
    present: Sequence[NDArray[np.float64]],
    absent: Sequence[NDArray[np.float64]],
    *,
    variance_regularizer: float,
) -> PrototypeHead:
    if len(present) < 2 or len(absent) < 2 or len(present) != len(absent):
        raise S7LiteV3Error("prototype head requires paired samples")
    present_array = np.stack(present)
    absent_array = np.stack(absent)
    present_mean = present_array.mean(axis=0)
    absent_mean = absent_array.mean(axis=0)
    pooled = np.concatenate([present_array, absent_array], axis=0)
    variance = pooled.var(axis=0)
    weight = (present_mean - absent_mean) / (variance + variance_regularizer)
    midpoint = (present_mean + absent_mean) / 2.0
    raw = np.abs((pooled - midpoint) @ weight)
    scale = max(float(np.median(raw)), 1e-8)
    return PrototypeHead(midpoint=midpoint, weight=weight, scale=scale)


def _decode_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as source:
        source.load()
        image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if image.shape != (32, 32, 3):
        raise S7LiteV3Error("v1 probe observation shape changed")
    return image


def _threshold(
    rows: Sequence[CalibrationRow],
    v1_config: S7Config,
    hierarchy: str,
    *,
    lesion_class: str,
    hypothesis: str,
    degradation: str,
    quality: str,
    source_max: bool,
) -> ThresholdResolution:
    return resolve_threshold(
        rows,
        v1_config,
        hierarchy,
        lesion_class=lesion_class,
        hypothesis=hypothesis,
        degradation=degradation,
        quality=quality,
        source_max=source_max,
    )


def _state(
    present_nonconformity: float,
    absent_nonconformity: float,
    present_threshold: float,
    absent_threshold: float,
) -> str:
    return fail_closed_state(
        positive_feasible=present_nonconformity <= present_threshold,
        negative_feasible=absent_nonconformity <= absent_threshold,
        positive_score=present_nonconformity,
        negative_score=absent_nonconformity,
        positive_threshold=present_threshold,
        negative_threshold=absent_threshold,
        positive_solver_status="optimal",
        negative_solver_status="optimal",
        solver_status="optimal",
        degradation_known=True,
    ).value


def run_v3(config: V3Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV3Error("S7 v3 output exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    tracemalloc.start()
    v1_config = S7Config.load(config.bindings["v1_config"])
    if v1_config.alpha != config.alpha:
        raise S7LiteV3Error("v3 alpha differs from v1")
    s6_config = _load_s6_contract(v1_config)
    io_audit = IOAudit()
    records, collisions, _domain = load_inputs(s6_config)
    selected_banks, selected_cases = select_bank_and_cases(records, collisions, s6_config)
    banks = build_lite_bank(selected_banks, s6_config, v1_config, io_audit)
    if any(len(selected_cases[lesion_class]) != 100 for lesion_class in LESION_CLASSES):
        raise S7LiteV3Error("v1 probe selection replay changed")
    features: dict[tuple[str, str, str], NDArray[np.float64]] = {}
    for lesion_class in LESION_CLASSES:
        for entry in banks[lesion_class]:
            for hypothesis in HYPOTHESES:
                for condition in CONDITIONS:
                    features[(entry.record.twin_id, hypothesis, condition)] = lesion_core_encoder(
                        entry.variants[hypothesis][condition].array
                    )
    calibration_rows: list[CalibrationRow] = []
    calibration_details: list[dict[str, object]] = []
    full_heads: dict[tuple[str, str], PrototypeHead] = {}
    for lesion_class in LESION_CLASSES:
        entries = banks[lesion_class]
        sources = sorted({entry.record.source_key for entry in entries})
        for condition in CONDITIONS:
            full_heads[(lesion_class, condition)] = fit_prototype_head(
                [features[(entry.record.twin_id, "PRESENT", condition)] for entry in entries],
                [features[(entry.record.twin_id, "ABSENT", condition)] for entry in entries],
                variance_regularizer=config.variance_regularizer,
            )
            for source_key in sources:
                train = [entry for entry in entries if entry.record.source_key != source_key]
                held = [entry for entry in entries if entry.record.source_key == source_key]
                head = fit_prototype_head(
                    [features[(entry.record.twin_id, "PRESENT", condition)] for entry in train],
                    [features[(entry.record.twin_id, "ABSENT", condition)] for entry in train],
                    variance_regularizer=config.variance_regularizer,
                )
                for entry in held:
                    for truth in HYPOTHESES:
                        feature = features[(entry.record.twin_id, truth, condition)]
                        logit = head.logit(feature)
                        nonconformity = -logit if truth == "PRESENT" else logit
                        quality = _quality_bin(entry.variants[truth][condition].array)
                        calibration_rows.append(
                            CalibrationRow(
                                lesion_class=lesion_class,
                                hypothesis=truth,
                                degradation=condition,
                                quality=quality,
                                source_key=source_key,
                                twin_id=entry.record.twin_id,
                                score=nonconformity,
                            )
                        )
                        calibration_details.append(
                            {
                                "lesion_class": lesion_class,
                                "source_truth": truth,
                                "condition": condition,
                                "quality": quality,
                                "source_key": source_key,
                                "twin_id": entry.record.twin_id,
                                "logit": logit,
                                "nonconformity": nonconformity,
                                "fit_excluded_source": True,
                            }
                        )
    if len(calibration_rows) != 768:
        raise S7LiteV3Error("v3 calibration does not contain 768 out-of-source rows")
    v2_decoded: object = json.loads(config.bindings["v2_results"].read_text(encoding="utf-8"))
    v2_rows = [
        _mapping(raw, "v2 row")
        for raw in _sequence(_mapping(v2_decoded, "v2 results").get("records"), "records")
    ]
    v1_root = config.bindings["v1_audit"].parent
    result_rows: list[dict[str, object]] = []
    probe_decodes = 0
    for raw in v2_rows:
        row = dict(raw)
        row["v2_record_sha256"] = sha256_bytes(_canonical(raw))
        if raw.get("expected_truth") == "OOD":
            for hierarchy in CALIBRATION_HIERARCHIES:
                row[f"state_{hierarchy.lower()}"] = EntailmentState.OUT_OF_MODEL.value
            row["state_region_primary"] = EntailmentState.OUT_OF_MODEL.value
            row["state_source_max"] = EntailmentState.OUT_OF_MODEL.value
            result_rows.append(row)
            continue
        relative = _safe_relative_string(raw.get("observation_png_path"), "observation path")
        observation = _decode_rgb(v1_root.joinpath(*PurePosixPath(relative).parts))
        probe_decodes += 1
        lesion_class = _string(raw.get("lesion_class"), "lesion class")
        condition = _string(raw.get("observation_condition"), "condition")
        quality = _string(raw.get("quality_bin"), "quality")
        logit = full_heads[(lesion_class, condition)].logit(lesion_core_encoder(observation))
        present_nonconformity = -logit
        absent_nonconformity = logit
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
                    degradation=condition,
                    quality=quality,
                    source_max=False,
                )
        for hierarchy in CALIBRATION_HIERARCHIES:
            present = resolutions["PRESENT"][hierarchy]
            absent = resolutions["ABSENT"][hierarchy]
            row[f"state_{hierarchy.lower()}"] = _state(
                present_nonconformity,
                absent_nonconformity,
                present.threshold,
                absent.threshold,
            )
        primary_present = resolutions["PRESENT"]["LESION_DEGRADATION_QUALITY"]
        primary_absent = resolutions["ABSENT"]["LESION_DEGRADATION_QUALITY"]
        group_present = _threshold(
            calibration_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="PRESENT",
            degradation=condition,
            quality=quality,
            source_max=True,
        )
        group_absent = _threshold(
            calibration_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="ABSENT",
            degradation=condition,
            quality=quality,
            source_max=True,
        )
        row.update(
            {
                "prototype_logit": logit,
                "positive_score": present_nonconformity,
                "negative_score": absent_nonconformity,
                "positive_threshold": primary_present.threshold,
                "negative_threshold": primary_absent.threshold,
                "positive_threshold_hierarchy": primary_present.used_hierarchy,
                "negative_threshold_hierarchy": primary_absent.used_hierarchy,
                "positive_p_value": _p_value(present_nonconformity, primary_present.values),
                "negative_p_value": _p_value(absent_nonconformity, primary_absent.values),
                "state_region_primary": row["state_lesion_degradation_quality"],
                "state_source_max": _state(
                    present_nonconformity,
                    absent_nonconformity,
                    group_present.threshold,
                    group_absent.threshold,
                ),
            }
        )
        result_rows.append(row)
    _apply_multi_region(result_rows, v1_config)
    v1_manifest = _audit_manifest(v1_root, config.bindings["v1_artifact_manifest"])
    v1_hashes = _v1_probe_hash_audit(v2_rows, v1_root)
    v2_root = config.bindings["v2_audit"].parent
    v2_manifest = _audit_manifest(v2_root, config.bindings["v2_artifact_manifest"])
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
        for row in result_rows
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
            for row in result_rows
            if row.get("lesion_class") == lesion_class
            and row.get("stratum") in {"PRESENT_GLARE_MODERATE_ID", "ABSENT_GLARE_MODERATE_ID"}
        ]
        moderate_coverage[lesion_class] = sum(
            row["state_region_primary"] in DETERMINATE_STATES for row in moderate
        ) / len(moderate)
    state_keys = {
        "UNADJUSTED_REGION": "state_region_primary",
        "BONFERRONI": "state_bonferroni",
        "HOLM": "state_holm",
        "SOURCE_MAX": "state_source_max",
    }
    policy_metrics = {
        policy: _policy_metrics(result_rows, state_key) for policy, state_key in state_keys.items()
    }
    hierarchy_metrics = {
        hierarchy: _policy_metrics(result_rows, f"state_{hierarchy.lower()}")
        for hierarchy in CALIBRATION_HIERARCHIES
    }
    primary_raw = policy_metrics["UNADJUSTED_REGION"]["development_fcr"]
    primary_fcr = float(primary_raw) if isinstance(primary_raw, int | float) else math.inf
    ood = [row for row in result_rows if row.get("expected_truth") == "OOD"]
    ood_pass = sum(row["state_region_primary"] == EntailmentState.OUT_OF_MODEL.value for row in ood)
    upstream_pass = (
        all(
            audit["mismatches"] == 0 and audit["unlisted"] == 0 and audit["missing"] == 0
            for audit in (v1_manifest, v2_manifest)
        )
        and v1_hashes["mismatches"] == 0
    )
    gate_pass = (
        upstream_pass
        and len(result_rows) == 560
        and probe_decodes == 400
        and ood_pass == len(ood) == 160
        and primary_fcr <= config.maximum_fcr
        and all(value >= config.minimum_moderate_coverage for value in moderate_coverage.values())
        and agreement_rate >= config.minimum_verify_agreement
    )
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "method": "SOURCE_EXCLUDED_PAIRED_DIAGONAL_LDA_PROTOTYPE",
        "feature_dimensions": 323,
        "variance_regularizer": config.variance_regularizer,
        "rows": calibration_details,
    }
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "probe_searches_rerun": 0,
        "probe_truth_used_for_training": False,
        "records": result_rows,
    }
    artifacts: dict[str, bytes] = {
        "prototype-calibration-v3.json": _canonical(calibration_payload),
        "lite-results-v3.json": _canonical(results_payload),
    }
    for lesion_class in LESION_CLASSES:
        source = v1_root / f"qc-{lesion_class.lower()}-lite-contact-sheet.png"
        artifacts[f"qc-{lesion_class.lower()}-lite-v3-contact-sheet.png"] = source.read_bytes()
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_observed": True,
            "v2_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        },
        "method": {
            "encoder": "FROZEN_LESION_CORE_MULTISCALE_NUMPY_V2",
            "feature_dimensions": 323,
            "head": "PAIRED_DIAGONAL_LDA_PROTOTYPE",
            "variance_regularizer": config.variance_regularizer,
            "learned_parameter_count": 4 * 3 * (323 * 2 + 1),
            "calibration_fit": "LEAVE_ONE_SOURCE_OUT",
            "calibration_rows": len(calibration_rows),
            "probe_searches_rerun": 0,
            "probe_truth_used_for_training": False,
        },
        "preservation": {
            "v1_manifest": v1_manifest,
            "v2_manifest": v2_manifest,
            "v1_probe_hashes": v1_hashes,
            "v1_v2_v3_probe_records": [len(v2_rows), len(v2_rows), len(result_rows)],
        },
        "calibration": {
            "hierarchies": list(CALIBRATION_HIERARCHIES),
            "hierarchy_metrics": hierarchy_metrics,
        },
        "multi_region": {"policy_metrics": policy_metrics},
        "development_gate_metrics": {
            "development_fcr": None if not math.isfinite(primary_fcr) else primary_fcr,
            "cluster_bootstrap": _cluster_bootstrap_fcr(
                result_rows, "state_region_primary", config.seed, config.bootstrap_replicates
            ),
            "moderate_glare_determinate_coverage_per_class": moderate_coverage,
            "lite_verify_shared_in_domain_records": len(shared),
            "lite_verify_exact_agreement_count": agreements,
            "lite_verify_exact_agreement_rate": agreement_rate,
            "ood_records": len(ood),
            "ood_out_of_model": ood_pass,
        },
        "resource": {
            "device": "CPU",
            "gpu_used": False,
            "peak_python_tracemalloc_bytes": peak_bytes,
            "wall_seconds": time.perf_counter() - start,
            "silent_resolution_change": False,
        },
        "io_audit": {
            "probe_observation_decodes": probe_decodes,
            "probe_searches_rerun": 0,
            "s4_twin_png_decodes_for_bank": io_audit.s4_twin_png_decodes,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "source_split_leaks": 0,
        },
        "gate": {
            "maximum_development_fcr": config.maximum_fcr,
            "minimum_moderate_coverage_per_class": config.minimum_moderate_coverage,
            "minimum_lite_verify_in_domain_agreement": config.minimum_verify_agreement,
            "automated": "PASS" if gate_pass else "FAIL",
            "ai_visual_qc": "PENDING",
            "s8_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "calibration_sha256": sha256_bytes(artifacts["prototype-calibration-v3.json"]),
            "results_sha256": sha256_bytes(artifacts["lite-results-v3.json"]),
        },
        "evidence_boundary": [
            "V3 is run-informed development work and not an independent replication.",
            "Prototype heads use only the source-isolated bank; probe truth is evaluation-only.",
            "V1 searches and witnesses are preserved and no probe search is rerun.",
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
        config = V3Config.load(args.config)
        audit = run_v3(config, output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"S7 Lite v3 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
