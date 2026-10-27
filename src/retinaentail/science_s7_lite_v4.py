"""Run-informed S7 v4 source-split conformal amendment over frozen v3 scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import yaml

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
from retinaentail.science_s6_verify import HYPOTHESES, LESION_CLASSES
from retinaentail.science_s7_lite import (
    CALIBRATION_HIERARCHIES,
    DETERMINATE_STATES,
    CalibrationRow,
    S7Config,
    ThresholdResolution,
    _apply_multi_region,
    _cluster_bootstrap_fcr,
    _p_value,
    _policy_metrics,
    resolve_threshold,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest, _v1_probe_hash_audit
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "4.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V4"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class S7LiteV4Error(ValueError):
    """Raised when the frozen v4 source-split contract fails closed."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV4Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV4Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV4Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV4Error(f"{name} is not SHA-256")
    return text


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV4Error(f"binding mismatch: {name}")
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
        raise S7LiteV4Error("v4 baseline drift")


@dataclass(frozen=True, slots=True)
class V4Config:
    path: Path
    config_sha256: str
    seed: int
    calibration_fraction: float
    alpha: float
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V4Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v4 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_SOURCE_SPLIT_AMENDMENT"
        ):
            raise S7LiteV4Error("unsupported S7 v4 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_v2_v3_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        }:
            raise S7LiteV4Error("v4 disclosure changed")
        split = _mapping(root.get("source_split"), "source split")
        if split.get("policy") != "SHA256_SEED_AND_SOURCE_KEY":
            raise S7LiteV4Error("v4 source split policy changed")
        if split.get("roles") != ["CALIBRATION_FIT", "DEVELOPMENT_VALIDATION"]:
            raise S7LiteV4Error("v4 roles changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        if calibration.get("ambiguous_policy") != "CONTRIBUTE_TO_BOTH_LABEL_SETS":
            raise S7LiteV4Error("v4 ambiguity calibration changed")
        if calibration.get("hierarchies") != list(CALIBRATION_HIERARCHIES):
            raise S7LiteV4Error("v4 hierarchies changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required = {
            "v4_prompt",
            "v4_code",
            "v4_audit_schema",
            "v1_config",
            "v1_audit",
            "v1_artifact_manifest",
            "v1_results",
            "v2_audit",
            "v2_artifact_manifest",
            "v3_config",
            "v3_audit",
            "v3_artifact_manifest",
            "v3_results",
        }
        if set(bindings) != required:
            raise S7LiteV4Error("v4 binding set changed")
        gate = _mapping(root.get("gate"), "gate")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            seed=_integer(root.get("seed"), "seed"),
            calibration_fraction=_number(split.get("calibration_fraction"), "calibration fraction"),
            alpha=_number(calibration.get("alpha"), "alpha"),
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


def source_role(seed: int, source_key: str, calibration_fraction: float) -> str:
    """Return a deterministic role without inspecting any outcome or label."""

    digest = hashlib.sha256(f"{seed}|{source_key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "CALIBRATION_FIT" if value < calibration_fraction else "DEVELOPMENT_VALIDATION"


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
    present_score: float, absent_score: float, present_threshold: float, absent_threshold: float
) -> str:
    return fail_closed_state(
        positive_feasible=present_score <= present_threshold,
        negative_feasible=absent_score <= absent_threshold,
        positive_score=present_score,
        negative_score=absent_score,
        positive_threshold=present_threshold,
        negative_threshold=absent_threshold,
        positive_solver_status="optimal",
        negative_solver_status="optimal",
        solver_status="optimal",
        degradation_known=True,
    ).value


def _build_calibration_rows(
    rows: Sequence[Mapping[str, object]], roles: Mapping[str, str]
) -> list[CalibrationRow]:
    calibration: list[CalibrationRow] = []
    for row in rows:
        source_key = _string(row.get("source_key"), "source key")
        if roles[source_key] != "CALIBRATION_FIT" or row.get("expected_truth") == "OOD":
            continue
        truth = _string(row.get("expected_truth"), "truth")
        labels = HYPOTHESES if truth == "AMBIGUOUS" else (truth,)
        for hypothesis in labels:
            score_name = "positive_score" if hypothesis == "PRESENT" else "negative_score"
            score = row.get(score_name)
            if not isinstance(score, int | float) or isinstance(score, bool):
                raise S7LiteV4Error("v3 calibration score is missing")
            calibration.append(
                CalibrationRow(
                    lesion_class=_string(row.get("lesion_class"), "lesion class"),
                    hypothesis=hypothesis,
                    degradation=_string(row.get("observation_condition"), "condition"),
                    quality=_string(row.get("quality_bin"), "quality"),
                    source_key=source_key,
                    twin_id=_string(row.get("twin_id"), "twin ID"),
                    score=float(score),
                )
            )
    return calibration


def run_v4(config: V4Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV4Error("S7 v4 output exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    v1_config = S7Config.load(config.bindings["v1_config"])
    if v1_config.alpha != config.alpha:
        raise S7LiteV4Error("v4 alpha differs from v1")
    v3_decoded: object = json.loads(config.bindings["v3_results"].read_text(encoding="utf-8"))
    v3_rows = [
        _mapping(raw, "v3 row")
        for raw in _sequence(_mapping(v3_decoded, "v3 results").get("records"), "records")
    ]
    source_keys = sorted({_string(row.get("source_key"), "source key") for row in v3_rows})
    roles = {
        source_key: source_role(config.seed, source_key, config.calibration_fraction)
        for source_key in source_keys
    }
    calibration_rows = _build_calibration_rows(v3_rows, roles)
    if not calibration_rows:
        raise S7LiteV4Error("v4 calibration split is empty")
    result_rows: list[dict[str, object]] = []
    for raw in v3_rows:
        row = dict(raw)
        source_key = _string(raw.get("source_key"), "source key")
        row["development_role"] = roles[source_key]
        row["v3_record_sha256"] = sha256_bytes(_canonical(raw))
        if raw.get("expected_truth") == "OOD":
            for hierarchy in CALIBRATION_HIERARCHIES:
                row[f"state_{hierarchy.lower()}"] = EntailmentState.OUT_OF_MODEL.value
            row["state_region_primary"] = EntailmentState.OUT_OF_MODEL.value
            row["state_source_max"] = EntailmentState.OUT_OF_MODEL.value
            result_rows.append(row)
            continue
        lesion_class = _string(raw.get("lesion_class"), "lesion class")
        degradation = _string(raw.get("observation_condition"), "condition")
        quality = _string(raw.get("quality_bin"), "quality")
        present_score_raw = raw.get("positive_score")
        absent_score_raw = raw.get("negative_score")
        if (
            not isinstance(present_score_raw, int | float)
            or isinstance(present_score_raw, bool)
            or not isinstance(absent_score_raw, int | float)
            or isinstance(absent_score_raw, bool)
        ):
            raise S7LiteV4Error("v3 nonconformity score is missing")
        present_score = float(present_score_raw)
        absent_score = float(absent_score_raw)
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
        result_rows.append(row)
    _apply_multi_region(result_rows, v1_config)
    validation_rows = [
        row
        for row in result_rows
        if row["development_role"] == "DEVELOPMENT_VALIDATION" and row["expected_truth"] != "OOD"
    ]
    calibration_sources = {key for key, role in roles.items() if role == "CALIBRATION_FIT"}
    validation_sources = {key for key, role in roles.items() if role == "DEVELOPMENT_VALIDATION"}
    source_overlap = len(calibration_sources.intersection(validation_sources))
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
    v1_root = config.bindings["v1_audit"].parent
    manifests = {
        "v1": _audit_manifest(v1_root, config.bindings["v1_artifact_manifest"]),
        "v2": _audit_manifest(
            config.bindings["v2_audit"].parent, config.bindings["v2_artifact_manifest"]
        ),
        "v3": _audit_manifest(
            config.bindings["v3_audit"].parent, config.bindings["v3_artifact_manifest"]
        ),
    }
    v1_hashes = _v1_probe_hash_audit(v3_rows, v1_root)
    upstream_pass = (
        all(
            audit["mismatches"] == 0 and audit["unlisted"] == 0 and audit["missing"] == 0
            for audit in manifests.values()
        )
        and v1_hashes["mismatches"] == 0
    )
    gate_pass = (
        upstream_pass
        and source_overlap == 0
        and len(result_rows) == 560
        and ood_pass == len(ood) == 160
        and validation_fcr <= config.maximum_fcr
        and all(value >= config.minimum_moderate_coverage for value in moderate_coverage.values())
        and agreement_rate >= config.minimum_verify_agreement
    )
    split_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "policy": "SHA256_SEED_AND_SOURCE_KEY",
        "seed": config.seed,
        "calibration_fraction": config.calibration_fraction,
        "roles": roles,
    }
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
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
        "probe_scores_recomputed": 0,
        "records": result_rows,
    }
    artifacts: dict[str, bytes] = {
        "source-split-v4.json": _canonical(split_payload),
        "split-conformal-calibration-v4.json": _canonical(calibration_payload),
        "lite-results-v4.json": _canonical(results_payload),
    }
    for lesion_class in LESION_CLASSES:
        source = v1_root / f"qc-{lesion_class.lower()}-lite-contact-sheet.png"
        artifacts[f"qc-{lesion_class.lower()}-lite-v4-contact-sheet.png"] = source.read_bytes()
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_v2_v3_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        },
        "source_split": {
            "policy": "SHA256_SEED_AND_SOURCE_KEY",
            "source_groups": len(source_keys),
            "calibration_sources": len(calibration_sources),
            "validation_sources": len(validation_sources),
            "source_role_overlap": source_overlap,
            "calibration_rows": len(calibration_rows),
            "validation_in_domain_rows": len(validation_rows),
        },
        "preservation": {
            "manifests": manifests,
            "v1_probe_hashes": v1_hashes,
            "probe_scores_recomputed": 0,
        },
        "calibration": {
            "hierarchies": list(CALIBRATION_HIERARCHIES),
            "ambiguous_policy": "CONTRIBUTE_TO_BOTH_LABEL_SETS",
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
            "probe_scores_recomputed": 0,
            "probe_searches_rerun": 0,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "source_split_leaks": source_overlap,
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
            "source_split_sha256": sha256_bytes(artifacts["source-split-v4.json"]),
            "calibration_sha256": sha256_bytes(artifacts["split-conformal-calibration-v4.json"]),
            "results_sha256": sha256_bytes(artifacts["lite-results-v4.json"]),
        },
        "evidence_boundary": [
            "V4 is run-informed development work and not an independent replication.",
            "Thresholds use calibration sources; gate metrics use disjoint validation sources.",
            "V3 scores and v1 searches/witnesses are immutable and not recomputed.",
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
        config = V4Config.load(args.config)
        audit = run_v4(config, output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"S7 Lite v4 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
