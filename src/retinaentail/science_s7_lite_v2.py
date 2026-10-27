"""Run-informed Science S7 Lite v2 contrastive conformal amendment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import tracemalloc
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
    FeatureCache,
    S7Config,
    S7LiteError,
    ThresholdResolution,
    _apply_multi_region,
    _cluster_bootstrap_fcr,
    _load_s6_contract,
    _p_value,
    _policy_metrics,
    build_lite_bank,
    resolve_threshold,
    search_lite,
)
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "2.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V2"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class S7LiteV2Error(ValueError):
    """Raised when the frozen v2 amendment cannot be replayed safely."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV2Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV2Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV2Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV2Error(f"{name} is not SHA-256")
    return text


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV2Error(f"binding mismatch: {name}")
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
        raise S7LiteV2Error("v2 baseline drift")


@dataclass(frozen=True, slots=True)
class V2Config:
    path: Path
    config_sha256: str
    seed: int
    alpha: float
    minimum_region_units: int
    minimum_group_units: int
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V2Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v2 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_AMENDMENT"
        ):
            raise S7LiteV2Error("unsupported S7 v2 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_observed": True,
            "v1_status": "BLOCKED_AUTOMATED_GATE",
            "independent_replication": False,
            "final_test_opened": False,
        }:
            raise S7LiteV2Error("v2 disclosure changed")
        amendment = _mapping(root.get("amendment"), "amendment")
        if amendment.get("method") != "LABEL_CONDITIONAL_CONTRASTIVE_CONFORMAL_SET":
            raise S7LiteV2Error("v2 method changed")
        if amendment.get("present_nonconformity") != "PRESENT_SCORE_MINUS_ABSENT_SCORE":
            raise S7LiteV2Error("v2 present score changed")
        if amendment.get("absent_nonconformity") != "ABSENT_SCORE_MINUS_PRESENT_SCORE":
            raise S7LiteV2Error("v2 absent score changed")
        if amendment.get("probe_policy") != "REUSE_V1_PROBES_SCORES_WITNESSES":
            raise S7LiteV2Error("v2 probe policy changed")
        if amendment.get("additional_searches") != "BANK_ONLY_OPPOSITE_HYPOTHESIS_768":
            raise S7LiteV2Error("v2 search scope changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        if calibration.get("hierarchies") != list(CALIBRATION_HIERARCHIES):
            raise S7LiteV2Error("v2 hierarchies changed")
        if calibration.get("rank_rule") != "CEIL_N_PLUS_1_TIMES_ONE_MINUS_ALPHA":
            raise S7LiteV2Error("v2 rank rule changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required = {
            "v2_prompt",
            "v2_code",
            "v2_audit_schema",
            "v1_config",
            "v1_code",
            "v1_audit",
            "v1_artifact_manifest",
            "v1_calibration",
            "v1_results",
        }
        if set(bindings) != required:
            raise S7LiteV2Error("v2 binding set changed")
        gate = _mapping(root.get("gate"), "gate")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            seed=_integer(root.get("seed"), "seed"),
            alpha=_number(calibration.get("alpha"), "alpha"),
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


def _audit_manifest(root: Path, manifest_path: Path) -> dict[str, int]:
    decoded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(decoded, "v1 artifact manifest")
    rows = _sequence(manifest.get("files"), "manifest files")
    listed: dict[str, Mapping[str, object]] = {}
    mismatches = 0
    verified_bytes = 0
    for raw in rows:
        row = _mapping(raw, "manifest row")
        relative = _safe_relative_string(row.get("relative_path"), "manifest path")
        path = root.joinpath(*PurePosixPath(relative).parts)
        listed[relative] = row
        expected_size = _integer(row.get("byte_size"), "manifest size")
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != _sha256(row.get("sha256"), "manifest SHA-256")
        ):
            mismatches += 1
        else:
            verified_bytes += expected_size
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    return {
        "entries": len(rows),
        "verified_bytes": verified_bytes,
        "mismatches": mismatches,
        "unlisted": len(actual.difference(listed)),
        "missing": len(set(listed).difference(actual)),
    }


def _v1_probe_hash_audit(rows: Sequence[Mapping[str, object]], root: Path) -> dict[str, int]:
    checks = 0
    mismatches = 0
    for row in rows:
        observation = root.joinpath(
            *PurePosixPath(
                _safe_relative_string(row.get("observation_png_path"), "observation")
            ).parts
        )
        checks += 1
        mismatches += int(
            not observation.is_file()
            or sha256_file(observation) != _sha256(row.get("observation_png_sha256"), "observation")
        )
        for hypothesis in HYPOTHESES:
            raw_search = row.get(f"{hypothesis.lower()}_search")
            if raw_search is None:
                continue
            search = _mapping(raw_search, "v1 search")
            candidate = root.joinpath(
                *PurePosixPath(
                    _safe_relative_string(search.get("candidate_png_path"), "candidate")
                ).parts
            )
            checks += 1
            mismatches += int(
                not candidate.is_file()
                or sha256_file(candidate)
                != _sha256(search.get("candidate_png_sha256"), "candidate SHA-256")
            )
    return {"checks": checks, "mismatches": mismatches}


def _resolve(
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


def _contrastive_decision(
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


def run_v2(config: V2Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV2Error("S7 v2 output directory exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    tracemalloc.start()
    v1_config = S7Config.load(config.bindings["v1_config"])
    if v1_config.alpha != config.alpha:
        raise S7LiteV2Error("v2 alpha differs from v1")
    s6_config = _load_s6_contract(v1_config)
    io_audit = IOAudit()
    records, collisions, _domain = load_inputs(s6_config)
    selected_banks, selected_cases = select_bank_and_cases(records, collisions, s6_config)
    banks = build_lite_bank(selected_banks, s6_config, v1_config, io_audit)
    cache = FeatureCache(v1_config)
    v1_cal_decoded: object = json.loads(
        config.bindings["v1_calibration"].read_text(encoding="utf-8")
    )
    v1_cal_rows = [
        _mapping(raw, "v1 calibration row")
        for raw in _sequence(_mapping(v1_cal_decoded, "v1 calibration").get("rows"), "rows")
    ]
    bank_lookup = {
        (lesion_class, entry.record.twin_id): entry
        for lesion_class in LESION_CLASSES
        for entry in banks[lesion_class]
    }
    contrast_rows: list[CalibrationRow] = []
    cross_rows: list[dict[str, object]] = []
    source_leaks = 0
    for row in v1_cal_rows:
        lesion_class = _string(row.get("lesion_class"), "lesion class")
        truth = _string(row.get("hypothesis"), "hypothesis")
        degradation = _string(row.get("degradation"), "degradation")
        twin_id = _string(row.get("twin_id"), "twin ID")
        entry = bank_lookup[(lesion_class, twin_id)]
        opposite = "ABSENT" if truth == "PRESENT" else "PRESENT"
        observed = entry.variants[truth][degradation].array
        opposite_result = search_lite(
            observed,
            banks[lesion_class],
            opposite,
            v1_config,
            cache,
            excluded_source_key=entry.record.source_key,
        )
        source_leaks += int(opposite_result.anchor.source_key == entry.record.source_key)
        source_leaks += sum(
            candidate.record.source_key == entry.record.source_key
            for candidate in banks[lesion_class]
            if candidate.record.twin_id in opposite_result.retrieved_ids
        )
        own_score = _number(row.get("score"), "v1 own score")
        nonconformity = own_score - opposite_result.score
        contrast_rows.append(
            CalibrationRow(
                lesion_class=lesion_class,
                hypothesis=truth,
                degradation=degradation,
                quality=_string(row.get("quality"), "quality"),
                source_key=entry.record.source_key,
                twin_id=twin_id,
                score=nonconformity,
            )
        )
        cross_rows.append(
            {
                "lesion_class": lesion_class,
                "source_truth": truth,
                "degradation": degradation,
                "quality": row.get("quality"),
                "source_key": entry.record.source_key,
                "twin_id": twin_id,
                "own_score_v1": own_score,
                "opposite_score_v2": opposite_result.score,
                "nonconformity": nonconformity,
                "opposite_anchor_twin_id": opposite_result.anchor.twin_id,
                "candidate_count": opposite_result.candidate_count,
                "optimization_gap": 0.0,
            }
        )
    if len(contrast_rows) != 768:
        raise S7LiteV2Error("v2 did not construct exactly 768 contrastive rows")
    v1_results_decoded: object = json.loads(
        config.bindings["v1_results"].read_text(encoding="utf-8")
    )
    v1_rows = [
        _mapping(raw, "v1 result row")
        for raw in _sequence(_mapping(v1_results_decoded, "v1 results").get("records"), "records")
    ]
    result_rows: list[dict[str, object]] = []
    for raw in v1_rows:
        row = dict(raw)
        row["v1_record_sha256"] = sha256_bytes(_canonical(raw))
        if raw.get("expected_truth") == "OOD":
            for hierarchy in CALIBRATION_HIERARCHIES:
                row[f"state_{hierarchy.lower()}"] = EntailmentState.OUT_OF_MODEL.value
            row["state_region_primary"] = EntailmentState.OUT_OF_MODEL.value
            row["state_source_max"] = EntailmentState.OUT_OF_MODEL.value
            result_rows.append(row)
            continue
        present_score = _number(raw.get("positive_score"), "v1 present score")
        absent_score = _number(raw.get("negative_score"), "v1 absent score")
        present_nonconformity = present_score - absent_score
        absent_nonconformity = absent_score - present_score
        lesion_class = _string(raw.get("lesion_class"), "lesion class")
        degradation = _string(raw.get("observation_condition"), "condition")
        quality = _string(raw.get("quality_bin"), "quality")
        resolutions: dict[str, dict[str, ThresholdResolution]] = {
            hypothesis: {} for hypothesis in HYPOTHESES
        }
        for hypothesis in HYPOTHESES:
            for hierarchy in CALIBRATION_HIERARCHIES:
                resolutions[hypothesis][hierarchy] = _resolve(
                    contrast_rows,
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
            row[f"state_{hierarchy.lower()}"] = _contrastive_decision(
                present_nonconformity,
                absent_nonconformity,
                present.threshold,
                absent.threshold,
            )
        primary_present = resolutions["PRESENT"]["LESION_DEGRADATION_QUALITY"]
        primary_absent = resolutions["ABSENT"]["LESION_DEGRADATION_QUALITY"]
        group_present = _resolve(
            contrast_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="PRESENT",
            degradation=degradation,
            quality=quality,
            source_max=True,
        )
        group_absent = _resolve(
            contrast_rows,
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
                "positive_score": present_nonconformity,
                "negative_score": absent_nonconformity,
                "positive_threshold": primary_present.threshold,
                "negative_threshold": primary_absent.threshold,
                "positive_threshold_hierarchy": primary_present.used_hierarchy,
                "negative_threshold_hierarchy": primary_absent.used_hierarchy,
                "positive_p_value": _p_value(
                    present_nonconformity,
                    primary_present.values,
                ),
                "negative_p_value": _p_value(
                    absent_nonconformity,
                    primary_absent.values,
                ),
                "state_region_primary": row["state_lesion_degradation_quality"],
                "state_source_max": _contrastive_decision(
                    present_nonconformity,
                    absent_nonconformity,
                    group_present.threshold,
                    group_absent.threshold,
                ),
            }
        )
        result_rows.append(row)
    _apply_multi_region(result_rows, v1_config)
    v1_audit_decoded: object = json.loads(config.bindings["v1_audit"].read_text(encoding="utf-8"))
    v1_audit = _mapping(v1_audit_decoded, "v1 audit")
    if v1_audit.get("status") != "BLOCKED_AUTOMATED_GATE":
        raise S7LiteV2Error("v1 failure disposition changed")
    v1_root = config.bindings["v1_audit"].parent
    manifest_audit = _audit_manifest(v1_root, config.bindings["v1_artifact_manifest"])
    probe_hash_audit = _v1_probe_hash_audit(v1_rows, v1_root)
    v2_states_decoded: object = json.loads(
        v1_config.bindings["s6_v2_results"].read_text(encoding="utf-8")
    )
    verify_states = {
        _string(_mapping(raw, "S6 row").get("case_id"), "case ID"): _string(
            _mapping(raw, "S6 row").get("state"), "state"
        )
        for raw in _sequence(_mapping(v2_states_decoded, "S6 results").get("records"), "S6 rows")
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
    preservation_pass = (
        manifest_audit["mismatches"] == 0
        and manifest_audit["unlisted"] == 0
        and manifest_audit["missing"] == 0
        and probe_hash_audit["mismatches"] == 0
        and len(result_rows) == 560
    )
    gate_pass = (
        preservation_pass
        and source_leaks == 0
        and len(cross_rows) == 768
        and ood_pass == len(ood) == 160
        and primary_fcr <= config.maximum_fcr
        and all(value >= config.minimum_moderate_coverage for value in moderate_coverage.values())
        and agreement_rate >= config.minimum_verify_agreement
    )
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "method": "LABEL_CONDITIONAL_CONTRASTIVE_CONFORMAL_SET",
        "alpha": config.alpha,
        "cross_hypothesis_rows": cross_rows,
    }
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "probe_searches_rerun": 0,
        "records": result_rows,
    }
    artifacts: dict[str, bytes] = {
        "contrastive-calibration-v2.json": _canonical(calibration_payload),
        "lite-results-v2.json": _canonical(results_payload),
    }
    for lesion_class in LESION_CLASSES:
        source = v1_root / f"qc-{lesion_class.lower()}-lite-contact-sheet.png"
        artifacts[f"qc-{lesion_class.lower()}-lite-v2-contact-sheet.png"] = source.read_bytes()
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_seconds = time.perf_counter() - start
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_observed": True,
            "v1_status": "BLOCKED_AUTOMATED_GATE",
            "independent_replication": False,
            "final_test_opened": False,
        },
        "method": {
            "name": "LABEL_CONDITIONAL_CONTRASTIVE_CONFORMAL_SET",
            "probe_searches_rerun": 0,
            "bank_only_cross_hypothesis_searches": len(cross_rows),
            "finite_zero_gap_searches": len(cross_rows),
            "source_leaks": source_leaks,
        },
        "preservation": {
            "v1_manifest": manifest_audit,
            "v1_probe_hashes": probe_hash_audit,
            "v1_probe_records": len(v1_rows),
            "v2_probe_records": len(result_rows),
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
            "learned_parameters": 0,
            "peak_python_tracemalloc_bytes": peak_bytes,
            "wall_seconds": wall_seconds,
            "silent_resolution_change": False,
        },
        "io_audit": {
            "probe_searches_rerun": 0,
            "s4_twin_png_decodes_for_bank_only_calibration": io_audit.s4_twin_png_decodes,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "source_split_leaks": source_leaks,
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
            "contrastive_calibration_sha256": sha256_bytes(
                artifacts["contrastive-calibration-v2.json"]
            ),
            "results_sha256": sha256_bytes(artifacts["lite-results-v2.json"]),
        },
        "evidence_boundary": [
            "V2 is run-informed development work and not an independent replication.",
            "All v1 probe scores, candidates and witness hashes are preserved without "
            "search rerun.",
            "The conformal prediction set is relative to the frozen finite Lite hypothesis class.",
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
        config = V2Config.load(args.config)
        audit = run_v2(config, output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError, S7LiteError) as error:
        print(f"S7 Lite v2 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
