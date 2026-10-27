"""Run-informed S6 v2 conditional-threshold amendment.

V2 preserves every v1 probe, finite-grid optimum, bound, and witness.  It fits
identity/glare thresholds independently from source-isolated exemplar-bank replay
and changes only threshold-conditioned feasibility and four-valued state.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
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
    make_labeled_contact_sheet,
)
from retinaentail.science_s6_verify import (
    ALL_STRATA,
    HYPOTHESES,
    LESION_CLASSES,
    PROJECT_ROOT,
    IOAudit,
    S6VerifyError,
    build_bank,
    load_inputs,
    search_finite_grid,
    select_bank_and_cases,
)
from retinaentail.science_s6_verify import S6Config as V1Config
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "2.0.0"
EXPERIMENT_ID: Final = "S6_RETINAENTAIL_VERIFY_V2"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_DEVELOPMENT_NONTRIVIALITY"
CONDITIONS: Final = ("IDENTITY", "S5_TARGET_GLARE_SEVERITY_1")


class S6V2Error(S6VerifyError):
    """Raised when the v2 amendment cannot preserve the v1 evidence chain."""


def _sha256_text(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S6V2Error(f"{name} is not SHA-256")
    return text


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S6V2Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S6V2Error(f"{name} must be finite and >= {minimum}")
    return result


def _verify_binding(binding: Mapping[str, object], name: str) -> Path:
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256_text(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S6V2Error(f"binding mismatch: {name}")
    return path


def _verify_baseline(expected: object) -> None:
    commit = _string(expected, "expected baseline commit")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != commit:
        raise S6V2Error("S6 v2 baseline drift")


@dataclass(frozen=True, slots=True)
class V2Config:
    path: Path
    payload: dict[str, object]
    config_sha256: str
    base: V1Config
    threshold_quantile: float
    threshold_multiplier: float
    minimum_ambiguity_agreement: float
    binding_paths: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V2Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S6 v2 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_AMENDMENT"
        ):
            raise S6V2Error("unsupported S6 v2 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "development disclosure")
        if disclosure != {
            "v1_observed": True,
            "reason": "IDENTITY_THRESHOLD_MISALIGNED_WITH_GLARE_CONDITION",
            "independent_replication": False,
            "final_test_opened": False,
        }:
            raise S6V2Error("S6 v2 disclosure changed")
        amendment = _mapping(root.get("amendment"), "amendment")
        if (
            amendment.get("conditions") != list(CONDITIONS)
            or amendment.get("calibration_policy")
            != "LEAVE_ONE_SOURCE_OUT_HIGHER_QUANTILE_TIMES_FROZEN_MULTIPLIER"
            or amendment.get("probe_search_policy") != "REUSE_V1_IMMUTABLE_OPTIMA_AND_WITNESSES"
            or amendment.get("condition_source") != "REGISTERED_OBSERVATION_TRACE_ONLY"
        ):
            raise S6V2Error("S6 v2 amendment changed")
        bindings = _mapping(root.get("bindings"), "bindings")
        binding_paths = {
            name: _verify_binding(_mapping(raw, f"binding {name}"), name)
            for name, raw in bindings.items()
        }
        required = {
            "v2_prompt",
            "v2_code",
            "v2_audit_schema",
            "v1_config",
            "v1_report",
            "v1_ai_qc",
            "v1_audit",
            "v1_artifact_manifest",
            "v1_results",
        }
        if set(binding_paths) != required:
            raise S6V2Error("S6 v2 binding set changed")
        base = V1Config.load(binding_paths["v1_config"])
        return cls(
            path=resolved,
            payload=dict(root),
            config_sha256=sha256_file(resolved),
            base=base,
            threshold_quantile=_number(amendment.get("threshold_quantile"), "quantile"),
            threshold_multiplier=_number(
                amendment.get("threshold_multiplier"), "threshold multiplier", 1.0
            ),
            minimum_ambiguity_agreement=_number(
                _mapping(root.get("gate"), "gate").get("minimum_ambiguity_agreement"),
                "ambiguity agreement gate",
            ),
            binding_paths=binding_paths,
        )


def _verify_v1_manifest(config: V2Config) -> dict[str, int]:
    root = config.binding_paths["v1_artifact_manifest"].parent
    decoded: object = json.loads(
        config.binding_paths["v1_artifact_manifest"].read_text(encoding="utf-8")
    )
    rows = _sequence(_mapping(decoded, "v1 artifact manifest").get("files"), "manifest files")
    listed: set[str] = set()
    verified_bytes = 0
    mismatches = 0
    for raw in rows:
        row = _mapping(raw, "manifest row")
        relative = _safe_relative_string(row.get("relative_path"), "artifact path")
        if relative in listed:
            raise S6V2Error("duplicate v1 artifact path")
        listed.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        expected_size = row.get("byte_size")
        expected_sha = _sha256_text(row.get("sha256"), "artifact SHA-256")
        if (
            not path.is_file()
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            mismatches += 1
        elif isinstance(expected_size, int):
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
    }


def calibrate_conditional_thresholds(
    config: V2Config,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, object]], IOAudit]:
    """Fit identity/glare thresholds without reading any v1 probe score."""

    io_audit = IOAudit()
    records, collisions, _ = load_inputs(config.base)
    selected_banks, _ = select_bank_and_cases(records, collisions, config.base)
    banks = build_bank(selected_banks, config.base, io_audit)
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    calibration_rows: list[dict[str, object]] = []
    for lesion_class in LESION_CLASSES:
        class_conditions: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            condition_thresholds: dict[str, float] = {}
            for hypothesis in HYPOTHESES:
                scores: list[float] = []
                for query in banks[lesion_class]:
                    observed = query.variants[hypothesis][condition].array
                    result = search_finite_grid(
                        observed,
                        banks[lesion_class],
                        hypothesis,
                        config.base,
                        excluded_source_key=query.record.source_key,
                    )
                    scores.append(result.score)
                    calibration_rows.append(
                        {
                            "lesion_class": lesion_class,
                            "condition": condition,
                            "hypothesis": hypothesis,
                            "query_twin_id": query.record.twin_id,
                            "query_source_key": query.record.source_key,
                            "best_anchor_twin_id": result.anchor.twin_id,
                            "best_anchor_source_key": result.anchor.source_key,
                            "score": result.score,
                            "candidate_count": result.candidate_count,
                            "optimization_gap": result.gap,
                        }
                    )
                quantile = float(
                    np.quantile(np.asarray(scores), config.threshold_quantile, method="higher")
                )
                condition_thresholds[hypothesis] = max(
                    1e-12, quantile * config.threshold_multiplier
                )
            class_conditions[condition] = condition_thresholds
        thresholds[lesion_class] = class_conditions
    return thresholds, calibration_rows, io_audit


def _png_bytes(image: NDArray[np.uint8]) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _load_png(path: Path) -> NDArray[np.uint8]:
    if not path.is_file():
        raise S6V2Error(f"missing v1 PNG: {path}")
    with Image.open(path) as source:
        source.load()
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def _condition_for(record: Mapping[str, object]) -> str:
    stratum = _string(record.get("stratum"), "v1 stratum")
    if stratum == "S5_TIER_A_AMBIGUOUS_ID":
        return "S5_TARGET_GLARE_SEVERITY_1"
    if stratum in {"PRESENT_ID", "ABSENT_ID"}:
        return "IDENTITY"
    raise S6V2Error("OOD record has no threshold condition")


def reclassify_v1_results(
    config: V2Config,
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[list[dict[str, object]], dict[str, bytes], dict[str, object]]:
    v1_root = config.binding_paths["v1_results"].parent
    decoded: object = json.loads(config.binding_paths["v1_results"].read_text(encoding="utf-8"))
    v1_results = _mapping(decoded, "v1 results")
    records = _sequence(v1_results.get("records"), "v1 result records")
    updated: list[dict[str, object]] = []
    artifacts: dict[str, bytes] = {}
    state_counts: dict[str, Counter[str]] = {
        lesion_class: Counter() for lesion_class in LESION_CLASSES
    }
    stratum_counts: dict[str, Counter[str]] = {
        lesion_class: Counter() for lesion_class in LESION_CLASSES
    }
    witness_hash_checks = 0
    witness_hash_mismatches = 0
    preserved_zero_gap = 0
    ood_records = 0
    ood_preserved = 0
    ambiguity_records = 0
    ambiguity_unidentifiable = 0
    qc: dict[str, dict[str, list[tuple[str, NDArray[np.uint8]]]]] = {
        lesion_class: {stratum: [] for stratum in ALL_STRATA} for lesion_class in LESION_CLASSES
    }
    for raw in records:
        original = _mapping(raw, "v1 result")
        record = copy.deepcopy(dict(original))
        lesion_class = _string(record.get("lesion_class"), "lesion class")
        stratum = _string(record.get("stratum"), "stratum")
        if lesion_class not in LESION_CLASSES or stratum not in ALL_STRATA:
            raise S6V2Error("v1 result class/stratum changed")
        observation_path = v1_root.joinpath(
            *PurePosixPath(
                _safe_relative_string(record.get("observation_png_path"), "observation path")
            ).parts
        )
        observation = _load_png(observation_path)
        if sha256_bytes(_png_bytes(observation)) != record.get("observation_png_sha256"):
            raise S6V2Error("v1 observation PNG hash mismatch")
        stratum_counts[lesion_class][stratum] += 1
        if stratum in {"NEAR_OOD", "FAR_OOD"}:
            ood_records += 1
            ood_preserved += int(
                record.get("state") == "OUT_OF_MODEL"
                and record.get("positive_search") is None
                and record.get("negative_search") is None
            )
            placeholder = np.zeros_like(observation)
            if len(qc[lesion_class][stratum]) < 6:
                qc[lesion_class][stratum].extend(
                    [
                        (f"{stratum} obs", observation),
                        ("positive OOM", placeholder),
                        ("negative OOM", placeholder),
                    ]
                )
        else:
            condition = _condition_for(record)
            search_records: dict[str, dict[str, object]] = {}
            for hypothesis, field in (
                ("PRESENT", "positive_search"),
                ("ABSENT", "negative_search"),
            ):
                search = copy.deepcopy(dict(_mapping(record.get(field), field)))
                candidate_relative = _safe_relative_string(
                    search.get("candidate_png_path"), "candidate path"
                )
                candidate_path = v1_root.joinpath(*PurePosixPath(candidate_relative).parts)
                witness_hash_checks += 1
                if (
                    not candidate_path.is_file()
                    or sha256_file(candidate_path) != search.get("candidate_png_sha256")
                    or search.get("independent_replay_exact") is not True
                    or search.get("predicate_provenance_valid") is not True
                ):
                    witness_hash_mismatches += 1
                threshold = thresholds[lesion_class][condition][hypothesis]
                score = _number(search.get("score"), "v1 search score")
                search["v1_threshold"] = search.get("threshold")
                search["threshold"] = threshold
                search["threshold_condition"] = condition
                search["feasible"] = score <= threshold
                preserved_zero_gap += int(
                    search.get("optimization_gap") == 0.0
                    and search.get("lower_bound") == search.get("upper_bound")
                )
                search_records[hypothesis] = search
                record[field] = search
            positive = search_records["PRESENT"]
            negative = search_records["ABSENT"]
            state = fail_closed_state(
                positive_feasible=cast(bool, positive["feasible"]),
                negative_feasible=cast(bool, negative["feasible"]),
                positive_score=cast(float, positive["score"]),
                negative_score=cast(float, negative["score"]),
                positive_threshold=cast(float, positive["threshold"]),
                negative_threshold=cast(float, negative["threshold"]),
                positive_solver_status="optimal",
                negative_solver_status="optimal",
                solver_status="optimal",
                degradation_known=True,
            )
            record["v1_state"] = record.get("state")
            record["state"] = state.value
            record["threshold_condition"] = condition
            if stratum == "S5_TIER_A_AMBIGUOUS_ID":
                ambiguity_records += 1
                ambiguity_unidentifiable += int(state is EntailmentState.RAW_UNIDENTIFIABLE)
            if len(qc[lesion_class][stratum]) < 6:
                positive_image = _load_png(
                    v1_root.joinpath(
                        *PurePosixPath(
                            _safe_relative_string(
                                positive.get("candidate_png_path"), "positive candidate"
                            )
                        ).parts
                    )
                )
                negative_image = _load_png(
                    v1_root.joinpath(
                        *PurePosixPath(
                            _safe_relative_string(
                                negative.get("candidate_png_path"), "negative candidate"
                            )
                        ).parts
                    )
                )
                qc[lesion_class][stratum].extend(
                    [
                        (f"{stratum} obs", observation),
                        (
                            f"P {positive['score']:.3f}/{positive['threshold']:.3f}",
                            positive_image,
                        ),
                        (
                            f"A {negative['score']:.3f}/{negative['threshold']:.3f}",
                            negative_image,
                        ),
                    ]
                )
        state_counts[lesion_class][_string(record.get("state"), "state")] += 1
        updated.append(record)
    for lesion_class in LESION_CLASSES:
        cells = [cell for stratum in ALL_STRATA for cell in qc[lesion_class][stratum]]
        artifacts[f"qc-{lesion_class.lower()}-verify-v2-contact-sheet.png"] = (
            make_labeled_contact_sheet(cells, columns=3)
        )
    diagnostics: dict[str, object] = {
        "record_count": len(updated),
        "state_counts_per_class": {
            lesion_class: dict(sorted(state_counts[lesion_class].items()))
            for lesion_class in LESION_CLASSES
        },
        "strata_counts_per_class": {
            lesion_class: dict(sorted(stratum_counts[lesion_class].items()))
            for lesion_class in LESION_CLASSES
        },
        "witness_hash_checks": witness_hash_checks,
        "witness_hash_mismatches": witness_hash_mismatches,
        "preserved_zero_gap_searches": preserved_zero_gap,
        "ood_records": ood_records,
        "ood_preserved_out_of_model": ood_preserved,
        "ambiguity_records": ambiguity_records,
        "ambiguity_raw_unidentifiable": ambiguity_unidentifiable,
        "ambiguity_agreement_rate": (
            ambiguity_unidentifiable / ambiguity_records if ambiguity_records else 0.0
        ),
    }
    return updated, artifacts, diagnostics


def run_v2(config: V2Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S6V2Error("S6 v2 output directory exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    manifest_audit = _verify_v1_manifest(config)
    v1_audit_decoded: object = json.loads(
        config.binding_paths["v1_audit"].read_text(encoding="utf-8")
    )
    v1_audit = _mapping(v1_audit_decoded, "v1 audit")
    if (
        v1_audit.get("status") != "READY_FOR_AI_QC"
        or _mapping(v1_audit.get("gate"), "v1 gate").get("automated") != "PASS"
    ):
        raise S6V2Error("S6 v1 mechanics gate is not PASS")
    thresholds, calibration_rows, calibration_io = calibrate_conditional_thresholds(config)
    updated, qc_artifacts, diagnostics = reclassify_v1_results(config, thresholds)
    threshold_payload = {
        "schema_version": SCHEMA_VERSION,
        "policy": "CONDITIONED_LEAVE_ONE_SOURCE_OUT_HIGHER_QUANTILE_TIMES_FROZEN_MULTIPLIER",
        "conditions": list(CONDITIONS),
        "quantile": config.threshold_quantile,
        "multiplier": config.threshold_multiplier,
        "thresholds": thresholds,
        "records": calibration_rows,
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_results_sha256": sha256_file(config.binding_paths["v1_results"]),
        "witness_artifact_root": "reports/generated/science/s6_retinaentail_verify_v1",
        "amendment_scope": "THRESHOLD_CONDITION_AND_FEASIBILITY_STATE_ONLY",
        "records": updated,
    }
    artifacts: dict[str, bytes] = {
        "conditional-thresholds-v2.json": _canonical(threshold_payload),
        "verify-results-v2.json": _canonical(result_payload),
        **qc_artifacts,
    }
    agreement = _number(diagnostics["ambiguity_agreement_rate"], "agreement")
    automated_gate = (
        manifest_audit["mismatches"] == 0
        and manifest_audit["unlisted"] == 0
        and diagnostics["record_count"] == 400
        and diagnostics["witness_hash_checks"] == 480
        and diagnostics["witness_hash_mismatches"] == 0
        and diagnostics["preserved_zero_gap_searches"] == 480
        and diagnostics["ood_records"] == 160
        and diagnostics["ood_preserved_out_of_model"] == 160
        and agreement >= config.minimum_ambiguity_agreement
    )
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if automated_gate else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        },
        "v1_manifest_audit": manifest_audit,
        "conditional_calibration": {
            "conditions": list(CONDITIONS),
            "calibration_searches": len(calibration_rows),
            "s4_twin_png_decodes": calibration_io.s4_twin_png_decodes,
            "source_isolation_violations": sum(
                row["query_source_key"] == row["best_anchor_source_key"] for row in calibration_rows
            ),
        },
        "preservation": diagnostics,
        "gate": {
            "minimum_ambiguity_agreement": config.minimum_ambiguity_agreement,
            "observed_ambiguity_agreement": agreement,
            "automated": "PASS" if automated_gate else "FAIL",
            "ai_visual_qc": "PENDING",
            "s6_closed": False,
            "s7_authorized": False,
        },
        "io_audit": {
            "v1_probe_searches_rerun": 0,
            "final_locked_test_reads": 0,
            "online_evaluation_reads": 0,
            "uwf_reads": 0,
            "dr_grade_values_used": 0,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "conditional_thresholds_sha256": sha256_bytes(
                artifacts["conditional-thresholds-v2.json"]
            ),
            "verify_results_v2_sha256": sha256_bytes(artifacts["verify-results-v2.json"]),
        },
        "evidence_boundary": [
            "V2 is run-informed development work and is not an independent replication.",
            "V2 changes thresholds/feasibility/state only; v1 optima, gaps and witnesses "
            "are immutable.",
            "Ambiguity agreement is a development adequacy diagnostic, not FCR or "
            "clinical performance.",
            "No final locked test, Lite, enhancer or paper-performance experiment ran.",
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
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Science S6 v2 failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 3


if __name__ == "__main__":
    raise SystemExit(main())
