"""Run post-S8 diagnostic ablations without changing the frozen S8 result."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from retinaentail.data.manifest import sha256_file
from retinaentail.witnesses.replay import canonical_json_bytes, write_bytes_fail_safe

EXPECTED_S8_RESULTS = "82ca0ff513cbc0b969895009468dcb0be283ed3d9df1d297103b98f7c85a7a3b"
EXPECTED_S8_CALIBRATION = "3cf5589b6553fafe7e9b5b0cf6bf49ee1d0e393e5e9973fef71ab738fdb733fe"
DETERMINATE = {"ENTAILED_PRESENT", "ENTAILED_ABSENT"}
FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
JOINT_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


class S9DiagnosticError(ValueError):
    """Raised when frozen S8 inputs or diagnostic rules change."""


def _load(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise S9DiagnosticError("diagnostic input is not an object")
    return value


def _false(state: str, truth: str) -> bool:
    return (state == "ENTAILED_PRESENT" and truth == "ABSENT") or (
        state == "ENTAILED_ABSENT" and truth == "PRESENT"
    )


def _state(
    positive: float, negative: float, positive_threshold: float, negative_threshold: float
) -> str:
    feasible = (positive <= positive_threshold, negative <= negative_threshold)
    return {
        (True, False): "ENTAILED_PRESENT",
        (False, True): "ENTAILED_ABSENT",
        (True, True): "RAW_UNIDENTIFIABLE",
        (False, False): "OUT_OF_MODEL",
    }[feasible]


def _metrics(states: Sequence[tuple[str, str]]) -> dict[str, object]:
    determinate = [(state, truth) for state, truth in states if state in DETERMINATE]
    false = sum(_false(state, truth) for state, truth in determinate)
    return {
        "total": len(states),
        "determinate": len(determinate),
        "coverage": len(determinate) / len(states) if states else 0.0,
        "false": false,
        "fcr": false / len(determinate) if determinate else None,
        "state_counts": dict(sorted(Counter(state for state, _ in states).items())),
    }


def _exact_two_sided_binomial(successes: int, total: int) -> float | None:
    if total == 0:
        return None
    tail = sum(math.comb(total, index) for index in range(0, min(successes, total - successes) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _paired_error_test(
    rows: Sequence[Mapping[str, object]], first_key: str, second_key: str
) -> dict[str, object]:
    common = [
        row
        for row in rows
        if row.get(first_key) in DETERMINATE and row.get(second_key) in DETERMINATE
    ]
    first_only_wrong = 0
    second_only_wrong = 0
    for row in common:
        truth = str(row["expected_truth"])
        first_wrong = _false(str(row[first_key]), truth)
        second_wrong = _false(str(row[second_key]), truth)
        first_only_wrong += int(first_wrong and not second_wrong)
        second_only_wrong += int(second_wrong and not first_wrong)
    discordant = first_only_wrong + second_only_wrong
    return {
        "common_determinate_rows": len(common),
        "first_wrong_second_correct": first_only_wrong,
        "first_correct_second_wrong": second_only_wrong,
        "discordant": discordant,
        "exact_two_sided_p": _exact_two_sided_binomial(first_only_wrong, discordant),
    }


def analyze(results_path: Path, calibration_path: Path) -> dict[str, object]:
    """Return frozen, explicitly post-hoc S9 diagnostics."""

    if sha256_file(results_path) != EXPECTED_S8_RESULTS:
        raise S9DiagnosticError("S8 results hash changed")
    if sha256_file(calibration_path) != EXPECTED_S8_CALIBRATION:
        raise S9DiagnosticError("S8 calibration hash changed")
    results = _load(results_path)
    rows_raw = results.get("rows")
    if not isinstance(rows_raw, list) or len(rows_raw) != 17442:
        raise S9DiagnosticError("S8 row count changed")
    rows = [row for row in rows_raw if isinstance(row, dict)]
    if len(rows) != len(rows_raw):
        raise S9DiagnosticError("S8 row type changed")

    selective_curve = []
    for factor in FACTORS:
        states = []
        for row in rows:
            forced = str(row["state_forced_nearest"])
            margin = float(row["selective_margin"])
            threshold = float(row["selective_margin_threshold"])
            state = forced if margin > factor * threshold else "RAW_UNIDENTIFIABLE"
            states.append((state, str(row["expected_truth"])))
        selective_curve.append({"factor": factor, **_metrics(states)})

    symmetric_curve = []
    for factor in FACTORS[1:]:
        states = [
            (
                _state(
                    float(row["positive_score"]),
                    float(row["negative_score"]),
                    factor * float(row["positive_threshold"]),
                    factor * float(row["negative_threshold"]),
                ),
                str(row["expected_truth"]),
            )
            for row in rows
        ]
        symmetric_curve.append({"factor": factor, **_metrics(states)})

    asymmetric_grid = []
    for positive_factor in JOINT_FACTORS:
        for negative_factor in JOINT_FACTORS:
            states = [
                (
                    _state(
                        float(row["positive_score"]),
                        float(row["negative_score"]),
                        positive_factor * float(row["positive_threshold"]),
                        negative_factor * float(row["negative_threshold"]),
                    ),
                    str(row["expected_truth"]),
                )
                for row in rows
            ]
            asymmetric_grid.append(
                {
                    "positive_factor": positive_factor,
                    "negative_factor": negative_factor,
                    **_metrics(states),
                }
            )

    by_condition: dict[str, object] = {}
    by_condition_lesion: dict[str, object] = {}
    for condition in sorted({str(row["condition_id"]) for row in rows}):
        group = [row for row in rows if row["condition_id"] == condition]
        by_condition[condition] = _metrics(
            [(str(row["state_four_valued"]), str(row["expected_truth"])) for row in group]
        )
        for lesion in ("HE", "EX", "MA", "SE"):
            lesion_group = [row for row in group if row["lesion_class"] == lesion]
            by_condition_lesion[f"{condition}|{lesion}"] = _metrics(
                [
                    (str(row["state_four_valued"]), str(row["expected_truth"]))
                    for row in lesion_group
                ]
            )
    by_severity: dict[str, object] = {}
    for severity in ("IDENTITY", "LIGHT", "MODERATE", "HEAVY"):
        group = [row for row in rows if row["severity_level"] == severity]
        by_severity[severity] = _metrics(
            [(str(row["state_four_valued"]), str(row["expected_truth"])) for row in group]
        )

    policy_metrics = {
        name: _metrics([(str(row[key]), str(row["expected_truth"])) for row in rows])
        for name, key in {
            "Four-valued": "state_four_valued",
            "Forced nearest": "state_forced_nearest",
            "Selective margin": "state_selective_margin",
            "Present only": "state_present_only",
            "Absent only": "state_absent_only",
        }.items()
    }
    recent_methods = [
        {
            "method": "MDA-Net",
            "year": 2024,
            "venue": "Medical Image Analysis",
            "official_url": "https://github.com/RuoyuGuo/MDA-Net",
            "code": "AVAILABLE",
            "checkpoint": "NOT_PUBLISHED_IN_OFFICIAL_README",
            "local_numeric_comparison": "NOT_RUN__FROM_SCRATCH_TRAINING_REQUIRED",
        },
        {
            "method": "FD3",
            "year": 2024,
            "venue": "arXiv",
            "official_url": "https://github.com/heeheee888/FD3",
            "code": "AVAILABLE",
            "checkpoint": "METHOD_SPECIFIC_CHECKPOINT_REQUIRED",
            "local_numeric_comparison": "NOT_RUN__ADM_DIFFUSION_ROUTE_EXCEEDS_3GB_UNVERIFIED",
        },
        {
            "method": "Contextual-OT",
            "year": 2025,
            "venue": "WACV",
            "official_url": "https://github.com/Retinal-Research/Contextual-OT",
            "code": "AVAILABLE",
            "checkpoint": "GOOGLE_DRIVE_LINK_AND_EXTERNAL_VGG19_DEPENDENCY",
            "local_numeric_comparison": "NOT_RUN__DEPENDENCY_AND_MEMORY_CONTRACT_NOT_REPRODUCED",
        },
        {
            "method": "EyeBench",
            "year": 2025,
            "venue": "arXiv",
            "official_url": "https://github.com/Retinal-Research/EyeBench",
            "code": "README_ONLY_AT_AUDIT_DATE",
            "checkpoint": "NOT_APPLICABLE_BENCHMARK",
            "local_numeric_comparison": "CONCEPTUAL_PROTOCOL_COMPARISON_ONLY",
        },
        {
            "method": "TPOT-DA",
            "year": 2025,
            "venue": "Official repository extension of ISBI TPOT",
            "official_url": "https://github.com/Retinal-Research/TPOT-DA",
            "code": "AVAILABLE",
            "checkpoint": "NOT_BOUND",
            "local_numeric_comparison": "NOT_RUN__REPOSITORY_LICENSE_ONHOLD",
        },
    ]
    return {
        "schema_version": "1.0.0",
        "experiment_id": "S9_BOUNDARY_DIAGNOSTICS_V1",
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC_NOT_CONFIRMATORY",
        "bindings": {
            "s8_results_sha256": EXPECTED_S8_RESULTS,
            "s8_calibration_sha256": EXPECTED_S8_CALIBRATION,
        },
        "row_count": len(rows),
        "policy_metrics": policy_metrics,
        "selective_risk_coverage": selective_curve,
        "symmetric_threshold_sensitivity": symmetric_curve,
        "asymmetric_threshold_grid": asymmetric_grid,
        "by_condition": by_condition,
        "by_condition_lesion": by_condition_lesion,
        "by_severity": by_severity,
        "paired_tests": {
            "four_valued_vs_forced": _paired_error_test(
                rows, "state_four_valued", "state_forced_nearest"
            ),
            "four_valued_vs_selective": _paired_error_test(
                rows, "state_four_valued", "state_selective_margin"
            ),
        },
        "recent_method_executability": recent_methods,
        "evidence_boundary": [
            "All threshold-factor analyses are post-S8 diagnostics, not a new confirmation.",
            "NOT_RUN methods have no local numeric result.",
            "No final locked-test content, clinical endpoint, or enhancer training was used.",
        ],
    }


def _source_csv(payload: Mapping[str, object]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("analysis", "key", "coverage", "fcr", "determinate", "false"))
    for analysis in ("selective_risk_coverage", "symmetric_threshold_sensitivity"):
        rows = payload[analysis]
        if not isinstance(rows, list):
            raise S9DiagnosticError("diagnostic curve is invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise S9DiagnosticError("diagnostic curve row is invalid")
            writer.writerow(
                (
                    analysis,
                    row["factor"],
                    row["coverage"],
                    row["fcr"],
                    row["determinate"],
                    row["false"],
                )
            )
    for analysis in ("by_condition", "by_severity", "policy_metrics"):
        rows = payload[analysis]
        if not isinstance(rows, dict):
            raise S9DiagnosticError("diagnostic grouping is invalid")
        for key, row in rows.items():
            if not isinstance(row, dict):
                raise S9DiagnosticError("diagnostic group row is invalid")
            writer.writerow(
                (analysis, key, row["coverage"], row["fcr"], row["determinate"], row["false"])
            )
    return stream.getvalue().encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise S9DiagnosticError("S9 output already exists")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = analyze(args.results, args.calibration)
    write_bytes_fail_safe(args.output_dir / "diagnostics-v1.json", canonical_json_bytes(payload))
    write_bytes_fail_safe(args.output_dir / "figure-source-data-v1.csv", _source_csv(payload))
    print(
        json.dumps({key: payload[key] for key in ("status", "row_count", "paired_tests")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
