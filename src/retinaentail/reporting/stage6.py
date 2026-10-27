"""Stage 6 complete-package verifier and synthetic pilot reporting.

The JSON and HTML emitted here are descriptive receipts.  They deliberately do
not serialize process capabilities.  A consumer must verify the complete Stage 6
package again, which in turn replays the complete Stage 2, Stage 3, and Stage 5
packages in the consuming process.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.pilot.stage6_gate import Stage6PreflightReport
from retinaentail.witnesses.replay import write_bytes_fail_safe

STAGE6_REPORT_SCHEMA_VERSION: Final = "6.0.0"
STAGE6_SCOPE: Final = "SYNTHETIC_ONLY"
STAGE6_METHOD: Final = "complete_package_capabilities_locked_pilot_v1"
STAGE6_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE6_PACKAGE"
STAGE6_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TOKEN: Final = object()
_ISSUED_REPORTS: dict[int, tuple[object, str]] = {}

_STAGE2_KEYS: Final = {
    "scope",
    "dataset_id",
    "dataset_version",
    "audit_status",
    "record_count",
    "manifest_sha256",
    "package_sha256",
    "package_file_count",
    "replay_verified",
    "capability_issued",
}
_STAGE3_KEYS: Final = {
    "scope",
    "package_sha256",
    "package_file_count",
    "calibration_artifact_sha256",
    "config_sha256",
    "data_sha256",
    "model_sha256",
    "alpha",
    "positive_threshold",
    "negative_threshold",
    "positive_sample_count",
    "negative_sample_count",
    "total_sample_count",
    "replay_verified",
    "capability_issued",
}
_STAGE5_KEYS: Final = {
    "scope",
    "pilot_id",
    "package_sha256",
    "package_file_count",
    "case_count",
    "replay_verified",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6PilotReport:
    """Issuer-bound report for the offline Stage 6 verification fixture."""

    run_id: str
    stage2_verifier: Mapping[str, object]
    stage3_verifier: Mapping[str, object]
    stage5_source: Mapping[str, object]
    pilot_audit: Mapping[str, object]
    stage6_config_sha256: str
    stage6_package_payload_sha256: str
    stage6_model_provenance_sha256: str
    schema_version: str = STAGE6_REPORT_SCHEMA_VERSION
    scope: str = STAGE6_SCOPE
    method: str = STAGE6_METHOD
    source_digest_method: str = STAGE6_DIGEST_METHOD
    thresholds_locked_before_pilot: bool = True
    thresholds_reestimated: bool = False
    test_split_used_for_calibration: bool = False
    test_tuning_prohibited: bool = True
    process_bound_capabilities_required: bool = True
    capabilities_serialized: bool = False
    detached_summaries_authoritative: bool = False
    synthetic_capabilities_authorize_real_data: bool = False
    synthetic_components_same_dataset_provenance: bool = False
    real_authorization_issued: bool = False
    dataset_root_probed: bool = False
    real_data_used: bool = False
    real_data_read_count: int = 0
    cpu_only: bool = True
    gpu_used: bool = False
    network_used: bool = False
    clinical_claim: bool = False
    individual_report_authoritative: bool = False
    authority_boundary: str = STAGE6_AUTHORITY_BOUNDARY
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        issued = _ISSUED_REPORTS.get(id(self))
        new_issue = _token is _REPORT_TOKEN
        if not new_issue and (issued is None or issued[0] is not self):
            raise TypeError("Stage6PilotReport must be created by its issuer")
        if self.schema_version != STAGE6_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 6 report schema version")
        if self.scope != STAGE6_SCOPE or self.method != STAGE6_METHOD:
            raise ValueError("Stage 6 fixture reports must retain synthetic scope and method")
        if not _identifier(self.run_id):
            raise ValueError("run_id must be a safe non-empty identifier")
        if self.source_digest_method != STAGE6_DIGEST_METHOD:
            raise ValueError("unsupported Stage 6 source digest method")
        for name in (
            "stage6_config_sha256",
            "stage6_package_payload_sha256",
            "stage6_model_provenance_sha256",
        ):
            _digest(getattr(self, name), name)

        required_flags = {
            "thresholds_locked_before_pilot": True,
            "thresholds_reestimated": False,
            "test_split_used_for_calibration": False,
            "test_tuning_prohibited": True,
            "process_bound_capabilities_required": True,
            "capabilities_serialized": False,
            "detached_summaries_authoritative": False,
            "synthetic_capabilities_authorize_real_data": False,
            "synthetic_components_same_dataset_provenance": False,
            "real_authorization_issued": False,
            "dataset_root_probed": False,
            "real_data_used": False,
            "cpu_only": True,
            "gpu_used": False,
            "network_used": False,
            "clinical_claim": False,
            "individual_report_authoritative": False,
        }
        if any(getattr(self, name) is not expected for name, expected in required_flags.items()):
            raise ValueError("Stage 6 safety flags are inconsistent")
        if self.real_data_read_count != 0:
            raise ValueError("the Stage 6 synthetic fixture cannot report real-data reads")
        if self.authority_boundary != STAGE6_AUTHORITY_BOUNDARY:
            raise ValueError("only a complete verified Stage 6 package is authoritative")

        stage2 = _json_mapping(self.stage2_verifier, "stage2_verifier")
        stage3 = _json_mapping(self.stage3_verifier, "stage3_verifier")
        stage5 = _json_mapping(self.stage5_source, "stage5_source")
        audit = _json_mapping(self.pilot_audit, "pilot_audit")
        _validate_stage2(stage2)
        _validate_stage3(stage3)
        _validate_stage5(stage5)
        _validate_pilot_audit(audit, stage3=stage3, stage5=stage5)
        object.__setattr__(self, "stage2_verifier", MappingProxyType(stage2))
        object.__setattr__(self, "stage3_verifier", MappingProxyType(stage3))
        object.__setattr__(self, "stage5_source", MappingProxyType(stage5))
        object.__setattr__(self, "pilot_audit", MappingProxyType(audit))

        fingerprint = _canonical(self._payload())
        if new_issue:
            _ISSUED_REPORTS[id(self)] = (self, fingerprint)
        elif issued is None or issued[1] != fingerprint:
            raise ValueError("issued Stage 6 report changed after issuance")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "method": self.method,
            "run_id": self.run_id,
            "source_digest_method": self.source_digest_method,
            "stage2_verifier": _json_copy(dict(self.stage2_verifier)),
            "stage3_verifier": _json_copy(dict(self.stage3_verifier)),
            "stage5_source": _json_copy(dict(self.stage5_source)),
            "pilot_audit": _json_copy(dict(self.pilot_audit)),
            "stage6_config_sha256": self.stage6_config_sha256,
            "stage6_package_payload_sha256": self.stage6_package_payload_sha256,
            "stage6_model_provenance_sha256": self.stage6_model_provenance_sha256,
            "thresholds_locked_before_pilot": self.thresholds_locked_before_pilot,
            "thresholds_reestimated": self.thresholds_reestimated,
            "test_split_used_for_calibration": self.test_split_used_for_calibration,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "process_bound_capabilities_required": self.process_bound_capabilities_required,
            "capabilities_serialized": self.capabilities_serialized,
            "detached_summaries_authoritative": self.detached_summaries_authoritative,
            "synthetic_capabilities_authorize_real_data": (
                self.synthetic_capabilities_authorize_real_data
            ),
            "synthetic_components_same_dataset_provenance": (
                self.synthetic_components_same_dataset_provenance
            ),
            "real_authorization_issued": self.real_authorization_issued,
            "dataset_root_probed": self.dataset_root_probed,
            "real_data_used": self.real_data_used,
            "real_data_read_count": self.real_data_read_count,
            "cpu_only": self.cpu_only,
            "gpu_used": self.gpu_used,
            "network_used": self.network_used,
            "clinical_claim": self.clinical_claim,
            "individual_report_authoritative": self.individual_report_authoritative,
            "authority_boundary": self.authority_boundary,
        }

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return self._payload()


def issue_stage6_pilot_report(
    *,
    run_id: str,
    stage2_verifier: Mapping[str, object],
    stage3_verifier: Mapping[str, object],
    stage5_source: Mapping[str, object],
    pilot_audit: Mapping[str, object],
    stage6_config_sha256: str,
    stage6_package_payload_sha256: str,
    stage6_model_provenance_sha256: str,
) -> Stage6PilotReport:
    """Issue a synthetic report after all complete-package replays passed."""

    return Stage6PilotReport(
        run_id=run_id,
        stage2_verifier=stage2_verifier,
        stage3_verifier=stage3_verifier,
        stage5_source=stage5_source,
        pilot_audit=pilot_audit,
        stage6_config_sha256=stage6_config_sha256,
        stage6_package_payload_sha256=stage6_package_payload_sha256,
        stage6_model_provenance_sha256=stage6_model_provenance_sha256,
        _token=_REPORT_TOKEN,
    )


def render_stage6_pilot_json(report: Stage6PilotReport) -> str:
    report = _require_report(report)
    return (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def render_stage6_pilot_html(report: Stage6PilotReport) -> str:
    report = _require_report(report)
    audit = report.pilot_audit
    evaluation = cast(Mapping[str, object], audit["evaluation"])
    state_counts = cast(Mapping[str, int], evaluation["state_counts"])
    state_intervals = cast(Mapping[str, Mapping[str, object]], evaluation["state_proportions"])
    rows: list[str] = []
    for state in EntailmentState:
        interval = state_intervals[state.value]
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(state.value)}</th>'
            f"<td>{state_counts[state.value]}</td>"
            f"<td>{escape(_interval_text(interval))}</td>"
            "</tr>"
        )
    stage2 = report.stage2_verifier
    stage3 = report.stage3_verifier
    stage5 = report.stage5_source
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 6 verifier pilot</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 6 complete-package verifier pilot</h1>
  <p><strong>Synthetic fixture only. No real dataset root was probed or read.</strong></p>
  <p>Run: <code>{escape(report.run_id, quote=True)}</code></p>
  <p>Stage 2 package: <code>{escape(cast(str, stage2["package_sha256"]))}</code>;
     Stage 3 package: <code>{escape(cast(str, stage3["package_sha256"]))}</code>;
     Stage 5 package: <code>{escape(cast(str, stage5["package_sha256"]))}</code>.</p>
  <p>Capabilities were regenerated in this process and were not serialized.
     Detached summaries are not authoritative.</p>
  <p>The independent synthetic components do not establish same-dataset real provenance and
     cannot authorize a real root.</p>
  <h2>Locked pilot four-state summary</h2>
  <table><thead><tr><th>State</th><th>Count</th><th>95% Wilson interval</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table>
  <h2>Failure and witness replay</h2>
  <p>Fail closed: {evaluation["fail_closed_count"]}; OOD:
     {evaluation["out_of_distribution_count"]}; dual witness replay passed:
     {evaluation["both_witness_replays_passed_count"]} / {evaluation["test_case_count"]}.</p>
  <p>Thresholds re-estimated on test: no. Test tuning: prohibited.</p>
  <p>Authority boundary: <code>{escape(report.authority_boundary)}</code>.</p>
</body>
</html>
"""


def render_stage6_preflight_json(report: Stage6PreflightReport) -> str:
    report = _require_preflight(report)
    return (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def render_stage6_preflight_html(report: Stage6PreflightReport) -> str:
    report = _require_preflight(report)
    reasons = "".join(f"<li><code>{escape(reason.value)}</code></li>" for reason in report.reasons)
    if not reasons:
        reasons = (
            "<li>Presence checks are ready; full in-process verification is still required.</li>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 6 preflight</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 6 zero-I/O preflight</h1>
  <p>Status: <strong>{escape(report.status.value)}</strong>.</p>
  <p>Dataset: <code>{escape(report.dataset_id or "not supplied")}</code>;
     version: <code>{escape(report.dataset_version or "not supplied")}</code>.</p>
  <p><strong>This report accepts no paths, probes no root, and authorizes no file open.</strong></p>
  <h2>Reasons</h2><ul>{reasons}</ul>
</body>
</html>
"""


def write_stage6_preflight_reports(
    report: Stage6PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    report = _require_preflight(report)
    if not isinstance(output_dir, Path) or not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("Stage 6 output and boundary types are invalid")
    json_path = boundary.require_safe(output_dir / "stage6-preflight.json")
    html_path = boundary.require_safe(output_dir / "stage6-preflight.html")
    payloads = (
        (json_path, render_stage6_preflight_json(report).encode("utf-8")),
        (html_path, render_stage6_preflight_html(report).encode("utf-8")),
    )
    for path, payload in payloads:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    for path, payload in payloads:
        write_bytes_fail_safe(path, payload)
    return json_path, html_path


def _validate_stage2(value: Mapping[str, object]) -> None:
    _keys(value, _STAGE2_KEYS, "stage2_verifier")
    if value["scope"] != "SYNTHETIC_FIXTURE" or value["audit_status"] != "PASS":
        raise ValueError("Stage 6 fixture requires a verified synthetic PASS Stage 2 package")
    for name in ("dataset_id", "dataset_version"):
        if not _identifier(value[name]):
            raise ValueError(f"stage2_verifier.{name} is invalid")
    _positive_int(value["record_count"], "stage2 record_count")
    _positive_int(value["package_file_count"], "stage2 package_file_count")
    for name in ("manifest_sha256", "package_sha256"):
        _digest(value[name], f"stage2_verifier.{name}")
    if value["replay_verified"] is not True or value["capability_issued"] is not True:
        raise ValueError("Stage 2 capability must have been issued by a complete replay")


def _validate_stage3(value: Mapping[str, object]) -> None:
    _keys(value, _STAGE3_KEYS, "stage3_verifier")
    if value["scope"] != "SYNTHETIC_TOY":
        raise ValueError("Stage 6 fixture requires synthetic Stage 3 scope")
    _positive_int(value["package_file_count"], "stage3 package_file_count")
    for name in (
        "package_sha256",
        "calibration_artifact_sha256",
        "config_sha256",
        "data_sha256",
        "model_sha256",
    ):
        _digest(value[name], f"stage3_verifier.{name}")
    alpha = _finite_number(value["alpha"], "stage3 alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Stage 3 alpha must lie strictly between zero and one")
    for name in ("positive_threshold", "negative_threshold"):
        if _finite_number(value[name], f"stage3 {name}") < 0.0:
            raise ValueError("Stage 3 thresholds must be non-negative")
    positive = _positive_int(value["positive_sample_count"], "positive_sample_count")
    negative = _positive_int(value["negative_sample_count"], "negative_sample_count")
    total = _positive_int(value["total_sample_count"], "total_sample_count")
    if total != positive + negative:
        raise ValueError("Stage 3 calibration sample counts are inconsistent")
    if value["replay_verified"] is not True or value["capability_issued"] is not True:
        raise ValueError("Stage 3 capability must have been issued by a complete replay")


def _validate_stage5(value: Mapping[str, object]) -> None:
    _keys(value, _STAGE5_KEYS, "stage5_source")
    if value["scope"] != "SYNTHETIC_ONLY" or not _identifier(value["pilot_id"]):
        raise ValueError("Stage 5 source scope or pilot ID is invalid")
    _digest(value["package_sha256"], "stage5_source.package_sha256")
    _positive_int(value["package_file_count"], "stage5 package_file_count")
    _positive_int(value["case_count"], "stage5 case_count")
    if value["replay_verified"] is not True:
        raise ValueError("Stage 5 source must be completely replay-verified")


def _validate_pilot_audit(
    value: Mapping[str, object],
    *,
    stage3: Mapping[str, object],
    stage5: Mapping[str, object],
) -> None:
    if (
        value.get("schema_version") != "5.0.0"
        or value.get("scope") != "SYNTHETIC_ONLY"
        or value.get("authority_boundary") != "COMPLETE_VERIFIED_STAGE5_PACKAGE"
        or value.get("clinical_claim") is not False
        or value.get("real_data_used") is not False
        or value.get("thresholds_reestimated") is not False
        or value.get("test_split_used_for_calibration") is not False
        or value.get("test_tuning_prohibited") is not True
    ):
        raise ValueError("embedded Stage 5 audit violates its locked synthetic contract")
    if value.get("pilot_id") != stage5["pilot_id"]:
        raise ValueError("Stage 5 source descriptor does not match its audit")
    evaluation = _mapping(value.get("evaluation"), "pilot_audit.evaluation")
    cases = value.get("case_index")
    if not isinstance(cases, list) or len(cases) != stage5["case_count"]:
        raise ValueError("Stage 5 case index does not match the source descriptor")
    locked = _mapping(value.get("locked_contract"), "pilot_audit.locked_contract")
    cross_bindings = {
        "alpha": "alpha",
        "positive_threshold": "positive_threshold",
        "negative_threshold": "negative_threshold",
        "positive_sample_count": "positive_sample_count",
        "negative_sample_count": "negative_sample_count",
        "total_sample_count": "total_sample_count",
        "calibration_artifact_hash": "calibration_artifact_sha256",
        "config_hash": "config_sha256",
        "data_hash": "data_sha256",
        "model_hash": "model_sha256",
    }
    for locked_name, stage3_name in cross_bindings.items():
        if locked.get(locked_name) != stage3[stage3_name]:
            raise ValueError("Stage 3 capability does not match the locked Stage 5 contract")
    test_count = _positive_int(evaluation.get("test_case_count"), "test_case_count")
    if test_count != len(cases):
        raise ValueError("Stage 5 evaluation count does not match its case index")
    state_counts = _mapping(evaluation.get("state_counts"), "state_counts")
    proportions = _mapping(evaluation.get("state_proportions"), "state_proportions")
    confidence = _finite_number(evaluation.get("confidence_level"), "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    recomputed = {state.value: 0 for state in EntailmentState}
    for item in cases:
        case = _mapping(item, "case_index item")
        state = case.get("state")
        if state not in recomputed:
            raise ValueError("case index contains an unknown four-state value")
        recomputed[state] += 1
    if state_counts != recomputed:
        raise ValueError("Stage 5 four-state counts do not match the case index")
    for state, count in recomputed.items():
        expected = wilson_score_proportion(count, test_count, confidence_level=confidence).to_dict()
        if proportions.get(state) != expected:
            raise ValueError("Stage 5 Wilson interval does not match its case count")


def _interval_text(value: Mapping[str, object]) -> str:
    estimate = value.get("estimate")
    lower = value.get("lower")
    upper = value.get("upper")
    if not all(
        isinstance(item, int | float) and not isinstance(item, bool)
        for item in (estimate, lower, upper)
    ):
        return "undefined"
    estimate_text = f"{float(cast(float, estimate)):.6f}"
    lower_text = f"{float(cast(float, lower)):.6f}"
    upper_text = f"{float(cast(float, upper)):.6f}"
    return f"{estimate_text} [{lower_text}, {upper_text}]"


def _require_report(value: object) -> Stage6PilotReport:
    if not isinstance(value, Stage6PilotReport):
        raise TypeError("report must be a Stage6PilotReport")
    value.__post_init__()
    return value


def _require_preflight(value: object) -> Stage6PreflightReport:
    if not isinstance(value, Stage6PreflightReport):
        raise TypeError("report must be a Stage6PreflightReport")
    value.__post_init__()
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are incomplete or unknown")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(cast(Mapping[str, object], value))


def _json_mapping(value: object, name: str) -> dict[str, object]:
    serializable = dict(value) if isinstance(value, Mapping) else value
    copied = _json_copy(serializable)
    if not isinstance(copied, dict) or any(not isinstance(key, str) for key in copied):
        raise ValueError(f"{name} must be a finite JSON object")
    return cast(dict[str, object], copied)


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 6 report values must be finite JSON") from error


def _canonical(value: object) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and not any(character in value for character in "\r\n\x00")
    )


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "STAGE6_AUTHORITY_BOUNDARY",
    "STAGE6_DIGEST_METHOD",
    "STAGE6_METHOD",
    "STAGE6_REPORT_SCHEMA_VERSION",
    "STAGE6_SCOPE",
    "Stage6PilotReport",
    "issue_stage6_pilot_report",
    "render_stage6_pilot_html",
    "render_stage6_pilot_json",
    "render_stage6_preflight_html",
    "render_stage6_preflight_json",
    "write_stage6_preflight_reports",
]
