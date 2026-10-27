"""Deterministic reporting for the Stage 5 locked small-pilot protocol.

Stage 5 reports are descriptive safety-audit artifacts.  They are issued from an
already verified Stage 4 package, carry its exact package digest, and cannot make a
clinical claim.  The complete verified package is the authority boundary; an
individual JSON or HTML file is not independently authoritative.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from retinaentail.calibration.protocol import EvaluationFailureReason
from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.pilot.gate import Stage5PreflightReport
from retinaentail.pilot.identity import (
    Stage5IdentityAudit,
    Stage5IdentityAuditStatus,
    audit_five_identity_split_isolation,
)
from retinaentail.witnesses.replay import write_bytes_fail_safe

STAGE5_PILOT_REPORT_SCHEMA_VERSION: Final = "5.0.0"
STAGE5_PILOT_SCOPE: Final = "SYNTHETIC_ONLY"
STAGE5_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE5_PACKAGE"
STAGE5_SOURCE_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_REASON_VALUES: Final = frozenset(item.value for item in EvaluationFailureReason)
_REPORT_TOKEN: Final = object()
_ISSUED_REPORTS: dict[int, tuple[object, str]] = {}


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5PilotReport:
    """Immutable, issuer-bound view of one synthetic locked-pilot audit."""

    pilot_id: str
    source_stage4_package_sha256: str
    source_stage4_file_count: int
    stage5_config_sha256: str
    pilot_data_sha256: str
    stage5_model_provenance_sha256: str
    locked_contract: Mapping[str, object]
    identity_isolation: Mapping[str, object]
    evaluation: Mapping[str, object]
    case_index: tuple[Mapping[str, object], ...]
    schema_version: str = STAGE5_PILOT_REPORT_SCHEMA_VERSION
    scope: str = STAGE5_PILOT_SCOPE
    split: str = "test"
    source_digest_method: str = STAGE5_SOURCE_DIGEST_METHOD
    thresholds_locked_before_pilot: bool = True
    thresholds_reestimated: bool = False
    test_split_used_for_calibration: bool = False
    test_tuning_prohibited: bool = True
    cpu_only: bool = True
    gpu_used: bool = False
    network_used: bool = False
    real_data_used: bool = False
    real_data_read_count: int = 0
    clinical_claim: bool = False
    individual_report_authoritative: bool = False
    authority_boundary: str = STAGE5_AUTHORITY_BOUNDARY
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        issued = _ISSUED_REPORTS.get(id(self))
        new_issue = _token is _REPORT_TOKEN
        if not new_issue and (issued is None or issued[0] is not self):
            raise TypeError("Stage5PilotReport can only be issued by issue_stage5_pilot_report")
        if self.schema_version != STAGE5_PILOT_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 5 report schema version")
        if self.scope != STAGE5_PILOT_SCOPE or self.split != "test":
            raise ValueError("Stage 5 toy pilot reports are synthetic test-only artifacts")
        if not _valid_identifier(self.pilot_id):
            raise ValueError("pilot_id must be a canonical non-empty identifier")
        for name in (
            "source_stage4_package_sha256",
            "stage5_config_sha256",
            "pilot_data_sha256",
            "stage5_model_provenance_sha256",
        ):
            if _SHA256_PATTERN.fullmatch(cast(str, getattr(self, name))) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.source_digest_method != STAGE5_SOURCE_DIGEST_METHOD:
            raise ValueError("unsupported source Stage 4 package digest method")
        if (
            isinstance(self.source_stage4_file_count, bool)
            or not isinstance(self.source_stage4_file_count, int)
            or self.source_stage4_file_count < 1
        ):
            raise ValueError("source_stage4_file_count must be a positive integer")
        constant_flags = {
            "thresholds_locked_before_pilot": True,
            "thresholds_reestimated": False,
            "test_split_used_for_calibration": False,
            "test_tuning_prohibited": True,
            "cpu_only": True,
            "gpu_used": False,
            "network_used": False,
            "real_data_used": False,
            "clinical_claim": False,
            "individual_report_authoritative": False,
        }
        if any(getattr(self, name) is not expected for name, expected in constant_flags.items()):
            raise ValueError("Stage 5 locked-pilot safety flags are inconsistent")
        if self.real_data_read_count != 0:
            raise ValueError("the Stage 5 toy pilot must not read real data")
        if self.authority_boundary != STAGE5_AUTHORITY_BOUNDARY:
            raise ValueError("only the complete verified Stage 5 package is authoritative")

        locked = _json_mapping(self.locked_contract, "locked_contract")
        identity = _json_mapping(self.identity_isolation, "identity_isolation")
        evaluation = _json_mapping(self.evaluation, "evaluation")
        cases = tuple(_json_mapping(item, "case_index item") for item in self.case_index)
        _validate_locked_contract(locked)
        _validate_identity(identity)
        _validate_evaluation(evaluation, locked, cases)
        object.__setattr__(self, "locked_contract", MappingProxyType(locked))
        object.__setattr__(self, "identity_isolation", MappingProxyType(identity))
        object.__setattr__(self, "evaluation", MappingProxyType(evaluation))
        object.__setattr__(
            self,
            "case_index",
            tuple(MappingProxyType(item) for item in cases),
        )

        fingerprint = _canonical_compact(self._payload_dict())
        if new_issue:
            _ISSUED_REPORTS[id(self)] = (self, fingerprint)
        elif issued is None or issued[1] != fingerprint:
            raise ValueError("issued Stage 5 report bindings changed after issuance")

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "split": self.split,
            "pilot_id": self.pilot_id,
            "source_stage4_package_sha256": self.source_stage4_package_sha256,
            "source_stage4_file_count": self.source_stage4_file_count,
            "source_digest_method": self.source_digest_method,
            "stage5_config_sha256": self.stage5_config_sha256,
            "pilot_data_sha256": self.pilot_data_sha256,
            "stage5_model_provenance_sha256": self.stage5_model_provenance_sha256,
            "locked_contract": _json_copy(dict(self.locked_contract)),
            "identity_isolation": _json_copy(dict(self.identity_isolation)),
            "evaluation": _json_copy(dict(self.evaluation)),
            "case_index": [_json_copy(dict(item)) for item in self.case_index],
            "positive_witness_replays_passed_count": sum(
                item["positive_witness_replay_passed"] is True for item in self.case_index
            ),
            "negative_witness_replays_passed_count": sum(
                item["negative_witness_replay_passed"] is True for item in self.case_index
            ),
            "thresholds_locked_before_pilot": self.thresholds_locked_before_pilot,
            "thresholds_reestimated": self.thresholds_reestimated,
            "test_split_used_for_calibration": self.test_split_used_for_calibration,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "cpu_only": self.cpu_only,
            "gpu_used": self.gpu_used,
            "network_used": self.network_used,
            "real_data_used": self.real_data_used,
            "real_data_read_count": self.real_data_read_count,
            "clinical_claim": self.clinical_claim,
            "individual_report_authoritative": self.individual_report_authoritative,
            "authority_boundary": self.authority_boundary,
        }

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return self._payload_dict()


def issue_stage5_pilot_report(
    *,
    pilot_id: str,
    source_stage4_package_sha256: str,
    source_stage4_file_count: int,
    stage5_config_sha256: str,
    pilot_data_sha256: str,
    stage5_model_provenance_sha256: str,
    locked_contract: Mapping[str, object],
    identity_isolation: Mapping[str, object],
    identity_records: Sequence[object],
    evaluation: Mapping[str, object],
    case_index: tuple[Mapping[str, object], ...],
) -> Stage5PilotReport:
    """Issue a Stage 5 report after strict structural and cross-field validation."""

    if isinstance(identity_records, str | bytes) or not isinstance(identity_records, Sequence):
        raise TypeError("identity_records must be a non-string sequence")
    recomputed_identity = audit_five_identity_split_isolation(tuple(identity_records))
    try:
        supplied_identity = Stage5IdentityAudit.from_dict(identity_isolation)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "identity_isolation must be a strict Stage5IdentityAudit payload"
        ) from error
    if supplied_identity.to_dict() != recomputed_identity.to_dict():
        raise ValueError("identity_isolation does not match the supplied identity records")
    return Stage5PilotReport(
        pilot_id=pilot_id,
        source_stage4_package_sha256=source_stage4_package_sha256,
        source_stage4_file_count=source_stage4_file_count,
        stage5_config_sha256=stage5_config_sha256,
        pilot_data_sha256=pilot_data_sha256,
        stage5_model_provenance_sha256=stage5_model_provenance_sha256,
        locked_contract=locked_contract,
        identity_isolation=supplied_identity.to_dict(),
        evaluation=evaluation,
        case_index=case_index,
        _token=_REPORT_TOKEN,
    )


def render_stage5_pilot_json(report: Stage5PilotReport) -> str:
    report = _require_report(report)
    return (
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_stage5_pilot_html(report: Stage5PilotReport) -> str:
    report = _require_report(report)
    evaluation = report.evaluation
    state_counts = cast(Mapping[str, object], evaluation["state_counts"])
    intervals = cast(Mapping[str, object], evaluation["state_proportions"])
    state_rows: list[str] = []
    for state in EntailmentState:
        interval = cast(Mapping[str, object], intervals[state.value])
        estimate = cast(float | None, interval["estimate"])
        lower = cast(float | None, interval["lower"])
        upper = cast(float | None, interval["upper"])
        interval_text = (
            "undefined"
            if estimate is None or lower is None or upper is None
            else f"{estimate:.6f} [{lower:.6f}, {upper:.6f}]"
        )
        state_rows.append(
            "<tr>"
            f'<th scope="row">{escape(state.value)}</th>'
            f"<td>{escape(str(state_counts[state.value]))}</td>"
            f"<td>{escape(interval_text)}</td>"
            "</tr>"
        )
    case_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(item['case_id']), quote=True)}</td>"
        f"<td>{escape(str(item['state']), quote=True)}</td>"
        f"<td>{escape(str(item['status']), quote=True)}</td>"
        f"<td>{escape(str(item['dual_witness_replay_passed']), quote=True)}</td>"
        "</tr>"
        for item in report.case_index
    )
    positive_replays_passed = sum(
        item["positive_witness_replay_passed"] is True for item in report.case_index
    )
    negative_replays_passed = sum(
        item["negative_witness_replay_passed"] is True for item in report.case_index
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 5 locked pilot</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 5 locked small-pilot audit</h1>
  <p><strong>Synthetic fixture only; this is not a clinical claim.</strong></p>
  <p>Pilot: <code>{escape(report.pilot_id, quote=True)}</code></p>
  <p>Source Stage 4 package: <code>{escape(report.source_stage4_package_sha256)}</code></p>
  <p>Authority boundary: <code>{escape(report.authority_boundary)}</code></p>
  <h2>Four-state summary and 95% Wilson intervals</h2>
  <table><thead><tr><th>State</th><th>Count</th><th>Proportion</th></tr></thead>
  <tbody>{"".join(state_rows)}</tbody></table>
  <h2>Fail-closed and OOD</h2>
  <p>Fail closed: {escape(str(evaluation["fail_closed_count"]))};
     OOD: {escape(str(evaluation["out_of_distribution_count"]))};
     positive replay passed: {positive_replays_passed};
     negative replay passed: {negative_replays_passed};
     dual replay passed: {escape(str(evaluation["both_witness_replays_passed_count"]))}.</p>
  <h2>Case replay index</h2>
  <table><thead><tr><th>Case</th><th>State</th><th>Status</th><th>Dual replay</th></tr></thead>
  <tbody>{case_rows}</tbody></table>
</body>
</html>
"""


def render_stage5_preflight_json(report: Stage5PreflightReport) -> str:
    report = _require_preflight(report)
    return (
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_stage5_preflight_html(report: Stage5PreflightReport) -> str:
    report = _require_preflight(report)
    payload = report.to_dict()
    checks = (
        ("Approved review", report.approved_review_provided),
        ("Dataset root declaration", report.dataset_root_provided),
        ("Versioned adapter spec", report.adapter_spec_provided),
        ("Stage 2 PASS audit", report.stage2_pass_audit_provided),
        ("APPROVED_REAL_DATA frozen card", report.frozen_data_card_provided),
        ("Valid Stage 2 access permit", report.stage2_access_permit_provided),
        ("Manual identity/label confirmation", report.manual_confirmation_provided),
        ("Verified Stage 3 package summary", report.stage3_package_summary_provided),
        ("Stage 4 locked-evidence summary", report.stage4_locked_evidence_provided),
    )
    check_rows = "\n".join(
        "<tr>"
        f'<th scope="row">{escape(label, quote=True)}</th>'
        f"<td>{'provided' if present else 'missing'}</td>"
        "</tr>"
        for label, present in checks
    )
    reasons = report.reasons
    reason_items = "".join(f"<li><code>{escape(item.value)}</code></li>" for item in reasons)
    if not reason_items:
        reason_items = (
            "<li>Presence checks complete; prerequisite verification is still required.</li>"
        )
    purposes = ", ".join(item.value for item in report.approved_review_purposes) or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 5 preflight</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 5 prerequisite preflight</h1>
  <p>Status: <strong>{escape(report.status.value)}</strong></p>
  <p>Context: <code>{escape(report.verification_context.value)}</code></p>
  <p>Dataset: <code>{escape(report.dataset_id or "not supplied")}</code>;
     version: <code>{escape(report.dataset_version or "not supplied")}</code>.</p>
  <p>Approved purposes: <code>{escape(purposes)}</code>.</p>
  <p><strong>This presence-only report never authorizes data access.</strong></p>
  <table><tbody>{check_rows}</tbody></table>
  <h2>Reasons</h2><ul>{reason_items}</ul>
  <p>Dataset root probed: {escape(str(payload["dataset_root_probed"]))};
     real data read: {escape(str(payload["real_data_read"]))};
     sample count: {escape(str(payload["sample_count"]))}.</p>
</body>
</html>
"""


def write_stage5_pilot_reports(
    report: Stage5PilotReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    report = _require_report(report)
    json_path = boundary.require_safe(output_dir / "stage5-pilot.json")
    html_path = boundary.require_safe(output_dir / "stage5-pilot.html")
    payloads = (
        (json_path, render_stage5_pilot_json(report).encode("utf-8")),
        (html_path, render_stage5_pilot_html(report).encode("utf-8")),
    )
    for path, payload in payloads:
        _preflight(path, payload)
    for path, payload in payloads:
        write_bytes_fail_safe(path, payload)
    return json_path, html_path


def write_stage5_preflight_reports(
    report: Stage5PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    report = _require_preflight(report)
    json_path = boundary.require_safe(output_dir / "stage5-preflight.json")
    html_path = boundary.require_safe(output_dir / "stage5-preflight.html")
    payloads = (
        (json_path, render_stage5_preflight_json(report).encode("utf-8")),
        (html_path, render_stage5_preflight_html(report).encode("utf-8")),
    )
    for path, payload in payloads:
        _preflight(path, payload)
    for path, payload in payloads:
        write_bytes_fail_safe(path, payload)
    return json_path, html_path


def _validate_locked_contract(value: Mapping[str, object]) -> None:
    required = {
        "alpha",
        "positive_threshold",
        "negative_threshold",
        "positive_sample_count",
        "negative_sample_count",
        "total_sample_count",
        "calibration_artifact_hash",
        "config_hash",
        "data_hash",
        "model_hash",
    }
    if set(value) != required:
        raise ValueError("locked_contract keys are incomplete or unknown")
    for name in ("alpha", "positive_threshold", "negative_threshold"):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            raise ValueError(f"locked_contract.{name} must be finite")
    alpha = float(cast(float, value["alpha"]))
    if not 0.0 < alpha < 1.0:
        raise ValueError("locked alpha must lie strictly between zero and one")
    if (
        float(cast(float, value["positive_threshold"])) < 0.0
        or float(cast(float, value["negative_threshold"])) < 0.0
    ):
        raise ValueError("locked thresholds must be non-negative")
    counts = [value[name] for name in ("positive_sample_count", "negative_sample_count")]
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in counts):
        raise ValueError("locked per-class calibration counts must be positive integers")
    total = value["total_sample_count"]
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total != sum(cast(list[int], counts))
    ):
        raise ValueError("locked total calibration count is inconsistent")
    for name in ("calibration_artifact_hash", "config_hash", "data_hash", "model_hash"):
        raw = value[name]
        if not isinstance(raw, str) or _SHA256_PATTERN.fullmatch(raw) is None:
            raise ValueError(f"locked_contract.{name} must be a lowercase SHA-256 digest")


def _validate_identity(value: Mapping[str, object]) -> None:
    try:
        audit = Stage5IdentityAudit.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 5 five-identity isolation payload is invalid") from error
    if audit.status is not Stage5IdentityAuditStatus.PASS or not audit.isolation_proven:
        raise ValueError("Stage 5 five-identity isolation must pass")
    if audit.reasons or audit.issues:
        raise ValueError("a passing Stage 5 identity audit cannot contain failures")
    if audit.record_count < 1:
        raise ValueError("identity audit record_count must be positive")
    if any(audit.unknown_counts.values()):
        raise ValueError("the synthetic five-identity fixture cannot contain unknown identities")


def _validate_evaluation(
    value: Mapping[str, object],
    locked: Mapping[str, object],
    cases: tuple[Mapping[str, object], ...],
) -> None:
    if (
        value.get("schema_version") != "4.0.0"
        or value.get("scope") != "SYNTHETIC_ONLY"
        or value.get("split") != "test"
        or value.get("clinical_claim") is not False
        or value.get("status") != "SUCCESS"
    ):
        raise ValueError("source Stage 4 evaluation scope or status is invalid")
    if value.get("locked_contract") != locked:
        raise ValueError("Stage 4 evaluation does not match the Stage 5 locked contract")
    test_count = value.get("test_case_count")
    if isinstance(test_count, bool) or not isinstance(test_count, int) or test_count < 1:
        raise ValueError("Stage 5 pilot requires a non-empty locked test set")
    if len(cases) != test_count:
        raise ValueError("case index does not cover every Stage 4 test case")
    case_ids = [item.get("case_id") for item in cases]
    if any(not _valid_identifier(item) for item in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("case index identifiers must be unique and canonical")
    expected_case_keys = {
        "pilot_index",
        "case_id",
        "reference_label",
        "state",
        "status",
        "failure_reasons",
        "dual_witness_replay_passed",
        "positive_witness_replay_passed",
        "negative_witness_replay_passed",
        "evaluation_sha256",
        "certificate_sha256",
        "dual_replay_sha256",
        "positive_witness_sha256",
        "negative_witness_sha256",
    }
    for index, item in enumerate(cases):
        if set(item) != expected_case_keys or item.get("pilot_index") != index:
            raise ValueError("case index shape or preregistered order is invalid")
        if item.get("state") not in {state.value for state in EntailmentState}:
            raise ValueError("case index contains an unknown entailment state")
        if item.get("status") not in {"SUCCESS", "FAIL_CLOSED"}:
            raise ValueError("case index contains an unknown evaluation status")
        if item.get("reference_label") not in {"LESION_PRESENT", "LESION_ABSENT", "UNKNOWN"}:
            raise ValueError("case index contains an unknown reference label")
        replay_flags = tuple(
            item.get(name)
            for name in (
                "dual_witness_replay_passed",
                "positive_witness_replay_passed",
                "negative_witness_replay_passed",
            )
        )
        if any(type(flag) is not bool for flag in replay_flags):  # noqa: E721
            raise ValueError("case index replay flags must be exact booleans")
        if replay_flags[0] is not (replay_flags[1] and replay_flags[2]):
            raise ValueError("case index dual replay must equal the conjunction of both sides")
        reasons = item.get("failure_reasons")
        if (
            not isinstance(reasons, list)
            or any(not _valid_identifier(reason) for reason in reasons)
            or len(reasons) != len(set(reasons))
        ):
            raise ValueError("case index failure reasons must be unique controlled strings")
        if any(reason not in _FAILURE_REASON_VALUES for reason in reasons):
            raise ValueError("case index contains an uncontrolled failure reason")
        if (item["status"] == "SUCCESS") != (not reasons):
            raise ValueError("case index status conflicts with failure reasons")
        if item["status"] == "SUCCESS" and replay_flags[0] is not True:
            raise ValueError("a successful case requires a passing dual witness replay")
        if item["status"] == "FAIL_CLOSED" and item["state"] != "OUT_OF_MODEL":
            raise ValueError("a fail-closed case must map externally to OUT_OF_MODEL")
        for name in (
            "evaluation_sha256",
            "certificate_sha256",
            "dual_replay_sha256",
            "positive_witness_sha256",
            "negative_witness_sha256",
        ):
            raw_digest = item.get(name)
            if not isinstance(raw_digest, str) or _SHA256_PATTERN.fullmatch(raw_digest) is None:
                raise ValueError("case index artifact hashes must be lowercase SHA-256 values")
    state_counts = value.get("state_counts")
    proportions = value.get("state_proportions")
    if not isinstance(state_counts, Mapping) or set(state_counts) != {
        state.value for state in EntailmentState
    }:
        raise ValueError("Stage 4 four-state counts are incomplete")
    if not isinstance(proportions, Mapping) or set(proportions) != set(state_counts):
        raise ValueError("Stage 4 state proportions are incomplete")
    confidence = value.get("confidence_level")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(confidence)
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("Stage 4 confidence level is invalid")
    counts: list[int] = []
    for state in EntailmentState:
        count = state_counts[state.value]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Stage 4 state counts must be non-negative integers")
        counts.append(count)
        expected = wilson_score_proportion(
            count,
            test_count,
            confidence_level=float(confidence),
        ).to_dict()
        if proportions[state.value] != expected:
            raise ValueError("Stage 4 Wilson interval does not match its state count")
    if sum(counts) != test_count:
        raise ValueError("Stage 4 four-state counts do not partition the pilot")
    case_state_counts = {state.value: 0 for state in EntailmentState}
    for item in cases:
        case_state_counts[cast(str, item["state"])] += 1
    if case_state_counts != dict(state_counts):
        raise ValueError("case index states do not match aggregate state counts")
    input_count = value.get("input_record_count")
    rejected_count = value.get("rejected_record_count")
    if input_count != test_count or rejected_count != 0:
        raise ValueError("verified Stage 4 source cannot contain rejected pilot records")
    reference_counts = value.get("reference_label_counts")
    expected_reference_counts = {
        "LESION_PRESENT": 0,
        "LESION_ABSENT": 0,
        "UNKNOWN": 0,
    }
    for item in cases:
        expected_reference_counts[cast(str, item["reference_label"])] += 1
    if reference_counts != expected_reference_counts:
        raise ValueError("Stage 4 reference-label counts do not match the bound case index")
    labeled_count = value.get("labeled_reference_count")
    if labeled_count != (
        expected_reference_counts["LESION_PRESENT"] + expected_reference_counts["LESION_ABSENT"]
    ):
        raise ValueError("Stage 4 labeled denominator must exclude UNKNOWN references")
    expected_failure_counts: dict[str, int] = {}
    for item in cases:
        for reason in cast(list[str], item["failure_reasons"]):
            expected_failure_counts[reason] = expected_failure_counts.get(reason, 0) + 1
    if value.get("failure_reason_counts") != dict(sorted(expected_failure_counts.items())):
        raise ValueError("Stage 4 failure-reason counts do not match the bound case index")
    if value.get("rejected_reason_counts") != {} or value.get("aggregation_failure_reasons") != []:
        raise ValueError("verified Stage 4 source cannot contain aggregation rejections")
    summary_counts = (
        "evaluable_count",
        "unevaluable_count",
        "fail_closed_count",
        "out_of_distribution_count",
        "both_witness_replays_passed_count",
        "one_or_more_witness_replays_failed_count",
    )
    for name in summary_counts:
        count_value = value.get(name)
        if (
            isinstance(count_value, bool)
            or not isinstance(count_value, int)
            or not 0 <= count_value <= test_count
        ):
            raise ValueError(f"Stage 4 {name} is invalid")
    numeric_counts = cast(Mapping[str, int], value)
    if numeric_counts["evaluable_count"] + numeric_counts["unevaluable_count"] != test_count:
        raise ValueError("Stage 4 evaluable and unevaluable counts must partition the pilot")
    if numeric_counts["fail_closed_count"] != numeric_counts["unevaluable_count"]:
        raise ValueError("Stage 4 fail-closed count must equal unevaluable count")
    if (
        numeric_counts["both_witness_replays_passed_count"]
        + numeric_counts["one_or_more_witness_replays_failed_count"]
        != test_count
    ):
        raise ValueError("Stage 4 witness replay counts must partition the pilot")
    expected_from_cases = {
        "evaluable_count": sum(item["status"] == "SUCCESS" for item in cases),
        "unevaluable_count": sum(item["status"] == "FAIL_CLOSED" for item in cases),
        "fail_closed_count": sum(item["status"] == "FAIL_CLOSED" for item in cases),
        "out_of_distribution_count": sum(
            "OUT_OF_DISTRIBUTION" in cast(list[object], item["failure_reasons"]) for item in cases
        ),
        "both_witness_replays_passed_count": sum(
            item["dual_witness_replay_passed"] is True for item in cases
        ),
        "one_or_more_witness_replays_failed_count": sum(
            item["dual_witness_replay_passed"] is False for item in cases
        ),
    }
    if any(numeric_counts[name] != expected for name, expected in expected_from_cases.items()):
        raise ValueError("Stage 4 summary counts do not match the bound case index")
    interval_fields = {
        "evaluable_proportion": "evaluable_count",
        "unevaluable_proportion": "unevaluable_count",
        "fail_closed_proportion": "fail_closed_count",
        "out_of_distribution_proportion": "out_of_distribution_count",
        "both_witness_replays_passed_proportion": "both_witness_replays_passed_count",
        "one_or_more_witness_replays_failed_proportion": (
            "one_or_more_witness_replays_failed_count"
        ),
    }
    for interval_name, count_name in interval_fields.items():
        expected_interval = wilson_score_proportion(
            numeric_counts[count_name],
            test_count,
            confidence_level=float(confidence),
        ).to_dict()
        if value.get(interval_name) != expected_interval:
            raise ValueError(f"Stage 4 {interval_name} conflicts with its bound count")


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    copied = _json_copy(dict(value))
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must serialize to a JSON object")
    return copied


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 5 report fields must be finite JSON values") from error


def _canonical_compact(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and not any(character in value for character in "\r\n\x00")
    )


def _require_report(value: object) -> Stage5PilotReport:
    if not isinstance(value, Stage5PilotReport):
        raise TypeError("report must be a Stage5PilotReport")
    value.__post_init__()
    return value


def _require_preflight(value: object) -> Stage5PreflightReport:
    if not isinstance(value, Stage5PreflightReport):
        raise TypeError("report must be a Stage5PreflightReport")
    value.__post_init__()
    return value


def _preflight(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")


__all__ = [
    "STAGE5_AUTHORITY_BOUNDARY",
    "STAGE5_PILOT_REPORT_SCHEMA_VERSION",
    "STAGE5_PILOT_SCOPE",
    "STAGE5_SOURCE_DIGEST_METHOD",
    "Stage5PilotReport",
    "issue_stage5_pilot_report",
    "render_stage5_pilot_html",
    "render_stage5_pilot_json",
    "render_stage5_preflight_html",
    "render_stage5_preflight_json",
    "write_stage5_pilot_reports",
    "write_stage5_preflight_reports",
]
