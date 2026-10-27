"""Deterministic reporting for the Stage 7 synthetic consumption transaction.

The emitted JSON and HTML are receipts, never serialized authorization objects.
Only replaying the complete Stage 7 package re-establishes the evidence chain.
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

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.pilot.identity import Stage5IdentityAudit, Stage5IdentityAuditStatus
from retinaentail.pilot.stage7_gate import Stage7PreflightReport
from retinaentail.witnesses.replay import write_bytes_fail_safe

STAGE7_REPORT_SCHEMA_VERSION: Final = "7.0.0"
STAGE7_SCOPE: Final = "SYNTHETIC_ONLY"
STAGE7_METHOD: Final = "synthetic_authorization_consumption_rebuild_locked_pilot_v1"
STAGE7_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE7_PACKAGE"
STAGE7_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TOKEN: Final = object()
_ISSUED_REPORTS: dict[int, tuple[object, str]] = {}

_SOURCE_KEYS: Final = {
    "scope",
    "run_id",
    "authority_boundary",
    "package_sha256",
    "package_file_count",
    "envelope_sha256",
    "complete_byte_replay_verified",
}
_AUTHORIZATION_KEYS: Final = {
    "scope",
    "ttl_seconds",
    "process_bound",
    "non_serializable",
    "one_shot",
    "issued",
    "consumed",
    "second_consumption_rejected",
    "authorizes_embedded_fixture_replay",
    "authorizes_real_data",
    "authorizes_physical_dataset_root",
    "detached_receipt_authoritative",
}
_REBUILD_KEYS: Final = {
    "mode",
    "dataset_id",
    "dataset_version",
    "record_count",
    "original_manifest_sha256",
    "rebuilt_manifest_sha256",
    "record_fingerprints_sha256",
    "canonical_equivalent",
    "byte_equivalent",
    "mismatch_count",
    "physical_dataset_root_used",
    "missing_label_policy",
}
_LOCKED_KEYS: Final = {
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


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7PilotReport:
    """Issuer-bound description of one synthetic-only Stage 7 transaction."""

    run_id: str
    stage6_source: Mapping[str, object]
    authorization_consumption: Mapping[str, object]
    rebuild_equivalence: Mapping[str, object]
    identity_isolation: Mapping[str, object]
    locked_contract: Mapping[str, object]
    evaluation: Mapping[str, object]
    case_index: tuple[Mapping[str, object], ...]
    stage7_config_sha256: str
    stage7_source_payload_sha256: str
    stage7_model_provenance_sha256: str
    schema_version: str = STAGE7_REPORT_SCHEMA_VERSION
    scope: str = STAGE7_SCOPE
    method: str = STAGE7_METHOD
    source_digest_method: str = STAGE7_DIGEST_METHOD
    split: str = "test"
    thresholds_locked_before_transaction: bool = True
    thresholds_reestimated: bool = False
    test_split_used_for_calibration: bool = False
    test_tuning_prohibited: bool = True
    synthetic_capability_authorizes_real_data: bool = False
    real_authorization_issued: bool = False
    dataset_root_probed: bool = False
    real_data_used: bool = False
    real_data_read_count: int = 0
    cpu_only: bool = True
    gpu_used: bool = False
    network_used: bool = False
    clinical_claim: bool = False
    individual_report_authoritative: bool = False
    authority_boundary: str = STAGE7_AUTHORITY_BOUNDARY
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        issued = _ISSUED_REPORTS.get(id(self))
        new_issue = _token is _REPORT_TOKEN
        if not new_issue and (issued is None or issued[0] is not self):
            raise TypeError("Stage7PilotReport must be created by its issuer")
        if (
            self.schema_version != STAGE7_REPORT_SCHEMA_VERSION
            or self.scope != STAGE7_SCOPE
            or self.method != STAGE7_METHOD
            or self.source_digest_method != STAGE7_DIGEST_METHOD
            or self.split != "test"
        ):
            raise ValueError("Stage 7 report version, scope, method, or split is invalid")
        if not _identifier(self.run_id):
            raise ValueError("run_id must be a safe non-empty identifier")
        for name in (
            "stage7_config_sha256",
            "stage7_source_payload_sha256",
            "stage7_model_provenance_sha256",
        ):
            _digest(getattr(self, name), name)
        required_flags = {
            "thresholds_locked_before_transaction": True,
            "thresholds_reestimated": False,
            "test_split_used_for_calibration": False,
            "test_tuning_prohibited": True,
            "synthetic_capability_authorizes_real_data": False,
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
            raise ValueError("Stage 7 safety flags are inconsistent")
        if self.real_data_read_count != 0:
            raise ValueError("the Stage 7 synthetic fixture cannot report real-data reads")
        if self.authority_boundary != STAGE7_AUTHORITY_BOUNDARY:
            raise ValueError("only a complete verified Stage 7 package is authoritative")

        source = _json_mapping(self.stage6_source, "stage6_source")
        authorization = _json_mapping(self.authorization_consumption, "authorization_consumption")
        rebuild = _json_mapping(self.rebuild_equivalence, "rebuild_equivalence")
        identity = _json_mapping(self.identity_isolation, "identity_isolation")
        locked = _json_mapping(self.locked_contract, "locked_contract")
        evaluation = _json_mapping(self.evaluation, "evaluation")
        cases = tuple(_json_mapping(item, "case_index item") for item in self.case_index)
        _validate_source(source)
        _validate_authorization(authorization)
        _validate_rebuild(rebuild)
        _validate_identity(identity)
        _validate_locked(locked)
        _validate_evaluation(evaluation, locked=locked, cases=cases)
        object.__setattr__(self, "stage6_source", MappingProxyType(source))
        object.__setattr__(self, "authorization_consumption", MappingProxyType(authorization))
        object.__setattr__(self, "rebuild_equivalence", MappingProxyType(rebuild))
        object.__setattr__(self, "identity_isolation", MappingProxyType(identity))
        object.__setattr__(self, "locked_contract", MappingProxyType(locked))
        object.__setattr__(self, "evaluation", MappingProxyType(evaluation))
        object.__setattr__(self, "case_index", tuple(MappingProxyType(item) for item in cases))

        fingerprint = _canonical(self._payload())
        if new_issue:
            _ISSUED_REPORTS[id(self)] = (self, fingerprint)
        elif issued is None or issued[1] != fingerprint:
            raise ValueError("issued Stage 7 report changed after issuance")

    def _payload(self) -> dict[str, object]:
        evaluation = dict(self.evaluation)
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "method": self.method,
            "run_id": self.run_id,
            "split": self.split,
            "source_digest_method": self.source_digest_method,
            "stage6_source": _json_copy(dict(self.stage6_source)),
            "authorization_consumption": _json_copy(dict(self.authorization_consumption)),
            "rebuild_equivalence": _json_copy(dict(self.rebuild_equivalence)),
            "identity_isolation": _json_copy(dict(self.identity_isolation)),
            "locked_contract": _json_copy(dict(self.locked_contract)),
            "evaluation": _json_copy(evaluation),
            "case_index": [_json_copy(dict(item)) for item in self.case_index],
            "stage7_config_sha256": self.stage7_config_sha256,
            "stage7_source_payload_sha256": self.stage7_source_payload_sha256,
            "stage7_model_provenance_sha256": self.stage7_model_provenance_sha256,
            "thresholds_locked_before_transaction": self.thresholds_locked_before_transaction,
            "thresholds_reestimated": self.thresholds_reestimated,
            "test_split_used_for_calibration": self.test_split_used_for_calibration,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "synthetic_capability_authorizes_real_data": (
                self.synthetic_capability_authorizes_real_data
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


def issue_stage7_pilot_report(
    *,
    run_id: str,
    stage6_source: Mapping[str, object],
    authorization_consumption: Mapping[str, object],
    rebuild_equivalence: Mapping[str, object],
    identity_isolation: Mapping[str, object],
    locked_contract: Mapping[str, object],
    evaluation: Mapping[str, object],
    case_index: Sequence[Mapping[str, object]],
    stage7_config_sha256: str,
    stage7_source_payload_sha256: str,
    stage7_model_provenance_sha256: str,
) -> Stage7PilotReport:
    """Issue a report only after its full synthetic transaction has completed."""

    raw_case_index: object = case_index
    if isinstance(raw_case_index, str | bytes) or not isinstance(raw_case_index, Sequence):
        raise TypeError("case_index must be a non-string sequence")
    return Stage7PilotReport(
        run_id=run_id,
        stage6_source=stage6_source,
        authorization_consumption=authorization_consumption,
        rebuild_equivalence=rebuild_equivalence,
        identity_isolation=identity_isolation,
        locked_contract=locked_contract,
        evaluation=evaluation,
        case_index=tuple(case_index),
        stage7_config_sha256=stage7_config_sha256,
        stage7_source_payload_sha256=stage7_source_payload_sha256,
        stage7_model_provenance_sha256=stage7_model_provenance_sha256,
        _token=_REPORT_TOKEN,
    )


def render_stage7_pilot_json(report: Stage7PilotReport) -> str:
    report = _require_report(report)
    return (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def render_stage7_pilot_html(report: Stage7PilotReport) -> str:
    report = _require_report(report)
    evaluation = report.evaluation
    state_counts = cast(Mapping[str, int], evaluation["state_counts"])
    intervals = cast(Mapping[str, Mapping[str, object]], evaluation["state_proportions"])
    rows = []
    for state in EntailmentState:
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(state.value)}</th>'
            f"<td>{state_counts[state.value]}</td>"
            f"<td>{escape(_interval_text(intervals[state.value]))}</td>"
            "</tr>"
        )
    source = report.stage6_source
    rebuild = report.rebuild_equivalence
    identity = report.identity_isolation
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 7 synthetic consumption pilot</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 7 authorization-consumption and rebuild transaction</h1>
  <p><strong>Synthetic fixture only. It cannot authorize real data, and no physical
     dataset root was probed or read.</strong></p>
  <p>Run: <code>{escape(report.run_id, quote=True)}</code></p>
  <p>Complete Stage 6 source: <code>{escape(cast(str, source["package_sha256"]))}</code>
     ({source["package_file_count"]} files), fully replayed byte-for-byte.</p>
  <p>The process-bound, non-serializable synthetic authorization was consumed once;
     a second consumption was rejected. Its detached receipt is not authority.</p>
  <p>Typed Stage 2 manifest rebuild: {rebuild["record_count"]} record(s),
     equivalent: {str(rebuild["canonical_equivalent"]).lower()}.</p>
  <p>Five-identity isolation: <strong>{escape(cast(str, identity["status"]))}</strong>.</p>
  <h2>Frozen test pilot</h2>
  <table><thead><tr><th>State</th><th>Count</th><th>95% Wilson interval</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table>
  <p>Fail closed: {evaluation["fail_closed_count"]}; OOD:
     {evaluation["out_of_distribution_count"]}; dual witness replay passed:
     {evaluation["both_witness_replays_passed_count"]} / {evaluation["test_case_count"]}.</p>
  <p>Thresholds re-estimated: no. Test tuning: prohibited.</p>
  <p>Authority boundary: <code>{escape(report.authority_boundary)}</code>.</p>
</body>
</html>
"""


def render_stage7_preflight_json(report: Stage7PreflightReport) -> str:
    """Serialize a zero-I/O presence report without any supplied path values."""

    report = _require_preflight(report)
    return (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def render_stage7_preflight_html(report: Stage7PreflightReport) -> str:
    """Render the Stage 7 BLOCKED/READY presence result as a path-free receipt."""

    report = _require_preflight(report)
    reasons = "".join(f"<li><code>{escape(reason.value)}</code></li>" for reason in report.reasons)
    if not reasons:
        reasons = (
            "<li>Presence is complete, but a live process-issued Stage 6 bound root "
            "must still be replayed and consumed.</li>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 7 preflight</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 7 zero-I/O preflight</h1>
  <p>Status: <strong>{escape(report.status.value)}</strong>.</p>
  <p>Dataset: <code>{escape(report.dataset_id or "not supplied")}</code>;
     version: <code>{escape(report.dataset_version or "not supplied")}</code>.</p>
  <p><strong>This receipt accepts no path authority, probes no root, consumes no
     Stage 6 binding, and authorizes no file open.</strong></p>
  <h2>Reasons</h2><ul>{reasons}</ul>
</body>
</html>
"""


def write_stage7_preflight_reports(
    report: Stage7PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    """Write deterministic JSON/HTML without overwriting non-identical evidence."""

    report = _require_preflight(report)
    if not isinstance(output_dir, Path) or not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("Stage 7 output and boundary types are invalid")
    json_path = boundary.require_safe(output_dir / "stage7-preflight.json")
    html_path = boundary.require_safe(output_dir / "stage7-preflight.html")
    payloads = (
        (json_path, render_stage7_preflight_json(report).encode("utf-8")),
        (html_path, render_stage7_preflight_html(report).encode("utf-8")),
    )
    for path, payload in payloads:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    for path, payload in payloads:
        write_bytes_fail_safe(path, payload)
    return json_path, html_path


def _validate_source(value: Mapping[str, object]) -> None:
    _keys(value, _SOURCE_KEYS, "stage6_source")
    if (
        value["scope"] != STAGE7_SCOPE
        or value["authority_boundary"] != "COMPLETE_VERIFIED_STAGE6_PACKAGE"
        or not _identifier(value["run_id"])
        or value["complete_byte_replay_verified"] is not True
    ):
        raise ValueError("Stage 6 source is not a complete verified synthetic package")
    _digest(value["package_sha256"], "stage6_source.package_sha256")
    _digest(value["envelope_sha256"], "stage6_source.envelope_sha256")
    _positive_int(value["package_file_count"], "stage6_source.package_file_count")


def _validate_authorization(value: Mapping[str, object]) -> None:
    _keys(value, _AUTHORIZATION_KEYS, "authorization_consumption")
    if value["scope"] != STAGE7_SCOPE or value["ttl_seconds"] != 5:
        raise ValueError("Stage 7 synthetic authorization scope or TTL is invalid")
    expected = {
        "process_bound": True,
        "non_serializable": True,
        "one_shot": True,
        "issued": True,
        "consumed": True,
        "second_consumption_rejected": True,
        "authorizes_embedded_fixture_replay": True,
        "authorizes_real_data": False,
        "authorizes_physical_dataset_root": False,
        "detached_receipt_authoritative": False,
    }
    if any(value[name] is not expected for name, expected in expected.items()):
        raise ValueError("Stage 7 synthetic authorization receipt is unsafe or incomplete")


def _validate_rebuild(value: Mapping[str, object]) -> None:
    _keys(value, _REBUILD_KEYS, "rebuild_equivalence")
    if (
        value["mode"] != "EMBEDDED_STAGE2_TYPED_ROUNDTRIP"
        or not _identifier(value["dataset_id"])
        or not _identifier(value["dataset_version"])
        or value["canonical_equivalent"] is not True
        or value["byte_equivalent"] is not True
        or value["mismatch_count"] != 0
        or value["physical_dataset_root_used"] is not False
        or value["missing_label_policy"] != "UNKNOWN_NEVER_NEGATIVE"
    ):
        raise ValueError("Stage 7 rebuild equivalence proof is invalid")
    _positive_int(value["record_count"], "rebuild_equivalence.record_count")
    original = _digest(value["original_manifest_sha256"], "original_manifest_sha256")
    rebuilt = _digest(value["rebuilt_manifest_sha256"], "rebuilt_manifest_sha256")
    _digest(value["record_fingerprints_sha256"], "record_fingerprints_sha256")
    if original != rebuilt:
        raise ValueError("rebuilt manifest digest differs from the frozen manifest")


def _validate_identity(value: Mapping[str, object]) -> None:
    try:
        audit = Stage5IdentityAudit.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 7 five-identity audit is malformed") from error
    if (
        audit.status is not Stage5IdentityAuditStatus.PASS
        or not audit.isolation_proven
        or audit.record_count < 1
        or audit.reasons
        or audit.issues
    ):
        raise ValueError("Stage 7 five-identity isolation did not pass")


def _validate_locked(value: Mapping[str, object]) -> None:
    _keys(value, _LOCKED_KEYS, "locked_contract")
    alpha = _finite(value["alpha"], "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("locked alpha must lie strictly between zero and one")
    for name in ("positive_threshold", "negative_threshold"):
        if _finite(value[name], name) < 0.0:
            raise ValueError("locked thresholds must be non-negative")
    positive = _positive_int(value["positive_sample_count"], "positive_sample_count")
    negative = _positive_int(value["negative_sample_count"], "negative_sample_count")
    if _positive_int(value["total_sample_count"], "total_sample_count") != positive + negative:
        raise ValueError("locked calibration sample counts are inconsistent")
    for name in (
        "calibration_artifact_hash",
        "config_hash",
        "data_hash",
        "model_hash",
    ):
        _digest(value[name], name)


def _validate_evaluation(
    value: Mapping[str, object],
    *,
    locked: Mapping[str, object],
    cases: tuple[Mapping[str, object], ...],
) -> None:
    if value.get("scope") != STAGE7_SCOPE or value.get("split") != "test":
        raise ValueError("Stage 7 evaluation must retain the synthetic frozen test split")
    if value.get("locked_contract") != dict(locked):
        raise ValueError("evaluation is not bound to the exact locked contract")
    count = _positive_int(value.get("test_case_count"), "test_case_count")
    if count != len(cases):
        raise ValueError("evaluation and case-index counts differ")
    state_counts = _mapping(value.get("state_counts"), "state_counts")
    intervals = _mapping(value.get("state_proportions"), "state_proportions")
    confidence = _finite(value.get("confidence_level"), "confidence_level")
    if confidence != 0.95:
        raise ValueError("Stage 7 fixture requires 95% Wilson intervals")
    recomputed = {state.value: 0 for state in EntailmentState}
    fail_closed = 0
    out_of_distribution = 0
    dual_replay = 0
    for index, case in enumerate(cases):
        if case.get("pilot_index") != index or not _identifier(case.get("case_id")):
            raise ValueError("case index ordering or case_id is invalid")
        state = case.get("state")
        if state not in recomputed:
            raise ValueError("case index contains an unknown entailment state")
        recomputed[state] += 1
        status = case.get("status")
        if status not in {"SUCCESS", "FAIL_CLOSED"}:
            raise ValueError("case status is invalid")
        if status == "FAIL_CLOSED":
            fail_closed += 1
        reasons = case.get("failure_reasons")
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise ValueError("case failure_reasons must be a controlled string array")
        if bool(reasons) is not (status == "FAIL_CLOSED"):
            raise ValueError("case failure reasons and fail-closed status disagree")
        if status == "FAIL_CLOSED" and state != EntailmentState.OUT_OF_MODEL.value:
            raise ValueError("a fail-closed case must expose OUT_OF_MODEL")
        if "OUT_OF_DISTRIBUTION" in reasons:
            out_of_distribution += 1
        positive_replay = case.get("positive_witness_replay_passed")
        negative_replay = case.get("negative_witness_replay_passed")
        dual_replay_passed = case.get("dual_witness_replay_passed")
        if (
            type(positive_replay) is not bool  # noqa: E721
            or type(negative_replay) is not bool  # noqa: E721
            or type(dual_replay_passed) is not bool  # noqa: E721
            or dual_replay_passed is not (positive_replay and negative_replay)
        ):
            raise ValueError("case witness replay booleans are inconsistent")
        if dual_replay_passed:
            dual_replay += 1
        for name in (
            "certificate_sha256",
            "dual_replay_sha256",
            "evaluation_sha256",
            "positive_witness_sha256",
            "negative_witness_sha256",
        ):
            _digest(case.get(name), f"case.{name}")
    if state_counts != recomputed:
        raise ValueError("four-state counts do not match the case index")
    for state, numerator in recomputed.items():
        expected = wilson_score_proportion(numerator, count, confidence_level=confidence).to_dict()
        if intervals.get(state) != expected:
            raise ValueError("Wilson interval does not match its state count")
    expected_counts = {
        "fail_closed_count": fail_closed,
        "out_of_distribution_count": out_of_distribution,
        "both_witness_replays_passed_count": dual_replay,
    }
    if any(value.get(name) != expected for name, expected in expected_counts.items()):
        raise ValueError("failure, OOD, or witness replay counts are inconsistent")
    expected_intervals = {
        "fail_closed_proportion": fail_closed,
        "out_of_distribution_proportion": out_of_distribution,
        "both_witness_replays_passed_proportion": dual_replay,
    }
    for name, numerator in expected_intervals.items():
        expected = wilson_score_proportion(numerator, count, confidence_level=confidence).to_dict()
        if value.get(name) != expected:
            raise ValueError(f"{name} does not match its count")


def _require_preflight(value: object) -> Stage7PreflightReport:
    if not isinstance(value, Stage7PreflightReport):
        raise TypeError("report must be a Stage7PreflightReport")
    value.__post_init__()
    return value


def _require_report(value: object) -> Stage7PilotReport:
    if not isinstance(value, Stage7PilotReport):
        raise TypeError("report must be a Stage7PilotReport")
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
    copied = _json_copy(dict(value) if isinstance(value, Mapping) else value)
    if not isinstance(copied, dict) or any(not isinstance(key, str) for key in copied):
        raise ValueError(f"{name} must be a finite JSON object")
    return cast(dict[str, object], copied)


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 7 report values must be finite JSON") from error


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _interval_text(value: Mapping[str, object]) -> str:
    estimate = _finite(value.get("estimate"), "estimate")
    lower = _finite(value.get("lower"), "lower")
    upper = _finite(value.get("upper"), "upper")
    return f"{estimate:.6f} [{lower:.6f}, {upper:.6f}]"


__all__ = [
    "STAGE7_AUTHORITY_BOUNDARY",
    "STAGE7_DIGEST_METHOD",
    "STAGE7_METHOD",
    "STAGE7_REPORT_SCHEMA_VERSION",
    "STAGE7_SCOPE",
    "Stage7PilotReport",
    "issue_stage7_pilot_report",
    "render_stage7_preflight_html",
    "render_stage7_preflight_json",
    "render_stage7_pilot_html",
    "render_stage7_pilot_json",
    "write_stage7_preflight_reports",
]
