"""Deterministic Stage 8 repeatability, stability, and preflight reporters."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage8_gate import Stage8PreflightReport
from retinaentail.stability.stage8 import Stage8StabilityReport
from retinaentail.witnesses.replay import write_bytes_fail_safe

STAGE8_REPORT_SCHEMA_VERSION: Final = "8.0.0"
STAGE8_REPORT_SCOPE: Final = "SYNTHETIC_FIXTURE"
STAGE8_REPORT_METHOD: Final = "synthetic_locked_pilot_repeatability_witness_stability_v1"
STAGE8_REPORT_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE8_SYNTHETIC_PACKAGE"

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TOKEN: Final = object()
_ISSUED_REPORTS: dict[int, tuple[object, str]] = {}


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8PilotReport:
    """Issuer-bound synthetic report; serialization conveys no authority."""

    run_id: str
    source_package: Mapping[str, object]
    capability_lifecycle: Mapping[str, object]
    stability_audit: Stage8StabilityReport
    replay_receipts: tuple[Mapping[str, object], ...]
    stage8_config_sha256: str
    stage8_observations_sha256: str
    stage8_model_provenance_sha256: str
    schema_version: str = STAGE8_REPORT_SCHEMA_VERSION
    scope: str = STAGE8_REPORT_SCOPE
    method: str = STAGE8_REPORT_METHOD
    preregistered_before_replay: bool = True
    exact_case_set_enforced: bool = True
    sample_replacement_performed: bool = False
    thresholds_reestimated: bool = False
    test_split_used_for_tuning: bool = False
    locked_baseline_only: bool = True
    process_capability_required: bool = True
    process_capability_serialized: bool = False
    detached_receipt_authoritative: bool = False
    synthetic_authorizes_real_data: bool = False
    synthetic_authorizes_physical_root: bool = False
    dataset_root_probed: bool = False
    real_data_read_count: int = 0
    cpu_only: bool = True
    gpu_used: bool = False
    network_used: bool = False
    clinical_claim: bool = False
    individual_report_authoritative: bool = False
    authority_boundary: str = STAGE8_REPORT_AUTHORITY_BOUNDARY
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        issued = _ISSUED_REPORTS.get(id(self))
        new_issue = _token is _REPORT_TOKEN
        if not new_issue and (issued is None or issued[0] is not self):
            raise TypeError("Stage8PilotReport must be created by its issuer")
        if (
            self.schema_version != STAGE8_REPORT_SCHEMA_VERSION
            or self.scope != STAGE8_REPORT_SCOPE
            or self.method != STAGE8_REPORT_METHOD
            or self.authority_boundary != STAGE8_REPORT_AUTHORITY_BOUNDARY
        ):
            raise ValueError("Stage 8 report metadata is invalid")
        if not _identifier(self.run_id):
            raise ValueError("run_id must be a canonical non-empty identifier")
        for name in (
            "stage8_config_sha256",
            "stage8_observations_sha256",
            "stage8_model_provenance_sha256",
        ):
            _digest(getattr(self, name), name)
        safety = {
            "preregistered_before_replay": True,
            "exact_case_set_enforced": True,
            "sample_replacement_performed": False,
            "thresholds_reestimated": False,
            "test_split_used_for_tuning": False,
            "locked_baseline_only": True,
            "process_capability_required": True,
            "process_capability_serialized": False,
            "detached_receipt_authoritative": False,
            "synthetic_authorizes_real_data": False,
            "synthetic_authorizes_physical_root": False,
            "dataset_root_probed": False,
            "cpu_only": True,
            "gpu_used": False,
            "network_used": False,
            "clinical_claim": False,
            "individual_report_authoritative": False,
        }
        if any(getattr(self, name) is not expected for name, expected in safety.items()):
            raise ValueError("Stage 8 safety flags are inconsistent")
        if self.real_data_read_count != 0:
            raise ValueError("the Stage 8 synthetic fixture cannot report real-data reads")
        if not isinstance(self.stability_audit, Stage8StabilityReport):
            raise TypeError("stability_audit must be a Stage8StabilityReport")
        self.stability_audit.__post_init__()
        source = _json_mapping(self.source_package, "source_package")
        capability = _json_mapping(self.capability_lifecycle, "capability_lifecycle")
        receipts = tuple(_json_mapping(value, "replay_receipt") for value in self.replay_receipts)
        _validate_source(source, self.stability_audit)
        _validate_capability(capability, self.stability_audit)
        _validate_receipts(receipts, self.stability_audit)
        object.__setattr__(self, "source_package", MappingProxyType(source))
        object.__setattr__(self, "capability_lifecycle", MappingProxyType(capability))
        object.__setattr__(
            self,
            "replay_receipts",
            tuple(MappingProxyType(value) for value in receipts),
        )
        fingerprint = _canonical(self._payload())
        if new_issue:
            _ISSUED_REPORTS[id(self)] = (self, fingerprint)
        elif issued is None or issued[1] != fingerprint:
            raise ValueError("issued Stage 8 report changed after issuance")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "method": self.method,
            "run_id": self.run_id,
            "source_package": _json_copy(dict(self.source_package)),
            "capability_lifecycle": _json_copy(dict(self.capability_lifecycle)),
            "preregistration": self.stability_audit.preregistration.to_dict(),
            "stability_audit": self.stability_audit.to_dict(),
            "replay_receipts": [_json_copy(dict(value)) for value in self.replay_receipts],
            "stage8_config_sha256": self.stage8_config_sha256,
            "stage8_observations_sha256": self.stage8_observations_sha256,
            "stage8_model_provenance_sha256": self.stage8_model_provenance_sha256,
            "preregistered_before_replay": self.preregistered_before_replay,
            "exact_case_set_enforced": self.exact_case_set_enforced,
            "sample_replacement_performed": self.sample_replacement_performed,
            "thresholds_reestimated": self.thresholds_reestimated,
            "test_split_used_for_tuning": self.test_split_used_for_tuning,
            "locked_baseline_only": self.locked_baseline_only,
            "process_capability_required": self.process_capability_required,
            "process_capability_serialized": self.process_capability_serialized,
            "detached_receipt_authoritative": self.detached_receipt_authoritative,
            "synthetic_authorizes_real_data": self.synthetic_authorizes_real_data,
            "synthetic_authorizes_physical_root": self.synthetic_authorizes_physical_root,
            "dataset_root_probed": self.dataset_root_probed,
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


def issue_stage8_pilot_report(
    *,
    run_id: str,
    source_package: Mapping[str, object],
    capability_lifecycle: Mapping[str, object],
    stability_audit: Stage8StabilityReport,
    replay_receipts: Sequence[Mapping[str, object]],
    stage8_config_sha256: str,
    stage8_observations_sha256: str,
    stage8_model_provenance_sha256: str,
) -> Stage8PilotReport:
    return Stage8PilotReport(
        run_id=run_id,
        source_package=source_package,
        capability_lifecycle=capability_lifecycle,
        stability_audit=stability_audit,
        replay_receipts=tuple(replay_receipts),
        stage8_config_sha256=stage8_config_sha256,
        stage8_observations_sha256=stage8_observations_sha256,
        stage8_model_provenance_sha256=stage8_model_provenance_sha256,
        _token=_REPORT_TOKEN,
    )


def render_stage8_pilot_json(report: Stage8PilotReport) -> str:
    report = _require_report(report)
    return (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def render_stage8_pilot_html(report: Stage8PilotReport) -> str:
    report = _require_report(report)
    stability = report.stability_audit
    transition_rows = "".join(
        "<tr>"
        f'<th scope="row">{escape(item.case_id)}</th>'
        f"<td>{escape(item.run_id)}</td>"
        f"<td>{escape(item.transition)}</td>"
        f"<td>{'yes' if item.repeat_fail_closed else 'no'}</td>"
        f"<td>{'yes' if item.repeat_out_of_distribution else 'no'}</td>"
        f"<td>{'yes' if item.positive_witness_replay_stable else 'no'}</td>"
        f"<td>{'yes' if item.negative_witness_replay_stable else 'no'}</td>"
        f"<td>{'yes' if item.dual_witness_replay_stable else 'no'}</td>"
        "<td><code>"
        f"{escape(item.baseline_positive_witness_sha256 or 'unavailable')}</code><br>"
        f"replay: {'yes' if item.baseline_positive_witness_replay_valid else 'no'}</td>"
        "<td><code>"
        f"{escape(item.repeat_positive_witness_sha256 or 'unavailable')}</code><br>"
        f"replay: {'yes' if item.repeat_positive_witness_replay_valid else 'no'}</td>"
        "<td><code>"
        f"{escape(item.baseline_negative_witness_sha256 or 'unavailable')}</code><br>"
        f"replay: {'yes' if item.baseline_negative_witness_replay_valid else 'no'}</td>"
        "<td><code>"
        f"{escape(item.repeat_negative_witness_sha256 or 'unavailable')}</code><br>"
        f"replay: {'yes' if item.repeat_negative_witness_replay_valid else 'no'}</td>"
        f"<td>{'yes' if item.dual_witness_hash_stable else 'no'}</td>"
        f"<td>{escape(', '.join(value.value for value in item.failure_reasons) or 'none')}</td>"
        "</tr>"
        for item in stability.case_transitions
    )
    metric_rows = "".join(
        "<tr>"
        f'<th scope="row">{escape(metric.value)}</th>'
        f"<td>{interval.numerator}/{interval.denominator}</td>"
        f"<td>{interval.estimate:.6f}</td>"
        f"<td>[{interval.lower:.6f}, {interval.upper:.6f}]</td>"
        "</tr>"
        for metric, interval in stability.metric_intervals.items()
    )
    event_rows = "".join(
        "<tr>"
        f'<th scope="row">{escape(event.value)}</th>'
        f"<td>{interval.numerator}/{interval.denominator}</td>"
        f"<td>{interval.estimate:.6f}</td>"
        f"<td>[{interval.lower:.6f}, {interval.upper:.6f}]</td>"
        "</tr>"
        for event, interval in stability.event_intervals.items()
    )
    source_hash = cast(str, report.source_package["package_sha256"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 8 repeatability audit</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .4rem .55rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Stage 8 frozen-pilot repeatability and dual-witness stability audit</h1>
  <p><strong>Synthetic fixture only. It authorizes neither real data nor a physical
     root.</strong></p>
  <p>Run: <code>{escape(report.run_id)}</code>; audit status:
     <strong>{escape(stability.status.value)}</strong>.</p>
  <p>Complete Stage 7 source package: <code>{escape(source_hash)}</code>.</p>
  <p>Fixed cases: {len(stability.preregistration.inclusion_case_ids)}; repeat runs:
     {len(stability.preregistration.replay_specs)}; expected transitions:
     {stability.expected_transition_count}.</p>
  <h2>Per-case four-state transitions and witness stability</h2>
  <table><thead><tr><th>Case</th><th>Run</th><th>Transition</th><th>Fail closed</th>
  <th>OOD</th><th>Positive replay stable</th><th>Negative replay stable</th>
  <th>Dual replay stable</th><th>Baseline + witness SHA / replay</th>
  <th>Repeat + witness SHA / replay</th><th>Baseline - witness SHA / replay</th>
  <th>Repeat - witness SHA / replay</th><th>Dual hash stable</th><th>Reasons</th></tr></thead>
  <tbody>{transition_rows}</tbody></table>
  <h2>Preregistered stability metrics</h2>
  <table><thead><tr><th>Metric</th><th>Count</th><th>Estimate</th><th>95% Wilson interval</th>
  </tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Fail-closed, OOD, OOM, and dual-replay events</h2>
  <table><thead><tr><th>Event</th><th>Count</th><th>Estimate</th><th>95% Wilson interval</th>
  </tr></thead><tbody>{event_rows}</tbody></table>
  <p>Thresholds re-estimated: no. Test tuning: prohibited. Sample replacement: none.</p>
  <p>Authority boundary: <code>{escape(report.authority_boundary)}</code>.</p>
</body>
</html>
"""


def render_stage8_preflight_json(report: Stage8PreflightReport) -> str:
    report = _require_preflight(report)
    return (
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )


def render_stage8_preflight_html(report: Stage8PreflightReport) -> str:
    report = _require_preflight(report)
    reasons = "".join(f"<li><code>{escape(reason.value)}</code></li>" for reason in report.reasons)
    if not reasons:
        reasons = (
            "<li>Presence checks are ready; live process-capability replay is still required.</li>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width, initial-scale=1"><title>RetinaEntail-DR Stage 8 preflight</title>
<style>:root {{ color-scheme: light dark; font-family: system-ui,sans-serif; }}
body {{ margin:2rem; line-height:1.45; }} code {{ overflow-wrap:anywhere; }}</style></head>
<body><h1>Stage 8 zero-I/O preflight</h1>
<p>Status: <strong>{escape(report.status.value)}</strong>.</p>
<p>Dataset: <code>{escape(report.dataset_id or "not supplied")}</code>; version:
<code>{escape(report.dataset_version or "not supplied")}</code>.</p>
<p><strong>This receipt accepts no path, reads no environment variable, probes no root,
and authorizes no file open.</strong></p><h2>Reasons</h2><ul>{reasons}</ul></body></html>
"""


def write_stage8_preflight_reports(
    report: Stage8PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    report = _require_preflight(report)
    if not isinstance(output_dir, Path) or not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("Stage 8 output and boundary types are invalid")
    json_path = boundary.require_safe(output_dir / "stage8-preflight.json")
    html_path = boundary.require_safe(output_dir / "stage8-preflight.html")
    payloads = (
        (json_path, render_stage8_preflight_json(report).encode("utf-8")),
        (html_path, render_stage8_preflight_html(report).encode("utf-8")),
    )
    for path, payload in payloads:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    for path, payload in payloads:
        write_bytes_fail_safe(path, payload)
    return json_path, html_path


def _validate_source(value: Mapping[str, object], audit: Stage8StabilityReport) -> None:
    required = {
        "schema_version",
        "scope",
        "dataset_id",
        "dataset_version",
        "package_sha256",
        "package_file_count",
        "stage2_package_sha256",
        "stage3_package_sha256",
        "physical_root_identity_sha256",
        "frozen_bindings",
        "fixture_only",
    }
    if set(value) != required:
        raise ValueError("source package descriptor fields are incomplete or unknown")
    prereg = audit.preregistration
    if (
        value["scope"] != "SYNTHETIC_FIXTURE"
        or value["fixture_only"] is not True
        or value["dataset_id"] != prereg.dataset_id
        or value["dataset_version"] != prereg.dataset_version
        or value["package_sha256"] != prereg.stage7_package_sha256
    ):
        raise ValueError("source package descriptor does not match the preregistration")
    if (
        isinstance(value["package_file_count"], bool)
        or not isinstance(value["package_file_count"], int)
        or value["package_file_count"] < 1
    ):
        raise ValueError("source package file count must be positive")
    for name in (
        "package_sha256",
        "stage2_package_sha256",
        "stage3_package_sha256",
        "physical_root_identity_sha256",
    ):
        _digest(value[name], f"source_package.{name}")


def _validate_capability(value: Mapping[str, object], audit: Stage8StabilityReport) -> None:
    required_true = (
        "process_bound",
        "non_serializable",
        "one_shot",
        "all_capabilities_fresh",
        "authorizes_embedded_package_replay",
    )
    required_false = (
        "authorizes_real_data",
        "authorizes_physical_dataset_root",
        "detached_receipt_authoritative",
        "authorizes_file_open",
    )
    if (
        value.get("scope") != "SYNTHETIC_FIXTURE"
        or any(value.get(name) is not True for name in required_true)
        or any(value.get(name) is not False for name in required_false)
    ):
        raise ValueError("synthetic process-capability lifecycle is incomplete or unsafe")
    issued = value.get("issued_count")
    if isinstance(issued, bool) or not isinstance(issued, int) or issued < 1:
        raise ValueError("Stage 8 must issue at least one fresh process capability")
    if issued != len(audit.preregistration.replay_specs):
        raise ValueError("each preregistered replay requires a fresh process capability")
    if any(
        value.get(name) != issued
        for name in (
            "consumed_count",
            "second_use_rejected_count",
            "complete_stage7_replay_count",
        )
    ):
        raise ValueError("every Stage 8 process capability must be consumed and replayed once")
    per_run = value.get("per_run")
    if not isinstance(per_run, list) or len(per_run) != issued:
        raise ValueError("each Stage 8 process capability requires a per-run receipt")
    run_ids: set[str] = set()
    for item, spec in zip(per_run, audit.preregistration.replay_specs, strict=True):
        receipt = _json_mapping(item, "capability per_run")
        run_id = receipt.get("run_id")
        if (
            not isinstance(run_id, str)
            or run_id in run_ids
            or run_id != spec.run_id
            or receipt.get("seed") != spec.seed
        ):
            raise ValueError("capability run IDs must be unique")
        run_ids.add(run_id)
        if (
            receipt.get("ttl_seconds") != 5
            or receipt.get("issued") is not True
            or receipt.get("consumed") is not True
            or receipt.get("complete_stage7_package_replayed") is not True
            or receipt.get("second_consumption_rejected") is not True
            or receipt.get("authorizes_real_data") is not False
            or receipt.get("authorizes_physical_dataset_root") is not False
        ):
            raise ValueError("per-run process-capability receipt is incomplete or unsafe")


def _validate_receipts(values: tuple[dict[str, object], ...], audit: Stage8StabilityReport) -> None:
    expected = audit.preregistration.replay_specs
    if len(values) != len(expected):
        raise ValueError("each preregistered repeat run needs one verifier receipt")
    for receipt, spec in zip(values, expected, strict=True):
        if (
            receipt.get("run_id") != spec.run_id
            or receipt.get("seed") != spec.seed
            or receipt.get("degradation_replay_id") != spec.degradation_replay_id
            or receipt.get("degradation_replay_set_sha256") != spec.degradation_replay_set_sha256
            or receipt.get("complete_stage7_package_replay_verified") is not True
            or receipt.get("locked_baseline_reexecuted") is not True
            or receipt.get("thresholds_reestimated") is not False
            or receipt.get("test_tuning_performed") is not False
            or receipt.get("sample_replacement_performed") is not False
        ):
            raise ValueError("repeat verifier receipt conflicts with preregistration")
        for name in (
            "stage7_package_sha256",
            "stage7_tree_sha256",
            "case_observations_sha256",
            "dual_witness_replay_set_sha256",
        ):
            _digest(receipt.get(name), f"replay_receipt.{name}")


def _require_report(value: object) -> Stage8PilotReport:
    if not isinstance(value, Stage8PilotReport):
        raise TypeError("report must be a Stage8PilotReport")
    value.__post_init__()
    return value


def _require_preflight(value: object) -> Stage8PreflightReport:
    if not isinstance(value, Stage8PreflightReport):
        raise TypeError("report must be a Stage8PreflightReport")
    value.__post_init__()
    return value


def _json_mapping(value: object, name: str) -> dict[str, object]:
    copied = _json_copy(dict(value) if isinstance(value, Mapping) else value)
    if not isinstance(copied, dict) or any(not isinstance(key, str) for key in copied):
        raise ValueError(f"{name} must be a finite JSON object")
    return cast(dict[str, object], copied)


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 8 report values must be finite JSON") from error


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


__all__ = [
    "STAGE8_REPORT_AUTHORITY_BOUNDARY",
    "STAGE8_REPORT_METHOD",
    "STAGE8_REPORT_SCHEMA_VERSION",
    "STAGE8_REPORT_SCOPE",
    "Stage8PilotReport",
    "issue_stage8_pilot_report",
    "render_stage8_pilot_html",
    "render_stage8_pilot_json",
    "render_stage8_preflight_html",
    "render_stage8_preflight_json",
    "write_stage8_preflight_reports",
]
