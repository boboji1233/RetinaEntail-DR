"""Deterministic JSON and escaped static HTML for Stage 4 evaluation audits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import TypeVar

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.gate import Stage4PreflightReport
from retinaentail.evaluation.metrics import (
    BinomialProportion,
    Stage4EvaluationReport,
)
from retinaentail.witnesses.replay import write_bytes_fail_safe

_K = TypeVar("_K")


def render_stage4_evaluation_json(report: Stage4EvaluationReport) -> str:
    """Render strict JSON; non-finite values cannot be serialized."""

    report = _require_report(report)
    return (
        json.dumps(
            report.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_stage4_preflight_json(report: Stage4PreflightReport) -> str:
    """Render a zero-I/O Stage 4 preflight as strict JSON."""

    report = _require_preflight(report)
    return (
        json.dumps(
            report.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_stage4_preflight_html(report: Stage4PreflightReport) -> str:
    """Render the path-free preflight as escaped, script-free static HTML."""

    report = _require_preflight(report)
    reasons = (
        "\n".join(
            f"<li><code>{escape(reason.value, quote=True)}</code></li>" for reason in report.reasons
        )
        or "<li>None; independent prerequisite verification is still required.</li>"
    )
    purposes = (
        ", ".join(
            f"<code>{escape(purpose.value, quote=True)}</code>"
            for purpose in report.approved_review_purposes
        )
        or "—"
    )
    dataset_id = escape(report.dataset_id or "—", quote=True)
    dataset_version = escape(report.dataset_version or "—", quote=True)
    status = escape(report.status.value, quote=True)
    presence_rows = (
        ("Approved review", report.approved_review_provided),
        ("Dataset root supplied", report.dataset_root_provided),
        ("Adapter spec", report.adapter_spec_provided),
        ("Stage 2 PASS audit", report.stage2_pass_audit_provided),
        ("Frozen data card", report.frozen_data_card_provided),
        ("Stage 3 calibration package", report.stage3_calibration_package_provided),
        ("Manual confirmations", report.manual_confirmation_record_provided),
        ("Locked evidence", report.locked_evidence_provided),
    )
    rows = "\n".join(
        f'<tr><th scope="row">{escape(label, quote=True)}</th><td>{str(present).lower()}</td></tr>'
        for label, present in presence_rows
    )
    evidence = report.locked_evidence
    evidence_html = "<p>No locked evidence supplied.</p>"
    if evidence is not None:
        evidence_html = (
            "<table><tbody>"
            '<tr><th scope="row">Binding SHA-256</th>'
            f"<td><code>{escape(evidence.binding_sha256(), quote=True)}</code></td></tr>"
            '<tr><th scope="row">Calibration artifact SHA-256</th>'
            f"<td><code>{escape(evidence.calibration_artifact_sha256, quote=True)}</code>"
            "</td></tr>"
            f'<tr><th scope="row">Alpha</th><td>{evidence.alpha}</td></tr>'
            '<tr><th scope="row">Thresholds (positive / negative)</th>'
            f"<td>{evidence.positive_threshold} / {evidence.negative_threshold}</td></tr>"
            "</tbody></table>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 4 preflight</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>RetinaEntail-DR Stage 4 preflight</h1>
  <p><strong>Status:</strong> <code>{status}</code></p>
  <p>This zero-I/O report never authorizes data access. Prerequisites verified:
  false; real data read: false; sample count: 0; test tuning prohibited: true.</p>
  <table><tbody>
    <tr><th scope="row">Dataset</th><td>{dataset_id}</td></tr>
    <tr><th scope="row">Exact version</th><td>{dataset_version}</td></tr>
    <tr><th scope="row">Approved purposes</th><td>{purposes}</td></tr>
    {rows}
  </tbody></table>
  <h2>Locked evidence summary</h2>
  {evidence_html}
  <h2>Reasons</h2>
  <ul>{reasons}</ul>
</body>
</html>
"""


def _interval_text(interval: BinomialProportion) -> str:
    if interval.estimate is None:
        return "— (zero denominator)"
    assert interval.lower is not None
    assert interval.upper is not None
    return f"{interval.estimate:.6f} [{interval.lower:.6f}, {interval.upper:.6f}]"


def _state_rows(report: Stage4EvaluationReport) -> str:
    return "\n".join(
        "<tr>"
        f"<td><code>{escape(state.value, quote=True)}</code></td>"
        f"<td>{report.state_counts[state]}</td>"
        f"<td>{report.test_case_count}</td>"
        f"<td>{escape(_interval_text(report.state_proportions[state]), quote=True)}</td>"
        "</tr>"
        for state in EntailmentState
    )


def _summary_rows(report: Stage4EvaluationReport) -> str:
    rows = (
        ("Input records", report.input_record_count, None),
        ("Accepted test cases", report.test_case_count, None),
        ("Rejected records", report.rejected_record_count, None),
        ("Evaluable", report.evaluable_count, report.evaluable_proportion),
        ("Unevaluable", report.unevaluable_count, report.unevaluable_proportion),
        ("Fail closed", report.fail_closed_count, report.fail_closed_proportion),
        (
            "Out of distribution",
            report.out_of_distribution_count,
            report.out_of_distribution_proportion,
        ),
        (
            "Both witness replays passed",
            report.both_witness_replays_passed_count,
            report.both_witness_replays_passed_proportion,
        ),
        (
            "One or more witness replays failed",
            report.one_or_more_witness_replays_failed_count,
            report.one_or_more_witness_replays_failed_proportion,
        ),
        ("Labeled references (UNKNOWN excluded)", report.labeled_reference_count, None),
    )
    rendered: list[str] = []
    for label, count, interval in rows:
        proportion = "—" if interval is None else _interval_text(interval)
        rendered.append(
            "<tr>"
            f'<th scope="row">{escape(label, quote=True)}</th>'
            f"<td>{count}</td>"
            f"<td>{escape(proportion, quote=True)}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def _mapping_rows(values: Mapping[_K, int], *, empty_message: str) -> str:
    if not values:
        return f'<tr><td colspan="2">{escape(empty_message, quote=True)}</td></tr>'
    return "\n".join(
        f"<tr><td><code>{escape(str(key), quote=True)}</code></td><td>{count}</td></tr>"
        for key, count in sorted(values.items(), key=lambda item: str(item[0]))
    )


def render_stage4_evaluation_html(report: Stage4EvaluationReport) -> str:
    """Render standalone static HTML with every dynamic value escaped."""

    report = _require_report(report)
    locked = report.locked_contract
    aggregation_reasons = (
        "\n".join(
            f"<li><code>{escape(reason.value, quote=True)}</code></li>"
            for reason in report.aggregation_failure_reasons
        )
        or "<li>None.</li>"
    )
    label_rows = _mapping_rows(
        {label.value: count for label, count in report.reference_label_counts.items()},
        empty_message="No labels.",
    )
    failure_rows = _mapping_rows(
        report.failure_reason_counts,
        empty_message="No test-case fail-closed reasons.",
    )
    rejected_rows = _mapping_rows(
        report.rejected_reason_counts,
        empty_message="No rejected records.",
    )
    safe_schema = escape(report.schema_version, quote=True)
    safe_status = escape(report.status.value, quote=True)
    safe_scope = escape(report.scope.value, quote=True)
    safe_interval = escape(report.interval_method, quote=True)
    safe_artifact_hash = escape(locked.calibration_artifact_hash, quote=True)
    safe_config_hash = escape(locked.config_hash, quote=True)
    safe_data_hash = escape(locked.data_hash, quote=True)
    safe_model_hash = escape(locked.model_hash, quote=True)
    calibration_counts = (
        f"{locked.positive_sample_count} / {locked.negative_sample_count} / "
        f"{locked.total_sample_count}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 4 locked test audit</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
    .warning {{ border: 2px solid #b66; padding: .8rem; }}
  </style>
</head>
<body>
  <h1>RetinaEntail-DR Stage 4 locked test audit</h1>
  <p class="warning"><strong>Research audit only.</strong> These descriptive Wilson
  intervals are not a clinical claim, performance guarantee, or medical-device result.</p>
  <table>
    <tbody>
      <tr><th scope="row">Schema</th><td><code>{safe_schema}</code></td></tr>
      <tr><th scope="row">Status</th><td><code>{safe_status}</code></td></tr>
      <tr><th scope="row">Scope</th><td><code>{safe_scope}</code></td></tr>
      <tr><th scope="row">Split</th><td><code>test</code></td></tr>
      <tr><th scope="row">Clinical claim</th><td>false</td></tr>
      <tr><th scope="row">Alpha</th><td>{locked.alpha}</td></tr>
      <tr><th scope="row">Positive threshold</th><td>{locked.positive_threshold}</td></tr>
      <tr><th scope="row">Negative threshold</th><td>{locked.negative_threshold}</td></tr>
      <tr>
        <th scope="row">Calibration counts (positive / negative / total)</th>
        <td>{calibration_counts}</td>
      </tr>
      <tr>
        <th scope="row">Calibration artifact SHA-256</th>
        <td><code>{safe_artifact_hash}</code></td>
      </tr>
      <tr><th scope="row">Config SHA-256</th><td><code>{safe_config_hash}</code></td></tr>
      <tr><th scope="row">Data SHA-256</th><td><code>{safe_data_hash}</code></td></tr>
      <tr><th scope="row">Model SHA-256</th><td><code>{safe_model_hash}</code></td></tr>
      <tr>
        <th scope="row">Interval</th>
        <td><code>{safe_interval}</code>, confidence={report.confidence_level}</td>
      </tr>
    </tbody>
  </table>
  <h2>Audit counts</h2>
  <table><thead><tr><th>Measure</th><th>Count</th><th>Proportion and interval</th></tr></thead>
    <tbody>{_summary_rows(report)}</tbody>
  </table>
  <h2>Four-state distribution</h2>
  <table><thead><tr>
    <th>State</th><th>Count</th><th>Denominator</th><th>Proportion and interval</th>
  </tr></thead>
    <tbody>{_state_rows(report)}</tbody>
  </table>
  <h2>Reference labels</h2>
  <p><code>UNKNOWN</code> is retained as unknown and is never counted as lesion-absent.</p>
  <table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{label_rows}</tbody></table>
  <h2>Test-case fail-closed reasons</h2>
  <table><thead><tr><th>Reason</th><th>Cases</th></tr></thead><tbody>{failure_rows}</tbody></table>
  <h2>Rejected-input reasons</h2>
  <table><thead><tr><th>Reason</th><th>Records</th></tr></thead><tbody>{rejected_rows}</tbody></table>
  <h2>Aggregation failures</h2>
  <ul>{aggregation_reasons}</ul>
</body>
</html>
"""


def write_stage4_evaluation_reports(
    report: Stage4EvaluationReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    """Preflight and write a matching JSON/HTML pair without overwriting differences."""

    report = _require_report(report)
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    json_path = boundary.require_safe(output_dir / "stage4-evaluation.json")
    html_path = boundary.require_safe(output_dir / "stage4-evaluation.html")
    json_payload = render_stage4_evaluation_json(report).encode("utf-8")
    html_payload = render_stage4_evaluation_html(report).encode("utf-8")
    _preflight(json_path, json_payload)
    _preflight(html_path, html_payload)
    write_bytes_fail_safe(json_path, json_payload)
    write_bytes_fail_safe(html_path, html_payload)
    return json_path, html_path


def write_stage4_preflight_reports(
    report: Stage4PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    """Preflight and write the path-free Stage 4 JSON/HTML preflight pair."""

    report = _require_preflight(report)
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    json_path = boundary.require_safe(output_dir / "stage4-preflight.json")
    html_path = boundary.require_safe(output_dir / "stage4-preflight.html")
    json_payload = render_stage4_preflight_json(report).encode("utf-8")
    html_payload = render_stage4_preflight_html(report).encode("utf-8")
    _preflight(json_path, json_payload)
    _preflight(html_path, html_payload)
    write_bytes_fail_safe(json_path, json_payload)
    write_bytes_fail_safe(html_path, html_payload)
    return json_path, html_path


def _require_report(value: object) -> Stage4EvaluationReport:
    if not isinstance(value, Stage4EvaluationReport):
        raise TypeError("report must be a Stage4EvaluationReport")
    # Re-run invariant checks in case an attacker bypassed frozen dataclass guards.
    value.__post_init__()
    return value


def _require_preflight(value: object) -> Stage4PreflightReport:
    if not isinstance(value, Stage4PreflightReport):
        raise TypeError("report must be a Stage4PreflightReport")
    value.__post_init__()
    return value


def _preflight(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")


__all__ = [
    "render_stage4_evaluation_html",
    "render_stage4_evaluation_json",
    "render_stage4_preflight_html",
    "render_stage4_preflight_json",
    "write_stage4_evaluation_reports",
    "write_stage4_preflight_reports",
]
