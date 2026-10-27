"""JSON and escaped HTML rendering for Stage 10 reports."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage10_gate import Stage10PreflightReport
from retinaentail.witnesses.replay import write_bytes_fail_safe


def _json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_stage10_preflight_json(report: Stage10PreflightReport) -> str:
    if not isinstance(report, Stage10PreflightReport):
        raise TypeError("report must be Stage10PreflightReport")
    return _json_text(report.to_dict())


def render_stage10_preflight_html(report: Stage10PreflightReport) -> str:
    if not isinstance(report, Stage10PreflightReport):
        raise TypeError("report must be Stage10PreflightReport")
    data = report.to_dict()
    reason_values = data["reasons"]
    if not isinstance(reason_values, list):
        raise ValueError("Stage 10 preflight reasons must be a list")
    reasons = "".join(f"<li>{html.escape(str(value))}</li>" for value in reason_values)
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Stage 10 preflight</title>'
        "</head><body><h1>Stage 10 REAL preflight</h1>"
        f"<p>Status: <strong>{html.escape(str(data['status']))}</strong></p>"
        f"<p>Dataset root probed: {str(data['dataset_root_probed']).lower()}</p>"
        f"<p>Real data read: {str(data['real_data_read']).lower()}</p>"
        f"<ul>{reasons}</ul></body></html>\n"
    )


def write_stage10_preflight_reports(
    report: Stage10PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    json_path = output_dir / "preflight.json"
    html_path = output_dir / "preflight.html"
    boundary.require_safe(json_path)
    boundary.require_safe(html_path)
    write_bytes_fail_safe(json_path, render_stage10_preflight_json(report).encode("utf-8"))
    write_bytes_fail_safe(html_path, render_stage10_preflight_html(report).encode("utf-8"))
    return json_path, html_path


def render_stage10_audit_json(report: Mapping[str, object]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    return _json_text(dict(report))


def render_stage10_audit_html(report: Mapping[str, object]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    diff = report.get("diff")
    diff_map = diff if isinstance(diff, Mapping) else {}
    rows = []
    comparisons = diff_map.get("case_comparisons")
    if isinstance(comparisons, list):
        for item in comparisons:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('case_id', '')))}</td>"
                f"<td>{html.escape(str(item.get('run_id', '')))}</td>"
                f"<td>{html.escape(str(item.get('runner_a_state', '')))}</td>"
                f"<td>{html.escape(str(item.get('runner_b_state', '')))}</td>"
                f"<td>{html.escape(str(item.get('effective_state', '')))}</td>"
                f"<td>{html.escape(', '.join(str(v) for v in item.get('differences', [])))}</td>"
                "</tr>"
            )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Stage 10 audit</title>'
        "<style>body{font-family:system-ui,sans-serif}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:.35rem;text-align:left}</style></head><body>"
        "<h1>Stage 10 local dual-runner portability audit</h1>"
        f"<p>Status: <strong>{html.escape(str(report.get('status', 'UNKNOWN')))}</strong></p>"
        f"<p>Physical hosts: {html.escape(str(report.get('physical_host_count', '')))}; "
        f"cross-host claim: {html.escape(str(report.get('cross_physical_host_claim', False)))}</p>"
        "<p>Deterministic file mismatches: "
        f"{html.escape(str(diff_map.get('deterministic_file_mismatch_count', '')))}</p>"
        "<table><thead><tr><th>Case</th><th>Run</th><th>Runner A</th><th>Runner B</th>"
        f"<th>Effective</th><th>Differences</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<p>This synthetic fixture is not clinical evidence and not a "
        "cross-physical-host result.</p>"
        "</body></html>\n"
    )
