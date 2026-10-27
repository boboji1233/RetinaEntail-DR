"""JSON and escaped HTML rendering for Stage 11 reports."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage11_gate import Stage11PreflightReport
from retinaentail.witnesses.replay import write_bytes_fail_safe


def _json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_stage11_preflight_json(report: Stage11PreflightReport) -> str:
    if not isinstance(report, Stage11PreflightReport):
        raise TypeError("report must be Stage11PreflightReport")
    return _json_text(report.to_dict())


def render_stage11_preflight_html(report: Stage11PreflightReport) -> str:
    if not isinstance(report, Stage11PreflightReport):
        raise TypeError("report must be Stage11PreflightReport")
    data = report.to_dict()
    raw_reasons = data["reasons"]
    if not isinstance(raw_reasons, list):
        raise ValueError("Stage 11 preflight reasons must be a list")
    reasons = "".join(f"<li>{html.escape(str(value))}</li>" for value in raw_reasons)
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Stage 11 preflight</title>'
        "</head><body><h1>Stage 11 REAL cross-host preflight</h1>"
        f"<p>Status: <strong>{html.escape(str(data['status']))}</strong></p>"
        f"<p>Dataset root probed: {str(data['dataset_root_probed']).lower()}</p>"
        f"<p>Real data read: {str(data['real_data_read']).lower()}</p>"
        f"<ul>{reasons}</ul></body></html>\n"
    )


def write_stage11_preflight_reports(
    report: Stage11PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    json_path = output_dir / "preflight.json"
    html_path = output_dir / "preflight.html"
    boundary.require_safe(json_path)
    boundary.require_safe(html_path)
    write_bytes_fail_safe(json_path, render_stage11_preflight_json(report).encode("utf-8"))
    write_bytes_fail_safe(html_path, render_stage11_preflight_html(report).encode("utf-8"))
    return json_path, html_path


def render_stage11_audit_json(report: Mapping[str, object]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    return _json_text(dict(report))


def render_stage11_audit_html(report: Mapping[str, object]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    comparison = report.get("baseline_comparison")
    comparison_map = comparison if isinstance(comparison, Mapping) else {}
    tamper = report.get("tamper_audit")
    tamper_map = tamper if isinstance(tamper, Mapping) else {}
    rows: list[str] = []
    raw_results = tamper_map.get("results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('scenario', '')))}</td>"
                f"<td>{html.escape(str(item.get('expected_code', '')))}</td>"
                f"<td>{html.escape(str(item.get('observed_code', '')))}</td>"
                f"<td>{html.escape(str(item.get('rejected', False)))}</td>"
                "</tr>"
            )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Stage 11 audit</title>'
        "<style>body{font-family:system-ui,sans-serif}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:.35rem;text-align:left}</style></head><body>"
        "<h1>Stage 11 local asymmetric-signature and rotation audit</h1>"
        f"<p>Status: <strong>{html.escape(str(report.get('status', 'UNKNOWN')))}</strong></p>"
        f"<p>Physical hosts: {html.escape(str(report.get('physical_host_count', '')))}; "
        "cross-host tested: "
        f"{html.escape(str(report.get('cross_physical_host_tested', False)))}</p>"
        f"<p>Frozen comparison: {html.escape(str(comparison_map.get('status', 'UNKNOWN')))}</p>"
        "<table><thead><tr><th>Tamper scenario</th><th>Expected</th><th>Observed</th>"
        f"<th>Rejected</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<p>The deterministic private-key derivation and trust anchor are public fixture material. "
        "This package is not production authentication, not a two-host result, "
        "and not clinical evidence."
        "</p></body></html>\n"
    )
