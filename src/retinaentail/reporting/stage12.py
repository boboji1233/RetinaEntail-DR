"""Stage 12 JSON and escaped HTML reports."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage12_gate import Stage12PreflightReport
from retinaentail.witnesses.replay import write_bytes_fail_safe


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_stage12_preflight_json(report: Stage12PreflightReport) -> str:
    return _json(report.to_dict())


def render_stage12_preflight_html(report: Stage12PreflightReport) -> str:
    payload = report.to_dict()
    raw_reasons = payload["reasons"]
    if not isinstance(raw_reasons, list):
        raise ValueError("Stage 12 preflight reasons must be a list")
    reasons = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in raw_reasons)
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Stage 12 preflight</title>'
        "</head><body><h1>Stage 12 production signer ceremony preflight</h1>"
        f"<p>Status: <strong>{html.escape(str(payload['status']))}</strong></p>"
        f"<p>Dataset root probed: {str(payload['dataset_root_probed']).lower()}</p>"
        f"<p>Real data read: {str(payload['real_data_read']).lower()}</p>"
        f"<ul>{reasons}</ul></body></html>\n"
    )


def write_stage12_preflight_reports(
    report: Stage12PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    json_path = output_dir / "preflight.json"
    html_path = output_dir / "preflight.html"
    boundary.require_safe(json_path)
    boundary.require_safe(html_path)
    write_bytes_fail_safe(json_path, render_stage12_preflight_json(report).encode())
    write_bytes_fail_safe(html_path, render_stage12_preflight_html(report).encode())
    return json_path, html_path


def render_stage12_audit_json(report: Mapping[str, object]) -> str:
    return _json(dict(report))


def render_stage12_audit_html(report: Mapping[str, object]) -> str:
    tamper = report.get("tamper_audit")
    results = tamper.get("results", []) if isinstance(tamper, Mapping) else []
    rows: list[str] = []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, Mapping):
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(item.get('scenario', '')))}</td>"
                    f"<td>{html.escape(str(item.get('expected_code', '')))}</td>"
                    f"<td>{html.escape(str(item.get('observed_code', '')))}</td>"
                    f"<td>{html.escape(str(item.get('rejected', False)))}</td>"
                    "</tr>"
                )
    comparison = report.get("recovered_runner_comparison")
    comparison_status = comparison.get("status") if isinstance(comparison, Mapping) else "UNKNOWN"
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Stage 12 audit</title>'
        "<style>body{font-family:system-ui,sans-serif}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:.35rem}</style></head><body>"
        "<h1>Stage 12 local random-signer compromise/recovery audit</h1>"
        f"<p>Status: <strong>{html.escape(str(report.get('status', 'UNKNOWN')))}</strong></p>"
        f"<p>Recovered runner comparison: {html.escape(str(comparison_status))}</p>"
        "<p>Independent processes: "
        f"{html.escape(str(report.get('independent_process_count', 0)))}</p>"
        "<table><thead><tr><th>Scenario</th><th>Expected</th><th>Observed</th>"
        f"<th>Rejected</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<p>Random private keys remained process-local. This is not HSM-backed, production "
        "authentication, RFC 6960 OCSP, a two-physical-host result, or clinical evidence.</p>"
        "</body></html>\n"
    )
