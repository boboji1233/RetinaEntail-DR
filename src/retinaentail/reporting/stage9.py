"""Deterministic JSON/HTML reporting for Stage 9 preflight and package diffs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from pathlib import Path

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage9_gate import Stage9PreflightReport
from retinaentail.witnesses.replay import write_bytes_fail_safe


def _json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_stage9_preflight_json(report: Stage9PreflightReport) -> str:
    if not isinstance(report, Stage9PreflightReport):
        raise TypeError("report must be Stage9PreflightReport")
    return _json_text(report.to_dict())


def render_stage9_preflight_html(report: Stage9PreflightReport) -> str:
    if not isinstance(report, Stage9PreflightReport):
        raise TypeError("report must be Stage9PreflightReport")
    reasons = "".join(f"<li><code>{escape(item.value)}</code></li>" for item in report.reasons)
    if not reasons:
        reasons = "<li>Presence checks complete; live capability replay is still required.</li>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Stage 9 preflight</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.45}}code{{overflow-wrap:anywhere}}</style>
</head><body><h1>RetinaEntail-DR Stage 9 preflight</h1>
<p>Status: <strong>{escape(report.status.value)}</strong></p>
<p>This presence-only gate accepts no path, reads no environment, probes no dataset root,
and grants no authority to detached PASS JSON.</p><ul>{reasons}</ul>
<p>Dataset root probed: <strong>false</strong>; real data read: <strong>false</strong>.</p>
</body></html>
"""


def write_stage9_preflight_reports(
    report: Stage9PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    if not isinstance(output_dir, Path) or not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("output_dir and boundary types are invalid")
    target = boundary.require_safe(output_dir)
    json_path = boundary.require_safe(target / "preflight.json")
    html_path = boundary.require_safe(target / "preflight.html")
    write_bytes_fail_safe(json_path, render_stage9_preflight_json(report).encode("utf-8"))
    write_bytes_fail_safe(html_path, render_stage9_preflight_html(report).encode("utf-8"))
    return json_path, html_path


def render_stage9_audit_json(report: Mapping[str, object]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    return _json_text(dict(report))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Stage 9 report contains a malformed mapping")
    return value


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Stage 9 report contains a malformed list")
    return value


def _comparison_differences(value: object) -> str:
    differences = _list(_mapping(value).get("differences"))
    return escape(", ".join(str(item) for item in differences), quote=True) or "—"


def render_stage9_audit_html(report: Mapping[str, object]) -> str:
    """Render the package diff without trusting identifiers or hashes as markup."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    diff = _mapping(report.get("diff"))
    session_a = _mapping(diff.get("session_a"))
    session_b = _mapping(diff.get("session_b"))
    metrics = _mapping(diff.get("metric_intervals"))
    comparisons = _list(diff.get("case_comparisons"))
    global_differences = _list(diff.get("global_differences"))
    metric_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(name), quote=True)}</code></td>"
        f"<td>{escape(str(_mapping(value).get('numerator')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(value).get('denominator')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(value).get('estimate')), quote=True)}</td>"
        f"<td>[{escape(str(_mapping(value).get('lower')), quote=True)}, "
        f"{escape(str(_mapping(value).get('upper')), quote=True)}]</td>"
        "</tr>"
        for name, value in sorted(metrics.items())
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(str(_mapping(item).get('case_id')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('run_id')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('session_a_state')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('session_b_state')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('effective_state')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('session_a_positive_score')), quote=True)} / "
        f"{escape(str(_mapping(item).get('session_a_negative_score')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('session_b_positive_score')), quote=True)} / "
        f"{escape(str(_mapping(item).get('session_b_negative_score')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('dual_witness_content_match')), quote=True)}</td>"
        f"<td>{escape(str(_mapping(item).get('dual_witness_replay_match')), quote=True)}</td>"
        "<td>"
        f"{_comparison_differences(item)}"
        "</td>"
        "</tr>"
        for item in comparisons
    )
    differences = ", ".join(escape(str(item), quote=True) for item in global_differences) or "none"
    status = escape(str(report.get("status")), quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Stage 9 independent-session diff</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4}}
table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #8888;padding:.4rem;vertical-align:top}}
code{{overflow-wrap:anywhere}}
</style>
</head><body><h1>RetinaEntail-DR Stage 9 independent-process/session diff</h1>
<p>Status: <strong>{status}</strong>. Clinical claim: false.</p>
<p>Deterministic differences fail closed. Global differences: <code>{differences}</code>.</p>
<h2>Session provenance</h2><table><thead><tr><th>Session</th><th>Producer PID</th>
<th>Session nonce SHA-256</th><th>Environment lock</th><th>Stage 8 package</th></tr></thead><tbody>
<tr><td>A</td><td>{escape(str(session_a.get("producer_process_id")))}</td>
<td><code>{escape(str(session_a.get("session_nonce_sha256")))}</code></td>
<td><code>{escape(str(session_a.get("environment_lock_sha256")))}</code></td>
<td><code>{escape(str(session_a.get("package_sha256")))}</code></td></tr>
<tr><td>B</td><td>{escape(str(session_b.get("producer_process_id")))}</td>
<td><code>{escape(str(session_b.get("session_nonce_sha256")))}</code></td>
<td><code>{escape(str(session_b.get("environment_lock_sha256")))}</code></td>
<td><code>{escape(str(session_b.get("package_sha256")))}</code></td></tr></tbody></table>
<h2>Wilson intervals</h2><table><thead><tr><th>Metric</th><th>Numerator</th>
<th>Denominator</th><th>Estimate</th><th>95% interval</th></tr></thead>
<tbody>{metric_rows}</tbody></table>
<h2>Per-case/run deterministic comparison</h2><table><thead><tr><th>Case</th><th>Run</th>
<th>A state</th><th>B state</th><th>Effective state</th><th>A positive / negative score</th>
<th>B positive / negative score</th><th>Dual witness content</th><th>Dual witness replay</th>
<th>Differences</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<p>No threshold re-estimation, test tuning, sample replacement, network, GPU, model training,
or real-data access was performed by this synthetic package.</p></body></html>
"""
