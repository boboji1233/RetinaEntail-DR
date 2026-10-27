"""Deterministic JSON and escaped static-HTML rendering for dataset audits."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

from retinaentail.data.audit import AuditIssue, AuditResult
from retinaentail.data.stage2_models import DatasetManifestV2
from retinaentail.witnesses.replay import write_bytes_fail_safe


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditWriteBoundary:
    """Explicit allow/deny boundary required for every report filesystem write."""

    allowed_root: Path
    forbidden_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_root, Path):
            raise TypeError("allowed_root must be a pathlib.Path")
        if any(not isinstance(path, Path) for path in self.forbidden_roots):
            raise TypeError("forbidden_roots must contain pathlib.Path values")
        object.__setattr__(self, "allowed_root", self.allowed_root.absolute())
        object.__setattr__(
            self,
            "forbidden_roots",
            tuple(path.absolute() for path in self.forbidden_roots),
        )

    def require_safe(self, path: Path) -> Path:
        candidate = path.absolute()
        allowed = self.allowed_root
        try:
            candidate.relative_to(allowed)
            resolved_candidate = candidate.resolve(strict=False)
            resolved_candidate.relative_to(allowed.resolve(strict=False))
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("audit report path escapes its allowed output root") from error
        for forbidden in self.forbidden_roots:
            resolved_forbidden = forbidden.resolve(strict=False)
            for child, parent in (
                (candidate, forbidden),
                (forbidden, candidate),
                (resolved_candidate, resolved_forbidden),
                (resolved_forbidden, resolved_candidate),
            ):
                try:
                    child.relative_to(parent)
                except ValueError:
                    continue
                raise ValueError("audit report path overlaps a forbidden data root")
        return candidate


def render_audit_json(result: AuditResult) -> str:
    """Render a stable JSON report without absolute dataset paths."""

    result = _require_result(result)
    return (
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_manifest_csv(manifest: DatasetManifestV2) -> str:
    """Render a stable one-row-per-label manifest review table."""

    if not isinstance(manifest, DatasetManifestV2):
        raise TypeError("manifest must be a DatasetManifestV2")
    fields = (
        "schema_version",
        "dataset_id",
        "dataset_version",
        "relative_path",
        "sha256",
        "byte_size",
        "width",
        "height",
        "channels",
        "image_format",
        "source_image_id",
        "patient_id",
        "eye_id",
        "followup_id",
        "intervention_pair_id",
        "split",
        "label_semantic",
        "label_availability",
        "label_relative_path",
        "label_sha256",
        "label_byte_size",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in manifest.records:
        base: dict[str, object] = {
            "schema_version": record.schema_version,
            "dataset_id": record.dataset_id,
            "dataset_version": manifest.dataset_version,
            "relative_path": record.relative_path,
            "sha256": record.sha256,
            "byte_size": record.byte_size,
            "width": record.width,
            "height": record.height,
            "channels": record.channels,
            "image_format": record.image_format,
            "source_image_id": record.source_image_id,
            "patient_id": record.patient_id,
            "eye_id": record.eye_id,
            "followup_id": record.followup_id,
            "intervention_pair_id": record.intervention_pair_id,
            "split": record.split,
        }
        labels = record.labels or (None,)
        for label in labels:
            row = dict(base)
            if label is not None:
                row.update(
                    {
                        "label_semantic": label.semantic.value,
                        "label_availability": label.availability.value,
                        "label_relative_path": label.relative_path,
                        "label_sha256": label.sha256,
                        "label_byte_size": label.byte_size,
                    }
                )
            writer.writerow(row)
    return stream.getvalue()


def _summary_rows(result: AuditResult) -> str:
    summary = result.summary
    rows = (
        ("Dataset", result.dataset_id or "—"),
        ("Dataset version", result.dataset_version or "—"),
        ("Manifest SHA-256", result.manifest_sha256 or "—"),
        ("Records", summary.record_count),
        ("Images checked", summary.image_files_checked),
        ("Labels checked", summary.label_files_checked),
        ("Masks decoded", summary.mask_files_decoded),
        ("Unknown labels", summary.unknown_label_count),
    )
    return "\n".join(
        f'<tr><th scope="row">{escape(label)}</th><td>{escape(str(value), quote=True)}</td></tr>'
        for label, value in rows
    )


def _issue_row(issue: AuditIssue) -> str:
    relative_path = issue.relative_path or "—"
    related = ", ".join(issue.related_paths) or "—"
    details = "; ".join(f"{name}={value}" for name, value in issue.details) or "—"
    severity = escape(issue.severity.value, quote=True)
    return (
        f'<tr class="severity-{severity.casefold()}">'
        f"<td>{severity}</td>"
        f"<td><code>{escape(issue.code, quote=True)}</code></td>"
        f"<td>{escape(relative_path, quote=True)}</td>"
        f"<td>{escape(related, quote=True)}</td>"
        f"<td>{escape(issue.message, quote=True)}</td>"
        f"<td>{escape(details, quote=True)}</td>"
        "</tr>"
    )


def render_audit_html(result: AuditResult, *, title: str = "Dataset audit report") -> str:
    """Render a standalone report; every dynamic string is HTML-escaped."""

    result = _require_result(result)
    if not title.strip():
        raise ValueError("title must be a non-empty string")
    if any(character in title for character in "\\/\r\n"):
        raise ValueError("title must not contain filesystem paths or line breaks")
    safe_title = escape(title.strip(), quote=True)
    status = escape(result.status.value, quote=True)
    issue_rows = "\n".join(_issue_row(issue) for issue in result.issues)
    if not issue_rows:
        issue_rows = '<tr><td colspan="6">No issues.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-block: 1rem 2rem; }}
    th, td {{ border: 1px solid #8888; padding: .45rem .6rem; text-align: left; }}
    thead th {{ position: sticky; top: 0; background: Canvas; }}
    code {{ overflow-wrap: anywhere; }}
    .status {{ font-weight: 700; }}
    .severity-critical, .severity-error {{ background: #d33a; }}
    .severity-warning {{ background: #e9a23baa; }}
    .status-blocked {{ background: #e9a23baa; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <p>Schema: <code>{escape(result.schema_version, quote=True)}</code></p>
  <p class="status status-{status.casefold()}">Overall status: {status}</p>
  <h2>Summary</h2>
  <table>
    <tbody>
{_summary_rows(result)}
    </tbody>
  </table>
  <h2>Issues</h2>
  <table>
    <thead>
      <tr>
        <th>Severity</th><th>Code</th><th>Path</th>
        <th>Related paths</th><th>Message</th><th>Details</th>
      </tr>
    </thead>
    <tbody>
{issue_rows}
    </tbody>
  </table>
</body>
</html>
"""


def _require_result(value: object) -> AuditResult:
    if not isinstance(value, AuditResult):
        raise TypeError("result must be an AuditResult")
    return value


def _require_output_path(path: object) -> Path:
    if not isinstance(path, Path):
        raise TypeError("output path must be a pathlib.Path")
    return path


def write_audit_json(
    result: AuditResult,
    output_path: Path,
    *,
    boundary: AuditWriteBoundary,
) -> Path:
    """Write a UTF-8 JSON report to an explicitly supplied output path."""

    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    target = boundary.require_safe(_require_output_path(output_path))
    write_bytes_fail_safe(target, render_audit_json(result).encode("utf-8"))
    return target


def write_audit_html(
    result: AuditResult,
    output_path: Path,
    *,
    boundary: AuditWriteBoundary,
    title: str = "Dataset audit report",
) -> Path:
    """Write a standalone escaped HTML report to an explicit output path."""

    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    target = boundary.require_safe(_require_output_path(output_path))
    write_bytes_fail_safe(target, render_audit_html(result, title=title).encode("utf-8"))
    return target


def write_manifest_csv(
    manifest: DatasetManifestV2,
    output_path: Path,
    *,
    boundary: AuditWriteBoundary,
) -> Path:
    """Write the root-free manifest review table within the audit boundary."""

    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    target = boundary.require_safe(_require_output_path(output_path))
    write_bytes_fail_safe(target, render_manifest_csv(manifest).encode("utf-8"))
    return target


def _preflight_artifact(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")


def write_audit_reports(
    result: AuditResult,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
    title: str = "Dataset audit report",
) -> tuple[Path, Path]:
    """Preflight and publish the matching JSON/HTML pair within one explicit boundary."""

    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    directory = _require_output_path(output_dir)
    json_path = boundary.require_safe(directory / "audit.json")
    html_path = boundary.require_safe(directory / "audit.html")
    json_payload = render_audit_json(result).encode("utf-8")
    html_payload = render_audit_html(result, title=title).encode("utf-8")
    _preflight_artifact(json_path, json_payload)
    _preflight_artifact(html_path, html_payload)
    write_bytes_fail_safe(json_path, json_payload)
    write_bytes_fail_safe(html_path, html_payload)
    return json_path, html_path


__all__ = [
    "AuditWriteBoundary",
    "render_audit_html",
    "render_audit_json",
    "render_manifest_csv",
    "write_audit_html",
    "write_audit_json",
    "write_audit_reports",
    "write_manifest_csv",
]
