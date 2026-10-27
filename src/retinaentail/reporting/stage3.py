"""Deterministic, path-safe Stage 3 report and artifact-bundle writers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from pathlib import Path, PurePosixPath

from retinaentail.calibration.stage3_gate import Stage3PreflightReport
from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.witnesses.replay import write_bytes_fail_safe


def render_stage3_preflight_json(report: Stage3PreflightReport) -> str:
    if not isinstance(report, Stage3PreflightReport):
        raise TypeError("report must be Stage3PreflightReport")
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


def render_stage3_preflight_html(report: Stage3PreflightReport) -> str:
    if not isinstance(report, Stage3PreflightReport):
        raise TypeError("report must be Stage3PreflightReport")
    reasons = (
        "\n".join(
            f"<li><code>{escape(reason.value, quote=True)}</code></li>" for reason in report.reasons
        )
        or "<li>None — inputs are only ready for the Stage 2 Gate.</li>"
    )
    dataset = escape(report.dataset_id or "—", quote=True)
    version = escape(report.dataset_version or "—", quote=True)
    status = escape(report.status.value, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetinaEntail-DR Stage 3 preflight</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #8888; padding: .4rem .6rem; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>RetinaEntail-DR Stage 3 preflight</h1>
  <p><strong>Status:</strong> <code>{status}</code></p>
  <table>
    <tbody>
      <tr><th>Dataset</th><td>{dataset}</td></tr>
      <tr><th>Exact version</th><td>{version}</td></tr>
      <tr><th>Real data read</th><td>false</td></tr>
      <tr><th>Sample count</th><td>0</td></tr>
      <tr><th>Authorizes access</th><td>false</td></tr>
    </tbody>
  </table>
  <h2>Reasons</h2>
  <ul>{reasons}</ul>
  <p>This presence check is not a license decision or a dataset-access capability.</p>
</body>
</html>
"""


def _normalized_bundle_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("bundle artifact names must be normalized relative POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("bundle artifact names must be normalized relative POSIX paths")
    return value


def _preflight_target(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")


def write_stage3_bundle(
    output_dir: Path,
    artifacts: Mapping[str, bytes],
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, ...]:
    """Preflight every artifact before creating any file in a Stage 3 package."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(boundary, AuditWriteBoundary):
        raise TypeError("boundary must be an AuditWriteBoundary")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("artifacts must be a non-empty mapping")
    prepared: list[tuple[Path, bytes]] = []
    for raw_name, payload in artifacts.items():
        name = _normalized_bundle_name(raw_name)
        if not isinstance(payload, bytes):
            raise TypeError("artifact payloads must be bytes")
        candidate = boundary.require_safe(output_dir.joinpath(*PurePosixPath(name).parts))
        prepared.append((candidate, payload))
    prepared.sort(key=lambda item: item[0].as_posix())
    for path, payload in prepared:
        _preflight_target(path, payload)
    for path, payload in prepared:
        write_bytes_fail_safe(path, payload)
    return tuple(path for path, _ in prepared)


def write_stage3_preflight_reports(
    report: Stage3PreflightReport,
    output_dir: Path,
    *,
    boundary: AuditWriteBoundary,
) -> tuple[Path, Path]:
    paths = write_stage3_bundle(
        output_dir,
        {
            "preflight.json": render_stage3_preflight_json(report).encode("utf-8"),
            "preflight.html": render_stage3_preflight_html(report).encode("utf-8"),
        },
        boundary=boundary,
    )
    by_name = {path.name: path for path in paths}
    return by_name["preflight.json"], by_name["preflight.html"]


__all__ = [
    "render_stage3_preflight_html",
    "render_stage3_preflight_json",
    "write_stage3_bundle",
    "write_stage3_preflight_reports",
]
