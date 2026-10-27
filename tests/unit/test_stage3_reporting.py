from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.calibration.stage3_gate import Stage3PreflightReport, assess_stage3_preflight
from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.reporting.stage3 import (
    render_stage3_preflight_html,
    write_stage3_bundle,
    write_stage3_preflight_reports,
)


def _blocked_report() -> Stage3PreflightReport:
    return assess_stage3_preflight(
        dataset_id=None,
        dataset_version=None,
        review_record_provided=False,
        dataset_root_provided=False,
        adapter_spec_provided=False,
    )


def test_stage3_preflight_reports_are_path_free_escaped_and_repeatable(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "reports"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    json_path, html_path = write_stage3_preflight_reports(
        _blocked_report(), output_dir, boundary=boundary
    )
    first_json = json_path.read_bytes()
    first_html = html_path.read_bytes()

    write_stage3_preflight_reports(_blocked_report(), output_dir, boundary=boundary)

    payload = json.loads(first_json)
    assert payload["status"] == "BLOCKED"
    assert payload["sample_count"] == 0
    assert payload["real_data_read"] is False
    assert str(tmp_path) not in first_json.decode("utf-8")
    assert str(tmp_path) not in first_html.decode("utf-8")
    assert render_stage3_preflight_html(_blocked_report()).startswith("<!doctype html>")


def test_stage3_bundle_rejects_escape_and_preflights_all_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "second.json").write_bytes(b"different")
    boundary = AuditWriteBoundary(allowed_root=tmp_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage3_bundle(
            output_dir,
            {"first.json": b"new", "second.json": b"expected"},
            boundary=boundary,
        )
    assert not (output_dir / "first.json").exists()

    with pytest.raises(ValueError, match="normalized relative POSIX"):
        write_stage3_bundle(
            output_dir,
            {"../escape.json": b"no"},
            boundary=boundary,
        )
