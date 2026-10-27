from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage8_gate import Stage8PreflightReport, assess_stage8_preflight
from retinaentail.reporting.stage8 import (
    render_stage8_preflight_html,
    render_stage8_preflight_json,
    write_stage8_preflight_reports,
)


def _blocked_report() -> Stage8PreflightReport:
    return assess_stage8_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_stage7_package_capability_provided=False,
        fresh_root_run_capability_provided=False,
        fixed_seed_preregistered=False,
        degradation_replay_set_preregistered=False,
        inclusion_rules_preregistered=False,
        stability_metrics_preregistered=False,
        locked_contract_provided=False,
    )


def test_stage8_preflight_reports_are_path_free_and_deterministic(tmp_path: Path) -> None:
    report = _blocked_report()
    rendered_json = render_stage8_preflight_json(report)
    rendered_html = render_stage8_preflight_html(report)
    payload = json.loads(rendered_json)

    assert payload["status"] == "BLOCKED"
    assert payload["accepts_paths"] is False
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False
    assert "COMPLETE_REAL_STAGE7_PACKAGE_CAPABILITY_NOT_PROVIDED" in payload["reasons"]
    assert "accepts no path" in rendered_html
    assert "C:\\secret-dataset" not in rendered_json
    assert "C:\\secret-dataset" not in rendered_html

    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    paths = write_stage8_preflight_reports(
        report,
        tmp_path,
        boundary=boundary,
    )
    assert paths[0].read_text(encoding="utf-8") == rendered_json
    assert paths[1].read_text(encoding="utf-8") == rendered_html
    assert (
        write_stage8_preflight_reports(
            report,
            tmp_path,
            boundary=boundary,
        )
        == paths
    )

    paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage8_preflight_reports(
            report,
            tmp_path,
            boundary=boundary,
        )
