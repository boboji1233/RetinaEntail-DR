from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage10_gate import Stage10PreflightReport, assess_stage10_preflight
from retinaentail.reporting.stage10 import (
    render_stage10_audit_html,
    render_stage10_preflight_html,
    render_stage10_preflight_json,
    write_stage10_preflight_reports,
)


def _blocked() -> Stage10PreflightReport:
    return assess_stage10_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage9_package_a_capability_present=False,
        complete_real_stage9_package_b_capability_present=False,
        fresh_runner_a_capability_present=False,
        fresh_runner_b_capability_present=False,
        two_approved_runners_confirmed=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
        signer_trust_policy_preregistered=False,
    )


def test_preflight_report_is_deterministic_fail_safe_and_non_overwriting(tmp_path: Path) -> None:
    report = _blocked()
    rendered_json = render_stage10_preflight_json(report)
    payload = json.loads(rendered_json)
    assert payload["status"] == "BLOCKED"
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    paths = write_stage10_preflight_reports(report, tmp_path, boundary=boundary)
    assert paths[0].read_text(encoding="utf-8") == rendered_json
    assert paths[1].read_text(encoding="utf-8") == render_stage10_preflight_html(report)
    paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage10_preflight_reports(report, tmp_path, boundary=boundary)


def test_audit_html_escapes_untrusted_case_values() -> None:
    report = {
        "status": "FAIL_CLOSED",
        "physical_host_count": 1,
        "cross_physical_host_claim": False,
        "diff": {
            "deterministic_file_mismatch_count": 1,
            "case_comparisons": [
                {
                    "case_id": "<case>",
                    "run_id": "run",
                    "runner_a_state": "OUT_OF_MODEL",
                    "runner_b_state": "OUT_OF_MODEL",
                    "effective_state": "OUT_OF_MODEL",
                    "differences": ["<SCORE>"],
                }
            ],
        },
    }
    rendered = render_stage10_audit_html(report)
    assert "<case>" not in rendered
    assert "&lt;case&gt;" in rendered
    assert "&lt;SCORE&gt;" in rendered
