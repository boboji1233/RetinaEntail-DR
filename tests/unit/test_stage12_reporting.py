from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage12_gate import assess_stage12_preflight
from retinaentail.reporting.stage12 import (
    render_stage12_audit_html,
    render_stage12_preflight_html,
    render_stage12_preflight_json,
    write_stage12_preflight_reports,
)


def _blocked():  # type: ignore[no-untyped-def]
    return assess_stage12_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage11_package_a_capability_present=False,
        complete_real_stage11_package_b_capability_present=False,
        fresh_runner_a_process_capability_present=False,
        fresh_runner_b_process_capability_present=False,
        two_governance_approved_hardware_runners_verified=False,
        hsm_or_controlled_signer_boundary_verified=False,
        trusted_ca_chains_verified=False,
        trusted_timestamp_policy_verified=False,
        current_ocsp_status_verified=False,
        current_crl_status_verified=False,
        complete_wheel_lock_hashes_verified=False,
        hardware_identity_verified=False,
        frozen_baseline_contract_preregistered=False,
        compromise_recovery_drill_preregistered=False,
    )


def test_preflight_report_is_deterministic_path_safe_and_non_overwriting(tmp_path: Path) -> None:
    report = _blocked()
    rendered = render_stage12_preflight_json(report)
    assert json.loads(rendered)["status"] == "BLOCKED"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    paths = write_stage12_preflight_reports(report, tmp_path, boundary=boundary)
    assert paths[0].read_text(encoding="utf-8") == rendered
    assert paths[1].read_text(encoding="utf-8") == render_stage12_preflight_html(report)
    paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage12_preflight_reports(report, tmp_path, boundary=boundary)


def test_audit_html_escapes_values_and_disclaims_production() -> None:
    rendered = render_stage12_audit_html(
        {
            "status": "<FAIL_CLOSED>",
            "independent_process_count": 2,
            "recovered_runner_comparison": {"status": "SUCCESS"},
            "tamper_audit": {
                "results": [
                    {
                        "scenario": "<mutation>",
                        "expected_code": "<INVALID>",
                        "observed_code": "<INVALID>",
                        "rejected": True,
                    }
                ]
            },
        }
    )
    assert "<mutation>" not in rendered
    assert "&lt;mutation&gt;" in rendered
    assert "not HSM-backed" in rendered
    assert "not" in rendered and "production authentication" in rendered
