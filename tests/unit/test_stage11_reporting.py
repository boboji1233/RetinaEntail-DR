from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage11_gate import assess_stage11_preflight
from retinaentail.reporting.stage11 import (
    render_stage11_audit_html,
    render_stage11_preflight_html,
    render_stage11_preflight_json,
    write_stage11_preflight_reports,
)


def _blocked():  # type: ignore[no-untyped-def]
    return assess_stage11_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage10_package_a_capability_present=False,
        complete_real_stage10_package_b_capability_present=False,
        fresh_host_a_process_capability_present=False,
        fresh_host_b_process_capability_present=False,
        two_approved_physical_hosts_confirmed=False,
        production_asymmetric_trust_anchor_verified=False,
        unique_runner_leaf_keys_verified=False,
        certificate_chains_verified=False,
        revocation_status_current=False,
        trusted_time_policy_verified=False,
        complete_wheel_lock_hashes_verified=False,
        host_environment_policy_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    )


def test_preflight_rendering_is_deterministic_fail_safe_and_non_overwriting(
    tmp_path: Path,
) -> None:
    report = _blocked()
    rendered = render_stage11_preflight_json(report)
    assert json.loads(rendered)["status"] == "BLOCKED"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    paths = write_stage11_preflight_reports(report, tmp_path, boundary=boundary)
    assert paths[0].read_text(encoding="utf-8") == rendered
    assert paths[1].read_text(encoding="utf-8") == render_stage11_preflight_html(report)
    paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage11_preflight_reports(report, tmp_path, boundary=boundary)


def test_audit_html_escapes_untrusted_values_and_states_fixture_limit() -> None:
    rendered = render_stage11_audit_html(
        {
            "status": "<FAIL_CLOSED>",
            "physical_host_count": 1,
            "cross_physical_host_tested": False,
            "baseline_comparison": {"status": "SUCCESS"},
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
    assert "not production authentication" in rendered
