from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.pilot.stage9_gate import Stage9PreflightReport, assess_stage9_preflight
from retinaentail.reporting.stage9 import (
    render_stage9_audit_html,
    render_stage9_preflight_html,
    render_stage9_preflight_json,
    write_stage9_preflight_reports,
)


def _blocked() -> Stage9PreflightReport:
    return assess_stage9_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_stage8_package_a_capability_provided=False,
        complete_stage8_package_b_capability_provided=False,
        fresh_session_a_capability_provided=False,
        fresh_session_b_capability_provided=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    )


def test_preflight_report_is_path_free_deterministic_and_fail_safe(tmp_path: Path) -> None:
    report = _blocked()
    rendered_json = render_stage9_preflight_json(report)
    rendered_html = render_stage9_preflight_html(report)
    payload = json.loads(rendered_json)
    assert payload["status"] == "BLOCKED"
    assert payload["accepts_paths"] is False
    assert payload["reads_environment"] is False
    assert payload["dataset_root_probed"] is False
    assert "accepts no path" in rendered_html

    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    paths = write_stage9_preflight_reports(report, tmp_path, boundary=boundary)
    assert paths[0].read_text(encoding="utf-8") == rendered_json
    assert paths[1].read_text(encoding="utf-8") == rendered_html
    paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage9_preflight_reports(report, tmp_path, boundary=boundary)


def test_audit_html_escapes_case_and_difference_values() -> None:
    interval = {
        "numerator": 1,
        "denominator": 1,
        "estimate": 1.0,
        "lower": 0.2,
        "upper": 1.0,
        "confidence_level": 0.95,
        "method": "wilson_score",
    }
    session = {
        "producer_process_id": 123,
        "session_nonce_sha256": "a" * 64,
        "environment_lock_sha256": "b" * 64,
        "package_sha256": "c" * 64,
    }
    report = {
        "status": "FAIL_CLOSED",
        "diff": {
            "session_a": session,
            "session_b": {**session, "producer_process_id": 456},
            "global_differences": ["<PACKAGE_HASH>"],
            "metric_intervals": {"STATE_EXACT_MATCH": interval},
            "case_comparisons": [
                {
                    "case_id": "<case>",
                    "run_id": "run",
                    "session_a_state": "OUT_OF_MODEL",
                    "session_b_state": "OUT_OF_MODEL",
                    "effective_state": "OUT_OF_MODEL",
                    "session_a_positive_score": 0.0,
                    "session_a_negative_score": 0.0,
                    "session_b_positive_score": 0.0,
                    "session_b_negative_score": 0.0,
                    "dual_witness_content_match": False,
                    "dual_witness_replay_match": False,
                    "differences": ["<WITNESS_CONTENT>"],
                }
            ],
        },
    }
    html = render_stage9_audit_html(report)
    assert "<case>" not in html
    assert "&lt;case&gt;" in html
    assert "&lt;WITNESS_CONTENT&gt;" in html
