from __future__ import annotations

import json
from pathlib import Path

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.evaluation.gate import assess_stage4_preflight
from retinaentail.evaluation.metrics import (
    LockedStage4Contract,
    Stage4EvaluationReport,
    Stage4EvaluationScope,
    aggregate_locked_test_evaluations,
)
from retinaentail.reporting.stage4 import (
    render_stage4_evaluation_html,
    render_stage4_evaluation_json,
    render_stage4_preflight_html,
    write_stage4_evaluation_reports,
    write_stage4_preflight_reports,
)

CONFIG_HASH = "1" * 64
DATA_HASH = "2" * 64
MODEL_HASH = "3" * 64


def _report() -> Stage4EvaluationReport:
    return aggregate_locked_test_evaluations(
        (),
        locked_contract=LockedStage4Contract(
            alpha=0.25,
            positive_threshold=0.5,
            negative_threshold=0.5,
            positive_sample_count=4,
            negative_sample_count=4,
            total_sample_count=8,
            calibration_artifact_hash="4" * 64,
            config_hash=CONFIG_HASH,
            data_hash=DATA_HASH,
            model_hash=MODEL_HASH,
        ),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    )


def test_json_and_static_html_are_deterministic_and_nonclinical() -> None:
    report = _report()
    first_json = render_stage4_evaluation_json(report)
    second_json = render_stage4_evaluation_json(report)
    html = render_stage4_evaluation_html(report)

    assert first_json == second_json
    payload = json.loads(first_json)
    assert payload["scope"] == "SYNTHETIC_ONLY"
    assert payload["split"] == "test"
    assert payload["clinical_claim"] is False
    assert payload["reference_label_counts"]["UNKNOWN"] == 0
    assert "not a clinical claim" in html
    assert "UNKNOWN" in html
    assert "<script" not in html


def test_report_pair_is_repeatable_and_refuses_nonidentical_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage4"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    json_path, html_path = write_stage4_evaluation_reports(_report(), output_dir, boundary=boundary)
    json_bytes = json_path.read_bytes()
    html_bytes = html_path.read_bytes()

    repeated = write_stage4_evaluation_reports(_report(), output_dir, boundary=boundary)
    assert repeated == (json_path, html_path)
    assert json_path.read_bytes() == json_bytes
    assert html_path.read_bytes() == html_bytes

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "stage4-evaluation.json").write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage4_evaluation_reports(_report(), other_dir, boundary=boundary)
    assert not (other_dir / "stage4-evaluation.html").exists()


def test_report_writer_rejects_output_escape(tmp_path: Path) -> None:
    boundary = AuditWriteBoundary(allowed_root=tmp_path / "allowed")
    with pytest.raises(ValueError, match="escapes"):
        write_stage4_evaluation_reports(
            _report(),
            tmp_path / "outside",
            boundary=boundary,
        )


def test_blocked_preflight_reports_are_path_free_static_and_repeatable(tmp_path: Path) -> None:
    preflight = assess_stage4_preflight(
        dataset_id=None,
        dataset_version=None,
        approved_review_provided=False,
        approved_review_purposes=(),
        dataset_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_audit_provided=False,
        frozen_data_card_provided=False,
        stage3_calibration_package_provided=False,
        manual_confirmation_record_provided=False,
        locked_evidence_provided=False,
        locked_evidence=None,
    )
    output_dir = tmp_path / "preflight"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    json_path, html_path = write_stage4_preflight_reports(preflight, output_dir, boundary=boundary)
    first_json = json_path.read_bytes()
    first_html = html_path.read_bytes()

    repeated = write_stage4_preflight_reports(preflight, output_dir, boundary=boundary)
    assert repeated == (json_path, html_path)
    payload = json.loads(first_json)
    assert payload["status"] == "BLOCKED"
    assert payload["authorizes_data_access"] is False
    assert payload["real_data_read"] is False
    assert payload["sample_count"] == 0
    assert str(tmp_path) not in first_json.decode("utf-8")
    assert str(tmp_path) not in first_html.decode("utf-8")
    assert "<script" not in render_stage4_preflight_html(preflight)
