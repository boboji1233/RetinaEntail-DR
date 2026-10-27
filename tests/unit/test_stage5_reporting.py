from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.pilot.gate import Stage5VerificationContext, assess_stage5_preflight
from retinaentail.pilot.identity import (
    Stage5IdentityRecord,
    audit_five_identity_split_isolation,
)
from retinaentail.reporting.stage5 import (
    Stage5PilotReport,
    issue_stage5_pilot_report,
    render_stage5_pilot_html,
    render_stage5_pilot_json,
    render_stage5_preflight_html,
    write_stage5_pilot_reports,
    write_stage5_preflight_reports,
)


def _locked() -> dict[str, object]:
    return {
        "alpha": 0.25,
        "positive_threshold": 0.01,
        "negative_threshold": 0.02,
        "positive_sample_count": 4,
        "negative_sample_count": 4,
        "total_sample_count": 8,
        "calibration_artifact_hash": "4" * 64,
        "config_hash": "1" * 64,
        "data_hash": "2" * 64,
        "model_hash": "3" * 64,
    }


def _identity() -> dict[str, object]:
    return cast(
        dict[str, object],
        audit_five_identity_split_isolation(_identity_records()).to_dict(),
    )


def _identity_records() -> tuple[Stage5IdentityRecord, ...]:
    splits = ("train", "calibration", "test", "test")
    return tuple(
        Stage5IdentityRecord(
            dataset_id="retinaentail_stage5_toy",
            relative_path=f"{split}/record-{index}.synthetic",
            split=split,
            source_image_id=f"source-{index}",
            patient_id=f"patient-{index}",
            eye_id="R",
            followup_id="baseline",
            intervention_pair_id=f"pair-{index}",
        )
        for index, split in enumerate(splits)
    )


def _cases() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "pilot_index": index,
            "case_id": f"case-{index}",
            "reference_label": "UNKNOWN",
            "state": state.value,
            "status": "SUCCESS",
            "failure_reasons": [],
            "dual_witness_replay_passed": True,
            "positive_witness_replay_passed": True,
            "negative_witness_replay_passed": True,
            "evaluation_sha256": f"{index + 1:x}" * 64,
            "certificate_sha256": f"{index + 5:x}" * 64,
            "dual_replay_sha256": f"{index + 9:x}" * 64,
            "positive_witness_sha256": f"{index + 1:x}" * 64,
            "negative_witness_sha256": f"{index + 5:x}" * 64,
        }
        for index, state in enumerate(EntailmentState)
    )


def _evaluation() -> dict[str, object]:
    state_counts = {state.value: 1 for state in EntailmentState}
    state_proportions = {
        state.value: wilson_score_proportion(1, 4, confidence_level=0.95).to_dict()
        for state in EntailmentState
    }
    return {
        "schema_version": "4.0.0",
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "clinical_claim": False,
        "status": "SUCCESS",
        "locked_contract": _locked(),
        "confidence_level": 0.95,
        "input_record_count": 4,
        "test_case_count": 4,
        "rejected_record_count": 0,
        "evaluable_count": 4,
        "unevaluable_count": 0,
        "fail_closed_count": 0,
        "out_of_distribution_count": 0,
        "both_witness_replays_passed_count": 4,
        "one_or_more_witness_replays_failed_count": 0,
        "labeled_reference_count": 0,
        "state_counts": state_counts,
        "reference_label_counts": {
            "LESION_PRESENT": 0,
            "LESION_ABSENT": 0,
            "UNKNOWN": 4,
        },
        "failure_reason_counts": {},
        "rejected_reason_counts": {},
        "aggregation_failure_reasons": [],
        "state_proportions": state_proportions,
        "evaluable_proportion": wilson_score_proportion(4, 4).to_dict(),
        "unevaluable_proportion": wilson_score_proportion(0, 4).to_dict(),
        "fail_closed_proportion": wilson_score_proportion(0, 4).to_dict(),
        "out_of_distribution_proportion": wilson_score_proportion(0, 4).to_dict(),
        "both_witness_replays_passed_proportion": wilson_score_proportion(4, 4).to_dict(),
        "one_or_more_witness_replays_failed_proportion": wilson_score_proportion(0, 4).to_dict(),
    }


def _issue(
    *,
    identity: dict[str, object] | None = None,
    identity_records: tuple[Stage5IdentityRecord, ...] | None = None,
    evaluation: dict[str, object] | None = None,
    cases: tuple[dict[str, object], ...] | None = None,
) -> Stage5PilotReport:
    return issue_stage5_pilot_report(
        pilot_id="stage5-toy-locked-pilot-001",
        source_stage4_package_sha256="a" * 64,
        source_stage4_file_count=88,
        stage5_config_sha256="b" * 64,
        pilot_data_sha256="c" * 64,
        stage5_model_provenance_sha256="d" * 64,
        locked_contract=_locked(),
        identity_isolation=_identity() if identity is None else identity,
        identity_records=_identity_records() if identity_records is None else identity_records,
        evaluation=_evaluation() if evaluation is None else evaluation,
        case_index=_cases() if cases is None else cases,
    )


def _report() -> Stage5PilotReport:
    return _issue()


def test_stage5_json_html_are_deterministic_static_and_nonclinical() -> None:
    report = _report()
    first = render_stage5_pilot_json(report)
    assert first == render_stage5_pilot_json(report)
    payload = json.loads(first)
    html = render_stage5_pilot_html(report)

    assert payload["scope"] == "SYNTHETIC_ONLY"
    assert payload["thresholds_reestimated"] is False
    assert payload["test_split_used_for_calibration"] is False
    assert payload["clinical_claim"] is False
    assert payload["authority_boundary"] == "COMPLETE_VERIFIED_STAGE5_PACKAGE"
    assert "not a clinical claim" in html
    assert "Wilson" in html
    assert "OUT_OF_MODEL" in html
    assert "<script" not in html


def test_stage5_report_pair_is_repeatable_and_fail_safe(tmp_path: Path) -> None:
    output = tmp_path / "reports"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    first = write_stage5_pilot_reports(_report(), output, boundary=boundary)
    before = tuple(path.read_bytes() for path in first)
    assert write_stage5_pilot_reports(_report(), output, boundary=boundary) == first
    assert tuple(path.read_bytes() for path in first) == before

    other = tmp_path / "other"
    other.mkdir()
    (other / "stage5-pilot.json").write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_stage5_pilot_reports(_report(), other, boundary=boundary)
    assert not (other / "stage5-pilot.html").exists()


def test_stage5_report_writer_rejects_escape_and_unissued_copy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        write_stage5_pilot_reports(
            _report(),
            tmp_path / "outside",
            boundary=AuditWriteBoundary(allowed_root=tmp_path / "allowed"),
        )
    with pytest.raises(TypeError, match="can only be issued"):
        replace(_report(), pilot_id="forged")


def test_stage5_report_detects_post_issue_interval_mutation() -> None:
    report = _report()
    evaluation = cast(dict[str, object], report.evaluation["state_proportions"])
    present = cast(dict[str, object], evaluation["ENTAILED_PRESENT"])
    present["upper"] = 0.99
    with pytest.raises(ValueError, match="Wilson interval|bindings changed"):
        render_stage5_pilot_json(report)


def test_stage5_issuer_rejects_identity_payload_not_bound_to_records() -> None:
    identity = _identity()
    identity["input_projection_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match the supplied identity records"):
        _issue(identity=identity)


def test_stage5_issuer_rejects_contradictory_case_and_dual_counts() -> None:
    evaluation = _evaluation()
    evaluation["both_witness_replays_passed_count"] = 3
    evaluation["one_or_more_witness_replays_failed_count"] = 1
    evaluation["both_witness_replays_passed_proportion"] = wilson_score_proportion(3, 4).to_dict()
    evaluation["one_or_more_witness_replays_failed_proportion"] = wilson_score_proportion(
        1, 4
    ).to_dict()
    with pytest.raises(ValueError, match="summary counts do not match"):
        _issue(evaluation=evaluation)


def test_stage5_issuer_rejects_malformed_case_replay_and_hash_fields() -> None:
    cases = _cases()
    cases[0]["dual_witness_replay_passed"] = False
    with pytest.raises(ValueError, match="conjunction"):
        _issue(cases=cases)

    cases = _cases()
    cases[0]["evaluation_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="artifact hashes"):
        _issue(cases=cases)


def test_stage5_blocked_preflight_is_path_free_static_and_fail_safe(tmp_path: Path) -> None:
    report = assess_stage5_preflight(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id=None,
        dataset_version=None,
        approved_review_provided=False,
        approved_review_purposes=(),
        dataset_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_audit_provided=False,
        frozen_data_card_provided=False,
        stage2_access_permit_provided=False,
        manual_confirmation_provided=False,
        stage3_package_summary_provided=False,
        stage4_locked_evidence_provided=False,
    )
    output = tmp_path / "preflight"
    boundary = AuditWriteBoundary(allowed_root=tmp_path)
    first = write_stage5_preflight_reports(report, output, boundary=boundary)
    before = tuple(path.read_bytes() for path in first)
    assert write_stage5_preflight_reports(report, output, boundary=boundary) == first
    assert tuple(path.read_bytes() for path in first) == before
    payload = json.loads(first[0].read_text(encoding="utf-8"))
    html = render_stage5_preflight_html(report)
    assert payload["status"] == "BLOCKED"
    assert payload["authorizes_data_access"] is False
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False
    assert payload["sample_count"] == 0
    assert str(tmp_path) not in first[0].read_text(encoding="utf-8")
    assert str(tmp_path) not in html
    assert "never authorizes data access" in html
    assert "<script" not in html
