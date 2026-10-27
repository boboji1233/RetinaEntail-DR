from __future__ import annotations

import pytest

from retinaentail.calibration.stage3_gate import (
    Stage3PreflightReason,
    Stage3PreflightReport,
    Stage3PreflightStatus,
    assess_stage3_preflight,
)


def test_missing_stage3_inputs_are_path_free_blocked_and_round_trip() -> None:
    report = assess_stage3_preflight(
        dataset_id=None,
        dataset_version=None,
        review_record_provided=False,
        dataset_root_provided=False,
        adapter_spec_provided=False,
    )

    assert report.status is Stage3PreflightStatus.BLOCKED
    assert set(report.reasons) == set(Stage3PreflightReason)
    assert report.authorizes_data_access is False
    assert report.to_dict()["real_data_read"] is False
    assert report.to_dict()["sample_count"] == 0
    assert Stage3PreflightReport.from_dict(report.to_dict()) == report


def test_complete_presence_only_means_ready_to_attempt_stage2_gate() -> None:
    report = assess_stage3_preflight(
        dataset_id="idrid",
        dataset_version="release-v1",
        review_record_provided=True,
        dataset_root_provided=True,
        adapter_spec_provided=True,
    )

    assert report.status is Stage3PreflightStatus.READY_FOR_STAGE2_GATE
    assert report.reasons == ()
    assert report.authorizes_data_access is False


def test_preflight_rejects_non_boolean_presence_and_unsafe_version() -> None:
    try:
        assess_stage3_preflight(
            dataset_id="idrid",
            dataset_version="release/v1",
            review_record_provided=True,
            dataset_root_provided=True,
            adapter_spec_provided=True,
        )
    except ValueError as error:
        assert "report-safe" in str(error)
    else:
        raise AssertionError("unsafe version must fail")

    try:
        assess_stage3_preflight(
            dataset_id="idrid",
            dataset_version="release-v1",
            review_record_provided=1,  # type: ignore[arg-type]
            dataset_root_provided=True,
            adapter_spec_provided=True,
        )
    except TypeError as error:
        assert "exact bool" in str(error)
    else:
        raise AssertionError("numeric presence flag must fail")


def test_preflight_reload_rejects_ready_or_reason_tampering() -> None:
    blocked = assess_stage3_preflight(
        dataset_id=None,
        dataset_version=None,
        review_record_provided=False,
        dataset_root_provided=False,
        adapter_spec_provided=False,
    ).to_dict()
    blocked["status"] = "READY_FOR_STAGE2_GATE"
    blocked["reasons"] = []
    with pytest.raises(ValueError, match="conflict with explicit input presence"):
        Stage3PreflightReport.from_dict(blocked)

    incomplete = assess_stage3_preflight(
        dataset_id=None,
        dataset_version=None,
        review_record_provided=False,
        dataset_root_provided=False,
        adapter_spec_provided=False,
    ).to_dict()
    reasons = incomplete["reasons"]
    assert isinstance(reasons, list)
    incomplete["reasons"] = reasons[1:]
    with pytest.raises(ValueError, match="conflict with explicit input presence"):
        Stage3PreflightReport.from_dict(incomplete)
