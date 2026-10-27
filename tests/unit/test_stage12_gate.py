from __future__ import annotations

import inspect

from retinaentail.pilot.stage12_gate import (
    Stage12PreflightStatus,
    assess_stage12_preflight,
)


def _preflight(*, ready: bool):  # type: ignore[no-untyped-def]
    return assess_stage12_preflight(
        dataset_id="approved-dataset" if ready else None,
        dataset_version="exact-v1" if ready else None,
        complete_real_stage11_package_a_capability_present=ready,
        complete_real_stage11_package_b_capability_present=ready,
        fresh_runner_a_process_capability_present=ready,
        fresh_runner_b_process_capability_present=ready,
        two_governance_approved_hardware_runners_verified=ready,
        hsm_or_controlled_signer_boundary_verified=ready,
        trusted_ca_chains_verified=ready,
        trusted_timestamp_policy_verified=ready,
        current_ocsp_status_verified=ready,
        current_crl_status_verified=ready,
        complete_wheel_lock_hashes_verified=ready,
        hardware_identity_verified=ready,
        frozen_baseline_contract_preregistered=ready,
        compromise_recovery_drill_preregistered=ready,
    )


def test_stage12_real_gate_is_zero_io_and_blocked_without_prerequisites() -> None:
    signature = inspect.signature(assess_stage12_preflight)
    assert all("path" not in name and "root" not in name for name in signature.parameters)
    report = _preflight(ready=False)
    payload = report.to_dict()
    assert report.status is Stage12PreflightStatus.BLOCKED
    assert len(report.reasons) == 16
    assert payload["accepts_paths"] is False
    assert payload["accepts_private_keys"] is False
    assert payload["reads_environment"] is False
    assert payload["authorizes_file_open"] is False
    assert payload["authorizes_real_data"] is False
    assert payload["dataset_root_probed"] is False
    assert payload["real_data_read"] is False
    assert payload["production_ceremony_executed"] is False


def test_complete_presence_only_reaches_live_production_verification_boundary() -> None:
    report = _preflight(ready=True)
    assert report.status is Stage12PreflightStatus.READY_FOR_PRODUCTION_CEREMONY_VERIFICATION
    assert report.reasons == ()
    assert report.to_dict()["physical_host_count_verified"] == 0


def test_non_boolean_presence_flag_fails_closed() -> None:
    report = assess_stage12_preflight(
        dataset_id="approved-dataset",
        dataset_version="exact-v1",
        complete_real_stage11_package_a_capability_present=True,
        complete_real_stage11_package_b_capability_present=True,
        fresh_runner_a_process_capability_present=True,
        fresh_runner_b_process_capability_present=True,
        two_governance_approved_hardware_runners_verified=True,
        hsm_or_controlled_signer_boundary_verified=True,
        trusted_ca_chains_verified=True,
        trusted_timestamp_policy_verified=True,
        current_ocsp_status_verified=True,
        current_crl_status_verified=True,
        complete_wheel_lock_hashes_verified=True,
        hardware_identity_verified=True,
        frozen_baseline_contract_preregistered=True,
        compromise_recovery_drill_preregistered=1,  # type: ignore[arg-type]
    )
    assert report.status is Stage12PreflightStatus.BLOCKED
    assert [reason.value for reason in report.reasons] == ["INPUT_INVALID"]
