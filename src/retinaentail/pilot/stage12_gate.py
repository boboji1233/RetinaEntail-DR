"""Zero-I/O Stage 12 production-ceremony prerequisite gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

STAGE12_PREFLIGHT_SCHEMA_VERSION: Final = "12.0.0"


class Stage12PreflightStatus(StrEnum):
    BLOCKED = "BLOCKED"
    READY_FOR_PRODUCTION_CEREMONY_VERIFICATION = "READY_FOR_PRODUCTION_CEREMONY_VERIFICATION"


class Stage12Reason(StrEnum):
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    COMPLETE_REAL_STAGE11_PACKAGE_A_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE11_PACKAGE_A_CAPABILITY_NOT_PROVIDED"
    )
    COMPLETE_REAL_STAGE11_PACKAGE_B_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE11_PACKAGE_B_CAPABILITY_NOT_PROVIDED"
    )
    FRESH_RUNNER_A_PROCESS_CAPABILITY_NOT_PROVIDED = (
        "FRESH_RUNNER_A_PROCESS_CAPABILITY_NOT_PROVIDED"
    )
    FRESH_RUNNER_B_PROCESS_CAPABILITY_NOT_PROVIDED = (
        "FRESH_RUNNER_B_PROCESS_CAPABILITY_NOT_PROVIDED"
    )
    TWO_GOVERNANCE_APPROVED_HARDWARE_RUNNERS_NOT_VERIFIED = (
        "TWO_GOVERNANCE_APPROVED_HARDWARE_RUNNERS_NOT_VERIFIED"
    )
    HSM_OR_CONTROLLED_SIGNER_BOUNDARY_NOT_VERIFIED = (
        "HSM_OR_CONTROLLED_SIGNER_BOUNDARY_NOT_VERIFIED"
    )
    TRUSTED_CA_CHAINS_NOT_VERIFIED = "TRUSTED_CA_CHAINS_NOT_VERIFIED"
    TRUSTED_TIMESTAMP_POLICY_NOT_VERIFIED = "TRUSTED_TIMESTAMP_POLICY_NOT_VERIFIED"
    CURRENT_OCSP_STATUS_NOT_VERIFIED = "CURRENT_OCSP_STATUS_NOT_VERIFIED"
    CURRENT_CRL_STATUS_NOT_VERIFIED = "CURRENT_CRL_STATUS_NOT_VERIFIED"
    COMPLETE_WHEEL_LOCK_HASHES_NOT_VERIFIED = "COMPLETE_WHEEL_LOCK_HASHES_NOT_VERIFIED"
    HARDWARE_IDENTITY_NOT_VERIFIED = "HARDWARE_IDENTITY_NOT_VERIFIED"
    FROZEN_BASELINE_CONTRACT_NOT_PREREGISTERED = "FROZEN_BASELINE_CONTRACT_NOT_PREREGISTERED"
    COMPROMISE_RECOVERY_DRILL_NOT_PREREGISTERED = "COMPROMISE_RECOVERY_DRILL_NOT_PREREGISTERED"
    INPUT_INVALID = "INPUT_INVALID"


def _identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(c in value for c in "\r\n\x00")


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage12PreflightReport:
    status: Stage12PreflightStatus
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage12Reason, ...]
    inputs_present: tuple[tuple[str, bool], ...]
    schema_version: str = STAGE12_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE12_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 12 preflight schema")
        if tuple(sorted(self.inputs_present)) != self.inputs_present:
            raise ValueError("Stage 12 inputs must be sorted")
        ready = self.status is Stage12PreflightStatus.READY_FOR_PRODUCTION_CEREMONY_VERIFICATION
        if ready == bool(self.reasons):
            raise ValueError("Stage 12 preflight status conflicts with reasons")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "required_scope": "APPROVED_REAL_DATA",
            "required_split": "test",
            "inputs_present": dict(self.inputs_present),
            "reasons": [item.value for item in self.reasons],
            "accepts_paths": False,
            "accepts_private_keys": False,
            "reads_environment": False,
            "authorizes_file_open": False,
            "authorizes_real_data": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "detached_json_authoritative": False,
            "historical_receipt_authoritative": False,
            "local_fixture_authorizes_production": False,
            "production_ceremony_executed": False,
            "physical_host_count_verified": 0,
        }


def assess_stage12_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    complete_real_stage11_package_a_capability_present: bool,
    complete_real_stage11_package_b_capability_present: bool,
    fresh_runner_a_process_capability_present: bool,
    fresh_runner_b_process_capability_present: bool,
    two_governance_approved_hardware_runners_verified: bool,
    hsm_or_controlled_signer_boundary_verified: bool,
    trusted_ca_chains_verified: bool,
    trusted_timestamp_policy_verified: bool,
    current_ocsp_status_verified: bool,
    current_crl_status_verified: bool,
    complete_wheel_lock_hashes_verified: bool,
    hardware_identity_verified: bool,
    frozen_baseline_contract_preregistered: bool,
    compromise_recovery_drill_preregistered: bool,
) -> Stage12PreflightReport:
    """Check explicit presence only; no argument can carry a path or private key."""

    flags = {
        "complete_real_stage11_package_a_capability_present": (
            complete_real_stage11_package_a_capability_present
        ),
        "complete_real_stage11_package_b_capability_present": (
            complete_real_stage11_package_b_capability_present
        ),
        "fresh_runner_a_process_capability_present": fresh_runner_a_process_capability_present,
        "fresh_runner_b_process_capability_present": fresh_runner_b_process_capability_present,
        "two_governance_approved_hardware_runners_verified": (
            two_governance_approved_hardware_runners_verified
        ),
        "hsm_or_controlled_signer_boundary_verified": hsm_or_controlled_signer_boundary_verified,
        "trusted_ca_chains_verified": trusted_ca_chains_verified,
        "trusted_timestamp_policy_verified": trusted_timestamp_policy_verified,
        "current_ocsp_status_verified": current_ocsp_status_verified,
        "current_crl_status_verified": current_crl_status_verified,
        "complete_wheel_lock_hashes_verified": complete_wheel_lock_hashes_verified,
        "hardware_identity_verified": hardware_identity_verified,
        "frozen_baseline_contract_preregistered": frozen_baseline_contract_preregistered,
        "compromise_recovery_drill_preregistered": compromise_recovery_drill_preregistered,
    }
    if any(type(value) is not bool for value in flags.values()):  # noqa: E721
        return Stage12PreflightReport(
            status=Stage12PreflightStatus.BLOCKED,
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage12Reason.INPUT_INVALID,),
            inputs_present=tuple(sorted((name, False) for name in flags)),
        )
    reasons: list[Stage12Reason] = []
    if not _identifier(dataset_id):
        reasons.append(Stage12Reason.DATASET_NOT_EXPLICIT)
    if not _identifier(dataset_version):
        reasons.append(Stage12Reason.DATASET_VERSION_NOT_EXPLICIT)
    requirements = (
        (
            flags["complete_real_stage11_package_a_capability_present"],
            Stage12Reason.COMPLETE_REAL_STAGE11_PACKAGE_A_CAPABILITY_NOT_PROVIDED,
        ),
        (
            flags["complete_real_stage11_package_b_capability_present"],
            Stage12Reason.COMPLETE_REAL_STAGE11_PACKAGE_B_CAPABILITY_NOT_PROVIDED,
        ),
        (
            flags["fresh_runner_a_process_capability_present"],
            Stage12Reason.FRESH_RUNNER_A_PROCESS_CAPABILITY_NOT_PROVIDED,
        ),
        (
            flags["fresh_runner_b_process_capability_present"],
            Stage12Reason.FRESH_RUNNER_B_PROCESS_CAPABILITY_NOT_PROVIDED,
        ),
        (
            flags["two_governance_approved_hardware_runners_verified"],
            Stage12Reason.TWO_GOVERNANCE_APPROVED_HARDWARE_RUNNERS_NOT_VERIFIED,
        ),
        (
            flags["hsm_or_controlled_signer_boundary_verified"],
            Stage12Reason.HSM_OR_CONTROLLED_SIGNER_BOUNDARY_NOT_VERIFIED,
        ),
        (flags["trusted_ca_chains_verified"], Stage12Reason.TRUSTED_CA_CHAINS_NOT_VERIFIED),
        (
            flags["trusted_timestamp_policy_verified"],
            Stage12Reason.TRUSTED_TIMESTAMP_POLICY_NOT_VERIFIED,
        ),
        (flags["current_ocsp_status_verified"], Stage12Reason.CURRENT_OCSP_STATUS_NOT_VERIFIED),
        (flags["current_crl_status_verified"], Stage12Reason.CURRENT_CRL_STATUS_NOT_VERIFIED),
        (
            flags["complete_wheel_lock_hashes_verified"],
            Stage12Reason.COMPLETE_WHEEL_LOCK_HASHES_NOT_VERIFIED,
        ),
        (flags["hardware_identity_verified"], Stage12Reason.HARDWARE_IDENTITY_NOT_VERIFIED),
        (
            flags["frozen_baseline_contract_preregistered"],
            Stage12Reason.FROZEN_BASELINE_CONTRACT_NOT_PREREGISTERED,
        ),
        (
            flags["compromise_recovery_drill_preregistered"],
            Stage12Reason.COMPROMISE_RECOVERY_DRILL_NOT_PREREGISTERED,
        ),
    )
    reasons.extend(reason for present, reason in requirements if not present)
    return Stage12PreflightReport(
        status=(
            Stage12PreflightStatus.BLOCKED
            if reasons
            else Stage12PreflightStatus.READY_FOR_PRODUCTION_CEREMONY_VERIFICATION
        ),
        dataset_id=dataset_id if _identifier(dataset_id) else None,
        dataset_version=dataset_version if _identifier(dataset_version) else None,
        reasons=tuple(reasons),
        inputs_present=tuple(sorted(flags.items())),
    )
