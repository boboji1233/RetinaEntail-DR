"""Zero-I/O Stage 11 preflight and process-bound host signer capabilities."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Never, TypeVar

STAGE11_PREFLIGHT_SCHEMA_VERSION: Final = "11.0.0"
_MAX_CAPABILITY_TTL_SECONDS: Final = 60.0
_SHA256_HEX: Final = frozenset("0123456789abcdef")
_RESULT = TypeVar("_RESULT")
_ISSUER_TOKEN: Final = object()
_REGISTRY: dict[int, str] = {}


class Stage11VerificationContext(StrEnum):
    REAL_PREREQUISITE = "REAL_PREREQUISITE"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class Stage11PreflightStatus(StrEnum):
    BLOCKED = "BLOCKED"
    READY_FOR_PRODUCTION_HOST_VERIFICATION = "READY_FOR_PRODUCTION_HOST_VERIFICATION"


class Stage11Reason(StrEnum):
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    COMPLETE_REAL_STAGE10_PACKAGE_A_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE10_PACKAGE_A_CAPABILITY_NOT_PROVIDED"
    )
    COMPLETE_REAL_STAGE10_PACKAGE_B_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE10_PACKAGE_B_CAPABILITY_NOT_PROVIDED"
    )
    FRESH_HOST_A_PROCESS_CAPABILITY_NOT_PROVIDED = "FRESH_HOST_A_PROCESS_CAPABILITY_NOT_PROVIDED"
    FRESH_HOST_B_PROCESS_CAPABILITY_NOT_PROVIDED = "FRESH_HOST_B_PROCESS_CAPABILITY_NOT_PROVIDED"
    TWO_APPROVED_PHYSICAL_HOSTS_NOT_CONFIRMED = "TWO_APPROVED_PHYSICAL_HOSTS_NOT_CONFIRMED"
    PRODUCTION_ASYMMETRIC_TRUST_ANCHOR_NOT_VERIFIED = (
        "PRODUCTION_ASYMMETRIC_TRUST_ANCHOR_NOT_VERIFIED"
    )
    UNIQUE_RUNNER_LEAF_KEYS_NOT_VERIFIED = "UNIQUE_RUNNER_LEAF_KEYS_NOT_VERIFIED"
    CERTIFICATE_CHAINS_NOT_VERIFIED = "CERTIFICATE_CHAINS_NOT_VERIFIED"
    REVOCATION_STATUS_NOT_CURRENT = "REVOCATION_STATUS_NOT_CURRENT"
    TRUSTED_TIME_POLICY_NOT_VERIFIED = "TRUSTED_TIME_POLICY_NOT_VERIFIED"
    COMPLETE_WHEEL_LOCK_HASHES_NOT_VERIFIED = "COMPLETE_WHEEL_LOCK_HASHES_NOT_VERIFIED"
    HOST_ENVIRONMENT_POLICY_NOT_PREREGISTERED = "HOST_ENVIRONMENT_POLICY_NOT_PREREGISTERED"
    CASE_SEED_REPLAY_NOT_PREREGISTERED = "CASE_SEED_REPLAY_NOT_PREREGISTERED"
    LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED = "LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED"
    INPUT_INVALID = "INPUT_INVALID"


class Stage11CapabilityError(PermissionError):
    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(c in value for c in "\r\n\x00")


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage11PreflightReport:
    status: Stage11PreflightStatus
    verification_context: Stage11VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage11Reason, ...]
    inputs_present: tuple[tuple[str, bool], ...]
    schema_version: str = STAGE11_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE11_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 11 preflight schema")
        if not isinstance(self.status, Stage11PreflightStatus) or not isinstance(
            self.verification_context, Stage11VerificationContext
        ):
            raise TypeError("Stage 11 preflight uses controlled enums")
        if not all(isinstance(reason, Stage11Reason) for reason in self.reasons):
            raise TypeError("Stage 11 preflight reasons use Stage11Reason")
        if tuple(sorted(self.inputs_present)) != self.inputs_present:
            raise ValueError("inputs_present must be sorted")
        ready = self.status is Stage11PreflightStatus.READY_FOR_PRODUCTION_HOST_VERIFICATION
        if ready == bool(self.reasons):
            raise ValueError("Stage 11 preflight status conflicts with reasons")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "required_scope": "APPROVED_REAL_DATA",
            "required_split": "test",
            "inputs_present": dict(self.inputs_present),
            "reasons": [reason.value for reason in self.reasons],
            "accepts_paths": False,
            "reads_environment": False,
            "authorizes_file_open": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "detached_json_authoritative": False,
            "historical_receipt_authoritative": False,
            "synthetic_signature_authorizes_real": False,
            "production_private_key_accepted_as_input": False,
            "two_physical_hosts_verified": False,
        }


def assess_stage11_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    complete_real_stage10_package_a_capability_present: bool,
    complete_real_stage10_package_b_capability_present: bool,
    fresh_host_a_process_capability_present: bool,
    fresh_host_b_process_capability_present: bool,
    two_approved_physical_hosts_confirmed: bool,
    production_asymmetric_trust_anchor_verified: bool,
    unique_runner_leaf_keys_verified: bool,
    certificate_chains_verified: bool,
    revocation_status_current: bool,
    trusted_time_policy_verified: bool,
    complete_wheel_lock_hashes_verified: bool,
    host_environment_policy_preregistered: bool,
    case_seed_replay_preregistered: bool,
    locked_hashes_thresholds_preregistered: bool,
    verification_context: Stage11VerificationContext = (
        Stage11VerificationContext.REAL_PREREQUISITE
    ),
) -> Stage11PreflightReport:
    """Presence-only preflight whose public signature cannot accept paths or private keys."""

    flags = {
        "case_seed_replay_preregistered": case_seed_replay_preregistered,
        "certificate_chains_verified": certificate_chains_verified,
        "complete_real_stage10_package_a_capability_present": (
            complete_real_stage10_package_a_capability_present
        ),
        "complete_real_stage10_package_b_capability_present": (
            complete_real_stage10_package_b_capability_present
        ),
        "complete_wheel_lock_hashes_verified": complete_wheel_lock_hashes_verified,
        "fresh_host_a_process_capability_present": fresh_host_a_process_capability_present,
        "fresh_host_b_process_capability_present": fresh_host_b_process_capability_present,
        "host_environment_policy_preregistered": host_environment_policy_preregistered,
        "locked_hashes_thresholds_preregistered": locked_hashes_thresholds_preregistered,
        "production_asymmetric_trust_anchor_verified": (
            production_asymmetric_trust_anchor_verified
        ),
        "revocation_status_current": revocation_status_current,
        "trusted_time_policy_verified": trusted_time_policy_verified,
        "two_approved_physical_hosts_confirmed": two_approved_physical_hosts_confirmed,
        "unique_runner_leaf_keys_verified": unique_runner_leaf_keys_verified,
    }
    if not isinstance(verification_context, Stage11VerificationContext) or any(
        type(value) is not bool
        for value in flags.values()  # noqa: E721
    ):
        return Stage11PreflightReport(
            status=Stage11PreflightStatus.BLOCKED,
            verification_context=Stage11VerificationContext.REAL_PREREQUISITE,
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage11Reason.INPUT_INVALID,),
            inputs_present=tuple(sorted((name, False) for name in flags)),
        )
    reasons: list[Stage11Reason] = []
    if not _valid_identifier(dataset_id):
        reasons.append(Stage11Reason.DATASET_NOT_EXPLICIT)
    if not _valid_identifier(dataset_version):
        reasons.append(Stage11Reason.DATASET_VERSION_NOT_EXPLICIT)
    requirements = (
        (
            complete_real_stage10_package_a_capability_present,
            Stage11Reason.COMPLETE_REAL_STAGE10_PACKAGE_A_CAPABILITY_NOT_PROVIDED,
        ),
        (
            complete_real_stage10_package_b_capability_present,
            Stage11Reason.COMPLETE_REAL_STAGE10_PACKAGE_B_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_host_a_process_capability_present,
            Stage11Reason.FRESH_HOST_A_PROCESS_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_host_b_process_capability_present,
            Stage11Reason.FRESH_HOST_B_PROCESS_CAPABILITY_NOT_PROVIDED,
        ),
        (
            two_approved_physical_hosts_confirmed,
            Stage11Reason.TWO_APPROVED_PHYSICAL_HOSTS_NOT_CONFIRMED,
        ),
        (
            production_asymmetric_trust_anchor_verified,
            Stage11Reason.PRODUCTION_ASYMMETRIC_TRUST_ANCHOR_NOT_VERIFIED,
        ),
        (unique_runner_leaf_keys_verified, Stage11Reason.UNIQUE_RUNNER_LEAF_KEYS_NOT_VERIFIED),
        (certificate_chains_verified, Stage11Reason.CERTIFICATE_CHAINS_NOT_VERIFIED),
        (revocation_status_current, Stage11Reason.REVOCATION_STATUS_NOT_CURRENT),
        (trusted_time_policy_verified, Stage11Reason.TRUSTED_TIME_POLICY_NOT_VERIFIED),
        (
            complete_wheel_lock_hashes_verified,
            Stage11Reason.COMPLETE_WHEEL_LOCK_HASHES_NOT_VERIFIED,
        ),
        (
            host_environment_policy_preregistered,
            Stage11Reason.HOST_ENVIRONMENT_POLICY_NOT_PREREGISTERED,
        ),
        (
            case_seed_replay_preregistered,
            Stage11Reason.CASE_SEED_REPLAY_NOT_PREREGISTERED,
        ),
        (
            locked_hashes_thresholds_preregistered,
            Stage11Reason.LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED,
        ),
    )
    reasons.extend(reason for present, reason in requirements if not present)
    return Stage11PreflightReport(
        status=(
            Stage11PreflightStatus.BLOCKED
            if reasons
            else Stage11PreflightStatus.READY_FOR_PRODUCTION_HOST_VERIFICATION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id if _valid_identifier(dataset_id) else None,
        dataset_version=dataset_version if _valid_identifier(dataset_version) else None,
        reasons=tuple(reasons),
        inputs_present=tuple(sorted(flags.items())),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage11RunnerAuthority:
    runner_id: str
    logical_host_id: str
    producer_process_id: int
    session_nonce_sha256: str
    dataset_id: str
    dataset_version: str
    scope: str
    split: str
    source_stage10_package_sha256: str
    stage10_environment_lock_sha256: str
    runtime_lock_sha256: str
    dependency_lock_sha256: str
    trust_anchor_sha256: str
    current_key_id: str
    baseline_projection_sha256: str
    signature_verified: bool

    def __post_init__(self) -> None:
        text_fields = (
            self.runner_id,
            self.logical_host_id,
            self.dataset_id,
            self.dataset_version,
            self.scope,
            self.split,
        )
        digest_fields = (
            self.session_nonce_sha256,
            self.source_stage10_package_sha256,
            self.stage10_environment_lock_sha256,
            self.runtime_lock_sha256,
            self.dependency_lock_sha256,
            self.trust_anchor_sha256,
            self.current_key_id,
            self.baseline_projection_sha256,
        )
        if not all(_valid_identifier(value) for value in text_fields):
            raise ValueError("Stage 11 runner authority identifier is invalid")
        if not all(
            len(value) == 64 and all(character in _SHA256_HEX for character in value)
            for value in digest_fields
        ):
            raise ValueError("Stage 11 runner authority digest is invalid")
        if self.producer_process_id <= 0 or not self.signature_verified:
            raise ValueError("Stage 11 runner authority is not verified")
        if self.scope != "SYNTHETIC_ONLY" or self.split != "test":
            raise ValueError("public Stage 11 authority is synthetic test-only")

    def to_dict(self) -> dict[str, object]:
        return {
            "runner_id": self.runner_id,
            "logical_host_id": self.logical_host_id,
            "producer_process_id": self.producer_process_id,
            "session_nonce_sha256": self.session_nonce_sha256,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "scope": self.scope,
            "split": self.split,
            "source_stage10_package_sha256": self.source_stage10_package_sha256,
            "stage10_environment_lock_sha256": self.stage10_environment_lock_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "trust_anchor_sha256": self.trust_anchor_sha256,
            "current_key_id": self.current_key_id,
            "baseline_projection_sha256": self.baseline_projection_sha256,
            "signature_verified": self.signature_verified,
        }


@dataclass(slots=True, kw_only=True)
class VerifiedStage10HostCapability:
    authority: Stage11RunnerAuthority
    verifier_process_id: int
    issued_monotonic: float
    expires_monotonic: float
    nonce: str
    _replay: Callable[[], Stage11RunnerAuthority] = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    consumed: bool = False

    @property
    def authorizes_file_open(self) -> bool:
        return False

    @property
    def authorizes_real_data(self) -> bool:
        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 11 capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 11 capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 11 capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 11 capabilities cannot be serialized")


def _fingerprint(value: VerifiedStage10HostCapability) -> str:
    payload = {
        "authority": value.authority.to_dict(),
        "verifier_process_id": value.verifier_process_id,
        "issued_monotonic": value.issued_monotonic,
        "expires_monotonic": value.expires_monotonic,
        "nonce": value.nonce,
    }
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def issue_synthetic_stage10_host_capability(
    replay: Callable[[], Stage11RunnerAuthority],
    *,
    ttl_seconds: float = 30.0,
) -> VerifiedStage10HostCapability:
    """Issue only local synthetic authority after complete Stage 10/envelope verification."""

    if (
        not callable(replay)
        or isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int | float)
    ):
        raise Stage11CapabilityError(("synthetic host issuer input is invalid",))
    ttl = float(ttl_seconds)
    if not 0.0 < ttl <= _MAX_CAPABILITY_TTL_SECONDS:
        raise Stage11CapabilityError(("host capability TTL is outside the allowed range",))
    authority = replay()
    if not isinstance(authority, Stage11RunnerAuthority) or authority.scope != "SYNTHETIC_ONLY":
        raise Stage11CapabilityError(("synthetic issuer rejects REAL or malformed authority",))
    now = time.monotonic()
    capability = VerifiedStage10HostCapability(
        authority=authority,
        verifier_process_id=os.getpid(),
        issued_monotonic=now,
        expires_monotonic=now + ttl,
        nonce=hashlib.sha256(f"{id(authority)}:{time.monotonic_ns()}".encode()).hexdigest(),
        _replay=replay,
        _token=_ISSUER_TOKEN,
    )
    _REGISTRY[id(capability)] = _fingerprint(capability)
    return capability


def host_capability_validation_reasons(value: object) -> tuple[str, ...]:
    if not isinstance(value, VerifiedStage10HostCapability):
        return ("not a live Stage 11 host capability",)
    reasons: list[str] = []
    if value._token is not _ISSUER_TOKEN or _REGISTRY.get(id(value)) != _fingerprint(value):
        reasons.append("host capability provenance is invalid")
    if value.verifier_process_id != os.getpid():
        reasons.append("host capability is bound to another verifier process")
    if value.consumed:
        reasons.append("host capability was already consumed")
    if time.monotonic() > value.expires_monotonic:
        reasons.append("host capability expired")
    if value.authorizes_file_open or value.authorizes_real_data:
        reasons.append("host capability illegally authorizes data or file access")
    return tuple(reasons)


def _pair_reasons(left: object, right: object) -> tuple[str, ...]:
    reasons = [
        *host_capability_validation_reasons(left),
        *host_capability_validation_reasons(right),
    ]
    if left is right:
        reasons.append("one host capability cannot fill both positions")
    if not isinstance(left, VerifiedStage10HostCapability) or not isinstance(
        right, VerifiedStage10HostCapability
    ):
        return tuple(dict.fromkeys(reasons))
    a = left.authority
    b = right.authority
    if a.runner_id == b.runner_id or a.session_nonce_sha256 == b.session_nonce_sha256:
        reasons.append("runner identities or session nonces are not independent")
    if a.current_key_id == b.current_key_id:
        reasons.append("runner leaf signing keys are shared")
    if (a.dataset_id, a.dataset_version, a.scope, a.split) != (
        b.dataset_id,
        b.dataset_version,
        b.scope,
        b.split,
    ):
        reasons.append("host capability scope or dataset differs")
    frozen = (
        "source_stage10_package_sha256",
        "stage10_environment_lock_sha256",
        "runtime_lock_sha256",
        "dependency_lock_sha256",
        "trust_anchor_sha256",
        "baseline_projection_sha256",
    )
    if any(getattr(a, name) != getattr(b, name) for name in frozen):
        reasons.append("host frozen contract differs")
    return tuple(dict.fromkeys(reasons))


def consume_stage11_host_pair(
    left: VerifiedStage10HostCapability,
    right: VerifiedStage10HostCapability,
    consumer: Callable[[Stage11RunnerAuthority, Stage11RunnerAuthority], _RESULT],
) -> _RESULT:
    """Atomically consume two live authorities and replay both before comparison."""

    if not callable(consumer):
        raise Stage11CapabilityError(("consumer is not callable",))
    reasons = _pair_reasons(left, right)
    if reasons:
        raise Stage11CapabilityError(reasons)
    left.consumed = True
    right.consumed = True
    replayed_left = left._replay()
    replayed_right = right._replay()
    if replayed_left != left.authority or replayed_right != right.authority:
        raise Stage11CapabilityError(("signed Stage 10 authority changed before consumption",))
    return consumer(replayed_left, replayed_right)


def host_capability_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, VerifiedStage10HostCapability):
        raise Stage11CapabilityError(("not a live Stage 11 host capability",))
    return {
        "schema_version": STAGE11_PREFLIGHT_SCHEMA_VERSION,
        "authority": value.authority.to_dict(),
        "verifier_process_id": value.verifier_process_id,
        "consumed": value.consumed,
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "authorizes_file_open": False,
        "authorizes_real_data": False,
    }
