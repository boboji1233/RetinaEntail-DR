"""Zero-I/O Stage 10 preflight and process-bound runner capabilities."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Never, TypeVar

from retinaentail.reproducibility.stage10 import Stage10RunnerSnapshot

STAGE10_PREFLIGHT_SCHEMA_VERSION: Final = "10.0.0"
_MAX_CAPABILITY_TTL_SECONDS: Final = 60.0
_RESULT = TypeVar("_RESULT")
_ISSUER_TOKEN: Final = object()
_REGISTRY: dict[int, str] = {}


class Stage10VerificationContext(StrEnum):
    REAL_PREREQUISITE = "REAL_PREREQUISITE"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class Stage10PreflightStatus(StrEnum):
    BLOCKED = "BLOCKED"
    READY_FOR_RUNNER_CAPABILITY_VERIFICATION = "READY_FOR_RUNNER_CAPABILITY_VERIFICATION"


class Stage10Reason(StrEnum):
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    COMPLETE_REAL_STAGE9_PACKAGE_A_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE9_PACKAGE_A_CAPABILITY_NOT_PROVIDED"
    )
    COMPLETE_REAL_STAGE9_PACKAGE_B_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE9_PACKAGE_B_CAPABILITY_NOT_PROVIDED"
    )
    FRESH_RUNNER_A_CAPABILITY_NOT_PROVIDED = "FRESH_RUNNER_A_CAPABILITY_NOT_PROVIDED"
    FRESH_RUNNER_B_CAPABILITY_NOT_PROVIDED = "FRESH_RUNNER_B_CAPABILITY_NOT_PROVIDED"
    TWO_APPROVED_RUNNERS_NOT_CONFIRMED = "TWO_APPROVED_RUNNERS_NOT_CONFIRMED"
    ENVIRONMENT_LOCK_NOT_PREREGISTERED = "ENVIRONMENT_LOCK_NOT_PREREGISTERED"
    CASE_SEED_REPLAY_NOT_PREREGISTERED = "CASE_SEED_REPLAY_NOT_PREREGISTERED"
    LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED = "LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED"
    SIGNER_TRUST_POLICY_NOT_PREREGISTERED = "SIGNER_TRUST_POLICY_NOT_PREREGISTERED"
    INPUT_INVALID = "INPUT_INVALID"


class Stage10CapabilityError(PermissionError):
    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(c in value for c in "\r\n\x00")


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage10PreflightReport:
    status: Stage10PreflightStatus
    verification_context: Stage10VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage10Reason, ...]
    inputs_present: tuple[tuple[str, bool], ...]
    schema_version: str = STAGE10_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE10_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 10 preflight schema")
        if not isinstance(self.status, Stage10PreflightStatus) or not isinstance(
            self.verification_context, Stage10VerificationContext
        ):
            raise TypeError("Stage 10 preflight uses controlled enums")
        if not all(isinstance(reason, Stage10Reason) for reason in self.reasons):
            raise TypeError("Stage 10 preflight reasons use Stage10Reason")
        if tuple(sorted(self.inputs_present)) != self.inputs_present:
            raise ValueError("inputs_present must be sorted")
        ready = self.status is Stage10PreflightStatus.READY_FOR_RUNNER_CAPABILITY_VERIFICATION
        if ready == bool(self.reasons):
            raise ValueError("Stage 10 preflight status conflicts with reasons")

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
            "dataset_root_probed": False,
            "real_data_read": False,
            "detached_json_authoritative": False,
            "historical_receipt_authoritative": False,
            "synthetic_signature_authorizes_real": False,
            "two_physical_runners_verified": False,
        }


def assess_stage10_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    complete_real_stage9_package_a_capability_present: bool,
    complete_real_stage9_package_b_capability_present: bool,
    fresh_runner_a_capability_present: bool,
    fresh_runner_b_capability_present: bool,
    two_approved_runners_confirmed: bool,
    environment_lock_preregistered: bool,
    case_seed_replay_preregistered: bool,
    locked_hashes_thresholds_preregistered: bool,
    signer_trust_policy_preregistered: bool,
    verification_context: Stage10VerificationContext = Stage10VerificationContext.REAL_PREREQUISITE,
) -> Stage10PreflightReport:
    """Presence-only preflight.  Its signature cannot accept a path or root."""

    flags = {
        "case_seed_replay_preregistered": case_seed_replay_preregistered,
        "complete_real_stage9_package_a_capability_present": (
            complete_real_stage9_package_a_capability_present
        ),
        "complete_real_stage9_package_b_capability_present": (
            complete_real_stage9_package_b_capability_present
        ),
        "environment_lock_preregistered": environment_lock_preregistered,
        "fresh_runner_a_capability_present": fresh_runner_a_capability_present,
        "fresh_runner_b_capability_present": fresh_runner_b_capability_present,
        "locked_hashes_thresholds_preregistered": locked_hashes_thresholds_preregistered,
        "signer_trust_policy_preregistered": signer_trust_policy_preregistered,
        "two_approved_runners_confirmed": two_approved_runners_confirmed,
    }
    if not isinstance(verification_context, Stage10VerificationContext) or any(
        type(value) is not bool
        for value in flags.values()  # noqa: E721
    ):
        return Stage10PreflightReport(
            status=Stage10PreflightStatus.BLOCKED,
            verification_context=Stage10VerificationContext.REAL_PREREQUISITE,
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage10Reason.INPUT_INVALID,),
            inputs_present=tuple(sorted((name, False) for name in flags)),
        )
    reasons: list[Stage10Reason] = []
    if not _valid_identifier(dataset_id):
        reasons.append(Stage10Reason.DATASET_NOT_EXPLICIT)
    if not _valid_identifier(dataset_version):
        reasons.append(Stage10Reason.DATASET_VERSION_NOT_EXPLICIT)
    requirements = (
        (
            complete_real_stage9_package_a_capability_present,
            Stage10Reason.COMPLETE_REAL_STAGE9_PACKAGE_A_CAPABILITY_NOT_PROVIDED,
        ),
        (
            complete_real_stage9_package_b_capability_present,
            Stage10Reason.COMPLETE_REAL_STAGE9_PACKAGE_B_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_runner_a_capability_present,
            Stage10Reason.FRESH_RUNNER_A_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_runner_b_capability_present,
            Stage10Reason.FRESH_RUNNER_B_CAPABILITY_NOT_PROVIDED,
        ),
        (two_approved_runners_confirmed, Stage10Reason.TWO_APPROVED_RUNNERS_NOT_CONFIRMED),
        (
            environment_lock_preregistered,
            Stage10Reason.ENVIRONMENT_LOCK_NOT_PREREGISTERED,
        ),
        (
            case_seed_replay_preregistered,
            Stage10Reason.CASE_SEED_REPLAY_NOT_PREREGISTERED,
        ),
        (
            locked_hashes_thresholds_preregistered,
            Stage10Reason.LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED,
        ),
        (
            signer_trust_policy_preregistered,
            Stage10Reason.SIGNER_TRUST_POLICY_NOT_PREREGISTERED,
        ),
    )
    reasons.extend(reason for present, reason in requirements if not present)
    return Stage10PreflightReport(
        status=(
            Stage10PreflightStatus.BLOCKED
            if reasons
            else Stage10PreflightStatus.READY_FOR_RUNNER_CAPABILITY_VERIFICATION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id if _valid_identifier(dataset_id) else None,
        dataset_version=dataset_version if _valid_identifier(dataset_version) else None,
        reasons=tuple(reasons),
        inputs_present=tuple(sorted(flags.items())),
    )


@dataclass(slots=True, kw_only=True)
class VerifiedStage9RunnerCapability:
    """Live verifier authority; never serializable and never grants root access."""

    snapshot: Stage10RunnerSnapshot
    verifier_process_id: int
    issued_monotonic: float
    expires_monotonic: float
    nonce: str
    _replay: Callable[[], Stage10RunnerSnapshot] = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    consumed: bool = False

    @property
    def authorizes_file_open(self) -> bool:
        return False

    @property
    def authorizes_real_data(self) -> bool:
        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 10 capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 10 capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 10 capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 10 capabilities cannot be serialized")


def _fingerprint(value: VerifiedStage9RunnerCapability) -> str:
    payload = {
        "runner": value.snapshot.authority_dict(),
        "verifier_process_id": value.verifier_process_id,
        "issued_monotonic": value.issued_monotonic,
        "expires_monotonic": value.expires_monotonic,
        "nonce": value.nonce,
    }
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def issue_synthetic_stage9_runner_capability(
    replay: Callable[[], Stage10RunnerSnapshot],
    *,
    ttl_seconds: float = 30.0,
) -> VerifiedStage9RunnerCapability:
    """Issue only SYNTHETIC authority after a complete Stage 9 replay."""

    if (
        not callable(replay)
        or isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int | float)
    ):
        raise Stage10CapabilityError(("synthetic runner issuer input is invalid",))
    ttl = float(ttl_seconds)
    if not 0.0 < ttl <= _MAX_CAPABILITY_TTL_SECONDS:
        raise Stage10CapabilityError(("runner capability TTL is outside the allowed range",))
    snapshot = replay()
    if not isinstance(snapshot, Stage10RunnerSnapshot) or snapshot.scope != "SYNTHETIC_ONLY":
        raise Stage10CapabilityError(("synthetic issuer rejects REAL or malformed snapshots",))
    now = time.monotonic()
    capability = VerifiedStage9RunnerCapability(
        snapshot=snapshot,
        verifier_process_id=os.getpid(),
        issued_monotonic=now,
        expires_monotonic=now + ttl,
        nonce=hashlib.sha256(f"{id(snapshot)}:{time.monotonic_ns()}".encode()).hexdigest(),
        _replay=replay,
        _token=_ISSUER_TOKEN,
    )
    _REGISTRY[id(capability)] = _fingerprint(capability)
    return capability


def runner_capability_validation_reasons(value: object) -> tuple[str, ...]:
    if not isinstance(value, VerifiedStage9RunnerCapability):
        return ("not a live Stage 10 runner capability",)
    reasons: list[str] = []
    if value._token is not _ISSUER_TOKEN or _REGISTRY.get(id(value)) != _fingerprint(value):
        reasons.append("runner capability provenance is invalid")
    if value.verifier_process_id != os.getpid():
        reasons.append("runner capability is bound to another verifier process")
    if value.consumed:
        reasons.append("runner capability was already consumed")
    if time.monotonic() > value.expires_monotonic:
        reasons.append("runner capability expired")
    if value.authorizes_file_open or value.authorizes_real_data:
        reasons.append("runner capability illegally authorizes data or file access")
    return tuple(reasons)


def _pair_reasons(
    left: object,
    right: object,
) -> tuple[str, ...]:
    reasons = [
        *runner_capability_validation_reasons(left),
        *runner_capability_validation_reasons(right),
    ]
    if left is right:
        reasons.append("runner capability cannot fill both pair positions")
    if not isinstance(left, VerifiedStage9RunnerCapability) or not isinstance(
        right, VerifiedStage9RunnerCapability
    ):
        return tuple(dict.fromkeys(reasons))
    a = left.snapshot
    b = right.snapshot
    if (
        a.runner_id == b.runner_id
        or a.producer_process_id == b.producer_process_id
        or a.session_nonce_sha256 == b.session_nonce_sha256
    ):
        reasons.append("runner capabilities are not independent")
    if (a.dataset_id, a.dataset_version, a.scope, a.split) != (
        b.dataset_id,
        b.dataset_version,
        b.scope,
        b.split,
    ):
        reasons.append("runner capability scope or dataset differs")
    if a.environment_lock_sha256 != b.environment_lock_sha256:
        reasons.append("runner environment locks differ")
    if a.source_stage9_package_sha256 != b.source_stage9_package_sha256:
        reasons.append("runner source Stage 9 authority differs")
    return tuple(dict.fromkeys(reasons))


def consume_stage10_runner_pair(
    left: VerifiedStage9RunnerCapability,
    right: VerifiedStage9RunnerCapability,
    consumer: Callable[[Stage10RunnerSnapshot, Stage10RunnerSnapshot], _RESULT],
) -> _RESULT:
    """Atomically consume two authorities and replay both before comparison."""

    if not callable(consumer):
        raise Stage10CapabilityError(("consumer is not callable",))
    reasons = _pair_reasons(left, right)
    if reasons:
        raise Stage10CapabilityError(reasons)
    left.consumed = True
    right.consumed = True
    replayed_left = left._replay()
    replayed_right = right._replay()
    if replayed_left != left.snapshot or replayed_right != right.snapshot:
        raise Stage10CapabilityError(("Stage 9 package changed between issuance and consumption",))
    return consumer(replayed_left, replayed_right)


def runner_capability_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, VerifiedStage9RunnerCapability):
        raise Stage10CapabilityError(("not a live Stage 10 runner capability",))
    return {
        "schema_version": STAGE10_PREFLIGHT_SCHEMA_VERSION,
        "runner": value.snapshot.authority_dict(),
        "verifier_process_id": value.verifier_process_id,
        "consumed": value.consumed,
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "authorizes_file_open": False,
        "authorizes_real_data": False,
    }
