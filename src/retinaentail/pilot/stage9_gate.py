"""Stage 9 dual-session capability gate with a zero-I/O REAL preflight."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Never, TypeVar

from retinaentail.reproducibility.stage9 import Stage9SessionSnapshot

STAGE9_GATE_SCHEMA_VERSION: Final = "9.0.0"
DEFAULT_STAGE9_CAPABILITY_TTL_SECONDS: Final = 30
MAX_STAGE9_CAPABILITY_TTL_SECONDS: Final = 60

_PROCESS_ID: Final = os.getpid()
_CAPABILITY_TOKEN: Final = object()
_RESULT = TypeVar("_RESULT")


class Stage9VerificationContext(StrEnum):
    REAL_PREREQUISITE = "REAL_PREREQUISITE"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class Stage9PreflightStatus(StrEnum):
    BLOCKED = "BLOCKED"
    READY_FOR_DUAL_SESSION_CAPABILITY_VERIFICATION = (
        "READY_FOR_DUAL_SESSION_CAPABILITY_VERIFICATION"
    )


class Stage9Reason(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    COMPLETE_REAL_STAGE8_PACKAGE_A_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE8_PACKAGE_A_CAPABILITY_NOT_PROVIDED"
    )
    COMPLETE_REAL_STAGE8_PACKAGE_B_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE8_PACKAGE_B_CAPABILITY_NOT_PROVIDED"
    )
    FRESH_SESSION_A_CAPABILITY_NOT_PROVIDED = "FRESH_SESSION_A_CAPABILITY_NOT_PROVIDED"
    FRESH_SESSION_B_CAPABILITY_NOT_PROVIDED = "FRESH_SESSION_B_CAPABILITY_NOT_PROVIDED"
    ENVIRONMENT_LOCK_NOT_PREREGISTERED = "ENVIRONMENT_LOCK_NOT_PREREGISTERED"
    CASE_SEED_REPLAY_NOT_PREREGISTERED = "CASE_SEED_REPLAY_NOT_PREREGISTERED"
    LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED = "LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED"


class Stage9CapabilityError(PermissionError):
    """Raised when live verifier authority is missing, stale, reused, or inconsistent."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(char in value for char in "\r\n\x00")


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9PreflightReport:
    status: Stage9PreflightStatus
    verification_context: Stage9VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage9Reason, ...]
    schema_version: str = STAGE9_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE9_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 9 preflight schema")
        if not isinstance(self.status, Stage9PreflightStatus):
            raise TypeError("status must be Stage9PreflightStatus")
        if not isinstance(self.verification_context, Stage9VerificationContext):
            raise TypeError("verification_context must be Stage9VerificationContext")
        if not all(isinstance(reason, Stage9Reason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage9Reason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("preflight reasons must be unique")
        ready = self.status is Stage9PreflightStatus.READY_FOR_DUAL_SESSION_CAPABILITY_VERIFICATION
        if ready == bool(self.reasons):
            raise ValueError("preflight status and reasons conflict")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "reasons": [reason.value for reason in self.reasons],
            "accepts_paths": False,
            "reads_environment": False,
            "authorizes_file_open": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "detached_pass_json_authoritative": False,
            "cross_session_capability_reuse_allowed": False,
            "synthetic_authorizes_real": False,
        }


def assess_stage9_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    complete_stage8_package_a_capability_provided: bool,
    complete_stage8_package_b_capability_provided: bool,
    fresh_session_a_capability_provided: bool,
    fresh_session_b_capability_provided: bool,
    environment_lock_preregistered: bool,
    case_seed_replay_preregistered: bool,
    locked_hashes_thresholds_preregistered: bool,
    verification_context: Stage9VerificationContext = Stage9VerificationContext.REAL_PREREQUISITE,
) -> Stage9PreflightReport:
    """Evaluate presence only; no supplied path or ambient environment is accepted."""

    flags = (
        complete_stage8_package_a_capability_provided,
        complete_stage8_package_b_capability_provided,
        fresh_session_a_capability_provided,
        fresh_session_b_capability_provided,
        environment_lock_preregistered,
        case_seed_replay_preregistered,
        locked_hashes_thresholds_preregistered,
    )
    if not isinstance(verification_context, Stage9VerificationContext) or any(
        type(item) is not bool
        for item in flags  # noqa: E721
    ):
        return Stage9PreflightReport(
            status=Stage9PreflightStatus.BLOCKED,
            verification_context=Stage9VerificationContext.REAL_PREREQUISITE,
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage9Reason.INPUT_INVALID,),
        )
    reasons: list[Stage9Reason] = []
    if not _valid_identifier(dataset_id):
        reasons.append(Stage9Reason.DATASET_NOT_EXPLICIT)
    if not _valid_identifier(dataset_version):
        reasons.append(Stage9Reason.DATASET_VERSION_NOT_EXPLICIT)
    requirements = (
        (
            complete_stage8_package_a_capability_provided,
            Stage9Reason.COMPLETE_REAL_STAGE8_PACKAGE_A_CAPABILITY_NOT_PROVIDED,
        ),
        (
            complete_stage8_package_b_capability_provided,
            Stage9Reason.COMPLETE_REAL_STAGE8_PACKAGE_B_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_session_a_capability_provided,
            Stage9Reason.FRESH_SESSION_A_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_session_b_capability_provided,
            Stage9Reason.FRESH_SESSION_B_CAPABILITY_NOT_PROVIDED,
        ),
        (environment_lock_preregistered, Stage9Reason.ENVIRONMENT_LOCK_NOT_PREREGISTERED),
        (
            case_seed_replay_preregistered,
            Stage9Reason.CASE_SEED_REPLAY_NOT_PREREGISTERED,
        ),
        (
            locked_hashes_thresholds_preregistered,
            Stage9Reason.LOCKED_HASHES_THRESHOLDS_NOT_PREREGISTERED,
        ),
    )
    reasons.extend(reason for provided, reason in requirements if not provided)
    normalized = tuple(dict.fromkeys(reasons))
    return Stage9PreflightReport(
        status=(
            Stage9PreflightStatus.BLOCKED
            if normalized
            else Stage9PreflightStatus.READY_FOR_DUAL_SESSION_CAPABILITY_VERIFICATION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id if _valid_identifier(dataset_id) else None,
        dataset_version=dataset_version if _valid_identifier(dataset_version) else None,
        reasons=normalized,
    )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedStage8SessionCapability:
    """Opaque current-verifier-process authority; it cannot cross a process boundary."""

    snapshot: Stage9SessionSnapshot
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _replay: Callable[[], Stage9SessionSnapshot] = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    @property
    def authorizes_file_open(self) -> bool:
        return False

    @property
    def authorizes_real_data(self) -> bool:
        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 9 session capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 9 session capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 9 session capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 9 session capabilities cannot be serialized")


_ISSUED: dict[int, tuple[VerifiedStage8SessionCapability, str]] = {}
_CONSUMED: set[int] = set()


def _fingerprint(value: VerifiedStage8SessionCapability) -> str:
    payload = {
        "snapshot": value.snapshot.authority_dict(),
        "issued_process_id": value.issued_process_id,
        "issued_monotonic_ns": value.issued_monotonic_ns,
        "expires_monotonic_ns": value.expires_monotonic_ns,
        "ttl_seconds": value.ttl_seconds,
        "replay_id": id(value._replay),
    }
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def issue_synthetic_stage8_session_capability(
    replay: Callable[[], Stage9SessionSnapshot],
    *,
    ttl_seconds: int = DEFAULT_STAGE9_CAPABILITY_TTL_SECONDS,
) -> VerifiedStage8SessionCapability:
    """Replay one complete synthetic Stage 8 package and issue no-root authority."""

    if (
        not callable(replay)
        or type(ttl_seconds) is not int  # noqa: E721
        or not 1 <= ttl_seconds <= MAX_STAGE9_CAPABILITY_TTL_SECONDS
    ):
        raise Stage9CapabilityError(("synthetic session issuer input is invalid",))
    try:
        snapshot = replay()
    except Exception as error:
        raise Stage9CapabilityError(
            ("complete synthetic Stage 8 package replay failed",)
        ) from error
    if not isinstance(snapshot, Stage9SessionSnapshot) or snapshot.scope != "SYNTHETIC_ONLY":
        raise Stage9CapabilityError(("synthetic issuer rejects REAL or malformed snapshots",))
    now = time.monotonic_ns()
    value = object.__new__(VerifiedStage8SessionCapability)
    for name, item in (
        ("snapshot", snapshot),
        ("issued_process_id", _PROCESS_ID),
        ("issued_monotonic_ns", now),
        ("expires_monotonic_ns", now + ttl_seconds * 1_000_000_000),
        ("ttl_seconds", ttl_seconds),
        ("_replay", replay),
        ("_token", _CAPABILITY_TOKEN),
    ):
        object.__setattr__(value, name, item)
    _ISSUED[id(value)] = (value, _fingerprint(value))
    return value


def session_capability_validation_reasons(value: object) -> tuple[str, ...]:
    reasons: list[str] = []
    if not isinstance(value, VerifiedStage8SessionCapability):
        return ("not a live Stage 9 session capability",)
    registered = _ISSUED.get(id(value))
    if registered is None or registered[0] is not value or registered[1] != _fingerprint(value):
        reasons.append("capability was not issued by this verifier process or was mutated")
    if value._token is not _CAPABILITY_TOKEN or value.issued_process_id != _PROCESS_ID:
        reasons.append("capability is process-detached")
    if id(value) in _CONSUMED:
        reasons.append("capability was already consumed")
    if time.monotonic_ns() > value.expires_monotonic_ns:
        reasons.append("capability expired")
    if value.snapshot.scope != "SYNTHETIC_ONLY":
        reasons.append("public synthetic capability cannot authorize REAL scope")
    return tuple(dict.fromkeys(reasons))


def _pair_reasons(
    first: object, second: object
) -> tuple[VerifiedStage8SessionCapability, VerifiedStage8SessionCapability, tuple[str, ...]]:
    reasons = [*session_capability_validation_reasons(first)]
    reasons.extend(session_capability_validation_reasons(second))
    if not isinstance(first, VerifiedStage8SessionCapability) or not isinstance(
        second, VerifiedStage8SessionCapability
    ):
        raise Stage9CapabilityError(tuple(dict.fromkeys(reasons)))
    if first is second:
        reasons.append("the same capability cannot be used for both sessions")
    left = first.snapshot
    right = second.snapshot
    if left.producer_process_id == right.producer_process_id:
        reasons.append("producer process IDs are not independent")
    if left.session_nonce_sha256 == right.session_nonce_sha256:
        reasons.append("session nonces are not independent")
    comparable = (
        "environment_lock_sha256",
        "dataset_id",
        "dataset_version",
        "scope",
        "split",
        "package_sha256",
        "package_file_count",
        "package_manifest_sha256",
        "stage8_audit_sha256",
        "stage8_observations_sha256",
        "stage8_preregistration_sha256",
        "source_stage7_package_sha256",
    )
    if any(getattr(left, name) != getattr(right, name) for name in comparable):
        reasons.append("session package, environment, scope, split, hash, or provenance drifted")
    return first, second, tuple(dict.fromkeys(reasons))


def consume_stage9_session_pair(
    first: object,
    second: object,
    consumer: Callable[[Stage9SessionSnapshot, Stage9SessionSnapshot], _RESULT],
) -> _RESULT:
    """Atomically consume two authorities, replay both packages, then compare snapshots."""

    left_cap, right_cap, reasons = _pair_reasons(first, second)
    if reasons or not callable(consumer):
        raise Stage9CapabilityError(reasons or ("consumer is not callable",))
    _CONSUMED.update((id(left_cap), id(right_cap)))
    try:
        replayed_left = left_cap._replay()
        replayed_right = right_cap._replay()
    except Exception as error:
        raise Stage9CapabilityError(
            ("complete Stage 8 package replay failed at consumption",)
        ) from error
    if replayed_left != left_cap.snapshot or replayed_right != right_cap.snapshot:
        raise Stage9CapabilityError(("Stage 8 package changed between issuance and consumption",))
    return consumer(replayed_left, replayed_right)


def session_capability_snapshot(value: object) -> dict[str, object]:
    """Return lifecycle metadata only; this detached receipt grants no authority."""

    if not isinstance(value, VerifiedStage8SessionCapability):
        raise TypeError("value must be a Stage 9 session capability")
    return {
        "schema_version": STAGE9_GATE_SCHEMA_VERSION,
        "issued_process_id": value.issued_process_id,
        "ttl_seconds": value.ttl_seconds,
        "consumed": id(value) in _CONSUMED,
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "authorizes_file_open": False,
        "authorizes_real_data": False,
        "detached_receipt_authoritative": False,
        "producer_process_id": value.snapshot.producer_process_id,
        "session_nonce_sha256": value.snapshot.session_nonce_sha256,
        "stage8_package_sha256": value.snapshot.package_sha256,
    }
