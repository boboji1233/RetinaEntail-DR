"""Stage 8 process-bound prerequisite and root-consumption boundary.

The presence-only preflight is deliberately incapable of accepting a path.  A
REAL run additionally needs two live objects from the current process:

* a complete Stage 7 package capability issued by a repository-owned verifier;
* a fresh :class:`~retinaentail.pilot.stage7_gate.Stage7RootTransaction` for the
  same dataset/version and physical-root identity.

This module intentionally has no public REAL Stage 7 package issuer.  The
repository currently contains only a complete synthetic Stage 7 verifier; a
future REAL package verifier must call the private issuer only after deriving the
descriptor from the verified bytes.  The public synthetic issuer is explicitly
fixture-only and can never be consumed under the REAL context.

The callback boundary is cooperative, not an operating-system sandbox.  Only
repository-owned read-only audit callbacks are trusted with the root for their
dynamic extent, and returned values are checked recursively for root disclosure.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import posixpath
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Never, TypeVar, cast

from retinaentail.pilot.stage6_gate import Stage6VerificationContext
from retinaentail.pilot.stage7_gate import (
    Stage7RootTransaction,
    Stage7RootTransactionError,
    Stage7RunEvidence,
    consume_stage7_root_transaction,
    transaction_validation_reasons,
)

STAGE8_GATE_SCHEMA_VERSION: Final = "8.0.0"
DEFAULT_STAGE8_CAPABILITY_TTL_SECONDS: Final = 5
MAX_STAGE8_CAPABILITY_TTL_SECONDS: Final = 5
STAGE8_NO_PHYSICAL_ROOT_SHA256: Final = hashlib.sha256(
    b"retinaentail-stage8:no-physical-dataset-root:v1"
).hexdigest()

_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_ID: Final = os.getpid()
_PACKAGE_TOKEN: Final = object()
_RUN_TOKEN: Final = object()
_BOUNDARY_LOCK: Final = threading.RLock()
_NO_RESULT: Final = object()

_T = TypeVar("_T")


class Stage8PackageScope(StrEnum):
    """Evidence scope; synthetic evidence is never substitutable for REAL."""

    APPROVED_REAL_DATA = "APPROVED_REAL_DATA"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class Stage8PreflightStatus(StrEnum):
    """Presence-only result; READY never grants file access."""

    BLOCKED = "BLOCKED"
    READY_FOR_PROCESS_CAPABILITY_VERIFICATION = "READY_FOR_PROCESS_CAPABILITY_VERIFICATION"


class Stage8Reason(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    COMPLETE_REAL_STAGE7_PACKAGE_CAPABILITY_NOT_PROVIDED = (
        "COMPLETE_REAL_STAGE7_PACKAGE_CAPABILITY_NOT_PROVIDED"
    )
    FRESH_ROOT_RUN_CAPABILITY_NOT_PROVIDED = "FRESH_ROOT_RUN_CAPABILITY_NOT_PROVIDED"
    FIXED_SEED_NOT_PREREGISTERED = "FIXED_SEED_NOT_PREREGISTERED"
    DEGRADATION_REPLAY_SET_NOT_PREREGISTERED = "DEGRADATION_REPLAY_SET_NOT_PREREGISTERED"
    INCLUSION_RULES_NOT_PREREGISTERED = "INCLUSION_RULES_NOT_PREREGISTERED"
    STABILITY_METRICS_NOT_PREREGISTERED = "STABILITY_METRICS_NOT_PREREGISTERED"
    LOCKED_CONTRACT_NOT_PROVIDED = "LOCKED_CONTRACT_NOT_PROVIDED"


class Stage8CapabilityError(PermissionError):
    """Fail-closed error at the Stage 8 authority boundary."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 8 capability rejected: " + "; ".join(reasons))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_dataset_id(value: object) -> bool:
    return isinstance(value, str) and _DATASET_ID_PATTERN.fullmatch(value) is not None


def _valid_dataset_version(value: object) -> bool:
    return isinstance(value, str) and _DATASET_VERSION_PATTERN.fullmatch(value) is not None


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _expected_scope(context: Stage6VerificationContext) -> Stage8PackageScope:
    return (
        Stage8PackageScope.APPROVED_REAL_DATA
        if context is Stage6VerificationContext.REAL_PREREQUISITE
        else Stage8PackageScope.SYNTHETIC_FIXTURE
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8PreflightReport:
    status: Stage8PreflightStatus
    verification_context: Stage6VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage8Reason, ...]
    schema_version: str = STAGE8_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 Gate schema")
        if not isinstance(self.status, Stage8PreflightStatus):
            raise TypeError("status must be Stage8PreflightStatus")
        if not isinstance(self.verification_context, Stage6VerificationContext):
            raise TypeError("verification_context must be Stage6VerificationContext")
        if not all(isinstance(reason, Stage8Reason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage8Reason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must be unique")
        ready = self.status is Stage8PreflightStatus.READY_FOR_PROCESS_CAPABILITY_VERIFICATION
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
            "detached_receipt_authoritative": False,
            "synthetic_authorizes_real": False,
        }


def assess_stage8_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    complete_stage7_package_capability_provided: bool,
    fresh_root_run_capability_provided: bool,
    fixed_seed_preregistered: bool,
    degradation_replay_set_preregistered: bool,
    inclusion_rules_preregistered: bool,
    stability_metrics_preregistered: bool,
    locked_contract_provided: bool,
    verification_context: Stage6VerificationContext = (Stage6VerificationContext.REAL_PREREQUISITE),
) -> Stage8PreflightReport:
    """Check presence flags and identifiers without accepting or probing a root."""

    flags = (
        complete_stage7_package_capability_provided,
        fresh_root_run_capability_provided,
        fixed_seed_preregistered,
        degradation_replay_set_preregistered,
        inclusion_rules_preregistered,
        stability_metrics_preregistered,
        locked_contract_provided,
    )
    if not isinstance(verification_context, Stage6VerificationContext) or any(
        type(value) is not bool
        for value in flags  # noqa: E721
    ):
        return Stage8PreflightReport(
            status=Stage8PreflightStatus.BLOCKED,
            verification_context=(
                verification_context
                if isinstance(verification_context, Stage6VerificationContext)
                else Stage6VerificationContext.REAL_PREREQUISITE
            ),
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage8Reason.INPUT_INVALID,),
        )
    reasons: list[Stage8Reason] = []
    if not _valid_dataset_id(dataset_id):
        reasons.append(Stage8Reason.DATASET_NOT_EXPLICIT)
    if not _valid_dataset_version(dataset_version):
        reasons.append(Stage8Reason.DATASET_VERSION_NOT_EXPLICIT)
    requirements = (
        (
            complete_stage7_package_capability_provided,
            Stage8Reason.COMPLETE_REAL_STAGE7_PACKAGE_CAPABILITY_NOT_PROVIDED,
        ),
        (
            fresh_root_run_capability_provided,
            Stage8Reason.FRESH_ROOT_RUN_CAPABILITY_NOT_PROVIDED,
        ),
        (fixed_seed_preregistered, Stage8Reason.FIXED_SEED_NOT_PREREGISTERED),
        (
            degradation_replay_set_preregistered,
            Stage8Reason.DEGRADATION_REPLAY_SET_NOT_PREREGISTERED,
        ),
        (
            inclusion_rules_preregistered,
            Stage8Reason.INCLUSION_RULES_NOT_PREREGISTERED,
        ),
        (
            stability_metrics_preregistered,
            Stage8Reason.STABILITY_METRICS_NOT_PREREGISTERED,
        ),
        (locked_contract_provided, Stage8Reason.LOCKED_CONTRACT_NOT_PROVIDED),
    )
    reasons.extend(reason for present, reason in requirements if not present)
    normalized = tuple(dict.fromkeys(reasons))
    return Stage8PreflightReport(
        status=(
            Stage8PreflightStatus.BLOCKED
            if normalized
            else Stage8PreflightStatus.READY_FOR_PROCESS_CAPABILITY_VERIFICATION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id if _valid_dataset_id(dataset_id) else None,
        dataset_version=(dataset_version if _valid_dataset_version(dataset_version) else None),
        reasons=normalized,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8FrozenBindings:
    """Pre-registered test, model, replay, inclusion, and metric bindings."""

    fixed_seed: int
    degradation_replay_set_sha256: str
    inclusion_rules_sha256: str
    stability_metrics_sha256: str
    locked_contract_sha256: str
    test_split_sha256: str
    config_hash: str
    data_hash: str
    model_hash: str
    provenance_sha256: str
    preregistration_sha256: str
    split: str = "test"
    schema_version: str = STAGE8_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_GATE_SCHEMA_VERSION or self.split != "test":
            raise ValueError("Stage 8 frozen binding version or split is invalid")
        if type(self.fixed_seed) is not int or not 0 <= self.fixed_seed < 2**63:  # noqa: E721
            raise ValueError("fixed_seed must be a non-negative signed 64-bit integer")
        for name in (
            "degradation_replay_set_sha256",
            "inclusion_rules_sha256",
            "stability_metrics_sha256",
            "locked_contract_sha256",
            "test_split_sha256",
            "config_hash",
            "data_hash",
            "model_hash",
            "provenance_sha256",
            "preregistration_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "fixed_seed": self.fixed_seed,
            "degradation_replay_set_sha256": self.degradation_replay_set_sha256,
            "inclusion_rules_sha256": self.inclusion_rules_sha256,
            "stability_metrics_sha256": self.stability_metrics_sha256,
            "locked_contract_sha256": self.locked_contract_sha256,
            "test_split_sha256": self.test_split_sha256,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "provenance_sha256": self.provenance_sha256,
            "preregistration_sha256": self.preregistration_sha256,
        }

    def binding_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7PackageDescriptor:
    """Descriptor derived from complete package bytes by a trusted verifier."""

    scope: Stage8PackageScope
    dataset_id: str
    dataset_version: str
    package_sha256: str
    package_file_count: int
    stage2_package_sha256: str
    stage3_package_sha256: str
    physical_root_identity_sha256: str
    frozen_bindings: Stage8FrozenBindings
    fixture_only: bool
    schema_version: str = STAGE8_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 7 package capability schema")
        if not isinstance(self.scope, Stage8PackageScope):
            raise TypeError("scope must be Stage8PackageScope")
        if not _valid_dataset_id(self.dataset_id) or not _valid_dataset_version(
            self.dataset_version
        ):
            raise ValueError("Stage 7 package dataset identity is invalid")
        if type(self.package_file_count) is not int or self.package_file_count <= 0:  # noqa: E721
            raise ValueError("package_file_count must be positive")
        for name in (
            "package_sha256",
            "stage2_package_sha256",
            "stage3_package_sha256",
            "physical_root_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.frozen_bindings, Stage8FrozenBindings):
            raise TypeError("frozen_bindings must be Stage8FrozenBindings")
        if type(self.fixture_only) is not bool:  # noqa: E721
            raise TypeError("fixture_only must be bool")
        if (self.scope is Stage8PackageScope.SYNTHETIC_FIXTURE) != self.fixture_only:
            raise ValueError("package scope and fixture_only are inconsistent")
        if (
            self.scope is Stage8PackageScope.SYNTHETIC_FIXTURE
            and self.physical_root_identity_sha256 != STAGE8_NO_PHYSICAL_ROOT_SHA256
        ):
            raise ValueError("synthetic package descriptor must bind no physical root")
        if (
            self.scope is Stage8PackageScope.APPROVED_REAL_DATA
            and self.physical_root_identity_sha256 == STAGE8_NO_PHYSICAL_ROOT_SHA256
        ):
            raise ValueError("REAL package descriptor must bind a physical root identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "package_sha256": self.package_sha256,
            "package_file_count": self.package_file_count,
            "stage2_package_sha256": self.stage2_package_sha256,
            "stage3_package_sha256": self.stage3_package_sha256,
            "physical_root_identity_sha256": self.physical_root_identity_sha256,
            "frozen_bindings": self.frozen_bindings.to_dict(),
            "fixture_only": self.fixture_only,
        }

    def binding_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True, init=False)
class VerifiedStage7PackageCapability:
    """Opaque complete-package replay capability; detached receipts have no authority."""

    descriptor: Stage7PackageDescriptor
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _replay: Callable[[], Stage7PackageDescriptor] = field(repr=False, compare=False)
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    @property
    def authorizes_file_open(self) -> bool:
        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 7 package capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 7 package capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 7 package capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 7 package capabilities cannot be serialized")


_ISSUED_PACKAGE_CAPABILITIES: dict[int, tuple[VerifiedStage7PackageCapability, str]] = {}
_CONSUMED_PACKAGE_CAPABILITIES: set[int] = set()


def _package_capability_fingerprint(value: VerifiedStage7PackageCapability) -> str:
    return _canonical_sha256(
        {
            "descriptor_sha256": value.descriptor.binding_sha256(),
            "issued_process_id": value.issued_process_id,
            "issued_monotonic_ns": value.issued_monotonic_ns,
            "expires_monotonic_ns": value.expires_monotonic_ns,
            "ttl_seconds": value.ttl_seconds,
            "replay_id": id(value._replay),
        }
    )


def _issue_package_capability(
    descriptor: Stage7PackageDescriptor,
    replay: Callable[[], Stage7PackageDescriptor],
    *,
    ttl_seconds: int,
    issuer_token: object,
) -> VerifiedStage7PackageCapability:
    """Private issuer boundary for a concrete, complete package verifier."""

    if (
        issuer_token is not _PACKAGE_TOKEN
        or not isinstance(descriptor, Stage7PackageDescriptor)
        or not callable(replay)
        or type(ttl_seconds) is not int  # noqa: E721
        or not 1 <= ttl_seconds <= MAX_STAGE8_CAPABILITY_TTL_SECONDS
    ):
        raise Stage8CapabilityError(("Stage 7 package issuer input is invalid",))
    try:
        replayed = replay()
    except Exception as error:
        raise Stage8CapabilityError(("complete Stage 7 package replay failed",)) from error
    if not isinstance(replayed, Stage7PackageDescriptor) or replayed != descriptor:
        raise Stage8CapabilityError(("replayed Stage 7 package descriptor does not match",))
    now = time.monotonic_ns()
    value = object.__new__(VerifiedStage7PackageCapability)
    assignments: tuple[tuple[str, object], ...] = (
        ("descriptor", descriptor),
        ("issued_process_id", _PROCESS_ID),
        ("issued_monotonic_ns", now),
        ("expires_monotonic_ns", now + ttl_seconds * 1_000_000_000),
        ("ttl_seconds", ttl_seconds),
        ("_replay", replay),
        ("_process_token", _PACKAGE_TOKEN),
        ("_token", _PACKAGE_TOKEN),
    )
    for name, item in assignments:
        object.__setattr__(value, name, item)
    _ISSUED_PACKAGE_CAPABILITIES[id(value)] = (
        value,
        _package_capability_fingerprint(value),
    )
    return value


def issue_synthetic_stage7_package_capability(
    replay: Callable[[], Stage7PackageDescriptor],
    *,
    ttl_seconds: int = DEFAULT_STAGE8_CAPABILITY_TTL_SECONDS,
) -> VerifiedStage7PackageCapability:
    """Issue fixture-only authority after an explicit complete synthetic replay.

    ``replay`` must be a repository-owned wrapper around the complete Stage 7 toy
    package verifier.  Its returned descriptor is replayed once at issuance and
    again immediately before Stage 8 binding.  This public helper deliberately
    rejects REAL descriptors, so it cannot bridge the missing REAL verifier.
    """

    if not callable(replay):
        raise Stage8CapabilityError(("synthetic Stage 7 replay must be callable",))
    try:
        descriptor = replay()
    except Exception as error:
        raise Stage8CapabilityError(
            ("complete synthetic Stage 7 package replay failed",)
        ) from error
    if (
        not isinstance(descriptor, Stage7PackageDescriptor)
        or descriptor.scope is not Stage8PackageScope.SYNTHETIC_FIXTURE
        or not descriptor.fixture_only
    ):
        raise Stage8CapabilityError(
            ("synthetic issuer accepts only a SYNTHETIC_FIXTURE descriptor",)
        )
    return _issue_package_capability(
        descriptor,
        replay,
        ttl_seconds=ttl_seconds,
        issuer_token=_PACKAGE_TOKEN,
    )


def package_capability_validation_reasons(
    value: object,
    *,
    required_context: object = Stage6VerificationContext.REAL_PREREQUISITE,
    expected_dataset_id: object | None = None,
    expected_dataset_version: object | None = None,
) -> tuple[str, ...]:
    """Validate issuance metadata without replaying package bytes or touching a root."""

    if not isinstance(required_context, Stage6VerificationContext):
        return ("required context is invalid",)
    if not isinstance(value, VerifiedStage7PackageCapability):
        return ("Stage 7 package capability has an invalid type",)
    issued = _ISSUED_PACKAGE_CAPABILITIES.get(id(value))
    if issued is None or issued[0] is not value:
        return ("Stage 7 package capability was not issued by a trusted verifier",)
    if id(value) in _CONSUMED_PACKAGE_CAPABILITIES:
        return ("Stage 7 package capability has already been consumed",)
    try:
        reasons: list[str] = []
        if issued[1] != _package_capability_fingerprint(value):
            reasons.append("Stage 7 package capability bindings changed after issuance")
        if value._process_token is not _PACKAGE_TOKEN or value._token is not _PACKAGE_TOKEN:
            reasons.append("Stage 7 package capability issuance identity is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("Stage 7 package capability belongs to a different process")
        if not 1 <= value.ttl_seconds <= MAX_STAGE8_CAPABILITY_TTL_SECONDS:
            reasons.append("Stage 7 package capability TTL is invalid")
        if value.expires_monotonic_ns - value.issued_monotonic_ns != (
            value.ttl_seconds * 1_000_000_000
        ):
            reasons.append("Stage 7 package capability expiry does not match TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("Stage 7 package capability has expired")
        if value.descriptor.scope is not _expected_scope(required_context):
            reasons.append(
                f"Stage 7 package scope must be {_expected_scope(required_context).value}"
            )
        if expected_dataset_id is not None and value.descriptor.dataset_id != expected_dataset_id:
            reasons.append("Stage 7 package dataset_id does not match")
        if (
            expected_dataset_version is not None
            and value.descriptor.dataset_version != expected_dataset_version
        ):
            reasons.append("Stage 7 package dataset_version does not match")
        return tuple(dict.fromkeys(reasons))
    except (AttributeError, TypeError, ValueError):
        return ("Stage 7 package capability fields are invalid",)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8SyntheticRunEvidence:
    """Fixture-only evidence; it contains no physical-root identity or authority."""

    dataset_id: str
    dataset_version: str
    stage7_package_descriptor: Stage7PackageDescriptor
    frozen_bindings: Stage8FrozenBindings
    package_descriptor_sha256: str
    schema_version: str = STAGE8_GATE_SCHEMA_VERSION
    verification_context: Stage6VerificationContext = Stage6VerificationContext.SYNTHETIC_FIXTURE

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 synthetic evidence schema")
        if self.verification_context is not Stage6VerificationContext.SYNTHETIC_FIXTURE:
            raise ValueError("synthetic evidence context is immutable")
        if not _valid_dataset_id(self.dataset_id) or not _valid_dataset_version(
            self.dataset_version
        ):
            raise ValueError("synthetic evidence dataset identity is invalid")
        descriptor = self.stage7_package_descriptor
        if (
            not isinstance(descriptor, Stage7PackageDescriptor)
            or descriptor.scope is not Stage8PackageScope.SYNTHETIC_FIXTURE
            or not descriptor.fixture_only
            or descriptor.dataset_id != self.dataset_id
            or descriptor.dataset_version != self.dataset_version
            or descriptor.physical_root_identity_sha256 != STAGE8_NO_PHYSICAL_ROOT_SHA256
        ):
            raise ValueError("synthetic evidence package descriptor is invalid")
        if descriptor.frozen_bindings != self.frozen_bindings:
            raise ValueError("synthetic evidence frozen bindings differ")
        _require_sha256(self.package_descriptor_sha256, "package_descriptor_sha256")
        if self.package_descriptor_sha256 != descriptor.binding_sha256():
            raise ValueError("synthetic package descriptor hash does not match")

    @property
    def authorizes_dataset_root_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "stage7_package": self.stage7_package_descriptor.to_dict(),
            "frozen_bindings": self.frozen_bindings.to_dict(),
            "package_descriptor_sha256": self.package_descriptor_sha256,
            "authorizes_dataset_root_access": False,
            "authorizes_real_data": False,
            "authorizes_physical_dataset_root": False,
        }

    def binding_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True, init=False)
class Stage8SyntheticRunCapability:
    """Fixture-only one-shot authority to replay an embedded Stage 7 package."""

    dataset_id: str
    dataset_version: str
    evidence: Stage8SyntheticRunEvidence
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _package_capability: VerifiedStage7PackageCapability = field(repr=False, compare=False)
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    @property
    def authorizes_file_open(self) -> bool:
        return False

    @property
    def authorizes_embedded_package_replay(self) -> bool:
        return True

    def __copy__(self) -> Never:
        raise TypeError("Stage 8 synthetic run capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 8 synthetic run capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 8 synthetic run capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 8 synthetic run capabilities cannot be serialized")


_ISSUED_SYNTHETIC_RUN_CAPABILITIES: dict[
    int,
    tuple[Stage8SyntheticRunCapability, str],
] = {}
_CONSUMED_SYNTHETIC_RUN_CAPABILITIES: set[int] = set()
_REPLAYED_SYNTHETIC_RUN_CAPABILITIES: set[int] = set()
_SECOND_USE_REJECTED_SYNTHETIC_RUN_CAPABILITIES: set[int] = set()


def _synthetic_run_fingerprint(value: Stage8SyntheticRunCapability) -> str:
    return _canonical_sha256(
        {
            "dataset_id": value.dataset_id,
            "dataset_version": value.dataset_version,
            "evidence_sha256": value.evidence.binding_sha256(),
            "issued_process_id": value.issued_process_id,
            "issued_monotonic_ns": value.issued_monotonic_ns,
            "expires_monotonic_ns": value.expires_monotonic_ns,
            "ttl_seconds": value.ttl_seconds,
            "package_capability_id": id(value._package_capability),
        }
    )


def begin_stage8_synthetic_run_capability(
    stage7_package_capability: object,
    *,
    expected_dataset_id: str,
    expected_dataset_version: str,
    frozen_bindings: Stage8FrozenBindings,
    ttl_seconds: int = DEFAULT_STAGE8_CAPABILITY_TTL_SECONDS,
) -> Stage8SyntheticRunCapability:
    """Consume fixture package authority into a fresh no-root run capability."""

    if (
        not _valid_dataset_id(expected_dataset_id)
        or not _valid_dataset_version(expected_dataset_version)
        or not isinstance(frozen_bindings, Stage8FrozenBindings)
        or type(ttl_seconds) is not int  # noqa: E721
        or not 1 <= ttl_seconds <= MAX_STAGE8_CAPABILITY_TTL_SECONDS
    ):
        raise Stage8CapabilityError(("Stage 8 synthetic run input is invalid",))
    reasons = package_capability_validation_reasons(
        stage7_package_capability,
        required_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
    )
    if reasons:
        raise Stage8CapabilityError(reasons)
    assert isinstance(stage7_package_capability, VerifiedStage7PackageCapability)
    descriptor = stage7_package_capability.descriptor
    if descriptor.frozen_bindings != frozen_bindings:
        raise Stage8CapabilityError(("synthetic package frozen bindings differ",))
    if descriptor.physical_root_identity_sha256 != STAGE8_NO_PHYSICAL_ROOT_SHA256:
        raise Stage8CapabilityError(("synthetic package no-root binding differs",))
    try:
        replayed = stage7_package_capability._replay()
    except Exception as error:
        raise Stage8CapabilityError(("complete synthetic Stage 7 replay failed",)) from error
    if not isinstance(replayed, Stage7PackageDescriptor) or replayed != descriptor:
        raise Stage8CapabilityError(("replayed synthetic Stage 7 descriptor changed",))
    now = time.monotonic_ns()
    deadline = min(
        now + ttl_seconds * 1_000_000_000,
        stage7_package_capability.expires_monotonic_ns,
    )
    if deadline <= now:
        raise Stage8CapabilityError(("synthetic package capability expired during binding",))
    evidence = Stage8SyntheticRunEvidence(
        dataset_id=expected_dataset_id,
        dataset_version=expected_dataset_version,
        stage7_package_descriptor=descriptor,
        frozen_bindings=frozen_bindings,
        package_descriptor_sha256=descriptor.binding_sha256(),
    )
    value = object.__new__(Stage8SyntheticRunCapability)
    assignments: tuple[tuple[str, object], ...] = (
        ("dataset_id", expected_dataset_id),
        ("dataset_version", expected_dataset_version),
        ("evidence", evidence),
        ("issued_process_id", _PROCESS_ID),
        ("issued_monotonic_ns", now),
        ("expires_monotonic_ns", deadline),
        ("ttl_seconds", ttl_seconds),
        ("_package_capability", stage7_package_capability),
        ("_process_token", _RUN_TOKEN),
        ("_token", _RUN_TOKEN),
    )
    for name, item in assignments:
        object.__setattr__(value, name, item)
    with _BOUNDARY_LOCK:
        if id(stage7_package_capability) in _CONSUMED_PACKAGE_CAPABILITIES:
            raise Stage8CapabilityError(("Stage 7 package capability has already been consumed",))
        _CONSUMED_PACKAGE_CAPABILITIES.add(id(stage7_package_capability))
        _ISSUED_SYNTHETIC_RUN_CAPABILITIES[id(value)] = (
            value,
            _synthetic_run_fingerprint(value),
        )
    return value


def synthetic_run_capability_validation_reasons(value: object) -> tuple[str, ...]:
    """Validate a fixture-only run without accepting or probing any path."""

    if not isinstance(value, Stage8SyntheticRunCapability):
        return ("Stage 8 synthetic run capability has an invalid type",)
    issued = _ISSUED_SYNTHETIC_RUN_CAPABILITIES.get(id(value))
    if issued is None or issued[0] is not value:
        return ("Stage 8 synthetic run capability was not issued by the Stage 8 Gate",)
    if id(value) in _CONSUMED_SYNTHETIC_RUN_CAPABILITIES:
        return ("Stage 8 synthetic run capability has already been consumed",)
    try:
        reasons: list[str] = []
        if issued[1] != _synthetic_run_fingerprint(value):
            reasons.append("Stage 8 synthetic run bindings changed after issuance")
        if value._process_token is not _RUN_TOKEN or value._token is not _RUN_TOKEN:
            reasons.append("Stage 8 synthetic run issuance identity is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("Stage 8 synthetic run belongs to a different process")
        if not 1 <= value.ttl_seconds <= MAX_STAGE8_CAPABILITY_TTL_SECONDS:
            reasons.append("Stage 8 synthetic run TTL is invalid")
        if not (
            value.issued_monotonic_ns
            < value.expires_monotonic_ns
            <= value.issued_monotonic_ns + value.ttl_seconds * 1_000_000_000
        ):
            reasons.append("Stage 8 synthetic run expiry does not match TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("Stage 8 synthetic run has expired")
        descriptor = value.evidence.stage7_package_descriptor
        if (
            descriptor.scope is not Stage8PackageScope.SYNTHETIC_FIXTURE
            or descriptor.physical_root_identity_sha256 != STAGE8_NO_PHYSICAL_ROOT_SHA256
        ):
            reasons.append("Stage 8 synthetic run scope or no-root binding changed")
        if id(value._package_capability) not in _CONSUMED_PACKAGE_CAPABILITIES:
            reasons.append("Stage 7 package capability was not consumed into this run")
        return tuple(dict.fromkeys(reasons))
    except (AttributeError, TypeError, ValueError):
        return ("Stage 8 synthetic run capability fields are invalid",)


def synthetic_run_capability_snapshot(value: object) -> dict[str, object]:
    """Return a detached lifecycle receipt; it never conveys replay authority."""

    if not isinstance(value, Stage8SyntheticRunCapability):
        raise Stage8CapabilityError(("Stage 8 synthetic run capability has an invalid type",))
    issued = _ISSUED_SYNTHETIC_RUN_CAPABILITIES.get(id(value))
    if issued is None or issued[0] is not value or issued[1] != _synthetic_run_fingerprint(value):
        raise Stage8CapabilityError(("Stage 8 synthetic run issuance is invalid",))
    return {
        "scope": Stage8PackageScope.SYNTHETIC_FIXTURE.value,
        "dataset_id": value.dataset_id,
        "dataset_version": value.dataset_version,
        "ttl_seconds": value.ttl_seconds,
        "process_bound": True,
        "non_serializable": True,
        "one_shot": True,
        "issued": True,
        "consumed": id(value) in _CONSUMED_SYNTHETIC_RUN_CAPABILITIES,
        "complete_stage7_package_replayed": (id(value) in _REPLAYED_SYNTHETIC_RUN_CAPABILITIES),
        "second_consumption_rejected": (
            id(value) in _SECOND_USE_REJECTED_SYNTHETIC_RUN_CAPABILITIES
        ),
        "authorizes_embedded_package_replay": True,
        "authorizes_real_data": False,
        "authorizes_physical_dataset_root": False,
        "detached_receipt_authoritative": False,
        "authorizes_file_open": False,
    }


def consume_stage8_synthetic_run_capability(
    value: object,
    consumer: Callable[[Stage8SyntheticRunEvidence], _T],
) -> _T:
    """Replay the complete embedded fixture package and run one no-root callback."""

    if not callable(consumer):
        raise Stage8CapabilityError(("consumer must be callable",))
    with _BOUNDARY_LOCK:
        reasons = synthetic_run_capability_validation_reasons(value)
        if reasons:
            if isinstance(value, Stage8SyntheticRunCapability) and (
                id(value) in _CONSUMED_SYNTHETIC_RUN_CAPABILITIES
            ):
                _SECOND_USE_REJECTED_SYNTHETIC_RUN_CAPABILITIES.add(id(value))
            raise Stage8CapabilityError(reasons)
        assert isinstance(value, Stage8SyntheticRunCapability)
        _CONSUMED_SYNTHETIC_RUN_CAPABILITIES.add(id(value))
    try:
        replayed = value._package_capability._replay()
    except Exception as error:
        raise Stage8CapabilityError(("complete synthetic Stage 7 replay failed",)) from error
    if replayed != value.evidence.stage7_package_descriptor:
        raise Stage8CapabilityError(("synthetic Stage 7 descriptor changed at consumption",))
    _REPLAYED_SYNTHETIC_RUN_CAPABILITIES.add(id(value))
    if time.monotonic_ns() >= value.expires_monotonic_ns:
        raise Stage8CapabilityError(("Stage 8 synthetic run expired before callback",))
    try:
        result = consumer(value.evidence)
    except Exception as error:
        raise Stage8CapabilityError(("trusted Stage 8 synthetic consumer failed",)) from error
    if time.monotonic_ns() >= value.expires_monotonic_ns:
        raise Stage8CapabilityError(("Stage 8 synthetic run expired during callback",))
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8RunEvidence:
    """Detached audit metadata.  Serialization of it conveys no authority."""

    verification_context: Stage6VerificationContext
    dataset_id: str
    dataset_version: str
    stage7_package_descriptor: Stage7PackageDescriptor
    frozen_bindings: Stage8FrozenBindings
    stage7_transaction_evidence_sha256: str
    schema_version: str = STAGE8_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 run evidence schema")
        if not isinstance(self.verification_context, Stage6VerificationContext):
            raise TypeError("verification_context must be Stage6VerificationContext")
        if not _valid_dataset_id(self.dataset_id) or not _valid_dataset_version(
            self.dataset_version
        ):
            raise ValueError("Stage 8 run evidence dataset identity is invalid")
        if self.stage7_package_descriptor.dataset_id != self.dataset_id or (
            self.stage7_package_descriptor.dataset_version != self.dataset_version
        ):
            raise ValueError("Stage 7 package and run evidence identity differ")
        if self.stage7_package_descriptor.scope is not _expected_scope(self.verification_context):
            raise ValueError("Stage 7 package and run evidence scope differ")
        if self.stage7_package_descriptor.frozen_bindings != self.frozen_bindings:
            raise ValueError("Stage 7 package and frozen run bindings differ")
        _require_sha256(
            self.stage7_transaction_evidence_sha256,
            "stage7_transaction_evidence_sha256",
        )

    @property
    def authorizes_dataset_root_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "stage7_package": self.stage7_package_descriptor.to_dict(),
            "frozen_bindings": self.frozen_bindings.to_dict(),
            "stage7_transaction_evidence_sha256": (self.stage7_transaction_evidence_sha256),
            "authorizes_dataset_root_access": False,
        }

    def binding_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True, init=False)
class Stage8RunCapability:
    """Opaque, process-bound, one-shot authority for a trusted Stage 8 callback."""

    dataset_id: str
    dataset_version: str
    verification_context: Stage6VerificationContext
    evidence: Stage8RunEvidence
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _stage7_transaction: Stage7RootTransaction = field(repr=False, compare=False)
    _package_capability: VerifiedStage7PackageCapability = field(repr=False, compare=False)
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    @property
    def authorizes_file_open(self) -> bool:
        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 8 run capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 8 run capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 8 run capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 8 run capabilities cannot be serialized")


_ISSUED_RUN_CAPABILITIES: dict[int, tuple[Stage8RunCapability, str]] = {}
_CONSUMED_RUN_CAPABILITIES: set[int] = set()


def _run_capability_fingerprint(value: Stage8RunCapability) -> str:
    return _canonical_sha256(
        {
            "dataset_id": value.dataset_id,
            "dataset_version": value.dataset_version,
            "verification_context": value.verification_context.value,
            "evidence_sha256": value.evidence.binding_sha256(),
            "issued_process_id": value.issued_process_id,
            "issued_monotonic_ns": value.issued_monotonic_ns,
            "expires_monotonic_ns": value.expires_monotonic_ns,
            "ttl_seconds": value.ttl_seconds,
            "stage7_transaction_id": id(value._stage7_transaction),
            "package_capability_id": id(value._package_capability),
        }
    )


def _transaction_pre_io_reasons(
    value: object,
    *,
    expected_dataset_id: str,
    expected_dataset_version: str,
    required_context: Stage6VerificationContext,
    descriptor: Stage7PackageDescriptor,
) -> tuple[str, ...]:
    if not isinstance(value, Stage7RootTransaction):
        return ("fresh Stage 7 root transaction has an invalid type",)
    try:
        evidence = value.evidence
        reasons: list[str] = []
        if value.dataset_id != expected_dataset_id:
            reasons.append("Stage 7 root transaction dataset_id does not match")
        if value.dataset_version != expected_dataset_version:
            reasons.append("Stage 7 root transaction dataset_version does not match")
        if value.verification_context is not required_context:
            reasons.append(f"Stage 7 root transaction context must be {required_context.value}")
        if evidence.dataset_id != expected_dataset_id:
            reasons.append("Stage 7 transaction evidence dataset_id does not match")
        if evidence.dataset_version != expected_dataset_version:
            reasons.append("Stage 7 transaction evidence dataset_version does not match")
        if evidence.verification_context is not required_context:
            reasons.append("Stage 7 transaction evidence context does not match")
        if (
            evidence.stage2_package_evidence.descriptor.package_sha256
            != descriptor.stage2_package_sha256
        ):
            reasons.append("Stage 2 package hash does not match the Stage 7 package")
        if evidence.stage3_package_descriptor.package_sha256 != descriptor.stage3_package_sha256:
            reasons.append("Stage 3 package hash does not match the Stage 7 package")
        if evidence.physical_root_identity_sha256 != descriptor.physical_root_identity_sha256:
            reasons.append("physical root identity does not match the Stage 7 package")
        stage3 = evidence.stage3_package_descriptor
        frozen = descriptor.frozen_bindings
        if stage3.config_hash != frozen.config_hash:
            reasons.append("locked config hash does not match Stage 3")
        if stage3.data_hash != frozen.data_hash:
            reasons.append("locked data hash does not match Stage 3")
        if stage3.model_hash != frozen.model_hash:
            reasons.append("locked model hash does not match Stage 3")
        return tuple(dict.fromkeys(reasons))
    except (AttributeError, TypeError, ValueError):
        return ("Stage 7 root transaction fields are invalid",)


def begin_stage8_run_capability(
    stage7_package_capability: object,
    stage7_root_transaction: object,
    *,
    expected_dataset_id: str,
    expected_dataset_version: str,
    frozen_bindings: Stage8FrozenBindings,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
    ttl_seconds: int = DEFAULT_STAGE8_CAPABILITY_TTL_SECONDS,
) -> Stage8RunCapability:
    """Replay package authority, bind a fresh transaction, and issue for at most 5 s."""

    if (
        not isinstance(required_context, Stage6VerificationContext)
        or not _valid_dataset_id(expected_dataset_id)
        or not _valid_dataset_version(expected_dataset_version)
        or not isinstance(frozen_bindings, Stage8FrozenBindings)
        or type(ttl_seconds) is not int  # noqa: E721
        or not 1 <= ttl_seconds <= MAX_STAGE8_CAPABILITY_TTL_SECONDS
    ):
        raise Stage8CapabilityError(("Stage 8 run capability input is invalid",))
    package_reasons = package_capability_validation_reasons(
        stage7_package_capability,
        required_context=required_context,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
    )
    if package_reasons:
        raise Stage8CapabilityError(package_reasons)
    assert isinstance(stage7_package_capability, VerifiedStage7PackageCapability)
    descriptor = stage7_package_capability.descriptor
    if descriptor.frozen_bindings != frozen_bindings:
        raise Stage8CapabilityError(("pre-registered frozen bindings do not match",))
    transaction_reasons = _transaction_pre_io_reasons(
        stage7_root_transaction,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
        required_context=required_context,
        descriptor=descriptor,
    )
    if transaction_reasons:
        raise Stage8CapabilityError(transaction_reasons)

    # Complete package replay precedes the first Stage 8-triggered dataset-root probe.
    try:
        replayed = stage7_package_capability._replay()
    except Exception as error:
        raise Stage8CapabilityError(("complete Stage 7 package replay failed",)) from error
    if not isinstance(replayed, Stage7PackageDescriptor) or replayed != descriptor:
        raise Stage8CapabilityError(("replayed Stage 7 package descriptor changed",))

    stage7_reasons = transaction_validation_reasons(
        stage7_root_transaction,
        required_context=required_context,
    )
    if stage7_reasons:
        raise Stage8CapabilityError(stage7_reasons)
    assert isinstance(stage7_root_transaction, Stage7RootTransaction)
    now = time.monotonic_ns()
    deadline = min(
        now + ttl_seconds * 1_000_000_000,
        stage7_package_capability.expires_monotonic_ns,
        stage7_root_transaction.expires_monotonic_ns,
    )
    if deadline <= now:
        raise Stage8CapabilityError(("a prerequisite capability expired during binding",))
    evidence = Stage8RunEvidence(
        verification_context=required_context,
        dataset_id=expected_dataset_id,
        dataset_version=expected_dataset_version,
        stage7_package_descriptor=descriptor,
        frozen_bindings=frozen_bindings,
        stage7_transaction_evidence_sha256=stage7_root_transaction.evidence.binding_sha256(),
    )
    value = object.__new__(Stage8RunCapability)
    assignments: tuple[tuple[str, object], ...] = (
        ("dataset_id", expected_dataset_id),
        ("dataset_version", expected_dataset_version),
        ("verification_context", required_context),
        ("evidence", evidence),
        ("issued_process_id", _PROCESS_ID),
        ("issued_monotonic_ns", now),
        ("expires_monotonic_ns", deadline),
        ("ttl_seconds", ttl_seconds),
        ("_stage7_transaction", stage7_root_transaction),
        ("_package_capability", stage7_package_capability),
        ("_process_token", _RUN_TOKEN),
        ("_token", _RUN_TOKEN),
    )
    for name, item in assignments:
        object.__setattr__(value, name, item)
    with _BOUNDARY_LOCK:
        if id(stage7_package_capability) in _CONSUMED_PACKAGE_CAPABILITIES:
            raise Stage8CapabilityError(("Stage 7 package capability has already been consumed",))
        _CONSUMED_PACKAGE_CAPABILITIES.add(id(stage7_package_capability))
        _ISSUED_RUN_CAPABILITIES[id(value)] = (value, _run_capability_fingerprint(value))
    return value


def _run_validation_reasons_without_io(
    value: object,
    *,
    required_context: Stage6VerificationContext,
) -> tuple[str, ...]:
    if not isinstance(value, Stage8RunCapability):
        return ("Stage 8 run capability has an invalid type",)
    issued = _ISSUED_RUN_CAPABILITIES.get(id(value))
    if issued is None or issued[0] is not value:
        return ("Stage 8 run capability was not issued by the Stage 8 Gate",)
    if id(value) in _CONSUMED_RUN_CAPABILITIES:
        return ("Stage 8 run capability has already been consumed",)
    try:
        reasons: list[str] = []
        if issued[1] != _run_capability_fingerprint(value):
            reasons.append("Stage 8 run capability bindings changed after issuance")
        if value._process_token is not _RUN_TOKEN or value._token is not _RUN_TOKEN:
            reasons.append("Stage 8 run capability issuance identity is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("Stage 8 run capability belongs to a different process")
        if value.verification_context is not required_context:
            reasons.append(f"Stage 8 run capability context must be {required_context.value}")
        if not 1 <= value.ttl_seconds <= MAX_STAGE8_CAPABILITY_TTL_SECONDS:
            reasons.append("Stage 8 run capability TTL is invalid")
        if not (
            value.issued_monotonic_ns
            < value.expires_monotonic_ns
            <= value.issued_monotonic_ns + value.ttl_seconds * 1_000_000_000
        ):
            reasons.append("Stage 8 run capability expiry does not match TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("Stage 8 run capability has expired")
        if value.evidence.dataset_id != value.dataset_id:
            reasons.append("Stage 8 evidence dataset_id changed")
        if value.evidence.dataset_version != value.dataset_version:
            reasons.append("Stage 8 evidence dataset_version changed")
        if value.evidence.verification_context is not value.verification_context:
            reasons.append("Stage 8 evidence context changed")
        if id(value._package_capability) not in _CONSUMED_PACKAGE_CAPABILITIES:
            reasons.append("Stage 7 package capability was not consumed into this run")
        return tuple(dict.fromkeys(reasons))
    except (AttributeError, TypeError, ValueError):
        return ("Stage 8 run capability fields are invalid",)


def run_capability_validation_reasons(
    value: object,
    *,
    required_context: object = Stage6VerificationContext.REAL_PREREQUISITE,
) -> tuple[str, ...]:
    """Validate local issuance before the final Stage 7 physical-root check."""

    if not isinstance(required_context, Stage6VerificationContext):
        return ("required context is invalid",)
    reasons = _run_validation_reasons_without_io(
        value,
        required_context=required_context,
    )
    if reasons:
        return reasons
    assert isinstance(value, Stage8RunCapability)
    return transaction_validation_reasons(
        value._stage7_transaction,
        required_context=required_context,
    )


def _text_is_root_or_descendant(value: str, root: Path) -> bool:
    """Compare path-shaped text lexically without filesystem access.

    Both Windows and POSIX forms are considered so a callback cannot evade the
    boundary by changing separator style.  Windows paths are case-insensitive;
    POSIX paths retain case sensitivity.  Relative/ordinary strings are never
    interpreted relative to the authorized root.
    """

    if not value or "\x00" in value:
        return False
    root_text = os.fspath(root)
    windows_root = ntpath.normpath(root_text)
    windows_value = ntpath.normpath(value)
    if ntpath.isabs(windows_root) and ntpath.isabs(windows_value):
        normalized_root = ntpath.normcase(windows_root).casefold()
        normalized_value = ntpath.normcase(windows_value).casefold()
        try:
            if ntpath.commonpath((normalized_root, normalized_value)) == normalized_root:
                return True
        except ValueError:
            pass

    # Backslashes are accepted as alternate separators at this serialization
    # boundary.  A real POSIX backslash-containing filename is therefore treated
    # conservatively as path-shaped evidence rather than returned to the caller.
    posix_root = posixpath.normpath(root_text.replace("\\", "/"))
    posix_value = posixpath.normpath(value.replace("\\", "/"))
    if posixpath.isabs(posix_root) and posixpath.isabs(posix_value):
        try:
            if posixpath.commonpath((posix_root, posix_value)) == posix_root:
                return True
        except ValueError:
            pass
    return False


def _result_discloses_root(value: object, root: Path, *, depth: int = 0) -> bool:
    """Allow only recursively inspected JSON-like values and known PathLike values.

    Unknown/custom objects are rejected conservatively.  Reflecting over
    ``__dict__``, slots, descriptors, or properties would create another
    attacker-controlled traversal surface and could execute code during a safety
    check, so the callback contract intentionally requires plain report values.
    """

    if depth > 8:
        return True
    if isinstance(value, os.PathLike):
        try:
            return _text_is_root_or_descendant(os.fspath(value), root)
        except TypeError:
            return True
    if isinstance(value, str):
        return _text_is_root_or_descendant(value, root)
    if isinstance(value, Mapping):
        return any(
            _result_discloses_root(key, root, depth=depth + 1)
            or _result_discloses_root(item, root, depth=depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_result_discloses_root(item, root, depth=depth + 1) for item in value)
    if value is None or isinstance(value, bool | int | float):
        return False
    return True


def consume_stage8_run_capability(
    value: object,
    consumer: Callable[[Path, Stage7RunEvidence, Stage8RunEvidence], _T],
    *,
    required_context: object = Stage6VerificationContext.REAL_PREREQUISITE,
) -> _T:
    """Run one trusted read-only callback; consume authority even when it fails."""

    if not callable(consumer):
        raise Stage8CapabilityError(("consumer must be callable",))
    if not isinstance(required_context, Stage6VerificationContext):
        raise Stage8CapabilityError(("required context is invalid",))
    with _BOUNDARY_LOCK:
        # Local issuance/TTL checks do not touch the physical root.  Marking the
        # run consumed before package replay ensures a replay failure cannot be
        # retried with the same authority.
        reasons = _run_validation_reasons_without_io(
            value,
            required_context=required_context,
        )
        if reasons:
            raise Stage8CapabilityError(reasons)
        assert isinstance(value, Stage8RunCapability)
        _CONSUMED_RUN_CAPABILITIES.add(id(value))

    # Re-establish complete Stage 7 package evidence before the first root lstat
    # performed inside consume_stage7_root_transaction.
    try:
        replayed = value._package_capability._replay()
    except Exception as error:
        raise Stage8CapabilityError(("complete Stage 7 package replay failed",)) from error
    if (
        not isinstance(replayed, Stage7PackageDescriptor)
        or replayed != value.evidence.stage7_package_descriptor
    ):
        raise Stage8CapabilityError(("replayed Stage 7 package descriptor changed",))
    if time.monotonic_ns() >= value.expires_monotonic_ns:
        raise Stage8CapabilityError(("Stage 8 run capability expired before root access",))

    observed_root: Path | None = None
    raw_result: object = _NO_RESULT
    callback_error: Exception | None = None

    def trusted_stage7_consumer(root: Path, stage7_evidence: Stage7RunEvidence) -> object:
        nonlocal observed_root
        observed_root = root
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            raise Stage8CapabilityError(("Stage 8 run capability expired before callback",))
        result = consumer(root, stage7_evidence, value.evidence)
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            raise Stage8CapabilityError(("Stage 8 run capability expired during callback",))
        return result

    try:
        raw_result = consume_stage7_root_transaction(
            value._stage7_transaction,
            trusted_stage7_consumer,
            required_context=required_context,
        )
    except (Stage7RootTransactionError, Stage8CapabilityError) as error:
        callback_error = error
    if callback_error is not None:
        raise Stage8CapabilityError(("trusted Stage 8 root consumer failed",)) from callback_error
    if time.monotonic_ns() >= value.expires_monotonic_ns:
        raise Stage8CapabilityError(("Stage 8 run capability expired after root callback",))
    assert raw_result is not _NO_RESULT
    result = cast(_T, raw_result)
    if observed_root is None or _result_discloses_root(result, observed_root):
        raise Stage8CapabilityError(("consumer returned or embedded the authorized root",))
    return result


__all__ = [
    "DEFAULT_STAGE8_CAPABILITY_TTL_SECONDS",
    "MAX_STAGE8_CAPABILITY_TTL_SECONDS",
    "STAGE8_GATE_SCHEMA_VERSION",
    "STAGE8_NO_PHYSICAL_ROOT_SHA256",
    "Stage7PackageDescriptor",
    "Stage8CapabilityError",
    "Stage8FrozenBindings",
    "Stage8PackageScope",
    "Stage8PreflightReport",
    "Stage8PreflightStatus",
    "Stage8Reason",
    "Stage8RunCapability",
    "Stage8RunEvidence",
    "Stage8SyntheticRunCapability",
    "Stage8SyntheticRunEvidence",
    "VerifiedStage7PackageCapability",
    "assess_stage8_preflight",
    "begin_stage8_run_capability",
    "begin_stage8_synthetic_run_capability",
    "consume_stage8_run_capability",
    "consume_stage8_synthetic_run_capability",
    "issue_synthetic_stage7_package_capability",
    "package_capability_validation_reasons",
    "run_capability_validation_reasons",
    "synthetic_run_capability_snapshot",
    "synthetic_run_capability_validation_reasons",
]
