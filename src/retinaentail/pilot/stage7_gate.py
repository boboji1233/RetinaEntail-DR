"""Stage 7 one-shot conversion of a Stage 6 bound root into a trusted callback.

The presence-only preflight accepts booleans and identifiers, never a path.  The
transaction boundary accepts only a live, process-issued :class:`Stage6BoundRoot`.
It replays the complete Stage 2 and Stage 3 packages before any dataset-root
inspection, consumes the Stage 6 binding within its five-second lifetime, and
retains the physical root only in an opaque, process-local transaction.

The root is exposed only for the dynamic extent of a trusted callback.  This is a
cooperative in-process authority boundary, not an OS sandbox: malicious Python code
could deliberately retain a callback argument.  Callers must therefore register
only repository-owned read-only rebuild/audit consumers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Never, TypeVar, cast

from retinaentail.calibration.stage3_package import (
    Stage3PackageDescriptor,
    Stage3PackageScope,
    replay_stage3_package,
)
from retinaentail.data.stage2_package import (
    Stage2PackageEvidence,
    Stage2PackageScope,
    replay_stage2_package_evidence,
)
from retinaentail.pilot.stage6_gate import (
    Stage6BoundRoot,
    Stage6RootBindingError,
    Stage6VerificationContext,
    bound_root_validation_reasons,
    consume_bound_stage6_root,
)

STAGE7_GATE_SCHEMA_VERSION: Final = "7.0.0"
DEFAULT_STAGE7_TRANSACTION_TTL_SECONDS: Final = 3_600
MAX_STAGE7_TRANSACTION_TTL_SECONDS: Final = 86_400

_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT_ATTRIBUTE: Final = 0x0400
_PROCESS_ID: Final = os.getpid()
_PROCESS_TOKEN: Final = object()
_TRANSACTION_TOKEN: Final = object()
_NO_RESULT: Final = object()
_BOUNDARY_LOCK: Final = threading.RLock()

_REAL_BINDING_NAMES: Final = frozenset(
    {
        "stage2_access_permit",
        "stage2_package",
        "stage3_package",
        "frozen_data_card",
        "manual_confirmation",
        "locked_evidence",
    }
)
_SYNTHETIC_BINDING_NAMES: Final = frozenset(
    {
        "independent_stage2_package",
        "independent_stage3_package",
        "synthetic_stage5_pilot",
    }
)

_T = TypeVar("_T")


class Stage7PreflightStatus(StrEnum):
    """Presence-only outcome; READY is not dataset-root authority."""

    BLOCKED = "BLOCKED"
    READY_FOR_BOUND_ROOT_CONSUMPTION = "READY_FOR_BOUND_ROOT_CONSUMPTION"


class Stage7Reason(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    THREE_PURPOSE_REVIEW_NOT_PROVIDED = "THREE_PURPOSE_REVIEW_NOT_PROVIDED"
    PHYSICAL_ROOT_NOT_PROVIDED = "PHYSICAL_ROOT_NOT_PROVIDED"
    ADAPTER_SPEC_NOT_PROVIDED = "ADAPTER_SPEC_NOT_PROVIDED"
    STAGE2_PASS_PACKAGE_NOT_PROVIDED = "STAGE2_PASS_PACKAGE_NOT_PROVIDED"
    APPROVED_REAL_DATA_CARD_NOT_PROVIDED = "APPROVED_REAL_DATA_CARD_NOT_PROVIDED"
    MANUAL_CONFIRMATION_NOT_PROVIDED = "MANUAL_CONFIRMATION_NOT_PROVIDED"
    STAGE3_PACKAGE_NOT_PROVIDED = "STAGE3_PACKAGE_NOT_PROVIDED"
    STAGE4_LOCKED_EVIDENCE_NOT_PROVIDED = "STAGE4_LOCKED_EVIDENCE_NOT_PROVIDED"
    STAGE6_BOUND_ROOT_NOT_PROVIDED = "STAGE6_BOUND_ROOT_NOT_PROVIDED"


class Stage7RootTransactionError(PermissionError):
    """Fail-closed error at the Stage 7 authority boundary."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 7 root transaction rejected: " + "; ".join(reasons))


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7PreflightReport:
    status: Stage7PreflightStatus
    verification_context: Stage6VerificationContext
    dataset_id: str | None
    dataset_version: str | None
    reasons: tuple[Stage7Reason, ...]
    schema_version: str = STAGE7_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE7_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 7 Gate schema")
        if not isinstance(self.status, Stage7PreflightStatus):
            raise TypeError("status must be Stage7PreflightStatus")
        if not isinstance(self.verification_context, Stage6VerificationContext):
            raise TypeError("verification_context must be Stage6VerificationContext")
        if not all(isinstance(reason, Stage7Reason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage7Reason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must be unique")
        ready = self.status is Stage7PreflightStatus.READY_FOR_BOUND_ROOT_CONSUMPTION
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
            "authorizes_file_open": False,
            "stage6_bound_root_consumed": False,
            "dataset_root_probed": False,
            "real_data_read": False,
            "synthetic_authorizes_real": False,
        }


def assess_stage7_preflight(
    *,
    verification_context: Stage6VerificationContext,
    dataset_id: str | None,
    dataset_version: str | None,
    three_purpose_review_provided: bool,
    physical_root_provided: bool,
    adapter_spec_provided: bool,
    stage2_pass_package_provided: bool,
    approved_real_data_card_provided: bool,
    manual_confirmation_provided: bool,
    stage3_calibration_package_provided: bool,
    stage4_locked_evidence_provided: bool,
    stage6_bound_root_provided: bool,
) -> Stage7PreflightReport:
    """Check identifiers and evidence-presence flags without accepting a path."""

    flags = (
        three_purpose_review_provided,
        physical_root_provided,
        adapter_spec_provided,
        stage2_pass_package_provided,
        approved_real_data_card_provided,
        manual_confirmation_provided,
        stage3_calibration_package_provided,
        stage4_locked_evidence_provided,
        stage6_bound_root_provided,
    )
    if not isinstance(verification_context, Stage6VerificationContext) or any(
        type(value) is not bool
        for value in flags  # noqa: E721
    ):
        return Stage7PreflightReport(
            status=Stage7PreflightStatus.BLOCKED,
            verification_context=(
                verification_context
                if isinstance(verification_context, Stage6VerificationContext)
                else Stage6VerificationContext.REAL_PREREQUISITE
            ),
            dataset_id=None,
            dataset_version=None,
            reasons=(Stage7Reason.INPUT_INVALID,),
        )
    reasons: list[Stage7Reason] = []
    if not _valid_dataset_id(dataset_id):
        reasons.append(Stage7Reason.DATASET_NOT_EXPLICIT)
    if not _valid_dataset_version(dataset_version):
        reasons.append(Stage7Reason.DATASET_VERSION_NOT_EXPLICIT)
    if verification_context is Stage6VerificationContext.REAL_PREREQUISITE:
        requirements = (
            (three_purpose_review_provided, Stage7Reason.THREE_PURPOSE_REVIEW_NOT_PROVIDED),
            (physical_root_provided, Stage7Reason.PHYSICAL_ROOT_NOT_PROVIDED),
            (adapter_spec_provided, Stage7Reason.ADAPTER_SPEC_NOT_PROVIDED),
            (stage2_pass_package_provided, Stage7Reason.STAGE2_PASS_PACKAGE_NOT_PROVIDED),
            (
                approved_real_data_card_provided,
                Stage7Reason.APPROVED_REAL_DATA_CARD_NOT_PROVIDED,
            ),
            (manual_confirmation_provided, Stage7Reason.MANUAL_CONFIRMATION_NOT_PROVIDED),
            (
                stage3_calibration_package_provided,
                Stage7Reason.STAGE3_PACKAGE_NOT_PROVIDED,
            ),
            (
                stage4_locked_evidence_provided,
                Stage7Reason.STAGE4_LOCKED_EVIDENCE_NOT_PROVIDED,
            ),
        )
        reasons.extend(reason for present, reason in requirements if not present)
    if not stage6_bound_root_provided:
        reasons.append(Stage7Reason.STAGE6_BOUND_ROOT_NOT_PROVIDED)
    normalized = tuple(dict.fromkeys(reasons))
    return Stage7PreflightReport(
        status=(
            Stage7PreflightStatus.BLOCKED
            if normalized
            else Stage7PreflightStatus.READY_FOR_BOUND_ROOT_CONSUMPTION
        ),
        verification_context=verification_context,
        dataset_id=dataset_id if _valid_dataset_id(dataset_id) else None,
        dataset_version=(dataset_version if _valid_dataset_version(dataset_version) else None),
        reasons=normalized,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7RunEvidence:
    """Immutable detached evidence passed to a trusted read-only callback.

    It intentionally contains the complete frozen Stage 2 evidence needed for
    record-by-record comparison.  Serialization of this value conveys no authority.
    """

    verification_context: Stage6VerificationContext
    dataset_id: str
    dataset_version: str
    stage2_package_evidence: Stage2PackageEvidence = field(repr=False, compare=False)
    stage3_package_descriptor: Stage3PackageDescriptor = field(repr=False, compare=False)
    stage6_bindings: tuple[tuple[str, str], ...]
    stage6_authorization_snapshot_sha256: str
    stage6_root_binding_sha256: str
    physical_root_identity_sha256: str
    schema_version: str = STAGE7_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE7_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 7 evidence schema")
        if not isinstance(self.verification_context, Stage6VerificationContext):
            raise TypeError("verification_context must be Stage6VerificationContext")
        if not _valid_dataset_id(self.dataset_id) or not _valid_dataset_version(
            self.dataset_version
        ):
            raise ValueError("Stage 7 evidence dataset identity is invalid")
        if not isinstance(self.stage2_package_evidence, Stage2PackageEvidence):
            raise TypeError("Stage 2 package evidence has an invalid type")
        if not isinstance(self.stage3_package_descriptor, Stage3PackageDescriptor):
            raise TypeError("Stage 3 package descriptor has an invalid type")
        if self.stage6_bindings != tuple(sorted(set(self.stage6_bindings))):
            raise ValueError("Stage 6 bindings must be unique and sorted")
        if not self.stage6_bindings or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or _SHA256_PATTERN.fullmatch(value) is None
            for name, value in self.stage6_bindings
        ):
            raise ValueError("Stage 6 bindings are invalid")
        for name in (
            "stage6_authorization_snapshot_sha256",
            "stage6_root_binding_sha256",
            "physical_root_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        binding_names_in_order = tuple(name for name, _ in self.stage6_bindings)
        binding_names = frozenset(binding_names_in_order)
        if len(binding_names) != len(binding_names_in_order):
            raise ValueError("Stage 6 binding names must be unique")
        binding_map = dict(self.stage6_bindings)
        if self.verification_context is Stage6VerificationContext.REAL_PREREQUISITE:
            if binding_names != _REAL_BINDING_NAMES:
                raise ValueError("real evidence does not contain the exact Stage 6 bindings")
            if (
                self.stage2_package_evidence.descriptor.scope
                is not Stage2PackageScope.APPROVED_REAL_DATA
                or self.stage3_package_descriptor.scope is not Stage3PackageScope.APPROVED_REAL_DATA
                or self.stage2_package_evidence.descriptor.dataset_id != self.dataset_id
                or self.stage2_package_evidence.descriptor.dataset_version != self.dataset_version
                or self.stage3_package_descriptor.dataset_id != self.dataset_id
                or self.stage3_package_descriptor.dataset_version != self.dataset_version
                or self.stage3_package_descriptor.fixture_only
                or self.stage3_package_descriptor.stage2_package_sha256
                != self.stage2_package_evidence.descriptor.package_sha256
                or self.stage3_package_descriptor.frozen_data_card_sha256
                != binding_map["frozen_data_card"]
            ):
                raise ValueError("real Stage 2/3 evidence scope or identity is invalid")
        elif (
            binding_names != _SYNTHETIC_BINDING_NAMES
            or self.stage2_package_evidence.descriptor.scope
            is not Stage2PackageScope.SYNTHETIC_FIXTURE
            or self.stage3_package_descriptor.scope is not Stage3PackageScope.SYNTHETIC_TOY
            or not self.stage3_package_descriptor.fixture_only
        ):
            raise ValueError("synthetic evidence does not contain exact independent bindings")

    @property
    def authorizes_dataset_root_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        binding_map = dict(self.stage6_bindings)
        return {
            "schema_version": self.schema_version,
            "verification_context": self.verification_context.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "stage2_dataset_id": self.stage2_package_evidence.descriptor.dataset_id,
            "stage2_dataset_version": self.stage2_package_evidence.descriptor.dataset_version,
            "stage2_package_sha256": self.stage2_package_evidence.descriptor.package_sha256,
            "stage2_manifest_sha256": self.stage2_package_evidence.descriptor.manifest_sha256,
            "stage2_audit_sha256": self.stage2_package_evidence.descriptor.audit_sha256,
            "stage2_evidence_sha256": _canonical_sha256(self.stage2_package_evidence.to_dict()),
            "stage3_dataset_id": self.stage3_package_descriptor.dataset_id,
            "stage3_dataset_version": self.stage3_package_descriptor.dataset_version,
            "stage3_package_sha256": self.stage3_package_descriptor.package_sha256,
            "stage3_calibration_artifact_sha256": (
                self.stage3_package_descriptor.calibration_artifact_hash
            ),
            "stage3_config_sha256": self.stage3_package_descriptor.config_hash,
            "stage3_data_sha256": self.stage3_package_descriptor.data_hash,
            "stage3_model_sha256": self.stage3_package_descriptor.model_hash,
            "stage6_bindings": binding_map,
            "stage6_authorization_snapshot_sha256": (self.stage6_authorization_snapshot_sha256),
            "stage6_root_binding_sha256": self.stage6_root_binding_sha256,
            "physical_root_identity_sha256": self.physical_root_identity_sha256,
            "frozen_data_card_binding_sha256": binding_map.get("frozen_data_card"),
            "manual_confirmation_binding_sha256": binding_map.get("manual_confirmation"),
            "stage4_locked_evidence_binding_sha256": binding_map.get("locked_evidence"),
            "authorizes_dataset_root_access": False,
        }

    def binding_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True, init=False)
class Stage7RootTransaction:
    """Opaque, process-bound, one-shot authority for one trusted callback."""

    dataset_id: str
    dataset_version: str
    verification_context: Stage6VerificationContext
    evidence: Stage7RunEvidence
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _root: Path = field(repr=False, compare=False)
    _physical_identity: tuple[int, int, int, int, int, int] = field(
        repr=False,
        compare=False,
    )
    _stage6_bound_root: Stage6BoundRoot = field(repr=False, compare=False)
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    @property
    def authorizes_file_open(self) -> bool:
        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 7 root transactions cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 7 root transactions cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 7 root transactions cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 7 root transactions cannot be serialized")


_ISSUED_TRANSACTIONS: dict[int, tuple[Stage7RootTransaction, str]] = {}
_RESERVED_BOUND_ROOT_IDS: set[int] = set()
_CONVERTED_BOUND_ROOT_IDS: set[int] = set()
_CONSUMED_TRANSACTION_IDS: set[int] = set()


def _issue_transaction(
    *,
    dataset_id: str,
    dataset_version: str,
    verification_context: Stage6VerificationContext,
    evidence: Stage7RunEvidence,
    root: Path,
    physical_identity: tuple[int, int, int, int, int, int],
    stage6_bound_root: Stage6BoundRoot,
    ttl_seconds: int,
) -> Stage7RootTransaction:
    now = time.monotonic_ns()
    value = object.__new__(Stage7RootTransaction)
    assignments: tuple[tuple[str, object], ...] = (
        ("dataset_id", dataset_id),
        ("dataset_version", dataset_version),
        ("verification_context", verification_context),
        ("evidence", evidence),
        ("issued_process_id", _PROCESS_ID),
        ("issued_monotonic_ns", now),
        ("expires_monotonic_ns", now + ttl_seconds * 1_000_000_000),
        ("ttl_seconds", ttl_seconds),
        ("_root", root),
        ("_physical_identity", physical_identity),
        ("_stage6_bound_root", stage6_bound_root),
        ("_process_token", _PROCESS_TOKEN),
        ("_token", _TRANSACTION_TOKEN),
    )
    for name, item in assignments:
        object.__setattr__(value, name, item)
    return value


def _transaction_fingerprint(value: Stage7RootTransaction) -> str:
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
            "root": value._root.as_posix(),
            "physical_identity": value._physical_identity,
            "stage6_bound_root_id": id(value._stage6_bound_root),
        }
    )


def _is_link_or_reparse(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _strict_physical_root_identity(root: Path) -> tuple[int, int, int, int, int, int]:
    final_metadata: os.stat_result | None = None
    for candidate in (*reversed(root.parents), root):
        try:
            metadata = os.lstat(candidate)
        except OSError as error:
            raise Stage7RootTransactionError(("root or an ancestor does not exist",)) from error
        if _is_link_or_reparse(metadata):
            raise Stage7RootTransactionError(("root path contains a link or reparse point",))
        if candidate == root:
            final_metadata = metadata
    if final_metadata is None or not stat.S_ISDIR(final_metadata.st_mode):
        raise Stage7RootTransactionError(("authorized root is not a directory",))
    return (
        final_metadata.st_dev,
        final_metadata.st_ino,
        final_metadata.st_mode,
        final_metadata.st_size,
        final_metadata.st_mtime_ns,
        final_metadata.st_ctime_ns,
    )


def _pre_io_bound_root_reasons(
    value: object,
    *,
    expected_dataset_id: str,
    expected_dataset_version: str,
    required_context: Stage6VerificationContext,
) -> tuple[str, ...]:
    if not isinstance(value, Stage6BoundRoot):
        return ("Stage 6 bound root has an invalid type",)
    try:
        authorization = value.authorization
        reasons: list[str] = []
        if authorization.verification_context is not required_context:
            reasons.append(f"Stage 6 context must be {required_context.value}")
        if authorization.dataset_id != expected_dataset_id:
            reasons.append("Stage 6 dataset_id does not match")
        if authorization.dataset_version != expected_dataset_version:
            reasons.append("Stage 6 dataset_version does not match")
        binding_names = frozenset(name for name, _ in authorization.binding_sha256)
        expected_names = (
            _REAL_BINDING_NAMES
            if required_context is Stage6VerificationContext.REAL_PREREQUISITE
            else _SYNTHETIC_BINDING_NAMES
        )
        if binding_names != expected_names:
            reasons.append("Stage 6 prerequisite binding set does not match the context")
        if required_context is Stage6VerificationContext.REAL_PREREQUISITE and (
            authorization._stage2_access_permit is None
            or authorization._frozen_data_card is None
            or authorization._manual_confirmation is None
            or authorization._locked_evidence is None
            or authorization._stage5_pilot_report is not None
        ):
            reasons.append("Stage 6 REAL authorization prerequisites are incomplete")
        return tuple(dict.fromkeys(reasons))
    except (AttributeError, TypeError, ValueError):
        return ("Stage 6 bound-root fields are invalid",)


def begin_stage7_root_transaction(
    value: object,
    *,
    expected_dataset_id: str,
    expected_dataset_version: str,
    required_context: Stage6VerificationContext = Stage6VerificationContext.REAL_PREREQUISITE,
    transaction_ttl_seconds: int = DEFAULT_STAGE7_TRANSACTION_TTL_SECONDS,
) -> Stage7RootTransaction:
    """Consume one live Stage 6 binding and issue one opaque callback transaction.

    All identifier, context, binding-set, TTL, and complete package checks occur
    before the first Stage 7-triggered dataset-root ``lstat``.  The Stage 6 bound
    root is consumed here, within its original maximum-five-second lifetime.  The
    independent Stage 7 TTL bounds later callback start/finish, not Stage 6 access.
    """

    if (
        not isinstance(required_context, Stage6VerificationContext)
        or not _valid_dataset_id(expected_dataset_id)
        or not _valid_dataset_version(expected_dataset_version)
        or type(transaction_ttl_seconds) is not int  # noqa: E721
        or not 1 <= transaction_ttl_seconds <= MAX_STAGE7_TRANSACTION_TTL_SECONDS
    ):
        raise Stage7RootTransactionError(("Stage 7 transaction input is invalid",))
    reasons = _pre_io_bound_root_reasons(
        value,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
        required_context=required_context,
    )
    if reasons:
        raise Stage7RootTransactionError(reasons)
    assert isinstance(value, Stage6BoundRoot)
    bound_id = id(value)
    with _BOUNDARY_LOCK:
        if bound_id in _RESERVED_BOUND_ROOT_IDS or bound_id in _CONVERTED_BOUND_ROOT_IDS:
            raise Stage7RootTransactionError(
                ("Stage 6 bound root already has a Stage 7 transaction",)
            )
        _RESERVED_BOUND_ROOT_IDS.add(bound_id)
    try:
        authorization = value.authorization
        try:
            stage2_evidence = replay_stage2_package_evidence(
                authorization._stage2_package_capability
            )
            stage3_descriptor = replay_stage3_package(authorization._stage3_package_capability)
        except (OSError, TypeError, ValueError) as error:
            raise Stage7RootTransactionError(
                ("complete Stage 2/3 replay failed before root consumption",)
            ) from error
        if required_context is Stage6VerificationContext.REAL_PREREQUISITE and (
            stage2_evidence.descriptor.scope is not Stage2PackageScope.APPROVED_REAL_DATA
            or stage3_descriptor.scope is not Stage3PackageScope.APPROVED_REAL_DATA
            or stage2_evidence.descriptor.dataset_id != expected_dataset_id
            or stage2_evidence.descriptor.dataset_version != expected_dataset_version
            or stage3_descriptor.dataset_id != expected_dataset_id
            or stage3_descriptor.dataset_version != expected_dataset_version
            or stage3_descriptor.fixture_only
        ):
            raise Stage7RootTransactionError(
                ("replayed package scope or dataset identity does not match REAL context",)
            )
        stage6_reasons = bound_root_validation_reasons(
            value,
            required_context=required_context,
        )
        if stage6_reasons:
            raise Stage7RootTransactionError(stage6_reasons)
        try:
            root = consume_bound_stage6_root(value, required_context=required_context)
        except Stage6RootBindingError as error:
            raise Stage7RootTransactionError(error.reasons) from error
        physical_identity = value.physical_identity
        if _strict_physical_root_identity(root) != physical_identity:
            raise Stage7RootTransactionError(
                ("physical root identity changed during Stage 6 consumption",)
            )
        evidence = Stage7RunEvidence(
            verification_context=required_context,
            dataset_id=expected_dataset_id,
            dataset_version=expected_dataset_version,
            stage2_package_evidence=stage2_evidence,
            stage3_package_descriptor=stage3_descriptor,
            stage6_bindings=authorization.binding_sha256,
            stage6_authorization_snapshot_sha256=value.authorization_snapshot_sha256,
            stage6_root_binding_sha256=authorization.root_binding_sha256,
            physical_root_identity_sha256=_canonical_sha256(physical_identity),
        )
        transaction = _issue_transaction(
            dataset_id=expected_dataset_id,
            dataset_version=expected_dataset_version,
            verification_context=required_context,
            evidence=evidence,
            root=root,
            physical_identity=physical_identity,
            stage6_bound_root=value,
            ttl_seconds=transaction_ttl_seconds,
        )
        with _BOUNDARY_LOCK:
            _ISSUED_TRANSACTIONS[id(transaction)] = (
                transaction,
                _transaction_fingerprint(transaction),
            )
            _CONVERTED_BOUND_ROOT_IDS.add(bound_id)
        return transaction
    finally:
        with _BOUNDARY_LOCK:
            _RESERVED_BOUND_ROOT_IDS.discard(bound_id)


def _transaction_validation_reasons_without_io(
    value: object,
    *,
    required_context: Stage6VerificationContext,
) -> tuple[str, ...]:
    if not isinstance(value, Stage7RootTransaction):
        return ("transaction has an invalid type",)
    issued = _ISSUED_TRANSACTIONS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("transaction was not issued by the Stage 7 Gate",)
    if id(value) in _CONSUMED_TRANSACTION_IDS:
        return ("transaction has already been consumed",)
    try:
        reasons: list[str] = []
        if issued[1] != _transaction_fingerprint(value):
            reasons.append("transaction bindings changed after issuance")
        if value._process_token is not _PROCESS_TOKEN or value._token is not _TRANSACTION_TOKEN:
            reasons.append("transaction issuance identity is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("transaction belongs to a different process")
        if value.verification_context is not required_context:
            reasons.append(f"transaction context must be {required_context.value}")
        if not 1 <= value.ttl_seconds <= MAX_STAGE7_TRANSACTION_TTL_SECONDS:
            reasons.append("transaction TTL is invalid")
        if value.expires_monotonic_ns - value.issued_monotonic_ns != (
            value.ttl_seconds * 1_000_000_000
        ):
            reasons.append("transaction expiry does not match TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("transaction has expired")
        if value.evidence.dataset_id != value.dataset_id:
            reasons.append("transaction evidence dataset_id changed")
        if value.evidence.dataset_version != value.dataset_version:
            reasons.append("transaction evidence dataset_version changed")
        if value.evidence.verification_context is not value.verification_context:
            reasons.append("transaction evidence context changed")
        if id(value._stage6_bound_root) not in _CONVERTED_BOUND_ROOT_IDS:
            reasons.append("transaction is not bound to a converted Stage 6 root")
        return tuple(dict.fromkeys(reasons))
    except (AttributeError, TypeError, ValueError):
        return ("transaction fields are invalid",)


def transaction_validation_reasons(
    value: object,
    *,
    required_context: object = Stage6VerificationContext.REAL_PREREQUISITE,
) -> tuple[str, ...]:
    """Validate issuance and then re-check physical identity as the final step."""

    if not isinstance(required_context, Stage6VerificationContext):
        return ("required context is invalid",)
    reasons = _transaction_validation_reasons_without_io(
        value,
        required_context=required_context,
    )
    if reasons:
        return reasons
    assert isinstance(value, Stage7RootTransaction)
    try:
        identity = _strict_physical_root_identity(value._root)
    except Stage7RootTransactionError as error:
        return error.reasons
    if identity != value._physical_identity:
        return ("transaction physical root identity changed",)
    return ()


def _result_discloses_root(value: object, root: Path) -> bool:
    if isinstance(value, os.PathLike):
        return True
    return isinstance(value, str) and value in {str(root), root.as_posix()}


def consume_stage7_root_transaction(
    value: object,
    consumer: Callable[[Path, Stage7RunEvidence], _T],
    *,
    required_context: object = Stage6VerificationContext.REAL_PREREQUISITE,
) -> _T:
    """Run exactly one trusted callback and never return the authorized root itself."""

    if not callable(consumer):
        raise Stage7RootTransactionError(("consumer must be callable",))
    with _BOUNDARY_LOCK:
        reasons = transaction_validation_reasons(value, required_context=required_context)
        if reasons:
            raise Stage7RootTransactionError(reasons)
        assert isinstance(value, Stage7RootTransaction)
        context = cast(Stage6VerificationContext, required_context)
        if value.verification_context is not context:
            raise Stage7RootTransactionError((f"transaction context must be {context.value}",))
        _CONSUMED_TRANSACTION_IDS.add(id(value))
    raw_result: object = _NO_RESULT
    consumer_error: Exception | None = None
    try:
        raw_result = consumer(value._root, value.evidence)
    except Exception as error:  # trusted callback errors still consume authority
        consumer_error = error
    finally:
        post_reasons: list[str] = []
        if os.getpid() != _PROCESS_ID:
            post_reasons.append("process changed during the Stage 7 callback")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            post_reasons.append("transaction expired during the Stage 7 callback")
        try:
            post_identity = _strict_physical_root_identity(value._root)
        except Stage7RootTransactionError as error:
            post_reasons.extend(error.reasons)
        else:
            if post_identity != value._physical_identity:
                post_reasons.append("physical root identity changed during callback")
        if post_reasons:
            raise Stage7RootTransactionError(tuple(dict.fromkeys(post_reasons)))
    if consumer_error is not None:
        raise Stage7RootTransactionError(("trusted root consumer failed",)) from consumer_error
    assert raw_result is not _NO_RESULT
    result = cast(_T, raw_result)
    if _result_discloses_root(result, value._root):
        raise Stage7RootTransactionError(("consumer returned the authorized root",))
    return result


__all__ = [
    "DEFAULT_STAGE7_TRANSACTION_TTL_SECONDS",
    "MAX_STAGE7_TRANSACTION_TTL_SECONDS",
    "STAGE7_GATE_SCHEMA_VERSION",
    "Stage7PreflightReport",
    "Stage7PreflightStatus",
    "Stage7Reason",
    "Stage7RootTransaction",
    "Stage7RootTransactionError",
    "Stage7RunEvidence",
    "assess_stage7_preflight",
    "begin_stage7_root_transaction",
    "consume_stage7_root_transaction",
    "transaction_validation_reasons",
]
