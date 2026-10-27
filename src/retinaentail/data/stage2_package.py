"""Complete Stage 2 manifest/audit package verification capabilities.

The verifier reads only an explicitly supplied evidence-package directory.  It
never resolves, stats, lists, or opens a dataset root.  A real-data package is
accepted only while the process-issued Stage 2 ``DatasetAccessPermit`` that
authorized the original read-only audit is still valid and exactly cross-bound.

Package descriptors and replay receipts are serializable evidence summaries,
not authority.  Authority is carried only by a live, process-bound
``VerifiedStage2PackageCapability`` registered by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Never

from retinaentail.data.audit import (
    AUDIT_SCHEMA_VERSION,
    AuditIssue,
    AuditResult,
    AuditStatus,
    AuditSummary,
    IssueSeverity,
)
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    permit_validation_reasons,
)
from retinaentail.data.stage2_models import (
    DATASET_LABEL_SEMANTICS,
    DatasetManifestV2,
    LabelAvailability,
    Stage2ContractError,
)

STAGE2_PACKAGE_SCHEMA_VERSION: Final = "1.0.0"
MAX_STAGE2_PACKAGE_TTL_SECONDS: Final = 300
DEFAULT_STAGE2_PACKAGE_TTL_SECONDS: Final = 60
MAX_STAGE2_MANIFEST_BYTES: Final = 64 * 1024 * 1024
MAX_STAGE2_AUDIT_BYTES: Final = 32 * 1024 * 1024

_EXPECTED_FILES: Final = ("audit.json", "manifest-v2.json")
_FILE_LIMITS: Final = {
    "audit.json": MAX_STAGE2_AUDIT_BYTES,
    "manifest-v2.json": MAX_STAGE2_MANIFEST_BYTES,
}
_REPARSE_POINT_ATTRIBUTE: Final = 0x0400
_CAPABILITY_TOKEN: Final = object()
_PROCESS_TOKEN: Final = object()
_PROCESS_NONCE: Final = secrets.token_hex(32)
_PROCESS_ID: Final = os.getpid()
_CAPABILITY_LOCK: Final = threading.RLock()


class Stage2PackageScope(StrEnum):
    """Separate self-contained synthetic evidence from licensed real evidence."""

    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    APPROVED_REAL_DATA = "APPROVED_REAL_DATA"


class Stage2PackageVerificationError(PermissionError):
    """Raised when a package or package capability fails closed."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 2 package verification rejected: " + "; ".join(reasons))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    return result


def _exact_mapping(value: object, *, name: str, fields: set[str]) -> dict[str, object]:
    result = _require_mapping(value, name=name)
    if set(result) != fields:
        raise ValueError(f"{name} has missing or unknown fields")
    return result


def _require_sequence(value: object, *, name: str) -> list[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return list(value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _decode_json(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8", errors="strict")
        decoded: object = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        return _require_mapping(decoded, name=name)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage2PackageFile:
    """One exact regular file in the canonical two-file package."""

    relative_path: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if self.relative_path not in _EXPECTED_FILES:
            raise ValueError("package file relative_path is not allowed")
        _require_sha256(self.sha256, "package file sha256")
        _require_positive_int(self.byte_size, "package file byte_size")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


def _file_manifest_payload(files: tuple[Stage2PackageFile, ...]) -> dict[str, object]:
    return {
        "schema_version": STAGE2_PACKAGE_SCHEMA_VERSION,
        "files": [item.to_dict() for item in files],
    }


def _descriptor_payload(
    *,
    scope: Stage2PackageScope,
    purpose: DataUsePurpose,
    dataset_id: str,
    dataset_version: str,
    record_count: int,
    manifest_sha256: str,
    audit_sha256: str,
    files: tuple[Stage2PackageFile, ...],
    file_manifest_sha256: str,
    registry_sha256: str | None,
    license_review_sha256: str | None,
    adapter_spec_sha256: str | None,
    root_binding_sha256: str | None,
    permit_binding_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": STAGE2_PACKAGE_SCHEMA_VERSION,
        "scope": scope.value,
        "purpose": purpose.value,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "record_count": record_count,
        "manifest_sha256": manifest_sha256,
        "audit_sha256": audit_sha256,
        "files": [item.to_dict() for item in files],
        "file_manifest_sha256": file_manifest_sha256,
        "registry_sha256": registry_sha256,
        "license_review_sha256": license_review_sha256,
        "adapter_spec_sha256": adapter_spec_sha256,
        "root_binding_sha256": root_binding_sha256,
        "permit_binding_sha256": permit_binding_sha256,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage2PackageDescriptor:
    """Primitive immutable digest snapshot; never authority when detached."""

    scope: Stage2PackageScope
    purpose: DataUsePurpose
    dataset_id: str
    dataset_version: str
    record_count: int
    manifest_sha256: str
    audit_sha256: str
    files: tuple[Stage2PackageFile, ...]
    file_manifest_sha256: str
    registry_sha256: str | None
    license_review_sha256: str | None
    adapter_spec_sha256: str | None
    root_binding_sha256: str | None
    permit_binding_sha256: str | None
    package_sha256: str
    schema_version: str = STAGE2_PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE2_PACKAGE_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 2 package schema_version")
        object.__setattr__(self, "scope", Stage2PackageScope(self.scope))
        object.__setattr__(self, "purpose", DataUsePurpose(self.purpose))
        if self.purpose is not DataUsePurpose.READ_ONLY_AUDIT:
            raise ValueError("Stage 2 package purpose must be read_only_audit")
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(self.dataset_version, str) or not self.dataset_version:
            raise ValueError("dataset_version must be a non-empty string")
        _require_positive_int(self.record_count, "record_count")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.audit_sha256, "audit_sha256")
        _require_sha256(self.file_manifest_sha256, "file_manifest_sha256")
        _require_sha256(self.package_sha256, "package_sha256")
        if not all(isinstance(item, Stage2PackageFile) for item in self.files):
            raise TypeError("files must contain Stage2PackageFile values")
        expected_names = tuple(item.relative_path for item in self.files)
        if expected_names != _EXPECTED_FILES or len(set(expected_names)) != len(expected_names):
            raise ValueError("files must be the exact sorted Stage 2 package file set")
        expected_file_manifest = _canonical_sha256(_file_manifest_payload(self.files))
        if self.file_manifest_sha256 != expected_file_manifest:
            raise ValueError("file_manifest_sha256 does not match files")
        bindings = (
            self.registry_sha256,
            self.license_review_sha256,
            self.adapter_spec_sha256,
            self.root_binding_sha256,
            self.permit_binding_sha256,
        )
        if self.scope is Stage2PackageScope.APPROVED_REAL_DATA:
            if any(item is None for item in bindings):
                raise ValueError("real package descriptor requires all permit bindings")
        elif any(item is not None for item in bindings):
            raise ValueError("synthetic package descriptor cannot carry permit bindings")
        for name, digest in zip(
            (
                "registry_sha256",
                "license_review_sha256",
                "adapter_spec_sha256",
                "root_binding_sha256",
                "permit_binding_sha256",
            ),
            bindings,
            strict=True,
        ):
            if digest is not None:
                _require_sha256(digest, name)
        expected_package = _canonical_sha256(
            _descriptor_payload(
                scope=self.scope,
                purpose=self.purpose,
                dataset_id=self.dataset_id,
                dataset_version=self.dataset_version,
                record_count=self.record_count,
                manifest_sha256=self.manifest_sha256,
                audit_sha256=self.audit_sha256,
                files=self.files,
                file_manifest_sha256=self.file_manifest_sha256,
                registry_sha256=self.registry_sha256,
                license_review_sha256=self.license_review_sha256,
                adapter_spec_sha256=self.adapter_spec_sha256,
                root_binding_sha256=self.root_binding_sha256,
                permit_binding_sha256=self.permit_binding_sha256,
            )
        )
        if self.package_sha256 != expected_package:
            raise ValueError("package_sha256 does not match descriptor bindings")

    def to_dict(self) -> dict[str, object]:
        """Return a non-authoritative portable evidence summary."""

        result = _descriptor_payload(
            scope=self.scope,
            purpose=self.purpose,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            record_count=self.record_count,
            manifest_sha256=self.manifest_sha256,
            audit_sha256=self.audit_sha256,
            files=self.files,
            file_manifest_sha256=self.file_manifest_sha256,
            registry_sha256=self.registry_sha256,
            license_review_sha256=self.license_review_sha256,
            adapter_spec_sha256=self.adapter_spec_sha256,
            root_binding_sha256=self.root_binding_sha256,
            permit_binding_sha256=self.permit_binding_sha256,
        )
        result["package_sha256"] = self.package_sha256
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedStage2PackageCapability:
    """Process-bound, expiring authority over one exact verified package."""

    descriptor: Stage2PackageDescriptor
    issued_process_id: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ttl_seconds: int
    _package_dir: Path = field(repr=False, compare=False)
    _permit: DatasetAccessPermit | None = field(repr=False, compare=False)
    _process_nonce: str = field(repr=False, compare=False)
    _issuance_nonce: str = field(repr=False, compare=False)
    _process_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CAPABILITY_TOKEN or self._process_token is not _PROCESS_TOKEN:
            raise TypeError(
                "VerifiedStage2PackageCapability can only be issued by "
                "verify_and_issue_stage2_package"
            )

    @property
    def scope(self) -> Stage2PackageScope:
        return self.descriptor.scope

    @property
    def purpose(self) -> DataUsePurpose:
        return self.descriptor.purpose

    @property
    def authorizes_dataset_root_access(self) -> bool:
        """Package verification never independently authorizes dataset-root I/O."""

        return False

    def __copy__(self) -> Never:
        raise TypeError("Stage 2 package capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("Stage 2 package capabilities cannot be copied")

    def __getstate__(self) -> Never:
        raise TypeError("Stage 2 package capabilities cannot be serialized")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 2 package capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> Never:
        del protocol
        raise TypeError("Stage 2 package capabilities cannot be serialized")


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage2PackageReplayResult:
    """Non-authoritative result of re-reading the complete package."""

    valid: bool
    reasons: tuple[str, ...]
    descriptor: Stage2PackageDescriptor | None = None

    def __post_init__(self) -> None:
        if self.valid != (not self.reasons and self.descriptor is not None):
            raise ValueError("replay result validity does not match reasons and descriptor")

    @property
    def authorizes_consumption(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "descriptor": None if self.descriptor is None else self.descriptor.to_dict(),
            "authorizes_consumption": False,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage2PackageEvidence:
    """Immutable evidence returned only after one complete non-consuming replay.

    This object is intentionally serializable and therefore is not authority.  Its
    purpose is to give Stage 7 both frozen objects needed for byte-independent,
    record-by-record rebuild comparison without exposing the package directory or
    its access permit.
    """

    manifest: DatasetManifestV2
    audit: AuditResult
    descriptor: Stage2PackageDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DatasetManifestV2):
            raise TypeError("evidence manifest must be DatasetManifestV2")
        if not isinstance(self.audit, AuditResult):
            raise TypeError("evidence audit must be AuditResult")
        if not isinstance(self.descriptor, Stage2PackageDescriptor):
            raise TypeError("evidence descriptor must be Stage2PackageDescriptor")
        _manifest_audit_crosscheck(self.manifest, self.audit)
        if (
            self.descriptor.dataset_id != self.manifest.dataset_id
            or self.descriptor.dataset_version != self.manifest.dataset_version
            or self.descriptor.record_count != len(self.manifest.records)
            or self.descriptor.manifest_sha256 != _canonical_sha256(self.manifest.to_dict())
            or self.descriptor.audit_sha256 != _canonical_sha256(self.audit.to_dict())
        ):
            raise ValueError("Stage 2 evidence does not match its package descriptor")

    @property
    def authorizes_dataset_root_access(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "audit": self.audit.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "authorizes_dataset_root_access": False,
        }


_ISSUED_CAPABILITIES: dict[
    int,
    tuple[VerifiedStage2PackageCapability, tuple[object, ...]],
] = {}
_CONSUMED_CAPABILITIES: set[int] = set()


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _REPARSE_POINT_ATTRIBUTE
    )


def _validate_package_directory(package_dir: Path) -> tuple[int, ...]:
    if not isinstance(package_dir, Path):
        raise TypeError("package_dir must be a pathlib.Path")
    if not package_dir.is_absolute() or ".." in package_dir.parts:
        raise ValueError("package_dir must be an absolute normalized path")
    chain = tuple(reversed((package_dir, *package_dir.parents)))
    try:
        for candidate in chain:
            metadata = os.lstat(candidate)
            if _is_link_or_reparse(metadata):
                raise ValueError("package path contains a link or reparse point")
        directory_metadata = os.lstat(package_dir)
    except OSError as error:
        raise ValueError("package directory is missing or unreadable") from error
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise ValueError("package_dir must be a regular directory")
    return _stat_identity(directory_metadata)


def _discover_exact_files(package_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    try:
        with os.scandir(package_dir) as entries:
            for entry in entries:
                if entry.name in discovered:
                    raise ValueError("package contains a filename collision")
                discovered[entry.name] = package_dir / entry.name
    except OSError as error:
        raise ValueError("package directory could not be listed") from error
    names = tuple(sorted(discovered))
    if names != _EXPECTED_FILES:
        missing = sorted(set(_EXPECTED_FILES).difference(discovered))
        unknown = sorted(set(discovered).difference(_EXPECTED_FILES))
        reasons: list[str] = []
        if missing:
            reasons.append("package is missing required files")
        if unknown:
            reasons.append("package contains unknown files")
        raise ValueError(" and ".join(reasons))
    return discovered


def _read_bounded_regular_file(path: Path, *, limit: int) -> tuple[bytes, str]:
    try:
        before_path = os.lstat(path)
    except OSError as error:
        raise ValueError("package file is missing or unreadable") from error
    if _is_link_or_reparse(before_path):
        raise ValueError("package file is a link or reparse point")
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError("package entry is not a regular file")
    if before_path.st_nlink > 1:
        raise ValueError("package file has multiple hard links")
    if before_path.st_size <= 0:
        raise ValueError("package file is empty")
    if before_path.st_size > limit:
        raise ValueError("package file exceeds its bounded size limit")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("package file could not be opened safely") from error
    try:
        before_stream = os.fstat(descriptor)
        if not stat.S_ISREG(before_stream.st_mode):
            raise ValueError("opened package entry is not a regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        digest = hashlib.sha256()
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise ValueError("package file exceeds its bounded size limit")
        after_stream = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise ValueError("package file changed during verification") from error
    identities = {
        _stat_identity(before_path),
        _stat_identity(before_stream),
        _stat_identity(after_stream),
        _stat_identity(after_path),
    }
    if len(identities) != 1 or len(payload) != before_path.st_size:
        raise ValueError("package file changed during verification")
    return payload, digest.hexdigest()


def _parse_audit(value: Mapping[str, object]) -> AuditResult:
    data = _exact_mapping(
        value,
        name="audit",
        fields={
            "schema_version",
            "status",
            "dataset_id",
            "dataset_version",
            "manifest_sha256",
            "summary",
            "issues",
        },
    )
    summary_data = _exact_mapping(
        data["summary"],
        name="audit summary",
        fields={
            "record_count",
            "image_files_checked",
            "label_files_checked",
            "mask_files_decoded",
            "unknown_label_count",
            "issue_counts",
        },
    )
    count_data = _require_mapping(summary_data["issue_counts"], name="audit issue_counts")
    issue_counts: list[tuple[str, int]] = []
    for severity, count in count_data.items():
        IssueSeverity(severity)
        if type(count) is not int or count < 0:  # noqa: E721
            raise ValueError("audit issue counts must be nonnegative integers")
        issue_counts.append((severity, count))
    issues: list[AuditIssue] = []
    for index, raw_issue in enumerate(_require_sequence(data["issues"], name="audit issues")):
        issue_data = _exact_mapping(
            raw_issue,
            name=f"audit issue {index}",
            fields={
                "severity",
                "code",
                "message",
                "relative_path",
                "related_paths",
                "details",
            },
        )
        related = _require_sequence(
            issue_data["related_paths"], name=f"audit issue {index} related_paths"
        )
        if not all(isinstance(item, str) for item in related):
            raise ValueError("audit related_paths must contain strings")
        related_paths = tuple(item for item in related if isinstance(item, str))
        detail_data = _require_mapping(issue_data["details"], name=f"audit issue {index} details")
        if not all(isinstance(item, str) for item in detail_data.values()):
            raise ValueError("audit issue details must contain strings")
        details = tuple((name, item) for name, item in detail_data.items() if isinstance(item, str))
        relative_path = issue_data["relative_path"]
        if relative_path is not None and not isinstance(relative_path, str):
            raise ValueError("audit issue relative_path must be a string or null")
        if not isinstance(issue_data["code"], str) or not isinstance(issue_data["message"], str):
            raise ValueError("audit issue code and message must be strings")
        if not isinstance(issue_data["severity"], str):
            raise ValueError("audit issue severity must be a string")
        issues.append(
            AuditIssue(
                severity=IssueSeverity(issue_data["severity"]),
                code=issue_data["code"],
                message=issue_data["message"],
                relative_path=relative_path,
                related_paths=related_paths,
                details=details,
            )
        )
    if data["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise ValueError("audit schema_version is unsupported")
    if not isinstance(data["status"], str):
        raise ValueError("audit status must be a string")
    for name in ("dataset_id", "dataset_version", "manifest_sha256"):
        if not isinstance(data[name], str):
            raise ValueError(f"audit {name} must be present")
    dataset_id = data["dataset_id"]
    dataset_version = data["dataset_version"]
    manifest_sha256 = data["manifest_sha256"]
    if not isinstance(dataset_id, str):
        raise AssertionError("unreachable audit dataset_id type")
    if not isinstance(dataset_version, str):
        raise AssertionError("unreachable audit dataset_version type")
    if not isinstance(manifest_sha256, str):
        raise AssertionError("unreachable audit manifest_sha256 type")
    summary = AuditSummary(
        record_count=summary_data["record_count"],  # type: ignore[arg-type]
        image_files_checked=summary_data["image_files_checked"],  # type: ignore[arg-type]
        label_files_checked=summary_data["label_files_checked"],  # type: ignore[arg-type]
        mask_files_decoded=summary_data["mask_files_decoded"],  # type: ignore[arg-type]
        unknown_label_count=summary_data["unknown_label_count"],  # type: ignore[arg-type]
        issue_counts=tuple(issue_counts),
    )
    return AuditResult(
        schema_version=AUDIT_SCHEMA_VERSION,
        status=AuditStatus(data["status"]),
        summary=summary,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_sha256=manifest_sha256,
        issues=tuple(issues),
    )


def _manifest_audit_crosscheck(manifest: DatasetManifestV2, audit: AuditResult) -> None:
    manifest_sha256 = _canonical_sha256(manifest.to_dict())
    if audit.status is not AuditStatus.PASS:
        raise ValueError("audit status must be PASS")
    if audit.dataset_id != manifest.dataset_id:
        raise ValueError("audit dataset_id does not match manifest")
    if audit.dataset_version != manifest.dataset_version:
        raise ValueError("audit dataset_version does not match manifest")
    if audit.manifest_sha256 != manifest_sha256:
        raise ValueError("audit manifest_sha256 does not match canonical manifest")
    record_count = len(manifest.records)
    available_labels = sum(
        label.availability is LabelAvailability.AVAILABLE
        for record in manifest.records
        for label in record.labels
    )
    unknown_labels = sum(
        label.availability is LabelAvailability.UNKNOWN
        for record in manifest.records
        for label in record.labels
    )
    mask_labels = sum(
        label.availability is LabelAvailability.AVAILABLE and label.semantic.value.endswith("_MASK")
        for record in manifest.records
        for label in record.labels
    )
    summary = audit.summary
    expected_counts = tuple(sorted(Counter(issue.severity.value for issue in audit.issues).items()))
    if summary.issue_counts != expected_counts:
        raise ValueError("audit issue counts do not exactly match issues")
    if summary.record_count != record_count or summary.image_files_checked != record_count:
        raise ValueError("audit image and record counts do not match complete manifest")
    if summary.label_files_checked != available_labels:
        raise ValueError("audit label count does not match complete manifest")
    if summary.unknown_label_count != unknown_labels:
        raise ValueError("audit unknown-label count does not match complete manifest")
    if summary.mask_files_decoded != mask_labels:
        raise ValueError("audit decoded-mask count does not match complete manifest")


def _registry_snapshot(permit: DatasetAccessPermit) -> dict[str, str]:
    registry = permit.registry
    return {
        "schema_version": registry.schema_version,
        "dataset_id": registry.dataset_id,
        "dataset_version": registry.dataset_version,
        "license_status": registry.license_status,
        "license_review_status": registry.license_review_status,
        "label_verification_status": registry.label_verification_status,
    }


def _permit_bindings(permit: DatasetAccessPermit) -> dict[str, str]:
    registry_sha256 = _canonical_sha256(_registry_snapshot(permit))
    review_sha256 = _canonical_sha256(permit.review.to_dict())
    spec_sha256 = _canonical_sha256(permit.spec.to_dict())
    root_binding_sha256 = _canonical_sha256(
        {
            "dataset_id": permit.registry.dataset_id,
            "dataset_version": permit.registry.dataset_version,
            "purpose": permit.purpose.value,
            "root": permit.root.as_posix(),
        }
    )
    permit_binding_sha256 = _canonical_sha256(
        {
            "registry_sha256": registry_sha256,
            "license_review_sha256": review_sha256,
            "adapter_spec_sha256": spec_sha256,
            "root_binding_sha256": root_binding_sha256,
            "purpose": permit.purpose.value,
        }
    )
    return {
        "registry_sha256": registry_sha256,
        "license_review_sha256": review_sha256,
        "adapter_spec_sha256": spec_sha256,
        "root_binding_sha256": root_binding_sha256,
        "permit_binding_sha256": permit_binding_sha256,
    }


def _manifest_permit_failure(
    manifest: DatasetManifestV2, permit: DatasetAccessPermit
) -> str | None:
    if manifest.dataset_id != permit.registry.dataset_id:
        return "manifest dataset_id does not match access permit"
    if manifest.dataset_version != permit.registry.dataset_version:
        return "manifest dataset_version does not match access permit"
    if manifest.dataset_id != permit.spec.dataset_id:
        return "manifest dataset_id does not match adapter spec"
    if manifest.dataset_id not in DATASET_LABEL_SEMANTICS:
        return "manifest dataset_id has no registered Stage 2 adapter"
    required_semantics = permit.spec.effective_label_semantics
    if len(manifest.records) != len(permit.spec.entries):
        return "manifest record count does not match adapter spec"
    entries = {entry.image_path.as_posix(): entry for entry in permit.spec.entries}
    if len(entries) != len(permit.spec.entries):
        return "adapter spec contains duplicate image paths"
    identity_fields = (
        "source_image_id",
        "split",
        "patient_id",
        "eye_id",
        "followup_id",
        "intervention_pair_id",
    )
    for record in manifest.records:
        entry = entries.get(record.relative_path)
        if entry is None:
            return "manifest image path is not authorized by adapter spec"
        if any(getattr(record, name) != getattr(entry, name) for name in identity_fields):
            return "manifest identities do not exactly match adapter spec"
        observed = {label.semantic: label for label in record.labels}
        if set(observed) != required_semantics:
            return "manifest label semantics do not match authorized task scope"
        expected = {label.semantic: label.relative_path for label in entry.labels}
        if not set(expected) <= set(observed):
            return "manifest is missing an adapter-declared label"
        for semantic, label in observed.items():
            expected_path = expected.get(semantic)
            if expected_path is None:
                if not (
                    label.availability is LabelAvailability.UNKNOWN
                    and label.relative_path is None
                    and label.sha256 is None
                    and label.byte_size is None
                ):
                    return "manifest unknown label state does not match adapter spec"
                continue
            if (
                label.availability is not LabelAvailability.AVAILABLE
                or label.relative_path != expected_path.as_posix()
            ):
                return "manifest available label state does not match adapter spec"
    return None


def _lexically_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _validate_scope_and_permit(
    *,
    package_dir: Path,
    scope: Stage2PackageScope,
    permit: DatasetAccessPermit | None,
) -> Mapping[str, str | None]:
    if scope is Stage2PackageScope.SYNTHETIC_FIXTURE:
        if permit is not None:
            raise ValueError("synthetic package cannot carry a real-data access permit")
        return {
            "registry_sha256": None,
            "license_review_sha256": None,
            "adapter_spec_sha256": None,
            "root_binding_sha256": None,
            "permit_binding_sha256": None,
        }
    if permit is None:
        raise ValueError("real package requires a process-issued DatasetAccessPermit")
    reasons = permit_validation_reasons(permit)
    if reasons:
        raise ValueError("real package access permit is invalid")
    if permit.purpose is not DataUsePurpose.READ_ONLY_AUDIT:
        raise ValueError("real package access permit purpose must be read_only_audit")
    if _lexically_overlap(package_dir, permit.root):
        raise ValueError("evidence package must not overlap the dataset root")
    return _permit_bindings(permit)


def _build_descriptor(
    *,
    scope: Stage2PackageScope,
    manifest: DatasetManifestV2,
    audit: AuditResult,
    file_payloads: Mapping[str, bytes],
    file_sha256: Mapping[str, str],
    permit_bindings: Mapping[str, str | None],
) -> Stage2PackageDescriptor:
    files = tuple(
        Stage2PackageFile(
            relative_path=name,
            sha256=file_sha256[name],
            byte_size=len(file_payloads[name]),
        )
        for name in _EXPECTED_FILES
    )
    file_manifest_sha256 = _canonical_sha256(_file_manifest_payload(files))
    purpose = DataUsePurpose.READ_ONLY_AUDIT
    record_count = len(manifest.records)
    manifest_sha256 = _canonical_sha256(manifest.to_dict())
    audit_sha256 = _canonical_sha256(audit.to_dict())
    registry_sha256 = permit_bindings["registry_sha256"]
    license_review_sha256 = permit_bindings["license_review_sha256"]
    adapter_spec_sha256 = permit_bindings["adapter_spec_sha256"]
    root_binding_sha256 = permit_bindings["root_binding_sha256"]
    permit_binding_sha256 = permit_bindings["permit_binding_sha256"]
    package_sha256 = _canonical_sha256(
        _descriptor_payload(
            scope=scope,
            purpose=purpose,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.dataset_version,
            record_count=record_count,
            manifest_sha256=manifest_sha256,
            audit_sha256=audit_sha256,
            files=files,
            file_manifest_sha256=file_manifest_sha256,
            registry_sha256=registry_sha256,
            license_review_sha256=license_review_sha256,
            adapter_spec_sha256=adapter_spec_sha256,
            root_binding_sha256=root_binding_sha256,
            permit_binding_sha256=permit_binding_sha256,
        )
    )
    return Stage2PackageDescriptor(
        scope=scope,
        purpose=purpose,
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        record_count=record_count,
        manifest_sha256=manifest_sha256,
        audit_sha256=audit_sha256,
        files=files,
        file_manifest_sha256=file_manifest_sha256,
        registry_sha256=registry_sha256,
        license_review_sha256=license_review_sha256,
        adapter_spec_sha256=adapter_spec_sha256,
        root_binding_sha256=root_binding_sha256,
        permit_binding_sha256=permit_binding_sha256,
        package_sha256=package_sha256,
    )


def _read_and_verify_package(
    package_dir: Path,
    *,
    scope: Stage2PackageScope,
    permit: DatasetAccessPermit | None,
) -> tuple[DatasetManifestV2, AuditResult, Stage2PackageDescriptor]:
    if not isinstance(package_dir, Path):
        raise TypeError("package_dir must be a pathlib.Path")
    if not package_dir.is_absolute() or ".." in package_dir.parts:
        raise ValueError("package_dir must be an absolute normalized path")
    permit_bindings = _validate_scope_and_permit(
        package_dir=package_dir,
        scope=scope,
        permit=permit,
    )
    directory_identity = _validate_package_directory(package_dir)
    paths = _discover_exact_files(package_dir)
    payloads: dict[str, bytes] = {}
    file_sha256: dict[str, str] = {}
    for name in _EXPECTED_FILES:
        payload, digest = _read_bounded_regular_file(paths[name], limit=_FILE_LIMITS[name])
        payloads[name] = payload
        file_sha256[name] = digest
    if _stat_identity(os.lstat(package_dir)) != directory_identity:
        raise ValueError("package directory changed during verification")
    if tuple(sorted(_discover_exact_files(package_dir))) != _EXPECTED_FILES:
        raise ValueError("package file set changed during verification")

    manifest_data = _decode_json(payloads["manifest-v2.json"], name="manifest-v2.json")
    audit_data = _decode_json(payloads["audit.json"], name="audit.json")
    try:
        manifest = DatasetManifestV2.from_dict(manifest_data)
        audit = _parse_audit(audit_data)
    except (Stage2ContractError, TypeError, ValueError) as error:
        raise ValueError("package evidence contract is invalid") from error
    _manifest_audit_crosscheck(manifest, audit)

    if scope is Stage2PackageScope.SYNTHETIC_FIXTURE:
        if not manifest.dataset_id.startswith(("synthetic_", "toy_")):
            raise ValueError("synthetic package dataset_id must use a synthetic prefix")
    else:
        if permit is None:
            raise AssertionError("unreachable real package permit state")
        failure = _manifest_permit_failure(manifest, permit)
        if failure is not None:
            raise ValueError(failure)
    descriptor = _build_descriptor(
        scope=scope,
        manifest=manifest,
        audit=audit,
        file_payloads=payloads,
        file_sha256=file_sha256,
        permit_bindings=permit_bindings,
    )
    return manifest, audit, descriptor


def _live_permit_binding(value: VerifiedStage2PackageCapability) -> str | None:
    if value._permit is None:
        return None
    return _permit_bindings(value._permit)["permit_binding_sha256"]


def _capability_fingerprint(value: VerifiedStage2PackageCapability) -> tuple[object, ...]:
    return (
        _canonical_sha256(value.descriptor.to_dict()),
        value._package_dir.as_posix(),
        _live_permit_binding(value),
        value.issued_process_id,
        value.issued_monotonic_ns,
        value.expires_monotonic_ns,
        value.ttl_seconds,
        value._process_nonce,
        value._issuance_nonce,
        value._process_token,
        value._token,
    )


def _validate_live_descriptor(value: object) -> None:
    if not isinstance(value, Stage2PackageDescriptor):
        raise TypeError("descriptor has an invalid type")
    Stage2PackageDescriptor(
        schema_version=value.schema_version,
        scope=value.scope,
        purpose=value.purpose,
        dataset_id=value.dataset_id,
        dataset_version=value.dataset_version,
        record_count=value.record_count,
        manifest_sha256=value.manifest_sha256,
        audit_sha256=value.audit_sha256,
        files=value.files,
        file_manifest_sha256=value.file_manifest_sha256,
        registry_sha256=value.registry_sha256,
        license_review_sha256=value.license_review_sha256,
        adapter_spec_sha256=value.adapter_spec_sha256,
        root_binding_sha256=value.root_binding_sha256,
        permit_binding_sha256=value.permit_binding_sha256,
        package_sha256=value.package_sha256,
    )


def _capability_validation_reasons_unlocked(value: object) -> tuple[str, ...]:
    if not isinstance(value, VerifiedStage2PackageCapability):
        return ("capability has an invalid type",)
    issued = _ISSUED_CAPABILITIES.get(id(value))
    if issued is None or issued[0] is not value:
        return ("capability was not issued by the Stage 2 package verifier",)
    if id(value) in _CONSUMED_CAPABILITIES:
        return ("capability has already been consumed",)
    try:
        if issued[1] != _capability_fingerprint(value):
            return ("capability bindings changed after issuance",)
        reasons: list[str] = []
        if value._token is not _CAPABILITY_TOKEN or value._process_token is not _PROCESS_TOKEN:
            reasons.append("capability issuance identity is invalid")
        if value._process_nonce != _PROCESS_NONCE:
            reasons.append("capability process nonce is invalid")
        if value.issued_process_id != _PROCESS_ID or os.getpid() != _PROCESS_ID:
            reasons.append("capability belongs to a different process")
        descriptor_object: object = value.descriptor
        try:
            _validate_live_descriptor(descriptor_object)
        except (TypeError, ValueError):
            reasons.append("capability descriptor bindings are invalid")
        if not isinstance(value._package_dir, Path) or not value._package_dir.is_absolute():
            reasons.append("capability package path binding is invalid")
        if type(value.ttl_seconds) is not int or not (  # noqa: E721
            1 <= value.ttl_seconds <= MAX_STAGE2_PACKAGE_TTL_SECONDS
        ):
            reasons.append("capability TTL is invalid")
        elif (
            value.expires_monotonic_ns - value.issued_monotonic_ns
            != value.ttl_seconds * 1_000_000_000
        ):
            reasons.append("capability expiry does not match TTL")
        if time.monotonic_ns() >= value.expires_monotonic_ns:
            reasons.append("capability has expired")
        if value.scope is Stage2PackageScope.APPROVED_REAL_DATA:
            if value._permit is None:
                reasons.append("real capability is missing its access permit")
            else:
                if permit_validation_reasons(value._permit):
                    reasons.append("real capability access permit is no longer valid")
                elif value._permit.purpose is not DataUsePurpose.READ_ONLY_AUDIT:
                    reasons.append("real capability access permit purpose changed")
                else:
                    live = _permit_bindings(value._permit)
                    descriptor = value.descriptor
                    expected = {
                        "registry_sha256": descriptor.registry_sha256,
                        "license_review_sha256": descriptor.license_review_sha256,
                        "adapter_spec_sha256": descriptor.adapter_spec_sha256,
                        "root_binding_sha256": descriptor.root_binding_sha256,
                        "permit_binding_sha256": descriptor.permit_binding_sha256,
                    }
                    if live != expected:
                        reasons.append("real capability permit bindings changed")
        elif value._permit is not None:
            reasons.append("synthetic capability cannot carry a real-data permit")
        return tuple(reasons)
    except (AttributeError, TypeError, ValueError):
        return ("capability fields are invalid",)


def stage2_package_capability_validation_reasons(value: object) -> tuple[str, ...]:
    """Validate issuance, mutation resistance, process binding, TTL, and permit."""

    with _CAPABILITY_LOCK:
        return _capability_validation_reasons_unlocked(value)


def stage2_package_capability_snapshot(
    value: object,
) -> Stage2PackageDescriptor:
    """Return the live descriptor only after capability validation.

    The returned descriptor remains non-authoritative when separated from its
    capability.  Consumers that depend on current package bytes must additionally
    call :func:`replay_stage2_package` or :func:`consume_stage2_package`.
    """

    reasons = stage2_package_capability_validation_reasons(value)
    if reasons:
        raise Stage2PackageVerificationError(reasons)
    if not isinstance(value, VerifiedStage2PackageCapability):
        raise AssertionError("unreachable capability type")
    return value.descriptor


def _require_ttl_seconds(value: object) -> int:
    if type(value) is not int or not (  # noqa: E721
        1 <= value <= MAX_STAGE2_PACKAGE_TTL_SECONDS
    ):
        raise ValueError(
            f"ttl_seconds must be an integer from 1 to {MAX_STAGE2_PACKAGE_TTL_SECONDS}"
        )
    return value


def verify_and_issue_stage2_package(
    package_dir: Path,
    *,
    scope: Stage2PackageScope,
    permit: DatasetAccessPermit | None = None,
    ttl_seconds: int = DEFAULT_STAGE2_PACKAGE_TTL_SECONDS,
) -> VerifiedStage2PackageCapability:
    """Fully verify a two-file package and issue short-lived process authority."""

    ttl = _require_ttl_seconds(ttl_seconds)
    try:
        normalized_scope = Stage2PackageScope(scope)
        _, _, descriptor = _read_and_verify_package(
            package_dir,
            scope=normalized_scope,
            permit=permit,
        )
    except (OSError, TypeError, ValueError) as error:
        raise Stage2PackageVerificationError((str(error),)) from error
    now = time.monotonic_ns()
    capability = VerifiedStage2PackageCapability(
        descriptor=descriptor,
        issued_process_id=_PROCESS_ID,
        issued_monotonic_ns=now,
        expires_monotonic_ns=now + ttl * 1_000_000_000,
        ttl_seconds=ttl,
        _package_dir=package_dir,
        _permit=permit,
        _process_nonce=_PROCESS_NONCE,
        _issuance_nonce=secrets.token_hex(32),
        _process_token=_PROCESS_TOKEN,
        _token=_CAPABILITY_TOKEN,
    )
    with _CAPABILITY_LOCK:
        _ISSUED_CAPABILITIES[id(capability)] = (
            capability,
            _capability_fingerprint(capability),
        )
    return capability


def _replay_verified(
    value: VerifiedStage2PackageCapability,
) -> tuple[DatasetManifestV2, AuditResult, Stage2PackageDescriptor]:
    manifest, audit, descriptor = _read_and_verify_package(
        value._package_dir,
        scope=value.scope,
        permit=value._permit,
    )
    if descriptor != value.descriptor:
        raise ValueError("replayed package does not match issued descriptor")
    reasons = stage2_package_capability_validation_reasons(value)
    if reasons:
        raise Stage2PackageVerificationError(reasons)
    return manifest, audit, descriptor


def replay_stage2_package(value: object) -> Stage2PackageReplayResult:
    """Re-read the same package without issuing or extending authority."""

    reasons = stage2_package_capability_validation_reasons(value)
    if reasons:
        return Stage2PackageReplayResult(valid=False, reasons=reasons)
    if not isinstance(value, VerifiedStage2PackageCapability):
        return Stage2PackageReplayResult(
            valid=False,
            reasons=("capability has an invalid type",),
        )
    try:
        _, _, descriptor = _replay_verified(value)
    except (OSError, TypeError, ValueError, Stage2PackageVerificationError) as error:
        if isinstance(error, Stage2PackageVerificationError):
            replay_reasons = error.reasons
        else:
            replay_reasons = (str(error),)
        return Stage2PackageReplayResult(valid=False, reasons=replay_reasons)
    return Stage2PackageReplayResult(valid=True, reasons=(), descriptor=descriptor)


def replay_stage2_package_manifest(value: object) -> DatasetManifestV2:
    """Fully replay without consuming authority and return the immutable manifest.

    This is the narrow membership-verification boundary for downstream stages.  It
    revalidates issuance identity, process/PID binding, monotonic TTL, current permit
    bindings, exact package bytes, and manifest/audit cross-bindings.  It neither
    extends the capability lifetime nor marks the capability consumed.
    """

    reasons = stage2_package_capability_validation_reasons(value)
    if reasons:
        raise Stage2PackageVerificationError(reasons)
    if not isinstance(value, VerifiedStage2PackageCapability):
        raise AssertionError("unreachable capability type")
    try:
        manifest, _, _ = _replay_verified(value)
    except Stage2PackageVerificationError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise Stage2PackageVerificationError((str(error),)) from error
    return manifest


def replay_stage2_package_evidence(value: object) -> Stage2PackageEvidence:
    """Fully replay and return the frozen manifest/audit pair without consuming.

    The returned dataclass is a detached evidence snapshot, never an access token.
    The live capability remains required for every later authoritative replay.
    """

    reasons = stage2_package_capability_validation_reasons(value)
    if reasons:
        raise Stage2PackageVerificationError(reasons)
    if not isinstance(value, VerifiedStage2PackageCapability):
        raise AssertionError("unreachable capability type")
    try:
        manifest, audit, descriptor = _replay_verified(value)
    except Stage2PackageVerificationError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise Stage2PackageVerificationError((str(error),)) from error
    return Stage2PackageEvidence(manifest=manifest, audit=audit, descriptor=descriptor)


def consume_stage2_package(value: object) -> DatasetManifestV2:
    """Replay and consume one capability exactly once, returning immutable manifest data."""

    reasons = stage2_package_capability_validation_reasons(value)
    if reasons:
        raise Stage2PackageVerificationError(reasons)
    if not isinstance(value, VerifiedStage2PackageCapability):
        raise AssertionError("unreachable capability type")
    try:
        manifest, _, _ = _replay_verified(value)
    except (OSError, TypeError, ValueError) as error:
        raise Stage2PackageVerificationError((str(error),)) from error
    with _CAPABILITY_LOCK:
        final_reasons = _capability_validation_reasons_unlocked(value)
        if final_reasons:
            raise Stage2PackageVerificationError(final_reasons)
        _CONSUMED_CAPABILITIES.add(id(value))
    return manifest


__all__ = [
    "DEFAULT_STAGE2_PACKAGE_TTL_SECONDS",
    "MAX_STAGE2_PACKAGE_TTL_SECONDS",
    "STAGE2_PACKAGE_SCHEMA_VERSION",
    "Stage2PackageDescriptor",
    "Stage2PackageEvidence",
    "Stage2PackageFile",
    "Stage2PackageReplayResult",
    "Stage2PackageScope",
    "Stage2PackageVerificationError",
    "VerifiedStage2PackageCapability",
    "consume_stage2_package",
    "replay_stage2_package",
    "replay_stage2_package_evidence",
    "replay_stage2_package_manifest",
    "stage2_package_capability_snapshot",
    "stage2_package_capability_validation_reasons",
    "verify_and_issue_stage2_package",
]
