"""Read-only, fail-closed auditing for versioned local dataset manifests.

The auditor never mutates dataset files and never stores the dataset root in its
result.  All reportable paths pass a containment check and are normalized to
POSIX-style paths relative to the supplied root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    permit_validation_reasons,
)
from retinaentail.data.stage2_models import (
    DATASET_LABEL_SEMANTICS,
    DatasetManifestV2,
    LabelAvailability,
    LabelRecordV2,
    ManifestRecordV2,
)

AUDIT_SCHEMA_VERSION = "2.0.0"
_DATASET_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ISSUE_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DETAIL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DATASET_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")


class IssueSeverity(StrEnum):
    """Severity of one deterministic audit finding."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditStatus(StrEnum):
    """Overall status; any unresolved warning or error closes the gate."""

    PASS = "PASS"
    FAIL_CLOSED = "FAIL_CLOSED"
    BLOCKED = "BLOCKED"


_SEVERITY_ORDER = {
    IssueSeverity.CRITICAL: 0,
    IssueSeverity.ERROR: 1,
    IssueSeverity.WARNING: 2,
    IssueSeverity.INFO: 3,
}


def _safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return None
    if any(part in {"", ".", ".."} for part in posix.parts):
        return None
    normalized = posix.as_posix()
    return normalized if normalized == value else None


def _path_free_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if any(character in normalized for character in "\\/\r\n"):
        raise ValueError(f"{name} must not contain filesystem paths or line breaks")
    return normalized


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """One root-redacted, JSON-safe audit issue."""

    severity: IssueSeverity
    code: str
    message: str
    relative_path: str | None = None
    related_paths: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        if not isinstance(self.code, str) or _ISSUE_CODE_PATTERN.fullmatch(self.code) is None:
            raise ValueError("issue code must be a normalized uppercase identifier")
        object.__setattr__(self, "message", _path_free_text(self.message, "issue message"))
        if self.relative_path is not None and _safe_relative_path(self.relative_path) is None:
            raise ValueError("issue relative_path must be a normalized relative path")
        normalized_paths: list[str] = []
        for path in self.related_paths:
            normalized = _safe_relative_path(path)
            if normalized is None:
                raise ValueError("issue related_paths must contain normalized relative paths")
            normalized_paths.append(normalized)
        object.__setattr__(self, "related_paths", tuple(sorted(set(normalized_paths))))
        normalized_details: list[tuple[str, str]] = []
        detail_names: set[str] = set()
        for name, value in self.details:
            if (
                not isinstance(name, str)
                or _DETAIL_NAME_PATTERN.fullmatch(name) is None
                or name in detail_names
            ):
                raise ValueError("issue detail names must be unique normalized identifiers")
            detail_names.add(name)
            normalized_details.append((name, _path_free_text(value, f"issue detail {name}")))
        object.__setattr__(self, "details", tuple(sorted(normalized_details)))

    def sort_key(self) -> tuple[object, ...]:
        return (
            _SEVERITY_ORDER[self.severity],
            self.code,
            self.relative_path or "",
            self.related_paths,
            self.details,
            self.message,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "relative_path": self.relative_path,
            "related_paths": list(self.related_paths),
            "details": {name: value for name, value in self.details},
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditSummary:
    """Small deterministic counters suitable for JSON and HTML reports."""

    record_count: int
    image_files_checked: int
    label_files_checked: int
    mask_files_decoded: int
    unknown_label_count: int
    issue_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "record_count",
            "image_files_checked",
            "label_files_checked",
            "mask_files_decoded",
            "unknown_label_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        normalized_counts: list[tuple[str, int]] = []
        for severity, count in self.issue_counts:
            IssueSeverity(severity)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("issue counts must be non-negative integers")
            normalized_counts.append((severity, count))
        object.__setattr__(self, "issue_counts", tuple(sorted(normalized_counts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "image_files_checked": self.image_files_checked,
            "label_files_checked": self.label_files_checked,
            "mask_files_decoded": self.mask_files_decoded,
            "unknown_label_count": self.unknown_label_count,
            "issue_counts": {name: count for name, count in self.issue_counts},
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditResult:
    """Versioned audit result with no absolute filesystem paths."""

    status: AuditStatus
    summary: AuditSummary
    dataset_id: str | None = None
    dataset_version: str | None = None
    manifest_sha256: str | None = None
    issues: tuple[AuditIssue, ...] = ()
    schema_version: str = AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported audit schema_version")
        object.__setattr__(self, "status", AuditStatus(self.status))
        if not isinstance(self.summary, AuditSummary):
            raise TypeError("summary must be an AuditSummary")
        if self.dataset_id is not None and _DATASET_ID_PATTERN.fullmatch(self.dataset_id) is None:
            raise ValueError("dataset_id must be a safe normalized identifier")
        if self.dataset_version is not None:
            dataset_version = _path_free_text(self.dataset_version, "dataset_version")
            if _DATASET_VERSION_PATTERN.fullmatch(dataset_version) is None:
                raise ValueError("dataset_version must be a report-safe release identifier")
            object.__setattr__(self, "dataset_version", dataset_version)
        if self.manifest_sha256 is not None and (
            _SHA256_PATTERN.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        if not all(isinstance(issue, AuditIssue) for issue in self.issues):
            raise TypeError("issues must contain only AuditIssue values")
        ordered_issues = tuple(sorted(self.issues, key=AuditIssue.sort_key))
        object.__setattr__(self, "issues", ordered_issues)
        has_unresolved_issue = any(
            issue.severity is not IssueSeverity.INFO for issue in ordered_issues
        )
        if self.status is AuditStatus.PASS and has_unresolved_issue:
            raise ValueError("PASS audit result cannot contain unresolved issues")
        expected_counts = tuple(
            sorted(Counter(issue.severity.value for issue in ordered_issues).items())
        )
        if self.summary.issue_counts != expected_counts:
            raise ValueError("summary issue_counts must exactly match issues")
        if self.status is AuditStatus.PASS and (
            self.summary.record_count == 0
            or self.summary.image_files_checked != self.summary.record_count
        ):
            raise ValueError("PASS audit result requires every non-empty manifest image checked")
        if self.status is AuditStatus.BLOCKED:
            summary_values = (
                self.summary.record_count,
                self.summary.image_files_checked,
                self.summary.label_files_checked,
                self.summary.mask_files_decoded,
                self.summary.unknown_label_count,
            )
            if any(summary_values) or not ordered_issues:
                raise ValueError("BLOCKED audit result requires zero file counts and gate issues")

    @property
    def passed(self) -> bool:
        return self.status is AuditStatus.PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
            "summary": self.summary.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class _DecodedImage:
    width: int
    height: int
    channels: int
    image_format: str
    mode: str
    colors: tuple[tuple[int, object], ...] | None = None
    palette: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    byte_size: int
    sha256: str
    identity: tuple[int, int, int, int, int]
    stable: bool


@dataclass(slots=True)
class _AuditCounters:
    image_files_checked: int = 0
    label_files_checked: int = 0
    mask_files_decoded: int = 0
    unknown_label_count: int = 0


def _detail(**values: object) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for name, value in values.items():
        text = str(value)
        if any(character in text for character in "\\/\r\n"):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            text = f"redacted-{digest}"
        normalized.append((name, text))
    return tuple(sorted(normalized))


def _resolve_under_root(root: Path, relative_path: str) -> Path | None:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _resolve_dataset_root(value: object) -> Path | None:
    if not isinstance(value, Path):
        return None
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_dir() else None


def _decode_image(path: Path, *, collect_colors: bool) -> _DecodedImage:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as candidate:
            image_format = candidate.format or "UNKNOWN"
            candidate.verify()
        with Image.open(path) as decoded:
            decoded.load()
            width, height = decoded.size
            channels = len(decoded.getbands())
            mode = decoded.mode
            raw_colors = decoded.getcolors(maxcolors=257) if collect_colors else None
            raw_palette = decoded.getpalette() if collect_colors and mode == "P" else None
    colors = None if raw_colors is None else tuple(raw_colors)
    palette = None if raw_palette is None else tuple(raw_palette)
    return _DecodedImage(width, height, channels, image_format, mode, colors, palette)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fingerprint_file(path: Path, *, chunk_size: int = 1024 * 1024) -> _FileFingerprint:
    """Hash one open handle in chunks and verify that the path still names that snapshot."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    path_after = path.stat()
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    path_identity = _stat_identity(path_after)
    return _FileFingerprint(
        byte_size=before.st_size,
        sha256=digest.hexdigest(),
        identity=before_identity,
        stable=before_identity == after_identity == path_identity,
    )


def _normalized_format(value: object) -> str:
    result = str(value).strip().upper().lstrip(".")
    return "JPEG" if result == "JPG" else result


def _semantic_text(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value).casefold()
    return str(value).casefold()


def _is_mask_label(label: LabelRecordV2) -> bool:
    semantic = _semantic_text(label.semantic)
    return semantic.endswith("_mask") or "segmentation" in semantic


def _identifier_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _make_result(
    *,
    record_count: int,
    issues: Iterable[AuditIssue],
    counters: _AuditCounters,
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    manifest_sha256: str | None = None,
) -> AuditResult:
    ordered = tuple(sorted(issues, key=AuditIssue.sort_key))
    severity_counts = Counter(issue.severity.value for issue in ordered)
    summary = AuditSummary(
        record_count=record_count,
        image_files_checked=counters.image_files_checked,
        label_files_checked=counters.label_files_checked,
        mask_files_decoded=counters.mask_files_decoded,
        unknown_label_count=counters.unknown_label_count,
        issue_counts=tuple(sorted(severity_counts.items())),
    )
    status = (
        AuditStatus.PASS
        if not any(issue.severity is not IssueSeverity.INFO for issue in ordered)
        else AuditStatus.FAIL_CLOSED
    )
    return AuditResult(
        status=status,
        summary=summary,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_sha256=manifest_sha256,
        issues=ordered,
    )


def blocked_audit_result(
    reasons: Iterable[str],
    *,
    dataset_id: str | None = None,
) -> AuditResult:
    """Build a deterministic zero-file report for a pre-filesystem gate denial.

    Reasons are deliberately constrained to single-line, path-free descriptions
    so callers cannot accidentally copy a local root into a report.
    """

    if dataset_id is not None and _DATASET_ID_PATTERN.fullmatch(dataset_id) is None:
        raise ValueError("dataset_id must be a safe normalized identifier")
    normalized_reasons: set[str] = set()
    for reason in reasons:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("blocked reasons must be non-empty strings")
        normalized = reason.strip()
        if len(normalized) > 240 or any(character in normalized for character in "\\/\r\n"):
            raise ValueError("blocked reasons must not contain paths or line breaks")
        normalized_reasons.add(normalized)
    if not normalized_reasons:
        raise ValueError("at least one blocked reason is required")
    issues = tuple(
        AuditIssue(
            severity=IssueSeverity.WARNING,
            code="ACCESS_GATE_BLOCKED",
            message=reason,
        )
        for reason in sorted(normalized_reasons)
    )
    counters = _AuditCounters()
    result = _make_result(
        record_count=0,
        issues=issues,
        counters=counters,
        dataset_id=dataset_id,
    )
    return AuditResult(
        status=AuditStatus.BLOCKED,
        summary=result.summary,
        dataset_id=dataset_id,
        issues=result.issues,
    )


def _append_issue(
    issues: list[AuditIssue],
    severity: IssueSeverity,
    code: str,
    message: str,
    *,
    relative_path: str | None = None,
    related_paths: Iterable[str] = (),
    details: tuple[tuple[str, str], ...] = (),
) -> None:
    issues.append(
        AuditIssue(
            severity=severity,
            code=code,
            message=message,
            relative_path=relative_path,
            related_paths=tuple(sorted(set(related_paths))),
            details=details,
        )
    )


def _audit_mask_pixels(
    decoded: _DecodedImage,
    issues: list[AuditIssue],
    *,
    relative_path: str,
) -> None:
    if decoded.colors is None:
        _append_issue(
            issues,
            IssueSeverity.WARNING,
            "MASK_VALUE_CARDINALITY_HIGH",
            "Mask has more than 257 distinct values; values require manual review.",
            relative_path=relative_path,
        )
        return
    values = {value for _, value in decoded.colors}
    if values == {0}:
        _append_issue(
            issues,
            IssueSeverity.WARNING,
            "MASK_ALL_ZERO",
            "Mask is all zero; this is not interpreted as a confirmed negative label.",
            relative_path=relative_path,
        )
    if 1 in values and 255 in values:
        _append_issue(
            issues,
            IssueSeverity.WARNING,
            "MASK_ENCODING_MIXED",
            "Mask mixes one-valued and 255-valued foreground encodings.",
            relative_path=relative_path,
        )
    allowed = {0, 1, 255}
    if not values.issubset(allowed):
        preview = ",".join(sorted(str(value) for value in values)[:8])
        _append_issue(
            issues,
            IssueSeverity.WARNING,
            "MASK_VALUE_UNEXPECTED",
            "Mask contains values outside the conservative binary-mask set.",
            relative_path=relative_path,
            details=_detail(values_preview=preview),
        )


def _audit_idrid_binary_mask_encoding(
    decoded: _DecodedImage,
    issues: list[AuditIssue],
    *,
    relative_path: str,
) -> bool:
    """Validate IDRiD's documented black/red binary TIFF encodings in memory."""

    if decoded.mode not in {"P", "RGBA"}:
        return False
    normalized_colors: list[tuple[int, object]] = []
    valid = decoded.colors is not None
    if decoded.mode == "P":
        palette = decoded.palette
        valid = valid and palette is not None and len(palette) >= 6
        if valid and decoded.colors is not None and palette is not None:
            expected_palette = {0: (0, 0, 0), 1: (255, 0, 0)}
            for count, value in decoded.colors:
                if not isinstance(value, int) or value not in expected_palette:
                    valid = False
                    break
                start = value * 3
                if tuple(palette[start : start + 3]) != expected_palette[value]:
                    valid = False
                    break
                normalized_colors.append((count, value))
    else:
        expected_rgba = {(0, 0, 0, 255): 0, (255, 0, 0, 255): 1}
        if valid and decoded.colors is not None:
            for count, value in decoded.colors:
                if not isinstance(value, tuple) or value not in expected_rgba:
                    valid = False
                    break
                normalized_colors.append((count, expected_rgba[value]))
    if not valid or not normalized_colors:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MASK_IDRID_ENCODING_INVALID",
            "IDRiD mask is not an exact opaque black/red binary encoding.",
            relative_path=relative_path,
            details=_detail(mode=decoded.mode),
        )
        return True
    normalized = _DecodedImage(
        decoded.width,
        decoded.height,
        1,
        decoded.image_format,
        "L",
        tuple(normalized_colors),
    )
    _audit_mask_pixels(normalized, issues, relative_path=relative_path)
    return True


def _audit_label(
    label: LabelRecordV2,
    *,
    dataset_id: str,
    root: Path,
    image_dimensions: tuple[int, int],
    issues: list[AuditIssue],
    counters: _AuditCounters,
) -> None:
    raw_path = label.relative_path
    if label.availability is LabelAvailability.UNKNOWN:
        counters.unknown_label_count += 1
        _append_issue(
            issues,
            IssueSeverity.WARNING,
            "LABEL_AVAILABILITY_UNKNOWN",
            "Label availability is UNKNOWN and must not be interpreted as lesion absence.",
            relative_path=_safe_relative_path(raw_path),
            details=_detail(semantic=_semantic_text(label.semantic)),
        )
    if raw_path is None:
        return
    relative_path = _safe_relative_path(raw_path)
    if relative_path is None:
        _append_issue(
            issues,
            IssueSeverity.CRITICAL,
            "LABEL_PATH_INVALID",
            "Label path is not a normalized relative path within the dataset root.",
        )
        return
    resolved = _resolve_under_root(root, relative_path)
    if resolved is None:
        _append_issue(
            issues,
            IssueSeverity.CRITICAL,
            "LABEL_PATH_ESCAPE",
            "Label path resolves outside the dataset root.",
            relative_path=relative_path,
        )
        return
    if not resolved.is_file():
        counters.unknown_label_count += int(label.availability is not LabelAvailability.UNKNOWN)
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "LABEL_FILE_MISSING",
            "Declared label file is missing; its state remains UNKNOWN.",
            relative_path=relative_path,
        )
        return
    counters.label_files_checked += 1
    try:
        fingerprint = _fingerprint_file(resolved)
    except OSError as error:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "LABEL_FILE_UNREADABLE",
            "Label file could not be read.",
            relative_path=relative_path,
            details=_detail(error_type=type(error).__name__),
        )
        return
    actual_size = fingerprint.byte_size
    actual_hash = fingerprint.sha256
    change_reported = False
    if not fingerprint.stable:
        _append_issue(
            issues,
            IssueSeverity.CRITICAL,
            "LABEL_CHANGED_DURING_AUDIT",
            "Label file identity or metadata changed while it was being hashed.",
            relative_path=relative_path,
        )
        change_reported = True
    if label.byte_size is not None and actual_size != label.byte_size:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "LABEL_SIZE_MISMATCH",
            "Label byte size does not match the manifest.",
            relative_path=relative_path,
            details=_detail(expected=label.byte_size, actual=actual_size),
        )
    if label.sha256 is not None and actual_hash != label.sha256:
        _append_issue(
            issues,
            IssueSeverity.CRITICAL,
            "LABEL_HASH_MISMATCH",
            "Label SHA-256 does not match the manifest.",
            relative_path=relative_path,
        )
    if not _is_mask_label(label):
        return
    try:
        decoded = _decode_image(resolved, collect_colors=True)
    except (
        MemoryError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MASK_DECODE_FAILED",
            "Mask could not be decoded completely by Pillow.",
            relative_path=relative_path,
            details=_detail(error_type=type(error).__name__),
        )
        return
    counters.mask_files_decoded += 1
    try:
        decoded_fingerprint = _fingerprint_file(resolved)
    except OSError as error:
        _append_issue(
            issues,
            IssueSeverity.CRITICAL,
            "LABEL_CHANGED_DURING_AUDIT",
            "Label file could not be revalidated after decoding.",
            relative_path=relative_path,
            details=_detail(error_type=type(error).__name__),
        )
    else:
        if decoded_fingerprint != fingerprint and not change_reported:
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "LABEL_CHANGED_DURING_AUDIT",
                "Label bytes or file identity changed between hashing and decoding.",
                relative_path=relative_path,
            )
    if (decoded.width, decoded.height) != image_dimensions:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MASK_DIMENSION_MISMATCH",
            "Mask dimensions do not match its image.",
            relative_path=relative_path,
            details=_detail(
                expected=f"{image_dimensions[0]}x{image_dimensions[1]}",
                actual=f"{decoded.width}x{decoded.height}",
            ),
        )
    if dataset_id == "idrid" and _audit_idrid_binary_mask_encoding(
        decoded,
        issues,
        relative_path=relative_path,
    ):
        return
    if decoded.channels != 1:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MASK_CHANNEL_MISMATCH",
            "Mask must decode as a single-channel image.",
            relative_path=relative_path,
            details=_detail(actual=decoded.channels),
        )
    if decoded.mode in {"P", "PA"}:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MASK_MODE_PALETTE",
            "Palette masks are ambiguous and require an explicit controlled conversion.",
            relative_path=relative_path,
        )
        return
    controlled_mode = decoded.mode in {"1", "L", "I"} or decoded.mode.startswith("I;16")
    if not controlled_mode:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MASK_MODE_UNEXPECTED",
            "Mask mode is not one of the controlled integer mask modes.",
            relative_path=relative_path,
            details=_detail(mode=decoded.mode),
        )
        return
    _audit_mask_pixels(decoded, issues, relative_path=relative_path)


def _audit_labels(
    labels: Iterable[LabelRecordV2],
    *,
    dataset_id: str,
    root: Path,
    image_dimensions: tuple[int, int],
    issues: list[AuditIssue],
    counters: _AuditCounters,
) -> None:
    for label in labels:
        _audit_label(
            label,
            dataset_id=dataset_id,
            root=root,
            image_dimensions=image_dimensions,
            issues=issues,
            counters=counters,
        )


def _audit_records_at_root(
    records: Iterable[ManifestRecordV2],
    *,
    dataset_root: Path,
    dataset_version: str | None = None,
    manifest_sha256: str | None = None,
) -> AuditResult:
    """Audit manifest records against one local root without mutating either.

    The function captures per-file failures as issues and returns a fail-closed
    result instead of treating unreadable or unknown labels as negative cases.
    """

    materialized: tuple[object, ...] = tuple(records)
    issues: list[AuditIssue] = []
    counters = _AuditCounters()
    if not materialized:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "MANIFEST_EMPTY",
            "Manifest contains no records.",
        )
    dataset_ids = {
        candidate.dataset_id
        for candidate in materialized
        if isinstance(candidate, ManifestRecordV2)
    }
    dataset_id = next(iter(dataset_ids)) if len(dataset_ids) == 1 else None
    if len(dataset_ids) > 1:
        _append_issue(
            issues,
            IssueSeverity.ERROR,
            "DATASET_ID_MIXED",
            "Manifest combines records from more than one dataset.",
            details=_detail(dataset_count=len(dataset_ids)),
        )
    root = _resolve_dataset_root(dataset_root)
    if root is None:
        _append_issue(
            issues,
            IssueSeverity.CRITICAL,
            "DATASET_ROOT_INVALID",
            "Dataset root cannot be resolved as a local directory.",
        )
        return _make_result(
            record_count=len(materialized),
            issues=issues,
            counters=counters,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            manifest_sha256=manifest_sha256,
        )

    paths: dict[str, list[ManifestRecordV2]] = defaultdict(list)
    actual_hash_paths: dict[str, set[str]] = defaultdict(set)
    basename_hash_paths: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    identifier_assignments: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for candidate in materialized:
        if not isinstance(candidate, ManifestRecordV2):
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "MANIFEST_RECORD_INVALID",
                "Manifest contains a value that is not ManifestRecordV2.",
            )
            continue
        relative_path = _safe_relative_path(candidate.relative_path)
        if relative_path is None:
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "IMAGE_PATH_INVALID",
                "Image path is not a normalized relative path within the dataset root.",
            )
            continue
        paths[relative_path].append(candidate)
        identifiers = (
            ("source_image_id", candidate.source_image_id),
            ("patient_id", candidate.patient_id),
            ("intervention_pair_id", candidate.intervention_pair_id),
        )
        for identifier_type, identifier in identifiers:
            if identifier is not None:
                identifier_assignments[(identifier_type, identifier)][candidate.split].add(
                    relative_path
                )

        resolved = _resolve_under_root(root, relative_path)
        if resolved is None:
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "IMAGE_PATH_ESCAPE",
                "Image path resolves outside the dataset root.",
                relative_path=relative_path,
            )
            _audit_labels(
                candidate.labels,
                dataset_id=candidate.dataset_id,
                root=root,
                image_dimensions=(candidate.width, candidate.height),
                issues=issues,
                counters=counters,
            )
            continue
        if not resolved.is_file():
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_FILE_MISSING",
                "Manifest image file is missing.",
                relative_path=relative_path,
            )
            _audit_labels(
                candidate.labels,
                dataset_id=candidate.dataset_id,
                root=root,
                image_dimensions=(candidate.width, candidate.height),
                issues=issues,
                counters=counters,
            )
            continue
        counters.image_files_checked += 1
        try:
            fingerprint = _fingerprint_file(resolved)
        except OSError as error:
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_FILE_UNREADABLE",
                "Image file could not be read.",
                relative_path=relative_path,
                details=_detail(error_type=type(error).__name__),
            )
            _audit_labels(
                candidate.labels,
                dataset_id=candidate.dataset_id,
                root=root,
                image_dimensions=(candidate.width, candidate.height),
                issues=issues,
                counters=counters,
            )
            continue
        actual_size = fingerprint.byte_size
        actual_hash = fingerprint.sha256
        change_reported = False
        if not fingerprint.stable:
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "IMAGE_CHANGED_DURING_AUDIT",
                "Image file identity or metadata changed while it was being hashed.",
                relative_path=relative_path,
            )
            change_reported = True
        actual_hash_paths[actual_hash].add(relative_path)
        basename_hash_paths[PurePosixPath(relative_path).name.casefold()][actual_hash].add(
            relative_path
        )
        if actual_size != candidate.byte_size:
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_SIZE_MISMATCH",
                "Image byte size does not match the manifest.",
                relative_path=relative_path,
                details=_detail(expected=candidate.byte_size, actual=actual_size),
            )
        if actual_hash != candidate.sha256:
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "IMAGE_HASH_MISMATCH",
                "Image SHA-256 does not match the manifest.",
                relative_path=relative_path,
            )
        try:
            decoded = _decode_image(resolved, collect_colors=False)
        except (
            MemoryError,
            OSError,
            SyntaxError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_DECODE_FAILED",
                "Image could not be decoded completely by Pillow.",
                relative_path=relative_path,
                details=_detail(error_type=type(error).__name__),
            )
            _audit_labels(
                candidate.labels,
                dataset_id=candidate.dataset_id,
                root=root,
                image_dimensions=(candidate.width, candidate.height),
                issues=issues,
                counters=counters,
            )
            continue
        try:
            decoded_fingerprint = _fingerprint_file(resolved)
        except OSError as error:
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "IMAGE_CHANGED_DURING_AUDIT",
                "Image file could not be revalidated after decoding.",
                relative_path=relative_path,
                details=_detail(error_type=type(error).__name__),
            )
        else:
            if decoded_fingerprint != fingerprint and not change_reported:
                _append_issue(
                    issues,
                    IssueSeverity.CRITICAL,
                    "IMAGE_CHANGED_DURING_AUDIT",
                    "Image bytes or file identity changed between hashing and decoding.",
                    relative_path=relative_path,
                )
        if (decoded.width, decoded.height) != (candidate.width, candidate.height):
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_DIMENSION_MISMATCH",
                "Decoded image dimensions do not match the manifest.",
                relative_path=relative_path,
                details=_detail(
                    expected=f"{candidate.width}x{candidate.height}",
                    actual=f"{decoded.width}x{decoded.height}",
                ),
            )
        if decoded.channels != candidate.channels:
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_CHANNEL_MISMATCH",
                "Decoded image channels do not match the manifest.",
                relative_path=relative_path,
                details=_detail(expected=candidate.channels, actual=decoded.channels),
            )
        if decoded.channels != 3:
            _append_issue(
                issues,
                IssueSeverity.WARNING,
                "IMAGE_CHANNEL_UNEXPECTED",
                "Fundus image is not three-channel RGB and requires manual review.",
                relative_path=relative_path,
                details=_detail(actual=decoded.channels),
            )
        if _normalized_format(decoded.image_format) != _normalized_format(candidate.image_format):
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_FORMAT_MISMATCH",
                "Decoded image format does not match the manifest.",
                relative_path=relative_path,
                details=_detail(
                    expected=_normalized_format(candidate.image_format),
                    actual=_normalized_format(decoded.image_format),
                ),
            )
        if not candidate.labels:
            counters.unknown_label_count += 1
            _append_issue(
                issues,
                IssueSeverity.WARNING,
                "LABEL_SET_UNDECLARED",
                "No label records are declared; label state remains UNKNOWN.",
                relative_path=relative_path,
            )
        _audit_labels(
            candidate.labels,
            dataset_id=candidate.dataset_id,
            root=root,
            image_dimensions=(decoded.width, decoded.height),
            issues=issues,
            counters=counters,
        )

    for relative_path, duplicate_records in paths.items():
        if len(duplicate_records) > 1:
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "RELATIVE_PATH_DUPLICATE",
                "Manifest contains the same relative image path more than once.",
                relative_path=relative_path,
                details=_detail(record_count=len(duplicate_records)),
            )
    for duplicate_paths in actual_hash_paths.values():
        if len(duplicate_paths) > 1:
            ordered_paths = tuple(sorted(duplicate_paths))
            _append_issue(
                issues,
                IssueSeverity.WARNING,
                "IMAGE_CONTENT_DUPLICATE",
                "Distinct relative paths contain identical image bytes.",
                relative_path=ordered_paths[0],
                related_paths=ordered_paths[1:],
            )
    for basename, hashes in basename_hash_paths.items():
        if len(hashes) > 1:
            conflicting_paths = sorted(path for group in hashes.values() for path in group)
            _append_issue(
                issues,
                IssueSeverity.ERROR,
                "IMAGE_BASENAME_CONFLICT",
                "The same image basename refers to different file contents.",
                relative_path=conflicting_paths[0],
                related_paths=conflicting_paths[1:],
                details=_detail(basename=basename),
            )
    for (identifier_type, identifier), split_paths in identifier_assignments.items():
        if len(split_paths) > 1:
            involved_paths = sorted(
                path for assigned_paths in split_paths.values() for path in assigned_paths
            )
            _append_issue(
                issues,
                IssueSeverity.CRITICAL,
                "SPLIT_LEAKAGE",
                "A protected identifier occurs in more than one split.",
                relative_path=involved_paths[0] if involved_paths else None,
                related_paths=involved_paths[1:],
                details=_detail(
                    identifier_type=identifier_type,
                    identifier_digest=_identifier_digest(identifier),
                    splits=",".join(sorted(split_paths)),
                ),
            )
    return _make_result(
        record_count=len(materialized),
        issues=issues,
        counters=counters,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_sha256=manifest_sha256,
    )


def _materialize_records(records: Iterable[object]) -> tuple[ManifestRecordV2, ...] | None:
    materialized = tuple(records)
    if not all(isinstance(record, ManifestRecordV2) for record in materialized):
        return None
    return tuple(record for record in materialized if isinstance(record, ManifestRecordV2))


def _manifest_digest(manifest: DatasetManifestV2) -> str:
    payload = json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_authorization_failure(
    manifest: DatasetManifestV2,
    permit: DatasetAccessPermit,
) -> str | None:
    if manifest.dataset_id != permit.registry.dataset_id:
        return "manifest dataset_id must match access permit"
    if manifest.dataset_version != permit.registry.dataset_version:
        return "manifest dataset_version must exactly match access permit"
    if manifest.dataset_id != permit.spec.dataset_id:
        return "manifest dataset_id must match explicit adapter spec"
    if manifest.dataset_id not in DATASET_LABEL_SEMANTICS:
        return "manifest dataset_id has no registered Stage 2 adapter"
    required_semantics = permit.spec.effective_label_semantics
    if len(manifest.records) != len(permit.spec.entries):
        return "manifest records must exactly match explicit adapter spec entries"
    entries_by_path = {entry.image_path.as_posix(): entry for entry in permit.spec.entries}
    if len(entries_by_path) != len(permit.spec.entries):
        return "adapter spec image paths must be unique"
    identity_fields = (
        "source_image_id",
        "split",
        "patient_id",
        "eye_id",
        "followup_id",
        "intervention_pair_id",
    )
    for record in manifest.records:
        entry = entries_by_path.get(record.relative_path)
        if entry is None:
            return "manifest image path is not authorized by adapter spec"
        if any(getattr(record, name) != getattr(entry, name) for name in identity_fields):
            return "manifest identities must exactly match adapter spec"
        if {label.semantic for label in record.labels} != required_semantics:
            return "manifest label semantics must exactly match the registered adapter"
        expected_labels = {
            label.semantic: (
                None if label.relative_path is None else label.relative_path.as_posix()
            )
            for label in entry.labels
        }
        observed_semantics = {label.semantic for label in record.labels}
        if not expected_labels.keys() <= observed_semantics:
            return "manifest is missing a label declared by the adapter spec"
        for label in record.labels:
            if label.semantic in expected_labels:
                expected_path = expected_labels[label.semantic]
                if expected_path is None and (
                    label.availability is LabelAvailability.UNKNOWN
                    and label.relative_path is None
                    and label.sha256 is None
                    and label.byte_size is None
                ):
                    continue
                if expected_path is None:
                    return "manifest label state does not match adapter spec"
                if label.relative_path != expected_path:
                    return "manifest label path is not authorized by adapter spec"
                if label.availability is not LabelAvailability.AVAILABLE:
                    return "manifest declared label must be AVAILABLE"
                continue
            if not (
                label.availability is LabelAvailability.UNKNOWN
                and label.relative_path is None
                and label.sha256 is None
                and label.byte_size is None
            ):
                return "manifest label semantic is not authorized by adapter spec"
    return None


def audit_manifest(
    manifest: DatasetManifestV2,
    *,
    permit: DatasetAccessPermit,
) -> AuditResult:
    """Audit real records only through a purpose-scoped access capability."""

    if not isinstance(permit, DatasetAccessPermit):
        raise TypeError("permit must be a DatasetAccessPermit")
    dataset_id = permit.registry.dataset_id
    permit_failures = permit_validation_reasons(permit)
    if permit_failures:
        return blocked_audit_result(permit_failures, dataset_id=dataset_id)
    if permit.purpose is not DataUsePurpose.READ_ONLY_AUDIT:
        return blocked_audit_result(
            ("access permit purpose must be read_only_audit",),
            dataset_id=dataset_id,
        )
    manifest_object: object = manifest
    if not isinstance(manifest_object, DatasetManifestV2):
        return blocked_audit_result(
            ("audit requires a DatasetManifestV2 envelope",),
            dataset_id=dataset_id,
        )
    if not permit.root.is_absolute():
        return blocked_audit_result(
            ("access permit root must be absolute",),
            dataset_id=dataset_id,
        )
    authorization_failure = _manifest_authorization_failure(manifest, permit)
    if authorization_failure is not None:
        return blocked_audit_result(
            (authorization_failure,),
            dataset_id=dataset_id,
        )
    try:
        _path_free_text(manifest.dataset_version, "dataset_version")
    except ValueError:
        return blocked_audit_result(
            ("manifest dataset_version is not report-safe",),
            dataset_id=dataset_id,
        )
    return _audit_records_at_root(
        manifest.records,
        dataset_root=permit.root,
        dataset_version=manifest.dataset_version,
        manifest_sha256=_manifest_digest(manifest),
    )


def _audit_synthetic_fixture(
    records: Iterable[ManifestRecordV2],
    *,
    fixture_root: Path,
) -> AuditResult:
    """Audit an explicitly synthetic fixture without a license permit.

    This narrow test-only boundary refuses ordinary dataset identifiers before
    resolving or reading ``fixture_root``.
    """

    manifest = records if isinstance(records, DatasetManifestV2) else None
    materialized = _materialize_records(records)
    if materialized is None:
        return blocked_audit_result(("synthetic fixture manifest is invalid",))
    if not materialized or any(
        not record.dataset_id.startswith(("synthetic_", "toy_")) for record in materialized
    ):
        return blocked_audit_result(
            ("synthetic fixture dataset_id must use toy_ or synthetic_ prefix",)
        )
    return _audit_records_at_root(
        materialized,
        dataset_root=fixture_root,
        dataset_version=None if manifest is None else manifest.dataset_version,
        manifest_sha256=None if manifest is None else _manifest_digest(manifest),
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditIssue",
    "AuditResult",
    "AuditStatus",
    "AuditSummary",
    "IssueSeverity",
    "audit_manifest",
    "blocked_audit_result",
]
