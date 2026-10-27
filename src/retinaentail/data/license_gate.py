"""Fail-closed capability gate that runs before any dataset-root access."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

from retinaentail.data.stage2_models import DatasetAdapterSpec

_PERMIT_TOKEN: Final = object()
_LICENSE_STATUSES: Final = frozenset({"unknown", "ambiguous", "verified", "restricted"})
_REVIEW_STATUSES: Final = frozenset({"pending_review", "reviewed"})
_LABEL_STATUSES: Final = frozenset({"pending_review", "verified"})
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_RFC3339_PATTERN: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_LICENSE_REVIEW_SCHEMA_VERSION: Final = "1.0.0"


class DataUsePurpose(StrEnum):
    """Controlled purposes that a human license review may authorize."""

    READ_ONLY_AUDIT = "read_only_audit"
    METADATA_VALIDATION = "metadata_validation"
    LESION_CALIBRATION = "lesion_calibration"
    LOCKED_TEST_EVALUATION = "locked_test_evaluation"


class LicenseReviewDecision(StrEnum):
    """Explicit human decision recorded independently from registry metadata."""

    APPROVED = "approved"
    DENIED = "denied"


class LicenseGateError(PermissionError):
    """Raised when registry evidence is insufficient for dataset-root access."""

    def __init__(self, dataset_id: str, reasons: tuple[str, ...]) -> None:
        self.dataset_id = dataset_id
        self.reasons = reasons
        super().__init__(f"dataset access denied for {dataset_id}: {'; '.join(reasons)}")


class LicenseMetadataError(ValueError):
    """Raised when license-gate registry metadata is malformed."""


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LicenseMetadataError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: object, name: str) -> str:
    digest = _nonempty(value, name)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise LicenseMetadataError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _dataset_id(value: object) -> str:
    dataset_id = _nonempty(value, "dataset_id")
    if _DATASET_ID_PATTERN.fullmatch(dataset_id) is None:
        raise LicenseMetadataError("dataset_id must match ^[a-z][a-z0-9_]*$")
    return dataset_id


def _dataset_version(value: object) -> str:
    version = _nonempty(value, "dataset_version")
    if _DATASET_VERSION_PATTERN.fullmatch(version) is None:
        raise LicenseMetadataError("dataset_version must be a report-safe release identifier")
    return version


def _purpose(value: object) -> DataUsePurpose:
    if not isinstance(value, str):
        raise LicenseMetadataError(f"invalid data-use purpose: {value!r}")
    try:
        return value if isinstance(value, DataUsePurpose) else DataUsePurpose(value)
    except ValueError as error:
        raise LicenseMetadataError(f"invalid data-use purpose: {value!r}") from error


def _review_decision(value: object) -> LicenseReviewDecision:
    if not isinstance(value, str):
        raise LicenseMetadataError(f"invalid review decision: {value!r}")
    try:
        return value if isinstance(value, LicenseReviewDecision) else LicenseReviewDecision(value)
    except ValueError as error:
        raise LicenseMetadataError(f"invalid review decision: {value!r}") from error


def _string_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LicenseMetadataError(f"{name} must contain a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise LicenseMetadataError(f"{name} keys must be strings")
        result[key] = item
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RegistryLicenseMetadata:
    """Immutable snapshot of registry fields used by the access decision."""

    schema_version: str
    dataset_id: str
    dataset_version: str
    license_status: str
    license_review_status: str
    label_verification_status: str

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise LicenseMetadataError("unsupported registry schema_version")
        for field_name in ("license_status", "license_review_status", "label_verification_status"):
            object.__setattr__(self, field_name, _nonempty(getattr(self, field_name), field_name))
        object.__setattr__(self, "dataset_id", _dataset_id(self.dataset_id))
        object.__setattr__(self, "dataset_version", _dataset_version(self.dataset_version))
        if self.license_status not in _LICENSE_STATUSES:
            raise LicenseMetadataError("invalid license_status")
        if self.license_review_status not in _REVIEW_STATUSES:
            raise LicenseMetadataError("invalid license_review_status")
        if self.label_verification_status not in _LABEL_STATUSES:
            raise LicenseMetadataError("invalid label_verification_status")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RegistryLicenseMetadata:
        required = {
            "schema_version",
            "dataset_id",
            "version",
            "license_status",
            "license_review_status",
            "label_verification_status",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise LicenseMetadataError(
                f"registry metadata is missing field(s): {', '.join(missing)}"
            )
        return cls(
            schema_version=_nonempty(value["schema_version"], "schema_version"),
            dataset_id=_dataset_id(value["dataset_id"]),
            dataset_version=_dataset_version(value["version"]),
            license_status=_nonempty(value["license_status"], "license_status"),
            license_review_status=_nonempty(
                value["license_review_status"], "license_review_status"
            ),
            label_verification_status=_nonempty(
                value["label_verification_status"], "label_verification_status"
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LicenseReviewRecord:
    """Versioned human review evidence required in addition to registry flags."""

    dataset_id: str
    dataset_version: str
    decision: LicenseReviewDecision
    permitted_purposes: tuple[DataUsePurpose, ...]
    terms_sha256: str
    evidence_sha256: str
    reviewer_id: str
    reviewed_at: str
    schema_version: str = _LICENSE_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _LICENSE_REVIEW_SCHEMA_VERSION:
            raise LicenseMetadataError("unsupported license review schema_version")
        object.__setattr__(self, "dataset_id", _dataset_id(self.dataset_id))
        object.__setattr__(self, "dataset_version", _dataset_version(self.dataset_version))
        decision = _review_decision(self.decision)
        object.__setattr__(self, "decision", decision)
        purposes: list[DataUsePurpose] = []
        seen: set[DataUsePurpose] = set()
        for raw_purpose in self.permitted_purposes:
            purpose = _purpose(raw_purpose)
            if purpose in seen:
                raise LicenseMetadataError(f"duplicate permitted purpose: {purpose.value}")
            seen.add(purpose)
            purposes.append(purpose)
        if decision is LicenseReviewDecision.APPROVED and not purposes:
            raise LicenseMetadataError("an approved review must permit at least one purpose")
        object.__setattr__(self, "permitted_purposes", tuple(purposes))
        object.__setattr__(self, "terms_sha256", _sha256(self.terms_sha256, "terms_sha256"))
        object.__setattr__(
            self,
            "evidence_sha256",
            _sha256(self.evidence_sha256, "evidence_sha256"),
        )
        object.__setattr__(self, "reviewer_id", _nonempty(self.reviewer_id, "reviewer_id"))
        reviewed_at = _nonempty(self.reviewed_at, "reviewed_at")
        if _RFC3339_PATTERN.fullmatch(reviewed_at) is None:
            raise LicenseMetadataError("reviewed_at must be an RFC 3339 timestamp")
        try:
            parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise LicenseMetadataError("reviewed_at must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise LicenseMetadataError("reviewed_at must include a timezone offset")
        object.__setattr__(self, "reviewed_at", reviewed_at)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "decision": self.decision.value,
            "permitted_purposes": [purpose.value for purpose in self.permitted_purposes],
            "terms_sha256": self.terms_sha256,
            "evidence_sha256": self.evidence_sha256,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> LicenseReviewRecord:
        required = {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "decision",
            "permitted_purposes",
            "terms_sha256",
            "evidence_sha256",
            "reviewer_id",
            "reviewed_at",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing:
            raise LicenseMetadataError(f"license review is missing field(s): {', '.join(missing)}")
        if unknown:
            raise LicenseMetadataError(f"license review has unknown field(s): {', '.join(unknown)}")
        raw_purposes = value["permitted_purposes"]
        if isinstance(raw_purposes, str | bytes) or not isinstance(raw_purposes, Sequence):
            raise LicenseMetadataError("permitted_purposes must be a sequence")
        return cls(
            schema_version=_nonempty(value["schema_version"], "schema_version"),
            dataset_id=_dataset_id(value["dataset_id"]),
            dataset_version=_dataset_version(value["dataset_version"]),
            decision=_review_decision(value["decision"]),
            permitted_purposes=tuple(_purpose(item) for item in raw_purposes),
            terms_sha256=_sha256(value["terms_sha256"], "terms_sha256"),
            evidence_sha256=_sha256(value["evidence_sha256"], "evidence_sha256"),
            reviewer_id=_nonempty(value["reviewer_id"], "reviewer_id"),
            reviewed_at=_nonempty(value["reviewed_at"], "reviewed_at"),
        )


def load_registry_license_metadata(path: Path) -> RegistryLicenseMetadata:
    """Load gate metadata from a registry file without consulting its dataset root."""

    if not isinstance(path, Path):
        raise TypeError("registry path must be a pathlib.Path")
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise LicenseMetadataError("registry YAML must contain a mapping")
    normalized: dict[str, object] = {}
    for key, item in decoded.items():
        if not isinstance(key, str):
            raise LicenseMetadataError("registry YAML keys must be strings")
        normalized[key] = item
    return RegistryLicenseMetadata.from_mapping(normalized)


def load_license_review_record(path: Path) -> LicenseReviewRecord:
    """Load a strict review record; this file is evidence, never an override flag."""

    if not isinstance(path, Path):
        raise TypeError("review path must be a pathlib.Path")
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LicenseReviewRecord.from_mapping(_string_mapping(decoded, "license review"))


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetAccessPermit:
    """Process-bound capability issued only by the license gate."""

    registry: RegistryLicenseMetadata
    review: LicenseReviewRecord
    purpose: DataUsePurpose
    root: Path
    spec: DatasetAdapterSpec
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _PERMIT_TOKEN:
            raise TypeError("DatasetAccessPermit can only be issued by authorize_dataset_access")
        if not self.root.is_absolute():
            raise ValueError("DatasetAccessPermit root must be absolute")


_ISSUED_PERMITS: dict[int, tuple[DatasetAccessPermit, tuple[str, ...]]] = {}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _registry_snapshot(registry: RegistryLicenseMetadata) -> dict[str, str]:
    return {
        "schema_version": registry.schema_version,
        "dataset_id": registry.dataset_id,
        "dataset_version": registry.dataset_version,
        "license_status": registry.license_status,
        "license_review_status": registry.license_review_status,
        "label_verification_status": registry.label_verification_status,
    }


def _permit_fingerprint(
    permit: DatasetAccessPermit,
) -> tuple[str, ...]:
    """Return a value snapshot without retaining mutable nested object aliases."""

    return (
        _canonical_sha256(_registry_snapshot(permit.registry)),
        _canonical_sha256(permit.review.to_dict()),
        permit.purpose.value,
        permit.root.as_posix(),
        _canonical_sha256(permit.spec.to_dict()),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class LicenseGateDecision:
    """Non-throwing, pre-I/O gate assessment suitable for reports and CLI output."""

    dataset_id: str
    purpose: DataUsePurpose
    permitted: bool
    reasons: tuple[str, ...]

    @property
    def status(self) -> str:
        return "PERMITTED" if self.permitted else "BLOCKED"

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "purpose": self.purpose.value,
            "status": self.status,
            "permitted": self.permitted,
            "reasons": list(self.reasons),
        }


def assess_dataset_access(
    registry: RegistryLicenseMetadata,
    *,
    review: LicenseReviewRecord | None,
    purpose: DataUsePurpose = DataUsePurpose.READ_ONLY_AUDIT,
    root_provided: bool,
    spec_provided: bool,
) -> LicenseGateDecision:
    """Assess only in-memory status flags and explicit-input presence."""

    if not isinstance(registry, RegistryLicenseMetadata):
        raise TypeError("registry must be RegistryLicenseMetadata")
    requested_purpose = _purpose(purpose)
    if review is not None and not isinstance(review, LicenseReviewRecord):
        raise TypeError("review must be a LicenseReviewRecord or None")
    if type(root_provided) is not bool or type(spec_provided) is not bool:  # noqa: E721
        raise TypeError("root_provided and spec_provided must be bool values")
    reasons: list[str] = []
    if registry.license_status != "verified":
        reasons.append("license_status must be verified")
    if registry.license_review_status != "reviewed":
        reasons.append("license_review_status must be reviewed")
    if registry.label_verification_status != "verified":
        reasons.append("label_verification_status must be verified")
    if review is None:
        reasons.append("an explicit LicenseReviewRecord is required")
    else:
        if review.dataset_id != registry.dataset_id:
            reasons.append("review dataset_id must match registry dataset_id")
        if review.dataset_version != registry.dataset_version:
            reasons.append("review dataset_version must exactly match registry version")
        if review.decision is not LicenseReviewDecision.APPROVED:
            reasons.append("license review decision must be approved")
        if requested_purpose not in review.permitted_purposes:
            reasons.append(f"license review must permit purpose {requested_purpose.value}")
    if not root_provided:
        reasons.append("caller must explicitly provide a pathlib.Path root")
    if not spec_provided:
        reasons.append("caller must explicitly provide a DatasetAdapterSpec")
    return LicenseGateDecision(
        dataset_id=registry.dataset_id,
        purpose=requested_purpose,
        permitted=not reasons,
        reasons=tuple(reasons),
    )


def permit_validation_reasons(value: object) -> tuple[str, ...]:
    """Revalidate issuance identity and every in-memory Gate binding at consumption."""

    if not isinstance(value, DatasetAccessPermit):
        return ("access capability has an invalid type",)
    issued = _ISSUED_PERMITS.get(id(value))
    if issued is None or issued[0] is not value:
        return ("access capability was not issued by the license gate",)
    try:
        if value._token is not _PERMIT_TOKEN:
            return ("access capability issuance identity is invalid",)
        if issued[1] != _permit_fingerprint(value):
            return ("access capability bindings changed after issuance",)
        decision = assess_dataset_access(
            value.registry,
            review=value.review,
            purpose=value.purpose,
            root_provided=isinstance(value.root, Path),
            spec_provided=isinstance(value.spec, DatasetAdapterSpec),
        )
        reasons = list(decision.reasons)
        if value.spec.dataset_id != value.registry.dataset_id:
            reasons.append("spec dataset_id must match registry dataset_id")
        if not value.root.is_absolute():
            reasons.append("access capability root must be absolute")
    except (AttributeError, TypeError, ValueError):
        return ("access capability fields are invalid",)
    return tuple(reasons)


def authorize_dataset_access(
    registry: RegistryLicenseMetadata,
    *,
    review: LicenseReviewRecord | None,
    purpose: DataUsePurpose = DataUsePurpose.READ_ONLY_AUDIT,
    root: Path | None,
    spec: DatasetAdapterSpec | None,
) -> DatasetAccessPermit:
    """Issue a capability without stat/glob/resolve/read access to ``root``.

    All three registry decisions, a matching purpose-scoped human review, and both
    explicit caller inputs are required.  A denial inspects only in-memory objects;
    actual filesystem validation is deferred to the capability-gated adapter.
    """

    if not isinstance(registry, RegistryLicenseMetadata):
        raise TypeError("registry must be RegistryLicenseMetadata")
    decision = assess_dataset_access(
        registry,
        review=review,
        purpose=purpose,
        root_provided=isinstance(root, Path),
        spec_provided=isinstance(spec, DatasetAdapterSpec),
    )
    reasons = list(decision.reasons)
    if isinstance(spec, DatasetAdapterSpec) and spec.dataset_id != registry.dataset_id:
        reasons.append("spec dataset_id must match registry dataset_id")
    if reasons:
        raise LicenseGateError(registry.dataset_id, tuple(reasons))
    if root is None or spec is None or review is None:
        raise AssertionError("unreachable gate state")
    permit = DatasetAccessPermit(
        registry=registry,
        review=review,
        purpose=_purpose(purpose),
        root=root.absolute(),
        spec=spec,
        _token=_PERMIT_TOKEN,
    )
    _ISSUED_PERMITS[id(permit)] = (permit, _permit_fingerprint(permit))
    return permit


__all__ = [
    "DatasetAccessPermit",
    "DataUsePurpose",
    "LicenseGateDecision",
    "LicenseGateError",
    "LicenseMetadataError",
    "LicenseReviewDecision",
    "LicenseReviewRecord",
    "RegistryLicenseMetadata",
    "assess_dataset_access",
    "authorize_dataset_access",
    "load_registry_license_metadata",
    "load_license_review_record",
    "permit_validation_reasons",
]
