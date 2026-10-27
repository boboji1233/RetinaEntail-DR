"""Full-package Stage 3 verification and process-bound calibration capability.

Detached summaries are deliberately insufficient.  The synthetic issuer scans the
exact package, rejects links/reparse points and unexpected entries, computes a
content-addressed file manifest, invokes the existing full regeneration/replay
verifier, scans again, and only then issues a short-lived capability.  Capability
validation is in-memory; :func:`replay_stage3_package` repeats complete package
verification at the point of use.

The APPROVED_REAL_DATA branch accepts only a process-bound receipt from the strict
four-file envelope verifier.  That verifier replays a full Stage 2 capability,
revalidates a frozen data card, and recomputes thresholds from calibration samples;
a fixture-marked envelope can be verified in tests but cannot issue real authority.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Never, SupportsIndex, cast

from retinaentail.calibration.data_card import (
    DatasetCardScope,
    FrozenDatasetCard,
    frozen_data_card_validation_reasons,
)
from retinaentail.calibration.protocol import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationArtifact,
    CalibrationLabel,
    CalibrationRunStatus,
    CalibrationSample,
    SampleSplit,
    calibrate_order_statistic,
)
from retinaentail.calibration.toy_fixture import Stage3ToyRunResult, verify_stage3_toy_package
from retinaentail.data.stage2_models import DatasetManifestV2, ManifestRecordV2

STAGE3_PACKAGE_CAPABILITY_SCHEMA_VERSION: Final = "6.0.0"
STAGE3_PACKAGE_MANIFEST_SCHEMA_VERSION: Final = "1.0.0"
STAGE3_REAL_PACKAGE_SCHEMA_VERSION: Final = "1.0.0"
STAGE3_PACKAGE_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"
DEFAULT_STAGE3_CAPABILITY_TTL_SECONDS: Final = 300
MAX_STAGE3_CAPABILITY_TTL_SECONDS: Final = 900
SYNTHETIC_STAGE3_DATASET_ID: Final = "retinaentail_stage3_toy"
SYNTHETIC_STAGE3_DATASET_VERSION: Final = "stage3-toy-3.0.0"

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_READ_CHUNK_BYTES: Final = 1024 * 1024
_REPARSE_POINT_ATTRIBUTE: Final = 0x0400
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_CAPABILITY_TOKEN: Final = object()
_PROCESS_NONCE: Final = object()
_PROCESS_ID: int = os.getpid()
_monotonic = time.monotonic
_CAPABILITY_LOCK: Final = threading.RLock()

_STAGE3_TOY_FILES: Final = frozenset(
    {
        "calibration_artifact.json",
        "calibration_samples.json",
        "certificate.json",
        "config_snapshot.json",
        "dual_replay.json",
        "evaluation.json",
        "fail_closed/missing_threshold.json",
        "fail_closed/unknown_degradation.json",
        "images/lesion_absent.png",
        "images/lesion_present.png",
        "images/observation.png",
        "model_provenance.json",
        "summary.json",
        "witnesses/negative.json",
        "witnesses/positive.json",
    }
)
_STAGE3_TOY_DIRECTORIES: Final = frozenset(
    parent.as_posix()
    for name in _STAGE3_TOY_FILES
    for parent in Path(name).parents
    if parent.as_posix() != "."
)
_STAGE3_REAL_FILES: Final = frozenset(
    {
        "calibration_artifact.json",
        "calibration_samples.json",
        "frozen_data_card.json",
        "package.json",
    }
)
_STAGE3_REAL_DIRECTORIES: Final = frozenset[str]()


class Stage3PackageScope(StrEnum):
    SYNTHETIC_TOY = "SYNTHETIC_TOY"
    APPROVED_REAL_DATA = "APPROVED_REAL_DATA"


class Stage3PackageCapabilityError(PermissionError):
    """Raised when a complete Stage 3 capability cannot be issued or consumed."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Stage 3 package capability rejected: " + "; ".join(reasons))


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3PackageFileEntry:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path(self.relative_path)
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= _MAX_ARTIFACT_BYTES:
            raise ValueError("Stage 3 package entry size is invalid")
        _require_sha256(self.sha256, "entry sha256")

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3PackageManifest:
    files: tuple[Stage3PackageFileEntry, ...]
    package_sha256: str
    digest_method: str = STAGE3_PACKAGE_DIGEST_METHOD
    schema_version: str = STAGE3_PACKAGE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_PACKAGE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 3 package manifest schema_version")
        if self.digest_method != STAGE3_PACKAGE_DIGEST_METHOD:
            raise ValueError("unsupported Stage 3 package digest method")
        if not self.files or any(
            not isinstance(item, Stage3PackageFileEntry) for item in self.files
        ):
            raise ValueError("Stage 3 package manifest requires typed file entries")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Stage 3 package manifest paths must be unique and sorted")
        _require_sha256(self.package_sha256, "package_sha256")
        if self.package_sha256 != _manifest_digest(self.files):
            raise ValueError("Stage 3 package digest does not match its file manifest")

    @property
    def file_count(self) -> int:
        return len(self.files)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "digest_method": self.digest_method,
            "file_count": self.file_count,
            "package_sha256": self.package_sha256,
            "files": [item.as_dict() for item in self.files],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3PackageDescriptor:
    """Immutable public descriptor; it is evidence metadata, never authority."""

    scope: Stage3PackageScope
    dataset_id: str
    dataset_version: str
    calibration_schema_version: str
    calibration_method_version: str
    protocol_scope: str
    config_hash: str
    data_hash: str
    model_hash: str
    calibration_artifact_hash: str
    positive_sample_count: int
    negative_sample_count: int
    total_sample_count: int
    alpha: float
    positive_threshold: float
    negative_threshold: float
    package_sha256: str
    package_file_count: int
    stage2_package_sha256: str | None = None
    frozen_data_card_sha256: str | None = None
    fixture_only: bool = False
    package_digest_method: str = STAGE3_PACKAGE_DIGEST_METHOD
    schema_version: str = STAGE3_PACKAGE_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_PACKAGE_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 3 capability snapshot schema_version")
        if not isinstance(self.scope, Stage3PackageScope):
            raise TypeError("Stage 3 package scope must use Stage3PackageScope")
        _require_dataset_id(self.dataset_id)
        _require_dataset_version(self.dataset_version)
        for name in (
            "calibration_schema_version",
            "calibration_method_version",
            "protocol_scope",
        ):
            _require_nonempty(getattr(self, name), name)
        for name in (
            "config_hash",
            "data_hash",
            "model_hash",
            "calibration_artifact_hash",
            "package_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        counts = (
            self.positive_sample_count,
            self.negative_sample_count,
            self.total_sample_count,
        )
        if (
            any(type(value) is not int or value < 1 for value in counts)
            or self.total_sample_count != self.positive_sample_count + self.negative_sample_count
        ):
            raise ValueError("Stage 3 calibration sample counts are inconsistent")
        for name in ("alpha", "positive_threshold", "negative_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.positive_threshold < 0.0 or self.negative_threshold < 0.0:
            raise ValueError("calibration thresholds must be non-negative")
        if type(self.package_file_count) is not int or self.package_file_count < 1:
            raise ValueError("package_file_count must be a positive integer")
        if self.package_digest_method != STAGE3_PACKAGE_DIGEST_METHOD:
            raise ValueError("unsupported Stage 3 package digest method")
        if type(self.fixture_only) is not bool:
            raise TypeError("fixture_only must be an exact boolean")
        if self.scope is Stage3PackageScope.SYNTHETIC_TOY and (
            self.dataset_id != SYNTHETIC_STAGE3_DATASET_ID
            or self.dataset_version != SYNTHETIC_STAGE3_DATASET_VERSION
            or self.protocol_scope != "TOY_PROTOCOL"
            or not self.fixture_only
            or self.stage2_package_sha256 is not None
            or self.frozen_data_card_sha256 is not None
        ):
            raise ValueError("synthetic Stage 3 descriptor identity/scope is invalid")
        if self.scope is Stage3PackageScope.APPROVED_REAL_DATA:
            if self.stage2_package_sha256 is None or self.frozen_data_card_sha256 is None:
                raise ValueError("real Stage 3 descriptor requires Stage 2/card bindings")
            _require_sha256(self.stage2_package_sha256, "stage2_package_sha256")
            _require_sha256(self.frozen_data_card_sha256, "frozen_data_card_sha256")

    def as_dict(self) -> dict[str, object]:
        """Return report metadata.  This mapping is intentionally not a capability."""

        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "calibration_schema_version": self.calibration_schema_version,
            "calibration_method_version": self.calibration_method_version,
            "protocol_scope": self.protocol_scope,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "calibration_artifact_hash": self.calibration_artifact_hash,
            "positive_sample_count": self.positive_sample_count,
            "negative_sample_count": self.negative_sample_count,
            "total_sample_count": self.total_sample_count,
            "alpha": self.alpha,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "package_sha256": self.package_sha256,
            "package_file_count": self.package_file_count,
            "stage2_package_sha256": self.stage2_package_sha256,
            "frozen_data_card_sha256": self.frozen_data_card_sha256,
            "fixture_only": self.fixture_only,
            "package_digest_method": self.package_digest_method,
        }

    def binding_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.as_dict()))


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedCompleteRealStage3Package:
    """Process-bound receipt issued only after full real-envelope recomputation."""

    descriptor: Stage3PackageDescriptor
    manifest: Stage3PackageManifest
    package_root: Path = field(repr=False, compare=False)
    package_root_identity: tuple[int, int] = field(repr=False)
    ttl_seconds: int
    issued_monotonic: float = field(repr=False)
    expires_monotonic: float = field(repr=False)
    issued_process_id: int = field(repr=False)
    _stage2_capability: object = field(repr=False, compare=False)
    _frozen_data_card: FrozenDatasetCard = field(repr=False, compare=False)
    _process_nonce: object = field(repr=False, compare=False)
    _issuance_nonce: object = field(repr=False, compare=False)
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        if _token is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedCompleteRealStage3Package can only be issued by "
                "verify_complete_real_stage3_package"
            )
        if (
            not isinstance(self.descriptor, Stage3PackageDescriptor)
            or self.descriptor.scope is not Stage3PackageScope.APPROVED_REAL_DATA
        ):
            raise TypeError("complete real-package descriptor is invalid")
        self.descriptor.__post_init__()
        if not isinstance(self.manifest, Stage3PackageManifest):
            raise TypeError("complete real-package manifest is invalid")
        self.manifest.__post_init__()
        if (
            self.descriptor.package_sha256 != self.manifest.package_sha256
            or self.descriptor.package_file_count != self.manifest.file_count
        ):
            raise ValueError("complete real-package descriptor/manifest mismatch")
        if not isinstance(self.package_root, Path) or not self.package_root.is_absolute():
            raise ValueError("complete real-package root must be absolute")
        if (
            not isinstance(self.package_root_identity, tuple)
            or len(self.package_root_identity) != 2
            or any(type(item) is not int or item < 0 for item in self.package_root_identity)
        ):
            raise ValueError("complete real-package root identity is invalid")
        _validate_ttl(self.ttl_seconds)
        if (
            isinstance(self.issued_monotonic, bool)
            or not isinstance(self.issued_monotonic, int | float)
            or not math.isfinite(self.issued_monotonic)
            or self.issued_monotonic < 0.0
            or isinstance(self.expires_monotonic, bool)
            or not isinstance(self.expires_monotonic, int | float)
            or not math.isfinite(self.expires_monotonic)
            or self.expires_monotonic != self.issued_monotonic + self.ttl_seconds
        ):
            raise ValueError("complete real-package TTL binding is invalid")
        if type(self.issued_process_id) is not int or self.issued_process_id < 1:
            raise ValueError("complete real-package process id is invalid")
        if not isinstance(self._frozen_data_card, FrozenDatasetCard):
            raise TypeError("complete real-package frozen card is invalid")
        if self._process_nonce is not _PROCESS_NONCE or self._issuance_nonce is None:
            raise TypeError("complete real-package process/issuance identity is invalid")

    def __copy__(self) -> Never:
        raise TypeError("complete Stage 3 package receipts cannot be copied")

    def __deepcopy__(self, _memo: object) -> Never:
        raise TypeError("complete Stage 3 package receipts cannot be deep-copied")

    def __reduce__(self) -> Never:
        raise TypeError("complete Stage 3 package receipts cannot be serialized")

    def __reduce_ex__(self, _protocol: SupportsIndex) -> Never:
        raise TypeError("complete Stage 3 package receipts cannot be serialized")

    def __getstate__(self) -> Never:
        raise TypeError("complete Stage 3 package receipts cannot be serialized")

    @property
    def authorizes_dataset_root_access(self) -> bool:
        """Evidence-package verification never grants dataset-root I/O."""

        return False


CompleteRealStage3PackageVerifier = VerifiedCompleteRealStage3Package


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedStage3PackageCapability:
    """Non-serializable, process-bound receipt for a fully replayed Stage 3 package."""

    descriptor: Stage3PackageDescriptor
    manifest: Stage3PackageManifest
    package_root: Path = field(repr=False, compare=False)
    package_root_identity: tuple[int, int] = field(repr=False)
    ttl_seconds: int
    issued_monotonic: float = field(repr=False)
    expires_monotonic: float = field(repr=False)
    issued_process_id: int = field(repr=False)
    _process_nonce: object = field(repr=False, compare=False)
    _issuance_nonce: object = field(repr=False, compare=False)
    _real_package_receipt: VerifiedCompleteRealStage3Package | None = field(
        default=None, repr=False, compare=False
    )
    _token: InitVar[object] = None

    def __post_init__(self, _token: object = None) -> None:
        if _token is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedStage3PackageCapability can only be issued by "
                "verify_and_issue_stage3_package"
            )
        if not isinstance(self.descriptor, Stage3PackageDescriptor):
            raise TypeError("capability descriptor has an invalid type")
        self.descriptor.__post_init__()
        if not isinstance(self.manifest, Stage3PackageManifest):
            raise TypeError("capability manifest has an invalid type")
        self.manifest.__post_init__()
        if (
            self.descriptor.package_sha256 != self.manifest.package_sha256
            or self.descriptor.package_file_count != self.manifest.file_count
        ):
            raise ValueError("capability snapshot does not match its package manifest")
        if not isinstance(self.package_root, Path) or not self.package_root.is_absolute():
            raise ValueError("capability package_root must be an absolute pathlib.Path")
        if (
            not isinstance(self.package_root_identity, tuple)
            or len(self.package_root_identity) != 2
            or any(type(item) is not int or item < 0 for item in self.package_root_identity)
        ):
            raise ValueError("capability package root identity is invalid")
        _validate_ttl(self.ttl_seconds)
        if (
            isinstance(self.issued_monotonic, bool)
            or not isinstance(self.issued_monotonic, int | float)
            or not math.isfinite(self.issued_monotonic)
            or self.issued_monotonic < 0.0
            or isinstance(self.expires_monotonic, bool)
            or not isinstance(self.expires_monotonic, int | float)
            or not math.isfinite(self.expires_monotonic)
            or self.expires_monotonic != self.issued_monotonic + self.ttl_seconds
        ):
            raise ValueError("capability monotonic TTL binding is invalid")
        if type(self.issued_process_id) is not int or self.issued_process_id < 1:
            raise ValueError("capability process id is invalid")
        if self._process_nonce is not _PROCESS_NONCE or self._issuance_nonce is None:
            raise TypeError("capability process/issuance identity is invalid")
        if self.descriptor.scope is Stage3PackageScope.SYNTHETIC_TOY:
            if self._real_package_receipt is not None:
                raise ValueError("synthetic capability cannot carry a real-package receipt")
        elif not isinstance(self._real_package_receipt, VerifiedCompleteRealStage3Package):
            raise ValueError("real capability requires a verified complete-package receipt")

    def __copy__(self) -> Never:
        raise TypeError("Stage 3 package capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> Never:
        raise TypeError("Stage 3 package capabilities cannot be deep-copied")

    def __reduce__(self) -> Never:
        raise TypeError("Stage 3 package capabilities cannot be serialized")

    def __reduce_ex__(self, _protocol: SupportsIndex) -> Never:
        raise TypeError("Stage 3 package capabilities cannot be serialized")

    def __getstate__(self) -> Never:
        raise TypeError("Stage 3 package capabilities cannot be serialized")

    @property
    def authorizes_dataset_root_access(self) -> bool:
        """Calibration-package authority is never dataset-root authority."""

        return False

    @property
    def snapshot(self) -> Stage3PackageDescriptor:
        """Compatibility name; the descriptor is metadata and not authority."""

        return self.descriptor


_ISSUED_CAPABILITIES: dict[
    int,
    tuple[VerifiedStage3PackageCapability, tuple[object, ...]],
] = {}
_ISSUED_COMPLETE_REAL_PACKAGES: dict[
    int,
    tuple[VerifiedCompleteRealStage3Package, tuple[object, ...]],
] = {}


def verify_and_issue_stage3_package(
    package_dir: Path,
    *,
    scope: Stage3PackageScope,
    ttl_seconds: int = DEFAULT_STAGE3_CAPABILITY_TTL_SECONDS,
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    stage2_capability: object = None,
    real_package_verifier: VerifiedCompleteRealStage3Package | None = None,
) -> VerifiedStage3PackageCapability:
    """Fully verify a package and issue a bounded process-local capability."""

    if not isinstance(scope, Stage3PackageScope):
        raise TypeError("scope must be Stage3PackageScope")
    _validate_ttl(ttl_seconds)
    if scope is Stage3PackageScope.APPROVED_REAL_DATA:
        return _issue_real_stage3_capability(
            package_dir=package_dir,
            ttl_seconds=ttl_seconds,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            stage2_capability=stage2_capability,
            real_package_receipt=real_package_verifier,
        )
    if not isinstance(package_dir, Path):
        raise TypeError("package_dir must be a pathlib.Path")
    if dataset_id not in (None, SYNTHETIC_STAGE3_DATASET_ID) or dataset_version not in (
        None,
        SYNTHETIC_STAGE3_DATASET_VERSION,
    ):
        raise Stage3PackageCapabilityError(
            ("synthetic Stage 3 dataset identity/version cannot be overridden",)
        )
    if stage2_capability is not None:
        _validate_optional_synthetic_stage2_capability(stage2_capability)
    if real_package_verifier is not None:
        raise Stage3PackageCapabilityError(
            ("synthetic Stage 3 issuance cannot carry a real-data package verifier",)
        )
    root, root_identity, before_files, before_manifest = _scan_complete_toy_package(package_dir)
    result = _full_replay_toy_package(root)
    if not result.verified or not result.replay.passed:
        raise Stage3PackageCapabilityError(("full Stage 3 witness replay did not verify",))
    root_after, identity_after, after_files, after_manifest = _scan_complete_toy_package(root)
    if (
        root_after != root
        or identity_after != root_identity
        or before_manifest != after_manifest
        or before_files != after_files
    ):
        raise Stage3PackageCapabilityError(
            ("Stage 3 package changed during complete verification",)
        )
    snapshot = _synthetic_snapshot(after_files, after_manifest)
    if snapshot.calibration_artifact_hash != result.calibration_artifact_hash:
        raise Stage3PackageCapabilityError(
            ("full verifier artifact hash does not match the package snapshot",)
        )
    issued = _monotonic()
    capability = VerifiedStage3PackageCapability(
        descriptor=snapshot,
        manifest=after_manifest,
        package_root=root,
        package_root_identity=root_identity,
        ttl_seconds=ttl_seconds,
        issued_monotonic=issued,
        expires_monotonic=issued + ttl_seconds,
        issued_process_id=os.getpid(),
        _process_nonce=_PROCESS_NONCE,
        _issuance_nonce=object(),
        _token=_CAPABILITY_TOKEN,
    )
    with _CAPABILITY_LOCK:
        _ISSUED_CAPABILITIES[id(capability)] = (
            capability,
            _capability_fingerprint(capability),
        )
    return capability


def stage3_package_capability_validation_reasons(
    value: object,
    *,
    expected_scope: Stage3PackageScope | None = None,
    expected_dataset_id: str | None = None,
    expected_dataset_version: str | None = None,
) -> tuple[str, ...]:
    """Validate issuance, process identity, immutable bindings, TTL, and expectations."""

    if not isinstance(value, VerifiedStage3PackageCapability):
        return ("Stage 3 package capability has an invalid type",)
    with _CAPABILITY_LOCK:
        issued = _ISSUED_CAPABILITIES.get(id(value))
    if issued is None or issued[0] is not value:
        return ("Stage 3 package capability was not issued by the full package verifier",)
    reasons: list[str] = []
    try:
        if issued[1] != _capability_fingerprint(value):
            reasons.append("Stage 3 package capability bindings changed after issuance")
        if value._process_nonce is not _PROCESS_NONCE:
            reasons.append("Stage 3 package capability process nonce is invalid")
        if value.issued_process_id != os.getpid() or value.issued_process_id != _PROCESS_ID:
            reasons.append("Stage 3 package capability belongs to a different process")
        if _monotonic() >= value.expires_monotonic:
            reasons.append("Stage 3 package capability has expired")
        if value.descriptor.scope is Stage3PackageScope.APPROVED_REAL_DATA:
            nested_reasons = complete_real_stage3_package_validation_reasons(
                value._real_package_receipt
            )
            if nested_reasons:
                reasons.append("complete real Stage 3 package receipt is invalid")
        if expected_scope is not None and value.descriptor.scope is not expected_scope:
            reasons.append("Stage 3 package capability scope does not match")
        if expected_dataset_id is not None and value.descriptor.dataset_id != expected_dataset_id:
            reasons.append("Stage 3 package capability dataset_id does not match")
        if (
            expected_dataset_version is not None
            and value.descriptor.dataset_version != expected_dataset_version
        ):
            reasons.append("Stage 3 package capability dataset_version does not match")
    except (AttributeError, TypeError, ValueError):
        return ("Stage 3 package capability fields are invalid",)
    return tuple(dict.fromkeys(reasons))


def require_stage3_package_capability(
    value: object,
    *,
    expected_scope: Stage3PackageScope | None = None,
    expected_dataset_id: str | None = None,
    expected_dataset_version: str | None = None,
) -> VerifiedStage3PackageCapability:
    reasons = stage3_package_capability_validation_reasons(
        value,
        expected_scope=expected_scope,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
    )
    if reasons:
        raise Stage3PackageCapabilityError(reasons)
    return cast(VerifiedStage3PackageCapability, value)


def stage3_package_capability_snapshot(
    value: object,
    *,
    expected_scope: Stage3PackageScope | None = None,
) -> Stage3PackageDescriptor:
    """Return the immutable descriptor only after capability validation."""

    return require_stage3_package_capability(value, expected_scope=expected_scope).descriptor


def replay_stage3_package(
    value: object,
) -> Stage3PackageDescriptor:
    """Repeat exact scan and full witness replay, then compare every issued binding."""

    capability = require_stage3_package_capability(value)
    if capability.descriptor.scope is Stage3PackageScope.APPROVED_REAL_DATA:
        receipt = capability._real_package_receipt
        if receipt is None:
            raise Stage3PackageCapabilityError(("real package receipt is missing",))
        descriptor = replay_complete_real_stage3_package(receipt)
        if descriptor != capability.descriptor or receipt.manifest != capability.manifest:
            raise Stage3PackageCapabilityError(
                ("complete real Stage 3 replay differs from the issued capability",)
            )
        require_stage3_package_capability(capability)
        return capability.descriptor
    root, root_identity, files, manifest = _scan_complete_toy_package(capability.package_root)
    if root != capability.package_root or root_identity != capability.package_root_identity:
        raise Stage3PackageCapabilityError(("Stage 3 package root identity changed",))
    result = _full_replay_toy_package(root)
    root_after, identity_after, files_after, manifest_after = _scan_complete_toy_package(root)
    if (
        root_after != root
        or identity_after != root_identity
        or files_after != files
        or manifest_after != manifest
    ):
        raise Stage3PackageCapabilityError(("Stage 3 package changed during replay",))
    snapshot = _synthetic_snapshot(files_after, manifest_after)
    if (
        not result.verified
        or not result.replay.passed
        or result.calibration_artifact_hash != snapshot.calibration_artifact_hash
        or manifest_after != capability.manifest
        or snapshot != capability.descriptor
    ):
        raise Stage3PackageCapabilityError(
            ("Stage 3 package replay does not match the issued snapshot",)
        )
    require_stage3_package_capability(capability)
    return capability.descriptor


def consume_stage3_package(
    value: object,
    *,
    expected_scope: Stage3PackageScope,
    expected_dataset_id: str,
    expected_dataset_version: str,
) -> Stage3PackageDescriptor:
    """Validate expectations and replay the complete package at the consumption point."""

    capability = require_stage3_package_capability(
        value,
        expected_scope=expected_scope,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
    )
    replayed = replay_stage3_package(capability)
    require_stage3_package_capability(
        capability,
        expected_scope=expected_scope,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_version=expected_dataset_version,
    )
    return replayed


def _synthetic_snapshot(
    files: Mapping[str, bytes],
    manifest: Stage3PackageManifest,
) -> Stage3PackageDescriptor:
    artifact = _decode_mapping(files["calibration_artifact.json"], "calibration artifact")
    required = {
        "schema_version",
        "method_version",
        "protocol_scope",
        "alpha",
        "positive_threshold",
        "negative_threshold",
        "positive_sample_count",
        "negative_sample_count",
        "total_sample_count",
        "positive_order_index",
        "negative_order_index",
        "config_hash",
        "data_hash",
        "model_hash",
        "status",
        "failure_reasons",
    }
    if set(artifact) != required:
        raise Stage3PackageCapabilityError(
            ("calibration artifact fields are incomplete or unknown",)
        )
    if artifact["status"] != "SUCCESS" or artifact["failure_reasons"] != []:
        raise Stage3PackageCapabilityError(("calibration artifact is not successful",))
    alpha = _require_finite(artifact["alpha"], "alpha", positive=True)
    positive_threshold = _require_finite(
        artifact["positive_threshold"], "positive_threshold", nonnegative=True
    )
    negative_threshold = _require_finite(
        artifact["negative_threshold"], "negative_threshold", nonnegative=True
    )
    positive_count = _require_positive_int(
        artifact["positive_sample_count"], "positive_sample_count"
    )
    negative_count = _require_positive_int(
        artifact["negative_sample_count"], "negative_sample_count"
    )
    total_count = _require_positive_int(artifact["total_sample_count"], "total_sample_count")
    if total_count != positive_count + negative_count:
        raise Stage3PackageCapabilityError(("calibration sample counts are inconsistent",))
    snapshot = Stage3PackageDescriptor(
        scope=Stage3PackageScope.SYNTHETIC_TOY,
        dataset_id=SYNTHETIC_STAGE3_DATASET_ID,
        dataset_version=SYNTHETIC_STAGE3_DATASET_VERSION,
        calibration_schema_version=_require_nonempty(
            artifact["schema_version"], "calibration schema_version"
        ),
        calibration_method_version=_require_nonempty(
            artifact["method_version"], "calibration method_version"
        ),
        protocol_scope=_require_nonempty(artifact["protocol_scope"], "protocol_scope"),
        config_hash=_require_sha256(artifact["config_hash"], "config_hash"),
        data_hash=_require_sha256(artifact["data_hash"], "data_hash"),
        model_hash=_require_sha256(artifact["model_hash"], "model_hash"),
        calibration_artifact_hash=_sha256_bytes(files["calibration_artifact.json"]),
        positive_sample_count=positive_count,
        negative_sample_count=negative_count,
        total_sample_count=total_count,
        alpha=alpha,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        package_sha256=manifest.package_sha256,
        package_file_count=manifest.file_count,
        fixture_only=True,
    )
    summary = _decode_mapping(files["summary.json"], "Stage 3 summary")
    expected_summary = {
        "schema_version": "3.0.0",
        "scope": "SYNTHETIC_TOY",
        "alpha": snapshot.alpha,
        "positive_threshold": snapshot.positive_threshold,
        "negative_threshold": snapshot.negative_threshold,
        "positive_sample_count": snapshot.positive_sample_count,
        "negative_sample_count": snapshot.negative_sample_count,
        "config_hash": snapshot.config_hash,
        "calibration_data_hash": snapshot.data_hash,
        "model_hash": snapshot.model_hash,
        "calibration_artifact_hash": snapshot.calibration_artifact_hash,
        "positive_witness_replay_passed": True,
        "negative_witness_replay_passed": True,
        "real_data_used": False,
        "clinical_claim": False,
    }
    if any(summary.get(name) != expected for name, expected in expected_summary.items()):
        raise Stage3PackageCapabilityError(
            ("Stage 3 summary does not match the fully verified calibration package",)
        )
    return snapshot


def _scan_complete_toy_package(
    package_dir: Path,
) -> tuple[Path, tuple[int, int], dict[str, bytes], Stage3PackageManifest]:
    return _scan_exact_package(
        package_dir,
        expected_files=_STAGE3_TOY_FILES,
        expected_directories=_STAGE3_TOY_DIRECTORIES,
    )


def _scan_complete_real_package(
    package_dir: Path,
) -> tuple[Path, tuple[int, int], dict[str, bytes], Stage3PackageManifest]:
    return _scan_exact_package(
        package_dir,
        expected_files=_STAGE3_REAL_FILES,
        expected_directories=_STAGE3_REAL_DIRECTORIES,
    )


def _scan_exact_package(
    package_dir: Path,
    *,
    expected_files: frozenset[str],
    expected_directories: frozenset[str],
) -> tuple[Path, tuple[int, int], dict[str, bytes], Stage3PackageManifest]:
    if not isinstance(package_dir, Path):
        raise TypeError("package_dir must be a pathlib.Path")
    _reject_link_components(package_dir)
    try:
        root_lstat = os.lstat(package_dir)
    except OSError as error:
        raise FileNotFoundError(package_dir) from error
    if _is_link_or_reparse(root_lstat) or not stat.S_ISDIR(root_lstat.st_mode):
        raise Stage3PackageCapabilityError(
            ("Stage 3 package root must be a regular non-link directory",)
        )
    try:
        root = package_dir.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise Stage3PackageCapabilityError(("Stage 3 package root cannot be resolved",)) from error
    root_identity = _directory_identity(root_lstat)
    discovered: dict[str, Path] = {}
    pending: list[tuple[Path, str]] = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise Stage3PackageCapabilityError(
                ("Stage 3 package directory cannot be enumerated",)
            ) from error
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            normalized = relative.replace("\\", "/")
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise Stage3PackageCapabilityError(
                    (f"Stage 3 package entry cannot be inspected: {normalized}",)
                ) from error
            if entry.is_symlink() or _is_link_or_reparse(entry_stat):
                raise Stage3PackageCapabilityError(
                    (f"Stage 3 package contains a link/reparse point: {normalized}",)
                )
            entry_path = Path(entry.path)
            if stat.S_ISDIR(entry_stat.st_mode):
                if normalized not in expected_directories:
                    raise Stage3PackageCapabilityError(
                        (f"Stage 3 package contains an unknown directory: {normalized}",)
                    )
                pending.append((entry_path, normalized))
            elif stat.S_ISREG(entry_stat.st_mode):
                if normalized not in expected_files:
                    raise Stage3PackageCapabilityError(
                        (f"Stage 3 package contains an unknown file: {normalized}",)
                    )
                discovered[normalized] = entry_path
            else:
                raise Stage3PackageCapabilityError(
                    (f"Stage 3 package contains an unsupported entry: {normalized}",)
                )
    missing = sorted(expected_files.difference(discovered))
    if missing:
        raise Stage3PackageCapabilityError((f"Stage 3 package files missing={missing!r}",))
    files = {
        name: _read_regular_bounded(discovered[name], relative_path=name)
        for name in sorted(expected_files)
    }
    manifest = _package_manifest(files, expected_files=expected_files)
    return root, root_identity, files, manifest


def _package_manifest(
    files: Mapping[str, bytes], *, expected_files: frozenset[str]
) -> Stage3PackageManifest:
    if set(files) != expected_files:
        raise Stage3PackageCapabilityError(("Stage 3 package files are incomplete or unknown",))
    entries = tuple(
        Stage3PackageFileEntry(
            relative_path=name,
            size_bytes=len(files[name]),
            sha256=_sha256_bytes(files[name]),
        )
        for name in sorted(files)
    )
    return Stage3PackageManifest(files=entries, package_sha256=_manifest_digest(entries))


def _manifest_digest(files: tuple[Stage3PackageFileEntry, ...]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "digest_method": STAGE3_PACKAGE_DIGEST_METHOD,
                "file_count": len(files),
                "files": [item.as_dict() for item in files],
            }
        )
    )


def _read_regular_bounded(path: Path, *, relative_path: str) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file cannot be inspected: {relative_path}",)
        ) from error
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file is not a regular non-link file: {relative_path}",)
        )
    if before.st_nlink != 1:
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file must not have multiple hard links: {relative_path}",)
        )
    if before.st_size > _MAX_ARTIFACT_BYTES:
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file exceeds size limit: {relative_path}",)
        )
    payload = bytearray()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(before):
                raise Stage3PackageCapabilityError(
                    (f"Stage 3 package file changed before read: {relative_path}",)
                )
            while True:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > _MAX_ARTIFACT_BYTES:
                    raise Stage3PackageCapabilityError(
                        (f"Stage 3 package file exceeds size limit: {relative_path}",)
                    )
            after_opened = os.fstat(descriptor)
            if _file_identity(after_opened) != _file_identity(before):
                raise Stage3PackageCapabilityError(
                    (f"Stage 3 package file changed during read: {relative_path}",)
                )
        finally:
            os.close(descriptor)
    except Stage3PackageCapabilityError:
        raise
    except OSError as error:
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file cannot be read: {relative_path}",)
        ) from error
    try:
        after = os.lstat(path)
    except OSError as error:
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file disappeared after read: {relative_path}",)
        ) from error
    if _file_identity(after) != _file_identity(before) or len(payload) != before.st_size:
        raise Stage3PackageCapabilityError(
            (f"Stage 3 package file changed during read: {relative_path}",)
        )
    return bytes(payload)


def _full_replay_toy_package(package_root: Path) -> Stage3ToyRunResult:
    """Normalize all expected full-verifier failures to the capability boundary."""

    try:
        return verify_stage3_toy_package(package_root)
    except (OSError, TypeError, ValueError) as error:
        raise Stage3PackageCapabilityError(
            ("complete Stage 3 toy package regeneration/replay failed",)
        ) from error


def _validate_optional_synthetic_stage2_capability(stage2_capability: object) -> None:
    """Validate optional synthetic provenance without binding unrelated identities."""

    try:
        stage2_module = importlib.import_module("retinaentail.data.stage2_package")
        stage2_scope_type = stage2_module.Stage2PackageScope
        stage2_capability_type = stage2_module.VerifiedStage2PackageCapability
        stage2_validation = stage2_module.stage2_package_capability_validation_reasons
        stage2_snapshot = stage2_module.stage2_package_capability_snapshot
        stage2_replay = stage2_module.replay_stage2_package
    except (AttributeError, ImportError) as error:
        raise Stage3PackageCapabilityError(
            ("full Stage 2 package capability verifier is unavailable",)
        ) from error
    if not isinstance(stage2_capability, stage2_capability_type):
        raise Stage3PackageCapabilityError(
            ("optional synthetic Stage 2 evidence is not a full package capability",)
        )
    if stage2_validation(stage2_capability):
        raise Stage3PackageCapabilityError(
            ("optional synthetic Stage 2 package capability is invalid",)
        )
    replay = stage2_replay(stage2_capability)
    if getattr(replay, "valid", False) is not True:
        raise Stage3PackageCapabilityError(("optional synthetic Stage 2 package replay failed",))
    descriptor = stage2_snapshot(stage2_capability)
    if getattr(descriptor, "scope", None) is not stage2_scope_type.SYNTHETIC_FIXTURE:
        raise Stage3PackageCapabilityError(
            ("optional Stage 2 package capability must be SYNTHETIC_FIXTURE",)
        )


def _validate_real_prerequisites(
    stage2_capability: object,
    frozen_data_card: object,
    *,
    replay_stage2: bool,
) -> tuple[object, str, DatasetManifestV2 | None]:
    """Validate process-issued Stage 2 and frozen-card evidence without root access."""

    try:
        stage2_module = importlib.import_module("retinaentail.data.stage2_package")
        stage2_scope_type = stage2_module.Stage2PackageScope
        stage2_capability_type = stage2_module.VerifiedStage2PackageCapability
        stage2_validation = stage2_module.stage2_package_capability_validation_reasons
        stage2_snapshot = stage2_module.stage2_package_capability_snapshot
        stage2_replay_fn = stage2_module.replay_stage2_package
        stage2_manifest_replay = stage2_module.replay_stage2_package_manifest
    except (AttributeError, ImportError) as error:
        raise Stage3PackageCapabilityError(
            ("full Stage 2 package capability verifier is unavailable",)
        ) from error
    if not isinstance(stage2_capability, stage2_capability_type):
        raise Stage3PackageCapabilityError(
            ("APPROVED_REAL_DATA requires a full Stage 2 package capability",)
        )
    if stage2_validation(stage2_capability):
        raise Stage3PackageCapabilityError(("full Stage 2 package capability is invalid",))
    stage2_manifest: DatasetManifestV2 | None = None
    if replay_stage2:
        replay = stage2_replay_fn(stage2_capability)
        if getattr(replay, "valid", False) is not True:
            raise Stage3PackageCapabilityError(("full Stage 2 package replay failed",))
        try:
            replayed_manifest = stage2_manifest_replay(stage2_capability)
        except (PermissionError, TypeError, ValueError) as error:
            raise Stage3PackageCapabilityError(("full Stage 2 manifest replay failed",)) from error
        if not isinstance(replayed_manifest, DatasetManifestV2):
            raise Stage3PackageCapabilityError(
                ("full Stage 2 manifest replay returned an invalid type",)
            )
        stage2_manifest = replayed_manifest
    stage2_descriptor = stage2_snapshot(stage2_capability)
    if getattr(stage2_descriptor, "scope", None) is not stage2_scope_type.APPROVED_REAL_DATA:
        raise Stage3PackageCapabilityError(
            ("Stage 2 package capability scope is not APPROVED_REAL_DATA",)
        )
    if not isinstance(frozen_data_card, FrozenDatasetCard):
        raise Stage3PackageCapabilityError(
            ("APPROVED_REAL_DATA requires a process-issued frozen dataset card",)
        )
    if frozen_data_card_validation_reasons(frozen_data_card):
        raise Stage3PackageCapabilityError(("frozen dataset card is invalid",))
    if frozen_data_card.scope is not DatasetCardScope.APPROVED_REAL_DATA:
        raise Stage3PackageCapabilityError(("frozen dataset card scope is not APPROVED_REAL_DATA",))
    comparisons = (
        ("dataset_id", frozen_data_card.dataset_id),
        ("dataset_version", frozen_data_card.dataset_version),
        ("record_count", frozen_data_card.sample_count),
        ("registry_sha256", frozen_data_card.registry_sha256),
        ("license_review_sha256", frozen_data_card.license_review_sha256),
        ("adapter_spec_sha256", frozen_data_card.adapter_spec_sha256),
    )
    if any(
        getattr(stage2_descriptor, field_name, None) != expected
        for field_name, expected in comparisons
    ):
        raise Stage3PackageCapabilityError(
            ("Stage 2 package and frozen dataset card bindings differ",)
        )
    card_payload = frozen_data_card.to_dict()
    card_sha256 = _sha256_bytes(_canonical_json_bytes(card_payload))
    return stage2_descriptor, card_sha256, stage2_manifest


def _verify_real_package_payloads(
    files: Mapping[str, bytes],
    manifest: Stage3PackageManifest,
    *,
    stage2_descriptor: object,
    stage2_manifest: DatasetManifestV2,
    frozen_data_card: FrozenDatasetCard,
    frozen_data_card_sha256: str,
) -> Stage3PackageDescriptor:
    if set(files) != _STAGE3_REAL_FILES:
        raise Stage3PackageCapabilityError(
            ("complete real Stage 3 package files are incomplete or unknown",)
        )
    envelope = _decode_exact_mapping(
        files["package.json"],
        "complete real Stage 3 package envelope",
        fields={
            "schema_version",
            "scope",
            "dataset_id",
            "dataset_version",
            "split",
            "fixture_only",
            "baseline_only",
            "clinical_claim",
            "package_digest_method",
            "package_file_count",
            "stage2_package_sha256",
            "frozen_data_card_sha256",
            "calibration_samples_sha256",
            "calibration_artifact_sha256",
            "alpha",
            "positive_threshold",
            "negative_threshold",
            "positive_sample_count",
            "negative_sample_count",
            "total_sample_count",
            "config_hash",
            "data_hash",
            "model_hash",
        },
    )
    if files["package.json"] != _canonical_json_bytes(envelope):
        raise Stage3PackageCapabilityError(("real package envelope is not canonical JSON",))
    if (
        envelope["schema_version"] != STAGE3_REAL_PACKAGE_SCHEMA_VERSION
        or envelope["scope"] != Stage3PackageScope.APPROVED_REAL_DATA.value
        or envelope["split"] != SampleSplit.CALIBRATION.value
        or envelope["baseline_only"] is not True
        or envelope["clinical_claim"] is not False
        or envelope["package_digest_method"] != STAGE3_PACKAGE_DIGEST_METHOD
        or envelope["package_file_count"] != len(_STAGE3_REAL_FILES)
    ):
        raise Stage3PackageCapabilityError(("real package envelope metadata is invalid",))
    fixture_only = envelope["fixture_only"]
    if type(fixture_only) is not bool:
        raise Stage3PackageCapabilityError(("real package fixture_only must be a boolean",))
    dataset_id = _require_dataset_id(envelope["dataset_id"])
    dataset_version = _require_dataset_version(envelope["dataset_version"])
    if (
        dataset_id != getattr(stage2_descriptor, "dataset_id", None)
        or dataset_version != getattr(stage2_descriptor, "dataset_version", None)
        or dataset_id != frozen_data_card.dataset_id
        or dataset_version != frozen_data_card.dataset_version
    ):
        raise Stage3PackageCapabilityError(
            ("real package dataset identity/version does not match prerequisites",)
        )
    stage2_package_sha256 = _require_sha256(
        envelope["stage2_package_sha256"], "stage2_package_sha256"
    )
    if stage2_package_sha256 != getattr(stage2_descriptor, "package_sha256", None):
        raise Stage3PackageCapabilityError(("real package Stage 2 digest does not match",))
    if (
        _require_sha256(envelope["frozen_data_card_sha256"], "frozen_data_card_sha256")
        != frozen_data_card_sha256
        or _sha256_bytes(files["frozen_data_card.json"]) != frozen_data_card_sha256
        or files["frozen_data_card.json"] != _canonical_json_bytes(frozen_data_card.to_dict())
    ):
        raise Stage3PackageCapabilityError(("real package frozen data card does not match",))
    card_payload = _decode_mapping(files["frozen_data_card.json"], "frozen data card")
    if card_payload != frozen_data_card.to_dict():
        raise Stage3PackageCapabilityError(("serialized frozen data card differs",))

    samples_payload = _decode_exact_mapping(
        files["calibration_samples.json"],
        "real calibration samples",
        fields={"schema_version", "dataset_id", "dataset_version", "split", "samples"},
    )
    if files["calibration_samples.json"] != _canonical_json_bytes(samples_payload):
        raise Stage3PackageCapabilityError(("real calibration samples are not canonical JSON",))
    if (
        samples_payload["schema_version"] != STAGE3_REAL_PACKAGE_SCHEMA_VERSION
        or samples_payload["dataset_id"] != dataset_id
        or samples_payload["dataset_version"] != dataset_version
        or samples_payload["split"] != SampleSplit.CALIBRATION.value
    ):
        raise Stage3PackageCapabilityError(("real calibration sample metadata is invalid",))
    samples = _parse_real_calibration_samples(samples_payload["samples"])
    _verify_calibration_sample_membership(samples, stage2_manifest)
    samples_sha256 = _sha256_bytes(files["calibration_samples.json"])
    if (
        _require_sha256(envelope["calibration_samples_sha256"], "calibration_samples_sha256")
        != samples_sha256
    ):
        raise Stage3PackageCapabilityError(("real calibration sample digest does not match",))

    alpha = _require_finite(envelope["alpha"], "alpha", positive=True)
    if alpha >= 1.0:
        raise Stage3PackageCapabilityError(("alpha must be less than one",))
    envelope_positive_threshold = _require_finite(
        envelope["positive_threshold"], "positive_threshold", nonnegative=True
    )
    envelope_negative_threshold = _require_finite(
        envelope["negative_threshold"], "negative_threshold", nonnegative=True
    )
    envelope_positive_count = _require_positive_int(
        envelope["positive_sample_count"], "positive_sample_count"
    )
    envelope_negative_count = _require_positive_int(
        envelope["negative_sample_count"], "negative_sample_count"
    )
    envelope_total_count = _require_positive_int(
        envelope["total_sample_count"], "total_sample_count"
    )
    config_hash = _require_sha256(envelope["config_hash"], "config_hash")
    data_hash = _require_sha256(envelope["data_hash"], "data_hash")
    model_hash = _require_sha256(envelope["model_hash"], "model_hash")
    if (
        config_hash != frozen_data_card.config_sha256
        or data_hash != frozen_data_card.data_sha256
        or model_hash != frozen_data_card.model_sha256
    ):
        raise Stage3PackageCapabilityError(
            ("real calibration hashes differ from the frozen dataset card",)
        )
    recomputed = calibrate_order_statistic(
        samples,
        alpha=alpha,
        config_hash=config_hash,
        data_hash=data_hash,
        model_hash=model_hash,
    )
    if (
        recomputed.status is not CalibrationRunStatus.SUCCESS
        or recomputed.failure_reasons
        or recomputed.protocol_scope.value != "TOY_PROTOCOL"
    ):
        raise Stage3PackageCapabilityError(
            ("real calibration samples fail closed during recomputation",)
        )
    artifact_payload = _decode_mapping(
        files["calibration_artifact.json"], "real calibration artifact"
    )
    if artifact_payload != recomputed.to_dict() or files[
        "calibration_artifact.json"
    ] != _canonical_json_bytes(recomputed.to_dict()):
        raise Stage3PackageCapabilityError(
            ("real calibration artifact differs from recomputation",)
        )
    artifact_sha256 = _sha256_bytes(files["calibration_artifact.json"])
    if (
        _require_sha256(envelope["calibration_artifact_sha256"], "calibration_artifact_sha256")
        != artifact_sha256
    ):
        raise Stage3PackageCapabilityError(("real calibration artifact digest differs",))
    if (
        envelope_positive_threshold != recomputed.positive_threshold
        or envelope_negative_threshold != recomputed.negative_threshold
        or envelope_positive_count != recomputed.positive_sample_count
        or envelope_negative_count != recomputed.negative_sample_count
        or envelope_total_count != recomputed.total_sample_count
    ):
        raise Stage3PackageCapabilityError(
            ("real package envelope calibration values differ from recomputation",)
        )
    _require_envelope_artifact_match(envelope, recomputed)
    assert recomputed.positive_threshold is not None
    assert recomputed.negative_threshold is not None
    return Stage3PackageDescriptor(
        scope=Stage3PackageScope.APPROVED_REAL_DATA,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        calibration_schema_version=recomputed.schema_version,
        calibration_method_version=recomputed.method_version,
        protocol_scope=recomputed.protocol_scope.value,
        config_hash=config_hash,
        data_hash=data_hash,
        model_hash=model_hash,
        calibration_artifact_hash=artifact_sha256,
        positive_sample_count=recomputed.positive_sample_count,
        negative_sample_count=recomputed.negative_sample_count,
        total_sample_count=recomputed.total_sample_count,
        alpha=alpha,
        positive_threshold=recomputed.positive_threshold,
        negative_threshold=recomputed.negative_threshold,
        package_sha256=manifest.package_sha256,
        package_file_count=manifest.file_count,
        stage2_package_sha256=stage2_package_sha256,
        frozen_data_card_sha256=frozen_data_card_sha256,
        fixture_only=fixture_only,
    )


def _parse_real_calibration_samples(value: object) -> tuple[CalibrationSample, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise Stage3PackageCapabilityError(("real calibration samples must be an array",))
    result: list[CalibrationSample] = []
    fields = {
        "schema_version",
        "sample_id",
        "split",
        "label",
        "positive_score",
        "negative_score",
        "source_image_id",
        "patient_id",
        "intervention_pair_id",
    }
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise Stage3PackageCapabilityError(
                (f"real calibration sample {index} must be an object",)
            )
        item = {key: entry for key, entry in raw.items() if isinstance(key, str)}
        if len(item) != len(raw) or set(item) != fields:
            raise Stage3PackageCapabilityError(
                (f"real calibration sample {index} fields are incomplete or unknown",)
            )
        if item["schema_version"] != CALIBRATION_SCHEMA_VERSION:
            raise Stage3PackageCapabilityError(
                (f"real calibration sample {index} schema is unsupported",)
            )
        sample_id = _require_nonempty(item["sample_id"], f"sample {index} id")
        if item["split"] != SampleSplit.CALIBRATION.value:
            raise Stage3PackageCapabilityError(
                (f"real calibration sample {index} is not in calibration split",)
            )
        try:
            label = CalibrationLabel(cast(str, item["label"]))
        except (TypeError, ValueError) as error:
            raise Stage3PackageCapabilityError(
                (f"real calibration sample {index} label is invalid",)
            ) from error
        if label is CalibrationLabel.UNKNOWN:
            raise Stage3PackageCapabilityError(
                (f"real calibration sample {index} label is unknown",)
            )
        result.append(
            CalibrationSample(
                schema_version=CALIBRATION_SCHEMA_VERSION,
                sample_id=sample_id,
                split=SampleSplit.CALIBRATION,
                label=label,
                positive_score=_require_finite(
                    item["positive_score"], f"sample {index} positive_score", nonnegative=True
                ),
                negative_score=_require_finite(
                    item["negative_score"], f"sample {index} negative_score", nonnegative=True
                ),
                source_image_id=_optional_nonempty(
                    item["source_image_id"], f"sample {index} source_image_id"
                ),
                patient_id=_optional_nonempty(item["patient_id"], f"sample {index} patient_id"),
                intervention_pair_id=_optional_nonempty(
                    item["intervention_pair_id"],
                    f"sample {index} intervention_pair_id",
                ),
            )
        )
    if not result:
        raise Stage3PackageCapabilityError(("real calibration sample array is empty",))
    return tuple(result)


def _verify_calibration_sample_membership(
    samples: tuple[CalibrationSample, ...],
    manifest: DatasetManifestV2,
) -> None:
    """Bind every score row to one unambiguous Stage 2 calibration record."""

    records_by_source: dict[str, ManifestRecordV2] = {}
    for record in manifest.records:
        if record.source_image_id in records_by_source:
            raise Stage3PackageCapabilityError(
                ("Stage 2 manifest contains duplicate source_image_id values",)
            )
        records_by_source[record.source_image_id] = record
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(sample_ids) != len(set(sample_ids)):
        raise Stage3PackageCapabilityError(
            ("real calibration package contains duplicate sample_id values",)
        )
    for sample in samples:
        if sample.source_image_id is None:
            raise Stage3PackageCapabilityError(
                ("real calibration samples require source_image_id membership",)
            )
        matched_record = records_by_source.get(sample.source_image_id)
        if matched_record is None:
            raise Stage3PackageCapabilityError(
                ("real calibration sample source_image_id is absent from Stage 2",)
            )
        if matched_record.split != SampleSplit.CALIBRATION.value:
            raise Stage3PackageCapabilityError(
                ("real calibration sample source_image_id is outside calibration split",)
            )
        if (
            sample.patient_id != matched_record.patient_id
            or sample.intervention_pair_id != matched_record.intervention_pair_id
        ):
            raise Stage3PackageCapabilityError(
                ("real calibration sample identity provenance differs from Stage 2",)
            )


def _require_envelope_artifact_match(
    envelope: Mapping[str, object], artifact: CalibrationArtifact
) -> None:
    expected: dict[str, object] = {
        "alpha": artifact.alpha,
        "positive_threshold": artifact.positive_threshold,
        "negative_threshold": artifact.negative_threshold,
        "positive_sample_count": artifact.positive_sample_count,
        "negative_sample_count": artifact.negative_sample_count,
        "total_sample_count": artifact.total_sample_count,
        "config_hash": artifact.config_hash,
        "data_hash": artifact.data_hash,
        "model_hash": artifact.model_hash,
    }
    if any(envelope[name] != expected_value for name, expected_value in expected.items()):
        raise Stage3PackageCapabilityError(
            ("real package envelope calibration bindings differ from recomputation",)
        )


def verify_complete_real_stage3_package(
    package_dir: Path,
    *,
    stage2_capability: object,
    frozen_data_card: FrozenDatasetCard,
    ttl_seconds: int = DEFAULT_STAGE3_CAPABILITY_TTL_SECONDS,
) -> VerifiedCompleteRealStage3Package:
    """Verify an exact real-data envelope and issue a short-lived replay receipt.

    This reads only the supplied evidence-package directories.  The Stage 2 package
    capability and frozen card are revalidated; no dataset root is accepted here.
    """

    _validate_ttl(ttl_seconds)
    stage2_descriptor, card_sha256, stage2_manifest = _validate_real_prerequisites(
        stage2_capability,
        frozen_data_card,
        replay_stage2=True,
    )
    if stage2_manifest is None:
        raise Stage3PackageCapabilityError(("full Stage 2 manifest replay is missing",))
    root, root_identity, before_files, before_manifest = _scan_complete_real_package(package_dir)
    descriptor = _verify_real_package_payloads(
        before_files,
        before_manifest,
        stage2_descriptor=stage2_descriptor,
        stage2_manifest=stage2_manifest,
        frozen_data_card=frozen_data_card,
        frozen_data_card_sha256=card_sha256,
    )
    after_stage2_descriptor, after_card_sha256, after_stage2_manifest = (
        _validate_real_prerequisites(
            stage2_capability,
            frozen_data_card,
            replay_stage2=True,
        )
    )
    if (
        after_stage2_descriptor != stage2_descriptor
        or after_card_sha256 != card_sha256
        or after_stage2_manifest != stage2_manifest
    ):
        raise Stage3PackageCapabilityError(
            ("Stage 2/card prerequisites changed during Stage 3 verification",)
        )
    root_after, identity_after, after_files, after_manifest = _scan_complete_real_package(root)
    if (
        root_after != root
        or identity_after != root_identity
        or before_files != after_files
        or before_manifest != after_manifest
    ):
        raise Stage3PackageCapabilityError(
            ("complete real Stage 3 package changed during verification",)
        )
    descriptor_after = _verify_real_package_payloads(
        after_files,
        after_manifest,
        stage2_descriptor=stage2_descriptor,
        stage2_manifest=stage2_manifest,
        frozen_data_card=frozen_data_card,
        frozen_data_card_sha256=card_sha256,
    )
    if descriptor_after != descriptor:
        raise Stage3PackageCapabilityError(
            ("complete real Stage 3 descriptor changed during verification",)
        )
    issued = _monotonic()
    receipt = VerifiedCompleteRealStage3Package(
        descriptor=descriptor,
        manifest=after_manifest,
        package_root=root,
        package_root_identity=root_identity,
        ttl_seconds=ttl_seconds,
        issued_monotonic=issued,
        expires_monotonic=issued + ttl_seconds,
        issued_process_id=os.getpid(),
        _stage2_capability=stage2_capability,
        _frozen_data_card=frozen_data_card,
        _process_nonce=_PROCESS_NONCE,
        _issuance_nonce=object(),
        _token=_CAPABILITY_TOKEN,
    )
    with _CAPABILITY_LOCK:
        _ISSUED_COMPLETE_REAL_PACKAGES[id(receipt)] = (
            receipt,
            _complete_real_fingerprint(receipt),
        )
    return receipt


def complete_real_stage3_package_validation_reasons(value: object) -> tuple[str, ...]:
    """Validate a complete real-package receipt without performing package I/O."""

    if not isinstance(value, VerifiedCompleteRealStage3Package):
        return ("complete real Stage 3 package receipt has an invalid type",)
    with _CAPABILITY_LOCK:
        issued = _ISSUED_COMPLETE_REAL_PACKAGES.get(id(value))
    if issued is None or issued[0] is not value:
        return ("complete real Stage 3 package receipt was not issued by the verifier",)
    reasons: list[str] = []
    try:
        if issued[1] != _complete_real_fingerprint(value):
            reasons.append("complete real Stage 3 package receipt bindings changed")
        if value._process_nonce is not _PROCESS_NONCE:
            reasons.append("complete real Stage 3 package receipt process nonce is invalid")
        if value.issued_process_id != os.getpid() or value.issued_process_id != _PROCESS_ID:
            reasons.append("complete real Stage 3 package receipt belongs to another process")
        if _monotonic() >= value.expires_monotonic:
            reasons.append("complete real Stage 3 package receipt has expired")
        _validate_real_prerequisites(
            value._stage2_capability,
            value._frozen_data_card,
            replay_stage2=False,
        )
    except Stage3PackageCapabilityError:
        reasons.append("complete real Stage 3 package prerequisites are invalid")
    except (AttributeError, TypeError, ValueError):
        return ("complete real Stage 3 package receipt fields are invalid",)
    return tuple(dict.fromkeys(reasons))


def complete_real_stage3_package_snapshot(
    value: object,
) -> Stage3PackageDescriptor:
    reasons = complete_real_stage3_package_validation_reasons(value)
    if reasons:
        raise Stage3PackageCapabilityError(reasons)
    return cast(VerifiedCompleteRealStage3Package, value).descriptor


def replay_complete_real_stage3_package(
    value: object,
) -> Stage3PackageDescriptor:
    """Re-read all envelope bytes and recompute calibration from serialized samples."""

    receipt = _require_complete_real_stage3_package(value)
    stage2_descriptor, card_sha256, stage2_manifest = _validate_real_prerequisites(
        receipt._stage2_capability,
        receipt._frozen_data_card,
        replay_stage2=True,
    )
    if stage2_manifest is None:
        raise Stage3PackageCapabilityError(("full Stage 2 manifest replay is missing",))
    root, root_identity, files, manifest = _scan_complete_real_package(receipt.package_root)
    if root != receipt.package_root or root_identity != receipt.package_root_identity:
        raise Stage3PackageCapabilityError(("complete real package root identity changed",))
    descriptor = _verify_real_package_payloads(
        files,
        manifest,
        stage2_descriptor=stage2_descriptor,
        stage2_manifest=stage2_manifest,
        frozen_data_card=receipt._frozen_data_card,
        frozen_data_card_sha256=card_sha256,
    )
    root_after, identity_after, files_after, manifest_after = _scan_complete_real_package(root)
    after_stage2_descriptor, after_card_sha256, after_stage2_manifest = (
        _validate_real_prerequisites(
            receipt._stage2_capability,
            receipt._frozen_data_card,
            replay_stage2=True,
        )
    )
    if (
        root_after != root
        or identity_after != root_identity
        or files_after != files
        or manifest_after != manifest
        or after_stage2_descriptor != stage2_descriptor
        or after_card_sha256 != card_sha256
        or after_stage2_manifest != stage2_manifest
        or descriptor != receipt.descriptor
        or manifest != receipt.manifest
    ):
        raise Stage3PackageCapabilityError(
            ("complete real Stage 3 package replay differs from issuance",)
        )
    _require_complete_real_stage3_package(receipt)
    return receipt.descriptor


def _require_complete_real_stage3_package(
    value: object,
) -> VerifiedCompleteRealStage3Package:
    reasons = complete_real_stage3_package_validation_reasons(value)
    if reasons:
        raise Stage3PackageCapabilityError(reasons)
    return cast(VerifiedCompleteRealStage3Package, value)


def _issue_real_stage3_capability(
    *,
    package_dir: Path,
    ttl_seconds: int,
    dataset_id: str | None,
    dataset_version: str | None,
    stage2_capability: object,
    real_package_receipt: VerifiedCompleteRealStage3Package | None,
) -> VerifiedStage3PackageCapability:
    """Issue REAL authority only from a freshly replayed complete receipt."""

    if dataset_id is None or dataset_version is None:
        raise Stage3PackageCapabilityError(
            ("APPROVED_REAL_DATA requires explicit dataset_id and dataset_version",)
        )
    try:
        _require_dataset_id(dataset_id)
        _require_dataset_version(dataset_version)
    except ValueError as error:
        raise Stage3PackageCapabilityError(
            ("APPROVED_REAL_DATA dataset identity/version is invalid",)
        ) from error
    if not isinstance(package_dir, Path):
        raise TypeError("package_dir must be a pathlib.Path")
    receipt = _require_complete_real_stage3_package(real_package_receipt)
    if stage2_capability is not receipt._stage2_capability:
        raise Stage3PackageCapabilityError(
            ("Stage 2 capability is not the one bound by the complete Stage 3 receipt",)
        )
    descriptor = replay_complete_real_stage3_package(receipt)
    if descriptor.fixture_only:
        raise Stage3PackageCapabilityError(
            ("synthetic fixture envelopes cannot issue APPROVED_REAL_DATA authority",)
        )
    if descriptor.dataset_id != dataset_id or descriptor.dataset_version != dataset_version:
        raise Stage3PackageCapabilityError(
            ("complete Stage 3 package dataset identity/version does not match",)
        )
    try:
        supplied_root = package_dir.absolute()
    except OSError as error:
        raise Stage3PackageCapabilityError(("Stage 3 package path is invalid",)) from error
    if supplied_root != receipt.package_root:
        raise Stage3PackageCapabilityError(
            ("complete Stage 3 receipt is bound to a different package root",)
        )
    issued = _monotonic()
    if issued + ttl_seconds > receipt.expires_monotonic:
        raise Stage3PackageCapabilityError(
            ("public Stage 3 capability TTL exceeds the complete receipt lifetime",)
        )
    capability = VerifiedStage3PackageCapability(
        descriptor=descriptor,
        manifest=receipt.manifest,
        package_root=receipt.package_root,
        package_root_identity=receipt.package_root_identity,
        ttl_seconds=ttl_seconds,
        issued_monotonic=issued,
        expires_monotonic=issued + ttl_seconds,
        issued_process_id=os.getpid(),
        _process_nonce=_PROCESS_NONCE,
        _issuance_nonce=object(),
        _real_package_receipt=receipt,
        _token=_CAPABILITY_TOKEN,
    )
    with _CAPABILITY_LOCK:
        _ISSUED_CAPABILITIES[id(capability)] = (
            capability,
            _capability_fingerprint(capability),
        )
    return capability


def _reject_link_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            value = os.lstat(current)
        except FileNotFoundError:
            if current == absolute:
                return
            raise
        except OSError as error:
            raise Stage3PackageCapabilityError(
                ("Stage 3 package path component cannot be inspected",)
            ) from error
        if _is_link_or_reparse(value):
            raise Stage3PackageCapabilityError(
                ("Stage 3 package path contains a link/reparse point",)
            )


def _is_link_or_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return stat.S_ISLNK(value.st_mode) or bool(attributes & _REPARSE_POINT_ATTRIBUTE)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_mode),
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _capability_fingerprint(value: VerifiedStage3PackageCapability) -> tuple[object, ...]:
    return (
        value.descriptor.binding_sha256(),
        _sha256_bytes(_canonical_json_bytes(value.manifest.as_dict())),
        value.package_root.as_posix(),
        value.package_root_identity,
        value.ttl_seconds,
        value.issued_monotonic,
        value.expires_monotonic,
        value.issued_process_id,
        value._process_nonce,
        value._issuance_nonce,
        None if value._real_package_receipt is None else id(value._real_package_receipt),
    )


def _complete_real_fingerprint(
    value: VerifiedCompleteRealStage3Package,
) -> tuple[object, ...]:
    return (
        value.descriptor.binding_sha256(),
        _sha256_bytes(_canonical_json_bytes(value.manifest.as_dict())),
        value.package_root.as_posix(),
        value.package_root_identity,
        value.ttl_seconds,
        value.issued_monotonic,
        value.expires_monotonic,
        value.issued_process_id,
        id(value._stage2_capability),
        _sha256_bytes(_canonical_json_bytes(value._frozen_data_card.to_dict())),
        value._process_nonce,
        value._issuance_nonce,
    )


def _validate_ttl(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_STAGE3_CAPABILITY_TTL_SECONDS:
        raise ValueError(
            f"ttl_seconds must be an integer in [1, {MAX_STAGE3_CAPABILITY_TTL_SECONDS}]"
        )
    return value


def _require_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("package relative_path must be normalized POSIX text")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("package relative_path must be normalized and relative")
    return value


def _require_nonempty(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n\x00")
    ):
        raise ValueError(f"{name} must be canonical non-empty text")
    return value


def _optional_nonempty(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty(value, name)


def _require_dataset_id(value: object) -> str:
    result = _require_nonempty(value, "dataset_id")
    if _DATASET_ID_PATTERN.fullmatch(result) is None:
        raise ValueError("dataset_id is invalid")
    return result


def _require_dataset_version(value: object) -> str:
    result = _require_nonempty(value, "dataset_version")
    if _DATASET_VERSION_PATTERN.fullmatch(result) is None:
        raise ValueError("dataset_version is invalid")
    return result


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise Stage3PackageCapabilityError((f"{name} must be a positive integer",))
    return value


def _require_finite(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage3PackageCapabilityError((f"{name} must be a finite number",))
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        raise Stage3PackageCapabilityError((f"{name} has an invalid finite range",))
    return result


def _decode_mapping(payload: bytes, name: str) -> dict[str, object]:
    try:
        value: object = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise Stage3PackageCapabilityError((f"{name} must be strict UTF-8 JSON",)) from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage3PackageCapabilityError((f"{name} must be a JSON object",))
    return {cast(str, key): item for key, item in value.items()}


def _decode_exact_mapping(
    payload: bytes,
    name: str,
    *,
    fields: set[str],
) -> dict[str, object]:
    result = _decode_mapping(payload, name)
    if set(result) != fields:
        raise Stage3PackageCapabilityError((f"{name} fields are incomplete or unknown",))
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"non-standard JSON constant: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "DEFAULT_STAGE3_CAPABILITY_TTL_SECONDS",
    "MAX_STAGE3_CAPABILITY_TTL_SECONDS",
    "STAGE3_PACKAGE_CAPABILITY_SCHEMA_VERSION",
    "STAGE3_PACKAGE_DIGEST_METHOD",
    "STAGE3_PACKAGE_MANIFEST_SCHEMA_VERSION",
    "STAGE3_REAL_PACKAGE_SCHEMA_VERSION",
    "SYNTHETIC_STAGE3_DATASET_ID",
    "SYNTHETIC_STAGE3_DATASET_VERSION",
    "CompleteRealStage3PackageVerifier",
    "Stage3PackageDescriptor",
    "Stage3PackageCapabilityError",
    "Stage3PackageFileEntry",
    "Stage3PackageManifest",
    "Stage3PackageScope",
    "VerifiedCompleteRealStage3Package",
    "VerifiedStage3PackageCapability",
    "complete_real_stage3_package_snapshot",
    "complete_real_stage3_package_validation_reasons",
    "consume_stage3_package",
    "replay_stage3_package",
    "require_stage3_package_capability",
    "stage3_package_capability_snapshot",
    "stage3_package_capability_validation_reasons",
    "replay_complete_real_stage3_package",
    "verify_complete_real_stage3_package",
    "verify_and_issue_stage3_package",
]
