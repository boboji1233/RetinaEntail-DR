"""Offline validation for metadata-only dataset registry templates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_DATASET_IDS = frozenset(
    {
        "idrid",
        "ddr",
        "deepdrid",
        "eyeq",
        "real_fundus",
        "fgadr",
        "maples_dr",
        "fives",
        "hrf_quality",
    }
)
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "name",
        "version",
        "source_url",
        "alternate_source_urls",
        "access_mode",
        "license_status",
        "license_review_status",
        "license_notes",
        "expected_labels",
        "label_verification_status",
        "root_env",
        "patient_id_available",
        "source_image_id_rule",
        "redistribution",
    }
)
_DATASET_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_ROOT_ENV = re.compile(r"^RETINAENTAIL_[A-Z0-9_]+_ROOT$")


class RegistryValidationError(ValueError):
    """Raised when registry metadata is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class RegistryValidationResult:
    path: Path
    dataset_id: str


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{name} must contain a YAML mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RegistryValidationError(f"{name} contains a non-string key")
        result[key] = item
    return result


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RegistryValidationError(f"{field} must be a YAML list")
    result = tuple(_nonempty(item, f"{field} item") for item in value)
    if not allow_empty and not result:
        raise RegistryValidationError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise RegistryValidationError(f"{field} must not contain duplicates")
    return result


def validate_registry_file(path: Path) -> RegistryValidationResult:
    """Validate one registry without downloading or probing its URLs."""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = _mapping(decoded, path.name)
    missing = REQUIRED_FIELDS.difference(data)
    unknown = set(data).difference(REQUIRED_FIELDS)
    if missing or unknown:
        raise RegistryValidationError(
            f"{path.name}: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    if data["schema_version"] != "1.0.0":
        raise RegistryValidationError(f"{path.name}: unsupported schema_version")
    dataset_id = _nonempty(data["dataset_id"], "dataset_id")
    if _DATASET_ID.fullmatch(dataset_id) is None:
        raise RegistryValidationError(f"{path.name}: invalid dataset_id")
    if path.stem != dataset_id:
        raise RegistryValidationError(f"{path.name}: filename must match dataset_id")
    for field in ("name", "version", "license_notes", "source_image_id_rule"):
        _nonempty(data[field], field)
    urls = (_nonempty(data["source_url"], "source_url"),) + _string_list(
        data["alternate_source_urls"], "alternate_source_urls", allow_empty=True
    )
    if any(not url.startswith("https://") for url in urls):
        raise RegistryValidationError(f"{path.name}: registry URLs must use HTTPS")
    if data["access_mode"] not in {"manual", "public_direct", "gated"}:
        raise RegistryValidationError(f"{path.name}: invalid access_mode")
    if data["license_status"] not in {"unknown", "ambiguous", "verified", "restricted"}:
        raise RegistryValidationError(f"{path.name}: invalid license_status")
    if data["license_review_status"] not in {"pending_review", "reviewed"}:
        raise RegistryValidationError(f"{path.name}: invalid license_review_status")
    if data["label_verification_status"] not in {"pending_review", "verified"}:
        raise RegistryValidationError(f"{path.name}: invalid label_verification_status")
    _string_list(data["expected_labels"], "expected_labels", allow_empty=False)
    root_env = _nonempty(data["root_env"], "root_env")
    if _ROOT_ENV.fullmatch(root_env) is None:
        raise RegistryValidationError(f"{path.name}: invalid root_env")
    patient_id_available = data["patient_id_available"]
    if patient_id_available != "unknown" and type(patient_id_available) is not bool:  # noqa: E721
        raise RegistryValidationError(f"{path.name}: invalid patient_id_available")
    if type(data["redistribution"]) is not bool:  # noqa: E721 - exact schema type
        raise RegistryValidationError(f"{path.name}: redistribution must be a bool")
    if dataset_id == "real_fundus" and (
        data["license_status"] != "ambiguous" or data["license_review_status"] != "pending_review"
    ):
        raise RegistryValidationError(
            f"{path.name}: ambiguous data licensing must remain pending human review"
        )
    return RegistryValidationResult(path=path, dataset_id=dataset_id)


def validate_registry_directory(directory: Path) -> tuple[RegistryValidationResult, ...]:
    if not isinstance(directory, Path) or not directory.is_dir():
        raise NotADirectoryError(directory)
    results = tuple(validate_registry_file(path) for path in sorted(directory.glob("*.yaml")))
    found = {result.dataset_id for result in results}
    if found != REQUIRED_DATASET_IDS:
        raise RegistryValidationError(
            f"registry set mismatch: missing={sorted(REQUIRED_DATASET_IDS - found)!r}, "
            f"unexpected={sorted(found - REQUIRED_DATASET_IDS)!r}"
        )
    return results


__all__ = [
    "REQUIRED_DATASET_IDS",
    "RegistryValidationError",
    "RegistryValidationResult",
    "validate_registry_directory",
    "validate_registry_file",
]
