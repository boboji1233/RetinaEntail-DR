"""Science S4 evidence-gap recovery contract-freeze v3 tooling.

This module safely materializes only DDR lesion-segmentation development
members and audits their read-only adapter plus exact-revision Paired Retina
image hashes.  It never extracts or reads DDR TEST member content and never
decodes a real Paired Retina image.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import stat
import sys
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np

import retinaentail.science_s4_recovery as recovery_preflight
import retinaentail.science_s4_recovery_acquisition as acquisition_v2
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "3.0.0"
EXPERIMENT_ID: Final = "S4_EVIDENCE_GAP_RECOVERY_CONTRACT_V3"
READY_STATUS: Final = "READY_FOR_S4_RECOVERY_IMAGE_AUDIT"
BLOCKED_STATUS: Final = "BLOCKED_PREREQUISITES"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class S4RecoveryContractError(RuntimeError):
    """Raised when the frozen contract cannot be replayed without guessing."""


@dataclass(frozen=True, slots=True)
class S4RecoveryContractResult:
    """Contract-audit disposition and generated artifact hashes."""

    status: str
    output_dir: Path
    artifact_hashes: dict[str, str]


def _mapping(value: object, name: str) -> dict[str, object]:
    try:
        return acquisition_v2._mapping(value, name)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _sequence(value: object, name: str) -> tuple[object, ...]:
    try:
        return acquisition_v2._sequence(value, name)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _string(value: object, name: str) -> str:
    try:
        return acquisition_v2._string(value, name)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    try:
        return acquisition_v2._integer(value, name, minimum=minimum)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S4RecoveryContractError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < minimum:
        raise S4RecoveryContractError(f"{name} is outside the frozen range")
    return rendered


def _load_config(path: Path) -> dict[str, object]:
    try:
        return acquisition_v2._load_config(path)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        return acquisition_v2._load_json(path, name)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_binding(binding: object, name: str) -> tuple[Path, str]:
    try:
        return acquisition_v2._verify_binding(binding, name)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _verify_baseline(value: object) -> None:
    try:
        recovery_preflight._verify_baseline(value)  # noqa: SLF001
    except recovery_preflight.S4RecoveryError as error:
        raise S4RecoveryContractError(str(error)) from error


def _external_root(path: Path, expected_name: str) -> Path:
    if not path.is_absolute():
        raise S4RecoveryContractError("contract root must be an explicit absolute path")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink() or resolved.name != expected_name:
        raise S4RecoveryContractError("contract root is not the frozen external directory")
    return resolved


def _external_leaf(path: Path, root_name: str, leaf_name: str) -> Path:
    try:
        return acquisition_v2._explicit_external_dir(path, root_name, leaf_name)  # noqa: SLF001
    except acquisition_v2.S4RecoveryAcquisitionError as error:
        raise S4RecoveryContractError(str(error)) from error


def _resolve_config(path: Path) -> tuple[Path, dict[str, object]]:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4RecoveryContractError("config path must remain in the project root") from error
    config = _load_config(resolved)
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise S4RecoveryContractError("unsupported S4 recovery contract config")
    if config.get("status") != "EXECUTABLE_CONTRACT_AUDIT":
        raise S4RecoveryContractError("S4 recovery contract audit is not enabled")
    _verify_baseline(config.get("expected_baseline_commit"))
    return resolved, config


def _replay_acquisition(
    config: Mapping[str, object], acquisition_metadata_root: Path
) -> dict[str, object]:
    bindings = _mapping(config.get("bindings"), "bindings")
    required = {
        "contract_prompt",
        "acquisition_config",
        "acquisition_report",
        "acquisition_audit",
        "acquisition_artifact_manifest",
        "acquisition_ddr_inventory",
        "acquisition_pair_metadata_audit",
    }
    if set(bindings) != required:
        raise S4RecoveryContractError("contract bindings changed")
    verified = {
        name: _verify_binding(binding, f"bindings.{name}") for name, binding in bindings.items()
    }
    acquisition_config = _load_config(verified["acquisition_config"][0])
    acquisition_audit = _load_json(verified["acquisition_audit"][0], "acquisition audit")
    if (
        acquisition_audit.get("status") != acquisition_v2.READY_STATUS
        or acquisition_audit.get("acquisition_ready") is not True
        or _mapping(acquisition_audit.get("source_recovery"), "source recovery").get(
            "all_parts_complete"
        )
        is not True
    ):
        raise S4RecoveryContractError("acquisition v2 is not ready for contract freeze")
    artifact_manifest = _load_json(
        verified["acquisition_artifact_manifest"][0], "acquisition artifact manifest"
    )
    artifact_entries = _sequence(artifact_manifest.get("files"), "acquisition artifact files")
    expected_artifacts = {
        "audit.json",
        "ddr-part-inventory.json",
        "tier-b-pair-metadata-audit.json",
    }
    actual_artifacts: set[str] = set()
    artifact_bytes = 0
    artifact_root = verified["acquisition_audit"][0].parent
    for raw in artifact_entries:
        entry = _mapping(raw, "acquisition artifact")
        relative = _string(entry.get("relative_path"), "artifact relative path")
        if relative in actual_artifacts or relative not in expected_artifacts:
            raise S4RecoveryContractError("acquisition artifact inventory changed")
        actual_artifacts.add(relative)
        path = artifact_root / relative
        size = _integer(entry.get("byte_size"), f"artifact {relative} size")
        digest = _string(entry.get("sha256"), f"artifact {relative} hash")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise S4RecoveryContractError(f"acquisition artifact drift: {relative}")
        artifact_bytes += size
    if actual_artifacts != expected_artifacts:
        raise S4RecoveryContractError("acquisition artifact inventory is incomplete")

    metadata_root = _external_leaf(
        acquisition_metadata_root,
        "s4_evidence_gap_recovery_acquisition_v2",
        "tier_b_metadata",
    )
    tier_b = _mapping(acquisition_config.get("tier_b"), "acquisition Tier-B")
    candidate = _mapping(tier_b.get("candidate"), "acquisition Tier-B candidate")
    tree_relative = _string(candidate.get("tree_path"), "acquisition tree path")
    tree_path = metadata_root / Path(tree_relative)
    tree_hash = _string(candidate.get("tree_sha256"), "acquisition tree hash")
    if not tree_path.is_file() or tree_path.is_symlink() or _sha256_file(tree_path) != tree_hash:
        raise S4RecoveryContractError("acquisition repository tree drift")
    pair_audit = _load_json(
        verified["acquisition_pair_metadata_audit"][0], "acquisition pair metadata audit"
    )
    if (
        pair_audit.get("pair_metadata_gate") != "PASS"
        or pair_audit.get("tier_b_status") != "NOT_AVAILABLE"
    ):
        raise S4RecoveryContractError("acquisition pair-metadata disposition changed")
    return {
        "status": "PASS",
        "verified_binding_count": len(verified),
        "acquisition_status": acquisition_v2.READY_STATUS,
        "acquisition_artifact_file_count": len(artifact_entries),
        "acquisition_artifact_byte_count": artifact_bytes,
        "tree_path": str(tree_path),
        "tree_sha256": tree_hash,
        "pair_audit": pair_audit,
    }


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise S4RecoveryContractError("unsafe ZIP member spelling")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise S4RecoveryContractError(f"unsafe ZIP member path: {name}")
    if ":" in pure.parts[0] or pure.parts[0] in {".", ""}:
        raise S4RecoveryContractError(f"unsafe ZIP member root: {name}")
    normalized = pure.as_posix()
    if name.endswith("/"):
        normalized += "/"
    return normalized


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _test_member(name: str) -> bool:
    lowered = name.lower()
    return "/test/" in lowered or lowered.endswith("/test.txt")


def _inspect_archive(archive_path: Path, ddr: Mapping[str, object]) -> dict[str, object]:
    archive = _mapping(ddr.get("archive"), "DDR archive")
    expected_size = _integer(archive.get("byte_size"), "DDR archive byte size", minimum=1)
    expected_hash = _string(archive.get("sha256"), "DDR archive SHA-256")
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size != expected_size
        or _sha256_file(archive_path) != expected_hash
    ):
        raise S4RecoveryContractError("DDR concatenated archive size or hash drift")
    extraction = _mapping(ddr.get("development_extraction"), "DDR development extraction")
    allowed = tuple(
        _string(item, "allowed extraction prefix")
        for item in _sequence(extraction.get("allowed_member_prefixes"), "allowed prefixes")
    )
    maximum_ratio = _number(
        archive.get("maximum_compression_ratio"), "maximum compression ratio", minimum=1.0
    )
    seen: set[str] = set()
    development: list[zipfile.ZipInfo] = []
    test_records: list[dict[str, object]] = []
    member_count = 0
    file_count = 0
    declared_bytes = 0
    encrypted = 0
    symlinks = 0
    with zipfile.ZipFile(archive_path) as bundle:
        infos = bundle.infolist()
        for info in infos:
            name = _safe_member_name(info)
            key = name.casefold()
            if key in seen:
                raise S4RecoveryContractError(f"duplicate normalized ZIP member: {name}")
            seen.add(key)
            member_count += 1
            if info.flag_bits & 1:
                encrypted += 1
            if _zip_is_symlink(info):
                symlinks += 1
            if not info.is_dir():
                file_count += 1
                declared_bytes += info.file_size
                ratio = info.file_size / max(1, info.compress_size)
                if ratio > maximum_ratio:
                    raise S4RecoveryContractError(f"ZIP compression ratio exceeds contract: {name}")
                if any(name.startswith(prefix) for prefix in allowed):
                    development.append(info)
            if _test_member(name):
                test_records.append(
                    {
                        "relative_path": name,
                        "crc32": f"{info.CRC:08x}",
                        "compressed_size": info.compress_size,
                        "uncompressed_size": info.file_size,
                        "is_directory": info.is_dir(),
                    }
                )
    expected = {
        "member_count": _integer(archive.get("expected_member_count"), "member count"),
        "file_count": _integer(archive.get("expected_file_count"), "file count"),
        "declared_bytes": _integer(
            archive.get("expected_declared_uncompressed_bytes"), "declared bytes"
        ),
        "encrypted": _integer(archive.get("encrypted_members"), "encrypted members"),
        "symlinks": _integer(archive.get("symlink_members"), "symlink members"),
    }
    actual = {
        "member_count": member_count,
        "file_count": file_count,
        "declared_bytes": declared_bytes,
        "encrypted": encrypted,
        "symlinks": symlinks,
    }
    if actual != expected:
        raise S4RecoveryContractError("DDR central-directory summary changed")
    test_contract = _mapping(ddr.get("test_central_directory_only"), "DDR test contract")
    expected_test_count = _integer(test_contract.get("expected_member_count"), "test member count")
    if len(test_records) != expected_test_count:
        raise S4RecoveryContractError("DDR TEST central-directory count changed")
    return {
        "archive_relative_path": _string(archive.get("relative_path"), "archive relative path"),
        "archive_byte_size": expected_size,
        "archive_sha256": expected_hash,
        "member_count": member_count,
        "file_count": file_count,
        "declared_uncompressed_bytes": declared_bytes,
        "encrypted_member_count": encrypted,
        "symlink_member_count": symlinks,
        "development_file_count": len(development),
        "development_members": development,
        "test_member_count": len(test_records),
        "test_content_reads": 0,
        "test_records": test_records,
    }


def extract_ddr_development(
    *, config_path: Path, contract_root: Path, development_root: Path
) -> int:
    """Selectively extract only frozen DDR lesion-segmentation development files."""

    _, config = _resolve_config(config_path)
    root_name = _string(config.get("external_root_name"), "external root name")
    root = _external_root(contract_root, root_name)
    output = _external_leaf(development_root, root_name, "ddr_development")
    if any(output.iterdir()):
        raise S4RecoveryContractError("refusing non-empty DDR development output")
    ddr = _mapping(config.get("ddr"), "DDR")
    archive_relative = _string(
        _mapping(ddr.get("archive"), "DDR archive").get("relative_path"),
        "DDR archive relative path",
    )
    archive_path = root / Path(archive_relative)
    inspected = _inspect_archive(archive_path, ddr)
    members = cast(list[zipfile.ZipInfo], inspected.pop("development_members"))
    extracted = 0
    with zipfile.ZipFile(archive_path) as bundle:
        for info in members:
            relative = PurePosixPath(_safe_member_name(info))
            target = output.joinpath(*relative.parts)
            if target.exists():
                raise FileExistsError(f"refusing existing extracted target: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = target.parent.resolve(strict=True)
            try:
                resolved_parent.relative_to(output)
            except ValueError as error:
                raise S4RecoveryContractError("extraction target escaped output root") from error
            with bundle.open(info, "r") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if target.stat().st_size != info.file_size:
                raise S4RecoveryContractError(f"extracted size mismatch: {info.filename}")
            extracted += 1
            if extracted % 250 == 0:
                print(f"DDR development extracted: {extracted}/{len(members)}")
    print(f"DDR development extraction complete: {extracted} files")
    return extracted


def _audit_development_adapter(root: Path, extraction: Mapping[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    file_count = 0
    total_bytes = 0
    seen_sources: set[str] = set()
    labels = tuple(
        _string(item, "DDR lesion label")
        for item in _sequence(extraction.get("lesion_labels"), "DDR lesion labels")
    )
    for split, folder in (("TRAIN", "train"), ("VALID", "valid")):
        base = root / "DDR-dataset" / "lesion_segmentation" / folder
        image_dir = base / "image"
        label_leaf = "label" if split == "TRAIN" else "segmentation label"
        if not image_dir.is_dir() or image_dir.is_symlink():
            raise S4RecoveryContractError(f"DDR {split} image directory is unavailable")
        images = sorted(image_dir.glob("*.jpg"), key=lambda value: value.name)
        split_counts[split] = len(images)
        for image in images:
            source_id = image.stem
            if source_id in seen_sources:
                raise S4RecoveryContractError(f"DDR source crosses split: {source_id}")
            seen_sources.add(source_id)
            image_size = image.stat().st_size
            record_labels: dict[str, object] = {}
            for lesion in labels:
                mask = base / label_leaf / lesion / f"{source_id}.tif"
                if not mask.is_file() or mask.is_symlink():
                    record_labels[lesion] = {"availability": "UNKNOWN", "relative_path": None}
                    continue
                mask_size = mask.stat().st_size
                record_labels[lesion] = {
                    "availability": "AVAILABLE",
                    "relative_path": mask.relative_to(root).as_posix(),
                    "byte_size": mask_size,
                    "sha256": _sha256_file(mask),
                }
                file_count += 1
                total_bytes += mask_size
            records.append(
                {
                    "dataset_id": "ddr",
                    "source_image_id": source_id,
                    "split": split,
                    "image": {
                        "relative_path": image.relative_to(root).as_posix(),
                        "byte_size": image_size,
                        "sha256": _sha256_file(image),
                    },
                    "labels": record_labels,
                }
            )
            file_count += 1
            total_bytes += image_size
    expected_counts = _mapping(extraction.get("expected_source_counts"), "source counts")
    if split_counts != {
        "TRAIN": _integer(expected_counts.get("TRAIN"), "TRAIN source count"),
        "VALID": _integer(expected_counts.get("VALID"), "VALID source count"),
    }:
        raise S4RecoveryContractError("DDR development source counts changed")
    expected_file_counts = _mapping(extraction.get("expected_file_counts"), "file counts")
    expected_total_files = sum(
        _integer(expected_file_counts.get(split), f"{split} file count")
        for split in ("TRAIN", "VALID")
    )
    if file_count != expected_total_files or len(records) != _integer(
        extraction.get("expected_total_source_count"), "total source count"
    ):
        raise S4RecoveryContractError("DDR development adapter inventory is incomplete")
    label_availability = {
        lesion: sum(
            _mapping(_mapping(record["labels"], "record labels")[lesion], "record label").get(
                "availability"
            )
            == "AVAILABLE"
            for record in records
        )
        for lesion in labels
    }
    return {
        "dataset_id": "ddr",
        "dataset_version": "ddr-oia-official-drive-2019-07-10-v1",
        "root": str(root),
        "split_source_counts": split_counts,
        "source_count": len(records),
        "file_count": file_count,
        "total_bytes_hashed": total_bytes,
        "label_availability_counts": label_availability,
        "source_split_leaks": 0,
        "missing_label_policy": "UNKNOWN",
        "records": records,
    }


def select_primary_pairs(
    accepted_pairs: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Choose one pair per subject-eye using only the frozen pair-id hash rule."""

    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    seen_pair_ids: set[str] = set()
    for raw in accepted_pairs:
        pair = dict(raw)
        pair_id = _string(pair.get("pair_id"), "pair ID")
        subject = _string(pair.get("subject_id"), "pair subject")
        eye = _string(pair.get("eye_laterality"), "pair eye")
        if pair_id in seen_pair_ids:
            raise S4RecoveryContractError(f"duplicate pair ID: {pair_id}")
        seen_pair_ids.add(pair_id)
        pair["selection_sha256"] = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
        grouped[(subject, eye)].append(pair)
    primary: list[dict[str, object]] = []
    supplemental: list[dict[str, object]] = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key], key=lambda item: (cast(str, item["selection_sha256"]), item["pair_id"])
        )
        primary.append(ordered[0])
        supplemental.extend(ordered[1:])
    return primary, supplemental


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Apply the frozen IEC sRGB transfer function to a synthetic or decoded array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise S4RecoveryContractError("sRGB values must be finite within [0, 1]")
    return np.where(array <= 0.04045, array / 12.92, ((array + 0.055) / 1.055) ** 2.4)


def conformal_residual_radius(
    unit_residuals: Sequence[Sequence[float]], *, alpha: float, minimum_units: int
) -> tuple[float, ...]:
    """Compute the frozen per-channel finite-sample conformal residual radius."""

    array = np.asarray(unit_residuals, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] < minimum_units
        or array.shape[1] < 1
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or not 0.0 < alpha < 1.0
    ):
        raise S4RecoveryContractError("invalid independent residual calibration matrix")
    rank = math.ceil((array.shape[0] + 1) * (1.0 - alpha))
    if rank > array.shape[0]:
        raise S4RecoveryContractError("finite conformal radius is unavailable at this sample size")
    ordered = np.sort(array, axis=0)
    return tuple(float(value) for value in ordered[rank - 1])


def calibrated_sets_overlap(
    left: Sequence[float], right: Sequence[float], radius: Sequence[float]
) -> bool:
    """Return whether channelwise symmetric calibrated acquisition-error sets overlap."""

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    bound = np.asarray(radius, dtype=np.float64)
    if (
        lhs.ndim != 1
        or lhs.shape != rhs.shape
        or lhs.shape != bound.shape
        or not np.isfinite(lhs).all()
        or not np.isfinite(rhs).all()
        or not np.isfinite(bound).all()
        or np.any(bound < 0.0)
    ):
        raise S4RecoveryContractError("invalid calibrated overlap witness inputs")
    return bool(np.all(np.abs(lhs - rhs) <= 2.0 * bound))


def _audit_paired_images(
    *,
    snapshot_root: Path,
    tree_path: Path,
    pair_audit: Mapping[str, object],
    contract: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    tree = _load_json(tree_path, "Paired Retina repository tree")
    files = _mapping(tree.get("files"), "Paired Retina tree files")
    image_entries = {
        path: _mapping(metadata, f"tree metadata for {path}")
        for path, metadata in files.items()
        if path.lower().endswith(".jpg")
    }
    expected_count = _integer(contract.get("expected_public_image_count"), "public image count")
    if len(image_entries) != expected_count:
        raise S4RecoveryContractError("Paired Retina public image count changed")
    records: list[dict[str, object]] = []
    total_bytes = 0
    for relative in sorted(image_entries):
        metadata = image_entries[relative]
        path = snapshot_root.joinpath(*PurePosixPath(relative).parts)
        expected_size = _integer(metadata.get("lfs_size"), f"{relative} LFS size", minimum=1)
        expected_hash = _string(metadata.get("lfs_sha256"), f"{relative} LFS SHA-256")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_size
            or _sha256_file(path) != expected_hash
        ):
            raise S4RecoveryContractError(f"Paired Retina image hash drift: {relative}")
        device = "eyer" if "/eyer/" in relative else "visucam" if "/visucam/" in relative else None
        if device is None:
            raise S4RecoveryContractError(f"unknown Paired Retina device path: {relative}")
        records.append(
            {
                "relative_path": relative,
                "device": device,
                "byte_size": expected_size,
                "sha256": expected_hash,
            }
        )
        total_bytes += expected_size
    expected_bytes = _integer(contract.get("expected_public_image_bytes"), "public image bytes")
    if total_bytes != expected_bytes:
        raise S4RecoveryContractError("Paired Retina public image byte count changed")
    actual_snapshot_images = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*.jpg")
        if path.is_file()
    }
    if actual_snapshot_images != set(image_entries):
        raise S4RecoveryContractError("Paired Retina snapshot image inventory changed")

    accepted_raw = _sequence(pair_audit.get("accepted_pairs"), "accepted pairs")
    accepted = [_mapping(item, "accepted pair") for item in accepted_raw]
    primary, supplemental = select_primary_pairs(accepted)
    if len(accepted) != _integer(
        contract.get("expected_accepted_pair_count"), "accepted pair count"
    ) or len(primary) != _integer(
        contract.get("expected_independent_subject_eye_count"), "subject-eye count"
    ):
        raise S4RecoveryContractError("Paired Retina pair contract counts changed")
    known_images = set(image_entries)
    for pair in accepted:
        for field in ("visucam_image_path", "eyer_image_path"):
            if _string(pair.get(field), field) not in known_images:
                raise S4RecoveryContractError("accepted pair references an unavailable image")
    pair_contract = {
        "repository_revision": _string(contract.get("repository_revision"), "repository revision"),
        "selection_rule": "MIN_SHA256_UTF8_PAIR_ID_PER_SUBJECT_EYE",
        "split": "CALIBRATION_ONLY",
        "accepted_pair_count": len(accepted),
        "primary_unit_count": len(primary),
        "supplemental_pair_count": len(supplemental),
        "minimum_primary_unit_count": _integer(
            contract.get("minimum_independent_subject_eye_count"), "minimum subject-eye count"
        ),
        "primary_pairs": primary,
        "supplemental_pairs": supplemental,
        "real_image_decodes": 0,
    }
    image_manifest = {
        "repository": _string(contract.get("repository"), "repository"),
        "repository_revision": pair_contract["repository_revision"],
        "snapshot_root": str(snapshot_root),
        "image_count": len(records),
        "total_bytes_hashed": total_bytes,
        "eyer_image_count": sum(record["device"] == "eyer" for record in records),
        "visucam_image_count": sum(record["device"] == "visucam" for record in records),
        "real_image_decodes": 0,
        "files": records,
    }
    if image_manifest["eyer_image_count"] != _integer(
        contract.get("expected_eyer_image_count"), "Eyer image count"
    ) or image_manifest["visucam_image_count"] != _integer(
        contract.get("expected_visucam_image_count"), "Visucam image count"
    ):
        raise S4RecoveryContractError("Paired Retina device counts changed")
    return image_manifest, pair_contract


def _contract_frozen(config: Mapping[str, object]) -> bool:
    image = _mapping(config.get("image_contract"), "image contract")
    registration = _mapping(image.get("registration"), "registration contract")
    overlap = _mapping(image.get("overlap_witness"), "overlap witness contract")
    paired = _mapping(config.get("paired_retina"), "Paired Retina contract")
    selection = _mapping(paired.get("primary_selection"), "primary selection")
    return (
        image.get("decoded_mode") == "RGB"
        and image.get("srgb_linearization") == "IEC_61966_2_1"
        and registration.get("fixed_device") == "visucam"
        and registration.get("moving_device") == "eyer"
        and registration.get("reflection_allowed") is False
        and overlap.get("status") == "FROZEN_IMPLEMENTATION_REQUIRED_BEFORE_REAL_IMAGE_AUDIT"
        and selection.get("rule") == "MIN_SHA256_UTF8_PAIR_ID_PER_SUBJECT_EYE"
        and selection.get("split") == "CALIBRATION_ONLY"
    )


def run_contract_audit(
    *,
    config_path: Path,
    contract_root: Path,
    acquisition_metadata_root: Path,
    output_dir: Path,
) -> S4RecoveryContractResult:
    """Audit the frozen v3 data contract without real-image decoding."""

    resolved_config, config = _resolve_config(config_path)
    frozen_output = (PROJECT_ROOT / _string(config.get("artifact_root"), "artifact root")).resolve()
    if output_dir.resolve() != frozen_output:
        raise S4RecoveryContractError("output directory changed")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite S4 contract output: {output_dir}")
    root_name = _string(config.get("external_root_name"), "external root name")
    root = _external_root(contract_root, root_name)
    upstream = _replay_acquisition(config, acquisition_metadata_root)
    ddr = _mapping(config.get("ddr"), "DDR")
    archive_relative = _string(
        _mapping(ddr.get("archive"), "DDR archive").get("relative_path"), "archive relative path"
    )
    central = _inspect_archive(root / Path(archive_relative), ddr)
    central.pop("development_members")
    development = _audit_development_adapter(
        _external_leaf(root / "ddr_development", root_name, "ddr_development"),
        _mapping(ddr.get("development_extraction"), "DDR development extraction"),
    )
    paired = _mapping(config.get("paired_retina"), "Paired Retina")
    snapshot = root / Path(_string(paired.get("snapshot_relative_path"), "snapshot path"))
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise S4RecoveryContractError("Paired Retina snapshot is unavailable")
    tree_path = Path(_string(upstream["tree_path"], "tree path"))
    image_manifest, pair_contract = _audit_paired_images(
        snapshot_root=snapshot,
        tree_path=tree_path,
        pair_audit=_mapping(upstream["pair_audit"], "pair audit"),
        contract=paired,
    )
    contract_frozen = _contract_frozen(config)
    ready = (
        development["source_count"] == 532
        and image_manifest["image_count"] == 399
        and pair_contract["primary_unit_count"] == 96
        and contract_frozen
    )
    status = READY_STATUS if ready else BLOCKED_STATUS
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "contract_frozen": contract_frozen,
        "s4_overall_status": BLOCKED_STATUS,
        "config_sha256": _sha256_file(resolved_config),
        "upstream_replay": {key: value for key, value in upstream.items() if key != "pair_audit"},
        "ddr": {
            "archive_hash_verified": True,
            "central_directory_member_count": central["member_count"],
            "development_source_count": development["source_count"],
            "development_file_count": development["file_count"],
            "source_split_leaks": development["source_split_leaks"],
            "test_central_directory_member_count": central["test_member_count"],
            "test_content_reads": 0,
            "development_adapter_status": "READY",
        },
        "tier_b": {
            "public_image_count": image_manifest["image_count"],
            "accepted_pair_count": pair_contract["accepted_pair_count"],
            "primary_unit_count": pair_contract["primary_unit_count"],
            "minimum_primary_unit_count": pair_contract["minimum_primary_unit_count"],
            "real_image_decodes": 0,
            "registration_status": "FROZEN_NOT_RUN",
            "overlap_witness_status": "FROZEN_NOT_FIT",
            "tier_b_status": "NOT_AVAILABLE",
        },
        "blockers": [
            "TIER_B_REAL_IMAGE_REGISTRATION_NOT_RUN",
            "TIER_B_CALIBRATED_OVERLAP_WITNESS_NOT_FIT",
            "DDR_RECOVERY_TWIN_NOT_RUN",
        ],
        "io_audit": {
            "ddr_archive_hash_reads": 1,
            "ddr_central_directory_reads": 1,
            "ddr_development_file_hash_reads": development["file_count"],
            "ddr_development_bytes_hashed": development["total_bytes_hashed"],
            "ddr_test_content_reads": 0,
            "ddr_test_image_decodes": 0,
            "paired_retina_file_hash_reads": image_manifest["image_count"],
            "paired_retina_bytes_hashed": image_manifest["total_bytes_hashed"],
            "paired_retina_image_decodes": 0,
            "locked_test_root_reads": 0,
            "locked_test_image_decodes": 0,
        },
        "downstream_authorization": {
            "s4_recovery_image_audit": ready,
            "s4_recovery_twin_execution": False,
            "s5_degradation_calibration": False,
            "s6_verify": False,
            "s7_lite": False,
            "enhancer_training": False,
            "locked_test_evaluation": False,
            "fcr_estimation": False,
            "paper_performance_experiment": False,
        },
        "evidence_boundary": [
            "Contract readiness is not Tier-B availability or Science S4 PASS.",
            "No Paired Retina image or DDR TEST member was decoded.",
            "Provider-asserted simultaneity has no per-image acquisition timestamp.",
            "Counts and synthetic fixtures are not FCR, medical, clinical, or "
            "paper-performance results.",
        ],
    }
    payloads: dict[str, bytes] = {
        "ddr-central-directory-audit.json": canonical_json_bytes(
            cast(dict[str, JSONValue], central)
        ),
        "ddr-development-adapter-manifest.json": canonical_json_bytes(
            cast(dict[str, JSONValue], development)
        ),
        "paired-retina-image-manifest.json": canonical_json_bytes(
            cast(dict[str, JSONValue], image_manifest)
        ),
        "tier-b-primary-pair-contract.json": canonical_json_bytes(
            cast(dict[str, JSONValue], pair_contract)
        ),
        "audit.json": canonical_json_bytes(cast(dict[str, JSONValue], audit)),
    }
    artifact_manifest = {
        "files": [
            {
                "relative_path": name,
                "byte_size": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, payload in sorted(payloads.items())
        ]
    }
    artifact_bytes = canonical_json_bytes(cast(dict[str, JSONValue], artifact_manifest))
    for name, payload in payloads.items():
        write_bytes_fail_safe(output_dir / name, payload)
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", artifact_bytes)
    hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    hashes["artifact-manifest.json"] = sha256_bytes(artifact_bytes)
    return S4RecoveryContractResult(status=status, output_dir=output_dir, artifact_hashes=hashes)


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded contract-v3 CLI."""

    parser = argparse.ArgumentParser(description="Science S4 recovery contract freeze v3.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-ddr")
    extract.add_argument("--config", type=Path, required=True)
    extract.add_argument("--contract-root", type=Path, required=True)
    extract.add_argument("--development-root", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--contract-root", type=Path, required=True)
    audit.add_argument("--acquisition-metadata-root", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded extraction or contract audit."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract-ddr":
            extract_ddr_development(
                config_path=cast(Path, args.config),
                contract_root=cast(Path, args.contract_root),
                development_root=cast(Path, args.development_root),
            )
            return 0
        result = run_contract_audit(
            config_path=cast(Path, args.config),
            contract_root=cast(Path, args.contract_root),
            acquisition_metadata_root=cast(Path, args.acquisition_metadata_root),
            output_dir=cast(Path, args.output_dir),
        )
    except (FileExistsError, OSError, S4RecoveryContractError, zipfile.BadZipFile) as error:
        print(f"Science S4 recovery contract fail-closed: {error}", file=sys.stderr)
        return 1
    print(f"Science S4 recovery contract status: {result.status}")
    for name, digest in result.artifact_hashes.items():
        print(f"{name}: {digest}")
    return 0 if result.status == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_STATUS",
    "READY_STATUS",
    "S4RecoveryContractError",
    "S4RecoveryContractResult",
    "build_parser",
    "calibrated_sets_overlap",
    "conformal_residual_radius",
    "extract_ddr_development",
    "main",
    "run_contract_audit",
    "select_primary_pairs",
    "srgb_to_linear",
]
