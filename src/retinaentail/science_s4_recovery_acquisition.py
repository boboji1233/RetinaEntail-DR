"""Science S4 evidence-gap recovery acquisition continuation v2 audit.

The audit hashes official DDR split parts and validates Paired Retina row-level
pair metadata against the exact public repository tree.  It does not concatenate
or extract DDR, decode images, or authorize twins or downstream experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import Final, cast

import retinaentail.science_s4_recovery as recovery_preflight
import retinaentail.science_s4_recovery_execution as execution_v1
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "2.0.0"
EXPERIMENT_ID: Final = "S4_EVIDENCE_GAP_RECOVERY_ACQUISITION_V2"
BLOCKED_STATUS: Final = "BLOCKED_PREREQUISITES"
READY_STATUS: Final = "READY_FOR_S4_RECOVERY_CONTRACT_FREEZE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
_EXCEL_EPOCH: Final = date(1899, 12, 30)


class S4RecoveryAcquisitionError(RuntimeError):
    """Raised when acquisition evidence cannot be verified without guessing."""


@dataclass(frozen=True, slots=True)
class S4RecoveryAcquisitionResult:
    """Generated acquisition disposition and artifact hashes."""

    status: str
    output_dir: Path
    artifact_hashes: dict[str, str]


def _mapping(value: object, name: str) -> dict[str, object]:
    try:
        return execution_v1._mapping(value, name)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _sequence(value: object, name: str) -> tuple[object, ...]:
    try:
        return execution_v1._sequence(value, name)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _string(value: object, name: str) -> str:
    try:
        return execution_v1._string(value, name)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    try:
        return execution_v1._integer(value, name, minimum=minimum)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, object]:
    try:
        return execution_v1._load_config(path)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _verify_baseline(value: object) -> None:
    try:
        recovery_preflight._verify_baseline(value)  # noqa: SLF001
    except recovery_preflight.S4RecoveryError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        return execution_v1._load_json(path, name)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _verify_binding(binding: object, name: str) -> tuple[Path, str]:
    try:
        return execution_v1._verify_binding(binding, name)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _explicit_external_dir(path: Path, root_name: str, leaf_name: str) -> Path:
    try:
        return execution_v1._explicit_external_dir(path, root_name, leaf_name)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error


def _replay_execution_v1(config: Mapping[str, object]) -> dict[str, object]:
    bindings = _mapping(config.get("bindings"), "bindings")
    required = {
        "acquisition_prompt",
        "execution_v1_config",
        "execution_v1_report",
        "execution_v1_audit",
        "execution_v1_artifact_manifest",
    }
    if set(bindings) != required:
        raise S4RecoveryAcquisitionError("acquisition bindings changed")
    verified = {
        name: _verify_binding(binding, f"bindings.{name}") for name, binding in bindings.items()
    }
    v1_config = _load_config(verified["execution_v1_config"][0])
    try:
        upstream = execution_v1._replay_preflight(v1_config)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error
    audit = _load_json(verified["execution_v1_audit"][0], "execution v1 audit")
    if audit.get("status") != BLOCKED_STATUS:
        raise S4RecoveryAcquisitionError("execution v1 disposition must remain blocked")
    manifest = _load_json(
        verified["execution_v1_artifact_manifest"][0], "execution v1 artifact manifest"
    )
    entries = _sequence(manifest.get("files"), "execution v1 artifact files")
    expected_names = {
        "audit.json",
        "ddr-part-inventory.json",
        "tier-b-candidate-inventory.json",
    }
    actual_names: set[str] = set()
    artifact_root = verified["execution_v1_audit"][0].parent
    artifact_bytes = 0
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"execution v1 artifact {index}")
        relative = _string(entry.get("relative_path"), f"artifact {index} path")
        if relative not in expected_names or relative in actual_names:
            raise S4RecoveryAcquisitionError("execution v1 artifact inventory changed")
        actual_names.add(relative)
        path = artifact_root / relative
        size = _integer(entry.get("byte_size"), f"artifact {relative} size")
        digest = _string(entry.get("sha256"), f"artifact {relative} hash")
        if not path.is_file() or path.is_symlink():
            raise S4RecoveryAcquisitionError(f"missing execution v1 artifact: {relative}")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise S4RecoveryAcquisitionError(f"execution v1 artifact drift: {relative}")
        artifact_bytes += size
    if actual_names != expected_names:
        raise S4RecoveryAcquisitionError("execution v1 artifact inventory is incomplete")
    return {
        "status": "PASS",
        "verified_binding_count": len(verified),
        "execution_v1_status": BLOCKED_STATUS,
        "execution_v1_artifact_file_count": len(entries),
        "execution_v1_artifact_byte_count": artifact_bytes,
        "preflight_and_s4_v1_replay": upstream,
    }


def _audit_ddr_parts(root: Path, source: Mapping[str, object]) -> dict[str, object]:
    parts = _sequence(source.get("parts"), "DDR parts")
    expected_count = _integer(source.get("expected_part_count"), "DDR expected count")
    if len(parts) != expected_count:
        raise S4RecoveryAcquisitionError("DDR expected part count changed")
    entries = tuple(root.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise S4RecoveryAcquisitionError("DDR parts root must contain only regular files")
    expected_names: set[str] = set()
    records: list[dict[str, object]] = []
    complete_count = 0
    total_bytes = 0
    for index, raw in enumerate(parts):
        part = _mapping(raw, f"DDR part {index}")
        name = _string(part.get("name"), f"DDR part {index} name")
        file_id = _string(part.get("file_id"), f"DDR part {index} file ID")
        expected_hash = _string(part.get("sha256"), f"DDR part {index} SHA-256")
        expected_size = _integer(part.get("byte_size"), f"DDR part {index} byte size", minimum=1)
        if name in expected_names:
            raise S4RecoveryAcquisitionError(f"duplicate DDR part name: {name}")
        expected_names.add(name)
        path = root / name
        record: dict[str, object] = {
            "name": name,
            "file_id": file_id,
            "expected_byte_size": expected_size,
            "expected_sha256": expected_hash,
        }
        if not path.is_file():
            record.update({"status": "MISSING", "byte_size": 0})
        else:
            size = path.stat().st_size
            digest = _sha256_file(path) if size > 0 else None
            matches = size == expected_size and digest == expected_hash
            record.update(
                {
                    "status": "COMPLETE_LOCAL_HASHED" if matches else "SIZE_OR_HASH_MISMATCH",
                    "byte_size": size,
                    "sha256": digest,
                }
            )
            if matches:
                complete_count += 1
                total_bytes += size
        records.append(record)
    unexpected = sorted(entry.name for entry in entries if entry.name not in expected_names)
    partials = [name for name in unexpected if name.endswith(".part")]
    return {
        "dataset_id": "ddr",
        "dataset_version": _string(source.get("dataset_version"), "DDR dataset version"),
        "root": str(root),
        "root_entry_count": len(entries),
        "expected_part_count": expected_count,
        "complete_part_count": complete_count,
        "total_complete_bytes": total_bytes,
        "unexpected_files": unexpected,
        "partial_files": partials,
        "all_parts_complete": complete_count == expected_count and not unexpected,
        "archive_concatenated": False,
        "archive_extracted": False,
        "parts": records,
    }


def _normalize_subject_id(value: str) -> str:
    try:
        number = float(value)
    except ValueError as error:
        raise S4RecoveryAcquisitionError("Paired Retina subject ID is not numeric") from error
    integer = int(number)
    if number != integer or integer <= 0:
        raise S4RecoveryAcquisitionError("Paired Retina subject ID is not a positive integer")
    return str(integer)


def _normalize_extended_image_id(value: str) -> str:
    """Recover an image id, including Excel-coerced 2024 day/month values."""

    try:
        number = float(value)
    except ValueError as error:
        raise S4RecoveryAcquisitionError("Paired Retina image ID is not numeric") from error
    if number >= 1000:
        converted = _EXCEL_EPOCH + timedelta(days=int(number))
        if converted.year != 2024:
            raise S4RecoveryAcquisitionError("unexpected Excel-coerced image ID year")
        return f"{converted.day}.{converted.month}"
    rendered = str(number)
    return rendered.removesuffix(".0")


def _load_hf_tree(path: Path, revision: str) -> set[str]:
    tree = _load_json(path, "Hugging Face repository tree")
    if tree.get("format_version") != 1:
        raise S4RecoveryAcquisitionError("unsupported Hugging Face tree format")
    files = _mapping(tree.get("files"), "Hugging Face tree files")
    if path.stem != revision:
        raise S4RecoveryAcquisitionError("Hugging Face tree revision path changed")
    paths = set(files)
    if any(
        PurePosixPath(item).is_absolute() or ".." in PurePosixPath(item).parts for item in paths
    ):
        raise S4RecoveryAcquisitionError("unsafe Hugging Face tree path")
    return paths


def _audit_pair_metadata(root: Path, tier_b: Mapping[str, object]) -> dict[str, object]:
    candidate = _mapping(tier_b.get("candidate"), "Tier-B candidate")
    revision = _string(candidate.get("repository_revision"), "repository revision")
    workbook_relative = _string(candidate.get("workbook_path"), "workbook path")
    tree_relative = _string(candidate.get("tree_path"), "tree path")
    readme_relative = _string(candidate.get("readme_path"), "README path")
    workbook = root / Path(workbook_relative)
    tree_path = root / Path(tree_relative)
    readme = root / Path(readme_relative)
    for path, label in (
        (workbook, "workbook"),
        (tree_path, "repository tree"),
        (readme, "README"),
    ):
        if not path.is_file() or path.is_symlink():
            raise S4RecoveryAcquisitionError(f"Paired Retina {label} is missing")
    workbook_hash = _sha256_file(workbook)
    tree_hash = _sha256_file(tree_path)
    readme_hash = _sha256_file(readme)
    if workbook_hash != _string(candidate.get("workbook_sha256"), "workbook SHA-256"):
        raise S4RecoveryAcquisitionError("Paired Retina workbook hash drift")
    if tree_hash != _string(candidate.get("tree_sha256"), "tree SHA-256"):
        raise S4RecoveryAcquisitionError("Paired Retina tree hash drift")
    if readme_hash != _string(candidate.get("readme_sha256"), "README SHA-256"):
        raise S4RecoveryAcquisitionError("Paired Retina README hash drift")
    readme_text = readme.read_text(encoding="utf-8")
    required_claims = (
        "taken simultaneously under identical conditions",
        "Each record contains matched images from both devices",
    )
    if any(claim not in readme_text for claim in required_claims):
        raise S4RecoveryAcquisitionError("Paired Retina provider pair assertion changed")
    try:
        rows = execution_v1._xlsx_rows(workbook)  # noqa: SLF001
    except execution_v1.S4RecoveryExecutionError as error:
        raise S4RecoveryAcquisitionError(str(error)) from error
    public_paths = _load_hf_tree(tree_path, revision)
    eyer_paths = {path for path in public_paths if "/eyer/" in path and path.endswith(".jpg")}
    visucam_paths = {path for path in public_paths if "/visucam/" in path and path.endswith(".jpg")}
    eyer_by_name = {PurePosixPath(path).name: path for path in eyer_paths}
    visucam_by_name = {PurePosixPath(path).name: path for path in visucam_paths}
    if len(eyer_by_name) != len(eyer_paths) or len(visucam_by_name) != len(visucam_paths):
        raise S4RecoveryAcquisitionError("public image basenames are not unique")

    required = (
        "pat_id",
        "extended_img_id_visucam",
        "camera_visucam",
        "extended_img_id_eyer",
        "camera_eyer",
        "eye",
        "center",
    )
    available_headers = set(rows[0]) if rows else set()
    missing_headers = sorted(set(required) - available_headers)
    if missing_headers:
        raise S4RecoveryAcquisitionError("Paired Retina workbook lacks required headers")
    accepted: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    used_pair_keys: set[tuple[str, str, str]] = set()
    used_images: set[str] = set()
    matched_public_images: set[str] = set()
    metadata_subjects: set[str] = set()
    metadata_subject_eyes: set[tuple[str, str]] = set()
    all_subjects: set[str] = set()
    all_subject_eyes: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        if any(not row[field].strip() for field in required):
            rejection_counts["MISSING_REQUIRED_FIELD"] += 1
            continue
        subject = _normalize_subject_id(row["pat_id"])
        eye = row["eye"].strip().lower()
        center = row["center"].strip().lower()
        if eye not in {"left", "right"}:
            raise S4RecoveryAcquisitionError("unknown Paired Retina laterality")
        if center not in {"macula", "optic"}:
            raise S4RecoveryAcquisitionError("unknown Paired Retina center")
        if row["camera_visucam"].strip().lower() != "visucam":
            raise S4RecoveryAcquisitionError("Visucam device role changed")
        if row["camera_eyer"].strip().lower() != "eyer":
            raise S4RecoveryAcquisitionError("Eyer device role changed")
        metadata_subjects.add(subject)
        metadata_subject_eyes.add((subject, eye))
        visucam_id = _normalize_extended_image_id(row["extended_img_id_visucam"])
        eyer_id = _normalize_extended_image_id(row["extended_img_id_eyer"])
        visucam_name = f"{visucam_id}.visucam.jpg"
        eyer_name = f"{eyer_id}.eyer.jpg"
        visucam_path = visucam_by_name.get(visucam_name)
        eyer_path = eyer_by_name.get(eyer_name)
        if visucam_path is not None:
            matched_public_images.add(visucam_path)
        if eyer_path is not None:
            matched_public_images.add(eyer_path)
        if visucam_path is None and eyer_path is None:
            rejection_counts["PAIR_NOT_IN_PUBLIC_IMAGE_SUBSET"] += 1
            continue
        if visucam_path is None:
            rejection_counts["MISSING_VISUCAM_PUBLIC_IMAGE"] += 1
            continue
        if eyer_path is None:
            rejection_counts["MISSING_EYER_PUBLIC_IMAGE"] += 1
            continue
        pair_key = (subject, eye, center)
        if pair_key in used_pair_keys:
            raise S4RecoveryAcquisitionError("duplicate subject/eye/center pair key")
        if visucam_path in used_images or eyer_path in used_images:
            raise S4RecoveryAcquisitionError("Paired Retina public image is reused")
        used_pair_keys.add(pair_key)
        used_images.update((visucam_path, eyer_path))
        all_subjects.add(subject)
        all_subject_eyes.add((subject, eye))
        accepted.append(
            {
                "row_number": index,
                "pair_id": f"{subject}:{eye}:{center}",
                "subject_id": subject,
                "eye_laterality": eye,
                "center": center,
                "tabletop_device": "visucam",
                "portable_device": "eyer",
                "visucam_image_path": visucam_path,
                "eyer_image_path": eyer_path,
            }
        )
    public_image_count = len(eyer_paths) + len(visucam_paths)
    unmatched_public = sorted((eyer_paths | visucam_paths) - matched_public_images)
    minimum = _integer(tier_b.get("minimum_independent_pairs"), "Tier-B minimum", minimum=1)
    pair_gate_pass = len(all_subject_eyes) >= minimum and not unmatched_public
    conditions = {
        "registration": _string(tier_b.get("registration_status"), "registration status"),
        "domain": _string(tier_b.get("domain_status"), "domain status"),
        "threshold": _string(tier_b.get("threshold_status"), "threshold status"),
        "overlap_witness": _string(tier_b.get("overlap_witness_status"), "overlap witness status"),
    }
    conditions_frozen = conditions == {
        "registration": "FROZEN",
        "domain": "FROZEN",
        "threshold": "FROZEN",
        "overlap_witness": "IMPLEMENTED",
    }
    return {
        "dataset_id": "paired_retina",
        "repository": _string(candidate.get("repository"), "repository"),
        "repository_revision": revision,
        "workbook": {
            "relative_path": workbook_relative,
            "byte_size": workbook.stat().st_size,
            "sha256": workbook_hash,
            "sheet_name": "paired_dataframe",
            "row_count": len(rows),
            "complete_required_row_count": len(rows) - rejection_counts["MISSING_REQUIRED_FIELD"],
            "declared_patient_count": _integer(
                candidate.get("declared_patient_count"), "declared patient count"
            ),
            "observed_patient_id_count": len(metadata_subjects),
            "observed_subject_eye_count": len(metadata_subject_eyes),
        },
        "repository_tree": {
            "relative_path": tree_relative,
            "byte_size": tree_path.stat().st_size,
            "sha256": tree_hash,
            "total_file_count": len(public_paths),
            "public_eyer_image_count": len(eyer_paths),
            "public_visucam_image_count": len(visucam_paths),
            "public_image_count": public_image_count,
        },
        "provider_pair_assertion": {
            "relative_path": readme_relative,
            "byte_size": readme.stat().st_size,
            "sha256": readme_hash,
            "claims_bound": list(required_claims),
            "timestamp_fields_present_in_workbook": False,
        },
        "accepted_pair_count": len(accepted),
        "accepted_patient_count": len(all_subjects),
        "independent_subject_eye_count": len(all_subject_eyes),
        "minimum_independent_pairs": minimum,
        "rejected_row_count": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "unmatched_public_image_count": len(unmatched_public),
        "unmatched_public_images": unmatched_public,
        "pair_metadata_gate": "PASS" if pair_gate_pass else "FAIL",
        "simultaneity_evidence": "PROVIDER_ASSERTION_BOUND_TO_EXACT_ROW_PAIR_REVISION",
        "simultaneity_timestamp_verified": False,
        "conditions": conditions,
        "conditions_frozen": conditions_frozen,
        "tier_b_status": "AVAILABLE" if pair_gate_pass and conditions_frozen else "NOT_AVAILABLE",
        "accepted_pairs": accepted,
    }


def run_acquisition_audit(
    *, config_path: Path, ddr_parts_root: Path, tier_b_metadata_root: Path, output_dir: Path
) -> S4RecoveryAcquisitionResult:
    """Audit v2 acquisition evidence without archive extraction or image decoding."""

    resolved_config = config_path.resolve(strict=True)
    try:
        resolved_config.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4RecoveryAcquisitionError("config path must remain in the project root") from error
    config = _load_config(resolved_config)
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise S4RecoveryAcquisitionError("unsupported S4 recovery acquisition config")
    if config.get("status") != "EXECUTABLE_ACQUISITION_AUDIT":
        raise S4RecoveryAcquisitionError("S4 acquisition audit is not enabled")
    _verify_baseline(config.get("expected_baseline_commit"))
    frozen_output = (PROJECT_ROOT / _string(config.get("artifact_root"), "artifact root")).resolve()
    if output_dir.resolve() != frozen_output:
        raise S4RecoveryAcquisitionError("output directory changed")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite S4 acquisition output: {output_dir}")
    root_name = _string(config.get("external_root_name"), "external root name")
    ddr_root = _explicit_external_dir(ddr_parts_root, root_name, "ddr_parts")
    tier_root = _explicit_external_dir(tier_b_metadata_root, root_name, "tier_b_metadata")
    upstream = _replay_execution_v1(config)
    ddr = _audit_ddr_parts(ddr_root, _mapping(config.get("source_recovery"), "source recovery"))
    tier_b = _audit_pair_metadata(tier_root, _mapping(config.get("tier_b"), "Tier-B"))
    acquisition_ready = ddr["all_parts_complete"] is True and tier_b["pair_metadata_gate"] == "PASS"
    status = READY_STATUS if acquisition_ready else BLOCKED_STATUS
    blockers: list[str] = []
    if ddr["all_parts_complete"] is not True:
        blockers.append("DDR_TRANSFER_AND_LOCAL_SHA256_INCOMPLETE")
    else:
        blockers.append("DDR_DEVELOPMENT_ADAPTER_NOT_BUILT")
    if tier_b["pair_metadata_gate"] != "PASS":
        blockers.append("INDEPENDENT_TIER_B_PAIR_METADATA_INSUFFICIENT")
    else:
        blockers.append("TIER_B_PAIR_IMAGES_NOT_ACQUIRED")
    if tier_b["tier_b_status"] != "AVAILABLE":
        blockers.append("TIER_B_CONTRACT_CONDITIONS_NOT_FROZEN")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "acquisition_ready": acquisition_ready,
        "s4_overall_status": BLOCKED_STATUS,
        "config_sha256": _sha256_file(resolved_config),
        "upstream_replay": upstream,
        "source_recovery": {
            "complete_part_count": ddr["complete_part_count"],
            "expected_part_count": ddr["expected_part_count"],
            "total_complete_bytes": ddr["total_complete_bytes"],
            "all_parts_complete": ddr["all_parts_complete"],
            "archive_concatenated": False,
            "archive_extracted": False,
            "development_adapter_built": False,
        },
        "tier_b_metadata": {
            "accepted_pair_count": tier_b["accepted_pair_count"],
            "independent_subject_eye_count": tier_b["independent_subject_eye_count"],
            "minimum_independent_pairs": tier_b["minimum_independent_pairs"],
            "pair_metadata_gate": tier_b["pair_metadata_gate"],
            "conditions_frozen": tier_b["conditions_frozen"],
            "tier_b_status": tier_b["tier_b_status"],
        },
        "blockers": blockers,
        "io_audit": {
            "ddr_part_file_reads": ddr["complete_part_count"],
            "ddr_part_bytes_hashed": ddr["total_complete_bytes"],
            "ddr_archive_concatenations": 0,
            "ddr_archive_extractions": 0,
            "ddr_dataset_root_reads": 0,
            "ddr_image_decodes": 0,
            "tier_b_metadata_file_reads": 3,
            "tier_b_image_decodes": 0,
            "fgadr_root_reads": 0,
            "locked_test_root_reads": 0,
            "locked_test_image_decodes": 0,
        },
        "downstream_authorization": {
            "s4_recovery_contract_freeze": acquisition_ready,
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
            "Acquisition readiness is not Science S4 PASS or Tier-B availability.",
            "Provider-asserted simultaneity is bound to exact row pairs but has no "
            "acquisition timestamps.",
            "No DDR archive was concatenated or extracted and no image was decoded.",
            "Counts and synthetic fixtures are not FCR, medical, clinical, or "
            "paper-performance results.",
        ],
    }
    ddr_bytes = canonical_json_bytes(cast(dict[str, JSONValue], ddr))
    tier_bytes = canonical_json_bytes(cast(dict[str, JSONValue], tier_b))
    audit_bytes = canonical_json_bytes(cast(dict[str, JSONValue], audit))
    artifact_manifest = {
        "files": [
            {
                "relative_path": "audit.json",
                "byte_size": len(audit_bytes),
                "sha256": sha256_bytes(audit_bytes),
            },
            {
                "relative_path": "ddr-part-inventory.json",
                "byte_size": len(ddr_bytes),
                "sha256": sha256_bytes(ddr_bytes),
            },
            {
                "relative_path": "tier-b-pair-metadata-audit.json",
                "byte_size": len(tier_bytes),
                "sha256": sha256_bytes(tier_bytes),
            },
        ]
    }
    artifact_bytes = canonical_json_bytes(cast(dict[str, JSONValue], artifact_manifest))
    write_bytes_fail_safe(output_dir / "ddr-part-inventory.json", ddr_bytes)
    write_bytes_fail_safe(output_dir / "tier-b-pair-metadata-audit.json", tier_bytes)
    write_bytes_fail_safe(output_dir / "audit.json", audit_bytes)
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", artifact_bytes)
    return S4RecoveryAcquisitionResult(
        status=status,
        output_dir=output_dir,
        artifact_hashes={
            "ddr-part-inventory.json": sha256_bytes(ddr_bytes),
            "tier-b-pair-metadata-audit.json": sha256_bytes(tier_bytes),
            "audit.json": sha256_bytes(audit_bytes),
            "artifact-manifest.json": sha256_bytes(artifact_bytes),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded acquisition-audit CLI."""

    parser = argparse.ArgumentParser(description="Audit Science S4 recovery acquisition v2.")
    parser.add_argument("command", choices=("audit",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ddr-parts-root", type=Path, required=True)
    parser.add_argument("--tier-b-metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the acquisition audit and report its bounded disposition."""

    args = build_parser().parse_args(argv)
    try:
        result = run_acquisition_audit(
            config_path=cast(Path, args.config),
            ddr_parts_root=cast(Path, args.ddr_parts_root),
            tier_b_metadata_root=cast(Path, args.tier_b_metadata_root),
            output_dir=cast(Path, args.output_dir),
        )
    except (FileExistsError, OSError, S4RecoveryAcquisitionError) as error:
        print(f"Science S4 recovery acquisition audit fail-closed: {error}", file=sys.stderr)
        return 1
    print(f"Science S4 recovery acquisition status: {result.status}")
    for name, digest in result.artifact_hashes.items():
        print(f"{name}: {digest}")
    return 0 if result.status == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_STATUS",
    "READY_STATUS",
    "S4RecoveryAcquisitionError",
    "S4RecoveryAcquisitionResult",
    "build_parser",
    "main",
    "run_acquisition_audit",
]
