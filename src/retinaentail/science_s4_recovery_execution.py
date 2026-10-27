"""Fail-closed Science S4 evidence-gap recovery execution audit.

This command inventories explicitly supplied DDR split parts and Tier-B metadata.
It never accepts an extracted DDR root, never reads locked-test pixels, and never
promotes a dataset-card statement into pair-level evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast
from xml.etree import ElementTree

import retinaentail.science_s4_recovery as preflight
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S4_EVIDENCE_GAP_RECOVERY_EXECUTION_V1"
BLOCKED_STATUS: Final = "BLOCKED_PREREQUISITES"
READY_STATUS: Final = "READY_FOR_S4_RECOVERY_TWIN_EXECUTION"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
_MAIN_NS: Final = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS: Final = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS: Final = "http://schemas.openxmlformats.org/package/2006/relationships"


class S4RecoveryExecutionError(RuntimeError):
    """Raised when execution evidence cannot be audited without guessing."""


@dataclass(frozen=True, slots=True)
class S4RecoveryExecutionResult:
    """Generated execution disposition and artifact hashes."""

    status: str
    output_dir: Path
    artifact_hashes: dict[str, str]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S4RecoveryExecutionError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise S4RecoveryExecutionError(f"{name} contains a non-string key")
        result[key] = item
    return result


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise S4RecoveryExecutionError(f"{name} must be a sequence")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S4RecoveryExecutionError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S4RecoveryExecutionError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, object]:
    try:
        return preflight._load_yaml(path, "S4 recovery execution config")  # noqa: SLF001
    except preflight.S4RecoveryError as error:
        raise S4RecoveryExecutionError(str(error)) from error


def _verify_binding(binding: object, name: str) -> tuple[Path, str]:
    try:
        return preflight._verify_binding(binding, name)  # noqa: SLF001
    except preflight.S4RecoveryError as error:
        raise S4RecoveryExecutionError(str(error)) from error


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        return preflight._load_json(path, name)  # noqa: SLF001
    except preflight.S4RecoveryError as error:
        raise S4RecoveryExecutionError(str(error)) from error


def _explicit_external_dir(path: Path, root_name: str, leaf_name: str) -> Path:
    if not path.is_absolute():
        raise S4RecoveryExecutionError(f"{leaf_name} root must be an explicit absolute path")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise S4RecoveryExecutionError(f"{leaf_name} root must be a regular directory")
    if resolved.name != leaf_name or resolved.parent.name != root_name:
        raise S4RecoveryExecutionError(f"{leaf_name} root must be the frozen child of {root_name}")
    return resolved


def _replay_preflight(config: Mapping[str, object]) -> dict[str, object]:
    bindings = _mapping(config.get("bindings"), "bindings")
    required = {
        "execution_prompt",
        "preflight_config",
        "preflight_report",
        "preflight_audit",
        "preflight_artifact_manifest",
        "preflight_measurement_manifest",
    }
    if set(bindings) != required:
        raise S4RecoveryExecutionError("execution bindings changed")
    verified = {
        name: _verify_binding(binding, f"bindings.{name}") for name, binding in bindings.items()
    }

    preflight_config = _load_config(verified["preflight_config"][0])
    try:
        upstream = preflight._replay_upstream(preflight_config)  # noqa: SLF001
    except preflight.S4RecoveryError as error:
        raise S4RecoveryExecutionError(str(error)) from error
    audit = _load_json(verified["preflight_audit"][0], "S4 recovery preflight audit")
    if audit.get("status") != BLOCKED_STATUS:
        raise S4RecoveryExecutionError("preflight disposition must remain blocked")
    measurement = _load_json(
        verified["preflight_measurement_manifest"][0], "preflight measurement manifest"
    )
    if _integer(measurement.get("pair_count"), "preflight pair_count") != 18:
        raise S4RecoveryExecutionError("HRF preflight pair count changed")

    artifact_manifest = _load_json(
        verified["preflight_artifact_manifest"][0], "preflight artifact manifest"
    )
    entries = _sequence(artifact_manifest.get("files"), "preflight artifact files")
    expected_names = {"audit.json", "measurement-error-candidate-manifest.json"}
    actual_names: set[str] = set()
    artifact_root = verified["preflight_audit"][0].parent
    artifact_bytes = 0
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"preflight artifact {index}")
        relative = _string(entry.get("relative_path"), f"preflight artifact {index} path")
        if relative not in expected_names or relative in actual_names:
            raise S4RecoveryExecutionError("preflight artifact inventory changed")
        actual_names.add(relative)
        path = artifact_root / relative
        if not path.is_file() or path.is_symlink():
            raise S4RecoveryExecutionError(f"missing preflight artifact: {relative}")
        size = _integer(entry.get("byte_size"), f"preflight artifact {relative} size")
        digest = _string(entry.get("sha256"), f"preflight artifact {relative} hash")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise S4RecoveryExecutionError(f"preflight artifact drift: {relative}")
        artifact_bytes += size
    if actual_names != expected_names:
        raise S4RecoveryExecutionError("preflight artifact inventory is incomplete")
    return {
        "status": "PASS",
        "verified_binding_count": len(verified),
        "s4_v1_replay": upstream,
        "preflight_status": BLOCKED_STATUS,
        "preflight_artifact_file_count": len(entries),
        "preflight_artifact_byte_count": artifact_bytes,
        "hrf_verified_pair_count": 18,
    }


def _audit_ddr_parts(root: Path, source: Mapping[str, object]) -> tuple[dict[str, object], int]:
    parts = _sequence(source.get("parts"), "DDR parts")
    expected_count = _integer(source.get("expected_part_count"), "DDR expected_part_count")
    if len(parts) != expected_count:
        raise S4RecoveryExecutionError("DDR expected part count changed")
    expected: dict[str, str] = {}
    for index, raw in enumerate(parts):
        part = _mapping(raw, f"DDR part {index}")
        name = _string(part.get("name"), f"DDR part {index} name")
        file_id = _string(part.get("file_id"), f"DDR part {index} file_id")
        if name in expected:
            raise S4RecoveryExecutionError(f"duplicate DDR part name: {name}")
        expected[name] = file_id

    entries = tuple(root.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise S4RecoveryExecutionError("DDR part root must contain only regular files")
    actual_names = {entry.name for entry in entries}
    unexpected = sorted(actual_names - set(expected))
    records: list[dict[str, object]] = []
    complete_count = 0
    total_bytes = 0
    for name, file_id in expected.items():
        path = root / name
        if not path.is_file():
            records.append({"name": name, "file_id": file_id, "status": "MISSING", "byte_size": 0})
            continue
        size = path.stat().st_size
        status = "COMPLETE_LOCAL_HASHED" if size > 0 else "EMPTY"
        record: dict[str, object] = {
            "name": name,
            "file_id": file_id,
            "status": status,
            "byte_size": size,
        }
        if size > 0:
            record["sha256"] = _sha256_file(path)
            complete_count += 1
            total_bytes += size
        records.append(record)
    all_complete = complete_count == expected_count and not unexpected
    manifest = {
        "dataset_id": _string(source.get("dataset_id"), "DDR dataset_id"),
        "dataset_version": _string(source.get("dataset_version"), "DDR dataset_version"),
        "root": str(root),
        "expected_part_count": expected_count,
        "root_entry_count": len(entries),
        "complete_part_count": complete_count,
        "total_complete_bytes": total_bytes,
        "unexpected_files": unexpected,
        "all_parts_complete": all_complete,
        "archive_merge_or_extract_performed": False,
        "parts": records,
    }
    return manifest, len(entries)


def _column_index(reference: str) -> int:
    match = re.match(r"^([A-Z]+)", reference)
    if match is None:
        raise S4RecoveryExecutionError(f"invalid XLSX cell reference: {reference}")
    index = 0
    for character in match.group(1):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _xlsx_rows(path: Path) -> list[dict[str, str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.findall(f"{{{_MAIN_NS}}}si"):
                    shared.append(
                        "".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t"))
                    )
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            sheet = workbook.find(f"{{{_MAIN_NS}}}sheets/{{{_MAIN_NS}}}sheet")
            if sheet is None:
                raise S4RecoveryExecutionError("XLSX contains no worksheet")
            relationship_id = sheet.attrib.get(f"{{{_REL_NS}}}id")
            relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            target = None
            for relation in relationships.findall(f"{{{_PKG_REL_NS}}}Relationship"):
                if relation.attrib.get("Id") == relationship_id:
                    target = relation.attrib.get("Target")
                    break
            if not target:
                raise S4RecoveryExecutionError("XLSX worksheet relationship is missing")
            worksheet_name = "xl/" + target.lstrip("/")
            worksheet = ElementTree.fromstring(archive.read(worksheet_name))
    except (KeyError, OSError, zipfile.BadZipFile, ElementTree.ParseError) as error:
        raise S4RecoveryExecutionError(f"cannot parse Tier-B XLSX metadata: {error}") from error

    raw_rows: list[list[str]] = []
    for row in worksheet.findall(f".//{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            reference = cell.attrib.get("r", "")
            index = _column_index(reference)
            cell_type = cell.attrib.get("t")
            value_node = cell.find(f"{{{_MAIN_NS}}}v")
            value = "" if value_node is None or value_node.text is None else value_node.text
            if cell_type == "s" and value:
                shared_index = int(value)
                if shared_index >= len(shared):
                    raise S4RecoveryExecutionError("XLSX shared-string index is invalid")
                value = shared[shared_index]
            elif cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{_MAIN_NS}}}t"))
            values[index] = value.strip()
        if values:
            width = max(values) + 1
            raw_rows.append([values.get(index, "") for index in range(width)])
    if not raw_rows:
        raise S4RecoveryExecutionError("Tier-B XLSX metadata is empty")
    headers = [header.strip() for header in raw_rows[0]]
    if not headers or any(not header for header in headers) or len(headers) != len(set(headers)):
        raise S4RecoveryExecutionError("Tier-B XLSX headers are empty or duplicated")
    return [
        {header: (row[index] if index < len(row) else "") for index, header in enumerate(headers)}
        for row in raw_rows[1:]
        if any(value for value in row)
    ]


def _audit_tier_b_metadata(
    root: Path, tier_b: Mapping[str, object]
) -> tuple[dict[str, object], int]:
    if tier_b.get("retinaentail_independent") is not True:
        raise S4RecoveryExecutionError("Tier-B candidate must remain RetinaEntail-independent")
    candidate = _mapping(tier_b.get("candidate"), "Tier-B candidate")
    expected_name = _string(candidate.get("expected_metadata_name"), "Tier-B metadata name")
    entries = tuple(root.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise S4RecoveryExecutionError("Tier-B metadata root must contain only regular files")
    unexpected = sorted(entry.name for entry in entries if entry.name != expected_name)
    path = root / expected_name
    required_fields = tuple(
        _string(value, "Tier-B required field")
        for value in _sequence(candidate.get("required_pair_fields"), "Tier-B required fields")
    )
    minimum = _integer(tier_b.get("minimum_independent_pairs"), "Tier-B minimum", minimum=1)
    pair_count = 0
    independent_units = 0
    metadata_status = "METADATA_UNAVAILABLE"
    metadata_record: dict[str, object] = {
        "name": expected_name,
        "status": "MISSING",
        "byte_size": 0,
    }
    missing_fields = list(required_fields)
    if path.is_file():
        size = path.stat().st_size
        metadata_record = {
            "name": expected_name,
            "status": "PRESENT",
            "byte_size": size,
            "sha256": _sha256_file(path),
        }
        rows = _xlsx_rows(path)
        available_fields = set(rows[0]) if rows else set()
        missing_fields = sorted(set(required_fields) - available_fields)
        if missing_fields:
            metadata_status = "PAIR_FIELDS_INCOMPLETE"
        else:
            roles = {
                _string(value, "Tier-B device role")
                for value in _sequence(candidate.get("allowed_device_roles"), "device roles")
            }
            groups: dict[tuple[str, str, str], set[str]] = {}
            images: set[str] = set()
            for row in rows:
                values = {field: row[field].strip() for field in required_fields}
                if any(not value for value in values.values()):
                    raise S4RecoveryExecutionError("Tier-B metadata contains empty pair fields")
                if values["device_role"] not in roles:
                    raise S4RecoveryExecutionError(
                        "Tier-B metadata contains an unknown device role"
                    )
                if values["image_path"] in images:
                    raise S4RecoveryExecutionError("Tier-B metadata reuses an image path")
                images.add(values["image_path"])
                key = (values["subject_id"], values["eye_laterality"], values["pair_id"])
                groups.setdefault(key, set()).add(values["device_role"])
            complete = {key for key, found_roles in groups.items() if found_roles == roles}
            pair_count = len(complete)
            independent_units = len({(subject, eye) for subject, eye, _pair in complete})
            metadata_status = (
                "PAIR_CONTRACT_VERIFIED"
                if pair_count == len(groups) and independent_units >= minimum
                else "PAIR_CONTRACT_INSUFFICIENT"
            )

    conditions_frozen = all(
        tier_b.get(key) not in {"NOT_FROZEN", "NOT_IMPLEMENTED"}
        for key in (
            "registration_status",
            "domain_status",
            "threshold_status",
            "overlap_witness_status",
        )
    )
    available = (
        metadata_status == "PAIR_CONTRACT_VERIFIED"
        and independent_units >= minimum
        and conditions_frozen
        and not unexpected
    )
    manifest = {
        "dataset_id": _string(candidate.get("dataset_id"), "Tier-B dataset_id"),
        "repository": _string(candidate.get("repository"), "Tier-B repository"),
        "repository_revision": _string(
            candidate.get("repository_revision"), "Tier-B repository revision"
        ),
        "data_upload_revision": _string(
            candidate.get("data_upload_revision"), "Tier-B data revision"
        ),
        "root": str(root),
        "root_entry_count": len(entries),
        "unexpected_files": unexpected,
        "metadata": metadata_record,
        "required_pair_fields": list(required_fields),
        "missing_pair_fields": missing_fields,
        "metadata_status": metadata_status,
        "verified_pair_count": pair_count,
        "independent_subject_eye_count": independent_units,
        "minimum_independent_pairs": minimum,
        "existing_hrf_pair_count": _integer(
            tier_b.get("existing_verified_hrf_pairs"), "existing HRF pairs"
        ),
        "conditions_frozen": conditions_frozen,
        "retinaentail_independent": True,
        "tier_b_status": "AVAILABLE" if available else "NOT_AVAILABLE",
    }
    return manifest, len(entries)


def run_execution_audit(
    *, config_path: Path, ddr_parts_root: Path, tier_b_metadata_root: Path, output_dir: Path
) -> S4RecoveryExecutionResult:
    """Audit explicit recovery inputs without reading extracted or locked-test roots."""

    resolved_config = config_path.resolve(strict=True)
    try:
        resolved_config.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4RecoveryExecutionError("config path must remain within the project root") from error
    config = _load_config(resolved_config)
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise S4RecoveryExecutionError("unsupported S4 recovery execution config")
    if (
        config.get("status") != "EXECUTABLE_BLOCKER_AUDIT"
        or config.get("execution_enabled") is not True
    ):
        raise S4RecoveryExecutionError("S4 recovery execution audit is not enabled")
    scope = _mapping(config.get("scope"), "scope")
    if scope.get("locked_test_root_reads_allowed") is not False:
        raise S4RecoveryExecutionError("locked-test root reads must remain forbidden")
    if scope.get("missing_label_policy") != "UNKNOWN":
        raise S4RecoveryExecutionError("missing labels must remain UNKNOWN")
    frozen_output = (PROJECT_ROOT / _string(config.get("artifact_root"), "artifact_root")).resolve(
        strict=False
    )
    if output_dir.resolve(strict=False) != frozen_output:
        raise S4RecoveryExecutionError("output directory must match the frozen artifact_root")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite recovery execution output: {output_dir}")

    root_name = _string(config.get("external_root_name"), "external_root_name")
    ddr_root = _explicit_external_dir(ddr_parts_root, root_name, "ddr_parts")
    tier_root = _explicit_external_dir(tier_b_metadata_root, root_name, "tier_b_metadata")
    upstream = _replay_preflight(config)
    ddr_manifest, ddr_entries = _audit_ddr_parts(
        ddr_root, _mapping(config.get("source_recovery"), "source_recovery")
    )
    tier_b_manifest, metadata_entries = _audit_tier_b_metadata(
        tier_root, _mapping(config.get("tier_b"), "tier_b")
    )
    blockers: list[str] = []
    if ddr_manifest["all_parts_complete"] is not True:
        blockers.append("DDR_TRANSFER_AND_LOCAL_SHA256_INCOMPLETE")
    if tier_b_manifest["tier_b_status"] != "AVAILABLE":
        blockers.append("INDEPENDENT_TIER_B_MEASUREMENT_ERROR_SET_NOT_AVAILABLE")
    status = BLOCKED_STATUS if blockers else READY_STATUS
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "config_sha256": _sha256_file(resolved_config),
        "upstream_replay": upstream,
        "source_recovery": {
            "dataset_id": ddr_manifest["dataset_id"],
            "all_parts_complete": ddr_manifest["all_parts_complete"],
            "complete_part_count": ddr_manifest["complete_part_count"],
            "expected_part_count": ddr_manifest["expected_part_count"],
            "archive_merge_or_extract_performed": False,
            "development_adapter_built": False,
        },
        "tier_b_candidate": {
            "dataset_id": tier_b_manifest["dataset_id"],
            "metadata_status": tier_b_manifest["metadata_status"],
            "independent_subject_eye_count": tier_b_manifest["independent_subject_eye_count"],
            "minimum_independent_pairs": tier_b_manifest["minimum_independent_pairs"],
            "conditions_frozen": tier_b_manifest["conditions_frozen"],
            "tier_b_status": tier_b_manifest["tier_b_status"],
        },
        "download_attempts": _mapping(config.get("download_attempts"), "download_attempts"),
        "io_audit": {
            "ddr_part_directory_entries": ddr_entries,
            "ddr_archive_reads": 0,
            "ddr_dataset_root_reads": 0,
            "ddr_image_decodes": 0,
            "tier_b_metadata_directory_entries": metadata_entries,
            "tier_b_image_decodes": 0,
            "fgadr_root_reads": 0,
            "locked_test_root_reads": 0,
            "locked_test_image_decodes": 0,
        },
        "blockers": blockers,
        "downstream_authorization": {
            "s4_recovery_twin_execution": status == READY_STATUS,
            "s5_degradation_calibration": False,
            "s6_verify": False,
            "s7_lite": False,
            "enhancer_training": False,
            "locked_test_evaluation": False,
            "fcr_estimation": False,
            "paper_performance_experiment": False,
        },
        "evidence_boundary": [
            "Zero-byte or missing downloads are not completed archives and have no "
            "fabricated hash.",
            "Dataset-card patient counts and single-image viewer rows are not pair-level evidence.",
            "No file count, pair count, component, twin, collision, or synthetic "
            "fixture is FCR, medical, clinical, or paper-performance evidence.",
            "No S5, Verify, Lite, enhancer training, locked-test evaluation, or FCR "
            "estimation was run.",
        ],
    }
    ddr_bytes = canonical_json_bytes(cast(dict[str, JSONValue], ddr_manifest))
    tier_bytes = canonical_json_bytes(cast(dict[str, JSONValue], tier_b_manifest))
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
                "relative_path": "tier-b-candidate-inventory.json",
                "byte_size": len(tier_bytes),
                "sha256": sha256_bytes(tier_bytes),
            },
        ]
    }
    artifact_bytes = canonical_json_bytes(cast(dict[str, JSONValue], artifact_manifest))
    write_bytes_fail_safe(output_dir / "ddr-part-inventory.json", ddr_bytes)
    write_bytes_fail_safe(output_dir / "tier-b-candidate-inventory.json", tier_bytes)
    write_bytes_fail_safe(output_dir / "audit.json", audit_bytes)
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", artifact_bytes)
    return S4RecoveryExecutionResult(
        status=status,
        output_dir=output_dir,
        artifact_hashes={
            "ddr-part-inventory.json": sha256_bytes(ddr_bytes),
            "tier-b-candidate-inventory.json": sha256_bytes(tier_bytes),
            "audit.json": sha256_bytes(audit_bytes),
            "artifact-manifest.json": sha256_bytes(artifact_bytes),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded execution-audit CLI parser."""

    parser = argparse.ArgumentParser(description="Audit Science S4 recovery execution inputs.")
    parser.add_argument("command", choices=("audit",))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/science/s4_evidence_gap_recovery_execution_v1.yaml"),
    )
    parser.add_argument("--ddr-parts-root", type=Path, required=True)
    parser.add_argument("--tier-b-metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded audit and report its fail-closed disposition."""

    args = build_parser().parse_args(argv)
    try:
        result = run_execution_audit(
            config_path=cast(Path, args.config),
            ddr_parts_root=cast(Path, args.ddr_parts_root),
            tier_b_metadata_root=cast(Path, args.tier_b_metadata_root),
            output_dir=cast(Path, args.output_dir),
        )
    except (FileExistsError, OSError, S4RecoveryExecutionError) as error:
        print(f"Science S4 recovery execution audit fail-closed: {error}", file=sys.stderr)
        return 1
    print(f"Science S4 recovery execution status: {result.status}")
    for name, digest in result.artifact_hashes.items():
        print(f"{name}: {digest}")
    return 0 if result.status == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_STATUS",
    "READY_STATUS",
    "S4RecoveryExecutionError",
    "S4RecoveryExecutionResult",
    "build_parser",
    "main",
    "run_execution_audit",
]
