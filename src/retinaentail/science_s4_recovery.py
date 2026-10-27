"""Fail-closed Science S4 evidence-gap recovery preflight.

The command replays the immutable S4 v1 evidence and audits an explicitly supplied
independent repeated-acquisition candidate.  It never accepts DDR/FGADR roots and
never reads locked-test pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml
from PIL import Image, UnidentifiedImageError

from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S4_EVIDENCE_GAP_RECOVERY_PREFLIGHT"
LESION_CLASSES: Final = ("HE", "EX", "MA", "SE")
BLOCKED_STATUS: Final = "BLOCKED_PREREQUISITES"
READY_STATUS: Final = "READY_FOR_S4_RECOVERY_EXECUTION"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class S4RecoveryError(RuntimeError):
    """Raised when recovery evidence cannot be replayed exactly."""


@dataclass(frozen=True, slots=True)
class S4RecoveryResult:
    """Generated recovery status and artifact hashes."""

    status: str
    output_dir: Path
    artifact_hashes: dict[str, str]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S4RecoveryError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise S4RecoveryError(f"{name} contains a non-string key")
        result[key] = item
    return result


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise S4RecoveryError(f"{name} must be a sequence")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S4RecoveryError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S4RecoveryError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise S4RecoveryError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path, name: str) -> dict[str, object]:
    try:
        decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise S4RecoveryError(f"cannot load {name}: {error}") from error
    return _mapping(decoded, name)


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S4RecoveryError(f"cannot load {name}: {error}") from error
    return _mapping(decoded, name)


def _project_file(relative: object, name: str) -> Path:
    value = _string(relative, name)
    pure = Path(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise S4RecoveryError(f"{name} must be a normalized project-relative path")
    path = (PROJECT_ROOT / pure).resolve(strict=False)
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4RecoveryError(f"{name} escapes the project root") from error
    if not path.is_file():
        raise S4RecoveryError(f"{name} is not a file: {value}")
    return path


def _verify_binding(binding: object, name: str) -> tuple[Path, str]:
    record = _mapping(binding, name)
    path = _project_file(record.get("path"), f"{name}.path")
    expected = _digest(record.get("sha256"), f"{name}.sha256")
    actual = _sha256_file(path)
    if actual != expected:
        raise S4RecoveryError(f"{name} hash mismatch: expected {expected}, got {actual}")
    return path, actual


def _verify_baseline(commit: object) -> str:
    expected = _string(commit, "expected_baseline_commit")
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise S4RecoveryError("expected_baseline_commit must be a lowercase Git object ID")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise S4RecoveryError("frozen S4 baseline is not an ancestor of HEAD")
    return expected


def _class_counts(value: object, name: str) -> dict[str, int]:
    record = _mapping(value, name)
    if set(record) != set(LESION_CLASSES):
        raise S4RecoveryError(f"{name} must contain exactly {list(LESION_CLASSES)!r}")
    return {lesion: _integer(record[lesion], f"{name}.{lesion}") for lesion in LESION_CLASSES}


def _audit_accepted_counts(summary: Mapping[str, object]) -> dict[str, int]:
    raw = _mapping(summary.get("accepted_counts"), "S4 audit accepted_counts")
    counts = {lesion: 0 for lesion in LESION_CLASSES}
    for key, value in raw.items():
        parts = key.split(":")
        if len(parts) != 2 or parts[1] not in counts:
            raise S4RecoveryError(f"unexpected S4 accepted-count key: {key}")
        counts[parts[1]] += _integer(value, f"S4 accepted_counts.{key}")
    return counts


def _verify_artifact_manifest(path: Path, artifact_root_value: object) -> tuple[int, int]:
    root_relative = _string(artifact_root_value, "S4 artifact_root")
    root = (PROJECT_ROOT / Path(root_relative)).resolve(strict=False)
    try:
        root.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4RecoveryError("S4 artifact root escapes project root") from error
    manifest = _load_json(path, "S4 artifact manifest")
    entries = _sequence(manifest.get("files"), "S4 artifact manifest files")
    if not entries:
        raise S4RecoveryError("S4 artifact manifest must not be empty")
    seen: set[str] = set()
    byte_count = 0
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"S4 artifact entry {index}")
        relative = _string(entry.get("relative_path"), f"artifact entry {index} path")
        pure = Path(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise S4RecoveryError(f"unsafe S4 artifact path: {relative}")
        if relative in seen:
            raise S4RecoveryError(f"duplicate S4 artifact path: {relative}")
        seen.add(relative)
        artifact = (root / pure).resolve(strict=False)
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise S4RecoveryError(f"S4 artifact escapes root: {relative}") from error
        if not artifact.is_file():
            raise S4RecoveryError(f"missing S4 artifact: {relative}")
        expected_size = _integer(entry.get("byte_size"), f"artifact {relative} byte_size")
        expected_hash = _digest(entry.get("sha256"), f"artifact {relative} sha256")
        actual_size = artifact.stat().st_size
        if actual_size != expected_size:
            raise S4RecoveryError(f"S4 artifact size mismatch: {relative}")
        if _sha256_file(artifact) != expected_hash:
            raise S4RecoveryError(f"S4 artifact hash mismatch: {relative}")
        byte_count += actual_size
    return len(entries), byte_count


def _replay_upstream(config: Mapping[str, object]) -> dict[str, object]:
    _verify_baseline(config.get("expected_baseline_commit"))
    bindings = _mapping(config.get("bindings"), "bindings")
    verified: dict[str, tuple[Path, str]] = {}
    for name, binding in bindings.items():
        verified[name] = _verify_binding(binding, f"bindings.{name}")

    audit = _load_json(verified["s4_audit"][0], "S4 audit")
    expected = _mapping(config.get("s4_v1_expected"), "s4_v1_expected")
    expected_status = _string(expected.get("status"), "s4_v1_expected.status")
    if audit.get("status") != expected_status:
        raise S4RecoveryError("S4 audit status does not match the frozen blocker")
    summary = _mapping(audit.get("summary"), "S4 audit summary")
    accepted = _audit_accepted_counts(summary)
    distinct = _class_counts(summary.get("accepted_distinct_sources"), "distinct sources")
    collisions = _class_counts(summary.get("raw_exact_collision_pairs"), "raw collisions")
    if accepted != _class_counts(expected.get("accepted_counts"), "expected accepted counts"):
        raise S4RecoveryError("S4 accepted counts changed")
    if distinct != _class_counts(expected.get("distinct_sources"), "expected distinct sources"):
        raise S4RecoveryError("S4 distinct-source counts changed")
    if collisions != _class_counts(expected.get("raw_exact_collisions"), "expected collisions"):
        raise S4RecoveryError("S4 collision counts changed")
    io_audit = _mapping(audit.get("io_audit"), "S4 io_audit")
    for key in ("locked_test_root_reads", "locked_test_mask_decodes", "source_split_leak_count"):
        if _integer(io_audit.get(key), f"S4 io_audit.{key}") != 0:
            raise S4RecoveryError(f"S4 {key} must remain zero")

    qc = _load_json(verified["s4_qc_record"][0], "S4 QC record")
    if (
        qc.get("decision") != "PASS_WITH_LIMITATIONS"
        or qc.get("stage_disposition") != expected_status
    ):
        raise S4RecoveryError("S4 QC record no longer preserves the blocked disposition")
    manifest_binding = _mapping(bindings["s4_artifact_manifest"], "S4 manifest binding")
    file_count, byte_count = _verify_artifact_manifest(
        verified["s4_artifact_manifest"][0], manifest_binding.get("artifact_root")
    )
    return {
        "status": "PASS",
        "verified_binding_count": len(verified),
        "artifact_file_count": file_count,
        "artifact_byte_count": byte_count,
        "s4_status": expected_status,
        "accepted_counts": accepted,
        "distinct_sources": distinct,
        "raw_exact_collisions": collisions,
    }


def _audit_measurement_root(
    root: Path, candidate: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], int]:
    if not root.is_absolute():
        raise S4RecoveryError("measurement root must be an explicit absolute path")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise S4RecoveryError("measurement root must be a directory")
    expected_ids = tuple(
        _integer(value, "measurement pair id", minimum=1)
        for value in _sequence(candidate.get("expected_pair_ids"), "expected_pair_ids")
    )
    if len(expected_ids) != len(set(expected_ids)):
        raise S4RecoveryError("measurement pair ids must be unique")
    roles = tuple(
        _string(value, "measurement role")
        for value in _sequence(candidate.get("expected_roles"), "expected_roles")
    )
    if roles != ("good", "bad"):
        raise S4RecoveryError("measurement roles must remain good/bad")
    expected_names = {f"{pair_id}_{role}.JPG" for pair_id in expected_ids for role in roles}
    entries = tuple(resolved.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise S4RecoveryError("measurement root must contain only regular files")
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise S4RecoveryError(
            f"measurement inventory mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )
    expected_file_count = _integer(candidate.get("expected_file_count"), "expected_file_count")
    if len(entries) != expected_file_count:
        raise S4RecoveryError("measurement file count mismatch")

    files: list[dict[str, object]] = []
    dimensions: dict[int, tuple[int, int]] = {}
    for pair_id in expected_ids:
        for role in roles:
            name = f"{pair_id}_{role}.JPG"
            path = resolved / name
            try:
                with Image.open(path) as image:
                    image.load()
                    mode = image.mode
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as error:
                raise S4RecoveryError(f"cannot decode measurement image {name}: {error}") from error
            if mode != "RGB" or width <= 0 or height <= 0:
                raise S4RecoveryError(f"measurement image {name} must be non-empty RGB")
            current = (width, height)
            previous = dimensions.setdefault(pair_id, current)
            if current != previous:
                raise S4RecoveryError(f"measurement pair {pair_id} dimensions differ")
            files.append(
                {
                    "relative_path": name,
                    "pair_id": pair_id,
                    "role": role,
                    "byte_size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "mode": mode,
                    "width": width,
                    "height": height,
                }
            )

    minimum = _integer(candidate.get("minimum_independent_pairs"), "minimum pairs", minimum=1)
    calculated_minimum = math.ceil(math.log(0.05) / math.log(0.95))
    if minimum != calculated_minimum:
        raise S4RecoveryError("frozen Tier-B minimum does not match its stated formula")
    pair_count = len(expected_ids)
    conditions_frozen = all(
        candidate.get(key) not in {"NOT_FROZEN", "NOT_IMPLEMENTED"}
        for key in (
            "registration_status",
            "domain_status",
            "threshold_status",
            "overlap_witness_status",
        )
    )
    independent = candidate.get("retinaentail_independent") is True
    tier_b_available = pair_count >= minimum and conditions_frozen and independent
    candidate_summary: dict[str, object] = {
        "dataset_id": _string(candidate.get("dataset_id"), "measurement dataset_id"),
        "dataset_version": _string(candidate.get("dataset_version"), "measurement version"),
        "pair_count": pair_count,
        "minimum_independent_pairs": minimum,
        "sample_sufficiency": pair_count >= minimum,
        "conditions_frozen": conditions_frozen,
        "retinaentail_independent": independent,
        "tier_b_status": "AVAILABLE" if tier_b_available else "NOT_AVAILABLE",
        "limitations": [
            "The 18 good/bad pairs include different acquisition settings and systematic defocus.",
            "No registration, domain, threshold, or overlap-witness rule is frozen in this "
            "preflight.",
            "This acquisition candidate is not lesion truth or performance evidence.",
        ],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": candidate_summary["dataset_id"],
        "dataset_version": candidate_summary["dataset_version"],
        "archive": {
            "name": _string(candidate.get("archive_name"), "archive_name"),
            "byte_size": _integer(candidate.get("archive_byte_size"), "archive_byte_size"),
            "sha256": _digest(candidate.get("archive_sha256"), "archive_sha256"),
        },
        "root": str(resolved),
        "pair_count": pair_count,
        "files": files,
    }
    return manifest, candidate_summary, len(entries)


def run_preflight(
    *, config_path: Path, measurement_root: Path, output_dir: Path
) -> S4RecoveryResult:
    """Replay S4 and inspect the explicitly supplied HRF candidate without downstream execution."""

    resolved_config = config_path.resolve(strict=True)
    try:
        resolved_config.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise S4RecoveryError("config path must remain within the project root") from error
    config = _load_yaml(resolved_config, "S4 recovery config")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise S4RecoveryError("unsupported S4 recovery config")
    if (
        config.get("status") != "EXECUTABLE_PREFLIGHT"
        or config.get("execution_enabled") is not True
    ):
        raise S4RecoveryError("S4 recovery preflight is not executable")
    scope = _mapping(config.get("scope"), "scope")
    if scope.get("lesion_classes") != list(LESION_CLASSES):
        raise S4RecoveryError("lesion class order changed")
    if scope.get("locked_test_root_reads_allowed") is not False:
        raise S4RecoveryError("locked-test root reads must remain forbidden")
    frozen_output = (PROJECT_ROOT / _string(config.get("artifact_root"), "artifact_root")).resolve(
        strict=False
    )
    if output_dir.resolve(strict=False) != frozen_output:
        raise S4RecoveryError("output directory must match the frozen artifact_root")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing recovery output: {output_dir}")

    upstream = _replay_upstream(config)
    candidate = _mapping(config.get("measurement_error_candidate"), "measurement candidate")
    measurement_manifest, measurement_summary, measurement_entries = _audit_measurement_root(
        measurement_root, candidate
    )
    source_recovery = _mapping(config.get("source_recovery"), "source_recovery")
    ddr = _mapping(source_recovery.get("ddr"), "source_recovery.ddr")
    fgadr = _mapping(source_recovery.get("fgadr"), "source_recovery.fgadr")
    blockers: list[str] = []
    if ddr.get("status") != "READY_FOR_ADAPTER_PREFLIGHT":
        blockers.append("DDR_TRANSFER_AND_LOCAL_SHA256_INCOMPLETE")
    if measurement_summary["tier_b_status"] != "AVAILABLE":
        blockers.append("INDEPENDENT_TIER_B_MEASUREMENT_ERROR_SET_NOT_AVAILABLE")
    status = BLOCKED_STATUS if blockers else READY_STATUS
    config_sha256 = _sha256_file(resolved_config)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "config_sha256": config_sha256,
        "upstream_replay": upstream,
        "source_recovery": {
            "order": source_recovery.get("order"),
            "ddr": {
                "status": ddr.get("status"),
                "expected_development_sources": ddr.get("expected_development_sources"),
                "official_part_count": len(_sequence(ddr.get("parts"), "DDR parts")),
                "root_read_count": 0,
            },
            "fgadr": {
                "status": fgadr.get("status"),
                "root_read_count": 0,
            },
        },
        "measurement_error_candidate": measurement_summary,
        "io_audit": {
            "measurement_root_entries": measurement_entries,
            "measurement_image_decodes": measurement_entries,
            "ddr_root_reads": 0,
            "fgadr_root_reads": 0,
            "locked_test_root_reads": 0,
            "locked_test_image_decodes": 0,
        },
        "blockers": blockers,
        "downstream_authorization": {
            "s4_recovery_execution": status == READY_STATUS,
            "s5_degradation_calibration": False,
            "s6_verify": False,
            "s7_lite": False,
            "enhancer_training": False,
            "locked_test_evaluation": False,
            "fcr_estimation": False,
            "paper_performance_experiment": False,
        },
        "evidence_boundary": [
            "The HRF pair count and hashes are acquisition-error preflight evidence, not lesion "
            "truth.",
            "No twin count, file count, collision count, or synthetic fixture is FCR, medical, "
            "clinical, or paper-performance evidence.",
            "No S5, Verify, Lite, enhancer training, locked-test evaluation, or FCR estimation "
            "was run.",
        ],
    }
    measurement_bytes = canonical_json_bytes(cast(dict[str, JSONValue], measurement_manifest))
    audit_bytes = canonical_json_bytes(cast(dict[str, JSONValue], audit))
    artifact_manifest = {
        "files": [
            {
                "relative_path": "audit.json",
                "byte_size": len(audit_bytes),
                "sha256": sha256_bytes(audit_bytes),
            },
            {
                "relative_path": "measurement-error-candidate-manifest.json",
                "byte_size": len(measurement_bytes),
                "sha256": sha256_bytes(measurement_bytes),
            },
        ]
    }
    artifact_bytes = canonical_json_bytes(cast(dict[str, JSONValue], artifact_manifest))
    write_bytes_fail_safe(
        output_dir / "measurement-error-candidate-manifest.json", measurement_bytes
    )
    write_bytes_fail_safe(output_dir / "audit.json", audit_bytes)
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", artifact_bytes)
    return S4RecoveryResult(
        status=status,
        output_dir=output_dir,
        artifact_hashes={
            "measurement-error-candidate-manifest.json": sha256_bytes(measurement_bytes),
            "audit.json": sha256_bytes(audit_bytes),
            "artifact-manifest.json": sha256_bytes(artifact_bytes),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded preflight CLI parser."""

    parser = argparse.ArgumentParser(description="Run Science S4 evidence-gap recovery preflight.")
    parser.add_argument("command", choices=("preflight",))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/science/s4_evidence_gap_recovery_preflight_v1.yaml"),
    )
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recovery preflight and report its fail-closed disposition."""

    args = build_parser().parse_args(argv)
    try:
        result = run_preflight(
            config_path=cast(Path, args.config),
            measurement_root=cast(Path, args.measurement_root),
            output_dir=cast(Path, args.output_dir),
        )
    except (FileExistsError, OSError, S4RecoveryError) as error:
        print(f"Science S4 recovery preflight fail-closed: {error}", file=sys.stderr)
        return 1
    print(f"Science S4 recovery preflight status: {result.status}")
    for name, digest in result.artifact_hashes.items():
        print(f"{name}: {digest}")
    return 0 if result.status == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_STATUS",
    "READY_STATUS",
    "S4RecoveryError",
    "S4RecoveryResult",
    "build_parser",
    "main",
    "run_preflight",
]
