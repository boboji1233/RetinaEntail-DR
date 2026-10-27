"""Zero-dataset-I/O preregistration preflight for Science S4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

import yaml

from retinaentail.data.manifest import sha256_file
from retinaentail.schemas import JSONValue
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "S4_CONTROLLED_BENCHMARK_SUPPORT_ORACLE"
_EXPECTED_CLASSES = ("HE", "EX", "MA", "SE")
_EXPECTED_SPLITS = ("TRAIN", "CALIBRATION")
_EXPECTED_AXES = ("SOURCE_TRUTH", "SUPPORT_TRUTH", "DOMAIN_TRUTH")
_EXPECTED_CONFIRMATORY_TIERS = ("EXACT_COLLISION", "CALIBRATED_SET_OVERLAP")
_EXPECTED_EXPLORATORY_TIERS = ("EMPIRICAL_EQUIVALENCE",)


class S4PreflightError(ValueError):
    """Raised when the frozen preflight contract cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class S4PreflightResult:
    """Deterministic S4 preflight result and generated artifact hashes."""

    status: str
    blockers: tuple[str, ...]
    payload: dict[str, JSONValue]
    artifact_hashes: dict[str, str]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S4PreflightError(f"{label} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise S4PreflightError(f"{label} keys must be strings")
        result[key] = item
    return result


def _sequence(value: object, label: str) -> list[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise S4PreflightError(f"{label} must be a sequence")
    return list(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S4PreflightError(f"{label} must be a non-empty string")
    return value.strip()


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - schema boundary requires exact bool
        raise S4PreflightError(f"{label} must be a bool")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise S4PreflightError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _git_oid(value: object, label: str) -> str:
    oid = _string(value, label)
    if len(oid) != 40 or any(char not in "0123456789abcdef" for char in oid):
        raise S4PreflightError(f"{label} must be a 40-character lowercase Git object ID")
    return oid


def _json_safe(value: object, label: str = "value") -> JSONValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise S4PreflightError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise S4PreflightError(f"{label} contains a non-string key")
            result[key] = _json_safe(item, f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_safe(item, f"{label} item") for item in value]
    raise S4PreflightError(f"{label} is not JSON-safe")


def _relative_path(value: object, label: str) -> str:
    raw = _string(value, label)
    if "\\" in raw:
        raise S4PreflightError(f"{label} must use normalized POSIX separators")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise S4PreflightError(f"{label} must be relative")
    if not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise S4PreflightError(f"{label} must not contain traversal")
    if posix.as_posix() != raw:
        raise S4PreflightError(f"{label} must be normalized")
    return raw


def _resolve_existing(project_root: Path, relative_path: str, label: str) -> Path:
    resolved_root = project_root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise S4PreflightError(f"{label} escapes project root") from error
    if not candidate.is_file():
        raise S4PreflightError(f"{label} is not a file: {relative_path}")
    return candidate


def _load_yaml(path: Path, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise S4PreflightError(f"cannot load {label}: {error}") from error
    return _mapping(value, label)


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S4PreflightError(f"cannot load {label}: {error}") from error
    return _mapping(value, label)


def _expected_sequence(value: object, expected: tuple[str, ...], label: str) -> list[str]:
    actual = [_string(item, f"{label} item") for item in _sequence(value, label)]
    if tuple(actual) != expected:
        raise S4PreflightError(f"{label} must equal {list(expected)!r}")
    return actual


def _binding_path(
    project_root: Path,
    bindings: dict[str, object],
    binding_id: str,
) -> tuple[dict[str, object], Path]:
    binding = _mapping(bindings.get(binding_id), f"bindings.{binding_id}")
    relative = _relative_path(binding.get("path"), f"bindings.{binding_id}.path")
    return binding, _resolve_existing(project_root, relative, f"bindings.{binding_id}.path")


def _git_commit_is_ancestor(project_root: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode not in {0, 1}:
        raise S4PreflightError("cannot verify the frozen S3 baseline commit")
    return completed.returncode == 0


def _check(checks: list[dict[str, JSONValue]], check_id: str, expected: str, actual: str) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "PASS" if actual == expected else "FAIL",
            "expected": expected,
            "actual": actual,
        }
    )


def _canonical_yaml_sha256(path: Path) -> str:
    decoded = _load_yaml(path, str(path))
    return sha256_bytes(canonical_json_bytes(_json_safe(decoded)))


def _implementation_bundle_sha256(project_root: Path, value: object) -> str:
    files = [_relative_path(item, "s3 implementation file") for item in _sequence(value, "files")]
    if not files:
        raise S4PreflightError("S3 implementation file list must not be empty")
    entries: list[JSONValue] = []
    for relative in files:
        path = _resolve_existing(project_root, relative, "s3 implementation file")
        entries.append({"relative_path": relative, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(entries))


def _validate_artifact_manifest(project_root: Path, binding: dict[str, object]) -> int:
    manifest_relative = _relative_path(binding.get("path"), "artifact manifest path")
    artifact_root_relative = _relative_path(binding.get("artifact_root"), "S3 artifact root")
    manifest_path = _resolve_existing(project_root, manifest_relative, "S3 artifact manifest")
    artifact_root = (project_root.resolve(strict=True) / artifact_root_relative).resolve(
        strict=True
    )
    try:
        artifact_root.relative_to(project_root.resolve(strict=True))
    except ValueError as error:
        raise S4PreflightError("S3 artifact root escapes project root") from error
    manifest = _load_json(manifest_path, "S3 artifact manifest")
    entries = _sequence(manifest.get("files"), "S3 artifact manifest files")
    if not entries:
        raise S4PreflightError("S3 artifact manifest must not be empty")
    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"artifact entry {index}")
        relative = _relative_path(entry.get("relative_path"), f"artifact entry {index} path")
        if relative in seen:
            raise S4PreflightError(f"duplicate S3 artifact path: {relative}")
        seen.add(relative)
        candidate = (artifact_root / relative).resolve(strict=True)
        try:
            candidate.relative_to(artifact_root)
        except ValueError as error:
            raise S4PreflightError(f"S3 artifact escapes root: {relative}") from error
        if not candidate.is_file():
            raise S4PreflightError(f"S3 artifact is not a file: {relative}")
        byte_size = entry.get("byte_size")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
            raise S4PreflightError(f"artifact entry {index} byte_size is invalid")
        if candidate.stat().st_size != byte_size:
            raise S4PreflightError(f"S3 artifact byte size mismatch: {relative}")
        expected_hash = _sha256(entry.get("sha256"), f"artifact entry {index} sha256")
        if sha256_file(candidate) != expected_hash:
            raise S4PreflightError(f"S3 artifact hash mismatch: {relative}")
    return len(entries)


def _validate_contract(config: dict[str, object]) -> None:
    if _string(config.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise S4PreflightError("unsupported S4 config schema version")
    if _string(config.get("experiment_id"), "experiment_id") != EXPERIMENT_ID:
        raise S4PreflightError("unexpected S4 experiment_id")
    if _string(config.get("status"), "status") not in {
        "BLOCKED_PREREQUISITES",
        "READY_FOR_S4_EXECUTION",
    }:
        raise S4PreflightError("unsupported frozen S4 config status")
    if not _exact_bool(config.get("preflight_only"), "preflight_only"):
        raise S4PreflightError("S4 config must be preflight-only")
    if _exact_bool(config.get("execution_enabled"), "execution_enabled"):
        raise S4PreflightError("S4 real execution must remain disabled")
    if _exact_bool(config.get("dataset_root_argument_allowed"), "dataset root argument"):
        raise S4PreflightError("S4 preflight must not accept a dataset root")
    if _exact_bool(config.get("environment_root_discovery_allowed"), "environment root discovery"):
        raise S4PreflightError("S4 preflight must not discover roots from the environment")
    scope = _mapping(config.get("scope"), "scope")
    _expected_sequence(scope.get("dataset_order"), ("IDRID",), "scope.dataset_order")
    _expected_sequence(scope.get("lesion_classes"), _EXPECTED_CLASSES, "scope.lesion_classes")
    _expected_sequence(
        scope.get("development_splits"), _EXPECTED_SPLITS, "scope.development_splits"
    )
    if not _exact_bool(scope.get("locked_test_membership_only"), "locked membership only"):
        raise S4PreflightError("locked test must be membership-only")
    if _exact_bool(scope.get("locked_test_root_reads_allowed"), "locked root reads"):
        raise S4PreflightError("locked-test root reads must be forbidden")
    truth = _mapping(config.get("truth_contract"), "truth_contract")
    _expected_sequence(truth.get("axes"), _EXPECTED_AXES, "truth axes")
    _expected_sequence(
        truth.get("confirmatory_oracle_tiers"),
        _EXPECTED_CONFIRMATORY_TIERS,
        "confirmatory tiers",
    )
    _expected_sequence(
        truth.get("exploratory_oracle_tiers"),
        _EXPECTED_EXPLORATORY_TIERS,
        "exploratory tiers",
    )
    if _string(truth.get("unresolved_state"), "unresolved state") != "ORACLE_UNRESOLVED":
        raise S4PreflightError("unresolved Oracle outcomes must remain ORACLE_UNRESOLVED")
    if not _exact_bool(truth.get("retinaentail_independent"), "Oracle independence"):
        raise S4PreflightError("support Oracle must be RetinaEntail-independent")


def run_s4_preflight(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    git_head: str | None = None,
) -> S4PreflightResult:
    """Replay frozen repository/S3 evidence without accepting or discovering a dataset root."""

    resolved_root = project_root.resolve(strict=True)
    resolved_config = config_path.resolve(strict=True)
    try:
        resolved_config.relative_to(resolved_root)
    except ValueError as error:
        raise S4PreflightError("config path must remain within project root") from error
    config = _load_yaml(resolved_config, "S4 config")
    _validate_contract(config)
    artifact_root_relative = _relative_path(config.get("artifact_root"), "artifact_root")
    expected_output = (resolved_root / artifact_root_relative).resolve(strict=False)
    if output_dir.resolve(strict=False) != expected_output:
        raise S4PreflightError("output directory must match the frozen artifact_root")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing S4 output directory: {output_dir}")

    checks: list[dict[str, JSONValue]] = []
    bindings = _mapping(config.get("bindings"), "bindings")
    expected_commit = _git_oid(
        bindings.get("expected_upstream_s3_commit"), "expected_upstream_s3_commit"
    )
    if git_head is None:
        commit_is_ancestor = _git_commit_is_ancestor(resolved_root, expected_commit)
    else:
        commit_is_ancestor = _git_oid(git_head, "git_head") == expected_commit
    _check(
        checks,
        "S3_BASELINE_COMMIT_ANCESTRY",
        "PRESENT_AND_ANCESTOR",
        "PRESENT_AND_ANCESTOR" if commit_is_ancestor else "MISSING_OR_NOT_ANCESTOR",
    )

    prompt_binding, prompt_path = _binding_path(resolved_root, bindings, "execution_prompt")
    _check(
        checks,
        "EXECUTION_PROMPT_SHA256",
        _sha256(prompt_binding.get("sha256"), "execution prompt sha256"),
        sha256_file(prompt_path),
    )

    owner_binding, owner_path = _binding_path(resolved_root, bindings, "owner_scope_confirmation")
    _check(
        checks,
        "OWNER_SCOPE_CONFIRMATION_SHA256",
        _sha256(owner_binding.get("sha256"), "owner scope confirmation sha256"),
        sha256_file(owner_path),
    )
    owner_record = _load_json(owner_path, "S4 owner scope confirmation")
    _check(
        checks,
        "OWNER_SCOPE_CONFIRMATION_DECISION",
        "PASS",
        _string(owner_record.get("decision"), "owner scope confirmation decision"),
    )

    catalog_qc_binding, catalog_qc_path = _binding_path(resolved_root, bindings, "ma_se_catalog_qc")
    _check(
        checks,
        "MA_SE_CATALOG_QC_SHA256",
        _sha256(catalog_qc_binding.get("sha256"), "MA/SE catalog QC sha256"),
        sha256_file(catalog_qc_path),
    )
    catalog_qc = _load_json(catalog_qc_path, "MA/SE catalog QC record")
    _check(
        checks,
        "MA_SE_CATALOG_QC_DECISION",
        "PASS",
        _string(catalog_qc.get("decision"), "MA/SE catalog QC decision"),
    )

    summary_binding, summary_path = _binding_path(resolved_root, bindings, "s3_summary_config")
    _check(
        checks,
        "S3_SUMMARY_CONFIG_SHA256",
        _sha256(summary_binding.get("sha256"), "S3 summary sha256"),
        sha256_file(summary_path),
    )
    summary = _load_yaml(summary_path, "S3 summary config")
    _check(checks, "S3_SUMMARY_STATUS", "PASS", _string(summary.get("status"), "S3 status"))

    run_binding, run_path = _binding_path(resolved_root, bindings, "s3_frozen_run_config")
    _check(
        checks,
        "S3_RUN_CONFIG_CANONICAL_SHA256",
        _sha256(run_binding.get("canonical_sha256"), "S3 run config canonical sha256"),
        _canonical_yaml_sha256(run_path),
    )

    qc_binding, qc_path = _binding_path(resolved_root, bindings, "s3_qc_record")
    _check(
        checks,
        "S3_QC_RECORD_SHA256",
        _sha256(qc_binding.get("sha256"), "S3 QC sha256"),
        sha256_file(qc_path),
    )
    qc_record = _load_json(qc_path, "S3 QC record")
    _check(checks, "S3_QC_DECISION", "PASS", _string(qc_record.get("decision"), "S3 decision"))

    audit_binding, audit_path = _binding_path(resolved_root, bindings, "s3_generated_audit")
    _check(
        checks,
        "S3_GENERATED_AUDIT_SHA256",
        _sha256(audit_binding.get("sha256"), "S3 audit sha256"),
        sha256_file(audit_path),
    )
    generated_audit = _load_json(audit_path, "S3 generated audit")
    _check(
        checks,
        "S3_GENERATED_AUDIT_STATUS",
        "READY_FOR_RESPONSIBLE_CONFIRMATION",
        _string(generated_audit.get("status"), "S3 generated audit status"),
    )

    artifact_binding, artifact_manifest_path = _binding_path(
        resolved_root, bindings, "s3_artifact_manifest"
    )
    _check(
        checks,
        "S3_ARTIFACT_MANIFEST_SHA256",
        _sha256(artifact_binding.get("sha256"), "S3 artifact manifest sha256"),
        sha256_file(artifact_manifest_path),
    )
    artifact_file_count = _validate_artifact_manifest(resolved_root, artifact_binding)
    _check(
        checks,
        "S3_ARTIFACT_ENTRIES",
        f"{artifact_file_count}/{artifact_file_count} matched",
        f"{artifact_file_count}/{artifact_file_count} matched",
    )

    pilot_binding, pilot_path = _binding_path(resolved_root, bindings, "s3_pilot_manifest")
    _check(
        checks,
        "S3_PILOT_MANIFEST_SHA256",
        _sha256(pilot_binding.get("sha256"), "S3 pilot manifest sha256"),
        sha256_file(pilot_path),
    )
    pilot = _load_json(pilot_path, "S3 pilot manifest")
    accepted = _sequence(pilot.get("accepted"), "S3 accepted twins")
    rejected = _sequence(pilot.get("rejected"), "S3 rejected attempts")
    _check(checks, "S3_ACCEPTED_REJECTED_COUNTS", "40/2", f"{len(accepted)}/{len(rejected)}")

    implementation = _mapping(bindings.get("s3_implementation"), "s3 implementation binding")
    _check(
        checks,
        "S3_IMPLEMENTATION_BUNDLE_SHA256",
        _sha256(implementation.get("canonical_bundle_sha256"), "implementation bundle sha256"),
        _implementation_bundle_sha256(resolved_root, implementation.get("files")),
    )

    oracle_spec_binding, oracle_spec_path = _binding_path(resolved_root, bindings, "oracle_spec")
    _check(
        checks,
        "ORACLE_SPEC_SHA256",
        _sha256(oracle_spec_binding.get("sha256"), "Oracle spec sha256"),
        sha256_file(oracle_spec_path),
    )
    oracle_code_binding, oracle_code_path = _binding_path(
        resolved_root, bindings, "oracle_implementation"
    )
    _check(
        checks,
        "ORACLE_IMPLEMENTATION_SHA256",
        _sha256(oracle_code_binding.get("sha256"), "Oracle implementation sha256"),
        sha256_file(oracle_code_path),
    )

    prerequisites: list[dict[str, JSONValue]] = []
    blockers: list[str] = []
    for index, raw_item in enumerate(_sequence(config.get("prerequisites"), "prerequisites")):
        item = _mapping(raw_item, f"prerequisite {index}")
        item_id = _string(item.get("id"), f"prerequisite {index} id")
        item_status = _string(item.get("status"), f"prerequisite {index} status")
        if item_status not in {"PASS", "MISSING", "FAIL"}:
            raise S4PreflightError(f"unsupported prerequisite status: {item_status}")
        blocking = _exact_bool(item.get("blocking"), f"prerequisite {index} blocking")
        evidence = _string(item.get("evidence"), f"prerequisite {index} evidence")
        prerequisites.append(
            {"id": item_id, "status": item_status, "blocking": blocking, "evidence": evidence}
        )
        if blocking and item_status != "PASS":
            blockers.append(item_id)

    failed_checks = [cast(str, item["id"]) for item in checks if item["status"] != "PASS"]
    if failed_checks:
        status = "FAIL_CLOSED"
        blockers = [f"UPSTREAM_CHECK_FAILED:{item}" for item in failed_checks] + blockers
    elif blockers:
        status = "BLOCKED_PREREQUISITES"
    else:
        status = "READY_FOR_S4_EXECUTION"

    scope = _mapping(config.get("scope"), "scope")
    truth = _mapping(config.get("truth_contract"), "truth_contract")
    prohibited = [
        _string(item, "prohibited item")
        for item in _sequence(config.get("prohibited"), "prohibited")
    ]
    config_sha256 = sha256_bytes(canonical_json_bytes(_json_safe(config)))
    payload: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "preflight_only": True,
        "config_sha256": config_sha256,
        "scope": {
            "dataset_order": _json_safe(scope.get("dataset_order")),
            "lesion_classes": _json_safe(scope.get("lesion_classes")),
            "development_splits": _json_safe(scope.get("development_splits")),
            "locked_test_membership_only": True,
        },
        "oracle_contract": {
            "truth_axes": _json_safe(truth.get("axes")),
            "confirmatory_tiers": _json_safe(truth.get("confirmatory_oracle_tiers")),
            "exploratory_tiers": _json_safe(truth.get("exploratory_oracle_tiers")),
            "unresolved_state": "ORACLE_UNRESOLVED",
            "retinaentail_independent": True,
            "forbidden_inputs": _json_safe(truth.get("forbidden_oracle_inputs")),
        },
        "upstream_checks": _json_safe(checks),
        "prerequisites": _json_safe(prerequisites),
        "blockers": _json_safe(blockers),
        "io_audit": {
            "dataset_root_argument_accepted": False,
            "environment_root_discovery": False,
            "dataset_root_reads": 0,
            "locked_test_root_reads": 0,
            "image_decodes": 0,
            "mask_decodes": 0,
            "s3_artifact_files_hashed": artifact_file_count,
        },
        "prohibited": _json_safe(prohibited),
    }
    preflight_bytes = canonical_json_bytes(payload)
    write_bytes_fail_safe(output_dir / "preflight.json", preflight_bytes)
    manifest: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "relative_path": "preflight.json",
                "byte_size": len(preflight_bytes),
                "sha256": sha256_bytes(preflight_bytes),
            }
        ],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", manifest_bytes)
    hashes = {
        "preflight.json": sha256_bytes(preflight_bytes),
        "artifact-manifest.json": sha256_bytes(manifest_bytes),
    }
    return S4PreflightResult(
        status=status,
        blockers=tuple(blockers),
        payload=payload,
        artifact_hashes=hashes,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retinaentail.science_s4",
        description="Run the zero-dataset-I/O Science S4 preregistration preflight.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="Replay repository/S3 bindings without accepting a dataset root."
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    return parser


def _cli_relative(value: Path, label: str) -> str:
    return _relative_path(value.as_posix(), label)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = _project_root()
    try:
        config_relative = _cli_relative(args.config, "--config")
        output_relative = _cli_relative(args.output_dir, "--output-dir")
        result = run_s4_preflight(
            project_root=project_root,
            config_path=project_root / config_relative,
            output_dir=project_root / output_relative,
        )
    except (FileExistsError, OSError, S4PreflightError) as error:
        print(f"Science S4 preflight fail-closed: {error}", file=sys.stderr)
        return 2
    print(f"status: {result.status}")
    print(f"blockers: {len(result.blockers)}")
    for blocker in result.blockers:
        print(f"- {blocker}")
    print("dataset root reads: 0")
    print("locked-test root reads: 0")
    print("image decodes: 0")
    print("mask decodes: 0")
    print(f"preflight hash: {result.artifact_hashes['preflight.json']}")
    print(f"artifact manifest hash: {result.artifact_hashes['artifact-manifest.json']}")
    return 0 if result.status == "READY_FOR_S4_EXECUTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
