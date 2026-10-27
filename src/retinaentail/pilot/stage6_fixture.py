"""Offline Stage 6 fixture for complete-package verifier capabilities.

This module never accepts a real dataset root.  It embeds already generated
synthetic Stage 2, Stage 3, and Stage 5 packages, regenerates both process-bound
verifier capabilities, and publishes a byte-addressed outer package.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

import yaml

from retinaentail.calibration.stage3_package import (
    Stage3PackageScope,
    replay_stage3_package,
    stage3_package_capability_snapshot,
    verify_and_issue_stage3_package,
)
from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.data.stage2_package import (
    Stage2PackageScope,
    replay_stage2_package,
    stage2_package_capability_snapshot,
    verify_and_issue_stage2_package,
)
from retinaentail.entailment.states import EntailmentState
from retinaentail.pilot.toy_fixture import verify_stage5_toy_package
from retinaentail.reporting.stage3 import write_stage3_bundle
from retinaentail.reporting.stage6 import (
    STAGE6_AUTHORITY_BOUNDARY,
    STAGE6_DIGEST_METHOD,
    STAGE6_METHOD,
    issue_stage6_pilot_report,
    render_stage6_pilot_html,
    render_stage6_pilot_json,
)
from retinaentail.witnesses.replay import sha256_bytes

STAGE6_TOY_CONFIG_VERSION: Final = "6.0"
STAGE6_PACKAGE_VERSION: Final = "6.0.0"
_MAX_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES: Final = 128 * 1024 * 1024
_MAX_PACKAGE_FILES: Final = 512
_MAX_DEPTH: Final = 16
_ROOT_FILES: Final = frozenset(
    {
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "model_provenance.json",
        "source_packages.json",
        "summary.json",
    }
)


class Stage6ToyConfigurationError(ValueError):
    """Raised when a synthetic Stage 6 declaration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6SourceContract:
    stage2_scope: str
    stage2_dataset_id: str
    stage2_dataset_version: str
    stage2_audit_status: str
    stage3_scope: str
    stage5_scope: str
    stage5_pilot_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage2_scope": self.stage2_scope,
            "stage2_dataset_id": self.stage2_dataset_id,
            "stage2_dataset_version": self.stage2_dataset_version,
            "stage2_audit_status": self.stage2_audit_status,
            "stage3_scope": self.stage3_scope,
            "stage5_scope": self.stage5_scope,
            "stage5_pilot_id": self.stage5_pilot_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage6ToyConfig:
    run_id: str
    source_contract: Stage6SourceContract
    confidence_level: float
    threshold_policy: str
    schema_version: str = STAGE6_TOY_CONFIG_VERSION
    method: str = STAGE6_METHOD
    process_bound_capabilities_required: bool = True
    capabilities_serialized: bool = False
    detached_summaries_authoritative: bool = False
    test_tuning_prohibited: bool = True
    real_data_access: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "method": self.method,
            "source_contract": self.source_contract.to_dict(),
            "confidence_level": self.confidence_level,
            "threshold_policy": self.threshold_policy,
            "process_bound_capabilities_required": self.process_bound_capabilities_required,
            "capabilities_serialized": self.capabilities_serialized,
            "detached_summaries_authoritative": self.detached_summaries_authoritative,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "real_data_access": self.real_data_access,
        }


@dataclass(frozen=True, slots=True)
class Stage6ToyRunResult:
    """Non-authoritative receipt from one full Stage 6 package verification."""

    output_dir: Path
    package_sha256: str
    package_file_count: int
    stage2_package_sha256: str
    stage3_package_sha256: str
    stage5_package_sha256: str
    case_count: int
    states: tuple[EntailmentState, ...]
    verified: bool


@dataclass(frozen=True, slots=True)
class _Prepared:
    config: Stage6ToyConfig
    root_artifacts: Mapping[str, bytes]
    stage2_descriptor: Mapping[str, object]
    stage3_descriptor: Mapping[str, object]
    stage5_descriptor: Mapping[str, object]
    states: tuple[EntailmentState, ...]


def parse_stage6_toy_config(value: object) -> Stage6ToyConfig:
    data = _strict_mapping(
        value,
        "Stage 6 config",
        {
            "schema_version",
            "run_id",
            "method",
            "source_contract",
            "confidence_level",
            "threshold_policy",
            "process_bound_capabilities_required",
            "capabilities_serialized",
            "detached_summaries_authoritative",
            "test_tuning_prohibited",
            "real_data_access",
        },
    )
    if data["schema_version"] != STAGE6_TOY_CONFIG_VERSION or data["method"] != STAGE6_METHOD:
        raise Stage6ToyConfigurationError("unsupported Stage 6 config version or method")
    source_data = _strict_mapping(
        data["source_contract"],
        "source_contract",
        {
            "stage2_scope",
            "stage2_dataset_id",
            "stage2_dataset_version",
            "stage2_audit_status",
            "stage3_scope",
            "stage5_scope",
            "stage5_pilot_id",
        },
    )
    source = Stage6SourceContract(
        stage2_scope=_text(source_data["stage2_scope"], "stage2_scope"),
        stage2_dataset_id=_text(source_data["stage2_dataset_id"], "stage2_dataset_id"),
        stage2_dataset_version=_text(
            source_data["stage2_dataset_version"], "stage2_dataset_version"
        ),
        stage2_audit_status=_text(source_data["stage2_audit_status"], "stage2_audit_status"),
        stage3_scope=_text(source_data["stage3_scope"], "stage3_scope"),
        stage5_scope=_text(source_data["stage5_scope"], "stage5_scope"),
        stage5_pilot_id=_text(source_data["stage5_pilot_id"], "stage5_pilot_id"),
    )
    expected_source = {
        "stage2_scope": "SYNTHETIC_FIXTURE",
        "stage2_dataset_id": "synthetic_stage2",
        "stage2_dataset_version": "synthetic-fixture-v1",
        "stage2_audit_status": "PASS",
        "stage3_scope": "SYNTHETIC_TOY",
        "stage5_scope": "SYNTHETIC_ONLY",
        "stage5_pilot_id": "stage5-toy-locked-pilot-001",
    }
    if source.to_dict() != expected_source:
        raise Stage6ToyConfigurationError("Stage 6 source contract is not the fixed fixture")
    confidence = _number(data["confidence_level"], "confidence_level")
    if confidence != 0.95:
        raise Stage6ToyConfigurationError("Stage 6 fixture confidence_level must be 0.95")
    config = Stage6ToyConfig(
        run_id=_text(data["run_id"], "run_id"),
        source_contract=source,
        confidence_level=confidence,
        threshold_policy=_text(data["threshold_policy"], "threshold_policy"),
        process_bound_capabilities_required=_boolean(
            data["process_bound_capabilities_required"],
            "process_bound_capabilities_required",
        ),
        capabilities_serialized=_boolean(
            data["capabilities_serialized"], "capabilities_serialized"
        ),
        detached_summaries_authoritative=_boolean(
            data["detached_summaries_authoritative"],
            "detached_summaries_authoritative",
        ),
        test_tuning_prohibited=_boolean(data["test_tuning_prohibited"], "test_tuning_prohibited"),
        real_data_access=_boolean(data["real_data_access"], "real_data_access"),
    )
    if (
        config.run_id != "stage6-toy-verifier-pilot-001"
        or config.threshold_policy != "LOCKED_STAGE3_NO_REESTIMATION"
        or config.process_bound_capabilities_required is not True
        or config.capabilities_serialized is not False
        or config.detached_summaries_authoritative is not False
        or config.test_tuning_prohibited is not True
        or config.real_data_access is not False
    ):
        raise Stage6ToyConfigurationError("Stage 6 fixture safety declarations are invalid")
    return config


def load_stage6_toy_config(path: Path) -> Stage6ToyConfig:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if len(payload) > 64 * 1024:
        raise Stage6ToyConfigurationError("Stage 6 config exceeds its size limit")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Stage6ToyConfigurationError("Stage 6 config is not strict UTF-8 YAML") from error
    return parse_stage6_toy_config(value)


def _prepare(
    config: Stage6ToyConfig,
    *,
    stage2_root: Path,
    stage3_root: Path,
    stage5_root: Path,
    stage2_files: Mapping[str, bytes],
    stage3_files: Mapping[str, bytes],
    stage5_files: Mapping[str, bytes],
) -> _Prepared:
    stage2_capability = verify_and_issue_stage2_package(
        stage2_root,
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
    )
    stage2_replay = replay_stage2_package(stage2_capability)
    if not stage2_replay.valid:
        raise ValueError("complete Stage 2 package replay did not verify")
    stage2_snapshot = stage2_package_capability_snapshot(stage2_capability)

    stage3_capability = verify_and_issue_stage3_package(
        stage3_root,
        scope=Stage3PackageScope.SYNTHETIC_TOY,
        stage2_capability=stage2_capability,
    )
    stage3_replay = replay_stage3_package(stage3_capability)
    stage3_snapshot = stage3_package_capability_snapshot(stage3_capability)
    if stage3_replay != stage3_snapshot:
        raise ValueError("complete Stage 3 package replay did not match its capability")

    verified_stage5 = verify_stage5_toy_package(stage5_root)
    if not verified_stage5.verified:
        raise ValueError("complete Stage 5 package replay did not verify")
    stage5_descriptor = _stage5_descriptor(stage5_files)
    pilot_audit = _json_mapping(stage5_files["audit.json"], "Stage 5 audit")

    stage2_descriptor = _normalized_stage2_descriptor(stage2_snapshot)
    stage3_descriptor = _normalized_stage3_descriptor(stage3_snapshot)
    _validate_config_sources(
        config,
        stage2=stage2_descriptor,
        stage3=stage3_descriptor,
        stage5=stage5_descriptor,
        pilot_audit=pilot_audit,
    )
    cases = pilot_audit.get("case_index")
    if not isinstance(cases, list):
        raise ValueError("Stage 5 audit case_index is missing")
    states = tuple(EntailmentState(cast(str, _mapping(item, "case")["state"])) for item in cases)

    config_bytes = _canonical_json_bytes(config.to_dict())
    config_sha256 = sha256_bytes(config_bytes)
    source_payload = _source_descriptor(stage2_files, stage3_files, stage5_files)
    source_payload_sha256 = sha256_bytes(_canonical_json_bytes(source_payload))
    model_provenance = {
        "schema_version": STAGE6_PACKAGE_VERSION,
        "component": "stage6_package_orchestrator",
        "method": STAGE6_METHOD,
        "model_kind": "none_orchestration_only",
        "trained": False,
        "pretrained_weights": False,
        "threshold_source": "verified_complete_stage3_package",
        "test_threshold_reestimation": False,
        "clinical_claim": False,
    }
    model_bytes = _canonical_json_bytes(model_provenance)
    model_sha256 = sha256_bytes(model_bytes)
    report = issue_stage6_pilot_report(
        run_id=config.run_id,
        stage2_verifier=stage2_descriptor,
        stage3_verifier=stage3_descriptor,
        stage5_source=stage5_descriptor,
        pilot_audit=pilot_audit,
        stage6_config_sha256=config_sha256,
        stage6_package_payload_sha256=source_payload_sha256,
        stage6_model_provenance_sha256=model_sha256,
    )
    summary = {
        "schema_version": STAGE6_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "run_id": config.run_id,
        "method": STAGE6_METHOD,
        "stage2_package_sha256": stage2_descriptor["package_sha256"],
        "stage3_package_sha256": stage3_descriptor["package_sha256"],
        "stage5_package_sha256": stage5_descriptor["package_sha256"],
        "stage6_config_sha256": config_sha256,
        "stage6_package_payload_sha256": source_payload_sha256,
        "stage6_model_provenance_sha256": model_sha256,
        "case_count": len(states),
        "state_counts": _mapping(
            _mapping(pilot_audit["evaluation"], "evaluation")["state_counts"],
            "state_counts",
        ),
        "fail_closed_count": _mapping(pilot_audit["evaluation"], "evaluation")["fail_closed_count"],
        "out_of_distribution_count": _mapping(pilot_audit["evaluation"], "evaluation")[
            "out_of_distribution_count"
        ],
        "both_witness_replays_passed_count": _mapping(pilot_audit["evaluation"], "evaluation")[
            "both_witness_replays_passed_count"
        ],
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "capabilities_serialized": False,
        "detached_summaries_authoritative": False,
        "real_data_gate": "BLOCKED_NO_EXPLICIT_APPROVED_DATASET",
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "authority_boundary": STAGE6_AUTHORITY_BOUNDARY,
        "clinical_claim": False,
    }
    sources = {
        "schema_version": STAGE6_PACKAGE_VERSION,
        "digest_method": STAGE6_DIGEST_METHOD,
        "stage2": dict(stage2_descriptor),
        "stage3": dict(stage3_descriptor),
        "stage5": dict(stage5_descriptor),
        "source_payload": source_payload,
    }
    artifacts = {
        "audit.html": render_stage6_pilot_html(report).encode("utf-8"),
        "audit.json": render_stage6_pilot_json(report).encode("utf-8"),
        "config_snapshot.json": config_bytes,
        "model_provenance.json": model_bytes,
        "source_packages.json": _canonical_json_bytes(sources),
        "summary.json": _canonical_json_bytes(summary),
    }
    return _Prepared(
        config=config,
        root_artifacts=artifacts,
        stage2_descriptor=stage2_descriptor,
        stage3_descriptor=stage3_descriptor,
        stage5_descriptor=stage5_descriptor,
        states=states,
    )


def run_stage6_toy_fixture(
    config_path: Path,
    stage2_package: Path,
    stage3_package: Path,
    stage5_package: Path,
    output_dir: Path,
) -> Stage6ToyRunResult:
    """Create an outer package only after all three complete sources replay."""

    paths = (config_path, stage2_package, stage3_package, stage5_package, output_dir)
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("all Stage 6 paths must be pathlib.Path values")
    source_roots = tuple(path.resolve(strict=True) for path in paths[:4])
    target = output_dir.resolve(strict=False)
    _require_nonoverlap((*source_roots, target))
    config = load_stage6_toy_config(config_path)
    stage2_files = _read_tree(stage2_package, label="Stage 2")
    stage3_files = _read_tree(stage3_package, label="Stage 3")
    stage5_files = _read_tree(stage5_package, label="Stage 5")
    prepared = _prepare(
        config,
        stage2_root=stage2_package,
        stage3_root=stage3_package,
        stage5_root=stage5_package,
        stage2_files=stage2_files,
        stage3_files=stage3_files,
        stage5_files=stage5_files,
    )
    payloads = dict(prepared.root_artifacts)
    payloads.update({f"stage2/package/{name}": data for name, data in stage2_files.items()})
    payloads.update({f"stage3/package/{name}": data for name, data in stage3_files.items()})
    payloads.update({f"stage5/package/{name}": data for name, data in stage5_files.items()})
    package_manifest = _package_manifest(payloads)
    payloads["package_manifest.json"] = _canonical_json_bytes(package_manifest)
    write_stage3_bundle(
        output_dir,
        payloads,
        boundary=AuditWriteBoundary(allowed_root=output_dir),
    )
    return verify_stage6_toy_package(output_dir)


def verify_stage6_toy_package(output_dir: Path) -> Stage6ToyRunResult:
    """Reissue both capabilities, replay Stage 5, and byte-compare the package."""

    actual = _read_tree(output_dir, label="Stage 6")
    if "package_manifest.json" not in actual:
        raise ValueError("Stage 6 package manifest is missing")
    manifest = _json_mapping(actual["package_manifest.json"], "Stage 6 package manifest")
    _verify_package_manifest(manifest, actual)
    root_files = {name: actual[name] for name in _ROOT_FILES}
    stage2_files = _prefixed(actual, "stage2/package/")
    stage3_files = _prefixed(actual, "stage3/package/")
    stage5_files = _prefixed(actual, "stage5/package/")
    if set(actual) != {
        "package_manifest.json",
        *_ROOT_FILES,
        *(f"stage2/package/{name}" for name in stage2_files),
        *(f"stage3/package/{name}" for name in stage3_files),
        *(f"stage5/package/{name}" for name in stage5_files),
    }:
        raise ValueError("Stage 6 package contains an unknown path")
    config = parse_stage6_toy_config(
        _decode_json(root_files["config_snapshot.json"], "Stage 6 config snapshot")
    )
    prepared = _prepare(
        config,
        stage2_root=output_dir / "stage2" / "package",
        stage3_root=output_dir / "stage3" / "package",
        stage5_root=output_dir / "stage5" / "package",
        stage2_files=stage2_files,
        stage3_files=stage3_files,
        stage5_files=stage5_files,
    )
    if dict(prepared.root_artifacts) != root_files:
        raise ValueError("Stage 6 root report artifacts do not regenerate byte-for-byte")
    expected_payloads = dict(prepared.root_artifacts)
    expected_payloads.update(
        {f"stage2/package/{name}": data for name, data in stage2_files.items()}
    )
    expected_payloads.update(
        {f"stage3/package/{name}": data for name, data in stage3_files.items()}
    )
    expected_payloads.update(
        {f"stage5/package/{name}": data for name, data in stage5_files.items()}
    )
    expected_manifest = _package_manifest(expected_payloads)
    if manifest != expected_manifest:
        raise ValueError("Stage 6 package manifest does not match regenerated artifacts")
    return Stage6ToyRunResult(
        output_dir=output_dir,
        package_sha256=cast(str, expected_manifest["package_sha256"]),
        package_file_count=len(actual),
        stage2_package_sha256=cast(str, prepared.stage2_descriptor["package_sha256"]),
        stage3_package_sha256=cast(str, prepared.stage3_descriptor["package_sha256"]),
        stage5_package_sha256=cast(str, prepared.stage5_descriptor["package_sha256"]),
        case_count=len(prepared.states),
        states=prepared.states,
        verified=True,
    )


def _normalized_stage2_descriptor(value: object) -> dict[str, object]:
    source = _object_mapping(value, "Stage 2 capability snapshot")
    return {
        "scope": _enum_text(source["scope"]),
        "dataset_id": source["dataset_id"],
        "dataset_version": source["dataset_version"],
        "audit_status": "PASS",
        "record_count": source["record_count"],
        "manifest_sha256": source["manifest_sha256"],
        "package_sha256": source["package_sha256"],
        "package_file_count": len(cast(Sequence[object], source["files"])),
        "replay_verified": True,
        "capability_issued": True,
    }


def _normalized_stage3_descriptor(value: object) -> dict[str, object]:
    source = _object_mapping(value, "Stage 3 capability snapshot")
    aliases = {
        "calibration_artifact_sha256": (
            "calibration_artifact_sha256",
            "calibration_artifact_hash",
            "artifact_sha256",
        ),
        "config_sha256": ("config_sha256", "config_hash"),
        "data_sha256": ("data_sha256", "data_hash"),
        "model_sha256": ("model_sha256", "model_hash"),
    }
    result: dict[str, object] = {
        "scope": _enum_text(source["scope"]),
        "package_sha256": source["package_sha256"],
        "package_file_count": source["package_file_count"],
        "alpha": source["alpha"],
        "positive_threshold": source["positive_threshold"],
        "negative_threshold": source["negative_threshold"],
        "positive_sample_count": source["positive_sample_count"],
        "negative_sample_count": source["negative_sample_count"],
        "total_sample_count": source["total_sample_count"],
        "replay_verified": True,
        "capability_issued": True,
    }
    for target, names in aliases.items():
        for name in names:
            if name in source:
                result[target] = source[name]
                break
        else:
            raise ValueError(f"Stage 3 capability snapshot is missing {target}")
    return result


def _stage5_descriptor(files: Mapping[str, bytes]) -> dict[str, object]:
    summary = _json_mapping(files["summary.json"], "Stage 5 summary")
    descriptor = _file_descriptor(files)
    return {
        "scope": summary.get("scope"),
        "pilot_id": summary.get("pilot_id"),
        "package_sha256": descriptor["package_sha256"],
        "package_file_count": descriptor["file_count"],
        "case_count": summary.get("case_count"),
        "replay_verified": True,
    }


def _validate_config_sources(
    config: Stage6ToyConfig,
    *,
    stage2: Mapping[str, object],
    stage3: Mapping[str, object],
    stage5: Mapping[str, object],
    pilot_audit: Mapping[str, object],
) -> None:
    source = config.source_contract
    observed = (
        stage2["scope"],
        stage2["dataset_id"],
        stage2["dataset_version"],
        stage2["audit_status"],
        stage3["scope"],
        stage5["scope"],
        stage5["pilot_id"],
    )
    expected = (
        source.stage2_scope,
        source.stage2_dataset_id,
        source.stage2_dataset_version,
        source.stage2_audit_status,
        source.stage3_scope,
        source.stage5_scope,
        source.stage5_pilot_id,
    )
    if observed != expected or pilot_audit.get("pilot_id") != source.stage5_pilot_id:
        raise ValueError("complete source packages do not match the Stage 6 config")


def _source_descriptor(
    stage2: Mapping[str, bytes],
    stage3: Mapping[str, bytes],
    stage5: Mapping[str, bytes],
) -> dict[str, object]:
    return {
        "schema_version": STAGE6_PACKAGE_VERSION,
        "digest_method": STAGE6_DIGEST_METHOD,
        "stage2": _file_descriptor(stage2),
        "stage3": _file_descriptor(stage3),
        "stage5": _file_descriptor(stage5),
    }


def _file_descriptor(files: Mapping[str, bytes]) -> dict[str, object]:
    entries = [
        {
            "relative_path": name,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in sorted(files.items())
    ]
    base = {"digest_method": STAGE6_DIGEST_METHOD, "file_count": len(entries), "files": entries}
    return {**base, "package_sha256": sha256_bytes(_canonical_json_bytes(base))}


def _package_manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    descriptor = _file_descriptor(payloads)
    return {
        "schema_version": STAGE6_PACKAGE_VERSION,
        "digest_method": descriptor["digest_method"],
        "file_count": descriptor["file_count"],
        "files": descriptor["files"],
        "package_sha256": descriptor["package_sha256"],
    }


def _verify_package_manifest(value: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    data = _strict_mapping(
        value,
        "Stage 6 package manifest",
        {"schema_version", "digest_method", "file_count", "files", "package_sha256"},
    )
    if (
        data["schema_version"] != STAGE6_PACKAGE_VERSION
        or data["digest_method"] != STAGE6_DIGEST_METHOD
    ):
        raise ValueError("Stage 6 package manifest version or method is invalid")
    payloads = {
        name: payload for name, payload in actual.items() if name != "package_manifest.json"
    }
    expected = _package_manifest(payloads)
    if dict(value) != expected:
        raise ValueError("Stage 6 package manifest does not match package bytes")


def _read_tree(root: Path, *, label: str) -> dict[str, bytes]:
    if not isinstance(root, Path):
        raise TypeError(f"{label} package root must be a pathlib.Path")
    _reject_link_components(root)
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or _is_link_or_reparse(root_stat):
        raise ValueError(f"{label} package root must be a physical directory")
    files: dict[str, bytes] = {}
    total = 0

    def visit(directory: Path, relative: PurePosixPath, depth: int) -> None:
        nonlocal total
        if depth > _MAX_DEPTH:
            raise ValueError(f"{label} package exceeds its depth limit")
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda item: item.name)
        for entry in ordered:
            # On Windows ``DirEntry.stat`` may report zero device/inode values while
            # ``lstat`` exposes the stable file identity used after the read.
            entry_stat = os.lstat(entry.path)
            if _is_link_or_reparse(entry_stat):
                raise ValueError(f"{label} package contains a link or reparse point")
            child_relative = relative / entry.name
            name = child_relative.as_posix()
            _relative_name(name)
            if entry.is_dir(follow_symlinks=False):
                before_count = len(files)
                visit(Path(entry.path), child_relative, depth + 1)
                if len(files) == before_count:
                    raise ValueError(f"{label} package contains an empty directory")
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"{label} package contains a non-regular entry")
            if entry_stat.st_size > _MAX_FILE_BYTES:
                raise ValueError(f"{label} package artifact exceeds its size limit")
            payload = _read_regular_bounded(Path(entry.path), entry_stat)
            post = os.lstat(entry.path)
            if _identity(entry_stat) != _identity(post) or len(payload) != entry_stat.st_size:
                raise ValueError(f"{label} package artifact changed while being read")
            total += len(payload)
            if total > _MAX_PACKAGE_BYTES or len(files) >= _MAX_PACKAGE_FILES:
                raise ValueError(f"{label} package exceeds its aggregate limit")
            files[name] = payload

    visit(root, PurePosixPath(), 0)
    if not files:
        raise ValueError(f"{label} package must not be empty")
    return files


def _read_regular_bounded(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(expected) or _is_link_or_reparse(opened):
            raise ValueError("package artifact identity changed before reading")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_FILE_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_FILE_BYTES:
                raise ValueError("package artifact exceeds its size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(after) != _identity(expected):
        raise ValueError("package artifact changed while being read")
    return b"".join(chunks)


def _reject_link_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if _is_link_or_reparse(metadata):
            raise ValueError("package path contains a link or reparse point")


def _prefixed(files: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    result = {
        name.removeprefix(prefix): payload
        for name, payload in files.items()
        if name.startswith(prefix)
    }
    if not result:
        raise ValueError(f"Stage 6 package is missing {prefix}")
    return result


def _is_link_or_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(value.st_mode) or bool(attributes & reparse)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _require_nonoverlap(paths: Sequence[Path]) -> None:
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Stage 6 source and output paths must not overlap")


def _relative_name(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError("package paths must be normalized relative POSIX paths")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _decode_json(payload: bytes, name: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite constant {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _unique_json_object(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("JSON object keys must be strings")
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_mapping(payload: bytes, name: str) -> dict[str, object]:
    return _mapping(_decode_json(payload, name), name)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(cast(Mapping[str, object], value))


def _object_mapping(value: object, name: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        return _mapping(value, name)
    fields = getattr(value, "__dataclass_fields__", None)
    if not isinstance(fields, Mapping):
        raise ValueError(f"{name} is not a descriptor")
    return {field: getattr(value, field) for field in fields if not field.startswith("_")}


def _strict_mapping(value: object, name: str, required: set[str]) -> dict[str, object]:
    result = _mapping(value, name)
    if set(result) != required:
        raise Stage6ToyConfigurationError(f"{name} fields are incomplete or unknown")
    return result


def _enum_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return _text(raw, "enum value")


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n\x00")
    ):
        raise Stage6ToyConfigurationError(f"{name} must be safe non-empty text")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise Stage6ToyConfigurationError(f"{name} must be finite")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721
        raise Stage6ToyConfigurationError(f"{name} must be an exact boolean")
    return value


__all__ = [
    "STAGE6_PACKAGE_VERSION",
    "STAGE6_TOY_CONFIG_VERSION",
    "Stage6SourceContract",
    "Stage6ToyConfig",
    "Stage6ToyConfigurationError",
    "Stage6ToyRunResult",
    "load_stage6_toy_config",
    "parse_stage6_toy_config",
    "run_stage6_toy_fixture",
    "verify_stage6_toy_package",
]
