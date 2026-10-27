"""Local dual-runner Stage 10 fixture and offline artifact-transfer verifier."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml

from retinaentail.pilot.stage9_fixture import verify_stage9_toy_package
from retinaentail.pilot.stage10_gate import (
    Stage10CapabilityError,
    consume_stage10_runner_pair,
    issue_synthetic_stage9_runner_capability,
    runner_capability_snapshot,
)
from retinaentail.reporting.stage10 import render_stage10_audit_html, render_stage10_audit_json
from retinaentail.reproducibility.stage9 import Stage9Observation, parse_stage8_observations
from retinaentail.reproducibility.stage10 import (
    STAGE10_SCHEMA_VERSION,
    Stage10DiffReport,
    Stage10DiffStatus,
    Stage10EnvironmentLock,
    Stage10RunnerSnapshot,
    canonical_json_bytes,
    canonical_sha256,
    compare_stage10_runners,
    parse_stage10_environment_lock,
)
from retinaentail.witnesses.replay import sha256_bytes, write_bytes_fail_safe

STAGE10_TOY_CONFIG_VERSION: Final = "10.0.0"
STAGE10_METHOD: Final = "local_dual_runner_signed_exchange_v1"
STAGE10_SCOPE: Final = "SYNTHETIC_FIXTURE"
STAGE10_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE10_SYNTHETIC_PACKAGE"
STAGE10_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"
STAGE10_SIGNATURE_ALGORITHM: Final = "HMAC-SHA256"
STAGE10_AUTHENTICATION_SCOPE: Final = "SYNTHETIC_SELF_CONSISTENCY_ONLY"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES: Final = 512 * 1024 * 1024
_MAX_PACKAGE_FILES: Final = 2048
_WORKER_TIMEOUT_SECONDS: Final = 900
_FIXTURE_KEY_NAMESPACE: Final = b"retinaentail-stage10-synthetic-fixture-only-v1"
_ROOT_FILES: Final = frozenset(
    {
        "artifact_diff.json",
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "environment_lock.json",
        "source_authority.json",
        "summary.json",
    }
)


class Stage10ToyConfigurationError(ValueError):
    """Raised when Stage 10 synthetic preregistration is malformed or drifts."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage10ToyConfig:
    run_id: str
    runner_ids: tuple[str, str]
    source_stage9_package_sha256: str
    source_stage9_package_file_count: int
    stage9_preregistration_sha256: str
    source_stage8_package_sha256: str
    source_stage7_package_sha256: str
    dataset_id: str
    dataset_version: str
    inclusion_case_ids: tuple[str, ...]
    replay_run_ids: tuple[str, ...]
    replay_seeds: tuple[int, ...]
    degradation_replay_set_sha256: str
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    config_hash: str
    data_hash: str
    model_hash: str
    allowed_stage9_provenance_paths: tuple[str, ...]
    exact_environment_fields: tuple[str, ...]
    confidence_level: float
    raw: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Stage10ToyRunResult:
    """Detached receipt; it grants neither runner nor data/root authority."""

    output_dir: Path
    package_sha256: str
    package_file_count: int
    runner_process_ids: tuple[int, int]
    runner_stage9_package_sha256: tuple[str, str]
    source_stage9_package_sha256: str
    case_count: int
    observation_comparison_count: int
    exact_file_match_count: int
    allowed_provenance_difference_count: int
    deterministic_file_mismatch_count: int
    status: str
    verified: bool


@dataclass(frozen=True, slots=True)
class _RunnerResult:
    runner_id: str
    process_id: int
    nonce_sha256: str
    receipt: Mapping[str, object]
    runner_dir: Path


def _strict_mapping(value: object, name: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage10ToyConfigurationError(f"{name} fields are incomplete or unknown")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00"):
        raise Stage10ToyConfigurationError(f"{name} must be safe non-empty text")
    return value


def _digest(value: object, name: str) -> str:
    text = _text(value, name)
    if _SHA256_RE.fullmatch(text) is None:
        raise Stage10ToyConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise Stage10ToyConfigurationError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage10ToyConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not result == result or result in (float("inf"), float("-inf")):
        raise Stage10ToyConfigurationError(f"{name} must be finite")
    return result


def parse_stage10_toy_config(value: object) -> Stage10ToyConfig:
    data = _strict_mapping(
        value,
        "Stage 10 config",
        {
            "schema_version",
            "run_id",
            "method",
            "scope",
            "dataset_id",
            "dataset_version",
            "split",
            "runner_ids",
            "physical_host_count",
            "cross_physical_host_claim",
            "source_contract",
            "environment_policy",
            "inclusion_case_ids",
            "replay_runs",
            "degradation_replay_set_sha256",
            "locked_contract",
            "artifact_exchange",
            "confidence_level",
            "locked_policy",
        },
    )
    source = _strict_mapping(
        data["source_contract"],
        "source_contract",
        {
            "stage9_package_sha256",
            "stage9_package_file_count",
            "stage9_preregistration_sha256",
            "stage8_package_sha256",
            "stage7_package_sha256",
        },
    )
    environment = _strict_mapping(
        data["environment_policy"],
        "environment_policy",
        {
            "capture_probe_before_runners",
            "python_hash_seed",
            "python_utf8_mode",
            "thread_count",
            "locale",
            "exact_match_fields",
        },
    )
    locked = _strict_mapping(
        data["locked_contract"],
        "locked_contract",
        {
            "alpha",
            "positive_threshold",
            "negative_threshold",
            "positive_sample_count",
            "negative_sample_count",
            "config_hash",
            "data_hash",
            "model_hash",
        },
    )
    exchange = _strict_mapping(
        data["artifact_exchange"],
        "artifact_exchange",
        {
            "envelope_payload_contains_data",
            "envelope_payload_contains_root",
            "integrity_algorithm",
            "authentication_scope",
            "offline_verification_required",
            "detached_envelope_authoritative",
            "real_key_material_embedded",
            "allowed_stage9_provenance_paths",
        },
    )
    policy = _strict_mapping(
        data["locked_policy"],
        "locked_policy",
        {
            "test_tuning_prohibited",
            "threshold_reestimation_prohibited",
            "sample_replacement_prohibited",
            "network_used",
            "gpu_used",
            "real_data_access",
            "deterministic_difference_fail_closed",
            "large_model_training_prohibited",
            "weights_download_prohibited",
        },
    )
    runner_values = data["runner_ids"]
    cases = data["inclusion_case_ids"]
    runs = data["replay_runs"]
    exact_fields = environment["exact_match_fields"]
    allowed_paths = exchange["allowed_stage9_provenance_paths"]
    if not isinstance(runner_values, list) or len(runner_values) != 2:
        raise Stage10ToyConfigurationError("Stage 10 requires exactly two runner IDs")
    if not isinstance(cases, list) or not cases or not isinstance(runs, list) or not runs:
        raise Stage10ToyConfigurationError("case and replay preregistration must be non-empty")
    if not isinstance(exact_fields, list) or not exact_fields:
        raise Stage10ToyConfigurationError("exact environment fields must be non-empty")
    if not isinstance(allowed_paths, list):
        raise Stage10ToyConfigurationError("allowed provenance paths must be a list")
    runners = tuple(_text(item, "runner_id") for item in runner_values)
    if len(set(runners)) != 2:
        raise Stage10ToyConfigurationError("runner IDs must be distinct")
    case_ids = tuple(_text(item, "inclusion_case_id") for item in cases)
    run_ids: list[str] = []
    seeds: list[int] = []
    for item in runs:
        run = _strict_mapping(item, "replay_run", {"run_id", "seed"})
        run_ids.append(_text(run["run_id"], "run_id"))
        seed = run["seed"]
        if type(seed) is not int or not 0 <= seed <= (2**32) - 1:  # noqa: E721
            raise Stage10ToyConfigurationError("replay seed must be a uint32 integer")
        seeds.append(seed)
    exact = tuple(_text(item, "exact environment field") for item in exact_fields)
    allowed = tuple(_text(item, "allowed provenance path") for item in allowed_paths)
    bool_values = {
        "capture_probe_before_runners": environment["capture_probe_before_runners"],
        "python_utf8_mode": environment["python_utf8_mode"],
        **exchange,
        **policy,
    }
    for non_bool in (
        "integrity_algorithm",
        "authentication_scope",
        "allowed_stage9_provenance_paths",
    ):
        bool_values.pop(non_bool)
    if any(type(item) is not bool for item in bool_values.values()):  # noqa: E721
        raise Stage10ToyConfigurationError("Stage 10 policy flags must be bool")
    config = Stage10ToyConfig(
        run_id=_text(data["run_id"], "run_id"),
        runner_ids=cast(tuple[str, str], runners),
        source_stage9_package_sha256=_digest(
            source["stage9_package_sha256"], "stage9_package_sha256"
        ),
        source_stage9_package_file_count=_positive_int(
            source["stage9_package_file_count"], "stage9_package_file_count"
        ),
        stage9_preregistration_sha256=_digest(
            source["stage9_preregistration_sha256"], "stage9_preregistration_sha256"
        ),
        source_stage8_package_sha256=_digest(
            source["stage8_package_sha256"], "stage8_package_sha256"
        ),
        source_stage7_package_sha256=_digest(
            source["stage7_package_sha256"], "stage7_package_sha256"
        ),
        dataset_id=_text(data["dataset_id"], "dataset_id"),
        dataset_version=_text(data["dataset_version"], "dataset_version"),
        inclusion_case_ids=case_ids,
        replay_run_ids=tuple(run_ids),
        replay_seeds=tuple(seeds),
        degradation_replay_set_sha256=_digest(
            data["degradation_replay_set_sha256"], "degradation_replay_set_sha256"
        ),
        alpha=_number(locked["alpha"], "alpha"),
        positive_threshold=_number(locked["positive_threshold"], "positive_threshold"),
        negative_threshold=_number(locked["negative_threshold"], "negative_threshold"),
        positive_sample_count=_positive_int(
            locked["positive_sample_count"], "positive_sample_count"
        ),
        negative_sample_count=_positive_int(
            locked["negative_sample_count"], "negative_sample_count"
        ),
        config_hash=_digest(locked["config_hash"], "config_hash"),
        data_hash=_digest(locked["data_hash"], "data_hash"),
        model_hash=_digest(locked["model_hash"], "model_hash"),
        allowed_stage9_provenance_paths=allowed,
        exact_environment_fields=exact,
        confidence_level=_number(data["confidence_level"], "confidence_level"),
        raw=data,
    )
    required_environment_fields = set(
        parse_stage10_environment_lock(
            {
                "schema_version": "10.0.0",
                "python_implementation": "x",
                "python_version": "x",
                "python_executable_sha256": "0" * 64,
                "pyproject_sha256": "0" * 64,
                "platform_system": "x",
                "platform_release": "x",
                "platform_machine": "x",
                "platform_processor": "x",
                "project_version": "x",
                "dependency_versions": {"x": "x"},
                "blas_configuration_sha256": "0" * 64,
                "thread_policy": {
                    "BLIS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
                "locale_name": "C",
                "preferred_encoding": "utf-8",
                "filesystem_encoding": "utf-8",
                "python_hash_seed": "0",
                "no_user_site": True,
                "dont_write_bytecode": True,
                "utf8_mode": 1,
                "optimize": 0,
            }
        ).to_dict()
    ) - {"schema_version"}
    if (
        data["schema_version"] != STAGE10_TOY_CONFIG_VERSION
        or config.run_id != "stage10-toy-runner-portability-001"
        or data["method"] != STAGE10_METHOD
        or data["scope"] != STAGE10_SCOPE
        or data["split"] != "test"
        or data["physical_host_count"] != 1
        or data["cross_physical_host_claim"] is not False
        or environment["capture_probe_before_runners"] is not True
        or environment["python_hash_seed"] != "0"
        or environment["python_utf8_mode"] is not True
        or environment["thread_count"] != 1
        or environment["locale"] != "C"
        or set(exact) != required_environment_fields
        or len(exact) != len(set(exact))
        or tuple(sorted(allowed))
        != tuple(
            sorted(
                {
                    "audit.html",
                    "audit.json",
                    "package_manifest.json",
                    "session_a.json",
                    "session_b.json",
                }
            )
        )
        or len(case_ids) != len(set(case_ids))
        or len(run_ids) != len(set(run_ids))
        or config.confidence_level != 0.95
        or exchange["envelope_payload_contains_data"] is not False
        or exchange["envelope_payload_contains_root"] is not False
        or exchange["integrity_algorithm"] != STAGE10_SIGNATURE_ALGORITHM
        or exchange["authentication_scope"] != STAGE10_AUTHENTICATION_SCOPE
        or exchange["offline_verification_required"] is not True
        or exchange["detached_envelope_authoritative"] is not False
        or exchange["real_key_material_embedded"] is not False
        or policy
        != {
            "test_tuning_prohibited": True,
            "threshold_reestimation_prohibited": True,
            "sample_replacement_prohibited": True,
            "network_used": False,
            "gpu_used": False,
            "real_data_access": False,
            "deterministic_difference_fail_closed": True,
            "large_model_training_prohibited": True,
            "weights_download_prohibited": True,
        }
    ):
        raise Stage10ToyConfigurationError("Stage 10 fixture safety declarations are invalid")
    return config


def load_stage10_toy_config(path: Path) -> Stage10ToyConfig:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if len(payload) > 64 * 1024:
        raise Stage10ToyConfigurationError("Stage 10 config exceeds its size limit")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Stage10ToyConfigurationError("Stage 10 config is not strict UTF-8 YAML") from error
    return parse_stage10_toy_config(value)


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("JSON object has a non-string or duplicate key")
        result[key] = value
    return result


def _decode_json(payload: bytes, name: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _json_mapping(payload: bytes, name: str) -> dict[str, object]:
    value = _decode_json(payload, name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _read_tree(root: Path, *, label: str) -> dict[str, bytes]:
    package_root = root.resolve(strict=True)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"{label} root must be a regular directory")
    result: dict[str, bytes] = {}
    total = 0
    for path in sorted(package_root.rglob("*")):
        info = path.stat(follow_symlinks=False)
        if path.is_symlink() or bool(
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"{label} contains a link or reparse point")
        if path.is_dir():
            continue
        if not path.is_file() or info.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"{label} contains a non-regular or oversized file")
        relative = path.relative_to(package_root).as_posix()
        if not relative or relative.startswith("/") or "\\" in relative or "\x00" in relative:
            raise ValueError(f"{label} contains an unsafe path")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(
            payload
        ) != info.st_size:
            raise ValueError(f"{label} changed while read")
        total += len(payload)
        if total > _MAX_PACKAGE_BYTES or len(result) >= _MAX_PACKAGE_FILES:
            raise ValueError(f"{label} exceeds package bounds")
        result[relative] = payload
    if not result:
        raise ValueError(f"{label} package is empty")
    return result


def _prefixed(files: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    return {
        name[len(prefix) :]: payload for name, payload in files.items() if name.startswith(prefix)
    }


def _file_manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    files = [
        {"relative_path": name, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        for name, payload in sorted(payloads.items())
    ]
    return {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "digest_method": STAGE10_DIGEST_METHOD,
        "file_count": len(files),
        "files": files,
        "package_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _verify_manifest(manifest: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    payloads = {
        name: payload for name, payload in actual.items() if name != "package_manifest.json"
    }
    if dict(manifest) != _file_manifest(payloads):
        raise ValueError("Stage 10 package manifest does not match complete package bytes")


def _normalize_stage9_session(value: object) -> object:
    if not isinstance(value, dict):
        raise ValueError("Stage 9 session provenance must be an object")
    result = json.loads(json.dumps(value, allow_nan=False))
    for key in ("producer_process_id", "parent_supervisor_process_id", "session_nonce_sha256"):
        if key not in result:
            raise ValueError("Stage 9 session provenance is incomplete")
        result[key] = "<RUNNER_DYNAMIC>"
    package = result.get("stage8_package")
    if not isinstance(package, dict):
        raise ValueError("Stage 9 package authority is missing")
    for key in ("session_id", "producer_process_id", "session_nonce_sha256"):
        if key not in package:
            raise ValueError("Stage 9 package authority is incomplete")
        package[key] = "<RUNNER_DYNAMIC>"
    return result


def _normalize_stage9_audit(value: object) -> object:
    if not isinstance(value, dict):
        raise ValueError("Stage 9 audit must be an object")
    result = json.loads(json.dumps(value, allow_nan=False))
    for key in ("session_a", "session_b"):
        result[key] = _normalize_stage9_session(result.get(key))
    diff = result.get("diff")
    lifecycle = result.get("capability_lifecycle")
    if not isinstance(diff, dict) or not isinstance(lifecycle, dict):
        raise ValueError("Stage 9 audit diff or lifecycle is missing")
    for key in ("session_a", "session_b"):
        session = diff.get(key)
        if not isinstance(session, dict):
            raise ValueError("Stage 9 diff session is missing")
        for field in ("session_id", "producer_process_id", "session_nonce_sha256"):
            if field not in session:
                raise ValueError("Stage 9 diff session is incomplete")
            session[field] = "<RUNNER_DYNAMIC>"
        cap = lifecycle.get(key)
        if not isinstance(cap, dict):
            raise ValueError("Stage 9 capability lifecycle is incomplete")
        for field in ("issued_process_id", "producer_process_id", "session_nonce_sha256"):
            if field not in cap:
                raise ValueError("Stage 9 capability provenance is incomplete")
            cap[field] = "<RUNNER_DYNAMIC>"
    return result


def _expected_preregistration(config: Stage10ToyConfig, actual: Mapping[str, object]) -> None:
    expected = {
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "split": "test",
        "inclusion_case_ids": list(config.inclusion_case_ids),
        "replay_run_ids": list(config.replay_run_ids),
        "replay_seeds": list(config.replay_seeds),
        "degradation_replay_set_sha256": config.degradation_replay_set_sha256,
        "alpha": config.alpha,
        "positive_threshold": config.positive_threshold,
        "negative_threshold": config.negative_threshold,
        "positive_sample_count": config.positive_sample_count,
        "negative_sample_count": config.negative_sample_count,
        "config_hash": config.config_hash,
        "data_hash": config.data_hash,
        "model_hash": config.model_hash,
        "source_stage7_package_sha256": config.source_stage7_package_sha256,
        "source_stage8_package_sha256": config.source_stage8_package_sha256,
        "confidence_level": config.confidence_level,
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage 9 preregistration differs from the frozen Stage 10 contract")
    if canonical_sha256(actual) != config.stage9_preregistration_sha256:
        raise ValueError("Stage 9 preregistration digest differs from Stage 10 source contract")


def _stage9_material(package_root: Path, config: Stage10ToyConfig) -> dict[str, object]:
    verified = verify_stage9_toy_package(package_root)
    files = _read_tree(package_root, label="Stage 9 runner artifact")
    audit = _json_mapping(files["audit.json"], "Stage 9 audit")
    preregistration = audit.get("preregistration")
    if not isinstance(preregistration, dict):
        raise ValueError("Stage 9 preregistration is missing")
    _expected_preregistration(config, preregistration)
    observations = parse_stage8_observations(
        _decode_json(files["session_a/stage8/observations.json"], "Stage 9 embedded observations")
    )
    expected_keys = tuple(
        (case_id, run_id)
        for run_id in ("stage7_locked_baseline", *config.replay_run_ids)
        for case_id in config.inclusion_case_ids
    )
    if tuple(item.key for item in observations) != expected_keys:
        raise ValueError("Stage 10 observation order differs from preregistration")
    deterministic_files = {
        name: sha256_bytes(payload)
        for name, payload in files.items()
        if name not in config.allowed_stage9_provenance_paths
    }
    projection = {
        "normalized_audit": _normalize_stage9_audit(audit),
        "normalized_session_a": _normalize_stage9_session(
            _decode_json(files["session_a.json"], "Stage 9 session A")
        ),
        "normalized_session_b": _normalize_stage9_session(
            _decode_json(files["session_b.json"], "Stage 9 session B")
        ),
        "deterministic_file_sha256": deterministic_files,
        "observations_sha256": canonical_sha256([item.canonical_sha256 for item in observations]),
    }
    manifest = _json_mapping(files["package_manifest.json"], "Stage 9 manifest")
    return {
        "verified": verified,
        "files": files,
        "audit": audit,
        "observations": observations,
        "package_sha256": verified.package_sha256,
        "package_file_count": len(files),
        "package_manifest_sha256": sha256_bytes(files["package_manifest.json"]),
        "audit_sha256": sha256_bytes(files["audit.json"]),
        "preregistration_sha256": canonical_sha256(preregistration),
        "source_stage7_package_sha256": verified.source_stage7_package_sha256,
        "source_stage8_package_sha256": verified.stage8_package_sha256,
        "deterministic_projection_sha256": canonical_sha256(projection),
        "manifest_declared_package_sha256": manifest.get("package_sha256"),
    }


def _fixture_key(config: Stage10ToyConfig, runner_id: str) -> bytes:
    return hashlib.sha256(
        _FIXTURE_KEY_NAMESPACE
        + b":"
        + config.source_stage9_package_sha256.encode("ascii")
        + b":"
        + runner_id.encode("utf-8")
    ).digest()


def create_synthetic_signed_envelope(
    *,
    config: Stage10ToyConfig,
    environment_lock: Stage10EnvironmentLock,
    runner_id: str,
    producer_process_id: int,
    session_nonce_sha256: str,
    stage9_package: Path,
) -> dict[str, object]:
    """Create a fixture-only HMAC envelope after complete Stage 9 verification."""

    if runner_id not in config.runner_ids:
        raise ValueError("runner ID is outside Stage 10 preregistration")
    material = _stage9_material(stage9_package, config)
    observations = cast(tuple[Stage9Observation, ...], material["observations"])
    payload = {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "runner_id": runner_id,
        "producer_process_id": producer_process_id,
        "session_nonce_sha256": _digest(session_nonce_sha256, "session_nonce_sha256"),
        "environment_lock_sha256": environment_lock.sha256(),
        "host_profile_sha256": environment_lock.host_profile_sha256(),
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "source_stage9_package_sha256": config.source_stage9_package_sha256,
        "stage9_package_sha256": material["package_sha256"],
        "stage9_package_file_count": material["package_file_count"],
        "stage9_manifest_sha256": material["package_manifest_sha256"],
        "stage9_audit_sha256": material["audit_sha256"],
        "stage9_preregistration_sha256": material["preregistration_sha256"],
        "source_stage7_package_sha256": material["source_stage7_package_sha256"],
        "source_stage8_package_sha256": material["source_stage8_package_sha256"],
        "deterministic_projection_sha256": material["deterministic_projection_sha256"],
        "observation_set_sha256": canonical_sha256(
            [item.canonical_sha256 for item in observations]
        ),
        "payload_contains_data": False,
        "payload_contains_root": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "detached_envelope_authoritative": False,
    }
    key = _fixture_key(config, runner_id)
    signature = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    return {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "integrity_algorithm": STAGE10_SIGNATURE_ALGORITHM,
        "authentication_scope": STAGE10_AUTHENTICATION_SCOPE,
        "key_id": hashlib.sha256(key).hexdigest(),
        "fixture_key_derivation_public": True,
        "production_authentication": False,
        "real_key_material_embedded": False,
        "payload": payload,
        "integrity_tag": signature,
    }


def _verify_envelope(
    envelope: object,
    *,
    config: Stage10ToyConfig,
    runner_id: str,
    expected_payload: Mapping[str, object],
) -> str:
    fields = {
        "schema_version",
        "integrity_algorithm",
        "authentication_scope",
        "key_id",
        "fixture_key_derivation_public",
        "production_authentication",
        "real_key_material_embedded",
        "payload",
        "integrity_tag",
    }
    if not isinstance(envelope, dict) or set(envelope) != fields:
        raise ValueError("Stage 10 envelope fields are incomplete or unknown")
    key = _fixture_key(config, runner_id)
    payload = envelope["payload"]
    if (
        envelope["schema_version"] != STAGE10_SCHEMA_VERSION
        or envelope["integrity_algorithm"] != STAGE10_SIGNATURE_ALGORITHM
        or envelope["authentication_scope"] != STAGE10_AUTHENTICATION_SCOPE
        or envelope["key_id"] != hashlib.sha256(key).hexdigest()
        or envelope["fixture_key_derivation_public"] is not True
        or envelope["production_authentication"] is not False
        or envelope["real_key_material_embedded"] is not False
        or not isinstance(payload, dict)
        or payload != dict(expected_payload)
    ):
        raise ValueError("Stage 10 synthetic envelope contract failed")
    expected = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    signature = envelope["integrity_tag"]
    if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
        raise ValueError("Stage 10 envelope HMAC verification failed")
    return canonical_sha256(envelope)


def _snapshot_from_runner(
    runner: _RunnerResult,
    *,
    config: Stage10ToyConfig,
    environment_lock: Stage10EnvironmentLock,
) -> Stage10RunnerSnapshot:
    material = _stage9_material(runner.runner_dir / "stage9", config)
    observations = cast(tuple[Stage9Observation, ...], material["observations"])
    expected_payload = {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "runner_id": runner.runner_id,
        "producer_process_id": runner.process_id,
        "session_nonce_sha256": runner.nonce_sha256,
        "environment_lock_sha256": environment_lock.sha256(),
        "host_profile_sha256": environment_lock.host_profile_sha256(),
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "source_stage9_package_sha256": config.source_stage9_package_sha256,
        "stage9_package_sha256": material["package_sha256"],
        "stage9_package_file_count": material["package_file_count"],
        "stage9_manifest_sha256": material["package_manifest_sha256"],
        "stage9_audit_sha256": material["audit_sha256"],
        "stage9_preregistration_sha256": material["preregistration_sha256"],
        "source_stage7_package_sha256": material["source_stage7_package_sha256"],
        "source_stage8_package_sha256": material["source_stage8_package_sha256"],
        "deterministic_projection_sha256": material["deterministic_projection_sha256"],
        "observation_set_sha256": canonical_sha256(
            [item.canonical_sha256 for item in observations]
        ),
        "payload_contains_data": False,
        "payload_contains_root": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "detached_envelope_authoritative": False,
    }
    envelope_path = runner.runner_dir / "envelope.json"
    envelope_bytes = envelope_path.read_bytes()
    envelope = _decode_json(envelope_bytes, f"{runner.runner_id} envelope")
    envelope_sha = _verify_envelope(
        envelope,
        config=config,
        runner_id=runner.runner_id,
        expected_payload=expected_payload,
    )
    return Stage10RunnerSnapshot(
        runner_id=runner.runner_id,
        producer_process_id=runner.process_id,
        session_nonce_sha256=runner.nonce_sha256,
        environment_lock_sha256=environment_lock.sha256(),
        host_profile_sha256=environment_lock.host_profile_sha256(),
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        scope="SYNTHETIC_ONLY",
        split="test",
        source_stage9_package_sha256=config.source_stage9_package_sha256,
        stage9_package_sha256=cast(str, material["package_sha256"]),
        stage9_package_file_count=cast(int, material["package_file_count"]),
        stage9_manifest_sha256=cast(str, material["package_manifest_sha256"]),
        stage9_audit_sha256=cast(str, material["audit_sha256"]),
        stage9_preregistration_sha256=cast(str, material["preregistration_sha256"]),
        source_stage7_package_sha256=cast(str, material["source_stage7_package_sha256"]),
        source_stage8_package_sha256=cast(str, material["source_stage8_package_sha256"]),
        deterministic_projection_sha256=cast(str, material["deterministic_projection_sha256"]),
        envelope_sha256=envelope_sha,
        signature_verified=True,
        observations=observations,
    )


def _artifact_diff(
    files_a: Mapping[str, bytes],
    files_b: Mapping[str, bytes],
    *,
    allowed_paths: tuple[str, ...],
) -> dict[str, object]:
    names = sorted(set(files_a) | set(files_b))
    rows: list[dict[str, object]] = []
    exact = 0
    allowed = 0
    mismatches = 0
    for name in names:
        left = files_a.get(name)
        right = files_b.get(name)
        if left is not None and right is not None and left == right:
            status_value = "EXACT_MATCH"
            exact += 1
        elif left is not None and right is not None and name in allowed_paths:
            status_value = "ALLOWED_RUNNER_PROVENANCE"
            allowed += 1
        elif left is None:
            status_value = "MISSING_RUNNER_A"
            mismatches += 1
        elif right is None:
            status_value = "MISSING_RUNNER_B"
            mismatches += 1
        else:
            status_value = "DETERMINISTIC_MISMATCH"
            mismatches += 1
        rows.append(
            {
                "relative_path": name,
                "classification": status_value,
                "runner_a_sha256": None if left is None else sha256_bytes(left),
                "runner_b_sha256": None if right is None else sha256_bytes(right),
                "runner_a_size_bytes": None if left is None else len(left),
                "runner_b_size_bytes": None if right is None else len(right),
            }
        )
    return {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "comparison_policy": "exact_except_preregistered_runner_provenance_paths_v1",
        "allowed_provenance_paths": list(allowed_paths),
        "compared_file_count": len(names),
        "exact_file_match_count": exact,
        "allowed_provenance_difference_count": allowed,
        "deterministic_file_mismatch_count": mismatches,
        "deterministic_difference_fail_closed": True,
        "files": rows,
    }


def _minimal_environment(nonce: str) -> dict[str, str]:
    names = (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "TEMP",
        "TMP",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
    )
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "LC_ALL": "C",
            "LANG": "C",
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "RETINAENTAIL_STAGE10_RUNNER_NONCE": nonce,
        }
    )
    return environment


def _run_probe(target: Path, project_root: Path) -> Stage10EnvironmentLock:
    nonce = secrets.token_hex(32)
    command = (
        sys.executable,
        "-m",
        "retinaentail.pilot.stage10_worker",
        "probe",
        "--output",
        str(target),
        "--expected-parent-pid",
        str(os.getpid()),
        "--nonce",
        nonce,
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=_minimal_environment(nonce),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        shell=False,
        timeout=60,
        check=False,
        creationflags=creationflags,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Stage 10 environment probe failed: {completed.stderr[-2000:]}")
    lines = tuple(line for line in completed.stdout.splitlines() if line.strip())
    if len(lines) != 1:
        raise RuntimeError("Stage 10 environment probe emitted unexpected output")
    receipt = _json_mapping(lines[0].encode(), "Stage 10 probe receipt")
    lock = parse_stage10_environment_lock(_decode_json(target.read_bytes(), "environment lock"))
    if (
        type(receipt.get("probe_process_id")) is not int  # noqa: E721
        or cast(int, receipt.get("probe_process_id")) <= 0
        or receipt.get("parent_process_id") != os.getpid()
        or receipt.get("nonce_sha256") != hashlib.sha256(nonce.encode()).hexdigest()
        or receipt.get("environment_lock_sha256") != lock.sha256()
        or receipt.get("dataset_root_probed") is not False
        or receipt.get("real_data_read") is not False
    ):
        raise RuntimeError("Stage 10 environment probe receipt failed closed")
    return lock


def _launch_runners(
    *,
    config_path: Path,
    source_stage9_package: Path,
    output_dir: Path,
    environment_lock_path: Path,
    project_root: Path,
    runner_ids: tuple[str, str],
) -> tuple[_RunnerResult, _RunnerResult]:
    processes: list[tuple[str, str, Path, subprocess.Popen[str]]] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    for runner_id in runner_ids:
        nonce = secrets.token_hex(32)
        runner_dir = output_dir / runner_id
        command = (
            sys.executable,
            "-m",
            "retinaentail.pilot.stage10_worker",
            "run",
            "--config",
            str(config_path),
            "--source-stage9-package",
            str(source_stage9_package),
            "--expected-environment-lock",
            str(environment_lock_path),
            "--output",
            str(runner_dir),
            "--runner-id",
            runner_id,
            "--runner-nonce",
            nonce,
            "--expected-parent-pid",
            str(os.getpid()),
        )
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=_minimal_environment(nonce),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            shell=False,
            creationflags=creationflags,
        )
        processes.append((runner_id, nonce, runner_dir, process))
    results: list[_RunnerResult] = []
    try:
        for runner_id, nonce, runner_dir, process in processes:
            stdout, stderr = process.communicate(timeout=_WORKER_TIMEOUT_SECONDS)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Stage 10 {runner_id} failed with code {process.returncode}: {stderr[-4000:]}"
                )
            lines = tuple(line for line in stdout.splitlines() if line.strip())
            if len(lines) != 1:
                raise RuntimeError(f"Stage 10 {runner_id} emitted unexpected output")
            receipt = _json_mapping(lines[0].encode(), f"{runner_id} receipt")
            nonce_sha = hashlib.sha256(nonce.encode()).hexdigest()
            if (
                receipt.get("runner_id") != runner_id
                or receipt.get("producer_process_id") != process.pid
                or receipt.get("parent_process_id") != os.getpid()
                or receipt.get("session_nonce_sha256") != nonce_sha
                or receipt.get("stage9_verified") is not True
                or receipt.get("stage9_status") != "SUCCESS"
                or receipt.get("signed_envelope_created") is not True
                or receipt.get("network_used") is not False
                or receipt.get("gpu_used") is not False
                or receipt.get("real_data_read") is not False
                or receipt.get("dataset_root_probed") is not False
                or receipt.get("detached_receipt_authoritative") is not False
            ):
                raise RuntimeError(f"Stage 10 {runner_id} receipt failed closed")
            results.append(
                _RunnerResult(
                    runner_id=runner_id,
                    process_id=process.pid,
                    nonce_sha256=nonce_sha,
                    receipt=receipt,
                    runner_dir=runner_dir,
                )
            )
    except BaseException:
        for _, _, _, process in processes:
            if process.poll() is None:
                process.kill()
        raise
    if len(results) != 2 or results[0].process_id == results[1].process_id:
        raise RuntimeError("Stage 10 local runner processes were not independent")
    return (results[0], results[1])


def _paths_nonoverlapping(values: Sequence[Path]) -> None:
    resolved = tuple(item.resolve(strict=False) for item in values)
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Stage 10 config, source, and output paths must not overlap")


def _summary(
    config: Stage10ToyConfig,
    diff: Stage10DiffReport,
    artifact_diff: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "run_id": config.run_id,
        "method": STAGE10_METHOD,
        "scope": STAGE10_SCOPE,
        "status": diff.status.value,
        "physical_host_count": 1,
        "local_runner_process_count": 2,
        "cross_physical_host_claim": False,
        "cross_physical_host_portability_tested": False,
        "case_count": len(config.inclusion_case_ids),
        "observation_comparison_count": len(diff.case_comparisons),
        "exact_file_match_count": artifact_diff["exact_file_match_count"],
        "allowed_provenance_difference_count": artifact_diff["allowed_provenance_difference_count"],
        "deterministic_file_mismatch_count": artifact_diff["deterministic_file_mismatch_count"],
        "signed_envelopes_verified": 2,
        "signature_authentication_scope": STAGE10_AUTHENTICATION_SCOPE,
        "production_authentication": False,
        "thresholds_reestimated": False,
        "test_split_used_for_tuning": False,
        "sample_replacement_performed": False,
        "network_used": False,
        "gpu_used": False,
        "large_model_trained": False,
        "weights_downloaded": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "clinical_claim": False,
        "synthetic_authorizes_real_data": False,
    }


def _audit(
    *,
    config: Stage10ToyConfig,
    environment_lock: Stage10EnvironmentLock,
    source_authority: Mapping[str, object],
    diff: Stage10DiffReport,
    artifact_diff: Mapping[str, object],
    lifecycle: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "run_id": config.run_id,
        "method": STAGE10_METHOD,
        "scope": STAGE10_SCOPE,
        "status": diff.status.value,
        "authority_boundary": STAGE10_AUTHORITY_BOUNDARY,
        "physical_host_count": 1,
        "local_runner_process_count": 2,
        "cross_physical_host_claim": False,
        "cross_physical_host_portability_tested": False,
        "clinical_claim": False,
        "environment_lock": environment_lock.to_dict(),
        "environment_lock_sha256": environment_lock.sha256(),
        "exact_environment_fields": list(config.exact_environment_fields),
        "source_authority": dict(source_authority),
        "capability_lifecycle": dict(lifecycle),
        "signature_policy": {
            "integrity_algorithm": STAGE10_SIGNATURE_ALGORITHM,
            "authentication_scope": STAGE10_AUTHENTICATION_SCOPE,
            "fixture_key_derivation_public": True,
            "production_authentication": False,
            "real_key_material_embedded": False,
            "payload_contains_data": False,
            "payload_contains_root": False,
            "offline_verification_completed": True,
        },
        "diff": diff.to_dict(),
        "artifact_diff_summary": {
            name: artifact_diff[name]
            for name in (
                "compared_file_count",
                "exact_file_match_count",
                "allowed_provenance_difference_count",
                "deterministic_file_mismatch_count",
            )
        },
        "test_tuning_performed": False,
        "thresholds_reestimated": False,
        "sample_replacement_performed": False,
        "network_used": False,
        "gpu_used": False,
        "large_model_trained": False,
        "weights_downloaded": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "detached_pass_json_authoritative": False,
        "synthetic_authorizes_real_data": False,
    }


def run_stage10_toy_fixture(
    config_path: Path,
    source_stage9_package: Path,
    output_dir: Path,
) -> Stage10ToyRunResult:
    """Run two local runner processes and verify their signed artifact exchange."""

    if any(not isinstance(item, Path) for item in (config_path, source_stage9_package, output_dir)):
        raise TypeError("all Stage 10 paths must be pathlib.Path values")
    config_root = config_path.resolve(strict=True)
    source_root = source_stage9_package.resolve(strict=True)
    target = output_dir.resolve(strict=False)
    _paths_nonoverlapping((config_root, source_root, target))
    if target.exists():
        raise FileExistsError(f"refusing to reuse a Stage 10 output directory: {target}")
    config = load_stage10_toy_config(config_root)
    source_verified = verify_stage9_toy_package(source_root)
    if (
        source_verified.package_sha256 != config.source_stage9_package_sha256
        or source_verified.package_file_count != config.source_stage9_package_file_count
        or source_verified.stage8_package_sha256 != config.source_stage8_package_sha256
        or source_verified.source_stage7_package_sha256 != config.source_stage7_package_sha256
        or source_verified.status != "SUCCESS"
    ):
        raise ValueError("source Stage 9 package differs from the frozen Stage 10 contract")
    project_root = Path(__file__).resolve().parents[3]
    environment_path = target / "environment_lock.json"
    environment_lock = _run_probe(environment_path, project_root)
    runner_a, runner_b = _launch_runners(
        config_path=config_root,
        source_stage9_package=source_root,
        output_dir=target,
        environment_lock_path=environment_path,
        project_root=project_root,
        runner_ids=config.runner_ids,
    )
    files_a = _read_tree(runner_a.runner_dir / "stage9", label="runner A Stage 9")
    files_b = _read_tree(runner_b.runner_dir / "stage9", label="runner B Stage 9")
    artifact_diff = _artifact_diff(
        files_a,
        files_b,
        allowed_paths=config.allowed_stage9_provenance_paths,
    )
    replay_counts = {runner_a.runner_id: 0, runner_b.runner_id: 0}

    def replay_a() -> Stage10RunnerSnapshot:
        replay_counts[runner_a.runner_id] += 1
        return _snapshot_from_runner(runner_a, config=config, environment_lock=environment_lock)

    def replay_b() -> Stage10RunnerSnapshot:
        replay_counts[runner_b.runner_id] += 1
        return _snapshot_from_runner(runner_b, config=config, environment_lock=environment_lock)

    capability_a = issue_synthetic_stage9_runner_capability(replay_a, ttl_seconds=60)
    capability_b = issue_synthetic_stage9_runner_capability(replay_b, ttl_seconds=60)
    diff = consume_stage10_runner_pair(
        capability_a,
        capability_b,
        lambda left, right: compare_stage10_runners(
            left,
            right,
            deterministic_file_mismatch_count=cast(
                int, artifact_diff["deterministic_file_mismatch_count"]
            ),
            allowed_provenance_difference_count=cast(
                int, artifact_diff["allowed_provenance_difference_count"]
            ),
            confidence_level=config.confidence_level,
        ),
    )
    reuse_rejected = False
    try:
        consume_stage10_runner_pair(
            capability_a,
            capability_b,
            lambda left, right: (left, right),
        )
    except Stage10CapabilityError:
        reuse_rejected = True
    if not reuse_rejected:
        raise AssertionError("Stage 10 runner capabilities became reusable")
    if diff.status is not Stage10DiffStatus.SUCCESS:
        raise ValueError("synthetic Stage 10 deterministic comparison failed closed")
    source_authority = {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "source_stage9_package_sha256": source_verified.package_sha256,
        "source_stage9_package_file_count": source_verified.package_file_count,
        "source_stage8_package_sha256": source_verified.stage8_package_sha256,
        "source_stage7_package_sha256": source_verified.source_stage7_package_sha256,
        "complete_source_stage9_replay_verified": True,
        "detached_source_summary_authoritative": False,
        "authorizes_real_data": False,
    }
    lifecycle = {
        "schema_version": STAGE10_SCHEMA_VERSION,
        "issued_count": 2,
        "consumed_count": 2,
        "second_pair_consumption_rejected": reuse_rejected,
        "supervisor_complete_stage9_replay_count": sum(replay_counts.values()),
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "all_capabilities_fresh": True,
        "cross_runner_reuse_allowed": False,
        "detached_receipt_authoritative": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "runner_a": runner_capability_snapshot(capability_a),
        "runner_b": runner_capability_snapshot(capability_b),
    }
    audit = _audit(
        config=config,
        environment_lock=environment_lock,
        source_authority=source_authority,
        diff=diff,
        artifact_diff=artifact_diff,
        lifecycle=lifecycle,
    )
    summary = _summary(config, diff, artifact_diff)
    root_payloads = {
        "artifact_diff.json": _pretty_json_bytes(artifact_diff),
        "audit.html": render_stage10_audit_html(audit).encode("utf-8"),
        "audit.json": render_stage10_audit_json(audit).encode("utf-8"),
        "config_snapshot.json": _pretty_json_bytes(dict(config.raw)),
        "environment_lock.json": _pretty_json_bytes(environment_lock.to_dict()),
        "source_authority.json": _pretty_json_bytes(source_authority),
        "summary.json": _pretty_json_bytes(summary),
    }
    for name, payload in root_payloads.items():
        write_bytes_fail_safe(target / name, payload)
    final_tree = _read_tree(target, label="Stage 10 generated package")
    manifest = _file_manifest(
        {name: payload for name, payload in final_tree.items() if name != "package_manifest.json"}
    )
    write_bytes_fail_safe(target / "package_manifest.json", _pretty_json_bytes(manifest))
    actual = _read_tree(target, label="Stage 10 generated package")
    _verify_manifest(manifest, actual)
    return Stage10ToyRunResult(
        output_dir=output_dir,
        package_sha256=cast(str, manifest["package_sha256"]),
        package_file_count=len(actual),
        runner_process_ids=(runner_a.process_id, runner_b.process_id),
        runner_stage9_package_sha256=(
            diff.runner_a.stage9_package_sha256,
            diff.runner_b.stage9_package_sha256,
        ),
        source_stage9_package_sha256=config.source_stage9_package_sha256,
        case_count=len(config.inclusion_case_ids),
        observation_comparison_count=len(diff.case_comparisons),
        exact_file_match_count=cast(int, artifact_diff["exact_file_match_count"]),
        allowed_provenance_difference_count=cast(
            int, artifact_diff["allowed_provenance_difference_count"]
        ),
        deterministic_file_mismatch_count=cast(
            int, artifact_diff["deterministic_file_mismatch_count"]
        ),
        status=diff.status.value,
        verified=True,
    )


def verify_stage10_toy_package(output_dir: Path) -> Stage10ToyRunResult:
    """Offline verifier: replays both embedded Stage 9 packages and both HMAC envelopes."""

    if not isinstance(output_dir, Path):
        raise TypeError("Stage 10 package root must be pathlib.Path")
    root = output_dir.resolve(strict=True)
    actual = _read_tree(root, label="Stage 10")
    manifest = _json_mapping(actual.get("package_manifest.json", b""), "Stage 10 manifest")
    _verify_manifest(manifest, actual)
    files_a = _prefixed(actual, "runner_a/stage9/")
    files_b = _prefixed(actual, "runner_b/stage9/")
    expected_paths = {
        "package_manifest.json",
        *_ROOT_FILES,
        "runner_a/envelope.json",
        "runner_b/envelope.json",
        *(f"runner_a/stage9/{name}" for name in files_a),
        *(f"runner_b/stage9/{name}" for name in files_b),
    }
    if set(actual) != expected_paths:
        raise ValueError("Stage 10 package contains an unknown path")
    config = parse_stage10_toy_config(
        _decode_json(actual["config_snapshot.json"], "Stage 10 config")
    )
    environment_lock = parse_stage10_environment_lock(
        _decode_json(actual["environment_lock.json"], "Stage 10 environment lock")
    )
    runner_results: list[_RunnerResult] = []
    audit = _json_mapping(actual["audit.json"], "Stage 10 audit")
    diff_payload = audit.get("diff")
    if not isinstance(diff_payload, dict):
        raise ValueError("Stage 10 audit diff is missing")
    for label, runner_id in (
        ("runner_a", config.runner_ids[0]),
        ("runner_b", config.runner_ids[1]),
    ):
        authority = diff_payload.get(label)
        if not isinstance(authority, dict):
            raise ValueError("Stage 10 runner authority is missing")
        runner_results.append(
            _RunnerResult(
                runner_id=runner_id,
                process_id=_positive_int(
                    authority.get("producer_process_id"), "producer_process_id"
                ),
                nonce_sha256=_digest(authority.get("session_nonce_sha256"), "session_nonce_sha256"),
                receipt={},
                runner_dir=root / label,
            )
        )
    snapshots = tuple(
        _snapshot_from_runner(item, config=config, environment_lock=environment_lock)
        for item in runner_results
    )
    artifact_diff = _artifact_diff(
        files_a,
        files_b,
        allowed_paths=config.allowed_stage9_provenance_paths,
    )
    if actual["artifact_diff.json"] != _pretty_json_bytes(artifact_diff):
        raise ValueError("Stage 10 artifact diff does not regenerate")
    diff = compare_stage10_runners(
        snapshots[0],
        snapshots[1],
        deterministic_file_mismatch_count=cast(
            int, artifact_diff["deterministic_file_mismatch_count"]
        ),
        allowed_provenance_difference_count=cast(
            int, artifact_diff["allowed_provenance_difference_count"]
        ),
        confidence_level=config.confidence_level,
    )
    if audit.get("diff") != diff.to_dict() or audit.get("status") != diff.status.value:
        raise ValueError("Stage 10 audit differs from recomputed exact diff")
    source_authority = _json_mapping(actual["source_authority.json"], "source authority")
    if audit.get("source_authority") != source_authority:
        raise ValueError("Stage 10 source authority differs from audit")
    lifecycle = audit.get("capability_lifecycle")
    if not isinstance(lifecycle, dict) or any(
        lifecycle.get(key) != value
        for key, value in {
            "issued_count": 2,
            "consumed_count": 2,
            "second_pair_consumption_rejected": True,
            "process_bound": True,
            "one_shot": True,
            "non_serializable": True,
            "all_capabilities_fresh": True,
            "cross_runner_reuse_allowed": False,
            "detached_receipt_authoritative": False,
            "authorizes_real_data": False,
            "authorizes_physical_root": False,
        }.items()
    ):
        raise ValueError("Stage 10 capability lifecycle is invalid")
    if actual["audit.html"] != render_stage10_audit_html(audit).encode("utf-8"):
        raise ValueError("Stage 10 HTML audit does not regenerate")
    summary = _summary(config, diff, artifact_diff)
    if actual["summary.json"] != _pretty_json_bytes(summary):
        raise ValueError("Stage 10 summary does not regenerate")
    package_sha = manifest.get("package_sha256")
    if not isinstance(package_sha, str):
        raise ValueError("Stage 10 package SHA-256 is missing")
    return Stage10ToyRunResult(
        output_dir=output_dir,
        package_sha256=package_sha,
        package_file_count=len(actual),
        runner_process_ids=(snapshots[0].producer_process_id, snapshots[1].producer_process_id),
        runner_stage9_package_sha256=(
            snapshots[0].stage9_package_sha256,
            snapshots[1].stage9_package_sha256,
        ),
        source_stage9_package_sha256=config.source_stage9_package_sha256,
        case_count=len(config.inclusion_case_ids),
        observation_comparison_count=len(diff.case_comparisons),
        exact_file_match_count=cast(int, artifact_diff["exact_file_match_count"]),
        allowed_provenance_difference_count=cast(
            int, artifact_diff["allowed_provenance_difference_count"]
        ),
        deterministic_file_mismatch_count=cast(
            int, artifact_diff["deterministic_file_mismatch_count"]
        ),
        status=diff.status.value,
        verified=True,
    )
