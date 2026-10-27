"""Stage 9 two-process synthetic fixture and complete-package verifier."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

import yaml

from retinaentail.pilot.stage8_fixture import verify_stage8_toy_package
from retinaentail.pilot.stage9_gate import (
    Stage9CapabilityError,
    consume_stage9_session_pair,
    issue_synthetic_stage8_session_capability,
    session_capability_snapshot,
)
from retinaentail.reporting.stage9 import render_stage9_audit_html, render_stage9_audit_json
from retinaentail.reproducibility.stage9 import (
    Stage9DiffReport,
    Stage9DiffStatus,
    Stage9EnvironmentLock,
    Stage9SessionSnapshot,
    capture_stage9_environment_lock,
    compare_stage8_sessions,
    environment_lock_json_bytes,
    parse_stage8_observations,
    parse_stage9_environment_lock,
)
from retinaentail.witnesses.replay import sha256_bytes, write_bytes_fail_safe

STAGE9_TOY_CONFIG_VERSION: Final = "9.0"
STAGE9_PACKAGE_VERSION: Final = "9.0.0"
STAGE9_METHOD: Final = "independent_process_stage8_package_diff_v1"
STAGE9_SCOPE: Final = "SYNTHETIC_FIXTURE"
STAGE9_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE9_SYNTHETIC_PACKAGE"
STAGE9_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"

_MAX_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES: Final = 256 * 1024 * 1024
_MAX_PACKAGE_FILES: Final = 768
_WORKER_TIMEOUT_SECONDS: Final = 300
_ROOT_FILES: Final = frozenset(
    {
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "environment_lock.json",
        "package_diff.json",
        "session_a.json",
        "session_b.json",
        "summary.json",
    }
)


class Stage9ToyConfigurationError(ValueError):
    """Raised when the synthetic Stage 9 preregistration drifts."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9ToyConfig:
    run_id: str
    source_stage8_package_sha256: str
    source_stage8_package_file_count: int
    source_stage7_package_sha256: str
    stage8_preregistration_sha256: str
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
    confidence_level: float
    raw: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Stage9ToyRunResult:
    """Detached verification receipt; it never grants dataset or process authority."""

    output_dir: Path
    package_sha256: str
    package_file_count: int
    stage8_package_sha256: str
    source_stage7_package_sha256: str
    producer_process_ids: tuple[int, int]
    case_count: int
    observation_comparison_count: int
    matching_stage8_file_count: int
    differing_stage8_file_count: int
    status: str
    verified: bool


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    session_id: str
    process_id: int
    nonce_sha256: str
    receipt: Mapping[str, object]
    package_dir: Path


def _strict_mapping(value: object, name: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage9ToyConfigurationError(f"{name} fields are incomplete or unknown")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\x00"):
        raise Stage9ToyConfigurationError(f"{name} must be safe non-empty text")
    return value


def _digest(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Stage9ToyConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise Stage9ToyConfigurationError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage9ToyConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not result == result or result in (float("inf"), float("-inf")):
        raise Stage9ToyConfigurationError(f"{name} must be finite")
    return result


def parse_stage9_toy_config(value: object) -> Stage9ToyConfig:
    data = _strict_mapping(
        value,
        "Stage 9 config",
        {
            "schema_version",
            "run_id",
            "method",
            "scope",
            "source_contract",
            "dataset_id",
            "dataset_version",
            "split",
            "session_count",
            "environment_lock",
            "inclusion_case_ids",
            "replay_runs",
            "degradation_replay_set_sha256",
            "locked_contract",
            "confidence_level",
            "locked_policy",
        },
    )
    source = _strict_mapping(
        data["source_contract"],
        "source_contract",
        {
            "stage8_package_sha256",
            "stage8_package_file_count",
            "stage7_package_sha256",
            "stage8_preregistration_sha256",
        },
    )
    environment = _strict_mapping(
        data["environment_lock"],
        "environment_lock",
        {"capture_before_sessions", "python_hash_seed"},
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
            "detached_receipt_authoritative",
        },
    )
    cases = data["inclusion_case_ids"]
    runs = data["replay_runs"]
    if not isinstance(cases, list) or not cases or not isinstance(runs, list) or not runs:
        raise Stage9ToyConfigurationError("case and replay preregistration must be non-empty")
    case_ids = tuple(_text(item, "inclusion_case_id") for item in cases)
    run_ids: list[str] = []
    seeds: list[int] = []
    for item in runs:
        run = _strict_mapping(item, "replay_run", {"run_id", "seed"})
        run_ids.append(_text(run["run_id"], "run_id"))
        seed = run["seed"]
        if type(seed) is not int or not 0 <= seed <= (2**32) - 1:  # noqa: E721
            raise Stage9ToyConfigurationError("replay seed must be a uint32 integer")
        seeds.append(seed)
    boolean_expectations = {
        **policy,
        "capture_before_sessions": environment["capture_before_sessions"],
    }
    if any(type(item) is not bool for item in boolean_expectations.values()):  # noqa: E721
        raise Stage9ToyConfigurationError("Stage 9 policy flags must be bool")
    config = Stage9ToyConfig(
        run_id=_text(data["run_id"], "run_id"),
        source_stage8_package_sha256=_digest(
            source["stage8_package_sha256"], "stage8_package_sha256"
        ),
        source_stage8_package_file_count=_positive_int(
            source["stage8_package_file_count"], "stage8_package_file_count"
        ),
        source_stage7_package_sha256=_digest(
            source["stage7_package_sha256"], "stage7_package_sha256"
        ),
        stage8_preregistration_sha256=_digest(
            source["stage8_preregistration_sha256"], "stage8_preregistration_sha256"
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
        confidence_level=_number(data["confidence_level"], "confidence_level"),
        raw=data,
    )
    if (
        data["schema_version"] != STAGE9_TOY_CONFIG_VERSION
        or config.run_id != "stage9-toy-independent-session-diff-001"
        or data["method"] != STAGE9_METHOD
        or data["scope"] != STAGE9_SCOPE
        or data["split"] != "test"
        or data["session_count"] != 2
        or environment != {"capture_before_sessions": True, "python_hash_seed": "0"}
        or len(case_ids) != len(set(case_ids))
        or len(run_ids) != len(set(run_ids))
        or config.confidence_level != 0.95
        or policy
        != {
            "test_tuning_prohibited": True,
            "threshold_reestimation_prohibited": True,
            "sample_replacement_prohibited": True,
            "network_used": False,
            "gpu_used": False,
            "real_data_access": False,
            "deterministic_difference_fail_closed": True,
            "detached_receipt_authoritative": False,
        }
    ):
        raise Stage9ToyConfigurationError("Stage 9 fixture safety declarations are invalid")
    return config


def load_stage9_toy_config(path: Path) -> Stage9ToyConfig:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if len(payload) > 64 * 1024:
        raise Stage9ToyConfigurationError("Stage 9 config exceeds its size limit")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Stage9ToyConfigurationError("Stage 9 config is not strict UTF-8 YAML") from error
    return parse_stage9_toy_config(value)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
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


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if not relative or relative.startswith("/") or "\\" in relative or "\x00" in relative:
        raise ValueError("package contains an unsafe relative path")
    return relative


def _read_tree(root: Path, *, label: str) -> dict[str, bytes]:
    package_root = root.resolve(strict=True)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"{label} root must be a regular directory")
    result: dict[str, bytes] = {}
    total = 0
    for path in sorted(package_root.rglob("*")):
        info = path.stat(follow_symlinks=False)
        if path.is_symlink() or bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"{label} contains a link or reparse point")
        if path.is_dir():
            continue
        if not path.is_file() or info.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"{label} contains a non-regular or oversized file")
        relative = _safe_relative(path, package_root)
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        if info.st_size != len(payload) or (info.st_size, info.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while it was read")
        total += len(payload)
        if total > _MAX_PACKAGE_BYTES or len(result) >= _MAX_PACKAGE_FILES:
            raise ValueError(f"{label} exceeds package bounds")
        result[relative] = payload
    if not result:
        raise ValueError(f"{label} package is empty")
    return result


def _file_manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    files = [
        {"relative_path": name, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        for name, payload in sorted(payloads.items())
    ]
    return {
        "schema_version": STAGE9_PACKAGE_VERSION,
        "digest_method": STAGE9_DIGEST_METHOD,
        "file_count": len(files),
        "files": files,
        "package_sha256": sha256_bytes(_canonical_compact_bytes(files)),
    }


def _verify_manifest(manifest: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    payloads = {
        name: payload for name, payload in actual.items() if name != "package_manifest.json"
    }
    if dict(manifest) != _file_manifest(payloads):
        raise ValueError("Stage 9 package manifest does not match complete package bytes")


def _prefixed(files: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    return {
        name[len(prefix) :]: payload for name, payload in files.items() if name.startswith(prefix)
    }


def _stage8_file_diff(
    files_a: Mapping[str, bytes], files_b: Mapping[str, bytes]
) -> dict[str, object]:
    names = sorted(set(files_a) | set(files_b))
    rows: list[dict[str, object]] = []
    matches = 0
    for name in names:
        left = files_a.get(name)
        right = files_b.get(name)
        if left is None:
            status = "MISSING_SESSION_A"
        elif right is None:
            status = "MISSING_SESSION_B"
        elif left == right:
            status = "MATCH"
            matches += 1
        else:
            status = "CONTENT_MISMATCH"
        rows.append(
            {
                "relative_path": name,
                "status": status,
                "session_a_sha256": None if left is None else sha256_bytes(left),
                "session_b_sha256": None if right is None else sha256_bytes(right),
                "session_a_size_bytes": None if left is None else len(left),
                "session_b_size_bytes": None if right is None else len(right),
            }
        )
    return {
        "schema_version": STAGE9_PACKAGE_VERSION,
        "deterministic_difference_fail_closed": True,
        "compared_file_count": len(names),
        "matching_file_count": matches,
        "differing_file_count": len(names) - matches,
        "files": rows,
    }


def _minimal_child_environment(session_nonce: str) -> dict[str, str]:
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
        "LC_ALL",
        "LANG",
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "RETINAENTAIL_STAGE9_SESSION_NONCE": session_nonce,
        }
    )
    return environment


def _launch_workers(
    *, stage8_config_path: Path, stage7_package: Path, output_dir: Path, project_root: Path
) -> tuple[_WorkerResult, _WorkerResult]:
    processes: list[tuple[str, str, Path, subprocess.Popen[str]]] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    for session_id in ("session_a", "session_b"):
        nonce = secrets.token_hex(32)
        package_dir = output_dir / session_id / "stage8"
        command = (
            sys.executable,
            "-m",
            "retinaentail.pilot.stage9_worker",
            "--config",
            str(stage8_config_path),
            "--stage7-package",
            str(stage7_package),
            "--output",
            str(package_dir),
            "--session-id",
            session_id,
            "--session-nonce",
            nonce,
            "--expected-parent-pid",
            str(os.getpid()),
        )
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=_minimal_child_environment(nonce),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            shell=False,
            creationflags=creationflags,
        )
        processes.append((session_id, nonce, package_dir, process))
    results: list[_WorkerResult] = []
    try:
        for session_id, nonce, package_dir, process in processes:
            stdout, stderr = process.communicate(timeout=_WORKER_TIMEOUT_SECONDS)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Stage 9 {session_id} worker failed with code {process.returncode}: "
                    f"{stderr[-2000:]}"
                )
            lines = tuple(line for line in stdout.splitlines() if line.strip())
            if len(lines) != 1:
                raise RuntimeError(f"Stage 9 {session_id} worker emitted unexpected output")
            receipt = _json_mapping(lines[0].encode("utf-8"), f"{session_id} receipt")
            expected_nonce = hashlib.sha256(nonce.encode("ascii")).hexdigest()
            if (
                receipt.get("session_id") != session_id
                or receipt.get("producer_process_id") != process.pid
                or receipt.get("parent_process_id") != os.getpid()
                or receipt.get("session_nonce_sha256") != expected_nonce
                or receipt.get("stage8_verified") is not True
                or receipt.get("stage8_status") != "SUCCESS"
                or receipt.get("network_used") is not False
                or receipt.get("gpu_used") is not False
                or receipt.get("real_data_read") is not False
                or receipt.get("detached_receipt_authoritative") is not False
            ):
                raise RuntimeError(f"Stage 9 {session_id} worker receipt failed closed")
            results.append(
                _WorkerResult(
                    session_id=session_id,
                    process_id=process.pid,
                    nonce_sha256=expected_nonce,
                    receipt=receipt,
                    package_dir=package_dir,
                )
            )
    except BaseException:
        for _, _, _, process in processes:
            if process.poll() is None:
                process.kill()
        raise
    if len(results) != 2 or results[0].process_id == results[1].process_id:
        raise RuntimeError("Stage 9 producer processes were not independent")
    return (results[0], results[1])


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _build_snapshot(
    worker: _WorkerResult,
    *,
    environment_lock: Stage9EnvironmentLock,
    config: Stage9ToyConfig,
) -> Stage9SessionSnapshot:
    verified = verify_stage8_toy_package(worker.package_dir)
    if (
        verified.package_sha256 != config.source_stage8_package_sha256
        or verified.package_file_count != config.source_stage8_package_file_count
        or verified.stage7_package_sha256 != config.source_stage7_package_sha256
        or verified.status != "SUCCESS"
        or not verified.verified
    ):
        raise ValueError("Stage 8 package differs from the frozen Stage 9 source contract")
    receipt_lock = parse_stage9_environment_lock(worker.receipt.get("environment_lock"))
    if (
        receipt_lock != environment_lock
        or worker.receipt.get("environment_lock_sha256") != environment_lock.sha256()
    ):
        expected_lock = environment_lock.to_dict()
        observed_lock = receipt_lock.to_dict()
        differences = tuple(
            name for name in expected_lock if expected_lock[name] != observed_lock.get(name)
        )
        raise ValueError(
            "producer environment differs from the preregistered environment lock: "
            + ",".join(differences)
        )
    observations_bytes = (worker.package_dir / "observations.json").read_bytes()
    preregistration_bytes = (worker.package_dir / "preregistration.json").read_bytes()
    audit_bytes = (worker.package_dir / "audit.json").read_bytes()
    manifest_bytes = (worker.package_dir / "package_manifest.json").read_bytes()
    preregistration = _json_mapping(preregistration_bytes, "Stage 8 preregistration")
    if (
        sha256_bytes(preregistration_bytes) != config.stage8_preregistration_sha256
        or preregistration.get("dataset_id") != config.dataset_id
        or preregistration.get("dataset_version") != config.dataset_version
        or preregistration.get("scope") != "SYNTHETIC_ONLY"
        or preregistration.get("split") != "test"
        or tuple(cast(list[object], preregistration.get("inclusion_case_ids")))
        != config.inclusion_case_ids
        or preregistration.get("alpha") != config.alpha
        or preregistration.get("positive_threshold") != config.positive_threshold
        or preregistration.get("negative_threshold") != config.negative_threshold
        or preregistration.get("positive_sample_count") != config.positive_sample_count
        or preregistration.get("negative_sample_count") != config.negative_sample_count
        or preregistration.get("config_hash") != config.config_hash
        or preregistration.get("data_hash") != config.data_hash
        or preregistration.get("model_hash") != config.model_hash
    ):
        raise ValueError("Stage 8 preregistration differs from the frozen Stage 9 contract")
    replay_specs = preregistration.get("replay_specs")
    if (
        not isinstance(replay_specs, list)
        or tuple(_mapping(item, "Stage 8 replay spec").get("run_id") for item in replay_specs)
        != config.replay_run_ids
        or tuple(_mapping(item, "Stage 8 replay spec").get("seed") for item in replay_specs)
        != config.replay_seeds
        or any(
            _mapping(item, "Stage 8 replay spec").get("degradation_replay_set_sha256")
            != config.degradation_replay_set_sha256
            for item in replay_specs
        )
    ):
        raise ValueError("Stage 8 seed or degradation replay preregistration drifted")
    observations = parse_stage8_observations(
        _decode_json(observations_bytes, "Stage 8 observations")
    )
    return Stage9SessionSnapshot(
        session_id=worker.session_id,
        producer_process_id=worker.process_id,
        session_nonce_sha256=worker.nonce_sha256,
        environment_lock_sha256=environment_lock.sha256(),
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        scope="SYNTHETIC_ONLY",
        split="test",
        package_sha256=verified.package_sha256,
        package_file_count=verified.package_file_count,
        package_manifest_sha256=sha256_bytes(manifest_bytes),
        stage8_audit_sha256=sha256_bytes(audit_bytes),
        stage8_observations_sha256=sha256_bytes(observations_bytes),
        stage8_preregistration_sha256=sha256_bytes(preregistration_bytes),
        source_stage7_package_sha256=verified.stage7_package_sha256,
        observations=observations,
    )


def _session_payload(worker: _WorkerResult, snapshot: Stage9SessionSnapshot) -> dict[str, object]:
    return {
        "schema_version": STAGE9_PACKAGE_VERSION,
        "session_id": worker.session_id,
        "producer_process_id": worker.process_id,
        "parent_supervisor_process_id": os.getpid(),
        "session_nonce_sha256": worker.nonce_sha256,
        "environment_lock_sha256": snapshot.environment_lock_sha256,
        "stage8_package": snapshot.authority_dict(),
        "producer_receipt_observed_directly": True,
        "complete_stage8_package_replay_verified_by_supervisor": True,
        "detached_receipt_authoritative": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
    }


def _package_summary(
    config: Stage9ToyConfig,
    diff: Stage9DiffReport,
    file_diff: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE9_PACKAGE_VERSION,
        "run_id": config.run_id,
        "method": STAGE9_METHOD,
        "scope": STAGE9_SCOPE,
        "status": diff.status.value,
        "case_count": len(config.inclusion_case_ids),
        "observation_comparison_count": len(diff.case_comparisons),
        "independent_processes": (
            diff.session_a.producer_process_id != diff.session_b.producer_process_id
        ),
        "independent_sessions": (
            diff.session_a.session_nonce_sha256 != diff.session_b.session_nonce_sha256
        ),
        "environment_lock_exact_match": (
            diff.session_a.environment_lock_sha256 == diff.session_b.environment_lock_sha256
        ),
        "stage8_package_exact_match": diff.session_a.package_sha256
        == diff.session_b.package_sha256,
        "matching_stage8_file_count": file_diff["matching_file_count"],
        "differing_stage8_file_count": file_diff["differing_file_count"],
        "deterministic_observation_match_count": sum(
            1 for item in diff.case_comparisons if item.deterministic_record_match
        ),
        "fail_closed_effective_count": sum(
            1 for item in diff.case_comparisons if item.effective_state.value == "OUT_OF_MODEL"
        ),
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


def _audit_payload(
    config: Stage9ToyConfig,
    environment_lock: Stage9EnvironmentLock,
    session_a: Mapping[str, object],
    session_b: Mapping[str, object],
    diff: Stage9DiffReport,
    file_diff: Mapping[str, object],
    lifecycle: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE9_PACKAGE_VERSION,
        "run_id": config.run_id,
        "method": STAGE9_METHOD,
        "scope": STAGE9_SCOPE,
        "status": diff.status.value,
        "authority_boundary": STAGE9_AUTHORITY_BOUNDARY,
        "clinical_claim": False,
        "preregistration": {
            "schema_version": STAGE9_PACKAGE_VERSION,
            "environment_lock_sha256": environment_lock.sha256(),
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
        },
        "session_a": dict(session_a),
        "session_b": dict(session_b),
        "capability_lifecycle": dict(lifecycle),
        "diff": diff.to_dict(),
        "package_diff_summary": {
            name: file_diff[name]
            for name in ("compared_file_count", "matching_file_count", "differing_file_count")
        },
        "deterministic_difference_fail_closed": True,
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


def _paths_nonoverlapping(values: Sequence[Path]) -> None:
    resolved = tuple(item.resolve(strict=False) for item in values)
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Stage 9 config, source, and output paths must not overlap")


def run_stage9_toy_fixture(
    config_path: Path,
    stage7_package: Path,
    output_dir: Path,
) -> Stage9ToyRunResult:
    """Run two fresh Python producers, consume two capabilities, and package the exact diff."""

    if any(not isinstance(item, Path) for item in (config_path, stage7_package, output_dir)):
        raise TypeError("all Stage 9 paths must be pathlib.Path values")
    config_root = config_path.resolve(strict=True)
    source_root = stage7_package.resolve(strict=True)
    target = output_dir.resolve(strict=False)
    _paths_nonoverlapping((config_root, source_root, target))
    if target.exists():
        raise FileExistsError(f"refusing to reuse a Stage 9 output directory: {target}")
    config = load_stage9_toy_config(config_root)
    environment_lock = replace(
        capture_stage9_environment_lock(python_hash_seed="0"),
        no_user_site=True,
        dont_write_bytecode=True,
    )
    project_root = Path(__file__).resolve().parents[3]
    stage8_config = (project_root / "configs" / "stage8_toy_repeatability_audit.yaml").resolve(
        strict=True
    )
    worker_a, worker_b = _launch_workers(
        stage8_config_path=stage8_config,
        stage7_package=source_root,
        output_dir=target,
        project_root=project_root,
    )
    replay_counts = {"session_a": 0, "session_b": 0}

    def replay_a() -> Stage9SessionSnapshot:
        replay_counts["session_a"] += 1
        return _build_snapshot(worker_a, environment_lock=environment_lock, config=config)

    def replay_b() -> Stage9SessionSnapshot:
        replay_counts["session_b"] += 1
        return _build_snapshot(worker_b, environment_lock=environment_lock, config=config)

    capability_a = issue_synthetic_stage8_session_capability(replay_a)
    capability_b = issue_synthetic_stage8_session_capability(replay_b)
    diff = consume_stage9_session_pair(
        capability_a,
        capability_b,
        lambda left, right: compare_stage8_sessions(
            left, right, confidence_level=config.confidence_level
        ),
    )
    reuse_rejected = False
    try:
        consume_stage9_session_pair(capability_a, capability_b, compare_stage8_sessions)
    except Stage9CapabilityError:
        reuse_rejected = True
    if not reuse_rejected:
        raise AssertionError("Stage 9 session capabilities became reusable")
    files_a = _read_tree(worker_a.package_dir, label="Stage 8 session A")
    files_b = _read_tree(worker_b.package_dir, label="Stage 8 session B")
    file_diff = _stage8_file_diff(files_a, files_b)
    if file_diff["differing_file_count"] != 0 or diff.status is not Stage9DiffStatus.SUCCESS:
        raise ValueError("synthetic Stage 9 deterministic comparison failed closed")
    lifecycle = {
        "schema_version": STAGE9_PACKAGE_VERSION,
        "issued_count": 2,
        "consumed_count": 2,
        "second_pair_consumption_rejected": reuse_rejected,
        "producer_complete_verification_count": 2,
        "supervisor_complete_replay_count": sum(replay_counts.values()),
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "all_capabilities_fresh": True,
        "cross_session_reuse_allowed": False,
        "detached_receipt_authoritative": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "session_a": session_capability_snapshot(capability_a),
        "session_b": session_capability_snapshot(capability_b),
    }
    session_a_payload = _session_payload(worker_a, diff.session_a)
    session_b_payload = _session_payload(worker_b, diff.session_b)
    audit = _audit_payload(
        config,
        environment_lock,
        session_a_payload,
        session_b_payload,
        diff,
        file_diff,
        lifecycle,
    )
    summary = _package_summary(config, diff, file_diff)
    root_payloads = {
        "audit.html": render_stage9_audit_html(audit).encode("utf-8"),
        "audit.json": render_stage9_audit_json(audit).encode("utf-8"),
        "config_snapshot.json": _canonical_json_bytes(dict(config.raw)),
        "environment_lock.json": environment_lock_json_bytes(environment_lock),
        "package_diff.json": _canonical_json_bytes(file_diff),
        "session_a.json": _canonical_json_bytes(session_a_payload),
        "session_b.json": _canonical_json_bytes(session_b_payload),
        "summary.json": _canonical_json_bytes(summary),
    }
    for name, payload in root_payloads.items():
        write_bytes_fail_safe(target / name, payload)
    package_files = {
        **root_payloads,
        **{f"session_a/stage8/{name}": payload for name, payload in files_a.items()},
        **{f"session_b/stage8/{name}": payload for name, payload in files_b.items()},
    }
    package_manifest = _file_manifest(package_files)
    write_bytes_fail_safe(target / "package_manifest.json", _canonical_json_bytes(package_manifest))
    final_tree = _read_tree(target, label="Stage 9 generated package")
    _verify_manifest(package_manifest, final_tree)
    return Stage9ToyRunResult(
        output_dir=output_dir,
        package_sha256=cast(str, package_manifest["package_sha256"]),
        package_file_count=len(final_tree),
        stage8_package_sha256=diff.session_a.package_sha256,
        source_stage7_package_sha256=diff.session_a.source_stage7_package_sha256,
        producer_process_ids=(
            diff.session_a.producer_process_id,
            diff.session_b.producer_process_id,
        ),
        case_count=len(config.inclusion_case_ids),
        observation_comparison_count=len(diff.case_comparisons),
        matching_stage8_file_count=cast(int, file_diff["matching_file_count"]),
        differing_stage8_file_count=file_diff["differing_file_count"],
        status=diff.status.value,
        verified=True,
    )


def _snapshot_from_package(
    *,
    session_id: str,
    package_dir: Path,
    session_payload: Mapping[str, object],
    environment_lock: Stage9EnvironmentLock,
    config: Stage9ToyConfig,
) -> Stage9SessionSnapshot:
    worker = _WorkerResult(
        session_id=session_id,
        process_id=_positive_int(session_payload.get("producer_process_id"), "producer_process_id"),
        nonce_sha256=_digest(session_payload.get("session_nonce_sha256"), "session_nonce_sha256"),
        receipt={
            "environment_lock": environment_lock.to_dict(),
            "environment_lock_sha256": environment_lock.sha256(),
        },
        package_dir=package_dir,
    )
    return _build_snapshot(worker, environment_lock=environment_lock, config=config)


def verify_stage9_toy_package(output_dir: Path) -> Stage9ToyRunResult:
    """Replay both embedded Stage 8 packages and recompute all deterministic comparisons."""

    if not isinstance(output_dir, Path):
        raise TypeError("Stage 9 package root must be a pathlib.Path")
    package_root = output_dir.resolve(strict=True)
    actual = _read_tree(package_root, label="Stage 9")
    manifest_bytes = actual.get("package_manifest.json")
    if manifest_bytes is None:
        raise ValueError("Stage 9 package manifest is missing")
    manifest = _json_mapping(manifest_bytes, "Stage 9 package manifest")
    _verify_manifest(manifest, actual)
    files_a = _prefixed(actual, "session_a/stage8/")
    files_b = _prefixed(actual, "session_b/stage8/")
    expected_paths = {
        "package_manifest.json",
        *_ROOT_FILES,
        *(f"session_a/stage8/{name}" for name in files_a),
        *(f"session_b/stage8/{name}" for name in files_b),
    }
    if set(actual) != expected_paths:
        raise ValueError("Stage 9 package contains an unknown path")
    config = parse_stage9_toy_config(
        _decode_json(actual["config_snapshot.json"], "Stage 9 config snapshot")
    )
    environment_lock = parse_stage9_environment_lock(
        _decode_json(actual["environment_lock.json"], "Stage 9 environment lock")
    )
    if environment_lock_json_bytes(environment_lock) != actual["environment_lock.json"]:
        raise ValueError("Stage 9 environment lock is not canonical")
    session_a_payload = _json_mapping(actual["session_a.json"], "Stage 9 session A")
    session_b_payload = _json_mapping(actual["session_b.json"], "Stage 9 session B")
    session_a = _snapshot_from_package(
        session_id="session_a",
        package_dir=package_root / "session_a" / "stage8",
        session_payload=session_a_payload,
        environment_lock=environment_lock,
        config=config,
    )
    session_b = _snapshot_from_package(
        session_id="session_b",
        package_dir=package_root / "session_b" / "stage8",
        session_payload=session_b_payload,
        environment_lock=environment_lock,
        config=config,
    )
    if session_a.authority_dict() != session_a_payload.get(
        "stage8_package"
    ) or session_b.authority_dict() != session_b_payload.get("stage8_package"):
        raise ValueError("Stage 9 session descriptor differs from complete package replay")
    diff = compare_stage8_sessions(session_a, session_b, confidence_level=config.confidence_level)
    file_diff = _stage8_file_diff(files_a, files_b)
    expected_file_diff = _canonical_json_bytes(file_diff)
    if actual["package_diff.json"] != expected_file_diff:
        raise ValueError("Stage 9 complete package diff does not regenerate")
    audit = _json_mapping(actual["audit.json"], "Stage 9 audit")
    if audit.get("diff") != diff.to_dict() or audit.get("status") != diff.status.value:
        raise ValueError("Stage 9 audit differs from recomputed deterministic diff")
    if audit.get("session_a") != session_a_payload or audit.get("session_b") != session_b_payload:
        raise ValueError("Stage 9 audit session provenance differs from replay")
    lifecycle = _mapping(audit.get("capability_lifecycle"), "capability_lifecycle")
    required_lifecycle = {
        "issued_count": 2,
        "consumed_count": 2,
        "second_pair_consumption_rejected": True,
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "all_capabilities_fresh": True,
        "cross_session_reuse_allowed": False,
        "detached_receipt_authoritative": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
    }
    if any(lifecycle.get(name) != value for name, value in required_lifecycle.items()):
        raise ValueError("Stage 9 capability lifecycle receipt is invalid")
    if actual["audit.html"] != render_stage9_audit_html(audit).encode("utf-8"):
        raise ValueError("Stage 9 HTML report does not regenerate")
    summary = _package_summary(config, diff, file_diff)
    if actual["summary.json"] != _canonical_json_bytes(summary):
        raise ValueError("Stage 9 summary does not regenerate")
    package_sha256 = cast(str, manifest.get("package_sha256"))
    if not isinstance(package_sha256, str):
        raise ValueError("Stage 9 package SHA-256 is missing")
    return Stage9ToyRunResult(
        output_dir=output_dir,
        package_sha256=package_sha256,
        package_file_count=len(actual),
        stage8_package_sha256=session_a.package_sha256,
        source_stage7_package_sha256=session_a.source_stage7_package_sha256,
        producer_process_ids=(session_a.producer_process_id, session_b.producer_process_id),
        case_count=len(config.inclusion_case_ids),
        observation_comparison_count=len(diff.case_comparisons),
        matching_stage8_file_count=cast(int, file_diff["matching_file_count"]),
        differing_stage8_file_count=cast(int, file_diff["differing_file_count"]),
        status=diff.status.value,
        verified=True,
    )
