"""Stage 12 two-process random signer compromise/recovery fixture."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml

from retinaentail.pilot.stage11_fixture import verify_stage11_toy_package
from retinaentail.reporting.stage12 import render_stage12_audit_html, render_stage12_audit_json
from retinaentail.reproducibility.stage12 import (
    STAGE12_SCHEMA_VERSION,
    Stage12VerificationCode,
    Stage12VerificationError,
    canonical_sha256,
    compare_recovered_runners,
    sha256_bytes,
    verify_runner_recovery_bundle,
)
from retinaentail.witnesses.replay import write_bytes_fail_safe

STAGE12_METHOD: Final = "two_process_random_ed25519_compromise_recovery_v1"
STAGE12_CONFIG_VERSION: Final = "12.0.0"
_MAX_FILE_BYTES: Final = 32 * 1024 * 1024
_MAX_TOTAL_BYTES: Final = 256 * 1024 * 1024
_MAX_FILES: Final = 1_000
_TAMPER_SCENARIOS: Final = (
    "recovered_payload_mutation",
    "recovered_signature_mutation",
    "recovery_proof_mutation",
    "status_signature_mutation",
    "untrusted_anchor",
    "generation2_certificate_mutation",
    "session_nonce_replay",
    "shared_recovery_key",
    "frozen_contract_mutation",
    "malformed_bundle_extra_field",
)


class Stage12ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage12Config:
    run_id: str
    dataset_id: str
    dataset_version: str
    runner_ids: tuple[str, str]
    trusted_time_epoch: int
    compromise_epoch: int
    recovery_epoch: int
    source_package_sha256: str
    source_package_file_count: int
    source_manifest_sha256: str
    source_audit_sha256: str
    source_summary_sha256: str
    frozen_payload: Mapping[str, object]
    tamper_scenarios: tuple[str, ...]
    raw: Mapping[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage12RunResult:
    output_dir: Path
    package_sha256: str
    package_file_count: int
    source_stage11_package_sha256: str
    source_stage11_package_file_count: int
    runner_process_ids: tuple[int, int]
    recovered_key_ids: tuple[str, str]
    tamper_scenario_count: int
    tamper_rejection_count: int
    comparison_status: str
    status: str
    verified: bool


def _mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage12ConfigurationError(f"{label} fields are incomplete or unknown")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00"):
        raise Stage12ConfigurationError(f"{label} is invalid")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise Stage12ConfigurationError(f"{label} is not SHA-256")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage12ConfigurationError(f"{label} is not a positive integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage12ConfigurationError(f"{label} is not numeric")
    result = float(value)
    if not result == result or result in (float("inf"), float("-inf")):
        raise Stage12ConfigurationError(f"{label} is not finite")
    return result


def parse_stage12_config(value: object) -> Stage12Config:
    root = _mapping(
        value,
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
            "independent_process_count",
            "cross_physical_host_claim",
            "production_authentication",
            "trusted_time_epoch",
            "compromise_epoch",
            "recovery_epoch",
            "source_contract",
            "locked_contract",
            "signer_policy",
            "tamper_scenarios",
            "locked_policy",
        },
        "Stage 12 config",
    )
    if (
        root["schema_version"] != STAGE12_CONFIG_VERSION
        or root["method"] != STAGE12_METHOD
        or root["scope"] != "SYNTHETIC_FIXTURE"
        or root["split"] != "test"
        or root["physical_host_count"] != 1
        or root["independent_process_count"] != 2
        or root["cross_physical_host_claim"] is not False
        or root["production_authentication"] is not False
    ):
        raise Stage12ConfigurationError("Stage 12 config exceeds local fixture scope")
    runner_values = root["runner_ids"]
    if not isinstance(runner_values, list) or len(runner_values) != 2:
        raise Stage12ConfigurationError("Stage 12 requires two runner IDs")
    runners = tuple(_text(item, "runner id") for item in runner_values)
    if len(set(runners)) != 2:
        raise Stage12ConfigurationError("Stage 12 runner IDs are not unique")
    source = _mapping(
        root["source_contract"],
        {
            "stage11_package_sha256",
            "stage11_package_file_count",
            "stage11_manifest_sha256",
            "stage11_audit_sha256",
            "stage11_summary_sha256",
        },
        "Stage 12 source contract",
    )
    locked = _mapping(
        root["locked_contract"],
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
        "Stage 12 frozen contract",
    )
    signer = _mapping(
        root["signer_policy"],
        {
            "adapter",
            "algorithm",
            "random_private_keys",
            "private_key_export_allowed",
            "private_key_serialization_allowed",
            "private_key_artifact_write_allowed",
            "hsm_backed",
            "production_authentication",
            "local_status_is_rfc6960_ocsp",
            "complete_wheel_archive_hashes",
        },
        "Stage 12 signer policy",
    )
    if (
        signer["adapter"] != "InMemoryEd25519Signer"
        or signer["algorithm"] != "Ed25519"
        or signer["random_private_keys"] is not True
        or any(
            signer[name] is not False
            for name in (
                "private_key_export_allowed",
                "private_key_serialization_allowed",
                "private_key_artifact_write_allowed",
                "hsm_backed",
                "production_authentication",
                "local_status_is_rfc6960_ocsp",
                "complete_wheel_archive_hashes",
            )
        )
    ):
        raise Stage12ConfigurationError("Stage 12 local signer policy is unsafe")
    policy = _mapping(
        root["locked_policy"],
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
        "Stage 12 locked policy",
    )
    if any(
        policy[name] is not True
        for name in (
            "test_tuning_prohibited",
            "threshold_reestimation_prohibited",
            "sample_replacement_prohibited",
            "deterministic_difference_fail_closed",
            "large_model_training_prohibited",
            "weights_download_prohibited",
        )
    ) or any(
        policy[name] is not False for name in ("network_used", "gpu_used", "real_data_access")
    ):
        raise Stage12ConfigurationError("Stage 12 locked policy is unsafe")
    trusted = _positive_int(root["trusted_time_epoch"], "trusted time")
    compromise = _positive_int(root["compromise_epoch"], "compromise time")
    recovery = _positive_int(root["recovery_epoch"], "recovery time")
    if not compromise < recovery <= trusted:
        raise Stage12ConfigurationError("Stage 12 recovery timeline is invalid")
    scenarios_value = root["tamper_scenarios"]
    if not isinstance(scenarios_value, list):
        raise Stage12ConfigurationError("Stage 12 tamper scenarios are invalid")
    scenarios = tuple(_text(item, "tamper scenario") for item in scenarios_value)
    if scenarios != _TAMPER_SCENARIOS:
        raise Stage12ConfigurationError("Stage 12 tamper preregistration differs")
    source_material = {
        "source_stage11_package_sha256": _digest(
            source["stage11_package_sha256"], "source package"
        ),
        "source_stage11_package_file_count": _positive_int(
            source["stage11_package_file_count"], "source file count"
        ),
        "source_stage11_manifest_sha256": _digest(
            source["stage11_manifest_sha256"], "source manifest"
        ),
        "source_stage11_audit_sha256": _digest(source["stage11_audit_sha256"], "source audit"),
        "source_stage11_summary_sha256": _digest(
            source["stage11_summary_sha256"], "source summary"
        ),
        "alpha": _finite(locked["alpha"], "alpha"),
        "positive_threshold": _finite(locked["positive_threshold"], "positive threshold"),
        "negative_threshold": _finite(locked["negative_threshold"], "negative threshold"),
        "positive_sample_count": _positive_int(
            locked["positive_sample_count"], "positive sample count"
        ),
        "negative_sample_count": _positive_int(
            locked["negative_sample_count"], "negative sample count"
        ),
        "config_hash": _digest(locked["config_hash"], "config hash"),
        "data_hash": _digest(locked["data_hash"], "data hash"),
        "model_hash": _digest(locked["model_hash"], "model hash"),
    }
    frozen = {**source_material, "frozen_contract_sha256": canonical_sha256(source_material)}
    return Stage12Config(
        run_id=_text(root["run_id"], "run id"),
        dataset_id=_text(root["dataset_id"], "dataset id"),
        dataset_version=_text(root["dataset_version"], "dataset version"),
        runner_ids=cast(tuple[str, str], runners),
        trusted_time_epoch=trusted,
        compromise_epoch=compromise,
        recovery_epoch=recovery,
        source_package_sha256=cast(str, frozen["source_stage11_package_sha256"]),
        source_package_file_count=cast(int, frozen["source_stage11_package_file_count"]),
        source_manifest_sha256=cast(str, frozen["source_stage11_manifest_sha256"]),
        source_audit_sha256=cast(str, frozen["source_stage11_audit_sha256"]),
        source_summary_sha256=cast(str, frozen["source_stage11_summary_sha256"]),
        frozen_payload=frozen,
        tamper_scenarios=scenarios,
        raw=root,
    )


def load_stage12_config(path: Path) -> Stage12Config:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return parse_stage12_config(value)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _unique_json(value: bytes, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[object, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for raw_key, item in pairs:
            if not isinstance(raw_key, str) or raw_key in result:
                raise ValueError(f"{label} contains invalid or duplicate JSON keys")
            result[raw_key] = item
        return result

    parsed = json.loads(value, object_pairs_hook=unique)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} is not a JSON object")
    return parsed


def _read_tree(root: Path, label: str) -> dict[str, bytes]:
    resolved = root.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{label} is not a safe directory")
    result: dict[str, bytes] = {}
    total = 0
    for path in sorted(resolved.rglob("*")):
        info = path.stat(follow_symlinks=False)
        if path.is_symlink() or bool(
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"{label} contains a link or reparse point")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} contains a non-regular file")
        relative = path.relative_to(resolved).as_posix()
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or ":" in relative
            or "\\" in relative
            or "\x00" in relative
        ):
            raise ValueError(f"{label} contains an unsafe path")
        size = info.st_size
        if size > _MAX_FILE_BYTES:
            raise ValueError(f"{label} file exceeds limit")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(
            payload
        ) != info.st_size:
            raise ValueError(f"{label} changed while read")
        total += len(payload)
        if total > _MAX_TOTAL_BYTES or len(result) >= _MAX_FILES:
            raise ValueError(f"{label} exceeds package limits")
        result[relative] = payload
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def _manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    entries = [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    ]
    core = {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "digest_method": "sha256-canonical-file-manifest-v1",
        "file_count": len(entries),
        "files": entries,
    }
    return {**core, "package_sha256": canonical_sha256(core)}


def _verify_manifest(manifest: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    expected = _manifest(actual)
    if manifest != expected:
        raise ValueError("Stage 12 package manifest differs")


def _source_material(root: Path, config: Stage12Config) -> dict[str, bytes]:
    result = verify_stage11_toy_package(root)
    if (
        result.package_sha256 != config.source_package_sha256
        or result.package_file_count != config.source_package_file_count
    ):
        raise ValueError("Stage 12 source Stage 11 package differs from preregistration")
    files = _read_tree(root, "Stage 11 source")
    checks = {
        "package_manifest.json": config.source_manifest_sha256,
        "audit.json": config.source_audit_sha256,
        "summary.json": config.source_summary_sha256,
    }
    for name, expected in checks.items():
        if sha256_bytes(files[name]) != expected:
            raise ValueError(f"Stage 12 source file differs: {name}")
    return files


def _worker_request(config: Stage12Config, runner_id: str) -> dict[str, object]:
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "runner_id": runner_id,
        "trusted_time_epoch": config.trusted_time_epoch,
        "compromise_epoch": config.compromise_epoch,
        "recovery_epoch": config.recovery_epoch,
        "frozen_payload": dict(config.frozen_payload),
    }


def _launch_workers(target: Path, config: Stage12Config) -> tuple[dict[str, object], ...]:
    processes: list[subprocess.Popen[str]] = []
    bundle_paths: list[Path] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for runner_id in config.runner_ids:
        request_path = target / "requests" / f"{runner_id}.json"
        bundle_path = target / runner_id / "bundle.json"
        write_bytes_fail_safe(request_path, _json_bytes(_worker_request(config, runner_id)))
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "retinaentail.pilot.stage12_worker",
                    "--request",
                    str(request_path),
                    "--output",
                    str(bundle_path),
                ],
                cwd=target.parent,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        )
        bundle_paths.append(bundle_path)
    for process in processes:
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("Stage 12 signer worker exceeded 30 seconds") from None
        if process.returncode != 0 or stdout or stderr:
            raise RuntimeError(
                f"Stage 12 signer worker failed: code={process.returncode}; "
                f"stdout={stdout[-500:]!r}; stderr={stderr[-500:]!r}"
            )
    return tuple(
        _unique_json(path.read_bytes(), f"{runner_id} bundle")
        for runner_id, path in zip(config.runner_ids, bundle_paths, strict=True)
    )


def _observed(call: Callable[[], object]) -> str:
    try:
        call()
    except Stage12VerificationError as error:
        return error.code.value
    except (TypeError, ValueError):
        return Stage12VerificationCode.MALFORMED_ARTIFACT.value
    return "UNEXPECTED_SUCCESS"


def _tamper_audit(
    bundles: tuple[dict[str, object], dict[str, object]],
    verified: tuple[dict[str, object], dict[str, object]],
    config: Stage12Config,
) -> dict[str, object]:
    def verify_a(value: object) -> object:
        return verify_runner_recovery_bundle(
            value,
            expected_runner_id=config.runner_ids[0],
            expected_frozen_payload=config.frozen_payload,
            trusted_time_epoch=config.trusted_time_epoch,
        )

    cases: list[tuple[str, str, Callable[[], object]]] = []
    payload_mutation = copy.deepcopy(bundles[0])
    cast(
        dict[str, object],
        cast(dict[str, object], payload_mutation["recovered_envelope"])["payload"],
    )["model_hash"] = "0" * 64
    cases.append(
        (
            "recovered_payload_mutation",
            Stage12VerificationCode.FROZEN_CONTRACT_MISMATCH.value,
            lambda: verify_a(payload_mutation),
        )
    )
    signature_mutation = copy.deepcopy(bundles[0])
    cast(dict[str, object], signature_mutation["recovered_envelope"])["signature_b64"] = "A" * 88
    cases.append(
        (
            "recovered_signature_mutation",
            Stage12VerificationCode.ENVELOPE_SIGNATURE_INVALID.value,
            lambda: verify_a(signature_mutation),
        )
    )
    proof_mutation = copy.deepcopy(bundles[0])
    cast(dict[str, object], proof_mutation["recovery_proof"])["root_signature_b64"] = "A" * 88
    cases.append(
        (
            "recovery_proof_mutation",
            Stage12VerificationCode.RECOVERY_PROOF_INVALID.value,
            lambda: verify_a(proof_mutation),
        )
    )
    status_mutation = copy.deepcopy(bundles[0])
    cast(dict[str, object], status_mutation["status_staple"])["signature_b64"] = "A" * 88
    cases.append(
        (
            "status_signature_mutation",
            Stage12VerificationCode.STATUS_SIGNATURE_INVALID.value,
            lambda: verify_a(status_mutation),
        )
    )
    anchor_mutation = copy.deepcopy(bundles[0])
    cast(dict[str, object], anchor_mutation["trust_anchor"])["signature_b64"] = "A" * 88
    cases.append(
        (
            "untrusted_anchor",
            Stage12VerificationCode.TRUST_ANCHOR_INVALID.value,
            lambda: verify_a(anchor_mutation),
        )
    )
    certificate_mutation = copy.deepcopy(bundles[0])
    cast(dict[str, object], certificate_mutation["generation2_certificate"])["signature_b64"] = (
        "A" * 88
    )
    cases.append(
        (
            "generation2_certificate_mutation",
            Stage12VerificationCode.CERTIFICATE_SIGNATURE_INVALID.value,
            lambda: verify_a(certificate_mutation),
        )
    )
    nonce_replay = copy.deepcopy(verified[1])
    nonce_replay["session_nonce_sha256"] = verified[0]["session_nonce_sha256"]
    cases.append(
        (
            "session_nonce_replay",
            Stage12VerificationCode.SESSION_NONCE_REPLAYED.value,
            lambda: _comparison_error(compare_recovered_runners(verified[0], nonce_replay)),
        )
    )
    shared_key = copy.deepcopy(verified[1])
    shared_key["recovered_key_id"] = verified[0]["recovered_key_id"]
    cases.append(
        (
            "shared_recovery_key",
            Stage12VerificationCode.SHARED_RECOVERY_KEY_REJECTED.value,
            lambda: _comparison_error(compare_recovered_runners(verified[0], shared_key)),
        )
    )
    frozen_mutation = copy.deepcopy(verified[1])
    cast(dict[str, object], frozen_mutation["frozen_payload"])["model_hash"] = "0" * 64
    cases.append(
        (
            "frozen_contract_mutation",
            Stage12VerificationCode.FROZEN_CONTRACT_MISMATCH.value,
            lambda: _comparison_error(compare_recovered_runners(verified[0], frozen_mutation)),
        )
    )
    malformed = copy.deepcopy(bundles[0])
    malformed["unexpected"] = False
    cases.append(
        (
            "malformed_bundle_extra_field",
            Stage12VerificationCode.MALFORMED_ARTIFACT.value,
            lambda: verify_a(malformed),
        )
    )
    if tuple(name for name, _, _ in cases) != config.tamper_scenarios:
        raise ValueError("Stage 12 runtime tamper cases differ from preregistration")
    results = []
    for name, expected, call in cases:
        observed = _observed(call)
        results.append(
            {
                "scenario": name,
                "expected_code": expected,
                "observed_code": observed,
                "rejected": observed == expected,
            }
        )
    rejected = sum(item["rejected"] is True for item in results)
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "status": "SUCCESS" if rejected == len(results) else "FAIL_CLOSED",
        "scenario_count": len(results),
        "rejection_count": rejected,
        "results": results,
        "failure_state": "OUT_OF_MODEL",
        "synthetic_only": True,
    }


def _comparison_error(comparison: Mapping[str, object]) -> None:
    differences = comparison.get("differences")
    if isinstance(differences, list) and differences:
        code = str(differences[0])
        try:
            enum_code = Stage12VerificationCode(code)
        except ValueError:
            raise Stage12VerificationError(
                Stage12VerificationCode.FROZEN_CONTRACT_MISMATCH, code
            ) from None
        raise Stage12VerificationError(enum_code, code)


def _audit(
    config: Stage12Config,
    verified: tuple[dict[str, object], dict[str, object]],
    comparison: Mapping[str, object],
    tamper: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "run_id": config.run_id,
        "method": STAGE12_METHOD,
        "scope": "SYNTHETIC_FIXTURE",
        "status": (
            "SUCCESS"
            if comparison["status"] == "SUCCESS" and tamper["status"] == "SUCCESS"
            else "FAIL_CLOSED"
        ),
        "failure_state": "OUT_OF_MODEL",
        "source_stage11_package_sha256": config.source_package_sha256,
        "source_stage11_package_file_count": config.source_package_file_count,
        "runner_results": list(verified),
        "runner_process_ids": [item["producer_process_id"] for item in verified],
        "independent_process_count": len({item["producer_process_id"] for item in verified}),
        "physical_host_count": 1,
        "cross_physical_host_tested": False,
        "cross_physical_host_claim": False,
        "signer_adapter_interoperability": "LOCAL_PROTOCOL_FIXTURE_ONLY",
        "random_ephemeral_private_keys": True,
        "private_keys_exported": False,
        "private_keys_written_to_artifacts": False,
        "hsm_backed": False,
        "trusted_timestamp_service_used": False,
        "rfc6960_ocsp_used": False,
        "local_signed_status_staple_verified": True,
        "compromised_generation": 1,
        "recovered_generation": 2,
        "compromised_keys_rejected_after_revocation": True,
        "recovered_runner_comparison": dict(comparison),
        "tamper_audit": dict(tamper),
        "production_authentication": False,
        "production_ceremony_executed": False,
        "complete_wheel_archive_hashes": False,
        "hardware_identity_attested": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "test_split_used_for_tuning": False,
        "thresholds_reestimated": False,
        "sample_replacement_performed": False,
        "network_used_during_fixture": False,
        "gpu_used": False,
        "weights_downloaded": False,
        "large_model_trained": False,
        "clinical_claim": False,
    }


def _summary(
    config: Stage12Config,
    audit: Mapping[str, object],
    tamper: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": audit["status"],
        "scope": "SYNTHETIC_FIXTURE",
        "source_stage11_package_sha256": config.source_package_sha256,
        "independent_process_count": audit["independent_process_count"],
        "physical_host_count": 1,
        "random_ephemeral_private_keys": True,
        "private_keys_exported": False,
        "compromised_generation": 1,
        "recovered_generation": 2,
        "tamper_scenario_count": tamper["scenario_count"],
        "tamper_rejection_count": tamper["rejection_count"],
        "production_authentication": False,
        "production_ceremony_executed": False,
        "real_data_read_count": 0,
        "clinical_claim": False,
    }


def _preregistration(config: Stage12Config) -> dict[str, object]:
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "run_id": config.run_id,
        "trusted_time_epoch": config.trusted_time_epoch,
        "compromise_epoch": config.compromise_epoch,
        "recovery_epoch": config.recovery_epoch,
        "runner_ids": list(config.runner_ids),
        "frozen_payload": dict(config.frozen_payload),
        "tamper_scenarios": list(config.tamper_scenarios),
        "failure_state": "OUT_OF_MODEL",
    }


def run_stage12_fixture(
    config_path: Path, source_stage11_package: Path, output_dir: Path
) -> Stage12RunResult:
    config = load_stage12_config(config_path)
    target = output_dir.resolve(strict=False)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite Stage 12 output: {target}")
    source_root = source_stage11_package.resolve(strict=True)
    if target == source_root or target in source_root.parents or source_root in target.parents:
        raise ValueError("Stage 12 source and output paths overlap")
    source_files = _source_material(source_root, config)
    target.mkdir(parents=True)
    raw_bundles = _launch_workers(target, config)
    bundles = cast(tuple[dict[str, object], dict[str, object]], raw_bundles)
    verified = cast(
        tuple[dict[str, object], dict[str, object]],
        tuple(
            verify_runner_recovery_bundle(
                bundle,
                expected_runner_id=runner_id,
                expected_frozen_payload=config.frozen_payload,
                trusted_time_epoch=config.trusted_time_epoch,
            )
            for runner_id, bundle in zip(config.runner_ids, bundles, strict=True)
        ),
    )
    comparison = compare_recovered_runners(*verified)
    tamper = _tamper_audit(bundles, verified, config)
    audit = _audit(config, verified, comparison, tamper)
    summary = _summary(config, audit, tamper)
    preregistration = _preregistration(config)
    payloads = {
        "audit.html": render_stage12_audit_html(audit).encode(),
        "audit.json": render_stage12_audit_json(audit).encode(),
        "config.json": _json_bytes(dict(config.raw)),
        "preregistration.json": _json_bytes(preregistration),
        "summary.json": _json_bytes(summary),
        "tamper_audit.json": _json_bytes(tamper),
        **{f"source_stage11/{name}": value for name, value in source_files.items()},
    }
    for name, payload in payloads.items():
        write_bytes_fail_safe(target / name, payload)
    actual_without_manifest = _read_tree(target, "Stage 12 output")
    manifest = _manifest(actual_without_manifest)
    write_bytes_fail_safe(target / "package_manifest.json", _json_bytes(manifest))
    result = verify_stage12_package(target)
    if not result.verified:
        raise RuntimeError("Stage 12 package did not verify")
    return result


def verify_stage12_package(output_dir: Path) -> Stage12RunResult:
    root = output_dir.resolve(strict=True)
    actual = _read_tree(root, "Stage 12 package")
    manifest = _unique_json(actual.pop("package_manifest.json"), "Stage 12 manifest")
    _verify_manifest(manifest, actual)
    config = parse_stage12_config(_unique_json(actual["config.json"], "Stage 12 config"))
    if actual["config.json"] != _json_bytes(dict(config.raw)):
        raise ValueError("Stage 12 config encoding differs")
    source_prefix = "source_stage11/"
    source_names = {name: value for name, value in actual.items() if name.startswith(source_prefix)}
    if len(source_names) != config.source_package_file_count:
        raise ValueError("Stage 12 embedded Stage 11 file count differs")
    embedded_root = root / "source_stage11"
    embedded_files = _source_material(embedded_root, config)
    expected_paths = {
        "audit.html",
        "audit.json",
        "config.json",
        "preregistration.json",
        "summary.json",
        "tamper_audit.json",
        *(f"requests/{runner_id}.json" for runner_id in config.runner_ids),
        *(f"{runner_id}/bundle.json" for runner_id in config.runner_ids),
        *(f"source_stage11/{name}" for name in embedded_files),
    }
    if set(actual) != expected_paths:
        raise ValueError("Stage 12 package paths differ")
    for runner_id in config.runner_ids:
        if actual[f"requests/{runner_id}.json"] != _json_bytes(_worker_request(config, runner_id)):
            raise ValueError(f"Stage 12 {runner_id} request differs")
    bundles = cast(
        tuple[dict[str, object], dict[str, object]],
        tuple(
            _unique_json(actual[f"{runner_id}/bundle.json"], f"{runner_id} bundle")
            for runner_id in config.runner_ids
        ),
    )
    verified = cast(
        tuple[dict[str, object], dict[str, object]],
        tuple(
            verify_runner_recovery_bundle(
                bundle,
                expected_runner_id=runner_id,
                expected_frozen_payload=config.frozen_payload,
                trusted_time_epoch=config.trusted_time_epoch,
            )
            for runner_id, bundle in zip(config.runner_ids, bundles, strict=True)
        ),
    )
    comparison = compare_recovered_runners(*verified)
    tamper = _tamper_audit(bundles, verified, config)
    audit = _audit(config, verified, comparison, tamper)
    if actual["audit.json"] != render_stage12_audit_json(audit).encode():
        raise ValueError("Stage 12 audit JSON differs")
    if actual["audit.html"] != render_stage12_audit_html(audit).encode():
        raise ValueError("Stage 12 audit HTML differs")
    if _unique_json(actual["tamper_audit.json"], "tamper audit") != tamper:
        raise ValueError("Stage 12 tamper audit differs")
    if actual["preregistration.json"] != _json_bytes(_preregistration(config)):
        raise ValueError("Stage 12 preregistration differs")
    if actual["summary.json"] != _json_bytes(_summary(config, audit, tamper)):
        raise ValueError("Stage 12 summary differs")
    pids = cast(tuple[int, int], tuple(cast(int, item["producer_process_id"]) for item in verified))
    keys = cast(tuple[str, str], tuple(cast(str, item["recovered_key_id"]) for item in verified))
    return Stage12RunResult(
        output_dir=root,
        package_sha256=cast(str, manifest["package_sha256"]),
        package_file_count=cast(int, manifest["file_count"]) + 1,
        source_stage11_package_sha256=config.source_package_sha256,
        source_stage11_package_file_count=config.source_package_file_count,
        runner_process_ids=pids,
        recovered_key_ids=keys,
        tamper_scenario_count=cast(int, tamper["scenario_count"]),
        tamper_rejection_count=cast(int, tamper["rejection_count"]),
        comparison_status=cast(str, comparison["status"]),
        status=cast(str, audit["status"]),
        verified=True,
    )
