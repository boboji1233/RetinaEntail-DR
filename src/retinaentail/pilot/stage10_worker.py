"""Isolated Stage 10 local runner and environment-probe entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
from pathlib import Path

from retinaentail.pilot.stage9_fixture import run_stage9_toy_fixture, verify_stage9_toy_package
from retinaentail.pilot.stage10_fixture import (
    create_synthetic_signed_envelope,
    load_stage10_toy_config,
)
from retinaentail.reproducibility.stage10 import (
    capture_stage10_environment_lock,
    parse_stage10_environment_lock,
)
from retinaentail.witnesses.replay import write_bytes_fail_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retinaentail-stage10-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--expected-parent-pid", type=int, required=True)
    probe.add_argument("--nonce", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--source-stage9-package", type=Path, required=True)
    run.add_argument("--expected-environment-lock", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--runner-id", required=True)
    run.add_argument("--runner-nonce", required=True)
    run.add_argument("--expected-parent-pid", type=int, required=True)
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _set_frozen_locale() -> None:
    if locale.setlocale(locale.LC_ALL, "C") != "C":
        raise RuntimeError("Stage 10 runner could not freeze the C locale")


def _probe(arguments: argparse.Namespace) -> int:
    if arguments.expected_parent_pid != os.getppid():
        raise RuntimeError("Stage 10 probe parent PID differs from the supervisor")
    _set_frozen_locale()
    lock = capture_stage10_environment_lock(_project_root())
    write_bytes_fail_safe(arguments.output, _canonical_pretty(lock.to_dict()))
    receipt = {
        "schema_version": "10.0.0",
        "probe_process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "nonce_sha256": hashlib.sha256(arguments.nonce.encode()).hexdigest(),
        "environment_lock_sha256": lock.sha256(),
        "dataset_root_probed": False,
        "real_data_read": False,
    }
    print(json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    if arguments.expected_parent_pid != os.getppid():
        raise RuntimeError("Stage 10 runner parent PID differs from the supervisor")
    if os.environ.get("RETINAENTAIL_STAGE10_RUNNER_NONCE") != arguments.runner_nonce:
        raise RuntimeError("Stage 10 runner nonce was not process-bound")
    _set_frozen_locale()
    config = load_stage10_toy_config(arguments.config.resolve(strict=True))
    if arguments.runner_id not in config.runner_ids:
        raise RuntimeError("Stage 10 runner ID is outside preregistration")
    expected_lock = parse_stage10_environment_lock(
        json.loads(arguments.expected_environment_lock.read_text(encoding="utf-8"))
    )
    observed_lock = capture_stage10_environment_lock(_project_root())
    if observed_lock != expected_lock:
        raise RuntimeError("Stage 10 runner environment differs from the preregistered probe")
    source = arguments.source_stage9_package.resolve(strict=True)
    source_verified = verify_stage9_toy_package(source)
    if source_verified.package_sha256 != config.source_stage9_package_sha256:
        raise RuntimeError("Stage 10 runner source Stage 9 package drifted")
    source_stage7 = source / "session_a" / "stage8" / "stage7" / "package"
    output = arguments.output.resolve(strict=False)
    stage9_output = output / "stage9"
    result = run_stage9_toy_fixture(
        _project_root() / "configs" / "stage9_toy_independent_session_diff.yaml",
        source_stage7,
        stage9_output,
    )
    verified = verify_stage9_toy_package(stage9_output)
    if result != verified or verified.status != "SUCCESS":
        raise RuntimeError("Stage 10 runner Stage 9 package replay failed")
    nonce_sha = hashlib.sha256(arguments.runner_nonce.encode()).hexdigest()
    envelope = create_synthetic_signed_envelope(
        config=config,
        environment_lock=observed_lock,
        runner_id=arguments.runner_id,
        producer_process_id=os.getpid(),
        session_nonce_sha256=nonce_sha,
        stage9_package=stage9_output,
    )
    write_bytes_fail_safe(output / "envelope.json", _canonical_pretty(envelope))
    receipt = {
        "schema_version": "10.0.0",
        "runner_id": arguments.runner_id,
        "producer_process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "session_nonce_sha256": nonce_sha,
        "environment_lock_sha256": observed_lock.sha256(),
        "stage9_package_sha256": verified.package_sha256,
        "stage9_verified": True,
        "stage9_status": verified.status,
        "signed_envelope_created": True,
        "network_used": False,
        "gpu_used": False,
        "real_data_read": False,
        "dataset_root_probed": False,
        "detached_receipt_authoritative": False,
    }
    print(json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "probe":
        return _probe(arguments)
    return _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
