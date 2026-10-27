"""Internal Stage 9 synthetic producer executed in a fresh Python process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from retinaentail.pilot.stage8_fixture import run_stage8_toy_fixture
from retinaentail.reproducibility.stage9 import capture_stage9_environment_lock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retinaentail-stage9-worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage7-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-nonce", required=True)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if os.getppid() != arguments.expected_parent_pid:
        raise RuntimeError("Stage 9 worker parent process identity changed")
    if arguments.session_id not in {"session_a", "session_b"}:
        raise ValueError("Stage 9 worker session ID is invalid")
    nonce = arguments.session_nonce
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(char not in "0123456789abcdef" for char in nonce)
    ):
        raise ValueError("Stage 9 worker nonce is invalid")
    environment_lock = capture_stage9_environment_lock(python_hash_seed="0")
    result = run_stage8_toy_fixture(
        arguments.config,
        arguments.stage7_package,
        arguments.output,
    )
    receipt = {
        "schema_version": "9.0.0",
        "session_id": arguments.session_id,
        "producer_process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "session_nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "environment_lock": environment_lock.to_dict(),
        "environment_lock_sha256": environment_lock.sha256(),
        "stage8_package_sha256": result.package_sha256,
        "stage8_package_file_count": result.package_file_count,
        "stage8_status": result.status,
        "stage8_verified": result.verified,
        "network_used": False,
        "gpu_used": False,
        "real_data_read": False,
        "detached_receipt_authoritative": False,
    }
    print(json.dumps(receipt, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
