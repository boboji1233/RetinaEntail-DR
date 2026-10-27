"""Isolated Stage 12 random-key signer worker for the local fixture."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import cast

from retinaentail.reproducibility.stage12 import (
    STAGE12_SCOPE,
    InMemoryEd25519Signer,
    build_ephemeral_certificate,
    build_ephemeral_trust_anchor,
    build_local_status_staple,
    build_recovery_proof,
    build_signed_envelope,
    canonical_sha256,
)

_REQUEST_FIELDS = {
    "schema_version",
    "runner_id",
    "trusted_time_epoch",
    "compromise_epoch",
    "recovery_epoch",
    "frozen_payload",
}


def run_worker(request_path: Path, output_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        raise ValueError("Stage 12 worker request is malformed")
    if request["schema_version"] != "12.0.0" or not isinstance(request["runner_id"], str):
        raise ValueError("Stage 12 worker request policy is invalid")
    frozen = request["frozen_payload"]
    if not isinstance(frozen, dict):
        raise ValueError("Stage 12 frozen payload is invalid")
    runner_id = request["runner_id"]
    trusted_time = int(cast(int, request["trusted_time_epoch"]))
    compromise_epoch = int(cast(int, request["compromise_epoch"]))
    recovery_epoch = int(cast(int, request["recovery_epoch"]))

    root = InMemoryEd25519Signer()
    compromised = InMemoryEd25519Signer()
    recovered = InMemoryEd25519Signer()
    trust_anchor = build_ephemeral_trust_anchor(
        runner_id=runner_id,
        trusted_time_epoch=trusted_time,
        root_signer=root,
    )
    generation1 = build_ephemeral_certificate(
        runner_id=runner_id,
        generation=1,
        signer=compromised,
        issuer=root,
        previous_key_id=None,
        trusted_time_epoch=trusted_time,
    )
    generation2 = build_ephemeral_certificate(
        runner_id=runner_id,
        generation=2,
        signer=recovered,
        issuer=root,
        previous_key_id=compromised.key_id,
        trusted_time_epoch=trusted_time,
    )
    recovery_proof = build_recovery_proof(
        runner_id=runner_id,
        old_certificate=generation1,
        new_certificate=generation2,
        root_signer=root,
        recovered_signer=recovered,
        compromise_epoch=compromise_epoch,
        recovery_epoch=recovery_epoch,
        source_stage11_package_sha256=cast(str, frozen["source_stage11_package_sha256"]),
        frozen_contract_sha256=cast(str, frozen["frozen_contract_sha256"]),
    )
    status_staple = build_local_status_staple(
        runner_id=runner_id,
        old_certificate=generation1,
        new_certificate=generation2,
        root_signer=root,
        compromise_epoch=compromise_epoch,
        trusted_time_epoch=trusted_time,
    )
    process_id = os.getpid()
    session_nonce = secrets.token_hex(32)
    pre_payload = {
        **frozen,
        "schema_version": "12.0.0",
        "runner_id": runner_id,
        "producer_process_id": process_id,
        "session_nonce_sha256": canonical_sha256(
            {"nonce": session_nonce, "runner": runner_id, "phase": "pre"}
        ),
        "phase": "PRE_COMPROMISE",
        "signing_generation": 1,
        "scope": STAGE12_SCOPE,
        "split": "test",
        "payload_contains_data": False,
        "payload_contains_root": False,
        "payload_contains_private_key": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "production_authentication": False,
    }
    recovered_payload = {
        **pre_payload,
        "session_nonce_sha256": canonical_sha256(
            {"nonce": session_nonce, "runner": runner_id, "phase": "recovered"}
        ),
        "phase": "RECOVERED",
        "signing_generation": 2,
    }
    bundle = {
        "trust_anchor": trust_anchor,
        "generation1_certificate": generation1,
        "generation2_certificate": generation2,
        "recovery_proof": recovery_proof,
        "status_staple": status_staple,
        "pre_compromise_envelope": build_signed_envelope(
            pre_payload, certificate=generation1, signer=compromised
        ),
        "recovered_envelope": build_signed_envelope(
            recovered_payload, certificate=generation2, signer=recovered
        ),
        "receipt": {
            "schema_version": "12.0.0",
            "runner_id": runner_id,
            "producer_process_id": process_id,
            "signer_adapter": "InMemoryEd25519Signer",
            "random_private_keys_generated": 3,
            "private_key_exported": False,
            "private_key_serialized": False,
            "private_key_written_to_artifact": False,
            "hsm_backed": False,
            "production_authentication": False,
            "physical_host_attested": False,
            "network_used": False,
            "gpu_used": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Stage 12 worker output: {output_path}")
    output_path.write_text(
        json.dumps(bundle, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_worker(arguments.request, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
