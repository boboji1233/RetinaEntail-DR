from __future__ import annotations

import copy
from pathlib import Path

import pytest

import retinaentail.pilot.stage10_fixture as fixture
from retinaentail.entailment.states import EntailmentState
from retinaentail.reproducibility.stage9 import Stage9Observation
from retinaentail.reproducibility.stage10 import STAGE10_THREAD_KEYS, Stage10EnvironmentLock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stage10_toy_runner_portability.yaml"


def _environment() -> Stage10EnvironmentLock:
    return Stage10EnvironmentLock(
        python_implementation="CPython",
        python_version="3.11.0",
        python_executable_sha256="1" * 64,
        pyproject_sha256="2" * 64,
        platform_system="Windows",
        platform_release="11",
        platform_machine="AMD64",
        platform_processor="AMD64",
        project_version="0.1.0",
        dependency_versions=(("Pillow", "1"), ("PyYAML", "1"), ("numpy", "1")),
        blas_configuration_sha256="3" * 64,
        thread_policy=tuple((name, "1") for name in STAGE10_THREAD_KEYS),
        locale_name="C",
        preferred_encoding="utf-8",
        filesystem_encoding="utf-8",
        python_hash_seed="0",
        no_user_site=True,
        dont_write_bytecode=True,
        utf8_mode=1,
        optimize=0,
    )


def _observation() -> Stage9Observation:
    return Stage9Observation(
        case_id="case",
        run_id="run",
        state=EntailmentState.OUT_OF_MODEL,
        positive_score=0.0,
        negative_score=0.0,
        fail_closed=True,
        out_of_distribution=True,
        positive_witness_sha256="4" * 64,
        negative_witness_sha256="5" * 64,
        positive_witness_replay_valid=False,
        negative_witness_replay_valid=False,
        canonical_sha256="6" * 64,
    )


def test_synthetic_envelope_contains_no_path_and_tamper_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = fixture.load_stage10_toy_config(CONFIG)
    material: dict[str, object] = {
        "package_sha256": "7" * 64,
        "package_file_count": 10,
        "package_manifest_sha256": "8" * 64,
        "audit_sha256": "9" * 64,
        "preregistration_sha256": config.stage9_preregistration_sha256,
        "source_stage7_package_sha256": config.source_stage7_package_sha256,
        "source_stage8_package_sha256": config.source_stage8_package_sha256,
        "deterministic_projection_sha256": "a" * 64,
        "observations": (_observation(),),
    }
    monkeypatch.setattr(fixture, "_stage9_material", lambda path, cfg: material)
    envelope = fixture.create_synthetic_signed_envelope(
        config=config,
        environment_lock=_environment(),
        runner_id="runner_a",
        producer_process_id=123,
        session_nonce_sha256="b" * 64,
        stage9_package=Path("not-read"),
    )
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    assert payload["payload_contains_data"] is False
    assert payload["payload_contains_root"] is False
    assert not any(
        "path" in key or "root" in key
        for key in payload
        if key not in {"payload_contains_root", "authorizes_physical_root"}
    )
    fixture._verify_envelope(
        envelope,
        config=config,
        runner_id="runner_a",
        expected_payload=payload,
    )
    tampered = copy.deepcopy(envelope)
    tampered["integrity_tag"] = "0" * 64
    with pytest.raises(ValueError, match="HMAC"):
        fixture._verify_envelope(
            tampered,
            config=config,
            runner_id="runner_a",
            expected_payload=payload,
        )


def test_unregistered_artifact_difference_is_deterministic_mismatch() -> None:
    diff = fixture._artifact_diff(
        {"deterministic.json": b"a", "audit.json": b"a"},
        {"deterministic.json": b"b", "audit.json": b"b"},
        allowed_paths=("audit.json",),
    )
    assert diff["allowed_provenance_difference_count"] == 1
    assert diff["deterministic_file_mismatch_count"] == 1
