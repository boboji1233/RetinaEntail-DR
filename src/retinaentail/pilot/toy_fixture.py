"""CPU/offline Stage 5 synthetic locked small-pilot package.

The fixture consumes one exact, independently verified Stage 4 toy package.  It
never reads a medical-data root, never estimates a threshold, and never invokes a
calibration or training routine.  Verification checks the exact outer file set,
the exact embedded Stage 4 package digest, the full Stage 4 replay verifier, five
identity classes, deterministic regeneration, and byte equality for every file.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml

from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.toy_fixture import verify_stage4_toy_package
from retinaentail.pilot.identity import (
    Stage5IdentityAuditStatus,
    Stage5IdentityRecord,
    audit_five_identity_split_isolation,
)
from retinaentail.reporting.stage5 import (
    STAGE5_AUTHORITY_BOUNDARY,
    STAGE5_SOURCE_DIGEST_METHOD,
    issue_stage5_pilot_report,
    render_stage5_pilot_html,
    render_stage5_pilot_json,
)
from retinaentail.witnesses.replay import sha256_bytes

STAGE5_TOY_CONFIG_VERSION: Final = "5.0"
STAGE5_TOY_PACKAGE_VERSION: Final = "5.0.0"
STAGE5_PILOT_METHOD: Final = "verified_stage4_locked_small_pilot_v1"
_MAX_STAGE5_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")

_CASE_IDS: Final = (
    "stage4-toy-entailed-present",
    "stage4-toy-entailed-absent",
    "stage4-toy-raw-unidentifiable",
    "stage4-toy-infeasible-out-of-model",
    "stage4-toy-domain-out-of-model",
    "stage4-toy-unknown-degradation",
    "stage4-toy-invalid-witness",
)
_CASE_FILES: Final = frozenset(
    {
        "case.json",
        "certificate.json",
        "dual_replay.json",
        "evaluation.json",
        "images/lesion_absent.png",
        "images/lesion_present.png",
        "images/observation.png",
        "witnesses/negative.json",
        "witnesses/positive.json",
    }
)
_STAGE3_FILES: Final = frozenset(
    {
        "calibration_artifact.json",
        "calibration_samples.json",
        "certificate.json",
        "config_snapshot.json",
        "dual_replay.json",
        "evaluation.json",
        "fail_closed/missing_threshold.json",
        "fail_closed/unknown_degradation.json",
        "images/lesion_absent.png",
        "images/lesion_present.png",
        "images/observation.png",
        "model_provenance.json",
        "summary.json",
        "witnesses/negative.json",
        "witnesses/positive.json",
    }
)


def _stage4_files() -> frozenset[str]:
    names = {
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "identity_isolation.json",
        "locked_calibration.json",
        "locked_calibration_artifact.json",
        "locked_calibration_summary.json",
        "model_provenance.json",
        "summary.json",
        "test_data.json",
    }
    names.update(f"stage3_lock/package/{name}" for name in _STAGE3_FILES)
    names.update(f"cases/{case_id}/{name}" for case_id in _CASE_IDS for name in _CASE_FILES)
    return frozenset(names)


_STAGE4_FILES: Final = _stage4_files()
_STAGE5_OWN_FILES: Final = frozenset(
    {
        "audit.html",
        "audit.json",
        "case_index.json",
        "config_snapshot.json",
        "identity_isolation.json",
        "locked_contract.json",
        "model_provenance.json",
        "pilot_data.json",
        "source_stage4_package.json",
        "summary.json",
    }
)
_STAGE5_FILES: Final = frozenset(
    {*_STAGE5_OWN_FILES, *(f"stage4_lock/package/{name}" for name in _STAGE4_FILES)}
)

_EXPECTED_SOURCE: Final[dict[str, object]] = {
    "package_schema_version": "4.0.0",
    "run_id": "stage4-toy-locked-test-001",
    "package_sha256": "bfb7b22393bf6bafd6944a6ef091fa2464d70b95a344ad5d78cbd0fb3d1713df",
    "package_file_count": 88,
    "stage4_config_sha256": "ea417983d85b9564d4ec8b217966dc31d734734fe074b12beb48c34cfe933680",
    "test_plan_sha256": "e1c667954725800f8a5a2d0e012313ccca738247678d26f9a6400aadcf927094",
    "test_data_sha256": "c10341f6f466c2222587568ffdd2715d7460b23e748b79683c9d47f2f73a20d1",
    "stage4_model_provenance_sha256": (
        "213dae8b86972b551b60042e69e2ec2551f34947c2bab61298c412d9ebadf23b"
    ),
    "calibration_artifact_sha256": (
        "c84762f2d44d345cd39732e1ce0f5a44003e0a635e482f28d3f8abcd6395e78e"
    ),
    "calibration_config_sha256": (
        "0ea540363d0a768bc1964c007f6a37a393f1b29b3c9a312ed92bd59a76bbab52"
    ),
    "calibration_data_sha256": ("ab4fa68cb4228df6a1db23caaeacb8d1c35cca6ec7da220e600a8bdcbc067ecd"),
    "calibration_model_sha256": (
        "ca2fa473841f6b79d8b327525fb14de0e099ac8d342538a5031d9a54783b81c1"
    ),
    "alpha": 0.25,
    "positive_threshold": 1.2299400515471258e-05,
    "negative_threshold": 1.2299400515471254e-05,
    "positive_sample_count": 4,
    "negative_sample_count": 4,
    "total_sample_count": 8,
}


class Stage5ToyConfigurationError(ValueError):
    """Raised when the preregistered synthetic pilot declaration is not exact."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5SourceLock:
    package_schema_version: str
    run_id: str
    package_sha256: str
    package_file_count: int
    stage4_config_sha256: str
    test_plan_sha256: str
    test_data_sha256: str
    stage4_model_provenance_sha256: str
    calibration_artifact_sha256: str
    calibration_config_sha256: str
    calibration_data_sha256: str
    calibration_model_sha256: str
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    total_sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _EXPECTED_SOURCE}

    def __post_init__(self) -> None:
        if self.to_dict() != _EXPECTED_SOURCE:
            raise Stage5ToyConfigurationError("source Stage 4 lock does not match fixture v5.0")


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage5ToyConfig:
    schema_version: str
    pilot_id: str
    source_stage4: Stage5SourceLock
    confidence_level: float
    train_identity_control_count: int
    pilot_case_ids: tuple[str, ...]
    threshold_policy: str
    test_tuning_prohibited: bool
    real_data_access: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "source_stage4": self.source_stage4.to_dict(),
            "confidence_level": self.confidence_level,
            "train_identity_control_count": self.train_identity_control_count,
            "pilot_case_ids": list(self.pilot_case_ids),
            "threshold_policy": self.threshold_policy,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "real_data_access": self.real_data_access,
        }


@dataclass(frozen=True, slots=True)
class Stage5ToyRunResult:
    """Convenience receipt, never an authority or data-access capability.

    ``verified`` records that the verifier completed during this call.  Consumers
    must re-run :func:`verify_stage5_toy_package` on the complete package at the
    point of use; constructing or retaining this value proves nothing by itself.
    """

    output_dir: Path
    case_count: int
    states: tuple[EntailmentState, ...]
    source_stage4_package_sha256: str
    pilot_data_sha256: str
    verified: bool


@dataclass(frozen=True, slots=True)
class _PreparedStage5:
    config: Stage5ToyConfig
    config_sha256: str
    source_descriptor: dict[str, object]
    pilot_data_sha256: str
    model_provenance_sha256: str
    states: tuple[EntailmentState, ...]
    artifacts: dict[str, bytes]


def parse_stage5_toy_config(value: object) -> Stage5ToyConfig:
    data = _strict_mapping(
        value,
        "Stage 5 toy config",
        {
            "schema_version",
            "pilot_id",
            "source_stage4",
            "confidence_level",
            "train_identity_control_count",
            "pilot_case_ids",
            "threshold_policy",
            "test_tuning_prohibited",
            "real_data_access",
        },
    )
    if data["schema_version"] != STAGE5_TOY_CONFIG_VERSION:
        raise Stage5ToyConfigurationError("unsupported Stage 5 toy config schema_version")
    source_data = _strict_mapping(data["source_stage4"], "source_stage4", set(_EXPECTED_SOURCE))
    source = Stage5SourceLock(
        package_schema_version=_string(
            source_data["package_schema_version"], "source_stage4.package_schema_version"
        ),
        run_id=_string(source_data["run_id"], "source_stage4.run_id"),
        package_sha256=_digest(source_data["package_sha256"], "source_stage4.package_sha256"),
        package_file_count=_integer(
            source_data["package_file_count"], "source_stage4.package_file_count", minimum=1
        ),
        stage4_config_sha256=_digest(
            source_data["stage4_config_sha256"], "source_stage4.stage4_config_sha256"
        ),
        test_plan_sha256=_digest(source_data["test_plan_sha256"], "source_stage4.test_plan_sha256"),
        test_data_sha256=_digest(source_data["test_data_sha256"], "source_stage4.test_data_sha256"),
        stage4_model_provenance_sha256=_digest(
            source_data["stage4_model_provenance_sha256"],
            "source_stage4.stage4_model_provenance_sha256",
        ),
        calibration_artifact_sha256=_digest(
            source_data["calibration_artifact_sha256"],
            "source_stage4.calibration_artifact_sha256",
        ),
        calibration_config_sha256=_digest(
            source_data["calibration_config_sha256"],
            "source_stage4.calibration_config_sha256",
        ),
        calibration_data_sha256=_digest(
            source_data["calibration_data_sha256"],
            "source_stage4.calibration_data_sha256",
        ),
        calibration_model_sha256=_digest(
            source_data["calibration_model_sha256"],
            "source_stage4.calibration_model_sha256",
        ),
        alpha=_number(source_data["alpha"], "source_stage4.alpha", minimum=0.0),
        positive_threshold=_number(
            source_data["positive_threshold"],
            "source_stage4.positive_threshold",
            minimum=0.0,
        ),
        negative_threshold=_number(
            source_data["negative_threshold"],
            "source_stage4.negative_threshold",
            minimum=0.0,
        ),
        positive_sample_count=_integer(
            source_data["positive_sample_count"],
            "source_stage4.positive_sample_count",
            minimum=1,
        ),
        negative_sample_count=_integer(
            source_data["negative_sample_count"],
            "source_stage4.negative_sample_count",
            minimum=1,
        ),
        total_sample_count=_integer(
            source_data["total_sample_count"],
            "source_stage4.total_sample_count",
            minimum=2,
        ),
    )
    raw_cases = data["pilot_case_ids"]
    if isinstance(raw_cases, str | bytes) or not isinstance(raw_cases, Sequence):
        raise Stage5ToyConfigurationError("pilot_case_ids must be a sequence")
    cases = tuple(_string(item, f"pilot_case_ids[{index}]") for index, item in enumerate(raw_cases))
    config = Stage5ToyConfig(
        schema_version=STAGE5_TOY_CONFIG_VERSION,
        pilot_id=_string(data["pilot_id"], "pilot_id"),
        source_stage4=source,
        confidence_level=_number(data["confidence_level"], "confidence_level", minimum=0.0),
        train_identity_control_count=_integer(
            data["train_identity_control_count"], "train_identity_control_count", minimum=1
        ),
        pilot_case_ids=cases,
        threshold_policy=_string(data["threshold_policy"], "threshold_policy"),
        test_tuning_prohibited=_boolean(data["test_tuning_prohibited"], "test_tuning_prohibited"),
        real_data_access=_boolean(data["real_data_access"], "real_data_access"),
    )
    if (
        config.pilot_id != "stage5-toy-locked-pilot-001"
        or config.confidence_level != 0.95
        or config.train_identity_control_count != 2
        or config.pilot_case_ids != _CASE_IDS
        or config.threshold_policy != "LOCKED_STAGE3_NO_REESTIMATION"
        or config.test_tuning_prohibited is not True
        or config.real_data_access is not False
    ):
        raise Stage5ToyConfigurationError("Stage 5 pilot declaration is not the locked v5.0 plan")
    return config


def load_stage5_toy_config(path: Path) -> Stage5ToyConfig:
    if not isinstance(path, Path):
        raise TypeError("config path must be a pathlib.Path")
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Stage5ToyConfigurationError("Stage 5 toy configuration cannot be loaded") from error
    return parse_stage5_toy_config(value)


def stage4_package_descriptor(files: Mapping[str, bytes]) -> dict[str, object]:
    """Build the canonical, content-addressed descriptor for an exact Stage 4 package."""

    if set(files) != _STAGE4_FILES:
        raise ValueError("source Stage 4 package file set is incomplete or contains unknown files")
    entries = [
        {
            "relative_path": name,
            "size_bytes": len(files[name]),
            "sha256": sha256_bytes(files[name]),
        }
        for name in sorted(files)
    ]
    digest_payload: dict[str, object] = {
        "digest_method": STAGE5_SOURCE_DIGEST_METHOD,
        "file_count": len(entries),
        "files": entries,
    }
    return {
        "schema_version": STAGE5_TOY_PACKAGE_VERSION,
        "source_stage": "STAGE4",
        "package_sha256": sha256_bytes(_canonical_json_bytes(digest_payload)),
        **digest_payload,
    }


def _locked_contract(source: Stage5SourceLock) -> dict[str, object]:
    return {
        "alpha": source.alpha,
        "positive_threshold": source.positive_threshold,
        "negative_threshold": source.negative_threshold,
        "positive_sample_count": source.positive_sample_count,
        "negative_sample_count": source.negative_sample_count,
        "total_sample_count": source.total_sample_count,
        "calibration_artifact_hash": source.calibration_artifact_sha256,
        "config_hash": source.calibration_config_sha256,
        "data_hash": source.calibration_data_sha256,
        "model_hash": source.calibration_model_sha256,
    }


def _identity_records(
    source_files: Mapping[str, bytes],
    config: Stage5ToyConfig,
) -> tuple[Stage5IdentityRecord, ...]:
    dataset_id = "retinaentail_stage5_toy"
    records: list[Stage5IdentityRecord] = []
    for index in range(config.train_identity_control_count):
        records.append(
            Stage5IdentityRecord(
                dataset_id=dataset_id,
                relative_path=f"identity-controls/train/control-{index:03d}.synthetic",
                split="train",
                source_image_id=f"stage5-train-source-{index:03d}",
                patient_id=f"stage5-train-patient-{index:03d}",
                eye_id=f"stage5-train-eye-{index:03d}",
                followup_id=f"stage5-train-followup-{index:03d}",
                intervention_pair_id=f"stage5-train-pair-{index:03d}",
            )
        )
    calibration = _mapping_json(
        source_files["stage3_lock/package/calibration_samples.json"],
        "Stage 3 calibration samples",
    )
    calibration_records = _sequence(calibration.get("records"), "calibration records")
    for index, raw in enumerate(calibration_records):
        outer = _mapping(raw, f"calibration records[{index}]")
        sample = _mapping(outer.get("sample"), f"calibration records[{index}].sample")
        sample_id = _string(sample.get("sample_id"), f"calibration records[{index}].sample_id")
        patient = _string(sample.get("patient_id"), f"calibration records[{index}].patient_id")
        records.append(
            Stage5IdentityRecord(
                dataset_id=dataset_id,
                relative_path=f"stage3/calibration/{sample_id}.synthetic",
                split="calibration",
                source_image_id=_string(
                    sample.get("source_image_id"),
                    f"calibration records[{index}].source_image_id",
                ),
                patient_id=patient,
                eye_id=f"{patient}-eye-R",
                followup_id=f"{patient}-eye-R-baseline",
                intervention_pair_id=_string(
                    sample.get("intervention_pair_id"),
                    f"calibration records[{index}].intervention_pair_id",
                ),
            )
        )
    test_data = _mapping_json(source_files["test_data.json"], "Stage 4 test data")
    test_records = _sequence(test_data.get("records"), "test records")
    for index, raw in enumerate(test_records):
        record = _mapping(raw, f"test records[{index}]")
        case_id = _string(record.get("case_id"), f"test records[{index}].case_id")
        patient = _string(record.get("patient_id"), f"test records[{index}].patient_id")
        records.append(
            Stage5IdentityRecord(
                dataset_id=dataset_id,
                relative_path=f"stage4/test/{case_id}.synthetic",
                split="test",
                source_image_id=_string(
                    record.get("source_image_id"), f"test records[{index}].source_image_id"
                ),
                patient_id=patient,
                eye_id=f"{patient}-eye-R",
                followup_id=f"{patient}-eye-R-baseline",
                intervention_pair_id=_string(
                    record.get("intervention_pair_id"),
                    f"test records[{index}].intervention_pair_id",
                ),
            )
        )
    return tuple(records)


def _case_index(source_files: Mapping[str, bytes]) -> tuple[dict[str, object], ...]:
    test_data = _mapping_json(source_files["test_data.json"], "Stage 4 test data")
    raw_records = _sequence(test_data.get("records"), "Stage 4 test records")
    by_case: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(raw_records):
        record = _mapping(raw, f"Stage 4 test records[{index}]")
        case_id = _string(record.get("case_id"), f"Stage 4 test records[{index}].case_id")
        by_case[case_id] = record
    if tuple(by_case) != _CASE_IDS:
        raise ValueError("Stage 4 test case membership or order differs from the locked pilot")
    result: list[dict[str, object]] = []
    for index, case_id in enumerate(_CASE_IDS):
        prefix = f"cases/{case_id}/"
        evaluation = _mapping_json(source_files[prefix + "evaluation.json"], "evaluation")
        replay = _mapping_json(source_files[prefix + "dual_replay.json"], "dual replay")
        positive = _mapping(replay.get("positive"), "positive replay")
        negative = _mapping(replay.get("negative"), "negative replay")
        state = _string(evaluation.get("state"), "evaluation.state")
        if state not in {item.value for item in EntailmentState}:
            raise ValueError("Stage 4 case has an unknown entailment state")
        observation_mode = by_case[case_id].get("observation_mode")
        reference_label = (
            "LESION_PRESENT"
            if observation_mode == "LESION_PRESENT"
            else "LESION_ABSENT"
            if observation_mode == "LESION_ABSENT"
            else "UNKNOWN"
        )
        result.append(
            {
                "pilot_index": index,
                "case_id": case_id,
                "reference_label": reference_label,
                "state": state,
                "status": _string(evaluation.get("status"), "evaluation.status"),
                "failure_reasons": list(_sequence(evaluation.get("failure_reasons"), "reasons")),
                "dual_witness_replay_passed": _boolean(replay.get("passed"), "replay.passed"),
                "positive_witness_replay_passed": _boolean(
                    positive.get("passed"), "positive replay.passed"
                ),
                "negative_witness_replay_passed": _boolean(
                    negative.get("passed"), "negative replay.passed"
                ),
                "evaluation_sha256": sha256_bytes(source_files[prefix + "evaluation.json"]),
                "certificate_sha256": sha256_bytes(source_files[prefix + "certificate.json"]),
                "dual_replay_sha256": sha256_bytes(source_files[prefix + "dual_replay.json"]),
                "positive_witness_sha256": sha256_bytes(
                    source_files[prefix + "witnesses/positive.json"]
                ),
                "negative_witness_sha256": sha256_bytes(
                    source_files[prefix + "witnesses/negative.json"]
                ),
            }
        )
    return tuple(result)


def _pilot_data(
    config: Stage5ToyConfig,
    descriptor: Mapping[str, object],
    records: tuple[Stage5IdentityRecord, ...],
    case_index: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "schema_version": STAGE5_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "pilot_id": config.pilot_id,
        "source_stage4_package_sha256": descriptor["package_sha256"],
        "split": "test",
        "threshold_policy": config.threshold_policy,
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "missing_label_policy": "UNKNOWN_NEVER_NEGATIVE",
        "identity_records_are_synthetic_metadata_controls": True,
        "identity_records": [record.to_dict() for record in records],
        "pilot_cases": list(case_index),
    }


def _model_provenance() -> tuple[str, dict[str, object]]:
    package_root = Path(__file__).resolve().parents[1]
    components = {
        "stage4_fixture": package_root / "evaluation" / "toy_fixture.py",
        "stage5_fixture": package_root / "pilot" / "toy_fixture.py",
        "stage5_identity": package_root / "pilot" / "identity.py",
        "stage5_reporting": package_root / "reporting" / "stage5.py",
    }
    payload: dict[str, object] = {
        "schema_version": STAGE5_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "method": STAGE5_PILOT_METHOD,
        "components": [
            {
                "component": name,
                "sha256": sha256_bytes(
                    path.read_text(encoding="utf-8")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .encode("utf-8")
                ),
            }
            for name, path in sorted(components.items())
        ],
    }
    return sha256_bytes(_canonical_json_bytes(payload)), payload


def _validate_source_bindings(
    config: Stage5ToyConfig,
    descriptor: Mapping[str, object],
    source_files: Mapping[str, bytes],
) -> None:
    source = config.source_stage4
    if (
        descriptor.get("package_sha256") != source.package_sha256
        or descriptor.get("file_count") != source.package_file_count
    ):
        raise ValueError("source Stage 4 package digest or file count does not match the lock")
    summary = _mapping_json(source_files["summary.json"], "Stage 4 summary")
    config_snapshot = _mapping_json(source_files["config_snapshot.json"], "Stage 4 config snapshot")
    required = {
        "schema_version": source.package_schema_version,
        "run_id": source.run_id,
        "stage4_config_sha256": source.stage4_config_sha256,
        "test_plan_sha256": source.test_plan_sha256,
        "test_data_sha256": source.test_data_sha256,
        "stage4_model_provenance_sha256": source.stage4_model_provenance_sha256,
        "calibration_artifact_sha256": source.calibration_artifact_sha256,
        "calibration_config_sha256": source.calibration_config_sha256,
        "calibration_data_sha256": source.calibration_data_sha256,
        "calibration_model_sha256": source.calibration_model_sha256,
        "alpha": source.alpha,
        "positive_threshold": source.positive_threshold,
        "negative_threshold": source.negative_threshold,
        "positive_calibration_sample_count": source.positive_sample_count,
        "negative_calibration_sample_count": source.negative_sample_count,
    }
    if any(summary.get(name) != expected for name, expected in required.items()):
        raise ValueError("Stage 4 summary provenance does not match the Stage 5 source lock")
    if config_snapshot.get("run_id") != source.run_id:
        raise ValueError("Stage 4 config run_id does not match the Stage 5 source lock")
    source_flags = {
        "thresholds_locked_before_test": True,
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "real_data_used": False,
        "real_data_read_count": 0,
        "clinical_claim": False,
        "authority_boundary": "COMPLETE_VERIFIED_STAGE4_PACKAGE",
    }
    if any(summary.get(name) != expected for name, expected in source_flags.items()):
        raise ValueError("Stage 4 package violates locked-test safety flags")


def _prepare(config: Stage5ToyConfig, source_files: Mapping[str, bytes]) -> _PreparedStage5:
    descriptor = stage4_package_descriptor(source_files)
    _validate_source_bindings(config, descriptor, source_files)
    identity_records = _identity_records(source_files, config)
    identity_audit = audit_five_identity_split_isolation(identity_records)
    if (
        identity_audit.status is not Stage5IdentityAuditStatus.PASS
        or not identity_audit.isolation_proven
    ):
        raise ValueError("synthetic five-identity split isolation failed closed")
    case_index = _case_index(source_files)
    states = tuple(EntailmentState(cast(str, item["state"])) for item in case_index)
    if set(states) != set(EntailmentState):
        raise ValueError("Stage 5 synthetic pilot must exercise all four entailment states")
    pilot_data = _pilot_data(config, descriptor, identity_records, case_index)
    pilot_data_sha256 = sha256_bytes(_canonical_json_bytes(pilot_data))
    config_sha256 = sha256_bytes(_canonical_json_bytes(config.to_dict()))
    model_sha256, model_provenance = _model_provenance()
    evaluation = _mapping_json(source_files["audit.json"], "Stage 4 evaluation")
    report = issue_stage5_pilot_report(
        pilot_id=config.pilot_id,
        source_stage4_package_sha256=cast(str, descriptor["package_sha256"]),
        source_stage4_file_count=cast(int, descriptor["file_count"]),
        stage5_config_sha256=config_sha256,
        pilot_data_sha256=pilot_data_sha256,
        stage5_model_provenance_sha256=model_sha256,
        locked_contract=_locked_contract(config.source_stage4),
        identity_isolation=identity_audit.to_dict(),
        identity_records=identity_records,
        evaluation=evaluation,
        case_index=case_index,
    )
    summary = {
        "schema_version": STAGE5_TOY_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "pilot_id": config.pilot_id,
        "method": STAGE5_PILOT_METHOD,
        "source_stage4_package_sha256": descriptor["package_sha256"],
        "source_stage4_file_count": descriptor["file_count"],
        "stage5_config_sha256": config_sha256,
        "pilot_data_sha256": pilot_data_sha256,
        "stage5_model_provenance_sha256": model_sha256,
        "case_count": len(case_index),
        "state_counts": evaluation["state_counts"],
        "fail_closed_count": evaluation["fail_closed_count"],
        "out_of_distribution_count": evaluation["out_of_distribution_count"],
        "both_witness_replays_passed_count": evaluation["both_witness_replays_passed_count"],
        "positive_witness_replays_passed_count": sum(
            item["positive_witness_replay_passed"] is True for item in case_index
        ),
        "negative_witness_replays_passed_count": sum(
            item["negative_witness_replay_passed"] is True for item in case_index
        ),
        "five_identity_isolation_passed": True,
        "alpha": config.source_stage4.alpha,
        "positive_threshold": config.source_stage4.positive_threshold,
        "negative_threshold": config.source_stage4.negative_threshold,
        "thresholds_locked_before_pilot": True,
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "cpu_only": True,
        "gpu_used": False,
        "network_used": False,
        "real_data_used": False,
        "real_data_read_count": 0,
        "real_data_gate": "BLOCKED_NO_EXPLICIT_APPROVED_DATASET",
        "individual_report_authoritative": False,
        "authority_boundary": STAGE5_AUTHORITY_BOUNDARY,
        "clinical_claim": False,
    }
    artifacts = {
        "audit.html": render_stage5_pilot_html(report).encode("utf-8"),
        "audit.json": render_stage5_pilot_json(report).encode("utf-8"),
        "case_index.json": _canonical_json_bytes(
            {"schema_version": STAGE5_TOY_PACKAGE_VERSION, "cases": list(case_index)}
        ),
        "config_snapshot.json": _canonical_json_bytes(config.to_dict()),
        "identity_isolation.json": _canonical_json_bytes(identity_audit.to_dict()),
        "locked_contract.json": _canonical_json_bytes(_locked_contract(config.source_stage4)),
        "model_provenance.json": _canonical_json_bytes(model_provenance),
        "pilot_data.json": _canonical_json_bytes(pilot_data),
        "source_stage4_package.json": _canonical_json_bytes(descriptor),
        "summary.json": _canonical_json_bytes(summary),
    }
    return _PreparedStage5(
        config=config,
        config_sha256=config_sha256,
        source_descriptor=descriptor,
        pilot_data_sha256=pilot_data_sha256,
        model_provenance_sha256=model_sha256,
        states=states,
        artifacts=artifacts,
    )


def verify_stage5_toy_package(output_dir: Path) -> Stage5ToyRunResult:
    """Reverify embedded Stage 4, regenerate Stage 5, and byte-compare all artifacts."""

    if not isinstance(output_dir, Path) or not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    actual = _file_mapping(output_dir, expected=_STAGE5_FILES, stage="Stage 5")
    config = parse_stage5_toy_config(
        _decode_json(actual["config_snapshot.json"], "Stage 5 config snapshot")
    )
    embedded_root = output_dir / "stage4_lock" / "package"
    verified_stage4 = verify_stage4_toy_package(embedded_root)
    if not verified_stage4.verified:
        raise ValueError("embedded source Stage 4 package did not independently verify")
    source_files = {name: actual[f"stage4_lock/package/{name}"] for name in sorted(_STAGE4_FILES)}
    prepared = _prepare(config, source_files)
    expected = dict(prepared.artifacts)
    expected.update(
        {f"stage4_lock/package/{name}": source_files[name] for name in sorted(source_files)}
    )
    if set(expected) != _STAGE5_FILES:
        raise RuntimeError("internal Stage 5 package file contract is inconsistent")
    for name, payload in expected.items():
        if actual[name] != payload:
            raise ValueError(f"Stage 5 package artifact mismatch: {name}")
    return Stage5ToyRunResult(
        output_dir=output_dir,
        case_count=len(prepared.states),
        states=prepared.states,
        source_stage4_package_sha256=cast(str, prepared.source_descriptor["package_sha256"]),
        pilot_data_sha256=prepared.pilot_data_sha256,
        verified=True,
    )


def run_stage5_toy_fixture(
    config_path: Path,
    stage4_package: Path,
    output_dir: Path,
) -> Stage5ToyRunResult:
    """Publish a deterministic pilot only after the read-only Stage 4 source verifies."""

    if not isinstance(stage4_package, Path) or not isinstance(output_dir, Path):
        raise TypeError("stage4_package and output_dir must be pathlib.Path values")
    try:
        source_root = stage4_package.resolve(strict=True)
        target_root = output_dir.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Stage 4 source and Stage 5 output paths must be resolvable") from error
    if (
        source_root == target_root
        or source_root in target_root.parents
        or target_root in source_root.parents
    ):
        raise ValueError("Stage 5 output must not overlap the read-only Stage 4 source package")
    config = load_stage5_toy_config(config_path)
    verified_stage4 = verify_stage4_toy_package(stage4_package)
    if not verified_stage4.verified:
        raise ValueError("source Stage 4 package did not independently verify")
    source_files = _file_mapping(stage4_package, expected=_STAGE4_FILES, stage="Stage 4")
    prepared = _prepare(config, source_files)
    artifacts = dict(prepared.artifacts)
    artifacts.update(
        {f"stage4_lock/package/{name}": source_files[name] for name in sorted(source_files)}
    )
    if set(artifacts) != _STAGE5_FILES:
        raise RuntimeError("internal Stage 5 package file contract is inconsistent")
    from retinaentail.data.audit_reporting import AuditWriteBoundary
    from retinaentail.reporting.stage3 import write_stage3_bundle

    write_stage3_bundle(
        output_dir,
        artifacts,
        boundary=AuditWriteBoundary(allowed_root=output_dir),
    )
    return verify_stage5_toy_package(output_dir)


def _file_mapping(root: Path, *, expected: frozenset[str], stage: str) -> dict[str, bytes]:
    if root.is_symlink():
        raise ValueError(f"{stage} package root cannot be a symbolic link")
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"{stage} package cannot contain symbolic links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{stage} package contains an unsupported entry: {relative}")
        if relative not in expected:
            raise ValueError(f"{stage} package contains an unknown file: {relative}")
        discovered[relative] = path
    missing = sorted(expected.difference(discovered))
    if missing:
        raise ValueError(f"{stage} package files missing={missing!r}")
    return {name: _read_bounded(discovered[name], stage=stage) for name in sorted(expected)}


def _read_bounded(path: Path, *, stage: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{stage} package artifacts cannot be symbolic links")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"{stage} package artifact cannot be inspected") from error
    if size > _MAX_STAGE5_ARTIFACT_BYTES:
        raise ValueError(f"{stage} package artifact exceeds size limit: {path.name}")
    try:
        with path.open("rb") as stream:
            payload = stream.read(_MAX_STAGE5_ARTIFACT_BYTES + 1)
    except OSError as error:
        raise ValueError(f"{stage} package artifact cannot be read") from error
    if len(payload) > _MAX_STAGE5_ARTIFACT_BYTES:
        raise ValueError(f"{stage} package artifact exceeds size limit: {path.name}")
    return payload


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _decode_json(payload: bytes, name: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error


def _mapping_json(payload: bytes, name: str) -> dict[str, object]:
    return _mapping(_decode_json(payload, name), name)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return {cast(str, key): item for key, item in value.items()}


def _strict_mapping(value: object, name: str, required: set[str]) -> dict[str, object]:
    result = _mapping(value, name)
    missing = required.difference(result)
    unknown = set(result).difference(required)
    if missing or unknown:
        raise Stage5ToyConfigurationError(
            f"{name} missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    return result


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n\x00")
    ):
        raise Stage5ToyConfigurationError(f"{name} must be a canonical non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    result = _string(value, name)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise Stage5ToyConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage5ToyConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage5ToyConfigurationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise Stage5ToyConfigurationError(f"{name} must be finite and >= {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise Stage5ToyConfigurationError(f"{name} must be boolean")
    return value


__all__ = [
    "STAGE5_PILOT_METHOD",
    "STAGE5_TOY_CONFIG_VERSION",
    "STAGE5_TOY_PACKAGE_VERSION",
    "Stage5SourceLock",
    "Stage5ToyConfig",
    "Stage5ToyConfigurationError",
    "Stage5ToyRunResult",
    "load_stage5_toy_config",
    "parse_stage5_toy_config",
    "run_stage5_toy_fixture",
    "stage4_package_descriptor",
    "verify_stage5_toy_package",
]
