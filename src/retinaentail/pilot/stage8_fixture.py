"""Offline Stage 8 repeatability and dual-witness stability fixture.

The fixture accepts only a complete Stage 7 synthetic package.  It replays the
locked baseline with a pre-registered case set, seeds, and degradation traces.
It cannot issue real-data authority and never accepts a physical dataset root.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

import yaml

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.entailment.states import EntailmentState
from retinaentail.pilot.stage7_fixture import verify_stage7_toy_package
from retinaentail.pilot.stage8_gate import (
    STAGE8_NO_PHYSICAL_ROOT_SHA256,
    Stage7PackageDescriptor,
    Stage8CapabilityError,
    Stage8FrozenBindings,
    Stage8PackageScope,
    Stage8SyntheticRunEvidence,
    begin_stage8_synthetic_run_capability,
    consume_stage8_synthetic_run_capability,
    issue_synthetic_stage7_package_capability,
    synthetic_run_capability_snapshot,
)
from retinaentail.reporting.stage3 import write_stage3_bundle
from retinaentail.reporting.stage8 import (
    issue_stage8_pilot_report,
    render_stage8_pilot_html,
    render_stage8_pilot_json,
)
from retinaentail.stability.stage8 import (
    STAGE8_BASELINE_RUN_ID,
    Stage8CaseObservation,
    Stage8Preregistration,
    Stage8ReplaySpec,
    Stage8Scope,
    Stage8StabilityMetric,
    Stage8StabilityReport,
    audit_stage8_repeatability,
)
from retinaentail.witnesses.replay import sha256_bytes

STAGE8_TOY_CONFIG_VERSION: Final = "8.0"
STAGE8_PACKAGE_VERSION: Final = "8.0.0"
STAGE8_METHOD: Final = "synthetic_locked_pilot_repeatability_witness_stability_v1"
STAGE8_SCOPE: Final = "SYNTHETIC_FIXTURE"
STAGE8_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE8_SYNTHETIC_PACKAGE"
STAGE8_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES: Final = 192 * 1024 * 1024
_MAX_PACKAGE_FILES: Final = 768
_MAX_DEPTH: Final = 24
_ROOT_FILES: Final = frozenset(
    {
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "model_provenance.json",
        "observations.json",
        "preregistration.json",
        "repeatability.json",
        "source_package.json",
        "summary.json",
    }
)


class Stage8ToyConfigurationError(ValueError):
    """Raised when the fixed Stage 8 synthetic declaration is unsafe."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8SourceContract:
    stage7_scope: str
    stage7_run_id: str
    stage7_authority_boundary: str
    stage7_package_sha256: str
    stage7_package_file_count: int
    stage7_audit_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage7_scope": self.stage7_scope,
            "stage7_run_id": self.stage7_run_id,
            "stage7_authority_boundary": self.stage7_authority_boundary,
            "stage7_package_sha256": self.stage7_package_sha256,
            "stage7_package_file_count": self.stage7_package_file_count,
            "stage7_audit_sha256": self.stage7_audit_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8ReplayRun:
    run_id: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "seed": self.seed}


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage8ToyConfig:
    run_id: str
    source_contract: Stage8SourceContract
    dataset_id: str
    dataset_version: str
    split: str
    inclusion_policy: str
    inclusion_case_ids: tuple[str, ...]
    degradation_replay_id: str
    degradation_replay_set_sha256: str
    replay_runs: tuple[Stage8ReplayRun, ...]
    confidence_level: float
    schema_version: str = STAGE8_TOY_CONFIG_VERSION
    method: str = STAGE8_METHOD
    scope: str = STAGE8_SCOPE
    alpha_and_thresholds_locked: bool = True
    config_data_model_hashes_locked: bool = True
    test_tuning_prohibited: bool = True
    threshold_reestimation_prohibited: bool = True
    sample_replacement_prohibited: bool = True
    network_used: bool = False
    gpu_used: bool = False
    real_data_access: bool = False

    def to_dict(self) -> dict[str, object]:
        metrics = {
            "confidence_level": self.confidence_level,
            "state_transition_matrix": True,
            "fail_closed_stability": True,
            "out_of_distribution_stability": True,
            "positive_witness_replay_stability": True,
            "negative_witness_replay_stability": True,
            "dual_witness_replay_stability": True,
            "positive_witness_hash_stability": True,
            "negative_witness_hash_stability": True,
            "dual_witness_hash_stability": True,
        }
        locked = {
            "alpha_and_thresholds_locked": self.alpha_and_thresholds_locked,
            "config_data_model_hashes_locked": self.config_data_model_hashes_locked,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "threshold_reestimation_prohibited": self.threshold_reestimation_prohibited,
            "sample_replacement_prohibited": self.sample_replacement_prohibited,
            "network_used": self.network_used,
            "gpu_used": self.gpu_used,
            "real_data_access": self.real_data_access,
        }
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "method": self.method,
            "scope": self.scope,
            "source_contract": self.source_contract.to_dict(),
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "inclusion_policy": self.inclusion_policy,
            "inclusion_case_ids": list(self.inclusion_case_ids),
            "degradation_replay_id": self.degradation_replay_id,
            "degradation_replay_set_sha256": self.degradation_replay_set_sha256,
            "replay_runs": [item.to_dict() for item in self.replay_runs],
            "metrics": metrics,
            "locked_policy": locked,
        }


@dataclass(frozen=True, slots=True)
class Stage8ToyRunResult:
    """Detached verification receipt; never a data-access capability."""

    output_dir: Path
    package_sha256: str
    package_file_count: int
    stage7_package_sha256: str
    case_count: int
    repeat_run_count: int
    observation_count: int
    status: str
    verified: bool


@dataclass(frozen=True, slots=True)
class _ReplaySnapshot:
    stage7_tree_sha256: str
    observations: tuple[Stage8CaseObservation, ...]


@dataclass(frozen=True, slots=True)
class _Prepared:
    config: Stage8ToyConfig
    root_artifacts: Mapping[str, bytes]
    source_descriptor: Stage7PackageDescriptor
    stability_report: Stage8StabilityReport
    replay_receipts: tuple[Mapping[str, object], ...]
    observation_count: int


def parse_stage8_toy_config(value: object) -> Stage8ToyConfig:
    """Parse the fixed synthetic preregistration with no permissive defaults."""

    data = _strict_mapping(
        value,
        "Stage 8 config",
        {
            "schema_version",
            "run_id",
            "method",
            "scope",
            "source_contract",
            "dataset_id",
            "dataset_version",
            "split",
            "inclusion_policy",
            "inclusion_case_ids",
            "degradation_replay_id",
            "degradation_replay_set_sha256",
            "replay_runs",
            "metrics",
            "locked_policy",
        },
    )
    source_raw = _strict_mapping(
        data["source_contract"],
        "source_contract",
        {
            "stage7_scope",
            "stage7_run_id",
            "stage7_authority_boundary",
            "stage7_package_sha256",
            "stage7_package_file_count",
            "stage7_audit_sha256",
        },
    )
    source = Stage8SourceContract(
        stage7_scope=_text(source_raw["stage7_scope"], "stage7_scope"),
        stage7_run_id=_text(source_raw["stage7_run_id"], "stage7_run_id"),
        stage7_authority_boundary=_text(
            source_raw["stage7_authority_boundary"], "stage7_authority_boundary"
        ),
        stage7_package_sha256=_digest(source_raw["stage7_package_sha256"], "stage7_package_sha256"),
        stage7_package_file_count=_positive_int(
            source_raw["stage7_package_file_count"], "stage7_package_file_count"
        ),
        stage7_audit_sha256=_digest(source_raw["stage7_audit_sha256"], "stage7_audit_sha256"),
    )
    case_values = data["inclusion_case_ids"]
    if not isinstance(case_values, list) or not case_values:
        raise Stage8ToyConfigurationError("inclusion_case_ids must be a non-empty array")
    case_ids = tuple(_text(item, "inclusion_case_id") for item in case_values)
    if len(case_ids) != len(set(case_ids)):
        raise Stage8ToyConfigurationError("inclusion_case_ids must be unique")
    run_values = data["replay_runs"]
    if not isinstance(run_values, list) or len(run_values) < 2:
        raise Stage8ToyConfigurationError("at least two replay runs must be preregistered")
    replay_runs: list[Stage8ReplayRun] = []
    for index, item in enumerate(run_values):
        raw = _strict_mapping(item, f"replay_runs[{index}]", {"run_id", "seed"})
        replay_runs.append(
            Stage8ReplayRun(
                run_id=_text(raw["run_id"], "replay run_id"),
                seed=_nonnegative_int(raw["seed"], "replay seed"),
            )
        )
    if len({item.run_id for item in replay_runs}) != len(replay_runs):
        raise Stage8ToyConfigurationError("replay run IDs must be unique")

    metrics = _strict_mapping(
        data["metrics"],
        "metrics",
        {
            "confidence_level",
            "state_transition_matrix",
            "fail_closed_stability",
            "out_of_distribution_stability",
            "positive_witness_replay_stability",
            "negative_witness_replay_stability",
            "dual_witness_replay_stability",
            "positive_witness_hash_stability",
            "negative_witness_hash_stability",
            "dual_witness_hash_stability",
        },
    )
    for name in set(metrics) - {"confidence_level"}:
        if _boolean(metrics[name], f"metrics.{name}") is not True:
            raise Stage8ToyConfigurationError("all preregistered stability metrics are required")
    confidence = _number(metrics["confidence_level"], "confidence_level")
    locked = _strict_mapping(
        data["locked_policy"],
        "locked_policy",
        {
            "alpha_and_thresholds_locked",
            "config_data_model_hashes_locked",
            "test_tuning_prohibited",
            "threshold_reestimation_prohibited",
            "sample_replacement_prohibited",
            "network_used",
            "gpu_used",
            "real_data_access",
        },
    )
    config = Stage8ToyConfig(
        schema_version=_text(data["schema_version"], "schema_version"),
        run_id=_text(data["run_id"], "run_id"),
        method=_text(data["method"], "method"),
        scope=_text(data["scope"], "scope"),
        source_contract=source,
        dataset_id=_text(data["dataset_id"], "dataset_id"),
        dataset_version=_text(data["dataset_version"], "dataset_version"),
        split=_text(data["split"], "split"),
        inclusion_policy=_text(data["inclusion_policy"], "inclusion_policy"),
        inclusion_case_ids=case_ids,
        degradation_replay_id=_text(data["degradation_replay_id"], "degradation_replay_id"),
        degradation_replay_set_sha256=_digest(
            data["degradation_replay_set_sha256"], "degradation_replay_set_sha256"
        ),
        replay_runs=tuple(replay_runs),
        confidence_level=confidence,
        alpha_and_thresholds_locked=_boolean(
            locked["alpha_and_thresholds_locked"], "alpha_and_thresholds_locked"
        ),
        config_data_model_hashes_locked=_boolean(
            locked["config_data_model_hashes_locked"], "config_data_model_hashes_locked"
        ),
        test_tuning_prohibited=_boolean(locked["test_tuning_prohibited"], "test_tuning_prohibited"),
        threshold_reestimation_prohibited=_boolean(
            locked["threshold_reestimation_prohibited"], "threshold_reestimation_prohibited"
        ),
        sample_replacement_prohibited=_boolean(
            locked["sample_replacement_prohibited"], "sample_replacement_prohibited"
        ),
        network_used=_boolean(locked["network_used"], "network_used"),
        gpu_used=_boolean(locked["gpu_used"], "gpu_used"),
        real_data_access=_boolean(locked["real_data_access"], "real_data_access"),
    )
    _validate_fixed_config(config)
    return config


def load_stage8_toy_config(path: Path) -> Stage8ToyConfig:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if len(payload) > 64 * 1024:
        raise Stage8ToyConfigurationError("Stage 8 config exceeds its size limit")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Stage8ToyConfigurationError("Stage 8 config is not strict UTF-8 YAML") from error
    return parse_stage8_toy_config(value)


def _prepare(
    config: Stage8ToyConfig,
    *,
    stage7_root: Path,
    stage7_files: Mapping[str, bytes],
) -> _Prepared:
    """Replay a complete Stage 7 package and consume one no-root capability."""

    verified = verify_stage7_toy_package(stage7_root)
    if dict(stage7_files) != _read_tree(stage7_root, label="Stage 7"):
        raise ValueError("Stage 7 package bytes changed during complete replay")
    audit = _json_mapping(stage7_files["audit.json"], "Stage 7 audit")
    stage6_summary = _json_mapping(
        stage7_files["stage6/package/summary.json"], "embedded Stage 6 summary"
    )
    _validate_stage7_source(config, audit, verified.package_sha256, len(stage7_files))
    locked = _mapping(audit.get("locked_contract"), "Stage 7 locked_contract")
    raw_cases = audit.get("case_index")
    if not isinstance(raw_cases, list):
        raise ValueError("Stage 7 case_index is missing")
    case_index = tuple(_mapping(value, "Stage 7 case") for value in raw_cases)
    if tuple(value.get("case_id") for value in case_index) != config.inclusion_case_ids:
        raise ValueError("Stage 7 test case set differs from the preregistered inclusion set")
    provenance_sha256 = sha256_bytes(stage7_files["audit.json"])
    if provenance_sha256 != config.source_contract.stage7_audit_sha256:
        raise ValueError("Stage 7 audit provenance hash differs from preregistration")
    trace_hashes = _case_trace_hashes(stage7_files, config.inclusion_case_ids)
    if sha256_bytes(_canonical_compact_bytes(trace_hashes)) != (
        config.degradation_replay_set_sha256
    ):
        raise ValueError("Stage 7 degradation replay set differs from preregistration")

    replay_specs = tuple(
        Stage8ReplaySpec(
            run_id=value.run_id,
            seed=value.seed,
            degradation_replay_id=config.degradation_replay_id,
            degradation_replay_set_sha256=config.degradation_replay_set_sha256,
        )
        for value in config.replay_runs
    )
    preregistration = Stage8Preregistration(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        scope=Stage8Scope.SYNTHETIC_ONLY,
        inclusion_case_ids=config.inclusion_case_ids,
        replay_specs=replay_specs,
        alpha=_finite_source(locked.get("alpha"), "locked alpha"),
        positive_threshold=_finite_source(locked.get("positive_threshold"), "positive_threshold"),
        negative_threshold=_finite_source(locked.get("negative_threshold"), "negative_threshold"),
        positive_sample_count=_positive_source_int(
            locked.get("positive_sample_count"), "positive_sample_count"
        ),
        negative_sample_count=_positive_source_int(
            locked.get("negative_sample_count"), "negative_sample_count"
        ),
        config_hash=_source_digest(locked.get("config_hash"), "config_hash"),
        data_hash=_source_digest(locked.get("data_hash"), "data_hash"),
        model_hash=_source_digest(locked.get("model_hash"), "model_hash"),
        stage7_package_sha256=verified.package_sha256,
        provenance_sha256=provenance_sha256,
        confidence_level=config.confidence_level,
    )
    config_bytes = _canonical_json_bytes(config.to_dict())
    config_sha256 = sha256_bytes(config_bytes)
    inclusion_sha256 = sha256_bytes(
        _canonical_compact_bytes(
            {
                "split": config.split,
                "inclusion_policy": config.inclusion_policy,
                "inclusion_case_ids": list(config.inclusion_case_ids),
            }
        )
    )
    metrics_sha256 = sha256_bytes(
        _canonical_compact_bytes([value.value for value in Stage8StabilityMetric])
    )
    frozen_bindings = Stage8FrozenBindings(
        fixed_seed=config.replay_runs[0].seed,
        degradation_replay_set_sha256=config.degradation_replay_set_sha256,
        inclusion_rules_sha256=inclusion_sha256,
        stability_metrics_sha256=metrics_sha256,
        locked_contract_sha256=sha256_bytes(_canonical_compact_bytes(locked)),
        test_split_sha256=sha256_bytes(_canonical_compact_bytes(list(case_index))),
        config_hash=preregistration.config_hash,
        data_hash=preregistration.data_hash,
        model_hash=preregistration.model_hash,
        provenance_sha256=provenance_sha256,
        preregistration_sha256=config_sha256,
    )
    stage2_package_sha256 = _source_digest(
        stage6_summary.get("stage2_package_sha256"), "Stage 2 package hash"
    )
    stage3_package_sha256 = _source_digest(
        stage6_summary.get("stage3_package_sha256"), "Stage 3 package hash"
    )
    descriptor = Stage7PackageDescriptor(
        scope=Stage8PackageScope.SYNTHETIC_FIXTURE,
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        package_sha256=verified.package_sha256,
        package_file_count=len(stage7_files),
        stage2_package_sha256=stage2_package_sha256,
        stage3_package_sha256=stage3_package_sha256,
        physical_root_identity_sha256=STAGE8_NO_PHYSICAL_ROOT_SHA256,
        frozen_bindings=frozen_bindings,
        fixture_only=True,
    )

    baseline = _extract_observations(
        stage7_files,
        config=config,
        preregistration=preregistration,
        run_id=STAGE8_BASELINE_RUN_ID,
        seed=None,
    )
    latest: dict[str, object] = {}
    replay_call_count = 0

    def replay_complete_stage7() -> Stage7PackageDescriptor:
        nonlocal replay_call_count
        replayed = verify_stage7_toy_package(stage7_root)
        fresh = _read_tree(stage7_root, label="Stage 7")
        if fresh != dict(stage7_files) or replayed.package_sha256 != descriptor.package_sha256:
            raise ValueError("complete Stage 7 package replay changed bytes or descriptor")
        fresh_observations = _extract_observations(
            fresh,
            config=config,
            preregistration=preregistration,
            run_id=STAGE8_BASELINE_RUN_ID,
            seed=None,
        )
        replay_call_count += 1
        latest["call_count"] = replay_call_count
        latest["snapshot"] = _ReplaySnapshot(
            stage7_tree_sha256=cast(str, _file_descriptor(fresh)["package_sha256"]),
            observations=fresh_observations,
        )
        return descriptor

    repeat_observations_values: list[Stage8CaseObservation] = []
    replay_receipts_values: list[dict[str, object]] = []
    lifecycle_values: list[dict[str, object]] = []
    for spec in preregistration.replay_specs:
        package_capability = issue_synthetic_stage7_package_capability(
            replay_complete_stage7,
            ttl_seconds=5,
        )
        run_capability = begin_stage8_synthetic_run_capability(
            package_capability,
            expected_dataset_id=config.dataset_id,
            expected_dataset_version=config.dataset_version,
            frozen_bindings=frozen_bindings,
            ttl_seconds=5,
        )
        count_before_consumption = replay_call_count
        current_spec = spec

        def consume_replay(
            evidence: Stage8SyntheticRunEvidence,
            bound_spec: Stage8ReplaySpec = current_spec,
            minimum_call_count: int = count_before_consumption,
        ) -> tuple[dict[str, object], tuple[Stage8CaseObservation, ...]]:
            if evidence.stage7_package_descriptor != descriptor:
                raise ValueError("Stage 8 synthetic evidence descriptor changed")
            snapshot = latest.get("snapshot")
            if not isinstance(snapshot, _ReplaySnapshot) or replay_call_count <= minimum_call_count:
                raise ValueError("consumption did not perform a fresh complete Stage 7 replay")
            repeat = tuple(
                _as_repeat_observation(value, bound_spec) for value in snapshot.observations
            )
            serialized_repeat = [_observation_to_dict(value) for value in repeat]
            witness_projection = [
                {
                    "case_id": value.case_id,
                    "positive_witness_sha256": value.positive_witness_sha256,
                    "negative_witness_sha256": value.negative_witness_sha256,
                    "positive_witness_replay_valid": value.positive_witness_replay_valid,
                    "negative_witness_replay_valid": value.negative_witness_replay_valid,
                }
                for value in repeat
            ]
            receipt = {
                "schema_version": STAGE8_PACKAGE_VERSION,
                "run_id": bound_spec.run_id,
                "seed": bound_spec.seed,
                "degradation_replay_id": bound_spec.degradation_replay_id,
                "degradation_replay_set_sha256": bound_spec.degradation_replay_set_sha256,
                "stage7_package_sha256": descriptor.package_sha256,
                "stage7_tree_sha256": snapshot.stage7_tree_sha256,
                "case_observations_sha256": sha256_bytes(
                    _canonical_compact_bytes(serialized_repeat)
                ),
                "dual_witness_replay_set_sha256": sha256_bytes(
                    _canonical_compact_bytes(witness_projection)
                ),
                "complete_stage7_package_replay_verified": True,
                "locked_baseline_reexecuted": True,
                "thresholds_reestimated": False,
                "test_tuning_performed": False,
                "sample_replacement_performed": False,
            }
            return receipt, repeat

        replay_receipt, run_observations = consume_stage8_synthetic_run_capability(
            run_capability, consume_replay
        )
        second_rejected = False
        try:
            consume_stage8_synthetic_run_capability(run_capability, consume_replay)
        except Stage8CapabilityError:
            second_rejected = True
        if not second_rejected:
            raise AssertionError("Stage 8 synthetic run capability became reusable")
        snapshot = synthetic_run_capability_snapshot(run_capability)
        if (
            snapshot.get("consumed") is not True
            or snapshot.get("complete_stage7_package_replayed") is not True
            or snapshot.get("second_consumption_rejected") is not True
        ):
            raise AssertionError("Stage 8 process-capability lifecycle receipt is incomplete")
        lifecycle_values.append({"run_id": spec.run_id, "seed": spec.seed, **snapshot})
        replay_receipts_values.append(replay_receipt)
        repeat_observations_values.extend(run_observations)

    repeat_observations = tuple(repeat_observations_values)
    replay_receipts = tuple(replay_receipts_values)
    stability_report = audit_stage8_repeatability(
        preregistration=preregistration,
        baseline_records=baseline,
        repeat_records=repeat_observations,
    )
    lifecycle = {
        "scope": Stage8PackageScope.SYNTHETIC_FIXTURE.value,
        "issued_count": len(lifecycle_values),
        "consumed_count": sum(value["consumed"] is True for value in lifecycle_values),
        "second_use_rejected_count": sum(
            value["second_consumption_rejected"] is True for value in lifecycle_values
        ),
        "complete_stage7_replay_count": sum(
            value["complete_stage7_package_replayed"] is True for value in lifecycle_values
        ),
        "process_bound": True,
        "non_serializable": True,
        "one_shot": True,
        "all_capabilities_fresh": True,
        "authorizes_embedded_package_replay": True,
        "authorizes_real_data": False,
        "authorizes_physical_dataset_root": False,
        "detached_receipt_authoritative": False,
        "authorizes_file_open": False,
        "per_run": lifecycle_values,
    }

    baseline_payload = [_observation_to_dict(value) for value in baseline]
    repeat_payload = [_observation_to_dict(value) for value in repeat_observations]
    observations_payload = {
        "schema_version": STAGE8_PACKAGE_VERSION,
        "scope": STAGE8_SCOPE,
        "source_stage7_package_sha256": descriptor.package_sha256,
        "baseline_records": baseline_payload,
        "repeat_records": repeat_payload,
        "replay_receipts": list(replay_receipts),
        "exact_case_set_enforced": True,
        "sample_replacement_performed": False,
    }
    observations_bytes = _canonical_json_bytes(observations_payload)
    observations_sha256 = sha256_bytes(observations_bytes)
    model_provenance = {
        "schema_version": STAGE8_PACKAGE_VERSION,
        "component": "stage8_synthetic_repeatability_orchestrator",
        "method": STAGE8_METHOD,
        "baseline": "locked_stage4_toy_dual_hypothesis_baseline",
        "trained": False,
        "pretrained_weights": False,
        "threshold_source": "frozen_stage7_locked_contract",
        "thresholds_reestimated": False,
        "test_tuning_performed": False,
        "network_used": False,
        "gpu_used": False,
        "clinical_claim": False,
    }
    model_bytes = _canonical_json_bytes(model_provenance)
    report = issue_stage8_pilot_report(
        run_id=config.run_id,
        source_package=descriptor.to_dict(),
        capability_lifecycle=lifecycle,
        stability_audit=stability_report,
        replay_receipts=replay_receipts,
        stage8_config_sha256=config_sha256,
        stage8_observations_sha256=observations_sha256,
        stage8_model_provenance_sha256=sha256_bytes(model_bytes),
    )
    summary = {
        "schema_version": STAGE8_PACKAGE_VERSION,
        "scope": STAGE8_SCOPE,
        "run_id": config.run_id,
        "method": STAGE8_METHOD,
        "status": stability_report.status.value,
        "source_stage7_package_sha256": descriptor.package_sha256,
        "source_stage7_package_file_count": descriptor.package_file_count,
        "case_count": len(config.inclusion_case_ids),
        "repeat_run_count": len(config.replay_runs),
        "observation_count": len(baseline) + len(repeat_observations),
        "transition_count": stability_report.expected_transition_count,
        "transition_matrix": dict(stability_report.transition_matrix),
        "stability_metric_intervals": {
            key.value: value.to_dict() for key, value in stability_report.metric_intervals.items()
        },
        "event_intervals": {
            key.value: value.to_dict() for key, value in stability_report.event_intervals.items()
        },
        "failure_reason_counts": {
            key.value: value for key, value in stability_report.failure_reason_counts.items()
        },
        "rejected_reason_counts": {
            key.value: value for key, value in stability_report.rejected_reason_counts.items()
        },
        "fresh_process_capability_count": len(config.replay_runs),
        "process_capability_consumed_count": len(config.replay_runs),
        "second_consumption_rejected_count": len(config.replay_runs),
        "complete_stage7_package_replayed_for_each_repeat": True,
        "thresholds_reestimated": False,
        "test_split_used_for_tuning": False,
        "sample_replacement_performed": False,
        "synthetic_authorizes_real_data": False,
        "synthetic_authorizes_physical_root": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "authority_boundary": STAGE8_AUTHORITY_BOUNDARY,
        "clinical_claim": False,
    }
    source_payload = {
        "schema_version": STAGE8_PACKAGE_VERSION,
        "descriptor": descriptor.to_dict(),
        "stage7_audit_sha256": provenance_sha256,
        "complete_stage7_package_replay_verified": True,
        "detached_descriptor_authoritative": False,
    }
    root_artifacts = {
        "audit.html": render_stage8_pilot_html(report).encode("utf-8"),
        "audit.json": render_stage8_pilot_json(report).encode("utf-8"),
        "config_snapshot.json": config_bytes,
        "model_provenance.json": model_bytes,
        "observations.json": observations_bytes,
        "preregistration.json": _canonical_json_bytes(preregistration.to_dict()),
        "repeatability.json": _canonical_json_bytes(stability_report.to_dict()),
        "source_package.json": _canonical_json_bytes(source_payload),
        "summary.json": _canonical_json_bytes(summary),
    }
    return _Prepared(
        config=config,
        root_artifacts=root_artifacts,
        source_descriptor=descriptor,
        stability_report=stability_report,
        replay_receipts=replay_receipts,
        observation_count=len(baseline) + len(repeat_observations),
    )


def run_stage8_toy_fixture(
    config_path: Path,
    stage7_package: Path,
    output_dir: Path,
) -> Stage8ToyRunResult:
    """Create a complete Stage 8 package after a fresh one-shot replay."""

    if any(not isinstance(value, Path) for value in (config_path, stage7_package, output_dir)):
        raise TypeError("all Stage 8 paths must be pathlib.Path values")
    config_root = config_path.resolve(strict=True)
    source_root = stage7_package.resolve(strict=True)
    target = output_dir.resolve(strict=False)
    _require_nonoverlap((config_root, source_root, target))
    config = load_stage8_toy_config(config_root)
    stage7_files = _read_tree(source_root, label="Stage 7")
    prepared = _prepare(config, stage7_root=source_root, stage7_files=stage7_files)
    payloads = dict(prepared.root_artifacts)
    payloads.update({f"stage7/package/{name}": data for name, data in stage7_files.items()})
    payloads["package_manifest.json"] = _canonical_json_bytes(_package_manifest(payloads))
    write_stage3_bundle(
        output_dir,
        payloads,
        boundary=AuditWriteBoundary(allowed_root=output_dir),
    )
    return verify_stage8_toy_package(output_dir)


def verify_stage8_toy_package(output_dir: Path) -> Stage8ToyRunResult:
    """Replay embedded Stage 7 and regenerate every Stage 8 artifact byte-for-byte."""

    if not isinstance(output_dir, Path):
        raise TypeError("Stage 8 package root must be a pathlib.Path")
    package_root = output_dir.resolve(strict=True)
    actual = _read_tree(package_root, label="Stage 8")
    manifest_bytes = actual.get("package_manifest.json")
    if manifest_bytes is None:
        raise ValueError("Stage 8 package manifest is missing")
    manifest = _json_mapping(manifest_bytes, "Stage 8 package manifest")
    _verify_package_manifest(manifest, actual)
    try:
        root_files = {name: actual[name] for name in _ROOT_FILES}
    except KeyError as error:
        raise ValueError("Stage 8 root artifact is missing") from error
    stage7_files = _prefixed(actual, "stage7/package/")
    expected_paths = {
        "package_manifest.json",
        *_ROOT_FILES,
        *(f"stage7/package/{name}" for name in stage7_files),
    }
    if set(actual) != expected_paths:
        raise ValueError("Stage 8 package contains an unknown path")
    config = parse_stage8_toy_config(
        _decode_json(root_files["config_snapshot.json"], "Stage 8 config snapshot")
    )
    prepared = _prepare(
        config,
        stage7_root=package_root / "stage7" / "package",
        stage7_files=stage7_files,
    )
    if dict(prepared.root_artifacts) != root_files:
        raise ValueError("Stage 8 root artifacts do not regenerate byte-for-byte")
    expected_payloads = dict(prepared.root_artifacts)
    expected_payloads.update(
        {f"stage7/package/{name}": data for name, data in stage7_files.items()}
    )
    expected_manifest = _package_manifest(expected_payloads)
    if manifest != expected_manifest:
        raise ValueError("Stage 8 package manifest does not match regenerated artifacts")
    return Stage8ToyRunResult(
        output_dir=output_dir,
        package_sha256=cast(str, expected_manifest["package_sha256"]),
        package_file_count=len(actual),
        stage7_package_sha256=prepared.source_descriptor.package_sha256,
        case_count=len(prepared.config.inclusion_case_ids),
        repeat_run_count=len(prepared.config.replay_runs),
        observation_count=prepared.observation_count,
        status=prepared.stability_report.status.value,
        verified=True,
    )


def _validate_stage7_source(
    config: Stage8ToyConfig,
    audit: Mapping[str, object],
    package_sha256: str,
    file_count: int,
) -> None:
    source = config.source_contract
    if (
        audit.get("schema_version") != "7.0.0"
        or audit.get("scope") != "SYNTHETIC_ONLY"
        or audit.get("run_id") != source.stage7_run_id
        or audit.get("authority_boundary") != source.stage7_authority_boundary
        or audit.get("dataset_root_probed") is not False
        or audit.get("real_data_read_count") != 0
        or audit.get("thresholds_reestimated") is not False
        or audit.get("test_split_used_for_calibration") is not False
        or audit.get("test_tuning_prohibited") is not True
        or audit.get("real_authorization_issued") is not False
        or audit.get("synthetic_capability_authorizes_real_data") is not False
        or source.stage7_scope != "SYNTHETIC_ONLY"
        or package_sha256 != source.stage7_package_sha256
        or file_count != source.stage7_package_file_count
    ):
        raise ValueError("Stage 7 source violates the fixed synthetic Stage 8 contract")
    rebuild = _mapping(audit.get("rebuild_equivalence"), "Stage 7 rebuild_equivalence")
    if (
        rebuild.get("dataset_id") != config.dataset_id
        or rebuild.get("dataset_version") != config.dataset_version
        or rebuild.get("canonical_equivalent") is not True
        or rebuild.get("byte_equivalent") is not True
        or rebuild.get("mismatch_count") != 0
        or rebuild.get("physical_dataset_root_used") is not False
    ):
        raise ValueError("Stage 7 source dataset identity or rebuild proof differs")


def _case_trace_hashes(files: Mapping[str, bytes], case_ids: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for case_id in case_ids:
        payload = files.get(_stage4_case_path(case_id, "case.json"))
        if payload is None:
            raise ValueError(f"Stage 7 source case artifact is missing: {case_id}")
        case = _json_mapping(payload, f"case {case_id}")
        values[case_id] = _source_digest(
            case.get("degradation_trace_sha256"), "case degradation trace"
        )
    return values


def _extract_observations(
    files: Mapping[str, bytes],
    *,
    config: Stage8ToyConfig,
    preregistration: Stage8Preregistration,
    run_id: str,
    seed: int | None,
) -> tuple[Stage8CaseObservation, ...]:
    audit = _json_mapping(files["audit.json"], "Stage 7 audit")
    raw_cases = audit.get("case_index")
    if not isinstance(raw_cases, list):
        raise ValueError("Stage 7 case index is missing")
    indexed = {
        cast(str, value["case_id"]): value
        for raw in raw_cases
        if isinstance((value := _mapping(raw, "Stage 7 case")).get("case_id"), str)
    }
    if set(indexed) != set(config.inclusion_case_ids) or len(indexed) != len(raw_cases):
        raise ValueError("Stage 7 case index is not the exact preregistered set")
    observations: list[Stage8CaseObservation] = []
    for case_id in config.inclusion_case_ids:
        index = indexed[case_id]
        case_payload = files.get(_stage4_case_path(case_id, "case.json"))
        evaluation_payload = files.get(_stage4_case_path(case_id, "evaluation.json"))
        if case_payload is None or evaluation_payload is None:
            raise ValueError(f"Stage 7 locked baseline artifacts are missing for {case_id}")
        case = _json_mapping(case_payload, f"case {case_id}")
        evaluation = _json_mapping(evaluation_payload, f"evaluation {case_id}")
        case_spec = _mapping(case.get("case_spec"), "Stage 7 case_spec")
        if case_spec.get("case_id") != case_id:
            raise ValueError("Stage 7 case artifact identity differs")
        fail_closed = index.get("status") == "FAIL_CLOSED"
        solver_succeeded = all(
            evaluation.get(name) == "toy_baseline"
            for name in ("solver_status", "positive_solver_status", "negative_solver_status")
        )
        actual_seed = seed if seed is not None else case.get("trace_seed")
        observation = Stage8CaseObservation(
            case_id=case_id,
            run_id=run_id,
            dataset_id=config.dataset_id,
            dataset_version=config.dataset_version,
            split=config.split,
            scope=Stage8Scope.SYNTHETIC_ONLY,
            seed=actual_seed,
            degradation_replay_id=(
                config.degradation_replay_id if seed is not None else "stage7_frozen"
            ),
            degradation_replay_set_sha256=config.degradation_replay_set_sha256,
            case_degradation_trace_sha256=case.get("degradation_trace_sha256"),
            state=index.get("state"),
            fail_closed=fail_closed,
            out_of_distribution=evaluation.get("out_of_distribution"),
            alpha=evaluation.get("alpha"),
            positive_score=evaluation.get("positive_score"),
            negative_score=evaluation.get("negative_score"),
            positive_threshold=evaluation.get("positive_threshold"),
            negative_threshold=evaluation.get("negative_threshold"),
            positive_sample_count=evaluation.get("positive_sample_count"),
            negative_sample_count=evaluation.get("negative_sample_count"),
            config_hash=evaluation.get("config_hash"),
            data_hash=evaluation.get("data_hash"),
            model_hash=evaluation.get("model_hash"),
            stage7_package_sha256=preregistration.stage7_package_sha256,
            provenance_sha256=preregistration.provenance_sha256,
            degradation_known=evaluation.get("degradation_known"),
            solver_succeeded=solver_succeeded,
            positive_witness_sha256=index.get("positive_witness_sha256"),
            negative_witness_sha256=index.get("negative_witness_sha256"),
            positive_witness_replay_valid=index.get("positive_witness_replay_passed"),
            negative_witness_replay_valid=index.get("negative_witness_replay_passed"),
        )
        observations.append(observation)
    return tuple(observations)


def _as_repeat_observation(
    value: Stage8CaseObservation, spec: Stage8ReplaySpec
) -> Stage8CaseObservation:
    payload = _observation_to_dict(value)
    payload.update(
        {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "degradation_replay_id": spec.degradation_replay_id,
            "degradation_replay_set_sha256": spec.degradation_replay_set_sha256,
            "scope": Stage8Scope.SYNTHETIC_ONLY,
        }
    )
    return Stage8CaseObservation(**payload)


def _observation_to_dict(value: Stage8CaseObservation) -> dict[str, object]:
    if not isinstance(value, Stage8CaseObservation):
        raise TypeError("value must be a Stage8CaseObservation")
    scope = value.scope.value if isinstance(value.scope, Stage8Scope) else value.scope
    state = value.state.value if isinstance(value.state, EntailmentState) else value.state
    return {
        "schema_version": value.schema_version,
        "case_id": value.case_id,
        "run_id": value.run_id,
        "dataset_id": value.dataset_id,
        "dataset_version": value.dataset_version,
        "split": value.split,
        "scope": scope,
        "seed": value.seed,
        "degradation_replay_id": value.degradation_replay_id,
        "degradation_replay_set_sha256": value.degradation_replay_set_sha256,
        "case_degradation_trace_sha256": value.case_degradation_trace_sha256,
        "state": state,
        "fail_closed": value.fail_closed,
        "out_of_distribution": value.out_of_distribution,
        "alpha": value.alpha,
        "positive_score": value.positive_score,
        "negative_score": value.negative_score,
        "positive_threshold": value.positive_threshold,
        "negative_threshold": value.negative_threshold,
        "positive_sample_count": value.positive_sample_count,
        "negative_sample_count": value.negative_sample_count,
        "config_hash": value.config_hash,
        "data_hash": value.data_hash,
        "model_hash": value.model_hash,
        "stage7_package_sha256": value.stage7_package_sha256,
        "provenance_sha256": value.provenance_sha256,
        "degradation_known": value.degradation_known,
        "solver_succeeded": value.solver_succeeded,
        "positive_witness_sha256": value.positive_witness_sha256,
        "negative_witness_sha256": value.negative_witness_sha256,
        "positive_witness_replay_valid": value.positive_witness_replay_valid,
        "negative_witness_replay_valid": value.negative_witness_replay_valid,
    }


def _stage4_case_path(case_id: str, artifact: str) -> str:
    return f"stage6/package/stage5/package/stage4_lock/package/cases/{case_id}/{artifact}"


def _validate_fixed_config(config: Stage8ToyConfig) -> None:
    expected_source = {
        "stage7_scope": "SYNTHETIC_ONLY",
        "stage7_run_id": "stage7-toy-authorization-consumption-001",
        "stage7_authority_boundary": "COMPLETE_VERIFIED_STAGE7_PACKAGE",
        "stage7_package_sha256": (
            "232506ae1514b8ff52b60621b73dcc3218c315277bd360efdacdf030c39eb61a"
        ),
        "stage7_package_file_count": 130,
        "stage7_audit_sha256": ("ada988dd84d27b99a9c78e593197521cd0a7feab090de6ac31991a6b9c99d9ad"),
    }
    expected_cases = (
        "stage4-toy-entailed-present",
        "stage4-toy-entailed-absent",
        "stage4-toy-raw-unidentifiable",
        "stage4-toy-infeasible-out-of-model",
        "stage4-toy-domain-out-of-model",
        "stage4-toy-unknown-degradation",
        "stage4-toy-invalid-witness",
    )
    if (
        config.schema_version != STAGE8_TOY_CONFIG_VERSION
        or config.run_id != "stage8-toy-repeatability-stability-001"
        or config.method != STAGE8_METHOD
        or config.scope != STAGE8_SCOPE
        or config.source_contract.to_dict() != expected_source
        or config.dataset_id != "synthetic_stage2"
        or config.dataset_version != "synthetic-fixture-v1"
        or config.split != "test"
        or config.inclusion_policy != "EXACT_FROZEN_CASE_SET_NO_REPLACEMENT"
        or config.inclusion_case_ids != expected_cases
        or config.degradation_replay_id != "frozen_stage7_test_trace_set_v1"
        or config.degradation_replay_set_sha256
        != "3ecb4261c82250937253dea330ba898317ed7903416d88529fe10ef8fd6a9e76"
        or tuple((item.run_id, item.seed) for item in config.replay_runs)
        != (("stage8-repeat-001", 20260824), ("stage8-repeat-002", 20260824))
        or config.confidence_level != 0.95
        or config.alpha_and_thresholds_locked is not True
        or config.config_data_model_hashes_locked is not True
        or config.test_tuning_prohibited is not True
        or config.threshold_reestimation_prohibited is not True
        or config.sample_replacement_prohibited is not True
        or config.network_used is not False
        or config.gpu_used is not False
        or config.real_data_access is not False
    ):
        raise Stage8ToyConfigurationError("Stage 8 fixture safety declarations are invalid")


def _require_nonoverlap(paths: Sequence[Path]) -> None:
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Stage 8 source and output paths must not overlap")


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
            metadata = os.lstat(entry.path)
            if _is_link_or_reparse(metadata):
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
            if metadata.st_size > _MAX_FILE_BYTES:
                raise ValueError(f"{label} package artifact exceeds its size limit")
            payload = _read_regular_bounded(Path(entry.path), metadata)
            after = os.lstat(entry.path)
            if _identity(metadata) != _identity(after) or len(payload) != metadata.st_size:
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
            chunks.append(chunk)
            size += len(chunk)
            if size > _MAX_FILE_BYTES:
                raise ValueError("package artifact exceeds its size limit")
        if size != opened.st_size:
            raise ValueError("package artifact size changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_link_components(path: Path) -> None:
    existing = path.resolve(strict=False)
    for parent in (existing, *existing.parents):
        if not parent.exists():
            continue
        metadata = os.lstat(parent)
        if _is_link_or_reparse(metadata):
            raise ValueError("package path contains a link or reparse point")


def _is_link_or_reparse(value: os.stat_result) -> bool:
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    attributes = int(getattr(value, "st_file_attributes", 0))
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


def _file_descriptor(files: Mapping[str, bytes]) -> dict[str, object]:
    entries = [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(files.items())
    ]
    base = {"digest_method": STAGE8_DIGEST_METHOD, "file_count": len(entries), "files": entries}
    return {**base, "package_sha256": sha256_bytes(_canonical_compact_bytes(base))}


def _package_manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    descriptor = _file_descriptor(payloads)
    return {
        "schema_version": STAGE8_PACKAGE_VERSION,
        "digest_method": descriptor["digest_method"],
        "file_count": descriptor["file_count"],
        "files": descriptor["files"],
        "package_sha256": descriptor["package_sha256"],
    }


def _verify_package_manifest(value: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    data = _strict_value_mapping(
        value,
        "Stage 8 package manifest",
        {"schema_version", "digest_method", "file_count", "files", "package_sha256"},
    )
    if (
        data["schema_version"] != STAGE8_PACKAGE_VERSION
        or data["digest_method"] != STAGE8_DIGEST_METHOD
    ):
        raise ValueError("Stage 8 package manifest version or method is invalid")
    payloads = {
        name: payload for name, payload in actual.items() if name != "package_manifest.json"
    }
    if dict(value) != _package_manifest(payloads):
        raise ValueError("Stage 8 package manifest does not match package bytes")


def _prefixed(files: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    result = {name[len(prefix) :]: data for name, data in files.items() if name.startswith(prefix)}
    if not result or any(not name for name in result):
        raise ValueError(f"embedded package {prefix!r} is missing or malformed")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
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


def _strict_mapping(value: object, name: str, required: set[str]) -> dict[str, object]:
    try:
        return _strict_value_mapping(value, name, required)
    except ValueError as error:
        raise Stage8ToyConfigurationError(str(error)) from error


def _strict_value_mapping(value: object, name: str, required: set[str]) -> dict[str, object]:
    result = _mapping(value, name)
    if set(result) != required:
        raise ValueError(f"{name} fields are incomplete or unknown")
    return result


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n\x00")
    ):
        raise Stage8ToyConfigurationError(f"{name} must be safe non-empty text")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Stage8ToyConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Stage8ToyConfigurationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage8ToyConfigurationError(f"{name} must be a non-negative integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise Stage8ToyConfigurationError(f"{name} must be finite")
    return float(value)


def _finite_source(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_source_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721
        raise Stage8ToyConfigurationError(f"{name} must be an exact boolean")
    return value


__all__ = [
    "STAGE8_AUTHORITY_BOUNDARY",
    "STAGE8_PACKAGE_VERSION",
    "STAGE8_TOY_CONFIG_VERSION",
    "Stage8ReplayRun",
    "Stage8SourceContract",
    "Stage8ToyConfig",
    "Stage8ToyConfigurationError",
    "Stage8ToyRunResult",
    "load_stage8_toy_config",
    "parse_stage8_toy_config",
    "run_stage8_toy_fixture",
    "verify_stage8_toy_package",
]
