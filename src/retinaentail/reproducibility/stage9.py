"""Fail-closed Stage 9 comparison of two independently produced Stage 8 packages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.witnesses.replay import sha256_bytes

STAGE9_SCHEMA_VERSION: Final = "9.0.0"
STAGE9_BASELINE_RUN_ID: Final = "stage7_locked_baseline"

_SHA256_LENGTH: Final = 64
_OBSERVATION_FIELDS: Final = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_version",
        "scope",
        "split",
        "case_id",
        "run_id",
        "seed",
        "degradation_replay_id",
        "degradation_replay_set_sha256",
        "case_degradation_trace_sha256",
        "alpha",
        "positive_score",
        "negative_score",
        "positive_threshold",
        "negative_threshold",
        "positive_sample_count",
        "negative_sample_count",
        "config_hash",
        "data_hash",
        "model_hash",
        "provenance_sha256",
        "stage7_package_sha256",
        "state",
        "fail_closed",
        "out_of_distribution",
        "solver_succeeded",
        "degradation_known",
        "positive_witness_sha256",
        "negative_witness_sha256",
        "positive_witness_replay_valid",
        "negative_witness_replay_valid",
    }
)


class Stage9DiffStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAIL_CLOSED = "FAIL_CLOSED"


class Stage9DifferenceKind(StrEnum):
    ENVIRONMENT_LOCK = "ENVIRONMENT_LOCK"
    DATASET_IDENTITY = "DATASET_IDENTITY"
    SCOPE_OR_SPLIT = "SCOPE_OR_SPLIT"
    PREREGISTRATION = "PREREGISTRATION"
    PACKAGE_HASH = "PACKAGE_HASH"
    PACKAGE_FILE_COUNT = "PACKAGE_FILE_COUNT"
    PACKAGE_MANIFEST = "PACKAGE_MANIFEST"
    SOURCE_STAGE7_PACKAGE = "SOURCE_STAGE7_PACKAGE"
    AUDIT_PROVENANCE = "AUDIT_PROVENANCE"
    OBSERVATION_SET = "OBSERVATION_SET"
    OBSERVATION_RECORD = "OBSERVATION_RECORD"
    STATE = "STATE"
    SCORE = "SCORE"
    FAIL_CLOSED_OR_OOD = "FAIL_CLOSED_OR_OOD"
    WITNESS_CONTENT = "WITNESS_CONTENT"
    WITNESS_REPLAY = "WITNESS_REPLAY"
    INVALID_INPUT = "INVALID_INPUT"


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(char in value for char in "\r\n\x00")


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_sha256(value: object, name: str) -> str:
    if not _valid_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _plain_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9EnvironmentLock:
    """Deterministic environment fields; process/session identity is deliberately excluded."""

    python_implementation: str
    python_version: str
    python_executable_sha256: str
    platform_system: str
    platform_release: str
    platform_machine: str
    project_version: str
    dependency_versions: tuple[tuple[str, str], ...]
    python_hash_seed: str
    no_user_site: bool
    dont_write_bytecode: bool
    optimize: int
    schema_version: str = STAGE9_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE9_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 9 environment-lock schema")
        for name in (
            "python_implementation",
            "python_version",
            "platform_system",
            "platform_release",
            "platform_machine",
            "project_version",
            "python_hash_seed",
        ):
            if not _valid_identifier(getattr(self, name)):
                raise ValueError(f"invalid environment-lock field: {name}")
        _require_sha256(self.python_executable_sha256, "python_executable_sha256")
        if self.python_hash_seed != "0":
            raise ValueError("Stage 9 requires PYTHONHASHSEED=0")
        if type(self.no_user_site) is not bool or type(self.dont_write_bytecode) is not bool:  # noqa: E721
            raise TypeError("environment flags must be bool")
        if type(self.optimize) is not int or self.optimize < 0:  # noqa: E721
            raise ValueError("optimize must be a non-negative integer")
        if tuple(sorted(self.dependency_versions)) != self.dependency_versions:
            raise ValueError("dependency_versions must be sorted")
        if len({name for name, _ in self.dependency_versions}) != len(self.dependency_versions):
            raise ValueError("dependency_versions must be unique")
        if not all(
            _valid_identifier(name) and _valid_identifier(version)
            for name, version in self.dependency_versions
        ):
            raise ValueError("invalid dependency lock entry")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_executable_sha256": self.python_executable_sha256,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "project_version": self.project_version,
            "dependency_versions": {name: version for name, version in self.dependency_versions},
            "python_hash_seed": self.python_hash_seed,
            "no_user_site": self.no_user_site,
            "dont_write_bytecode": self.dont_write_bytecode,
            "optimize": self.optimize,
        }

    def sha256(self) -> str:
        return _canonical_digest(self.to_dict())


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def capture_stage9_environment_lock(*, python_hash_seed: str = "0") -> Stage9EnvironmentLock:
    """Capture an explicit lock; callers choose the seed without consulting ambient env vars."""

    executable = Path(sys.executable).resolve(strict=True)
    dependencies = tuple(
        sorted(
            (
                ("numpy", _distribution_version("numpy")),
                ("Pillow", _distribution_version("Pillow")),
                ("PyYAML", _distribution_version("PyYAML")),
                ("retinaentail-dr", _distribution_version("retinaentail-dr")),
            )
        )
    )
    return Stage9EnvironmentLock(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_executable_sha256=sha256_file(executable),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine() or "unknown",
        project_version=_distribution_version("retinaentail-dr"),
        dependency_versions=dependencies,
        python_hash_seed=python_hash_seed,
        no_user_site=bool(sys.flags.no_user_site),
        dont_write_bytecode=bool(sys.flags.dont_write_bytecode),
        optimize=sys.flags.optimize,
    )


def parse_stage9_environment_lock(value: object) -> Stage9EnvironmentLock:
    """Parse a closed serialized environment lock."""

    required = {
        "schema_version",
        "python_implementation",
        "python_version",
        "python_executable_sha256",
        "platform_system",
        "platform_release",
        "platform_machine",
        "project_version",
        "dependency_versions",
        "python_hash_seed",
        "no_user_site",
        "dont_write_bytecode",
        "optimize",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("environment lock fields are incomplete or unknown")
    dependencies = value["dependency_versions"]
    if not isinstance(dependencies, dict) or not all(
        isinstance(name, str) and isinstance(version, str) for name, version in dependencies.items()
    ):
        raise ValueError("dependency_versions must be a string mapping")
    for name in (
        "schema_version",
        "python_implementation",
        "python_version",
        "python_executable_sha256",
        "platform_system",
        "platform_release",
        "platform_machine",
        "project_version",
        "python_hash_seed",
    ):
        if not isinstance(value[name], str):
            raise ValueError(f"environment lock {name} must be text")
    if type(value["no_user_site"]) is not bool or type(value["dont_write_bytecode"]) is not bool:  # noqa: E721
        raise ValueError("environment lock boolean flags are invalid")
    if type(value["optimize"]) is not int:  # noqa: E721
        raise ValueError("environment lock optimize is invalid")
    return Stage9EnvironmentLock(
        schema_version=value["schema_version"],
        python_implementation=value["python_implementation"],
        python_version=value["python_version"],
        python_executable_sha256=value["python_executable_sha256"],
        platform_system=value["platform_system"],
        platform_release=value["platform_release"],
        platform_machine=value["platform_machine"],
        project_version=value["project_version"],
        dependency_versions=tuple(sorted(dependencies.items())),
        python_hash_seed=value["python_hash_seed"],
        no_user_site=value["no_user_site"],
        dont_write_bytecode=value["dont_write_bytecode"],
        optimize=value["optimize"],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9Observation:
    case_id: str
    run_id: str
    state: EntailmentState
    positive_score: float
    negative_score: float
    fail_closed: bool
    out_of_distribution: bool
    positive_witness_sha256: str
    negative_witness_sha256: str
    positive_witness_replay_valid: bool
    negative_witness_replay_valid: bool
    canonical_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.case_id, self.run_id)


def _parse_observation(value: object) -> Stage9Observation:
    if not isinstance(value, dict) or set(value) != _OBSERVATION_FIELDS:
        raise ValueError("Stage 8 observation fields are incomplete or unknown")
    if value.get("schema_version") != "8.0.0":
        raise ValueError("Stage 8 observation schema is not 8.0.0")
    case_id = value.get("case_id")
    run_id = value.get("run_id")
    if not _valid_identifier(case_id) or not _valid_identifier(run_id):
        raise ValueError("Stage 8 observation identifier is invalid")
    try:
        state = EntailmentState(cast(str, value.get("state")))
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 8 observation state is invalid") from error
    positive_score = _finite_number(value.get("positive_score"), "positive_score")
    negative_score = _finite_number(value.get("negative_score"), "negative_score")
    for name in ("alpha", "positive_threshold", "negative_threshold"):
        _finite_number(value.get(name), name)
    for name in ("positive_sample_count", "negative_sample_count"):
        _plain_positive_int(value.get(name), name)
    seed = value.get("seed")
    if type(seed) is not int or not 0 <= seed <= (2**32) - 1:  # noqa: E721
        raise ValueError("seed must be a uint32 integer")
    for name in (
        "degradation_replay_set_sha256",
        "case_degradation_trace_sha256",
        "config_hash",
        "data_hash",
        "model_hash",
        "provenance_sha256",
        "stage7_package_sha256",
        "positive_witness_sha256",
        "negative_witness_sha256",
    ):
        _require_sha256(value.get(name), name)
    for name in (
        "fail_closed",
        "out_of_distribution",
        "solver_succeeded",
        "degradation_known",
        "positive_witness_replay_valid",
        "negative_witness_replay_valid",
    ):
        if type(value.get(name)) is not bool:  # noqa: E721
            raise ValueError(f"{name} must be bool")
    return Stage9Observation(
        case_id=cast(str, case_id),
        run_id=cast(str, run_id),
        state=state,
        positive_score=positive_score,
        negative_score=negative_score,
        fail_closed=value["fail_closed"],
        out_of_distribution=value["out_of_distribution"],
        positive_witness_sha256=value["positive_witness_sha256"],
        negative_witness_sha256=value["negative_witness_sha256"],
        positive_witness_replay_valid=value["positive_witness_replay_valid"],
        negative_witness_replay_valid=value["negative_witness_replay_valid"],
        canonical_sha256=_canonical_digest(value),
    )


def parse_stage8_observations(value: object) -> tuple[Stage9Observation, ...]:
    """Strictly parse the complete baseline + repeat observation set."""

    required = {
        "schema_version",
        "scope",
        "source_stage7_package_sha256",
        "exact_case_set_enforced",
        "sample_replacement_performed",
        "baseline_records",
        "repeat_records",
        "replay_receipts",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Stage 8 observations root is incomplete or unknown")
    if (
        value["schema_version"] != "8.0.0"
        or value["scope"] not in {"SYNTHETIC_FIXTURE", "APPROVED_REAL_DATA"}
        or not _valid_sha256(value["source_stage7_package_sha256"])
        or value["exact_case_set_enforced"] is not True
        or value["sample_replacement_performed"] is not False
        or not isinstance(value["replay_receipts"], list)
    ):
        raise ValueError("Stage 8 observation package controls are invalid")
    baseline = value["baseline_records"]
    repeats = value["repeat_records"]
    if (
        not isinstance(baseline, list)
        or not baseline
        or not isinstance(repeats, list)
        or not repeats
    ):
        raise ValueError("Stage 8 observations must include baseline and repeat records")
    result = tuple(_parse_observation(item) for item in (*baseline, *repeats))
    keys = tuple(item.key for item in result)
    if len(keys) != len(set(keys)):
        raise ValueError("Stage 8 observations contain a duplicate case/run key")
    baseline_cases = tuple(item.case_id for item in result if item.run_id == STAGE9_BASELINE_RUN_ID)
    if not baseline_cases or len(baseline_cases) != len(set(baseline_cases)):
        raise ValueError("Stage 8 baseline set is invalid")
    repeat_runs = {item.run_id for item in result if item.run_id != STAGE9_BASELINE_RUN_ID}
    if not repeat_runs:
        raise ValueError("Stage 8 repeat set is empty")
    for run_id in repeat_runs:
        if tuple(item.case_id for item in result if item.run_id == run_id) != baseline_cases:
            raise ValueError("Stage 8 repeat case order differs from the frozen baseline")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9SessionSnapshot:
    """Bound descriptor produced only after complete Stage 8 package replay."""

    session_id: str
    producer_process_id: int
    session_nonce_sha256: str
    environment_lock_sha256: str
    dataset_id: str
    dataset_version: str
    scope: str
    split: str
    package_sha256: str
    package_file_count: int
    package_manifest_sha256: str
    stage8_audit_sha256: str
    stage8_observations_sha256: str
    stage8_preregistration_sha256: str
    source_stage7_package_sha256: str
    observations: tuple[Stage9Observation, ...]
    schema_version: str = STAGE9_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE9_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 9 session schema")
        for name in ("session_id", "dataset_id", "dataset_version", "scope", "split"):
            if not _valid_identifier(getattr(self, name)):
                raise ValueError(f"invalid Stage 9 session field: {name}")
        _plain_positive_int(self.producer_process_id, "producer_process_id")
        _plain_positive_int(self.package_file_count, "package_file_count")
        for name in (
            "session_nonce_sha256",
            "environment_lock_sha256",
            "package_sha256",
            "package_manifest_sha256",
            "stage8_audit_sha256",
            "stage8_observations_sha256",
            "stage8_preregistration_sha256",
            "source_stage7_package_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.split != "test" or self.scope not in {"SYNTHETIC_ONLY", "APPROVED_REAL_DATA"}:
            raise ValueError("Stage 9 accepts only a frozen test scope")
        if not self.observations:
            raise ValueError("Stage 9 session observations cannot be empty")
        if len({item.key for item in self.observations}) != len(self.observations):
            raise ValueError("Stage 9 session observations contain duplicate keys")

    def authority_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "producer_process_id": self.producer_process_id,
            "session_nonce_sha256": self.session_nonce_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "scope": self.scope,
            "split": self.split,
            "package_sha256": self.package_sha256,
            "package_file_count": self.package_file_count,
            "package_manifest_sha256": self.package_manifest_sha256,
            "stage8_audit_sha256": self.stage8_audit_sha256,
            "stage8_observations_sha256": self.stage8_observations_sha256,
            "stage8_preregistration_sha256": self.stage8_preregistration_sha256,
            "source_stage7_package_sha256": self.source_stage7_package_sha256,
            "complete_stage8_package_replay_verified": True,
            "detached_receipt_authoritative": False,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9CaseComparison:
    case_id: str
    run_id: str
    session_a_state: EntailmentState
    session_b_state: EntailmentState
    effective_state: EntailmentState
    session_a_positive_score: float
    session_b_positive_score: float
    session_a_negative_score: float
    session_b_negative_score: float
    state_match: bool
    score_match: bool
    fail_closed_ood_match: bool
    dual_witness_content_match: bool
    dual_witness_replay_match: bool
    deterministic_record_match: bool
    differences: tuple[Stage9DifferenceKind, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STAGE9_SCHEMA_VERSION,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "session_a_state": self.session_a_state.value,
            "session_b_state": self.session_b_state.value,
            "effective_state": self.effective_state.value,
            "session_a_positive_score": self.session_a_positive_score,
            "session_b_positive_score": self.session_b_positive_score,
            "session_a_negative_score": self.session_a_negative_score,
            "session_b_negative_score": self.session_b_negative_score,
            "state_match": self.state_match,
            "score_match": self.score_match,
            "fail_closed_ood_match": self.fail_closed_ood_match,
            "dual_witness_content_match": self.dual_witness_content_match,
            "dual_witness_replay_match": self.dual_witness_replay_match,
            "deterministic_record_match": self.deterministic_record_match,
            "differences": [item.value for item in self.differences],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage9DiffReport:
    status: Stage9DiffStatus
    session_a: Stage9SessionSnapshot
    session_b: Stage9SessionSnapshot
    global_differences: tuple[Stage9DifferenceKind, ...]
    case_comparisons: tuple[Stage9CaseComparison, ...]
    transition_matrix: tuple[tuple[str, int], ...]
    metric_intervals: tuple[tuple[str, dict[str, object]], ...]
    confidence_level: float
    schema_version: str = STAGE9_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "clinical_claim": False,
            "deterministic_difference_fail_closed": True,
            "confidence_level": self.confidence_level,
            "session_a": self.session_a.authority_dict(),
            "session_b": self.session_b.authority_dict(),
            "independent_processes": (
                self.session_a.producer_process_id != self.session_b.producer_process_id
            ),
            "independent_sessions": (
                self.session_a.session_nonce_sha256 != self.session_b.session_nonce_sha256
            ),
            "global_differences": [item.value for item in self.global_differences],
            "comparison_count": len(self.case_comparisons),
            "case_comparisons": [item.to_dict() for item in self.case_comparisons],
            "transition_matrix": dict(self.transition_matrix),
            "metric_intervals": dict(self.metric_intervals),
        }


def _append_once(values: list[Stage9DifferenceKind], item: Stage9DifferenceKind) -> None:
    if item not in values:
        values.append(item)


def _global_differences(
    a: Stage9SessionSnapshot, b: Stage9SessionSnapshot
) -> tuple[Stage9DifferenceKind, ...]:
    values: list[Stage9DifferenceKind] = []
    if a.environment_lock_sha256 != b.environment_lock_sha256:
        values.append(Stage9DifferenceKind.ENVIRONMENT_LOCK)
    if (a.dataset_id, a.dataset_version) != (b.dataset_id, b.dataset_version):
        values.append(Stage9DifferenceKind.DATASET_IDENTITY)
    if (a.scope, a.split) != (b.scope, b.split):
        values.append(Stage9DifferenceKind.SCOPE_OR_SPLIT)
    if a.stage8_preregistration_sha256 != b.stage8_preregistration_sha256:
        values.append(Stage9DifferenceKind.PREREGISTRATION)
    if a.package_sha256 != b.package_sha256:
        values.append(Stage9DifferenceKind.PACKAGE_HASH)
    if a.package_file_count != b.package_file_count:
        values.append(Stage9DifferenceKind.PACKAGE_FILE_COUNT)
    if a.package_manifest_sha256 != b.package_manifest_sha256:
        values.append(Stage9DifferenceKind.PACKAGE_MANIFEST)
    if a.source_stage7_package_sha256 != b.source_stage7_package_sha256:
        values.append(Stage9DifferenceKind.SOURCE_STAGE7_PACKAGE)
    if a.stage8_audit_sha256 != b.stage8_audit_sha256:
        values.append(Stage9DifferenceKind.AUDIT_PROVENANCE)
    if a.stage8_observations_sha256 != b.stage8_observations_sha256:
        values.append(Stage9DifferenceKind.OBSERVATION_SET)
    if (
        a.producer_process_id == b.producer_process_id
        or a.session_nonce_sha256 == b.session_nonce_sha256
    ):
        values.append(Stage9DifferenceKind.INVALID_INPUT)
    return tuple(values)


def _comparison(
    a: Stage9Observation, b: Stage9Observation, *, global_drift: bool
) -> Stage9CaseComparison:
    differences: list[Stage9DifferenceKind] = []
    state_match = a.state is b.state
    score_match = a.positive_score == b.positive_score and a.negative_score == b.negative_score
    fail_closed_ood_match = (
        a.fail_closed == b.fail_closed and a.out_of_distribution == b.out_of_distribution
    )
    witness_content_match = (
        a.positive_witness_sha256 == b.positive_witness_sha256
        and a.negative_witness_sha256 == b.negative_witness_sha256
    )
    witness_replay_match = (
        a.positive_witness_replay_valid == b.positive_witness_replay_valid
        and a.negative_witness_replay_valid == b.negative_witness_replay_valid
    )
    record_match = a.canonical_sha256 == b.canonical_sha256
    if not state_match:
        differences.append(Stage9DifferenceKind.STATE)
    if not score_match:
        differences.append(Stage9DifferenceKind.SCORE)
    if not fail_closed_ood_match:
        differences.append(Stage9DifferenceKind.FAIL_CLOSED_OR_OOD)
    if not witness_content_match:
        differences.append(Stage9DifferenceKind.WITNESS_CONTENT)
    if not witness_replay_match:
        differences.append(Stage9DifferenceKind.WITNESS_REPLAY)
    if not record_match:
        differences.append(Stage9DifferenceKind.OBSERVATION_RECORD)
    deterministic_match = not differences and not global_drift
    return Stage9CaseComparison(
        case_id=a.case_id,
        run_id=a.run_id,
        session_a_state=a.state,
        session_b_state=b.state,
        effective_state=(b.state if deterministic_match else EntailmentState.OUT_OF_MODEL),
        session_a_positive_score=a.positive_score,
        session_b_positive_score=b.positive_score,
        session_a_negative_score=a.negative_score,
        session_b_negative_score=b.negative_score,
        state_match=state_match,
        score_match=score_match,
        fail_closed_ood_match=fail_closed_ood_match,
        dual_witness_content_match=witness_content_match,
        dual_witness_replay_match=witness_replay_match,
        deterministic_record_match=deterministic_match,
        differences=tuple(differences),
    )


def _transition_matrix(values: tuple[Stage9CaseComparison, ...]) -> tuple[tuple[str, int], ...]:
    counts = {
        f"{first.value}->{second.value}": 0
        for first in EntailmentState
        for second in EntailmentState
    }
    for value in values:
        counts[f"{value.session_a_state.value}->{value.session_b_state.value}"] += 1
    return tuple(sorted(counts.items()))


def compare_stage8_sessions(
    session_a: Stage9SessionSnapshot,
    session_b: Stage9SessionSnapshot,
    *,
    confidence_level: float = 0.95,
) -> Stage9DiffReport:
    """Compare frozen deterministic fields and fail closed on every difference."""

    if not isinstance(session_a, Stage9SessionSnapshot) or not isinstance(
        session_b, Stage9SessionSnapshot
    ):
        raise TypeError("Stage 9 comparison requires two verified session snapshots")
    confidence = _finite_number(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    global_differences = list(_global_differences(session_a, session_b))
    map_a = {item.key: item for item in session_a.observations}
    map_b = {item.key: item for item in session_b.observations}
    if tuple(map_a) != tuple(map_b):
        _append_once(global_differences, Stage9DifferenceKind.OBSERVATION_SET)
    keys = tuple(dict.fromkeys((*map_a, *map_b)))
    comparisons: list[Stage9CaseComparison] = []
    global_drift = bool(global_differences)
    for key in keys:
        left = map_a.get(key)
        right = map_b.get(key)
        if left is None or right is None:
            _append_once(global_differences, Stage9DifferenceKind.OBSERVATION_SET)
            continue
        comparisons.append(_comparison(left, right, global_drift=global_drift))
    if len(comparisons) != len(keys):
        global_drift = True
        comparisons = [
            _comparison(map_a[key], map_b[key], global_drift=True)
            for key in keys
            if key in map_a and key in map_b
        ]
    denominator = len(keys)
    metric_getters: dict[str, Callable[[Stage9CaseComparison], bool]] = {
        "STATE_EXACT_MATCH": lambda item: item.state_match,
        "SCORE_EXACT_MATCH": lambda item: item.score_match,
        "FAIL_CLOSED_OOD_EXACT_MATCH": lambda item: item.fail_closed_ood_match,
        "DUAL_WITNESS_CONTENT_EXACT_MATCH": lambda item: item.dual_witness_content_match,
        "DUAL_WITNESS_REPLAY_EXACT_MATCH": lambda item: item.dual_witness_replay_match,
        "FULL_DETERMINISTIC_RECORD_EXACT_MATCH": lambda item: item.deterministic_record_match,
    }
    intervals: list[tuple[str, dict[str, object]]] = []
    for name, getter in metric_getters.items():
        numerator = sum(1 for item in comparisons if getter(item))
        intervals.append(
            (
                name,
                wilson_score_proportion(
                    numerator, denominator, confidence_level=confidence
                ).to_dict(),
            )
        )
    final_global = tuple(global_differences)
    success = (
        not final_global
        and len(comparisons) == denominator
        and all(item.deterministic_record_match for item in comparisons)
    )
    return Stage9DiffReport(
        status=Stage9DiffStatus.SUCCESS if success else Stage9DiffStatus.FAIL_CLOSED,
        session_a=session_a,
        session_b=session_b,
        global_differences=final_global,
        case_comparisons=tuple(comparisons),
        transition_matrix=_transition_matrix(tuple(comparisons)),
        metric_intervals=tuple(intervals),
        confidence_level=confidence,
    )


def environment_lock_json_bytes(value: Stage9EnvironmentLock) -> bytes:
    """Serialize a validated lock for a package artifact."""

    if not isinstance(value, Stage9EnvironmentLock):
        raise TypeError("value must be Stage9EnvironmentLock")
    return (
        json.dumps(
            value.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def environment_lock_file_sha256(path: Path) -> str:
    """Hash a serialized lock without loading unrelated files."""

    return sha256_bytes(path.read_bytes())
