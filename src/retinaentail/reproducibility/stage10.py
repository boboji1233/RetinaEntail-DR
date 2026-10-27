"""Stage 10 runner-portability comparison contracts.

This module compares complete, independently replayed Stage 9 results.  Runner and
process identity are expected to differ; every preregistered deterministic field is
compared exactly.  There is deliberately no numeric tolerance.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import locale
import math
import os
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import numpy as np

from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.reproducibility.stage9 import Stage9Observation

STAGE10_SCHEMA_VERSION: Final = "10.0.0"
STAGE10_INTERVAL_METHOD: Final = "wilson_score"
STAGE10_THREAD_KEYS: Final = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON representation used by Stage 10 signatures."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _valid_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(c in value for c in "\r\n\x00")


def _require_text(value: object, name: str) -> str:
    if not _valid_text(value):
        raise ValueError(f"{name} must be safe non-empty text")
    return cast(str, value)


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:  # noqa: E721
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _blas_configuration_sha256() -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    normalized = "\n".join(line.rstrip() for line in stream.getvalue().splitlines()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage10EnvironmentLock:
    """Frozen environment policy captured before either baseline runner starts."""

    python_implementation: str
    python_version: str
    python_executable_sha256: str
    pyproject_sha256: str
    platform_system: str
    platform_release: str
    platform_machine: str
    platform_processor: str
    project_version: str
    dependency_versions: tuple[tuple[str, str], ...]
    blas_configuration_sha256: str
    thread_policy: tuple[tuple[str, str], ...]
    locale_name: str
    preferred_encoding: str
    filesystem_encoding: str
    python_hash_seed: str
    no_user_site: bool
    dont_write_bytecode: bool
    utf8_mode: int
    optimize: int
    schema_version: str = STAGE10_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE10_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 10 environment-lock schema")
        for name in (
            "python_implementation",
            "python_version",
            "platform_system",
            "platform_release",
            "platform_machine",
            "platform_processor",
            "project_version",
            "locale_name",
            "preferred_encoding",
            "filesystem_encoding",
            "python_hash_seed",
        ):
            _require_text(getattr(self, name), name)
        for name in (
            "python_executable_sha256",
            "pyproject_sha256",
            "blas_configuration_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.python_hash_seed != "0":
            raise ValueError("Stage 10 requires PYTHONHASHSEED=0")
        if self.thread_policy != tuple((name, "1") for name in STAGE10_THREAD_KEYS):
            raise ValueError("Stage 10 thread policy must pin every declared backend to one thread")
        if tuple(sorted(self.dependency_versions)) != self.dependency_versions:
            raise ValueError("dependency versions must be sorted")
        if len({name for name, _ in self.dependency_versions}) != len(self.dependency_versions):
            raise ValueError("dependency versions must be unique")
        if not all(
            _valid_text(name) and _valid_text(version) for name, version in self.dependency_versions
        ):
            raise ValueError("dependency lock contains invalid text")
        if type(self.no_user_site) is not bool or type(self.dont_write_bytecode) is not bool:  # noqa: E721
            raise TypeError("environment flags must be bool")
        if type(self.utf8_mode) is not int or self.utf8_mode != 1:  # noqa: E721
            raise ValueError("Stage 10 requires Python UTF-8 mode")
        if type(self.optimize) is not int or self.optimize < 0:  # noqa: E721
            raise ValueError("optimize must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_executable_sha256": self.python_executable_sha256,
            "pyproject_sha256": self.pyproject_sha256,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "platform_processor": self.platform_processor,
            "project_version": self.project_version,
            "dependency_versions": dict(self.dependency_versions),
            "blas_configuration_sha256": self.blas_configuration_sha256,
            "thread_policy": dict(self.thread_policy),
            "locale_name": self.locale_name,
            "preferred_encoding": self.preferred_encoding,
            "filesystem_encoding": self.filesystem_encoding,
            "python_hash_seed": self.python_hash_seed,
            "no_user_site": self.no_user_site,
            "dont_write_bytecode": self.dont_write_bytecode,
            "utf8_mode": self.utf8_mode,
            "optimize": self.optimize,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def host_profile_sha256(self) -> str:
        return canonical_sha256(
            {
                "platform_system": self.platform_system,
                "platform_release": self.platform_release,
                "platform_machine": self.platform_machine,
                "platform_processor": self.platform_processor,
                "blas_configuration_sha256": self.blas_configuration_sha256,
            }
        )


def capture_stage10_environment_lock(project_root: Path) -> Stage10EnvironmentLock:
    """Capture actual runner state; this is never called by the zero-I/O REAL preflight."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path")
    root = project_root.resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    pyproject = (root / "pyproject.toml").resolve(strict=True)
    thread_policy = tuple((name, os.environ.get(name, "")) for name in STAGE10_THREAD_KEYS)
    processor = platform.processor().strip() or platform.machine().strip() or "unknown"
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
    return Stage10EnvironmentLock(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_executable_sha256=sha256_file(executable),
        pyproject_sha256=sha256_file(pyproject),
        platform_system=platform.system() or "unknown",
        platform_release=platform.release() or "unknown",
        platform_machine=platform.machine() or "unknown",
        platform_processor=processor,
        project_version=_distribution_version("retinaentail-dr"),
        dependency_versions=dependencies,
        blas_configuration_sha256=_blas_configuration_sha256(),
        thread_policy=thread_policy,
        locale_name=locale.setlocale(locale.LC_ALL, None) or "unknown",
        preferred_encoding=locale.getpreferredencoding(False) or "unknown",
        filesystem_encoding=sys.getfilesystemencoding() or "unknown",
        python_hash_seed=os.environ.get("PYTHONHASHSEED", ""),
        no_user_site=bool(sys.flags.no_user_site),
        dont_write_bytecode=bool(sys.flags.dont_write_bytecode),
        utf8_mode=sys.flags.utf8_mode,
        optimize=sys.flags.optimize,
    )


def parse_stage10_environment_lock(value: object) -> Stage10EnvironmentLock:
    fields = {
        "schema_version",
        "python_implementation",
        "python_version",
        "python_executable_sha256",
        "pyproject_sha256",
        "platform_system",
        "platform_release",
        "platform_machine",
        "platform_processor",
        "project_version",
        "dependency_versions",
        "blas_configuration_sha256",
        "thread_policy",
        "locale_name",
        "preferred_encoding",
        "filesystem_encoding",
        "python_hash_seed",
        "no_user_site",
        "dont_write_bytecode",
        "utf8_mode",
        "optimize",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Stage 10 environment lock fields are incomplete or unknown")
    dependencies = value["dependency_versions"]
    threads = value["thread_policy"]
    if not isinstance(dependencies, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in dependencies.items()
    ):
        raise ValueError("dependency_versions must be a string mapping")
    if not isinstance(threads, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in threads.items()
    ):
        raise ValueError("thread_policy must be a string mapping")
    kwargs = {
        name: value[name] for name in fields if name not in {"dependency_versions", "thread_policy"}
    }
    if not all(
        isinstance(kwargs[name], str)
        for name in fields
        if name
        not in {
            "dependency_versions",
            "thread_policy",
            "no_user_site",
            "dont_write_bytecode",
            "utf8_mode",
            "optimize",
        }
    ):
        raise ValueError("Stage 10 environment text fields are invalid")
    if type(value["no_user_site"]) is not bool or type(value["dont_write_bytecode"]) is not bool:  # noqa: E721
        raise ValueError("Stage 10 environment boolean fields are invalid")
    if type(value["utf8_mode"]) is not int or type(value["optimize"]) is not int:  # noqa: E721
        raise ValueError("Stage 10 environment integer fields are invalid")
    return Stage10EnvironmentLock(
        dependency_versions=tuple(sorted(dependencies.items())),
        thread_policy=tuple(sorted(threads.items())),
        **kwargs,
    )


class Stage10DiffStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAIL_CLOSED = "FAIL_CLOSED"


class Stage10DifferenceKind(StrEnum):
    ENVIRONMENT_LOCK = "ENVIRONMENT_LOCK"
    DATASET_IDENTITY = "DATASET_IDENTITY"
    SCOPE_OR_SPLIT = "SCOPE_OR_SPLIT"
    PREREGISTRATION = "PREREGISTRATION"
    SOURCE_STAGE9_PACKAGE = "SOURCE_STAGE9_PACKAGE"
    SOURCE_PROVENANCE = "SOURCE_PROVENANCE"
    SIGNATURE = "SIGNATURE"
    RUNNER_INDEPENDENCE = "RUNNER_INDEPENDENCE"
    DETERMINISTIC_PACKAGE_CONTENT = "DETERMINISTIC_PACKAGE_CONTENT"
    OBSERVATION_SET = "OBSERVATION_SET"
    STATE = "STATE"
    SCORE = "SCORE"
    FAIL_CLOSED_OR_OOD = "FAIL_CLOSED_OR_OOD"
    WITNESS_CONTENT = "WITNESS_CONTENT"
    WITNESS_REPLAY = "WITNESS_REPLAY"
    OBSERVATION_RECORD = "OBSERVATION_RECORD"


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage10RunnerSnapshot:
    runner_id: str
    producer_process_id: int
    session_nonce_sha256: str
    environment_lock_sha256: str
    host_profile_sha256: str
    dataset_id: str
    dataset_version: str
    scope: str
    split: str
    source_stage9_package_sha256: str
    stage9_package_sha256: str
    stage9_package_file_count: int
    stage9_manifest_sha256: str
    stage9_audit_sha256: str
    stage9_preregistration_sha256: str
    source_stage7_package_sha256: str
    source_stage8_package_sha256: str
    deterministic_projection_sha256: str
    envelope_sha256: str
    signature_verified: bool
    observations: tuple[Stage9Observation, ...]
    schema_version: str = STAGE10_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE10_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 10 runner schema")
        for name in ("runner_id", "dataset_id", "dataset_version", "scope", "split"):
            _require_text(getattr(self, name), name)
        _positive_int(self.producer_process_id, "producer_process_id")
        _positive_int(self.stage9_package_file_count, "stage9_package_file_count")
        for name in (
            "session_nonce_sha256",
            "environment_lock_sha256",
            "host_profile_sha256",
            "source_stage9_package_sha256",
            "stage9_package_sha256",
            "stage9_manifest_sha256",
            "stage9_audit_sha256",
            "stage9_preregistration_sha256",
            "source_stage7_package_sha256",
            "source_stage8_package_sha256",
            "deterministic_projection_sha256",
            "envelope_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.scope not in {"SYNTHETIC_ONLY", "APPROVED_REAL_DATA"} or self.split != "test":
            raise ValueError("Stage 10 accepts only frozen test observations")
        if type(self.signature_verified) is not bool or not self.signature_verified:  # noqa: E721
            raise ValueError("runner envelope signature must be verified")
        if not self.observations or len({item.key for item in self.observations}) != len(
            self.observations
        ):
            raise ValueError("runner observations must be non-empty and unique")

    def authority_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runner_id": self.runner_id,
            "producer_process_id": self.producer_process_id,
            "session_nonce_sha256": self.session_nonce_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "host_profile_sha256": self.host_profile_sha256,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "scope": self.scope,
            "split": self.split,
            "source_stage9_package_sha256": self.source_stage9_package_sha256,
            "stage9_package_sha256": self.stage9_package_sha256,
            "stage9_package_file_count": self.stage9_package_file_count,
            "stage9_manifest_sha256": self.stage9_manifest_sha256,
            "stage9_audit_sha256": self.stage9_audit_sha256,
            "stage9_preregistration_sha256": self.stage9_preregistration_sha256,
            "source_stage7_package_sha256": self.source_stage7_package_sha256,
            "source_stage8_package_sha256": self.source_stage8_package_sha256,
            "deterministic_projection_sha256": self.deterministic_projection_sha256,
            "envelope_sha256": self.envelope_sha256,
            "signature_verified": self.signature_verified,
            "complete_stage9_package_replay_verified": True,
            "detached_receipt_authoritative": False,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage10CaseComparison:
    case_id: str
    run_id: str
    runner_a_state: EntailmentState
    runner_b_state: EntailmentState
    effective_state: EntailmentState
    runner_a_positive_score: float
    runner_b_positive_score: float
    runner_a_negative_score: float
    runner_b_negative_score: float
    state_match: bool
    score_match: bool
    fail_closed_ood_match: bool
    dual_witness_content_match: bool
    dual_witness_replay_match: bool
    deterministic_record_match: bool
    differences: tuple[Stage10DifferenceKind, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STAGE10_SCHEMA_VERSION,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "runner_a_state": self.runner_a_state.value,
            "runner_b_state": self.runner_b_state.value,
            "effective_state": self.effective_state.value,
            "runner_a_positive_score": self.runner_a_positive_score,
            "runner_b_positive_score": self.runner_b_positive_score,
            "runner_a_negative_score": self.runner_a_negative_score,
            "runner_b_negative_score": self.runner_b_negative_score,
            "state_match": self.state_match,
            "score_match": self.score_match,
            "fail_closed_ood_match": self.fail_closed_ood_match,
            "dual_witness_content_match": self.dual_witness_content_match,
            "dual_witness_replay_match": self.dual_witness_replay_match,
            "deterministic_record_match": self.deterministic_record_match,
            "differences": [item.value for item in self.differences],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage10DiffReport:
    status: Stage10DiffStatus
    runner_a: Stage10RunnerSnapshot
    runner_b: Stage10RunnerSnapshot
    global_differences: tuple[Stage10DifferenceKind, ...]
    case_comparisons: tuple[Stage10CaseComparison, ...]
    transition_matrix: tuple[tuple[str, int], ...]
    metric_intervals: tuple[tuple[str, dict[str, object]], ...]
    deterministic_file_mismatch_count: int
    allowed_provenance_difference_count: int
    confidence_level: float
    schema_version: str = STAGE10_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "clinical_claim": False,
            "cross_physical_host_claim": False,
            "deterministic_difference_fail_closed": True,
            "confidence_level": self.confidence_level,
            "runner_a": self.runner_a.authority_dict(),
            "runner_b": self.runner_b.authority_dict(),
            "independent_runner_processes": self.runner_a.producer_process_id
            != self.runner_b.producer_process_id,
            "independent_runner_sessions": self.runner_a.session_nonce_sha256
            != self.runner_b.session_nonce_sha256,
            "distinct_runner_ids": self.runner_a.runner_id != self.runner_b.runner_id,
            "global_differences": [item.value for item in self.global_differences],
            "comparison_count": len(self.case_comparisons),
            "case_comparisons": [item.to_dict() for item in self.case_comparisons],
            "transition_matrix": dict(self.transition_matrix),
            "metric_intervals": dict(self.metric_intervals),
            "deterministic_file_mismatch_count": self.deterministic_file_mismatch_count,
            "allowed_provenance_difference_count": self.allowed_provenance_difference_count,
        }


def compare_stage10_runners(
    runner_a: Stage10RunnerSnapshot,
    runner_b: Stage10RunnerSnapshot,
    *,
    deterministic_file_mismatch_count: int,
    allowed_provenance_difference_count: int,
    confidence_level: float = 0.95,
) -> Stage10DiffReport:
    """Compare exact deterministic fields and preserve a fixed denominator on failure."""

    if not isinstance(runner_a, Stage10RunnerSnapshot) or not isinstance(
        runner_b, Stage10RunnerSnapshot
    ):
        raise TypeError("Stage 10 requires two verified runner snapshots")
    confidence = _finite(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if type(deterministic_file_mismatch_count) is not int or deterministic_file_mismatch_count < 0:  # noqa: E721
        raise ValueError("deterministic_file_mismatch_count must be non-negative")
    if (
        type(allowed_provenance_difference_count) is not int
        or allowed_provenance_difference_count < 0
    ):  # noqa: E721
        raise ValueError("allowed_provenance_difference_count must be non-negative")
    global_differences: list[Stage10DifferenceKind] = []

    def add(kind: Stage10DifferenceKind) -> None:
        if kind not in global_differences:
            global_differences.append(kind)

    if runner_a.environment_lock_sha256 != runner_b.environment_lock_sha256:
        add(Stage10DifferenceKind.ENVIRONMENT_LOCK)
    if (runner_a.dataset_id, runner_a.dataset_version) != (
        runner_b.dataset_id,
        runner_b.dataset_version,
    ):
        add(Stage10DifferenceKind.DATASET_IDENTITY)
    if (runner_a.scope, runner_a.split) != (runner_b.scope, runner_b.split):
        add(Stage10DifferenceKind.SCOPE_OR_SPLIT)
    if runner_a.stage9_preregistration_sha256 != runner_b.stage9_preregistration_sha256:
        add(Stage10DifferenceKind.PREREGISTRATION)
    if runner_a.source_stage9_package_sha256 != runner_b.source_stage9_package_sha256:
        add(Stage10DifferenceKind.SOURCE_STAGE9_PACKAGE)
    if (
        runner_a.source_stage7_package_sha256 != runner_b.source_stage7_package_sha256
        or runner_a.source_stage8_package_sha256 != runner_b.source_stage8_package_sha256
    ):
        add(Stage10DifferenceKind.SOURCE_PROVENANCE)
    if not runner_a.signature_verified or not runner_b.signature_verified:
        add(Stage10DifferenceKind.SIGNATURE)
    if (
        runner_a.runner_id == runner_b.runner_id
        or runner_a.producer_process_id == runner_b.producer_process_id
        or runner_a.session_nonce_sha256 == runner_b.session_nonce_sha256
    ):
        add(Stage10DifferenceKind.RUNNER_INDEPENDENCE)
    if (
        runner_a.deterministic_projection_sha256 != runner_b.deterministic_projection_sha256
        or deterministic_file_mismatch_count
    ):
        add(Stage10DifferenceKind.DETERMINISTIC_PACKAGE_CONTENT)
    map_a = {item.key: item for item in runner_a.observations}
    map_b = {item.key: item for item in runner_b.observations}
    if tuple(map_a) != tuple(map_b):
        add(Stage10DifferenceKind.OBSERVATION_SET)
    keys = tuple(dict.fromkeys((*map_a, *map_b)))
    comparisons: list[Stage10CaseComparison] = []
    global_drift = bool(global_differences)
    for key in keys:
        left = map_a.get(key)
        right = map_b.get(key)
        if left is None or right is None:
            add(Stage10DifferenceKind.OBSERVATION_SET)
            continue
        differences: list[Stage10DifferenceKind] = []
        state_match = left.state is right.state
        score_match = (
            left.positive_score == right.positive_score
            and left.negative_score == right.negative_score
        )
        fail_match = (
            left.fail_closed == right.fail_closed
            and left.out_of_distribution == right.out_of_distribution
        )
        witness_content_match = (
            left.positive_witness_sha256 == right.positive_witness_sha256
            and left.negative_witness_sha256 == right.negative_witness_sha256
        )
        witness_replay_match = (
            left.positive_witness_replay_valid == right.positive_witness_replay_valid
            and left.negative_witness_replay_valid == right.negative_witness_replay_valid
        )
        record_match = left.canonical_sha256 == right.canonical_sha256
        for matches, kind in (
            (state_match, Stage10DifferenceKind.STATE),
            (score_match, Stage10DifferenceKind.SCORE),
            (fail_match, Stage10DifferenceKind.FAIL_CLOSED_OR_OOD),
            (witness_content_match, Stage10DifferenceKind.WITNESS_CONTENT),
            (witness_replay_match, Stage10DifferenceKind.WITNESS_REPLAY),
            (record_match, Stage10DifferenceKind.OBSERVATION_RECORD),
        ):
            if not matches:
                differences.append(kind)
        deterministic_match = not differences and not global_drift
        comparisons.append(
            Stage10CaseComparison(
                case_id=left.case_id,
                run_id=left.run_id,
                runner_a_state=left.state,
                runner_b_state=right.state,
                effective_state=right.state
                if deterministic_match
                else EntailmentState.OUT_OF_MODEL,
                runner_a_positive_score=left.positive_score,
                runner_b_positive_score=right.positive_score,
                runner_a_negative_score=left.negative_score,
                runner_b_negative_score=right.negative_score,
                state_match=state_match,
                score_match=score_match,
                fail_closed_ood_match=fail_match,
                dual_witness_content_match=witness_content_match,
                dual_witness_replay_match=witness_replay_match,
                deterministic_record_match=deterministic_match,
                differences=tuple(differences),
            )
        )
    if len(comparisons) != len(runner_a.observations) or len(comparisons) != len(
        runner_b.observations
    ):
        add(Stage10DifferenceKind.OBSERVATION_SET)
        global_drift = True
    if global_drift:
        comparisons = [
            replace(
                item,
                effective_state=EntailmentState.OUT_OF_MODEL,
                deterministic_record_match=False,
            )
            for item in comparisons
        ]
    frozen = tuple(comparisons)
    matrix = {
        f"{first.value}->{second.value}": 0
        for first in EntailmentState
        for second in EntailmentState
    }
    for item in frozen:
        matrix[f"{item.runner_a_state.value}->{item.runner_b_state.value}"] += 1
    metric_getters: dict[str, Callable[[Stage10CaseComparison], bool]] = {
        "state_exact_match": lambda item: item.state_match,
        "score_exact_match": lambda item: item.score_match,
        "fail_closed_ood_exact_match": lambda item: item.fail_closed_ood_match,
        "dual_witness_content_exact_match": lambda item: item.dual_witness_content_match,
        "dual_witness_replay_result_exact_match": lambda item: item.dual_witness_replay_match,
        "full_deterministic_record_exact_match": lambda item: item.deterministic_record_match,
    }
    intervals = tuple(
        (
            name,
            wilson_score_proportion(
                sum(1 for item in frozen if getter(item)),
                len(runner_a.observations),
                confidence_level=confidence,
            ).to_dict(),
        )
        for name, getter in metric_getters.items()
    )
    success = not global_differences and len(frozen) == len(runner_a.observations)
    return Stage10DiffReport(
        status=Stage10DiffStatus.SUCCESS if success else Stage10DiffStatus.FAIL_CLOSED,
        runner_a=runner_a,
        runner_b=runner_b,
        global_differences=tuple(global_differences),
        case_comparisons=frozen,
        transition_matrix=tuple(sorted(matrix.items())),
        metric_intervals=intervals,
        deterministic_file_mismatch_count=deterministic_file_mismatch_count,
        allowed_provenance_difference_count=allowed_provenance_difference_count,
        confidence_level=confidence,
    )
