"""Science S7 RetinaEntail-Lite v1 development-only runner.

The runner consumes the hash-bound S4--S6 development evidence and never accepts a
dataset or final-test root.  Its deterministic encoder and exhaustive finite search
are deliberately small, interpretable, CPU-only, and model-relative.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import OperationTrace, apply_and_record, replay_trace
from retinaentail.entailment.states import EntailmentState, fail_closed_state
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
    _write_artifacts,
    make_labeled_contact_sheet,
    normalize_regional_observation,
    regional_context_bbox,
)
from retinaentail.science_s5_degradation_calibration_v2 import frozen_family_parameters
from retinaentail.science_s6_verify import (
    HYPOTHESES,
    LESION_CLASSES,
    BankEntry,
    IOAudit,
    S6Config,
    SelectedCase,
    TwinRecord,
    Variant,
    _decode_side,
    _digest,
    build_bank,
    load_inputs,
    make_query_observation,
    select_bank_and_cases,
    transform_candidate,
)
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V1"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
CALIBRATION_HIERARCHIES: Final = (
    "GLOBAL",
    "LESION_CLASS",
    "LESION_DEGRADATION",
    "LESION_DEGRADATION_QUALITY",
)
MODERATE_STRATA: Final = (
    "PRESENT_GLARE_MODERATE_ID",
    "ABSENT_GLARE_MODERATE_ID",
)
IN_DOMAIN_STRATA: Final = (
    "PRESENT_ID",
    "ABSENT_ID",
    "S5_TIER_A_AMBIGUOUS_ID",
    *MODERATE_STRATA,
)
OOD_STRATA: Final = ("NEAR_OOD", "FAR_OOD")
S7_STRATA: Final = (*IN_DOMAIN_STRATA, *OOD_STRATA)
DETERMINATE_STATES: Final = {
    EntailmentState.ENTAILED_PRESENT.value,
    EntailmentState.ENTAILED_ABSENT.value,
}


class S7LiteError(ValueError):
    """Raised when the frozen S7 contract cannot execute safely."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteError(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteError(f"{name} is not SHA-256")
    return text


def _verify_baseline(expected: object) -> None:
    commit = _string(expected, "expected baseline commit")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = completed.stdout.strip()
    if observed != commit:
        raise S7LiteError(f"baseline drift: expected {commit}, got {observed}")


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteError(f"binding mismatch: {name}")
    return path


@dataclass(frozen=True, slots=True)
class S7Config:
    """Validated S7 v1 development contract."""

    path: Path
    config_sha256: str
    seed: int
    retrieval_top_k: int
    traces: tuple[tuple[str, float], ...]
    translations: tuple[int, ...]
    gains: tuple[float, ...]
    biases: tuple[int, ...]
    alpha: float
    minimum_region_units: int
    minimum_group_units: int
    bootstrap_replicates: int
    gate_max_fcr: float
    gate_min_moderate_coverage: float
    gate_min_verify_agreement: float
    qc_cases_per_stratum: int
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> S7Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_DEVELOPMENT_LITE"
        ):
            raise S7LiteError("unsupported S7 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        scope = _mapping(root.get("scope"), "scope")
        if scope != {
            "splits": ["TRAIN", "CALIBRATION"],
            "locked_test_reads": 0,
            "final_test_status": "NOT_OPENED",
            "paper_performance": False,
            "development_fcr_only": True,
        }:
            raise S7LiteError("S7 scope changed")
        method = _mapping(root.get("method"), "method")
        encoder = _mapping(method.get("encoder"), "encoder")
        if encoder != {
            "kind": "FROZEN_INTERPRETABLE_MULTISCALE_NUMPY_V1",
            "learned_parameters": 0,
            "input_wh": [32, 32],
            "silent_resolution_change": False,
        }:
            raise S7LiteError("S7 encoder contract changed")
        retrieval = _mapping(method.get("retrieval"), "retrieval")
        if retrieval.get("policy") != "SOURCE_ISOLATED_CONTEXT_FEATURE_TOP_K":
            raise S7LiteError("S7 retrieval contract changed")
        search = _mapping(method.get("search"), "search")
        if search.get("positive_negative") != "INDEPENDENT_EXHAUSTIVE_FINITE_SEARCH":
            raise S7LiteError("S7 search contract changed")
        trace_rows = _sequence(search.get("traces"), "traces")
        traces = tuple(
            (
                _string(_mapping(row, "trace").get("name"), "trace name"),
                _number(_mapping(row, "trace").get("severity"), "trace severity"),
            )
            for row in trace_rows
        )
        expected_traces = (
            ("IDENTITY", 0.0),
            ("GLARE_MODERATE", 0.6),
            ("GLARE_SEVERE", 1.0),
        )
        if traces != expected_traces:
            raise S7LiteError("S7 trace grid changed")
        translations = tuple(
            _integer(item, "translation", -1)
            for item in _sequence(search.get("translations_xy_px"), "translations")
        )
        gains = tuple(_number(item, "gain") for item in _sequence(search.get("gains"), "gains"))
        biases = tuple(
            _integer(item, "bias", -2) for item in _sequence(search.get("biases_uint8"), "biases")
        )
        if translations != (-1, 0, 1) or gains != (1.0,) or biases != (-2, 0, 2):
            raise S7LiteError("S7 finite search grid changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        if calibration.get("hierarchies") != list(CALIBRATION_HIERARCHIES):
            raise S7LiteError("S7 calibration hierarchy changed")
        if calibration.get("unit") != "SOURCE_IMAGE":
            raise S7LiteError("S7 calibration unit changed")
        if calibration.get("rank_rule") != "CEIL_N_PLUS_1_TIMES_ONE_MINUS_ALPHA":
            raise S7LiteError("S7 conformal rank rule changed")
        if calibration.get("fallback") != "MOST_GRANULAR_TO_PARENT_TO_GLOBAL":
            raise S7LiteError("S7 fallback rule changed")
        multi = _mapping(root.get("multi_region"), "multi-region")
        if multi.get("policies") != ["UNADJUSTED_REGION", "BONFERRONI", "HOLM", "SOURCE_MAX"]:
            raise S7LiteError("S7 multi-region policies changed")
        gate = _mapping(root.get("gate"), "gate")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(value, name) for name, value in bindings_raw.items()}
        required = {
            "execution_prompt",
            "s7_code",
            "s7_audit_schema",
            "s6_v1_config",
            "s6_v1_audit",
            "s6_v1_artifact_manifest",
            "s6_v1_results",
            "s6_v2_config",
            "s6_v2_audit",
            "s6_v2_artifact_manifest",
            "s6_v2_results",
            "s6_v2_ai_qc",
            "s6_v2_report",
        }
        if set(bindings) != required:
            raise S7LiteError("S7 binding set changed")
        qc = _mapping(root.get("qc"), "QC")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            seed=_integer(root.get("seed"), "seed"),
            retrieval_top_k=_integer(retrieval.get("top_k"), "retrieval top-k", 1),
            traces=traces,
            translations=translations,
            gains=gains,
            biases=biases,
            alpha=_number(calibration.get("alpha"), "alpha"),
            minimum_region_units=_integer(
                calibration.get("minimum_region_units"), "minimum region units", 1
            ),
            minimum_group_units=_integer(
                calibration.get("minimum_group_units"), "minimum group units", 1
            ),
            bootstrap_replicates=_integer(
                _mapping(root.get("statistics"), "statistics").get("bootstrap_replicates"),
                "bootstrap replicates",
                1,
            ),
            gate_max_fcr=_number(gate.get("maximum_development_fcr"), "FCR gate"),
            gate_min_moderate_coverage=_number(
                gate.get("minimum_moderate_coverage_per_class"), "coverage gate"
            ),
            gate_min_verify_agreement=_number(
                gate.get("minimum_lite_verify_in_domain_agreement"), "agreement gate"
            ),
            qc_cases_per_stratum=_integer(qc.get("cases_per_stratum"), "QC cases", 1),
            bindings=bindings,
        )


@dataclass(frozen=True, slots=True)
class LiteSearchResult:
    hypothesis: str
    score: float
    anchor: TwinRecord
    trace_name: str
    trace: OperationTrace
    dx: int
    dy: int
    gain: float
    bias: int
    candidate: NDArray[np.uint8]
    retrieved_ids: tuple[str, ...]
    candidate_count: int


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    lesion_class: str
    hypothesis: str
    degradation: str
    quality: str
    source_key: str
    twin_id: str
    score: float


@dataclass(frozen=True, slots=True)
class ThresholdResolution:
    requested_hierarchy: str
    used_hierarchy: str
    threshold: float
    unit_count: int
    rank: int
    values: tuple[float, ...]


def _load_s6_contract(config: S7Config) -> S6Config:
    """Reconstruct the frozen S6 selection contract without its historical HEAD guard."""

    path = config.bindings["s6_v1_config"]
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(decoded, "S6 v1 config")
    if root.get("experiment_id") != "S6_RETINAENTAIL_VERIFY_V1":
        raise S7LiteError("S6 v1 experiment binding changed")
    selection = _mapping(root.get("selection"), "S6 selection")
    hypothesis = _mapping(root.get("hypothesis_class"), "S6 hypothesis")
    scoring = _mapping(root.get("scoring"), "S6 scoring")
    bindings_raw = _mapping(root.get("bindings"), "S6 bindings")
    binding_paths = {
        name: _verify_binding(value, f"nested_s6_{name}") for name, value in bindings_raw.items()
    }
    weights_raw = _mapping(scoring.get("weights"), "S6 weights")
    return S6Config(
        path=path,
        payload=dict(root),
        config_sha256=sha256_file(path),
        seed=_integer(selection.get("seed"), "S6 seed"),
        regions_per_class=_integer(selection.get("regions_per_class"), "S6 regions", 1),
        per_stratum=20,
        bank_size_per_class=_integer(hypothesis.get("bank_size_per_class"), "S6 bank", 2),
        bank_max_per_source=_integer(hypothesis.get("bank_max_per_source"), "S6 bank/source", 1),
        retrieval_top_k=_integer(hypothesis.get("retrieval_top_k"), "S6 retrieval", 1),
        context_margin=_integer(hypothesis.get("context_margin_px"), "S6 margin", 1),
        observation_wh=(32, 32),
        translations=(-1, 0, 1),
        gains=(0.98, 1.0, 1.02),
        biases=(-2, 0, 2),
        score_weights={
            name: _number(value, f"S6 weight {name}") for name, value in weights_raw.items()
        },
        threshold_quantile=_number(scoring.get("threshold_quantile"), "S6 quantile"),
        threshold_multiplier=_number(scoring.get("threshold_multiplier"), "S6 multiplier"),
        minimum_replay_rate=0.95,
        minimum_predicate_rate=0.95,
        qc_cases_per_stratum=2,
        binding_paths=binding_paths,
    )


def _core_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return ((xx - (width - 1) / 2.0) / (0.27 * width)) ** 2 + (
        (yy - (height - 1) / 2.0) / (0.27 * height)
    ) ** 2 <= 1.0


def frozen_encoder(image: NDArray[np.uint8], *, context_only: bool = False) -> NDArray[np.float64]:
    """Return the complete deterministic, parameter-free Lite v1 feature vector."""

    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
        raise S7LiteError("Lite encoder requires 32x32 RGB uint8")
    rgb = image.astype(np.float64) / 255.0
    mask = ~_core_mask(image.shape[:2]) if context_only else np.ones((32, 32), dtype=bool)
    selected = rgb[mask]
    features: list[float] = []
    for row_block in np.array_split(np.arange(32), 4):
        for column_block in np.array_split(np.arange(32), 4):
            block_mask = mask[np.ix_(row_block, column_block)]
            block = rgb[np.ix_(row_block, column_block)]
            if block_mask.any():
                features.extend(float(value) for value in block[block_mask].mean(axis=0))
            else:
                features.extend((0.0, 0.0, 0.0))
    features.extend(float(value) for value in selected.mean(axis=0))
    features.extend(float(value) for value in selected.std(axis=0))
    for channel in range(3):
        histogram, _ = np.histogram(selected[:, channel], bins=8, range=(0.0, 1.0))
        features.extend(float(value) / selected.shape[0] for value in histogram)
    green = rgb[:, :, 1]
    padded = np.pad(green, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    gradient = np.hypot(gx, gy) / math.sqrt(2.0)
    for row_block in np.array_split(np.arange(32), 4):
        for column_block in np.array_split(np.arange(32), 4):
            block_mask = mask[np.ix_(row_block, column_block)]
            block = gradient[np.ix_(row_block, column_block)]
            features.append(float(block[block_mask].mean()) if block_mask.any() else 0.0)
    core = _core_mask(image.shape[:2])
    context = ~core
    contrast = rgb[core].mean(axis=0) - rgb[context].mean(axis=0)
    features.extend(float(value) for value in contrast)
    features.append(float(np.mean(image.max(axis=2) >= 250)))
    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (98,) or not np.isfinite(vector).all():
        raise S7LiteError("Lite encoder emitted an invalid feature vector")
    return vector


def _quality_bin(image: NDArray[np.uint8]) -> str:
    channel_range = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
    glare = np.mean((image.max(axis=2) >= 250) & (channel_range <= 18))
    return "HIGH_GLARE" if float(glare) >= 0.02 else "LOW_GLARE"


def _glare_variant(
    record: TwinRecord,
    hypothesis: str,
    severity: float,
    s6_config: S6Config,
    config: S7Config,
    io_audit: IOAudit,
) -> Variant:
    source = _decode_side(record, hypothesis, io_audit)
    bbox = regional_context_bbox(record.raw, margin=s6_config.context_margin, patch_wh=(256, 256))
    parameters = cast(
        dict[str, str | int | float | bool], frozen_family_parameters("glare_reflection", severity)
    )
    parameters = {
        **parameters,
        "center_x": ((bbox[0] + bbox[2]) / 2.0) / source.shape[1],
        "center_y": ((bbox[1] + bbox[3]) / 2.0) / source.shape[0],
    }
    seed = int(_digest(config.seed, "S7_GLARE", severity, record.twin_id, hypothesis)[:16], 16)
    output, step = apply_and_record(
        source,
        "glare_reflection",
        severity=severity,
        seed=seed,
        parameters=parameters,
    )
    return Variant(normalize_regional_observation(output, bbox, (32, 32)), OperationTrace((step,)))


def build_lite_bank(
    selected: Mapping[str, Sequence[TwinRecord]],
    s6_config: S6Config,
    config: S7Config,
    io_audit: IOAudit,
) -> dict[str, list[BankEntry]]:
    base = build_bank(selected, s6_config, io_audit)
    result: dict[str, list[BankEntry]] = {}
    for lesion_class, entries in base.items():
        expanded: list[BankEntry] = []
        for entry in entries:
            hypotheses: dict[str, Mapping[str, Variant]] = {}
            for hypothesis in HYPOTHESES:
                hypotheses[hypothesis] = {
                    "IDENTITY": entry.variants[hypothesis]["IDENTITY"],
                    "GLARE_MODERATE": _glare_variant(
                        entry.record, hypothesis, 0.6, s6_config, config, io_audit
                    ),
                    "GLARE_SEVERE": entry.variants[hypothesis]["S5_TARGET_GLARE_SEVERITY_1"],
                }
            expanded.append(BankEntry(record=entry.record, variants=hypotheses))
        result[lesion_class] = expanded
    return result


class FeatureCache:
    """Cache immutable candidate and retrieval features for the finite bank."""

    def __init__(self, config: S7Config) -> None:
        self.config = config
        self.candidates: dict[tuple[object, ...], NDArray[np.float64]] = {}
        self.context: dict[tuple[str, str], NDArray[np.float64]] = {}

    def context_feature(self, entry: BankEntry, hypothesis: str) -> NDArray[np.float64]:
        key = (entry.record.twin_id, hypothesis)
        if key not in self.context:
            self.context[key] = frozen_encoder(
                entry.variants[hypothesis]["IDENTITY"].array, context_only=True
            )
        return self.context[key]

    def candidate_feature(
        self,
        entry: BankEntry,
        hypothesis: str,
        trace_name: str,
        dx: int,
        dy: int,
        gain: float,
        bias: int,
    ) -> NDArray[np.float64]:
        key = (entry.record.twin_id, hypothesis, trace_name, dx, dy, gain, bias)
        if key not in self.candidates:
            candidate = transform_candidate(
                entry.variants[hypothesis][trace_name].array,
                dx=dx,
                dy=dy,
                gain=gain,
                bias=bias,
            )
            self.candidates[key] = frozen_encoder(candidate)
        return self.candidates[key]


def search_lite(
    observed: NDArray[np.uint8],
    entries: Sequence[BankEntry],
    hypothesis: str,
    config: S7Config,
    cache: FeatureCache,
    *,
    excluded_source_key: str,
) -> LiteSearchResult:
    """Exhaust the frozen retrieved Lite v1 hypothesis class."""

    observed_context = frozen_encoder(observed, context_only=True)
    observed_feature = frozen_encoder(observed)
    eligible = [entry for entry in entries if entry.record.source_key != excluded_source_key]
    ranked = sorted(
        eligible,
        key=lambda entry: (
            float(np.mean(np.abs(observed_context - cache.context_feature(entry, hypothesis)))),
            _digest(config.seed, "LITE_RETRIEVE", hypothesis, entry.record.twin_id),
        ),
    )
    retrieved = ranked[: config.retrieval_top_k]
    if len(retrieved) != config.retrieval_top_k:
        raise S7LiteError("insufficient source-isolated Lite exemplars")
    best: tuple[float, BankEntry, str, int, int, float, int] | None = None
    count = 0
    for entry in retrieved:
        for trace_name, _severity in config.traces:
            for dy in config.translations:
                for dx in config.translations:
                    for gain in config.gains:
                        for bias in config.biases:
                            candidate_feature = cache.candidate_feature(
                                entry, hypothesis, trace_name, dx, dy, gain, bias
                            )
                            score = float(np.mean(np.abs(observed_feature - candidate_feature)))
                            count += 1
                            key = (score, entry.record.twin_id, trace_name, dx, dy, gain, bias)
                            if best is None or key < (
                                best[0],
                                best[1].record.twin_id,
                                best[2],
                                best[3],
                                best[4],
                                best[5],
                                best[6],
                            ):
                                best = (score, entry, trace_name, dx, dy, gain, bias)
    if best is None or not math.isfinite(best[0]):
        raise S7LiteError("Lite search did not produce a finite optimum")
    score, entry, trace_name, dx, dy, gain, bias = best
    variant = entry.variants[hypothesis][trace_name]
    candidate = transform_candidate(variant.array, dx=dx, dy=dy, gain=gain, bias=bias)
    return LiteSearchResult(
        hypothesis=hypothesis,
        score=score,
        anchor=entry.record,
        trace_name=trace_name,
        trace=variant.trace,
        dx=dx,
        dy=dy,
        gain=gain,
        bias=bias,
        candidate=candidate,
        retrieved_ids=tuple(item.record.twin_id for item in retrieved),
        candidate_count=count,
    )


def _calibration_condition(trace_name: str) -> str:
    if trace_name not in {
        name
        for name, _severity in (("IDENTITY", 0.0), ("GLARE_MODERATE", 0.6), ("GLARE_SEVERE", 1.0))
    }:
        raise S7LiteError("unknown calibration condition")
    return trace_name


def calibrate(
    banks: Mapping[str, Sequence[BankEntry]], config: S7Config, cache: FeatureCache
) -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    for lesion_class in LESION_CLASSES:
        for query in banks[lesion_class]:
            for hypothesis in HYPOTHESES:
                for trace_name, _severity in config.traces:
                    observed = query.variants[hypothesis][trace_name].array
                    result = search_lite(
                        observed,
                        banks[lesion_class],
                        hypothesis,
                        config,
                        cache,
                        excluded_source_key=query.record.source_key,
                    )
                    rows.append(
                        CalibrationRow(
                            lesion_class=lesion_class,
                            hypothesis=hypothesis,
                            degradation=_calibration_condition(trace_name),
                            quality=_quality_bin(observed),
                            source_key=query.record.source_key,
                            twin_id=query.record.twin_id,
                            score=result.score,
                        )
                    )
    return rows


def _hierarchy_matches(
    row: CalibrationRow,
    hierarchy: str,
    *,
    lesion_class: str,
    hypothesis: str,
    degradation: str,
    quality: str,
) -> bool:
    if row.hypothesis != hypothesis:
        return False
    if hierarchy == "GLOBAL":
        return True
    if row.lesion_class != lesion_class:
        return False
    if hierarchy == "LESION_CLASS":
        return True
    if row.degradation != degradation:
        return False
    if hierarchy == "LESION_DEGRADATION":
        return True
    if hierarchy == "LESION_DEGRADATION_QUALITY":
        return row.quality == quality
    raise S7LiteError(f"unknown calibration hierarchy: {hierarchy}")


def resolve_threshold(
    rows: Sequence[CalibrationRow],
    config: S7Config,
    requested_hierarchy: str,
    *,
    lesion_class: str,
    hypothesis: str,
    degradation: str,
    quality: str,
    source_max: bool,
) -> ThresholdResolution:
    requested_index = CALIBRATION_HIERARCHIES.index(requested_hierarchy)
    for hierarchy in reversed(CALIBRATION_HIERARCHIES[: requested_index + 1]):
        matched = [
            row.score
            for row in rows
            if _hierarchy_matches(
                row,
                hierarchy,
                lesion_class=lesion_class,
                hypothesis=hypothesis,
                degradation=degradation,
                quality=quality,
            )
        ]
        if source_max:
            per_source: dict[str, float] = {}
            for row in rows:
                if _hierarchy_matches(
                    row,
                    hierarchy,
                    lesion_class=lesion_class,
                    hypothesis=hypothesis,
                    degradation=degradation,
                    quality=quality,
                ):
                    per_source[row.source_key] = max(per_source.get(row.source_key, 0.0), row.score)
            matched = list(per_source.values())
            minimum = config.minimum_group_units
        else:
            minimum = config.minimum_region_units
        rank = math.ceil((len(matched) + 1) * (1.0 - config.alpha))
        if len(matched) >= minimum and rank <= len(matched):
            values = tuple(sorted(matched))
            return ThresholdResolution(
                requested_hierarchy=requested_hierarchy,
                used_hierarchy=hierarchy,
                threshold=values[rank - 1],
                unit_count=len(values),
                rank=rank,
                values=values,
            )
    raise S7LiteError("no finite conformal threshold is available after global fallback")


def _moderate_observation(
    case: SelectedCase,
    hypothesis: str,
    s6_config: S6Config,
    config: S7Config,
    io_audit: IOAudit,
) -> NDArray[np.uint8]:
    return _glare_variant(case.record, hypothesis, 0.6, s6_config, config, io_audit).array


def _case_id(case: SelectedCase, stratum: str) -> str:
    return f"s6-{case.record.lesion_class.lower()}-{stratum.lower()}-{case.record.twin_id}"


def _expected_truth(stratum: str) -> str:
    if stratum.startswith("PRESENT"):
        return "PRESENT"
    if stratum.startswith("ABSENT"):
        return "ABSENT"
    if stratum == "S5_TIER_A_AMBIGUOUS_ID":
        return "AMBIGUOUS"
    return "OOD"


def _false_certainty(state: str, truth: str) -> bool:
    if state not in DETERMINATE_STATES:
        return False
    if truth == "PRESENT":
        return state != EntailmentState.ENTAILED_PRESENT.value
    if truth == "ABSENT":
        return state != EntailmentState.ENTAILED_ABSENT.value
    return True


def _decision(
    positive_score: float,
    negative_score: float,
    positive_threshold: float,
    negative_threshold: float,
) -> str:
    return fail_closed_state(
        positive_feasible=positive_score <= positive_threshold,
        negative_feasible=negative_score <= negative_threshold,
        positive_score=positive_score,
        negative_score=negative_score,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        positive_solver_status="optimal",
        negative_solver_status="optimal",
        solver_status="optimal",
        degradation_known=True,
    ).value


def _p_value(score: float, values: Sequence[float]) -> float:
    return (1.0 + sum(value >= score for value in values)) / (len(values) + 1.0)


def _png_bytes(image: NDArray[np.uint8]) -> bytes:
    from io import BytesIO

    from PIL import Image

    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _search_record(result: LiteSearchResult, relative_path: str) -> dict[str, object]:
    return {
        "hypothesis": result.hypothesis,
        "score": result.score,
        "lower_bound": result.score,
        "upper_bound": result.score,
        "optimization_gap": 0.0,
        "solver_status": "OPTIMAL_RETRIEVED_FINITE_LITE_GRID",
        "candidate_count": result.candidate_count,
        "retrieved_exemplar_ids": list(result.retrieved_ids),
        "anchor": {
            "package_id": result.anchor.package_id,
            "twin_id": result.anchor.twin_id,
            "source_key": result.anchor.source_key,
            "split": result.anchor.split,
        },
        "trace_name": result.trace_name,
        "trace": result.trace.to_dict(),
        "geometry": {"dx": result.dx, "dy": result.dy},
        "color": {"gain": result.gain, "bias": result.bias},
        "candidate_array_sha256": sha256_bytes(result.candidate.tobytes()),
        "candidate_png_path": relative_path,
        "candidate_png_sha256": sha256_bytes(_png_bytes(result.candidate)),
    }


def replay_lite_witness(
    result: LiteSearchResult,
    s6_config: S6Config,
    io_audit: IOAudit,
) -> bool:
    source = _decode_side(result.anchor, result.hypothesis, io_audit)
    io_audit.witness_replay_png_decodes += 1
    io_audit.s4_twin_png_decodes -= 1
    bbox = regional_context_bbox(
        result.anchor.raw, margin=s6_config.context_margin, patch_wh=(256, 256)
    )
    normalized = normalize_regional_observation(replay_trace(source, result.trace), bbox, (32, 32))
    candidate = transform_candidate(
        normalized,
        dx=result.dx,
        dy=result.dy,
        gain=result.gain,
        bias=result.bias,
    )
    return np.array_equal(candidate, result.candidate)


def _predicate_provenance_valid(result: LiteSearchResult) -> bool:
    gate = _mapping(result.anchor.raw.get("automated_gate"), "S4 automated gate")
    common = (
        gate.get("complete_pair") is True
        and gate.get("finite_uint8_outputs") is True
        and gate.get("outside_target_edit_exact_preservation") is True
        and gate.get("target_edit_other_lesion_overlap_px") == 0
    )
    if result.hypothesis == "PRESENT":
        return common and gate.get("target_exact_preservation_in_sham") is True
    changed = gate.get("x_minus_changed_pixel_count")
    return common and isinstance(changed, int) and not isinstance(changed, bool) and changed > 0


def _cluster_bootstrap_fcr(
    rows: Sequence[Mapping[str, object]], state_key: str, seed: int, replicates: int
) -> dict[str, object]:
    by_source: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("expected_truth") != "OOD":
            by_source[_string(row.get("source_key"), "source key")].append(row)
    sources = sorted(by_source)
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = generator.choice(sources, size=len(sources), replace=True)
        determinate = 0
        false = 0
        for source in sampled:
            for row in by_source[str(source)]:
                state = _string(row.get(state_key), "bootstrap state")
                if state in DETERMINATE_STATES:
                    determinate += 1
                    false += int(
                        _false_certainty(state, _string(row.get("expected_truth"), "truth"))
                    )
        if determinate:
            estimates.append(false / determinate)
    if not estimates:
        return {"replicates_requested": replicates, "replicates_defined": 0, "ci95": None}
    return {
        "replicates_requested": replicates,
        "replicates_defined": len(estimates),
        "ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
    }


def _policy_metrics(rows: Sequence[Mapping[str, object]], state_key: str) -> dict[str, object]:
    in_domain = [row for row in rows if row.get("expected_truth") != "OOD"]
    determinate = [row for row in in_domain if row.get(state_key) in DETERMINATE_STATES]
    false = [
        row
        for row in determinate
        if _false_certainty(
            _string(row.get(state_key), "state"), _string(row.get("expected_truth"), "truth")
        )
    ]
    sources = defaultdict(list)
    for row in in_domain:
        sources[_string(row.get("source_key"), "source")].append(row)
    false_sources = sum(
        any(
            _false_certainty(
                _string(row.get(state_key), "state"),
                _string(row.get("expected_truth"), "truth"),
            )
            for row in group
        )
        for group in sources.values()
    )
    claimed_sources = sum(
        any(row.get(state_key) in DETERMINATE_STATES for row in group) for group in sources.values()
    )
    return {
        "in_domain_regions": len(in_domain),
        "determinate_regions": len(determinate),
        "determinate_coverage": len(determinate) / len(in_domain) if in_domain else 0.0,
        "false_certainties": len(false),
        "development_fcr": len(false) / len(determinate) if determinate else None,
        "source_groups": len(sources),
        "source_groups_with_determinate": claimed_sources,
        "source_groups_with_false_certainty": false_sources,
        "family_wise_false_certainty_rate": false_sources / len(sources) if sources else 0.0,
    }


def _apply_multi_region(rows: list[dict[str, object]], config: S7Config) -> None:
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["expected_truth"] != "OOD":
            by_source[cast(str, row["source_key"])].append(index)
        else:
            row["state_bonferroni"] = EntailmentState.OUT_OF_MODEL.value
            row["state_holm"] = EntailmentState.OUT_OF_MODEL.value
    for indices in by_source.values():
        tests: list[tuple[float, int, str]] = []
        for index in indices:
            row = rows[index]
            tests.append((cast(float, row["positive_p_value"]), index, "PRESENT"))
            tests.append((cast(float, row["negative_p_value"]), index, "ABSENT"))
        test_count = len(tests)
        bonf_flags: dict[tuple[int, str], bool] = {
            (index, hypothesis): p_value > config.alpha / test_count
            for p_value, index, hypothesis in tests
        }
        rejected: set[tuple[int, str]] = set()
        for rank, (p_value, index, hypothesis) in enumerate(sorted(tests), start=1):
            if p_value <= config.alpha / (test_count - rank + 1):
                rejected.add((index, hypothesis))
            else:
                break
        for index in indices:
            row = rows[index]
            row["state_bonferroni"] = fail_closed_state(
                positive_feasible=bonf_flags[(index, "PRESENT")],
                negative_feasible=bonf_flags[(index, "ABSENT")],
                positive_score=cast(float, row["positive_score"]),
                negative_score=cast(float, row["negative_score"]),
                positive_threshold=cast(float, row["positive_threshold"]),
                negative_threshold=cast(float, row["negative_threshold"]),
                positive_solver_status="optimal",
                negative_solver_status="optimal",
            ).value
            row["state_holm"] = fail_closed_state(
                positive_feasible=(index, "PRESENT") not in rejected,
                negative_feasible=(index, "ABSENT") not in rejected,
                positive_score=cast(float, row["positive_score"]),
                negative_score=cast(float, row["negative_score"]),
                positive_threshold=cast(float, row["positive_threshold"]),
                negative_threshold=cast(float, row["negative_threshold"]),
                positive_solver_status="optimal",
                negative_solver_status="optimal",
            ).value


def _failure_injection_audit() -> dict[str, str]:
    valid = {
        "positive_feasible": True,
        "negative_feasible": False,
        "positive_score": 0.1,
        "negative_score": 0.2,
        "positive_threshold": 0.15,
        "negative_threshold": 0.15,
        "positive_solver_status": "optimal",
        "negative_solver_status": "optimal",
    }
    scenarios: dict[str, dict[str, object]] = {
        "missing_threshold": {**valid, "positive_threshold": None},
        "nonfinite_score": {**valid, "negative_score": math.nan},
        "failed_solver": {**valid, "positive_solver_status": "failed"},
        "unknown_degradation": {**valid, "degradation_known": False},
    }
    results: dict[str, str] = {}
    for name, inputs in scenarios.items():
        results[name] = fail_closed_state(
            positive_feasible=cast(bool | None, inputs.get("positive_feasible")),
            negative_feasible=cast(bool | None, inputs.get("negative_feasible")),
            positive_score=cast(float | None, inputs.get("positive_score")),
            negative_score=cast(float | None, inputs.get("negative_score")),
            positive_threshold=cast(float | None, inputs.get("positive_threshold")),
            negative_threshold=cast(float | None, inputs.get("negative_threshold")),
            positive_solver_status=cast(str | None, inputs.get("positive_solver_status")),
            negative_solver_status=cast(str | None, inputs.get("negative_solver_status")),
            degradation_known=cast(bool, inputs.get("degradation_known", True)),
        ).value
    return results


def run_s7(config: S7Config, *, output_dir: Path) -> dict[str, object]:
    """Execute the frozen S7 v1 development run."""

    output = output_dir.resolve()
    if output.exists():
        raise S7LiteError("S7 output directory already exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    tracemalloc.start()
    s6_config = _load_s6_contract(config)
    io_audit = IOAudit()
    records, collisions, _domain = load_inputs(s6_config)
    selected_banks, selected_cases = select_bank_and_cases(records, collisions, s6_config)
    banks = build_lite_bank(selected_banks, s6_config, config, io_audit)
    cache = FeatureCache(config)
    calibration_rows = calibrate(banks, config, cache)
    artifacts: dict[str, bytes] = {}
    result_rows: list[dict[str, object]] = []
    witness_checks = 0
    witness_passes = 0
    predicate_checks = 0
    predicate_passes = 0
    source_leaks = 0
    completed_searches = 0
    zero_gap_searches = 0
    qc_images: dict[str, dict[str, list[tuple[str, NDArray[np.uint8]]]]] = {
        lesion_class: {stratum: [] for stratum in S7_STRATA} for lesion_class in LESION_CLASSES
    }
    for lesion_class in LESION_CLASSES:
        expanded_cases: list[tuple[SelectedCase, str]] = []
        for case in selected_cases[lesion_class]:
            expanded_cases.append((case, case.stratum))
            if case.stratum == "PRESENT_ID":
                expanded_cases.append((case, "PRESENT_GLARE_MODERATE_ID"))
            elif case.stratum == "ABSENT_ID":
                expanded_cases.append((case, "ABSENT_GLARE_MODERATE_ID"))
        for case, stratum in expanded_cases:
            expected = _expected_truth(stratum)
            case_id = _case_id(case, stratum)
            if stratum in MODERATE_STRATA:
                hypothesis = "PRESENT" if stratum.startswith("PRESENT") else "ABSENT"
                observed = _moderate_observation(case, hypothesis, s6_config, config, io_audit)
                trace_record: dict[str, object] = {
                    "family": "glare_reflection",
                    "severity": 0.6,
                    "domain": "ID",
                    "source_truth": hypothesis,
                }
                domain_known = True
                condition = "GLARE_MODERATE"
            else:
                observed, raw_trace, domain_known = make_query_observation(
                    case, collisions, s6_config, io_audit
                )
                trace_record = dict(raw_trace)
                condition = "GLARE_SEVERE" if stratum == "S5_TIER_A_AMBIGUOUS_ID" else "IDENTITY"
            observation_relative = f"cases/{lesion_class.lower()}/{case_id}/observation.png"
            artifacts[observation_relative] = _png_bytes(observed)
            row: dict[str, object] = {
                "case_id": case_id,
                "lesion_class": lesion_class,
                "stratum": stratum,
                "expected_truth": expected,
                "source_key": case.record.source_key,
                "source_image_id": case.record.source_image_id,
                "twin_id": case.record.twin_id,
                "split": case.record.split,
                "observation_trace": trace_record,
                "observation_condition": condition,
                "quality_bin": _quality_bin(observed),
                "observation_png_path": observation_relative,
                "observation_png_sha256": sha256_bytes(artifacts[observation_relative]),
                "observation_array_sha256": sha256_bytes(observed.tobytes()),
            }
            if not domain_known:
                for hierarchy in CALIBRATION_HIERARCHIES:
                    row[f"state_{hierarchy.lower()}"] = EntailmentState.OUT_OF_MODEL.value
                row["state_region_primary"] = EntailmentState.OUT_OF_MODEL.value
                row["state_source_max"] = EntailmentState.OUT_OF_MODEL.value
                row["positive_search"] = None
                row["negative_search"] = None
                black = np.zeros_like(observed)
                qc_images[lesion_class][stratum].append(
                    (case_id, np.stack([observed, black, black]))
                )
                result_rows.append(row)
                continue
            searches: dict[str, LiteSearchResult] = {}
            resolutions: dict[str, dict[str, ThresholdResolution]] = {}
            for hypothesis in HYPOTHESES:
                search = search_lite(
                    observed,
                    banks[lesion_class],
                    hypothesis,
                    config,
                    cache,
                    excluded_source_key=case.record.source_key,
                )
                searches[hypothesis] = search
                completed_searches += 1
                zero_gap_searches += 1
                source_leaks += int(search.anchor.source_key == case.record.source_key)
                source_leaks += sum(
                    entry.record.source_key == case.record.source_key
                    for entry in banks[lesion_class]
                    if entry.record.twin_id in search.retrieved_ids
                )
                witness_checks += 1
                witness_passes += int(replay_lite_witness(search, s6_config, io_audit))
                predicate_checks += 1
                predicate_passes += int(_predicate_provenance_valid(search))
                relative = f"cases/{lesion_class.lower()}/{case_id}/best-{hypothesis.lower()}.png"
                artifacts[relative] = _png_bytes(search.candidate)
                row[f"{hypothesis.lower()}_search"] = _search_record(search, relative)
                resolutions[hypothesis] = {}
                for hierarchy in CALIBRATION_HIERARCHIES:
                    resolutions[hypothesis][hierarchy] = resolve_threshold(
                        calibration_rows,
                        config,
                        hierarchy,
                        lesion_class=lesion_class,
                        hypothesis=hypothesis,
                        degradation=condition,
                        quality=cast(str, row["quality_bin"]),
                        source_max=False,
                    )
            for hierarchy in CALIBRATION_HIERARCHIES:
                positive = resolutions["PRESENT"][hierarchy]
                negative = resolutions["ABSENT"][hierarchy]
                row[f"state_{hierarchy.lower()}"] = _decision(
                    searches["PRESENT"].score,
                    searches["ABSENT"].score,
                    positive.threshold,
                    negative.threshold,
                )
            primary_positive = resolutions["PRESENT"]["LESION_DEGRADATION_QUALITY"]
            primary_negative = resolutions["ABSENT"]["LESION_DEGRADATION_QUALITY"]
            group_positive = resolve_threshold(
                calibration_rows,
                config,
                "LESION_DEGRADATION_QUALITY",
                lesion_class=lesion_class,
                hypothesis="PRESENT",
                degradation=condition,
                quality=cast(str, row["quality_bin"]),
                source_max=True,
            )
            group_negative = resolve_threshold(
                calibration_rows,
                config,
                "LESION_DEGRADATION_QUALITY",
                lesion_class=lesion_class,
                hypothesis="ABSENT",
                degradation=condition,
                quality=cast(str, row["quality_bin"]),
                source_max=True,
            )
            row.update(
                {
                    "positive_score": searches["PRESENT"].score,
                    "negative_score": searches["ABSENT"].score,
                    "positive_threshold": primary_positive.threshold,
                    "negative_threshold": primary_negative.threshold,
                    "positive_threshold_hierarchy": primary_positive.used_hierarchy,
                    "negative_threshold_hierarchy": primary_negative.used_hierarchy,
                    "positive_p_value": _p_value(
                        searches["PRESENT"].score, primary_positive.values
                    ),
                    "negative_p_value": _p_value(searches["ABSENT"].score, primary_negative.values),
                    "state_region_primary": row["state_lesion_degradation_quality"],
                    "state_source_max": _decision(
                        searches["PRESENT"].score,
                        searches["ABSENT"].score,
                        group_positive.threshold,
                        group_negative.threshold,
                    ),
                    "source_max_positive_hierarchy": group_positive.used_hierarchy,
                    "source_max_negative_hierarchy": group_negative.used_hierarchy,
                }
            )
            qc_images[lesion_class][stratum].append(
                (
                    case_id,
                    np.stack(
                        [observed, searches["PRESENT"].candidate, searches["ABSENT"].candidate]
                    ),
                )
            )
            result_rows.append(row)
    _apply_multi_region(result_rows, config)
    v2_decoded: object = json.loads(config.bindings["s6_v2_results"].read_text(encoding="utf-8"))
    v2_states = {
        _string(_mapping(raw, "S6 v2 row").get("case_id"), "S6 case ID"): _string(
            _mapping(raw, "S6 v2 row").get("state"), "S6 state"
        )
        for raw in _sequence(_mapping(v2_decoded, "S6 v2 results").get("records"), "S6 rows")
    }
    shared_in_domain = [
        row
        for row in result_rows
        if row["case_id"] in v2_states
        and row["stratum"] in {"PRESENT_ID", "ABSENT_ID", "S5_TIER_A_AMBIGUOUS_ID"}
    ]
    agreement_count = sum(
        row["state_region_primary"] == v2_states[cast(str, row["case_id"])]
        for row in shared_in_domain
    )
    agreement_rate = agreement_count / len(shared_in_domain) if shared_in_domain else 0.0
    moderate_coverage: dict[str, float] = {}
    for lesion_class in LESION_CLASSES:
        moderate = [
            row
            for row in result_rows
            if row["lesion_class"] == lesion_class and row["stratum"] in MODERATE_STRATA
        ]
        moderate_coverage[lesion_class] = sum(
            row["state_region_primary"] in DETERMINATE_STATES for row in moderate
        ) / len(moderate)
    state_keys = {
        "UNADJUSTED_REGION": "state_region_primary",
        "BONFERRONI": "state_bonferroni",
        "HOLM": "state_holm",
        "SOURCE_MAX": "state_source_max",
    }
    policy_metrics = {
        policy: _policy_metrics(result_rows, state_key) for policy, state_key in state_keys.items()
    }
    hierarchy_metrics = {
        hierarchy: _policy_metrics(result_rows, f"state_{hierarchy.lower()}")
        for hierarchy in CALIBRATION_HIERARCHIES
    }
    primary_fcr_raw = policy_metrics["UNADJUSTED_REGION"]["development_fcr"]
    primary_fcr = float(primary_fcr_raw) if isinstance(primary_fcr_raw, int | float) else math.inf
    ood_rows = [row for row in result_rows if row["expected_truth"] == "OOD"]
    ood_fail_closed = sum(
        row["state_region_primary"] == EntailmentState.OUT_OF_MODEL.value for row in ood_rows
    )
    replay_rate = witness_passes / witness_checks if witness_checks else 0.0
    predicate_rate = predicate_passes / predicate_checks if predicate_checks else 0.0
    failure_audit = _failure_injection_audit()
    automated_gate = (
        replay_rate == 1.0
        and predicate_rate == 1.0
        and ood_fail_closed == len(ood_rows)
        and source_leaks == 0
        and completed_searches == zero_gap_searches
        and primary_fcr <= config.gate_max_fcr
        and all(value >= config.gate_min_moderate_coverage for value in moderate_coverage.values())
        and agreement_rate >= config.gate_min_verify_agreement
        and all(value == EntailmentState.OUT_OF_MODEL.value for value in failure_audit.values())
    )
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "alpha": config.alpha,
        "rank_rule": "CEIL_N_PLUS_1_TIMES_ONE_MINUS_ALPHA",
        "minimum_region_units": config.minimum_region_units,
        "minimum_group_units": config.minimum_group_units,
        "rows": [
            {
                "lesion_class": row.lesion_class,
                "hypothesis": row.hypothesis,
                "degradation": row.degradation,
                "quality": row.quality,
                "source_key": row.source_key,
                "twin_id": row.twin_id,
                "score": row.score,
            }
            for row in calibration_rows
        ],
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": "TRAIN_CALIBRATION_DEVELOPMENT_ONLY",
        "primary_region_hierarchy": "LESION_DEGRADATION_QUALITY_WITH_FROZEN_FALLBACK",
        "records": result_rows,
    }
    encoder_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "FROZEN_INTERPRETABLE_MULTISCALE_NUMPY_V1",
        "input_shape": [32, 32, 3],
        "feature_dimensions": 98,
        "components": [
            "4x4_rgb_block_means",
            "rgb_global_mean_std",
            "8_bin_rgb_histograms",
            "4x4_green_gradient_means",
            "central_context_rgb_contrast",
            "saturation_fraction",
        ],
        "learned_parameters": 0,
        "device": "CPU",
    }
    artifacts["conditional-calibration-v1.json"] = _canonical(calibration_payload)
    artifacts["lite-results-v1.json"] = _canonical(result_payload)
    artifacts["frozen-encoder-spec-v1.json"] = _canonical(encoder_payload)
    for lesion_class in LESION_CLASSES:
        panels: list[tuple[str, NDArray[np.uint8]]] = []
        for stratum in S7_STRATA:
            selected = sorted(qc_images[lesion_class][stratum], key=lambda item: item[0])[
                : config.qc_cases_per_stratum
            ]
            for _case_id_value, images in selected:
                panels.extend(
                    [
                        (f"{stratum[:12]} obs", images[0]),
                        (f"{stratum[:12]} +", images[1]),
                        (f"{stratum[:12]} -", images[2]),
                    ]
                )
        artifacts[f"qc-{lesion_class.lower()}-lite-contact-sheet.png"] = make_labeled_contact_sheet(
            panels, columns=3, panel_size=128
        )
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if automated_gate else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "scope": {
            "splits": ["TRAIN", "CALIBRATION"],
            "final_test_status": "NOT_OPENED",
            "paper_performance": False,
            "development_fcr_only": True,
        },
        "selection": {
            "bank_entries_per_class": {
                lesion_class: len(banks[lesion_class]) for lesion_class in LESION_CLASSES
            },
            "records": len(result_rows),
            "strata_per_class": {
                lesion_class: dict(
                    sorted(
                        Counter(
                            cast(str, row["stratum"])
                            for row in result_rows
                            if row["lesion_class"] == lesion_class
                        ).items()
                    )
                )
                for lesion_class in LESION_CLASSES
            },
        },
        "encoder": encoder_payload,
        "search": {
            "completed_searches": completed_searches,
            "finite_zero_gap_searches": zero_gap_searches,
            "candidate_grid_per_retrieved_exemplar": (
                len(config.traces)
                * len(config.translations) ** 2
                * len(config.gains)
                * len(config.biases)
            ),
            "source_leaks": source_leaks,
            "global_claim": "RETRIEVED_FINITE_LITE_HYPOTHESIS_CLASS_ONLY",
        },
        "calibration": {
            "search_rows": len(calibration_rows),
            "hierarchies": list(CALIBRATION_HIERARCHIES),
            "primary": "LESION_DEGRADATION_QUALITY_WITH_FROZEN_FALLBACK",
            "hierarchy_metrics": hierarchy_metrics,
        },
        "multi_region": {"policy_metrics": policy_metrics},
        "witness": {
            "replay_checks": witness_checks,
            "replay_passes": witness_passes,
            "replay_rate": replay_rate,
            "predicate_checks": predicate_checks,
            "predicate_passes": predicate_passes,
            "predicate_rate": predicate_rate,
        },
        "development_gate_metrics": {
            "development_fcr": None if not math.isfinite(primary_fcr) else primary_fcr,
            "cluster_bootstrap": _cluster_bootstrap_fcr(
                result_rows,
                "state_region_primary",
                config.seed,
                config.bootstrap_replicates,
            ),
            "moderate_glare_determinate_coverage_per_class": moderate_coverage,
            "lite_verify_shared_in_domain_records": len(shared_in_domain),
            "lite_verify_exact_agreement_count": agreement_count,
            "lite_verify_exact_agreement_rate": agreement_rate,
            "ood_records": len(ood_rows),
            "ood_out_of_model": ood_fail_closed,
        },
        "failure_injection_states": failure_audit,
        "resource": {
            "device": "CPU",
            "gpu_used": False,
            "learned_parameters": 0,
            "peak_python_tracemalloc_bytes": peak_bytes,
            "wall_seconds": elapsed,
            "silent_resolution_change": False,
        },
        "io_audit": {
            "s4_twin_png_decodes": io_audit.s4_twin_png_decodes,
            "s5_witness_png_decodes": io_audit.s5_witness_png_decodes,
            "witness_replay_png_decodes": io_audit.witness_replay_png_decodes,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "source_split_leaks": source_leaks,
        },
        "gate": {
            "maximum_development_fcr": config.gate_max_fcr,
            "minimum_moderate_coverage_per_class": config.gate_min_moderate_coverage,
            "minimum_lite_verify_in_domain_agreement": config.gate_min_verify_agreement,
            "automated": "PASS" if automated_gate else "FAIL",
            "ai_visual_qc": "PENDING",
            "s8_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "encoder_spec_sha256": sha256_bytes(artifacts["frozen-encoder-spec-v1.json"]),
            "calibration_sha256": sha256_bytes(artifacts["conditional-calibration-v1.json"]),
            "results_sha256": sha256_bytes(artifacts["lite-results-v1.json"]),
        },
        "evidence_boundary": [
            "All outcomes are TRAIN/CALIBRATION development diagnostics.",
            "The Lite optimum is exact only inside the declared retrieved finite hypothesis class.",
            "The development FCR diagnostic is not a final-test, clinical, medical, "
            "or paper result.",
            "No enhancer was trained and no final locked test was opened.",
        ],
    }
    artifacts["audit.json"] = _canonical(audit)
    _write_artifacts(output, artifacts)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = S7Config.load(args.config)
        audit = run_s7(config, output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"S7 Lite failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
