"""Science S6 RetinaEntail-Verify v1 development-only finite-grid runner.

The runner consumes only hash-bound S4/S5 generated evidence.  It never accepts a
dataset root or a locked-test root.  Its zero optimization gap is global only inside
the explicitly retrieved finite hypothesis grid, not over the image space.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import (
    OperationTrace,
    apply_and_record,
    replay_trace,
)
from retinaentail.entailment.states import EntailmentState, fail_closed_state
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _decode_twin_png,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
    _twin_output,
    _write_artifacts,
    make_labeled_contact_sheet,
    normalize_regional_observation,
    regional_context_bbox,
)
from retinaentail.science_s5_degradation_calibration_v2 import frozen_family_parameters
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S6_RETINAENTAIL_VERIFY_V1"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
LESION_CLASSES: Final = ("HE", "EX", "MA", "SE")
CANONICAL_SPLITS: Final = ("TRAIN", "CALIBRATION")
IN_DOMAIN_STRATA: Final = (
    "PRESENT_ID",
    "ABSENT_ID",
    "S5_TIER_A_AMBIGUOUS_ID",
)
OOD_STRATA: Final = ("NEAR_OOD", "FAR_OOD")
ALL_STRATA: Final = (*IN_DOMAIN_STRATA, *OOD_STRATA)
HYPOTHESES: Final = ("PRESENT", "ABSENT")


class S6VerifyError(ValueError):
    """Raised when the frozen S6 contract cannot be executed safely."""


def _sha256_text(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S6VerifyError(f"{name} is not SHA-256")
    return text


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S6VerifyError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S6VerifyError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S6VerifyError(f"{name} must be finite and >= {minimum}")
    return result


def _verify_binding(binding: Mapping[str, object], name: str) -> Path:
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256_text(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S6VerifyError(f"binding mismatch: {name}")
    return path


def _verify_baseline(expected: object) -> None:
    commit = _string(expected, "expected baseline commit")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != commit:
        raise S6VerifyError(f"baseline drift: expected {commit}, got {completed.stdout.strip()}")


def _tuple_ints(value: object, name: str) -> tuple[int, ...]:
    raw = _sequence(value, name)
    return tuple(_integer(item, f"{name} item", -1000) for item in raw)


def _tuple_numbers(value: object, name: str) -> tuple[float, ...]:
    raw = _sequence(value, name)
    return tuple(_number(item, f"{name} item", -1000.0) for item in raw)


@dataclass(frozen=True, slots=True)
class S6Config:
    """Validated, hash-bound S6 development contract."""

    path: Path
    payload: dict[str, object]
    config_sha256: str
    seed: int
    regions_per_class: int
    per_stratum: int
    bank_size_per_class: int
    bank_max_per_source: int
    retrieval_top_k: int
    context_margin: int
    observation_wh: tuple[int, int]
    translations: tuple[int, ...]
    gains: tuple[float, ...]
    biases: tuple[int, ...]
    score_weights: dict[str, float]
    threshold_quantile: float
    threshold_multiplier: float
    minimum_replay_rate: float
    minimum_predicate_rate: float
    qc_cases_per_stratum: int
    binding_paths: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> S6Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S6 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_DEVELOPMENT_VERIFY"
        ):
            raise S6VerifyError("unsupported S6 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        scope = _mapping(root.get("scope"), "scope")
        if scope != {
            "splits": ["TRAIN", "CALIBRATION"],
            "locked_test_reads": 0,
            "final_test_status": "NOT_OPENED",
            "oracle_region_only": True,
            "paper_performance": False,
        }:
            raise S6VerifyError("S6 scope changed")
        selection = _mapping(root.get("selection"), "selection")
        strata = _mapping(selection.get("strata_per_class"), "strata")
        if set(strata) != set(ALL_STRATA) or any(
            _integer(strata.get(name), f"stratum {name}", 1) != 20 for name in ALL_STRATA
        ):
            raise S6VerifyError("S6 strata changed")
        hypothesis = _mapping(root.get("hypothesis_class"), "hypothesis class")
        geometry = _mapping(hypothesis.get("geometry"), "geometry")
        color = _mapping(hypothesis.get("color"), "color")
        degradation = _mapping(hypothesis.get("degradation"), "degradation")
        if (
            hypothesis.get("bank_policy") != "SOURCE_ISOLATED_HASH_BANK__BACKGROUND_TOP_K_RETRIEVAL"
            or hypothesis.get("finite_grid_scope")
            != "RETRIEVED_EXEMPLARS_X_TRACE_X_TRANSLATION_X_GAIN_X_BIAS"
            or degradation.get("id_traces") != ["IDENTITY", "S5_TARGET_GLARE_SEVERITY_1"]
            or degradation.get("near_ood") != "HAZE_SEVERITY_0P8_ABOVE_ID_MAX_0P6"
            or degradation.get("far_ood") != "UNREGISTERED_INTENSITY_INVERSION"
        ):
            raise S6VerifyError("S6 hypothesis/degradation class changed")
        translations = _tuple_ints(geometry.get("translations_xy_px"), "translations")
        gains = _tuple_numbers(color.get("gains"), "gains")
        biases = _tuple_ints(color.get("biases_uint8"), "biases")
        if translations != (-1, 0, 1) or gains != (0.98, 1.0, 1.02) or biases != (-2, 0, 2):
            raise S6VerifyError("S6 geometry/color grid changed")
        scoring = _mapping(root.get("scoring"), "scoring")
        weights_raw = _mapping(scoring.get("weights"), "score weights")
        required_terms = {"pixel", "frequency", "vessel", "lesion", "degradation"}
        if set(weights_raw) != required_terms:
            raise S6VerifyError("S6 score terms changed")
        weights = {
            name: _number(weights_raw.get(name), f"weight {name}") for name in required_terms
        }
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise S6VerifyError("S6 score weights must sum to one")
        solver = _mapping(root.get("solver"), "solver")
        if (
            solver.get("strategy") != "EXHAUSTIVE_RETRIEVED_FINITE_GRID"
            or solver.get("lower_bound") != "EXACT_FINITE_GRID_MINIMUM"
            or solver.get("global_claim") != "RETRIEVED_FINITE_HYPOTHESIS_CLASS_ONLY"
        ):
            raise S6VerifyError("S6 solver contract changed")
        gate = _mapping(root.get("gate"), "gate")
        bindings = _mapping(root.get("bindings"), "bindings")
        binding_paths = {
            name: _verify_binding(_mapping(raw, f"binding {name}"), name)
            for name, raw in bindings.items()
        }
        required_bindings = {
            "execution_prompt",
            "s6_code",
            "s6_audit_schema",
            "s4_v1_twin_manifest",
            "s4_v1_artifact_manifest",
            "s4_v5_twin_manifest",
            "s4_v5_artifact_manifest",
            "s5_v2_config",
            "s5_v2_report",
            "s5_v2_ai_qc",
            "s5_v2_audit",
            "s5_v2_artifact_manifest",
            "s5_v2_degradation_domain",
            "s5_v2_regional_oracle",
            "preregistration_v2",
            "controlled_twin_oracle_spec",
        }
        if set(binding_paths) != required_bindings:
            raise S6VerifyError("S6 binding set changed")
        observation = _tuple_ints(hypothesis.get("observation_wh"), "observation size")
        if observation != (32, 32):
            raise S6VerifyError("S6 observation size changed")
        return cls(
            path=resolved,
            payload=dict(root),
            config_sha256=sha256_file(resolved),
            seed=_integer(selection.get("seed"), "selection seed"),
            regions_per_class=_integer(selection.get("regions_per_class"), "regions per class", 1),
            per_stratum=20,
            bank_size_per_class=_integer(
                hypothesis.get("bank_size_per_class"), "bank size per class", 2
            ),
            bank_max_per_source=_integer(
                hypothesis.get("bank_max_per_source"), "bank maximum per source", 1
            ),
            retrieval_top_k=_integer(hypothesis.get("retrieval_top_k"), "retrieval top k", 1),
            context_margin=_integer(hypothesis.get("context_margin_px"), "context margin", 1),
            observation_wh=observation,
            translations=translations,
            gains=gains,
            biases=biases,
            score_weights=weights,
            threshold_quantile=_number(scoring.get("threshold_quantile"), "threshold quantile"),
            threshold_multiplier=_number(
                scoring.get("threshold_multiplier"), "threshold multiplier", 1.0
            ),
            minimum_replay_rate=_number(gate.get("minimum_dual_witness_replay"), "replay gate"),
            minimum_predicate_rate=_number(
                gate.get("minimum_predicate_validity"), "predicate gate"
            ),
            qc_cases_per_stratum=_integer(
                _mapping(root.get("qc"), "QC").get("cases_per_stratum"),
                "QC cases per stratum",
                1,
            ),
            binding_paths=binding_paths,
        )


@dataclass(frozen=True, slots=True)
class TwinRecord:
    package_id: str
    root: Path
    raw: Mapping[str, object]
    twin_id: str
    lesion_class: str
    split: str
    source_image_id: str

    @property
    def source_key(self) -> str:
        return f"{self.package_id}:{self.source_image_id}"


@dataclass(frozen=True, slots=True)
class SelectedCase:
    record: TwinRecord
    stratum: str


@dataclass(frozen=True, slots=True)
class Variant:
    array: NDArray[np.uint8]
    trace: OperationTrace


@dataclass(frozen=True, slots=True)
class BankEntry:
    record: TwinRecord
    variants: Mapping[str, Mapping[str, Variant]]


@dataclass(frozen=True, slots=True)
class SearchResult:
    hypothesis: str
    score: float
    terms: Mapping[str, float]
    lower_bound: float
    upper_bound: float
    gap: float
    candidate_count: int
    retrieved_ids: tuple[str, ...]
    anchor: TwinRecord
    trace_name: str
    trace: OperationTrace
    dx: int
    dy: int
    gain: float
    bias: int
    candidate: NDArray[np.uint8]


@dataclass(slots=True)
class IOAudit:
    s4_twin_png_decodes: int = 0
    s5_witness_png_decodes: int = 0
    witness_replay_png_decodes: int = 0


def _digest(seed: int, *parts: object) -> str:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_package(path: Path, package_id: str) -> list[TwinRecord]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(decoded, f"{package_id} manifest")
    accepted = _sequence(root.get("accepted"), f"{package_id} accepted")
    records: list[TwinRecord] = []
    for raw in accepted:
        item = _mapping(raw, f"{package_id} twin")
        lesion_class = _string(item.get("lesion_class"), "lesion class")
        split = _string(item.get("split"), "split")
        if lesion_class not in LESION_CLASSES or split not in CANONICAL_SPLITS:
            raise S6VerifyError("S4 twin class/split is outside the S6 scope")
        records.append(
            TwinRecord(
                package_id=package_id,
                root=path.parent,
                raw=item,
                twin_id=_string(item.get("twin_id"), "twin ID"),
                lesion_class=lesion_class,
                split=split,
                source_image_id=_string(item.get("source_image_id"), "source image ID"),
            )
        )
    return records


def load_inputs(
    config: S6Config,
) -> tuple[list[TwinRecord], dict[str, Mapping[str, object]], Mapping[str, object]]:
    """Load only hash-bound S4/S5 generated manifests."""

    records = [
        *_load_package(config.binding_paths["s4_v1_twin_manifest"], "S4_IDRID_V1"),
        *_load_package(config.binding_paths["s4_v5_twin_manifest"], "S4_DDR_V5"),
    ]
    by_id: dict[str, TwinRecord] = {}
    for record in records:
        if record.twin_id in by_id:
            raise S6VerifyError(f"duplicate S4 twin ID: {record.twin_id}")
        by_id[record.twin_id] = record
    oracle_decoded: object = json.loads(
        config.binding_paths["s5_v2_regional_oracle"].read_text(encoding="utf-8")
    )
    oracle_root = _mapping(oracle_decoded, "S5 Oracle")
    collisions: dict[str, Mapping[str, object]] = {}
    for raw in _sequence(oracle_root.get("records"), "S5 Oracle records"):
        item = _mapping(raw, "S5 Oracle record")
        if item.get("support_truth") != "OBSERVATION_AMBIGUOUS":
            continue
        twin_id = _string(item.get("twin_id"), "S5 twin ID")
        if twin_id not in by_id or by_id[twin_id].package_id != "S4_DDR_V5":
            raise S6VerifyError("S5 collision does not bind an S4 DDR v5 twin")
        collisions[twin_id] = item
    domain_decoded: object = json.loads(
        config.binding_paths["s5_v2_degradation_domain"].read_text(encoding="utf-8")
    )
    domain = _mapping(domain_decoded, "S5 degradation domain")
    if domain.get("domain_gate") is not True:
        raise S6VerifyError("S5 degradation-domain gate is not PASS")
    families = _mapping(domain.get("families"), "S5 families")
    haze = _mapping(families.get("haze_veil"), "S5 haze family")
    if (
        max(
            _number(value, "haze severity")
            for value in _sequence(haze.get("id_severity_grid"), "haze grid")
        )
        != 0.6
    ):
        raise S6VerifyError("S5 haze ID maximum changed")
    return records, collisions, domain


def _eligible(record: TwinRecord, config: S6Config) -> bool:
    try:
        left, top, right, bottom = regional_context_bbox(
            record.raw, margin=config.context_margin, patch_wh=(256, 256)
        )
    except ValueError:
        return False
    return left > 0 and top > 0 and right < 256 and bottom < 256


def select_bank_and_cases(
    records: Sequence[TwinRecord],
    collisions: Mapping[str, Mapping[str, object]],
    config: S6Config,
) -> tuple[dict[str, list[TwinRecord]], dict[str, list[SelectedCase]]]:
    """Freeze source-isolated bank and exactly 100 unique regions per class."""

    banks: dict[str, list[TwinRecord]] = {}
    cases: dict[str, list[SelectedCase]] = {}
    for lesion_class in LESION_CLASSES:
        class_records = [
            record
            for record in records
            if record.lesion_class == lesion_class and _eligible(record, config)
        ]
        collision_records = [record for record in class_records if record.twin_id in collisions]
        collision_sources = {record.source_key for record in collision_records}
        ordered_bank = sorted(
            (record for record in class_records if record.split == "TRAIN"),
            key=lambda record: (
                record.source_key in collision_sources,
                _digest(config.seed, "BANK", lesion_class, record.source_key, record.twin_id),
            ),
        )
        bank: list[TwinRecord] = []
        per_source: Counter[str] = Counter()
        for record in ordered_bank:
            if per_source[record.source_key] >= config.bank_max_per_source:
                continue
            remaining_collision_sources = collision_sources.difference(
                {item.source_key for item in bank}, {record.source_key}
            )
            if record.source_key in collision_sources and len(remaining_collision_sources) < 20:
                continue
            bank.append(record)
            per_source[record.source_key] += 1
            if len(bank) == config.bank_size_per_class:
                break
        if len(bank) != config.bank_size_per_class or len(per_source) < 2:
            raise S6VerifyError(f"insufficient source-diverse {lesion_class} exemplar bank")
        bank_sources = set(per_source)
        ambiguous = sorted(
            (record for record in collision_records if record.source_key not in bank_sources),
            key=lambda record: _digest(config.seed, "AMBIGUOUS", lesion_class, record.twin_id),
        )[: config.per_stratum]
        if len(ambiguous) != config.per_stratum:
            raise S6VerifyError(f"insufficient source-isolated {lesion_class} S5 collisions")
        selected_ids = {record.twin_id for record in ambiguous}
        remainder = sorted(
            (
                record
                for record in class_records
                if record.source_key not in bank_sources and record.twin_id not in selected_ids
            ),
            key=lambda record: _digest(config.seed, "PROBE", lesion_class, record.twin_id),
        )
        required_remainder = config.regions_per_class - config.per_stratum
        if len(remainder) < required_remainder:
            raise S6VerifyError(f"insufficient source-isolated {lesion_class} development probes")
        strata_records: dict[str, list[TwinRecord]] = {
            "S5_TIER_A_AMBIGUOUS_ID": ambiguous,
        }
        cursor = 0
        for stratum in ("PRESENT_ID", "ABSENT_ID", "NEAR_OOD", "FAR_OOD"):
            strata_records[stratum] = remainder[cursor : cursor + config.per_stratum]
            cursor += config.per_stratum
        class_cases = [
            SelectedCase(record=record, stratum=stratum)
            for stratum in ALL_STRATA
            for record in strata_records[stratum]
        ]
        if len({case.record.twin_id for case in class_cases}) != config.regions_per_class:
            raise S6VerifyError("S6 case selection contains duplicate target regions")
        if any(case.record.source_key in bank_sources for case in class_cases):
            raise S6VerifyError("S6 bank/probe source isolation failed")
        banks[lesion_class] = bank
        cases[lesion_class] = class_cases
    return banks, cases


def _decode_side(record: TwinRecord, hypothesis: str, io_audit: IOAudit) -> NDArray[np.uint8]:
    name = "x_plus_sham" if hypothesis == "PRESENT" else "x_minus_removed"
    path, expected = _twin_output(record.raw, name, record.root)
    array = _decode_twin_png(path, expected)
    io_audit.s4_twin_png_decodes += 1
    return array


def _target_glare_trace(
    record: TwinRecord, config: S6Config, image: NDArray[np.uint8]
) -> OperationTrace:
    bbox = regional_context_bbox(record.raw, margin=config.context_margin, patch_wh=(256, 256))
    parameters = cast(
        dict[str, str | int | float | bool],
        frozen_family_parameters("glare_reflection", 1.0),
    )
    parameters = {
        **parameters,
        "center_x": ((bbox[0] + bbox[2]) / 2.0) / image.shape[1],
        "center_y": ((bbox[1] + bbox[3]) / 2.0) / image.shape[0],
    }
    seed = int(_digest(config.seed, "BANK_GLARE", record.twin_id)[:16], 16)
    _, step = apply_and_record(
        image,
        "glare_reflection",
        severity=1.0,
        seed=seed,
        parameters=parameters,
    )
    return OperationTrace((step,))


def build_bank(
    records: Mapping[str, Sequence[TwinRecord]], config: S6Config, io_audit: IOAudit
) -> dict[str, list[BankEntry]]:
    """Decode and freeze identity/glare variants for both hypotheses."""

    result: dict[str, list[BankEntry]] = {}
    for lesion_class, class_records in records.items():
        entries: list[BankEntry] = []
        for record in class_records:
            by_hypothesis: dict[str, Mapping[str, Variant]] = {}
            for hypothesis in HYPOTHESES:
                source = _decode_side(record, hypothesis, io_audit)
                bbox = regional_context_bbox(
                    record.raw, margin=config.context_margin, patch_wh=(256, 256)
                )
                identity = normalize_regional_observation(source, bbox, config.observation_wh)
                glare_trace = _target_glare_trace(record, config, source)
                glare = normalize_regional_observation(
                    replay_trace(source, glare_trace), bbox, config.observation_wh
                )
                by_hypothesis[hypothesis] = {
                    "IDENTITY": Variant(identity, OperationTrace()),
                    "S5_TARGET_GLARE_SEVERITY_1": Variant(glare, glare_trace),
                }
            entries.append(BankEntry(record=record, variants=by_hypothesis))
        result[lesion_class] = entries
    return result


def _core_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    height, width = shape
    y, x = np.ogrid[:height, :width]
    return ((x - (width - 1) / 2.0) / (0.27 * width)) ** 2 + (
        (y - (height - 1) / 2.0) / (0.27 * height)
    ) ** 2 <= 1.0


def _sobel_green(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    green = image[:, :, 1].astype(np.float64)
    padded = np.pad(green, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.hypot(gx, gy)


def score_terms(observed: NDArray[np.uint8], candidate: NDArray[np.uint8]) -> dict[str, float]:
    """Return the five frozen normalized discrepancy terms."""

    if (
        observed.shape != candidate.shape
        or observed.dtype != np.uint8
        or candidate.dtype != np.uint8
    ):
        raise S6VerifyError("S6 score inputs must be shape-matched RGB uint8")
    observed_float = observed.astype(np.float64)
    candidate_float = candidate.astype(np.float64)
    difference = np.abs(observed_float - candidate_float)
    core = _core_mask(observed.shape[:2])
    observed_fft = np.log1p(np.abs(np.fft.rfft2(observed_float.mean(axis=2))))
    candidate_fft = np.log1p(np.abs(np.fft.rfft2(candidate_float.mean(axis=2))))
    frequency_scale = max(float(np.mean(observed_fft) + np.mean(candidate_fft)), 1.0)
    vessel_difference = np.abs(_sobel_green(observed) - _sobel_green(candidate))
    signatures = []
    for array in (observed_float, candidate_float):
        luminance = array.mean(axis=2)
        channel_range = array.max(axis=2) - array.min(axis=2)
        signatures.append(
            np.asarray(
                [
                    array.mean() / 255.0,
                    array.std() / 128.0,
                    np.mean(luminance >= 220.0),
                    np.mean(channel_range) / 255.0,
                ],
                dtype=np.float64,
            )
        )
    return {
        "pixel": float(np.mean(difference) / 255.0),
        "frequency": float(np.mean(np.abs(observed_fft - candidate_fft)) / frequency_scale),
        "vessel": float(np.mean(vessel_difference) / 1442.5),
        "lesion": float(np.mean(difference[core]) / 255.0),
        "degradation": float(np.mean(np.abs(signatures[0] - signatures[1]))),
    }


def weighted_score(terms: Mapping[str, float], weights: Mapping[str, float]) -> float:
    score = float(sum(weights[name] * terms[name] for name in weights))
    if not math.isfinite(score) or score < 0.0:
        raise S6VerifyError("S6 score is invalid")
    return score


def _background_distance(observed: NDArray[np.uint8], candidate: NDArray[np.uint8]) -> float:
    core = _core_mask(observed.shape[:2])
    difference = np.abs(observed.astype(np.int16) - candidate.astype(np.int16))
    outside = np.broadcast_to((~core)[:, :, None], difference.shape)
    return float(np.mean(difference[outside]) / 255.0)


def transform_candidate(
    image: NDArray[np.uint8], *, dx: int, dy: int, gain: float, bias: int
) -> NDArray[np.uint8]:
    """Apply one exact frozen integer-translation/color-grid transform."""

    height, width = image.shape[:2]
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    shifted = padded[1 - dy : 1 - dy + height, 1 - dx : 1 - dx + width]
    if dx == 0 and dy == 0 and gain == 1.0 and bias == 0:
        return shifted.copy()
    return np.clip(np.rint(shifted.astype(np.float64) * gain + bias), 0, 255).astype(np.uint8)


def search_finite_grid(
    observed: NDArray[np.uint8],
    entries: Sequence[BankEntry],
    hypothesis: str,
    config: S6Config,
    *,
    excluded_source_key: str,
) -> SearchResult:
    """Exhaust the declared retrieved finite grid and return its exact minimum."""

    eligible = [entry for entry in entries if entry.record.source_key != excluded_source_key]
    ranked = sorted(
        eligible,
        key=lambda entry: (
            _background_distance(observed, entry.variants[hypothesis]["IDENTITY"].array),
            _digest(config.seed, "RETRIEVE", hypothesis, entry.record.twin_id),
        ),
    )
    retrieved = ranked[: config.retrieval_top_k]
    if len(retrieved) != config.retrieval_top_k:
        raise S6VerifyError("insufficient source-isolated retrieved exemplars")
    best: (
        tuple[
            float,
            dict[str, float],
            BankEntry,
            str,
            Variant,
            int,
            int,
            float,
            int,
            NDArray[np.uint8],
        ]
        | None
    ) = None
    candidate_count = 0
    for entry in retrieved:
        for trace_name in ("IDENTITY", "S5_TARGET_GLARE_SEVERITY_1"):
            variant = entry.variants[hypothesis][trace_name]
            for dy in config.translations:
                for dx in config.translations:
                    for gain in config.gains:
                        for bias in config.biases:
                            candidate = transform_candidate(
                                variant.array, dx=dx, dy=dy, gain=gain, bias=bias
                            )
                            terms = score_terms(observed, candidate)
                            score = weighted_score(terms, config.score_weights)
                            candidate_count += 1
                            if best is None or (
                                score,
                                entry.record.twin_id,
                                trace_name,
                                dx,
                                dy,
                                gain,
                                bias,
                            ) < (
                                best[0],
                                best[2].record.twin_id,
                                best[3],
                                best[5],
                                best[6],
                                best[7],
                                best[8],
                            ):
                                best = (
                                    score,
                                    terms,
                                    entry,
                                    trace_name,
                                    variant,
                                    dx,
                                    dy,
                                    gain,
                                    bias,
                                    candidate,
                                )
    if best is None:
        raise S6VerifyError("finite-grid search produced no candidate")
    score, terms, entry, trace_name, variant, dx, dy, gain, bias, candidate = best
    return SearchResult(
        hypothesis=hypothesis,
        score=score,
        terms=terms,
        lower_bound=score,
        upper_bound=score,
        gap=0.0,
        candidate_count=candidate_count,
        retrieved_ids=tuple(item.record.twin_id for item in retrieved),
        anchor=entry.record,
        trace_name=trace_name,
        trace=variant.trace,
        dx=dx,
        dy=dy,
        gain=gain,
        bias=bias,
        candidate=candidate,
    )


def calibrate_thresholds(
    banks: Mapping[str, Sequence[BankEntry]], config: S6Config
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    """Fit same-hypothesis leave-one-source-out thresholds on the development bank."""

    thresholds: dict[str, dict[str, float]] = {}
    records: list[dict[str, object]] = []
    for lesion_class in LESION_CLASSES:
        class_thresholds: dict[str, float] = {}
        for hypothesis in HYPOTHESES:
            scores: list[float] = []
            for query in banks[lesion_class]:
                observed = query.variants[hypothesis]["IDENTITY"].array
                result = search_finite_grid(
                    observed,
                    banks[lesion_class],
                    hypothesis,
                    config,
                    excluded_source_key=query.record.source_key,
                )
                scores.append(result.score)
                records.append(
                    {
                        "lesion_class": lesion_class,
                        "hypothesis": hypothesis,
                        "query_twin_id": query.record.twin_id,
                        "query_source_key": query.record.source_key,
                        "best_anchor_twin_id": result.anchor.twin_id,
                        "best_anchor_source_key": result.anchor.source_key,
                        "score": result.score,
                        "candidate_count": result.candidate_count,
                    }
                )
            quantile = float(
                np.quantile(np.asarray(scores), config.threshold_quantile, method="higher")
            )
            class_thresholds[hypothesis] = max(1e-12, quantile * config.threshold_multiplier)
        thresholds[lesion_class] = class_thresholds
    return thresholds, records


def _png_bytes(image: NDArray[np.uint8]) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _load_ambiguous_observation(
    case: SelectedCase,
    collision: Mapping[str, object],
    config: S6Config,
    io_audit: IOAudit,
) -> tuple[NDArray[np.uint8], dict[str, object]]:
    evidence_root = config.binding_paths["s5_v2_regional_oracle"].parent / "oracle-witnesses"
    witness_dir = evidence_root / case.record.lesion_class.lower() / case.record.twin_id
    evidence_path = witness_dir / "evidence.json"
    if not evidence_path.is_file():
        raise S6VerifyError("S5 ambiguous evidence file is missing")
    evidence_decoded: object = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence = _mapping(evidence_decoded, "S5 ambiguous evidence")
    if (
        evidence.get("twin_id") != case.record.twin_id
        or evidence.get("support_truth") != "OBSERVATION_AMBIGUOUS"
        or evidence.get("evidence_sha256") != collision.get("evidence_sha256")
    ):
        raise S6VerifyError("S5 ambiguous evidence binding mismatch")
    observations: list[NDArray[np.uint8]] = []
    for side in ("x_plus", "x_minus"):
        path = witness_dir / f"{side}.png"
        if not path.is_file():
            raise S6VerifyError("S5 ambiguous witness PNG is missing")
        with Image.open(path) as source:
            source.load()
            array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        io_audit.s5_witness_png_decodes += 1
        observations.append(array)
    if not np.array_equal(observations[0], observations[1]):
        raise S6VerifyError("S5 Tier-A ambiguous observations are not equal")
    plus = _decode_side(case.record, "PRESENT", io_audit)
    minus = _decode_side(case.record, "ABSENT", io_audit)
    bbox_raw = _sequence(evidence.get("regional_bbox_xyxy"), "S5 regional bbox")
    bbox = cast(
        tuple[int, int, int, int], tuple(_integer(value, "bbox coordinate") for value in bbox_raw)
    )
    plus_trace = OperationTrace.from_dict(
        cast(Mapping[str, object], _mapping(evidence.get("x_plus_trace"), "plus trace"))
    )
    minus_trace = OperationTrace.from_dict(
        cast(Mapping[str, object], _mapping(evidence.get("x_minus_trace"), "minus trace"))
    )
    plus_replay = normalize_regional_observation(
        replay_trace(plus, plus_trace), bbox, config.observation_wh
    )
    minus_replay = normalize_regional_observation(
        replay_trace(minus, minus_trace), bbox, config.observation_wh
    )
    if not np.array_equal(plus_replay, observations[0]) or not np.array_equal(
        minus_replay, observations[1]
    ):
        raise S6VerifyError("S5 ambiguous observation independent replay mismatch")
    return observations[0], {
        "family": "glare_reflection",
        "severity": 1.0,
        "domain": "ID",
        "support_truth": "OBSERVATION_AMBIGUOUS",
        "oracle_method_id": evidence.get("oracle_method_id"),
        "evidence_sha256": evidence.get("evidence_sha256"),
    }


def make_query_observation(
    case: SelectedCase,
    collisions: Mapping[str, Mapping[str, object]],
    config: S6Config,
    io_audit: IOAudit,
) -> tuple[NDArray[np.uint8], dict[str, object], bool]:
    """Materialize one frozen probe observation and its domain disposition."""

    record = case.record
    bbox = regional_context_bbox(record.raw, margin=config.context_margin, patch_wh=(256, 256))
    if case.stratum == "S5_TIER_A_AMBIGUOUS_ID":
        return (
            *_load_ambiguous_observation(case, collisions[record.twin_id], config, io_audit),
            True,
        )
    hypothesis = "ABSENT" if case.stratum == "ABSENT_ID" else "PRESENT"
    source = _decode_side(record, hypothesis, io_audit)
    if case.stratum in {"PRESENT_ID", "ABSENT_ID"}:
        return (
            normalize_regional_observation(source, bbox, config.observation_wh),
            {
                "family": "identity",
                "severity": 0.0,
                "domain": "ID",
                "source_truth": hypothesis,
            },
            True,
        )
    if case.stratum == "NEAR_OOD":
        parameters = cast(
            dict[str, str | int | float | bool], frozen_family_parameters("haze_veil", 0.8)
        )
        output, step = apply_and_record(
            source,
            "haze_veil",
            severity=0.8,
            seed=int(_digest(config.seed, "NEAR_OOD", record.twin_id)[:16], 16),
            parameters=parameters,
        )
        return (
            normalize_regional_observation(output, bbox, config.observation_wh),
            {
                "family": "haze_veil",
                "severity": 0.8,
                "domain": "NEAR_OOD",
                "trace": OperationTrace((step,)).to_dict(),
            },
            False,
        )
    if case.stratum == "FAR_OOD":
        output = np.subtract(255, source, dtype=np.uint8)
        return (
            normalize_regional_observation(output, bbox, config.observation_wh),
            {
                "family": "unregistered_intensity_inversion",
                "severity": None,
                "domain": "FAR_OOD",
            },
            False,
        )
    raise S6VerifyError(f"unknown S6 stratum: {case.stratum}")


def _predicate_valid(result: SearchResult) -> bool:
    gate = _mapping(result.anchor.raw.get("automated_gate"), "S4 automated gate")
    common = (
        gate.get("complete_pair") is True
        and gate.get("finite_uint8_outputs") is True
        and gate.get("outside_target_edit_exact_preservation") is True
        and gate.get("target_edit_other_lesion_overlap_px") == 0
    )
    if result.hypothesis == "PRESENT":
        return common and gate.get("target_exact_preservation_in_sham") is True
    return common and _integer(gate.get("x_minus_changed_pixel_count"), "minus changed", 1) > 0


def _witness_record(
    result: SearchResult,
    observed: NDArray[np.uint8],
    threshold: float,
    relative_path: str,
) -> dict[str, object]:
    output_path, output_sha = _twin_output(
        result.anchor.raw,
        "x_plus_sham" if result.hypothesis == "PRESENT" else "x_minus_removed",
        result.anchor.root,
    )
    return {
        "hypothesis": result.hypothesis,
        "feasible": result.score <= threshold,
        "score": result.score,
        "threshold": threshold,
        "score_terms": dict(result.terms),
        "upper_bound": result.upper_bound,
        "lower_bound": result.lower_bound,
        "optimization_gap": result.gap,
        "solver_status": "OPTIMAL_RETRIEVED_FINITE_GRID",
        "candidate_count": result.candidate_count,
        "retrieved_exemplar_ids": list(result.retrieved_ids),
        "anchor": {
            "package_id": result.anchor.package_id,
            "twin_id": result.anchor.twin_id,
            "source_key": result.anchor.source_key,
            "split": result.anchor.split,
            "input_relative_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
            "input_png_sha256": output_sha,
        },
        "trace_name": result.trace_name,
        "trace": result.trace.to_dict(),
        "geometry": {"dx": result.dx, "dy": result.dy},
        "color": {"gain": result.gain, "bias": result.bias},
        "observed_array_sha256": sha256_bytes(observed.tobytes()),
        "candidate_array_sha256": sha256_bytes(result.candidate.tobytes()),
        "candidate_png_path": relative_path,
        "candidate_png_sha256": sha256_bytes(_png_bytes(result.candidate)),
        "predicate_provenance_valid": _predicate_valid(result),
    }


def replay_witness(
    record: Mapping[str, object],
    result: SearchResult,
    observed: NDArray[np.uint8],
    config: S6Config,
    io_audit: IOAudit,
) -> bool:
    """Independently decode and reconstruct one best-candidate witness."""

    source = _decode_side(result.anchor, result.hypothesis, io_audit)
    io_audit.witness_replay_png_decodes += 1
    io_audit.s4_twin_png_decodes -= 1
    bbox = regional_context_bbox(
        result.anchor.raw, margin=config.context_margin, patch_wh=(256, 256)
    )
    degraded = replay_trace(source, result.trace)
    normalized = normalize_regional_observation(degraded, bbox, config.observation_wh)
    candidate = transform_candidate(
        normalized, dx=result.dx, dy=result.dy, gain=result.gain, bias=result.bias
    )
    terms = score_terms(observed, candidate)
    score = weighted_score(terms, config.score_weights)
    return (
        sha256_bytes(candidate.tobytes()) == record.get("candidate_array_sha256")
        and sha256_bytes(_png_bytes(candidate)) == record.get("candidate_png_sha256")
        and math.isclose(score, _number(record.get("score"), "witness score"), abs_tol=1e-15)
        and all(
            math.isclose(
                terms[name],
                _number(_mapping(record.get("score_terms"), "terms")[name], f"term {name}"),
                abs_tol=1e-15,
            )
            for name in config.score_weights
        )
    )


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
        "solver_status": "optimal",
        "degradation_known": True,
    }
    scenarios: dict[str, dict[str, object]] = {
        "missing_threshold": {**valid, "positive_threshold": None},
        "nonfinite_score": {**valid, "negative_score": math.nan},
        "unknown_degradation": {**valid, "degradation_known": False},
        "failed_solver": {**valid, "solver_status": "failed"},
        "missing_feasibility": {**valid, "positive_feasible": None},
    }
    results: dict[str, str] = {}
    for name, inputs in scenarios.items():
        results[name] = fail_closed_state(
            positive_feasible=cast(bool | None, inputs["positive_feasible"]),
            negative_feasible=cast(bool | None, inputs["negative_feasible"]),
            positive_score=cast(float | None, inputs["positive_score"]),
            negative_score=cast(float | None, inputs["negative_score"]),
            positive_threshold=cast(float | None, inputs["positive_threshold"]),
            negative_threshold=cast(float | None, inputs["negative_threshold"]),
            positive_solver_status=cast(str | None, inputs["positive_solver_status"]),
            negative_solver_status=cast(str | None, inputs["negative_solver_status"]),
            solver_status=cast(str | None, inputs["solver_status"]),
            degradation_known=cast(bool, inputs["degradation_known"]),
        ).value
    return results


def run_s6(config: S6Config, *, output_dir: Path) -> dict[str, object]:
    """Execute S6 v1 without any raw dataset or final-test access."""

    output = output_dir.resolve()
    if output.exists():
        raise S6VerifyError("S6 output directory already exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    io_audit = IOAudit()
    records, collisions, _domain = load_inputs(config)
    selected_banks, selected_cases = select_bank_and_cases(records, collisions, config)
    banks = build_bank(selected_banks, config, io_audit)
    thresholds, threshold_rows = calibrate_thresholds(banks, config)
    artifacts: dict[str, bytes] = {}
    result_rows: list[dict[str, object]] = []
    state_counts: dict[str, Counter[str]] = {
        lesion_class: Counter() for lesion_class in LESION_CLASSES
    }
    stratum_counts: dict[str, Counter[str]] = {
        lesion_class: Counter() for lesion_class in LESION_CLASSES
    }
    replay_checks = 0
    replay_passes = 0
    predicate_checks = 0
    predicate_passes = 0
    completed_searches = 0
    finite_gap_searches = 0
    ood_cases = 0
    ood_out_of_model = 0
    ambiguity_cases = 0
    ambiguity_raw_unidentifiable = 0
    qc_rows: dict[str, dict[str, list[tuple[str, NDArray[np.uint8]]]]] = {
        lesion_class: {stratum: [] for stratum in ALL_STRATA} for lesion_class in LESION_CLASSES
    }
    for lesion_class in LESION_CLASSES:
        for case in selected_cases[lesion_class]:
            observed, trace_record, domain_known = make_query_observation(
                case, collisions, config, io_audit
            )
            case_id = f"s6-{lesion_class.lower()}-{case.stratum.lower()}-{case.record.twin_id}"
            observation_relative = f"cases/{lesion_class.lower()}/{case_id}/observation.png"
            artifacts[observation_relative] = _png_bytes(observed)
            row: dict[str, object] = {
                "case_id": case_id,
                "twin_id": case.record.twin_id,
                "component_id": case.record.raw.get("component_id"),
                "package_id": case.record.package_id,
                "source_image_id": case.record.source_image_id,
                "source_key": case.record.source_key,
                "lesion_class": lesion_class,
                "split": case.record.split,
                "stratum": case.stratum,
                "observation_trace": trace_record,
                "observation_array_sha256": sha256_bytes(observed.tobytes()),
                "observation_png_path": observation_relative,
                "observation_png_sha256": sha256_bytes(artifacts[observation_relative]),
            }
            stratum_counts[lesion_class][case.stratum] += 1
            if not domain_known:
                state = fail_closed_state(
                    positive_feasible=None,
                    negative_feasible=None,
                    positive_score=None,
                    negative_score=None,
                    positive_threshold=None,
                    negative_threshold=None,
                    positive_solver_status=None,
                    negative_solver_status=None,
                    solver_status="failed",
                    degradation_known=False,
                )
                row.update(
                    {
                        "state": state.value,
                        "internal_status": "FAIL_CLOSED_UNKNOWN_DEGRADATION",
                        "failure_reasons": [
                            "NEAR_OOD_ABOVE_FROZEN_ID_MAXIMUM"
                            if case.stratum == "NEAR_OOD"
                            else "UNREGISTERED_DEGRADATION_FAMILY"
                        ],
                        "positive_search": None,
                        "negative_search": None,
                    }
                )
                ood_cases += 1
                ood_out_of_model += int(state is EntailmentState.OUT_OF_MODEL)
                placeholder = np.zeros_like(observed)
                if len(qc_rows[lesion_class][case.stratum]) < config.qc_cases_per_stratum:
                    qc_rows[lesion_class][case.stratum].extend(
                        [
                            (f"{case.stratum} obs", observed),
                            ("positive OOM", placeholder),
                            ("negative OOM", placeholder),
                        ]
                    )
            else:
                search_results: dict[str, SearchResult] = {}
                witness_records: dict[str, dict[str, object]] = {}
                for hypothesis in HYPOTHESES:
                    result = search_finite_grid(
                        observed,
                        banks[lesion_class],
                        hypothesis,
                        config,
                        excluded_source_key=case.record.source_key,
                    )
                    search_results[hypothesis] = result
                    candidate_relative = (
                        f"cases/{lesion_class.lower()}/{case_id}/best-{hypothesis.lower()}.png"
                    )
                    artifacts[candidate_relative] = _png_bytes(result.candidate)
                    witness = _witness_record(
                        result,
                        observed,
                        thresholds[lesion_class][hypothesis],
                        candidate_relative,
                    )
                    witness_records[hypothesis] = witness
                    replay_checks += 1
                    replay_ok = replay_witness(witness, result, observed, config, io_audit)
                    replay_passes += int(replay_ok)
                    witness["independent_replay_exact"] = replay_ok
                    predicate_checks += 1
                    predicate_passes += int(witness["predicate_provenance_valid"] is True)
                    completed_searches += 1
                    finite_gap_searches += int(
                        math.isfinite(result.lower_bound)
                        and math.isfinite(result.upper_bound)
                        and result.gap == 0.0
                    )
                positive = witness_records["PRESENT"]
                negative = witness_records["ABSENT"]
                state = fail_closed_state(
                    positive_feasible=cast(bool, positive["feasible"]),
                    negative_feasible=cast(bool, negative["feasible"]),
                    positive_score=cast(float, positive["score"]),
                    negative_score=cast(float, negative["score"]),
                    positive_threshold=cast(float, positive["threshold"]),
                    negative_threshold=cast(float, negative["threshold"]),
                    positive_solver_status="optimal",
                    negative_solver_status="optimal",
                    solver_status="optimal",
                    degradation_known=True,
                )
                row.update(
                    {
                        "state": state.value,
                        "internal_status": "COMPLETED_RETRIEVED_FINITE_GRID",
                        "failure_reasons": [],
                        "positive_search": positive,
                        "negative_search": negative,
                    }
                )
                if case.stratum == "S5_TIER_A_AMBIGUOUS_ID":
                    ambiguity_cases += 1
                    ambiguity_raw_unidentifiable += int(state is EntailmentState.RAW_UNIDENTIFIABLE)
                if len(qc_rows[lesion_class][case.stratum]) < 3 * config.qc_cases_per_stratum:
                    qc_rows[lesion_class][case.stratum].extend(
                        [
                            (f"{case.stratum} obs", observed),
                            (
                                f"P {positive['score']:.3f}/{positive['threshold']:.3f}",
                                search_results["PRESENT"].candidate,
                            ),
                            (
                                f"A {negative['score']:.3f}/{negative['threshold']:.3f}",
                                search_results["ABSENT"].candidate,
                            ),
                        ]
                    )
            state_counts[lesion_class][cast(str, row["state"])] += 1
            result_rows.append(row)
    for lesion_class in LESION_CLASSES:
        cells = [cell for stratum in ALL_STRATA for cell in qc_rows[lesion_class][stratum]]
        artifacts[f"qc-{lesion_class.lower()}-verify-contact-sheet.png"] = (
            make_labeled_contact_sheet(cells, columns=3)
        )
    bank_payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": "SOURCE_ISOLATED_HASH_BANK__BACKGROUND_TOP_K_RETRIEVAL",
        "classes": {
            lesion_class: [
                {
                    "package_id": record.package_id,
                    "twin_id": record.twin_id,
                    "source_key": record.source_key,
                    "split": record.split,
                    "lesion_class": record.lesion_class,
                }
                for record in selected_banks[lesion_class]
            ]
            for lesion_class in LESION_CLASSES
        },
    }
    threshold_payload = {
        "schema_version": SCHEMA_VERSION,
        "policy": "LEAVE_ONE_SOURCE_OUT_HIGHER_QUANTILE_TIMES_FROZEN_MULTIPLIER",
        "quantile": config.threshold_quantile,
        "multiplier": config.threshold_multiplier,
        "thresholds": thresholds,
        "records": threshold_rows,
    }
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_class": "RETRIEVED_FINITE_GRID_V1",
        "global_claim": "RETRIEVED_FINITE_HYPOTHESIS_CLASS_ONLY",
        "records": result_rows,
    }
    artifacts["exemplar-bank-v1.json"] = _canonical(bank_payload)
    artifacts["threshold-calibration-v1.json"] = _canonical(threshold_payload)
    artifacts["verify-results-v1.json"] = _canonical(results_payload)
    replay_rate = replay_passes / replay_checks if replay_checks else 0.0
    predicate_rate = predicate_passes / predicate_checks if predicate_checks else 0.0
    failure_audit = _failure_injection_audit()
    exact_selection = all(
        len(selected_cases[lesion_class]) == config.regions_per_class
        and all(
            stratum_counts[lesion_class][stratum] == config.per_stratum for stratum in ALL_STRATA
        )
        for lesion_class in LESION_CLASSES
    )
    bank_probe_isolation = all(
        {record.source_key for record in selected_banks[lesion_class]}.isdisjoint(
            {case.record.source_key for case in selected_cases[lesion_class]}
        )
        for lesion_class in LESION_CLASSES
    )
    automated_gate = (
        exact_selection
        and bank_probe_isolation
        and replay_rate >= config.minimum_replay_rate
        and predicate_rate >= config.minimum_predicate_rate
        and ood_cases == ood_out_of_model
        and completed_searches == finite_gap_searches
        and all(value == "OUT_OF_MODEL" for value in failure_audit.values())
    )
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if automated_gate else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "scope": {
            "splits": list(CANONICAL_SPLITS),
            "final_test_status": "NOT_OPENED",
            "oracle_region_only": True,
            "paper_performance": False,
        },
        "selection": {
            "regions_per_class": {
                lesion_class: len(selected_cases[lesion_class]) for lesion_class in LESION_CLASSES
            },
            "strata_per_class": {
                lesion_class: dict(sorted(stratum_counts[lesion_class].items()))
                for lesion_class in LESION_CLASSES
            },
            "bank_entries_per_class": {
                lesion_class: len(selected_banks[lesion_class]) for lesion_class in LESION_CLASSES
            },
            "bank_probe_source_isolation": bank_probe_isolation,
            "unique_region_gate": exact_selection,
        },
        "solver": {
            "strategy": "EXHAUSTIVE_RETRIEVED_FINITE_GRID",
            "global_claim": "RETRIEVED_FINITE_HYPOTHESIS_CLASS_ONLY",
            "completed_searches": completed_searches,
            "finite_zero_gap_searches": finite_gap_searches,
            "failure_injection_states": failure_audit,
        },
        "witness": {
            "replay_checks": replay_checks,
            "replay_passes": replay_passes,
            "replay_rate": replay_rate,
            "predicate_checks": predicate_checks,
            "predicate_passes": predicate_passes,
            "predicate_validity_rate": predicate_rate,
        },
        "development_diagnostics": {
            "state_counts_per_class": {
                lesion_class: dict(sorted(state_counts[lesion_class].items()))
                for lesion_class in LESION_CLASSES
            },
            "s5_tier_a_ambiguous_cases": ambiguity_cases,
            "s5_tier_a_raw_unidentifiable": ambiguity_raw_unidentifiable,
            "s5_tier_a_ambiguity_agreement_rate": (
                ambiguity_raw_unidentifiable / ambiguity_cases if ambiguity_cases else 0.0
            ),
            "disposition": "DEVELOPMENT_DIAGNOSTIC_NOT_FCR_OR_PAPER_PERFORMANCE",
        },
        "io_audit": {
            "s4_twin_png_decodes": io_audit.s4_twin_png_decodes,
            "s5_witness_png_decodes": io_audit.s5_witness_png_decodes,
            "witness_replay_png_decodes": io_audit.witness_replay_png_decodes,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "uwf_reads": 0,
            "dr_grade_values_used": 0,
            "source_split_leaks": 0,
        },
        "gate": {
            "automated": "PASS" if automated_gate else "FAIL",
            "ai_visual_qc": "PENDING",
            "s7_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "exemplar_bank_sha256": sha256_bytes(artifacts["exemplar-bank-v1.json"]),
            "threshold_calibration_sha256": sha256_bytes(
                artifacts["threshold-calibration-v1.json"]
            ),
            "verify_results_sha256": sha256_bytes(artifacts["verify-results-v1.json"]),
        },
        "evidence_boundary": [
            "The zero gap is exact only inside the retrieved finite hypothesis grid.",
            "This run uses development TRAIN/CALIBRATION controlled-twin regions only.",
            "S5 ambiguity agreement is a development diagnostic, not FCR or medical performance.",
            "No final locked test, Lite, enhancer, clinical, or paper-performance experiment ran.",
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
        config = S6Config.load(args.config)
        audit = run_s6(config, output_dir=args.output_dir)
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Science S6 failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 3


if __name__ == "__main__":
    raise SystemExit(main())
