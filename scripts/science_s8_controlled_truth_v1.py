"""Construct the prospective Science S8 controlled-truth target tranche."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import science_s7_lite_v7 as core
import science_s7_lite_v8 as v8
import science_s7_lite_v9 as v9
import torch
import yaml
from numpy.typing import NDArray

import retinaentail.science_s4_ddr_recovery as ddr
import retinaentail.science_s4_twins as s4_twins
from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import OperationTrace, apply_and_record, replay_trace
from retinaentail.interventions.controlled_twin_pilot import (
    QCCell,
    TwinRejected,
    _contact_sheet,
)
from retinaentail.science_s5_degradation_calibration import _mapping, _sequence, _string
from retinaentail.science_s5_degradation_calibration_v2 import frozen_family_parameters
from retinaentail.science_s6_verify import LESION_CLASSES
from retinaentail.science_s7_lite import _cluster_bootstrap_fcr, _policy_metrics
from retinaentail.science_s8_controlled_truth import (
    ConditionSpec,
    apply_policies,
    fit_selective_margin_thresholds,
    fit_source_max_thresholds,
    frozen_conditions,
    one_sided_zero_error_upper,
    wilson_interval,
)
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "1.0.0"
EXPERIMENT_ID: Final = "S8_CONTROLLED_TRUTH_BOUNDARY_V1"
MINIMUM_ACCEPTED_SOURCES: Final = {"HE": 120, "EX": 80, "MA": 160, "SE": 20}
CLASS_INDEX: Final = {lesion: index for index, lesion in enumerate(LESION_CLASSES)}
POLICIES: Final = {
    "four_valued": "state_four_valued",
    "forced_nearest": "state_forced_nearest",
    "selective_margin": "state_selective_margin",
    "present_only": "state_present_only",
    "absent_only": "state_absent_only",
}


class S8ControlledTruthRunError(ValueError):
    """Raised when the prospective S8 construction contract cannot be replayed."""


@dataclass(frozen=True, slots=True)
class S8Config:
    """Minimal config surface shared with the frozen DDR twin constructor."""

    path: Path
    config_sha256: str
    seed: int
    artifact_root: str
    expected_baseline_commit: str
    alpha: float
    bootstrap_replicates: int
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> S8Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S8 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("status") != "PREREGISTERED_NOT_RUN"
        ):
            raise S8ControlledTruthRunError("unsupported S8 config")
        seed = root.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != 2026082621:
            raise S8ControlledTruthRunError("S8 seed changed")
        bindings: dict[str, Path] = {}
        for name, raw in _mapping(root.get("bindings"), "S8 bindings").items():
            binding = _mapping(raw, f"binding {name}")
            relative = _string(binding.get("path"), f"{name} path")
            target = resolved.parents[2] / Path(relative)
            expected = _string(binding.get("sha256"), f"{name} hash")
            if not target.is_file() or sha256_file(target) != expected:
                raise S8ControlledTruthRunError(f"S8 binding mismatch: {name}")
            bindings[name] = target
        required = {
            "membership",
            "component_inventory",
            "s4_ddr_config",
            "s4_ddr_adapter",
            "v8_source_roles",
            "v8_adapter",
            "v8_config",
        }
        if set(bindings) != required:
            raise S8ControlledTruthRunError("S8 binding set changed")
        construction = _mapping(root.get("construction"), "construction")
        if construction.get("minimum_accepted_sources") != MINIMUM_ACCEPTED_SOURCES:
            raise S8ControlledTruthRunError("S8 construction minima changed")
        calibration = _mapping(root.get("calibration"), "calibration")
        statistics = _mapping(root.get("statistics"), "statistics")
        return cls(
            resolved,
            sha256_file(resolved),
            seed,
            _string(root.get("artifact_root"), "artifact root"),
            _string(root.get("expected_baseline_commit"), "baseline commit"),
            float(cast(float, calibration.get("alpha"))),
            int(cast(int, statistics.get("bootstrap_replicates"))),
            bindings,
        )


@dataclass(slots=True)
class ConstructionResult:
    """In-memory prospective construction result."""

    accepted: list[dict[str, object]]
    rejected: list[dict[str, object]]
    not_attempted: list[dict[str, object]]
    artifacts: dict[str, bytes]
    qc_cells: dict[str, list[QCCell]]
    io_counts: dict[str, int]

    def summary(self) -> dict[str, object]:
        accepted_counts = Counter(row["lesion_class"] for row in self.accepted)
        rejected_counts = Counter(row["reason_code"] for row in self.rejected)
        gate = all(
            accepted_counts[lesion] >= MINIMUM_ACCEPTED_SOURCES[lesion] for lesion in LESION_CLASSES
        )
        return {
            "status": "READY_FOR_FORMAL_RUN" if gate else "BLOCKED_CONSTRUCTION",
            "accepted": len(self.accepted),
            "accepted_by_class": {lesion: accepted_counts[lesion] for lesion in LESION_CLASSES},
            "minimum_accepted_sources": MINIMUM_ACCEPTED_SOURCES,
            "rejected_attempts": len(self.rejected),
            "rejected_by_reason": dict(sorted(rejected_counts.items())),
            "not_attempted_after_acceptance": len(self.not_attempted),
            "generated_artifact_count_in_memory": len(self.artifacts),
            "io_counts": self.io_counts,
            "locked_test_root_reads": 0,
            "model_runs": 0,
            "outputs_written": 0,
        }


def _resolved_candidates(config: S8Config) -> list[list[Mapping[str, object]]]:
    membership = core._load_json(config.bindings["membership"], "S8 membership")
    inventory = core._load_json(config.bindings["component_inventory"], "component inventory")
    lookup = {
        _string(record.get("component_id"), "component ID"): record
        for raw in _sequence(inventory.get("records"), "inventory records")
        for record in (_mapping(raw, "inventory record"),)
    }
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for raw in _sequence(membership.get("candidate_records"), "membership candidates"):
        selected = _mapping(raw, "membership candidate")
        component_id = _string(selected.get("component_id"), "component ID")
        record = lookup.get(component_id)
        if record is None:
            raise S8ControlledTruthRunError("membership component is absent from inventory")
        if any(
            record.get(key) != selected.get(key)
            for key in ("source_image_id", "lesion_class", "split")
        ):
            raise S8ControlledTruthRunError("membership-to-inventory identity mismatch")
        grouped[
            (
                _string(selected.get("source_key"), "source key"),
                _string(selected.get("lesion_class"), "lesion class"),
            )
        ].append({**record, "candidate_rank": selected.get("candidate_rank")})
    result: list[list[Mapping[str, object]]] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda row: int(cast(int, row["candidate_rank"])))
        ranks = [row["candidate_rank"] for row in ordered]
        if ranks != list(range(1, len(ranks) + 1)) or len(ranks) > 10:
            raise S8ControlledTruthRunError("membership candidate ranks changed")
        result.append(ordered)
    return result


def build_prospective_tranche(config: S8Config) -> ConstructionResult:
    """Try only frozen candidates and accept the first constructible per source/class."""

    runtime = v9._ddr_runtime_config(config)  # noqa: SLF001
    adapter = core._load_json(config.bindings["s4_ddr_adapter"], "DDR adapter")
    adapter_records = ddr._adapter_records(runtime, adapter)  # noqa: SLF001
    adapter_lookup = {
        _string(record.get("source_image_id"), "source ID"): record for record in adapter_records
    }
    development = Path(_string(adapter.get("root"), "DDR development root")).resolve(strict=True)
    if (
        development.name != runtime.development_leaf_name
        or development.parent.name != runtime.external_root_name
        or development.is_symlink()
    ):
        raise S8ControlledTruthRunError("DDR development root binding changed")

    io_counts: Counter[str] = Counter()
    contexts: dict[str, s4_twins.SourceContext] = {}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    not_attempted: list[dict[str, object]] = []
    artifacts: dict[str, bytes] = {}
    qc_cells: dict[str, list[QCCell]] = defaultdict(list)
    for candidates in _resolved_candidates(config):
        accepted_rank: int | None = None
        for record in candidates:
            rank = int(cast(int, record["candidate_rank"]))
            if accepted_rank is not None:
                not_attempted.append(
                    {
                        "component_id": record["component_id"],
                        "source_image_id": record["source_image_id"],
                        "lesion_class": record["lesion_class"],
                        "candidate_rank": rank,
                        "reason_code": "NOT_ATTEMPTED_AFTER_ACCEPTANCE",
                    }
                )
                continue
            source_id = _string(record.get("source_image_id"), "source ID")
            context = contexts.get(source_id)
            if context is None:
                adapter_record = adapter_lookup.get(source_id)
                if adapter_record is None:
                    raise S8ControlledTruthRunError("S8 source is absent from DDR adapter")
                context = ddr._load_context(  # noqa: SLF001
                    config=runtime,
                    development=development,
                    adapter_record=adapter_record,
                    io_counts=io_counts,
                )
                contexts[source_id] = context
            try:
                built = s4_twins._build_one(  # noqa: SLF001
                    context=context, record=record, config=runtime.twin
                )
            except TwinRejected as error:
                if error.reason_code not in runtime.twin.rejection_taxonomy:
                    raise S8ControlledTruthRunError("unregistered S8 rejection") from error
                rejected.append(
                    {
                        "component_id": record["component_id"],
                        "source_image_id": source_id,
                        "lesion_class": record["lesion_class"],
                        "candidate_rank": rank,
                        "reason_code": error.reason_code,
                    }
                )
                continue
            accepted_rank = rank
            built.record["experiment_id"] = EXPERIMENT_ID
            built.record["dataset_id"] = "ddr"
            built.record["dataset_version"] = runtime.twin.dataset_version
            built.record["candidate_rank"] = rank
            accepted.append(cast(dict[str, object], built.record))
            for relative, payload in built.artifacts.items():
                if relative in artifacts:
                    raise S8ControlledTruthRunError("S8 artifact path collision")
                artifacts[relative] = payload
            qc_cells[_string(record.get("lesion_class"), "lesion class")].append(built.qc_cell)
    return ConstructionResult(
        accepted,
        rejected,
        not_attempted,
        artifacts,
        dict(qc_cells),
        dict(io_counts),
    )


@dataclass(frozen=True, slots=True)
class ViewRecord:
    """One normalized observation and its replay binding."""

    array: NDArray[np.uint8]
    trace_sha256: str
    seeds: tuple[int, ...]


def _verify_formal_head(config: S8Config) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.path.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != config.expected_baseline_commit:
        raise S8ControlledTruthRunError("S8 formal baseline drift")


def _write_construction(output: Path, result: ConstructionResult) -> list[core.TwinItem]:
    for relative, payload in sorted(result.artifacts.items()):
        write_bytes_fail_safe(output.joinpath(*PurePosixPath(relative).parts), payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "accepted": result.accepted,
        "rejected_attempts": result.rejected,
        "not_attempted_after_acceptance": result.not_attempted,
    }
    write_bytes_fail_safe(output / "twin-manifest-v1.json", canonical_json_bytes(manifest))
    for lesion in LESION_CLASSES:
        cells = sorted(result.qc_cells.get(lesion, []), key=lambda cell: cell.twin_id)[:24]
        if cells:
            write_bytes_fail_safe(
                output / f"qc-{lesion.lower()}-controlled-twins.png",
                _contact_sheet(cells, columns=3, panel_size=160),
            )
    return [
        core.TwinItem(
            package_id="S8_DDR_V1",
            root=output,
            raw=record,
            source_key=f"S4_DDR_V5:{record['source_image_id']}",
            source_image_id=_string(record.get("source_image_id"), "source ID"),
            lesion_class=_string(record.get("lesion_class"), "lesion class"),
            twin_id=_string(record.get("twin_id"), "twin ID"),
        )
        for record in result.accepted
    ]


def _v8_training_items(config: S8Config) -> tuple[list[core.TwinItem], list[core.TwinItem]]:
    v8_root = _mapping(
        yaml.safe_load(config.bindings["v8_config"].read_text(encoding="utf-8")),
        "v8 config",
    )
    v8_bindings = _mapping(v8_root.get("bindings"), "v8 bindings")
    project = config.path.parents[2]

    def bound(name: str) -> Path:
        raw = _mapping(v8_bindings.get(name), f"v8 binding {name}")
        path = project.joinpath(*PurePosixPath(_string(raw.get("path"), "path")).parts)
        if not path.is_file() or sha256_file(path) != _string(raw.get("sha256"), "hash"):
            raise S8ControlledTruthRunError(f"v8 upstream binding changed: {name}")
        return path

    items = core._load_twins(
        "S4_IDRID_V1", bound("s4_idrid_twin_manifest"), manifest_name="S4 IDRiD v1"
    ) + core._load_twins("S4_DDR_V5", bound("s4_ddr_twin_manifest"), manifest_name="S4 DDR v5")
    roles_payload = core._load_json(config.bindings["v8_source_roles"], "v8 roles")
    roles_raw = _mapping(roles_payload.get("roles"), "v8 role mapping")
    roles = {key: _string(value, "v8 role") for key, value in roles_raw.items()}
    train = core._select_records(items, roles, "MODEL_TRAIN", 2, 2026082618)
    calibration = core._select_records(items, roles, "CONFORMAL_FIT", 4, 2026082618)
    if (len(train), len(calibration)) != (266, 118):
        raise S8ControlledTruthRunError("v8 train/calibration memberships changed")
    return train, calibration


def _load_frozen_adapter(config: S8Config) -> v8.FrozenResNetAdapter:
    v8_root = _mapping(
        yaml.safe_load(config.bindings["v8_config"].read_text(encoding="utf-8")),
        "v8 config",
    )
    weights = _mapping(_mapping(v8_root.get("method"), "method").get("weight_binding"), "weights")
    weight_path = Path.home().joinpath(
        *PurePosixPath(_string(weights.get("home_relative_path"), "weight path")).parts
    )
    if not weight_path.is_file() or sha256_file(weight_path) != _string(
        weights.get("sha256"), "weight hash"
    ):
        raise S8ControlledTruthRunError("frozen ResNet-18 weight changed")
    model = v8.FrozenResNetAdapter(32, weight_path)
    payload = torch.load(config.bindings["v8_adapter"], map_location="cpu", weights_only=True)
    state = _mapping(payload, "v8 adapter")
    adapter_state = state.get("adapter_state")
    if not isinstance(adapter_state, dict):
        raise S8ControlledTruthRunError("v8 adapter state is invalid")
    incompatible = model.load_state_dict(adapter_state, strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("backbone.") for key in incompatible.missing_keys
    ):
        raise S8ControlledTruthRunError("v8 adapter replay is incompatible")
    model.eval()
    return model


def _condition_view(
    raw: NDArray[np.uint8],
    item: core.TwinItem,
    hypothesis: str,
    condition: ConditionSpec,
    config: S8Config,
    replay_counts: Counter[str],
) -> ViewRecord:
    current = raw
    steps = []
    seeds: list[int] = []
    bbox = core.regional_context_bbox(item.raw, margin=8, patch_wh=(256, 256))
    for index, operation in enumerate(condition.operations):
        seed = int(
            core._digest(
                config.seed,
                "S8_VIEW",
                "|".join(
                    (
                        condition.condition_id,
                        operation,
                        str(index),
                        item.twin_id,
                        hypothesis,
                    )
                ),
            )[:16],
            16,
        )
        parameters = frozen_family_parameters(operation, condition.severity)
        if operation == "glare_reflection":
            parameters = {
                **cast(dict[str, str | int | float | bool], parameters),
                "center_x": ((bbox[0] + bbox[2]) / 2.0) / 256.0,
                "center_y": ((bbox[1] + bbox[3]) / 2.0) / 256.0,
            }
        current, step = apply_and_record(
            current,
            operation,
            severity=condition.severity,
            seed=seed,
            parameters=parameters,
        )
        steps.append(step)
        seeds.append(seed)
    trace = OperationTrace(tuple(steps))
    replay_counts["checks"] += 1
    if not np.array_equal(current, replay_trace(raw, trace)):
        replay_counts["mismatches"] += 1
        raise S8ControlledTruthRunError("S8 degradation trace replay failed")
    normalized = core._normalized_context(current, item, 128)
    return ViewRecord(
        normalized,
        sha256_bytes(canonical_json_bytes(trace.to_dict())),
        tuple(seeds),
    )


def _raw_cache(
    items: Sequence[core.TwinItem], io_audit: core.IOAudit
) -> dict[tuple[str, str, str], NDArray[np.uint8]]:
    return {
        (item.package_id, item.twin_id, hypothesis): core._decode_side(item, hypothesis, io_audit)
        for item in items
        for hypothesis in ("PRESENT", "ABSENT")
    }


def _encode(
    model: v8.FrozenResNetAdapter,
    images: Sequence[NDArray[np.uint8]],
    lesions: Sequence[str],
) -> NDArray[np.float32]:
    chunks: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            classes = torch.tensor(
                [CLASS_INDEX[value] for value in lesions[start : start + 64]],
                dtype=torch.long,
            )
            embedding, _ = model(core._tensor(np.stack(images[start : start + 64])), classes)
            chunks.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def _score_condition(
    *,
    model: v8.FrozenResNetAdapter,
    condition: ConditionSpec,
    train: Sequence[core.TwinItem],
    queries: Sequence[core.TwinItem],
    role: str,
    cache: Mapping[tuple[str, str, str], NDArray[np.uint8]],
    config: S8Config,
    replay_counts: Counter[str],
) -> list[dict[str, object]]:
    descriptors = [
        (item, hypothesis) for item in (*train, *queries) for hypothesis in ("PRESENT", "ABSENT")
    ]
    views = [
        _condition_view(
            cache[(item.package_id, item.twin_id, hypothesis)],
            item,
            hypothesis,
            condition,
            config,
            replay_counts,
        )
        for item, hypothesis in descriptors
    ]
    embeddings = _encode(
        model,
        [view.array for view in views],
        [item.lesion_class for item, _ in descriptors],
    )
    train_count = len(train) * 2
    bank: dict[tuple[str, str], list[tuple[core.TwinItem, NDArray[np.float32]]]] = defaultdict(list)
    for (item, hypothesis), embedding in zip(
        descriptors[:train_count], embeddings[:train_count], strict=True
    ):
        bank[(item.lesion_class, hypothesis)].append((item, embedding))
    rows: list[dict[str, object]] = []
    for index, ((item, truth), query, view) in enumerate(
        zip(descriptors[train_count:], embeddings[train_count:], views[train_count:], strict=True)
    ):
        scores: dict[str, float] = {}
        witnesses: dict[str, dict[str, object]] = {}
        for hypothesis in ("PRESENT", "ABSENT"):
            eligible = [
                entry
                for entry in bank[(item.lesion_class, hypothesis)]
                if entry[0].source_key != item.source_key
            ]
            if not eligible:
                raise S8ControlledTruthRunError("S8 witness bank lacks source-isolated support")
            distances = np.asarray(
                [1.0 - float(np.dot(query, entry[1])) for entry in eligible],
                dtype=np.float64,
            )
            selected = min(
                range(len(eligible)),
                key=lambda value: (float(distances[value]), eligible[value][0].twin_id),
            )
            witness = eligible[selected][0]
            if witness.source_key == item.source_key or not math.isfinite(
                float(distances[selected])
            ):
                raise S8ControlledTruthRunError("S8 witness isolation or score failed")
            scores[hypothesis] = float(distances[selected])
            witnesses[hypothesis] = {
                "source_key": witness.source_key,
                "twin_id": witness.twin_id,
            }
        rows.append(
            {
                "row_id": f"{role}:{condition.condition_id}:{item.twin_id}:{truth}",
                "role": role,
                "source_key": item.source_key,
                "source_image_id": item.source_image_id,
                "twin_id": item.twin_id,
                "lesion_class": item.lesion_class,
                "expected_truth": truth,
                "condition_id": condition.condition_id,
                "degradation_family": condition.family,
                "severity_level": condition.severity_level,
                "severity": condition.severity,
                "positive_score": scores["PRESENT"],
                "negative_score": scores["ABSENT"],
                "positive_witness": witnesses["PRESENT"],
                "negative_witness": witnesses["ABSENT"],
                "observation_sha256": sha256_bytes(view.array.tobytes()),
                "trace_sha256": view.trace_sha256,
                "degradation_seeds": list(view.seeds),
                "query_index_within_condition": index,
            }
        )
    return rows


def _stratified_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    groupings = {
        "overall": (),
        "lesion": ("lesion_class",),
        "truth": ("expected_truth",),
        "family": ("degradation_family",),
        "severity": ("severity_level",),
        "condition": ("condition_id",),
        "lesion_truth": ("lesion_class", "expected_truth"),
    }
    for policy, state_key in POLICIES.items():
        policy_result: dict[str, object] = {}
        for name, fields in groupings.items():
            grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
            for row in rows:
                grouped[tuple(row.get(field) for field in fields)].append(row)
            metrics: dict[str, object] = {}
            for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
                summary = _policy_metrics(group, state_key)
                determinate = int(cast(int, summary["determinate_regions"]))
                false = int(cast(int, summary["false_certainties"]))
                total = int(cast(int, summary["in_domain_regions"]))
                summary["coverage_wilson95"] = list(wilson_interval(determinate, total))
                summary["fcr_wilson95"] = (
                    list(wilson_interval(false, determinate)) if determinate else None
                )
                summary["zero_error_one_sided_upper95"] = (
                    one_sided_zero_error_upper(determinate) if false == 0 else None
                )
                label = "ALL" if not fields else "|".join(str(value) for value in key)
                metrics[label] = summary
            policy_result[name] = metrics
        policy_result["source_cluster_bootstrap"] = _cluster_bootstrap_fcr(
            rows, state_key, 2026082621, 2000
        )
        result[policy] = policy_result
    return result


def _artifact_manifest(output: Path) -> dict[str, object]:
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        entries.append(
            {
                "relative_path": relative,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "entries": entries,
    }


def run_formal(config: S8Config, output_dir: Path) -> dict[str, object]:
    """Run construction, development-only calibration, and prospective S8 scoring."""

    _verify_formal_head(config)
    output = output_dir.resolve()
    expected = (config.path.parents[2] / config.artifact_root).resolve()
    if output != expected or output.exists():
        raise S8ControlledTruthRunError("S8 output path changed or already exists")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

    construction = build_prospective_tranche(config)
    summary = construction.summary()
    if summary["status"] != "READY_FOR_FORMAL_RUN":
        raise S8ControlledTruthRunError("S8 construction gate failed")
    test_items = _write_construction(output, construction)
    train_items, calibration_items = _v8_training_items(config)
    source_sets = {
        "train": {item.source_key for item in train_items},
        "calibration": {item.source_key for item in calibration_items},
        "test": {item.source_key for item in test_items},
    }
    overlap = sum(
        len(source_sets[first].intersection(source_sets[second]))
        for first, second in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    )
    if overlap:
        raise S8ControlledTruthRunError("S8 source-role overlap")
    io_audit = core.IOAudit()
    all_items = [*train_items, *calibration_items, *test_items]
    cache = _raw_cache(all_items, io_audit)
    model = _load_frozen_adapter(config)
    replay_counts: Counter[str] = Counter()
    calibration_rows: list[dict[str, object]] = []
    for condition in frozen_conditions():
        calibration_rows.extend(
            _score_condition(
                model=model,
                condition=condition,
                train=train_items,
                queries=calibration_items,
                role="CONFORMAL_FIT",
                cache=cache,
                config=config,
                replay_counts=replay_counts,
            )
        )
    thresholds = fit_source_max_thresholds(calibration_rows, alpha=config.alpha)
    selective = fit_selective_margin_thresholds(calibration_rows, alpha=config.alpha)
    write_bytes_fail_safe(
        output / "source-max-calibration-v1.json",
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "method": "SOURCE_MAX_CLASS_CONDITION_HYPOTHESIS_SPLIT_CONFORMAL",
                "alpha": config.alpha,
                "thresholds": thresholds,
                "selective_margin_thresholds": selective,
            }
        ),
    )

    test_rows: list[dict[str, object]] = []
    for condition in frozen_conditions():
        raw = _score_condition(
            model=model,
            condition=condition,
            train=train_items,
            queries=test_items,
            role="PROSPECTIVE_S8_TEST",
            cache=cache,
            config=config,
            replay_counts=replay_counts,
        )
        test_rows.extend(apply_policies(row, thresholds, selective) for row in raw)
    metrics = _stratified_metrics(test_rows)
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_DEVELOPMENT_BOUNDARY_RESULT",
        "construction": summary,
        "condition_count": len(frozen_conditions()),
        "calibration_row_count": len(calibration_rows),
        "test_row_count": len(test_rows),
        "metrics": metrics,
        "rows": test_rows,
    }
    write_bytes_fail_safe(
        output / "controlled-truth-results-v1.json", canonical_json_bytes(results)
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "config_sha256": config.config_sha256,
        "bindings": {name: sha256_file(path) for name, path in config.bindings.items()},
        "source_role_counts": {name: len(values) for name, values in source_sets.items()},
        "source_role_overlap": overlap,
        "twin_png_decodes": io_audit.twin_png_decodes,
        "twin_png_hash_mismatches": io_audit.twin_png_hash_mismatches,
        "degradation_replay_checks": replay_counts["checks"],
        "degradation_replay_mismatches": replay_counts["mismatches"],
        "query_witness_source_collisions": 0,
        "locked_test_root_reads": 0,
        "model_training_steps": 0,
        "adapter_sha256": sha256_file(config.bindings["v8_adapter"]),
    }
    write_bytes_fail_safe(output / "audit.json", canonical_json_bytes(audit))
    write_bytes_fail_safe(
        output / "artifact-manifest-v1.json", canonical_json_bytes(_artifact_manifest(output))
    )
    return {"audit": audit, "results": results, "output": str(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = S8Config.load(args.config)
    if args.preflight_only:
        result = build_prospective_tranche(config)
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
        return 0 if result.summary()["status"] == "READY_FOR_FORMAL_RUN" else 2
    completed = run_formal(config, cast(Path, args.output_dir))
    result_summary = cast(Mapping[str, object], completed["results"])
    print(
        json.dumps(
            {
                "status": result_summary["status"],
                "construction": result_summary["construction"],
                "condition_count": result_summary["condition_count"],
                "test_row_count": result_summary["test_row_count"],
                "output": completed["output"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
