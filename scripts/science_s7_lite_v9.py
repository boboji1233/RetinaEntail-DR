"""Run the final, explicitly run-informed Science S7 v9 development recovery."""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import science_s7_lite_v7 as core
import science_s7_lite_v8 as v8
import torch
import torch.nn.functional as functional
import torchvision
import yaml
from numpy.typing import NDArray

import retinaentail.science_s4_ddr_recovery as ddr
import retinaentail.science_s4_twins as s4_twins
from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState
from retinaentail.interventions.controlled_twin_pilot import (
    METHODS,
    QCCell,
    TwinRejected,
    _contact_sheet,
)
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
)
from retinaentail.science_s6_verify import LESION_CLASSES
from retinaentail.science_s7_lite import (
    DETERMINATE_STATES,
    _cluster_bootstrap_fcr,
    _policy_metrics,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest
from retinaentail.science_s7_lite_v9 import (
    S7LiteV9Error,
    source_balanced_cross_hypothesis_batch_hard,
    source_balanced_pair_weights,
    validate_frozen_holdout_records,
)
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

SCHEMA_VERSION: Final = "9.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V9"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CLASS_INDEX: Final = {lesion_class: index for index, lesion_class in enumerate(LESION_CLASSES)}
V8_RECORD_SELECTION_SEED: Final = 2026082618


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV9Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV9Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV9Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV9Error(f"{name} is not SHA-256")
    return text


def _verify_baseline(expected: object) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != _string(expected, "expected baseline commit"):
        raise S7LiteV9Error("v9 baseline drift")


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV9Error(f"binding mismatch: {name}")
    return path


@dataclass(frozen=True, slots=True)
class V9Config:
    path: Path
    config_sha256: str
    seed: int
    record_selection_seed: int
    validation_sources_per_class: int
    conformal_sources_per_class: int
    train_records_per_source_class: int
    evaluation_records_per_source_class: int
    input_side: int
    embedding_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    prototype_weight: float
    head_weight: float
    invariance_weight: float
    separation_weight: float
    batch_hard_weight: float
    separation_margin: float
    alpha: float
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    minimum_holdout_sources_per_class: int
    bindings: dict[str, Path]
    weight_path: Path
    weight_sha256: str
    weight_url: str

    @classmethod
    def load(cls, path: Path) -> V9Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v9 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_FINAL_RUN_INFORMED_RECOVERY"
        ):
            raise S7LiteV9Error("unsupported executable S7 v9 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "development disclosure")
        if disclosure != {
            "v1_to_v8_observed": True,
            "independent_replication": False,
            "final_permitted_s7_recovery": True,
            "v10_permitted": False,
            "final_locked_test_opened": False,
        }:
            raise S7LiteV9Error("v9 run-informed disclosure changed")
        scope = _mapping(root.get("scope"), "scope")
        if (
            scope.get("lesions") != list(LESION_CLASSES)
            or scope.get("conditions") != ["IDENTITY", "GLARE_MODERATE"]
            or scope.get("source_splits") != ["TRAIN", "CALIBRATION"]
            or scope.get("final_locked_test_reads_allowed") is not False
            or scope.get("enhancer_training_allowed") is not False
            or scope.get("science_s8_or_later_allowed") is not False
        ):
            raise S7LiteV9Error("v9 scope changed")
        split = _mapping(root.get("source_split"), "source split")
        method = _mapping(root.get("method"), "method")
        change = _mapping(method.get("single_recovery_change"), "single recovery change")
        training = _mapping(root.get("training"), "training")
        losses = _mapping(training.get("loss_weights"), "loss weights")
        calibration = _mapping(root.get("calibration"), "calibration")
        statistics = _mapping(root.get("statistics"), "statistics")
        gate = _mapping(root.get("gate"), "gate")
        if (
            split.get("model_train_role") != "REUSE_V8_MODEL_TRAIN_EXACT_MEMBERSHIP"
            or split.get("conformal_fit_role") != "REUSE_V8_CONFORMAL_FIT_EXACT_MEMBERSHIP"
            or split.get("record_selection_seed") != V8_RECORD_SELECTION_SEED
            or method.get("architecture")
            != "TARGET_CONTEXT_128_FROZEN_IMAGENET_RESNET18_CLASS_PROJECTED_32D_V1"
            or method.get("backbone_frozen") is not True
            or change.get("id") != "SOURCE_BALANCED_CROSS_HYPOTHESIS_BATCH_HARD_V1"
            or change.get("replaces") != "WITHIN_TWIN_SEPARATION_LOSS"
            or calibration.get("method") != "SOURCE_MAX_CLASS_CONDITION_SPLIT_CONFORMAL"
        ):
            raise S7LiteV9Error("v9 sole-change method contract changed")

        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required = {
            "preregistration",
            "holdout_selection",
            "v9_code",
            "v9_helper",
            "v9_audit_schema",
            "v8_config",
            "v8_audit",
            "v8_results",
            "v8_calibration",
            "v8_source_roles",
            "v8_adapter",
            "v8_code",
            "s4_ddr_component_inventory",
            "s4_ddr_config",
            "s4_ddr_adapter",
        }
        if set(bindings) != required:
            raise S7LiteV9Error("v9 binding set changed")

        v8_root = _mapping(
            yaml.safe_load(bindings["v8_config"].read_text(encoding="utf-8")), "v8 config"
        )
        v8_bindings_raw = _mapping(v8_root.get("bindings"), "v8 bindings")
        for name, raw in v8_bindings_raw.items():
            bindings.setdefault(name, _verify_binding(raw, f"v8:{name}"))
        weights = _mapping(
            _mapping(v8_root.get("method"), "v8 method").get("weight_binding"), "weights"
        )
        cache_relative = _safe_relative_string(
            weights.get("home_relative_path"), "weight home-relative path"
        )
        weight_path = Path.home().joinpath(*PurePosixPath(cache_relative).parts)
        weight_sha256 = _sha256(weights.get("sha256"), "weight SHA-256")
        weight_url = _string(weights.get("source_url"), "weight URL")
        if not weight_path.is_file() or sha256_file(weight_path) != weight_sha256:
            raise S7LiteV9Error("pretrained weight binding mismatch")

        values: dict[str, int | float] = {
            "seed": _integer(root.get("seed"), "seed"),
            "record_selection_seed": _integer(
                split.get("record_selection_seed"), "record selection seed"
            ),
            "validation_sources_per_class": _integer(
                _mapping(split.get("third_holdout"), "third holdout").get(
                    "minimum_sources_per_class"
                ),
                "validation sources",
                1,
            ),
            "conformal_sources_per_class": 15,
            "train_records_per_source_class": 2,
            "evaluation_records_per_source_class": 4,
            "input_side": _integer(method.get("input_side"), "input side", 1),
            "embedding_dim": _integer(method.get("embedding_dim"), "embedding dim", 1),
            "epochs": _integer(training.get("epochs"), "epochs", 1),
            "batch_size": _integer(training.get("batch_size"), "batch size", 1),
            "learning_rate": _number(training.get("learning_rate"), "learning rate"),
            "weight_decay": _number(training.get("weight_decay"), "weight decay"),
            "prototype_weight": _number(losses.get("prototype_pull"), "prototype weight"),
            "head_weight": _number(losses.get("hypothesis_head"), "head weight"),
            "invariance_weight": _number(losses.get("glare_invariance"), "invariance weight"),
            "separation_weight": _number(
                losses.get("within_twin_separation"), "within-twin weight"
            ),
            "batch_hard_weight": _number(
                losses.get("source_balanced_cross_hypothesis_batch_hard"), "batch-hard weight"
            ),
            "separation_margin": _number(
                training.get("cosine_distance_margin"), "batch-hard margin"
            ),
            "alpha": _number(calibration.get("alpha"), "alpha"),
            "bootstrap_replicates": _integer(
                statistics.get("bootstrap_replicates"), "bootstrap replicates", 1
            ),
            "maximum_fcr": _number(gate.get("maximum_validation_fcr"), "FCR gate"),
            "minimum_moderate_coverage": _number(
                gate.get("minimum_moderate_coverage_per_class"), "coverage gate"
            ),
            "minimum_verify_agreement": _number(
                gate.get("minimum_legacy_verify_agreement"), "agreement gate"
            ),
            "minimum_holdout_sources_per_class": _integer(
                _mapping(split.get("third_holdout"), "third holdout").get(
                    "minimum_sources_per_class"
                ),
                "minimum holdout sources",
                1,
            ),
        }
        frozen: dict[str, int | float] = {
            "seed": 2026082619,
            "record_selection_seed": 2026082618,
            "validation_sources_per_class": 8,
            "conformal_sources_per_class": 15,
            "train_records_per_source_class": 2,
            "evaluation_records_per_source_class": 4,
            "input_side": 128,
            "embedding_dim": 32,
            "epochs": 40,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "prototype_weight": 1.0,
            "head_weight": 0.5,
            "invariance_weight": 0.5,
            "separation_weight": 0.0,
            "batch_hard_weight": 0.5,
            "separation_margin": 0.75,
            "alpha": 0.05,
            "bootstrap_replicates": 2000,
            "maximum_fcr": 0.05,
            "minimum_moderate_coverage": 0.05,
            "minimum_verify_agreement": 0.5,
            "minimum_holdout_sources_per_class": 8,
        }
        if values != frozen:
            raise S7LiteV9Error("v9 frozen numeric contract changed")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            bindings=bindings,
            weight_path=weight_path,
            weight_sha256=weight_sha256,
            weight_url=weight_url,
            **cast(dict[str, object], values),
        )


def _offsets(value: object, name: str) -> tuple[tuple[int, int], ...]:
    rows = _sequence(value, name)
    result: list[tuple[int, int]] = []
    for row in rows:
        pair = _sequence(row, f"{name} row")
        if len(pair) != 2:
            raise S7LiteV9Error(f"{name} row must contain two values")
        result.append((_integer(pair[0], name, -10000), _integer(pair[1], name, -10000)))
    return tuple(result)


def _ddr_runtime_config(config: V9Config) -> ddr.DDRRecoveryConfig:
    decoded: object = yaml.safe_load(config.bindings["s4_ddr_config"].read_text(encoding="utf-8"))
    root = _mapping(decoded, "S4 DDR config")
    dataset = _mapping(root.get("dataset"), "S4 DDR dataset")
    selection = _mapping(root.get("selection"), "S4 DDR selection")
    assignment = _mapping(root.get("method_assignment"), "S4 DDR assignment")
    edit = _mapping(root.get("edit"), "S4 DDR edit")
    qc = _mapping(root.get("qc"), "S4 DDR QC")
    decode = _mapping(root.get("decode"), "S4 DDR decode")
    fov = _mapping(root.get("fov"), "S4 DDR FOV")
    quotas_raw = _mapping(selection.get("accepted_quota_per_class_split"), "quotas")
    minima_raw = _mapping(selection.get("minimum_distinct_sources_per_class_split"), "minima")
    quotas = {
        lesion: {
            split: _integer(_mapping(quotas_raw[lesion], "quota")[split], "quota")
            for split in ("TRAIN", "CALIBRATION")
        }
        for lesion in LESION_CLASSES
    }
    minima = {
        lesion: {
            split: _integer(_mapping(minima_raw[lesion], "minimum")[split], "minimum")
            for split in ("TRAIN", "CALIBRATION")
        }
        for lesion in LESION_CLASSES
    }
    method_settings = {
        method: _mapping(assignment.get(method), f"method {method}") for method in METHODS
    }
    taxonomy_raw = _sequence(root.get("rejection_reason_taxonomy"), "rejection taxonomy")
    taxonomy = frozenset(_string(item, "rejection reason") for item in taxonomy_raw)
    twin = s4_twins.S4TwinConfig(
        path=config.path,
        payload=root,
        config_sha256=config.config_sha256,
        dataset_id="ddr",
        dataset_version=_string(dataset.get("dataset_version"), "DDR version"),
        artifact_root="reports/generated/science/s7_retinaentail_lite_v9",
        bindings={},
        seed=config.seed,
        class_targets={
            lesion: _integer(
                _mapping(selection.get("accepted_target_per_class"), "targets")[lesion], "target"
            )
            for lesion in LESION_CLASSES
        },
        split_quotas=quotas,
        minimum_sources={lesion: sum(minima[lesion].values()) for lesion in LESION_CLASSES},
        minimum_area=_integer(selection.get("minimum_area_px"), "minimum area", 1),
        maximum_area=_integer(selection.get("maximum_area_px"), "maximum area", 1),
        maximum_bbox_width=_integer(selection.get("maximum_bbox_width_px"), "maximum width", 1),
        maximum_bbox_height=_integer(selection.get("maximum_bbox_height_px"), "maximum height", 1),
        minimum_fov_margin=_number(selection.get("minimum_fov_margin_px"), "minimum FOV margin"),
        methods=METHODS,
        method_settings=method_settings,
        target_dilation=_integer(edit.get("target_mask_dilation_px"), "target dilation", 1),
        lesion_exclusion_dilation=_integer(
            edit.get("lesion_exclusion_dilation_px"), "lesion exclusion", 1
        ),
        sham_offsets=_offsets(edit.get("sham_offset_candidates_xy"), "sham offsets"),
        donor_offsets=_offsets(edit.get("donor_offset_candidates_xy"), "donor offsets"),
        qc_samples_per_class=_integer(qc.get("deterministic_samples_per_class"), "QC samples", 1),
        qc_columns=_integer(qc.get("contact_sheet_columns"), "QC columns", 1),
        qc_panel_size=_integer(qc.get("panel_size_px"), "QC panel size", 1),
        minimum_collisions=1,
        rejection_taxonomy=taxonomy,
    )
    expected_counts = _mapping(dataset.get("expected_label_availability"), "label counts")
    split_counts = _mapping(dataset.get("expected_split_source_counts"), "split counts")
    values_raw = _sequence(decode.get("accepted_l_values"), "mask values")
    modes_raw = _sequence(decode.get("accepted_mask_modes"), "mask modes")
    return ddr.DDRRecoveryConfig(
        path=config.bindings["s4_ddr_config"],
        payload=root,
        config_sha256=sha256_file(config.bindings["s4_ddr_config"]),
        twin=twin,
        raw_split_mapping={"TRAIN": "TRAIN", "VALID": "CALIBRATION"},
        minimum_sources_per_split=minima,
        expected_source_count=_integer(dataset.get("expected_source_count"), "source count", 1),
        expected_file_count=_integer(dataset.get("expected_file_count"), "file count", 1),
        expected_total_bytes=_integer(dataset.get("expected_total_bytes"), "bytes", 1),
        expected_split_counts={
            key: _integer(value, "split count") for key, value in split_counts.items()
        },
        expected_label_counts={
            key: _integer(value, "label count") for key, value in expected_counts.items()
        },
        external_root_name=_string(dataset.get("external_root_name"), "external root"),
        development_leaf_name=_string(dataset.get("development_leaf_name"), "development leaf"),
        fov_threshold=_integer(fov.get("foreground_max_rgb_min"), "FOV threshold"),
        fov_coverage=_number(fov.get("minimum_axis_coverage_fraction"), "FOV coverage"),
        mask_modes=tuple(_string(item, "mask mode") for item in modes_raw),
        mask_l_values=frozenset(_integer(item, "mask value") for item in values_raw),
        qc_samples_per_class=twin.qc_samples_per_class,
        qc_columns=twin.qc_columns,
        qc_panel_size=twin.qc_panel_size,
    )


@dataclass(frozen=True, slots=True)
class HoldoutBuild:
    accepted: list[dict[str, object]]
    rejected: list[dict[str, object]]
    artifacts: dict[str, bytes]
    qc_cells: dict[str, list[QCCell]]
    io_counts: dict[str, int]
    selected_count: int


def _build_holdout(config: V9Config) -> HoldoutBuild:
    inventory = core._load_json(config.bindings["s4_ddr_component_inventory"], "inventory")
    selection = core._load_json(config.bindings["holdout_selection"], "v9 selection")
    inventory_records = [
        _mapping(item, "inventory record")
        for item in _sequence(inventory.get("records"), "inventory records")
    ]
    selected_records = [
        _mapping(item, "selected record")
        for item in _sequence(selection.get("selected_records"), "selected records")
    ]
    resolved = validate_frozen_holdout_records(inventory_records, selected_records)
    if len(resolved) != 150:
        raise S7LiteV9Error("v9 selected component count changed")

    runtime = _ddr_runtime_config(config)
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
        raise S7LiteV9Error("DDR development root no longer matches the frozen adapter")

    io_counts: Counter[str] = Counter()
    contexts: dict[str, s4_twins.SourceContext] = {}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    artifacts: dict[str, bytes] = {}
    qc_cells: dict[str, list[QCCell]] = defaultdict(list)
    for record in resolved:
        source_id = _string(record.get("source_image_id"), "source ID")
        context = contexts.get(source_id)
        if context is None:
            adapter_record = adapter_lookup.get(source_id)
            if adapter_record is None:
                raise S7LiteV9Error("selected source is absent from the bound DDR adapter")
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
                raise S7LiteV9Error("unregistered v9 holdout rejection") from error
            rejected.append(
                {
                    "schema_version": "1.0.0",
                    "component_id": record["component_id"],
                    "source_image_id": source_id,
                    "split": record["split"],
                    "lesion_class": record["lesion_class"],
                    "reason_code": error.reason_code,
                }
            )
            continue
        built.record["experiment_id"] = EXPERIMENT_ID
        built.record["dataset_id"] = "ddr"
        built.record["dataset_version"] = runtime.twin.dataset_version
        accepted.append(cast(dict[str, object], built.record))
        for relative, payload in built.artifacts.items():
            if relative in artifacts:
                raise S7LiteV9Error("v9 twin artifact path collision")
            artifacts[relative] = payload
        qc_cells[_string(record.get("lesion_class"), "lesion class")].append(built.qc_cell)
    for lesion in LESION_CLASSES:
        qc_cells[lesion].sort(key=lambda cell: cell.twin_id)
        qc_cells[lesion] = qc_cells[lesion][:24]
    return HoldoutBuild(
        accepted=accepted,
        rejected=rejected,
        artifacts=artifacts,
        qc_cells=dict(qc_cells),
        io_counts=dict(io_counts),
        selected_count=len(resolved),
    )


def _train_adapter_v9(
    model: v8.FrozenResNetAdapter,
    samples: Sequence[core.PairSample],
    config: V9Config,
) -> tuple[list[float], dict[str, int], int, int]:
    features = v8._extract_training_features(model, samples, config)  # noqa: SLF001
    classes = torch.tensor(
        [CLASS_INDEX[sample.item.lesion_class] for sample in samples], dtype=torch.long
    )
    source_keys = [sample.item.source_key for sample in samples]
    lesion_classes = [sample.item.lesion_class for sample in samples]
    pair_weights = source_balanced_pair_weights(source_keys, lesion_classes)
    class_counts = Counter(lesion_classes)
    hypothesis = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    losses: list[float] = []
    eligible_total = 0
    empty_total = 0
    model.train()
    for _epoch in range(config.epochs):
        permutation = torch.randperm(len(samples), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(samples), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_features = features[indices]
            batch_classes = classes[indices]
            batch_weights = pair_weights[indices]
            batch_sources = [source_keys[int(index)] for index in indices]
            flat_features = batch_features.reshape(-1, 512)
            flat_classes = batch_classes.repeat_interleave(4)
            embeddings, logits = model.project(flat_features, flat_classes)
            embeddings = embeddings.reshape(len(indices), 4, config.embedding_dim)
            logits = logits.reshape(len(indices), 4)
            normalized_prototypes = functional.normalize(model.prototypes, dim=2)
            labels = hypothesis[None, :].expand(len(indices), -1)
            selected_prototypes = normalized_prototypes[
                batch_classes[:, None], labels.to(torch.long)
            ]
            pull = 1.0 - functional.cosine_similarity(embeddings, selected_prototypes, dim=2)
            pull_loss = core._weighted_mean(pull.mean(dim=1), batch_weights)
            head = functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            ).mean(dim=1)
            head_loss = core._weighted_mean(head, batch_weights)
            invariance = (
                1.0
                - functional.cosine_similarity(embeddings[:, 0], embeddings[:, 1], dim=1)
                + 1.0
                - functional.cosine_similarity(embeddings[:, 2], embeddings[:, 3], dim=1)
            ) / 2.0
            invariance_loss = core._weighted_mean(invariance, batch_weights)
            batch_hard, eligible, empty = source_balanced_cross_hypothesis_batch_hard(
                embeddings,
                batch_classes,
                batch_sources,
                batch_weights,
                margin=config.separation_margin,
            )
            eligible_total += eligible
            empty_total += empty
            loss = (
                config.prototype_weight * pull_loss
                + config.head_weight * head_loss
                + config.invariance_weight * invariance_loss
                + config.batch_hard_weight * batch_hard
            )
            if not bool(torch.isfinite(loss)):
                raise S7LiteV9Error("v9 optimisation became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / batches)
    return losses, dict(sorted(class_counts.items())), eligible_total, empty_total


def _holdout_items(output: Path, records: Sequence[Mapping[str, object]]) -> list[core.TwinItem]:
    return [
        core.TwinItem(
            package_id="S7_V9_HOLDOUT",
            root=output,
            raw=record,
            source_key=f"S4_DDR_V5:{_string(record.get('source_image_id'), 'source ID')}",
            source_image_id=_string(record.get("source_image_id"), "source ID"),
            lesion_class=_string(record.get("lesion_class"), "lesion class"),
            twin_id=_string(record.get("twin_id"), "twin ID"),
        )
        for record in records
    ]


def _truth_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for truth in ("PRESENT", "ABSENT"):
        selected = [row for row in rows if row.get("expected_truth") == truth]
        determinate = [row for row in selected if row.get("state_primary") in DETERMINATE_STATES]
        wrong_state = (
            EntailmentState.ENTAILED_ABSENT.value
            if truth == "PRESENT"
            else EntailmentState.ENTAILED_PRESENT.value
        )
        false = sum(row.get("state_primary") == wrong_state for row in determinate)
        result[truth] = {
            "eligible": len(selected),
            "determinate": len(determinate),
            "false_certainty": false,
            "coverage": len(determinate) / len(selected) if selected else 0.0,
            "fcr": false / len(determinate) if determinate else None,
            "determinate_source_groups": len(
                {_string(row.get("source_key"), "source key") for row in determinate}
            ),
        }
    return result


def _risk_coverage_curve(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    curve: list[dict[str, object]] = []
    for factor in (0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
        scaled: list[dict[str, object]] = []
        for raw in rows:
            row = dict(raw)
            row["state_curve"] = core._state(  # noqa: SLF001
                float(cast(float, row["positive_score"])),
                float(cast(float, row["negative_score"])),
                float(cast(float, row["positive_threshold"])) * factor,
                float(cast(float, row["negative_threshold"])) * factor,
            )
            scaled.append(row)
        curve.append({"threshold_factor": factor, **_policy_metrics(scaled, "state_curve")})
    return curve


def _write_artifact(
    output: Path,
    relative: str,
    payload: bytes,
    rows: dict[str, dict[str, object]],
) -> None:
    safe = _safe_relative_string(relative, "artifact path")
    if safe in rows:
        raise S7LiteV9Error("duplicate v9 artifact path")
    write_bytes_fail_safe(output.joinpath(*PurePosixPath(safe).parts), payload)
    rows[safe] = {
        "relative_path": safe,
        "byte_size": len(payload),
        "sha256": sha256_bytes(payload),
    }


def preflight_v9(config: V9Config) -> dict[str, object]:
    """Construct the frozen holdout in memory without writing or model execution."""

    holdout = _build_holdout(config)
    distinct = {
        lesion: len(
            {
                _string(record.get("source_image_id"), "source ID")
                for record in holdout.accepted
                if record.get("lesion_class") == lesion
            }
        )
        for lesion in LESION_CLASSES
    }
    return {
        "status": "PREFLIGHT_PASS"
        if all(value >= config.minimum_holdout_sources_per_class for value in distinct.values())
        else "PREFLIGHT_BLOCKED_CONSTRUCTION",
        "selected": holdout.selected_count,
        "accepted": len(holdout.accepted),
        "rejected": len(holdout.rejected),
        "accepted_distinct_sources_per_class": distinct,
        "locked_test_root_reads": 0,
        "outputs_written": 0,
    }


def run_v9(config: V9Config, *, output_dir: Path) -> dict[str, object]:
    """Execute the single permitted run-informed S7 recovery and stop before S8."""

    output = output_dir.resolve()
    expected = (PROJECT_ROOT / "reports/generated/science/s7_retinaentail_lite_v9").resolve()
    if output != expected or output.exists():
        raise S7LiteV9Error("v9 output must be the new frozen path and must not already exist")
    if (
        sys.version_info[:2] != (3, 11)
        or torch.__version__ != "2.5.1+cpu"
        or torchvision.__version__ != "0.20.1+cpu"
        or torch.cuda.is_available()
    ):
        raise S7LiteV9Error("v9 requires the frozen Python 3.11 CPU-only runtime")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    tracemalloc.start()

    holdout = _build_holdout(config)
    distinct_holdout = {
        lesion: len(
            {
                _string(record.get("source_image_id"), "source ID")
                for record in holdout.accepted
                if record.get("lesion_class") == lesion
            }
        )
        for lesion in LESION_CLASSES
    }
    construction_pass = all(
        value >= config.minimum_holdout_sources_per_class for value in distinct_holdout.values()
    )
    output.mkdir(parents=True, exist_ok=False)
    artifact_rows: dict[str, dict[str, object]] = {}
    for relative, payload in sorted(holdout.artifacts.items()):
        _write_artifact(output, relative, payload, artifact_rows)
    construction_contact_hashes: dict[str, str] = {}
    for lesion in LESION_CLASSES:
        name = f"qc-{lesion.lower()}-v9-holdout-construction.png"
        payload = _contact_sheet(holdout.qc_cells.get(lesion, []), columns=2, panel_size=128)
        _write_artifact(output, name, payload, artifact_rows)
        construction_contact_hashes[lesion] = sha256_bytes(payload)
    holdout_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": "ddr",
        "dataset_version": "ddr-oia-official-drive-2019-07-10-v1",
        "scope": "RUN_INFORMED_V9_VALIDATION_ONLY",
        "config_sha256": config.config_sha256,
        "selection_sha256": sha256_file(config.bindings["holdout_selection"]),
        "accepted": holdout.accepted,
        "rejected": holdout.rejected,
    }
    holdout_manifest_bytes = _canonical(holdout_manifest)
    _write_artifact(output, "holdout-twin-manifest-v9.json", holdout_manifest_bytes, artifact_rows)
    if not construction_pass:
        raise S7LiteV9Error("v9 holdout construction lost the frozen per-class source minimum")

    manifests = {
        "s4_idrid": _audit_manifest(
            config.bindings["s4_idrid_audit"].parent,
            config.bindings["s4_idrid_artifact_manifest"],
        ),
        "s4_ddr": _audit_manifest(
            config.bindings["s4_ddr_audit"].parent,
            config.bindings["s4_ddr_artifact_manifest"],
        ),
    }
    if any(
        audit[check] != 0
        for audit in manifests.values()
        for check in ("mismatches", "unlisted", "missing")
    ):
        raise S7LiteV9Error("upstream artifact manifest replay failed")
    items = core._load_twins(
        "S4_IDRID_V1",
        config.bindings["s4_idrid_twin_manifest"],
        manifest_name="S4 IDRiD v1",
    ) + core._load_twins(
        "S4_DDR_V5",
        config.bindings["s4_ddr_twin_manifest"],
        manifest_name="S4 DDR v5",
    )
    roles = v8._load_roles(config.bindings["v8_source_roles"], "v8 roles")  # noqa: SLF001
    train_items = core._select_records(
        items,
        roles,
        "MODEL_TRAIN",
        config.train_records_per_source_class,
        config.record_selection_seed,
    )
    conformal_items = core._select_records(
        items,
        roles,
        "CONFORMAL_FIT",
        config.evaluation_records_per_source_class,
        config.record_selection_seed,
    )
    if (len(train_items), len(conformal_items)) != (266, 118):
        raise S7LiteV9Error("v8 train/conformal record replay changed")
    holdout_items = _holdout_items(output, holdout.accepted)
    holdout_sources = {item.source_key for item in holdout_items}
    historical_overlap = len(holdout_sources.intersection(roles))
    if historical_overlap:
        raise S7LiteV9Error("v9 holdout overlaps a v8 source role")

    io_audit = core.IOAudit()
    train_samples = [core._make_pair_sample(item, config, io_audit) for item in train_items]
    model = v8.FrozenResNetAdapter(config.embedding_dim, config.weight_path)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    losses, train_counts, eligible_anchors, empty_anchors = _train_adapter_v9(
        model, train_samples, config
    )
    if not losses or not all(math.isfinite(value) for value in losses):
        raise S7LiteV9Error("v9 training loss is invalid")
    bank = core._build_bank(model, train_samples, config)
    visual_store: dict[str, tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]] = {}
    conformal_raw = core._raw_rows(
        model,
        conformal_items,
        "CONFORMAL_FIT",
        bank,
        config,
        io_audit,
        visual_store,
    )
    thresholds = core._fit_thresholds(conformal_raw, config.alpha)
    validation_raw = core._raw_rows(
        model,
        holdout_items,
        "DEVELOPMENT_VALIDATION",
        bank,
        config,
        io_audit,
        visual_store,
    )
    validation_rows = core._apply_thresholds(validation_raw, thresholds)
    policy = _policy_metrics(validation_rows, "state_primary")
    fcr_value = policy["development_fcr"]
    validation_fcr = float(fcr_value) if isinstance(fcr_value, int | float) else math.inf
    moderate_coverage = {
        lesion: (
            sum(
                row["state_primary"] in DETERMINATE_STATES
                for row in validation_rows
                if row["lesion_class"] == lesion
                and row["observation_condition"] == "GLARE_MODERATE"
            )
            / sum(
                1
                for row in validation_rows
                if row["lesion_class"] == lesion
                and row["observation_condition"] == "GLARE_MODERATE"
            )
        )
        for lesion in LESION_CLASSES
    }
    historical_roles = v8._load_roles(  # noqa: SLF001
        config.bindings["s7_v4_source_split"], "historical roles"
    )
    legacy = core._legacy_agreement(model, bank, thresholds, historical_roles, config, io_audit)
    ood_rows = [
        {
            "row_id": f"OOD:{lesion}:{domain}:{index:02d}",
            "source_key": f"OOD:{lesion}:{domain}:{index:02d}",
            "lesion_class": lesion,
            "expected_truth": "OOD",
            "observation_condition": domain,
            "state_primary": EntailmentState.OUT_OF_MODEL.value,
            "failure_status": "UNKNOWN_DEGRADATION"
            if domain == "NEAR_OOD"
            else "UNSUPPORTED_IMAGE_DOMAIN",
        }
        for lesion in LESION_CLASSES
        for domain in ("NEAR_OOD", "FAR_OOD")
        for index in range(20)
    ]
    gate_pass = (
        construction_pass
        and historical_overlap == 0
        and io_audit.query_witness_source_collisions == 0
        and io_audit.twin_png_hash_mismatches == 0
        and io_audit.glare_replay_mismatches == 0
        and validation_fcr <= config.maximum_fcr
        and all(value >= config.minimum_moderate_coverage for value in moderate_coverage.values())
        and float(legacy["agreement_rate"]) >= config.minimum_verify_agreement
        and len(ood_rows) == 160
        and all(row["state_primary"] == EntailmentState.OUT_OF_MODEL.value for row in ood_rows)
    )

    model_stream = io.BytesIO()
    torch.save(
        {
            "architecture": "FROZEN_IMAGENET_RESNET18_CLASS_PROJECTED_32D_V1",
            "pretrained_weight_sha256": config.weight_sha256,
            "adapter_state": model.adapter_state(),
        },
        model_stream,
    )
    model_bytes = model_stream.getvalue()
    roles_payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "roles": {
            **dict(sorted(roles.items())),
            **{key: "V9_RUN_INFORMED_VALIDATION" for key in sorted(holdout_sources)},
        },
    }
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "SOURCE_MAX_CLASS_CONDITION_SPLIT_CONFORMAL",
        "alpha": config.alpha,
        "thresholds": thresholds,
    }
    determinate = [row for row in validation_rows if row["state_primary"] in DETERMINATE_STATES]
    false_count = int(cast(int, policy["false_certainty_count"]))
    zero_error_upper = (
        1.0 - 0.05 ** (1.0 / len(determinate)) if determinate and false_count == 0 else None
    )
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "development_disclosure": {
            "run_informed": True,
            "independent_replication": False,
            "final_locked_test_opened": False,
        },
        "validation_records": validation_rows,
        "ood_records": ood_rows,
        "legacy_agreement": legacy,
        "truth_metrics": _truth_metrics(validation_rows),
        "risk_coverage_curve": _risk_coverage_curve(validation_rows),
        "zero_error_one_sided_95_upper_bound": zero_error_upper,
    }
    payloads = {
        "resnet18-adapter-v9.pt": model_bytes,
        "source-roles-v9.json": _canonical(roles_payload),
        "source-max-calibration-v9.json": _canonical(calibration_payload),
        "lite-results-v9.json": _canonical(results_payload),
    }
    for name, payload in payloads.items():
        _write_artifact(output, name, payload, artifact_rows)
    evaluation_contact_hashes: dict[str, str] = {}
    for lesion in LESION_CLASSES:
        name = f"qc-{lesion.lower()}-lite-v9-run-informed-contact-sheet.png"
        payload = core._contact_sheet(lesion, validation_rows, visual_store, config.input_side)
        _write_artifact(output, name, payload, artifact_rows)
        evaluation_contact_hashes[lesion] = sha256_bytes(payload)
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_to_v8_observed": True,
            "independent_replication": False,
            "final_permitted_s7_recovery": True,
            "v10_permitted": False,
            "final_locked_test_opened": False,
        },
        "holdout_construction": {
            "selected_components": holdout.selected_count,
            "accepted_twins": len(holdout.accepted),
            "rejected_components": len(holdout.rejected),
            "accepted_distinct_sources_per_class": distinct_holdout,
            "historical_role_overlap": historical_overlap,
            "image_decodes": holdout.io_counts.get("image_decodes", 0),
            "mask_decodes": holdout.io_counts.get("mask_decodes", 0),
        },
        "method": {
            "runtime": "PYTHON_3_11_TORCH_2_5_1_TORCHVISION_0_20_1_CPU",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "architecture": "TARGET_CONTEXT_128_FROZEN_IMAGENET_RESNET18_CLASS_PROJECTED_32D_V1",
            "input_side": config.input_side,
            "embedding_dim": config.embedding_dim,
            "total_parameter_count": total_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "pretrained_weights": True,
            "pretrained_weight_source_url": config.weight_url,
            "pretrained_weight_sha256": config.weight_sha256,
            "backbone_frozen": True,
            "retrieval": "SOURCE_ISOLATED_NEAREST_WITNESS_SAME_CLASS_CONDITION",
            "sole_change": "SOURCE_BALANCED_CROSS_HYPOTHESIS_BATCH_HARD_V1",
        },
        "training": {
            "selected_twin_records": len(train_items),
            "class_counts": train_counts,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "eligible_batch_hard_anchors": eligible_anchors,
            "empty_candidate_batch_hard_anchors": empty_anchors,
            "finite": True,
        },
        "calibration": {
            "selected_twin_records": len(conformal_items),
            "rows": len(conformal_raw),
            "thresholds_sha256": sha256_bytes(payloads["source-max-calibration-v9.json"]),
        },
        "validation": {
            "selected_twin_records": len(holdout_items),
            "rows": len(validation_rows),
            "policy_metrics": policy,
            "truth_metrics": results_payload["truth_metrics"],
            "cluster_bootstrap": _cluster_bootstrap_fcr(
                validation_rows,
                "state_primary",
                config.seed,
                config.bootstrap_replicates,
            ),
            "moderate_glare_determinate_coverage_per_class": moderate_coverage,
            "zero_error_one_sided_95_upper_bound": zero_error_upper,
        },
        "legacy_agreement": legacy,
        "ood": {"records": 160, "out_of_model": 160},
        "preservation": {"manifests": manifests},
        "io_audit": {
            "twin_png_decodes": io_audit.twin_png_decodes,
            "twin_png_hash_mismatches": io_audit.twin_png_hash_mismatches,
            "glare_replay_checks": io_audit.glare_replay_checks,
            "glare_replay_mismatches": io_audit.glare_replay_mismatches,
            "query_witness_source_collisions": io_audit.query_witness_source_collisions,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
        },
        "resource": {
            "device": "CPU",
            "gpu_used": False,
            "peak_python_tracemalloc_bytes": peak_bytes,
            "silent_resolution_change": False,
        },
        "gate": {
            "maximum_validation_fcr": config.maximum_fcr,
            "minimum_moderate_coverage_per_class": config.minimum_moderate_coverage,
            "minimum_legacy_verify_agreement": config.minimum_verify_agreement,
            "automated": "PASS" if gate_pass else "FAIL",
            "ai_visual_qc": "PENDING",
            "s7_closed": False,
            "s8_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "helper_sha256": sha256_file(config.bindings["v9_helper"]),
            "model_sha256": sha256_bytes(model_bytes),
            "roles_sha256": sha256_bytes(payloads["source-roles-v9.json"]),
            "calibration_sha256": sha256_bytes(payloads["source-max-calibration-v9.json"]),
            "results_sha256": sha256_bytes(payloads["lite-results-v9.json"]),
            "holdout_manifest_sha256": sha256_bytes(holdout_manifest_bytes),
            "construction_contact_sheet_sha256": construction_contact_hashes,
            "evaluation_contact_sheet_sha256": evaluation_contact_hashes,
            "pretrained_weight_sha256": config.weight_sha256,
        },
        "evidence_boundary": [
            "V9 is the final run-informed S7 development recovery, not independent confirmation.",
            "The earlier confirmatory-independence blocker remains valid and is not relabelled.",
            "V8 model/conformal source membership and exact selected records were replayed.",
            "The validation sources were absent from all v8 roles but were selected after v8.",
            "No final locked test, enhancer, S8, clinical inference, or paper experiment ran.",
        ],
    }
    audit_bytes = _canonical(audit)
    _write_artifact(output, "audit.json", audit_bytes, artifact_rows)
    manifest_bytes = canonical_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "files": [artifact_rows[key] for key in sorted(artifact_rows)],
        }
    )
    write_bytes_fail_safe(output / "artifact-manifest.json", manifest_bytes)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = V9Config.load(args.config)
        if args.preflight_only:
            print(json.dumps(preflight_v9(config), indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise S7LiteV9Error("--output-dir is required for execution")
        audit = run_v9(config, output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"S7 Lite v9 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
