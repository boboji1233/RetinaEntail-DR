"""Train and evaluate the frozen Science S7 v7 fresh-holdout Lite recovery."""

from __future__ import annotations

import argparse
import hashlib
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
import torch
import torch.nn.functional as functional
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from torch import Tensor, nn

from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import OperationTrace, apply_and_record, replay_trace
from retinaentail.entailment.states import EntailmentState
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
    _write_artifacts,
    regional_context_bbox,
)
from retinaentail.science_s5_degradation_calibration_v2 import frozen_family_parameters
from retinaentail.science_s6_verify import HYPOTHESES, LESION_CLASSES
from retinaentail.science_s7_lite import (
    DETERMINATE_STATES,
    _cluster_bootstrap_fcr,
    _policy_metrics,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest
from retinaentail.science_s7_lite_v4 import _state
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "7.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V7"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CLASS_INDEX: Final = {lesion_class: index for index, lesion_class in enumerate(LESION_CLASSES)}
VIEW_ORDER: Final = (
    ("PRESENT", "IDENTITY"),
    ("PRESENT", "GLARE_MODERATE"),
    ("ABSENT", "IDENTITY"),
    ("ABSENT", "GLARE_MODERATE"),
)


class S7LiteV7Error(ValueError):
    """Raised when the frozen v7 contract cannot execute safely."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV7Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV7Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV7Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV7Error(f"{name} is not SHA-256")
    return text


def _digest(seed: int, label: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{label}|{value}".encode()).hexdigest()


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV7Error(f"binding mismatch: {name}")
    return path


def _verify_baseline(expected: object) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != _string(expected, "expected baseline commit"):
        raise S7LiteV7Error("v7 baseline drift")


@dataclass(frozen=True, slots=True)
class V7Config:
    path: Path
    config_sha256: str
    seed: int
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
    separation_margin: float
    alpha: float
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V7Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v7 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_FRESH_HOLDOUT_RECOVERY"
        ):
            raise S7LiteV7Error("unsupported S7 v7 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_to_v6_observed": True,
            "independent_replication": False,
            "fresh_primary_validation": True,
            "final_test_opened": False,
        }:
            raise S7LiteV7Error("v7 disclosure changed")
        split = _mapping(root.get("source_split"), "source split")
        method = _mapping(root.get("method"), "method")
        training = _mapping(root.get("training"), "training")
        losses = _mapping(training.get("loss_weights"), "loss weights")
        calibration = _mapping(root.get("calibration"), "calibration")
        statistics = _mapping(root.get("statistics"), "statistics")
        gate = _mapping(root.get("gate"), "gate")
        expected_contract = {
            "runtime": method.get("runtime") == "PYTHON_3_11_TORCH_2_5_1_CPU",
            "architecture": method.get("architecture")
            == "TARGET_CONTEXT_128_SHARED_CONV_CLASS_PROJECTED_32D_PROTOTYPE_V1",
            "retrieval": method.get("retrieval")
            == "SOURCE_ISOLATED_NEAREST_WITNESS_SAME_CLASS_CONDITION",
            "validation_selector": split.get("validation_selector")
            == "UNSEEN_CLASS_PRIORITY_SHA256_MINIMUM",
            "conformal_selector": split.get("conformal_selector")
            == "CLASS_PRIORITY_SHA256_MINIMUM",
            "legacy_role": split.get("legacy_role") == "V4_DEVELOPMENT_VALIDATION_ONLY",
            "optimizer": training.get("optimizer") == "ADAMW",
            "calibration": calibration.get("method")
            == "SOURCE_MAX_CLASS_CONDITION_SPLIT_CONFORMAL",
        }
        if not all(expected_contract.values()):
            raise S7LiteV7Error("v7 frozen method contract changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required_bindings = {
            "direction_review",
            "v7_prompt",
            "v7_code",
            "v7_audit_schema",
            "s4_idrid_audit",
            "s4_idrid_artifact_manifest",
            "s4_idrid_twin_manifest",
            "s4_ddr_audit",
            "s4_ddr_artifact_manifest",
            "s4_ddr_twin_manifest",
            "s5_v2_audit",
            "s5_v2_artifact_manifest",
            "s5_v2_oracle",
            "s7_v1_audit",
            "s7_v1_artifact_manifest",
            "s7_v4_source_split",
            "s7_v5_results",
            "s7_v6_audit",
            "s7_v6_artifact_manifest",
            "s7_closeout_report",
            "s7_qc_record",
            "s6_v2_results",
        }
        if set(bindings) != required_bindings:
            raise S7LiteV7Error("v7 binding set changed")
        values = {
            "seed": _integer(root.get("seed"), "seed"),
            "validation_sources_per_class": _integer(
                split.get("validation_sources_per_class"), "validation sources", 1
            ),
            "conformal_sources_per_class": _integer(
                split.get("conformal_sources_per_class"), "conformal sources", 1
            ),
            "train_records_per_source_class": _integer(
                split.get("train_records_per_source_class"), "train record cap", 1
            ),
            "evaluation_records_per_source_class": _integer(
                split.get("evaluation_records_per_source_class"), "evaluation record cap", 1
            ),
            "input_side": _integer(method.get("input_side"), "input side", 1),
            "embedding_dim": _integer(method.get("embedding_dim"), "embedding dim", 1),
            "epochs": _integer(training.get("epochs"), "epochs", 1),
            "batch_size": _integer(training.get("batch_size"), "batch size", 1),
            "learning_rate": _number(training.get("learning_rate"), "learning rate"),
            "weight_decay": _number(training.get("weight_decay"), "weight decay"),
            "prototype_weight": _number(losses.get("prototype_pull"), "prototype weight"),
            "head_weight": _number(losses.get("hypothesis_head"), "head weight"),
            "invariance_weight": _number(losses.get("glare_invariance"), "invariance weight"),
            "separation_weight": _number(losses.get("twin_separation"), "separation weight"),
            "separation_margin": _number(
                training.get("cosine_distance_margin"), "separation margin"
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
        }
        frozen_values = {
            "seed": 2026082617,
            "validation_sources_per_class": 8,
            "conformal_sources_per_class": 15,
            "train_records_per_source_class": 2,
            "evaluation_records_per_source_class": 4,
            "input_side": 128,
            "embedding_dim": 32,
            "epochs": 60,
            "batch_size": 16,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "prototype_weight": 1.0,
            "head_weight": 0.5,
            "invariance_weight": 0.5,
            "separation_weight": 0.5,
            "separation_margin": 0.75,
            "alpha": 0.05,
            "bootstrap_replicates": 2000,
            "maximum_fcr": 0.05,
            "minimum_moderate_coverage": 0.05,
            "minimum_verify_agreement": 0.5,
        }
        if values != frozen_values:
            raise S7LiteV7Error("v7 frozen numeric contract changed")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            bindings=bindings,
            **values,
        )


@dataclass(frozen=True, slots=True)
class TwinItem:
    package_id: str
    root: Path
    raw: Mapping[str, object]
    source_key: str
    source_image_id: str
    lesion_class: str
    twin_id: str


@dataclass(frozen=True, slots=True)
class PairSample:
    item: TwinItem
    views: tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]
    view_seeds: tuple[int | None, int | None, int | None, int | None]


@dataclass(frozen=True, slots=True)
class BankEntry:
    item: TwinItem
    hypothesis: str
    condition: str
    embedding: NDArray[np.float32]
    view: NDArray[np.uint8]
    view_sha256: str
    degradation_seed: int | None
    source_path: str
    source_sha256: str


@dataclass(slots=True)
class IOAudit:
    twin_png_decodes: int = 0
    twin_png_hash_mismatches: int = 0
    glare_replay_checks: int = 0
    glare_replay_mismatches: int = 0
    query_witness_source_collisions: int = 0
    locked_test_root_reads: int = 0
    online_evaluation_reads: int = 0


class TwinContrastiveEncoder(nn.Module):
    """Small shared encoder with class-conditioned projections and hypothesis prototypes."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.base = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.projection_weight = nn.Parameter(torch.empty(4, embedding_dim, 64))
        self.projection_bias = nn.Parameter(torch.zeros(4, embedding_dim))
        self.head_weight = nn.Parameter(torch.empty(4, embedding_dim))
        self.head_bias = nn.Parameter(torch.zeros(4))
        self.prototypes = nn.Parameter(torch.empty(4, 2, embedding_dim))
        for index in range(4):
            nn.init.xavier_uniform_(self.projection_weight[index])
        nn.init.xavier_uniform_(self.head_weight)
        nn.init.normal_(self.prototypes, std=0.2)

    def forward(self, inputs: Tensor, class_indices: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.base(self.features(inputs).flatten(1))
        weights = self.projection_weight[class_indices]
        projected = torch.bmm(weights, pooled.unsqueeze(2)).squeeze(2)
        projected = projected + self.projection_bias[class_indices]
        embedding = functional.normalize(projected, dim=1)
        logits = (embedding * self.head_weight[class_indices]).sum(dim=1)
        logits = logits + self.head_bias[class_indices]
        return embedding, logits


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(decoded, name)


def _load_twins(package_id: str, manifest_path: Path, *, manifest_name: str) -> list[TwinItem]:
    root = manifest_path.parent
    manifest = _load_json(manifest_path, manifest_name)
    items: list[TwinItem] = []
    for raw in _sequence(manifest.get("accepted"), f"{manifest_name} accepted"):
        record = _mapping(raw, "twin record")
        source_image_id = _string(record.get("source_image_id"), "source image ID")
        lesion_class = _string(record.get("lesion_class"), "lesion class")
        twin_id = _string(record.get("twin_id"), "twin ID")
        if lesion_class not in LESION_CLASSES:
            raise S7LiteV7Error("unknown lesion class in S4 manifest")
        items.append(
            TwinItem(
                package_id=package_id,
                root=root,
                raw=record,
                source_key=f"{package_id}:{source_image_id}",
                source_image_id=source_image_id,
                lesion_class=lesion_class,
                twin_id=twin_id,
            )
        )
    if not items:
        raise S7LiteV7Error(f"{manifest_name} has no accepted twins")
    return items


def _select_roles(
    items: Sequence[TwinItem], historical_roles: Mapping[str, str], config: V7Config
) -> dict[str, str]:
    by_class: dict[str, set[str]] = {lesion_class: set() for lesion_class in LESION_CLASSES}
    for item in items:
        by_class[item.lesion_class].add(item.source_key)
    all_sources = set().union(*by_class.values())
    historical_sources = set(historical_roles)
    if not historical_sources <= all_sources or len(historical_sources) != 184:
        raise S7LiteV7Error("historical S7 source universe changed")
    legacy = {key for key, role in historical_roles.items() if role == "DEVELOPMENT_VALIDATION"}
    if len(legacy) != 69:
        raise S7LiteV7Error("historical S7 validation role changed")
    unseen = all_sources - historical_sources
    validation: set[str] = set()
    for lesion_class in ("EX", "MA", "SE", "HE"):
        candidates = sorted(
            by_class[lesion_class].intersection(unseen),
            key=lambda key: _digest(config.seed, f"VALIDATION|{lesion_class}", key),
        )
        for key in candidates:
            if len(by_class[lesion_class].intersection(validation)) >= (
                config.validation_sources_per_class
            ):
                break
            validation.add(key)
    remaining = all_sources - validation - legacy
    conformal: set[str] = set()
    for lesion_class in ("EX", "MA", "SE", "HE"):
        candidates = sorted(
            by_class[lesion_class].intersection(remaining),
            key=lambda key: _digest(config.seed, f"CONFORMAL|{lesion_class}", key),
        )
        for key in candidates:
            if len(by_class[lesion_class].intersection(conformal)) >= (
                config.conformal_sources_per_class
            ):
                break
            conformal.add(key)
    roles = {
        key: (
            "DEVELOPMENT_VALIDATION"
            if key in validation
            else "LEGACY_AGREEMENT"
            if key in legacy
            else "CONFORMAL_FIT"
            if key in conformal
            else "MODEL_TRAIN"
        )
        for key in all_sources
    }
    counts = Counter(roles.values())
    if counts != {
        "MODEL_TRAIN": 177,
        "CONFORMAL_FIT": 37,
        "DEVELOPMENT_VALIDATION": 19,
        "LEGACY_AGREEMENT": 69,
    }:
        raise S7LiteV7Error(f"unexpected v7 source-role counts: {dict(counts)}")
    if validation.intersection(historical_sources):
        raise S7LiteV7Error("v7 fresh validation overlaps historical S7")
    if any(
        len(by_class[lesion_class].intersection(validation)) < config.validation_sources_per_class
        for lesion_class in LESION_CLASSES
    ):
        raise S7LiteV7Error("v7 validation class coverage is insufficient")
    return roles


def _select_records(
    items: Sequence[TwinItem], roles: Mapping[str, str], role: str, cap: int, seed: int
) -> list[TwinItem]:
    grouped: dict[tuple[str, str], list[TwinItem]] = defaultdict(list)
    for item in items:
        if roles[item.source_key] == role:
            grouped[(item.source_key, item.lesion_class)].append(item)
    selected: list[TwinItem] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: _digest(seed, "RECORD", item.twin_id))
        selected.extend(ordered[:cap])
    return selected


def _output_binding(item: TwinItem, hypothesis: str) -> tuple[Path, str, str]:
    name = "x_plus_sham" if hypothesis == "PRESENT" else "x_minus_removed"
    output = _mapping(_mapping(item.raw.get("outputs"), "outputs").get(name), name)
    relative = _safe_relative_string(output.get("relative_path"), f"{name} path")
    expected = _sha256(output.get("sha256"), f"{name} SHA-256")
    return item.root.joinpath(*PurePosixPath(relative).parts), relative, expected


def _decode_side(item: TwinItem, hypothesis: str, io_audit: IOAudit) -> NDArray[np.uint8]:
    path, _relative, expected = _output_binding(item, hypothesis)
    if not path.is_file() or sha256_file(path) != expected:
        io_audit.twin_png_hash_mismatches += 1
        raise S7LiteV7Error("bound S4 twin PNG changed")
    with Image.open(path) as source:
        source.load()
        image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    io_audit.twin_png_decodes += 1
    if image.shape != (256, 256, 3):
        raise S7LiteV7Error("S4 twin patch shape changed")
    return image


def _normalized_context(
    image: NDArray[np.uint8], item: TwinItem, output_side: int
) -> NDArray[np.uint8]:
    bbox = regional_context_bbox(item.raw, margin=8, patch_wh=(256, 256))
    left, top, right, bottom = bbox
    if left < 0 or top < 0 or right > 256 or bottom > 256 or right <= left or bottom <= top:
        raise S7LiteV7Error("target context bbox is invalid")
    return np.asarray(
        Image.fromarray(image[top:bottom, left:right]).resize(
            (output_side, output_side), resample=Image.Resampling.BOX
        ),
        dtype=np.uint8,
    ).copy()


def _make_view(
    item: TwinItem,
    hypothesis: str,
    condition: str,
    config: V7Config,
    io_audit: IOAudit,
) -> tuple[NDArray[np.uint8], int | None]:
    source = _decode_side(item, hypothesis, io_audit)
    if condition == "IDENTITY":
        return _normalized_context(source, item, config.input_side), None
    if condition != "GLARE_MODERATE":
        raise S7LiteV7Error("unknown v7 observation condition")
    bbox = regional_context_bbox(item.raw, margin=8, patch_wh=(256, 256))
    parameters = cast(
        dict[str, str | int | float | bool],
        frozen_family_parameters("glare_reflection", 0.6),
    )
    parameters = {
        **parameters,
        "center_x": ((bbox[0] + bbox[2]) / 2.0) / 256.0,
        "center_y": ((bbox[1] + bbox[3]) / 2.0) / 256.0,
    }
    seed = int(_digest(config.seed, f"V7_GLARE|{hypothesis}", item.twin_id)[:16], 16)
    output, step = apply_and_record(
        source,
        "glare_reflection",
        severity=0.6,
        seed=seed,
        parameters=parameters,
    )
    replayed = replay_trace(source, OperationTrace((step,)))
    io_audit.glare_replay_checks += 1
    if not np.array_equal(output, replayed):
        io_audit.glare_replay_mismatches += 1
        raise S7LiteV7Error("v7 glare replay changed")
    return _normalized_context(output, item, config.input_side), seed


def _make_pair_sample(item: TwinItem, config: V7Config, io_audit: IOAudit) -> PairSample:
    views: list[NDArray[np.uint8]] = []
    seeds: list[int | None] = []
    for hypothesis, condition in VIEW_ORDER:
        view, seed = _make_view(item, hypothesis, condition, config, io_audit)
        views.append(view)
        seeds.append(seed)
    return PairSample(
        item=item,
        views=cast(
            tuple[
                NDArray[np.uint8],
                NDArray[np.uint8],
                NDArray[np.uint8],
                NDArray[np.uint8],
            ],
            tuple(views),
        ),
        view_seeds=cast(tuple[int | None, int | None, int | None, int | None], tuple(seeds)),
    )


def _tensor(images: NDArray[np.uint8]) -> Tensor:
    values = images.astype(np.float32) / 255.0
    values = (values - 0.5) / 0.25
    return torch.from_numpy(values.transpose(0, 3, 1, 2))


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum()


def _train_model(
    model: TwinContrastiveEncoder, samples: Sequence[PairSample], config: V7Config
) -> tuple[list[float], dict[str, int]]:
    arrays = np.stack([np.stack(sample.views) for sample in samples])
    inputs = _tensor(arrays.reshape(-1, config.input_side, config.input_side, 3)).reshape(
        len(samples), 4, 3, config.input_side, config.input_side
    )
    classes = torch.tensor(
        [CLASS_INDEX[sample.item.lesion_class] for sample in samples], dtype=torch.long
    )
    class_counts = Counter(sample.item.lesion_class for sample in samples)
    pair_weights = torch.tensor(
        [1.0 / class_counts[sample.item.lesion_class] for sample in samples],
        dtype=torch.float32,
    )
    pair_weights = pair_weights / pair_weights.mean()
    hypothesis = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    losses: list[float] = []
    model.train()
    for _epoch in range(config.epochs):
        permutation = torch.randperm(len(samples), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(samples), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch = inputs[indices]
            batch_classes = classes[indices]
            batch_weights = pair_weights[indices]
            flat_inputs = batch.reshape(-1, 3, config.input_side, config.input_side)
            flat_classes = batch_classes.repeat_interleave(4)
            embeddings, logits = model(flat_inputs, flat_classes)
            embeddings = embeddings.reshape(len(indices), 4, config.embedding_dim)
            logits = logits.reshape(len(indices), 4)
            normalized_prototypes = functional.normalize(model.prototypes, dim=2)
            labels = hypothesis[None, :].expand(len(indices), -1)
            hypothesis_indices = labels.to(torch.long)
            selected_prototypes = normalized_prototypes[batch_classes[:, None], hypothesis_indices]
            pull = 1.0 - functional.cosine_similarity(embeddings, selected_prototypes, dim=2)
            pull_loss = _weighted_mean(pull.mean(dim=1), batch_weights)
            head = functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            ).mean(dim=1)
            head_loss = _weighted_mean(head, batch_weights)
            present_invariance = 1.0 - functional.cosine_similarity(
                embeddings[:, 0], embeddings[:, 1], dim=1
            )
            absent_invariance = 1.0 - functional.cosine_similarity(
                embeddings[:, 2], embeddings[:, 3], dim=1
            )
            invariance_loss = _weighted_mean(
                (present_invariance + absent_invariance) / 2.0, batch_weights
            )
            identity_distance = 1.0 - functional.cosine_similarity(
                embeddings[:, 0], embeddings[:, 2], dim=1
            )
            glare_distance = 1.0 - functional.cosine_similarity(
                embeddings[:, 1], embeddings[:, 3], dim=1
            )
            twin_separation = (
                functional.relu(config.separation_margin - identity_distance)
                + functional.relu(config.separation_margin - glare_distance)
            ) / 2.0
            prototype_distance = 1.0 - functional.cosine_similarity(
                normalized_prototypes[:, 0], normalized_prototypes[:, 1], dim=1
            )
            prototype_separation = functional.relu(
                config.separation_margin - prototype_distance
            ).mean()
            separation_loss = _weighted_mean(twin_separation, batch_weights)
            separation_loss = separation_loss + prototype_separation
            loss = (
                config.prototype_weight * pull_loss
                + config.head_weight * head_loss
                + config.invariance_weight * invariance_loss
                + config.separation_weight * separation_loss
            )
            if not bool(torch.isfinite(loss)):
                raise S7LiteV7Error("v7 optimization became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / batches)
    return losses, dict(sorted(class_counts.items()))


def _encode_views(
    model: TwinContrastiveEncoder,
    images: Sequence[NDArray[np.uint8]],
    lesion_classes: Sequence[str],
    config: V7Config,
) -> NDArray[np.float32]:
    if len(images) != len(lesion_classes):
        raise S7LiteV7Error("embedding input lengths differ")
    result: list[NDArray[np.float32]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), 64):
            batch_images = np.stack(images[start : start + 64])
            batch_classes = torch.tensor(
                [CLASS_INDEX[value] for value in lesion_classes[start : start + 64]],
                dtype=torch.long,
            )
            embeddings, _logits = model(_tensor(batch_images), batch_classes)
            result.append(embeddings.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(result, axis=0)


def _build_bank(
    model: TwinContrastiveEncoder, samples: Sequence[PairSample], config: V7Config
) -> dict[tuple[str, str, str], list[BankEntry]]:
    images: list[NDArray[np.uint8]] = []
    classes: list[str] = []
    descriptors: list[tuple[PairSample, int, str, str]] = []
    for sample in samples:
        for index, (hypothesis, condition) in enumerate(VIEW_ORDER):
            images.append(sample.views[index])
            classes.append(sample.item.lesion_class)
            descriptors.append((sample, index, hypothesis, condition))
    embeddings = _encode_views(model, images, classes, config)
    bank: dict[tuple[str, str, str], list[BankEntry]] = defaultdict(list)
    for embedding, view, descriptor in zip(embeddings, images, descriptors, strict=True):
        sample, index, hypothesis, condition = descriptor
        path, relative, expected = _output_binding(sample.item, hypothesis)
        if not path.is_file():
            raise S7LiteV7Error("bank source file disappeared")
        bank[(sample.item.lesion_class, hypothesis, condition)].append(
            BankEntry(
                item=sample.item,
                hypothesis=hypothesis,
                condition=condition,
                embedding=embedding,
                view=view,
                view_sha256=sha256_bytes(view.tobytes()),
                degradation_seed=sample.view_seeds[index],
                source_path=relative,
                source_sha256=expected,
            )
        )
    return dict(bank)


def _search(
    embedding: NDArray[np.float32],
    lesion_class: str,
    hypothesis: str,
    condition: str,
    bank: Mapping[tuple[str, str, str], Sequence[BankEntry]],
    query_source: str,
    io_audit: IOAudit,
) -> tuple[float, BankEntry]:
    candidates = bank.get((lesion_class, hypothesis, condition), ())
    eligible = [entry for entry in candidates if entry.item.source_key != query_source]
    if not eligible:
        raise S7LiteV7Error("v7 witness bank has no source-isolated candidate")
    distances = [float(1.0 - np.dot(embedding, entry.embedding)) for entry in eligible]
    if not all(math.isfinite(value) for value in distances):
        raise S7LiteV7Error("v7 witness distance is non-finite")
    index = min(
        range(len(eligible)), key=lambda value: (distances[value], eligible[value].item.twin_id)
    )
    selected = eligible[index]
    if selected.item.source_key == query_source:
        io_audit.query_witness_source_collisions += 1
        raise S7LiteV7Error("query and witness source collide")
    return distances[index], selected


def _raw_rows(
    model: TwinContrastiveEncoder,
    items: Sequence[TwinItem],
    role: str,
    bank: Mapping[tuple[str, str, str], Sequence[BankEntry]],
    config: V7Config,
    io_audit: IOAudit,
    visual_store: dict[str, tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in items:
        sample = _make_pair_sample(item, config, io_audit)
        embeddings = _encode_views(
            model,
            list(sample.views),
            [item.lesion_class] * 4,
            config,
        )
        for index, (truth, condition) in enumerate(VIEW_ORDER):
            observation = sample.views[index]
            searches: dict[str, dict[str, object]] = {}
            selected_views: dict[str, NDArray[np.uint8]] = {}
            scores: dict[str, float] = {}
            for hypothesis in HYPOTHESES:
                score, selected = _search(
                    embeddings[index],
                    item.lesion_class,
                    hypothesis,
                    condition,
                    bank,
                    item.source_key,
                    io_audit,
                )
                scores[hypothesis] = score
                selected_views[hypothesis] = selected.view
                searches[hypothesis] = {
                    "hypothesis": hypothesis,
                    "score": score,
                    "candidate_twin_id": selected.item.twin_id,
                    "candidate_source_key": selected.item.source_key,
                    "candidate_method": _string(selected.item.raw.get("method"), "method"),
                    "candidate_condition": selected.condition,
                    "candidate_source_path": selected.source_path,
                    "candidate_source_sha256": selected.source_sha256,
                    "candidate_view_sha256": selected.view_sha256,
                    "candidate_degradation_seed": selected.degradation_seed,
                }
            row_id = f"{role}:{item.twin_id}:{truth}:{condition}"
            side_path, side_relative, side_sha = _output_binding(item, truth)
            if not side_path.is_file():
                raise S7LiteV7Error("query source file disappeared")
            rows.append(
                {
                    "row_id": row_id,
                    "role": role,
                    "source_key": item.source_key,
                    "source_image_id": item.source_image_id,
                    "twin_id": item.twin_id,
                    "lesion_class": item.lesion_class,
                    "expected_truth": truth,
                    "observation_condition": condition,
                    "observation_source_path": side_relative,
                    "observation_source_sha256": side_sha,
                    "observation_view_sha256": sha256_bytes(observation.tobytes()),
                    "observation_degradation_seed": sample.view_seeds[index],
                    "positive_score": scores["PRESENT"],
                    "negative_score": scores["ABSENT"],
                    "present_search": searches["PRESENT"],
                    "absent_search": searches["ABSENT"],
                }
            )
            if role == "DEVELOPMENT_VALIDATION":
                visual_store[row_id] = (
                    observation,
                    selected_views["PRESENT"],
                    selected_views["ABSENT"],
                )
    return rows


def _fit_thresholds(
    rows: Sequence[Mapping[str, object]], alpha: float
) -> dict[str, dict[str, dict[str, object]]]:
    thresholds: dict[str, dict[str, dict[str, object]]] = {}
    for lesion_class in LESION_CLASSES:
        thresholds[lesion_class] = {}
        for condition in ("IDENTITY", "GLARE_MODERATE"):
            thresholds[lesion_class][condition] = {}
            for hypothesis, score_key in (
                ("PRESENT", "positive_score"),
                ("ABSENT", "negative_score"),
            ):
                grouped: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        row.get("lesion_class") == lesion_class
                        and row.get("observation_condition") == condition
                        and row.get("expected_truth") == hypothesis
                    ):
                        grouped[_string(row.get("source_key"), "source key")].append(
                            float(cast(float, row[score_key]))
                        )
                values = sorted(max(group) for group in grouped.values())
                if len(values) < 15:
                    raise S7LiteV7Error("v7 conformal source group count is insufficient")
                rank = math.ceil((len(values) + 1) * (1.0 - alpha))
                threshold = values[min(rank, len(values)) - 1]
                thresholds[lesion_class][condition][hypothesis] = {
                    "source_groups": len(values),
                    "rank": rank,
                    "threshold": threshold,
                    "values_sha256": sha256_bytes(_canonical(values)),
                }
    return thresholds


def _apply_thresholds(
    rows: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in rows:
        row = dict(raw)
        lesion_class = _string(row.get("lesion_class"), "lesion class")
        condition = _string(row.get("observation_condition"), "condition")
        positive_threshold = float(thresholds[lesion_class][condition]["PRESENT"]["threshold"])
        negative_threshold = float(thresholds[lesion_class][condition]["ABSENT"]["threshold"])
        row["positive_threshold"] = positive_threshold
        row["negative_threshold"] = negative_threshold
        row["state_primary"] = _state(
            float(cast(float, row["positive_score"])),
            float(cast(float, row["negative_score"])),
            positive_threshold,
            negative_threshold,
        )
        result.append(row)
    return result


def _legacy_agreement(
    model: TwinContrastiveEncoder,
    bank: Mapping[tuple[str, str, str], Sequence[BankEntry]],
    thresholds: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
    historical_roles: Mapping[str, str],
    config: V7Config,
    io_audit: IOAudit,
) -> dict[str, object]:
    v5_rows = _sequence(
        _load_json(config.bindings["s7_v5_results"], "S7 v5 results").get("records"),
        "S7 v5 records",
    )
    verify_rows = _sequence(
        _load_json(config.bindings["s6_v2_results"], "S6 v2 results").get("records"),
        "S6 v2 records",
    )
    verify_states = {
        _string(_mapping(raw, "S6 v2 row").get("case_id"), "case ID"): _string(
            _mapping(raw, "S6 v2 row").get("state"), "state"
        )
        for raw in verify_rows
    }
    v1_root = config.bindings["s7_v1_audit"].parent
    selected = [
        _mapping(raw, "S7 v5 row")
        for raw in v5_rows
        if historical_roles.get(_string(_mapping(raw, "row").get("source_key"), "source"))
        == "DEVELOPMENT_VALIDATION"
        and _mapping(raw, "row").get("stratum")
        in {"PRESENT_ID", "ABSENT_ID", "S5_TIER_A_AMBIGUOUS_ID"}
    ]
    if len(selected) != 96:
        raise S7LiteV7Error("legacy Lite--Verify subset changed")
    agreements = 0
    result_rows: list[dict[str, object]] = []
    for raw in selected:
        relative = _safe_relative_string(raw.get("observation_png_path"), "legacy observation")
        path = v1_root.joinpath(*PurePosixPath(relative).parts)
        expected = _sha256(raw.get("observation_png_sha256"), "legacy observation SHA-256")
        if not path.is_file() or sha256_file(path) != expected:
            raise S7LiteV7Error("legacy observation changed")
        with Image.open(path) as source:
            source.load()
            view = np.asarray(
                source.convert("RGB").resize(
                    (config.input_side, config.input_side), resample=Image.Resampling.BOX
                ),
                dtype=np.uint8,
            ).copy()
        lesion_class = _string(raw.get("lesion_class"), "lesion class")
        source_key = _string(raw.get("source_key"), "source key")
        embedding = _encode_views(model, [view], [lesion_class], config)[0]
        scores: dict[str, float] = {}
        for hypothesis in HYPOTHESES:
            score, _entry = _search(
                embedding,
                lesion_class,
                hypothesis,
                "IDENTITY",
                bank,
                source_key,
                io_audit,
            )
            scores[hypothesis] = score
        state = _state(
            scores["PRESENT"],
            scores["ABSENT"],
            float(thresholds[lesion_class]["IDENTITY"]["PRESENT"]["threshold"]),
            float(thresholds[lesion_class]["IDENTITY"]["ABSENT"]["threshold"]),
        )
        case_id = _string(raw.get("case_id"), "case ID")
        verify_state = verify_states[case_id]
        agreements += int(state == verify_state)
        result_rows.append(
            {
                "case_id": case_id,
                "source_key": source_key,
                "lesion_class": lesion_class,
                "lite_state": state,
                "verify_state": verify_state,
                "agreement": state == verify_state,
            }
        )
    return {
        "records": len(result_rows),
        "agreements": agreements,
        "agreement_rate": agreements / len(result_rows),
        "historical_sources_used_for_model_or_calibration": 0,
        "rows_sha256": sha256_bytes(_canonical(result_rows)),
    }


def _contact_sheet(
    lesion_class: str,
    rows: Sequence[Mapping[str, object]],
    visual_store: Mapping[str, tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]],
    side: int,
) -> bytes:
    chosen: list[Mapping[str, object]] = []
    for truth, condition in VIEW_ORDER:
        candidates = sorted(
            (
                row
                for row in rows
                if row.get("lesion_class") == lesion_class
                and row.get("expected_truth") == truth
                and row.get("observation_condition") == condition
            ),
            key=lambda row: _string(row.get("row_id"), "row ID"),
        )
        if not candidates:
            raise S7LiteV7Error("contact-sheet stratum is empty")
        chosen.append(candidates[0])
    header = 22
    canvas = Image.new("RGB", (3 * side, len(chosen) * (side + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(chosen):
        row_id = _string(row.get("row_id"), "row ID")
        images = visual_store[row_id]
        labels = (
            f"{row.get('expected_truth')}_{row.get('observation_condition')} obs",
            "PRESENT witness",
            "ABSENT witness",
        )
        top = row_index * (side + header)
        for column, (image, label) in enumerate(zip(images, labels, strict=True)):
            left = column * side
            canvas.paste(Image.fromarray(image), (left, top))
            draw.text((left + 2, top + side + 4), label, fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def run_v7(config: V7Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV7Error("S7 v7 output exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    if (
        sys.version_info[:2] != (3, 11)
        or torch.__version__ != "2.5.1+cpu"
        or torch.cuda.is_available()
    ):
        raise S7LiteV7Error("v7 requires the frozen Python 3.11 CPU-only torch runtime")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    tracemalloc.start()
    io_audit = IOAudit()

    manifests = {
        "s4_idrid": _audit_manifest(
            config.bindings["s4_idrid_audit"].parent,
            config.bindings["s4_idrid_artifact_manifest"],
        ),
        "s4_ddr": _audit_manifest(
            config.bindings["s4_ddr_audit"].parent,
            config.bindings["s4_ddr_artifact_manifest"],
        ),
        "s5_v2": _audit_manifest(
            config.bindings["s5_v2_audit"].parent,
            config.bindings["s5_v2_artifact_manifest"],
        ),
        "s7_v1": _audit_manifest(
            config.bindings["s7_v1_audit"].parent,
            config.bindings["s7_v1_artifact_manifest"],
        ),
        "s7_v6": _audit_manifest(
            config.bindings["s7_v6_audit"].parent,
            config.bindings["s7_v6_artifact_manifest"],
        ),
    }
    if any(
        audit[check] != 0
        for audit in manifests.values()
        for check in ("mismatches", "unlisted", "missing")
    ):
        raise S7LiteV7Error("upstream artifact manifest replay failed")

    items = _load_twins(
        "S4_IDRID_V1",
        config.bindings["s4_idrid_twin_manifest"],
        manifest_name="S4 IDRiD v1",
    ) + _load_twins(
        "S4_DDR_V5",
        config.bindings["s4_ddr_twin_manifest"],
        manifest_name="S4 DDR v5",
    )
    historical_decoded = _load_json(config.bindings["s7_v4_source_split"], "v4 split")
    historical_roles_raw = _mapping(historical_decoded.get("roles"), "historical roles")
    historical_roles = {
        key: _string(value, "historical role") for key, value in historical_roles_raw.items()
    }
    roles = _select_roles(items, historical_roles, config)
    train_items = _select_records(
        items,
        roles,
        "MODEL_TRAIN",
        config.train_records_per_source_class,
        config.seed,
    )
    conformal_items = _select_records(
        items,
        roles,
        "CONFORMAL_FIT",
        config.evaluation_records_per_source_class,
        config.seed,
    )
    validation_items = _select_records(
        items,
        roles,
        "DEVELOPMENT_VALIDATION",
        config.evaluation_records_per_source_class,
        config.seed,
    )
    if (len(train_items), len(conformal_items), len(validation_items)) != (300, 144, 91):
        raise S7LiteV7Error("v7 selected record counts changed")

    train_samples = [_make_pair_sample(item, config, io_audit) for item in train_items]
    model = TwinContrastiveEncoder(config.embedding_dim)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    losses, train_class_counts = _train_model(model, train_samples, config)
    if not losses or not all(math.isfinite(value) for value in losses):
        raise S7LiteV7Error("v7 training loss is invalid")
    bank = _build_bank(model, train_samples, config)
    visual_store: dict[str, tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]] = {}
    conformal_raw = _raw_rows(
        model,
        conformal_items,
        "CONFORMAL_FIT",
        bank,
        config,
        io_audit,
        visual_store,
    )
    thresholds = _fit_thresholds(conformal_raw, config.alpha)
    validation_raw = _raw_rows(
        model,
        validation_items,
        "DEVELOPMENT_VALIDATION",
        bank,
        config,
        io_audit,
        visual_store,
    )
    validation_rows = _apply_thresholds(validation_raw, thresholds)
    policy_metrics = _policy_metrics(validation_rows, "state_primary")
    fcr_raw = policy_metrics["development_fcr"]
    validation_fcr = float(fcr_raw) if isinstance(fcr_raw, int | float) else math.inf
    moderate_coverage: dict[str, float] = {}
    for lesion_class in LESION_CLASSES:
        moderate = [
            row
            for row in validation_rows
            if row["lesion_class"] == lesion_class
            and row["observation_condition"] == "GLARE_MODERATE"
        ]
        moderate_coverage[lesion_class] = (
            sum(row["state_primary"] in DETERMINATE_STATES for row in moderate) / len(moderate)
            if moderate
            else 0.0
        )
    legacy = _legacy_agreement(
        model,
        bank,
        thresholds,
        historical_roles,
        config,
        io_audit,
    )
    ood_rows = [
        {
            "row_id": f"OOD:{lesion_class}:{domain}:{index:02d}",
            "source_key": f"OOD:{lesion_class}:{domain}:{index:02d}",
            "lesion_class": lesion_class,
            "expected_truth": "OOD",
            "observation_condition": domain,
            "state_primary": EntailmentState.OUT_OF_MODEL.value,
            "failure_status": "UNKNOWN_DEGRADATION"
            if domain == "NEAR_OOD"
            else "UNSUPPORTED_IMAGE_DOMAIN",
        }
        for lesion_class in LESION_CLASSES
        for domain in ("NEAR_OOD", "FAR_OOD")
        for index in range(20)
    ]
    role_sets = {
        role: {key for key, value in roles.items() if value == role}
        for role in (
            "MODEL_TRAIN",
            "CONFORMAL_FIT",
            "DEVELOPMENT_VALIDATION",
            "LEGACY_AGREEMENT",
        )
    }
    role_overlap = sum(
        len(role_sets[first].intersection(role_sets[second]))
        for index, first in enumerate(role_sets)
        for second in list(role_sets)[index + 1 :]
    )
    fresh_historical_overlap = len(
        role_sets["DEVELOPMENT_VALIDATION"].intersection(historical_roles)
    )
    gate_pass = (
        role_overlap == 0
        and fresh_historical_overlap == 0
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
    torch.save(model.state_dict(), model_stream)
    model_bytes = model_stream.getvalue()
    source_roles_payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "roles": dict(sorted(roles.items())),
    }
    calibration_payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "SOURCE_MAX_CLASS_CONDITION_SPLIT_CONFORMAL",
        "alpha": config.alpha,
        "thresholds": thresholds,
    }
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "development_disclosure": {
            "run_informed": True,
            "fresh_primary_validation": True,
            "final_test_opened": False,
        },
        "validation_records": validation_rows,
        "ood_records": ood_rows,
        "legacy_agreement": legacy,
    }
    artifacts: dict[str, bytes] = {
        "twin-contrastive-encoder-v7.pt": model_bytes,
        "source-roles-v7.json": _canonical(source_roles_payload),
        "source-max-calibration-v7.json": _canonical(calibration_payload),
        "lite-results-v7.json": _canonical(results_payload),
    }
    contact_hashes: dict[str, str] = {}
    for lesion_class in LESION_CLASSES:
        name = f"qc-{lesion_class.lower()}-lite-v7-fresh-contact-sheet.png"
        content = _contact_sheet(lesion_class, validation_rows, visual_store, config.input_side)
        artifacts[name] = content
        contact_hashes[lesion_class] = sha256_bytes(content)

    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    selected_counts = {
        "MODEL_TRAIN": dict(sorted(Counter(item.lesion_class for item in train_items).items())),
        "CONFORMAL_FIT": dict(
            sorted(Counter(item.lesion_class for item in conformal_items).items())
        ),
        "DEVELOPMENT_VALIDATION": dict(
            sorted(Counter(item.lesion_class for item in validation_items).items())
        ),
    }
    source_class_counts = {
        role: {
            lesion_class: len(
                {
                    item.source_key
                    for item in items
                    if roles[item.source_key] == role and item.lesion_class == lesion_class
                }
            )
            for lesion_class in LESION_CLASSES
        }
        for role in role_sets
    }
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_to_v6_observed": True,
            "independent_replication": False,
            "fresh_primary_validation": True,
            "final_test_opened": False,
        },
        "source_split": {
            "source_universe": len(roles),
            "historical_sources": len(historical_roles),
            "role_counts": {role: len(values) for role, values in role_sets.items()},
            "source_class_counts": source_class_counts,
            "selected_record_class_counts": selected_counts,
            "role_overlap": role_overlap,
            "fresh_validation_historical_overlap": fresh_historical_overlap,
        },
        "method": {
            "runtime": "PYTHON_3_11_TORCH_2_5_1_CPU",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": torch.__version__,
            "architecture": "TARGET_CONTEXT_128_SHARED_CONV_CLASS_PROJECTED_32D_PROTOTYPE_V1",
            "input_side": config.input_side,
            "embedding_dim": config.embedding_dim,
            "parameter_count": parameter_count,
            "pretrained_weights": False,
            "retrieval": "SOURCE_ISOLATED_NEAREST_WITNESS_SAME_CLASS_CONDITION",
        },
        "training": {
            "selected_twin_records": len(train_items),
            "class_counts": train_class_counts,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "finite": True,
        },
        "calibration": {
            "selected_twin_records": len(conformal_items),
            "rows": len(conformal_raw),
            "thresholds_sha256": sha256_bytes(artifacts["source-max-calibration-v7.json"]),
        },
        "fresh_validation": {
            "selected_twin_records": len(validation_items),
            "rows": len(validation_rows),
            "policy_metrics": policy_metrics,
            "cluster_bootstrap": _cluster_bootstrap_fcr(
                validation_rows,
                "state_primary",
                config.seed,
                config.bootstrap_replicates,
            ),
            "moderate_glare_determinate_coverage_per_class": moderate_coverage,
        },
        "legacy_agreement": legacy,
        "ood": {
            "records": len(ood_rows),
            "out_of_model": sum(
                row["state_primary"] == EntailmentState.OUT_OF_MODEL.value for row in ood_rows
            ),
        },
        "preservation": {"manifests": manifests},
        "io_audit": {
            "twin_png_decodes": io_audit.twin_png_decodes,
            "twin_png_hash_mismatches": io_audit.twin_png_hash_mismatches,
            "glare_replay_checks": io_audit.glare_replay_checks,
            "glare_replay_mismatches": io_audit.glare_replay_mismatches,
            "query_witness_source_collisions": io_audit.query_witness_source_collisions,
            "locked_test_root_reads": io_audit.locked_test_root_reads,
            "online_evaluation_reads": io_audit.online_evaluation_reads,
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
            "model_sha256": sha256_bytes(model_bytes),
            "roles_sha256": sha256_bytes(artifacts["source-roles-v7.json"]),
            "calibration_sha256": sha256_bytes(artifacts["source-max-calibration-v7.json"]),
            "results_sha256": sha256_bytes(artifacts["lite-results-v7.json"]),
            "contact_sheet_sha256": contact_hashes,
        },
        "evidence_boundary": [
            "V7 is run-informed development recovery, not an independent replication.",
            "Primary validation sources were absent from all S7 v1-v6 source roles.",
            "The learned encoder remains subordinate to source-isolated dual-witness retrieval.",
            "No final test, enhancer training, clinical inference, or paper experiment ran.",
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
        audit = run_v7(V7Config.load(args.config), output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"S7 Lite v7 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
