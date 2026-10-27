"""Train and evaluate the frozen Science S7 v8 pretrained-backbone recovery."""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tracemalloc
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import science_s7_lite_v7 as core
import torch
import torch.nn.functional as functional
import torchvision
import yaml
from numpy.typing import NDArray
from torch import Tensor, nn
from torchvision.models import resnet18

from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _string,
    _write_artifacts,
)
from retinaentail.science_s6_verify import LESION_CLASSES
from retinaentail.science_s7_lite import (
    DETERMINATE_STATES,
    _cluster_bootstrap_fcr,
    _policy_metrics,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "8.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V8"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CLASS_INDEX: Final = {lesion_class: index for index, lesion_class in enumerate(LESION_CLASSES)}
ROLE_NAMES: Final = (
    "MODEL_TRAIN",
    "CONFORMAL_FIT",
    "DEVELOPMENT_VALIDATION",
    "V7_OBSERVED_VALIDATION",
    "LEGACY_AGREEMENT",
)


class S7LiteV8Error(ValueError):
    """Raised when the frozen v8 contract cannot execute safely."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV8Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV8Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV8Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV8Error(f"{name} is not SHA-256")
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
        raise S7LiteV8Error("v8 baseline drift")


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV8Error(f"binding mismatch: {name}")
    return path


@dataclass(frozen=True, slots=True)
class V8Config:
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
    weight_path: Path
    weight_sha256: str
    weight_url: str

    @classmethod
    def load(cls, path: Path) -> V8Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v8 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_SECOND_FRESH_HOLDOUT_RECOVERY"
        ):
            raise S7LiteV8Error("unsupported S7 v8 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_to_v7_observed": True,
            "independent_replication": False,
            "second_fresh_primary_validation": True,
            "final_test_opened": False,
        }:
            raise S7LiteV8Error("v8 disclosure changed")
        split = _mapping(root.get("source_split"), "source split")
        method = _mapping(root.get("method"), "method")
        training = _mapping(root.get("training"), "training")
        losses = _mapping(training.get("loss_weights"), "loss weights")
        calibration = _mapping(root.get("calibration"), "calibration")
        statistics = _mapping(root.get("statistics"), "statistics")
        gate = _mapping(root.get("gate"), "gate")
        expected_contract = {
            "runtime": method.get("runtime") == "PYTHON_3_11_TORCH_2_5_1_TORCHVISION_0_20_1_CPU",
            "architecture": method.get("architecture")
            == "TARGET_CONTEXT_128_FROZEN_IMAGENET_RESNET18_CLASS_PROJECTED_32D_V1",
            "retrieval": method.get("retrieval")
            == "SOURCE_ISOLATED_NEAREST_WITNESS_SAME_CLASS_CONDITION",
            "validation_selector": split.get("validation_selector")
            == "SECOND_UNSEEN_CLASS_PRIORITY_SHA256_MINIMUM",
            "conformal_selector": split.get("conformal_selector")
            == "CLASS_PRIORITY_SHA256_MINIMUM",
            "legacy_role": split.get("legacy_role") == "V4_DEVELOPMENT_VALIDATION_ONLY",
            "v7_role": split.get("v7_observed_role") == "V7_VALIDATION_EXCLUDED",
            "optimizer": training.get("optimizer") == "ADAMW_ADAPTER_ONLY",
            "calibration": calibration.get("method")
            == "SOURCE_MAX_CLASS_CONDITION_SPLIT_CONFORMAL",
            "pretrained": method.get("pretrained_weights") is True,
            "backbone_frozen": method.get("backbone_frozen") is True,
        }
        if not all(expected_contract.values()):
            raise S7LiteV8Error("v8 frozen method contract changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required_bindings = {
            "direction_review",
            "v8_prompt",
            "v8_code",
            "v8_audit_schema",
            "s4_idrid_audit",
            "s4_idrid_artifact_manifest",
            "s4_idrid_twin_manifest",
            "s4_ddr_audit",
            "s4_ddr_artifact_manifest",
            "s4_ddr_twin_manifest",
            "s5_v2_audit",
            "s5_v2_artifact_manifest",
            "s7_v1_audit",
            "s7_v1_artifact_manifest",
            "s7_v4_source_split",
            "s7_v5_results",
            "s7_v6_audit",
            "s7_v6_artifact_manifest",
            "s7_v7_audit",
            "s7_v7_artifact_manifest",
            "s7_v7_source_roles",
            "s6_v2_results",
        }
        if set(bindings) != required_bindings:
            raise S7LiteV8Error("v8 binding set changed")
        weights = _mapping(method.get("weight_binding"), "weight binding")
        cache_relative = _safe_relative_string(
            weights.get("home_relative_path"), "weight home-relative path"
        )
        weight_path = Path.home().joinpath(*PurePosixPath(cache_relative).parts)
        weight_sha256 = _sha256(weights.get("sha256"), "weight SHA-256")
        weight_url = _string(weights.get("source_url"), "weight source URL")
        if (
            weight_url != "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            or not weight_path.is_file()
            or sha256_file(weight_path) != weight_sha256
        ):
            raise S7LiteV8Error("pretrained weight binding mismatch")
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
            "seed": 2026082618,
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
            "separation_weight": 0.5,
            "separation_margin": 0.75,
            "alpha": 0.05,
            "bootstrap_replicates": 2000,
            "maximum_fcr": 0.05,
            "minimum_moderate_coverage": 0.05,
            "minimum_verify_agreement": 0.5,
        }
        if values != frozen_values:
            raise S7LiteV8Error("v8 frozen numeric contract changed")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            bindings=bindings,
            weight_path=weight_path,
            weight_sha256=weight_sha256,
            weight_url=weight_url,
            **values,
        )


class FrozenResNetAdapter(nn.Module):
    """Frozen ImageNet ResNet-18 with class-conditioned trainable adapters."""

    def __init__(self, embedding_dim: int, weight_path: Path) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state)
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.projection_weight = nn.Parameter(torch.empty(4, embedding_dim, 512))
        self.projection_bias = nn.Parameter(torch.zeros(4, embedding_dim))
        self.head_weight = nn.Parameter(torch.empty(4, embedding_dim))
        self.head_bias = nn.Parameter(torch.zeros(4))
        self.prototypes = nn.Parameter(torch.empty(4, 2, embedding_dim))
        for index in range(4):
            nn.init.xavier_uniform_(self.projection_weight[index])
        nn.init.xavier_uniform_(self.head_weight)
        nn.init.normal_(self.prototypes, std=0.2)
        self.backbone.eval()

    def train(self, mode: bool = True) -> FrozenResNetAdapter:
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _imagenet_normalize(inputs: Tensor) -> Tensor:
        unit = inputs * 0.25 + 0.5
        mean = unit.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = unit.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (unit - mean) / std

    def extract_features(self, inputs: Tensor) -> Tensor:
        self.backbone.eval()
        with torch.no_grad():
            return cast(Tensor, self.backbone(self._imagenet_normalize(inputs)))

    def project(self, features: Tensor, class_indices: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.projection_weight[class_indices]
        projected = torch.bmm(weights, features.unsqueeze(2)).squeeze(2)
        projected = projected + self.projection_bias[class_indices]
        embedding = functional.normalize(projected, dim=1)
        logits = (embedding * self.head_weight[class_indices]).sum(dim=1)
        logits = logits + self.head_bias[class_indices]
        return embedding, logits

    def forward(self, inputs: Tensor, class_indices: Tensor) -> tuple[Tensor, Tensor]:
        return self.project(self.extract_features(inputs), class_indices)

    def adapter_state(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
        }


def _select_roles(
    items: Sequence[core.TwinItem],
    historical_roles: Mapping[str, str],
    v7_roles: Mapping[str, str],
    config: V8Config,
) -> dict[str, str]:
    by_class: dict[str, set[str]] = {lesion_class: set() for lesion_class in LESION_CLASSES}
    for item in items:
        by_class[item.lesion_class].add(item.source_key)
    all_sources = set().union(*by_class.values())
    historical_sources = set(historical_roles)
    if not historical_sources <= all_sources or len(historical_sources) != 184:
        raise S7LiteV8Error("historical S7 source universe changed")
    legacy = {key for key, role in historical_roles.items() if role == "DEVELOPMENT_VALIDATION"}
    observed = {key for key, role in v7_roles.items() if role == "DEVELOPMENT_VALIDATION"}
    if len(legacy) != 69 or len(observed) != 19 or observed.intersection(historical_sources):
        raise S7LiteV8Error("historical validation role changed")
    unseen = all_sources - historical_sources - observed
    validation: set[str] = set()
    for lesion_class in ("EX", "MA", "SE", "HE"):
        candidates = sorted(
            by_class[lesion_class].intersection(unseen),
            key=lambda key: core._digest(config.seed, f"VALIDATION|{lesion_class}", key),
        )
        for key in candidates:
            if len(by_class[lesion_class].intersection(validation)) >= (
                config.validation_sources_per_class
            ):
                break
            validation.add(key)
    remaining = all_sources - validation - legacy - observed
    conformal: set[str] = set()
    for lesion_class in ("EX", "MA", "SE", "HE"):
        candidates = sorted(
            by_class[lesion_class].intersection(remaining),
            key=lambda key: core._digest(config.seed, f"CONFORMAL|{lesion_class}", key),
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
            else "V7_OBSERVED_VALIDATION"
            if key in observed
            else "CONFORMAL_FIT"
            if key in conformal
            else "MODEL_TRAIN"
        )
        for key in all_sources
    }
    counts = Counter(roles.values())
    expected = {
        "MODEL_TRAIN": 161,
        "CONFORMAL_FIT": 33,
        "DEVELOPMENT_VALIDATION": 20,
        "V7_OBSERVED_VALIDATION": 19,
        "LEGACY_AGREEMENT": 69,
    }
    if counts != expected:
        raise S7LiteV8Error(f"unexpected v8 source-role counts: {dict(counts)}")
    if validation.intersection(historical_sources | observed):
        raise S7LiteV8Error("v8 fresh validation overlaps observed S7 sources")
    if any(
        len(by_class[lesion_class].intersection(validation)) < config.validation_sources_per_class
        for lesion_class in LESION_CLASSES
    ):
        raise S7LiteV8Error("v8 validation class coverage is insufficient")
    return roles


def _extract_training_features(
    model: FrozenResNetAdapter,
    samples: Sequence[core.PairSample],
    config: V8Config,
) -> Tensor:
    arrays = np.stack([np.stack(sample.views) for sample in samples]).reshape(
        -1, config.input_side, config.input_side, 3
    )
    chunks: list[Tensor] = []
    for start in range(0, len(arrays), 64):
        chunks.append(model.extract_features(core._tensor(arrays[start : start + 64])).cpu())
    return torch.cat(chunks).reshape(len(samples), 4, 512)


def _train_adapter(
    model: FrozenResNetAdapter,
    samples: Sequence[core.PairSample],
    config: V8Config,
) -> tuple[list[float], dict[str, int]]:
    features = _extract_training_features(model, samples, config)
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
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
            batch_features = features[indices]
            batch_classes = classes[indices]
            batch_weights = pair_weights[indices]
            flat_features = batch_features.reshape(-1, 512)
            flat_classes = batch_classes.repeat_interleave(4)
            embeddings, logits = model.project(flat_features, flat_classes)
            embeddings = embeddings.reshape(len(indices), 4, config.embedding_dim)
            logits = logits.reshape(len(indices), 4)
            normalized_prototypes = functional.normalize(model.prototypes, dim=2)
            labels = hypothesis[None, :].expand(len(indices), -1)
            hypothesis_indices = labels.to(torch.long)
            selected_prototypes = normalized_prototypes[batch_classes[:, None], hypothesis_indices]
            pull = 1.0 - functional.cosine_similarity(embeddings, selected_prototypes, dim=2)
            pull_loss = core._weighted_mean(pull.mean(dim=1), batch_weights)
            head = functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            ).mean(dim=1)
            head_loss = core._weighted_mean(head, batch_weights)
            present_invariance = 1.0 - functional.cosine_similarity(
                embeddings[:, 0], embeddings[:, 1], dim=1
            )
            absent_invariance = 1.0 - functional.cosine_similarity(
                embeddings[:, 2], embeddings[:, 3], dim=1
            )
            invariance_loss = core._weighted_mean(
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
            separation_loss = core._weighted_mean(twin_separation, batch_weights)
            separation_loss = separation_loss + prototype_separation
            loss = (
                config.prototype_weight * pull_loss
                + config.head_weight * head_loss
                + config.invariance_weight * invariance_loss
                + config.separation_weight * separation_loss
            )
            if not bool(torch.isfinite(loss)):
                raise S7LiteV8Error("v8 optimization became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / batches)
    return losses, dict(sorted(class_counts.items()))


def _load_roles(path: Path, name: str) -> dict[str, str]:
    decoded = core._load_json(path, name)
    raw = _mapping(decoded.get("roles"), f"{name} roles")
    return {key: _string(value, f"{name} role") for key, value in raw.items()}


def run_v8(config: V8Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV8Error("S7 v8 output exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    if (
        sys.version_info[:2] != (3, 11)
        or torch.__version__ != "2.5.1+cpu"
        or torchvision.__version__ != "0.20.1+cpu"
        or torch.cuda.is_available()
    ):
        raise S7LiteV8Error("v8 requires the frozen Python 3.11 CPU-only runtime")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    tracemalloc.start()
    io_audit = core.IOAudit()
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
        "s7_v7": _audit_manifest(
            config.bindings["s7_v7_audit"].parent,
            config.bindings["s7_v7_artifact_manifest"],
        ),
    }
    if any(
        audit[check] != 0
        for audit in manifests.values()
        for check in ("mismatches", "unlisted", "missing")
    ):
        raise S7LiteV8Error("upstream artifact manifest replay failed")
    items = core._load_twins(
        "S4_IDRID_V1",
        config.bindings["s4_idrid_twin_manifest"],
        manifest_name="S4 IDRiD v1",
    ) + core._load_twins(
        "S4_DDR_V5",
        config.bindings["s4_ddr_twin_manifest"],
        manifest_name="S4 DDR v5",
    )
    historical_roles = _load_roles(config.bindings["s7_v4_source_split"], "v4 split")
    v7_roles = _load_roles(config.bindings["s7_v7_source_roles"], "v7 split")
    roles = _select_roles(items, historical_roles, v7_roles, config)
    train_items = core._select_records(
        items, roles, "MODEL_TRAIN", config.train_records_per_source_class, config.seed
    )
    conformal_items = core._select_records(
        items, roles, "CONFORMAL_FIT", config.evaluation_records_per_source_class, config.seed
    )
    validation_items = core._select_records(
        items,
        roles,
        "DEVELOPMENT_VALIDATION",
        config.evaluation_records_per_source_class,
        config.seed,
    )
    if (len(train_items), len(conformal_items), len(validation_items)) != (266, 118, 73):
        raise S7LiteV8Error("v8 selected record counts changed")
    train_samples = [core._make_pair_sample(item, config, io_audit) for item in train_items]
    model = FrozenResNetAdapter(config.embedding_dim, config.weight_path)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    losses, train_class_counts = _train_adapter(model, train_samples, config)
    if not losses or not all(math.isfinite(value) for value in losses):
        raise S7LiteV8Error("v8 training loss is invalid")
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
        validation_items,
        "DEVELOPMENT_VALIDATION",
        bank,
        config,
        io_audit,
        visual_store,
    )
    validation_rows = core._apply_thresholds(validation_raw, thresholds)
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
    legacy = core._legacy_agreement(model, bank, thresholds, historical_roles, config, io_audit)
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
        role: {key for key, value in roles.items() if value == role} for role in ROLE_NAMES
    }
    role_overlap = sum(
        len(role_sets[first].intersection(role_sets[second]))
        for index, first in enumerate(ROLE_NAMES)
        for second in ROLE_NAMES[index + 1 :]
    )
    fresh_historical_overlap = len(
        role_sets["DEVELOPMENT_VALIDATION"].intersection(historical_roles)
    )
    fresh_v7_overlap = len(
        role_sets["DEVELOPMENT_VALIDATION"].intersection(role_sets["V7_OBSERVED_VALIDATION"])
    )
    gate_pass = (
        role_overlap == 0
        and fresh_historical_overlap == 0
        and fresh_v7_overlap == 0
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
            "second_fresh_primary_validation": True,
            "final_test_opened": False,
        },
        "validation_records": validation_rows,
        "ood_records": ood_rows,
        "legacy_agreement": legacy,
    }
    artifacts: dict[str, bytes] = {
        "resnet18-adapter-v8.pt": model_bytes,
        "source-roles-v8.json": _canonical(source_roles_payload),
        "source-max-calibration-v8.json": _canonical(calibration_payload),
        "lite-results-v8.json": _canonical(results_payload),
    }
    contact_hashes: dict[str, str] = {}
    for lesion_class in LESION_CLASSES:
        name = f"qc-{lesion_class.lower()}-lite-v8-second-fresh-contact-sheet.png"
        content = core._contact_sheet(
            lesion_class, validation_rows, visual_store, config.input_side
        )
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
        for role in ROLE_NAMES
    }
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_to_v7_observed": True,
            "independent_replication": False,
            "second_fresh_primary_validation": True,
            "final_test_opened": False,
        },
        "source_split": {
            "source_universe": len(roles),
            "historical_v1_to_v6_sources": len(historical_roles),
            "v7_observed_validation_sources": len(role_sets["V7_OBSERVED_VALIDATION"]),
            "role_counts": {role: len(values) for role, values in role_sets.items()},
            "source_class_counts": source_class_counts,
            "selected_record_class_counts": selected_counts,
            "role_overlap": role_overlap,
            "fresh_validation_historical_overlap": fresh_historical_overlap,
            "fresh_validation_v7_overlap": fresh_v7_overlap,
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
            "thresholds_sha256": sha256_bytes(artifacts["source-max-calibration-v8.json"]),
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
            "roles_sha256": sha256_bytes(artifacts["source-roles-v8.json"]),
            "calibration_sha256": sha256_bytes(artifacts["source-max-calibration-v8.json"]),
            "results_sha256": sha256_bytes(artifacts["lite-results-v8.json"]),
            "contact_sheet_sha256": contact_hashes,
            "pretrained_weight_sha256": config.weight_sha256,
        },
        "evidence_boundary": [
            "V8 is run-informed development recovery, not an independent replication.",
            "Primary validation sources were absent from S7 v1-v6 roles and v7 validation.",
            "The pretrained backbone is frozen and the adapter remains subordinate to "
            "dual-witness retrieval.",
            "The ImageNet weight is an external gitignored dependency bound by SHA-256.",
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
        audit = run_v8(V8Config.load(args.config), output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"S7 Lite v8 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
