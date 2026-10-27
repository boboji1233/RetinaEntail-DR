"""Pure helpers for the final run-informed Science S7 v9 recovery.

The executable runner remains in ``scripts/`` because it binds a specific
PyTorch/torchvision runtime.  The functions in this module are deliberately
small and testable: they validate the frozen holdout membership and implement
the sole v9 optimisation change without performing data access.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor

from retinaentail.science_s6_verify import LESION_CLASSES


class S7LiteV9Error(ValueError):
    """Raised when the frozen v9 contract cannot be replayed safely."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise S7LiteV9Error(f"{name} must be a non-empty string")
    return value


def validate_frozen_holdout_records(
    inventory_records: Sequence[Mapping[str, object]],
    selected_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Replay exact component/source/class/split membership without decoding data."""

    inventory: dict[str, Mapping[str, object]] = {}
    for raw in inventory_records:
        component_id = _text(raw.get("component_id"), "inventory component ID")
        if component_id in inventory:
            raise S7LiteV9Error("component inventory contains duplicate IDs")
        inventory[component_id] = raw

    resolved: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in selected_records:
        component_id = _text(raw.get("component_id"), "selected component ID")
        if component_id in seen:
            raise S7LiteV9Error("v9 holdout contains duplicate component IDs")
        seen.add(component_id)
        source_key = _text(raw.get("source_key"), "selected source key")
        if not source_key.startswith("S4_DDR_V5:"):
            raise S7LiteV9Error("v9 holdout contains a non-DDR source key")
        selected_source = source_key.removeprefix("S4_DDR_V5:")
        selected_class = _text(raw.get("lesion_class"), "selected lesion class")
        selected_split = _text(raw.get("split"), "selected split")
        if selected_class not in LESION_CLASSES or selected_split not in {
            "TRAIN",
            "CALIBRATION",
        }:
            raise S7LiteV9Error("v9 holdout class/split is outside the frozen scope")
        record = inventory.get(component_id)
        if record is None:
            raise S7LiteV9Error("v9 holdout component is absent from the bound inventory")
        if (
            record.get("source_image_id") != selected_source
            or record.get("lesion_class") != selected_class
            or record.get("split") != selected_split
            or record.get("inclusion_status") != "INCLUDED"
        ):
            raise S7LiteV9Error("v9 holdout component provenance does not replay")
        resolved.append(dict(record))
    if not resolved:
        raise S7LiteV9Error("v9 holdout is empty")
    return resolved


def source_balanced_pair_weights(
    source_keys: Sequence[str], lesion_classes: Sequence[str]
) -> Tensor:
    """Return inverse source/class-frequency weights with unit mean."""

    if not source_keys or len(source_keys) != len(lesion_classes):
        raise S7LiteV9Error("source/class weight inputs differ or are empty")
    counts = Counter(zip(source_keys, lesion_classes, strict=True))
    values = torch.tensor(
        [
            1.0 / counts[(source, lesion)]
            for source, lesion in zip(source_keys, lesion_classes, strict=True)
        ],
        dtype=torch.float32,
    )
    return values / values.mean()


def source_balanced_cross_hypothesis_batch_hard(
    embeddings: Tensor,
    class_indices: Tensor,
    source_keys: Sequence[str],
    pair_weights: Tensor,
    *,
    margin: float,
) -> tuple[Tensor, int, int]:
    """Compute the frozen cross-source, opposite-hypothesis batch-hard term.

    ``embeddings`` has shape ``[batch, 4, dim]`` in the frozen order
    present/identity, present/glare, absent/identity, absent/glare.  For every
    anchor, candidates must have the same lesion class and condition, the
    opposite hypothesis, and a different source.  Anchors without a legal
    candidate contribute exactly zero and are counted for the audit.
    """

    if (
        embeddings.ndim != 3
        or embeddings.shape[1] != 4
        or class_indices.ndim != 1
        or pair_weights.ndim != 1
        or embeddings.shape[0] != class_indices.shape[0]
        or embeddings.shape[0] != pair_weights.shape[0]
        or embeddings.shape[0] != len(source_keys)
        or not margin > 0.0
    ):
        raise S7LiteV9Error("batch-hard inputs violate the frozen shape contract")
    batch = embeddings.shape[0]
    same_class = class_indices[:, None] == class_indices[None, :]
    different_source = torch.tensor(
        [
            [source_keys[row] != source_keys[column] for column in range(batch)]
            for row in range(batch)
        ],
        dtype=torch.bool,
        device=embeddings.device,
    )
    legal = same_class & different_source
    losses: list[Tensor] = []
    weights: list[Tensor] = []
    eligible = 0
    empty = 0
    for present_index, absent_index in ((0, 2), (1, 3)):
        present = functional.normalize(embeddings[:, present_index], dim=1)
        absent = functional.normalize(embeddings[:, absent_index], dim=1)
        distance = 1.0 - present @ absent.T
        for matrix in (distance, distance.T):
            masked = matrix.masked_fill(~legal, torch.inf)
            minimum = masked.min(dim=1).values
            valid = torch.isfinite(minimum)
            eligible += int(valid.sum().item())
            empty += int((~valid).sum().item())
            if bool(valid.any()):
                losses.append(functional.relu(margin - minimum[valid]))
                weights.append(pair_weights[valid])
    if not losses:
        return embeddings.sum() * 0.0, eligible, empty
    joined_losses = torch.cat(losses)
    joined_weights = torch.cat(weights).to(joined_losses)
    loss = (joined_losses * joined_weights).sum() / joined_weights.sum()
    if not bool(torch.isfinite(loss)):
        raise S7LiteV9Error("batch-hard loss became non-finite")
    return loss, eligible, empty
