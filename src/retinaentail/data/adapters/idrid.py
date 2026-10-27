"""Explicit IDRiD adapter with controlled label semantics."""

from __future__ import annotations

from retinaentail.data.adapters.base import ExplicitDatasetAdapter
from retinaentail.data.stage2_models import DATASET_LABEL_SEMANTICS


class IDRiDAdapter(ExplicitDatasetAdapter):
    """Read only IDRiD entries and labels explicitly named by an authorized spec."""

    DATASET_ID = "idrid"
    LABEL_SEMANTICS = DATASET_LABEL_SEMANTICS["idrid"]


__all__ = ["IDRiDAdapter"]
