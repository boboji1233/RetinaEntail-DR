"""Explicit DeepDRiD adapter with controlled quality and paired-view semantics."""

from __future__ import annotations

from retinaentail.data.adapters.base import ExplicitDatasetAdapter
from retinaentail.data.stage2_models import DATASET_LABEL_SEMANTICS


class DeepDRiDAdapter(ExplicitDatasetAdapter):
    """Read only DeepDRiD entries and labels explicitly named by an authorized spec."""

    DATASET_ID = "deepdrid"
    LABEL_SEMANTICS = DATASET_LABEL_SEMANTICS["deepdrid"]


__all__ = ["DeepDRiDAdapter"]
