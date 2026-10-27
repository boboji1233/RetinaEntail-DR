"""Explicit DDR adapter; current ambiguous registry metadata cannot pass its gate."""

from __future__ import annotations

from retinaentail.data.adapters.base import ExplicitDatasetAdapter
from retinaentail.data.stage2_models import DATASET_LABEL_SEMANTICS


class DDRAdapter(ExplicitDatasetAdapter):
    """Read only DDR entries and labels explicitly named by an authorized spec."""

    DATASET_ID = "ddr"
    LABEL_SEMANTICS = DATASET_LABEL_SEMANTICS["ddr"]


__all__ = ["DDRAdapter"]
