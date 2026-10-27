"""Extend the frozen S8 construction candidates without decoding any data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import science_s8_freeze_locked_membership as v1

from retinaentail.science_s5_degradation_calibration import _mapping, _sequence, _string
from retinaentail.science_s6_verify import LESION_CLASSES
from retinaentail.witnesses.replay import canonical_json_bytes, write_bytes_fail_safe

MAX_CANDIDATES = 10


def freeze_v2(
    *, inventory_path: Path, v8_roles_path: Path, v9_selection_path: Path
) -> dict[str, object]:
    """Preserve v1 ranking and extend every source/class uniformly to top ten."""

    inventory = v1._load(inventory_path, "component inventory")  # noqa: SLF001
    roles = _mapping(v1._load(v8_roles_path, "v8 roles").get("roles"), "v8 roles")  # noqa: SLF001
    selection = v1._load(v9_selection_path, "v9 selection")  # noqa: SLF001
    v9_sources = {
        _string(_mapping(item, "v9 source").get("source_key"), "v9 source key")
        for item in _sequence(selection.get("selected_sources"), "v9 selected sources")
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for raw in _sequence(inventory.get("records"), "inventory records"):
        record = dict(_mapping(raw, "inventory record"))
        if record.get("inclusion_status") != "INCLUDED":
            continue
        source_id = _string(record.get("source_image_id"), "source ID")
        source_key = f"{v1.PACKAGE_ID}:{source_id}"
        lesion = _string(record.get("lesion_class"), "lesion class")
        if source_key not in roles and source_key not in v9_sources:
            grouped[(source_key, lesion)].append(record)
    candidates: list[dict[str, object]] = []
    sources: dict[str, set[str]] = defaultdict(set)
    for (source_key, lesion), records in sorted(grouped.items()):
        ordered = sorted(
            records,
            key=lambda row: v1._digest(  # noqa: SLF001
                lesion,
                source_key,
                _string(row.get("component_id"), "component ID"),
            ),
        )
        for rank, record in enumerate(ordered[:MAX_CANDIDATES], start=1):
            component_id = _string(record.get("component_id"), "component ID")
            candidates.append(
                {
                    "source_key": source_key,
                    "source_image_id": record["source_image_id"],
                    "split": record["split"],
                    "lesion_class": lesion,
                    "component_id": component_id,
                    "candidate_rank": rank,
                    "selection_digest": v1._digest(lesion, source_key, component_id),  # noqa: SLF001
                }
            )
        sources[source_key].add(lesion)
    counts = Counter(lesion for values in sources.values() for lesion in cast(set[str], values))
    return {
        "schema_version": "2.0.0",
        "experiment_id": "S8_CONTROLLED_TRUTH_LOCKED_MEMBERSHIP_V2",
        "status": "FROZEN_CONSTRUCTION_ONLY_AMENDMENT_NOT_EVALUATED",
        "seed": v1.SEED,
        "amendment": {
            "trigger": "V1_TOP3_CONSTRUCTION_PREFLIGHT_SE_28_BELOW_FROZEN_MINIMUM_30",
            "model_scores_observed": 0,
            "thresholds_fitted": 0,
            "uniform_change": "TOP3_TO_TOP10_FOR_EVERY_SOURCE_CLASS",
            "source_universe_changed": False,
            "ranking_changed": False,
        },
        "selection_contract": {
            "source_rule": "EXACT_V1_SOURCE_UNIVERSE",
            "component_rule": "V1_SHA256_ASCENDING_TOP10_CANDIDATES_PER_SOURCE_CLASS",
            "construction_rule": "FIRST_CONSTRUCTIBLE_CANDIDATE_PER_SOURCE_CLASS",
            "replacement_scope": "WITHIN_FROZEN_TOP10_ONLY",
            "final_locked_test_reads": 0,
        },
        "summary": {
            "locked_source_count": len(sources),
            "locked_source_counts_by_class": {lesion: counts[lesion] for lesion in LESION_CLASSES},
            "candidate_component_count": len(candidates),
            "maximum_accepted_target_count": len(grouped),
        },
        "sources": [
            {"source_key": key, "lesion_classes": sorted(values)}
            for key, values in sorted(sources.items())
        ],
        "candidate_records": candidates,
        "io_audit": {
            "image_decodes": 0,
            "mask_decodes": 0,
            "twin_writes": 0,
            "model_runs": 0,
            "locked_test_content_reads": 0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--v8-roles", type=Path, required=True)
    parser.add_argument("--v9-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise v1.S8MembershipError("S8 v2 membership output already exists")
    payload = freeze_v2(
        inventory_path=args.inventory,
        v8_roles_path=args.v8_roles,
        v9_selection_path=args.v9_selection,
    )
    write_bytes_fail_safe(args.output, canonical_json_bytes(payload))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
