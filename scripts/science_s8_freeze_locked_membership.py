"""Freeze the prospective Science S8 source/component membership without decoding data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from retinaentail.data.manifest import sha256_file
from retinaentail.science_s5_degradation_calibration import _mapping, _sequence, _string
from retinaentail.science_s6_verify import LESION_CLASSES
from retinaentail.witnesses.replay import canonical_json_bytes, write_bytes_fail_safe

SEED: Final = 2026082620
PACKAGE_ID: Final = "S4_DDR_V5"


class S8MembershipError(ValueError):
    """Raised when prospective locked membership cannot be frozen safely."""


def _load(path: Path, name: str) -> Mapping[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(decoded, name)


def _digest(lesion: str, source_key: str, component_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}|S8_RECORD|{lesion}|{source_key}|{component_id}".encode()
    ).hexdigest()


def freeze_membership(
    *, inventory_path: Path, v8_roles_path: Path, v9_selection_path: Path
) -> dict[str, object]:
    """Return a deterministic all-remaining-source S8 membership payload."""

    inventory = _load(inventory_path, "component inventory")
    roles = _mapping(_load(v8_roles_path, "v8 roles").get("roles"), "v8 role mapping")
    selection = _load(v9_selection_path, "v9 selection")
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
        source_key = f"{PACKAGE_ID}:{source_id}"
        lesion = _string(record.get("lesion_class"), "lesion class")
        if lesion not in LESION_CLASSES:
            raise S8MembershipError("inventory contains an unknown lesion class")
        if source_key in roles or source_key in v9_sources:
            continue
        grouped[(source_key, lesion)].append(record)
    if not grouped:
        raise S8MembershipError("no prospective S8 sources remain")

    candidates: list[dict[str, object]] = []
    sources: dict[str, set[str]] = defaultdict(set)
    for (source_key, lesion), records in sorted(grouped.items()):
        ordered = sorted(
            records,
            key=lambda record: _digest(
                lesion,
                source_key,
                _string(record.get("component_id"), "component ID"),
            ),
        )
        for rank, record in enumerate(ordered[:3], start=1):
            candidates.append(
                {
                    "source_key": source_key,
                    "source_image_id": _string(record.get("source_image_id"), "source ID"),
                    "split": _string(record.get("split"), "split"),
                    "lesion_class": lesion,
                    "component_id": _string(record.get("component_id"), "component ID"),
                    "candidate_rank": rank,
                    "selection_digest": _digest(
                        lesion,
                        source_key,
                        _string(record.get("component_id"), "component ID"),
                    ),
                }
            )
        sources[source_key].add(lesion)
    class_source_counts = Counter(
        lesion for lesions in sources.values() for lesion in cast(set[str], lesions)
    )
    if any(class_source_counts[lesion] < 30 for lesion in LESION_CLASSES):
        raise S8MembershipError("prospective S8 source support is insufficient")
    return {
        "schema_version": "1.0.0",
        "experiment_id": "S8_CONTROLLED_TRUTH_LOCKED_MEMBERSHIP_V1",
        "status": "FROZEN_NOT_DECODED_NOT_EVALUATED",
        "seed": SEED,
        "selection_contract": {
            "source_rule": "ALL_INCLUDED_DDR_SOURCES_ABSENT_FROM_V8_ROLES_AND_V9_HOLDOUT",
            "component_rule": "SHA256_ASCENDING_TOP3_CANDIDATES_PER_SOURCE_CLASS",
            "construction_rule": "FIRST_CONSTRUCTIBLE_CANDIDATE_PER_SOURCE_CLASS",
            "replacement_scope": "WITHIN_FROZEN_TOP3_ONLY",
            "source_cluster_unit": "SOURCE_IMAGE",
            "final_locked_test_reads": 0,
        },
        "bindings": {
            "component_inventory_sha256": sha256_file(inventory_path),
            "v8_source_roles_sha256": sha256_file(v8_roles_path),
            "v9_holdout_selection_sha256": sha256_file(v9_selection_path),
        },
        "summary": {
            "locked_source_count": len(sources),
            "locked_source_counts_by_class": {
                lesion: class_source_counts[lesion] for lesion in LESION_CLASSES
            },
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
        "evidence_boundary": [
            "This file freezes membership metadata only; no image or mask was decoded.",
            "No controlled twin, model output, FCR, clinical, medical, or paper result exists.",
            "All remaining eligible sources are included, preventing post-v9 "
            "source cherry-picking.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--v8-roles", type=Path, required=True)
    parser.add_argument("--v9-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise S8MembershipError("S8 membership output already exists")
    payload = freeze_membership(
        inventory_path=args.inventory,
        v8_roles_path=args.v8_roles,
        v9_selection_path=args.v9_selection,
    )
    write_bytes_fail_safe(args.output, canonical_json_bytes(payload))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
