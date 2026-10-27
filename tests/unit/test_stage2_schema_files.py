import json
from pathlib import Path

from retinaentail.data.audit import AuditStatus
from retinaentail.data.license_gate import DataUsePurpose, LicenseReviewDecision
from retinaentail.data.stage2_models import LabelSemantic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"


def _load(name: str) -> dict[str, object]:
    decoded = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def test_all_stage2_json_schemas_decode_and_are_closed() -> None:
    for name in (
        "adapter-spec-v2.schema.json",
        "audit-report-v2.schema.json",
        "license-review-v1.schema.json",
        "manifest-v2.schema.json",
    ):
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False


def test_schema_controlled_enums_match_python_contracts() -> None:
    manifest = _load("manifest-v2.schema.json")
    manifest_defs = manifest["$defs"]
    assert isinstance(manifest_defs, dict)
    semantic = manifest_defs["semantic"]
    assert isinstance(semantic, dict)
    assert set(semantic["enum"]) == {item.value for item in LabelSemantic}

    adapter = _load("adapter-spec-v2.schema.json")
    adapter_defs = adapter["$defs"]
    assert isinstance(adapter_defs, dict)
    label = adapter_defs["label"]
    assert isinstance(label, dict)
    label_properties = label["properties"]
    assert isinstance(label_properties, dict)
    adapter_semantic = label_properties["semantic"]
    assert isinstance(adapter_semantic, dict)
    assert adapter_semantic["$ref"] == "#/$defs/semantic"
    scoped_semantic = adapter_defs["semantic"]
    assert isinstance(scoped_semantic, dict)
    assert set(scoped_semantic["enum"]) == {item.value for item in LabelSemantic}

    audit = _load("audit-report-v2.schema.json")
    audit_properties = audit["properties"]
    assert isinstance(audit_properties, dict)
    status = audit_properties["status"]
    assert isinstance(status, dict)
    assert set(status["enum"]) == {item.value for item in AuditStatus}

    review = _load("license-review-v1.schema.json")
    review_properties = review["properties"]
    assert isinstance(review_properties, dict)
    purposes = review_properties["permitted_purposes"]
    decision = review_properties["decision"]
    assert isinstance(purposes, dict)
    assert isinstance(decision, dict)
    purpose_items = purposes["items"]
    assert isinstance(purpose_items, dict)
    assert set(purpose_items["enum"]) == {item.value for item in DataUsePurpose}
    assert set(decision["enum"]) == {item.value for item in LicenseReviewDecision}
