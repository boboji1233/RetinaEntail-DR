"""Offline diagnostics and fail-closed RetinaEntail-DR stage tools."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from retinaentail.calibration.stage3_gate import (
    Stage3PreflightStatus,
    assess_stage3_preflight,
)
from retinaentail.calibration.toy_fixture import (
    run_stage3_toy_fixture,
    verify_stage3_toy_package,
)
from retinaentail.data.adapters import DDRAdapter, DeepDRiDAdapter, IDRiDAdapter
from retinaentail.data.adapters.base import AdapterContractError, ExplicitDatasetAdapter
from retinaentail.data.audit import (
    AuditIssue,
    AuditResult,
    AuditStatus,
    AuditSummary,
    IssueSeverity,
    audit_manifest,
    blocked_audit_result,
)
from retinaentail.data.audit_reporting import (
    AuditWriteBoundary,
    write_audit_reports,
    write_manifest_csv,
)
from retinaentail.data.license_gate import (
    DataUsePurpose,
    LicenseGateError,
    LicenseMetadataError,
    assess_dataset_access,
    authorize_dataset_access,
    load_license_review_record,
    load_registry_license_metadata,
)
from retinaentail.data.registry import REQUIRED_DATASET_IDS, validate_registry_directory
from retinaentail.data.stage2_fixture import run_stage2_fixture
from retinaentail.data.stage2_models import (
    DatasetAdapterSpec,
    Stage2ContractError,
    load_dataset_adapter_spec,
)
from retinaentail.evaluation.gate import (
    Stage4PreflightStatus,
    Stage4ReviewPurpose,
    assess_stage4_preflight,
)
from retinaentail.evaluation.toy_fixture import (
    run_stage4_toy_fixture,
    verify_stage4_toy_package,
)
from retinaentail.pilot.gate import (
    Stage5PreflightStatus,
    Stage5VerificationContext,
    assess_stage5_preflight,
)
from retinaentail.pilot.stage6_fixture import (
    run_stage6_toy_fixture,
    verify_stage6_toy_package,
)
from retinaentail.pilot.stage6_gate import (
    REQUIRED_STAGE6_REVIEW_PURPOSES,
    Stage6PreflightStatus,
    Stage6VerificationContext,
    assess_stage6_preflight,
)
from retinaentail.pilot.stage7_fixture import (
    run_stage7_toy_fixture,
    verify_stage7_toy_package,
)
from retinaentail.pilot.stage7_gate import (
    Stage7PreflightStatus,
    assess_stage7_preflight,
)
from retinaentail.pilot.stage8_fixture import (
    run_stage8_toy_fixture,
    verify_stage8_toy_package,
)
from retinaentail.pilot.stage8_gate import (
    Stage8PreflightStatus,
    assess_stage8_preflight,
)
from retinaentail.pilot.stage9_fixture import (
    run_stage9_toy_fixture,
    verify_stage9_toy_package,
)
from retinaentail.pilot.stage9_gate import (
    Stage9PreflightStatus,
    assess_stage9_preflight,
)
from retinaentail.pilot.stage10_fixture import (
    run_stage10_toy_fixture,
    verify_stage10_toy_package,
)
from retinaentail.pilot.stage10_gate import (
    Stage10PreflightStatus,
    assess_stage10_preflight,
)
from retinaentail.pilot.stage11_fixture import (
    run_stage11_toy_fixture,
    verify_stage11_toy_package,
)
from retinaentail.pilot.stage11_gate import (
    Stage11PreflightStatus,
    assess_stage11_preflight,
)
from retinaentail.pilot.stage12_fixture import (
    run_stage12_fixture,
    verify_stage12_package,
)
from retinaentail.pilot.stage12_gate import (
    Stage12PreflightStatus,
    assess_stage12_preflight,
)
from retinaentail.pilot.toy_fixture import (
    run_stage5_toy_fixture,
    verify_stage5_toy_package,
)
from retinaentail.reporting.stage3 import write_stage3_preflight_reports
from retinaentail.reporting.stage4 import write_stage4_preflight_reports
from retinaentail.reporting.stage5 import write_stage5_preflight_reports
from retinaentail.reporting.stage6 import write_stage6_preflight_reports
from retinaentail.reporting.stage7 import write_stage7_preflight_reports
from retinaentail.reporting.stage8 import write_stage8_preflight_reports
from retinaentail.reporting.stage9 import write_stage9_preflight_reports
from retinaentail.reporting.stage10 import write_stage10_preflight_reports
from retinaentail.reporting.stage11 import write_stage11_preflight_reports
from retinaentail.reporting.stage12 import write_stage12_preflight_reports
from retinaentail.smoke import run_smoke
from retinaentail.witnesses.replay import canonical_json_bytes, write_bytes_fail_safe

_STAGE2_DATASET_ADAPTERS: dict[str, type[ExplicitDatasetAdapter]] = {
    "idrid": IDRiDAdapter,
    "ddr": DDRAdapter,
    "deepdrid": DeepDRiDAdapter,
}


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("could not locate the RetinaEntail-DR project root")


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def doctor() -> int:
    root = _project_root()
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    registry_files = tuple(sorted((root / "data_registry" / "datasets").glob("*.yaml")))
    report = {
        "project": "RetinaEntail-DR",
        "stage": (
            "0/1 + 2-read-only-audit + 3-calibration + 4-locked-evaluation"
            " + 5-prerequisite-pilot-scaffold + 6-complete-package-verifiers"
            " + 7-authorization-consumption-rebuild"
            " + 8-repeatability-witness-stability"
            " + 9-independent-session-package-diff"
            " + 10-runner-portability-signed-exchange"
            " + 11-asymmetric-rotation-cross-host-gate"
            " + 12-signer-interoperability-compromise-recovery"
        ),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_supported": supported_python,
        "platform": platform.platform(),
        "cpu_only_smoke": True,
        "gpu_required": False,
        "gpu_vram_limit_gb": 6,
        "network_required": False,
        "real_data_required": False,
        "real_data_access_requires_license_review": True,
        "stage3_real_data_status": "BLOCKED_UNTIL_EXPLICIT_APPROVED_INPUTS",
        "stage3_toy_calibration_available": True,
        "stage4_real_data_status": "BLOCKED_UNTIL_ALL_LOCKED_EVIDENCE_IS_VERIFIED",
        "stage4_toy_locked_evaluation_available": True,
        "stage5_real_data_status": "BLOCKED_REAL_VERIFIER_CAPABILITIES_UNAVAILABLE",
        "stage5_toy_locked_pilot_available": True,
        "stage6_real_data_status": "BLOCKED_NO_EXPLICIT_APPROVED_INPUTS",
        "stage6_complete_package_capabilities_available": True,
        "stage6_toy_verifier_pilot_available": True,
        "stage7_real_data_status": "BLOCKED_NO_EXPLICIT_APPROVED_INPUTS",
        "stage7_zero_io_preflight_available": True,
        "stage7_root_transaction_capability_available": True,
        "stage7_toy_consumption_pilot_available": True,
        "stage8_real_data_status": "BLOCKED_NO_REAL_STAGE7_PACKAGE_OR_FRESH_CAPABILITIES",
        "stage8_zero_io_preflight_available": True,
        "stage8_process_capability_boundary_available": True,
        "stage8_repeatability_audit_available": True,
        "stage8_toy_repeatability_package_available": True,
        "stage9_real_data_status": "BLOCKED_NO_TWO_REAL_STAGE8_PACKAGES_OR_FRESH_CAPABILITIES",
        "stage9_zero_io_preflight_available": True,
        "stage9_independent_process_fixture_available": True,
        "stage9_complete_package_diff_available": True,
        "stage10_real_data_status": "BLOCKED_NO_TWO_REAL_STAGE9_PACKAGES_OR_APPROVED_RUNNERS",
        "stage10_zero_io_preflight_available": True,
        "stage10_local_dual_runner_fixture_available": True,
        "stage10_offline_signed_exchange_verifier_available": True,
        "stage10_cross_physical_host_claim": False,
        "stage11_real_data_status": (
            "BLOCKED_NO_TWO_REAL_STAGE10_PACKAGES_APPROVED_HOSTS_OR_PRODUCTION_PKI"
        ),
        "stage11_zero_io_preflight_available": True,
        "stage11_local_ed25519_rotation_fixture_available": True,
        "stage11_offline_artifact_verifier_available": True,
        "stage11_cross_physical_host_claim": False,
        "stage11_production_authentication_claim": False,
        "stage12_real_data_status": (
            "BLOCKED_NO_TWO_REAL_STAGE11_PACKAGES_HARDWARE_RUNNERS_OR_PRODUCTION_TRUST"
        ),
        "stage12_zero_io_preflight_available": True,
        "stage12_two_process_random_signer_fixture_available": True,
        "stage12_offline_recovery_verifier_available": True,
        "stage12_cross_physical_host_claim": False,
        "stage12_production_authentication_claim": False,
        "registry_templates": len(registry_files),
        "dependencies": {
            "numpy": _version("numpy"),
            "Pillow": _version("Pillow"),
            "PyYAML": _version("PyYAML"),
            "cryptography": _version("cryptography"),
        },
        "status": (
            "ok"
            if supported_python and len(registry_files) == len(REQUIRED_DATASET_IDS)
            else "failed"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


def registry_validate() -> int:
    directory = _project_root() / "data_registry" / "datasets"
    results = validate_registry_directory(directory)
    print(f"registry validation: {len(results)}/{len(results)} valid")
    for result in results:
        print(f"  OK {result.dataset_id}: {result.path.name}")
    return 0


def smoke(config_path: Path | None, output_dir: Path | None) -> int:
    root = _project_root()
    resolved_config = config_path or root / "configs" / "smoke.yaml"
    resolved_output = output_dir or root / "outputs" / "smoke"
    result = run_smoke(resolved_config, resolved_output)
    print(f"smoke state: {result.state.value}")
    print(f"certificate: {result.certificate_path}")
    for replay in result.replay_results:
        print(
            f"witness {replay.witness_id}: passed={replay.passed} "
            f"replay_error={replay.replay_error:.12g}"
        )
    return 0


def _safe_generated_output(base: Path, requested: Path | None, default_name: str) -> Path:
    trusted_root = _project_root().resolve(strict=True)
    base_absolute = base.absolute()
    candidate = (base / default_name if requested is None else requested).absolute()
    try:
        candidate.relative_to(base_absolute)
    except ValueError as error:
        raise ValueError(f"generated output must stay under {base_absolute}") from error
    resolved_base = base_absolute.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_base.relative_to(trusted_root)
        resolved_candidate.relative_to(resolved_base)
    except ValueError as error:
        raise ValueError("generated output cannot escape the trusted project tree") from error
    return candidate


def _data_report_output(dataset_id: str, output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage2"
    return _safe_generated_output(base, output_dir, dataset_id)


def _paths_overlap_lexically(first: Path, second: Path) -> bool:
    first_absolute = first.absolute()
    second_absolute = second.absolute()
    for candidate, parent in ((first_absolute, second_absolute), (second_absolute, first_absolute)):
        try:
            candidate.relative_to(parent)
        except ValueError:
            continue
        return True
    return False


def _write_stage2_report(
    result: AuditResult,
    output_dir: Path,
    *,
    title: str,
    forbidden_roots: tuple[Path, ...] = (),
) -> None:
    boundary = AuditWriteBoundary(
        allowed_root=_project_root() / "reports" / "generated" / "stage2",
        forbidden_roots=forbidden_roots,
    )
    write_audit_reports(result, output_dir, boundary=boundary, title=title)


def _adapter_failure_result(dataset_id: str, error: BaseException) -> AuditResult:
    issue = AuditIssue(
        severity=IssueSeverity.CRITICAL,
        code="ADAPTER_BUILD_FAILED",
        message="The authorized adapter could not build a complete manifest; no claim was made.",
        details=(("error_type", type(error).__name__),),
    )
    return AuditResult(
        status=AuditStatus.FAIL_CLOSED,
        dataset_id=dataset_id,
        summary=AuditSummary(
            record_count=0,
            image_files_checked=0,
            label_files_checked=0,
            mask_files_decoded=0,
            unknown_label_count=0,
            issue_counts=((IssueSeverity.CRITICAL.value, 1),),
        ),
        issues=(issue,),
    )


def _blocked_data_audit(
    dataset_id: str,
    reasons: tuple[str, ...],
    output_dir: Path,
) -> int:
    result = blocked_audit_result(reasons, dataset_id=dataset_id)
    _write_stage2_report(
        result,
        output_dir,
        title=f"RetinaEntail-DR {dataset_id} data audit — BLOCKED",
    )
    print(f"data audit status: {result.status.value}")
    print(f"dataset: {dataset_id}")
    for reason in reasons:
        print(f"  BLOCKED {reason}")
    print(f"json report: {output_dir / 'audit.json'}")
    print(f"html report: {output_dir / 'audit.html'}")
    return 2


def data_audit(
    dataset_id: str,
    *,
    dataset_root: Path | None,
    spec_path: Path | None,
    review_path: Path | None,
    output_dir: Path | None,
) -> int:
    """Run the gate before loading a spec or touching an explicit dataset root."""

    if dataset_id not in _STAGE2_DATASET_ADAPTERS:
        raise ValueError(f"unsupported Stage 2 dataset: {dataset_id}")
    resolved_output = _data_report_output(dataset_id, output_dir)
    if dataset_root is not None and _paths_overlap_lexically(dataset_root, resolved_output):
        raise ValueError("report output must be separate from the dataset root")

    registry_path = _project_root() / "data_registry" / "datasets" / f"{dataset_id}.yaml"
    registry = load_registry_license_metadata(registry_path)
    registry_blocked_reasons: list[str] = []
    if registry.license_status != "verified":
        registry_blocked_reasons.append("license_status must be verified")
    if registry.license_review_status != "reviewed":
        registry_blocked_reasons.append("license_review_status must be reviewed")
    if registry.label_verification_status != "verified":
        registry_blocked_reasons.append("label_verification_status must be verified")
    if review_path is None:
        registry_blocked_reasons.append("an explicit license review record is required")
    if dataset_root is None:
        registry_blocked_reasons.append("an explicit dataset root is required")
    if spec_path is None:
        registry_blocked_reasons.append("an explicit adapter spec is required")
    if registry_blocked_reasons:
        return _blocked_data_audit(
            dataset_id,
            tuple(registry_blocked_reasons),
            resolved_output,
        )

    if review_path is None or dataset_root is None or spec_path is None:
        raise AssertionError("unreachable Stage 2 CLI gate state")
    try:
        review = load_license_review_record(review_path)
    except (OSError, ValueError, TypeError):
        return _blocked_data_audit(
            dataset_id,
            ("the explicit license review record is missing or invalid",),
            resolved_output,
        )
    decision = assess_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.READ_ONLY_AUDIT,
        root_provided=True,
        spec_provided=True,
    )
    if not decision.permitted:
        return _blocked_data_audit(dataset_id, decision.reasons, resolved_output)

    try:
        spec: DatasetAdapterSpec = load_dataset_adapter_spec(spec_path)
    except (OSError, ValueError, TypeError):
        return _blocked_data_audit(
            dataset_id,
            ("the explicit adapter spec is missing or invalid",),
            resolved_output,
        )
    try:
        permit = authorize_dataset_access(
            registry,
            review=review,
            purpose=DataUsePurpose.READ_ONLY_AUDIT,
            root=dataset_root,
            spec=spec,
        )
    except LicenseGateError as error:
        return _blocked_data_audit(dataset_id, error.reasons, resolved_output)

    try:
        adapter = _STAGE2_DATASET_ADAPTERS[dataset_id](permit)
    except AdapterContractError:
        return _blocked_data_audit(
            dataset_id,
            ("the adapter spec is incompatible with the selected dataset adapter",),
            resolved_output,
        )
    manifest_written = False
    try:
        resolved_root = dataset_root.resolve(strict=True)
        if not resolved_root.is_dir():
            raise NotADirectoryError(dataset_root)
    except OSError:
        return _blocked_data_audit(
            dataset_id,
            ("the explicit dataset root is missing or is not a directory",),
            resolved_output,
        )
    try:
        resolved_report = resolved_output.resolve(strict=False)
        if _paths_overlap_lexically(resolved_root, resolved_report):
            raise ValueError("report output must be separate from the dataset root")
        manifest = adapter.build_versioned_manifest()
        result = audit_manifest(manifest, permit=permit)
    except (AdapterContractError, OSError, RuntimeError, Stage2ContractError) as error:
        result = _adapter_failure_result(dataset_id, error)
    else:
        write_bytes_fail_safe(
            resolved_output / "manifest-v2.json",
            canonical_json_bytes(manifest.to_dict()),
        )
        write_manifest_csv(
            manifest,
            resolved_output / "manifest-v2.csv",
            boundary=AuditWriteBoundary(
                allowed_root=_project_root() / "reports" / "generated" / "stage2",
                forbidden_roots=(permit.root,),
            ),
        )
        manifest_written = True

    _write_stage2_report(
        result,
        resolved_output,
        title=f"RetinaEntail-DR {dataset_id} data audit",
        forbidden_roots=(permit.root,),
    )
    print(f"data audit status: {result.status.value}")
    print(f"dataset: {dataset_id}")
    print(f"records: {result.summary.record_count}")
    print(f"issues: {len(result.issues)}")
    print(f"json report: {resolved_output / 'audit.json'}")
    print(f"html report: {resolved_output / 'audit.html'}")
    if manifest_written:
        print(f"manifest json: {resolved_output / 'manifest-v2.json'}")
        print(f"manifest csv: {resolved_output / 'manifest-v2.csv'}")
    if result.status is AuditStatus.PASS:
        return 0
    return 2 if result.status is AuditStatus.BLOCKED else 1


def data_fixture(output_dir: Path | None) -> int:
    resolved_output = _safe_generated_output(
        _project_root() / "outputs", output_dir, "stage2-fixture"
    )
    result = run_stage2_fixture(resolved_output)
    print(f"fixture audit status: {result.audit_result.status.value}")
    print(f"manifest: {result.manifest_path}")
    print(f"json report: {result.audit_json_path}")
    print(f"html report: {result.audit_html_path}")
    return 0


def _stage3_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage3"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage3_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage3"
    return _safe_generated_output(base, output_dir, "toy-calibration-001")


def calibration_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    dataset_root: Path | None,
    spec_path: Path | None,
    review_path: Path | None,
    output_dir: Path | None,
) -> int:
    """Check explicit input presence without opening any supplied data or metadata path."""

    report = assess_stage3_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        review_record_provided=review_path is not None,
        dataset_root_provided=dataset_root is not None,
        adapter_spec_provided=spec_path is not None,
    )
    resolved_output = _stage3_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage3")
    json_path, html_path = write_stage3_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 3 preflight status: {report.status.value}")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage3PreflightStatus.READY_FOR_STAGE2_GATE:
        print("  NOT AUTHORIZED: inputs are only ready for the Stage 2 license Gate")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def calibration_fixture(config_path: Path | None, output_dir: Path | None) -> int:
    root = _project_root()
    resolved_config = config_path or root / "configs" / "stage3_toy_calibration.yaml"
    resolved_output = _stage3_fixture_output(output_dir)
    result = run_stage3_toy_fixture(resolved_config, resolved_output)
    print(f"Stage 3 toy calibration state: {result.state.value}")
    print(f"calibration artifact hash: {result.calibration_artifact_hash}")
    print(f"dual witness replay passed: {result.replay.passed}")
    print(f"package verified: {result.verified}")
    print(f"certificate: {result.certificate_path}")
    return 0


def calibration_verify(package: Path) -> int:
    resolved_package = _stage3_fixture_output(package)
    result = verify_stage3_toy_package(resolved_package)
    print(f"Stage 3 package state: {result.state.value}")
    print(f"dual witness replay passed: {result.replay.passed}")
    print(f"package verified: {result.verified}")
    print(f"certificate: {result.certificate_path}")
    return 0


def _stage4_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage4"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage4_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage4"
    return _safe_generated_output(base, output_dir, "toy-locked-test-001")


def evaluation_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    review_path: Path | None,
    review_purposes: tuple[str, ...],
    dataset_root: Path | None,
    spec_path: Path | None,
    stage2_audit_path: Path | None,
    frozen_data_card_path: Path | None,
    stage3_package_path: Path | None,
    manual_confirmation_path: Path | None,
    locked_evidence_path: Path | None,
    output_dir: Path | None,
) -> int:
    """Check Stage 4 input presence without opening any supplied evidence or data path."""

    purposes = tuple(Stage4ReviewPurpose(value) for value in review_purposes)
    report = assess_stage4_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_provided=review_path is not None,
        approved_review_purposes=purposes,
        dataset_root_provided=dataset_root is not None,
        adapter_spec_provided=spec_path is not None,
        stage2_pass_audit_provided=stage2_audit_path is not None,
        frozen_data_card_provided=frozen_data_card_path is not None,
        stage3_calibration_package_provided=stage3_package_path is not None,
        manual_confirmation_record_provided=manual_confirmation_path is not None,
        locked_evidence_provided=locked_evidence_path is not None,
        locked_evidence=None,
    )
    resolved_output = _stage4_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage4")
    json_path, html_path = write_stage4_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 4 preflight status: {report.status.value}")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage4PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION:
        print("  NOT AUTHORIZED: supplied evidence must still pass exact hash verification")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def evaluation_fixture(
    config_path: Path | None,
    stage3_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    resolved_config = config_path or root / "configs" / "stage4_toy_locked_test.yaml"
    resolved_stage3 = _stage3_fixture_output(stage3_package)
    resolved_output = _stage4_fixture_output(output_dir)
    result = run_stage4_toy_fixture(resolved_config, resolved_stage3, resolved_output)
    print(f"Stage 4 toy case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"locked calibration artifact hash: {result.locked_calibration_artifact_hash}")
    print(f"test data hash: {result.test_data_hash}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def evaluation_verify(package: Path) -> int:
    resolved_package = _stage4_fixture_output(package)
    result = verify_stage4_toy_package(resolved_package)
    print(f"Stage 4 package case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"locked calibration artifact hash: {result.locked_calibration_artifact_hash}")
    print(f"test data hash: {result.test_data_hash}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def _stage5_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage5"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage5_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage5"
    return _safe_generated_output(base, output_dir, "toy-locked-pilot-001")


def pilot_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    review_path: Path | None,
    review_purposes: tuple[str, ...],
    dataset_root: Path | None,
    spec_path: Path | None,
    stage2_audit_path: Path | None,
    frozen_data_card_path: Path | None,
    manual_confirmation_path: Path | None,
    stage3_package_summary_path: Path | None,
    locked_evidence_path: Path | None,
    output_dir: Path | None,
) -> int:
    """Report explicit Stage 5 input presence without opening any supplied path."""

    purposes = tuple(DataUsePurpose(value) for value in review_purposes)
    report = assess_stage5_preflight(
        verification_context=Stage5VerificationContext.REAL_PREREQUISITE,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_provided=review_path is not None,
        approved_review_purposes=purposes,
        dataset_root_provided=dataset_root is not None,
        adapter_spec_provided=spec_path is not None,
        stage2_pass_audit_provided=stage2_audit_path is not None,
        frozen_data_card_provided=frozen_data_card_path is not None,
        stage2_access_permit_provided=False,
        manual_confirmation_provided=manual_confirmation_path is not None,
        stage3_package_summary_provided=stage3_package_summary_path is not None,
        stage4_locked_evidence_provided=locked_evidence_path is not None,
    )
    resolved_output = _stage5_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage5")
    json_path, html_path = write_stage5_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 5 preflight status: {report.status.value}")
    print("prerequisites verified: false")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage5PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION:
        print("  NOT AUTHORIZED: exact evidence and a process-issued Stage 2 permit must verify")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def pilot_fixture(
    config_path: Path | None,
    stage4_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    resolved_config = config_path or root / "configs" / "stage5_toy_locked_pilot.yaml"
    resolved_stage4 = _stage4_fixture_output(stage4_package)
    resolved_output = _stage5_fixture_output(output_dir)
    result = run_stage5_toy_fixture(resolved_config, resolved_stage4, resolved_output)
    print(f"Stage 5 toy pilot case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"source Stage 4 package hash: {result.source_stage4_package_sha256}")
    print(f"pilot data hash: {result.pilot_data_sha256}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def pilot_verify(package: Path) -> int:
    resolved_package = _stage5_fixture_output(package)
    result = verify_stage5_toy_package(resolved_package)
    print(f"Stage 5 package case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"source Stage 4 package hash: {result.source_stage4_package_sha256}")
    print(f"pilot data hash: {result.pilot_data_sha256}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def _stage6_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage6"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage6_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage6"
    return _safe_generated_output(base, output_dir, "toy-verifier-pilot-001")


def _stage6_source_package(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("Stage 6 source package must be a pathlib.Path")
    trusted = (_project_root() / "outputs").resolve(strict=True)
    resolved = path.absolute().resolve(strict=True)
    try:
        resolved.relative_to(trusted)
    except ValueError as error:
        raise ValueError("Stage 6 synthetic sources must stay under project outputs") from error
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def stage6_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    review_purposes: tuple[str, ...],
    dataset_root: Path | None,
    stage2_package_path: Path | None,
    stage3_package_path: Path | None,
    frozen_data_card_path: Path | None,
    manual_confirmation_path: Path | None,
    locked_evidence_path: Path | None,
    output_dir: Path | None,
) -> int:
    """Emit a Stage 6 zero-I/O report; path presence never becomes capability."""

    # These values are intentionally not opened or resolved. A filesystem path is
    # not a process-issued verifier capability, even when the caller supplied it.
    del stage2_package_path, stage3_package_path
    report = assess_stage6_preflight(
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        approved_review_purposes=tuple(DataUsePurpose(item) for item in review_purposes),
        dataset_root_provided=dataset_root is not None,
        stage2_access_permit_provided=False,
        stage2_package_capability_provided=False,
        stage3_package_capability_provided=False,
        frozen_data_card_provided=frozen_data_card_path is not None,
        manual_confirmation_provided=manual_confirmation_path is not None,
        locked_evidence_provided=locked_evidence_path is not None,
    )
    resolved_output = _stage6_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage6")
    json_path, html_path = write_stage6_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 6 preflight status: {report.status.value}")
    print("prerequisites verified: false")
    print("authorization issued: false")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage6PreflightStatus.READY_FOR_PREREQUISITE_VERIFICATION:
        print("  NOT AUTHORIZED: complete package capabilities must verify in this process")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage6_fixture(
    config_path: Path | None,
    stage2_package: Path,
    stage3_package: Path,
    stage5_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (config_path or root / "configs" / "stage6_toy_verifier_pilot.yaml").absolute()
    result = run_stage6_toy_fixture(
        config,
        _stage6_source_package(stage2_package),
        _stage6_source_package(stage3_package),
        _stage6_source_package(stage5_package),
        _stage6_fixture_output(output_dir),
    )
    print(f"Stage 6 toy verifier pilot case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"Stage 2 complete package hash: {result.stage2_package_sha256}")
    print(f"Stage 3 complete package hash: {result.stage3_package_sha256}")
    print(f"Stage 5 complete package hash: {result.stage5_package_sha256}")
    print(f"Stage 6 complete package hash: {result.package_sha256}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def stage6_verify(package: Path) -> int:
    result = verify_stage6_toy_package(_stage6_fixture_output(package))
    print(f"Stage 6 package case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"Stage 6 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def _stage7_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage7"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage7_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage7"
    return _safe_generated_output(base, output_dir, "toy-consumption-pilot-001")


def stage7_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    review_path: Path | None,
    review_purposes: tuple[str, ...],
    dataset_root: Path | None,
    spec_path: Path | None,
    stage2_package_path: Path | None,
    frozen_data_card_path: Path | None,
    manual_confirmation_path: Path | None,
    stage3_package_path: Path | None,
    stage4_locked_evidence_path: Path | None,
    output_dir: Path | None,
) -> int:
    """Emit a path-free Stage 7 report without opening any supplied evidence."""

    approved_purposes = {DataUsePurpose(item) for item in review_purposes}
    report = assess_stage7_preflight(
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        three_purpose_review_provided=(
            review_path is not None and set(REQUIRED_STAGE6_REVIEW_PURPOSES) <= approved_purposes
        ),
        physical_root_provided=dataset_root is not None,
        adapter_spec_provided=spec_path is not None,
        stage2_pass_package_provided=stage2_package_path is not None,
        approved_real_data_card_provided=frozen_data_card_path is not None,
        manual_confirmation_provided=manual_confirmation_path is not None,
        stage3_calibration_package_provided=stage3_package_path is not None,
        stage4_locked_evidence_provided=stage4_locked_evidence_path is not None,
        # A live Stage6BoundRoot cannot cross a CLI/JSON boundary. Paths and
        # receipts supplied here are deliberately never treated as capability.
        stage6_bound_root_provided=False,
    )
    resolved_output = _stage7_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage7")
    json_path, html_path = write_stage7_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 7 preflight status: {report.status.value}")
    print("Stage 6 bound root consumed: false")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage7PreflightStatus.READY_FOR_BOUND_ROOT_CONSUMPTION:
        print("  NOT AUTHORIZED: a live process-issued Stage 6 bound root is still required")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage7_fixture(
    config_path: Path | None,
    stage6_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (config_path or root / "configs" / "stage7_toy_consumption_pilot.yaml").absolute()
    result = run_stage7_toy_fixture(
        config,
        _stage6_source_package(stage6_package),
        _stage7_fixture_output(output_dir),
    )
    print(f"Stage 7 synthetic consumption pilot case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"rebuilt Stage 2 records: {result.rebuilt_record_count}")
    print(f"five-identity records: {result.identity_record_count}")
    print(f"Stage 6 complete package hash: {result.stage6_package_sha256}")
    print(f"Stage 7 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def stage7_verify(package: Path) -> int:
    result = verify_stage7_toy_package(_stage7_fixture_output(package))
    print(f"Stage 7 package case count: {result.case_count}")
    print(f"states: {','.join(state.value for state in result.states)}")
    print(f"Stage 7 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def _stage8_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage8"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage8_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage8"
    return _safe_generated_output(base, output_dir, "toy-repeatability-audit-001")


def stage8_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    output_dir: Path | None,
) -> int:
    """Emit a zero-I/O Stage 8 report; a CLI path can never be a live capability."""

    report = assess_stage8_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        complete_stage7_package_capability_provided=False,
        fresh_root_run_capability_provided=False,
        fixed_seed_preregistered=False,
        degradation_replay_set_preregistered=False,
        inclusion_rules_preregistered=False,
        stability_metrics_preregistered=False,
        locked_contract_provided=False,
    )
    resolved_output = _stage8_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage8")
    json_path, html_path = write_stage8_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 8 preflight status: {report.status.value}")
    print("Stage 7 REAL package capability consumed: false")
    print("fresh process capability consumed: false")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage8PreflightStatus.READY_FOR_PROCESS_CAPABILITY_VERIFICATION:
        print("  NOT AUTHORIZED: live current-process capability replay is still required")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage8_fixture(
    config_path: Path | None,
    stage7_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (config_path or root / "configs" / "stage8_toy_repeatability_audit.yaml").absolute()
    result = run_stage8_toy_fixture(
        config,
        _stage7_fixture_output(stage7_package),
        _stage8_fixture_output(output_dir),
    )
    print(f"Stage 8 synthetic repeatability status: {result.status}")
    print(f"cases: {result.case_count}")
    print(f"repeat runs: {result.repeat_run_count}")
    print(f"observations: {result.observation_count}")
    print(f"Stage 7 source package hash: {result.stage7_package_sha256}")
    print(f"Stage 8 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def stage8_verify(package: Path) -> int:
    result = verify_stage8_toy_package(_stage8_fixture_output(package))
    print(f"Stage 8 package status: {result.status}")
    print(f"cases: {result.case_count}")
    print(f"repeat runs: {result.repeat_run_count}")
    print(f"observations: {result.observation_count}")
    print(f"Stage 8 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    return 0


def _stage9_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage9"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage9_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage9"
    return _safe_generated_output(base, output_dir, "toy-independent-session-diff-001")


def stage9_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    output_dir: Path | None,
) -> int:
    """Emit a zero-I/O REAL report; CLI paths can never become live capabilities."""

    report = assess_stage9_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        complete_stage8_package_a_capability_provided=False,
        complete_stage8_package_b_capability_provided=False,
        fresh_session_a_capability_provided=False,
        fresh_session_b_capability_provided=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    )
    resolved_output = _stage9_preflight_output(output_dir)
    boundary = AuditWriteBoundary(allowed_root=_project_root() / "reports" / "generated" / "stage9")
    json_path, html_path = write_stage9_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 9 preflight status: {report.status.value}")
    print("complete REAL Stage 8 package capabilities consumed: 0/2")
    print("fresh independent-session capabilities consumed: 0/2")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage9PreflightStatus.READY_FOR_DUAL_SESSION_CAPABILITY_VERIFICATION:
        print("  NOT AUTHORIZED: two live current-process verifier capabilities are still required")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage9_fixture(
    config_path: Path | None,
    stage7_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (
        config_path or root / "configs" / "stage9_toy_independent_session_diff.yaml"
    ).absolute()
    result = run_stage9_toy_fixture(
        config,
        _stage7_fixture_output(stage7_package),
        _stage9_fixture_output(output_dir),
    )
    print(f"Stage 9 synthetic independent-session status: {result.status}")
    print(
        f"producer process IDs: {result.producer_process_ids[0]},{result.producer_process_ids[1]}"
    )
    print(f"cases: {result.case_count}")
    print(f"observation comparisons: {result.observation_comparison_count}")
    print(f"matching Stage 8 files: {result.matching_stage8_file_count}")
    print(f"differing Stage 8 files: {result.differing_stage8_file_count}")
    print(f"Stage 8 package hash: {result.stage8_package_sha256}")
    print(f"Stage 9 package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"package diff: {result.output_dir / 'package_diff.json'}")
    return 0


def stage9_verify(package: Path) -> int:
    result = verify_stage9_toy_package(_stage9_fixture_output(package))
    print(f"Stage 9 package status: {result.status}")
    print(
        f"producer process IDs: {result.producer_process_ids[0]},{result.producer_process_ids[1]}"
    )
    print(f"observation comparisons: {result.observation_comparison_count}")
    print(f"matching Stage 8 files: {result.matching_stage8_file_count}")
    print(f"differing Stage 8 files: {result.differing_stage8_file_count}")
    print(f"Stage 9 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"package diff: {result.output_dir / 'package_diff.json'}")
    return 0


def _stage10_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage10"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage10_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage10"
    return _safe_generated_output(base, output_dir, "toy-runner-portability-001")


def stage10_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    output_dir: Path | None,
) -> int:
    """Emit a path-free REAL report; the CLI cannot mint live runner authority."""

    report = assess_stage10_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        complete_real_stage9_package_a_capability_present=False,
        complete_real_stage9_package_b_capability_present=False,
        fresh_runner_a_capability_present=False,
        fresh_runner_b_capability_present=False,
        two_approved_runners_confirmed=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
        signer_trust_policy_preregistered=False,
    )
    resolved_output = _stage10_preflight_output(output_dir)
    boundary = AuditWriteBoundary(
        allowed_root=_project_root() / "reports" / "generated" / "stage10"
    )
    json_path, html_path = write_stage10_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 10 preflight status: {report.status.value}")
    print("complete REAL Stage 9 package capabilities consumed: 0/2")
    print("fresh approved-runner capabilities consumed: 0/2")
    print("approved physical runners confirmed: 0/2")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage10PreflightStatus.READY_FOR_RUNNER_CAPABILITY_VERIFICATION:
        print("  NOT AUTHORIZED: live current-process REAL verifier capabilities are required")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage10_fixture(
    config_path: Path | None,
    stage9_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (config_path or root / "configs" / "stage10_toy_runner_portability.yaml").absolute()
    result = run_stage10_toy_fixture(
        config,
        _stage9_fixture_output(stage9_package),
        _stage10_fixture_output(output_dir),
    )
    print(f"Stage 10 local dual-runner status: {result.status}")
    print(f"runner process IDs: {result.runner_process_ids[0]},{result.runner_process_ids[1]}")
    print("physical host count: 1")
    print("cross-physical-host portability tested: false")
    print(f"cases: {result.case_count}")
    print(f"observation comparisons: {result.observation_comparison_count}")
    print(f"exact Stage 9 files: {result.exact_file_match_count}")
    print(f"allowed provenance differences: {result.allowed_provenance_difference_count}")
    print(f"deterministic file mismatches: {result.deterministic_file_mismatch_count}")
    print(f"Stage 10 package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"artifact diff: {result.output_dir / 'artifact_diff.json'}")
    return 0


def stage10_verify(package: Path) -> int:
    result = verify_stage10_toy_package(_stage10_fixture_output(package))
    print(f"Stage 10 package status: {result.status}")
    print(f"runner process IDs: {result.runner_process_ids[0]},{result.runner_process_ids[1]}")
    print("physical host count: 1")
    print("cross-physical-host portability tested: false")
    print(f"observation comparisons: {result.observation_comparison_count}")
    print(f"exact Stage 9 files: {result.exact_file_match_count}")
    print(f"allowed provenance differences: {result.allowed_provenance_difference_count}")
    print(f"deterministic file mismatches: {result.deterministic_file_mismatch_count}")
    print(f"Stage 10 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"artifact diff: {result.output_dir / 'artifact_diff.json'}")
    return 0


def _stage11_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage11"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage11_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage11"
    return _safe_generated_output(base, output_dir, "toy-asymmetric-exchange-001")


def stage11_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    output_dir: Path | None,
) -> int:
    """Emit the zero-I/O REAL report; the CLI cannot ingest production private keys."""

    report = assess_stage11_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        complete_real_stage10_package_a_capability_present=False,
        complete_real_stage10_package_b_capability_present=False,
        fresh_host_a_process_capability_present=False,
        fresh_host_b_process_capability_present=False,
        two_approved_physical_hosts_confirmed=False,
        production_asymmetric_trust_anchor_verified=False,
        unique_runner_leaf_keys_verified=False,
        certificate_chains_verified=False,
        revocation_status_current=False,
        trusted_time_policy_verified=False,
        complete_wheel_lock_hashes_verified=False,
        host_environment_policy_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    )
    resolved_output = _stage11_preflight_output(output_dir)
    boundary = AuditWriteBoundary(
        allowed_root=_project_root() / "reports" / "generated" / "stage11"
    )
    json_path, html_path = write_stage11_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 11 preflight status: {report.status.value}")
    print("complete REAL Stage 10 package capabilities consumed: 0/2")
    print("fresh approved-host process capabilities consumed: 0/2")
    print("approved physical hosts confirmed: 0/2")
    print("production asymmetric trust anchors verified: 0")
    print("complete wheel/lock hashes verified: false")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage11PreflightStatus.READY_FOR_PRODUCTION_HOST_VERIFICATION:
        print("  NOT AUTHORIZED: live production verifier capabilities are still required")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage11_fixture(
    config_path: Path | None,
    stage10_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (config_path or root / "configs" / "stage11_toy_asymmetric_exchange.yaml").absolute()
    result = run_stage11_toy_fixture(
        config,
        _stage10_fixture_output(stage10_package),
        _stage11_fixture_output(output_dir),
    )
    print(f"Stage 11 local asymmetric fixture status: {result.status}")
    print("physical host count: 1")
    print("cross-physical-host portability tested: false")
    print("production authentication: false")
    print("signature algorithm: Ed25519")
    print(f"unique runner leaf keys: {result.runner_key_ids[0]},{result.runner_key_ids[1]}")
    print(f"tamper rejections: {result.tamper_rejection_count}/{result.tamper_scenario_count}")
    print(f"frozen baseline status: {result.baseline_status}")
    print(f"Stage 11 package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"tamper audit: {result.output_dir / 'tamper_audit.json'}")
    return 0


def stage11_verify(package: Path) -> int:
    result = verify_stage11_toy_package(_stage11_fixture_output(package))
    print(f"Stage 11 package status: {result.status}")
    print("physical host count: 1")
    print("cross-physical-host portability tested: false")
    print("production authentication: false")
    print("signature algorithm: Ed25519")
    print(f"unique runner leaf keys: {result.runner_key_ids[0]},{result.runner_key_ids[1]}")
    print(f"tamper rejections: {result.tamper_rejection_count}/{result.tamper_scenario_count}")
    print(f"frozen baseline status: {result.baseline_status}")
    print(f"Stage 11 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"tamper audit: {result.output_dir / 'tamper_audit.json'}")
    return 0


def _stage12_preflight_output(output_dir: Path | None) -> Path:
    base = _project_root() / "reports" / "generated" / "stage12"
    return _safe_generated_output(base, output_dir, "current-preflight")


def _stage12_fixture_output(output_dir: Path | None) -> Path:
    base = _project_root() / "outputs" / "stage12"
    return _safe_generated_output(base, output_dir, "toy-signer-recovery-001")


def stage12_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    output_dir: Path | None,
) -> int:
    """Emit a path-free Gate report; CLI values are never live capabilities."""

    report = assess_stage12_preflight(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        complete_real_stage11_package_a_capability_present=False,
        complete_real_stage11_package_b_capability_present=False,
        fresh_runner_a_process_capability_present=False,
        fresh_runner_b_process_capability_present=False,
        two_governance_approved_hardware_runners_verified=False,
        hsm_or_controlled_signer_boundary_verified=False,
        trusted_ca_chains_verified=False,
        trusted_timestamp_policy_verified=False,
        current_ocsp_status_verified=False,
        current_crl_status_verified=False,
        complete_wheel_lock_hashes_verified=False,
        hardware_identity_verified=False,
        frozen_baseline_contract_preregistered=False,
        compromise_recovery_drill_preregistered=False,
    )
    resolved_output = _stage12_preflight_output(output_dir)
    boundary = AuditWriteBoundary(
        allowed_root=_project_root() / "reports" / "generated" / "stage12"
    )
    json_path, html_path = write_stage12_preflight_reports(
        report,
        resolved_output,
        boundary=boundary,
    )
    print(f"Stage 12 preflight status: {report.status.value}")
    print("complete REAL Stage 11 package capabilities consumed: 0/2")
    print("fresh approved-hardware-runner capabilities consumed: 0/2")
    print("HSM/controlled production signer boundaries verified: 0/2")
    print("current OCSP/CRL status verified: false")
    print("dataset root probed: false")
    print("real data read: false")
    print("sample count: 0")
    for reason in report.reasons:
        print(f"  BLOCKED {reason.value}")
    if report.status is Stage12PreflightStatus.READY_FOR_PRODUCTION_CEREMONY_VERIFICATION:
        print("  NOT AUTHORIZED: live production verifier capabilities are still required")
    print(f"json report: {json_path}")
    print(f"html report: {html_path}")
    return 2


def stage12_fixture(
    config_path: Path | None,
    stage11_package: Path,
    output_dir: Path | None,
) -> int:
    root = _project_root()
    config = (config_path or root / "configs" / "stage12_toy_signer_recovery.yaml").absolute()
    result = run_stage12_fixture(
        config,
        _stage11_fixture_output(stage11_package),
        _stage12_fixture_output(output_dir),
    )
    print(f"Stage 12 local signer recovery status: {result.status}")
    print(f"independent process IDs: {result.runner_process_ids[0]},{result.runner_process_ids[1]}")
    print("physical host count: 1")
    print("cross-physical-host ceremony executed: false")
    print("production authentication: false")
    print("signature algorithm: Ed25519")
    print(f"recovered key IDs: {result.recovered_key_ids[0]},{result.recovered_key_ids[1]}")
    print(f"tamper rejections: {result.tamper_rejection_count}/{result.tamper_scenario_count}")
    print(f"recovered runner comparison: {result.comparison_status}")
    print(f"Stage 12 package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"tamper audit: {result.output_dir / 'tamper_audit.json'}")
    return 0


def stage12_verify(package: Path) -> int:
    result = verify_stage12_package(_stage12_fixture_output(package))
    print(f"Stage 12 package status: {result.status}")
    print(f"independent process IDs: {result.runner_process_ids[0]},{result.runner_process_ids[1]}")
    print("physical host count: 1")
    print("production authentication: false")
    print(f"recovered key IDs: {result.recovered_key_ids[0]},{result.recovered_key_ids[1]}")
    print(f"tamper rejections: {result.tamper_rejection_count}/{result.tamper_scenario_count}")
    print(f"recovered runner comparison: {result.comparison_status}")
    print(f"Stage 12 complete package hash: {result.package_sha256}")
    print(f"package files: {result.package_file_count}")
    print(f"package verified: {result.verified}")
    print(f"json report: {result.output_dir / 'audit.json'}")
    print(f"html report: {result.output_dir / 'audit.html'}")
    print(f"tamper audit: {result.output_dir / 'tamper_audit.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retinaentail", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check the local offline RetinaEntail environment")
    smoke_parser = commands.add_parser("smoke", help="run the CPU-only synthetic pipeline")
    smoke_parser.add_argument("--config", type=Path, default=None)
    smoke_parser.add_argument("--output-dir", type=Path, default=None)
    registry_parser = commands.add_parser("registry", help="validate metadata registries")
    registry_commands = registry_parser.add_subparsers(dest="registry_command", required=True)
    registry_commands.add_parser("validate", help="validate all eight dataset templates")
    data_parser = commands.add_parser("data", help="run license-gated Stage 2 data tools")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    audit_parser = data_commands.add_parser(
        "audit",
        help="audit one explicitly authorized local dataset or emit a BLOCKED report",
    )
    audit_parser.add_argument("--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), required=True)
    audit_parser.add_argument("--root", type=Path, default=None)
    audit_parser.add_argument("--spec", type=Path, default=None)
    audit_parser.add_argument("--review-record", type=Path, default=None)
    audit_parser.add_argument("--output-dir", type=Path, default=None)
    fixture_parser = data_commands.add_parser(
        "fixture", help="generate and audit a deterministic synthetic local fixture"
    )
    fixture_parser.add_argument("--output-dir", type=Path, default=None)
    calibration_parser = commands.add_parser(
        "calibration", help="run fail-closed Stage 3 calibration tools"
    )
    calibration_commands = calibration_parser.add_subparsers(
        dest="calibration_command", required=True
    )
    preflight_parser = calibration_commands.add_parser(
        "preflight",
        help="check Stage 3 input presence without reading supplied paths",
    )
    preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    preflight_parser.add_argument("--dataset-version", default=None)
    preflight_parser.add_argument("--root", type=Path, default=None)
    preflight_parser.add_argument("--spec", type=Path, default=None)
    preflight_parser.add_argument("--review-record", type=Path, default=None)
    preflight_parser.add_argument("--output-dir", type=Path, default=None)
    calibration_fixture_parser = calibration_commands.add_parser(
        "fixture",
        help="run the deterministic CPU-only synthetic calibration package",
    )
    calibration_fixture_parser.add_argument("--config", type=Path, default=None)
    calibration_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    verify_parser = calibration_commands.add_parser(
        "verify", help="independently regenerate and replay a Stage 3 toy package"
    )
    verify_parser.add_argument("--package", type=Path, required=True)
    evaluation_parser = commands.add_parser(
        "evaluation", help="run fail-closed Stage 4 locked-test tools"
    )
    evaluation_commands = evaluation_parser.add_subparsers(dest="evaluation_command", required=True)
    evaluation_preflight_parser = evaluation_commands.add_parser(
        "preflight",
        help="check Stage 4 prerequisite presence without reading supplied paths",
    )
    evaluation_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    evaluation_preflight_parser.add_argument("--dataset-version", default=None)
    evaluation_preflight_parser.add_argument("--root", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--spec", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--review-record", type=Path, default=None)
    evaluation_preflight_parser.add_argument(
        "--review-purpose",
        action="append",
        choices=tuple(item.value for item in Stage4ReviewPurpose),
        default=[],
    )
    evaluation_preflight_parser.add_argument("--stage2-audit", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--frozen-data-card", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--stage3-package", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--manual-confirmation", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--locked-evidence", type=Path, default=None)
    evaluation_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    evaluation_fixture_parser = evaluation_commands.add_parser(
        "fixture",
        help="run a deterministic CPU-only synthetic locked-test package",
    )
    evaluation_fixture_parser.add_argument("--config", type=Path, default=None)
    evaluation_fixture_parser.add_argument("--stage3-package", type=Path, required=True)
    evaluation_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    evaluation_verify_parser = evaluation_commands.add_parser(
        "verify",
        help="independently regenerate and replay a Stage 4 toy package",
    )
    evaluation_verify_parser.add_argument("--package", type=Path, required=True)
    pilot_parser = commands.add_parser(
        "pilot", help="run fail-closed Stage 5 prerequisite and locked-pilot tools"
    )
    pilot_commands = pilot_parser.add_subparsers(dest="pilot_command", required=True)
    pilot_preflight_parser = pilot_commands.add_parser(
        "preflight",
        help="check Stage 5 prerequisite presence without reading supplied paths",
    )
    pilot_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    pilot_preflight_parser.add_argument("--dataset-version", default=None)
    pilot_preflight_parser.add_argument("--root", type=Path, default=None)
    pilot_preflight_parser.add_argument("--spec", type=Path, default=None)
    pilot_preflight_parser.add_argument("--review-record", type=Path, default=None)
    pilot_preflight_parser.add_argument(
        "--review-purpose",
        action="append",
        choices=tuple(item.value for item in DataUsePurpose),
        default=[],
    )
    pilot_preflight_parser.add_argument("--stage2-audit", type=Path, default=None)
    pilot_preflight_parser.add_argument("--frozen-data-card", type=Path, default=None)
    pilot_preflight_parser.add_argument("--manual-confirmation", type=Path, default=None)
    pilot_preflight_parser.add_argument("--stage3-package-summary", type=Path, default=None)
    pilot_preflight_parser.add_argument("--locked-evidence", type=Path, default=None)
    pilot_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    pilot_fixture_parser = pilot_commands.add_parser(
        "fixture",
        help="run a deterministic CPU-only synthetic locked-pilot package",
    )
    pilot_fixture_parser.add_argument("--config", type=Path, default=None)
    pilot_fixture_parser.add_argument("--stage4-package", type=Path, required=True)
    pilot_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    pilot_verify_parser = pilot_commands.add_parser(
        "verify",
        help="independently regenerate a Stage 5 toy pilot and replay its Stage 4 source",
    )
    pilot_verify_parser.add_argument("--package", type=Path, required=True)
    stage6_parser = commands.add_parser(
        "stage6",
        help="run complete-package capability and verifier-pilot tools",
    )
    stage6_commands = stage6_parser.add_subparsers(dest="stage6_command", required=True)
    stage6_preflight_parser = stage6_commands.add_parser(
        "preflight",
        help="report Stage 6 explicit presence without opening any supplied path",
    )
    stage6_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage6_preflight_parser.add_argument("--dataset-version", default=None)
    stage6_preflight_parser.add_argument("--root", type=Path, default=None)
    stage6_preflight_parser.add_argument(
        "--review-purpose",
        action="append",
        choices=tuple(item.value for item in DataUsePurpose),
        default=[],
    )
    stage6_preflight_parser.add_argument("--stage2-package", type=Path, default=None)
    stage6_preflight_parser.add_argument("--stage3-package", type=Path, default=None)
    stage6_preflight_parser.add_argument("--frozen-data-card", type=Path, default=None)
    stage6_preflight_parser.add_argument("--manual-confirmation", type=Path, default=None)
    stage6_preflight_parser.add_argument("--locked-evidence", type=Path, default=None)
    stage6_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage6_fixture_parser = stage6_commands.add_parser(
        "fixture",
        help="embed and replay complete synthetic Stage 2, Stage 3, and Stage 5 packages",
    )
    stage6_fixture_parser.add_argument("--config", type=Path, default=None)
    stage6_fixture_parser.add_argument("--stage2-package", type=Path, required=True)
    stage6_fixture_parser.add_argument("--stage3-package", type=Path, required=True)
    stage6_fixture_parser.add_argument("--stage5-package", type=Path, required=True)
    stage6_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage6_verify_parser = stage6_commands.add_parser(
        "verify",
        help="reissue complete-package capabilities and byte-verify a Stage 6 package",
    )
    stage6_verify_parser.add_argument("--package", type=Path, required=True)
    stage7_parser = commands.add_parser(
        "stage7",
        help="run authorization-consumption, rebuild, and frozen-pilot tools",
    )
    stage7_commands = stage7_parser.add_subparsers(dest="stage7_command", required=True)
    stage7_preflight_parser = stage7_commands.add_parser(
        "preflight",
        help="report Stage 7 presence without opening any supplied path",
    )
    stage7_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage7_preflight_parser.add_argument("--dataset-version", default=None)
    stage7_preflight_parser.add_argument("--root", type=Path, default=None)
    stage7_preflight_parser.add_argument("--spec", type=Path, default=None)
    stage7_preflight_parser.add_argument("--review-record", type=Path, default=None)
    stage7_preflight_parser.add_argument(
        "--review-purpose",
        action="append",
        choices=tuple(item.value for item in DataUsePurpose),
        default=[],
    )
    stage7_preflight_parser.add_argument("--stage2-package", type=Path, default=None)
    stage7_preflight_parser.add_argument("--frozen-data-card", type=Path, default=None)
    stage7_preflight_parser.add_argument("--manual-confirmation", type=Path, default=None)
    stage7_preflight_parser.add_argument("--stage3-package", type=Path, default=None)
    stage7_preflight_parser.add_argument("--stage4-locked-evidence", type=Path, default=None)
    stage7_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage7_fixture_parser = stage7_commands.add_parser(
        "fixture",
        help="consume and replay one complete synthetic Stage 6 package",
    )
    stage7_fixture_parser.add_argument("--config", type=Path, default=None)
    stage7_fixture_parser.add_argument("--stage6-package", type=Path, required=True)
    stage7_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage7_verify_parser = stage7_commands.add_parser(
        "verify",
        help="byte-replay a complete Stage 7 synthetic package",
    )
    stage7_verify_parser.add_argument("--package", type=Path, required=True)
    stage8_parser = commands.add_parser(
        "stage8",
        help="run frozen-pilot repeatability and dual-witness stability tools",
    )
    stage8_commands = stage8_parser.add_subparsers(dest="stage8_command", required=True)
    stage8_preflight_parser = stage8_commands.add_parser(
        "preflight",
        help="report Stage 8 presence without accepting a path or opening a root",
    )
    stage8_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage8_preflight_parser.add_argument("--dataset-version", default=None)
    stage8_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage8_fixture_parser = stage8_commands.add_parser(
        "fixture",
        help="repeat a complete locked Stage 7 synthetic package under preregistration",
    )
    stage8_fixture_parser.add_argument("--config", type=Path, default=None)
    stage8_fixture_parser.add_argument("--stage7-package", type=Path, required=True)
    stage8_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage8_verify_parser = stage8_commands.add_parser(
        "verify",
        help="byte-replay a complete Stage 8 synthetic repeatability package",
    )
    stage8_verify_parser.add_argument("--package", type=Path, required=True)
    stage9_parser = commands.add_parser(
        "stage9",
        help="run independent-process/session reproducibility and complete-package diff tools",
    )
    stage9_commands = stage9_parser.add_subparsers(dest="stage9_command", required=True)
    stage9_preflight_parser = stage9_commands.add_parser(
        "preflight",
        help="report Stage 9 REAL prerequisite presence without accepting package or root paths",
    )
    stage9_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage9_preflight_parser.add_argument("--dataset-version", default=None)
    stage9_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage9_fixture_parser = stage9_commands.add_parser(
        "fixture",
        help="launch two independent Python producers and diff their complete Stage 8 packages",
    )
    stage9_fixture_parser.add_argument("--config", type=Path, default=None)
    stage9_fixture_parser.add_argument("--stage7-package", type=Path, required=True)
    stage9_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage9_verify_parser = stage9_commands.add_parser(
        "verify",
        help="replay both embedded Stage 8 packages and verify the complete Stage 9 diff",
    )
    stage9_verify_parser.add_argument("--package", type=Path, required=True)
    stage10_parser = commands.add_parser(
        "stage10",
        help="audit local runner portability and signed offline artifact exchange",
    )
    stage10_commands = stage10_parser.add_subparsers(dest="stage10_command", required=True)
    stage10_preflight_parser = stage10_commands.add_parser(
        "preflight",
        help="report Stage 10 REAL prerequisites without accepting package or root paths",
    )
    stage10_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage10_preflight_parser.add_argument("--dataset-version", default=None)
    stage10_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage10_fixture_parser = stage10_commands.add_parser(
        "fixture",
        help="launch two local runner processes and verify signed Stage 9 artifact exchange",
    )
    stage10_fixture_parser.add_argument("--config", type=Path, default=None)
    stage10_fixture_parser.add_argument("--stage9-package", type=Path, required=True)
    stage10_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage10_verify_parser = stage10_commands.add_parser(
        "verify",
        help="offline-replay both embedded Stage 9 packages, envelopes, and exact diff",
    )
    stage10_verify_parser.add_argument("--package", type=Path, required=True)
    stage11_parser = commands.add_parser(
        "stage11",
        help="audit asymmetric signing, rotation, revocation, and cross-host prerequisites",
    )
    stage11_commands = stage11_parser.add_subparsers(dest="stage11_command", required=True)
    stage11_preflight_parser = stage11_commands.add_parser(
        "preflight",
        help="report Stage 11 REAL prerequisites without paths or private-key input",
    )
    stage11_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage11_preflight_parser.add_argument("--dataset-version", default=None)
    stage11_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage11_fixture_parser = stage11_commands.add_parser(
        "fixture",
        help="run the local Ed25519 rotation/revocation/tamper fixture",
    )
    stage11_fixture_parser.add_argument("--config", type=Path, default=None)
    stage11_fixture_parser.add_argument("--stage10-package", type=Path, required=True)
    stage11_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage11_verify_parser = stage11_commands.add_parser(
        "verify",
        help="offline-verify Stage 10 source, PKI artifacts, tamper audit, and package",
    )
    stage11_verify_parser.add_argument("--package", type=Path, required=True)
    stage12_parser = commands.add_parser(
        "stage12",
        help="audit signer interoperability, compromise recovery, and production prerequisites",
    )
    stage12_commands = stage12_parser.add_subparsers(dest="stage12_command", required=True)
    stage12_preflight_parser = stage12_commands.add_parser(
        "preflight",
        help="report Stage 12 REAL prerequisites without paths or private-key input",
    )
    stage12_preflight_parser.add_argument(
        "--dataset", choices=tuple(_STAGE2_DATASET_ADAPTERS), default=None
    )
    stage12_preflight_parser.add_argument("--dataset-version", default=None)
    stage12_preflight_parser.add_argument("--output-dir", type=Path, default=None)
    stage12_fixture_parser = stage12_commands.add_parser(
        "fixture",
        help="run two process-local random Ed25519 compromise/recovery adapters",
    )
    stage12_fixture_parser.add_argument("--config", type=Path, default=None)
    stage12_fixture_parser.add_argument("--stage11-package", type=Path, required=True)
    stage12_fixture_parser.add_argument("--output-dir", type=Path, default=None)
    stage12_verify_parser = stage12_commands.add_parser(
        "verify",
        help="offline-verify source, signer recovery, tamper audit, and complete package",
    )
    stage12_verify_parser.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "doctor":
            return doctor()
        if arguments.command == "smoke":
            return smoke(arguments.config, arguments.output_dir)
        if arguments.command == "registry" and arguments.registry_command == "validate":
            return registry_validate()
        if arguments.command == "data" and arguments.data_command == "audit":
            return data_audit(
                arguments.dataset,
                dataset_root=arguments.root,
                spec_path=arguments.spec,
                review_path=arguments.review_record,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "data" and arguments.data_command == "fixture":
            return data_fixture(arguments.output_dir)
        if arguments.command == "calibration" and arguments.calibration_command == "preflight":
            return calibration_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                dataset_root=arguments.root,
                spec_path=arguments.spec,
                review_path=arguments.review_record,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "calibration" and arguments.calibration_command == "fixture":
            return calibration_fixture(arguments.config, arguments.output_dir)
        if arguments.command == "calibration" and arguments.calibration_command == "verify":
            return calibration_verify(arguments.package)
        if arguments.command == "evaluation" and arguments.evaluation_command == "preflight":
            return evaluation_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                review_path=arguments.review_record,
                review_purposes=tuple(arguments.review_purpose),
                dataset_root=arguments.root,
                spec_path=arguments.spec,
                stage2_audit_path=arguments.stage2_audit,
                frozen_data_card_path=arguments.frozen_data_card,
                stage3_package_path=arguments.stage3_package,
                manual_confirmation_path=arguments.manual_confirmation,
                locked_evidence_path=arguments.locked_evidence,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "evaluation" and arguments.evaluation_command == "fixture":
            return evaluation_fixture(
                arguments.config,
                arguments.stage3_package,
                arguments.output_dir,
            )
        if arguments.command == "evaluation" and arguments.evaluation_command == "verify":
            return evaluation_verify(arguments.package)
        if arguments.command == "pilot" and arguments.pilot_command == "preflight":
            return pilot_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                review_path=arguments.review_record,
                review_purposes=tuple(arguments.review_purpose),
                dataset_root=arguments.root,
                spec_path=arguments.spec,
                stage2_audit_path=arguments.stage2_audit,
                frozen_data_card_path=arguments.frozen_data_card,
                manual_confirmation_path=arguments.manual_confirmation,
                stage3_package_summary_path=arguments.stage3_package_summary,
                locked_evidence_path=arguments.locked_evidence,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "pilot" and arguments.pilot_command == "fixture":
            return pilot_fixture(
                arguments.config,
                arguments.stage4_package,
                arguments.output_dir,
            )
        if arguments.command == "pilot" and arguments.pilot_command == "verify":
            return pilot_verify(arguments.package)
        if arguments.command == "stage6" and arguments.stage6_command == "preflight":
            return stage6_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                review_purposes=tuple(arguments.review_purpose),
                dataset_root=arguments.root,
                stage2_package_path=arguments.stage2_package,
                stage3_package_path=arguments.stage3_package,
                frozen_data_card_path=arguments.frozen_data_card,
                manual_confirmation_path=arguments.manual_confirmation,
                locked_evidence_path=arguments.locked_evidence,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage6" and arguments.stage6_command == "fixture":
            return stage6_fixture(
                arguments.config,
                arguments.stage2_package,
                arguments.stage3_package,
                arguments.stage5_package,
                arguments.output_dir,
            )
        if arguments.command == "stage6" and arguments.stage6_command == "verify":
            return stage6_verify(arguments.package)
        if arguments.command == "stage7" and arguments.stage7_command == "preflight":
            return stage7_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                review_path=arguments.review_record,
                review_purposes=tuple(arguments.review_purpose),
                dataset_root=arguments.root,
                spec_path=arguments.spec,
                stage2_package_path=arguments.stage2_package,
                frozen_data_card_path=arguments.frozen_data_card,
                manual_confirmation_path=arguments.manual_confirmation,
                stage3_package_path=arguments.stage3_package,
                stage4_locked_evidence_path=arguments.stage4_locked_evidence,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage7" and arguments.stage7_command == "fixture":
            return stage7_fixture(
                arguments.config,
                arguments.stage6_package,
                arguments.output_dir,
            )
        if arguments.command == "stage7" and arguments.stage7_command == "verify":
            return stage7_verify(arguments.package)
        if arguments.command == "stage8" and arguments.stage8_command == "preflight":
            return stage8_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage8" and arguments.stage8_command == "fixture":
            return stage8_fixture(
                arguments.config,
                arguments.stage7_package,
                arguments.output_dir,
            )
        if arguments.command == "stage8" and arguments.stage8_command == "verify":
            return stage8_verify(arguments.package)
        if arguments.command == "stage9" and arguments.stage9_command == "preflight":
            return stage9_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage9" and arguments.stage9_command == "fixture":
            return stage9_fixture(
                arguments.config,
                arguments.stage7_package,
                arguments.output_dir,
            )
        if arguments.command == "stage9" and arguments.stage9_command == "verify":
            return stage9_verify(arguments.package)
        if arguments.command == "stage10" and arguments.stage10_command == "preflight":
            return stage10_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage10" and arguments.stage10_command == "fixture":
            return stage10_fixture(
                arguments.config,
                arguments.stage9_package,
                arguments.output_dir,
            )
        if arguments.command == "stage10" and arguments.stage10_command == "verify":
            return stage10_verify(arguments.package)
        if arguments.command == "stage11" and arguments.stage11_command == "preflight":
            return stage11_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage11" and arguments.stage11_command == "fixture":
            return stage11_fixture(
                arguments.config,
                arguments.stage10_package,
                arguments.output_dir,
            )
        if arguments.command == "stage11" and arguments.stage11_command == "verify":
            return stage11_verify(arguments.package)
        if arguments.command == "stage12" and arguments.stage12_command == "preflight":
            return stage12_preflight(
                dataset_id=arguments.dataset,
                dataset_version=arguments.dataset_version,
                output_dir=arguments.output_dir,
            )
        if arguments.command == "stage12" and arguments.stage12_command == "fixture":
            return stage12_fixture(
                arguments.config,
                arguments.stage11_package,
                arguments.output_dir,
            )
        if arguments.command == "stage12" and arguments.stage12_command == "verify":
            return stage12_verify(arguments.package)
    except (LicenseMetadataError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    parser.error("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
