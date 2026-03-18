from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aurora_vtol.effectiveness import (
    GeometrySeedSpec,
    NominalEffectivenessTable,
    build_effectiveness_table_from_geometry_seed,
    effectiveness_table_to_payload,
    geometry_seed_spec_to_payload,
    load_effectiveness_table,
    load_geometry_seed_spec,
    summarize_effectiveness_table,
    summarize_geometry_seed_spec,
)


DEFAULT_GEOMETRY_SEED_SPEC = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "effectiveness_specs"
    / "aurora_ring32_geometry_seed_v1.json"
)
DEFAULT_CANDIDATE_TEMPLATE_SPEC_NAME = "aurora-vtol-ring32-candidate-template-v1"
DEFAULT_CANDIDATE_TEMPLATE_PROVENANCE = (
    "candidate template; replace with actual CAD-, CFD-, or bench-derived provenance before use"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_input_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = _repo_root() / value
    return value.resolve()


def _relative_repo_path(path: str | Path) -> str:
    value = Path(path).resolve()
    repo_root = _repo_root()
    try:
        return str(value.relative_to(repo_root))
    except ValueError:
        return str(value)


def _summary_stats(values) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _weight_summary(weights) -> dict:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D weights, got shape {arr.shape}")
    if arr.size == 0:
        return {
            "row_sum_min": 0.0,
            "row_sum_max": 0.0,
            "row_sum_mean": 0.0,
            "column_sum_min": 0.0,
            "column_sum_max": 0.0,
            "column_sum_mean": 0.0,
            "dominant_segment_weight_min": 0.0,
            "dominant_segment_weight_max": 0.0,
            "dominant_segment_weight_mean": 0.0,
            "support_segments_min": 0.0,
            "support_segments_max": 0.0,
            "support_segments_mean": 0.0,
        }
    row_sums = arr.sum(axis=1)
    column_sums = arr.sum(axis=0)
    dominant_segment_weight = arr.max(axis=0)
    row_max = arr.max(axis=1)
    support_segments = []
    for row_idx in range(arr.shape[0]):
        threshold = max(float(row_max[row_idx]) * 0.1, 1e-9)
        support_segments.append(int(np.count_nonzero(arr[row_idx] >= threshold)))
    return {
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
        "row_sum_mean": float(np.mean(row_sums)),
        "column_sum_min": float(np.min(column_sums)),
        "column_sum_max": float(np.max(column_sums)),
        "column_sum_mean": float(np.mean(column_sums)),
        "dominant_segment_weight_min": float(np.min(dominant_segment_weight)),
        "dominant_segment_weight_max": float(np.max(dominant_segment_weight)),
        "dominant_segment_weight_mean": float(np.mean(dominant_segment_weight)),
        "support_segments_min": float(np.min(support_segments)),
        "support_segments_max": float(np.max(support_segments)),
        "support_segments_mean": float(np.mean(support_segments)),
    }


def _component_scale_summary(table: NominalEffectivenessTable) -> dict:
    return {
        "axial": _summary_stats(table.axial_scale_by_segment),
        "radial": _summary_stats(table.radial_scale_by_segment),
        "tangential": _summary_stats(table.tangential_scale_by_segment),
    }


def _build_report_warnings(
    *,
    spec: GeometrySeedSpec | None,
    table: NominalEffectivenessTable,
    fan_weight_summary: dict,
    plenum_weight_summary: dict,
    component_scale_summary: dict,
) -> list[str]:
    warnings: list[str] = []
    provenance = str(spec.provenance if spec is not None else table.provenance).lower()
    if "provisional" in provenance:
        warnings.append(
            "Effectiveness source is still provisional, not CAD-, CFD-, or bench-validated hardware truth."
        )
    if all(
        abs(component_scale_summary[name]["min"] - 1.0) <= 1e-9
        and abs(component_scale_summary[name]["max"] - 1.0) <= 1e-9
        for name in ("axial", "radial", "tangential")
    ):
        warnings.append(
            "Axial, radial, and tangential component scales are still uniform unity values."
        )
    if (
        abs(fan_weight_summary["column_sum_min"] - 1.0) > 1e-6
        or abs(fan_weight_summary["column_sum_max"] - 1.0) > 1e-6
    ):
        warnings.append("Fan-to-segment weights are not column-normalized within tolerance.")
    if (
        abs(plenum_weight_summary["column_sum_min"] - 1.0) > 1e-6
        or abs(plenum_weight_summary["column_sum_max"] - 1.0) > 1e-6
    ):
        warnings.append("Plenum-to-segment weights are not column-normalized within tolerance.")
    if (
        spec is not None
        and np.all(spec.plenum_half_span_deg <= 1e-9)
        and np.all(spec.plenum_sigma_deg <= 1e-9)
    ):
        warnings.append(
            "Nominal plenum coupling is still point-owned with no spillover in the current geometry seed."
        )
    return warnings


def _resolve_effectiveness_source(
    *,
    spec_path: str | Path | None = None,
    table_path: str | Path | None = None,
    default_to_geometry_seed: bool,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None]:
    if spec_path and table_path:
        raise ValueError("Provide either spec_path or table_path, not both.")
    if not spec_path and not table_path:
        if not default_to_geometry_seed:
            raise ValueError("Provide at least one of --candidate-spec or --candidate-table.")
        spec_path = DEFAULT_GEOMETRY_SEED_SPEC

    spec: GeometrySeedSpec | None = None
    if table_path:
        resolved_source = Path(table_path).resolve()
        table = load_effectiveness_table(resolved_source)
        source_kind = "table"
    else:
        resolved_source = Path(spec_path).resolve()
        spec = load_geometry_seed_spec(resolved_source)
        table = build_effectiveness_table_from_geometry_seed(spec)
        source_kind = "geometry-seed"

    payload = {
        "source_kind": source_kind,
        "source_path": _relative_repo_path(resolved_source),
        "spec_summary": summarize_geometry_seed_spec(spec) if spec is not None else None,
        "table_summary": summarize_effectiveness_table(table),
    }
    return payload, table, spec


def build_effectiveness_report(
    *,
    spec_path: str | Path | None = None,
    table_path: str | Path | None = None,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None]:
    source_payload, table, spec = _resolve_effectiveness_source(
        spec_path=spec_path,
        table_path=table_path,
        default_to_geometry_seed=True,
    )
    fan_weight_summary = _weight_summary(table.fan_segment_weights)
    plenum_weight_summary = _weight_summary(table.plenum_segment_weights)
    component_scale_summary = _component_scale_summary(table)
    report = {
        **source_payload,
        "fan_weight_summary": fan_weight_summary,
        "plenum_weight_summary": plenum_weight_summary,
        "component_scale_summary": component_scale_summary,
        "warnings": _build_report_warnings(
            spec=spec,
            table=table,
            fan_weight_summary=fan_weight_summary,
            plenum_weight_summary=plenum_weight_summary,
            component_scale_summary=component_scale_summary,
        ),
    }
    return report, table, spec


def _delta_summary(baseline_table: NominalEffectivenessTable, candidate_table: NominalEffectivenessTable) -> dict:
    return {
        "fan_weight_max_abs_delta": float(np.max(np.abs(candidate_table.fan_segment_weights - baseline_table.fan_segment_weights))),
        "fan_weight_mean_abs_delta": float(np.mean(np.abs(candidate_table.fan_segment_weights - baseline_table.fan_segment_weights))),
        "plenum_weight_max_abs_delta": float(np.max(np.abs(candidate_table.plenum_segment_weights - baseline_table.plenum_segment_weights))),
        "plenum_weight_mean_abs_delta": float(np.mean(np.abs(candidate_table.plenum_segment_weights - baseline_table.plenum_segment_weights))),
        "axial_scale_max_abs_delta": float(np.max(np.abs(candidate_table.axial_scale_by_segment - baseline_table.axial_scale_by_segment))),
        "axial_scale_mean_abs_delta": float(np.mean(np.abs(candidate_table.axial_scale_by_segment - baseline_table.axial_scale_by_segment))),
        "radial_scale_max_abs_delta": float(np.max(np.abs(candidate_table.radial_scale_by_segment - baseline_table.radial_scale_by_segment))),
        "radial_scale_mean_abs_delta": float(np.mean(np.abs(candidate_table.radial_scale_by_segment - baseline_table.radial_scale_by_segment))),
        "tangential_scale_max_abs_delta": float(np.max(np.abs(candidate_table.tangential_scale_by_segment - baseline_table.tangential_scale_by_segment))),
        "tangential_scale_mean_abs_delta": float(np.mean(np.abs(candidate_table.tangential_scale_by_segment - baseline_table.tangential_scale_by_segment))),
    }


def _build_comparison_warnings(
    *,
    baseline_payload: dict,
    baseline_table: NominalEffectivenessTable,
    baseline_spec: GeometrySeedSpec | None,
    candidate_payload: dict,
    candidate_table: NominalEffectivenessTable,
    candidate_spec: GeometrySeedSpec | None,
    compatibility: dict,
    delta_summary: dict | None,
) -> list[str]:
    warnings: list[str] = []
    for label, spec, table in (
        ("baseline", baseline_spec, baseline_table),
        ("candidate", candidate_spec, candidate_table),
    ):
        provenance = str(spec.provenance if spec is not None else table.provenance).lower()
        if "provisional" in provenance:
            warnings.append(
                f"{label.title()} effectiveness source is still provisional, not CAD-, CFD-, or bench-validated hardware truth."
            )
    if not compatibility["comparable"]:
        warnings.append(
            "Baseline and candidate sources are not shape-compatible, so weight and scale deltas were not computed."
        )
        return warnings
    assert delta_summary is not None
    if all(abs(delta_summary[key]) <= 1e-12 for key in delta_summary):
        warnings.append("Candidate effectiveness source matches the baseline exactly within numerical tolerance.")
    if candidate_payload["source_path"] == baseline_payload["source_path"]:
        warnings.append("Baseline and candidate resolve to the same source path.")
    if (
        candidate_spec is not None
        and np.all(candidate_spec.plenum_half_span_deg <= 1e-9)
        and np.all(candidate_spec.plenum_sigma_deg <= 1e-9)
    ):
        warnings.append("Candidate nominal plenum coupling is still point-owned with no spillover.")
    if (
        np.allclose(candidate_table.axial_scale_by_segment, 1.0)
        and np.allclose(candidate_table.radial_scale_by_segment, 1.0)
        and np.allclose(candidate_table.tangential_scale_by_segment, 1.0)
    ):
        warnings.append("Candidate component scales are still uniform unity values.")
    return warnings


def build_effectiveness_comparison_report(
    *,
    candidate_spec_path: str | Path | None = None,
    candidate_table_path: str | Path | None = None,
    baseline_spec_path: str | Path | None = None,
    baseline_table_path: str | Path | None = None,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None, NominalEffectivenessTable, GeometrySeedSpec | None]:
    baseline_payload, baseline_table, baseline_spec = _resolve_effectiveness_source(
        spec_path=baseline_spec_path,
        table_path=baseline_table_path,
        default_to_geometry_seed=True,
    )
    candidate_payload, candidate_table, candidate_spec = _resolve_effectiveness_source(
        spec_path=candidate_spec_path,
        table_path=candidate_table_path,
        default_to_geometry_seed=False,
    )
    compatibility = {
        "segment_count_match": int(baseline_table.segment_count) == int(candidate_table.segment_count),
        "fan_count_match": int(baseline_table.fan_count) == int(candidate_table.fan_count),
        "plenum_count_match": int(baseline_table.plenum_count) == int(candidate_table.plenum_count),
    }
    compatibility["comparable"] = all(compatibility.values())
    delta_summary = _delta_summary(baseline_table, candidate_table) if compatibility["comparable"] else None
    report = {
        "baseline": baseline_payload,
        "candidate": candidate_payload,
        "compatibility": compatibility,
        "delta_summary": delta_summary,
        "warnings": _build_comparison_warnings(
            baseline_payload=baseline_payload,
            baseline_table=baseline_table,
            baseline_spec=baseline_spec,
            candidate_payload=candidate_payload,
            candidate_table=candidate_table,
            candidate_spec=candidate_spec,
            compatibility=compatibility,
            delta_summary=delta_summary,
        ),
    }
    return report, baseline_table, baseline_spec, candidate_table, candidate_spec


def _is_placeholder_text(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    markers = ("todo", "replace", "template", "placeholder", "tbd")
    return any(marker in text for marker in markers)


def build_effectiveness_validation_report(
    *,
    candidate_spec_path: str | Path | None = None,
    candidate_table_path: str | Path | None = None,
    baseline_spec_path: str | Path | None = None,
    baseline_table_path: str | Path | None = None,
    delta_tolerance: float = 1e-9,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None, NominalEffectivenessTable, GeometrySeedSpec | None]:
    if delta_tolerance < 0.0:
        raise ValueError("delta_tolerance must be non-negative")
    baseline_payload, baseline_table, baseline_spec = _resolve_effectiveness_source(
        spec_path=baseline_spec_path,
        table_path=baseline_table_path,
        default_to_geometry_seed=True,
    )
    candidate_payload, candidate_table, candidate_spec = _resolve_effectiveness_source(
        spec_path=candidate_spec_path,
        table_path=candidate_table_path,
        default_to_geometry_seed=False,
    )
    compatibility = {
        "segment_count_match": int(baseline_table.segment_count) == int(candidate_table.segment_count),
        "fan_count_match": int(baseline_table.fan_count) == int(candidate_table.fan_count),
        "plenum_count_match": int(baseline_table.plenum_count) == int(candidate_table.plenum_count),
    }
    compatibility["comparable"] = all(compatibility.values())
    delta_summary = _delta_summary(baseline_table, candidate_table) if compatibility["comparable"] else None

    if candidate_spec is not None:
        candidate_identity = str(candidate_spec.spec_name)
        candidate_provenance = str(candidate_spec.provenance)
    else:
        candidate_identity = str(candidate_payload["table_summary"]["table_name"])
        candidate_provenance = str(candidate_table.provenance)

    blocking_issues: list[str] = []
    advisory_issues: list[str] = []
    passed_checks: list[str] = []

    if _is_placeholder_text(candidate_identity):
        blocking_issues.append("Candidate identity still looks like a template or placeholder name.")
    else:
        passed_checks.append("Candidate identity is not obviously template-like.")

    if _is_placeholder_text(candidate_provenance):
        blocking_issues.append("Candidate provenance still contains placeholder language or missing detail.")
    else:
        passed_checks.append("Candidate provenance is populated beyond template placeholders.")

    if not compatibility["comparable"]:
        blocking_issues.append("Candidate is not shape-compatible with the current baseline.")
    else:
        passed_checks.append("Candidate is shape-compatible with the current baseline.")
        assert delta_summary is not None
        if all(abs(value) <= delta_tolerance for value in delta_summary.values()):
            blocking_issues.append("Candidate does not differ from the current baseline within the configured tolerance.")
        else:
            passed_checks.append("Candidate differs from the current baseline beyond the configured tolerance.")

    if (
        np.allclose(candidate_table.axial_scale_by_segment, 1.0)
        and np.allclose(candidate_table.radial_scale_by_segment, 1.0)
        and np.allclose(candidate_table.tangential_scale_by_segment, 1.0)
    ):
        advisory_issues.append("Candidate component scales are still uniform unity values.")
    else:
        passed_checks.append("Candidate includes non-unity component-scale information.")

    if candidate_spec is not None and np.all(candidate_spec.plenum_half_span_deg <= 1e-9) and np.all(candidate_spec.plenum_sigma_deg <= 1e-9):
        advisory_issues.append("Candidate nominal plenum coupling is still point-owned with no spillover.")
    elif candidate_spec is not None:
        passed_checks.append("Candidate includes non-trivial plenum coupling assumptions.")

    if blocking_issues:
        status = "risk"
    elif advisory_issues:
        status = "caution"
    else:
        status = "pass"

    report = {
        "status": status,
        "delta_tolerance": float(delta_tolerance),
        "baseline": baseline_payload,
        "candidate": candidate_payload,
        "candidate_identity": candidate_identity,
        "candidate_provenance": candidate_provenance,
        "compatibility": compatibility,
        "delta_summary": delta_summary,
        "blocking_issues": blocking_issues,
        "advisory_issues": advisory_issues,
        "passed_checks": passed_checks,
        "warnings": list(blocking_issues) + list(advisory_issues),
    }
    return report, baseline_table, baseline_spec, candidate_table, candidate_spec

def infer_effectiveness_summary_format(path: str, format_name: str) -> str:
    if format_name != "auto":
        return format_name
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "text"


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _render_mapping_section(title: str, mapping: dict, *, format_name: str) -> str:
    rows = [[key, _stringify(value)] for key, value in mapping.items()]
    if format_name == "markdown":
        lines = [f"## {title}", "", "| metric | value |", "| --- | --- |"]
        lines.extend(f"| {key} | {value} |" for key, value in rows)
        lines.append("")
        return "\n".join(lines)
    widths = [
        max(len("metric"), *(len(key) for key, _ in rows)),
        max(len("value"), *(len(value) for _, value in rows)),
    ]
    lines = [
        title,
        f"{'metric'.ljust(widths[0])}  {'value'.ljust(widths[1])}",
        f"{'-' * widths[0]}  {'-' * widths[1]}",
    ]
    lines.extend(f"{key.ljust(widths[0])}  {value.ljust(widths[1])}" for key, value in rows)
    lines.append("")
    return "\n".join(lines)


def render_effectiveness_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    source_mapping = {
        "source_kind": report.get("source_kind"),
        "source_path": report.get("source_path"),
    }
    sections.append(_render_mapping_section("Source", source_mapping, format_name=format_name))
    spec_summary = report.get("spec_summary")
    if isinstance(spec_summary, dict):
        sections.append(_render_mapping_section("Geometry Seed Summary", spec_summary, format_name=format_name))
    sections.append(_render_mapping_section("Effectiveness Table Summary", report.get("table_summary", {}), format_name=format_name))
    sections.append(_render_mapping_section("Fan Weight Summary", report.get("fan_weight_summary", {}), format_name=format_name))
    sections.append(_render_mapping_section("Plenum Weight Summary", report.get("plenum_weight_summary", {}), format_name=format_name))
    component_summary = report.get("component_scale_summary", {})
    for name in ("axial", "radial", "tangential"):
        if isinstance(component_summary.get(name), dict):
            sections.append(_render_mapping_section(f"{name.title()} Scale Summary", component_summary[name], format_name=format_name))
    warnings = list(report.get("warnings", []))
    if format_name == "markdown":
        lines = ["# effectiveness report", ""]
        lines.extend(section for section in sections if section)
        lines.append("## Warnings")
        lines.append("")
        if warnings:
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines = ["effectiveness report", ""]
    lines.extend(section for section in sections if section)
    lines.append("Warnings")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def render_effectiveness_comparison_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    baseline = dict(report.get("baseline", {}))
    candidate = dict(report.get("candidate", {}))
    sections.append(_render_mapping_section(
        "Baseline Source",
        {"source_kind": baseline.get("source_kind"), "source_path": baseline.get("source_path")},
        format_name=format_name,
    ))
    if isinstance(baseline.get("spec_summary"), dict):
        sections.append(_render_mapping_section("Baseline Geometry Seed Summary", baseline["spec_summary"], format_name=format_name))
    sections.append(_render_mapping_section("Baseline Table Summary", baseline.get("table_summary", {}), format_name=format_name))
    sections.append(_render_mapping_section(
        "Candidate Source",
        {"source_kind": candidate.get("source_kind"), "source_path": candidate.get("source_path")},
        format_name=format_name,
    ))
    if isinstance(candidate.get("spec_summary"), dict):
        sections.append(_render_mapping_section("Candidate Geometry Seed Summary", candidate["spec_summary"], format_name=format_name))
    sections.append(_render_mapping_section("Candidate Table Summary", candidate.get("table_summary", {}), format_name=format_name))
    sections.append(_render_mapping_section("Compatibility", report.get("compatibility", {}), format_name=format_name))
    if isinstance(report.get("delta_summary"), dict):
        sections.append(_render_mapping_section("Delta Summary", report["delta_summary"], format_name=format_name))
    warnings = list(report.get("warnings", []))
    title = "# effectiveness comparison" if format_name == "markdown" else "effectiveness comparison"
    lines = [title, ""]
    lines.extend(section for section in sections if section)
    warning_title = "## Warnings" if format_name == "markdown" else "Warnings"
    lines.append(warning_title)
    lines.append("")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_effectiveness_report_outputs(
    report: dict,
    table: NominalEffectivenessTable,
    *,
    spec: GeometrySeedSpec | None = None,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
    table_out: str = "",
    source_out: str = "",
) -> dict:
    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table_payload = effectiveness_table_to_payload(table)
    source_payload = geometry_seed_spec_to_payload(spec) if spec is not None else table_payload

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update(
            {
                "summary_json": str(out_path / "summary.json"),
                "summary_markdown": str(out_path / "summary.md"),
                "materialized_table": str(out_path / "materialized_table.json"),
                "source_payload": str(out_path / ("source_spec.json" if spec is not None else "source_table.json")),
            }
        )

    if table_out:
        artifacts["materialized_table"] = str(Path(table_out))
    if source_out:
        artifacts["source_payload"] = str(Path(source_out))
    if summary_out:
        artifacts["summary_custom"] = str(Path(summary_out))

    if artifacts:
        updated["artifacts"] = artifacts

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(markdown + ("" if markdown.endswith("\n") else "\n"), encoding="utf-8")
        write_json(Path(artifacts["materialized_table"]), table_payload)
        write_json(Path(artifacts["source_payload"]), source_payload)

    if table_out:
        write_json(Path(table_out), table_payload)
    if source_out:
        write_json(Path(source_out), source_payload)
    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
        updated["summary_format"] = resolved_format

    return updated


def write_effectiveness_comparison_outputs(
    report: dict,
    baseline_table: NominalEffectivenessTable,
    candidate_table: NominalEffectivenessTable,
    *,
    baseline_spec: GeometrySeedSpec | None = None,
    candidate_spec: GeometrySeedSpec | None = None,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
    baseline_table_out: str = "",
    candidate_table_out: str = "",
    baseline_source_out: str = "",
    candidate_source_out: str = "",
) -> dict:
    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    baseline_table_payload = effectiveness_table_to_payload(baseline_table)
    candidate_table_payload = effectiveness_table_to_payload(candidate_table)
    baseline_source_payload = geometry_seed_spec_to_payload(baseline_spec) if baseline_spec is not None else baseline_table_payload
    candidate_source_payload = geometry_seed_spec_to_payload(candidate_spec) if candidate_spec is not None else candidate_table_payload

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update(
            {
                "summary_json": str(out_path / "summary.json"),
                "summary_markdown": str(out_path / "summary.md"),
                "baseline_table": str(out_path / "baseline_table.json"),
                "candidate_table": str(out_path / "candidate_table.json"),
                "baseline_source": str(out_path / ("baseline_spec.json" if baseline_spec is not None else "baseline_source_table.json")),
                "candidate_source": str(out_path / ("candidate_spec.json" if candidate_spec is not None else "candidate_source_table.json")),
            }
        )

    if baseline_table_out:
        artifacts["baseline_table"] = str(Path(baseline_table_out))
    if candidate_table_out:
        artifacts["candidate_table"] = str(Path(candidate_table_out))
    if baseline_source_out:
        artifacts["baseline_source"] = str(Path(baseline_source_out))
    if candidate_source_out:
        artifacts["candidate_source"] = str(Path(candidate_source_out))
    if summary_out:
        artifacts["summary_custom"] = str(Path(summary_out))

    if artifacts:
        updated["artifacts"] = artifacts

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_comparison_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(markdown + ("" if markdown.endswith("\n") else "\n"), encoding="utf-8")
        write_json(Path(artifacts["baseline_table"]), baseline_table_payload)
        write_json(Path(artifacts["candidate_table"]), candidate_table_payload)
        write_json(Path(artifacts["baseline_source"]), baseline_source_payload)
        write_json(Path(artifacts["candidate_source"]), candidate_source_payload)

    if baseline_table_out:
        write_json(Path(baseline_table_out), baseline_table_payload)
    if candidate_table_out:
        write_json(Path(candidate_table_out), candidate_table_payload)
    if baseline_source_out:
        write_json(Path(baseline_source_out), baseline_source_payload)
    if candidate_source_out:
        write_json(Path(candidate_source_out), candidate_source_payload)
    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_comparison_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
        updated["summary_format"] = resolved_format

    return updated

def render_effectiveness_validation_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    sections.append(_render_mapping_section("Validation Result", {
        "status": report.get("status"),
        "delta_tolerance": report.get("delta_tolerance"),
        "candidate_identity": report.get("candidate_identity"),
    }, format_name=format_name))
    baseline = dict(report.get("baseline", {}))
    candidate = dict(report.get("candidate", {}))
    sections.append(_render_mapping_section("Baseline Source", {
        "source_kind": baseline.get("source_kind"),
        "source_path": baseline.get("source_path"),
    }, format_name=format_name))
    sections.append(_render_mapping_section("Candidate Source", {
        "source_kind": candidate.get("source_kind"),
        "source_path": candidate.get("source_path"),
    }, format_name=format_name))
    sections.append(_render_mapping_section("Compatibility", report.get("compatibility", {}), format_name=format_name))
    if isinstance(report.get("delta_summary"), dict):
        sections.append(_render_mapping_section("Delta Summary", report["delta_summary"], format_name=format_name))
    if format_name == "markdown":
        lines = ["# effectiveness validation", ""]
        lines.extend(section for section in sections if section)
        lines.append("## Blocking Issues")
        lines.append("")
        blocking = list(report.get("blocking_issues", []))
        lines.extend(f"- {item}" for item in blocking) if blocking else lines.append("- none")
        lines.append("")
        lines.append("## Advisory Issues")
        lines.append("")
        advisory = list(report.get("advisory_issues", []))
        lines.extend(f"- {item}" for item in advisory) if advisory else lines.append("- none")
        lines.append("")
        lines.append("## Passed Checks")
        lines.append("")
        passed = list(report.get("passed_checks", []))
        lines.extend(f"- {item}" for item in passed) if passed else lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines = ["effectiveness validation", ""]
    lines.extend(section for section in sections if section)
    for title, key in (("Blocking Issues", "blocking_issues"), ("Advisory Issues", "advisory_issues"), ("Passed Checks", "passed_checks")):
        lines.append(title)
        items = list(report.get(key, []))
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def write_effectiveness_validation_outputs(
    report: dict,
    baseline_table: NominalEffectivenessTable,
    candidate_table: NominalEffectivenessTable,
    *,
    baseline_spec: GeometrySeedSpec | None = None,
    candidate_spec: GeometrySeedSpec | None = None,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
) -> dict:
    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    baseline_table_payload = effectiveness_table_to_payload(baseline_table)
    candidate_table_payload = effectiveness_table_to_payload(candidate_table)
    baseline_source_payload = geometry_seed_spec_to_payload(baseline_spec) if baseline_spec is not None else baseline_table_payload
    candidate_source_payload = geometry_seed_spec_to_payload(candidate_spec) if candidate_spec is not None else candidate_table_payload

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update({
            "summary_json": str(out_path / "summary.json"),
            "summary_markdown": str(out_path / "summary.md"),
            "baseline_table": str(out_path / "baseline_table.json"),
            "candidate_table": str(out_path / "candidate_table.json"),
            "baseline_source": str(out_path / ("baseline_spec.json" if baseline_spec is not None else "baseline_source_table.json")),
            "candidate_source": str(out_path / ("candidate_spec.json" if candidate_spec is not None else "candidate_source_table.json")),
        })

    if artifacts:
        updated["artifacts"] = artifacts

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_validation_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(
            markdown + ("" if markdown.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        write_json(Path(artifacts["baseline_table"]), baseline_table_payload)
        write_json(Path(artifacts["candidate_table"]), candidate_table_payload)
        write_json(Path(artifacts["baseline_source"]), baseline_source_payload)
        write_json(Path(artifacts["candidate_source"]), candidate_source_payload)

    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_validation_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        updated["summary_format"] = resolved_format

    return updated





def _parse_candidate_provenance_note(note_path: str | Path | None) -> dict | None:
    if not note_path:
        return None
    resolved = Path(note_path).resolve()
    text = resolved.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        normalized_key = str(key).strip()
        if normalized_key.lower() == "todo":
            continue
        fields[normalized_key] = str(value).strip().strip("`")
    return {
        "note_path": _relative_repo_path(resolved),
        "fields": fields,
        "raw_text": text,
    }


def _recognized_source_type(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"cad", "cfd", "bench", "mixed"}


def _is_material_delta_significant(delta_summary: dict | None, tolerance: float) -> bool:
    if delta_summary is None:
        return False
    return any(abs(float(value)) > tolerance for value in delta_summary.values())


def build_effectiveness_adoption_report(
    *,
    candidate_spec_path: str | Path | None = None,
    candidate_table_path: str | Path | None = None,
    candidate_note_path: str | Path | None = None,
    baseline_spec_path: str | Path | None = None,
    baseline_table_path: str | Path | None = None,
    delta_tolerance: float = 1e-9,
    material_delta_tolerance: float = 1e-6,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None, NominalEffectivenessTable, GeometrySeedSpec | None, dict | None]:
    if material_delta_tolerance < 0.0:
        raise ValueError("material_delta_tolerance must be non-negative")

    validation_report, baseline_table, baseline_spec, candidate_table, candidate_spec = build_effectiveness_validation_report(
        candidate_spec_path=candidate_spec_path,
        candidate_table_path=candidate_table_path,
        baseline_spec_path=baseline_spec_path,
        baseline_table_path=baseline_table_path,
        delta_tolerance=delta_tolerance,
    )
    candidate_note = _parse_candidate_provenance_note(candidate_note_path)

    blocking_issues: list[str] = []
    review_notes: list[str] = []
    passed_checks: list[str] = []

    if validation_report["status"] == "risk":
        blocking_issues.append("Candidate failed the validation gate and should not be adopted.")
    elif validation_report["status"] == "caution":
        review_notes.append("Candidate passed validation with advisory issues that still require review.")
    else:
        passed_checks.append("Candidate passed the validation gate without advisory issues.")

    note_summary = None
    if candidate_note is None:
        blocking_issues.append("Candidate provenance note is required for adoption decisions.")
    else:
        note_fields = dict(candidate_note.get("fields", {}))
        note_summary = {
            "note_path": candidate_note["note_path"],
            "fields": note_fields,
        }
        passed_checks.append("Candidate provenance note provided.")

        required_fields = (
            "source_type",
            "source_reference",
            "extraction_method",
            "source_revision",
            "validation_state",
            "reviewer",
        )
        missing_fields = [field for field in required_fields if _is_placeholder_text(note_fields.get(field))]
        if missing_fields:
            blocking_issues.append(
                "Candidate provenance note still has placeholder or missing fields: " + ", ".join(missing_fields) + "."
            )
        else:
            passed_checks.append("Candidate provenance note includes the required evidence fields.")

        note_spec_name = note_fields.get("spec_name")
        if not _is_placeholder_text(note_spec_name):
            if str(note_spec_name).strip() == str(validation_report["candidate_identity"]).strip():
                passed_checks.append("Candidate provenance note identity matches the candidate identity.")
            else:
                review_notes.append("Candidate provenance note spec_name does not match the candidate identity.")

        note_source_type = note_fields.get("source_type")
        if not _is_placeholder_text(note_source_type):
            if _recognized_source_type(note_source_type):
                passed_checks.append("Candidate provenance note uses a recognized source_type.")
            else:
                review_notes.append("Candidate provenance note source_type is not one of CAD / CFD / bench / mixed.")

        validation_state = str(note_fields.get("validation_state", "")).strip().lower()
        if validation_state in {"reviewed", "accepted"}:
            passed_checks.append("Candidate provenance note records a reviewed or accepted validation state.")
        elif validation_state and not _is_placeholder_text(validation_state):
            review_notes.append(f"Candidate validation_state is '{validation_state}', not reviewed/accepted yet.")

    material_delta_significant = _is_material_delta_significant(
        validation_report.get("delta_summary"),
        material_delta_tolerance,
    )
    if validation_report["compatibility"]["comparable"]:
        if material_delta_significant:
            passed_checks.append("Candidate differs materially from the current baseline beyond the adoption tolerance.")
        else:
            review_notes.append("Candidate does not materially differ from the current baseline beyond the adoption tolerance.")

    if blocking_issues:
        adoption_status = "rejected"
    elif review_notes:
        adoption_status = "needs-review"
    else:
        adoption_status = "adoptable"

    report = {
        "adoption_status": adoption_status,
        "validation_status": validation_report["status"],
        "delta_tolerance": float(delta_tolerance),
        "material_delta_tolerance": float(material_delta_tolerance),
        "material_delta_significant": bool(material_delta_significant),
        "baseline": validation_report["baseline"],
        "candidate": validation_report["candidate"],
        "candidate_identity": validation_report["candidate_identity"],
        "candidate_provenance": validation_report["candidate_provenance"],
        "candidate_note": note_summary,
        "compatibility": validation_report["compatibility"],
        "delta_summary": validation_report["delta_summary"],
        "validation_blocking_issues": list(validation_report.get("blocking_issues", [])),
        "validation_advisory_issues": list(validation_report.get("advisory_issues", [])),
        "blocking_issues": blocking_issues,
        "review_notes": review_notes,
        "passed_checks": passed_checks,
        "warnings": list(blocking_issues) + list(review_notes),
    }
    return report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note


def render_effectiveness_adoption_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    sections.append(_render_mapping_section("Adoption Result", {
        "adoption_status": report.get("adoption_status"),
        "validation_status": report.get("validation_status"),
        "material_delta_tolerance": report.get("material_delta_tolerance"),
        "material_delta_significant": report.get("material_delta_significant"),
        "candidate_identity": report.get("candidate_identity"),
    }, format_name=format_name))
    baseline = dict(report.get("baseline", {}))
    candidate = dict(report.get("candidate", {}))
    sections.append(_render_mapping_section("Baseline Source", {
        "source_kind": baseline.get("source_kind"),
        "source_path": baseline.get("source_path"),
    }, format_name=format_name))
    sections.append(_render_mapping_section("Candidate Source", {
        "source_kind": candidate.get("source_kind"),
        "source_path": candidate.get("source_path"),
    }, format_name=format_name))
    note = report.get("candidate_note")
    if isinstance(note, dict):
        note_fields = dict(note.get("fields", {}))
        sections.append(_render_mapping_section("Candidate Note", {
            "note_path": note.get("note_path"),
            "source_type": note_fields.get("source_type"),
            "validation_state": note_fields.get("validation_state"),
            "reviewer": note_fields.get("reviewer"),
        }, format_name=format_name))
    sections.append(_render_mapping_section("Compatibility", report.get("compatibility", {}), format_name=format_name))
    if isinstance(report.get("delta_summary"), dict):
        sections.append(_render_mapping_section("Delta Summary", report["delta_summary"], format_name=format_name))
    if format_name == "markdown":
        lines = ["# effectiveness adoption assessment", ""]
        lines.extend(section for section in sections if section)
        lines.append("## Blocking Issues")
        lines.append("")
        blocking = list(report.get("blocking_issues", []))
        lines.extend(f"- {item}" for item in blocking) if blocking else lines.append("- none")
        lines.append("")
        lines.append("## Review Notes")
        lines.append("")
        review_notes = list(report.get("review_notes", []))
        lines.extend(f"- {item}" for item in review_notes) if review_notes else lines.append("- none")
        lines.append("")
        lines.append("## Passed Checks")
        lines.append("")
        passed = list(report.get("passed_checks", []))
        lines.extend(f"- {item}" for item in passed) if passed else lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines = ["effectiveness adoption assessment", ""]
    lines.extend(section for section in sections if section)
    for title, key in (("Blocking Issues", "blocking_issues"), ("Review Notes", "review_notes"), ("Passed Checks", "passed_checks")):
        lines.append(title)
        items = list(report.get(key, []))
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def write_effectiveness_adoption_outputs(
    report: dict,
    baseline_table: NominalEffectivenessTable,
    candidate_table: NominalEffectivenessTable,
    *,
    baseline_spec: GeometrySeedSpec | None = None,
    candidate_spec: GeometrySeedSpec | None = None,
    candidate_note: dict | None = None,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
) -> dict:
    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    baseline_table_payload = effectiveness_table_to_payload(baseline_table)
    candidate_table_payload = effectiveness_table_to_payload(candidate_table)
    baseline_source_payload = geometry_seed_spec_to_payload(baseline_spec) if baseline_spec is not None else baseline_table_payload
    candidate_source_payload = geometry_seed_spec_to_payload(candidate_spec) if candidate_spec is not None else candidate_table_payload

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update({
            "summary_json": str(out_path / "summary.json"),
            "summary_markdown": str(out_path / "summary.md"),
            "baseline_table": str(out_path / "baseline_table.json"),
            "candidate_table": str(out_path / "candidate_table.json"),
            "baseline_source": str(out_path / ("baseline_spec.json" if baseline_spec is not None else "baseline_source_table.json")),
            "candidate_source": str(out_path / ("candidate_spec.json" if candidate_spec is not None else "candidate_source_table.json")),
        })
        if candidate_note is not None:
            artifacts["candidate_note_fields"] = str(out_path / "candidate_note_fields.json")
            artifacts["candidate_note_copy"] = str(out_path / "candidate_note.md")

    if artifacts:
        updated["artifacts"] = artifacts

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_adoption_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(
            markdown + ("" if markdown.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        write_json(Path(artifacts["baseline_table"]), baseline_table_payload)
        write_json(Path(artifacts["candidate_table"]), candidate_table_payload)
        write_json(Path(artifacts["baseline_source"]), baseline_source_payload)
        write_json(Path(artifacts["candidate_source"]), candidate_source_payload)
        if candidate_note is not None:
            write_json(Path(artifacts["candidate_note_fields"]), dict(candidate_note.get("fields", {})))
            note_text = str(candidate_note.get("raw_text", ""))
            Path(artifacts["candidate_note_copy"]).write_text(
                note_text + ("" if note_text.endswith("\n") else "\n"),
                encoding="utf-8",
            )

    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_adoption_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        updated["summary_format"] = resolved_format

    return updated



def build_effectiveness_promotion_report(
    *,
    candidate_spec_path: str | Path | None = None,
    candidate_table_path: str | Path | None = None,
    candidate_note_path: str | Path | None = None,
    baseline_spec_path: str | Path | None = None,
    baseline_table_path: str | Path | None = None,
    delta_tolerance: float = 1e-9,
    material_delta_tolerance: float = 1e-6,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None, NominalEffectivenessTable, GeometrySeedSpec | None, dict | None]:
    adoption_report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_adoption_report(
        candidate_spec_path=candidate_spec_path,
        candidate_table_path=candidate_table_path,
        candidate_note_path=candidate_note_path,
        baseline_spec_path=baseline_spec_path,
        baseline_table_path=baseline_table_path,
        delta_tolerance=delta_tolerance,
        material_delta_tolerance=material_delta_tolerance,
    )

    blocking_issues: list[str] = []
    review_notes: list[str] = []
    passed_checks: list[str] = []

    if adoption_report["adoption_status"] != "adoptable":
        blocking_issues.append(
            f"Candidate adoption status is {adoption_report['adoption_status']}, so promotion is blocked."
        )
    else:
        passed_checks.append("Candidate is adoptable according to the adoption gate.")

    baseline_kind = str(adoption_report["baseline"].get("source_kind") or "")
    candidate_kind = str(adoption_report["candidate"].get("source_kind") or "")
    target_path = str(adoption_report["baseline"].get("source_path") or "")
    if baseline_kind != candidate_kind:
        blocking_issues.append(
            "Candidate source kind does not match the baseline target kind; stage promotion only after choosing a compatible baseline target."
        )
    else:
        passed_checks.append("Candidate source kind matches the baseline target kind.")

    if not target_path:
        blocking_issues.append("Baseline target path is not available, so promotion cannot be staged.")
        staged_replacement_basename = ""
    else:
        staged_replacement_basename = Path(target_path).name
        passed_checks.append("Baseline target path resolved for staged replacement.")

    if candidate_note is None:
        review_notes.append("Promotion pack will not include a candidate note copy because no note was supplied.")
    else:
        passed_checks.append("Candidate note is available for the promotion pack.")

    if blocking_issues:
        promotion_status = "blocked"
    else:
        promotion_status = "staged"

    report = {
        "promotion_status": promotion_status,
        "adoption_status": adoption_report["adoption_status"],
        "target_kind": baseline_kind,
        "target_path": target_path,
        "staged_replacement_basename": staged_replacement_basename,
        "delta_tolerance": float(delta_tolerance),
        "material_delta_tolerance": float(material_delta_tolerance),
        "baseline": adoption_report["baseline"],
        "candidate": adoption_report["candidate"],
        "candidate_identity": adoption_report["candidate_identity"],
        "candidate_provenance": adoption_report["candidate_provenance"],
        "candidate_note": adoption_report.get("candidate_note"),
        "compatibility": adoption_report["compatibility"],
        "delta_summary": adoption_report["delta_summary"],
        "adoption_blocking_issues": list(adoption_report.get("blocking_issues", [])),
        "adoption_review_notes": list(adoption_report.get("review_notes", [])),
        "blocking_issues": blocking_issues,
        "review_notes": review_notes,
        "passed_checks": passed_checks,
        "warnings": list(blocking_issues) + list(review_notes),
    }
    return report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note


def render_effectiveness_promotion_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    sections.append(_render_mapping_section("Promotion Result", {
        "promotion_status": report.get("promotion_status"),
        "adoption_status": report.get("adoption_status"),
        "target_kind": report.get("target_kind"),
        "target_path": report.get("target_path"),
        "staged_replacement_basename": report.get("staged_replacement_basename"),
        "candidate_identity": report.get("candidate_identity"),
    }, format_name=format_name))
    baseline = dict(report.get("baseline", {}))
    candidate = dict(report.get("candidate", {}))
    sections.append(_render_mapping_section("Baseline Source", {
        "source_kind": baseline.get("source_kind"),
        "source_path": baseline.get("source_path"),
    }, format_name=format_name))
    sections.append(_render_mapping_section("Candidate Source", {
        "source_kind": candidate.get("source_kind"),
        "source_path": candidate.get("source_path"),
    }, format_name=format_name))
    note = report.get("candidate_note")
    if isinstance(note, dict):
        note_fields = dict(note.get("fields", {}))
        sections.append(_render_mapping_section("Candidate Note", {
            "note_path": note.get("note_path"),
            "source_type": note_fields.get("source_type"),
            "validation_state": note_fields.get("validation_state"),
            "reviewer": note_fields.get("reviewer"),
        }, format_name=format_name))
    if isinstance(report.get("delta_summary"), dict):
        sections.append(_render_mapping_section("Delta Summary", report["delta_summary"], format_name=format_name))
    if format_name == "markdown":
        lines = ["# effectiveness promotion assessment", ""]
        lines.extend(section for section in sections if section)
        lines.append("## Blocking Issues")
        lines.append("")
        blocking = list(report.get("blocking_issues", []))
        lines.extend(f"- {item}" for item in blocking) if blocking else lines.append("- none")
        lines.append("")
        lines.append("## Review Notes")
        lines.append("")
        review_notes = list(report.get("review_notes", []))
        lines.extend(f"- {item}" for item in review_notes) if review_notes else lines.append("- none")
        lines.append("")
        lines.append("## Passed Checks")
        lines.append("")
        passed = list(report.get("passed_checks", []))
        lines.extend(f"- {item}" for item in passed) if passed else lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines = ["effectiveness promotion assessment", ""]
    lines.extend(section for section in sections if section)
    for title, key in (("Blocking Issues", "blocking_issues"), ("Review Notes", "review_notes"), ("Passed Checks", "passed_checks")):
        lines.append(title)
        items = list(report.get(key, []))
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def write_effectiveness_promotion_outputs(
    report: dict,
    baseline_table: NominalEffectivenessTable,
    candidate_table: NominalEffectivenessTable,
    *,
    baseline_spec: GeometrySeedSpec | None = None,
    candidate_spec: GeometrySeedSpec | None = None,
    candidate_note: dict | None = None,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
) -> dict:
    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    baseline_source_payload = geometry_seed_spec_to_payload(baseline_spec) if baseline_spec is not None else effectiveness_table_to_payload(baseline_table)
    candidate_source_payload = geometry_seed_spec_to_payload(candidate_spec) if candidate_spec is not None else effectiveness_table_to_payload(candidate_table)
    staged_payload = candidate_source_payload

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update({
            "summary_json": str(out_path / "summary.json"),
            "summary_markdown": str(out_path / "summary.md"),
            "promotion_manifest": str(out_path / "promotion_manifest.json"),
            "baseline_source": str(out_path / ("baseline_spec.json" if baseline_spec is not None else "baseline_source_table.json")),
            "candidate_source": str(out_path / ("candidate_spec.json" if candidate_spec is not None else "candidate_table.json")),
        })
        if candidate_note is not None:
            artifacts["candidate_note_copy"] = str(out_path / "candidate_note.md")
        staged_name = str(report.get("staged_replacement_basename") or "")
        if updated.get("promotion_status") == "staged" and staged_name:
            artifacts["staged_replacement"] = str(out_path / "staged" / staged_name)

    if artifacts:
        updated["artifacts"] = artifacts

    manifest = {
        "promotion_status": updated.get("promotion_status"),
        "adoption_status": updated.get("adoption_status"),
        "target_kind": updated.get("target_kind"),
        "target_path": updated.get("target_path"),
        "staged_replacement": artifacts.get("staged_replacement"),
        "candidate_identity": updated.get("candidate_identity"),
        "candidate_source_path": updated.get("candidate", {}).get("source_path"),
        "candidate_note_path": updated.get("candidate_note", {}).get("note_path") if isinstance(updated.get("candidate_note"), dict) else None,
    }

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_promotion_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(
            markdown + ("" if markdown.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        write_json(Path(artifacts["promotion_manifest"]), manifest)
        write_json(Path(artifacts["baseline_source"]), baseline_source_payload)
        write_json(Path(artifacts["candidate_source"]), candidate_source_payload)
        if candidate_note is not None:
            note_text = str(candidate_note.get("raw_text", ""))
            Path(artifacts["candidate_note_copy"]).write_text(
                note_text + ("" if note_text.endswith("\n") else "\n"),
                encoding="utf-8",
            )
        if updated.get("promotion_status") == "staged" and artifacts.get("staged_replacement"):
            write_json(Path(artifacts["staged_replacement"]), staged_payload)

    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_promotion_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        updated["summary_format"] = resolved_format

    return updated


def build_effectiveness_switch_report(
    *,
    promotion_manifest_path: str | Path | None = None,
    promotion_dir: str | Path | None = None,
    target_path_override: str | Path | None = None,
) -> tuple[dict, dict]:
    if bool(promotion_manifest_path) == bool(promotion_dir):
        raise ValueError("Provide exactly one of --promotion-manifest or --promotion-dir.")

    manifest_input = Path(promotion_manifest_path) if promotion_manifest_path else Path(promotion_dir) / "promotion_manifest.json"
    manifest_resolved = _resolve_repo_input_path(manifest_input)
    if not manifest_resolved.exists():
        raise ValueError(f"Promotion manifest not found: {manifest_input}")

    manifest = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    blocking_issues: list[str] = []
    review_notes: list[str] = []
    passed_checks: list[str] = []

    promotion_status = str(manifest.get("promotion_status") or "")
    if promotion_status != "staged":
        blocking_issues.append(
            f"Promotion manifest status is {promotion_status or 'unknown'}, so the baseline switch is blocked."
        )
    else:
        passed_checks.append("Promotion manifest is staged and eligible for baseline switching.")

    target_kind = str(manifest.get("target_kind") or "")
    if target_kind not in {"table", "geometry-seed"}:
        blocking_issues.append(
            "Promotion target kind is not supported for the baseline switch workflow."
        )
    else:
        passed_checks.append("Promotion target kind is supported for baseline switching.")

    manifest_target_value = str(manifest.get("target_path") or "")
    target_input: Path | None = None
    target_resolved: Path | None = None
    if target_path_override:
        target_input = Path(target_path_override)
        review_notes.append("Target path override is active for this baseline switch assessment.")
    elif manifest_target_value:
        target_input = Path(manifest_target_value)
    else:
        blocking_issues.append("Promotion manifest does not include a target path for the baseline switch.")

    if target_input is not None:
        target_resolved = _resolve_repo_input_path(target_input)
        if not target_resolved.exists():
            blocking_issues.append(
                f"Target baseline path does not exist: {_relative_repo_path(target_resolved)}"
            )
        else:
            passed_checks.append("Target baseline path exists and can be backed up before switching.")

    staged_value = str(manifest.get("staged_replacement") or "")
    staged_resolved: Path | None = None
    if not staged_value:
        blocking_issues.append("Promotion manifest does not include a staged replacement path.")
    else:
        staged_resolved = _resolve_repo_input_path(staged_value)
        if not staged_resolved.exists():
            blocking_issues.append(
                f"Staged replacement path does not exist: {_relative_repo_path(staged_resolved)}"
            )
        else:
            passed_checks.append("Staged replacement artifact exists and is ready to apply.")

    candidate_note_value = str(manifest.get("candidate_note_path") or "")
    candidate_note_resolved: Path | None = None
    if candidate_note_value:
        candidate_note_resolved = _resolve_repo_input_path(candidate_note_value)
        if candidate_note_resolved.exists():
            passed_checks.append("Candidate note is available for the switch pack provenance trail.")
        else:
            review_notes.append(
                "Candidate note path from the promotion manifest does not exist; the switch pack will omit the copied note."
            )
    else:
        review_notes.append(
            "Promotion manifest does not include a candidate note path; the switch pack will omit the copied note."
        )

    if target_resolved is not None and staged_resolved is not None:
        if target_resolved.suffix.lower() != staged_resolved.suffix.lower():
            review_notes.append(
                "Target and staged replacement use different file suffixes; verify the target kind before applying the switch."
            )

    switch_status = "blocked" if blocking_issues else "ready"
    report = {
        "switch_status": switch_status,
        "promotion_status": promotion_status,
        "target_kind": target_kind,
        "promotion_manifest_path": _relative_repo_path(manifest_resolved),
        "target_path": _relative_repo_path(target_resolved) if target_resolved is not None else manifest_target_value,
        "staged_replacement_path": _relative_repo_path(staged_resolved) if staged_resolved is not None else staged_value,
        "target_overridden": bool(target_path_override),
        "candidate_identity": manifest.get("candidate_identity"),
        "candidate_source_path": manifest.get("candidate_source_path"),
        "candidate_note_path": manifest.get("candidate_note_path"),
        "blocking_issues": blocking_issues,
        "review_notes": review_notes,
        "passed_checks": passed_checks,
        "warnings": list(blocking_issues) + list(review_notes),
    }
    switch_context = {
        "promotion_manifest_path": manifest_resolved,
        "target_path": target_resolved,
        "staged_replacement_path": staged_resolved,
        "candidate_note_path": candidate_note_resolved,
        "manifest": manifest,
    }
    return report, switch_context


def render_effectiveness_switch_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    sections.append(_render_mapping_section("Switch Result", {
        "switch_status": report.get("switch_status"),
        "promotion_status": report.get("promotion_status"),
        "target_kind": report.get("target_kind"),
        "target_path": report.get("target_path"),
        "staged_replacement_path": report.get("staged_replacement_path"),
        "candidate_identity": report.get("candidate_identity"),
        "target_overridden": report.get("target_overridden"),
    }, format_name=format_name))
    sections.append(_render_mapping_section("Promotion Pack", {
        "promotion_manifest_path": report.get("promotion_manifest_path"),
        "candidate_source_path": report.get("candidate_source_path"),
        "candidate_note_path": report.get("candidate_note_path"),
    }, format_name=format_name))
    if format_name == "markdown":
        lines = ["# effectiveness baseline switch assessment", ""]
        lines.extend(section for section in sections if section)
        lines.append("## Blocking Issues")
        lines.append("")
        blocking = list(report.get("blocking_issues", []))
        lines.extend(f"- {item}" for item in blocking) if blocking else lines.append("- none")
        lines.append("")
        lines.append("## Review Notes")
        lines.append("")
        review_notes = list(report.get("review_notes", []))
        lines.extend(f"- {item}" for item in review_notes) if review_notes else lines.append("- none")
        lines.append("")
        lines.append("## Passed Checks")
        lines.append("")
        passed = list(report.get("passed_checks", []))
        lines.extend(f"- {item}" for item in passed) if passed else lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines = ["effectiveness baseline switch assessment", ""]
    lines.extend(section for section in sections if section)
    for title, key in (("Blocking Issues", "blocking_issues"), ("Review Notes", "review_notes"), ("Passed Checks", "passed_checks")):
        lines.append(title)
        items = list(report.get(key, []))
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def write_effectiveness_switch_outputs(
    report: dict,
    switch_context: dict,
    *,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
    apply: bool = True,
) -> dict:
    if apply and not out_dir:
        raise ValueError("out_dir is required when apply=True so the baseline switch keeps a rollback trail.")

    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))
    promotion_manifest_path = Path(switch_context["promotion_manifest_path"])
    target_path = switch_context.get("target_path")
    staged_replacement_path = switch_context.get("staged_replacement_path")
    candidate_note_path = switch_context.get("candidate_note_path")
    manifest_payload = dict(switch_context.get("manifest", {}))

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update({
            "summary_json": str(out_path / "summary.json"),
            "summary_markdown": str(out_path / "summary.md"),
            "switch_manifest": str(out_path / "switch_manifest.json"),
            "input_promotion_manifest": str(out_path / "input_promotion_manifest.json"),
        })
        if candidate_note_path is not None and Path(candidate_note_path).exists():
            artifacts["candidate_note_copy"] = str(out_path / "candidate_note.md")
        if apply and updated.get("switch_status") != "blocked" and target_path is not None:
            target_name = Path(target_path).name
            artifacts["previous_target_backup"] = str(out_path / "backup" / target_name)
            artifacts["applied_target_snapshot"] = str(out_path / "applied" / target_name)

    if artifacts:
        updated["artifacts"] = artifacts

    if apply and updated.get("switch_status") != "blocked":
        if target_path is None or staged_replacement_path is None:
            raise ValueError("Switch context is incomplete; target and staged replacement paths are required.")
        copy_file(Path(target_path), Path(artifacts["previous_target_backup"]))
        copy_file(Path(staged_replacement_path), Path(target_path))
        copy_file(Path(target_path), Path(artifacts["applied_target_snapshot"]))
        updated["switch_status"] = "applied"
        passed_checks = list(updated.get("passed_checks", []))
        applied_msg = "Staged replacement applied to target baseline path."
        if applied_msg not in passed_checks:
            passed_checks.append(applied_msg)
        updated["passed_checks"] = passed_checks

    switch_manifest = {
        "switch_status": updated.get("switch_status"),
        "promotion_status": updated.get("promotion_status"),
        "target_kind": updated.get("target_kind"),
        "target_path": updated.get("target_path"),
        "staged_replacement_path": updated.get("staged_replacement_path"),
        "promotion_manifest_path": updated.get("promotion_manifest_path"),
        "candidate_identity": updated.get("candidate_identity"),
        "candidate_source_path": updated.get("candidate_source_path"),
        "candidate_note_path": updated.get("candidate_note_path"),
        "previous_target_backup": artifacts.get("previous_target_backup"),
        "applied_target_snapshot": artifacts.get("applied_target_snapshot"),
        "rollback_ready": bool(artifacts.get("previous_target_backup")),
    }

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_switch_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(
            markdown + ("" if markdown.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        write_json(Path(artifacts["switch_manifest"]), switch_manifest)
        copy_file(promotion_manifest_path, Path(artifacts["input_promotion_manifest"]))
        if artifacts.get("candidate_note_copy") and candidate_note_path is not None and Path(candidate_note_path).exists():
            copy_file(Path(candidate_note_path), Path(artifacts["candidate_note_copy"]))

    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_switch_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        updated["summary_format"] = resolved_format

    return updated


def build_effectiveness_rollback_report(
    *,
    switch_manifest_path: str | Path | None = None,
    switch_dir: str | Path | None = None,
    target_path_override: str | Path | None = None,
) -> tuple[dict, dict]:
    if bool(switch_manifest_path) == bool(switch_dir):
        raise ValueError("Provide exactly one of --switch-manifest or --switch-dir.")

    manifest_input = Path(switch_manifest_path) if switch_manifest_path else Path(switch_dir) / "switch_manifest.json"
    manifest_resolved = _resolve_repo_input_path(manifest_input)
    if not manifest_resolved.exists():
        raise ValueError(f"Switch manifest not found: {manifest_input}")

    manifest = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    blocking_issues: list[str] = []
    review_notes: list[str] = []
    passed_checks: list[str] = []

    switch_status = str(manifest.get("switch_status") or "")
    if switch_status != "applied":
        blocking_issues.append(
            f"Switch manifest status is {switch_status or 'unknown'}, so rollback is blocked."
        )
    else:
        passed_checks.append("Switch manifest is applied and eligible for rollback.")

    target_kind = str(manifest.get("target_kind") or "")
    if target_kind not in {"table", "geometry-seed"}:
        blocking_issues.append(
            "Switch target kind is not supported for the rollback workflow."
        )
    else:
        passed_checks.append("Switch target kind is supported for rollback.")

    manifest_target_value = str(manifest.get("target_path") or "")
    target_input: Path | None = None
    target_resolved: Path | None = None
    if target_path_override:
        target_input = Path(target_path_override)
        review_notes.append("Target path override is active for this rollback assessment.")
    elif manifest_target_value:
        target_input = Path(manifest_target_value)
    else:
        blocking_issues.append("Switch manifest does not include a target path for rollback.")

    if target_input is not None:
        target_resolved = _resolve_repo_input_path(target_input)
        if not target_resolved.exists():
            blocking_issues.append(
                f"Target baseline path does not exist: {_relative_repo_path(target_resolved)}"
            )
        else:
            passed_checks.append("Target baseline path exists and can be backed up before rollback.")

    rollback_source_value = str(manifest.get("previous_target_backup") or "")
    rollback_source_resolved: Path | None = None
    if not rollback_source_value:
        blocking_issues.append("Switch manifest does not include a rollback source path.")
    else:
        rollback_source_resolved = _resolve_repo_input_path(rollback_source_value)
        if not rollback_source_resolved.exists():
            blocking_issues.append(
                f"Rollback source path does not exist: {_relative_repo_path(rollback_source_resolved)}"
            )
        else:
            passed_checks.append("Rollback source artifact exists and is ready to restore.")

    candidate_note_value = str(manifest.get("candidate_note_path") or "")
    candidate_note_resolved: Path | None = None
    if candidate_note_value:
        candidate_note_resolved = _resolve_repo_input_path(candidate_note_value)
        if candidate_note_resolved.exists():
            passed_checks.append("Candidate note is available for the rollback pack provenance trail.")
        else:
            review_notes.append(
                "Candidate note path from the switch manifest does not exist; the rollback pack will omit the copied note."
            )
    else:
        review_notes.append(
            "Switch manifest does not include a candidate note path; the rollback pack will omit the copied note."
        )

    if target_resolved is not None and rollback_source_resolved is not None:
        if target_resolved.suffix.lower() != rollback_source_resolved.suffix.lower():
            review_notes.append(
                "Target and rollback source use different file suffixes; verify the target kind before applying rollback."
            )

    rollback_status = "blocked" if blocking_issues else "ready"
    report = {
        "rollback_status": rollback_status,
        "switch_status": switch_status,
        "target_kind": target_kind,
        "switch_manifest_path": _relative_repo_path(manifest_resolved),
        "target_path": _relative_repo_path(target_resolved) if target_resolved is not None else manifest_target_value,
        "rollback_source_path": _relative_repo_path(rollback_source_resolved) if rollback_source_resolved is not None else rollback_source_value,
        "target_overridden": bool(target_path_override),
        "candidate_identity": manifest.get("candidate_identity"),
        "candidate_source_path": manifest.get("candidate_source_path"),
        "candidate_note_path": manifest.get("candidate_note_path"),
        "promotion_manifest_path": manifest.get("promotion_manifest_path"),
        "blocking_issues": blocking_issues,
        "review_notes": review_notes,
        "passed_checks": passed_checks,
        "warnings": list(blocking_issues) + list(review_notes),
    }
    rollback_context = {
        "switch_manifest_path": manifest_resolved,
        "target_path": target_resolved,
        "rollback_source_path": rollback_source_resolved,
        "candidate_note_path": candidate_note_resolved,
        "manifest": manifest,
    }
    return report, rollback_context


def render_effectiveness_rollback_report(report: dict, *, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(report, indent=2)
    sections = []
    sections.append(_render_mapping_section("Rollback Result", {
        "rollback_status": report.get("rollback_status"),
        "switch_status": report.get("switch_status"),
        "target_kind": report.get("target_kind"),
        "target_path": report.get("target_path"),
        "rollback_source_path": report.get("rollback_source_path"),
        "candidate_identity": report.get("candidate_identity"),
        "target_overridden": report.get("target_overridden"),
    }, format_name=format_name))
    sections.append(_render_mapping_section("Switch Pack", {
        "switch_manifest_path": report.get("switch_manifest_path"),
        "promotion_manifest_path": report.get("promotion_manifest_path"),
        "candidate_source_path": report.get("candidate_source_path"),
        "candidate_note_path": report.get("candidate_note_path"),
    }, format_name=format_name))
    if format_name == "markdown":
        lines = ["# effectiveness baseline rollback assessment", ""]
        lines.extend(section for section in sections if section)
        lines.append("## Blocking Issues")
        lines.append("")
        blocking = list(report.get("blocking_issues", []))
        lines.extend(f"- {item}" for item in blocking) if blocking else lines.append("- none")
        lines.append("")
        lines.append("## Review Notes")
        lines.append("")
        review_notes = list(report.get("review_notes", []))
        lines.extend(f"- {item}" for item in review_notes) if review_notes else lines.append("- none")
        lines.append("")
        lines.append("## Passed Checks")
        lines.append("")
        passed = list(report.get("passed_checks", []))
        lines.extend(f"- {item}" for item in passed) if passed else lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines = ["effectiveness baseline rollback assessment", ""]
    lines.extend(section for section in sections if section)
    for title, key in (("Blocking Issues", "blocking_issues"), ("Review Notes", "review_notes"), ("Passed Checks", "passed_checks")):
        lines.append(title)
        items = list(report.get(key, []))
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def write_effectiveness_rollback_outputs(
    report: dict,
    rollback_context: dict,
    *,
    out_dir: str = "",
    summary_out: str = "",
    summary_format: str = "auto",
    apply: bool = True,
) -> dict:
    if apply and not out_dir:
        raise ValueError("out_dir is required when apply=True so the baseline rollback keeps a forward-recovery trail.")

    updated = dict(report)
    artifacts = dict(updated.get("artifacts", {}))
    switch_manifest_path = Path(rollback_context["switch_manifest_path"])
    target_path = rollback_context.get("target_path")
    rollback_source_path = rollback_context.get("rollback_source_path")
    candidate_note_path = rollback_context.get("candidate_note_path")

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts.update({
            "summary_json": str(out_path / "summary.json"),
            "summary_markdown": str(out_path / "summary.md"),
            "rollback_manifest": str(out_path / "rollback_manifest.json"),
            "input_switch_manifest": str(out_path / "input_switch_manifest.json"),
        })
        if candidate_note_path is not None and Path(candidate_note_path).exists():
            artifacts["candidate_note_copy"] = str(out_path / "candidate_note.md")
        if apply and updated.get("rollback_status") != "blocked" and target_path is not None:
            target_name = Path(target_path).name
            artifacts["pre_rollback_target_backup"] = str(out_path / "pre_rollback" / target_name)
            artifacts["restored_target_snapshot"] = str(out_path / "restored" / target_name)

    if artifacts:
        updated["artifacts"] = artifacts

    if apply and updated.get("rollback_status") != "blocked":
        if target_path is None or rollback_source_path is None:
            raise ValueError("Rollback context is incomplete; target and rollback source paths are required.")
        copy_file(Path(target_path), Path(artifacts["pre_rollback_target_backup"]))
        copy_file(Path(rollback_source_path), Path(target_path))
        copy_file(Path(target_path), Path(artifacts["restored_target_snapshot"]))
        updated["rollback_status"] = "applied"
        passed_checks = list(updated.get("passed_checks", []))
        applied_msg = "Rollback source applied to target baseline path."
        if applied_msg not in passed_checks:
            passed_checks.append(applied_msg)
        updated["passed_checks"] = passed_checks

    rollback_manifest = {
        "rollback_status": updated.get("rollback_status"),
        "switch_status": updated.get("switch_status"),
        "target_kind": updated.get("target_kind"),
        "target_path": updated.get("target_path"),
        "rollback_source_path": updated.get("rollback_source_path"),
        "switch_manifest_path": updated.get("switch_manifest_path"),
        "promotion_manifest_path": updated.get("promotion_manifest_path"),
        "candidate_identity": updated.get("candidate_identity"),
        "candidate_source_path": updated.get("candidate_source_path"),
        "candidate_note_path": updated.get("candidate_note_path"),
        "pre_rollback_target_backup": artifacts.get("pre_rollback_target_backup"),
        "restored_target_snapshot": artifacts.get("restored_target_snapshot"),
        "forward_switch_ready": bool(artifacts.get("pre_rollback_target_backup")),
    }

    if out_dir:
        write_json(Path(artifacts["summary_json"]), updated)
        markdown = render_effectiveness_rollback_report(updated, format_name="markdown")
        Path(artifacts["summary_markdown"]).write_text(
            markdown + ("" if markdown.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        write_json(Path(artifacts["rollback_manifest"]), rollback_manifest)
        copy_file(switch_manifest_path, Path(artifacts["input_switch_manifest"]))
        if artifacts.get("candidate_note_copy") and candidate_note_path is not None and Path(candidate_note_path).exists():
            copy_file(Path(candidate_note_path), Path(artifacts["candidate_note_copy"]))

    if summary_out:
        resolved_format = infer_effectiveness_summary_format(summary_out, summary_format)
        rendered = render_effectiveness_rollback_report(updated, format_name=resolved_format)
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        updated["summary_format"] = resolved_format

    return updated


def build_effectiveness_candidate_template(
    *,
    spec_name: str | None = None,
    provenance: str | None = None,
) -> dict:
    spec = load_geometry_seed_spec(DEFAULT_GEOMETRY_SEED_SPEC)
    payload = geometry_seed_spec_to_payload(spec)
    payload["spec_name"] = str(spec_name or DEFAULT_CANDIDATE_TEMPLATE_SPEC_NAME)
    payload["provenance"] = str(provenance or DEFAULT_CANDIDATE_TEMPLATE_PROVENANCE)
    return payload


def render_effectiveness_candidate_provenance_note(
    *,
    spec_name: str,
    spec_path: str,
) -> str:
    lines = [
        "# Aurora VTOL effectiveness candidate provenance note",
        "",
        "Use this note alongside a candidate effectiveness spec before promoting it into the live comparison flow.",
        "",
        "## Candidate identity",
        "",
        f"- spec_name: `{spec_name}`",
        f"- spec_path: `{spec_path}`",
        "- owner: TODO",
        "- date_utc: TODO",
        "",
        "## Source evidence",
        "",
        "- source_type: TODO (CAD / CFD / bench / mixed)",
        "- source_reference: TODO",
        "- extraction_method: TODO",
        "- source_revision: TODO",
        "",
        "## Assumptions",
        "",
        "- TODO: fan footprint assumptions",
        "- TODO: plenum coupling assumptions",
        "- TODO: component scale assumptions",
        "",
        "## Validation status",
        "",
        "- validation_state: TODO (unreviewed / provisional / reviewed / accepted)",
        "- reviewer: TODO",
        "- known_limitations: TODO",
        "",
        "## Change summary vs baseline",
        "",
        "- TODO: what changed relative to the current geometry-seeded baseline",
        "- TODO: expected impact on allocator behavior",
        "",
    ]
    return "\n".join(lines)


def write_effectiveness_candidate_template_outputs(
    *,
    out_dir: str = "",
    spec_out: str = "",
    note_out: str = "",
    spec_name: str | None = None,
    provenance: str | None = None,
) -> dict:
    payload = build_effectiveness_candidate_template(spec_name=spec_name, provenance=provenance)
    artifacts: dict[str, str] = {}

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts["candidate_spec"] = str(out_path / "candidate_spec.json")
        artifacts["provenance_note"] = str(out_path / "provenance_template.md")
    if spec_out:
        artifacts["candidate_spec"] = str(Path(spec_out))
    if note_out:
        artifacts["provenance_note"] = str(Path(note_out))
    if not artifacts:
        raise ValueError("Provide --out-dir, --spec-out, or --note-out.")

    if "candidate_spec" in artifacts:
        spec_path = Path(artifacts["candidate_spec"])
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        spec_path = Path("candidate_spec.json")

    if "provenance_note" in artifacts:
        note_path = Path(artifacts["provenance_note"])
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note = render_effectiveness_candidate_provenance_note(
            spec_name=str(payload["spec_name"]),
            spec_path=str(_relative_repo_path(spec_path) if spec_path.is_absolute() else spec_path),
        )
        note_path.write_text(note + ("" if note.endswith("\n") else "\n"), encoding="utf-8")

    return {
        "template_spec_name": str(payload["spec_name"]),
        "provenance_placeholder": str(payload["provenance"]),
        "warnings": [
            "Template values are seeded from the current geometry baseline and must be replaced with real candidate evidence before comparison or adoption.",
            "Do not treat the generated candidate template as validated hardware truth.",
        ],
        "artifacts": artifacts,
    }

