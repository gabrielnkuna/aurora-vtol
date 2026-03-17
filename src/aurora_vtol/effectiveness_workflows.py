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


def _relative_repo_path(path: str | Path) -> str:
    value = Path(path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
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


def _build_warnings(
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


def build_effectiveness_report(
    *,
    spec_path: str | Path | None = None,
    table_path: str | Path | None = None,
) -> tuple[dict, NominalEffectivenessTable, GeometrySeedSpec | None]:
    if spec_path and table_path:
        raise ValueError("Provide either spec_path or table_path, not both.")

    spec: GeometrySeedSpec | None = None
    if table_path:
        resolved_source = Path(table_path).resolve()
        table = load_effectiveness_table(resolved_source)
        source_kind = "table"
    else:
        resolved_source = Path(spec_path).resolve() if spec_path else DEFAULT_GEOMETRY_SEED_SPEC.resolve()
        spec = load_geometry_seed_spec(resolved_source)
        table = build_effectiveness_table_from_geometry_seed(spec)
        source_kind = "geometry-seed"

    fan_weight_summary = _weight_summary(table.fan_segment_weights)
    plenum_weight_summary = _weight_summary(table.plenum_segment_weights)
    component_scale_summary = _component_scale_summary(table)
    report = {
        "source_kind": source_kind,
        "source_path": _relative_repo_path(resolved_source),
        "spec_summary": summarize_geometry_seed_spec(spec) if spec is not None else None,
        "table_summary": summarize_effectiveness_table(table),
        "fan_weight_summary": fan_weight_summary,
        "plenum_weight_summary": plenum_weight_summary,
        "component_scale_summary": component_scale_summary,
        "warnings": _build_warnings(
            spec=spec,
            table=table,
            fan_weight_summary=fan_weight_summary,
            plenum_weight_summary=plenum_weight_summary,
            component_scale_summary=component_scale_summary,
        ),
    }
    return report, table, spec


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
        sections.append(
            _render_mapping_section("Geometry Seed Summary", spec_summary, format_name=format_name)
        )
    sections.append(
        _render_mapping_section(
            "Effectiveness Table Summary",
            report.get("table_summary", {}),
            format_name=format_name,
        )
    )
    sections.append(
        _render_mapping_section(
            "Fan Weight Summary",
            report.get("fan_weight_summary", {}),
            format_name=format_name,
        )
    )
    sections.append(
        _render_mapping_section(
            "Plenum Weight Summary",
            report.get("plenum_weight_summary", {}),
            format_name=format_name,
        )
    )
    component_summary = report.get("component_scale_summary", {})
    for name in ("axial", "radial", "tangential"):
        if isinstance(component_summary.get(name), dict):
            sections.append(
                _render_mapping_section(
                    f"{name.title()} Scale Summary",
                    component_summary[name],
                    format_name=format_name,
                )
            )
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
                "source_payload": str(
                    out_path / ("source_spec.json" if spec is not None else "source_table.json")
                ),
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
        Path(artifacts["summary_markdown"]).write_text(
            markdown + ("" if markdown.endswith("\n") else "\n"),
            encoding="utf-8",
        )
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
        summary_path.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
        updated["summary_format"] = resolved_format

    return updated
