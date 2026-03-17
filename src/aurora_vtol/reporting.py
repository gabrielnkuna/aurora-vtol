from __future__ import annotations

import csv
import io
from pathlib import Path

from aurora_vtol.allocator.power_system import PowerSystemParams


REQUIREMENTS_TABLE_COLUMNS = [
    ('battery_full_v', 'battery_full_v'),
    ('battery_empty_v', 'battery_empty_v'),
    ('capacity_kwh', 'capacity_kwh'),
    ('internal_resistance_ohm', 'internal_resistance_ohm'),
    ('hover_power_kw', 'hover_power_kw'),
    ('peak_power_kw', 'peak_power_kw'),
    ('aux_power_kw', 'aux_power_kw'),
    ('feasible', 'feasible'),
    ('required_continuous_power_kw', 'required_continuous_power_kw'),
    ('best_selected_aggressiveness', 'best_selected_aggressiveness'),
    ('best_report_status', 'best_report_status'),
    ('best_continuous_power_p95_pct', 'best_continuous_power_p95_pct'),
    ('best_final_goal_error_m', 'best_final_goal_error_m'),
]

FAULT_ENVELOPE_COLUMNS = [
    ('case', 'case'),
    ('order', 'order'),
    ('families', 'families'),
    ('feasible', 'feasible'),
    ('report_status', 'report_status'),
    ('selected_aggressiveness', 'selected_aggressiveness'),
    ('continuous_power_p95_pct', 'continuous_power_p95_pct'),
    ('flap_limit_usage_peak_pct', 'flap_limit_usage_peak_pct'),
    ('xy_tracking_rms_n', 'xy_tracking_rms_n'),
    ('final_goal_error_m', 'final_goal_error_m'),
    ('arrival_time_s', 'arrival_time_s'),
    ('severity_score', 'severity_score'),
]

FAULT_THRESHOLD_COLUMNS = [
    ('case', 'case'),
    ('source', 'source'),
    ('order', 'order'),
    ('families', 'families'),
    ('best_feasible', 'best_feasible'),
    ('meets_threshold', 'meets_threshold'),
    ('required_continuous_power_kw', 'required_continuous_power_kw'),
    ('battery_full_v', 'battery_full_v'),
    ('capacity_kwh', 'capacity_kwh'),
    ('internal_resistance_ohm', 'internal_resistance_ohm'),
    ('hover_power_kw', 'hover_power_kw'),
    ('peak_power_kw', 'peak_power_kw'),
    ('selected_aggressiveness', 'selected_aggressiveness'),
    ('report_status', 'report_status'),
    ('continuous_power_p95_pct', 'continuous_power_p95_pct'),
    ('final_goal_error_m', 'final_goal_error_m'),
    ('arrival_time_s', 'arrival_time_s'),
]

PRESET_SUMMARY_COLUMNS = [
    ('preset', 'preset'),
    ('fault_case', 'fault_case'),
    ('cases', 'cases'),
    ('feasible_cases', 'feasible_cases'),
    ('best_feasible', 'best_feasible'),
    ('battery_full_v', 'battery_full_v'),
    ('capacity_kwh', 'capacity_kwh'),
    ('internal_resistance_ohm', 'internal_resistance_ohm'),
    ('hover_power_kw', 'hover_power_kw'),
    ('continuous_power_kw', 'continuous_power_kw'),
    ('peak_power_kw', 'peak_power_kw'),
    ('selected_aggressiveness', 'selected_aggressiveness'),
    ('report_status', 'report_status'),
    ('continuous_power_p95_pct', 'continuous_power_p95_pct'),
    ('best_final_goal_error_m', 'best_final_goal_error_m'),
    ('best_arrival_time_s', 'best_arrival_time_s'),
]

FAULT_THRESHOLD_PACK_COLUMNS = [
    ('preset', 'preset'),
    ('summary_source', 'summary_source'),
    ('case', 'case'),
    ('source', 'source'),
    ('order', 'order'),
    ('families', 'families'),
    ('best_feasible', 'best_feasible'),
    ('meets_threshold', 'meets_threshold'),
    ('required_continuous_power_kw', 'required_continuous_power_kw'),
    ('battery_full_v', 'battery_full_v'),
    ('capacity_kwh', 'capacity_kwh'),
    ('internal_resistance_ohm', 'internal_resistance_ohm'),
    ('hover_power_kw', 'hover_power_kw'),
    ('peak_power_kw', 'peak_power_kw'),
    ('selected_aggressiveness', 'selected_aggressiveness'),
    ('report_status', 'report_status'),
    ('continuous_power_p95_pct', 'continuous_power_p95_pct'),
    ('final_goal_error_m', 'final_goal_error_m'),
    ('arrival_time_s', 'arrival_time_s'),
]


def summarize_power_params(power: PowerSystemParams) -> dict:
    return {
        'battery_full_v': float(power.battery_full_v),
        'battery_empty_v': float(power.battery_empty_v),
        'capacity_kwh': float(power.capacity_kwh),
        'internal_resistance_ohm': float(power.internal_resistance_ohm),
        'hover_power_kw': float(power.hover_power_w) / 1000.0,
        'continuous_power_kw': float(power.continuous_power_w) / 1000.0,
        'peak_power_kw': float(power.peak_power_w) / 1000.0,
        'aux_power_kw': float(power.aux_power_w) / 1000.0,
    }


def flatten_requirements_row(row: dict) -> dict:
    model = row.get('power_model', {})
    return {
        'battery_full_v': model.get('battery_full_v'),
        'battery_empty_v': model.get('battery_empty_v'),
        'capacity_kwh': model.get('capacity_kwh'),
        'internal_resistance_ohm': model.get('internal_resistance_ohm'),
        'hover_power_kw': model.get('hover_power_kw'),
        'peak_power_kw': model.get('peak_power_kw'),
        'aux_power_kw': model.get('aux_power_kw'),
        'feasible': bool(row.get('feasible', False)),
        'required_continuous_power_kw': row.get('required_continuous_power_kw'),
        'best_selected_aggressiveness': row.get('best_selected_aggressiveness'),
        'best_report_status': row.get('best_report_status'),
        'best_continuous_power_p95_pct': row.get('best_continuous_power_p95_pct'),
        'best_final_goal_error_m': row.get('best_final_goal_error_m'),
    }


def stringify_table_value(value) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f'{value:.6f}'.rstrip('0').rstrip('.')
    return str(value)


def infer_requirements_table_format(path: str, format_name: str) -> str:
    if format_name != 'auto':
        return format_name
    suffix = Path(path).suffix.lower()
    if suffix == '.csv':
        return 'csv'
    if suffix in {'.md', '.markdown'}:
        return 'markdown'
    return 'text'


def _render_table(matrix: list[list[str]], *, format_name: str, title: str) -> str:
    headers = matrix[0]
    widths = [
        max(len(matrix[r][c]) for r in range(len(matrix)))
        for c in range(len(headers))
    ]

    if format_name == 'csv':
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        for row in matrix:
            writer.writerow(row)
        return buffer.getvalue()

    if format_name == 'markdown':
        def format_row(values: list[str]) -> str:
            return '| ' + ' | '.join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + ' |'

        lines = [
            f'# {title}',
            '',
            format_row(headers),
            '| ' + ' | '.join('-' * widths[idx] for idx in range(len(headers))) + ' |',
        ]
        for row in matrix[1:]:
            lines.append(format_row(row))
        lines.append('')
        return '\n'.join(lines)

    def format_plain_row(values: list[str]) -> str:
        return '  '.join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [title, format_plain_row(headers), format_plain_row(['-' * width for width in widths])]
    for row in matrix[1:]:
        lines.append(format_plain_row(row))
    lines.append('')
    return '\n'.join(lines)


def _row_values(row: dict, keys: list[str], *, join_lists: bool = False) -> list[str]:
    values = []
    for key in keys:
        value = row.get(key)
        if join_lists and isinstance(value, list):
            value = ','.join(str(item) for item in value)
        values.append(stringify_table_value(value))
    return values


def render_requirements_table(rows: list[dict], *, format_name: str, preset: str) -> str:
    headers = [label for label, _key in REQUIREMENTS_TABLE_COLUMNS]
    keys = [key for _label, key in REQUIREMENTS_TABLE_COLUMNS]
    flat_rows = [flatten_requirements_row(row) for row in rows]
    matrix = [headers] + [_row_values(row, keys) for row in flat_rows]
    return _render_table(matrix, format_name=format_name, title=f'{preset} power requirements')


def render_fault_envelope_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in FAULT_ENVELOPE_COLUMNS]
    keys = [key for _label, key in FAULT_ENVELOPE_COLUMNS]
    matrix = [headers] + [_row_values(row, keys, join_lists=True) for row in rows]
    return _render_table(matrix, format_name=format_name, title=title)


def render_fault_threshold_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in FAULT_THRESHOLD_COLUMNS]
    keys = [key for _label, key in FAULT_THRESHOLD_COLUMNS]
    matrix = [headers] + [_row_values(row, keys) for row in rows]
    return _render_table(matrix, format_name=format_name, title=title)


def render_preset_summary_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in PRESET_SUMMARY_COLUMNS]
    keys = [key for _label, key in PRESET_SUMMARY_COLUMNS]
    matrix = [headers] + [_row_values(row, keys) for row in rows]
    return _render_table(matrix, format_name=format_name, title=title)


def render_fault_threshold_pack_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in FAULT_THRESHOLD_PACK_COLUMNS]
    keys = [key for _label, key in FAULT_THRESHOLD_PACK_COLUMNS]
    matrix = [headers] + [_row_values(row, keys) for row in rows]
    return _render_table(matrix, format_name=format_name, title=title)
