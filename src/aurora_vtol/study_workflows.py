from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import typer

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.engineering import tune_coordinate_mission
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.mission_planning import MissionObstacle
from aurora_vtol.allocator.power_system import PowerSystemParams
from aurora_vtol.reporting import infer_requirements_table_format, render_requirements_table, summarize_power_params


def build_power_params(
    *,
    battery_full_v: float | None = None,
    battery_empty_v: float | None = None,
    capacity_kwh: float | None = None,
    internal_resistance_ohm: float | None = None,
    hover_power_kw: float | None = None,
    continuous_power_kw: float | None = None,
    peak_power_kw: float | None = None,
    aux_power_kw: float | None = None,
) -> PowerSystemParams:
    base = PowerSystemParams()
    overrides = {}
    if battery_full_v is not None:
        overrides['battery_full_v'] = float(battery_full_v)
    if battery_empty_v is not None:
        overrides['battery_empty_v'] = float(battery_empty_v)
    if capacity_kwh is not None:
        overrides['capacity_kwh'] = float(capacity_kwh)
    if internal_resistance_ohm is not None:
        overrides['internal_resistance_ohm'] = float(internal_resistance_ohm)
    if hover_power_kw is not None:
        overrides['hover_power_w'] = 1000.0 * float(hover_power_kw)
    if continuous_power_kw is not None:
        overrides['continuous_power_w'] = 1000.0 * float(continuous_power_kw)
    if peak_power_kw is not None:
        overrides['peak_power_w'] = 1000.0 * float(peak_power_kw)
    if aux_power_kw is not None:
        overrides['aux_power_w'] = 1000.0 * float(aux_power_kw)
    return replace(base, **overrides) if overrides else base


def build_coordinate_mission_kwargs(
    *,
    dest_x_m: float,
    dest_y_m: float,
    dest_z_m: float,
    start_x_m: float,
    start_y_m: float,
    start_z_m: float,
    total_s: float,
    yaw_hold_deg: float,
    mz_nm: float,
    cruise_alt_m: float,
    max_speed_mps: float,
    arrival_radius_m: float,
    slow_radius_m: float,
    descent_radius_m: float,
    hold_s: float,
    fxy_max_n: float,
    pos_k_n_per_m: float,
    vel_k_n_per_mps: float,
    obstacles: list[MissionObstacle],
    lim: ActuatorLimits,
    pl: PlenumModel,
    power: PowerSystemParams,
    fault: FaultSpec,
) -> dict:
    return {
        'dest_x_m': dest_x_m,
        'dest_y_m': dest_y_m,
        'dest_z_m': dest_z_m,
        'start_x_m': start_x_m,
        'start_y_m': start_y_m,
        'start_z_m': start_z_m,
        'total_s': total_s,
        'yaw_hold_deg': yaw_hold_deg,
        'mz_nm': mz_nm,
        'cruise_alt_m': cruise_alt_m,
        'max_speed_mps': max_speed_mps,
        'arrival_radius_m': arrival_radius_m,
        'slow_radius_m': slow_radius_m,
        'descent_radius_m': descent_radius_m,
        'hold_s': hold_s,
        'fxy_max_n': fxy_max_n,
        'pos_k_n_per_m': pos_k_n_per_m,
        'vel_k_n_per_mps': vel_k_n_per_mps,
        'obstacles': obstacles,
        'lim': lim,
        'pl': pl,
        'power': power,
        'fault': fault,
    }


def _fault_status_rank(status: str | None) -> int:
    return {'pass': 0, 'caution': 1, 'risk': 2}.get(status or '', 3)


FAULT_THRESHOLD_STATUS_VALUES = ('feasible', 'pass', 'caution', 'risk')


def _normalize_required_status(required_status: str | None) -> str:
    value = str(required_status or 'feasible').strip().lower()
    if value not in FAULT_THRESHOLD_STATUS_VALUES:
        raise typer.BadParameter(
            '--required-status must be one of: ' + ', '.join(FAULT_THRESHOLD_STATUS_VALUES)
        )
    return value


def _status_meets_threshold(*, feasible: bool, status: str | None, required_status: str) -> bool:
    normalized = _normalize_required_status(required_status)
    if normalized == 'feasible':
        return bool(feasible)
    return bool(feasible) and _fault_status_rank(status) <= _fault_status_rank(normalized)



def build_power_sweep_report_for_fault(
    *,
    preset: str,
    preset_cfg: dict,
    obstacles: list[MissionObstacle],
    fault_label: str,
    fault: FaultSpec,
    fault_summary: dict,
    required_status: str,
    power_target_pct: float,
    flap_target_pct: float,
    tune_iterations: int,
    tune_min_aggressiveness: float,
    battery_full_v: list[float],
    battery_empty_v: list[float],
    capacity_kwh: list[float],
    internal_resistance_ohm: list[float],
    hover_power_kw: list[float],
    continuous_power_kw: list[float],
    peak_power_kw: list[float],
    aux_power_kw: list[float],
    top: int,
) -> tuple[dict, dict | None]:
    default_power = PowerSystemParams()
    lim = ActuatorLimits(alpha_rate_deg_s=350.0)
    pl = PlenumModel(tau_s=0.08)

    def values_or_default(values, default):
        return list(values) if values else [default]

    grid = list(product(
        values_or_default(battery_full_v, default_power.battery_full_v),
        values_or_default(battery_empty_v, default_power.battery_empty_v),
        values_or_default(capacity_kwh, default_power.capacity_kwh),
        values_or_default(internal_resistance_ohm, default_power.internal_resistance_ohm),
        values_or_default(hover_power_kw, default_power.hover_power_w / 1000.0),
        values_or_default(continuous_power_kw, default_power.continuous_power_w / 1000.0),
        values_or_default(peak_power_kw, default_power.peak_power_w / 1000.0),
        values_or_default(aux_power_kw, default_power.aux_power_w / 1000.0),
    ))

    results = []
    best_candidate = None
    summary_payload = dict(fault_summary)
    normalized_required_status = _normalize_required_status(required_status)

    def selection_key(entry: dict) -> tuple[float, ...]:
        if entry.get('meets_threshold'):
            return (
                0.0,
                float(entry['power_model']['continuous_power_kw']),
                float(_fault_status_rank(entry.get('report_status'))),
                -float(entry.get('selected_aggressiveness') or 0.0),
            )
        if entry.get('feasible'):
            return (
                1.0,
                float(_fault_status_rank(entry.get('report_status'))),
                float(entry['power_model']['continuous_power_kw']),
                -float(entry.get('selected_aggressiveness') or 0.0),
            )
        return (
            2.0,
            float(entry.get('selected_score') or 1e9),
            float(_fault_status_rank(entry.get('report_status'))),
            -float(entry.get('selected_aggressiveness') or 0.0),
        )

    for combo in grid:
        power = build_power_params(
            battery_full_v=combo[0],
            battery_empty_v=combo[1],
            capacity_kwh=combo[2],
            internal_resistance_ohm=combo[3],
            hover_power_kw=combo[4],
            continuous_power_kw=combo[5],
            peak_power_kw=combo[6],
            aux_power_kw=combo[7],
        )
        mission_kwargs = build_coordinate_mission_kwargs(
            dest_x_m=float(preset_cfg['dest_x_m']),
            dest_y_m=float(preset_cfg['dest_y_m']),
            dest_z_m=float(preset_cfg.get('dest_z_m', 0.0)),
            start_x_m=0.0,
            start_y_m=0.0,
            start_z_m=0.0,
            total_s=float(preset_cfg['total_s']),
            yaw_hold_deg=0.0,
            mz_nm=0.0,
            cruise_alt_m=float(preset_cfg['cruise_alt_m']),
            max_speed_mps=float(preset_cfg['max_speed_mps']),
            arrival_radius_m=float(preset_cfg['arrival_radius_m']),
            slow_radius_m=float(preset_cfg['slow_radius_m']),
            descent_radius_m=float(preset_cfg['descent_radius_m']),
            hold_s=float(preset_cfg['hold_s']),
            fxy_max_n=float(preset_cfg['fxy_max_n']),
            pos_k_n_per_m=float(preset_cfg['pos_k_n_per_m']),
            vel_k_n_per_mps=float(preset_cfg['vel_k_n_per_mps']),
            obstacles=obstacles,
            lim=lim,
            pl=pl,
            power=power,
            fault=fault,
        )
        meta, hist, report, tuning = tune_coordinate_mission(
            mission_kwargs,
            target_continuous_power_p95_pct=power_target_pct,
            target_flap_peak_pct=flap_target_pct,
            min_aggressiveness=tune_min_aggressiveness,
            max_iterations=tune_iterations,
        )
        feasible = bool(tuning.get('envelope', {}).get('feasible', False))
        report_status = report.get('status')
        entry = {
            'fault_case': dict(summary_payload),
            'power_model': summarize_power_params(power),
            'feasible': feasible,
            'meets_threshold': _status_meets_threshold(
                feasible=feasible,
                status=report_status,
                required_status=normalized_required_status,
            ),
            'selected_aggressiveness': tuning.get('selected', {}).get('aggressiveness'),
            'selected_score': tuning.get('selected', {}).get('score'),
            'report_status': report_status,
            'warnings': list(report.get('warnings', [])),
            'continuous_power_p95_pct': report.get('thermal', {}).get('continuous_power_p95_pct'),
            'power_margin_min_kw': report.get('thermal', {}).get('power_margin_min_kw'),
            'flap_limit_usage_peak_pct': report.get('flaps', {}).get('limit_usage_peak_pct'),
            'xy_tracking_rms_n': report.get('vectoring', {}).get('xy_tracking_rms_n'),
            'final_goal_error_m': report.get('mission', {}).get('final_goal_error_m'),
            'arrival_time_s': report.get('mission', {}).get('arrival_time_s'),
        }
        bundle = {'entry': entry, 'meta': meta, 'hist': hist}
        results.append(bundle)
        if best_candidate is None:
            best_candidate = bundle
        elif selection_key(entry) < selection_key(best_candidate['entry']):
            best_candidate = bundle

    ranked = sorted(results, key=lambda item: selection_key(item['entry']))
    feasible_count = sum(1 for item in ranked if item['entry']['feasible'])

    requirements_groups = {}
    for item in ranked:
        model = item['entry']['power_model']
        key = (
            model['battery_full_v'],
            model['battery_empty_v'],
            model['capacity_kwh'],
            model['internal_resistance_ohm'],
            model['hover_power_kw'],
            model['peak_power_kw'],
            model['aux_power_kw'],
        )
        bucket = requirements_groups.setdefault(key, {'all': [], 'feasible': [], 'qualified': []})
        bucket['all'].append(item['entry'])
        if item['entry']['feasible']:
            bucket['feasible'].append(item['entry'])
        if item['entry']['meets_threshold']:
            bucket['qualified'].append(item['entry'])

    requirements_table = []
    for key, bucket in requirements_groups.items():
        feasible_entries = sorted(bucket['feasible'], key=lambda entry: entry['power_model']['continuous_power_kw'])
        qualified_entries = sorted(bucket['qualified'], key=lambda entry: entry['power_model']['continuous_power_kw'])
        best_effort_entry = min(bucket['all'], key=lambda entry: (entry['selected_score'], -float(entry['selected_aggressiveness'] or 0.0)))
        best_entry = qualified_entries[0] if qualified_entries else (feasible_entries[0] if feasible_entries else best_effort_entry)
        base_model = {
            'battery_full_v': key[0],
            'battery_empty_v': key[1],
            'capacity_kwh': key[2],
            'internal_resistance_ohm': key[3],
            'hover_power_kw': key[4],
            'peak_power_kw': key[5],
            'aux_power_kw': key[6],
        }
        row = {
            'power_model': base_model,
            'feasible': bool(feasible_entries),
            'meets_threshold': bool(qualified_entries),
            'required_status': normalized_required_status,
            'required_continuous_power_kw': (qualified_entries[0]['power_model']['continuous_power_kw'] if qualified_entries else None),
            'best_selected_aggressiveness': best_entry['selected_aggressiveness'],
            'best_report_status': best_entry['report_status'],
            'best_continuous_power_p95_pct': best_entry['continuous_power_p95_pct'],
            'best_final_goal_error_m': best_entry['final_goal_error_m'],
            'best_arrival_time_s': best_entry['arrival_time_s'],
        }
        requirements_table.append(row)

    requirements_table.sort(key=lambda row: (
        0 if row['meets_threshold'] else 1,
        0 if row['feasible'] else 1,
        row['required_continuous_power_kw'] if row['required_continuous_power_kw'] is not None else 1e9,
        row['power_model']['hover_power_kw'],
        row['power_model']['internal_resistance_ohm'],
    ))

    report = {
        'preset': preset,
        'fault_case': dict(summary_payload),
        'targets': {
            'required_status': normalized_required_status,
            'continuous_power_p95_pct': float(power_target_pct),
            'flap_limit_usage_peak_pct': float(flap_target_pct),
            'tune_iterations': int(tune_iterations),
            'tune_min_aggressiveness': float(tune_min_aggressiveness),
        },
        'search_space': {
            'cases': len(ranked),
            'feasible_cases': feasible_count,
            'requirement_rows': len(requirements_table),
        },
        'best': (best_candidate['entry'] if best_candidate is not None else None),
        'requirements_table': requirements_table,
        'results': [item['entry'] for item in ranked[:max(1, int(top))]],
    }
    return report, best_candidate



def write_power_sweep_outputs(
    report: dict,
    *,
    preset: str,
    out: str = "",
    requirements_out: str = "",
    requirements_format: str = "auto",
    title_label: str | None = None,
) -> dict:
    if requirements_out:
        chosen_format = infer_requirements_table_format(requirements_out, requirements_format)
        rendered = render_requirements_table(
            report['requirements_table'],
            format_name=chosen_format,
            preset=(title_label or preset),
        )
        Path(requirements_out).write_text(rendered, encoding='utf-8')
        report['requirements_out'] = requirements_out
        report['requirements_format'] = chosen_format
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


