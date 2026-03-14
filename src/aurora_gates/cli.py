from __future__ import annotations
import csv
import io
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Optional
import numpy as np
import typer

from aurora_gates.allocator.metrics import yaw_track_coupling_mean_abs
from aurora_gates.allocator.engineering import assess_trace, render_assessment, tune_coordinate_mission
from aurora_gates.bridge import build_qgc_wpl_mission, build_setpoint_script, inspect_script, load_script_jsonl, load_trace_json, stream_script_ardupilot_sitl, stream_script_mavlink, summarize_qgc_wpl_mission, summarize_script, summarize_setpoints, write_qgc_wpl_mission, write_script_jsonl, write_script_svg
from aurora_gates.allocator.sim import MissionObstacle, PowerSystemParams, run_coordinate_mission_v5, run_demo, run_step_redirect_v3, run_step_test_v3, run_repel_test_v4, run_step_snap_v3
from aurora_gates.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_gates.allocator.field import RepelField
from aurora_gates.allocator.faults import FaultSpec
from aurora_gates.allocator.model import RingGeometry
from aurora_gates.allocator.trace import save_trace_json

app = typer.Typer(help="Project Aurora Allocator V1–V4 CLI")
alloc_app = typer.Typer(help="Allocator demos")
bridge_app = typer.Typer(help="Bridge planned traces into dry-run or live setpoint streams")
app.add_typer(alloc_app, name="alloc")
app.add_typer(bridge_app, name="bridge")

COORDINATE_PRESETS = {
    "medium": {
        "dest_x_m": 120.0,
        "dest_y_m": 40.0,
        "dest_z_m": 0.0,
        "total_s": 45.0,
        "cruise_alt_m": 18.0,
        "max_speed_mps": 8.0,
        "arrival_radius_m": 2.0,
        "slow_radius_m": 22.0,
        "descent_radius_m": 28.0,
        "hold_s": 2.0,
        "fxy_max_n": 3600.0,
        "pos_k_n_per_m": 100.0,
        "vel_k_n_per_mps": 1000.0,
        "obstacles": ["60,18,14,180,650,28"],
    },
    "long": {
        "dest_x_m": 180.0,
        "dest_y_m": 60.0,
        "dest_z_m": 0.0,
        "total_s": 70.0,
        "cruise_alt_m": 25.0,
        "max_speed_mps": 10.0,
        "arrival_radius_m": 2.0,
        "slow_radius_m": 28.0,
        "descent_radius_m": 35.0,
        "hold_s": 2.0,
        "fxy_max_n": 3800.0,
        "pos_k_n_per_m": 90.0,
        "vel_k_n_per_mps": 1100.0,
        "obstacles": ["95,25,18,180,700,35"],
    },
    "very-long": {
        "dest_x_m": 300.0,
        "dest_y_m": 120.0,
        "dest_z_m": 0.0,
        "total_s": 110.0,
        "cruise_alt_m": 35.0,
        "max_speed_mps": 12.0,
        "arrival_radius_m": 3.0,
        "slow_radius_m": 40.0,
        "descent_radius_m": 50.0,
        "hold_s": 3.0,
        "fxy_max_n": 4000.0,
        "pos_k_n_per_m": 80.0,
        "vel_k_n_per_mps": 1200.0,
        "obstacles": ["170,70,25,180,700,45"],
    },
}


FAULT_CASE_PRESETS = {
    "nominal": {
        "description": "No injected actuator faults.",
        "kwargs": {},
    },
    "dead-fan-0": {
        "description": "Fan group 0 fully failed.",
        "kwargs": {"dead_fan_group": 0, "dead_fan_scale": 0.0},
    },
    "dead-fan-1": {
        "description": "Fan group 1 fully failed.",
        "kwargs": {"dead_fan_group": 1, "dead_fan_scale": 0.0},
    },
    "stuck-flap-0": {
        "description": "Flap 0 stuck at 20 degrees.",
        "kwargs": {"stuck_flap_idx": 0, "stuck_flap_alpha_deg": 20.0},
    },
    "stuck-flap-8": {
        "description": "Flap 8 stuck at 20 degrees.",
        "kwargs": {"stuck_flap_idx": 8, "stuck_flap_alpha_deg": 20.0},
    },
    "slow-flap-0": {
        "description": "Flap 0 responds at 35 percent of nominal rate.",
        "kwargs": {"slow_flap_idx": 0, "slow_flap_rate_scale": 0.35},
    },
    "slow-flap-8": {
        "description": "Flap 8 responds at 35 percent of nominal rate.",
        "kwargs": {"slow_flap_idx": 8, "slow_flap_rate_scale": 0.35},
    },
    "weak-flap-0": {
        "description": "Flap 0 has 55 percent vectoring authority.",
        "kwargs": {"weak_flap_idx": 0, "weak_flap_scale": 0.55},
    },
    "weak-flap-8": {
        "description": "Flap 8 has 55 percent vectoring authority.",
        "kwargs": {"weak_flap_idx": 8, "weak_flap_scale": 0.55},
    },
    "biased-flap-0": {
        "description": "Flap 0 carries an 8 degree bias offset.",
        "kwargs": {"biased_flap_idx": 0, "biased_flap_offset_deg": 8.0},
    },
    "biased-flap-8": {
        "description": "Flap 8 carries an 8 degree bias offset.",
        "kwargs": {"biased_flap_idx": 8, "biased_flap_offset_deg": 8.0},
    },
    "plenum-sector-0": {
        "description": "Plenum sector 0 reduced to 70 percent thrust.",
        "kwargs": {"plenum_sector_idx": 0, "plenum_sector_scale": 0.7},
    },
    "plenum-sector-1": {
        "description": "Plenum sector 1 reduced to 70 percent thrust.",
        "kwargs": {"plenum_sector_idx": 1, "plenum_sector_scale": 0.7},
    },
}

CORE_FAULT_CASES = ["nominal", "dead-fan-0", "stuck-flap-0", "plenum-sector-0"]
STRESS_FAULT_CASES = [
    "dead-fan-0+stuck-flap-8",
    "dead-fan-0+plenum-sector-1",
    "stuck-flap-8+plenum-sector-1",
    "dead-fan-0+stuck-flap-8+plenum-sector-1",
]
MANEUVER_FAULT_CASES = [
    "nominal",
    "slow-flap-0",
    "weak-flap-0",
    "biased-flap-0",
    "stuck-flap-0",
    "dead-fan-0",
    "plenum-sector-0",
    "dead-fan-0+slow-flap-8",
    "dead-fan-0+biased-flap-8",
    "weak-flap-8+plenum-sector-1",
]
FAULT_CASE_SETS = {
    "nominal": ["nominal"],
    "core": CORE_FAULT_CASES,
    "stress": STRESS_FAULT_CASES,
    "maneuver": MANEUVER_FAULT_CASES,
    "all": CORE_FAULT_CASES + STRESS_FAULT_CASES + MANEUVER_FAULT_CASES,
}


def _fault_case_sort_key(name: str) -> tuple[int, str]:
    order = {case: idx for idx, case in enumerate(CORE_FAULT_CASES + STRESS_FAULT_CASES)}
    return (order.get(name, len(order)), name)


def resolve_fault_case(case_name: str) -> tuple[str, dict]:
    parts = [part.strip() for part in case_name.split("+") if part.strip()]
    if not parts:
        raise typer.BadParameter("--fault-case must not be empty")
    if parts == ["nominal"]:
        spec = FAULT_CASE_PRESETS["nominal"]
        return "nominal", {
            "description": spec["description"],
            "kwargs": {},
            "components": ["nominal"],
        }

    merged_kwargs: dict[str, float | int] = {}
    components: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part == "nominal":
            continue
        spec = FAULT_CASE_PRESETS.get(part)
        if spec is None:
            allowed = ", ".join(sorted(FAULT_CASE_PRESETS))
            raise typer.BadParameter(
                f"--fault-case must use known components joined by '+': {allowed}"
            )
        if part in seen:
            continue
        seen.add(part)
        components.append(part)
        for key, value in spec["kwargs"].items():
            if key in merged_kwargs and merged_kwargs[key] != value:
                raise typer.BadParameter(f"conflicting composite --fault-case '{case_name}'")
            merged_kwargs[key] = value

    if not components:
        spec = FAULT_CASE_PRESETS["nominal"]
        return "nominal", {
            "description": spec["description"],
            "kwargs": {},
            "components": ["nominal"],
        }

    components.sort(key=_fault_case_sort_key)
    normalized = "+".join(components)
    description = " + ".join(FAULT_CASE_PRESETS[name]["description"] for name in components)
    return normalized, {
        "description": description,
        "kwargs": merged_kwargs,
        "components": components,
    }


def build_fault_spec(case_name: str) -> FaultSpec:
    _normalized, spec = resolve_fault_case(case_name)
    return FaultSpec(**spec["kwargs"])


def summarize_fault_case(case_name: str, fault: FaultSpec | None = None) -> dict:
    normalized_case, spec = resolve_fault_case(case_name)
    fault = fault or FaultSpec(**spec["kwargs"])
    return {
        'case': normalized_case,
        'components': list(spec['components']),
        'description': spec['description'],
        'stuck_flap_idx': fault.stuck_flap_idx,
        'stuck_flap_alpha_deg': fault.stuck_flap_alpha_deg,
        'slow_flap_idx': fault.slow_flap_idx,
        'slow_flap_rate_scale': fault.slow_flap_rate_scale,
        'weak_flap_idx': fault.weak_flap_idx,
        'weak_flap_scale': fault.weak_flap_scale,
        'biased_flap_idx': fault.biased_flap_idx,
        'biased_flap_offset_deg': fault.biased_flap_offset_deg,
        'dead_fan_group': fault.dead_fan_group,
        'dead_fan_scale': fault.dead_fan_scale,
        'plenum_sector_idx': fault.plenum_sector_idx,
        'plenum_sector_scale': fault.plenum_sector_scale,
    }


def fault_spec_from_summary(summary: dict | None) -> FaultSpec:
    data = summary or {}
    return FaultSpec(
        stuck_flap_idx=(None if data.get('stuck_flap_idx') is None else int(data['stuck_flap_idx'])),
        stuck_flap_alpha_deg=float(data.get('stuck_flap_alpha_deg', 0.0) or 0.0),
        slow_flap_idx=(None if data.get('slow_flap_idx') is None else int(data['slow_flap_idx'])),
        slow_flap_rate_scale=float(data.get('slow_flap_rate_scale', 0.5) or 0.5),
        weak_flap_idx=(None if data.get('weak_flap_idx') is None else int(data['weak_flap_idx'])),
        weak_flap_scale=float(data.get('weak_flap_scale', 0.7) or 0.7),
        biased_flap_idx=(None if data.get('biased_flap_idx') is None else int(data['biased_flap_idx'])),
        biased_flap_offset_deg=float(data.get('biased_flap_offset_deg', 0.0) or 0.0),
        dead_fan_group=(None if data.get('dead_fan_group') is None else int(data['dead_fan_group'])),
        dead_fan_scale=float(data.get('dead_fan_scale', 0.0) or 0.0),
        plenum_sector_idx=(None if data.get('plenum_sector_idx') is None else int(data['plenum_sector_idx'])),
        plenum_sector_scale=float(data.get('plenum_sector_scale', 0.7) or 0.7),
    )


def select_fault_cases(fault_cases: list[str], fault_set: str) -> list[str]:
    if fault_cases:
        requested = list(fault_cases)
    else:
        requested = FAULT_CASE_SETS.get(fault_set, [])
        if not requested:
            allowed = ", ".join(sorted(FAULT_CASE_SETS))
            raise typer.BadParameter(f"--fault-set must be one of: {allowed}")

    selected: list[str] = []
    seen: set[str] = set()
    for name in requested:
        normalized, _spec = resolve_fault_case(name)
        if normalized not in seen:
            seen.add(normalized)
            selected.append(normalized)
    return selected


MANEUVER_DEFAULT_ANGLES = [45.0, 90.0, 135.0, 180.0]
MANEUVER_IGNORED_WARNINGS = {
    'final mission position error exceeded 1.5x arrival radius',
    'vehicle did not settle below 0.75 m/s at the end of the run',
    'minimum obstacle clearance fell below 2 m',
}


def _status_rank(status: str) -> int:
    return {'pass': 0, 'caution': 1, 'risk': 2}.get(status, 3)


def _slugify_case(value: str) -> str:
    slug = ''.join(ch if ch.isalnum() else '-' for ch in value.lower())
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug.strip('-')


def _maneuver_scalar(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def assess_maneuver_result(meta: dict, hist: dict, profile: str) -> dict:
    engineering = assess_trace(meta, hist)
    warnings = [w for w in engineering['warnings'] if w not in MANEUVER_IGNORED_WARNINGS]
    step_metrics = meta.get('step_metrics', {})
    turn_delta = abs(((float(meta.get('dir_b_deg', 0.0)) - float(meta.get('dir_a_deg', 0.0)) + 180.0) % 360.0) - 180.0)
    requires_reversal = (profile == 'step-snap') or turn_delta >= 135.0

    if requires_reversal and step_metrics.get('t_reversal_s') is None:
        warnings.append('maneuver did not establish reverse motion after the redirect')
    if step_metrics.get('t90_dir_s') is None:
        warnings.append('maneuver track did not align within 20 degrees of the target direction')
    if profile == 'step-snap' and step_metrics.get('t_to_speed_below_thr_s') is None:
        warnings.append('snap-stop phase never dropped below the configured stop threshold')

    yaw_track_decoupling = step_metrics.get('yaw_track_coupling_mean_abs_deg')
    yaw_hold_deg = float(meta.get('yaw_hold_deg', 0.0))
    yaw_hist = np.asarray(hist.get('yaw_deg', []), dtype=float)
    yaw_rate_hist = np.asarray(hist.get('yaw_rate_deg_s', []), dtype=float)
    yaw_hold_error = None
    yaw_rate_p95 = None
    if yaw_hist.size:
        yaw_err = np.abs((yaw_hist - yaw_hold_deg + 180.0) % 360.0 - 180.0)
        yaw_hold_error = float(np.mean(yaw_err))
    if yaw_rate_hist.size:
        yaw_rate_p95 = float(np.percentile(np.abs(yaw_rate_hist), 95.0))

    if yaw_hold_error is not None and yaw_hold_error > 5.0:
        warnings.append('yaw hold error exceeded 5 degrees during the maneuver')
    if yaw_rate_p95 is not None and yaw_rate_p95 > 5.0:
        warnings.append('yaw rate exceeded 5 deg/s during the maneuver')

    status = 'pass' if not warnings else ('caution' if len(warnings) <= 3 else 'risk')
    return {
        'status': status,
        'warnings': warnings,
        'maneuver': {
            'turn_delta_deg': turn_delta,
            't_to_speed_below_thr_s': step_metrics.get('t_to_speed_below_thr_s'),
            'snap_stop_distance_m': step_metrics.get('snap_stop_distance_m'),
            't_reversal_s': step_metrics.get('t_reversal_s'),
            't90_dir_s': step_metrics.get('t90_dir_s'),
            'min_speed_transition_mps': step_metrics.get('min_speed_transition_mps'),
            'peak_speed_mps': step_metrics.get('peak_speed_mps'),
            'yaw_hold_error_mean_abs_deg': yaw_hold_error,
            'yaw_rate_p95_deg_s': yaw_rate_p95,
            'yaw_track_decoupling_mean_abs_deg': yaw_track_decoupling,
        },
        'engineering': engineering,
    }


def _run_maneuver_profile(
    profile: str,
    *,
    dir_a_deg: float,
    dir_b_deg: float,
    fxy_n: float,
    step_time_s: float,
    total_s: float,
    yaw_hold_deg: float,
    mz_nm: float,
    lim: ActuatorLimits,
    pl: PlenumModel,
    fault: FaultSpec,
    snap_stop_s: float = 0.8,
    brake_gain: float = 1.2,
    speed_stop_thr_mps: float = 0.2,
    redirect_hold_frac: float = -1.0,
    redirect_steer_scale: float = 1.0,
    redirect_time_s: float = 1.6,
    redirect_speed_scale: float = 0.88,
    redirect_cross_gain: float = 1.0,
):
    if profile == 'step':
        return run_step_test_v3(
            dir_deg_a=dir_a_deg,
            dir_deg_b=dir_b_deg,
            fxy_n=fxy_n,
            step_time_s=step_time_s,
            total_s=total_s,
            yaw_hold_deg=yaw_hold_deg,
            mz_nm=mz_nm,
            lim=lim,
            pl=pl,
            fault=fault,
        )
    if profile == 'step-snap':
        return run_step_snap_v3(
            dir_deg_a=dir_a_deg,
            dir_deg_b=dir_b_deg,
            fxy_n=fxy_n,
            step_time_s=step_time_s,
            snap_stop_s=snap_stop_s,
            total_s=total_s,
            yaw_hold_deg=yaw_hold_deg,
            mz_nm=mz_nm,
            lim=lim,
            pl=pl,
            fault=fault,
            brake_gain=brake_gain,
            speed_stop_thr_mps=speed_stop_thr_mps,
            redirect_hold_frac=redirect_hold_frac,
            redirect_steer_scale=redirect_steer_scale,
        )
    if profile in {'step-redirect', 'step-redirect-eco'}:
        return run_step_redirect_v3(
            dir_deg_a=dir_a_deg,
            dir_deg_b=dir_b_deg,
            fxy_n=fxy_n,
            step_time_s=step_time_s,
            redirect_time_s=redirect_time_s,
            total_s=total_s,
            yaw_hold_deg=yaw_hold_deg,
            mz_nm=mz_nm,
            lim=lim,
            pl=pl,
            fault=fault,
            redirect_speed_scale=redirect_speed_scale,
            redirect_cross_gain=redirect_cross_gain,
        )
    raise ValueError(f'unknown maneuver profile: {profile}')


def _maneuver_tune_snapshot(params: dict, aggressiveness: float) -> dict:
    snap = {
        'aggressiveness': float(aggressiveness),
        'fxy_n': float(params['fxy_n']),
        'step_time_s': float(params['step_time_s']),
        'total_s': float(params['total_s']),
    }
    if 'snap_stop_s' in params:
        snap['snap_stop_s'] = float(params['snap_stop_s'])
    if 'brake_gain' in params:
        snap['brake_gain'] = float(params['brake_gain'])
    if 'speed_stop_thr_mps' in params:
        snap['speed_stop_thr_mps'] = float(params['speed_stop_thr_mps'])
    if 'redirect_hold_frac' in params:
        snap['redirect_hold_frac'] = float(params['redirect_hold_frac'])
    if 'redirect_steer_scale' in params:
        snap['redirect_steer_scale'] = float(params['redirect_steer_scale'])
    if 'redirect_time_s' in params:
        snap['redirect_time_s'] = float(params['redirect_time_s'])
    if 'redirect_speed_scale' in params:
        snap['redirect_speed_scale'] = float(params['redirect_speed_scale'])
    if 'redirect_cross_gain' in params:
        snap['redirect_cross_gain'] = float(params['redirect_cross_gain'])
    return snap


def _apply_maneuver_aggressiveness(base_kwargs: dict, aggressiveness: float, profile: str) -> dict:
    g = max(0.10, min(1.0, float(aggressiveness)))
    params = dict(base_kwargs)
    turn_delta = abs(((float(base_kwargs['dir_b_deg']) - float(base_kwargs['dir_a_deg']) + 180.0) % 360.0) - 180.0)
    turn_ratio = turn_delta / 180.0
    params['fxy_n'] = float(base_kwargs['fxy_n'] * (0.22 + 0.78 * g) * (1.0 - 0.25 * turn_ratio * (1.0 - 0.5 * g)))
    params['total_s'] = float(base_kwargs['total_s'] * (1.0 + (0.55 + 0.55 * turn_ratio) * (1.0 - g)))
    if profile == 'step-snap':
        base_hold_frac = float(base_kwargs.get('redirect_hold_frac', -1.0))
        base_steer_scale = float(base_kwargs.get('redirect_steer_scale', 1.0))
        params['snap_stop_s'] = float(base_kwargs['snap_stop_s'] * (1.05 + (2.2 + 1.8 * turn_ratio) * (1.0 - g) + 0.35 * turn_ratio))
        params['brake_gain'] = float(base_kwargs['brake_gain'] * (0.74 + 0.26 * g + 0.08 * turn_ratio))
        if base_hold_frac < 0.0:
            params['redirect_hold_frac'] = float(np.clip(0.60 + 0.25 * turn_ratio + 0.18 * (1.0 - g), 0.55, 0.98))
        else:
            params['redirect_hold_frac'] = float(np.clip(base_hold_frac, 0.0, 0.98))
        params['redirect_steer_scale'] = float(np.clip(base_steer_scale * (0.55 + 0.45 * g) * (1.0 - 0.25 * turn_ratio), 0.15, 1.5))
    elif profile in {'step-redirect', 'step-redirect-eco'}:
        base_time = float(base_kwargs.get('redirect_time_s', 1.6))
        base_speed_scale = float(base_kwargs.get('redirect_speed_scale', 0.88))
        base_cross_gain = float(base_kwargs.get('redirect_cross_gain', 1.0))
        if profile == 'step-redirect-eco':
            params['fxy_n'] *= 0.62 + 0.22 * g - 0.20 * turn_ratio * (1.0 - g)
            params['total_s'] *= 1.0 + 0.95 * turn_ratio * (1.0 - g)
            params['redirect_time_s'] = float(base_time * (1.25 + (2.55 + 2.05 * turn_ratio) * (1.0 - g) + 0.45 * turn_ratio))
            params['redirect_speed_scale'] = float(np.clip(base_speed_scale * (0.58 + 0.22 * g - 0.14 * turn_ratio * (1.0 - g)), 0.38, 0.88))
            params['redirect_cross_gain'] = float(np.clip(base_cross_gain * (1.12 + 0.65 * turn_ratio + 0.42 * (1.0 - g)), 1.00, 2.60))
        else:
            params['fxy_n'] *= 0.80 + 0.20 * g - 0.14 * turn_ratio * (1.0 - g)
            params['total_s'] *= 1.0 + 0.42 * turn_ratio * (1.0 - g)
            params['redirect_time_s'] = float(base_time * (1.05 + (1.85 + 1.55 * turn_ratio) * (1.0 - g) + 0.30 * turn_ratio))
            params['redirect_speed_scale'] = float(np.clip(base_speed_scale * (0.72 + 0.28 * g - 0.12 * turn_ratio * (1.0 - g)), 0.48, 1.0))
            params['redirect_cross_gain'] = float(np.clip(base_cross_gain * (0.98 + 0.42 * turn_ratio + 0.26 * (1.0 - g)), 0.80, 2.10))
    return params


def _maneuver_tune_passes(
    assessment: dict,
    *,
    profile: str,
    target_continuous_power_p95_pct: float,
    target_flap_peak_pct: float,
    target_fan_tracking_pct: float,
    target_xy_tracking_rms_n: float,
    target_yaw_coupling_deg: float,
) -> bool:
    engineering = assessment.get('engineering', {})
    maneuver = assessment.get('maneuver', {})
    thermal = engineering.get('thermal', {})
    flaps = engineering.get('flaps', {})
    fans = engineering.get('fans', {})
    vectoring = engineering.get('vectoring', {})

    continuous_p95 = _maneuver_scalar(thermal.get('continuous_power_p95_pct'))
    flap_peak = _maneuver_scalar(flaps.get('limit_usage_peak_pct'))
    flap_tracking = _maneuver_scalar(flaps.get('tracking_rms_deg'))
    fan_tracking_pct = _maneuver_scalar(fans.get('tracking_rms_pct_mean_cmd'))
    fan_response_min_pct = _maneuver_scalar(fans.get('response_min_pct'))
    xy_tracking_rms = _maneuver_scalar(vectoring.get('xy_tracking_rms_n'))
    alignment_p95_deg = _maneuver_scalar(vectoring.get('alignment_p95_deg'))
    yaw_hold_error = _maneuver_scalar(maneuver.get('yaw_hold_error_mean_abs_deg'))
    turn_delta = _maneuver_scalar(maneuver.get('turn_delta_deg')) or 0.0
    requires_reversal = (profile == 'step-snap') or turn_delta >= 135.0

    return (
        (not requires_reversal or maneuver.get('t_reversal_s') is not None)
        and maneuver.get('t90_dir_s') is not None
        and (profile != 'step-snap' or maneuver.get('t_to_speed_below_thr_s') is not None)
        and continuous_p95 is not None and continuous_p95 <= target_continuous_power_p95_pct
        and flap_peak is not None and flap_peak <= target_flap_peak_pct
        and flap_tracking is not None and flap_tracking <= 2.0
        and fan_tracking_pct is not None and fan_tracking_pct <= target_fan_tracking_pct
        and fan_response_min_pct is not None and fan_response_min_pct >= 90.0
        and xy_tracking_rms is not None and xy_tracking_rms <= target_xy_tracking_rms_n
        and alignment_p95_deg is not None and alignment_p95_deg <= 20.0
        and yaw_hold_error is not None and yaw_hold_error <= target_yaw_coupling_deg
    )


def _maneuver_tune_penalty(
    assessment: dict,
    *,
    profile: str,
    target_continuous_power_p95_pct: float,
    target_flap_peak_pct: float,
    target_fan_tracking_pct: float,
    target_xy_tracking_rms_n: float,
    target_yaw_coupling_deg: float,
) -> float:
    engineering = assessment.get('engineering', {})
    maneuver = assessment.get('maneuver', {})
    thermal = engineering.get('thermal', {})
    flaps = engineering.get('flaps', {})
    fans = engineering.get('fans', {})
    vectoring = engineering.get('vectoring', {})

    continuous_p95 = _maneuver_scalar(thermal.get('continuous_power_p95_pct'))
    flap_peak = _maneuver_scalar(flaps.get('limit_usage_peak_pct'))
    flap_tracking = _maneuver_scalar(flaps.get('tracking_rms_deg'))
    fan_tracking_pct = _maneuver_scalar(fans.get('tracking_rms_pct_mean_cmd'))
    fan_response_min_pct = _maneuver_scalar(fans.get('response_min_pct'))
    xy_tracking_rms = _maneuver_scalar(vectoring.get('xy_tracking_rms_n'))
    alignment_p95_deg = _maneuver_scalar(vectoring.get('alignment_p95_deg'))
    yaw_hold_error = _maneuver_scalar(maneuver.get('yaw_hold_error_mean_abs_deg'))
    turn_delta = _maneuver_scalar(maneuver.get('turn_delta_deg'))
    continuous_p95 = 1e9 if continuous_p95 is None else continuous_p95
    flap_peak = 1e9 if flap_peak is None else flap_peak
    flap_tracking = 1e9 if flap_tracking is None else flap_tracking
    fan_tracking_pct = 1e9 if fan_tracking_pct is None else fan_tracking_pct
    xy_tracking_rms = 1e9 if xy_tracking_rms is None else xy_tracking_rms
    alignment_p95_deg = 1e9 if alignment_p95_deg is None else alignment_p95_deg
    yaw_hold_error = 1e9 if yaw_hold_error is None else yaw_hold_error
    turn_delta = 0.0 if turn_delta is None else turn_delta
    requires_reversal = (profile == 'step-snap') or turn_delta >= 135.0

    penalty = 0.0
    if requires_reversal and maneuver.get('t_reversal_s') is None:
        penalty += 4000.0
    if maneuver.get('t90_dir_s') is None:
        penalty += 4000.0
    if profile == 'step-snap' and maneuver.get('t_to_speed_below_thr_s') is None:
        penalty += 3500.0
    penalty += 2.5 * max(0.0, continuous_p95 - target_continuous_power_p95_pct) ** 2
    penalty += 0.90 * max(0.0, flap_peak - target_flap_peak_pct) ** 2
    penalty += 50.0 * max(0.0, flap_tracking - 2.0) ** 2
    penalty += 2.0 * max(0.0, fan_tracking_pct - target_fan_tracking_pct) ** 2
    if fan_response_min_pct is None:
        penalty += 500.0
    else:
        penalty += 4.0 * max(0.0, 90.0 - fan_response_min_pct) ** 2
    penalty += 0.02 * max(0.0, xy_tracking_rms - target_xy_tracking_rms_n) ** 2
    penalty += 1.5 * max(0.0, alignment_p95_deg - 20.0) ** 2
    penalty += 3.0 * max(0.0, yaw_hold_error - target_yaw_coupling_deg) ** 2
    return float(penalty)


def _evaluate_maneuver_aggressiveness(
    profile: str,
    base_kwargs: dict,
    aggressiveness: float,
    *,
    target_continuous_power_p95_pct: float,
    target_flap_peak_pct: float,
    target_fan_tracking_pct: float,
    target_xy_tracking_rms_n: float,
    target_yaw_coupling_deg: float,
) -> dict:
    params = _apply_maneuver_aggressiveness(base_kwargs, aggressiveness, profile)
    meta, hist = _run_maneuver_profile(profile, **params)
    assessment = assess_maneuver_result(meta, hist, profile)
    passed = _maneuver_tune_passes(
        assessment,
        profile=profile,
        target_continuous_power_p95_pct=target_continuous_power_p95_pct,
        target_flap_peak_pct=target_flap_peak_pct,
        target_fan_tracking_pct=target_fan_tracking_pct,
        target_xy_tracking_rms_n=target_xy_tracking_rms_n,
        target_yaw_coupling_deg=target_yaw_coupling_deg,
    )
    score = _maneuver_tune_penalty(
        assessment,
        profile=profile,
        target_continuous_power_p95_pct=target_continuous_power_p95_pct,
        target_flap_peak_pct=target_flap_peak_pct,
        target_fan_tracking_pct=target_fan_tracking_pct,
        target_xy_tracking_rms_n=target_xy_tracking_rms_n,
        target_yaw_coupling_deg=target_yaw_coupling_deg,
    )
    record = {
        'aggressiveness': float(aggressiveness),
        'passed': bool(passed),
        'score': float(score),
        'status': assessment.get('status'),
        'params': _maneuver_tune_snapshot(params, aggressiveness),
        't_to_speed_below_thr_s': _maneuver_scalar(assessment.get('maneuver', {}).get('t_to_speed_below_thr_s')),
        'snap_stop_distance_m': _maneuver_scalar(assessment.get('maneuver', {}).get('snap_stop_distance_m')),
        't_reversal_s': _maneuver_scalar(assessment.get('maneuver', {}).get('t_reversal_s')),
        't90_dir_s': _maneuver_scalar(assessment.get('maneuver', {}).get('t90_dir_s')),
        'yaw_hold_error_mean_abs_deg': _maneuver_scalar(assessment.get('maneuver', {}).get('yaw_hold_error_mean_abs_deg')),
        'yaw_track_decoupling_mean_abs_deg': _maneuver_scalar(assessment.get('maneuver', {}).get('yaw_track_decoupling_mean_abs_deg')),
        'continuous_power_p95_pct': _maneuver_scalar(assessment.get('engineering', {}).get('thermal', {}).get('continuous_power_p95_pct')),
        'flap_limit_usage_peak_pct': _maneuver_scalar(assessment.get('engineering', {}).get('flaps', {}).get('limit_usage_peak_pct')),
        'fan_tracking_rms_pct_mean_cmd': _maneuver_scalar(assessment.get('engineering', {}).get('fans', {}).get('tracking_rms_pct_mean_cmd')),
        'xy_tracking_rms_n': _maneuver_scalar(assessment.get('engineering', {}).get('vectoring', {}).get('xy_tracking_rms_n')),
        'warnings': list(assessment.get('warnings', [])),
    }
    return {
        'meta': meta,
        'hist': hist,
        'assessment': assessment,
        'record': record,
    }


def tune_maneuver_profile(
    profile: str,
    base_kwargs: dict,
    *,
    target_continuous_power_p95_pct: float = 100.0,
    target_flap_peak_pct: float = 90.0,
    target_fan_tracking_pct: float = 10.0,
    target_xy_tracking_rms_n: float = 400.0,
    target_yaw_coupling_deg: float = 15.0,
    min_aggressiveness: float = 0.20,
    max_iterations: int = 6,
) -> tuple[dict, dict, dict, dict]:
    min_aggressiveness = max(0.10, min(1.0, float(min_aggressiveness)))
    coarse_samples = max(4, min(10, int(max_iterations) + 2))
    coarse_values = np.linspace(1.0, min_aggressiveness, num=coarse_samples)

    evaluations: list[dict] = []
    bundles_by_key: dict[float, dict] = {}
    best_bundle = None
    best_pass_bundle = None

    def evaluate(g: float) -> dict:
        key = round(float(g), 6)
        bundle = bundles_by_key.get(key)
        if bundle is None:
            bundle = _evaluate_maneuver_aggressiveness(
                profile,
                base_kwargs,
                float(g),
                target_continuous_power_p95_pct=target_continuous_power_p95_pct,
                target_flap_peak_pct=target_flap_peak_pct,
                target_fan_tracking_pct=target_fan_tracking_pct,
                target_xy_tracking_rms_n=target_xy_tracking_rms_n,
                target_yaw_coupling_deg=target_yaw_coupling_deg,
            )
            bundles_by_key[key] = bundle
            evaluations.append(bundle['record'])
        return bundle

    def bundle_key(bundle: dict) -> tuple[float, float, float]:
        rec = bundle['record']
        return (
            float(_status_rank(rec.get('status'))),
            float(rec.get('score') or 0.0),
            -float(rec.get('aggressiveness') or 0.0),
        )

    coarse_bundles = [evaluate(g) for g in coarse_values]
    for bundle in coarse_bundles:
        if best_bundle is None or bundle_key(bundle) < bundle_key(best_bundle):
            best_bundle = bundle
        if bundle['record']['passed'] and (
            best_pass_bundle is None or bundle['record']['aggressiveness'] > best_pass_bundle['record']['aggressiveness']
        ):
            best_pass_bundle = bundle

    bracket_low = None
    bracket_high = None
    for idx in range(1, len(coarse_bundles)):
        upper = coarse_bundles[idx - 1]
        lower = coarse_bundles[idx]
        if (not upper['record']['passed']) and lower['record']['passed']:
            bracket_high = upper['record']['aggressiveness']
            bracket_low = lower['record']['aggressiveness']
            break

    if best_pass_bundle is not None and bracket_low is not None and bracket_high is not None:
        low_bundle = bundles_by_key[round(bracket_low, 6)]
        high_bundle = bundles_by_key[round(bracket_high, 6)]
        for _ in range(max(0, int(max_iterations))):
            mid = 0.5 * (low_bundle['record']['aggressiveness'] + high_bundle['record']['aggressiveness'])
            if abs(high_bundle['record']['aggressiveness'] - low_bundle['record']['aggressiveness']) < 0.01:
                break
            mid_bundle = evaluate(mid)
            if mid_bundle['record']['passed']:
                low_bundle = mid_bundle
                if best_pass_bundle is None or mid_bundle['record']['aggressiveness'] > best_pass_bundle['record']['aggressiveness']:
                    best_pass_bundle = mid_bundle
            else:
                high_bundle = mid_bundle
            if best_bundle is None or bundle_key(mid_bundle) < bundle_key(best_bundle):
                best_bundle = mid_bundle

    selected = best_pass_bundle or best_bundle
    iterations = sorted(evaluations, key=lambda item: item['aggressiveness'], reverse=True)
    feasible = best_pass_bundle is not None
    tuning = {
        'enabled': True,
        'mode': 'maneuver-envelope-search',
        'status': ('tuned' if feasible else 'best-effort'),
        'profile': profile,
        'targets': {
            'continuous_power_p95_pct': float(target_continuous_power_p95_pct),
            'flap_limit_usage_peak_pct': float(target_flap_peak_pct),
            'fan_tracking_rms_pct_mean_cmd': float(target_fan_tracking_pct),
            'xy_tracking_rms_n': float(target_xy_tracking_rms_n),
            'yaw_hold_error_mean_abs_deg': float(target_yaw_coupling_deg),
            'min_aggressiveness': float(min_aggressiveness),
            'coarse_samples': int(coarse_samples),
            'refine_iterations': int(max_iterations),
        },
        'baseline': iterations[0],
        'selected': selected['record'],
        'envelope': {
            'feasible': bool(feasible),
            'max_sustainable_aggressiveness': (best_pass_bundle['record']['aggressiveness'] if feasible else None),
            'margin_to_full_scale': (1.0 - best_pass_bundle['record']['aggressiveness'] if feasible else None),
        },
        'iterations': iterations,
    }
    return selected['meta'], selected['hist'], selected['assessment'], tuning


def render_maneuver_pack_markdown(payload: dict) -> str:
    lines = [
        f"# Maneuver Pack: {payload['profile']}",
        '',
        f"Angles: {', '.join(str(int(a)) if float(a).is_integer() else str(a) for a in payload['angles_deg'])}",
        f"Fault cases: {len(payload['fault_cases'])}",
        f"Mode: {'maneuver-safe' if payload.get('maneuver_safe') else 'raw'}",
        f"Worst status: {payload.get('worst_status', 'n/a')}",
        '',
        '| Status | Fault Case | Dir B (deg) | Aggr | Stop Time (s) | Reversal (s) | Align (s) | Yaw Hold Err (deg) | Flap Peak (%) | Fan Track (%) | Power P95 (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for rec in payload['results']:
        lines.append(
            '| {status} | {fault_case} | {dir_b_deg:.0f} | {aggr} | {stop} | {reversal} | {align} | {yaw} | {flap} | {fan} | {power} |'.format(
                status=rec['status'],
                fault_case=rec['fault_case'],
                dir_b_deg=rec['dir_b_deg'],
                aggr=('n/a' if rec.get('selected_aggressiveness') is None else f"{rec['selected_aggressiveness']:.3f}"),
                stop=('n/a' if rec['t_to_speed_below_thr_s'] is None else f"{rec['t_to_speed_below_thr_s']:.2f}"),
                reversal=('n/a' if rec['t_reversal_s'] is None else f"{rec['t_reversal_s']:.2f}"),
                align=('n/a' if rec['t90_dir_s'] is None else f"{rec['t90_dir_s']:.2f}"),
                yaw=('n/a' if rec.get('yaw_hold_error_mean_abs_deg') is None else f"{rec['yaw_hold_error_mean_abs_deg']:.2f}"),
                flap=('n/a' if rec['flap_limit_usage_peak_pct'] is None else f"{rec['flap_limit_usage_peak_pct']:.1f}"),
                fan=('n/a' if rec['fan_tracking_rms_pct_mean_cmd'] is None else f"{rec['fan_tracking_rms_pct_mean_cmd']:.1f}"),
                power=('n/a' if rec['continuous_power_p95_pct'] is None else f"{rec['continuous_power_p95_pct']:.1f}"),
            )
        )
    return '\n'.join(lines) + '\n'


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


def _stringify_table_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip('0').rstrip('.')
    return str(value)


def infer_requirements_table_format(path: str, format_name: str) -> str:
    if format_name != "auto":
        return format_name
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "text"


def render_requirements_table(
    rows: list[dict],
    *,
    format_name: str,
    preset: str,
) -> str:
    flat_rows = [flatten_requirements_row(row) for row in rows]
    headers = [label for label, _key in REQUIREMENTS_TABLE_COLUMNS]
    keys = [key for _label, key in REQUIREMENTS_TABLE_COLUMNS]

    if format_name == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        for row in flat_rows:
            writer.writerow([_stringify_table_value(row.get(key)) for key in keys])
        return buffer.getvalue()

    matrix = [headers]
    for row in flat_rows:
        matrix.append([_stringify_table_value(row.get(key)) for key in keys])

    widths = [
        max(len(matrix[r][c]) for r in range(len(matrix)))
        for c in range(len(headers))
    ]

    if format_name == "markdown":
        def format_row(values: list[str]) -> str:
            return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

        lines = [
            f"# {preset} power requirements",
            "",
            format_row(headers),
            "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
        ]
        for row in matrix[1:]:
            lines.append(format_row(row))
        lines.append("")
        return "\n".join(lines)

    def format_plain_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [
        f"{preset} power requirements",
        format_plain_row(headers),
        format_plain_row(["-" * width for width in widths]),
    ]
    for row in matrix[1:]:
        lines.append(format_plain_row(row))
    lines.append("")
    return "\n".join(lines)


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

MP_GRAPH_PRESETS = {
    "overview": {
        "description": "Core mission and energy channels for quick health checks.",
        "channels": ["speed", "goal_m", "obs_m", "bat_v", "soc_pct", "pwr_kw", "thr_pct"],
    },
    "battery-power": {
        "description": "Battery, power draw, remaining energy, and thrust derating.",
        "channels": ["bat_v", "bat_a", "soc_pct", "pwr_kw", "eng_wh", "thr_pct"],
    },
    "thermal-power": {
        "description": "Continuous power pressure, thermal headroom, and fan response.",
        "channels": ["cont_pct", "pwr_marg", "therm_pct", "temp_max", "temp_avg", "rsp_pct"],
    },
    "fan-tracking": {
        "description": "Commanded versus achieved fan thrust summaries.",
        "channels": ["cmd_avg", "fan_avg", "cmd_stdp", "fan_stdp", "cmd_maxp", "fan_maxp"],
    },
    "force-tracking": {
        "description": "Requested versus achieved vehicle force in XYZ.",
        "channels": ["fx_cmd", "fx_ach", "fx_err", "fy_cmd", "fy_ach", "fy_err", "fz_cmd", "fz_ach", "fz_err"],
    },
    "flap-tracking": {
        "description": "Commanded versus achieved flap summary and sampled segments.",
        "channels": ["a_cmd", "a_rms", "c0", "a0", "c2", "a2", "c4", "a4", "c6", "a6"],
    },
}

for fan_start in range(0, 16, 8):
    fan_stop = fan_start + 7
    MP_GRAPH_PRESETS[f"fan-actual-{fan_start}-{fan_stop}"] = {
        "description": f"Actual thrust for fans {fan_start}-{fan_stop}.",
        "channels": [f"f{idx:02d}" for idx in range(fan_start, fan_stop + 1)],
    }
    MP_GRAPH_PRESETS[f"fan-command-{fan_start}-{fan_stop}"] = {
        "description": f"Commanded thrust for fans {fan_start}-{fan_stop}.",
        "channels": [f"fc{idx:02d}" for idx in range(fan_start, fan_stop + 1)],
    }

for flap_start in range(0, 32, 8):
    flap_stop = flap_start + 7
    MP_GRAPH_PRESETS[f"flap-actual-{flap_start}-{flap_stop}"] = {
        "description": f"Actual flap angles for segments {flap_start}-{flap_stop}.",
        "channels": [f"a{idx:02d}" for idx in range(flap_start, flap_stop + 1)],
    }
    MP_GRAPH_PRESETS[f"flap-command-{flap_start}-{flap_stop}"] = {
        "description": f"Commanded flap angles for segments {flap_start}-{flap_stop}.",
        "channels": [f"c{idx:02d}" for idx in range(flap_start, flap_stop + 1)],
    }

MP_GRAPH_CHANNEL_SOURCES = {
    "speed": "speed",
    "goal_m": "dist_to_goal",
    "obs_m": "nearest_obstacle_m",
    "bat_v": "battery_v",
    "bat_a": "battery_a",
    "soc_pct": "battery_soc",
    "pwr_kw": "power_w",
    "eng_wh": "energy_wh",
    "thr_pct": "thrust_scale_pct",
    "cont_pct": "continuous_power_pct",
    "pwr_marg": "power_margin_kw",
    "therm_pct": "thermal_scale_pct",
    "temp_max": "fan_temp_max_c",
    "temp_avg": "fan_temp_mean_c",
    "rsp_pct": "fan_response_pct",
    "cmd_avg": "fan_cmd_16",
    "cmd_stdp": "fan_cmd_16",
    "cmd_maxp": "fan_cmd_16",
    "fan_avg": "fan_thrust_16",
    "fan_stdp": "fan_thrust_16",
    "fan_maxp": "fan_thrust_16",
    "fx_cmd": "fx_cmd",
    "fy_cmd": "fy_cmd",
    "fz_cmd": "fz_cmd",
    "fx_ach": "fx_ach",
    "fy_ach": "fy_ach",
    "fz_ach": "fz_ach",
    "fx_err": "fx_err",
    "fy_err": "fy_err",
    "fz_err": "fz_err",
    "a_cmd": "alpha_cmd_rms",
    "a_rms": "alpha_deg_rms",
    "c0": "alpha_cmd_deg_32",
    "c2": "alpha_cmd_deg_32",
    "c4": "alpha_cmd_deg_32",
    "c6": "alpha_cmd_deg_32",
    "a0": "alpha_deg_32",
    "a2": "alpha_deg_32",
    "a4": "alpha_deg_32",
    "a6": "alpha_deg_32",
}

for fan_idx in range(16):
    MP_GRAPH_CHANNEL_SOURCES[f"f{fan_idx:02d}"] = "fan_thrust_16"
    MP_GRAPH_CHANNEL_SOURCES[f"fc{fan_idx:02d}"] = "fan_cmd_16"

for flap_idx in range(32):
    MP_GRAPH_CHANNEL_SOURCES[f"a{flap_idx:02d}"] = "alpha_deg_32"
    MP_GRAPH_CHANNEL_SOURCES[f"c{flap_idx:02d}"] = "alpha_cmd_deg_32"



def mp_graph_channel_available(hist: dict, channel: str) -> bool:
    source = MP_GRAPH_CHANNEL_SOURCES.get(channel)
    if source is None:
        return False
    values = hist.get(source, [])
    return bool(values)


def mp_graph_ui_label(channel: str) -> str:
    return f"MAV_{channel.upper()}"


def build_mp_graph_preset_report(hist: dict | None = None, preset_name: str | None = None) -> dict:
    selected = MP_GRAPH_PRESETS if preset_name is None else {preset_name: MP_GRAPH_PRESETS[preset_name]}
    report = {}
    for name, spec in selected.items():
        channels = list(spec["channels"])
        entry = {
            "description": spec["description"],
            "channels": channels,
            "paste": ", ".join(channels),
        }
        if hist is not None:
            available = [channel for channel in channels if mp_graph_channel_available(hist, channel)]
            missing = [channel for channel in channels if channel not in available]
            entry["available"] = available
            entry["missing"] = missing
        report[name] = entry
    return report


def render_mp_graph_preset_report(report: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2)
    if fmt == "paste":
        lines = [entry["paste"] for entry in report.values()]
        return "\n".join(lines) + "\n"
    if fmt == "mp-ui":
        lines = [f"{name}: {', '.join(mp_graph_ui_label(channel) for channel in entry['channels'])}" for name, entry in report.items()]
        return "\n".join(lines) + "\n"

    lines: list[str] = []
    if fmt == "md":
        lines.append("# Mission Planner Graph Presets")
    else:
        lines.append("Mission Planner Graph Presets")
    for name, entry in report.items():
        if fmt == "md":
            lines.append("")
            lines.append(f"## {name}")
            lines.append(entry["description"])
            lines.append("")
            lines.append(f"Paste: `{entry['paste']}`")
            if "available" in entry:
                lines.append("")
                lines.append(f"Available: `{', '.join(entry['available'])}`" if entry["available"] else "Available: none")
                if entry["missing"]:
                    lines.append(f"Missing: `{', '.join(entry['missing'])}`")
        else:
            lines.append("")
            lines.append(f"[{name}]")
            lines.append(f"description = {entry['description']}")
            lines.append(f"paste = {entry['paste']}")
            if "available" in entry:
                lines.append(f"available = {', '.join(entry['available'])}")
                lines.append(f"missing = {', '.join(entry['missing'])}")
    return "\n".join(lines) + "\n"


@alloc_app.command("assess")
def alloc_assess(
    trace: str = typer.Option(..., "--trace", help="Trace JSON path to assess"),
    out: str = typer.Option("", "--out", help="Optional output path for the rendered assessment"),
    format: str = typer.Option("text", "--format", help="Output format: json, text, or md"),
):
    fmt = format.lower().strip()
    if fmt not in {"json", "text", "md"}:
        raise typer.BadParameter("--format must be one of: json, text, md")
    meta, hist = load_trace_json(trace)
    report = assess_trace(meta, hist)
    rendered = render_assessment(report, fmt)
    if out:
        Path(out).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@bridge_app.command("inspect")
def bridge_inspect(
    script: str = typer.Option(..., "--script", help="Bridge JSONL setpoint script path"),
    svg_out: str = typer.Option("", "--svg-out", help="Optional SVG top-down path output"),
    fail_on_issues: bool = typer.Option(False, "--fail-on-issues/--no-fail-on-issues", help="Exit non-zero if validation finds issues"),
):
    setpoints = load_script_jsonl(script)
    report = inspect_script(setpoints)
    if svg_out:
        write_script_svg(svg_out, setpoints, title=Path(script).name)
        report["svg_out"] = svg_out
    typer.echo(json.dumps(report, indent=2))
    if fail_on_issues and not report.get("ok", False):
        raise typer.Exit(code=1)


@bridge_app.command("mp-graphs")
def bridge_mp_graphs(
    trace: str = typer.Option("", "--trace", help="Optional trace JSON path to validate which graph channels are available"),
    preset: str = typer.Option("", "--preset", help="Optional preset name to output by itself"),
    out: str = typer.Option("", "--out", help="Optional output path for the rendered preset list"),
    format: str = typer.Option("json", "--format", help="Output format: json, text, or md"),
):
    fmt = format.lower().strip()
    if fmt not in {"json", "text", "md", "paste", "mp-ui"}:
        raise typer.BadParameter("--format must be one of: json, text, md, paste, mp-ui")
    preset_name = preset.strip() or None
    if preset_name is not None and preset_name not in MP_GRAPH_PRESETS:
        raise typer.BadParameter(f"unknown preset: {preset_name}")

    hist = None
    trace_path = trace.strip()
    if trace_path:
        _meta, hist = load_trace_json(trace_path)

    preset_report = build_mp_graph_preset_report(hist=hist, preset_name=preset_name)
    report = {
        "trace": (trace_path or None),
        "presets": preset_report,
    }
    rendered = render_mp_graph_preset_report(preset_report, fmt)
    if out:
        Path(out).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@bridge_app.command("trace")
def bridge_trace(
    trace: str = typer.Option(..., "--trace", help="Planned trace JSON path"),
    jsonl_out: str = typer.Option("", "--jsonl-out", help="Optional JSONL setpoint script output path"),
    mavlink_out: str = typer.Option("", "--mavlink-out", help="Optional MAVLink connection string for live setpoint streaming"),
    hold_final_s: float = typer.Option(2.0, "--hold-final-s", help="How long to hold the final setpoint after the trace ends (s)"),
    rate_hz: float = typer.Option(20.0, "--rate-hz", help="Bridge output rate in Hz; use <=0 for native trace rate"),
    realtime: bool = typer.Option(True, "--realtime/--no-realtime", help="When streaming MAVLink, honor trace timing or send as fast as possible"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Print a bridge summary even when outputs are requested"),
    target_system: int = typer.Option(1, "--target-system", help="Target MAVLink system id for live setpoint streaming"),
    target_component: int = typer.Option(1, "--target-component", help="Target MAVLink component id for live setpoint streaming"),
    source_system: int = typer.Option(245, "--source-system", help="Source MAVLink system id for the bridge"),
    source_component: int = typer.Option(190, "--source-component", help="Source MAVLink component id for the bridge"),
    heartbeat_hz: float = typer.Option(1.0, "--heartbeat-hz", help="Heartbeat rate while streaming MAVLink setpoints"),
):
    meta, hist = load_trace_json(trace)
    script = build_setpoint_script(
        meta,
        hist,
        hold_final_s=hold_final_s,
        rate_hz=(None if rate_hz <= 0.0 else rate_hz),
    )
    summary = summarize_script(meta, script)

    if jsonl_out:
        write_script_jsonl(jsonl_out, script)
        summary["jsonl_out"] = jsonl_out

    if dry_run or (not jsonl_out and not mavlink_out):
        typer.echo(json.dumps(summary, indent=2))

    if mavlink_out:
        stream_script_mavlink(
            script,
            out=mavlink_out,
            target_system=target_system,
            target_component=target_component,
            source_system=source_system,
            source_component=source_component,
            heartbeat_hz=heartbeat_hz,
            realtime=realtime,
        )



@bridge_app.command("mp")
def bridge_mp(
    trace: str = typer.Option(..., "--trace", help="Planned trace JSON path"),
    out: str = typer.Option(..., "--out", help="Mission Planner mission file output path (.waypoints/.txt)"),
    home_lat: float = typer.Option(..., "--home-lat", help="Home latitude in degrees"),
    home_lon: float = typer.Option(..., "--home-lon", help="Home longitude in degrees"),
    home_alt_m: float = typer.Option(0.0, "--home-alt-m", help="Home altitude reference in meters"),
    takeoff_alt_m: float = typer.Option(0.0, "--takeoff-alt-m", help="Takeoff altitude for the MP mission. 0 uses the trace cruise altitude"),
    cruise_alt_m: float = typer.Option(0.0, "--cruise-alt-m", help="Cruise altitude override for mission waypoints"),
    waypoint_spacing_m: float = typer.Option(10.0, "--waypoint-spacing-m", help="Route spacing when sampling mission waypoints from the trace"),
    trace_route: bool = typer.Option(False, "--trace-route/--planner-route", help="Use trace-sampled waypoints instead of the default interpolated planner route"),
    spline: bool = typer.Option(True, "--spline/--no-spline", help="Use SPLINE_WAYPOINT for the route to reduce stop-and-go behavior in AUTO"),
    speed_mps: float = typer.Option(8.0, "--speed-mps", help="Mission speed command in m/s for AUTO"),
    acceptance_radius_m: float = typer.Option(4.0, "--acceptance-radius-m", help="Waypoint acceptance radius in meters for the MP mission"),
    approach_distance_m: float = typer.Option(18.0, "--approach-distance-m", help="Distance before destination to insert a lower approach waypoint"),
    descent_alt_m: float = typer.Option(0.0, "--descent-alt-m", help="Final approach altitude before LAND. 0 uses an automatic descent altitude"),
    land: bool = typer.Option(True, "--land/--no-land", help="Append a LAND command at the destination"),
):
    meta, hist = load_trace_json(trace)
    mission_items = build_qgc_wpl_mission(
        meta,
        hist,
        home_lat_deg=home_lat,
        home_lon_deg=home_lon,
        home_alt_m=home_alt_m,
        takeoff_alt_m=takeoff_alt_m,
        cruise_alt_m=cruise_alt_m,
        waypoint_spacing_m=waypoint_spacing_m,
        land=land,
        prefer_trace_points=trace_route,
        approach_distance_m=approach_distance_m,
        descent_alt_m=descent_alt_m,
        use_spline=spline,
        speed_mps=speed_mps,
        acceptance_radius_m=acceptance_radius_m,
    )
    write_qgc_wpl_mission(out, mission_items)
    typer.echo(json.dumps(summarize_qgc_wpl_mission(mission_items, out_path=out, home_lat_deg=home_lat, home_lon_deg=home_lon), indent=2))


@bridge_app.command("sitl")
def bridge_sitl(
    trace: str = typer.Option("", "--trace", help="Planned trace JSON path to convert into setpoints"),
    script: str = typer.Option("", "--script", help="Existing bridge JSONL setpoint script path"),
    jsonl_out: str = typer.Option("", "--jsonl-out", help="Optional JSONL output when building from --trace"),
    connect: str = typer.Option("tcp:127.0.0.1:5760", "--connect", help="ArduPilot SITL MAVLink connection string"),
    hold_final_s: float = typer.Option(2.0, "--hold-final-s", help="How long to hold the final setpoint after the trace ends (s)"),
    rate_hz: float = typer.Option(20.0, "--rate-hz", help="Bridge output rate in Hz; use <=0 for native trace rate"),
    realtime: bool = typer.Option(True, "--realtime/--no-realtime", help="Honor trace timing or send setpoints as fast as possible"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Print the resolved SITL bridge plan before executing"),
    execute: bool = typer.Option(False, "--execute/--no-execute", help="Actually connect to SITL after printing the resolved plan"),
    target_system: int = typer.Option(0, "--target-system", help="Target MAVLink system id. 0 = auto from heartbeat"),
    target_component: int = typer.Option(0, "--target-component", help="Target MAVLink component id. 0 = auto from heartbeat"),
    source_system: int = typer.Option(245, "--source-system", help="Source MAVLink system id for the bridge"),
    source_component: int = typer.Option(190, "--source-component", help="Source MAVLink component id for the bridge"),
    heartbeat_hz: float = typer.Option(1.0, "--heartbeat-hz", help="Heartbeat rate while streaming MAVLink setpoints"),
    wait_heartbeat_s: float = typer.Option(30.0, "--wait-heartbeat-s", help="How long to wait for the SITL heartbeat"),
    mode: str = typer.Option("GUIDED", "--mode", help="ArduPilot mode to enter before streaming; empty skips the mode change"),
    mode_timeout_s: float = typer.Option(10.0, "--mode-timeout-s", help="How long to wait for the requested mode"),
    arm: bool = typer.Option(True, "--arm/--no-arm", help="Arm SITL before streaming the setpoint script"),
    arm_timeout_s: float = typer.Option(15.0, "--arm-timeout-s", help="How long to wait for arming"),
    takeoff_alt_m: float = typer.Option(0.0, "--takeoff-alt-m", help="Optional GUIDED takeoff altitude before streaming; 0 disables takeoff"),
    takeoff_timeout_s: float = typer.Option(45.0, "--takeoff-timeout-s", help="How long to wait for the takeoff altitude"),
    takeoff_strict: bool = typer.Option(False, "--takeoff-strict/--no-takeoff-strict", help="Fail if SITL does not confirm climbing after the takeoff command"),
    land_mode: str = typer.Option("", "--land-mode", help="Optional mode to switch to after the script finishes, e.g. LAND"),
    land_timeout_s: float = typer.Option(20.0, "--land-timeout-s", help="How long to wait for the post-run mode change"),
    disarm: bool = typer.Option(False, "--disarm/--no-disarm", help="Disarm after the run finishes"),
    disarm_timeout_s: float = typer.Option(15.0, "--disarm-timeout-s", help="How long to wait for disarming"),
):
    if bool(trace) == bool(script):
        raise typer.BadParameter("provide exactly one of --trace or --script")
    if script and jsonl_out:
        raise typer.BadParameter("--jsonl-out can only be used with --trace")

    if trace:
        meta, hist = load_trace_json(trace)
        setpoints = build_setpoint_script(
            meta,
            hist,
            hold_final_s=hold_final_s,
            rate_hz=(None if rate_hz <= 0.0 else rate_hz),
        )
        summary = summarize_script(meta, setpoints)
        summary["input"] = {"trace": trace}
        if jsonl_out:
            write_script_jsonl(jsonl_out, setpoints)
            summary["jsonl_out"] = jsonl_out
    else:
        setpoints = load_script_jsonl(script)
        summary = summarize_setpoints(setpoints)
        summary["input"] = {"script": script}

    summary["sitl"] = {
        "connect": connect,
        "target_system": (target_system or "auto"),
        "target_component": (target_component or "auto"),
        "source_system": source_system,
        "source_component": source_component,
        "mode": mode or None,
        "arm": arm,
        "takeoff_alt_m": takeoff_alt_m,
        "takeoff_strict": takeoff_strict,
        "land_mode": land_mode or None,
        "disarm": disarm,
        "realtime": realtime,
        "execute": execute,
    }

    if dry_run or not execute:
        typer.echo(json.dumps(summary, indent=2))
    if not execute:
        return

    result = stream_script_ardupilot_sitl(
        setpoints,
        connect=connect,
        target_system=target_system,
        target_component=target_component,
        source_system=source_system,
        source_component=source_component,
        heartbeat_hz=heartbeat_hz,
        realtime=realtime,
        wait_heartbeat_s=wait_heartbeat_s,
        mode=mode,
        mode_timeout_s=mode_timeout_s,
        arm=arm,
        arm_timeout_s=arm_timeout_s,
        takeoff_alt_m=takeoff_alt_m,
        takeoff_timeout_s=takeoff_timeout_s,
        takeoff_strict=takeoff_strict,
        land_mode=land_mode,
        land_timeout_s=land_timeout_s,
        disarm=disarm,
        disarm_timeout_s=disarm_timeout_s,
    )
    typer.echo(json.dumps({"sitl_result": result}, indent=2))

def parse_mission_obstacles(specs: list[str]) -> list[MissionObstacle]:
    obstacles: list[MissionObstacle] = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) < 3 or len(parts) > 6:
            raise typer.BadParameter("--obstacle must be x,y,radius[,k[,swirl[,influence]]]")
        try:
            x_m = float(parts[0])
            y_m = float(parts[1])
            radius_m = float(parts[2])
            k_n_per_m = float(parts[3]) if len(parts) >= 4 else 180.0
            swirl_n = float(parts[4]) if len(parts) >= 5 else 900.0
            influence_m = float(parts[5]) if len(parts) >= 6 else max(20.0, 1.5 * radius_m)
        except ValueError as exc:
            raise typer.BadParameter(f"invalid --obstacle '{spec}'") from exc
        obstacles.append(MissionObstacle(x_m=x_m, y_m=y_m, radius_m=radius_m, k_n_per_m=k_n_per_m, swirl_n=swirl_n, influence_m=influence_m))
    return obstacles


@alloc_app.command("demo")
def alloc_demo(
    dir_deg: float = typer.Option(90.0, "--dir-deg", help="Direction of travel in BODY frame degrees (0=+X)"),
    fxy_n: float = typer.Option(3000.0, "--fxy", help="Desired lateral force magnitude (N)"),
    duration_s: float = typer.Option(6.0, "--duration-s", help="Sim duration (s)"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg", help="Yaw hold (deg)"),
    version: str = typer.Option("v2", "--version", help="v1 or v2"),
    mz_nm: float = typer.Option(0.0, "--mz-nm", help="Desired yaw moment (N*m). 0 = near-zero yaw trim"),
):
    hist = run_demo(
        dir_deg=dir_deg,
        fxy_n=fxy_n,
        duration_s=duration_s,
        yaw_hold_deg=yaw_hold_deg,
        mz_nm=mz_nm,
        version=version,
    )
    coupling = yaw_track_coupling_mean_abs(hist)
    out = {
        "allocator_version": version,
        "dir_deg": dir_deg,
        "fxy_n": fxy_n,
        "mz_nm_cmd": mz_nm,
        "duration_s": duration_s,
        "yaw_hold_deg": yaw_hold_deg,
        "yaw_track_coupling_mean_abs_deg": coupling,
        "final": {
            "x_m": hist["x"][-1],
            "y_m": hist["y"][-1],
            "vx_mps": hist["vx"][-1],
            "vy_mps": hist["vy"][-1],
            "alpha_deg_rms": hist["alpha_deg_rms"][-1],
            "ft_tan_rms": hist.get("ft_tan_rms", [0])[-1],
            "mz_est_nm": hist["mz_est"][-1],
        },
    }
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("step")
def alloc_step(
    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg", help="Direction before step (deg)"),
    dir_b_deg: float = typer.Option(180.0, "--dir-b-deg", help="Direction after step (deg)"),
    fxy_n: float = typer.Option(3000.0, "--fxy", help="Lateral force magnitude (N)"),
    step_time_s: float = typer.Option(3.0, "--step-time-s", help="Time of direction change (s)"),
    total_s: float = typer.Option(8.0, "--total-s", help="Total sim time (s)"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm", help="Desired yaw moment (N*m). 0 = near-zero yaw trim"),
    alpha_rate_deg_s: float = typer.Option(200.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.12, "--plenum-tau-s"),
    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or slow-flap-0"),
    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),
    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),
    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),
    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),
    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),
    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),
):
    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)
    fault = build_fault_spec(normalized_fault_case)
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    base_kwargs = {
        'dir_a_deg': dir_a_deg,
        'dir_b_deg': dir_b_deg,
        'fxy_n': fxy_n,
        'step_time_s': step_time_s,
        'total_s': total_s,
        'yaw_hold_deg': yaw_hold_deg,
        'mz_nm': mz_nm,
        'lim': lim,
        'pl': pl,
        'fault': fault,
    }
    if maneuver_safe:
        out, _hist, assessment, tuning = tune_maneuver_profile(
            'step',
            base_kwargs,
            target_continuous_power_p95_pct=power_target_pct,
            target_flap_peak_pct=flap_target_pct,
            target_fan_tracking_pct=fan_target_pct,
            target_xy_tracking_rms_n=xy_target_n,
            target_yaw_coupling_deg=yaw_target_deg,
            min_aggressiveness=tune_min_aggressiveness,
            max_iterations=tune_iterations,
        )
        out['maneuver_tuning'] = tuning
    else:
        out, _hist = _run_maneuver_profile('step', **base_kwargs)
        assessment = assess_maneuver_result(out, _hist, 'step')
    out['fault'] = summarize_fault_case(normalized_fault_case, fault)
    out['maneuver_assessment'] = assessment
    if trace_out:
        save_trace_json(trace_out, meta=out, hist=_hist)
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("coordinates")
def alloc_coordinates(
    preset: str = typer.Option("", "--preset", help="Mission preset: medium, long, or very-long"),
    dest_x_m: Optional[float] = typer.Option(None, "--dest-x", help="Destination east offset from home (m)"),
    dest_y_m: Optional[float] = typer.Option(None, "--dest-y", help="Destination north offset from home (m)"),
    dest_z_m: float = typer.Option(0.0, "--dest-z", help="Destination landing altitude relative to home (m)"),
    start_x_m: float = typer.Option(0.0, "--start-x", help="Start east offset from home (m)"),
    start_y_m: float = typer.Option(0.0, "--start-y", help="Start north offset from home (m)"),
    start_z_m: float = typer.Option(0.0, "--start-z", help="Start altitude relative to home (m)"),
    total_s: float = typer.Option(30.0, "--total-s", help="Mission sim duration (s)"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm", help="Desired yaw moment (N*m). Keep 0 for UFO-style no-yaw motion"),
    cruise_alt_m: float = typer.Option(12.0, "--cruise-alt-m", help="Transit altitude before final approach (m)"),
    max_speed_mps: float = typer.Option(8.0, "--max-speed-mps", help="Horizontal cruise speed cap (m/s)"),
    arrival_radius_m: float = typer.Option(1.5, "--arrival-radius-m", help="Horizontal arrival radius (m)"),
    slow_radius_m: float = typer.Option(14.0, "--slow-radius-m", help="Radius where the craft starts slowing for arrival (m)"),
    descent_radius_m: float = typer.Option(18.0, "--descent-radius-m", help="Radius where the craft starts descending to the landing altitude (m)"),
    hold_s: float = typer.Option(2.0, "--hold-s", help="How long to hold at the destination before ending the trace (s)"),
    fxy_max_n: float = typer.Option(3500.0, "--fxy-max", help="Maximum horizontal force command (N)"),
    pos_k_n_per_m: float = typer.Option(120.0, "--pos-k", help="Position guidance gain (N/m)"),
    vel_k_n_per_mps: float = typer.Option(900.0, "--vel-k", help="Velocity damping gain (N per m/s)"),
    obstacle: list[str] = typer.Option([], "--obstacle", help="Obstacle as x,y,radius[,k[,swirl[,influence]]]. Repeat for multiple obstacles."),
    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),
    power_safe: bool = typer.Option(False, "--power-safe/--no-power-safe", help="Auto-tune mission aggressiveness against sustained power and flap targets"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --power-safe is enabled"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --power-safe is enabled"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --power-safe envelope search"),
    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the power-safe search is allowed to test"),
    battery_full_v: Optional[float] = typer.Option(None, "--battery-full-v", help="Override full-pack voltage for the power model"),
    battery_empty_v: Optional[float] = typer.Option(None, "--battery-empty-v", help="Override empty-pack voltage for the power model"),
    capacity_kwh: Optional[float] = typer.Option(None, "--capacity-kwh", help="Override battery capacity for the power model"),
    internal_resistance_ohm: Optional[float] = typer.Option(None, "--internal-resistance-ohm", help="Override pack internal resistance for the power model"),
    hover_power_kw: Optional[float] = typer.Option(None, "--hover-power-kw", help="Override hover power draw for the power model"),
    continuous_power_kw: Optional[float] = typer.Option(None, "--continuous-power-kw", help="Override continuous power capability for the power model"),
    peak_power_kw: Optional[float] = typer.Option(None, "--peak-power-kw", help="Override peak power capability for the power model"),
    aux_power_kw: Optional[float] = typer.Option(None, "--aux-power-kw", help="Override auxiliary power draw for the power model"),
    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or dead-fan-0+stuck-flap-0"),
    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),
):
    preset_cfg = {}
    if preset:
        if preset not in COORDINATE_PRESETS:
            raise typer.BadParameter("--preset must be one of: medium, long, very-long")
        preset_cfg = COORDINATE_PRESETS[preset]
    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)
    fault_case = normalized_fault_case

    def pick(current, key, default):
        if current is None:
            return preset_cfg.get(key, default)
        if preset_cfg and current == default:
            return preset_cfg.get(key, current)
        return current

    dest_x_m = pick(dest_x_m, "dest_x_m", None)
    dest_y_m = pick(dest_y_m, "dest_y_m", None)
    if dest_x_m is None or dest_y_m is None:
        raise typer.BadParameter("--dest-x and --dest-y are required unless --preset is used")

    dest_z_m = pick(dest_z_m, "dest_z_m", 0.0)
    total_s = pick(total_s, "total_s", 30.0)
    cruise_alt_m = pick(cruise_alt_m, "cruise_alt_m", 12.0)
    max_speed_mps = pick(max_speed_mps, "max_speed_mps", 8.0)
    arrival_radius_m = pick(arrival_radius_m, "arrival_radius_m", 1.5)
    slow_radius_m = pick(slow_radius_m, "slow_radius_m", 14.0)
    descent_radius_m = pick(descent_radius_m, "descent_radius_m", 18.0)
    hold_s = pick(hold_s, "hold_s", 2.0)
    fxy_max_n = pick(fxy_max_n, "fxy_max_n", 3500.0)
    pos_k_n_per_m = pick(pos_k_n_per_m, "pos_k_n_per_m", 120.0)
    vel_k_n_per_mps = pick(vel_k_n_per_mps, "vel_k_n_per_mps", 900.0)
    if preset_cfg and not obstacle:
        obstacle = list(preset_cfg.get("obstacles", []))

    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    obstacles = parse_mission_obstacles(obstacle)
    power = build_power_params(
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
    )
    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)
    fault = build_fault_spec(normalized_fault_case)

    mission_kwargs = build_coordinate_mission_kwargs(
        dest_x_m=dest_x_m,
        dest_y_m=dest_y_m,
        dest_z_m=dest_z_m,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        start_z_m=start_z_m,
        total_s=total_s,
        yaw_hold_deg=yaw_hold_deg,
        mz_nm=mz_nm,
        cruise_alt_m=cruise_alt_m,
        max_speed_mps=max_speed_mps,
        arrival_radius_m=arrival_radius_m,
        slow_radius_m=slow_radius_m,
        descent_radius_m=descent_radius_m,
        hold_s=hold_s,
        fxy_max_n=fxy_max_n,
        pos_k_n_per_m=pos_k_n_per_m,
        vel_k_n_per_mps=vel_k_n_per_mps,
        obstacles=obstacles,
        lim=lim,
        pl=pl,
        power=power,
        fault=fault,
    )

    if power_safe:
        out, _hist, report, tuning = tune_coordinate_mission(
            mission_kwargs,
            target_continuous_power_p95_pct=power_target_pct,
            target_flap_peak_pct=flap_target_pct,
            min_aggressiveness=tune_min_aggressiveness,
            max_iterations=tune_iterations,
        )
        tuning["report_status"] = report.get("status")
        tuning["report_warnings"] = list(report.get("warnings", []))
        out["tuning"] = tuning
    else:
        out, _hist = run_coordinate_mission_v5(**mission_kwargs)
    if preset:
        out.setdefault("mission", {})["preset"] = preset
    out["power_model"] = summarize_power_params(power)
    out["fault_case"] = summarize_fault_case(normalized_fault_case, fault)
    if trace_out:
        save_trace_json(trace_out, meta=out, hist=_hist)
    typer.echo(json.dumps(out, indent=2))


def build_power_sweep_report_for_fault(
    *,
    preset: str,
    fault_label: str,
    fault: FaultSpec,
    fault_summary: dict | None,
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
    if preset not in COORDINATE_PRESETS:
        raise typer.BadParameter('--preset must be one of: medium, long, very-long')

    preset_cfg = COORDINATE_PRESETS[preset]
    default_power = PowerSystemParams()
    lim = ActuatorLimits(alpha_rate_deg_s=350.0)
    pl = PlenumModel(tau_s=0.08)
    obstacles = parse_mission_obstacles(list(preset_cfg.get('obstacles', [])))

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
    summary_payload = dict(fault_summary) if fault_summary is not None else summarize_fault_case(fault_label, fault)
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


def build_power_sweep_report(
    *,
    preset: str,
    fault_case: str,
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
    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)
    fault = build_fault_spec(normalized_fault_case)
    fault_summary = summarize_fault_case(normalized_fault_case, fault)
    return build_power_sweep_report_for_fault(
        preset=preset,
        fault_label=normalized_fault_case,
        fault=fault,
        fault_summary=fault_summary,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        top=top,
    )


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


def render_fault_envelope_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in FAULT_ENVELOPE_COLUMNS]
    keys = [key for _label, key in FAULT_ENVELOPE_COLUMNS]

    def normalize(row: dict) -> list[str]:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                value = ','.join(str(item) for item in value)
            values.append(_stringify_table_value(value))
        return values

    matrix = [headers] + [normalize(row) for row in rows]
    widths = [max(len(matrix[r][c]) for r in range(len(matrix))) for c in range(len(headers))]

    if format_name == 'csv':
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(headers)
        for row in matrix[1:] :
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


def _fault_case_slug(label: str) -> str:
    chars = []
    for char in label.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append('-')
    slug = ''.join(chars)
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug.strip('-') or 'fault-case'


def _fault_value_token(value: float) -> str:
    text = f'{float(value):.2f}'.rstrip('0').rstrip('.')
    return text.replace('-', 'm').replace('.', 'p')


def _fault_index_values(values: list[int], *, count: int, step: int, label: str) -> list[int]:
    if values:
        indices = sorted({int(value) for value in values})
    else:
        stride = max(1, int(step))
        indices = list(range(0, count, stride))
    invalid = [value for value in indices if value < 0 or value >= count]
    if invalid:
        raise typer.BadParameter(f'{label} must be between 0 and {count - 1}')
    return indices


def _fault_severity_values(values: list[float], default: float) -> list[float]:
    return sorted({float(value) for value in (values or [default])})


def _fault_component_catalog(
    *,
    fan_groups: list[int],
    flap_indices: list[int],
    plenum_indices: list[int],
    dead_fan_scales: list[float],
    stuck_flap_angles_deg: list[float],
    plenum_scales: list[float],
) -> dict[str, list[dict]]:
    catalog = {'fan': [], 'flap': [], 'plenum': []}

    for group in fan_groups:
        for scale in dead_fan_scales:
            catalog['fan'].append({
                'family': 'fan',
                'label': f'dead-fan-{group:02d}-x{_fault_value_token(scale)}',
                'description': f'fan group {group} scale {scale:.2f}',
                'kwargs': {'dead_fan_group': int(group), 'dead_fan_scale': float(scale)},
            })

    for idx in flap_indices:
        for angle in stuck_flap_angles_deg:
            catalog['flap'].append({
                'family': 'flap',
                'label': f'stuck-flap-{idx:02d}-a{_fault_value_token(angle)}',
                'description': f'flap {idx} stuck at {angle:.1f} deg',
                'kwargs': {'stuck_flap_idx': int(idx), 'stuck_flap_alpha_deg': float(angle)},
            })

    for idx in plenum_indices:
        for scale in plenum_scales:
            catalog['plenum'].append({
                'family': 'plenum',
                'label': f'plenum-sector-{idx:02d}-x{_fault_value_token(scale)}',
                'description': f'plenum sector {idx} scale {scale:.2f}',
                'kwargs': {'plenum_sector_idx': int(idx), 'plenum_sector_scale': float(scale)},
            })

    return catalog


def _compose_fault_envelope_case(parts: list[dict]) -> dict:
    kwargs: dict[str, float | int] = {}
    for part in parts:
        for key, value in part['kwargs'].items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f'conflicting fault parts for {key}')
            kwargs[key] = value
    labels = [part['label'] for part in parts]
    descriptions = [part['description'] for part in parts]
    families = [part['family'] for part in parts]
    fault = FaultSpec(**kwargs)
    return {
        'case': '+'.join(labels),
        'components': labels,
        'families': families,
        'description': ' + '.join(descriptions),
        'fault': fault,
        'parts': parts,
    }


def _summarize_dynamic_fault_case(case_name: str, fault: FaultSpec, *, components: list[str], description: str, families: list[str]) -> dict:
    return {
        'case': case_name,
        'components': list(components),
        'families': list(families),
        'description': description,
        'stuck_flap_idx': fault.stuck_flap_idx,
        'stuck_flap_alpha_deg': fault.stuck_flap_alpha_deg,
        'dead_fan_group': fault.dead_fan_group,
        'dead_fan_scale': fault.dead_fan_scale,
        'plenum_sector_idx': fault.plenum_sector_idx,
        'plenum_sector_scale': fault.plenum_sector_scale,
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


def _fault_envelope_score(entry: dict) -> float:
    score = 0.0
    if not entry.get('feasible', False):
        score += 1_000_000.0
    score += float(_fault_status_rank(entry.get('report_status'))) * 100_000.0
    score += float(entry.get('continuous_power_p95_pct') or 0.0) * 100.0
    score += float(entry.get('final_goal_error_m') or 0.0) * 1000.0
    score += max(0.0, 1.0 - float(entry.get('selected_aggressiveness') or 0.0)) * 10_000.0
    score += float(entry.get('arrival_time_s') or 0.0)
    return score


def _fault_envelope_manifest(
    *,
    preset: str,
    power: PowerSystemParams,
    power_target_pct: float,
    flap_target_pct: float,
    tune_iterations: int,
    tune_min_aggressiveness: float,
    fan_indices: list[int],
    flap_indices: list[int],
    plenum_indices: list[int],
    fan_scales: list[float],
    flap_angles: list[float],
    plenum_scales: list[float],
    top_per_family: int,
    include_pairs: bool,
    include_triples: bool,
) -> dict:
    return {
        'preset': preset,
        'power_model': summarize_power_params(power),
        'targets': {
            'required_status': str(required_status),
            'continuous_power_p95_pct': float(power_target_pct),
            'flap_limit_usage_peak_pct': float(flap_target_pct),
            'tune_iterations': int(tune_iterations),
            'tune_min_aggressiveness': float(tune_min_aggressiveness),
        },
        'sampling': {
            'fan_groups': list(fan_indices),
            'flap_indices': list(flap_indices),
            'plenum_sector_indices': list(plenum_indices),
            'dead_fan_scale': list(fan_scales),
            'stuck_flap_alpha_deg': list(flap_angles),
            'plenum_sector_scale': list(plenum_scales),
            'top_per_family': int(top_per_family),
            'include_pairs': bool(include_pairs),
            'include_triples': bool(include_triples),
        },
    }


def _prepare_fault_envelope_cache(cache_dir: Path, manifest: dict, *, resume: bool) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if resume and existing != manifest:
            raise typer.BadParameter(
                'fault-envelope cache manifest mismatch; use a different --out-dir or --no-resume'
            )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {'path': str(manifest_path), 'manifest': manifest}


def _fault_envelope_cache_paths(cache_dir: Path, case_name: str) -> tuple[Path, Path]:
    slug = _fault_case_slug(case_name)
    return cache_dir / f'{slug}.json', cache_dir / f'{slug}.trace.json'


def _load_fault_envelope_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding='utf-8'))


def _write_fault_envelope_cache(cache_path: Path, trace_path: Path, bundle: dict) -> None:
    payload = {
        'entry': bundle['entry'],
        'parts': bundle['parts'],
        'trace_path': str(trace_path),
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _evaluate_fault_envelope_case(
    *,
    preset: str,
    preset_cfg: dict,
    case: dict,
    power: PowerSystemParams,
    lim: ActuatorLimits,
    pl: PlenumModel,
    obstacles: list[MissionObstacle],
    power_target_pct: float,
    flap_target_pct: float,
    tune_iterations: int,
    tune_min_aggressiveness: float,
) -> dict:
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
        fault=case['fault'],
    )
    meta, hist, report, tuning = tune_coordinate_mission(
        mission_kwargs,
        target_continuous_power_p95_pct=power_target_pct,
        target_flap_peak_pct=flap_target_pct,
        min_aggressiveness=tune_min_aggressiveness,
        max_iterations=tune_iterations,
    )
    entry = {
        'preset': preset,
        'case': case['case'],
        'order': len(case['components']),
        'families': list(case['families']),
        'fault_case': _summarize_dynamic_fault_case(
            case['case'],
            case['fault'],
            components=case['components'],
            description=case['description'],
            families=case['families'],
        ),
        'feasible': bool(tuning.get('envelope', {}).get('feasible', False)),
        'selected_aggressiveness': tuning.get('selected', {}).get('aggressiveness'),
        'selected_score': tuning.get('selected', {}).get('score'),
        'report_status': report.get('status'),
        'warnings': list(report.get('warnings', [])),
        'continuous_power_p95_pct': report.get('thermal', {}).get('continuous_power_p95_pct'),
        'power_margin_min_kw': report.get('thermal', {}).get('power_margin_min_kw'),
        'flap_limit_usage_peak_pct': report.get('flaps', {}).get('limit_usage_peak_pct'),
        'xy_tracking_rms_n': report.get('vectoring', {}).get('xy_tracking_rms_n'),
        'final_goal_error_m': report.get('mission', {}).get('final_goal_error_m'),
        'arrival_time_s': report.get('mission', {}).get('arrival_time_s'),
    }
    entry['severity_score'] = _fault_envelope_score(entry)
    return {'entry': entry, 'meta': meta, 'hist': hist, 'parts': case['parts']}


def build_fault_envelope_report(
    *,
    preset: str,
    power_target_pct: float,
    flap_target_pct: float,
    tune_iterations: int,
    tune_min_aggressiveness: float,
    battery_full_v: float | None,
    battery_empty_v: float | None,
    capacity_kwh: float | None,
    internal_resistance_ohm: float | None,
    hover_power_kw: float | None,
    continuous_power_kw: float | None,
    peak_power_kw: float | None,
    aux_power_kw: float | None,
    dead_fan_scale: list[float],
    stuck_flap_alpha_deg: list[float],
    plenum_sector_scale: list[float],
    fan_group: list[int],
    flap_idx: list[int],
    plenum_sector_idx: list[int],
    flap_step: int,
    plenum_step: int,
    top_per_family: int,
    include_pairs: bool,
    include_triples: bool,
    top: int,
    cache_dir: str | None = None,
    resume: bool = True,
    max_new_cases: int = 0,
) -> tuple[dict, list[dict]]:
    if preset not in COORDINATE_PRESETS:
        raise typer.BadParameter('--preset must be one of: medium, long, very-long')

    preset_cfg = COORDINATE_PRESETS[preset]
    geom = RingGeometry()
    lim = ActuatorLimits(alpha_rate_deg_s=350.0)
    pl = PlenumModel(tau_s=0.08)
    obstacles = parse_mission_obstacles(list(preset_cfg.get('obstacles', [])))
    power = build_power_params(
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
    )

    fan_indices = _fault_index_values(fan_group, count=geom.n_segments // 2, step=1, label='--fan-group')
    flap_indices = _fault_index_values(flap_idx, count=geom.n_segments, step=flap_step, label='--flap-idx')
    plenum_indices = _fault_index_values(plenum_sector_idx, count=geom.n_segments, step=plenum_step, label='--plenum-sector-idx')
    fan_scales = _fault_severity_values(dead_fan_scale, 0.0)
    flap_angles = _fault_severity_values(stuck_flap_alpha_deg, 20.0)
    plenum_scales = _fault_severity_values(plenum_sector_scale, 0.7)

    manifest = _fault_envelope_manifest(
        preset=preset,
        power=power,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        fan_indices=fan_indices,
        flap_indices=flap_indices,
        plenum_indices=plenum_indices,
        fan_scales=fan_scales,
        flap_angles=flap_angles,
        plenum_scales=plenum_scales,
        top_per_family=top_per_family,
        include_pairs=include_pairs,
        include_triples=include_triples,
    )
    cache_state = None
    cache_base = Path(cache_dir) if cache_dir else None
    if cache_base is not None:
        cache_state = _prepare_fault_envelope_cache(cache_base, manifest, resume=resume)

    catalog = _fault_component_catalog(
        fan_groups=fan_indices,
        flap_indices=flap_indices,
        plenum_indices=plenum_indices,
        dead_fan_scales=fan_scales,
        stuck_flap_angles_deg=flap_angles,
        plenum_scales=plenum_scales,
    )

    evaluated: list[dict] = []
    seen_cases: set[str] = set()
    cached_cases = 0
    new_cases = 0
    budget_exhausted = False

    def record(parts: list[dict]) -> dict | None:
        nonlocal cached_cases, new_cases, budget_exhausted
        case = _compose_fault_envelope_case(parts)
        case_name = case['case']
        if case_name in seen_cases:
            return None

        cache_payload = None
        cache_path = None
        trace_path = None
        if cache_base is not None:
            cache_path, trace_path = _fault_envelope_cache_paths(cache_base, case_name)
            if resume:
                cache_payload = _load_fault_envelope_cache(cache_path)
        if cache_payload is not None:
            bundle = {
                'entry': cache_payload['entry'],
                'meta': None,
                'hist': None,
                'parts': case['parts'],
                'cache_trace': cache_payload.get('trace_path'),
            }
            evaluated.append(bundle)
            seen_cases.add(case_name)
            cached_cases += 1
            return bundle

        if max_new_cases > 0 and new_cases >= max_new_cases:
            budget_exhausted = True
            return None

        bundle = _evaluate_fault_envelope_case(
            preset=preset,
            preset_cfg=preset_cfg,
            case=case,
            power=power,
            lim=lim,
            pl=pl,
            obstacles=obstacles,
            power_target_pct=power_target_pct,
            flap_target_pct=flap_target_pct,
            tune_iterations=tune_iterations,
            tune_min_aggressiveness=tune_min_aggressiveness,
        )
        if cache_base is not None and cache_path is not None and trace_path is not None:
            save_trace_json(str(trace_path), meta=bundle['meta'], hist=bundle['hist'])
            bundle['cache_trace'] = str(trace_path)
            _write_fault_envelope_cache(cache_path, trace_path, bundle)
        evaluated.append(bundle)
        seen_cases.add(case_name)
        new_cases += 1
        return bundle

    family_ranked: dict[str, list[dict]] = {}
    single_counts: dict[str, int] = {}
    family_bundles = {family: [] for family in ('fan', 'flap', 'plenum')}
    max_single_candidates = max((len(catalog[family]) for family in family_bundles), default=0)
    for idx in range(max_single_candidates):
        for family in ('fan', 'flap', 'plenum'):
            if idx >= len(catalog[family]):
                continue
            bundle = record([catalog[family][idx]])
            if bundle is not None:
                family_bundles[family].append(bundle)
    for family, bundles in family_bundles.items():
        bundles.sort(key=lambda item: item['entry']['severity_score'], reverse=True)
        family_ranked[family] = bundles[:max(1, int(top_per_family))]
        single_counts[family] = len(bundles)

    pair_count = 0
    if include_pairs:
        for left_family, right_family in (('fan', 'flap'), ('fan', 'plenum'), ('flap', 'plenum')):
            for left in family_ranked[left_family]:
                for right in family_ranked[right_family]:
                    bundle = record(left['parts'] + right['parts'])
                    if bundle is not None:
                        pair_count += 1

    triple_count = 0
    if include_triples and family_ranked['fan'] and family_ranked['flap'] and family_ranked['plenum']:
        for fan_bundle in family_ranked['fan']:
            for flap_bundle in family_ranked['flap']:
                for plenum_bundle in family_ranked['plenum']:
                    bundle = record(fan_bundle['parts'] + flap_bundle['parts'] + plenum_bundle['parts'])
                    if bundle is not None:
                        triple_count += 1

    ranked = sorted(evaluated, key=lambda item: item['entry']['severity_score'], reverse=True)
    top_entries = [bundle['entry'] for bundle in ranked[:max(1, int(top))]]

    report = {
        'preset': preset,
        'power_model': summarize_power_params(power),
        'targets': {
            'continuous_power_p95_pct': float(power_target_pct),
            'flap_limit_usage_peak_pct': float(flap_target_pct),
            'tune_iterations': int(tune_iterations),
            'tune_min_aggressiveness': float(tune_min_aggressiveness),
        },
        'sampling': {
            'fan_groups': fan_indices,
            'flap_indices': flap_indices,
            'plenum_sector_indices': plenum_indices,
            'dead_fan_scale': fan_scales,
            'stuck_flap_alpha_deg': flap_angles,
            'plenum_sector_scale': plenum_scales,
            'flap_step': int(max(1, flap_step)),
            'plenum_step': int(max(1, plenum_step)),
            'top_per_family': int(top_per_family),
            'include_pairs': bool(include_pairs),
            'include_triples': bool(include_triples),
        },
        'search_space': {
            'cases': len(evaluated),
            'single_cases': int(sum(single_counts.values())),
            'pair_cases': int(pair_count),
            'triple_cases': int(triple_count),
            'cached_cases': int(cached_cases),
            'new_cases': int(new_cases),
            'complete': not budget_exhausted,
            'max_new_cases': int(max_new_cases),
        },
        'cache': ({
            'dir': str(cache_base),
            'resume': bool(resume),
            'manifest': (cache_state['path'] if cache_state is not None else ''),
        } if cache_base is not None else None),
        'worst_single_by_family': {
            family: (family_ranked[family][0]['entry'] if family_ranked[family] else None)
            for family in ('fan', 'flap', 'plenum')
        },
        'top_cases': top_entries,
    }
    return report, ranked



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


def render_fault_threshold_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in FAULT_THRESHOLD_COLUMNS]
    keys = [key for _label, key in FAULT_THRESHOLD_COLUMNS]

    if format_name == 'csv':
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(headers)
        for row in rows:
            writer.writerow([_stringify_table_value(row.get(key)) for key in keys])
        return buffer.getvalue()

    matrix = [headers]
    for row in rows:
        matrix.append([_stringify_table_value(row.get(key)) for key in keys])

    widths = [
        max(len(matrix[r][c]) for r in range(len(matrix)))
        for c in range(len(headers))
    ]

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


def _default_continuous_power_sweep(base_continuous_power_kw: float) -> list[float]:
    candidates = {
        max(5.0, float(base_continuous_power_kw) - 15.0),
        max(5.0, float(base_continuous_power_kw) - 10.0),
        max(5.0, float(base_continuous_power_kw) - 5.0),
        float(base_continuous_power_kw),
        float(base_continuous_power_kw) + 5.0,
        float(base_continuous_power_kw) + 10.0,
    }
    return sorted(candidates)


def load_fault_threshold_cases(
    summary_path: str,
    *,
    case_names: list[str],
    top_cases: int,
    include_family_worst: bool,
) -> tuple[dict, list[dict]]:
    report = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    if 'preset' not in report:
        raise typer.BadParameter('--summary must point to a fault-envelope summary.json')

    available: dict[str, dict] = {}
    ordered_names: list[str] = []

    def add_entry(entry: dict | None, source: str) -> None:
        if not entry:
            return
        fault_summary = dict(entry.get('fault_case') or {})
        case_name = str(fault_summary.get('case') or entry.get('case') or '').strip()
        if not case_name:
            return
        if case_name not in available:
            families = list(entry.get('families') or fault_summary.get('families') or [])
            available[case_name] = {
                'case': case_name,
                'fault_summary': fault_summary,
                'order': int(entry.get('order') or len(fault_summary.get('components', [])) or 0),
                'families': families,
                'severity_score': entry.get('severity_score'),
                'sources': [],
            }
            ordered_names.append(case_name)
        item = available[case_name]
        if source not in item['sources']:
            item['sources'].append(source)
        if not item['families'] and fault_summary.get('families'):
            item['families'] = list(fault_summary['families'])
        if not item['order']:
            item['order'] = int(entry.get('order') or len(item['families']) or 0)
        if item.get('severity_score') is None and entry.get('severity_score') is not None:
            item['severity_score'] = entry.get('severity_score')

    if include_family_worst:
        for family, entry in (report.get('worst_single_by_family') or {}).items():
            add_entry(entry, f'family:{family}')

    for idx, entry in enumerate((report.get('top_cases') or [])[:max(0, int(top_cases))], start=1):
        add_entry(entry, f'top:{idx}')

    if case_names:
        selected = []
        missing = []
        for case_name in case_names:
            if case_name not in available:
                missing.append(case_name)
            else:
                selected.append(available[case_name])
        if missing:
            raise typer.BadParameter(
                'requested --case values are not present in the supplied summary: ' + ', '.join(missing)
            )
        return report, selected

    return report, [available[name] for name in ordered_names]


def _fault_threshold_manifest(
    *,
    summary_path: str,
    preset: str,
    selected_cases: list[dict],
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
) -> dict:
    return {
        'source_summary': str(summary_path),
        'preset': preset,
        'selected_cases': [item['case'] for item in selected_cases],
        'targets': {
            'continuous_power_p95_pct': float(power_target_pct),
            'flap_limit_usage_peak_pct': float(flap_target_pct),
            'tune_iterations': int(tune_iterations),
            'tune_min_aggressiveness': float(tune_min_aggressiveness),
        },
        'sweep_values': {
            'battery_full_v': list(battery_full_v),
            'battery_empty_v': list(battery_empty_v),
            'capacity_kwh': list(capacity_kwh),
            'internal_resistance_ohm': list(internal_resistance_ohm),
            'hover_power_kw': list(hover_power_kw),
            'continuous_power_kw': list(continuous_power_kw),
            'peak_power_kw': list(peak_power_kw),
            'aux_power_kw': list(aux_power_kw),
        },
        'top': int(top),
    }


def _fault_threshold_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f'{_fault_case_slug(case_name)}.json'


def _write_fault_threshold_cache(cache_path: Path, payload: dict) -> None:
    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


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
    ('final_goal_error_m', 'final_goal_error_m'),
    ('arrival_time_s', 'arrival_time_s'),
]


def render_preset_summary_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in PRESET_SUMMARY_COLUMNS]
    keys = [key for _label, key in PRESET_SUMMARY_COLUMNS]

    if format_name == 'csv':
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(headers)
        for row in rows:
            writer.writerow([_stringify_table_value(row.get(key)) for key in keys])
        return buffer.getvalue()

    matrix = [headers]
    for row in rows:
        matrix.append([_stringify_table_value(row.get(key)) for key in keys])

    widths = [
        max(len(matrix[r][c]) for r in range(len(matrix)))
        for c in range(len(headers))
    ]

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

    lines = [
        title,
        format_plain_row(headers),
        format_plain_row(['-' * width for width in widths]),
    ]
    for row in matrix[1:]:
        lines.append(format_plain_row(row))
    lines.append('')
    return '\n'.join(lines)


@alloc_app.command("power-sweep")
def alloc_power_sweep(
    preset: str = typer.Option("medium", "--preset", help="Coordinate mission preset to evaluate"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage for the sustainable envelope search"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage for the sustainable envelope search"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for the sustainable envelope search"),
    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the sustainable envelope search is allowed to test"),
    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages"),
    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages"),
    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities"),
    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances"),
    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions"),
    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities"),
    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities"),
    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws"),
    top: int = typer.Option(10, "--top", help="How many ranked sweep results to print"),
    out: str = typer.Option("", "--out", help="Optional JSON output path for the sweep report"),
    requirements_out: str = typer.Option("", "--requirements-out", help="Optional compact requirements-table output path (.csv, .md, .txt)"),
    requirements_format: str = typer.Option("auto", "--requirements-format", help="Requirements-table output format: auto, csv, markdown, text"),
    best_trace_out: str = typer.Option("", "--best-trace-out", help="Optional JSON trace path for the best sweep candidate"),
    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or dead-fan-0+stuck-flap-0"),
):
    report, best_candidate = build_power_sweep_report(
        preset=preset,
        fault_case=fault_case,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        top=top,
    )

    if best_trace_out and best_candidate is not None:
        save_trace_json(best_trace_out, meta=best_candidate['meta'], hist=best_candidate['hist'])
        report['best_trace_out'] = best_trace_out

    title_label = preset if fault_case == 'nominal' else f"{preset} {fault_case}"
    write_power_sweep_outputs(
        report,
        preset=preset,
        out=out,
        requirements_out=requirements_out,
        requirements_format=requirements_format,
        title_label=title_label,
    )
    typer.echo(json.dumps(report, indent=2))


@alloc_app.command("sizing-pack")
def alloc_sizing_pack(
    preset: list[str] = typer.Option([], "--preset", help="Repeat to limit presets; defaults to medium, long, and very-long"),
    fault_case: list[str] = typer.Option([], "--fault-case", help="Repeat to include fault cases or composites like dead-fan-0+stuck-flap-0; overrides --fault-set when provided"),
    fault_set: str = typer.Option("nominal", "--fault-set", help="Fault set to use when --fault-case is not provided: nominal, core, stress, all"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage for the sustainable envelope search"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage for the sustainable envelope search"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for the sustainable envelope search"),
    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the sustainable envelope search is allowed to test"),
    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages"),
    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages"),
    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities"),
    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances"),
    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions"),
    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities"),
    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities"),
    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws"),
    top: int = typer.Option(10, "--top", help="How many ranked sweep results to retain per preset"),
    out_dir: str = typer.Option("runs/power_sizing_pack", "--out-dir", help="Directory for per-preset and combined sizing artifacts"),
):
    selected_presets = list(preset) if preset else list(COORDINATE_PRESETS.keys())
    invalid = [name for name in selected_presets if name not in COORDINATE_PRESETS]
    if invalid:
        raise typer.BadParameter("--preset must be one of: medium, long, very-long")
    selected_fault_cases = select_fault_cases(list(fault_case), fault_set)

    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    preset_reports = []

    for preset_name in selected_presets:
        for fault_name in selected_fault_cases:
            report, best_candidate = build_power_sweep_report(
                preset=preset_name,
                fault_case=fault_name,
                power_target_pct=power_target_pct,
                flap_target_pct=flap_target_pct,
                tune_iterations=tune_iterations,
                tune_min_aggressiveness=tune_min_aggressiveness,
                battery_full_v=battery_full_v,
                battery_empty_v=battery_empty_v,
                capacity_kwh=capacity_kwh,
                internal_resistance_ohm=internal_resistance_ohm,
                hover_power_kw=hover_power_kw,
                continuous_power_kw=continuous_power_kw,
                peak_power_kw=peak_power_kw,
                aux_power_kw=aux_power_kw,
                top=top,
            )

            file_stub = f'{preset_name}_{fault_name}'
            title_label = preset_name if fault_name == 'nominal' else f"{preset_name} {fault_name}"
            preset_json = base_dir / f'{file_stub}_power_requirements.json'
            preset_md = base_dir / f'{file_stub}_power_requirements.md'
            preset_csv = base_dir / f'{file_stub}_power_requirements.csv'
            best_trace = base_dir / f'{file_stub}_best_trace.json'

            if best_candidate is not None:
                save_trace_json(str(best_trace), meta=best_candidate['meta'], hist=best_candidate['hist'])
                report['best_trace_out'] = str(best_trace)

            write_power_sweep_outputs(
                report,
                preset=preset_name,
                out=str(preset_json),
                requirements_out=str(preset_md),
                requirements_format='markdown',
                title_label=title_label,
            )
            render_csv = render_requirements_table(
                report['requirements_table'],
                format_name='csv',
                preset=title_label,
            )
            preset_csv.write_text(render_csv, encoding='utf-8')

            best = report.get('best') or {}
            best_model = best.get('power_model', {})
            summary_row = {
                'preset': preset_name,
                'fault_case': fault_name,
                'cases': report.get('search_space', {}).get('cases'),
                'feasible_cases': report.get('search_space', {}).get('feasible_cases'),
                'best_feasible': bool(best.get('feasible', False)),
                'battery_full_v': best_model.get('battery_full_v'),
                'capacity_kwh': best_model.get('capacity_kwh'),
                'internal_resistance_ohm': best_model.get('internal_resistance_ohm'),
                'hover_power_kw': best_model.get('hover_power_kw'),
                'continuous_power_kw': best_model.get('continuous_power_kw'),
                'peak_power_kw': best_model.get('peak_power_kw'),
                'selected_aggressiveness': best.get('selected_aggressiveness'),
                'report_status': best.get('report_status'),
                'continuous_power_p95_pct': best.get('continuous_power_p95_pct'),
                'final_goal_error_m': best.get('final_goal_error_m'),
                'arrival_time_s': best.get('arrival_time_s'),
            }
            summary_rows.append(summary_row)
            preset_reports.append({
                'preset': preset_name,
                'fault_case': report.get('fault_case', {}),
                'search_space': report.get('search_space', {}),
                'best': best,
                'artifacts': {
                    'json': str(preset_json),
                    'markdown': str(preset_md),
                    'csv': str(preset_csv),
                    'best_trace': (str(best_trace) if best_candidate is not None else ''),
                },
            })

    summary_rows.sort(key=lambda row: (
        0 if row.get('best_feasible') else 1,
        row.get('fault_case', ''),
        row.get('continuous_power_kw') if row.get('continuous_power_kw') is not None else 1e9,
        row.get('preset', ''),
    ))

    summary_report = {
        'targets': {
            'continuous_power_p95_pct': float(power_target_pct),
            'flap_limit_usage_peak_pct': float(flap_target_pct),
            'tune_iterations': int(tune_iterations),
            'tune_min_aggressiveness': float(tune_min_aggressiveness),
        },
        'presets': preset_reports,
        'summary_rows': summary_rows,
    }

    summary_json = base_dir / 'summary.json'
    summary_md = base_dir / 'summary.md'
    summary_csv = base_dir / 'summary.csv'
    summary_json.write_text(json.dumps(summary_report, indent=2), encoding='utf-8')
    summary_md.write_text(
        render_preset_summary_table(summary_rows, format_name='markdown', title='power sizing pack summary'),
        encoding='utf-8',
    )
    summary_csv.write_text(
        render_preset_summary_table(summary_rows, format_name='csv', title='power sizing pack summary'),
        encoding='utf-8',
    )
    summary_report['summary_artifacts'] = {
        'json': str(summary_json),
        'markdown': str(summary_md),
        'csv': str(summary_csv),
    }
    typer.echo(json.dumps(summary_report, indent=2))


@alloc_app.command("fault-envelope")
def alloc_fault_envelope(
    preset: str = typer.Option("very-long", "--preset", help="Coordinate mission preset to evaluate"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage for the sustainable envelope search"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage for the sustainable envelope search"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for the sustainable envelope search"),
    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the sustainable envelope search is allowed to test"),
    battery_full_v: Optional[float] = typer.Option(None, "--battery-full-v", help="Override full-pack voltage for the power model"),
    battery_empty_v: Optional[float] = typer.Option(None, "--battery-empty-v", help="Override empty-pack voltage for the power model"),
    capacity_kwh: Optional[float] = typer.Option(None, "--capacity-kwh", help="Override battery capacity for the power model"),
    internal_resistance_ohm: Optional[float] = typer.Option(None, "--internal-resistance-ohm", help="Override pack internal resistance for the power model"),
    hover_power_kw: Optional[float] = typer.Option(None, "--hover-power-kw", help="Override hover power draw for the power model"),
    continuous_power_kw: Optional[float] = typer.Option(None, "--continuous-power-kw", help="Override continuous power capability for the power model"),
    peak_power_kw: Optional[float] = typer.Option(None, "--peak-power-kw", help="Override peak power capability for the power model"),
    aux_power_kw: Optional[float] = typer.Option(None, "--aux-power-kw", help="Override auxiliary power draw for the power model"),
    dead_fan_scale: list[float] = typer.Option([], "--dead-fan-scale", help="Repeat to test multiple dead-fan severity values; default 0.0"),
    stuck_flap_alpha_deg: list[float] = typer.Option([], "--stuck-flap-alpha-deg", help="Repeat to test multiple stuck-flap angles; default 20 deg"),
    plenum_sector_scale: list[float] = typer.Option([], "--plenum-sector-scale", help="Repeat to test multiple plenum-sector thrust scales; default 0.7"),
    fan_group: list[int] = typer.Option([], "--fan-group", help="Repeat to limit fan groups; default all 16 groups"),
    flap_idx: list[int] = typer.Option([], "--flap-idx", help="Repeat to limit flap indices; default uses the coarse grid from --flap-step"),
    plenum_sector_idx: list[int] = typer.Option([], "--plenum-sector-idx", help="Repeat to limit plenum sectors; default uses the coarse grid from --plenum-step"),
    flap_step: int = typer.Option(4, "--flap-step", help="Stride for default flap-index sampling when --flap-idx is not provided"),
    plenum_step: int = typer.Option(4, "--plenum-step", help="Stride for default plenum-sector sampling when --plenum-sector-idx is not provided"),
    top_per_family: int = typer.Option(2, "--top-per-family", help="Worst single-fault cases per family to promote into pair and triple combinations"),
    include_pairs: bool = typer.Option(True, "--pairs/--no-pairs", help="Evaluate pair combinations built from the worst single-fault cases"),
    include_triples: bool = typer.Option(False, "--triples/--no-triples", help="Evaluate triple combinations built from the worst single-fault cases"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse cached fault-envelope cases from the output directory when available"),
    max_new_cases: int = typer.Option(0, "--max-new-cases", help="Limit newly evaluated cases per run for resumable searches; 0 means no limit"),
    top: int = typer.Option(12, "--top", help="How many ranked worst cases to keep in the summary"),
    save_traces: int = typer.Option(5, "--save-traces", help="How many worst-case traces to write into the output directory"),
    out_dir: str = typer.Option("runs/fault_envelope", "--out-dir", help="Directory for the fault-envelope report and worst-case traces"),
):
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / 'cache'

    report, ranked = build_fault_envelope_report(
        preset=preset,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        dead_fan_scale=dead_fan_scale,
        stuck_flap_alpha_deg=stuck_flap_alpha_deg,
        plenum_sector_scale=plenum_sector_scale,
        fan_group=fan_group,
        flap_idx=flap_idx,
        plenum_sector_idx=plenum_sector_idx,
        flap_step=flap_step,
        plenum_step=plenum_step,
        top_per_family=top_per_family,
        include_pairs=include_pairs,
        include_triples=include_triples,
        top=top,
        cache_dir=str(cache_dir),
        resume=resume,
        max_new_cases=max_new_cases,
    )

    trace_paths = []
    for idx, bundle in enumerate(ranked[:max(0, int(save_traces))], start=1):
        trace_path = base_dir / f'{idx:02d}_{_fault_case_slug(bundle["entry"]["case"])}.json'
        cache_trace = bundle.get('cache_trace')
        if cache_trace and Path(cache_trace).exists():
            trace_path.write_text(Path(cache_trace).read_text(encoding='utf-8'), encoding='utf-8')
        elif bundle.get('meta') is not None and bundle.get('hist') is not None:
            save_trace_json(str(trace_path), meta=bundle['meta'], hist=bundle['hist'])
        else:
            continue
        bundle['entry']['trace'] = str(trace_path)
        trace_paths.append(str(trace_path))

    summary_json = base_dir / 'summary.json'
    summary_md = base_dir / 'summary.md'
    summary_csv = base_dir / 'summary.csv'
    summary_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary_md.write_text(
        render_fault_envelope_table(report['top_cases'], format_name='markdown', title=f'{preset} fault envelope summary'),
        encoding='utf-8',
    )
    summary_csv.write_text(
        render_fault_envelope_table(report['top_cases'], format_name='csv', title=f'{preset} fault envelope summary'),
        encoding='utf-8',
    )
    report['artifacts'] = {
        'json': str(summary_json),
        'markdown': str(summary_md),
        'csv': str(summary_csv),
        'cache_dir': str(cache_dir),
        'traces': trace_paths,
    }
    typer.echo(json.dumps(report, indent=2))



def run_fault_threshold_report(
    *,
    summary: str,
    case_names: list[str],
    top_cases: int,
    include_family_worst: bool,
    required_status: str,
    power_target_pct: Optional[float],
    flap_target_pct: Optional[float],
    tune_iterations: Optional[int],
    tune_min_aggressiveness: Optional[float],
    battery_full_v: list[float],
    battery_empty_v: list[float],
    capacity_kwh: list[float],
    internal_resistance_ohm: list[float],
    hover_power_kw: list[float],
    continuous_power_kw: list[float],
    peak_power_kw: list[float],
    aux_power_kw: list[float],
    top: int,
    resume: bool,
    max_new_cases: int,
    out_dir: str,
) -> dict:
    envelope_report, selected_cases = load_fault_threshold_cases(
        summary,
        case_names=list(case_names),
        top_cases=top_cases,
        include_family_worst=include_family_worst,
    )
    if not selected_cases:
        raise typer.BadParameter('no fault cases were selected from the supplied summary')

    default_power = PowerSystemParams()
    base_power = envelope_report.get('power_model', {})
    base_targets = envelope_report.get('targets', {})
    preset = str(envelope_report['preset'])

    resolved_required_status = _normalize_required_status(base_targets.get('required_status', required_status))
    resolved_power_target_pct = float(base_targets.get('continuous_power_p95_pct', 100.0) if power_target_pct is None else power_target_pct)
    resolved_flap_target_pct = float(base_targets.get('flap_limit_usage_peak_pct', 90.0) if flap_target_pct is None else flap_target_pct)
    resolved_tune_iterations = int(base_targets.get('tune_iterations', 6) if tune_iterations is None else tune_iterations)
    resolved_tune_min_aggressiveness = float(base_targets.get('tune_min_aggressiveness', 0.25) if tune_min_aggressiveness is None else tune_min_aggressiveness)

    battery_full_values = list(battery_full_v) if battery_full_v else [float(base_power.get('battery_full_v', default_power.battery_full_v))]
    battery_empty_values = list(battery_empty_v) if battery_empty_v else [float(base_power.get('battery_empty_v', default_power.battery_empty_v))]
    capacity_values = list(capacity_kwh) if capacity_kwh else [float(base_power.get('capacity_kwh', default_power.capacity_kwh))]
    resistance_values = list(internal_resistance_ohm) if internal_resistance_ohm else [float(base_power.get('internal_resistance_ohm', default_power.internal_resistance_ohm))]
    hover_values = list(hover_power_kw) if hover_power_kw else [float(base_power.get('hover_power_kw', default_power.hover_power_w / 1000.0))]
    base_continuous_kw = float(base_power.get('continuous_power_kw', default_power.continuous_power_w / 1000.0))
    continuous_values = list(continuous_power_kw) if continuous_power_kw else _default_continuous_power_sweep(base_continuous_kw)
    peak_values = list(peak_power_kw) if peak_power_kw else [float(base_power.get('peak_power_kw', default_power.peak_power_w / 1000.0))]
    aux_values = list(aux_power_kw) if aux_power_kw else [float(base_power.get('aux_power_kw', default_power.aux_power_w / 1000.0))]

    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / 'cache'
    manifest = _fault_threshold_manifest(
        summary_path=summary,
        preset=preset,
        selected_cases=selected_cases,
        required_status=resolved_required_status,
        power_target_pct=resolved_power_target_pct,
        flap_target_pct=resolved_flap_target_pct,
        tune_iterations=resolved_tune_iterations,
        tune_min_aggressiveness=resolved_tune_min_aggressiveness,
        battery_full_v=battery_full_values,
        battery_empty_v=battery_empty_values,
        capacity_kwh=capacity_values,
        internal_resistance_ohm=resistance_values,
        hover_power_kw=hover_values,
        continuous_power_kw=continuous_values,
        peak_power_kw=peak_values,
        aux_power_kw=aux_values,
        top=top,
    )
    cache_state = _prepare_fault_envelope_cache(cache_dir, manifest, resume=resume)

    summary_rows = []
    case_reports = []
    cached_cases = 0
    new_cases = 0
    budget_exhausted = False

    for case_info in selected_cases:
        case_name = case_info['case']
        cache_path = _fault_threshold_cache_path(cache_dir, case_name)
        cache_payload = _load_fault_envelope_cache(cache_path) if resume else None
        if cache_payload is not None:
            summary_rows.append(dict(cache_payload['summary_row']))
            case_reports.append(dict(cache_payload['case_report']))
            cached_cases += 1
            continue

        if max_new_cases < 0:
            budget_exhausted = True
            break
        if max_new_cases > 0 and new_cases >= max_new_cases:
            budget_exhausted = True
            break

        fault_summary = dict(case_info['fault_summary'])
        fault = fault_spec_from_summary(fault_summary)
        report, best_candidate = build_power_sweep_report_for_fault(
            preset=preset,
            fault_label=case_name,
            fault=fault,
            fault_summary=fault_summary,
            required_status=resolved_required_status,
            power_target_pct=resolved_power_target_pct,
            flap_target_pct=resolved_flap_target_pct,
            tune_iterations=resolved_tune_iterations,
            tune_min_aggressiveness=resolved_tune_min_aggressiveness,
            battery_full_v=battery_full_values,
            battery_empty_v=battery_empty_values,
            capacity_kwh=capacity_values,
            internal_resistance_ohm=resistance_values,
            hover_power_kw=hover_values,
            continuous_power_kw=continuous_values,
            peak_power_kw=peak_values,
            aux_power_kw=aux_values,
            top=top,
        )

        file_stub = _fault_case_slug(case_name)
        case_json = base_dir / f'{file_stub}_threshold.json'
        case_md = base_dir / f'{file_stub}_requirements.md'
        case_csv = base_dir / f'{file_stub}_requirements.csv'
        best_trace = base_dir / f'{file_stub}_best_trace.json'

        if best_candidate is not None:
            save_trace_json(str(best_trace), meta=best_candidate['meta'], hist=best_candidate['hist'])
            report['best_trace_out'] = str(best_trace)

        title_label = f'{preset} {case_name}'
        write_power_sweep_outputs(
            report,
            preset=preset,
            out=str(case_json),
            requirements_out=str(case_md),
            requirements_format='markdown',
            title_label=title_label,
        )
        case_csv.write_text(
            render_requirements_table(report['requirements_table'], format_name='csv', preset=title_label),
            encoding='utf-8',
        )

        best = report.get('best') or {}
        best_model = best.get('power_model', {})
        summary_row = {
            'case': case_name,
            'source': ','.join(case_info.get('sources', [])),
            'order': case_info.get('order'),
            'families': ','.join(case_info.get('families', [])),
            'best_feasible': bool(best.get('feasible', False)),
            'meets_threshold': bool(best.get('meets_threshold', False)),
            'required_continuous_power_kw': (best_model.get('continuous_power_kw') if best.get('meets_threshold') else None),
            'battery_full_v': best_model.get('battery_full_v'),
            'capacity_kwh': best_model.get('capacity_kwh'),
            'internal_resistance_ohm': best_model.get('internal_resistance_ohm'),
            'hover_power_kw': best_model.get('hover_power_kw'),
            'peak_power_kw': best_model.get('peak_power_kw'),
            'selected_aggressiveness': best.get('selected_aggressiveness'),
            'report_status': best.get('report_status'),
            'continuous_power_p95_pct': best.get('continuous_power_p95_pct'),
            'final_goal_error_m': best.get('final_goal_error_m'),
            'arrival_time_s': best.get('arrival_time_s'),
            'severity_score': case_info.get('severity_score'),
        }
        case_report = {
            'case': case_name,
            'fault_case': fault_summary,
            'sources': list(case_info.get('sources', [])),
            'severity_score': case_info.get('severity_score'),
            'search_space': report.get('search_space', {}),
            'best': best,
            'artifacts': {
                'json': str(case_json),
                'markdown': str(case_md),
                'csv': str(case_csv),
                'best_trace': (str(best_trace) if best_candidate is not None else ''),
            },
        }
        summary_rows.append(summary_row)
        case_reports.append(case_report)
        _write_fault_threshold_cache(cache_path, {
            'report': report,
            'summary_row': summary_row,
            'case_report': case_report,
        })
        new_cases += 1

    summary_rows.sort(key=lambda row: (
        0 if row.get('meets_threshold') else 1,
        0 if row.get('best_feasible') else 1,
        row.get('required_continuous_power_kw') if row.get('required_continuous_power_kw') is not None else 1e9,
        -(float(row.get('severity_score') or 0.0)),
        row.get('case', ''),
    ))

    completed_case_names = {item['case'] for item in case_reports}
    pending_cases = [item['case'] for item in selected_cases if item['case'] not in completed_case_names]
    summary_report = {
        'preset': preset,
        'source_summary': summary,
        'selected_cases': [item['case'] for item in selected_cases],
        'pending_cases': pending_cases,
        'targets': {
            'required_status': resolved_required_status,
            'continuous_power_p95_pct': resolved_power_target_pct,
            'flap_limit_usage_peak_pct': resolved_flap_target_pct,
            'tune_iterations': resolved_tune_iterations,
            'tune_min_aggressiveness': resolved_tune_min_aggressiveness,
        },
        'sweep_values': {
            'battery_full_v': battery_full_values,
            'battery_empty_v': battery_empty_values,
            'capacity_kwh': capacity_values,
            'internal_resistance_ohm': resistance_values,
            'hover_power_kw': hover_values,
            'continuous_power_kw': continuous_values,
            'peak_power_kw': peak_values,
            'aux_power_kw': aux_values,
        },
        'search_space': {
            'selected_cases': len(selected_cases),
            'completed_cases': len(case_reports),
            'cached_cases': int(cached_cases),
            'new_cases': int(new_cases),
            'complete': not budget_exhausted and not pending_cases,
            'max_new_cases': int(max_new_cases),
        },
        'cache': {
            'dir': str(cache_dir),
            'resume': bool(resume),
            'manifest': cache_state['path'],
        },
        'cases': case_reports,
        'summary_rows': summary_rows,
    }

    summary_json = base_dir / 'summary.json'
    summary_md = base_dir / 'summary.md'
    summary_csv = base_dir / 'summary.csv'
    summary_report['summary_artifacts'] = {
        'json': str(summary_json),
        'markdown': str(summary_md),
        'csv': str(summary_csv),
    }
    summary_json.write_text(json.dumps(summary_report, indent=2), encoding='utf-8')
    summary_md.write_text(
        render_fault_threshold_table(summary_rows, format_name='markdown', title=f'{preset} fault threshold summary'),
        encoding='utf-8',
    )
    summary_csv.write_text(
        render_fault_threshold_table(summary_rows, format_name='csv', title=f'{preset} fault threshold summary'),
        encoding='utf-8',
    )
    return summary_report


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


def render_fault_threshold_pack_table(rows: list[dict], *, format_name: str, title: str) -> str:
    headers = [label for label, _key in FAULT_THRESHOLD_PACK_COLUMNS]
    keys = [key for _label, key in FAULT_THRESHOLD_PACK_COLUMNS]

    if format_name == 'csv':
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(headers)
        for row in rows:
            writer.writerow([_stringify_table_value(row.get(key)) for key in keys])
        return buffer.getvalue()

    matrix = [headers]
    for row in rows:
        matrix.append([_stringify_table_value(row.get(key)) for key in keys])

    widths = [
        max(len(matrix[r][c]) for r in range(len(matrix)))
        for c in range(len(headers))
    ]

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


@alloc_app.command("fault-threshold")
def alloc_fault_threshold(
    summary: str = typer.Option(..., "--summary", help="Path to a completed fault-envelope summary.json"),
    case: list[str] = typer.Option([], "--case", help="Repeat to select specific cases from the supplied fault-envelope summary"),
    top_cases: int = typer.Option(3, "--top-cases", help="How many top fault-envelope cases to evaluate when --case is not provided"),
    include_family_worst: bool = typer.Option(True, "--include-family-worst/--no-include-family-worst", help="Also include the worst single case per family from the supplied summary"),
    required_status: str = typer.Option('feasible', "--required-status", help="Threshold qualification: feasible, pass, caution, or risk"),
    power_target_pct: Optional[float] = typer.Option(None, "--power-target-pct", help="Override the target p95 continuous-power percentage; defaults to the summary target"),
    flap_target_pct: Optional[float] = typer.Option(None, "--flap-target-pct", help="Override the target peak flap-limit usage percentage; defaults to the summary target"),
    tune_iterations: Optional[int] = typer.Option(None, "--tune-iters", help="Override the tuning iterations; defaults to the summary target"),
    tune_min_aggressiveness: Optional[float] = typer.Option(None, "--tune-min-aggr", help="Override the minimum aggressiveness; defaults to the summary target"),
    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages; defaults to the summary power model"),
    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages; defaults to the summary power model"),
    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities; defaults to the summary power model"),
    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances; defaults to the summary power model"),
    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions; defaults to the summary power model"),
    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities; defaults to a threshold sweep around the summary power model"),
    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities; defaults to the summary power model"),
    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws; defaults to the summary power model"),
    top: int = typer.Option(10, "--top", help="How many ranked sweep results to retain per fault case"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse cached fault-threshold cases from the output directory when available"),
    max_new_cases: int = typer.Option(0, "--max-new-cases", help="Limit newly evaluated fault-threshold cases per run; 0 means no limit"),
    out_dir: str = typer.Option("runs/fault_threshold", "--out-dir", help="Directory for the worst-case threshold artifacts"),
):
    summary_report = run_fault_threshold_report(
        summary=summary,
        case_names=list(case),
        top_cases=top_cases,
        include_family_worst=include_family_worst,
        required_status=required_status,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        top=top,
        resume=resume,
        max_new_cases=max_new_cases,
        out_dir=out_dir,
    )
    typer.echo(json.dumps(summary_report, indent=2))


@alloc_app.command("fault-threshold-pack")
def alloc_fault_threshold_pack(
    summary: list[str] = typer.Option([], "--summary", help="Repeat to provide completed fault-envelope summary.json files; defaults to auto-discovering runs/fault_envelope*/summary.json"),
    preset: list[str] = typer.Option([], "--preset", help="Optional preset filter when auto-discovering or limiting supplied summaries"),
    case: list[str] = typer.Option([], "--case", help="Repeat to select specific cases from every supplied summary"),
    top_cases: int = typer.Option(3, "--top-cases", help="How many top fault-envelope cases to evaluate per summary when --case is not provided"),
    include_family_worst: bool = typer.Option(True, "--include-family-worst/--no-include-family-worst", help="Also include the worst single case per family from each supplied summary"),
    required_status: str = typer.Option('feasible', "--required-status", help="Threshold qualification: feasible, pass, caution, or risk"),
    power_target_pct: Optional[float] = typer.Option(None, "--power-target-pct", help="Override the target p95 continuous-power percentage; defaults to each summary target"),
    flap_target_pct: Optional[float] = typer.Option(None, "--flap-target-pct", help="Override the target peak flap-limit usage percentage; defaults to each summary target"),
    tune_iterations: Optional[int] = typer.Option(None, "--tune-iters", help="Override the tuning iterations; defaults to each summary target"),
    tune_min_aggressiveness: Optional[float] = typer.Option(None, "--tune-min-aggr", help="Override the minimum aggressiveness; defaults to each summary target"),
    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages; defaults to each summary power model"),
    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages; defaults to each summary power model"),
    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities; defaults to each summary power model"),
    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances; defaults to each summary power model"),
    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions; defaults to each summary power model"),
    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities; defaults to a threshold sweep around each summary power model"),
    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities; defaults to each summary power model"),
    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws; defaults to each summary power model"),
    top: int = typer.Option(10, "--top", help="How many ranked sweep results to retain per fault case"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse cached fault-threshold cases from each preset output directory when available"),
    max_new_cases: int = typer.Option(0, "--max-new-cases", help="Limit newly evaluated fault-threshold cases per summary run; 0 means no limit"),
    max_new_cases_total: int = typer.Option(0, "--max-new-cases-total", help="Limit newly evaluated fault-threshold cases across the whole pack run; 0 means no limit"),
    out_dir: str = typer.Option("runs/fault_threshold_pack", "--out-dir", help="Directory for per-summary and combined threshold artifacts"),
):
    requested_presets = set(preset)
    summary_paths = [Path(item) for item in summary] if summary else sorted(Path('runs').glob('fault_envelope*/summary.json'))
    if not summary_paths:
        raise typer.BadParameter('no fault-envelope summaries were provided or discovered')

    resolved_summaries = []
    seen = set()
    for summary_path in summary_paths:
        if not summary_path.exists():
            raise typer.BadParameter(f'fault-envelope summary does not exist: {summary_path}')
        envelope_report = json.loads(summary_path.read_text(encoding='utf-8'))
        current_preset = str(envelope_report.get('preset') or '').strip()
        if not current_preset:
            raise typer.BadParameter(f'fault-envelope summary missing preset: {summary_path}')
        if requested_presets and current_preset not in requested_presets:
            continue
        key = (current_preset, str(summary_path))
        if key in seen:
            continue
        seen.add(key)
        resolved_summaries.append({
            'preset': current_preset,
            'summary_path': str(summary_path),
            'summary_source': summary_path.parent.name,
        })

    if not resolved_summaries:
        allowed = ', '.join(sorted(requested_presets)) if requested_presets else 'discovered summaries'
        raise typer.BadParameter(f'no summaries matched the requested presets: {allowed}')

    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    pack_rows = []
    preset_reports = []
    total_cached_cases = 0
    total_new_cases = 0
    all_complete = True
    remaining_total_budget = (int(max_new_cases_total) if max_new_cases_total > 0 else None)

    for item in resolved_summaries:
        preset_name = item['preset']
        summary_path = item['summary_path']
        summary_source = item['summary_source']
        preset_dir = base_dir / f'{preset_name}__{_fault_case_slug(summary_source)}'
        if remaining_total_budget is None:
            per_summary_budget = max_new_cases
        elif remaining_total_budget <= 0:
            per_summary_budget = -1
        elif max_new_cases > 0:
            per_summary_budget = min(int(max_new_cases), int(remaining_total_budget))
        else:
            per_summary_budget = int(remaining_total_budget)

        report = run_fault_threshold_report(
            summary=summary_path,
            case_names=list(case),
            top_cases=top_cases,
            include_family_worst=include_family_worst,
            required_status=required_status,
            power_target_pct=power_target_pct,
            flap_target_pct=flap_target_pct,
            tune_iterations=tune_iterations,
            tune_min_aggressiveness=tune_min_aggressiveness,
            battery_full_v=battery_full_v,
            battery_empty_v=battery_empty_v,
            capacity_kwh=capacity_kwh,
            internal_resistance_ohm=internal_resistance_ohm,
            hover_power_kw=hover_power_kw,
            continuous_power_kw=continuous_power_kw,
            peak_power_kw=peak_power_kw,
            aux_power_kw=aux_power_kw,
            top=top,
            resume=resume,
            max_new_cases=per_summary_budget,
            out_dir=str(preset_dir),
        )
        search_space = report.get('search_space', {})
        total_cached_cases += int(search_space.get('cached_cases', 0))
        total_new_cases += int(search_space.get('new_cases', 0))
        if remaining_total_budget is not None:
            remaining_total_budget -= int(search_space.get('new_cases', 0))
        all_complete = all_complete and bool(search_space.get('complete', False))

        for row in report.get('summary_rows', []):
            pack_rows.append({
                'preset': preset_name,
                'summary_source': summary_source,
                'case': row.get('case'),
                'source': row.get('source'),
                'order': row.get('order'),
                'families': row.get('families'),
                'best_feasible': row.get('best_feasible'),
                'meets_threshold': row.get('meets_threshold'),
                'required_continuous_power_kw': row.get('required_continuous_power_kw'),
                'battery_full_v': row.get('battery_full_v'),
                'capacity_kwh': row.get('capacity_kwh'),
                'internal_resistance_ohm': row.get('internal_resistance_ohm'),
                'hover_power_kw': row.get('hover_power_kw'),
                'peak_power_kw': row.get('peak_power_kw'),
                'selected_aggressiveness': row.get('selected_aggressiveness'),
                'report_status': row.get('report_status'),
                'continuous_power_p95_pct': row.get('continuous_power_p95_pct'),
                'final_goal_error_m': row.get('final_goal_error_m'),
                'arrival_time_s': row.get('arrival_time_s'),
                'severity_score': row.get('severity_score'),
            })

        preset_reports.append({
            'preset': preset_name,
            'summary_source': summary_source,
            'source_summary': summary_path,
            'search_space': search_space,
            'pending_cases': report.get('pending_cases', []),
            'summary_artifacts': report.get('summary_artifacts', {}),
        })

    pack_rows.sort(key=lambda row: (
        0 if row.get('meets_threshold') else 1,
        0 if row.get('best_feasible') else 1,
        row.get('required_continuous_power_kw') if row.get('required_continuous_power_kw') is not None else 1e9,
        row.get('preset', ''),
        -(float(row.get('severity_score') or 0.0)),
        row.get('case', ''),
    ))

    pack_report = {
        'summaries': preset_reports,
        'summary_rows': pack_rows,
        'search_space': {
            'summaries': len(resolved_summaries),
            'cached_cases': int(total_cached_cases),
            'new_cases': int(total_new_cases),
            'complete': bool(all_complete),
            'max_new_cases_per_summary': int(max_new_cases),
            'max_new_cases_total': int(max_new_cases_total),
        },
    }
    summary_json = base_dir / 'summary.json'
    summary_md = base_dir / 'summary.md'
    summary_csv = base_dir / 'summary.csv'
    pack_report['summary_artifacts'] = {
        'json': str(summary_json),
        'markdown': str(summary_md),
        'csv': str(summary_csv),
    }
    summary_json.write_text(json.dumps(pack_report, indent=2), encoding='utf-8')
    summary_md.write_text(
        render_fault_threshold_pack_table(pack_rows, format_name='markdown', title='fault threshold pack summary'),
        encoding='utf-8',
    )
    summary_csv.write_text(
        render_fault_threshold_pack_table(pack_rows, format_name='csv', title='fault threshold pack summary'),
        encoding='utf-8',
    )
    typer.echo(json.dumps(pack_report, indent=2))


@alloc_app.command("repel")
def alloc_repel(
    obstacle_x_m: float = typer.Option(30.0, "--ox"),
    obstacle_y_m: float = typer.Option(0.0, "--oy"),
    total_s: float = typer.Option(12.0, "--total-s"),
    init_vx_mps: float = typer.Option(1.0, "--init-vx"),
    init_vy_mps: float = typer.Option(0.0, "--init-vy"),
    radius_m: float = typer.Option(30.0, "--radius-m"),
    k_n_per_m: float = typer.Option(120.0, "--k"),
    fxy_max_n: float = typer.Option(4000.0, "--fxy-max"),
    stuck_flap_idx: int = typer.Option(-1, "--stuck-flap-idx"),
    stuck_flap_alpha_deg: float = typer.Option(0.0, "--stuck-flap-alpha-deg"),
    dead_fan_group: int = typer.Option(-1, "--dead-fan-group"),
    dead_fan_scale: float = typer.Option(0.0, "--dead-fan-scale"),
    plenum_sector_idx: int = typer.Option(-1, "--plenum-sector-idx"),
    plenum_sector_scale: float = typer.Option(0.7, "--plenum-sector-scale"),
    alpha_rate_deg_s: float = typer.Option(200.0, "--alpha-rate-deg-s", help="Actuator flap rate (deg/s)"),
    plenum_tau_s: float = typer.Option(0.12, "--plenum-tau-s", help="Plenum lag time constant (s)"),
    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path (for dashboard replay)"),
):
    field = RepelField(k_n_per_m=k_n_per_m, radius_m=radius_m, fxy_max_n=fxy_max_n)
    fault = FaultSpec(
        stuck_flap_idx=(None if stuck_flap_idx < 0 else stuck_flap_idx),
        stuck_flap_alpha_deg=stuck_flap_alpha_deg,
        dead_fan_group=(None if dead_fan_group < 0 else dead_fan_group),
        dead_fan_scale=dead_fan_scale,
        plenum_sector_idx=(None if plenum_sector_idx < 0 else plenum_sector_idx),
        plenum_sector_scale=plenum_sector_scale,
    )
    # build actuator limits / plenum objects from CLI options
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    out, _hist = run_repel_test_v4(
        obstacle_x_m=obstacle_x_m,
        obstacle_y_m=obstacle_y_m,
        total_s=total_s,
        initial_vx_mps=init_vx_mps,
        initial_vy_mps=init_vy_mps,
        field=field,
        fault=fault,
        lim=lim,
        pl=pl,
        trace_out=(trace_out if trace_out else None),
    )
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("step-redirect")
def alloc_step_redirect(
    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg"),
    dir_b_deg: float = typer.Option(90.0, "--dir-b-deg"),
    fxy_n: float = typer.Option(2200.0, "--fxy"),
    step_time_s: float = typer.Option(3.0, "--step-time-s"),
    redirect_time_s: float = typer.Option(1.6, "--redirect-time-s", help="Duration of the smooth redirect blend phase"),
    total_s: float = typer.Option(9.0, "--total-s"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm"),
    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),
    redirect_speed_scale: float = typer.Option(0.88, "--redirect-speed-scale", help="Fraction of the pre-step speed preserved through the redirect blend"),
    redirect_cross_gain: float = typer.Option(1.0, "--redirect-cross-gain", help="Cross-velocity cleanup gain during the redirect blend"),
    eco: bool = typer.Option(False, "--eco/--no-eco", help="Use the dedicated low-power eco redirect profile"),
    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or weak-flap-0"),
    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),
    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),
    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),
    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),
    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),
    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),
):
    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)
    fault = build_fault_spec(normalized_fault_case)
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    base_kwargs = {
        'dir_a_deg': dir_a_deg,
        'dir_b_deg': dir_b_deg,
        'fxy_n': fxy_n,
        'step_time_s': step_time_s,
        'redirect_time_s': redirect_time_s,
        'total_s': total_s,
        'yaw_hold_deg': yaw_hold_deg,
        'mz_nm': mz_nm,
        'lim': lim,
        'pl': pl,
        'fault': fault,
        'redirect_speed_scale': redirect_speed_scale,
        'redirect_cross_gain': redirect_cross_gain,
    }
    profile_name = 'step-redirect-eco' if eco else 'step-redirect'
    if maneuver_safe:
        out, _hist, assessment, tuning = tune_maneuver_profile(
            profile_name,
            base_kwargs,
            target_continuous_power_p95_pct=power_target_pct,
            target_flap_peak_pct=flap_target_pct,
            target_fan_tracking_pct=fan_target_pct,
            target_xy_tracking_rms_n=xy_target_n,
            target_yaw_coupling_deg=yaw_target_deg,
            min_aggressiveness=tune_min_aggressiveness,
            max_iterations=tune_iterations,
        )
        out['maneuver_tuning'] = tuning
    else:
        out, _hist = _run_maneuver_profile(profile_name, **base_kwargs)
        assessment = assess_maneuver_result(out, _hist, profile_name)
    out['fault'] = summarize_fault_case(normalized_fault_case, fault)
    out['maneuver_assessment'] = assessment
    if trace_out:
        save_trace_json(trace_out, meta=out, hist=_hist)
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("step-snap")
def alloc_step_snap(
    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg"),
    dir_b_deg: float = typer.Option(180.0, "--dir-b-deg"),
    fxy_n: float = typer.Option(3000.0, "--fxy"),
    step_time_s: float = typer.Option(3.0, "--step-time-s"),
    snap_stop_s: float = typer.Option(0.8, "--snap-stop-s", help="Duration of snap-stop brake phase"),
    total_s: float = typer.Option(9.0, "--total-s"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm"),
    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),
    brake_gain: float = typer.Option(1.2, "--brake-gain", help="Multiplier on fxy during snap-stop"),
    speed_stop_thr_mps: float = typer.Option(0.2, "--stop-thr-mps", help="Speed threshold considered 'stopped'"),
    redirect_hold_frac: float = typer.Option(-1.0, "--redirect-hold-frac", help="Fraction of the snap window kept as pure braking before steering toward the new direction. Negative = auto"),
    redirect_steer_scale: float = typer.Option(1.0, "--redirect-steer-scale", help="Scaling applied to the snap revector blend during sharp redirects"),
    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or slow-flap-0"),
    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),
    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),
    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),
    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),
    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),
    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),
):
    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)
    fault = build_fault_spec(normalized_fault_case)
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    base_kwargs = {
        'dir_a_deg': dir_a_deg,
        'dir_b_deg': dir_b_deg,
        'fxy_n': fxy_n,
        'step_time_s': step_time_s,
        'snap_stop_s': snap_stop_s,
        'total_s': total_s,
        'yaw_hold_deg': yaw_hold_deg,
        'mz_nm': mz_nm,
        'lim': lim,
        'pl': pl,
        'fault': fault,
        'brake_gain': brake_gain,
        'speed_stop_thr_mps': speed_stop_thr_mps,
        'redirect_hold_frac': redirect_hold_frac,
        'redirect_steer_scale': redirect_steer_scale,
    }

    if maneuver_safe:
        out, _hist, assessment, tuning = tune_maneuver_profile(
            'step-snap',
            base_kwargs,
            target_continuous_power_p95_pct=power_target_pct,
            target_flap_peak_pct=flap_target_pct,
            target_fan_tracking_pct=fan_target_pct,
            target_xy_tracking_rms_n=xy_target_n,
            target_yaw_coupling_deg=yaw_target_deg,
            min_aggressiveness=tune_min_aggressiveness,
            max_iterations=tune_iterations,
        )
        out['maneuver_tuning'] = tuning
    else:
        out, _hist = _run_maneuver_profile('step-snap', **base_kwargs)
        assessment = assess_maneuver_result(out, _hist, 'step-snap')
    out['fault'] = summarize_fault_case(normalized_fault_case, fault)
    out['maneuver_assessment'] = assessment
    if trace_out:
        save_trace_json(trace_out, meta=out, hist=_hist)
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("maneuver-pack")
def alloc_maneuver_pack(
    profile: str = typer.Option("step-snap", "--profile", help="Maneuver profile: step, step-snap, step-redirect, or step-redirect-eco"),
    dir_b_deg: list[float] = typer.Option([], "--dir-b-deg", help="Redirect target angle(s). Repeat; defaults to 45, 90, 135, 180."),
    fault_case: list[str] = typer.Option([], "--fault-case", help="Fault case(s). Repeat; defaults to the maneuver-focused set."),
    fault_set: str = typer.Option("maneuver", "--fault-set", help="Fault set: maneuver, nominal, core, stress, or all"),
    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg"),
    fxy_n: float = typer.Option(3000.0, "--fxy"),
    step_time_s: float = typer.Option(3.0, "--step-time-s"),
    snap_stop_s: float = typer.Option(0.8, "--snap-stop-s", help="Duration of snap-stop brake phase for step-snap"),
    total_s: float = typer.Option(9.0, "--total-s"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm"),
    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),
    brake_gain: float = typer.Option(1.2, "--brake-gain"),
    speed_stop_thr_mps: float = typer.Option(0.2, "--stop-thr-mps"),
    redirect_time_s: float = typer.Option(1.6, "--redirect-time-s", help="Duration of the smooth redirect blend phase for step-redirect"),
    redirect_speed_scale: float = typer.Option(0.88, "--redirect-speed-scale", help="Fraction of the pre-step speed preserved through the redirect blend for step-redirect"),
    redirect_cross_gain: float = typer.Option(1.0, "--redirect-cross-gain", help="Cross-velocity cleanup gain during the redirect blend for step-redirect"),
    redirect_hold_frac: float = typer.Option(-1.0, "--redirect-hold-frac", help="Fraction of the snap window kept as pure braking before steering toward the new direction. Negative = auto"),
    redirect_steer_scale: float = typer.Option(1.0, "--redirect-steer-scale", help="Scaling applied to the snap revector blend during sharp redirects"),
    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),
    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),
    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),
    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),
    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),
    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),
    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),
    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),
    save_traces: bool = typer.Option(False, "--save-traces/--no-save-traces", help="Write per-case trace JSON files next to the summary"),
    out_dir: str = typer.Option("runs/maneuver_pack", "--out-dir", help="Output directory for summary artifacts"),
):
    profile = profile.strip().lower()
    if profile not in {"step", "step-snap", "step-redirect", "step-redirect-eco"}:
        raise typer.BadParameter("--profile must be 'step', 'step-snap', 'step-redirect', or 'step-redirect-eco'")

    angles = [float(angle) for angle in (dir_b_deg or MANEUVER_DEFAULT_ANGLES)]
    selected_fault_cases = select_fault_cases(list(fault_case), fault_set)
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for angle in angles:
        for case_name in selected_fault_cases:
            fault = build_fault_spec(case_name)
            base_kwargs = {
                'dir_a_deg': dir_a_deg,
                'dir_b_deg': angle,
                'fxy_n': fxy_n,
                'step_time_s': step_time_s,
                'total_s': total_s,
                'yaw_hold_deg': yaw_hold_deg,
                'mz_nm': mz_nm,
                'lim': lim,
                'pl': pl,
                'fault': fault,
                'snap_stop_s': snap_stop_s,
                'brake_gain': brake_gain,
                'speed_stop_thr_mps': speed_stop_thr_mps,
                'redirect_time_s': redirect_time_s,
                'redirect_speed_scale': redirect_speed_scale,
                'redirect_cross_gain': redirect_cross_gain,
                'redirect_hold_frac': redirect_hold_frac,
                'redirect_steer_scale': redirect_steer_scale,
            }
            if maneuver_safe:
                meta, hist, assessment, tuning = tune_maneuver_profile(
                    profile,
                    base_kwargs,
                    target_continuous_power_p95_pct=power_target_pct,
                    target_flap_peak_pct=flap_target_pct,
                    target_fan_tracking_pct=fan_target_pct,
                    target_xy_tracking_rms_n=xy_target_n,
                    target_yaw_coupling_deg=yaw_target_deg,
                    min_aggressiveness=tune_min_aggressiveness,
                    max_iterations=tune_iterations,
                )
                meta['maneuver_tuning'] = tuning
            else:
                meta, hist = _run_maneuver_profile(profile, **base_kwargs)
                assessment = assess_maneuver_result(meta, hist, profile)
                tuning = None
            meta['fault'] = summarize_fault_case(case_name, fault)
            meta['maneuver_assessment'] = assessment
            engineering = assessment['engineering']
            maneuver = assessment['maneuver']
            result = {
                'profile': profile,
                'dir_a_deg': float(dir_a_deg),
                'dir_b_deg': float(angle),
                'fault_case': case_name,
                'status': assessment['status'],
                'warnings': list(assessment['warnings']),
                'fault': meta['fault'],
                'selected_aggressiveness': (tuning['selected']['aggressiveness'] if tuning else 1.0),
                'tuning_status': (tuning['status'] if tuning else 'raw'),
                't_to_speed_below_thr_s': maneuver['t_to_speed_below_thr_s'],
                'snap_stop_distance_m': maneuver['snap_stop_distance_m'],
                't_reversal_s': maneuver['t_reversal_s'],
                't90_dir_s': maneuver['t90_dir_s'],
                'min_speed_transition_mps': maneuver['min_speed_transition_mps'],
                'peak_speed_mps': maneuver['peak_speed_mps'],
                'yaw_hold_error_mean_abs_deg': maneuver['yaw_hold_error_mean_abs_deg'],
                'yaw_track_decoupling_mean_abs_deg': maneuver['yaw_track_decoupling_mean_abs_deg'],
                'xy_tracking_rms_n': engineering['vectoring']['xy_tracking_rms_n'],
                'alignment_p95_deg': engineering['vectoring']['alignment_p95_deg'],
                'flap_limit_usage_peak_pct': engineering['flaps']['limit_usage_peak_pct'],
                'flap_tracking_rms_deg': engineering['flaps']['tracking_rms_deg'],
                'fan_tracking_rms_pct_mean_cmd': engineering['fans']['tracking_rms_pct_mean_cmd'],
                'fan_response_min_pct': engineering['fans']['response_min_pct'],
                'continuous_power_p95_pct': engineering['thermal']['continuous_power_p95_pct'],
                'thrust_scale_min_pct': engineering['power']['thrust_scale_min_pct'],
            }
            results.append(result)
            if save_traces:
                trace_name = f"{profile}_{_slugify_case(case_name)}_{int(round(angle))}.json"
                save_trace_json(str(out_path / trace_name), meta=meta, hist=hist)

    results.sort(
        key=lambda rec: (
            _status_rank(rec['status']),
            -(rec['selected_aggressiveness'] or 0.0),
            -(rec['continuous_power_p95_pct'] or 0.0),
            -(rec['fan_tracking_rms_pct_mean_cmd'] or 0.0),
            -(rec['flap_limit_usage_peak_pct'] or 0.0),
            -(rec['yaw_hold_error_mean_abs_deg'] or 0.0),
            rec['fault_case'],
            rec['dir_b_deg'],
        )
    )

    summary = {
        'profile': profile,
        'angles_deg': angles,
        'fault_set': fault_set,
        'fault_cases': selected_fault_cases,
        'maneuver_safe': bool(maneuver_safe),
        'worst_status': (results[-1]['status'] if results else None),
        'counts': {
            'pass': sum(1 for rec in results if rec['status'] == 'pass'),
            'caution': sum(1 for rec in results if rec['status'] == 'caution'),
            'risk': sum(1 for rec in results if rec['status'] == 'risk'),
        },
        'results': results,
    }

    (out_path / 'summary.json').write_text(json.dumps(summary, indent=2))
    fieldnames = [
        'profile', 'dir_a_deg', 'dir_b_deg', 'fault_case', 'status', 'selected_aggressiveness', 'tuning_status',
        't_to_speed_below_thr_s', 'snap_stop_distance_m', 't_reversal_s', 't90_dir_s',
        'min_speed_transition_mps', 'peak_speed_mps', 'yaw_hold_error_mean_abs_deg', 'yaw_track_decoupling_mean_abs_deg',
        'xy_tracking_rms_n', 'alignment_p95_deg', 'flap_limit_usage_peak_pct',
        'flap_tracking_rms_deg', 'fan_tracking_rms_pct_mean_cmd', 'fan_response_min_pct',
        'continuous_power_p95_pct', 'thrust_scale_min_pct', 'warnings',
    ]
    with (out_path / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            row = dict(rec)
            row['warnings'] = ' | '.join(rec['warnings'])
            writer.writerow({key: row.get(key) for key in fieldnames})

    (out_path / 'summary.md').write_text(render_maneuver_pack_markdown(summary))
    typer.echo(json.dumps({
        'profile': profile,
        'out_dir': str(out_path),
        'maneuver_safe': bool(maneuver_safe),
        'cases': len(results),
        'worst_status': summary['worst_status'],
        'counts': summary['counts'],
    }, indent=2))
