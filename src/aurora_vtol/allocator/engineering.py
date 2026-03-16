from __future__ import annotations

import json
import math

import numpy as np

from .model import RingGeometry
from .sim import run_coordinate_mission_v5


def _series(hist: dict, key: str) -> np.ndarray:
    values = hist.get(key, [])
    if not values:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def _matrix(hist: dict, key: str) -> np.ndarray:
    values = hist.get(key, [])
    if not values:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _scalar(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _rms(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _percentile(values: np.ndarray, pct: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, pct))


def _final_mission_error(meta: dict, hist: dict) -> float | None:
    arrival = meta.get('arrival', {})
    final_goal_error_m = _scalar(arrival.get('final_goal_error_m'))
    if final_goal_error_m is not None:
        return final_goal_error_m
    mission = meta.get('mission', {})
    destination = mission.get('destination', {})
    dest_x = _scalar(destination.get('x_m'))
    dest_y = _scalar(destination.get('y_m'))
    x = _series(hist, 'x')
    y = _series(hist, 'y')
    if dest_x is None or dest_y is None or x.size == 0 or y.size == 0:
        return None
    return float(math.hypot(x[-1] - dest_x, y[-1] - dest_y))


def assess_trace(meta: dict, hist: dict, geom: RingGeometry | None = None) -> dict:
    geom = geom or RingGeometry()
    sim = meta.get('sim', {})
    mission = meta.get('mission', {})
    arrival = meta.get('arrival', {})
    power_cfg = meta.get('power', {})

    mass_kg = float(sim.get('mass_kg', 800.0))
    gravity = float(sim.get('gravity', 9.81))
    dt_s = float(sim.get('dt_s', 0.02))
    weight_n = mass_kg * gravity
    alpha_limit_deg = float(getattr(geom, 'alpha_max_deg', 25.0))

    fx_cmd = _series(hist, 'fx_cmd')
    fy_cmd = _series(hist, 'fy_cmd')
    fz_cmd = _series(hist, 'fz_cmd')
    fx_ach = _series(hist, 'fx_ach')
    fy_ach = _series(hist, 'fy_ach')
    fz_ach = _series(hist, 'fz_ach')
    vx = _series(hist, 'vx')
    vy = _series(hist, 'vy')
    vz = _series(hist, 'vz')
    speed = _series(hist, 'speed')
    z = _series(hist, 'z')

    battery_v = _series(hist, 'battery_v')
    battery_a = _series(hist, 'battery_a')
    battery_soc = _series(hist, 'battery_soc')
    power_w = _series(hist, 'power_w')
    energy_wh = _series(hist, 'energy_wh')
    thrust_scale_pct = _series(hist, 'thrust_scale_pct')
    continuous_power_raw_pct = _series(hist, 'continuous_power_raw_pct')
    if continuous_power_raw_pct.size == 0:
        continuous_power_raw_pct = _series(hist, 'continuous_power_pct')
    sustained_power_pct = _series(hist, 'sustained_power_pct')
    if sustained_power_pct.size == 0:
        sustained_power_pct = _series(hist, 'continuous_power_pct')
    power_margin_kw = _series(hist, 'power_margin_kw')
    burst_reserve_pct = _series(hist, 'burst_reserve_pct')
    burst_clip_pct = _series(hist, 'burst_clip_pct')
    burst_active_time_hist = _series(hist, 'burst_active_time_s')
    thermal_scale_pct = _series(hist, 'thermal_scale_pct')
    fan_temp_max_c = _series(hist, 'fan_temp_max_c')
    fan_temp_mean_c = _series(hist, 'fan_temp_mean_c')
    fan_response_pct = _series(hist, 'fan_response_pct')

    guard_scale = _series(hist, 'guard_scale')
    budget_ratio = _series(hist, 'budget_ratio')
    flap_guard_scale = _series(hist, 'flap_guard_scale')
    power_guard_scale = _series(hist, 'power_guard_scale')
    thermal_guard_scale = _series(hist, 'thermal_guard_scale')
    supply_guard_scale = _series(hist, 'supply_guard_scale')
    speed_guard_scale = _series(hist, 'speed_guard_scale')
    gain_guard_scale = _series(hist, 'gain_guard_scale')
    goal_force_scale = _series(hist, 'goal_force_scale')
    guard_continuous_ratio = _series(hist, 'continuous_power_ratio')
    guard_flap_usage_ratio = _series(hist, 'flap_usage_ratio')

    fan_actual = _matrix(hist, 'fan_actual_16')
    if fan_actual.size == 0:
        fan_actual = _matrix(hist, 'fan_thrust_16')
    fan_cmd = _matrix(hist, 'fan_cmd_16')
    flap_actual = _matrix(hist, 'alpha_deg_32')
    flap_cmd = _matrix(hist, 'alpha_cmd_deg_32')

    xy_cmd_mag = np.hypot(fx_cmd, fy_cmd) if fx_cmd.size and fy_cmd.size else np.asarray([], dtype=float)
    xy_ach_mag = np.hypot(fx_ach, fy_ach) if fx_ach.size and fy_ach.size else np.asarray([], dtype=float)
    xy_err_mag = np.hypot(fx_ach - fx_cmd, fy_ach - fy_cmd) if fx_cmd.size and fx_ach.size else np.asarray([], dtype=float)
    active_xy = xy_cmd_mag > 50.0 if xy_cmd_mag.size else np.asarray([], dtype=bool)

    alignment_deg = np.asarray([], dtype=float)
    gain_pct = np.asarray([], dtype=float)
    if active_xy.size and np.any(active_xy) and xy_ach_mag.size:
        safe_cmd = np.maximum(xy_cmd_mag[active_xy], 1e-6)
        safe_ach = np.maximum(xy_ach_mag[active_xy], 1e-6)
        dot = fx_cmd[active_xy] * fx_ach[active_xy] + fy_cmd[active_xy] * fy_ach[active_xy]
        cos_ang = np.clip(dot / (safe_cmd * safe_ach), -1.0, 1.0)
        alignment_deg = np.degrees(np.arccos(cos_ang))
        gain_pct = 100.0 * xy_ach_mag[active_xy] / safe_cmd

    lift_support_pct = np.asarray([], dtype=float)
    lift_margin_n = np.asarray([], dtype=float)
    if fz_ach.size:
        lift_support_pct = 100.0 * fz_ach / max(weight_n, 1e-6)
        lift_margin_n = fz_ach - weight_n

    flap_abs = np.abs(flap_actual) if flap_actual.size else np.empty((0, 0), dtype=float)
    flap_cmd_tracking = flap_actual - flap_cmd if flap_actual.size and flap_cmd.shape == flap_actual.shape else np.empty((0, 0), dtype=float)
    flap_limit_usage_pct = 100.0 * flap_abs / max(alpha_limit_deg, 1e-6) if flap_abs.size else np.empty((0, 0), dtype=float)
    flap_hot_mask = np.any(flap_limit_usage_pct >= 80.0, axis=1) if flap_limit_usage_pct.size else np.asarray([], dtype=bool)

    fan_tracking = fan_actual - fan_cmd if fan_actual.size and fan_cmd.shape == fan_actual.shape else np.empty((0, 0), dtype=float)
    fan_tracking_rms_n = _rms(fan_tracking.reshape(-1)) if fan_tracking.size else None
    mean_fan_cmd = float(np.mean(np.abs(fan_cmd))) if fan_cmd.size else 0.0
    fan_tracking_rms_pct_mean_cmd = (100.0 * fan_tracking_rms_n / mean_fan_cmd) if (fan_tracking_rms_n is not None and mean_fan_cmd > 1e-6) else None
    fan_balance_pct = np.asarray([], dtype=float)
    if fan_actual.size:
        fan_mean = np.maximum(np.mean(fan_actual, axis=1), 1e-6)
        fan_balance_pct = 100.0 * np.std(fan_actual, axis=1) / fan_mean

    mission_error_m = _final_mission_error(meta, hist)
    final_speed_mps = _scalar(speed[-1]) if speed.size else _scalar(math.hypot(vx[-1], vy[-1]) if vx.size and vy.size else None)
    final_alt_m = _scalar(z[-1]) if z.size else None
    arrival_radius_m = _scalar(meta.get('guidance', {}).get('arrival_radius_m'))
    min_obstacle_clearance_m = _scalar(arrival.get('min_obstacle_clearance_m'))

    guard_active_mask = budget_ratio < 0.999 if budget_ratio.size else np.asarray([], dtype=bool)
    guard_heavy_mask = budget_ratio < 0.75 if budget_ratio.size else np.asarray([], dtype=bool)
    burst_clip_mask = burst_clip_pct > 0.1 if burst_clip_pct.size else np.asarray([], dtype=bool)
    burst_active_mask = continuous_power_raw_pct > 100.0 if continuous_power_raw_pct.size else np.asarray([], dtype=bool)
    burst_clip_time_s = _scalar(np.sum(burst_clip_mask) * dt_s if burst_clip_mask.size else 0.0)
    burst_active_time_s = _scalar(np.sum(burst_active_mask) * dt_s if burst_active_mask.size else (np.sum(burst_active_time_hist) if burst_active_time_hist.size else 0.0))
    burst_reserve_min_pct = _scalar(np.min(burst_reserve_pct) if burst_reserve_pct.size else None)

    warnings: list[str] = []
    if thrust_scale_pct.size and float(np.min(thrust_scale_pct)) < 90.0:
        warnings.append('power-thrust derating fell below 90%')
    if burst_clip_time_s is not None and burst_clip_time_s > 0.05:
        warnings.append('burst-power reserve was exhausted and thrust had to be clipped')
    elif continuous_power_raw_pct.size and float(np.percentile(continuous_power_raw_pct, 95.0)) > 100.0:
        if burst_reserve_min_pct is not None and burst_reserve_min_pct < 25.0:
            warnings.append('continuous power demand relied too heavily on burst reserve')
    if thermal_scale_pct.size and float(np.min(thermal_scale_pct)) < 95.0:
        warnings.append('thermal derating pulled available thrust below 95%')
    if battery_v.size and power_cfg:
        battery_empty_v = float(power_cfg.get('battery_empty_v', 46.0))
        if float(np.min(battery_v)) <= battery_empty_v + 1.0:
            warnings.append('battery voltage approached the configured empty threshold')
    if flap_limit_usage_pct.size and float(np.max(flap_limit_usage_pct)) >= 95.0:
        warnings.append('flap deflection reached at least 95% of the configured limit')
    if flap_cmd_tracking.size:
        flap_tracking_rms_deg = _rms(flap_cmd_tracking.reshape(-1))
        if flap_tracking_rms_deg is not None and flap_tracking_rms_deg > 2.0:
            warnings.append('flap command tracking RMS exceeded 2 degrees')
    if fan_tracking_rms_pct_mean_cmd is not None and fan_tracking_rms_pct_mean_cmd > 10.0:
        warnings.append('fan thrust tracking RMS exceeded 10% of mean command')
    if fan_response_pct.size and float(np.min(fan_response_pct)) < 90.0:
        warnings.append('fan response fell below 90% of commanded thrust during the run')
    if alignment_deg.size and float(np.percentile(alignment_deg, 95)) > 20.0:
        warnings.append('vector alignment p95 exceeded 20 degrees')
    if xy_err_mag.size and active_xy.size and np.any(active_xy):
        tracking_rms_n = _rms(xy_err_mag[active_xy])
        if tracking_rms_n is not None and tracking_rms_n > 400.0:
            warnings.append('XY force tracking RMS exceeded 400 N during active vectoring')
    if mission_error_m is not None and arrival_radius_m is not None and mission_error_m > arrival_radius_m * 1.5:
        warnings.append('final mission position error exceeded 1.5x arrival radius')
    if final_speed_mps is not None and final_speed_mps > 0.75:
        warnings.append('vehicle did not settle below 0.75 m/s at the end of the run')
    if min_obstacle_clearance_m is not None and min_obstacle_clearance_m < 2.0:
        warnings.append('minimum obstacle clearance fell below 2 m')
    if fan_temp_max_c.size and float(np.max(fan_temp_max_c)) >= 85.0:
        warnings.append('fan temperature entered the thermal soft-limit region')

    status = 'pass' if not warnings else ('caution' if len(warnings) <= 3 else 'risk')

    return {
        'status': status,
        'trace_version': meta.get('version'),
        'warnings': warnings,
        'mission': {
            'preset': mission.get('preset'),
            'arrival_time_s': _scalar(arrival.get('arrival_time_s')),
            'arrived': bool(arrival.get('arrived', False)),
            'hold_complete': bool(arrival.get('hold_complete', False)),
            'final_goal_error_m': mission_error_m,
            'final_alt_m': final_alt_m,
            'final_speed_mps': final_speed_mps,
            'arrival_radius_m': arrival_radius_m,
            'min_obstacle_clearance_m': min_obstacle_clearance_m,
        },
        'lift': {
            'weight_n': float(weight_n),
            'support_mean_pct': _scalar(np.mean(lift_support_pct) if lift_support_pct.size else None),
            'support_min_pct': _scalar(np.min(lift_support_pct) if lift_support_pct.size else None),
            'support_p05_pct': _percentile(lift_support_pct, 5.0),
            'margin_min_n': _scalar(np.min(lift_margin_n) if lift_margin_n.size else None),
            'margin_p05_n': _percentile(lift_margin_n, 5.0),
        },
        'vectoring': {
            'xy_command_peak_n': _scalar(np.max(xy_cmd_mag) if xy_cmd_mag.size else None),
            'xy_achieved_peak_n': _scalar(np.max(xy_ach_mag) if xy_ach_mag.size else None),
            'xy_tracking_rms_n': _rms(xy_err_mag[active_xy]) if active_xy.size and np.any(active_xy) else None,
            'alignment_mean_deg': _scalar(np.mean(alignment_deg) if alignment_deg.size else None),
            'alignment_p95_deg': _percentile(alignment_deg, 95.0),
            'gain_mean_pct': _scalar(np.mean(gain_pct) if gain_pct.size else None),
        },
        'flaps': {
            'limit_deg': float(alpha_limit_deg),
            'actual_peak_deg': _scalar(np.max(flap_abs) if flap_abs.size else None),
            'limit_usage_peak_pct': _scalar(np.max(flap_limit_usage_pct) if flap_limit_usage_pct.size else None),
            'limit_usage_p95_pct': _percentile(flap_limit_usage_pct.reshape(-1), 95.0) if flap_limit_usage_pct.size else None,
            'tracking_rms_deg': _rms(flap_cmd_tracking.reshape(-1)) if flap_cmd_tracking.size else None,
            'time_above_80pct_s': _scalar(np.sum(flap_hot_mask) * dt_s if flap_hot_mask.size else 0.0),
        },
        'fans': {
            'actual_peak_n': _scalar(np.max(fan_actual) if fan_actual.size else None),
            'command_peak_n': _scalar(np.max(fan_cmd) if fan_cmd.size else None),
            'command_mean_n': _scalar(mean_fan_cmd if fan_cmd.size else None),
            'tracking_rms_n': fan_tracking_rms_n,
            'tracking_rms_pct_mean_cmd': _scalar(fan_tracking_rms_pct_mean_cmd),
            'balance_mean_pct': _scalar(np.mean(fan_balance_pct) if fan_balance_pct.size else None),
            'balance_peak_pct': _scalar(np.max(fan_balance_pct) if fan_balance_pct.size else None),
            'response_min_pct': _scalar(np.min(fan_response_pct) if fan_response_pct.size else None),
            'response_mean_pct': _scalar(np.mean(fan_response_pct) if fan_response_pct.size else None),
        },
        'thermal': {
            'fan_temp_mean_c': _scalar(np.mean(fan_temp_mean_c) if fan_temp_mean_c.size else None),
            'fan_temp_peak_c': _scalar(np.max(fan_temp_max_c) if fan_temp_max_c.size else None),
            'thermal_scale_min_pct': _scalar(np.min(thermal_scale_pct) if thermal_scale_pct.size else None),
            'thermal_scale_p05_pct': _percentile(thermal_scale_pct, 5.0),
            'continuous_power_p95_pct': _percentile(sustained_power_pct, 95.0),
            'continuous_power_peak_pct': _scalar(np.max(sustained_power_pct) if sustained_power_pct.size else None),
            'continuous_power_raw_p95_pct': _percentile(continuous_power_raw_pct, 95.0),
            'continuous_power_raw_peak_pct': _scalar(np.max(continuous_power_raw_pct) if continuous_power_raw_pct.size else None),
            'burst_reserve_min_pct': _scalar(np.min(burst_reserve_pct) if burst_reserve_pct.size else None),
            'burst_reserve_p05_pct': _percentile(burst_reserve_pct, 5.0),
            'burst_clip_time_s': burst_clip_time_s,
            'burst_active_time_s': burst_active_time_s,
            'power_margin_min_kw': _scalar(np.min(power_margin_kw) if power_margin_kw.size else None),
        },
        'guard': {
            'budget_mean_pct': _scalar(100.0 * np.mean(budget_ratio) if budget_ratio.size else None),
            'budget_min_pct': _scalar(100.0 * np.min(budget_ratio) if budget_ratio.size else None),
            'guard_active_time_s': _scalar(np.sum(guard_active_mask) * dt_s if guard_active_mask.size else 0.0),
            'guard_heavy_time_s': _scalar(np.sum(guard_heavy_mask) * dt_s if guard_heavy_mask.size else 0.0),
            'speed_guard_mean_pct': _scalar(100.0 * np.mean(speed_guard_scale) if speed_guard_scale.size else None),
            'gain_guard_mean_pct': _scalar(100.0 * np.mean(gain_guard_scale) if gain_guard_scale.size else None),
            'goal_force_scale_p05_pct': _percentile(100.0 * goal_force_scale, 5.0) if goal_force_scale.size else None,
            'flap_guard_min_pct': _scalar(100.0 * np.min(flap_guard_scale) if flap_guard_scale.size else None),
            'power_guard_min_pct': _scalar(100.0 * np.min(power_guard_scale) if power_guard_scale.size else None),
            'thermal_guard_min_pct': _scalar(100.0 * np.min(thermal_guard_scale) if thermal_guard_scale.size else None),
            'supply_guard_min_pct': _scalar(100.0 * np.min(supply_guard_scale) if supply_guard_scale.size else None),
            'guard_continuous_power_peak_pct': _scalar(100.0 * np.max(guard_continuous_ratio) if guard_continuous_ratio.size else None),
            'guard_flap_usage_peak_pct': _scalar(100.0 * np.max(guard_flap_usage_ratio) if guard_flap_usage_ratio.size else None),
        },
        'power': {
            'voltage_min_v': _scalar(np.min(battery_v) if battery_v.size else None),
            'voltage_mean_v': _scalar(np.mean(battery_v) if battery_v.size else None),
            'current_peak_a': _scalar(np.max(battery_a) if battery_a.size else None),
            'power_peak_kw': _scalar(np.max(power_w) / 1000.0 if power_w.size else None),
            'energy_used_wh': _scalar(energy_wh[-1] if energy_wh.size else None),
            'soc_end_pct': _scalar(battery_soc[-1] if battery_soc.size else None),
            'thrust_scale_min_pct': _scalar(np.min(thrust_scale_pct) if thrust_scale_pct.size else None),
        },
    }



def _coordinate_tune_snapshot(params: dict, aggressiveness: float) -> dict:
    snap = {
        'aggressiveness': float(aggressiveness),
    }
    for key in (
        'max_speed_mps',
        'fxy_max_n',
        'pos_k_n_per_m',
        'vel_k_n_per_mps',
        'slow_radius_m',
        'descent_radius_m',
        'total_s',
        'hold_s',
    ):
        if key in params:
            snap[key] = float(params[key])
    return snap


def _apply_coordinate_aggressiveness(base_kwargs: dict, aggressiveness: float) -> dict:
    g = max(0.25, min(1.0, float(aggressiveness)))
    params = dict(base_kwargs)
    params['max_speed_mps'] = float(base_kwargs['max_speed_mps'] * (0.45 + 0.55 * g))
    params['fxy_max_n'] = float(base_kwargs['fxy_max_n'] * (0.55 + 0.45 * g))
    params['pos_k_n_per_m'] = float(base_kwargs['pos_k_n_per_m'] * (0.60 + 0.40 * g))
    params['vel_k_n_per_mps'] = float(base_kwargs['vel_k_n_per_mps'] * (0.80 + 0.20 * g))
    params['slow_radius_m'] = float(base_kwargs['slow_radius_m'] * (1.0 + 0.85 * (1.0 - g)))
    params['descent_radius_m'] = float(max(
        base_kwargs['descent_radius_m'] * (1.0 + 0.35 * (1.0 - g)),
        params['slow_radius_m'] * 1.15,
    ))
    params['total_s'] = float(base_kwargs['total_s'] * (1.0 + 0.35 * (1.0 - g)))
    params['hold_s'] = float(base_kwargs.get('hold_s', 2.0) + 1.5 * (1.0 - g))
    return params


def _coordinate_tune_passes(
    report: dict,
    target_continuous_power_p95_pct: float,
    target_flap_peak_pct: float,
    target_fan_tracking_pct: float,
    max_final_goal_error_ratio: float,
) -> bool:
    mission = report.get('mission', {})
    thermal = report.get('thermal', {})
    flaps = report.get('flaps', {})
    fans = report.get('fans', {})
    arrived = bool(mission.get('arrived', False)) and bool(mission.get('hold_complete', False))
    continuous_p95 = _scalar(thermal.get('continuous_power_p95_pct'))
    flap_peak = _scalar(flaps.get('limit_usage_peak_pct'))
    fan_tracking_pct = _scalar(fans.get('tracking_rms_pct_mean_cmd'))
    final_goal_error = _scalar(mission.get('final_goal_error_m'))
    arrival_radius = max(1.0, _scalar(mission.get('arrival_radius_m')) or 1.0)
    final_speed = _scalar(mission.get('final_speed_mps'))
    allowed_goal_error = max(arrival_radius, arrival_radius * max_final_goal_error_ratio)
    return (
        arrived
        and continuous_p95 is not None and continuous_p95 <= target_continuous_power_p95_pct
        and flap_peak is not None and flap_peak <= target_flap_peak_pct
        and fan_tracking_pct is not None and fan_tracking_pct <= target_fan_tracking_pct
        and final_goal_error is not None and final_goal_error <= allowed_goal_error
        and final_speed is not None and final_speed <= 0.75
    )


def _coordinate_tune_penalty(
    report: dict,
    target_continuous_power_p95_pct: float,
    target_flap_peak_pct: float,
    target_fan_tracking_pct: float,
    max_final_goal_error_ratio: float,
) -> float:
    mission = report.get('mission', {})
    thermal = report.get('thermal', {})
    flaps = report.get('flaps', {})
    fans = report.get('fans', {})
    guard = report.get('guard', {})

    continuous_p95 = _scalar(thermal.get('continuous_power_p95_pct')) or 1e9
    flap_peak = _scalar(flaps.get('limit_usage_peak_pct')) or 1e9
    fan_tracking_pct = _scalar(fans.get('tracking_rms_pct_mean_cmd')) or 1e9
    final_goal_error = _scalar(mission.get('final_goal_error_m')) or 1e9
    arrival_radius = max(1.0, _scalar(mission.get('arrival_radius_m')) or 1.0)
    final_speed = _scalar(mission.get('final_speed_mps')) or 1e9
    guard_heavy_time_s = _scalar(guard.get('guard_heavy_time_s')) or 0.0
    arrived = bool(mission.get('arrived', False)) and bool(mission.get('hold_complete', False))

    allowed_goal_error = max(arrival_radius, arrival_radius * max_final_goal_error_ratio)
    penalty = 0.0
    if not arrived:
        penalty += 5000.0
    penalty += 2.0 * max(0.0, continuous_p95 - target_continuous_power_p95_pct) ** 2
    penalty += 0.75 * max(0.0, flap_peak - target_flap_peak_pct) ** 2
    penalty += 1.5 * max(0.0, fan_tracking_pct - target_fan_tracking_pct) ** 2
    penalty += 300.0 * max(0.0, final_goal_error - allowed_goal_error) ** 2
    penalty += 400.0 * max(0.0, final_speed - 0.75) ** 2
    penalty += 0.25 * guard_heavy_time_s
    return float(penalty)


def _evaluate_coordinate_aggressiveness(
    base_kwargs: dict,
    aggressiveness: float,
    *,
    target_continuous_power_p95_pct: float,
    target_flap_peak_pct: float,
    target_fan_tracking_pct: float,
    max_final_goal_error_ratio: float,
) -> dict:
    params = _apply_coordinate_aggressiveness(base_kwargs, aggressiveness)
    meta, hist = run_coordinate_mission_v5(**params)
    report = assess_trace(meta, hist)
    passed = _coordinate_tune_passes(
        report,
        target_continuous_power_p95_pct=target_continuous_power_p95_pct,
        target_flap_peak_pct=target_flap_peak_pct,
        target_fan_tracking_pct=target_fan_tracking_pct,
        max_final_goal_error_ratio=max_final_goal_error_ratio,
    )
    score = _coordinate_tune_penalty(
        report,
        target_continuous_power_p95_pct=target_continuous_power_p95_pct,
        target_flap_peak_pct=target_flap_peak_pct,
        target_fan_tracking_pct=target_fan_tracking_pct,
        max_final_goal_error_ratio=max_final_goal_error_ratio,
    )
    record = {
        'aggressiveness': float(aggressiveness),
        'passed': bool(passed),
        'score': float(score),
        'params': _coordinate_tune_snapshot(params, aggressiveness),
        'arrival_time_s': _scalar(report.get('mission', {}).get('arrival_time_s')),
        'final_goal_error_m': _scalar(report.get('mission', {}).get('final_goal_error_m')),
        'continuous_power_p95_pct': _scalar(report.get('thermal', {}).get('continuous_power_p95_pct')),
        'flap_limit_usage_peak_pct': _scalar(report.get('flaps', {}).get('limit_usage_peak_pct')),
        'fan_tracking_rms_pct_mean_cmd': _scalar(report.get('fans', {}).get('tracking_rms_pct_mean_cmd')),
        'xy_tracking_rms_n': _scalar(report.get('vectoring', {}).get('xy_tracking_rms_n')),
        'guard_heavy_time_s': _scalar(report.get('guard', {}).get('guard_heavy_time_s')),
        'warnings': list(report.get('warnings', [])),
    }
    return {
        'meta': meta,
        'hist': hist,
        'report': report,
        'record': record,
    }


def tune_coordinate_mission(
    base_kwargs: dict,
    *,
    target_continuous_power_p95_pct: float = 100.0,
    target_flap_peak_pct: float = 90.0,
    target_fan_tracking_pct: float = 10.0,
    max_final_goal_error_ratio: float = 1.5,
    min_aggressiveness: float = 0.25,
    max_iterations: int = 6,
) -> tuple[dict, dict, dict, dict]:
    min_aggressiveness = max(0.25, min(1.0, float(min_aggressiveness)))
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
            bundle = _evaluate_coordinate_aggressiveness(
                base_kwargs,
                float(g),
                target_continuous_power_p95_pct=target_continuous_power_p95_pct,
                target_flap_peak_pct=target_flap_peak_pct,
                target_fan_tracking_pct=target_fan_tracking_pct,
                max_final_goal_error_ratio=max_final_goal_error_ratio,
            )
            bundles_by_key[key] = bundle
            evaluations.append(bundle['record'])
        return bundle

    coarse_bundles = [evaluate(g) for g in coarse_values]
    for bundle in coarse_bundles:
        if best_bundle is None or bundle['record']['score'] < best_bundle['record']['score']:
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
            if best_bundle is None or mid_bundle['record']['score'] < best_bundle['record']['score']:
                best_bundle = mid_bundle
        bracket_low = low_bundle['record']['aggressiveness']
        bracket_high = high_bundle['record']['aggressiveness']

    selected = best_pass_bundle or best_bundle
    iterations = sorted(evaluations, key=lambda item: item['aggressiveness'], reverse=True)
    feasible = best_pass_bundle is not None
    tuning = {
        'enabled': True,
        'mode': 'sustainable-envelope-search',
        'status': ('tuned' if feasible else 'best-effort'),
        'targets': {
            'continuous_power_p95_pct': float(target_continuous_power_p95_pct),
            'flap_limit_usage_peak_pct': float(target_flap_peak_pct),
            'fan_tracking_rms_pct_mean_cmd': float(target_fan_tracking_pct),
            'max_final_goal_error_ratio': float(max_final_goal_error_ratio),
            'min_aggressiveness': float(min_aggressiveness),
            'coarse_samples': int(coarse_samples),
            'refine_iterations': int(max_iterations),
        },
        'baseline': iterations[0],
        'selected': selected['record'],
        'envelope': {
            'feasible': bool(feasible),
            'max_sustainable_aggressiveness': (best_pass_bundle['record']['aggressiveness'] if feasible else None),
            'next_unsafe_aggressiveness': (bracket_high if feasible else None),
            'search_floor_aggressiveness': float(min_aggressiveness),
            'margin_to_full_scale': (1.0 - best_pass_bundle['record']['aggressiveness'] if feasible else None),
        },
        'iterations': iterations,
    }
    return selected['meta'], selected['hist'], selected['report'], tuning


def render_assessment(report: dict, fmt: str) -> str:
    if fmt == 'json':
        return json.dumps(report, indent=2)

    warnings = report.get('warnings', [])
    heading = 'Engineering Trace Assessment' if fmt == 'text' else '# Engineering Trace Assessment'
    lines = [heading]
    lines.append('')
    lines.append(f"status = {report.get('status')}") if fmt == 'text' else lines.append(f"Status: `{report.get('status')}`")
    lines.append(f"trace_version = {report.get('trace_version')}") if fmt == 'text' else lines.append(f"Trace version: `{report.get('trace_version')}`")

    def add_section(name: str, values: dict):
        if fmt == 'md':
            lines.append('')
            lines.append(f'## {name}')
            for key, value in values.items():
                lines.append(f'- `{key}` = `{value}`')
        else:
            lines.append('')
            lines.append(f'[{name}]')
            for key, value in values.items():
                lines.append(f'{key} = {value}')

    add_section('mission', report.get('mission', {}))
    add_section('lift', report.get('lift', {}))
    add_section('vectoring', report.get('vectoring', {}))
    add_section('flaps', report.get('flaps', {}))
    add_section('fans', report.get('fans', {}))
    add_section('thermal', report.get('thermal', {}))
    add_section('guard', report.get('guard', {}))
    add_section('power', report.get('power', {}))

    if fmt == 'md':
        lines.append('')
        lines.append('## warnings')
        if warnings:
            for warning in warnings:
                lines.append(f'- {warning}')
        else:
            lines.append('- none')
    else:
        lines.append('')
        lines.append('[warnings]')
        if warnings:
            lines.extend(warnings)
        else:
            lines.append('none')

    return '\n'.join(lines) + '\n'
