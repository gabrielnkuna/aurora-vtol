from __future__ import annotations

import numpy as np

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.engineering import assess_trace
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.sim import run_step_redirect_v3, run_step_snap_v3, run_step_test_v3

MANEUVER_IGNORED_WARNINGS = {
    'final mission position error exceeded 1.5x arrival radius',
    'vehicle did not settle below 0.75 m/s at the end of the run',
    'minimum obstacle clearance fell below 2 m',
}


def _status_rank(status: str) -> int:
    return {'pass': 0, 'caution': 1, 'risk': 2}.get(status, 3)


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
    if profile in {'step-snap', 'step-snap-eco'}:
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
    if profile in {'step-snap', 'step-snap-eco'}:
        base_hold_frac = float(base_kwargs.get('redirect_hold_frac', -1.0))
        base_steer_scale = float(base_kwargs.get('redirect_steer_scale', 1.0))
        if profile == 'step-snap-eco':
            params['fxy_n'] *= 0.84 + 0.08 * g - 0.08 * turn_ratio * (1.0 - g)
            params['total_s'] *= 1.08 + 0.36 * turn_ratio * (1.0 - g)
            params['snap_stop_s'] = float(base_kwargs['snap_stop_s'] * (1.10 + (1.75 + 1.30 * turn_ratio) * (1.0 - g) + 0.20 * turn_ratio))
            params['brake_gain'] = float(base_kwargs['brake_gain'] * (0.68 + 0.18 * g + 0.05 * turn_ratio))
            if base_hold_frac < 0.0:
                params['redirect_hold_frac'] = float(np.clip(0.66 + 0.22 * turn_ratio + 0.16 * (1.0 - g), 0.62, 0.985))
            else:
                params['redirect_hold_frac'] = float(np.clip(max(base_hold_frac, 0.66), 0.0, 0.985))
            params['redirect_steer_scale'] = float(np.clip(base_steer_scale * (0.56 + 0.18 * g) * (1.0 - 0.12 * turn_ratio), 0.12, 1.1))
        else:
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
    burst_reserve_min_pct = _maneuver_scalar(thermal.get('burst_reserve_min_pct'))
    burst_clip_time_s = _maneuver_scalar(thermal.get('burst_clip_time_s'))
    flap_peak = _maneuver_scalar(flaps.get('limit_usage_peak_pct'))
    flap_tracking = _maneuver_scalar(flaps.get('tracking_rms_deg'))
    fan_tracking_pct = _maneuver_scalar(fans.get('tracking_rms_pct_mean_cmd'))
    fan_response_min_pct = _maneuver_scalar(fans.get('response_min_pct'))
    xy_tracking_rms = _maneuver_scalar(vectoring.get('xy_tracking_rms_n'))
    alignment_p95_deg = _maneuver_scalar(vectoring.get('alignment_p95_deg'))
    yaw_hold_error = _maneuver_scalar(maneuver.get('yaw_hold_error_mean_abs_deg'))
    turn_delta = _maneuver_scalar(maneuver.get('turn_delta_deg')) or 0.0
    requires_reversal = (profile == 'step-snap') or turn_delta >= 135.0
    if burst_reserve_min_pct is not None and burst_clip_time_s is not None:
        power_ok = burst_clip_time_s <= 0.05 and burst_reserve_min_pct >= 25.0
    else:
        power_ok = continuous_p95 is not None and continuous_p95 <= target_continuous_power_p95_pct

    return (
        (not requires_reversal or maneuver.get('t_reversal_s') is not None)
        and maneuver.get('t90_dir_s') is not None
        and (profile != 'step-snap' or maneuver.get('t_to_speed_below_thr_s') is not None)
        and power_ok
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
    burst_reserve_min_pct = _maneuver_scalar(thermal.get('burst_reserve_min_pct'))
    burst_clip_time_s = _maneuver_scalar(thermal.get('burst_clip_time_s'))
    flap_peak = _maneuver_scalar(flaps.get('limit_usage_peak_pct'))
    flap_tracking = _maneuver_scalar(flaps.get('tracking_rms_deg'))
    fan_tracking_pct = _maneuver_scalar(fans.get('tracking_rms_pct_mean_cmd'))
    fan_response_min_pct = _maneuver_scalar(fans.get('response_min_pct'))
    xy_tracking_rms = _maneuver_scalar(vectoring.get('xy_tracking_rms_n'))
    alignment_p95_deg = _maneuver_scalar(vectoring.get('alignment_p95_deg'))
    yaw_hold_error = _maneuver_scalar(maneuver.get('yaw_hold_error_mean_abs_deg'))
    turn_delta = _maneuver_scalar(maneuver.get('turn_delta_deg'))
    continuous_p95 = 1e9 if continuous_p95 is None else continuous_p95
    burst_reserve_min_pct = -1e9 if burst_reserve_min_pct is None else burst_reserve_min_pct
    burst_clip_time_s = 1e9 if burst_clip_time_s is None else burst_clip_time_s
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
    penalty += 0.40 * max(0.0, continuous_p95 - target_continuous_power_p95_pct) ** 2
    penalty += 1.5 * max(0.0, 25.0 - burst_reserve_min_pct) ** 2
    penalty += 8000.0 * max(0.0, burst_clip_time_s - 0.05) ** 2
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
        '| Status | Fault Case | Dir B (deg) | Aggr | Stop Time (s) | Reversal (s) | Align (s) | Yaw Hold Err (deg) | Flap Peak (%) | Fan Track (%) | Sust. P95 (%) | Raw P95 (%) | Burst Min (%) | Clip Time (s) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for rec in payload['results']:
        lines.append(
            '| {status} | {fault_case} | {dir_b_deg:.0f} | {aggr} | {stop} | {reversal} | {align} | {yaw} | {flap} | {fan} | {power} | {power_raw} | {burst_min} | {burst_clip} |'.format(
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
                power_raw=('n/a' if rec.get('continuous_power_raw_p95_pct') is None else f"{rec['continuous_power_raw_p95_pct']:.1f}"),
                burst_min=('n/a' if rec.get('burst_reserve_min_pct') is None else f"{rec['burst_reserve_min_pct']:.1f}"),
                burst_clip=('n/a' if rec.get('burst_clip_time_s') is None else f"{rec['burst_clip_time_s']:.2f}"),
            )
        )
    return '\n'.join(lines) + '\n'
