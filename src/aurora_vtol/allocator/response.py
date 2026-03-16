from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StepMetrics:
    response_latency_s: float | None
    peak_speed_mps: float | None
    overshoot_ratio: float | None
    settle_time_s: float | None
    yaw_track_coupling_mean_abs_deg: float | None

def first_time_above(t: np.ndarray, y: np.ndarray, thr: float) -> float | None:
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return None
    return float(t[idx[0]])

def settle_time(t: np.ndarray, y: np.ndarray, band_lo: float, band_hi: float, hold_s: float) -> float | None:
    for i in range(len(t)):
        t0 = t[i]
        t1 = t0 + hold_s
        j = np.searchsorted(t, t1, side="left")
        if j <= i:
            continue
        seg = y[i:j]
        if np.all((seg >= band_lo) & (seg <= band_hi)):
            return float(t0)
    return None

def compute_step_metrics(t: np.ndarray, speed: np.ndarray, target_speed: float, yaw_coupling: float | None) -> StepMetrics:
    if t.size == 0:
        return StepMetrics(None, None, None, None, yaw_coupling)
    peak = float(np.max(speed)) if speed.size else None
    lat = first_time_above(t, speed, 0.30 * target_speed)
    overshoot = None
    if peak is not None and target_speed > 1e-6:
        overshoot = float((peak - target_speed) / target_speed)
    st = settle_time(t, speed, 0.90 * target_speed, 1.10 * target_speed, hold_s=1.0)
    settle = float(st) if st is not None else None
    return StepMetrics(lat, peak, overshoot, settle, yaw_coupling)
