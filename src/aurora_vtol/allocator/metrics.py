from __future__ import annotations

import math

import numpy as np


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def track_angle_deg(vx: float, vy: float) -> float:
    return (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0


def track_angle_series_deg(vx, vy) -> np.ndarray:
    vx_arr = np.asarray(vx, float)
    vy_arr = np.asarray(vy, float)
    if vx_arr.shape != vy_arr.shape:
        raise ValueError('vx and vy must have the same shape')
    return (np.degrees(np.arctan2(vy_arr, vx_arr)) + 360.0) % 360.0


def time_to_track_alignment_s(
    t_s,
    vx,
    vy,
    speed,
    *,
    start_idx: int,
    target_deg: float,
    speed_threshold_mps: float = 0.5,
    angle_threshold_deg: float = 20.0,
) -> float | None:
    t_arr = np.asarray(t_s, float)
    sp_arr = np.asarray(speed, float)
    track_deg = track_angle_series_deg(vx, vy)
    for i in range(start_idx, len(t_arr)):
        if sp_arr[i] > speed_threshold_mps and abs(wrap180(float(track_deg[i] - target_deg))) <= angle_threshold_deg:
            return float(t_arr[i] - t_arr[start_idx])
    return None


def time_to_positive_target_projection_s(t_s, vx, vy, *, start_idx: int, target_deg: float) -> float | None:
    t_arr = np.asarray(t_s, float)
    vx_arr = np.asarray(vx, float)
    vy_arr = np.asarray(vy, float)
    tx = math.cos(math.radians(target_deg))
    ty = math.sin(math.radians(target_deg))
    for i in range(start_idx, len(t_arr)):
        proj = float(vx_arr[i] * tx + vy_arr[i] * ty)
        if proj > 0.0:
            return float(t_arr[i] - t_arr[start_idx])
    return None


def yaw_track_coupling_mean_abs(hist: dict) -> float:
    yaw = np.array(hist.get("yaw_deg", []), float)
    vx = np.array(hist.get("vx", []), float)
    vy = np.array(hist.get("vy", []), float)
    if yaw.size == 0 or vx.size == 0:
        return float("nan")
    sp = np.hypot(vx, vy)
    m = sp > 0.2
    if not np.any(m):
        return float("nan")
    ta = np.array([track_angle_deg(vx[i], vy[i]) for i in range(len(vx))], float)
    err = np.array([abs(wrap180(float(yaw[i] - ta[i]))) for i in range(len(yaw))], float)
    return float(np.mean(err[m]))
