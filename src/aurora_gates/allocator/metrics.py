from __future__ import annotations
import math
import numpy as np

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def track_angle_deg(vx: float, vy: float) -> float:
    return (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0

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
