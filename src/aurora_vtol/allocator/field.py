from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RepelField:
    k_n_per_m: float = 120.0
    radius_m: float = 30.0
    fxy_max_n: float = 4000.0
    soft_zone_m: float = 5.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def repel_force_xy(field: RepelField, x_m: float, y_m: float, ox_m: float, oy_m: float) -> tuple[float, float]:
    dx = x_m - ox_m
    dy = y_m - oy_m
    d = math.hypot(dx, dy)
    if d >= field.radius_m:
        return (0.0, 0.0)
    pen = field.radius_m - d
    d_eff = max(d, field.soft_zone_m)
    # add a kick at entry to make the field feel like a hard wall
    kick = 0.35 * field.fxy_max_n  # 35% of max force immediately at boundary
    mag = clamp(kick + field.k_n_per_m * pen, 0.0, field.fxy_max_n)
    fx = mag * (dx / d_eff)
    fy = mag * (dy / d_eff)
    return (fx, fy)
