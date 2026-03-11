from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .model import RingGeometry, segment_angles_rad, thrust_vectors_body, net_force_and_yaw_moment

@dataclass(frozen=True)
class AllocationRequest:
    fx_n: float
    fy_n: float
    fz_n: float
    mz_nm: float = 0.0

@dataclass(frozen=True)
class AllocationResult:
    thrust_per_seg_n: np.ndarray
    alpha_rad: np.ndarray
    dir_rad: np.ndarray
    net_force_n: np.ndarray
    net_mz_nm: float
    note: str

@dataclass(frozen=True)
class AllocationResultV2(AllocationResult):
    ft_tan_per_seg_n: np.ndarray

def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def allocate_v1(geom: RingGeometry, req: AllocationRequest) -> AllocationResult:
    n = geom.n_segments
    theta = segment_angles_rad(n)
    fxy = math.hypot(req.fx_n, req.fy_n)
    thrust_base = req.fz_n / n

    if fxy < 1e-6:
        thrust = np.full(n, thrust_base, dtype=float)
        alpha = np.zeros(n, dtype=float)
        dirr = theta.copy()
        forces = thrust_vectors_body(geom, thrust, alpha, dirr)
        net, mz = net_force_and_yaw_moment(geom, forces)
        return AllocationResult(thrust, alpha, dirr, net, mz, "hover-only")

    phi = math.atan2(req.fy_n, req.fx_n)
    pattern = np.cos(theta - phi)
    alpha_max = math.radians(geom.alpha_max_deg)

    def fxy_for_k(k: float) -> float:
        a = clamp(k * pattern, -alpha_max, alpha_max)
        return float(np.sum(thrust_base * np.sin(a) * pattern))

    lo, hi = 0.0, 10.0
    target = fxy
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        val = fxy_for_k(mid)
        if val < target:
            lo = mid
        else:
            hi = mid

    alpha = clamp(hi * pattern, -alpha_max, alpha_max)
    thrust = np.full(n, thrust_base, dtype=float)
    dirr = theta.copy()
    forces = thrust_vectors_body(geom, thrust, alpha, dirr)
    net, mz = net_force_and_yaw_moment(geom, forces)
    return AllocationResult(thrust, alpha, dirr, net, mz, "V1: cosine alpha distribution")

def allocate_v2(geom: RingGeometry, req: AllocationRequest) -> AllocationResultV2:
    v1 = allocate_v1(geom, req)
    n = geom.n_segments
    R = geom.radius_m
    thrust = v1.thrust_per_seg_n

    if abs(req.mz_nm) < 1e-6:
        ft = np.zeros(n, dtype=float)
    else:
        ft = np.full(n, float(req.mz_nm) / (R * n), dtype=float)

    ft = np.clip(ft, -0.20 * thrust, 0.20 * thrust)

    theta = segment_angles_rad(n)
    fx_t = ft * (-np.sin(theta))
    fy_t = ft * (np.cos(theta))
    f_tan = np.stack([fx_t, fy_t, np.zeros(n, dtype=float)], axis=1)

    forces_v1 = thrust_vectors_body(geom, v1.thrust_per_seg_n, v1.alpha_rad, v1.dir_rad)
    forces = forces_v1 + f_tan
    net, mz = net_force_and_yaw_moment(geom, forces)

    return AllocationResultV2(
        thrust_per_seg_n=v1.thrust_per_seg_n,
        alpha_rad=v1.alpha_rad,
        dir_rad=v1.dir_rad,
        net_force_n=net,
        net_mz_nm=mz,
        note="V2: V1 + tangential swirl ring",
        ft_tan_per_seg_n=ft,
    )
