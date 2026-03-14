from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .model import RingGeometry, segment_angles_rad, thrust_vectors_body, net_force_and_yaw_moment
from .faults import FaultSpec

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

def _fault_has_effect(fault: FaultSpec | None) -> bool:
    if fault is None:
        return False
    return any((
        fault.stuck_flap_idx is not None,
        fault.dead_fan_group is not None,
        fault.plenum_sector_idx is not None,
    ))


def _segment_fault_scales(n: int, fault: FaultSpec | None) -> np.ndarray:
    scale = np.ones(n, dtype=float)
    if fault is None:
        return scale
    if fault.dead_fan_group is not None:
        g = int(fault.dead_fan_group)
        for idx in (2 * g, 2 * g + 1):
            if 0 <= idx < n:
                scale[idx] *= float(fault.dead_fan_scale)
    if fault.plenum_sector_idx is not None:
        idx = int(fault.plenum_sector_idx)
        if 0 <= idx < n:
            scale[idx] *= float(fault.plenum_sector_scale)
    return np.clip(scale, 0.0, None)


def _solve_radial_components_limited(theta: np.ndarray, target_xy: np.ndarray, caps: np.ndarray) -> np.ndarray:
    radial = np.zeros_like(caps, dtype=float)
    residual = np.asarray(target_xy, dtype=float).copy()
    active = [idx for idx, cap in enumerate(caps) if cap > 1e-9]

    while active and np.linalg.norm(residual) > 1e-6:
        A = np.stack([np.cos(theta[active]), np.sin(theta[active])], axis=0)
        pseudo = A.T @ np.linalg.pinv(A @ A.T)
        candidate = pseudo @ residual

        saturated = []
        for local_idx, seg_idx in enumerate(active):
            cap = float(caps[seg_idx])
            value = float(candidate[local_idx])
            if abs(value) > cap + 1e-9:
                radial[seg_idx] = math.copysign(cap, value)
                residual -= radial[seg_idx] * np.array([math.cos(theta[seg_idx]), math.sin(theta[seg_idx])], dtype=float)
                saturated.append(seg_idx)

        if not saturated:
            for local_idx, seg_idx in enumerate(active):
                radial[seg_idx] = float(candidate[local_idx])
            residual[:] = 0.0
            break

        active = [idx for idx in active if idx not in saturated]

    return radial


def _allocate_with_faults(geom: RingGeometry, req: AllocationRequest, fault: FaultSpec) -> AllocationResultV2:
    n = geom.n_segments
    theta = segment_angles_rad(n)
    thrust_scale = _segment_fault_scales(n, fault)
    active = thrust_scale > 1e-6
    if not np.any(active):
        zero = np.zeros(n, dtype=float)
        forces = thrust_vectors_body(geom, zero, zero, theta)
        net, mz = net_force_and_yaw_moment(geom, forces)
        return AllocationResultV2(
            thrust_per_seg_n=zero,
            alpha_rad=zero,
            dir_rad=theta.copy(),
            net_force_n=net,
            net_mz_nm=mz,
            note='V2 fault-aware: no active segments',
            ft_tan_per_seg_n=zero,
        )

    alpha_max = math.radians(geom.alpha_max_deg)
    tan_alpha_max = math.tan(alpha_max)
    z_weights = active.astype(float)
    stuck_idx = None
    stuck_alpha = 0.0
    if fault.stuck_flap_idx is not None and 0 <= int(fault.stuck_flap_idx) < n and active[int(fault.stuck_flap_idx)]:
        stuck_idx = int(fault.stuck_flap_idx)
        stuck_alpha = math.radians(float(fault.stuck_flap_alpha_deg))
        severity = min(1.0, abs(math.tan(stuck_alpha)) / max(tan_alpha_max, 1e-6))
        z_weights[stuck_idx] *= max(0.15, 1.0 - 0.85 * severity)

    if float(np.sum(z_weights)) <= 1e-9:
        z_weights = active.astype(float)
    z_actual = np.zeros(n, dtype=float)
    z_actual[active] = float(req.fz_n) * z_weights[active] / float(np.sum(z_weights[active]))

    radial_actual = np.zeros(n, dtype=float)
    fixed_xy = np.zeros(2, dtype=float)
    if stuck_idx is not None:
        radial_actual[stuck_idx] = z_actual[stuck_idx] * math.tan(stuck_alpha)
        fixed_xy += radial_actual[stuck_idx] * np.array([math.cos(theta[stuck_idx]), math.sin(theta[stuck_idx])], dtype=float)

    target_xy = np.array([float(req.fx_n), float(req.fy_n)], dtype=float) - fixed_xy
    free_mask = active.copy()
    if stuck_idx is not None:
        free_mask[stuck_idx] = False
    caps = z_actual * tan_alpha_max
    radial_free = _solve_radial_components_limited(theta, target_xy, np.where(free_mask, caps, 0.0))
    radial_actual += radial_free

    alpha_actual = np.zeros(n, dtype=float)
    thrust_actual = np.zeros(n, dtype=float)
    active_idx = np.where(active)[0]
    alpha_actual[active_idx] = np.arctan2(radial_actual[active_idx], z_actual[active_idx])
    thrust_actual[active_idx] = np.hypot(z_actual[active_idx], radial_actual[active_idx])

    thrust_cmd = np.zeros(n, dtype=float)
    thrust_cmd[active_idx] = thrust_actual[active_idx] / thrust_scale[active_idx]
    alpha_cmd = alpha_actual.copy()
    if stuck_idx is not None:
        alpha_cmd[stuck_idx] = stuck_alpha

    forces = thrust_vectors_body(geom, thrust_actual, alpha_actual, theta)
    net, mz = net_force_and_yaw_moment(geom, forces)

    if abs(req.mz_nm) < 1e-6:
        ft_cmd = np.zeros(n, dtype=float)
    else:
        ft_cmd = np.full(n, float(req.mz_nm) / (geom.radius_m * n), dtype=float)
        ft_cmd = np.clip(ft_cmd, -0.20 * np.maximum(thrust_cmd, 1e-6), 0.20 * np.maximum(thrust_cmd, 1e-6))

    fx_t = ft_cmd * (-np.sin(theta))
    fy_t = ft_cmd * (np.cos(theta))
    net = net + np.array([np.sum(fx_t), np.sum(fy_t), 0.0], dtype=float)
    mz += float(np.sum((geom.radius_m * np.cos(theta)) * fy_t - (geom.radius_m * np.sin(theta)) * fx_t))

    return AllocationResultV2(
        thrust_per_seg_n=thrust_cmd,
        alpha_rad=alpha_cmd,
        dir_rad=theta.copy(),
        net_force_n=net,
        net_mz_nm=mz,
        note='V2 fault-aware constrained allocation',
        ft_tan_per_seg_n=ft_cmd,
    )

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

def allocate_v2(geom: RingGeometry, req: AllocationRequest, fault: FaultSpec | None = None) -> AllocationResultV2:
    if _fault_has_effect(fault):
        return _allocate_with_faults(geom, req, fault or FaultSpec())

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
