from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .faults import FaultSpec

@dataclass
class ActuatorLimits:
    alpha_rate_deg_s: float = 200.0
    ft_rate_n_s: float = 8000.0

@dataclass
class PlenumModel:
    tau_s: float = 0.12
    min_scale: float = 0.85
    max_scale: float = 1.15

@dataclass
class AllocatorState:
    alpha_rad: np.ndarray
    ft_tan_per_seg_n: np.ndarray
    plenum_scale: np.ndarray

    @staticmethod
    def init(n: int) -> "AllocatorState":
        return AllocatorState(
            alpha_rad=np.zeros(n, dtype=float),
            ft_tan_per_seg_n=np.zeros(n, dtype=float),
            plenum_scale=np.ones(n, dtype=float),
        )

def rate_limit(current: np.ndarray, target: np.ndarray, max_rate_per_s: float, dt_s: float) -> np.ndarray:
    max_step = max_rate_per_s * dt_s
    delta = target - current
    delta = np.clip(delta, -max_step, max_step)
    return current + delta

def rate_limit_per_axis(current: np.ndarray, target: np.ndarray, max_rate_per_s: np.ndarray, dt_s: float) -> np.ndarray:
    max_step = max_rate_per_s * dt_s
    delta = target - current
    delta = np.clip(delta, -max_step, max_step)
    return current + delta

def apply_actuator_limits(
    state: AllocatorState,
    alpha_target_rad: np.ndarray,
    ft_target_n: np.ndarray,
    lim: ActuatorLimits,
    dt_s: float,
    fault: FaultSpec | None = None,
) -> AllocatorState:
    alpha_rate_rad_s = np.deg2rad(lim.alpha_rate_deg_s)
    if fault is not None and fault.slow_flap_idx is not None:
        alpha_rate_rad_s_vec = np.full_like(alpha_target_rad, alpha_rate_rad_s, dtype=float)
        i = int(fault.slow_flap_idx)
        if 0 <= i < len(alpha_rate_rad_s_vec):
            alpha_rate_rad_s_vec[i] *= float(fault.slow_flap_rate_scale)
        alpha_next = rate_limit_per_axis(state.alpha_rad, alpha_target_rad, alpha_rate_rad_s_vec, dt_s)
    else:
        alpha_next = rate_limit(state.alpha_rad, alpha_target_rad, alpha_rate_rad_s, dt_s)
    ft_next = rate_limit(state.ft_tan_per_seg_n, ft_target_n, lim.ft_rate_n_s, dt_s)
    return AllocatorState(alpha_next, ft_next, state.plenum_scale.copy())

def apply_plenum_lag(state: AllocatorState, plenum_cmd_scale: np.ndarray, pl: PlenumModel, dt_s: float) -> AllocatorState:
    k = 0.0 if pl.tau_s <= 1e-6 else min(1.0, dt_s / pl.tau_s)
    p = state.plenum_scale + (plenum_cmd_scale - state.plenum_scale) * k
    p = np.clip(p, pl.min_scale, pl.max_scale)
    return AllocatorState(state.alpha_rad.copy(), state.ft_tan_per_seg_n.copy(), p)
