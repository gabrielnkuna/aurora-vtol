from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FaultSpec:
    stuck_flap_idx: int | None = None
    stuck_flap_alpha_deg: float = 0.0
    dead_fan_group: int | None = None
    dead_fan_scale: float = 0.0
    plenum_sector_idx: int | None = None
    plenum_sector_scale: float = 0.7

def apply_faults_to_alpha(alpha_rad: np.ndarray, fault: FaultSpec) -> np.ndarray:
    a = alpha_rad.copy()
    if fault.stuck_flap_idx is not None:
        i = int(fault.stuck_flap_idx)
        if 0 <= i < len(a):
            a[i] = np.deg2rad(float(fault.stuck_flap_alpha_deg))
    return a

def apply_faults_to_thrust(thrust_per_seg_n: np.ndarray, fault: FaultSpec) -> np.ndarray:
    t = thrust_per_seg_n.copy()
    if fault.dead_fan_group is not None:
        g = int(fault.dead_fan_group)
        if 0 <= g < 16:
            s0 = 2 * g
            s1 = 2 * g + 1
            scale = float(fault.dead_fan_scale)
            if 0 <= s0 < len(t):
                t[s0] *= scale
            if 0 <= s1 < len(t):
                t[s1] *= scale
    if fault.plenum_sector_idx is not None:
        i = int(fault.plenum_sector_idx)
        if 0 <= i < len(t):
            t[i] *= float(fault.plenum_sector_scale)
    return t
