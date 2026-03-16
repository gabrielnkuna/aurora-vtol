from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..topology import default_ring_topology

_ALPHA_LIMIT_RAD = np.deg2rad(25.0)

@dataclass(frozen=True)
class FaultSpec:
    stuck_flap_idx: int | None = None
    stuck_flap_alpha_deg: float = 0.0
    slow_flap_idx: int | None = None
    slow_flap_rate_scale: float = 0.5
    weak_flap_idx: int | None = None
    weak_flap_scale: float = 0.7
    biased_flap_idx: int | None = None
    biased_flap_offset_deg: float = 0.0
    dead_fan_group: int | None = None
    dead_fan_scale: float = 0.0
    plenum_sector_idx: int | None = None
    plenum_sector_scale: float = 0.7


def _clip_alpha(alpha_rad: np.ndarray) -> np.ndarray:
    return np.clip(alpha_rad, -_ALPHA_LIMIT_RAD, _ALPHA_LIMIT_RAD)


def apply_command_faults_to_alpha(alpha_rad: np.ndarray, fault: FaultSpec) -> np.ndarray:
    a = alpha_rad.copy()
    if fault.weak_flap_idx is not None:
        i = int(fault.weak_flap_idx)
        if 0 <= i < len(a):
            scale = max(0.05, float(fault.weak_flap_scale))
            a[i] /= scale
    if fault.biased_flap_idx is not None:
        i = int(fault.biased_flap_idx)
        if 0 <= i < len(a):
            a[i] -= np.deg2rad(float(fault.biased_flap_offset_deg))
    if fault.stuck_flap_idx is not None:
        i = int(fault.stuck_flap_idx)
        if 0 <= i < len(a):
            a[i] = np.deg2rad(float(fault.stuck_flap_alpha_deg))
    return _clip_alpha(a)


def apply_faults_to_alpha(alpha_rad: np.ndarray, fault: FaultSpec) -> np.ndarray:
    a = alpha_rad.copy()
    if fault.weak_flap_idx is not None:
        i = int(fault.weak_flap_idx)
        if 0 <= i < len(a):
            a[i] *= float(fault.weak_flap_scale)
    if fault.biased_flap_idx is not None:
        i = int(fault.biased_flap_idx)
        if 0 <= i < len(a):
            a[i] += np.deg2rad(float(fault.biased_flap_offset_deg))
    if fault.stuck_flap_idx is not None:
        i = int(fault.stuck_flap_idx)
        if 0 <= i < len(a):
            a[i] = np.deg2rad(float(fault.stuck_flap_alpha_deg))
    return _clip_alpha(a)

def apply_faults_to_thrust(thrust_per_seg_n: np.ndarray, fault: FaultSpec) -> np.ndarray:
    t = thrust_per_seg_n.copy()
    topology = default_ring_topology(len(t))
    scale = topology.segment_effectiveness_scales(fault)
    if scale.shape == t.shape:
        t *= scale
    return t
