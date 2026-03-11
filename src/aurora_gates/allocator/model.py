from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class RingGeometry:
    n_segments: int = 32
    radius_m: float = 2.5
    # increased max flap deflection to allow higher braking authority
    alpha_max_deg: float = 25.0

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def segment_angles_rad(n: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)

def tangential_unit_vectors(n: int) -> np.ndarray:
    theta = segment_angles_rad(n)
    tx = -np.sin(theta)
    ty = np.cos(theta)
    return np.stack([tx, ty], axis=1)

def thrust_vectors_body(geom: RingGeometry, thrust_per_seg_n: np.ndarray, alpha_rad: np.ndarray, dir_rad: np.ndarray) -> np.ndarray:
    n = geom.n_segments
    fz = thrust_per_seg_n * np.cos(alpha_rad)
    fxy = thrust_per_seg_n * np.sin(alpha_rad)
    fx = fxy * np.cos(dir_rad)
    fy = fxy * np.sin(dir_rad)
    return np.stack([fx, fy, fz], axis=1)

def net_force_and_yaw_moment(geom: RingGeometry, forces_body: np.ndarray) -> tuple[np.ndarray, float]:
    n = geom.n_segments
    theta = segment_angles_rad(n)
    R = geom.radius_m
    rx = R * np.cos(theta)
    ry = R * np.sin(theta)
    fx = forces_body[:, 0]
    fy = forces_body[:, 1]
    mz = float(np.sum(rx * fy - ry * fx))
    net = np.sum(forces_body, axis=0)
    return net, mz
