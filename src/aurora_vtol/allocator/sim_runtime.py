from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..topology import default_ring_topology

@dataclass(frozen=True)
class SimParams:
    mass_kg: float = 800.0
    dt_s: float = 0.02
    drag_coeff: float = 0.8
    gravity: float = 9.81
    yaw_inertia_kg_m2: float = 2500.0
    yaw_damping_nm_per_rad_s: float = 800.0
    z_drag_coeff: float = 1.2
    z_hold_kp_n_per_m: float = 1500.0
    z_hold_kd_n_per_mps: float = 2200.0

@dataclass
class SimState:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    yaw_deg: float = 0.0
    yaw_rate_deg_s: float = 0.0


def clip_force_xy(fx: float, fy: float, fxy_max_n: float) -> tuple[float, float]:
    mag = math.hypot(fx, fy)
    if mag <= max(1e-9, fxy_max_n):
        return fx, fy
    scale = fxy_max_n / mag
    return fx * scale, fy * scale


def step_vehicle(state: SimState, fx: float, fy: float, fz: float, mz: float, p: SimParams, z_ref_m: float = 0.0, vz_ref_mps: float = 0.0) -> None:
    fx_net = fx - p.drag_coeff * state.vx_mps
    fy_net = fy - p.drag_coeff * state.vy_mps
    fz_hold = p.z_hold_kp_n_per_m * (z_ref_m - state.z_m) + p.z_hold_kd_n_per_mps * (vz_ref_mps - state.vz_mps)
    fz_net = fz + fz_hold - p.mass_kg * p.gravity - p.z_drag_coeff * state.vz_mps
    ax = fx_net / p.mass_kg
    ay = fy_net / p.mass_kg
    az = fz_net / p.mass_kg
    yaw_rate_rad_s = math.radians(state.yaw_rate_deg_s)
    yaw_acc_rad_s2 = (mz - p.yaw_damping_nm_per_rad_s * yaw_rate_rad_s) / p.yaw_inertia_kg_m2
    state.vx_mps += ax * p.dt_s
    state.vy_mps += ay * p.dt_s
    state.vz_mps += az * p.dt_s
    state.x_m += state.vx_mps * p.dt_s
    state.y_m += state.vy_mps * p.dt_s
    state.z_m += state.vz_mps * p.dt_s
    if abs(state.z_m - z_ref_m) < 0.25 and abs(state.vz_mps - vz_ref_mps) < 0.5:
        state.z_m = z_ref_m
        state.vz_mps = vz_ref_mps
    yaw_rate_rad_s += yaw_acc_rad_s2 * p.dt_s
    state.yaw_rate_deg_s = math.degrees(yaw_rate_rad_s)
    state.yaw_deg = (state.yaw_deg + state.yaw_rate_deg_s * p.dt_s + 180.0) % 360.0 - 180.0


def pair_segments_to_fans(values, topology=None) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    active_topology = default_ring_topology(int(arr.size)) if topology is None else topology
    return active_topology.segment_values_to_fan_means(arr)


def fan_means_to_segments(fan_mean_n, segment_targets_n, topology=None) -> np.ndarray:
    targets = np.asarray(segment_targets_n, dtype=float)
    if targets.size == 0:
        return targets.copy()
    active_topology = default_ring_topology(int(targets.size)) if topology is None else topology
    return active_topology.distribute_fan_means_to_segments(fan_mean_n, targets)


def rate_limit_xy_force(prev_fx: float, prev_fy: float, target_fx: float, target_fy: float, max_rate_n_s: float, dt_s: float) -> tuple[float, float]:
    max_step = max(1e-6, max_rate_n_s * dt_s)
    dx = float(target_fx - prev_fx)
    dy = float(target_fy - prev_fy)
    mag = math.hypot(dx, dy)
    if mag <= max_step:
        return float(target_fx), float(target_fy)
    scale = max_step / mag
    return float(prev_fx + dx * scale), float(prev_fy + dy * scale)


def append_engineering_telemetry(hist: dict, telemetry: dict, fx_cmd: float, fy_cmd: float, fz_cmd: float, net) -> None:
    hist.setdefault("fz_cmd", []).append(float(fz_cmd))
    hist.setdefault("fx_ach", []).append(float(net[0]))
    hist.setdefault("fy_ach", []).append(float(net[1]))
    hist.setdefault("fz_ach", []).append(float(net[2]))
    hist.setdefault("fx_err", []).append(float(net[0] - fx_cmd))
    hist.setdefault("fy_err", []).append(float(net[1] - fy_cmd))
    hist.setdefault("fz_err", []).append(float(net[2] - fz_cmd))
    hist.setdefault("fan_cmd_16", []).append(list(telemetry["fan_cmd_16"]))
    hist.setdefault("fan_actual_16", []).append(list(telemetry["fan_actual_16"]))
    hist.setdefault("fan_temp_c_16", []).append(list(telemetry["fan_temp_c_16"]))
    hist.setdefault("fan_thermal_scale_16", []).append(list(telemetry["fan_thermal_scale_16"]))
    hist.setdefault("alpha_cmd_deg_32", []).append(list(telemetry["alpha_cmd_deg_32"]))
    hist.setdefault("alpha_cmd_rms", []).append(float(telemetry["alpha_cmd_rms"]))
    for key in (
        "battery_v",
        "battery_a",
        "battery_soc",
        "power_w",
        "energy_wh",
        "thrust_scale_pct",
        "continuous_power_pct",
        "continuous_power_raw_pct",
        "sustained_power_pct",
        "burst_reserve_pct",
        "burst_clip_pct",
        "burst_active_time_s",
        "power_margin_kw",
        "thermal_scale_pct",
        "fan_temp_max_c",
        "fan_temp_mean_c",
        "fan_response_pct",
    ):
        hist.setdefault(key, []).append(float(telemetry[key]))
