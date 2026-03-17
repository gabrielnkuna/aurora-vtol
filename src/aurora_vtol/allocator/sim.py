from __future__ import annotations
from dataclasses import dataclass, field as dc_field
import math
import numpy as np

from .model import RingGeometry, segment_angles_rad, thrust_vectors_body, net_force_and_yaw_moment
from .allocate import AllocationRequest, allocate_v1, allocate_v2
from .metrics import yaw_track_coupling_mean_abs
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel, apply_actuator_limits, apply_plenum_lag
from .response import compute_step_metrics
from .field import RepelField, repel_force_xy
from .faults import FaultSpec, apply_command_faults_to_alpha, apply_faults_to_alpha, apply_faults_to_thrust
from .trace import save_trace_json
from ..icd import ActuatorHealthState, EstimatedVehicleState, GuidanceTarget, RedirectTarget
from ..topology import default_ring_topology
from ..effectiveness import effectiveness_table_for_topology, hardware_assumptions_payload
from ..vehicle_controller import XYVehicleControllerGains, command_directional_force, track_redirect_velocity, track_step_snap_brake, track_step_snap_reverse, track_xy_position

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


@dataclass(frozen=True)
class MissionObstacle:
    x_m: float
    y_m: float
    radius_m: float = 12.0
    k_n_per_m: float = 180.0
    fxy_max_n: float = 2500.0
    swirl_n: float = 900.0
    influence_m: float = 20.0


@dataclass(frozen=True)
class PowerSystemParams:
    battery_full_v: float = 58.8
    battery_empty_v: float = 46.0
    capacity_kwh: float = 15.0
    internal_resistance_ohm: float = 0.0012
    hover_power_w: float = 110000.0
    continuous_power_w: float = 125000.0
    peak_power_w: float = 165000.0
    burst_duration_s: float = 4.0
    burst_recharge_tau_s: float = 18.0
    aux_power_w: float = 2500.0
    thrust_power_exponent: float = 1.35
    flap_power_scale: float = 0.10
    tangential_power_scale: float = 0.04
    min_supply_scale: float = 0.80
    fan_spool_tau_s: float = 0.18
    fan_heat_fraction: float = 0.11
    overload_heat_gain: float = 0.45
    fan_thermal_capacity_j_per_c: float = 5500.0
    fan_cooling_w_per_c: float = 28.0
    ambient_temp_c: float = 28.0
    thermal_limit_start_c: float = 85.0
    thermal_limit_full_c: float = 110.0
    min_thermal_scale: float = 0.65


@dataclass
class PowerSystemState:
    soc_frac: float = 1.0
    voltage_v: float = 58.8
    current_a: float = 0.0
    power_w: float = 2500.0
    energy_used_wh: float = 0.0
    thrust_scale: float = 1.0
    burst_reserve_j: float = 0.0
    fan_mean_n: np.ndarray = dc_field(default_factory=lambda: np.zeros(16, dtype=float))
    fan_temp_c: np.ndarray = dc_field(default_factory=lambda: np.full(16, 28.0, dtype=float))
    thermal_scale: np.ndarray = dc_field(default_factory=lambda: np.ones(16, dtype=float))

    @classmethod
    def init(cls, power: "PowerSystemParams", fan_count: int = 16) -> "PowerSystemState":
        return cls(
            soc_frac=1.0,
            voltage_v=power.battery_full_v,
            current_a=0.0,
            power_w=power.aux_power_w,
            energy_used_wh=0.0,
            thrust_scale=1.0,
            burst_reserve_j=burst_capacity_j(power),
            fan_mean_n=np.zeros(fan_count, dtype=float),
            fan_temp_c=np.full(fan_count, power.ambient_temp_c, dtype=float),
            thermal_scale=np.ones(fan_count, dtype=float),
        )


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def burst_capacity_j(power: PowerSystemParams) -> float:
    return max(0.0, (power.peak_power_w - power.continuous_power_w) * max(power.burst_duration_s, 0.0))


def sustained_power_ratio(power: PowerSystemParams, power_w: float, burst_reserve_j: float) -> float:
    continuous_limit = max(power.continuous_power_w, 1e-6)
    raw_ratio = float(power_w) / continuous_limit
    if raw_ratio <= 1.0:
        return raw_ratio
    capacity = burst_capacity_j(power)
    if capacity <= 1e-9:
        return raw_ratio
    reserve_frac = max(0.0, min(1.0, float(burst_reserve_j) / capacity))
    overload = raw_ratio - 1.0
    return 1.0 + overload * (1.0 - reserve_frac)


def clip_main_power_to_budget(
    fan_actual_mean: np.ndarray,
    thrust_pre_power: np.ndarray,
    load_multiplier: float,
    power_budget_w: float,
    power: PowerSystemParams,
    topology,
    hover_per_seg_n: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    budget_w = max(power.aux_power_w, float(power_budget_w))
    peak_budget_w = max(1e-6, budget_w - power.aux_power_w)
    main_load_w = max(1e-6, budget_w - power.aux_power_w)
    scale = 1.0
    power_w = power.aux_power_w
    thrust_actual = topology.distribute_fan_means_to_segments(fan_actual_mean, thrust_pre_power)
    thrust_ratio = np.clip(thrust_actual / hover_per_seg_n, 0.0, None)
    mean_thrust_ratio = float(np.mean(thrust_ratio)) if thrust_ratio.size else 0.0
    main_power_w = power.hover_power_w * (mean_thrust_ratio ** power.thrust_power_exponent)
    power_w = power.aux_power_w + main_power_w * load_multiplier
    if power_w > budget_w:
        main_load_w = max(1e-6, power_w - power.aux_power_w)
        scale = (peak_budget_w / main_load_w) ** (1.0 / max(power.thrust_power_exponent, 1e-6))
        scale = max(power.min_supply_scale, min(1.0, scale))
        fan_actual_mean = fan_actual_mean * scale
        thrust_actual = topology.distribute_fan_means_to_segments(fan_actual_mean, thrust_pre_power)
        thrust_ratio = np.clip(thrust_actual / hover_per_seg_n, 0.0, None)
        mean_thrust_ratio = float(np.mean(thrust_ratio)) if thrust_ratio.size else 0.0
        main_power_w = power.hover_power_w * (mean_thrust_ratio ** power.thrust_power_exponent)
        power_w = power.aux_power_w + main_power_w * load_multiplier
    return fan_actual_mean, thrust_actual, float(power_w), float(scale)


def clip_force_xy(fx: float, fy: float, fxy_max_n: float) -> tuple[float, float]:
    mag = math.hypot(fx, fy)
    if mag <= max(1e-9, fxy_max_n):
        return fx, fy
    scale = fxy_max_n / mag
    return fx * scale, fy * scale


def mission_safety_force(x_m: float, y_m: float, obstacles: list[MissionObstacle]) -> tuple[float, float, float | None, int, float]:
    if not obstacles:
        return 0.0, 0.0, None, 0, 0.0

    fx_total = 0.0
    fy_total = 0.0
    nearest_clearance_m = None
    active_count = 0
    max_threat = 0.0

    for obs in obstacles:
        dist_center = float(math.hypot(x_m - obs.x_m, y_m - obs.y_m))
        clearance_m = dist_center - obs.radius_m
        if nearest_clearance_m is None or clearance_m < nearest_clearance_m:
            nearest_clearance_m = clearance_m

        safety_radius_m = obs.radius_m + max(4.0, 0.35 * obs.influence_m)
        field = RepelField(k_n_per_m=obs.k_n_per_m, radius_m=safety_radius_m, fxy_max_n=obs.fxy_max_n)
        fx_rep, fy_rep = repel_force_xy(field, x_m, y_m, obs.x_m, obs.y_m)
        fx_total += float(fx_rep)
        fy_total += float(fy_rep)

        if dist_center <= safety_radius_m:
            active_count += 1
        threat = 1.0 - max(0.0, dist_center - obs.radius_m) / max(obs.influence_m, 1e-6)
        max_threat = max(max_threat, smoothstep01(threat))

    return fx_total, fy_total, nearest_clearance_m, active_count, max_threat


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-9:
        return float(math.hypot(px - ax, py - ay)), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return float(math.hypot(px - qx, py - qy)), t


def route_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += float(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]))
    return total


def route_penalty(points: list[tuple[float, float]], obstacles: list[MissionObstacle], clearance_m: float) -> float:
    penalty = 0.0
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        for obs in obstacles:
            dist_m, frac = point_segment_distance(obs.x_m, obs.y_m, ax, ay, bx, by)
            required_m = obs.radius_m + clearance_m
            if 0.02 < frac < 0.98 and dist_m < required_m:
                penalty += (required_m - dist_m) ** 2 * 200.0
    return penalty


def choose_bypass_waypoints(ax: float, ay: float, bx: float, by: float, obs: MissionObstacle, clearance_m: float, obstacles: list[MissionObstacle]) -> list[tuple[float, float]]:
    seg_dx = bx - ax
    seg_dy = by - ay
    seg_len = max(1e-6, float(math.hypot(seg_dx, seg_dy)))
    ux = seg_dx / seg_len
    uy = seg_dy / seg_len
    px = -uy
    py = ux
    offset_m = obs.radius_m + clearance_m
    lead_m = max(6.0, clearance_m * 1.5, min(obs.influence_m, offset_m * 1.5))

    best_score = None
    best_points = None
    for side in (-1.0, 1.0):
        wp1 = (obs.x_m - ux * lead_m + side * px * offset_m, obs.y_m - uy * lead_m + side * py * offset_m)
        wp2 = (obs.x_m + ux * lead_m + side * px * offset_m, obs.y_m + uy * lead_m + side * py * offset_m)
        points = [(ax, ay), wp1, wp2, (bx, by)]
        score = route_length(points) + route_penalty(points, obstacles, clearance_m)
        if best_score is None or score < best_score:
            best_score = score
            best_points = [wp1, wp2]

    return best_points if best_points is not None else []


def plan_route_waypoints(start_x_m: float, start_y_m: float, dest_x_m: float, dest_y_m: float, obstacles: list[MissionObstacle], clearance_m: float) -> list[tuple[float, float]]:
    route = [(start_x_m, start_y_m), (dest_x_m, dest_y_m)]
    max_insertions = max(4, len(obstacles) * 4)

    for _ in range(max_insertions):
        best_block = None
        for seg_idx in range(len(route) - 1):
            ax, ay = route[seg_idx]
            bx, by = route[seg_idx + 1]
            for obs in obstacles:
                dist_m, frac = point_segment_distance(obs.x_m, obs.y_m, ax, ay, bx, by)
                required_m = obs.radius_m + clearance_m
                if 0.05 < frac < 0.95 and dist_m < required_m:
                    severity = required_m - dist_m
                    if best_block is None or severity > best_block['severity']:
                        best_block = {
                            'seg_idx': seg_idx,
                            'ax': ax,
                            'ay': ay,
                            'bx': bx,
                            'by': by,
                            'obs': obs,
                            'severity': severity,
                        }

        if best_block is None:
            break

        bypass = choose_bypass_waypoints(
            best_block['ax'],
            best_block['ay'],
            best_block['bx'],
            best_block['by'],
            best_block['obs'],
            clearance_m,
            obstacles,
        )
        if not bypass:
            break
        route = route[:best_block['seg_idx'] + 1] + bypass + route[best_block['seg_idx'] + 1:]
        if len(route) >= 18:
            break

    simplified = route[:]
    changed = True
    while changed and len(simplified) > 2:
        changed = False
        for i in range(1, len(simplified) - 1):
            direct = [simplified[i - 1], simplified[i + 1]]
            current = [simplified[i - 1], simplified[i], simplified[i + 1]]
            direct_score = route_length(direct) + route_penalty(direct, obstacles, clearance_m)
            current_score = route_length(current) + route_penalty(current, obstacles, clearance_m)
            if direct_score <= current_score + 0.5:
                del simplified[i]
                changed = True
                break

    cleaned = [simplified[0]]
    for pt in simplified[1:]:
        if math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 0.75:
            cleaned.append(pt)
    return cleaned


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


def temperature_to_thermal_scale(temp_c, power: PowerSystemParams) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    span = max(1e-6, power.thermal_limit_full_c - power.thermal_limit_start_c)
    hot_x = np.clip((temp - power.thermal_limit_start_c) / span, 0.0, 1.0)
    hot_frac = hot_x * hot_x * (3.0 - 2.0 * hot_x)
    scale = power.min_thermal_scale + (1.0 - power.min_thermal_scale) * (1.0 - hot_frac)
    return np.clip(scale, power.min_thermal_scale, 1.0)


def init_hover_power_state(power: PowerSystemParams, geom: RingGeometry, sim: SimParams) -> PowerSystemState:
    fan_count = default_ring_topology(geom.n_segments).fan_count
    hover_seg_n = sim.mass_kg * sim.gravity / max(1, geom.n_segments)
    hover_power_w = power.hover_power_w + power.aux_power_w
    return PowerSystemState(
        soc_frac=1.0,
        voltage_v=power.battery_full_v,
        current_a=hover_power_w / max(power.battery_full_v, 1.0),
        power_w=hover_power_w,
        energy_used_wh=0.0,
        thrust_scale=1.0,
        burst_reserve_j=burst_capacity_j(power),
        fan_mean_n=np.full(fan_count, hover_seg_n, dtype=float),
        fan_temp_c=np.full(fan_count, power.ambient_temp_c, dtype=float),
        thermal_scale=np.ones(fan_count, dtype=float),
    )


def rate_limit_xy_force(prev_fx: float, prev_fy: float, target_fx: float, target_fy: float, max_rate_n_s: float, dt_s: float) -> tuple[float, float]:
    max_step = max(1e-6, max_rate_n_s * dt_s)
    dx = float(target_fx - prev_fx)
    dy = float(target_fy - prev_fy)
    mag = math.hypot(dx, dy)
    if mag <= max_step:
        return float(target_fx), float(target_fy)
    scale = max_step / mag
    return float(prev_fx + dx * scale), float(prev_fy + dy * scale)


def fault_motion_guard(geom: RingGeometry, fault: FaultSpec | None) -> dict[str, float]:
    if fault is None:
        return {
            'fault_guard_scale': 1.0,
            'fault_response_scale': 1.0,
            'fault_available_scale': 1.0,
            'fault_asymmetry': 0.0,
            'dead_align_scale': 1.0,
            'dead_cross_scale': 1.0,
            'dead_align_speed_floor_mps': 0.55,
            'plenum_power_trim': 1.0,
            'plenum_revector_trim': 1.0,
            'plenum_align_speed_floor_mps': 0.0,
            'plenum_brake_trim': 1.0,
        }

    topology = default_ring_topology(geom.n_segments)
    scale = topology.segment_effectiveness_scales(fault)

    theta = segment_angles_rad(geom.n_segments)
    available_scale = float(np.clip(np.mean(scale), 0.0, 1.0))
    weight_sum = float(np.sum(scale))
    if weight_sum > 1e-9:
        asymmetry = float(math.hypot(float(np.sum(scale * np.cos(theta))), float(np.sum(scale * np.sin(theta)))) / weight_sum)
    else:
        asymmetry = 1.0

    dead_guard = 1.0
    dead_align_scale = 1.0
    dead_cross_scale = 1.0
    dead_align_speed_floor_mps = 0.55
    if fault.dead_fan_group is not None:
        idxs = list(topology.fan_segments(int(fault.dead_fan_group)))
        local_scale = float(np.mean(scale[idxs])) if idxs else 1.0
        lost_scale = float(np.clip(1.0 - local_scale, 0.0, 1.0))
        dead_guard = 0.60 + 0.40 * local_scale
        dead_align_scale = 1.0 + 0.45 * lost_scale
        dead_cross_scale = 1.0 - 0.24 * lost_scale
        dead_align_speed_floor_mps = 0.55 + 0.22 * lost_scale

    plenum_guard = 1.0
    plenum_power_trim = 1.0
    plenum_revector_trim = 1.0
    plenum_align_speed_floor_mps = 0.0
    plenum_brake_trim = 1.0
    if fault.plenum_sector_idx is not None:
        idxs = list(topology.plenum_segments(int(fault.plenum_sector_idx)))
        local_scale = float(np.mean(scale[idxs])) if idxs else 1.0
        lost_scale = float(np.clip(1.0 - local_scale, 0.0, 1.0))
        plenum_guard = 0.82 + 0.18 * local_scale
        plenum_power_trim = 1.0 - 0.40 * lost_scale
        plenum_revector_trim = 1.0 - 0.22 * lost_scale
        plenum_align_speed_floor_mps = 0.48 + 0.18 * lost_scale
        plenum_brake_trim = 1.0 - 0.18 * lost_scale

    stuck_guard = 1.0
    if fault.stuck_flap_idx is not None:
        alpha_limit_rad = max(1e-6, math.radians(geom.alpha_max_deg))
        stuck_severity = min(1.0, abs(math.radians(float(fault.stuck_flap_alpha_deg))) / alpha_limit_rad)
        stuck_guard = 1.0 - 0.18 * stuck_severity

    authority_guard = min(available_scale, dead_guard, plenum_guard, stuck_guard, 1.0 - 2.8 * asymmetry)
    response_guard = min(dead_guard, 0.78 + 0.22 * available_scale, 1.0 - 4.5 * asymmetry)
    return {
        'fault_guard_scale': float(np.clip(authority_guard, 0.55, 1.0)),
        'fault_response_scale': float(np.clip(response_guard, 0.40, 1.0)),
        'fault_available_scale': float(available_scale),
        'fault_asymmetry': float(np.clip(asymmetry, 0.0, 1.0)),
        'dead_align_scale': float(np.clip(dead_align_scale, 1.0, 1.5)),
        'dead_cross_scale': float(np.clip(dead_cross_scale, 0.70, 1.0)),
        'dead_align_speed_floor_mps': float(np.clip(dead_align_speed_floor_mps, 0.55, 0.85)),
        'plenum_power_trim': float(np.clip(plenum_power_trim, 0.84, 1.0)),
        'plenum_revector_trim': float(np.clip(plenum_revector_trim, 0.88, 1.0)),
        'plenum_align_speed_floor_mps': float(np.clip(plenum_align_speed_floor_mps, 0.0, 0.65)),
        'plenum_brake_trim': float(np.clip(plenum_brake_trim, 0.88, 1.0)),
    }


def guidance_force_budget(power_state: PowerSystemState, state: AllocatorState, geom: RingGeometry, power: PowerSystemParams, fxy_max_n: float, fault: FaultSpec | None = None) -> tuple[float, dict[str, float]]:
    alpha_limit_rad = max(1e-6, math.radians(geom.alpha_max_deg))
    flap_use = float(np.max(np.abs(state.alpha_rad)) / alpha_limit_rad) if state.alpha_rad.size else 0.0
    flap_guard = 1.0 - 0.60 * smoothstep01((flap_use - 0.60) / 0.22)
    flap_guard = float(np.clip(flap_guard, 0.42, 1.0))

    if power.continuous_power_w > 1e-6:
        continuous_ratio_raw = float(power_state.power_w) / power.continuous_power_w
    else:
        continuous_ratio_raw = 0.0
    burst_capacity = burst_capacity_j(power)
    burst_reserve_j = float(getattr(power_state, 'burst_reserve_j', burst_capacity))
    burst_reserve_j = float(np.clip(burst_reserve_j, 0.0, burst_capacity)) if burst_capacity > 0.0 else 0.0
    burst_reserve_frac = burst_reserve_j / burst_capacity if burst_capacity > 1e-9 else 1.0
    continuous_ratio = sustained_power_ratio(power, float(power_state.power_w), burst_reserve_j)
    power_guard = 1.0 - 0.62 * smoothstep01((continuous_ratio - 0.80) / 0.18)
    power_guard = float(np.clip(power_guard, 0.40, 1.0))

    thermal_guard = float(np.clip(np.mean(power_state.thermal_scale), max(power.min_thermal_scale, 0.55), 1.0))
    supply_guard = float(np.clip(power_state.thrust_scale, 0.60, 1.0))
    fault_guard = fault_motion_guard(geom, fault)
    combined_guard = min(flap_guard, power_guard, thermal_guard, supply_guard, fault_guard['fault_guard_scale'])
    budget_ratio = float(np.clip(combined_guard, 0.25, 1.0))
    budget_n = float(max(0.25 * fxy_max_n, budget_ratio * fxy_max_n))
    return budget_n, {
        'guard_scale': float(combined_guard),
        'budget_ratio': float(budget_ratio),
        'flap_guard_scale': float(flap_guard),
        'power_guard_scale': float(power_guard),
        'thermal_guard_scale': float(thermal_guard),
        'supply_guard_scale': float(supply_guard),
        'fault_guard_scale': float(fault_guard['fault_guard_scale']),
        'fault_response_scale': float(fault_guard['fault_response_scale']),
        'fault_available_scale': float(fault_guard['fault_available_scale']),
        'fault_asymmetry_pct': float(100.0 * fault_guard['fault_asymmetry']),
        'dead_align_scale': float(fault_guard['dead_align_scale']),
        'dead_cross_scale': float(fault_guard['dead_cross_scale']),
        'dead_align_speed_floor_mps': float(fault_guard['dead_align_speed_floor_mps']),
        'plenum_power_trim': float(fault_guard['plenum_power_trim']),
        'plenum_revector_trim': float(fault_guard['plenum_revector_trim']),
        'plenum_align_speed_floor_mps': float(fault_guard['plenum_align_speed_floor_mps']),
        'plenum_brake_trim': float(fault_guard['plenum_brake_trim']),
        'continuous_power_ratio': float(continuous_ratio),
        'continuous_power_raw_ratio': float(continuous_ratio_raw),
        'burst_reserve_ratio': float(burst_reserve_frac),
        'flap_usage_ratio': float(flap_use),
    }


def apply_power_system(
    thrust_cmd_n,
    thrust_pre_power_n,
    alpha_cmd_rad,
    alpha_actual_rad,
    ft_actual_n,
    sim: SimParams,
    geom: RingGeometry,
    power: PowerSystemParams,
    power_state: PowerSystemState,
    dt_s: float,
):
    thrust_cmd = np.asarray(thrust_cmd_n, dtype=float)
    thrust_pre_power = np.asarray(thrust_pre_power_n, dtype=float)
    alpha_cmd = np.asarray(alpha_cmd_rad, dtype=float)
    alpha_actual = np.asarray(alpha_actual_rad, dtype=float)
    ft_actual = np.asarray(ft_actual_n, dtype=float)

    hover_per_seg_n = max(1e-6, sim.mass_kg * sim.gravity / max(1, geom.n_segments))
    alpha_limit_rad = max(1e-6, math.radians(geom.alpha_max_deg))
    tangential_scale_n = max(1e-6, 0.35 * hover_per_seg_n)
    fan_count = default_ring_topology(geom.n_segments).fan_count

    flap_activity = float(np.mean(np.abs(alpha_actual)) / alpha_limit_rad) if alpha_actual.size else 0.0
    flap_tracking = float(np.mean(np.abs(alpha_cmd - alpha_actual)) / alpha_limit_rad) if alpha_actual.size else 0.0
    tangential_activity = float(np.mean(np.abs(ft_actual)) / tangential_scale_n) if ft_actual.size else 0.0
    load_multiplier = (
        1.0
        + power.flap_power_scale * flap_activity
        + 0.05 * flap_tracking
        + power.tangential_power_scale * tangential_activity
    )

    topology = default_ring_topology(geom.n_segments)
    fan_cmd_mean = np.asarray(topology.segment_values_to_fan_means(thrust_cmd), dtype=float)
    fan_pre_power_mean = np.asarray(topology.segment_values_to_fan_means(thrust_pre_power), dtype=float)
    if fan_cmd_mean.size != fan_count:
        fan_cmd_mean = np.resize(fan_cmd_mean, fan_count)
    if fan_pre_power_mean.size != fan_count:
        fan_pre_power_mean = np.resize(fan_pre_power_mean, fan_count)

    prev_fan_mean = np.asarray(power_state.fan_mean_n, dtype=float)
    if prev_fan_mean.size != fan_count:
        prev_fan_mean = np.zeros(fan_count, dtype=float)
    prev_fan_temp = np.asarray(power_state.fan_temp_c, dtype=float)
    if prev_fan_temp.size != fan_count:
        prev_fan_temp = np.full(fan_count, power.ambient_temp_c, dtype=float)
    prev_thermal_scale = np.asarray(power_state.thermal_scale, dtype=float)
    if prev_thermal_scale.size != fan_count:
        prev_thermal_scale = np.ones(fan_count, dtype=float)

    soc_frac = max(0.0, min(1.0, float(power_state.soc_frac)))
    span_v = max(1e-6, power.battery_full_v - power.battery_empty_v)
    open_circuit_v = power.battery_empty_v + span_v * soc_frac
    supply_scale = power.min_supply_scale + (1.0 - power.min_supply_scale) * smoothstep01((float(power_state.voltage_v) - power.battery_empty_v) / span_v)
    supply_scale = max(power.min_supply_scale, min(1.0, float(supply_scale)))

    fan_target_mean = np.clip(fan_pre_power_mean, 0.0, None) * supply_scale * prev_thermal_scale
    spool_k = 1.0 if power.fan_spool_tau_s <= 1e-6 else min(1.0, dt_s / power.fan_spool_tau_s)
    fan_actual_mean = prev_fan_mean + (fan_target_mean - prev_fan_mean) * spool_k

    burst_capacity = burst_capacity_j(power)
    burst_reserve_j = float(getattr(power_state, 'burst_reserve_j', burst_capacity))
    burst_reserve_j = float(np.clip(burst_reserve_j, 0.0, burst_capacity)) if burst_capacity > 0.0 else 0.0

    fan_actual_mean, thrust_actual, power_w, peak_clip_scale = clip_main_power_to_budget(
        fan_actual_mean,
        thrust_pre_power,
        load_multiplier,
        power.peak_power_w,
        power,
        topology,
        hover_per_seg_n,
    )
    continuous_limit_w = max(power.continuous_power_w, 1e-6)
    continuous_power_raw_pct = 100.0 * power_w / continuous_limit_w
    burst_clip_pct = 0.0
    burst_active_time_s = 0.0
    if power_w > power.continuous_power_w and dt_s > 0.0:
        burst_active_time_s = float(dt_s)
        burst_needed_j = (power_w - power.continuous_power_w) * dt_s
        if burst_capacity <= 1e-9:
            allowed_power_w = power.continuous_power_w
        else:
            available_burst_j = max(0.0, burst_reserve_j)
            if burst_needed_j <= available_burst_j + 1e-9:
                burst_reserve_j = max(0.0, available_burst_j - burst_needed_j)
                allowed_power_w = power_w
            else:
                allowed_power_w = power.continuous_power_w + available_burst_j / dt_s
                burst_reserve_j = 0.0
        if allowed_power_w + 1e-9 < power_w:
            burst_clip_pct = 100.0 * max(0.0, power_w - allowed_power_w) / max(power_w, 1e-6)
            fan_actual_mean, thrust_actual, power_w, _burst_scale = clip_main_power_to_budget(
                fan_actual_mean,
                thrust_pre_power,
                load_multiplier,
                min(power.peak_power_w, allowed_power_w),
                power,
                topology,
                hover_per_seg_n,
            )
    elif burst_capacity > 1e-9 and power.burst_recharge_tau_s > 1e-6 and power_w < power.continuous_power_w:
        headroom_j = (power.continuous_power_w - power_w) * dt_s
        recharge_cap_j = burst_capacity - burst_reserve_j
        recharge_tau_j = burst_capacity * dt_s / power.burst_recharge_tau_s
        burst_reserve_j += min(recharge_cap_j, headroom_j, recharge_tau_j)

    burst_reserve_j = float(np.clip(burst_reserve_j, 0.0, burst_capacity)) if burst_capacity > 0.0 else 0.0
    sustained_power_pct = 100.0 * sustained_power_ratio(power, power_w, burst_reserve_j)
    burst_reserve_pct = 100.0 * burst_reserve_j / burst_capacity if burst_capacity > 1e-9 else 100.0

    fan_power_share_w = max(0.0, power_w - power.aux_power_w) / max(1, fan_count)
    overload_ratio = max(0.0, power_w / max(power.continuous_power_w, 1e-6) - 1.0)
    heating_w = fan_power_share_w * (power.fan_heat_fraction + power.overload_heat_gain * overload_ratio)
    cooling_w = power.fan_cooling_w_per_c * np.maximum(prev_fan_temp - power.ambient_temp_c, 0.0)
    fan_temp_next = prev_fan_temp + (heating_w - cooling_w) * dt_s / max(power.fan_thermal_capacity_j_per_c, 1e-6)
    thermal_scale_next = temperature_to_thermal_scale(fan_temp_next, power)

    current_a = power_w / max(open_circuit_v, 1.0)
    loaded_voltage_v = max(power.battery_empty_v * 0.5, open_circuit_v - current_a * power.internal_resistance_ohm)
    supply_scale_next = power.min_supply_scale + (1.0 - power.min_supply_scale) * smoothstep01((loaded_voltage_v - power.battery_empty_v) / span_v)
    supply_scale_next = max(power.min_supply_scale, min(1.0, float(supply_scale_next)))

    energy_used_wh = float(power_state.energy_used_wh) + power_w * dt_s / 3600.0
    capacity_wh = max(1e-6, power.capacity_kwh * 1000.0)
    soc_frac = max(0.0, 1.0 - energy_used_wh / capacity_wh)
    open_circuit_v = power.battery_empty_v + span_v * soc_frac
    loaded_voltage_v = max(power.battery_empty_v * 0.5, open_circuit_v - current_a * power.internal_resistance_ohm)

    fan_response_pct = 100.0
    active_fans = fan_cmd_mean > 1e-6
    if np.any(active_fans):
        fan_response_pct = float(100.0 * np.mean(fan_actual_mean[active_fans] / fan_cmd_mean[active_fans]))

    telemetry = {
        "battery_v": float(loaded_voltage_v),
        "battery_a": float(current_a),
        "battery_soc": float(100.0 * soc_frac),
        "power_w": float(power_w),
        "energy_wh": float(energy_used_wh),
        "thrust_scale_pct": float(100.0 * min(supply_scale_next, float(np.mean(thermal_scale_next)))) ,
        "continuous_power_pct": float(sustained_power_pct),
        "continuous_power_raw_pct": float(continuous_power_raw_pct),
        "sustained_power_pct": float(sustained_power_pct),
        "burst_reserve_pct": float(burst_reserve_pct),
        "burst_clip_pct": float(burst_clip_pct),
        "burst_active_time_s": float(burst_active_time_s),
        "power_margin_kw": float((power.continuous_power_w - power_w) / 1000.0),
        "thermal_scale_pct": float(100.0 * np.mean(thermal_scale_next)),
        "fan_temp_max_c": float(np.max(fan_temp_next)) if fan_temp_next.size else float(power.ambient_temp_c),
        "fan_temp_mean_c": float(np.mean(fan_temp_next)) if fan_temp_next.size else float(power.ambient_temp_c),
        "fan_response_pct": float(fan_response_pct),
        "fan_cmd_16": [float(v) for v in fan_cmd_mean],
        "fan_actual_16": [float(v) for v in fan_actual_mean],
        "fan_temp_c_16": [float(v) for v in fan_temp_next],
        "fan_thermal_scale_16": [float(100.0 * v) for v in thermal_scale_next],
        "alpha_cmd_deg_32": [float(v) for v in np.degrees(alpha_cmd)],
        "alpha_cmd_rms": float(np.sqrt(np.mean(np.degrees(alpha_cmd) ** 2))) if alpha_cmd.size else 0.0,
    }
    updated_state = PowerSystemState(
        soc_frac=soc_frac,
        voltage_v=float(loaded_voltage_v),
        current_a=float(current_a),
        power_w=float(power_w),
        energy_used_wh=float(energy_used_wh),
        thrust_scale=float(min(supply_scale_next, float(np.mean(thermal_scale_next)))),
        burst_reserve_j=float(burst_reserve_j),
        fan_mean_n=np.asarray(fan_actual_mean, dtype=float),
        fan_temp_c=np.asarray(fan_temp_next, dtype=float),
        thermal_scale=np.asarray(thermal_scale_next, dtype=float),
    )
    return thrust_actual, updated_state, telemetry


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


def run_demo(dir_deg: float, fxy_n: float, duration_s: float, yaw_hold_deg: float = 0.0, mz_nm: float = 0.0, version: str = "v2", geom: RingGeometry | None = None, sim: SimParams | None = None):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    st = SimState(yaw_deg=yaw_hold_deg)
    steps = int(duration_s / sim.dt_s)
    hist = {"t": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [], "yaw_deg": [], "yaw_rate_deg_s": [], "mz_est": [], "alpha_deg_rms": [], "alpha_deg_32": [], "ft_tan_32": []}

    phi = math.radians(dir_deg)
    fx_cmd = fxy_n * math.cos(phi)
    fy_cmd = fxy_n * math.sin(phi)
    fz_cmd = sim.mass_kg * sim.gravity
    theta = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)

    for k in range(steps):
        t = k * sim.dt_s
        req = AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm)
        # allocate using the chosen allocator version; v1 does not compute tangential forces
        # and therefore has no ft_tan_per_seg_n attribute.
        if version.lower() == "v1":
            alloc = allocate_v1(geom, req, topology=topology, effectiveness=effectiveness)
            ft_rms = 0.0
            ft_vals = [0.0] * geom.n_segments
        else:
            alloc = allocate_v2(geom, req, topology=topology, effectiveness=effectiveness)
            ft_rms = float(np.sqrt(np.mean(alloc.ft_tan_per_seg_n ** 2)))
            ft_vals = list(alloc.ft_tan_per_seg_n)

        forces = thrust_vectors_body(geom, alloc.thrust_per_seg_n, alloc.alpha_rad, theta)
        if version.lower() != "v1":
            fx_t = alloc.ft_tan_per_seg_n * (-np.sin(theta))
            fy_t = alloc.ft_tan_per_seg_n * (np.cos(theta))
            forces = forces + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)

        net, mz_est = net_force_and_yaw_moment(geom, forces)
        thrust, power_state, telemetry = apply_power_system(
            alloc.thrust_per_seg_n,
            alloc.thrust_per_seg_n,
            alloc.alpha_rad,
            alloc.alpha_rad,
            np.asarray(ft_vals, dtype=float),
            sim,
            geom,
            power,
            power_state,
            sim.dt_s,
        )
        forces = thrust_vectors_body(geom, thrust, alloc.alpha_rad, theta)
        if version.lower() != "v1":
            forces = forces + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
        net, mz_est = net_force_and_yaw_moment(geom, forces)
        step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim)

        hist.setdefault("fan_thrust_16", []).append(list(telemetry["fan_actual_16"]))
        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m); hist["z"].append(st.z_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["vz"].append(st.vz_mps)
        hist["yaw_deg"].append(st.yaw_deg); hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alloc.alpha_rad) ** 2))))
        # record full flap/thrust arrays
        hist["alpha_deg_32"].append(list(np.degrees(alloc.alpha_rad)))
        hist["ft_tan_32"].append(ft_vals)
        hist.setdefault("ft_tan_rms", []).append(ft_rms)
        append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net)

    return hist

def run_step_test_v3(dir_deg_a: float = 0.0, dir_deg_b: float = 180.0, fxy_n: float = 3000.0, step_time_s: float = 3.0, total_s: float = 8.0, yaw_hold_deg: float = 0.0, mz_nm: float = 0.0, version_note: str = "v3", geom=None, sim=None, lim: ActuatorLimits | None = None, pl: PlenumModel | None = None, fault: FaultSpec | None = None):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    fault = fault or FaultSpec()
    st = SimState(yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    step_k = int(step_time_s / sim.dt_s)
    theta = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)
    fz_cmd = sim.mass_kg * sim.gravity

    hist = {"t": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [], "yaw_deg": [], "yaw_rate_deg_s": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [], "speed": [], "cmd_dir_deg": [], "alpha_deg_32": [], "ft_tan_32": [], "fan_thrust_16": []}

    for k in range(steps):
        t = k * sim.dt_s
        dir_deg = dir_deg_a if k < step_k else dir_deg_b
        phi = math.radians(dir_deg)
        fx_cmd = fxy_n * math.cos(phi)
        fy_cmd = fxy_n * math.sin(phi)

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm), fault=fault, topology=topology, effectiveness=effectiveness)
        alpha_target = apply_command_faults_to_alpha(alloc.alpha_rad, fault)
        thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault, topology=topology)
        ft_target = alloc.ft_tan_per_seg_n

        state = apply_actuator_limits(state, alpha_target, ft_target, lim, sim.dt_s, fault=fault)

        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)
        alpha_actual = apply_faults_to_alpha(state.alpha_rad, fault)

        thrust_cmd = thrust_target
        thrust_pre_power = thrust_cmd * state.plenum_scale
        thrust, power_state, telemetry = apply_power_system(
            thrust_cmd,
            thrust_pre_power,
            alpha_target,
            alpha_actual,
            state.ft_tan_per_seg_n,
            sim,
            geom,
            power,
            power_state,
            sim.dt_s,
        )
        hist["fan_thrust_16"].append(list(telemetry["fan_actual_16"]))

        forces_main = thrust_vectors_body(geom, thrust, alpha_actual, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
        net, mz_est = net_force_and_yaw_moment(geom, forces)

        step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim)
        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m); hist["z"].append(st.z_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["vz"].append(st.vz_mps)
        hist["yaw_deg"].append(st.yaw_deg); hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alpha_actual) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        # full per-segment data for visualization
        hist["alpha_deg_32"].append(list(np.degrees(alpha_actual)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["speed"].append(speed)
        hist["cmd_dir_deg"].append(dir_deg)
        append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net)

    coupling = yaw_track_coupling_mean_abs(hist)
    t_arr = np.array(hist["t"], float)
    vx_arr = np.array(hist["vx"], float)
    vy_arr = np.array(hist["vy"], float)
    sp_arr = np.array(hist["speed"], float)

    # Gate-style V3 metrics: direction reversal without yaw
    step_idx = int(step_time_s / sim.dt_s)
    step_t = float(t_arr[step_idx])

    # direction of travel (deg)
    track_deg = (np.degrees(np.arctan2(vy_arr, vx_arr)) + 360.0) % 360.0
    target_deg = (dir_deg_b + 360.0) % 360.0
    def ang_err(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    # t90_dir_s: first time track is within 20 deg of target AND speed > 0.5m/s
    t90_dir = None
    for i in range(step_idx, len(t_arr)):
        if sp_arr[i] > 0.5 and ang_err(track_deg[i], target_deg) <= 20.0:
            t90_dir = float(t_arr[i] - step_t)
            break

    # t_reversal_s: first time vx crosses 0 (for 0->180 step on X)
    t_reversal = None
    for i in range(step_idx, len(t_arr)):
        if vx_arr[i] < 0.0:
            t_reversal = float(t_arr[i] - step_t)
            break

    # min speed during transition (first 2 seconds after step)
    win_end = min(len(t_arr) - 1, step_idx + int(2.0 / sim.dt_s))
    min_speed_transition = float(np.min(sp_arr[step_idx:win_end+1])) if win_end > step_idx else None

    out = {"version": version_note, "dir_a_deg": dir_deg_a, "dir_b_deg": dir_deg_b, "fxy_n": fxy_n, "step_time_s": step_time_s, "total_s": total_s, "yaw_hold_deg": yaw_hold_deg, "mz_nm_cmd": mz_nm,
           "actuator_limits": {"alpha_rate_deg_s": lim.alpha_rate_deg_s, "ft_rate_n_s": lim.ft_rate_n_s},
           "plenum_model": {"tau_s": pl.tau_s, "min_scale": pl.min_scale, "max_scale": pl.max_scale},
           "power": power.__dict__,
           "step_metrics": {"t90_dir_s": t90_dir, "t_reversal_s": t_reversal, "min_speed_transition_mps": min_speed_transition, "peak_speed_mps": float(np.max(sp_arr)) if sp_arr.size else None, "yaw_track_coupling_mean_abs_deg": coupling},
           "final": {"x_m": hist["x"][-1], "y_m": hist["y"][-1], "z_m": hist["z"][-1], "vx_mps": hist["vx"][-1], "vy_mps": hist["vy"][-1], "vz_mps": hist["vz"][-1], "yaw_deg": hist["yaw_deg"][-1], "yaw_rate_deg_s": hist["yaw_rate_deg_s"][-1], "alpha_deg_rms": hist["alpha_deg_rms"][-1], "ft_tan_rms": hist["ft_tan_rms"][-1], "mz_est_nm": hist["mz_est"][-1]}}
    return out, hist


# new step-snap function for Aurora abrupt stop + reverse

def run_step_snap_v3(
    dir_deg_a: float = 0.0,
    dir_deg_b: float = 180.0,
    fxy_n: float = 3000.0,
    step_time_s: float = 3.0,
    snap_stop_s: float = 0.8,
    total_s: float = 9.0,
    yaw_hold_deg: float = 0.0,
    mz_nm: float = 0.0,
    geom=None,
    sim=None,
    lim: ActuatorLimits | None = None,
    pl: PlenumModel | None = None,
    fault: FaultSpec | None = None,
    brake_gain: float = 1.2,
    speed_stop_thr_mps: float = 0.2,
    redirect_hold_frac: float = -1.0,
    redirect_steer_scale: float = 1.0,
):
    """
    V3 Step-Snap maneuver:
    A) Travel in dir A until step_time_s
    B) Snap-stop for snap_stop_s by commanding force opposite current velocity
    C) Reverse into dir B for remainder

    Metrics focus on "UFO-like" abrupt stop + reverse without yaw.
    """
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    fault = fault or FaultSpec()

    st = SimState(yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    step_idx = int(step_time_s / sim.dt_s)
    snap_end_idx = int((step_time_s + snap_stop_s) / sim.dt_s)

    theta = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)
    fz_cmd = sim.mass_kg * sim.gravity

    hist = {
        "t": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [], "speed": [],
        "yaw_deg": [], "yaw_rate_deg_s": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [],
        "alpha_deg_32": [], "ft_tan_32": [], "fan_thrust_16": [],
        "cmd_phase": [], "cmd_dir_deg": [], "fx_cmd": [], "fy_cmd": [],
        "guard_scale": [], "flap_guard_scale": [], "power_guard_scale": [],
        "thermal_guard_scale": [], "supply_guard_scale": [], "fault_guard_scale": [],
        "fault_response_scale": [], "fault_available_scale": [], "fault_asymmetry_pct": [],
        "fxy_budget_n": [], "budget_ratio": [], "speed_guard_scale": [], "gain_guard_scale": [],
        "continuous_power_ratio": [], "flap_usage_ratio": []
    }

    # helpers
    def ang_err(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    def smoothstep_local(u: float) -> float:
        u = max(0.0, min(1.0, float(u)))
        return u * u * (3.0 - 2.0 * u)

    def unit_or_default(x: float, y: float, default_xy: tuple[float, float]) -> tuple[float, float]:
        mag = math.hypot(x, y)
        if mag < 1e-6:
            return default_xy
        return (x / mag, y / mag)

    target_dir_rad = math.radians(dir_deg_b)
    target_unit = (math.cos(target_dir_rad), math.sin(target_dir_rad))
    nominal_brake_dir_deg = (dir_deg_a + 180.0) % 360.0
    nominal_brake_rad = math.radians(nominal_brake_dir_deg)
    nominal_brake_unit = (math.cos(nominal_brake_rad), math.sin(nominal_brake_rad))
    redirect_sep_deg = ang_err(nominal_brake_dir_deg, dir_deg_b)
    turn_ratio = redirect_sep_deg / 180.0
    if redirect_hold_frac < 0.0:
        hold_frac = float(np.clip(0.65 + 0.35 * turn_ratio, 0.55, 0.98))
    else:
        hold_frac = float(np.clip(redirect_hold_frac, 0.0, 0.95))
    steer_scale = float(np.clip(redirect_steer_scale, 0.0, 1.5))
    fault_profile = fault_motion_guard(geom, fault)
    phase_c_ramp_steps = max(
        1,
        int((0.75 + 0.95 * turn_ratio + 0.35 * hold_frac) / max(1e-6, fault_profile['dead_align_scale']) / sim.dt_s),
    )
    step_speed_ref = 0.0
    command_rate_n_s = 9000.0
    command_fx_prev = 0.0
    command_fy_prev = 0.0
    snap_power_ratio_filt = 1.0

    for k in range(steps):
        t = k * sim.dt_s
        fxy_budget_n, guard = guidance_force_budget(power_state, state, geom, power, fxy_n, fault=fault)
        budget_ratio = float(guard["budget_ratio"])
        speed_guard_scale = 0.38 + 0.62 * smoothstep_local((budget_ratio - 0.25) / 0.75)
        gain_guard_scale = 0.30 + 0.70 * smoothstep_local((budget_ratio - 0.25) / 0.75)
        power_pressure = smoothstep_local((guard["continuous_power_ratio"] - 0.88) / 0.12)
        power_guard_scale = float(guard["power_guard_scale"])
        snap_power_ratio_filt = 0.92 * snap_power_ratio_filt + 0.08 * float(guard["continuous_power_ratio"])
        power_priority_scale = 1.0 - 0.32 * smoothstep_local((snap_power_ratio_filt - 0.94) / 0.08)
        power_priority_scale = float(np.clip(power_priority_scale, 0.66, 1.0))
        dead_align_scale = float(guard["dead_align_scale"])
        dead_cross_scale = float(guard["dead_cross_scale"])
        dead_align_speed_floor_mps = float(guard["dead_align_speed_floor_mps"])
        plenum_power_trim = float(guard["plenum_power_trim"])
        plenum_revector_trim = float(guard["plenum_revector_trim"])
        plenum_align_speed_floor_mps = float(guard["plenum_align_speed_floor_mps"])
        plenum_brake_trim = float(guard["plenum_brake_trim"])
        speed_guard_scale *= 1.0 - 0.18 * power_pressure
        gain_guard_scale *= 1.0 - 0.24 * power_pressure
        speed_guard_scale *= 0.78 + 0.22 * power_guard_scale
        gain_guard_scale *= 0.70 + 0.30 * power_guard_scale
        speed_guard_scale *= 0.72 + 0.28 * power_priority_scale
        gain_guard_scale *= 0.62 + 0.38 * power_priority_scale
        speed_guard_scale *= float(guard["fault_guard_scale"])
        gain_guard_scale *= float(guard["fault_guard_scale"])
        speed_guard_scale *= plenum_power_trim
        gain_guard_scale *= plenum_power_trim
        fxy_budget_n *= 0.82 + 0.18 * power_guard_scale
        fxy_budget_n *= 0.70 + 0.30 * power_priority_scale
        fxy_budget_n *= plenum_power_trim

        maneuver_state = EstimatedVehicleState(
            x_m=st.x_m,
            y_m=st.y_m,
            z_m=st.z_m,
            vx_mps=st.vx_mps,
            vy_mps=st.vy_mps,
            vz_mps=st.vz_mps,
            yaw_deg=st.yaw_deg,
            yaw_rate_deg_s=st.yaw_rate_deg_s,
            battery_soc_pct=100.0 * power_state.soc_frac,
            bus_voltage_v=power_state.voltage_v,
            continuous_power_ratio=float(guard["continuous_power_ratio"]),
            thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
            fault_available_scale=float(guard["fault_available_scale"]),
            fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
        )
        maneuver_health = ActuatorHealthState(
            lateral_budget_n=fxy_budget_n,
            guard_scale=float(guard["guard_scale"]),
            response_scale=float(guard["fault_response_scale"]),
            continuous_power_ratio=float(guard["continuous_power_ratio"]),
            thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
            supply_scale_pct=100.0 * float(guard["supply_guard_scale"]),
            fault_available_scale=float(guard["fault_available_scale"]),
            fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
        )

        if k < step_idx:
            phase = "A"
            dir_deg = dir_deg_a
            phase_control = command_directional_force(
                math.cos(math.radians(dir_deg_a)),
                math.sin(math.radians(dir_deg_a)),
                maneuver_health,
                fxy_command_n=fxy_n,
                force_scale=gain_guard_scale,
                fz_n=fz_cmd,
                mz_nm=mz_nm,
                source="maneuver-controller.step-snap-cruise",
            )
            fx_raw = phase_control.desired_wrench.fx_n
            fy_raw = phase_control.desired_wrench.fy_n

        elif k < snap_end_idx:
            phase = "B"
            if step_speed_ref <= 1e-6:
                step_speed_ref = max(speed_stop_thr_mps, float(math.hypot(st.vx_mps, st.vy_mps)))
            progress = (k - step_idx) / max(1, snap_end_idx - step_idx - 1)
            steer_progress = smoothstep_local((progress - hold_frac) / max(1e-6, 1.0 - hold_frac))
            steer_mix = float(np.clip(0.35 * steer_progress * steer_scale, 0.0, 0.45))
            phase_control = track_step_snap_brake(
                maneuver_state,
                target_ux=target_unit[0],
                target_uy=target_unit[1],
                nominal_brake_ux=nominal_brake_unit[0],
                nominal_brake_uy=nominal_brake_unit[1],
                steer_mix=steer_mix,
                health=ActuatorHealthState(
                    lateral_budget_n=fxy_budget_n * brake_gain,
                    guard_scale=maneuver_health.guard_scale,
                    response_scale=maneuver_health.response_scale,
                    continuous_power_ratio=maneuver_health.continuous_power_ratio,
                    thermal_scale_pct=maneuver_health.thermal_scale_pct,
                    supply_scale_pct=maneuver_health.supply_scale_pct,
                    fault_available_scale=maneuver_health.fault_available_scale,
                    fault_asymmetry_pct=maneuver_health.fault_asymmetry_pct,
                ),
                fxy_command_n=fxy_n,
                brake_gain=brake_gain * gain_guard_scale * power_priority_scale * plenum_revector_trim * plenum_brake_trim * (1.0 - 0.20 * steer_mix),
                fz_n=fz_cmd,
                mz_nm=mz_nm,
                source="maneuver-controller.step-snap-brake",
            )
            fx_raw = phase_control.desired_wrench.fx_n
            fy_raw = phase_control.desired_wrench.fy_n
            dir_deg = (math.degrees(math.atan2(phase_control.command_uy, phase_control.command_ux)) + 360.0) % 360.0

        else:
            phase = "C"
            if step_speed_ref <= 1e-6:
                step_speed_ref = max(speed_stop_thr_mps, float(math.hypot(st.vx_mps, st.vy_mps)))
            dir_deg = dir_deg_b
            phase_c_progress = smoothstep_local((k - snap_end_idx) / phase_c_ramp_steps)
            desired_along_speed = step_speed_ref * phase_c_progress * speed_guard_scale
            if dead_align_scale > 1.0:
                align_capture = smoothstep_local((phase_c_progress - 0.18) / 0.42)
                desired_along_speed = max(
                    desired_along_speed,
                    dead_align_speed_floor_mps * align_capture,
                )
            if plenum_align_speed_floor_mps > 1e-6:
                plenum_align_capture = smoothstep_local((phase_c_progress - 0.24) / 0.34)
                desired_along_speed = max(
                    desired_along_speed,
                    plenum_align_speed_floor_mps * plenum_align_capture,
                )
            cross_gain = (0.75 + 0.45 * turn_ratio) * dead_cross_scale * plenum_revector_trim
            reverse_force_scale = (0.40 + 0.60 * phase_c_progress) * gain_guard_scale * power_priority_scale * plenum_revector_trim
            if dead_align_scale > 1.0:
                reverse_force_scale *= 0.90 + 0.10 * dead_align_scale
            phase_control = track_step_snap_reverse(
                maneuver_state,
                target_ux=target_unit[0],
                target_uy=target_unit[1],
                reference_speed_mps=step_speed_ref,
                desired_along_speed_mps=desired_along_speed,
                cross_gain=cross_gain,
                health=maneuver_health,
                fxy_command_n=fxy_n,
                force_scale=reverse_force_scale,
                fz_n=fz_cmd,
                mz_nm=mz_nm,
                source="maneuver-controller.step-snap-reverse",
            )
            fx_raw = phase_control.desired_wrench.fx_n
            fy_raw = phase_control.desired_wrench.fy_n

        fx_raw, fy_raw = clip_force_xy(fx_raw, fy_raw, fxy_budget_n)
        command_rate_active_n_s = command_rate_n_s * (0.50 + 0.50 * min(speed_guard_scale, gain_guard_scale))
        command_rate_active_n_s *= float(guard["fault_response_scale"])
        fx_cmd, fy_cmd = rate_limit_xy_force(command_fx_prev, command_fy_prev, fx_raw, fy_raw, command_rate_active_n_s, sim.dt_s)
        command_fx_prev, command_fy_prev = fx_cmd, fy_cmd

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm), fault=fault, topology=topology, effectiveness=effectiveness)
        alpha_target = apply_command_faults_to_alpha(alloc.alpha_rad, fault)
        thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault, topology=topology)
        ft_target = alloc.ft_tan_per_seg_n

        # actuator limits
        state = apply_actuator_limits(state, alpha_target, ft_target, lim, sim.dt_s, fault=fault)

        # plenum lag
        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)
        alpha_actual = apply_faults_to_alpha(state.alpha_rad, fault)

        thrust_cmd = thrust_target
        thrust_pre_power = thrust_cmd * state.plenum_scale
        thrust, power_state, telemetry = apply_power_system(
            thrust_cmd,
            thrust_pre_power,
            alpha_target,
            alpha_actual,
            state.ft_tan_per_seg_n,
            sim,
            geom,
            power,
            power_state,
            sim.dt_s,
        )
        hist["fan_thrust_16"].append(list(telemetry["fan_actual_16"]))

        forces_main = thrust_vectors_body(geom, thrust, alpha_actual, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)

        net, mz_est = net_force_and_yaw_moment(geom, forces)
        step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim)

        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m); hist["z"].append(st.z_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["vz"].append(st.vz_mps)
        hist["speed"].append(speed)
        hist["yaw_deg"].append(st.yaw_deg); hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alpha_actual) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        # per-segment arrays for snap visualization
        hist["alpha_deg_32"].append(list(np.degrees(alpha_actual)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["cmd_phase"].append(phase)
        hist["cmd_dir_deg"].append(dir_deg if isinstance(dir_deg, float) else float(dir_deg))
        hist["fx_cmd"].append(float(fx_cmd)); hist["fy_cmd"].append(float(fy_cmd))
        hist["guard_scale"].append(float(guard["guard_scale"]))
        hist["flap_guard_scale"].append(float(guard["flap_guard_scale"]))
        hist["power_guard_scale"].append(float(guard["power_guard_scale"]))
        hist["thermal_guard_scale"].append(float(guard["thermal_guard_scale"]))
        hist["supply_guard_scale"].append(float(guard["supply_guard_scale"]))
        hist["fault_guard_scale"].append(float(guard["fault_guard_scale"]))
        hist["fault_response_scale"].append(float(guard["fault_response_scale"]))
        hist["fault_available_scale"].append(float(guard["fault_available_scale"]))
        hist["fault_asymmetry_pct"].append(float(guard["fault_asymmetry_pct"]))
        hist["fxy_budget_n"].append(float(fxy_budget_n))
        hist["budget_ratio"].append(float(guard["budget_ratio"]))
        hist["speed_guard_scale"].append(float(speed_guard_scale))
        hist["gain_guard_scale"].append(float(gain_guard_scale))
        hist["continuous_power_ratio"].append(float(guard["continuous_power_ratio"]))
        hist["flap_usage_ratio"].append(float(guard["flap_usage_ratio"]))
        append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net)

    # ----- Gate metrics -----
    t_arr = np.array(hist["t"], float)
    vx_arr = np.array(hist["vx"], float)
    vy_arr = np.array(hist["vy"], float)
    sp_arr = np.array(hist["speed"], float)
    x_arr = np.array(hist["x"], float)
    y_arr = np.array(hist["y"], float)

    # track angle deg
    track_deg = (np.degrees(np.arctan2(vy_arr, vx_arr)) + 360.0) % 360.0
    target_deg = (dir_deg_b + 360.0) % 360.0

    step_t = float(t_arr[step_idx])
    snap_end_t = float(t_arr[min(snap_end_idx, len(t_arr)-1)])

    # 1) time to speed below threshold (after snap begins)
    t_to_stop = None
    for i in range(step_idx, len(t_arr)):
        if sp_arr[i] <= speed_stop_thr_mps:
            t_to_stop = float(t_arr[i] - step_t)
            break

    # 2) snap stop distance (distance traveled between step start and first stop-threshold hit, or snap end)
    if t_to_stop is not None:
        stop_idx = step_idx + int(t_to_stop / sim.dt_s)
        stop_idx = min(stop_idx, len(t_arr)-1)
    else:
        stop_idx = min(snap_end_idx, len(t_arr)-1)

    dx = x_arr[stop_idx] - x_arr[step_idx]
    dy = y_arr[stop_idx] - y_arr[step_idx]
    snap_stop_distance = float(math.hypot(dx, dy))

    # 3) time to reversal (vx sign for 0->180 on X; general case uses dot with target dir)
    # General: want velocity to have positive projection on target direction
    tx = math.cos(math.radians(target_deg))
    ty = math.sin(math.radians(target_deg))

    t_reversal = None
    for i in range(step_idx, len(t_arr)):
        proj = vx_arr[i]*tx + vy_arr[i]*ty
        if proj > 0.0:
            t_reversal = float(t_arr[i] - step_t)
            break

    # 4) time to align direction within 20 deg (and speed > 0.5)
    t90_dir = None
    for i in range(step_idx, len(t_arr)):
        if sp_arr[i] > 0.5 and ang_err(track_deg[i], target_deg) <= 20.0:
            t90_dir = float(t_arr[i] - step_t)
            break

    coupling = yaw_track_coupling_mean_abs(hist)

    # compute speeds at key times for reporting
    speed_at_step = float(sp_arr[step_idx]) if step_idx < len(sp_arr) else None
    speed_at_snap_end = float(sp_arr[snap_end_idx]) if snap_end_idx < len(sp_arr) else None

    out = {
        "version": "v3-step-snap",
        "dir_a_deg": dir_deg_a,
        "dir_b_deg": dir_deg_b,
        "fxy_n": fxy_n,
        "step_time_s": step_time_s,
        "snap_stop_s": snap_stop_s,
        "total_s": total_s,
        "yaw_hold_deg": yaw_hold_deg,
        "mz_nm_cmd": mz_nm,
        "brake_gain": brake_gain,
        "speed_stop_thr_mps": speed_stop_thr_mps,
        "actuator_limits": {"alpha_rate_deg_s": lim.alpha_rate_deg_s, "ft_rate_n_s": lim.ft_rate_n_s},
        "plenum_model": {"tau_s": pl.tau_s, "min_scale": pl.min_scale, "max_scale": pl.max_scale},
        "power": power.__dict__,
        "redirect_shaping": {
            "hold_frac": hold_frac,
            "steer_scale": steer_scale,
            "turn_separation_deg": redirect_sep_deg,
            "command_rate_n_s": command_rate_n_s,
            "guard_mode": "budget-aware-step-snap",
        },
        "step_metrics": {
            "speed_at_step_mps": speed_at_step,
            "speed_at_snap_end_mps": speed_at_snap_end,
            "t_to_speed_below_thr_s": t_to_stop,
            "snap_stop_distance_m": snap_stop_distance,
            "t_reversal_s": t_reversal,
            "t90_dir_s": t90_dir,
            "peak_speed_mps": float(np.max(sp_arr)) if sp_arr.size else None,
            "yaw_track_coupling_mean_abs_deg": coupling,
        },
        "final": {
            "x_m": float(x_arr[-1]),
            "y_m": float(y_arr[-1]),
            "z_m": float(hist["z"][-1]),
            "vx_mps": float(vx_arr[-1]),
            "vy_mps": float(vy_arr[-1]),
            "vz_mps": float(hist["vz"][-1]),
            "yaw_deg": float(hist["yaw_deg"][-1]),
            "yaw_rate_deg_s": float(hist["yaw_rate_deg_s"][-1]),
            "alpha_deg_rms": float(hist["alpha_deg_rms"][-1]),
            "ft_tan_rms": float(hist["ft_tan_rms"][-1]),
            "mz_est_nm": float(hist["mz_est"][-1]),
        },
        "timestamps": {
            "step_t": step_t,
            "snap_end_t": snap_end_t,
        },
        "hardware_assumptions": hardware_assumptions_payload(topology, effectiveness),
    }

    return out, hist


def run_step_redirect_v3(
    dir_deg_a: float = 0.0,
    dir_deg_b: float = 90.0,
    fxy_n: float = 2200.0,
    step_time_s: float = 3.0,
    redirect_time_s: float = 1.6,
    total_s: float = 9.0,
    yaw_hold_deg: float = 0.0,
    mz_nm: float = 0.0,
    geom=None,
    sim=None,
    lim: ActuatorLimits | None = None,
    pl: PlenumModel | None = None,
    fault: FaultSpec | None = None,
    redirect_speed_scale: float = 0.88,
    redirect_cross_gain: float = 1.0,
):
    """
    V3 Step-Redirect maneuver:
    A) Travel in dir A until step_time_s
    B) Smoothly rotate the desired velocity vector toward dir B without forcing a full stop
    C) Hold the new direction while keeping yaw fixed

    This better matches a holonomic UFO craft for 45/90-degree sidestep redirects.
    """
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    fault = fault or FaultSpec()

    st = SimState(yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    step_idx = int(step_time_s / sim.dt_s)
    redirect_end_idx = int((step_time_s + redirect_time_s) / sim.dt_s)

    theta = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)
    fz_cmd = sim.mass_kg * sim.gravity

    hist = {
        "t": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [], "speed": [],
        "yaw_deg": [], "yaw_rate_deg_s": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [],
        "alpha_deg_32": [], "ft_tan_32": [], "fan_thrust_16": [],
        "cmd_phase": [], "cmd_dir_deg": [], "fx_cmd": [], "fy_cmd": []
    }

    def ang_err(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    def smoothstep_local(u: float) -> float:
        u = max(0.0, min(1.0, float(u)))
        return u * u * (3.0 - 2.0 * u)

    def unit_or_default(x: float, y: float, default_xy: tuple[float, float]) -> tuple[float, float]:
        mag = math.hypot(x, y)
        if mag < 1e-6:
            return default_xy
        return (x / mag, y / mag)

    start_unit = (math.cos(math.radians(dir_deg_a)), math.sin(math.radians(dir_deg_a)))
    target_unit = (math.cos(math.radians(dir_deg_b)), math.sin(math.radians(dir_deg_b)))
    redirect_sep_deg = ang_err(dir_deg_a, dir_deg_b)
    turn_ratio = redirect_sep_deg / 180.0
    redirect_steps = max(1, redirect_end_idx - step_idx)
    settle_steps = max(1, int((0.60 + 0.50 * turn_ratio) / sim.dt_s))
    step_speed_ref = 0.0
    command_rate_n_s = 9000.0
    command_fx_prev = 0.0
    command_fy_prev = 0.0
    redirect_power_ratio_filt = 1.0

    for k in range(steps):
        t = k * sim.dt_s
        fxy_budget_n, guard = guidance_force_budget(power_state, state, geom, power, fxy_n, fault=fault)
        budget_ratio = float(guard["budget_ratio"])
        speed_guard_scale = 0.40 + 0.60 * smoothstep_local((budget_ratio - 0.25) / 0.75)
        gain_guard_scale = 0.34 + 0.66 * smoothstep_local((budget_ratio - 0.25) / 0.75)
        power_pressure = smoothstep_local((guard["continuous_power_ratio"] - 0.88) / 0.12)
        power_guard_scale = float(guard["power_guard_scale"])
        redirect_power_ratio_filt = 0.92 * redirect_power_ratio_filt + 0.08 * float(guard["continuous_power_ratio"])
        power_priority_scale = 1.0 - 0.30 * smoothstep_local((redirect_power_ratio_filt - 0.94) / 0.08)
        power_priority_scale = float(np.clip(power_priority_scale, 0.68, 1.0))
        speed_guard_scale *= 1.0 - 0.18 * power_pressure
        gain_guard_scale *= 1.0 - 0.24 * power_pressure
        speed_guard_scale *= 0.78 + 0.22 * power_guard_scale
        gain_guard_scale *= 0.70 + 0.30 * power_guard_scale
        speed_guard_scale *= 0.72 + 0.28 * power_priority_scale
        gain_guard_scale *= 0.62 + 0.38 * power_priority_scale
        speed_guard_scale *= float(guard["fault_guard_scale"])
        gain_guard_scale *= float(guard["fault_guard_scale"])
        fxy_budget_n *= 0.82 + 0.18 * power_guard_scale
        fxy_budget_n *= 0.70 + 0.30 * power_priority_scale

        if k < step_idx:
            phase = "A"
            dir_deg = dir_deg_a
            phi = math.radians(dir_deg)
            fx_raw = fxy_n * gain_guard_scale * math.cos(phi)
            fy_raw = fxy_n * gain_guard_scale * math.sin(phi)
        else:
            if step_speed_ref <= 1e-6:
                step_speed_ref = max(0.75, float(math.hypot(st.vx_mps, st.vy_mps)))

            if k < redirect_end_idx:
                phase = "R"
                progress = smoothstep_local((k - step_idx) / max(1, redirect_steps - 1))
            else:
                phase = "C"
                progress = 1.0

            desired_dir_x = (1.0 - progress) * start_unit[0] + progress * target_unit[0]
            desired_dir_y = (1.0 - progress) * start_unit[1] + progress * target_unit[1]
            desired_ux, desired_uy = unit_or_default(desired_dir_x, desired_dir_y, target_unit)

            desired_speed = step_speed_ref * (redirect_speed_scale + (1.0 - redirect_speed_scale) * progress)
            desired_speed *= speed_guard_scale

            if phase == "R":
                phase_scale = (0.50 + 0.40 * progress) * (1.0 - 0.12 * turn_ratio * (1.0 - progress))
            else:
                settle_progress = smoothstep_local((k - redirect_end_idx) / settle_steps)
                phase_scale = 0.78 + 0.22 * settle_progress

            phase_scale *= gain_guard_scale
            maneuver_health = ActuatorHealthState(
                lateral_budget_n=fxy_budget_n,
                guard_scale=float(guard["guard_scale"]),
                response_scale=float(guard["fault_response_scale"]),
                continuous_power_ratio=float(guard["continuous_power_ratio"]),
                thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
                supply_scale_pct=100.0 * float(guard["supply_guard_scale"]),
                fault_available_scale=float(guard["fault_available_scale"]),
                fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
            )
            maneuver_state = EstimatedVehicleState(
                x_m=st.x_m,
                y_m=st.y_m,
                z_m=st.z_m,
                vx_mps=st.vx_mps,
                vy_mps=st.vy_mps,
                vz_mps=st.vz_mps,
                yaw_deg=st.yaw_deg,
                yaw_rate_deg_s=st.yaw_rate_deg_s,
                battery_soc_pct=100.0 * power_state.soc_frac,
                bus_voltage_v=power_state.voltage_v,
                continuous_power_ratio=float(guard["continuous_power_ratio"]),
                thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
                fault_available_scale=float(guard["fault_available_scale"]),
                fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
            )
            redirect_target = RedirectTarget(
                desired_ux=desired_ux,
                desired_uy=desired_uy,
                desired_speed_mps=desired_speed,
                reference_speed_mps=step_speed_ref,
                cross_gain=redirect_cross_gain,
                hold_yaw_deg=yaw_hold_deg,
                phase=phase,
                force_scale=phase_scale,
            )
            redirect_control = track_redirect_velocity(
                maneuver_state,
                redirect_target,
                maneuver_health,
                fxy_command_n=fxy_n,
                mz_nm=mz_nm,
                source="maneuver-controller.step-redirect",
            )
            fx_raw = redirect_control.desired_wrench.fx_n
            fy_raw = redirect_control.desired_wrench.fy_n
            dir_deg = (math.degrees(math.atan2(redirect_control.command_uy, redirect_control.command_ux)) + 360.0) % 360.0

        fx_raw, fy_raw = clip_force_xy(fx_raw, fy_raw, fxy_budget_n)
        command_rate_active_n_s = command_rate_n_s * (0.55 + 0.45 * min(speed_guard_scale, gain_guard_scale))
        command_rate_active_n_s *= float(guard["fault_response_scale"])
        fx_cmd, fy_cmd = rate_limit_xy_force(command_fx_prev, command_fy_prev, fx_raw, fy_raw, command_rate_active_n_s, sim.dt_s)
        command_fx_prev, command_fy_prev = fx_cmd, fy_cmd

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm), fault=fault, topology=topology, effectiveness=effectiveness)
        alpha_target = apply_command_faults_to_alpha(alloc.alpha_rad, fault)
        thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault, topology=topology)
        ft_target = alloc.ft_tan_per_seg_n

        state = apply_actuator_limits(state, alpha_target, ft_target, lim, sim.dt_s, fault=fault)

        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)
        alpha_actual = apply_faults_to_alpha(state.alpha_rad, fault)

        thrust_cmd = thrust_target
        thrust_pre_power = thrust_cmd * state.plenum_scale
        thrust, power_state, telemetry = apply_power_system(
            thrust_cmd,
            thrust_pre_power,
            alpha_target,
            alpha_actual,
            state.ft_tan_per_seg_n,
            sim,
            geom,
            power,
            power_state,
            sim.dt_s,
        )
        hist["fan_thrust_16"].append(list(telemetry["fan_actual_16"]))

        forces_main = thrust_vectors_body(geom, thrust, alpha_actual, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)

        net, mz_est = net_force_and_yaw_moment(geom, forces)
        step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim)

        speed = float(math.hypot(st.vx_mps, st.vy_mps))
        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m); hist["z"].append(st.z_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["vz"].append(st.vz_mps)
        hist["speed"].append(speed)
        hist["yaw_deg"].append(st.yaw_deg); hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alpha_actual) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        hist["alpha_deg_32"].append(list(np.degrees(alpha_actual)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["cmd_phase"].append(phase)
        hist["cmd_dir_deg"].append(float(dir_deg))
        hist["fx_cmd"].append(float(fx_cmd)); hist["fy_cmd"].append(float(fy_cmd))
        append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net)

    t_arr = np.array(hist["t"], float)
    vx_arr = np.array(hist["vx"], float)
    vy_arr = np.array(hist["vy"], float)
    sp_arr = np.array(hist["speed"], float)
    x_arr = np.array(hist["x"], float)
    y_arr = np.array(hist["y"], float)
    track_deg = (np.degrees(np.arctan2(vy_arr, vx_arr)) + 360.0) % 360.0
    target_deg = (dir_deg_b + 360.0) % 360.0

    step_t = float(t_arr[step_idx])
    redirect_end_t = float(t_arr[min(redirect_end_idx, len(t_arr)-1)])

    tx = math.cos(math.radians(target_deg))
    ty = math.sin(math.radians(target_deg))
    t_reversal = None
    for i in range(step_idx, len(t_arr)):
        proj = vx_arr[i] * tx + vy_arr[i] * ty
        if proj > 0.0:
            t_reversal = float(t_arr[i] - step_t)
            break

    t90_dir = None
    for i in range(step_idx, len(t_arr)):
        if sp_arr[i] > 0.5 and ang_err(track_deg[i], target_deg) <= 20.0:
            t90_dir = float(t_arr[i] - step_t)
            break

    t_to_stop = None
    for i in range(step_idx, min(len(t_arr), redirect_end_idx + 1)):
        if sp_arr[i] <= 0.2:
            t_to_stop = float(t_arr[i] - step_t)
            break

    stop_idx = min(redirect_end_idx, len(t_arr) - 1)
    dx = x_arr[stop_idx] - x_arr[step_idx]
    dy = y_arr[stop_idx] - y_arr[step_idx]
    redirect_distance = float(math.hypot(dx, dy))

    win = sp_arr[step_idx:stop_idx + 1] if stop_idx >= step_idx else np.array([], dtype=float)
    min_speed_transition = float(np.min(win)) if win.size else None
    peak_speed_transition = float(np.max(win)) if win.size else None
    coupling = yaw_track_coupling_mean_abs(hist)

    out = {
        "version": "v3-step-redirect",
        "dir_a_deg": dir_deg_a,
        "dir_b_deg": dir_deg_b,
        "fxy_n": fxy_n,
        "step_time_s": step_time_s,
        "redirect_time_s": redirect_time_s,
        "total_s": total_s,
        "yaw_hold_deg": yaw_hold_deg,
        "mz_nm_cmd": mz_nm,
        "redirect_speed_scale": redirect_speed_scale,
        "redirect_cross_gain": redirect_cross_gain,
        "actuator_limits": {"alpha_rate_deg_s": lim.alpha_rate_deg_s, "ft_rate_n_s": lim.ft_rate_n_s},
        "plenum_model": {"tau_s": pl.tau_s, "min_scale": pl.min_scale, "max_scale": pl.max_scale},
        "power": power.__dict__,
        "redirect_shaping": {
            "speed_scale": redirect_speed_scale,
            "cross_gain": redirect_cross_gain,
            "turn_separation_deg": redirect_sep_deg,
        },
        "step_metrics": {
            "speed_at_step_mps": float(sp_arr[step_idx]) if step_idx < len(sp_arr) else None,
            "speed_at_redirect_end_mps": float(sp_arr[stop_idx]) if stop_idx < len(sp_arr) else None,
            "t_to_speed_below_thr_s": t_to_stop,
            "snap_stop_distance_m": redirect_distance,
            "t_reversal_s": t_reversal,
            "t90_dir_s": t90_dir,
            "min_speed_transition_mps": min_speed_transition,
            "peak_speed_transition_mps": peak_speed_transition,
            "peak_speed_mps": float(np.max(sp_arr)) if sp_arr.size else None,
            "yaw_track_coupling_mean_abs_deg": coupling,
        },
        "final": {
            "x_m": float(x_arr[-1]),
            "y_m": float(y_arr[-1]),
            "z_m": float(hist["z"][-1]),
            "vx_mps": float(vx_arr[-1]),
            "vy_mps": float(vy_arr[-1]),
            "vz_mps": float(hist["vz"][-1]),
            "yaw_deg": float(hist["yaw_deg"][-1]),
            "yaw_rate_deg_s": float(hist["yaw_rate_deg_s"][-1]),
            "alpha_deg_rms": float(hist["alpha_deg_rms"][-1]),
            "ft_tan_rms": float(hist["ft_tan_rms"][-1]),
            "mz_est_nm": float(hist["mz_est"][-1]),
        },
        "timestamps": {
            "step_t": step_t,
            "redirect_end_t": redirect_end_t,
        },
        "hardware_assumptions": hardware_assumptions_payload(topology, effectiveness),
    }

    return out, hist

def run_coordinate_mission_v5(
    dest_x_m: float = 60.0,
    dest_y_m: float = 0.0,
    dest_z_m: float = 0.0,
    start_x_m: float = 0.0,
    start_y_m: float = 0.0,
    start_z_m: float = 0.0,
    total_s: float = 30.0,
    yaw_hold_deg: float = 0.0,
    mz_nm: float = 0.0,
    cruise_alt_m: float = 12.0,
    max_speed_mps: float = 8.0,
    arrival_radius_m: float = 1.5,
    slow_radius_m: float = 14.0,
    descent_radius_m: float = 18.0,
    hold_s: float = 2.0,
    fxy_max_n: float = 3500.0,
    pos_k_n_per_m: float = 120.0,
    vel_k_n_per_mps: float = 900.0,
    obstacles: list[MissionObstacle] | None = None,
    geom=None,
    sim=None,
    lim: ActuatorLimits | None = None,
    pl: PlenumModel | None = None,
    power: PowerSystemParams | None = None,
    fault: FaultSpec | None = None,
):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = power or PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    obstacles = obstacles or []
    fault = fault or FaultSpec()

    st = SimState(x_m=start_x_m, y_m=start_y_m, z_m=start_z_m, yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    theta = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)
    fz_cmd = sim.mass_kg * sim.gravity
    transit_alt_m = max(cruise_alt_m, start_z_m, dest_z_m)
    planner_clearance_m = max(4.0, arrival_radius_m * 2.0)
    route_xy = plan_route_waypoints(start_x_m, start_y_m, dest_x_m, dest_y_m, obstacles, planner_clearance_m)
    goal_idx = 1 if len(route_xy) > 1 else 0
    command_rate_n_s = 9000.0
    command_fx_prev = 0.0
    command_fy_prev = 0.0
    controller_gains = XYVehicleControllerGains(
        pos_k_n_per_m=pos_k_n_per_m,
        vel_k_n_per_mps=vel_k_n_per_mps,
    )

    hist = {
        "t": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [], "speed": [],
        "yaw_deg": [], "yaw_rate_deg_s": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [],
        "alpha_deg_32": [], "ft_tan_32": [], "fan_thrust_16": [],
        "fx_cmd": [], "fy_cmd": [], "dist_to_goal": [], "dist_to_subgoal": [], "nearest_obstacle_m": [],
        "avoid_fx": [], "avoid_fy": [], "target_z_m": [], "mission_phase": [],
        "subgoal_x_m": [], "subgoal_y_m": [], "route_goal_idx": [],
        "guard_scale": [], "flap_guard_scale": [], "power_guard_scale": [],
        "thermal_guard_scale": [], "supply_guard_scale": [], "fault_guard_scale": [],
        "fault_response_scale": [], "fault_available_scale": [], "fault_asymmetry_pct": [],
        "fxy_budget_n": [], "budget_ratio": [], "speed_guard_scale": [], "gain_guard_scale": [],
        "goal_force_scale": [], "continuous_power_ratio": [], "flap_usage_ratio": []
    }

    arrival_time_s = None
    hold_start_s = None

    for k in range(steps):
        t = k * sim.dt_s
        dist_to_goal_m = float(math.hypot(dest_x_m - st.x_m, dest_y_m - st.y_m))
        speed_mps = float(math.hypot(st.vx_mps, st.vy_mps))

        while goal_idx < len(route_xy) - 1:
            subgoal_x_m, subgoal_y_m = route_xy[goal_idx]
            subgoal_dist_m = float(math.hypot(subgoal_x_m - st.x_m, subgoal_y_m - st.y_m))
            subgoal_radius_m = max(3.0, planner_clearance_m * 0.9)
            if subgoal_dist_m <= subgoal_radius_m:
                goal_idx += 1
            else:
                break

        goal_x_m, goal_y_m = route_xy[goal_idx]
        is_final_goal = goal_idx == len(route_xy) - 1
        goal_dx = goal_x_m - st.x_m
        goal_dy = goal_y_m - st.y_m
        dist_to_subgoal_m = float(math.hypot(goal_dx, goal_dy))

        if dist_to_subgoal_m > 1e-6:
            goal_ux = goal_dx / dist_to_subgoal_m
            goal_uy = goal_dy / dist_to_subgoal_m
        else:
            goal_ux = 0.0
            goal_uy = 0.0

        if is_final_goal:
            if slow_radius_m > 1e-6:
                speed_scale = 1.0 if dist_to_goal_m >= slow_radius_m else smoothstep01(dist_to_goal_m / slow_radius_m)
            else:
                speed_scale = 1.0
        else:
            turn_radius_m = max(8.0, planner_clearance_m * 2.5)
            if dist_to_subgoal_m >= turn_radius_m:
                speed_scale = 1.0
            else:
                speed_scale = 0.55 + 0.45 * smoothstep01(dist_to_subgoal_m / turn_radius_m)
        desired_speed_mps = max_speed_mps * speed_scale

        if is_final_goal:
            if descent_radius_m > 1e-6 and dist_to_goal_m < descent_radius_m:
                alt_blend = smoothstep01(1.0 - dist_to_goal_m / descent_radius_m)
            elif dist_to_goal_m <= arrival_radius_m:
                alt_blend = 1.0
            else:
                alt_blend = 0.0
            z_target_m = transit_alt_m + (dest_z_m - transit_alt_m) * alt_blend
            if dist_to_goal_m <= arrival_radius_m:
                z_target_m = dest_z_m
        else:
            z_target_m = transit_alt_m

        fx_safety, fy_safety, nearest_obstacle_m, active_safety, threat_scale = mission_safety_force(st.x_m, st.y_m, obstacles)
        fxy_budget_n, guard = guidance_force_budget(power_state, state, geom, power, fxy_max_n, fault=fault)
        budget_ratio = float(guard["budget_ratio"])
        speed_guard_scale = 0.38 + 0.62 * smoothstep01((budget_ratio - 0.25) / 0.75)
        gain_guard_scale = 0.32 + 0.68 * smoothstep01((budget_ratio - 0.25) / 0.75)
        power_pressure = smoothstep01((guard["continuous_power_ratio"] - 0.90) / 0.12)
        speed_guard_scale *= 1.0 - 0.18 * power_pressure
        gain_guard_scale *= 1.0 - 0.24 * power_pressure
        speed_guard_scale *= float(guard["fault_guard_scale"])
        gain_guard_scale *= float(guard["fault_guard_scale"])
        desired_speed_mps *= max(0.45, 1.0 - 0.45 * threat_scale)
        desired_speed_mps *= speed_guard_scale
        desired_vx_mps = desired_speed_mps * goal_ux
        desired_vy_mps = desired_speed_mps * goal_uy
        goal_scale = max(0.45, 1.0 - 0.55 * threat_scale) * gain_guard_scale
        safety_mag_n = float(math.hypot(fx_safety, fy_safety))
        goal_budget_n = max(0.25 * fxy_budget_n, fxy_budget_n - 0.60 * safety_mag_n)
        estimated_state = EstimatedVehicleState(
            x_m=st.x_m,
            y_m=st.y_m,
            z_m=st.z_m,
            vx_mps=st.vx_mps,
            vy_mps=st.vy_mps,
            vz_mps=st.vz_mps,
            yaw_deg=st.yaw_deg,
            yaw_rate_deg_s=st.yaw_rate_deg_s,
            battery_soc_pct=100.0 * power_state.soc_frac,
            bus_voltage_v=power_state.voltage_v,
            continuous_power_ratio=float(guard["continuous_power_ratio"]),
            thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
            fault_available_scale=float(guard["fault_available_scale"]),
            fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
        )
        guidance_target = GuidanceTarget(
            goal_x_m=goal_x_m,
            goal_y_m=goal_y_m,
            goal_z_m=z_target_m,
            desired_vx_mps=desired_vx_mps,
            desired_vy_mps=desired_vy_mps,
            desired_vz_mps=0.0,
            hold_yaw_deg=yaw_hold_deg,
            mode="route-track",
            force_scale=goal_scale,
        )
        actuator_health = ActuatorHealthState(
            lateral_budget_n=goal_budget_n,
            guard_scale=float(guard["guard_scale"]),
            response_scale=float(guard["fault_response_scale"]),
            continuous_power_ratio=float(guard["continuous_power_ratio"]),
            thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
            supply_scale_pct=100.0 * float(guard["supply_guard_scale"]),
            fault_available_scale=float(guard["fault_available_scale"]),
            fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
        )
        goal_control = track_xy_position(
            estimated_state,
            guidance_target,
            controller_gains,
            actuator_health,
            fz_n=fz_cmd,
            mz_nm=mz_nm,
            source="coordinate-mission.track-route",
        )
        fx_goal = goal_control.desired_wrench.fx_n
        fy_goal = goal_control.desired_wrench.fy_n
        goal_force_scale = float(np.clip(goal_control.clip_scale, 0.30, 1.0))
        fx_raw, fy_raw = clip_force_xy(fx_goal + fx_safety, fy_goal + fy_safety, fxy_budget_n)
        command_rate_active_n_s = command_rate_n_s * (0.50 + 0.50 * min(speed_guard_scale, goal_force_scale))
        command_rate_active_n_s *= float(guard["fault_response_scale"])
        fx_cmd, fy_cmd = rate_limit_xy_force(command_fx_prev, command_fy_prev, fx_raw, fy_raw, command_rate_active_n_s, sim.dt_s)
        command_fx_prev, command_fy_prev = fx_cmd, fy_cmd

        arrived_now = is_final_goal and dist_to_goal_m <= arrival_radius_m and abs(st.z_m - dest_z_m) <= 0.5 and speed_mps <= 0.35
        if arrived_now:
            fx_cmd = 0.0
            fy_cmd = 0.0
            command_fx_prev = 0.0
            command_fy_prev = 0.0
            if hold_start_s is None:
                hold_start_s = t
            if arrival_time_s is None:
                arrival_time_s = t
        else:
            hold_start_s = None

        if arrived_now:
            phase = "hold"
        elif not is_final_goal:
            phase = "bypass"
        elif active_safety > 0:
            phase = "safety"
        elif dist_to_goal_m <= descent_radius_m:
            phase = "approach"
        elif z_target_m > st.z_m + 0.5:
            phase = "climb"
        else:
            phase = "cruise"

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm), fault=fault, topology=topology, effectiveness=effectiveness)
        alpha_target = apply_command_faults_to_alpha(alloc.alpha_rad, fault)
        thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault, topology=topology)
        ft_target = alloc.ft_tan_per_seg_n
        state = apply_actuator_limits(state, alpha_target, ft_target, lim, sim.dt_s, fault=fault)

        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)
        alpha_actual = apply_faults_to_alpha(state.alpha_rad, fault)

        thrust_cmd = thrust_target
        thrust_pre_power = thrust_cmd * state.plenum_scale
        thrust, power_state, telemetry = apply_power_system(
            thrust_cmd,
            thrust_pre_power,
            alpha_target,
            alpha_actual,
            state.ft_tan_per_seg_n,
            sim,
            geom,
            power,
            power_state,
            sim.dt_s,
        )
        hist["fan_thrust_16"].append(list(telemetry["fan_actual_16"]))

        forces_main = thrust_vectors_body(geom, thrust, alpha_actual, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
        net, mz_est = net_force_and_yaw_moment(geom, forces)

        step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim, z_ref_m=z_target_m, vz_ref_mps=0.0)
        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m); hist["z"].append(st.z_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["vz"].append(st.vz_mps)
        hist["speed"].append(speed)
        hist["yaw_deg"].append(st.yaw_deg); hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alpha_actual) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        hist["alpha_deg_32"].append(list(np.degrees(alpha_actual)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["fx_cmd"].append(float(fx_cmd)); hist["fy_cmd"].append(float(fy_cmd))
        hist["dist_to_goal"].append(dist_to_goal_m)
        hist["dist_to_subgoal"].append(dist_to_subgoal_m)
        hist["nearest_obstacle_m"].append(None if nearest_obstacle_m is None else float(nearest_obstacle_m))
        hist["avoid_fx"].append(float(fx_safety)); hist["avoid_fy"].append(float(fy_safety))
        hist["target_z_m"].append(float(z_target_m))
        hist["mission_phase"].append(phase)
        hist["subgoal_x_m"].append(float(goal_x_m)); hist["subgoal_y_m"].append(float(goal_y_m))
        hist["route_goal_idx"].append(int(goal_idx))
        hist["guard_scale"].append(float(guard["guard_scale"]))
        hist["flap_guard_scale"].append(float(guard["flap_guard_scale"]))
        hist["power_guard_scale"].append(float(guard["power_guard_scale"]))
        hist["thermal_guard_scale"].append(float(guard["thermal_guard_scale"]))
        hist["supply_guard_scale"].append(float(guard["supply_guard_scale"]))
        hist["fault_guard_scale"].append(float(guard["fault_guard_scale"]))
        hist["fault_response_scale"].append(float(guard["fault_response_scale"]))
        hist["fault_available_scale"].append(float(guard["fault_available_scale"]))
        hist["fault_asymmetry_pct"].append(float(guard["fault_asymmetry_pct"]))
        hist["fxy_budget_n"].append(float(fxy_budget_n))
        hist["budget_ratio"].append(float(guard["budget_ratio"]))
        hist["speed_guard_scale"].append(float(speed_guard_scale))
        hist["gain_guard_scale"].append(float(gain_guard_scale))
        hist["goal_force_scale"].append(float(goal_force_scale))
        hist["continuous_power_ratio"].append(float(guard["continuous_power_ratio"]))
        hist["flap_usage_ratio"].append(float(guard["flap_usage_ratio"]))
        append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net)

        if hold_start_s is not None and (t - hold_start_s) >= hold_s:
            break

    x_arr = np.array(hist["x"], float)
    y_arr = np.array(hist["y"], float)
    z_arr = np.array(hist["z"], float)
    vx_arr = np.array(hist["vx"], float)
    vy_arr = np.array(hist["vy"], float)
    sp_arr = np.array(hist["speed"], float)
    final_goal_error_m = float(math.hypot(dest_x_m - x_arr[-1], dest_y_m - y_arr[-1]))
    final_alt_error_m = float(abs(dest_z_m - z_arr[-1]))
    path_length_m = float(np.sum(np.hypot(np.diff(x_arr), np.diff(y_arr)))) if len(x_arr) > 1 else 0.0
    clearances = [float(v) for v in hist["nearest_obstacle_m"] if v is not None]
    min_clearance_m = min(clearances) if clearances else None
    hold_complete = bool(arrival_time_s is not None and hist["t"][-1] - arrival_time_s >= hold_s)

    out = {
        "version": "v5-coordinates",
        "mission": {
            "start": {"x_m": start_x_m, "y_m": start_y_m, "z_m": start_z_m},
            "destination": {"x_m": dest_x_m, "y_m": dest_y_m, "z_m": dest_z_m},
            "total_s": total_s,
            "yaw_hold_deg": yaw_hold_deg,
            "mz_nm_cmd": mz_nm,
            "cruise_alt_m": cruise_alt_m,
            "hold_s": hold_s,
        },
        "guidance": {
            "max_speed_mps": max_speed_mps,
            "fxy_max_n": fxy_max_n,
            "arrival_radius_m": arrival_radius_m,
            "slow_radius_m": slow_radius_m,
            "descent_radius_m": descent_radius_m,
            "pos_k_n_per_m": pos_k_n_per_m,
            "vel_k_n_per_mps": vel_k_n_per_mps,
            "planner_clearance_m": planner_clearance_m,
            "command_rate_n_s": command_rate_n_s,
            "guard_floor_scale": 0.25,
            "guard_mode": "budget-aware-speed-shaping",
        },
        "planner": {
            "waypoints": [{"x_m": float(x), "y_m": float(y)} for x, y in route_xy],
            "num_bypass_waypoints": max(0, len(route_xy) - 2),
        },
        "obstacles": [obs.__dict__ for obs in obstacles],
        "sim": {
            "dt_s": sim.dt_s,
            "mass_kg": sim.mass_kg,
            "drag_coeff": sim.drag_coeff,
            "gravity": sim.gravity,
            "yaw_inertia_kg_m2": sim.yaw_inertia_kg_m2,
            "yaw_damping_nm_per_rad_s": sim.yaw_damping_nm_per_rad_s,
            "z_drag_coeff": sim.z_drag_coeff,
            "z_hold_kp_n_per_m": sim.z_hold_kp_n_per_m,
            "z_hold_kd_n_per_mps": sim.z_hold_kd_n_per_mps,
        },
        "limits": lim.__dict__,
        "plenum": pl.__dict__,
        "power": power.__dict__,
        "fault": {
            "stuck_flap_idx": fault.stuck_flap_idx,
            "stuck_flap_alpha_deg": fault.stuck_flap_alpha_deg,
            "dead_fan_group": fault.dead_fan_group,
            "dead_fan_scale": fault.dead_fan_scale,
            "plenum_sector_idx": fault.plenum_sector_idx,
            "plenum_sector_scale": fault.plenum_sector_scale,
        },
        "arrival": {
            "arrived": bool(arrival_time_s is not None),
            "arrival_time_s": arrival_time_s,
            "hold_complete": hold_complete,
            "final_goal_error_m": final_goal_error_m,
            "final_alt_error_m": final_alt_error_m,
            "min_obstacle_clearance_m": min_clearance_m,
            "path_length_m": path_length_m,
            "peak_speed_mps": float(np.max(sp_arr)) if sp_arr.size else None,
        },
        "final": {
            "x_m": float(x_arr[-1]),
            "y_m": float(y_arr[-1]),
            "z_m": float(z_arr[-1]),
            "vx_mps": float(vx_arr[-1]),
            "vy_mps": float(vy_arr[-1]),
            "vz_mps": float(hist["vz"][-1]),
            "yaw_deg": float(hist["yaw_deg"][-1]),
            "yaw_rate_deg_s": float(hist["yaw_rate_deg_s"][-1]),
            "alpha_deg_rms": float(hist["alpha_deg_rms"][-1]),
            "ft_tan_rms": float(hist["ft_tan_rms"][-1]),
            "mz_est_nm": float(hist["mz_est"][-1]),
        },
        "hardware_assumptions": hardware_assumptions_payload(topology, effectiveness),
    }
    return out, hist


def run_repel_test_v4(obstacle_x_m: float = 30.0, obstacle_y_m: float = 0.0, initial_x_m: float = 0.0, initial_y_m: float = 0.0, initial_vx_mps: float = 1.0, initial_vy_mps: float = 0.0, yaw_hold_deg: float = 0.0, total_s: float = 12.0, mz_nm: float = 0.0,
                      field: RepelField | None = None, fault: FaultSpec | None = None, trace_out: str | None = None, geom=None, sim=None, lim: ActuatorLimits | None = None, pl: PlenumModel | None = None):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    field = field or RepelField()
    fault = fault or FaultSpec()

    st = SimState(x_m=initial_x_m, y_m=initial_y_m, vx_mps=initial_vx_mps, vy_mps=initial_vy_mps, yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    fz_cmd = sim.mass_kg * sim.gravity
    theta = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)

    hist = {"t": [], "x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": [], "speed": [], "yaw_deg": [], "yaw_rate_deg_s": [], "fx_cmd": [], "fy_cmd": [], "dist_to_obstacle": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [], "alpha_deg_32": [], "ft_tan_32": [], "faults": [], "fan_thrust_16": []}

    for k in range(steps):
        t = k * sim.dt_s
        dist = float(math.hypot(st.x_m - obstacle_x_m, st.y_m - obstacle_y_m))
        fx_cmd, fy_cmd = repel_force_xy(field, st.x_m, st.y_m, obstacle_x_m, obstacle_y_m)

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm), fault=fault, topology=topology, effectiveness=effectiveness)
        alpha_target = apply_command_faults_to_alpha(alloc.alpha_rad, fault)
        thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault, topology=topology)
        ft_target = alloc.ft_tan_per_seg_n

        state = apply_actuator_limits(state, alpha_target, ft_target, lim, sim.dt_s, fault=fault)
        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)
        alpha_actual = apply_faults_to_alpha(state.alpha_rad, fault)

        thrust_cmd = thrust_target
        thrust_pre_power = thrust_cmd * state.plenum_scale
        thrust, power_state, telemetry = apply_power_system(
            thrust_cmd,
            thrust_pre_power,
            alpha_target,
            alpha_actual,
            state.ft_tan_per_seg_n,
            sim,
            geom,
            power,
            power_state,
            sim.dt_s,
        )
        hist["fan_thrust_16"].append(list(telemetry["fan_actual_16"]))
        forces_main = thrust_vectors_body(geom, thrust, alpha_actual, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
        net, mz_est = net_force_and_yaw_moment(geom, forces)

        step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim)
        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m); hist["z"].append(st.z_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["vz"].append(st.vz_mps); hist["speed"].append(speed)
        hist["yaw_deg"].append(st.yaw_deg); hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
        hist["fx_cmd"].append(fx_cmd); hist["fy_cmd"].append(fy_cmd)
        hist["dist_to_obstacle"].append(dist)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alpha_actual) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        # per-segment arrays for visualization
        hist["alpha_deg_32"].append(list(np.degrees(alpha_actual)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["faults"].append({"stuck_flap_idx": fault.stuck_flap_idx, "dead_fan_group": fault.dead_fan_group, "plenum_sector_idx": fault.plenum_sector_idx})
        append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net)

    coupling = yaw_track_coupling_mean_abs(hist)
    meta = {"version": "v4", "obstacle": {"x_m": obstacle_x_m, "y_m": obstacle_y_m}, "field": field.__dict__, "fault": fault.__dict__,
            "yaw_track_coupling_mean_abs_deg": coupling, "sim": {"dt_s": sim.dt_s, "mass_kg": sim.mass_kg, "drag_coeff": sim.drag_coeff, "gravity": sim.gravity, "yaw_inertia_kg_m2": sim.yaw_inertia_kg_m2, "yaw_damping_nm_per_rad_s": sim.yaw_damping_nm_per_rad_s, "z_drag_coeff": sim.z_drag_coeff, "z_hold_kp_n_per_m": sim.z_hold_kp_n_per_m, "z_hold_kd_n_per_mps": sim.z_hold_kd_n_per_mps}, "limits": lim.__dict__, "plenum": pl.__dict__, "power": power.__dict__, "hardware_assumptions": hardware_assumptions_payload(topology, effectiveness)}
    if trace_out:
        save_trace_json(trace_out, meta=meta, hist=hist)

    dist_arr = np.array(hist["dist_to_obstacle"], float)
    t_arr = np.array(hist["t"], float)
    sp_arr = np.array(hist["speed"], float)

    # --- Gate-style radial metrics (Gate D) ---
    # Radial velocity: positive means moving away from obstacle.
    x_arr = np.array(hist["x"], float)
    y_arr = np.array(hist["y"], float)
    vx_arr = np.array(hist["vx"], float)
    vy_arr = np.array(hist["vy"], float)

    # unit vector from obstacle -> craft
    dx_arr = x_arr - obstacle_x_m
    dy_arr = y_arr - obstacle_y_m
    dist_arr = np.hypot(dx_arr, dy_arr)
    dist_safe = np.maximum(dist_arr, 1e-6)

    ux = dx_arr / dist_safe
    uy = dy_arr / dist_safe

    v_rad = vx_arr * ux + vy_arr * uy  # radial velocity (m/s)

    # entry index
    inside = np.where(dist_arr <= field.radius_m)[0]
    enter_idx = int(inside[0]) if inside.size else None
    t_enter = float(t_arr[enter_idx]) if enter_idx is not None else None

    v_rad_enter = float(v_rad[enter_idx]) if enter_idx is not None else None

    # recede: first time after entry where v_rad > 0
    t_recede = None
    recede_idx = None
    if enter_idx is not None:
        for i in range(enter_idx, len(v_rad)):
            if float(v_rad[i]) > 0.0:
                t_recede = float(t_arr[i])
                recede_idx = i
                break

    # peak outward radial acceleration after entry
    a_rad_peak = None
    if enter_idx is not None and len(v_rad) >= enter_idx + 3:
        dt = np.diff(t_arr)
        dv = np.diff(v_rad)
        good = dt > 1e-6
        a = np.full_like(dv, np.nan, dtype=float)
        a[good] = dv[good] / dt[good]
        # take peak outward accel after entry
        a_after = a[enter_idx:]
        if np.isfinite(np.nanmax(a_after)):
            a_rad_peak = float(np.nanmax(a_after))

    # time to outward radial speed threshold (e.g. +0.5 m/s)
    t_to_vrad_away_0p5 = None
    if enter_idx is not None:
        thr = 0.5
        for i in range(enter_idx, len(v_rad)):
            if float(v_rad[i]) >= thr:
                t_to_vrad_away_0p5 = float(t_arr[i] - t_arr[enter_idx])
                break

    # old speed-difference based response (kept for compatibility)
    t_resp = None
    if t_enter is not None:
        v0 = float(sp_arr[0])
        for i in range(inside[0], len(sp_arr)):
            if abs(float(sp_arr[i]) - v0) >= 0.3:
                t_resp = float(t_arr[i])
                break

    latency = None
    if t_enter is not None and t_recede is not None:
        latency = float(t_recede - t_enter)

    out = {"meta": meta, "headline": {"enter_radius_time_s": t_enter,
                                     "response_time_s": t_resp,
                                     "response_latency_s": (None if (t_enter is None or t_resp is None) else float(t_resp - t_enter)),
                                     "recede_time_s": t_recede,
                                     "recede_latency_s": latency,
                                     # Gate-style additions:
                                     "v_rad_enter_mps": v_rad_enter,
                                     "a_rad_peak_mps2": a_rad_peak,
                                     "t_to_vrad_away_0p5_s": t_to_vrad_away_0p5,
                                     "repel_speed_mps_median": float(np.median(sp_arr)) if sp_arr.size else None,
                                     "repel_speed_mps_max": float(np.max(sp_arr)) if sp_arr.size else None,
                                     "yaw_track_coupling_mean_abs_deg": coupling}}
    return out, hist