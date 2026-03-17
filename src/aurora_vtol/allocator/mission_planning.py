from __future__ import annotations

from dataclasses import dataclass
import math

from .field import RepelField, repel_force_xy


def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class MissionObstacle:
    x_m: float
    y_m: float
    radius_m: float = 12.0
    k_n_per_m: float = 180.0
    fxy_max_n: float = 2500.0
    swirl_n: float = 900.0
    influence_m: float = 20.0



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
        max_threat = max(max_threat, _smoothstep01(threat))

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
    flap_guard = 1.0 - 0.60 * _smoothstep01((flap_use - 0.60) / 0.22)
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
    power_guard = 1.0 - 0.62 * _smoothstep01((continuous_ratio - 0.80) / 0.18)
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
    supply_scale = power.min_supply_scale + (1.0 - power.min_supply_scale) * _smoothstep01((float(power_state.voltage_v) - power.battery_empty_v) / span_v)
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
    supply_scale_next = power.min_supply_scale + (1.0 - power.min_supply_scale) * _smoothstep01((loaded_voltage_v - power.battery_empty_v) / span_v)
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


