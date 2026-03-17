from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .power_system import smoothstep01
from .sim_runtime import clip_force_xy, rate_limit_xy_force
from ..icd import ActuatorHealthState, EstimatedVehicleState, GuidanceTarget
from ..vehicle_controller import XYVehicleControllerGains, track_xy_position


@dataclass(frozen=True)
class CoordinateRouteGoal:
    goal_idx: int
    goal_x_m: float
    goal_y_m: float
    is_final_goal: bool
    dist_to_subgoal_m: float
    goal_ux: float
    goal_uy: float


@dataclass(frozen=True)
class CoordinateGuidanceCommand:
    desired_speed_mps: float
    z_target_m: float
    speed_guard_scale: float
    gain_guard_scale: float
    goal_force_scale: float
    goal_budget_n: float
    fx_cmd: float
    fy_cmd: float
    desired_vx_mps: float
    desired_vy_mps: float


def build_coordinate_history() -> dict[str, list]:
    return {
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


def advance_route_goal(
    x_m: float,
    y_m: float,
    route_xy: list[tuple[float, float]],
    goal_idx: int,
    planner_clearance_m: float,
) -> int:
    subgoal_radius_m = max(3.0, planner_clearance_m * 0.9)
    while goal_idx < len(route_xy) - 1:
        subgoal_x_m, subgoal_y_m = route_xy[goal_idx]
        subgoal_dist_m = float(math.hypot(subgoal_x_m - x_m, subgoal_y_m - y_m))
        if subgoal_dist_m <= subgoal_radius_m:
            goal_idx += 1
        else:
            break
    return goal_idx


def resolve_route_goal(
    x_m: float,
    y_m: float,
    route_xy: list[tuple[float, float]],
    goal_idx: int,
) -> CoordinateRouteGoal:
    goal_x_m, goal_y_m = route_xy[goal_idx]
    goal_dx = goal_x_m - x_m
    goal_dy = goal_y_m - y_m
    dist_to_subgoal_m = float(math.hypot(goal_dx, goal_dy))
    if dist_to_subgoal_m > 1e-6:
        goal_ux = goal_dx / dist_to_subgoal_m
        goal_uy = goal_dy / dist_to_subgoal_m
    else:
        goal_ux = 0.0
        goal_uy = 0.0
    return CoordinateRouteGoal(
        goal_idx=goal_idx,
        goal_x_m=goal_x_m,
        goal_y_m=goal_y_m,
        is_final_goal=goal_idx == len(route_xy) - 1,
        dist_to_subgoal_m=dist_to_subgoal_m,
        goal_ux=goal_ux,
        goal_uy=goal_uy,
    )


def compute_desired_speed_mps(
    max_speed_mps: float,
    slow_radius_m: float,
    planner_clearance_m: float,
    dist_to_goal_m: float,
    route_goal: CoordinateRouteGoal,
) -> float:
    if route_goal.is_final_goal:
        if slow_radius_m > 1e-6:
            speed_scale = 1.0 if dist_to_goal_m >= slow_radius_m else smoothstep01(dist_to_goal_m / slow_radius_m)
        else:
            speed_scale = 1.0
    else:
        turn_radius_m = max(8.0, planner_clearance_m * 2.5)
        if route_goal.dist_to_subgoal_m >= turn_radius_m:
            speed_scale = 1.0
        else:
            speed_scale = 0.55 + 0.45 * smoothstep01(route_goal.dist_to_subgoal_m / turn_radius_m)
    return max_speed_mps * speed_scale


def compute_target_altitude_m(
    transit_alt_m: float,
    dest_z_m: float,
    descent_radius_m: float,
    arrival_radius_m: float,
    dist_to_goal_m: float,
    is_final_goal: bool,
) -> float:
    if not is_final_goal:
        return transit_alt_m
    if descent_radius_m > 1e-6 and dist_to_goal_m < descent_radius_m:
        alt_blend = smoothstep01(1.0 - dist_to_goal_m / descent_radius_m)
    elif dist_to_goal_m <= arrival_radius_m:
        alt_blend = 1.0
    else:
        alt_blend = 0.0
    z_target_m = transit_alt_m + (dest_z_m - transit_alt_m) * alt_blend
    if dist_to_goal_m <= arrival_radius_m:
        z_target_m = dest_z_m
    return z_target_m

def apply_coordinate_guard_scaling(
    desired_speed_mps: float,
    guard: dict[str, float],
    threat_scale: float,
) -> tuple[float, float, float]:
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
    return desired_speed_mps, speed_guard_scale, gain_guard_scale


def compute_coordinate_guidance_command(
    *,
    st,
    route_goal: CoordinateRouteGoal,
    dist_to_goal_m: float,
    transit_alt_m: float,
    dest_z_m: float,
    descent_radius_m: float,
    arrival_radius_m: float,
    max_speed_mps: float,
    slow_radius_m: float,
    planner_clearance_m: float,
    threat_scale: float,
    fxy_budget_n: float,
    guard: dict[str, float],
    fx_safety: float,
    fy_safety: float,
    yaw_hold_deg: float,
    mz_nm: float,
    controller_gains: XYVehicleControllerGains,
    fz_cmd: float,
    command_fx_prev: float,
    command_fy_prev: float,
    command_rate_n_s: float,
    dt_s: float,
) -> CoordinateGuidanceCommand:
    desired_speed_mps = compute_desired_speed_mps(
        max_speed_mps,
        slow_radius_m,
        planner_clearance_m,
        dist_to_goal_m,
        route_goal,
    )
    z_target_m = compute_target_altitude_m(
        transit_alt_m,
        dest_z_m,
        descent_radius_m,
        arrival_radius_m,
        dist_to_goal_m,
        route_goal.is_final_goal,
    )
    desired_speed_mps, speed_guard_scale, gain_guard_scale = apply_coordinate_guard_scaling(
        desired_speed_mps,
        guard,
        threat_scale,
    )
    desired_vx_mps = desired_speed_mps * route_goal.goal_ux
    desired_vy_mps = desired_speed_mps * route_goal.goal_uy
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
        battery_soc_pct=100.0 * float(guard.get("soc_frac", 1.0)),
        bus_voltage_v=float(guard.get("bus_voltage_v", 0.0)),
        continuous_power_ratio=float(guard["continuous_power_ratio"]),
        thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
        fault_available_scale=float(guard["fault_available_scale"]),
        fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
    )
    guidance_target = GuidanceTarget(
        goal_x_m=route_goal.goal_x_m,
        goal_y_m=route_goal.goal_y_m,
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
    fx_cmd, fy_cmd = rate_limit_xy_force(
        command_fx_prev,
        command_fy_prev,
        fx_raw,
        fy_raw,
        command_rate_active_n_s,
        dt_s,
    )
    return CoordinateGuidanceCommand(
        desired_speed_mps=desired_speed_mps,
        z_target_m=z_target_m,
        speed_guard_scale=speed_guard_scale,
        gain_guard_scale=gain_guard_scale,
        goal_force_scale=goal_force_scale,
        goal_budget_n=goal_budget_n,
        fx_cmd=fx_cmd,
        fy_cmd=fy_cmd,
        desired_vx_mps=desired_vx_mps,
        desired_vy_mps=desired_vy_mps,
    )

def classify_coordinate_phase(
    route_goal: CoordinateRouteGoal,
    dist_to_goal_m: float,
    descent_radius_m: float,
    z_target_m: float,
    z_m: float,
    active_safety: float,
) -> str:
    if not route_goal.is_final_goal:
        return "bypass"
    if active_safety > 0:
        return "safety"
    if dist_to_goal_m <= descent_radius_m:
        return "approach"
    if z_target_m > z_m + 0.5:
        return "climb"
    return "cruise"


def build_coordinate_output(
    *,
    hist: dict[str, list],
    start_x_m: float,
    start_y_m: float,
    start_z_m: float,
    dest_x_m: float,
    dest_y_m: float,
    dest_z_m: float,
    total_s: float,
    yaw_hold_deg: float,
    mz_nm: float,
    cruise_alt_m: float,
    hold_s: float,
    max_speed_mps: float,
    fxy_max_n: float,
    arrival_radius_m: float,
    slow_radius_m: float,
    descent_radius_m: float,
    pos_k_n_per_m: float,
    vel_k_n_per_mps: float,
    planner_clearance_m: float,
    command_rate_n_s: float,
    route_xy: list[tuple[float, float]],
    obstacles: list,
    sim,
    lim,
    pl,
    power,
    fault,
    arrival_time_s: float | None,
    hardware_assumptions: dict,
) -> dict:
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

    return {
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
        "hardware_assumptions": hardware_assumptions,
    }
