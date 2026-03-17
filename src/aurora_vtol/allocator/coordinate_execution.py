from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .allocate import AllocationRequest, allocate_v2
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel, apply_actuator_limits, apply_plenum_lag
from .faults import FaultSpec, apply_command_faults_to_alpha, apply_faults_to_alpha, apply_faults_to_thrust
from .model import net_force_and_yaw_moment, thrust_vectors_body
from .power_system import apply_power_system
from .sim_runtime import append_engineering_telemetry, step_vehicle


@dataclass(frozen=True)
class CoordinateExecutionResult:
    allocator_state: AllocatorState
    power_state: object
    alpha_actual_rad: np.ndarray
    net_force_n: np.ndarray
    mz_est_nm: float
    telemetry: dict
    speed_mps: float


def execute_coordinate_step(
    *,
    st,
    allocator_state: AllocatorState,
    geom,
    sim,
    lim: ActuatorLimits,
    pl: PlenumModel,
    power,
    power_state,
    fault: FaultSpec,
    topology,
    effectiveness,
    theta_rad: np.ndarray,
    fx_cmd: float,
    fy_cmd: float,
    fz_cmd: float,
    mz_nm: float,
    z_target_m: float,
) -> CoordinateExecutionResult:
    alloc = allocate_v2(
        geom,
        AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm),
        fault=fault,
        topology=topology,
        effectiveness=effectiveness,
    )
    alpha_target = apply_command_faults_to_alpha(alloc.alpha_rad, fault)
    thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault, topology=topology)
    ft_target = alloc.ft_tan_per_seg_n
    allocator_state = apply_actuator_limits(allocator_state, alpha_target, ft_target, lim, sim.dt_s, fault=fault)

    demand = 1.0 + 0.20 * (np.abs(allocator_state.alpha_rad) / max(1e-6, np.max(np.abs(allocator_state.alpha_rad))))
    demand = np.clip(demand, pl.min_scale, pl.max_scale)
    allocator_state = apply_plenum_lag(allocator_state, demand, pl, sim.dt_s)
    alpha_actual = apply_faults_to_alpha(allocator_state.alpha_rad, fault)

    thrust_cmd = thrust_target
    thrust_pre_power = thrust_cmd * allocator_state.plenum_scale
    thrust, power_state, telemetry = apply_power_system(
        thrust_cmd,
        thrust_pre_power,
        alpha_target,
        alpha_actual,
        allocator_state.ft_tan_per_seg_n,
        sim,
        geom,
        power,
        power_state,
        sim.dt_s,
        topology=topology,
    )

    forces_main = thrust_vectors_body(geom, thrust, alpha_actual, theta_rad)
    fx_t = allocator_state.ft_tan_per_seg_n * (-np.sin(theta_rad))
    fy_t = allocator_state.ft_tan_per_seg_n * (np.cos(theta_rad))
    forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
    net, mz_est = net_force_and_yaw_moment(geom, forces)

    step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim, z_ref_m=z_target_m, vz_ref_mps=0.0)
    speed_mps = float(math.hypot(st.vx_mps, st.vy_mps))

    return CoordinateExecutionResult(
        allocator_state=allocator_state,
        power_state=power_state,
        alpha_actual_rad=alpha_actual,
        net_force_n=net,
        mz_est_nm=float(mz_est),
        telemetry=telemetry,
        speed_mps=speed_mps,
    )


def append_coordinate_step_history(
    *,
    hist: dict[str, list],
    t_s: float,
    st,
    route_goal,
    dist_to_goal_m: float,
    nearest_obstacle_m: float | None,
    fx_safety: float,
    fy_safety: float,
    z_target_m: float,
    phase: str,
    guard: dict[str, float],
    fxy_budget_n: float,
    guidance,
    fx_cmd: float,
    fy_cmd: float,
    fz_cmd: float,
    execution: CoordinateExecutionResult,
) -> None:
    hist["fan_thrust_16"].append(list(execution.telemetry["fan_actual_16"]))
    hist["t"].append(t_s)
    hist["x"].append(st.x_m)
    hist["y"].append(st.y_m)
    hist["z"].append(st.z_m)
    hist["vx"].append(st.vx_mps)
    hist["vy"].append(st.vy_mps)
    hist["vz"].append(st.vz_mps)
    hist["speed"].append(execution.speed_mps)
    hist["yaw_deg"].append(st.yaw_deg)
    hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
    hist["mz_est"].append(float(execution.mz_est_nm))
    hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(execution.alpha_actual_rad) ** 2))))
    hist["ft_tan_rms"].append(float(np.sqrt(np.mean(execution.allocator_state.ft_tan_per_seg_n ** 2))))
    hist["alpha_deg_32"].append(list(np.degrees(execution.alpha_actual_rad)))
    hist["ft_tan_32"].append(list(execution.allocator_state.ft_tan_per_seg_n))
    hist["fx_cmd"].append(float(fx_cmd))
    hist["fy_cmd"].append(float(fy_cmd))
    hist["dist_to_goal"].append(dist_to_goal_m)
    hist["dist_to_subgoal"].append(route_goal.dist_to_subgoal_m)
    hist["nearest_obstacle_m"].append(None if nearest_obstacle_m is None else float(nearest_obstacle_m))
    hist["avoid_fx"].append(float(fx_safety))
    hist["avoid_fy"].append(float(fy_safety))
    hist["target_z_m"].append(float(z_target_m))
    hist["mission_phase"].append(phase)
    hist["subgoal_x_m"].append(float(route_goal.goal_x_m))
    hist["subgoal_y_m"].append(float(route_goal.goal_y_m))
    hist["route_goal_idx"].append(int(route_goal.goal_idx))
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
    hist["speed_guard_scale"].append(float(guidance.speed_guard_scale))
    hist["gain_guard_scale"].append(float(guidance.gain_guard_scale))
    hist["goal_force_scale"].append(float(guidance.goal_force_scale))
    hist["continuous_power_ratio"].append(float(guard["continuous_power_ratio"]))
    hist["flap_usage_ratio"].append(float(guard["flap_usage_ratio"]))
    append_engineering_telemetry(hist, execution.telemetry, fx_cmd, fy_cmd, fz_cmd, execution.net_force_n)
