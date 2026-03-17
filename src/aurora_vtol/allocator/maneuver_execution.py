from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .allocate import AllocationRequest, allocate_v2
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel, apply_actuator_limits, apply_plenum_lag
from .faults import FaultSpec, apply_command_faults_to_alpha, apply_faults_to_alpha, apply_faults_to_thrust
from .model import net_force_and_yaw_moment, thrust_vectors_body
from .power_system import apply_power_system
from .sim_runtime import step_vehicle


@dataclass(frozen=True)
class ManeuverExecutionResult:
    allocator_state: AllocatorState
    power_state: object
    alpha_actual_rad: np.ndarray
    net_force_n: np.ndarray
    mz_est_nm: float
    telemetry: dict
    speed_mps: float


def execute_maneuver_step(
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
) -> ManeuverExecutionResult:
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

    step_vehicle(st, float(net[0]), float(net[1]), float(net[2]), float(mz_est), sim)
    speed_mps = float(math.hypot(st.vx_mps, st.vy_mps))

    return ManeuverExecutionResult(
        allocator_state=allocator_state,
        power_state=power_state,
        alpha_actual_rad=alpha_actual,
        net_force_n=net,
        mz_est_nm=float(mz_est),
        telemetry=telemetry,
        speed_mps=speed_mps,
    )
