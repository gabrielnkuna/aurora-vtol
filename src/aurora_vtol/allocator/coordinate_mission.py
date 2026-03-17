from __future__ import annotations

import math

from .coordinate_execution import append_coordinate_step_history, execute_coordinate_step
from .coordinate_support import (
    advance_route_goal,
    build_coordinate_history,
    build_coordinate_output,
    classify_coordinate_phase,
    compute_coordinate_guidance_command,
    resolve_route_goal,
)
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel
from .faults import FaultSpec
from .mission_planning import MissionObstacle, mission_safety_force, plan_route_waypoints
from .model import RingGeometry, segment_angles_rad
from .power_system import PowerSystemParams, guidance_force_budget, init_hover_power_state
from .sim_runtime import SimParams, SimState
from ..effectiveness import effectiveness_table_for_topology, hardware_assumptions_payload
from ..topology import default_ring_topology
from ..vehicle_controller import XYVehicleControllerGains


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
    allocator_state = AllocatorState.init(geom.n_segments)

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

    hist = build_coordinate_history()
    arrival_time_s = None
    hold_start_s = None

    for k in range(steps):
        t = k * sim.dt_s
        dist_to_goal_m = float(math.hypot(dest_x_m - st.x_m, dest_y_m - st.y_m))
        speed_mps = float(math.hypot(st.vx_mps, st.vy_mps))

        goal_idx = advance_route_goal(st.x_m, st.y_m, route_xy, goal_idx, planner_clearance_m)
        route_goal = resolve_route_goal(st.x_m, st.y_m, route_xy, goal_idx)

        fx_safety, fy_safety, nearest_obstacle_m, active_safety, threat_scale = mission_safety_force(
            st.x_m,
            st.y_m,
            obstacles,
        )
        fxy_budget_n, guard = guidance_force_budget(power_state, allocator_state, geom, power, fxy_max_n, fault=fault)
        guard["soc_frac"] = power_state.soc_frac
        guard["bus_voltage_v"] = power_state.voltage_v
        guidance = compute_coordinate_guidance_command(
            st=st,
            route_goal=route_goal,
            dist_to_goal_m=dist_to_goal_m,
            transit_alt_m=transit_alt_m,
            dest_z_m=dest_z_m,
            descent_radius_m=descent_radius_m,
            arrival_radius_m=arrival_radius_m,
            max_speed_mps=max_speed_mps,
            slow_radius_m=slow_radius_m,
            planner_clearance_m=planner_clearance_m,
            threat_scale=threat_scale,
            fxy_budget_n=fxy_budget_n,
            guard=guard,
            fx_safety=fx_safety,
            fy_safety=fy_safety,
            yaw_hold_deg=yaw_hold_deg,
            mz_nm=mz_nm,
            controller_gains=controller_gains,
            fz_cmd=fz_cmd,
            command_fx_prev=command_fx_prev,
            command_fy_prev=command_fy_prev,
            command_rate_n_s=command_rate_n_s,
            dt_s=sim.dt_s,
        )
        fx_cmd = guidance.fx_cmd
        fy_cmd = guidance.fy_cmd
        command_fx_prev = fx_cmd
        command_fy_prev = fy_cmd
        z_target_m = guidance.z_target_m

        arrived_now = (
            route_goal.is_final_goal
            and dist_to_goal_m <= arrival_radius_m
            and abs(st.z_m - dest_z_m) <= 0.5
            and speed_mps <= 0.35
        )
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
        else:
            phase = classify_coordinate_phase(
                route_goal,
                dist_to_goal_m,
                descent_radius_m,
                z_target_m,
                st.z_m,
                active_safety,
            )

        execution = execute_coordinate_step(
            st=st,
            allocator_state=allocator_state,
            geom=geom,
            sim=sim,
            lim=lim,
            pl=pl,
            power=power,
            power_state=power_state,
            fault=fault,
            topology=topology,
            effectiveness=effectiveness,
            theta_rad=theta,
            fx_cmd=fx_cmd,
            fy_cmd=fy_cmd,
            fz_cmd=fz_cmd,
            mz_nm=mz_nm,
            z_target_m=z_target_m,
        )
        allocator_state = execution.allocator_state
        power_state = execution.power_state
        append_coordinate_step_history(
            hist=hist,
            t_s=t,
            st=st,
            route_goal=route_goal,
            dist_to_goal_m=dist_to_goal_m,
            nearest_obstacle_m=nearest_obstacle_m,
            fx_safety=fx_safety,
            fy_safety=fy_safety,
            z_target_m=z_target_m,
            phase=phase,
            guard=guard,
            fxy_budget_n=fxy_budget_n,
            guidance=guidance,
            fx_cmd=fx_cmd,
            fy_cmd=fy_cmd,
            fz_cmd=fz_cmd,
            execution=execution,
        )

        if hold_start_s is not None and (t - hold_start_s) >= hold_s:
            break

    out = build_coordinate_output(
        hist=hist,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        start_z_m=start_z_m,
        dest_x_m=dest_x_m,
        dest_y_m=dest_y_m,
        dest_z_m=dest_z_m,
        total_s=total_s,
        yaw_hold_deg=yaw_hold_deg,
        mz_nm=mz_nm,
        cruise_alt_m=cruise_alt_m,
        hold_s=hold_s,
        max_speed_mps=max_speed_mps,
        fxy_max_n=fxy_max_n,
        arrival_radius_m=arrival_radius_m,
        slow_radius_m=slow_radius_m,
        descent_radius_m=descent_radius_m,
        pos_k_n_per_m=pos_k_n_per_m,
        vel_k_n_per_mps=vel_k_n_per_mps,
        planner_clearance_m=planner_clearance_m,
        command_rate_n_s=command_rate_n_s,
        route_xy=route_xy,
        obstacles=obstacles,
        sim=sim,
        lim=lim,
        pl=pl,
        power=power,
        fault=fault,
        arrival_time_s=arrival_time_s,
        hardware_assumptions=hardware_assumptions_payload(topology, effectiveness),
    )
    return out, hist
