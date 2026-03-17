from __future__ import annotations

import math

from .coordinate_execution import append_coordinate_step_history, execute_coordinate_step
from .coordinate_support import (
    advance_route_goal,
    build_coordinate_mission_setup,
    build_coordinate_output,
    classify_coordinate_phase,
    compute_coordinate_guidance_command,
    resolve_coordinate_arrival_state,
    resolve_route_goal,
)
from .dynamics import ActuatorLimits, PlenumModel
from .faults import FaultSpec
from .mission_planning import MissionObstacle, mission_safety_force
from .power_system import PowerSystemParams, guidance_force_budget
from ..effectiveness import hardware_assumptions_payload


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
    setup = build_coordinate_mission_setup(
        dest_x_m=dest_x_m,
        dest_y_m=dest_y_m,
        dest_z_m=dest_z_m,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        start_z_m=start_z_m,
        total_s=total_s,
        yaw_hold_deg=yaw_hold_deg,
        cruise_alt_m=cruise_alt_m,
        arrival_radius_m=arrival_radius_m,
        pos_k_n_per_m=pos_k_n_per_m,
        vel_k_n_per_mps=vel_k_n_per_mps,
        obstacles=obstacles,
        geom=geom,
        sim=sim,
        lim=lim,
        pl=pl,
        power=power,
        fault=fault,
    )
    geom = setup.geom
    sim = setup.sim
    power = setup.power
    power_state = setup.power_state
    lim = setup.lim
    pl = setup.pl
    obstacles = setup.obstacles
    fault = setup.fault
    st = setup.st
    allocator_state = setup.allocator_state
    theta = setup.theta_rad
    topology = setup.topology
    effectiveness = setup.effectiveness
    fz_cmd = setup.fz_cmd
    transit_alt_m = setup.transit_alt_m
    planner_clearance_m = setup.planner_clearance_m
    route_xy = setup.route_xy
    goal_idx = setup.goal_idx
    command_rate_n_s = setup.command_rate_n_s
    command_fx_prev = setup.command_fx_prev
    command_fy_prev = setup.command_fy_prev
    controller_gains = setup.controller_gains
    hist = setup.hist
    arrival_time_s = setup.arrival_time_s
    hold_start_s = setup.hold_start_s

    for k in range(setup.steps):
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

        arrival = resolve_coordinate_arrival_state(
            route_goal=route_goal,
            dist_to_goal_m=dist_to_goal_m,
            arrival_radius_m=arrival_radius_m,
            dest_z_m=dest_z_m,
            z_m=st.z_m,
            speed_mps=speed_mps,
            t_s=t,
            hold_start_s=hold_start_s,
            arrival_time_s=arrival_time_s,
            fx_cmd=fx_cmd,
            fy_cmd=fy_cmd,
            command_fx_prev=command_fx_prev,
            command_fy_prev=command_fy_prev,
            descent_radius_m=descent_radius_m,
            z_target_m=z_target_m,
            active_safety=active_safety,
        )
        arrival_time_s = arrival.arrival_time_s
        hold_start_s = arrival.hold_start_s
        fx_cmd = arrival.fx_cmd
        fy_cmd = arrival.fy_cmd
        command_fx_prev = arrival.command_fx_prev
        command_fy_prev = arrival.command_fy_prev
        phase = arrival.phase

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
