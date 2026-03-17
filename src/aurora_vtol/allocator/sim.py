from __future__ import annotations
import math
import numpy as np

from .model import RingGeometry, segment_angles_rad, thrust_vectors_body, net_force_and_yaw_moment
from .allocate import AllocationRequest, allocate_v1, allocate_v2
from .metrics import yaw_track_coupling_mean_abs
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel, apply_actuator_limits, apply_plenum_lag
from .response import compute_step_metrics
from .field import RepelField, repel_force_xy
from .mission_planning import MissionObstacle, mission_safety_force, plan_route_waypoints
from .faults import FaultSpec, apply_command_faults_to_alpha, apply_faults_to_alpha, apply_faults_to_thrust
from .trace import save_trace_json
from .sim_runtime import SimParams, SimState, append_engineering_telemetry, clip_force_xy, fan_means_to_segments, pair_segments_to_fans, rate_limit_xy_force, step_vehicle
from .power_system import PowerSystemParams, apply_power_system, guidance_force_budget, init_hover_power_state, smoothstep01
from ..icd import ActuatorHealthState, EstimatedVehicleState, GuidanceTarget, RedirectTarget
from ..topology import default_ring_topology
from ..effectiveness import effectiveness_table_for_topology, hardware_assumptions_payload
from ..vehicle_controller import XYVehicleControllerGains, command_directional_force, track_redirect_velocity, track_step_snap_brake, track_step_snap_reverse, track_xy_position

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
            topology=topology,
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

