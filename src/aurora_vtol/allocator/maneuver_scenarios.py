from __future__ import annotations

import math

import numpy as np

from .allocate import AllocationRequest, allocate_v1, allocate_v2
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel, apply_actuator_limits, apply_plenum_lag
from .faults import FaultSpec, apply_command_faults_to_alpha, apply_faults_to_alpha, apply_faults_to_thrust
from .field import RepelField, repel_force_xy
from .metrics import yaw_track_coupling_mean_abs
from .model import RingGeometry, net_force_and_yaw_moment, segment_angles_rad, thrust_vectors_body
from .power_system import PowerSystemParams, apply_power_system, guidance_force_budget, init_hover_power_state
from .response import compute_step_metrics
from .sim_runtime import SimParams, SimState, append_engineering_telemetry, clip_force_xy, rate_limit_xy_force, step_vehicle
from ..effectiveness import effectiveness_table_for_topology, hardware_assumptions_payload
from ..icd import ActuatorHealthState, EstimatedVehicleState, RedirectTarget
from ..topology import default_ring_topology
from ..vehicle_controller import command_directional_force, track_redirect_velocity, track_step_snap_brake, track_step_snap_reverse

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
            topology=topology,
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
            topology=topology,
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
            topology=topology,
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
            topology=topology,
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
            topology=topology,
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
