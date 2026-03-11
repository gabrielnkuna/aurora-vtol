from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .model import RingGeometry, segment_angles_rad, thrust_vectors_body, net_force_and_yaw_moment
from .allocate import AllocationRequest, allocate_v1, allocate_v2
from .metrics import yaw_track_coupling_mean_abs
from .dynamics import AllocatorState, ActuatorLimits, PlenumModel, apply_actuator_limits, apply_plenum_lag
from .response import compute_step_metrics
from .field import RepelField, repel_force_xy
from .faults import FaultSpec, apply_faults_to_alpha, apply_faults_to_thrust
from .trace import save_trace_json

@dataclass(frozen=True)
class SimParams:
    mass_kg: float = 800.0
    dt_s: float = 0.02
    drag_coeff: float = 0.8
    gravity: float = 9.81

@dataclass
class SimState:
    x_m: float = 0.0
    y_m: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    yaw_deg: float = 0.0

def step_xy(state: SimState, fx: float, fy: float, p: SimParams) -> None:
    fx_net = fx - p.drag_coeff * state.vx_mps
    fy_net = fy - p.drag_coeff * state.vy_mps
    ax = fx_net / p.mass_kg
    ay = fy_net / p.mass_kg
    state.vx_mps += ax * p.dt_s
    state.vy_mps += ay * p.dt_s
    state.x_m += state.vx_mps * p.dt_s
    state.y_m += state.vy_mps * p.dt_s

def run_demo(dir_deg: float, fxy_n: float, duration_s: float, yaw_hold_deg: float = 0.0, mz_nm: float = 0.0, version: str = "v2", geom: RingGeometry | None = None, sim: SimParams | None = None):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    st = SimState(yaw_deg=yaw_hold_deg)
    steps = int(duration_s / sim.dt_s)
    hist = {"t": [], "x": [], "y": [], "vx": [], "vy": [], "yaw_deg": [], "mz_est": [], "alpha_deg_rms": [], "alpha_deg_32": [], "ft_tan_32": []}

    phi = math.radians(dir_deg)
    fx_cmd = fxy_n * math.cos(phi)
    fy_cmd = fxy_n * math.sin(phi)
    fz_cmd = 800.0 * sim.gravity
    theta = segment_angles_rad(geom.n_segments)

    for k in range(steps):
        t = k * sim.dt_s
        req = AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm)
        # allocate using the chosen allocator version; v1 does not compute tangential forces
        # and therefore has no ft_tan_per_seg_n attribute.
        if version.lower() == "v1":
            alloc = allocate_v1(geom, req)
            ft_rms = 0.0
            ft_vals = [0.0] * geom.n_segments
        else:
            alloc = allocate_v2(geom, req)
            ft_rms = float(np.sqrt(np.mean(alloc.ft_tan_per_seg_n ** 2)))
            ft_vals = list(alloc.ft_tan_per_seg_n)

        forces = thrust_vectors_body(geom, alloc.thrust_per_seg_n, alloc.alpha_rad, theta)
        if version.lower() != "v1":
            fx_t = alloc.ft_tan_per_seg_n * (-np.sin(theta))
            fy_t = alloc.ft_tan_per_seg_n * (np.cos(theta))
            forces = forces + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)

        net, mz_est = net_force_and_yaw_moment(geom, forces)
        step_xy(st, float(net[0]), float(net[1]), sim)

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps)
        hist["yaw_deg"].append(st.yaw_deg)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alloc.alpha_rad) ** 2))))
        # record full flap/thrust arrays
        hist["alpha_deg_32"].append(list(np.degrees(alloc.alpha_rad)))
        hist["ft_tan_32"].append(ft_vals)
        hist.setdefault("ft_tan_rms", []).append(ft_rms)

    return hist

def run_step_test_v3(dir_deg_a: float = 0.0, dir_deg_b: float = 180.0, fxy_n: float = 3000.0, step_time_s: float = 3.0, total_s: float = 8.0, yaw_hold_deg: float = 0.0, mz_nm: float = 0.0, version_note: str = "v3", geom=None, sim=None, lim: ActuatorLimits | None = None, pl: PlenumModel | None = None):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    st = SimState(yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    step_k = int(step_time_s / sim.dt_s)
    theta = segment_angles_rad(geom.n_segments)
    fz_cmd = 800.0 * sim.gravity

    hist = {"t": [], "x": [], "y": [], "vx": [], "vy": [], "yaw_deg": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [], "speed": [], "cmd_dir_deg": [], "alpha_deg_32": [], "ft_tan_32": [], "fan_thrust_16": []}

    for k in range(steps):
        t = k * sim.dt_s
        dir_deg = dir_deg_a if k < step_k else dir_deg_b
        phi = math.radians(dir_deg)
        fx_cmd = fxy_n * math.cos(phi)
        fy_cmd = fxy_n * math.sin(phi)

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm))

        state = apply_actuator_limits(state, alloc.alpha_rad, alloc.ft_tan_per_seg_n, lim, sim.dt_s)

        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)

        thrust = alloc.thrust_per_seg_n * state.plenum_scale
        # fan logging for step-test
        fan16 = []
        for j in range(16):
            fan16.append(float(0.5 * (thrust[2*j] + thrust[2*j + 1])))
        hist["fan_thrust_16"].append(fan16)

        forces_main = thrust_vectors_body(geom, thrust, state.alpha_rad, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
        net, mz_est = net_force_and_yaw_moment(geom, forces)

        step_xy(st, float(net[0]), float(net[1]), sim)
        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps)
        hist["yaw_deg"].append(st.yaw_deg)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(state.alpha_rad) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        # full per-segment data for visualization
        hist["alpha_deg_32"].append(list(np.degrees(state.alpha_rad)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["speed"].append(speed)
        hist["cmd_dir_deg"].append(dir_deg)

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

    # t90_dir_s: first time track is within 20° of target AND speed > 0.5m/s
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
           "step_metrics": {"t90_dir_s": t90_dir, "t_reversal_s": t_reversal, "min_speed_transition_mps": min_speed_transition, "peak_speed_mps": float(np.max(sp_arr)) if sp_arr.size else None, "yaw_track_coupling_mean_abs_deg": coupling},
           "final": {"x_m": hist["x"][-1], "y_m": hist["y"][-1], "vx_mps": hist["vx"][-1], "vy_mps": hist["vy"][-1], "alpha_deg_rms": hist["alpha_deg_rms"][-1], "ft_tan_rms": hist["ft_tan_rms"][-1], "mz_est_nm": hist["mz_est"][-1]}}
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
    brake_gain: float = 1.2,
    speed_stop_thr_mps: float = 0.2,
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
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()

    st = SimState(yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    step_idx = int(step_time_s / sim.dt_s)
    snap_end_idx = int((step_time_s + snap_stop_s) / sim.dt_s)

    theta = segment_angles_rad(geom.n_segments)
    fz_cmd = 800.0 * sim.gravity

    hist = {
        "t": [], "x": [], "y": [], "vx": [], "vy": [], "speed": [],
        "yaw_deg": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [],
        "alpha_deg_32": [], "ft_tan_32": [], "fan_thrust_16": [],
        "cmd_phase": [], "cmd_dir_deg": [], "fx_cmd": [], "fy_cmd": []
    }

    # helpers
    def ang_err(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    for k in range(steps):
        t = k * sim.dt_s

        # Decide phase
        if k < step_idx:
            phase = "A"
            dir_deg = dir_deg_a
            phi = math.radians(dir_deg)
            fx_cmd = fxy_n * math.cos(phi)
            fy_cmd = fxy_n * math.sin(phi)

        elif k < snap_end_idx:
            phase = "B"  # snap-stop (brake opposite current velocity)
            sp = math.hypot(st.vx_mps, st.vy_mps)
            if sp < 1e-6:
                fx_cmd, fy_cmd = 0.0, 0.0
            else:
                # braking force opposite velocity direction
                ux = st.vx_mps / sp
                uy = st.vy_mps / sp
                mag = brake_gain * fxy_n
                fx_cmd = -mag * ux
                fy_cmd = -mag * uy
            # keep cmd_dir_deg for logging (track direction target not meaningful here)
            dir_deg = float("nan")

        else:
            phase = "C"
            dir_deg = dir_deg_b
            phi = math.radians(dir_deg)
            fx_cmd = fxy_n * math.cos(phi)
            fy_cmd = fxy_n * math.sin(phi)

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm))

        # actuator limits
        state = apply_actuator_limits(state, alloc.alpha_rad, alloc.ft_tan_per_seg_n, lim, sim.dt_s)

        # plenum lag
        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)

        thrust = alloc.thrust_per_seg_n * state.plenum_scale
        # record 16‑fan pairs
        fan16 = []
        for j in range(16):
            fan16.append(float(0.5 * (thrust[2*j] + thrust[2*j + 1])))
        hist["fan_thrust_16"].append(fan16)

        forces_main = thrust_vectors_body(geom, thrust, state.alpha_rad, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)

        net, mz_est = net_force_and_yaw_moment(geom, forces)
        step_xy(st, float(net[0]), float(net[1]), sim)

        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps)
        hist["speed"].append(speed)
        hist["yaw_deg"].append(st.yaw_deg)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(state.alpha_rad) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        # per-segment arrays for snap visualization
        hist["alpha_deg_32"].append(list(np.degrees(state.alpha_rad)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["cmd_phase"].append(phase)
        hist["cmd_dir_deg"].append(dir_deg if isinstance(dir_deg, float) else float(dir_deg))
        hist["fx_cmd"].append(float(fx_cmd)); hist["fy_cmd"].append(float(fy_cmd))

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

    # 4) time to align direction within 20° (and speed > 0.5)
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
            "vx_mps": float(vx_arr[-1]),
            "vy_mps": float(vy_arr[-1]),
            "alpha_deg_rms": float(hist["alpha_deg_rms"][-1]),
            "ft_tan_rms": float(hist["ft_tan_rms"][-1]),
            "mz_est_nm": float(hist["mz_est"][-1]),
        },
        "timestamps": {
            "step_t": step_t,
            "snap_end_t": snap_end_t,
        }
    }

    return out, hist

def run_repel_test_v4(obstacle_x_m: float = 30.0, obstacle_y_m: float = 0.0, initial_x_m: float = 0.0, initial_y_m: float = 0.0, initial_vx_mps: float = 1.0, initial_vy_mps: float = 0.0, yaw_hold_deg: float = 0.0, total_s: float = 12.0, mz_nm: float = 0.0,
                      field: RepelField | None = None, fault: FaultSpec | None = None, trace_out: str | None = None, geom=None, sim=None, lim: ActuatorLimits | None = None, pl: PlenumModel | None = None):
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    field = field or RepelField()
    fault = fault or FaultSpec()

    st = SimState(x_m=initial_x_m, y_m=initial_y_m, vx_mps=initial_vx_mps, vy_mps=initial_vy_mps, yaw_deg=yaw_hold_deg)
    state = AllocatorState.init(geom.n_segments)

    steps = int(total_s / sim.dt_s)
    fz_cmd = 800.0 * sim.gravity
    theta = segment_angles_rad(geom.n_segments)

    hist = {"t": [], "x": [], "y": [], "vx": [], "vy": [], "speed": [], "yaw_deg": [], "fx_cmd": [], "fy_cmd": [], "dist_to_obstacle": [], "mz_est": [], "alpha_deg_rms": [], "ft_tan_rms": [], "alpha_deg_32": [], "ft_tan_32": [], "faults": [], "fan_thrust_16": []}

    for k in range(steps):
        t = k * sim.dt_s
        dist = float(math.hypot(st.x_m - obstacle_x_m, st.y_m - obstacle_y_m))
        fx_cmd, fy_cmd = repel_force_xy(field, st.x_m, st.y_m, obstacle_x_m, obstacle_y_m)

        alloc = allocate_v2(geom, AllocationRequest(fx_cmd, fy_cmd, fz_cmd, mz_nm))
        alpha_target = apply_faults_to_alpha(alloc.alpha_rad, fault)
        thrust_target = apply_faults_to_thrust(alloc.thrust_per_seg_n, fault)
        ft_target = alloc.ft_tan_per_seg_n

        state = apply_actuator_limits(state, alpha_target, ft_target, lim, sim.dt_s)
        demand = 1.0 + 0.20 * (np.abs(state.alpha_rad) / max(1e-6, np.max(np.abs(state.alpha_rad))))
        demand = np.clip(demand, pl.min_scale, pl.max_scale)
        state = apply_plenum_lag(state, demand, pl, sim.dt_s)

        thrust = thrust_target * state.plenum_scale
        # log 16-fan thrust
        fan16 = []
        for j in range(16):
            fan16.append(float(0.5 * (thrust[2*j] + thrust[2*j + 1])))
        hist["fan_thrust_16"].append(fan16)
        forces_main = thrust_vectors_body(geom, thrust, state.alpha_rad, theta)
        fx_t = state.ft_tan_per_seg_n * (-np.sin(theta))
        fy_t = state.ft_tan_per_seg_n * (np.cos(theta))
        forces = forces_main + np.stack([fx_t, fy_t, np.zeros_like(fx_t)], axis=1)
        net, mz_est = net_force_and_yaw_moment(geom, forces)

        step_xy(st, float(net[0]), float(net[1]), sim)
        speed = float(math.hypot(st.vx_mps, st.vy_mps))

        hist["t"].append(t)
        hist["x"].append(st.x_m); hist["y"].append(st.y_m)
        hist["vx"].append(st.vx_mps); hist["vy"].append(st.vy_mps); hist["speed"].append(speed)
        hist["yaw_deg"].append(st.yaw_deg)
        hist["fx_cmd"].append(fx_cmd); hist["fy_cmd"].append(fy_cmd)
        hist["dist_to_obstacle"].append(dist)
        hist["mz_est"].append(float(mz_est))
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(state.alpha_rad) ** 2))))
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(state.ft_tan_per_seg_n ** 2))))
        # per-segment arrays for visualization
        hist["alpha_deg_32"].append(list(np.degrees(state.alpha_rad)))
        hist["ft_tan_32"].append(list(state.ft_tan_per_seg_n))
        hist["faults"].append({"stuck_flap_idx": fault.stuck_flap_idx, "dead_fan_group": fault.dead_fan_group, "plenum_sector_idx": fault.plenum_sector_idx})

    coupling = yaw_track_coupling_mean_abs(hist)
    meta = {"version": "v4", "obstacle": {"x_m": obstacle_x_m, "y_m": obstacle_y_m}, "field": field.__dict__, "fault": fault.__dict__,
            "yaw_track_coupling_mean_abs_deg": coupling, "sim": {"dt_s": sim.dt_s, "mass_kg": sim.mass_kg, "drag_coeff": sim.drag_coeff}, "limits": lim.__dict__, "plenum": pl.__dict__}
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

    # old speed‑difference based response (kept for compatibility)
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