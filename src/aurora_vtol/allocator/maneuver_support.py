from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .dynamics import AllocatorState, ActuatorLimits, PlenumModel
from .faults import FaultSpec
from .model import RingGeometry, segment_angles_rad
from .power_system import PowerSystemParams, init_hover_power_state, smoothstep01
from .sim_runtime import SimParams, SimState, append_engineering_telemetry
from ..effectiveness import effectiveness_table_for_topology
from ..icd import ActuatorHealthState, EstimatedVehicleState, RedirectTarget
from ..topology import default_ring_topology
from ..vehicle_controller import command_directional_force, track_redirect_velocity, track_step_snap_brake, track_step_snap_reverse


@dataclass(frozen=True)
class StatefulManeuverSetup:
    geom: RingGeometry
    sim: SimParams
    power: PowerSystemParams
    power_state: object
    lim: ActuatorLimits
    pl: PlenumModel
    fault: FaultSpec
    st: SimState
    allocator_state: AllocatorState
    steps: int
    theta_rad: object
    topology: object
    effectiveness: object
    fz_cmd: float


@dataclass(frozen=True)
class TurnGeometry:
    start_unit: tuple[float, float]
    target_unit: tuple[float, float]
    nominal_brake_dir_deg: float
    nominal_brake_unit: tuple[float, float]
    separation_deg: float
    turn_ratio: float


@dataclass(frozen=True)
class StepSnapShaping:
    hold_frac: float
    steer_scale: float
    redirect_sep_deg: float
    turn_ratio: float
    phase_c_ramp_steps: int


@dataclass(frozen=True)
class StepRedirectShaping:
    redirect_sep_deg: float
    turn_ratio: float
    redirect_steps: int
    settle_steps: int


@dataclass(frozen=True)
class StepSnapGuardProfile:
    fxy_budget_n: float
    speed_guard_scale: float
    gain_guard_scale: float
    power_priority_scale: float
    power_ratio_filt: float
    dead_align_scale: float
    dead_cross_scale: float
    dead_align_speed_floor_mps: float
    plenum_power_trim: float
    plenum_revector_trim: float
    plenum_align_speed_floor_mps: float
    plenum_brake_trim: float


@dataclass(frozen=True)
class StepRedirectGuardProfile:
    fxy_budget_n: float
    speed_guard_scale: float
    gain_guard_scale: float
    power_priority_scale: float
    power_ratio_filt: float


@dataclass(frozen=True)
class ManeuverPhaseCommand:
    phase: str
    dir_deg: float
    fx_raw: float
    fy_raw: float
    step_speed_ref: float


def heading_error_deg(a_deg: float, b_deg: float) -> float:
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(d)


def smoothstep_local(u: float) -> float:
    return smoothstep01(float(u))


def unit_or_default(x: float, y: float, default_xy: tuple[float, float]) -> tuple[float, float]:
    mag = math.hypot(x, y)
    if mag < 1e-6:
        return default_xy
    return (x / mag, y / mag)


def build_turn_geometry(dir_deg_a: float, dir_deg_b: float) -> TurnGeometry:
    start_unit = (math.cos(math.radians(dir_deg_a)), math.sin(math.radians(dir_deg_a)))
    target_unit = (math.cos(math.radians(dir_deg_b)), math.sin(math.radians(dir_deg_b)))
    nominal_brake_dir_deg = (dir_deg_a + 180.0) % 360.0
    nominal_brake_unit = (
        math.cos(math.radians(nominal_brake_dir_deg)),
        math.sin(math.radians(nominal_brake_dir_deg)),
    )
    separation_deg = heading_error_deg(dir_deg_a, dir_deg_b)
    turn_ratio = separation_deg / 180.0
    return TurnGeometry(
        start_unit=start_unit,
        target_unit=target_unit,
        nominal_brake_dir_deg=nominal_brake_dir_deg,
        nominal_brake_unit=nominal_brake_unit,
        separation_deg=separation_deg,
        turn_ratio=turn_ratio,
    )


def build_step_snap_shaping(
    *,
    turn: TurnGeometry,
    redirect_hold_frac: float,
    redirect_steer_scale: float,
    fault_profile: Mapping[str, float],
    dt_s: float,
) -> StepSnapShaping:
    if redirect_hold_frac < 0.0:
        hold_frac = float(np.clip(0.65 + 0.35 * turn.turn_ratio, 0.55, 0.98))
    else:
        hold_frac = float(np.clip(redirect_hold_frac, 0.0, 0.95))
    steer_scale = float(np.clip(redirect_steer_scale, 0.0, 1.5))
    phase_c_ramp_steps = max(
        1,
        int((0.75 + 0.95 * turn.turn_ratio + 0.35 * hold_frac) / max(1e-6, float(fault_profile['dead_align_scale'])) / dt_s),
    )
    target_dir_deg = (math.degrees(math.atan2(turn.target_unit[1], turn.target_unit[0])) + 360.0) % 360.0
    redirect_sep_deg = heading_error_deg(turn.nominal_brake_dir_deg, target_dir_deg)
    return StepSnapShaping(
        hold_frac=hold_frac,
        steer_scale=steer_scale,
        redirect_sep_deg=redirect_sep_deg,
        turn_ratio=redirect_sep_deg / 180.0,
        phase_c_ramp_steps=phase_c_ramp_steps,
    )


def build_step_redirect_shaping(
    *,
    turn: TurnGeometry,
    step_idx: int,
    redirect_end_idx: int,
    dt_s: float,
) -> StepRedirectShaping:
    redirect_steps = max(1, redirect_end_idx - step_idx)
    settle_steps = max(1, int((0.60 + 0.50 * turn.turn_ratio) / dt_s))
    return StepRedirectShaping(
        redirect_sep_deg=turn.separation_deg,
        turn_ratio=turn.turn_ratio,
        redirect_steps=redirect_steps,
        settle_steps=settle_steps,
    )


def build_stateful_maneuver_setup(
    *,
    total_s: float,
    yaw_hold_deg: float,
    geom=None,
    sim=None,
    lim: ActuatorLimits | None = None,
    pl: PlenumModel | None = None,
    fault: FaultSpec | None = None,
) -> StatefulManeuverSetup:
    geom = geom or RingGeometry()
    sim = sim or SimParams()
    power = PowerSystemParams()
    power_state = init_hover_power_state(power, geom, sim)
    lim = lim or ActuatorLimits()
    pl = pl or PlenumModel()
    fault = fault or FaultSpec()
    st = SimState(yaw_deg=yaw_hold_deg)
    allocator_state = AllocatorState.init(geom.n_segments)
    steps = int(total_s / sim.dt_s)
    theta_rad = segment_angles_rad(geom.n_segments)
    topology = default_ring_topology(geom.n_segments)
    effectiveness = effectiveness_table_for_topology(topology)
    fz_cmd = sim.mass_kg * sim.gravity
    return StatefulManeuverSetup(
        geom=geom,
        sim=sim,
        power=power,
        power_state=power_state,
        lim=lim,
        pl=pl,
        fault=fault,
        st=st,
        allocator_state=allocator_state,
        steps=steps,
        theta_rad=theta_rad,
        topology=topology,
        effectiveness=effectiveness,
        fz_cmd=fz_cmd,
    )


def build_maneuver_state(st: SimState, power_state: object, guard: Mapping[str, float]) -> EstimatedVehicleState:
    return EstimatedVehicleState(
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


def build_maneuver_health(*, fxy_budget_n: float, guard: Mapping[str, float]) -> ActuatorHealthState:
    return ActuatorHealthState(
        lateral_budget_n=fxy_budget_n,
        guard_scale=float(guard["guard_scale"]),
        response_scale=float(guard["fault_response_scale"]),
        continuous_power_ratio=float(guard["continuous_power_ratio"]),
        thermal_scale_pct=100.0 * float(guard["thermal_guard_scale"]),
        supply_scale_pct=100.0 * float(guard["supply_guard_scale"]),
        fault_available_scale=float(guard["fault_available_scale"]),
        fault_asymmetry_pct=float(guard["fault_asymmetry_pct"]),
    )


def scale_maneuver_health_lateral_budget(health: ActuatorHealthState, scale: float) -> ActuatorHealthState:
    return ActuatorHealthState(
        lateral_budget_n=float(health.lateral_budget_n) * float(scale),
        guard_scale=health.guard_scale,
        response_scale=health.response_scale,
        continuous_power_ratio=health.continuous_power_ratio,
        thermal_scale_pct=health.thermal_scale_pct,
        supply_scale_pct=health.supply_scale_pct,
        fault_available_scale=health.fault_available_scale,
        fault_asymmetry_pct=health.fault_asymmetry_pct,
    )


def build_step_snap_guard_profile(
    *,
    initial_budget_n: float,
    guard: Mapping[str, float],
    power_ratio_filt: float,
) -> StepSnapGuardProfile:
    budget_ratio = float(guard["budget_ratio"])
    speed_guard_scale = 0.38 + 0.62 * smoothstep_local((budget_ratio - 0.25) / 0.75)
    gain_guard_scale = 0.30 + 0.70 * smoothstep_local((budget_ratio - 0.25) / 0.75)
    power_pressure = smoothstep_local((float(guard["continuous_power_ratio"]) - 0.88) / 0.12)
    power_guard_scale = float(guard["power_guard_scale"])
    next_power_ratio_filt = 0.92 * power_ratio_filt + 0.08 * float(guard["continuous_power_ratio"])
    power_priority_scale = 1.0 - 0.32 * smoothstep_local((next_power_ratio_filt - 0.94) / 0.08)
    power_priority_scale = float(np.clip(power_priority_scale, 0.66, 1.0))
    plenum_power_trim = float(guard["plenum_power_trim"])
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
    fxy_budget_n = initial_budget_n
    fxy_budget_n *= 0.82 + 0.18 * power_guard_scale
    fxy_budget_n *= 0.70 + 0.30 * power_priority_scale
    fxy_budget_n *= plenum_power_trim
    return StepSnapGuardProfile(
        fxy_budget_n=fxy_budget_n,
        speed_guard_scale=speed_guard_scale,
        gain_guard_scale=gain_guard_scale,
        power_priority_scale=power_priority_scale,
        power_ratio_filt=next_power_ratio_filt,
        dead_align_scale=float(guard["dead_align_scale"]),
        dead_cross_scale=float(guard["dead_cross_scale"]),
        dead_align_speed_floor_mps=float(guard["dead_align_speed_floor_mps"]),
        plenum_power_trim=plenum_power_trim,
        plenum_revector_trim=float(guard["plenum_revector_trim"]),
        plenum_align_speed_floor_mps=float(guard["plenum_align_speed_floor_mps"]),
        plenum_brake_trim=float(guard["plenum_brake_trim"]),
    )


def build_step_redirect_guard_profile(
    *,
    initial_budget_n: float,
    guard: Mapping[str, float],
    power_ratio_filt: float,
) -> StepRedirectGuardProfile:
    budget_ratio = float(guard["budget_ratio"])
    speed_guard_scale = 0.40 + 0.60 * smoothstep_local((budget_ratio - 0.25) / 0.75)
    gain_guard_scale = 0.34 + 0.66 * smoothstep_local((budget_ratio - 0.25) / 0.75)
    power_pressure = smoothstep_local((float(guard["continuous_power_ratio"]) - 0.88) / 0.12)
    power_guard_scale = float(guard["power_guard_scale"])
    next_power_ratio_filt = 0.92 * power_ratio_filt + 0.08 * float(guard["continuous_power_ratio"])
    power_priority_scale = 1.0 - 0.30 * smoothstep_local((next_power_ratio_filt - 0.94) / 0.08)
    power_priority_scale = float(np.clip(power_priority_scale, 0.68, 1.0))
    speed_guard_scale *= 1.0 - 0.18 * power_pressure
    gain_guard_scale *= 1.0 - 0.24 * power_pressure
    speed_guard_scale *= 0.78 + 0.22 * power_guard_scale
    gain_guard_scale *= 0.70 + 0.30 * power_guard_scale
    speed_guard_scale *= 0.72 + 0.28 * power_priority_scale
    gain_guard_scale *= 0.62 + 0.38 * power_priority_scale
    speed_guard_scale *= float(guard["fault_guard_scale"])
    gain_guard_scale *= float(guard["fault_guard_scale"])
    fxy_budget_n = initial_budget_n
    fxy_budget_n *= 0.82 + 0.18 * power_guard_scale
    fxy_budget_n *= 0.70 + 0.30 * power_priority_scale
    return StepRedirectGuardProfile(
        fxy_budget_n=fxy_budget_n,
        speed_guard_scale=speed_guard_scale,
        gain_guard_scale=gain_guard_scale,
        power_priority_scale=power_priority_scale,
        power_ratio_filt=next_power_ratio_filt,
    )


def compute_step_snap_phase_command(
    *,
    k: int,
    step_idx: int,
    snap_end_idx: int,
    dir_deg_a: float,
    dir_deg_b: float,
    step_speed_ref: float,
    speed_stop_thr_mps: float,
    fxy_n: float,
    fz_cmd: float,
    mz_nm: float,
    brake_gain: float,
    maneuver_state: EstimatedVehicleState,
    maneuver_health: ActuatorHealthState,
    snap_shaping: StepSnapShaping,
    turn: TurnGeometry,
    guard_profile: StepSnapGuardProfile,
    st: SimState,
) -> ManeuverPhaseCommand:
    if k < step_idx:
        phase = "A"
        dir_deg = dir_deg_a
        phase_control = command_directional_force(
            math.cos(math.radians(dir_deg_a)),
            math.sin(math.radians(dir_deg_a)),
            maneuver_health,
            fxy_command_n=fxy_n,
            force_scale=guard_profile.gain_guard_scale,
            fz_n=fz_cmd,
            mz_nm=mz_nm,
            source="maneuver-controller.step-snap-cruise",
        )
        return ManeuverPhaseCommand(
            phase=phase,
            dir_deg=dir_deg,
            fx_raw=phase_control.raw_fx_n,
            fy_raw=phase_control.raw_fy_n,
            step_speed_ref=step_speed_ref,
        )

    if step_speed_ref <= 1e-6:
        step_speed_ref = max(speed_stop_thr_mps, float(math.hypot(st.vx_mps, st.vy_mps)))

    if k < snap_end_idx:
        phase = "B"
        progress = (k - step_idx) / max(1, snap_end_idx - step_idx - 1)
        steer_progress = smoothstep_local((progress - snap_shaping.hold_frac) / max(1e-6, 1.0 - snap_shaping.hold_frac))
        steer_mix = float(np.clip(0.35 * steer_progress * snap_shaping.steer_scale, 0.0, 0.45))
        phase_control = track_step_snap_brake(
            maneuver_state,
            target_ux=turn.target_unit[0],
            target_uy=turn.target_unit[1],
            nominal_brake_ux=turn.nominal_brake_unit[0],
            nominal_brake_uy=turn.nominal_brake_unit[1],
            steer_mix=steer_mix,
            health=scale_maneuver_health_lateral_budget(maneuver_health, brake_gain),
            fxy_command_n=fxy_n,
            brake_gain=brake_gain * guard_profile.gain_guard_scale * guard_profile.power_priority_scale * guard_profile.plenum_revector_trim * guard_profile.plenum_brake_trim * (1.0 - 0.20 * steer_mix),
            fz_n=fz_cmd,
            mz_nm=mz_nm,
            source="maneuver-controller.step-snap-brake",
        )
        dir_deg = (math.degrees(math.atan2(phase_control.command_uy, phase_control.command_ux)) + 360.0) % 360.0
        return ManeuverPhaseCommand(
            phase=phase,
            dir_deg=dir_deg,
            fx_raw=phase_control.raw_fx_n,
            fy_raw=phase_control.raw_fy_n,
            step_speed_ref=step_speed_ref,
        )

    phase = "C"
    dir_deg = dir_deg_b
    phase_c_progress = smoothstep_local((k - snap_end_idx) / snap_shaping.phase_c_ramp_steps)
    desired_along_speed = step_speed_ref * phase_c_progress * guard_profile.speed_guard_scale
    if guard_profile.dead_align_scale > 1.0:
        align_capture = smoothstep_local((phase_c_progress - 0.18) / 0.42)
        desired_along_speed = max(desired_along_speed, guard_profile.dead_align_speed_floor_mps * align_capture)
    if guard_profile.plenum_align_speed_floor_mps > 1e-6:
        plenum_align_capture = smoothstep_local((phase_c_progress - 0.24) / 0.34)
        desired_along_speed = max(desired_along_speed, guard_profile.plenum_align_speed_floor_mps * plenum_align_capture)
    cross_gain = (0.75 + 0.45 * snap_shaping.turn_ratio) * guard_profile.dead_cross_scale * guard_profile.plenum_revector_trim
    reverse_force_scale = (0.40 + 0.60 * phase_c_progress) * guard_profile.gain_guard_scale * guard_profile.power_priority_scale * guard_profile.plenum_revector_trim
    if guard_profile.dead_align_scale > 1.0:
        reverse_force_scale *= 0.90 + 0.10 * guard_profile.dead_align_scale
    phase_control = track_step_snap_reverse(
        maneuver_state,
        target_ux=turn.target_unit[0],
        target_uy=turn.target_unit[1],
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
    return ManeuverPhaseCommand(
        phase=phase,
        dir_deg=dir_deg,
        fx_raw=phase_control.raw_fx_n,
        fy_raw=phase_control.raw_fy_n,
        step_speed_ref=step_speed_ref,
    )


def compute_step_redirect_phase_command(
    *,
    k: int,
    step_idx: int,
    redirect_end_idx: int,
    dir_deg_a: float,
    yaw_hold_deg: float,
    step_speed_ref: float,
    fxy_n: float,
    mz_nm: float,
    redirect_speed_scale: float,
    redirect_cross_gain: float,
    maneuver_state: EstimatedVehicleState,
    maneuver_health: ActuatorHealthState,
    redirect_shaping: StepRedirectShaping,
    turn: TurnGeometry,
    guard_profile: StepRedirectGuardProfile,
    st: SimState,
) -> ManeuverPhaseCommand:
    if k < step_idx:
        phase = "A"
        dir_deg = dir_deg_a
        phi = math.radians(dir_deg)
        return ManeuverPhaseCommand(
            phase=phase,
            dir_deg=dir_deg,
            fx_raw=fxy_n * guard_profile.gain_guard_scale * math.cos(phi),
            fy_raw=fxy_n * guard_profile.gain_guard_scale * math.sin(phi),
            step_speed_ref=step_speed_ref,
        )

    if step_speed_ref <= 1e-6:
        step_speed_ref = max(0.75, float(math.hypot(st.vx_mps, st.vy_mps)))

    if k < redirect_end_idx:
        phase = "R"
        progress = smoothstep_local((k - step_idx) / max(1, redirect_shaping.redirect_steps - 1))
    else:
        phase = "C"
        progress = 1.0

    desired_dir_x = (1.0 - progress) * turn.start_unit[0] + progress * turn.target_unit[0]
    desired_dir_y = (1.0 - progress) * turn.start_unit[1] + progress * turn.target_unit[1]
    desired_ux, desired_uy = unit_or_default(desired_dir_x, desired_dir_y, turn.target_unit)
    desired_speed = step_speed_ref * (redirect_speed_scale + (1.0 - redirect_speed_scale) * progress)
    desired_speed *= guard_profile.speed_guard_scale

    if phase == "R":
        phase_scale = (0.50 + 0.40 * progress) * (1.0 - 0.12 * redirect_shaping.turn_ratio * (1.0 - progress))
    else:
        settle_progress = smoothstep_local((k - redirect_end_idx) / redirect_shaping.settle_steps)
        phase_scale = 0.78 + 0.22 * settle_progress
    phase_scale *= guard_profile.gain_guard_scale

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
    dir_deg = (math.degrees(math.atan2(redirect_control.command_uy, redirect_control.command_ux)) + 360.0) % 360.0
    return ManeuverPhaseCommand(
        phase=phase,
        dir_deg=dir_deg,
        fx_raw=redirect_control.raw_fx_n,
        fy_raw=redirect_control.raw_fy_n,
        step_speed_ref=step_speed_ref,
    )


def append_stateful_maneuver_history(
    hist: dict,
    *,
    t: float,
    st: SimState,
    speed: float | None,
    mz_est: float,
    alpha_actual_rad,
    ft_tan_per_seg_n,
    telemetry: Mapping[str, object],
    fx_cmd: float,
    fy_cmd: float,
    fz_cmd: float,
    net_force_n,
    extras: Mapping[str, object] | None = None,
) -> None:
    alpha_actual_rad = np.asarray(alpha_actual_rad, dtype=float)
    ft_tan_per_seg_n = np.asarray(ft_tan_per_seg_n, dtype=float)

    if "fan_thrust_16" in hist and "fan_actual_16" in telemetry:
        hist["fan_thrust_16"].append(list(telemetry["fan_actual_16"]))

    if "t" in hist:
        hist["t"].append(float(t))
    if "x" in hist:
        hist["x"].append(st.x_m)
    if "y" in hist:
        hist["y"].append(st.y_m)
    if "z" in hist:
        hist["z"].append(st.z_m)
    if "vx" in hist:
        hist["vx"].append(st.vx_mps)
    if "vy" in hist:
        hist["vy"].append(st.vy_mps)
    if "vz" in hist:
        hist["vz"].append(st.vz_mps)
    if "speed" in hist and speed is not None:
        hist["speed"].append(float(speed))
    if "yaw_deg" in hist:
        hist["yaw_deg"].append(st.yaw_deg)
    if "yaw_rate_deg_s" in hist:
        hist["yaw_rate_deg_s"].append(st.yaw_rate_deg_s)
    if "mz_est" in hist:
        hist["mz_est"].append(float(mz_est))
    if "alpha_deg_rms" in hist:
        hist["alpha_deg_rms"].append(float(np.sqrt(np.mean(np.degrees(alpha_actual_rad) ** 2))))
    if "ft_tan_rms" in hist:
        hist["ft_tan_rms"].append(float(np.sqrt(np.mean(ft_tan_per_seg_n ** 2))))
    if "alpha_deg_32" in hist:
        hist["alpha_deg_32"].append(list(np.degrees(alpha_actual_rad)))
    if "ft_tan_32" in hist:
        hist["ft_tan_32"].append(list(ft_tan_per_seg_n))
    if "fx_cmd" in hist:
        hist["fx_cmd"].append(float(fx_cmd))
    if "fy_cmd" in hist:
        hist["fy_cmd"].append(float(fy_cmd))

    if extras:
        for key, value in extras.items():
            if key in hist:
                hist[key].append(value)

    append_engineering_telemetry(hist, telemetry, fx_cmd, fy_cmd, fz_cmd, net_force_n)
