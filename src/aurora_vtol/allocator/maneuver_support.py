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
from ..icd import ActuatorHealthState, EstimatedVehicleState
from ..topology import default_ring_topology


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
