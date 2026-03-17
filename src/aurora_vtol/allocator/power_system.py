from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import math
from typing import TYPE_CHECKING

import numpy as np

from .dynamics import AllocatorState
from .faults import FaultSpec
from .model import RingGeometry, segment_angles_rad
from ..topology import RingActuatorTopology, default_ring_topology

if TYPE_CHECKING:
    from .sim_runtime import SimParams

@dataclass(frozen=True)
class PowerSystemParams:
    battery_full_v: float = 58.8
    battery_empty_v: float = 46.0
    capacity_kwh: float = 15.0
    internal_resistance_ohm: float = 0.0012
    hover_power_w: float = 110000.0
    continuous_power_w: float = 125000.0
    peak_power_w: float = 165000.0
    burst_duration_s: float = 4.0
    burst_recharge_tau_s: float = 18.0
    aux_power_w: float = 2500.0
    thrust_power_exponent: float = 1.35
    flap_power_scale: float = 0.10
    tangential_power_scale: float = 0.04
    min_supply_scale: float = 0.80
    fan_spool_tau_s: float = 0.18
    fan_heat_fraction: float = 0.11
    overload_heat_gain: float = 0.45
    fan_thermal_capacity_j_per_c: float = 5500.0
    fan_cooling_w_per_c: float = 28.0
    ambient_temp_c: float = 28.0
    thermal_limit_start_c: float = 85.0
    thermal_limit_full_c: float = 110.0
    min_thermal_scale: float = 0.65


@dataclass
class PowerSystemState:
    soc_frac: float = 1.0
    voltage_v: float = 58.8
    current_a: float = 0.0
    power_w: float = 2500.0
    energy_used_wh: float = 0.0
    thrust_scale: float = 1.0
    burst_reserve_j: float = 0.0
    fan_mean_n: np.ndarray = dc_field(default_factory=lambda: np.zeros(16, dtype=float))
    fan_temp_c: np.ndarray = dc_field(default_factory=lambda: np.full(16, 28.0, dtype=float))
    thermal_scale: np.ndarray = dc_field(default_factory=lambda: np.ones(16, dtype=float))

    @classmethod
    def init(cls, power: "PowerSystemParams", fan_count: int = 16) -> "PowerSystemState":
        return cls(
            soc_frac=1.0,
            voltage_v=power.battery_full_v,
            current_a=0.0,
            power_w=power.aux_power_w,
            energy_used_wh=0.0,
            thrust_scale=1.0,
            burst_reserve_j=burst_capacity_j(power),
            fan_mean_n=np.zeros(fan_count, dtype=float),
            fan_temp_c=np.full(fan_count, power.ambient_temp_c, dtype=float),
            thermal_scale=np.ones(fan_count, dtype=float),
        )


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def burst_capacity_j(power: PowerSystemParams) -> float:
    return max(0.0, (power.peak_power_w - power.continuous_power_w) * max(power.burst_duration_s, 0.0))


def sustained_power_ratio(power: PowerSystemParams, power_w: float, burst_reserve_j: float) -> float:
    continuous_limit = max(power.continuous_power_w, 1e-6)
    raw_ratio = float(power_w) / continuous_limit
    if raw_ratio <= 1.0:
        return raw_ratio
    capacity = burst_capacity_j(power)
    if capacity <= 1e-9:
        return raw_ratio
    reserve_frac = max(0.0, min(1.0, float(burst_reserve_j) / capacity))
    overload = raw_ratio - 1.0
    return 1.0 + overload * (1.0 - reserve_frac)


def clip_main_power_to_budget(
    fan_actual_mean: np.ndarray,
    thrust_pre_power: np.ndarray,
    load_multiplier: float,
    power_budget_w: float,
    power: PowerSystemParams,
    topology,
    hover_per_seg_n: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    budget_w = max(power.aux_power_w, float(power_budget_w))
    peak_budget_w = max(1e-6, budget_w - power.aux_power_w)
    main_load_w = max(1e-6, budget_w - power.aux_power_w)
    scale = 1.0
    power_w = power.aux_power_w
    thrust_actual = topology.distribute_fan_means_to_segments(fan_actual_mean, thrust_pre_power)
    thrust_ratio = np.clip(thrust_actual / hover_per_seg_n, 0.0, None)
    mean_thrust_ratio = float(np.mean(thrust_ratio)) if thrust_ratio.size else 0.0
    main_power_w = power.hover_power_w * (mean_thrust_ratio ** power.thrust_power_exponent)
    power_w = power.aux_power_w + main_power_w * load_multiplier
    if power_w > budget_w:
        main_load_w = max(1e-6, power_w - power.aux_power_w)
        scale = (peak_budget_w / main_load_w) ** (1.0 / max(power.thrust_power_exponent, 1e-6))
        scale = max(power.min_supply_scale, min(1.0, scale))
        fan_actual_mean = fan_actual_mean * scale
        thrust_actual = topology.distribute_fan_means_to_segments(fan_actual_mean, thrust_pre_power)
        thrust_ratio = np.clip(thrust_actual / hover_per_seg_n, 0.0, None)
        mean_thrust_ratio = float(np.mean(thrust_ratio)) if thrust_ratio.size else 0.0
        main_power_w = power.hover_power_w * (mean_thrust_ratio ** power.thrust_power_exponent)
        power_w = power.aux_power_w + main_power_w * load_multiplier
    return fan_actual_mean, thrust_actual, float(power_w), float(scale)


def temperature_to_thermal_scale(temp_c, power: PowerSystemParams) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    span = max(1e-6, power.thermal_limit_full_c - power.thermal_limit_start_c)
    hot_x = np.clip((temp - power.thermal_limit_start_c) / span, 0.0, 1.0)
    hot_frac = hot_x * hot_x * (3.0 - 2.0 * hot_x)
    scale = power.min_thermal_scale + (1.0 - power.min_thermal_scale) * (1.0 - hot_frac)
    return np.clip(scale, power.min_thermal_scale, 1.0)


def init_hover_power_state(power: PowerSystemParams, geom: RingGeometry, sim: SimParams, topology: RingActuatorTopology | None = None) -> PowerSystemState:
    active_topology = default_ring_topology(geom.n_segments) if topology is None else topology
    fan_count = active_topology.fan_count
    hover_seg_n = sim.mass_kg * sim.gravity / max(1, geom.n_segments)
    hover_power_w = power.hover_power_w + power.aux_power_w
    return PowerSystemState(
        soc_frac=1.0,
        voltage_v=power.battery_full_v,
        current_a=hover_power_w / max(power.battery_full_v, 1.0),
        power_w=hover_power_w,
        energy_used_wh=0.0,
        thrust_scale=1.0,
        burst_reserve_j=burst_capacity_j(power),
        fan_mean_n=np.full(fan_count, hover_seg_n, dtype=float),
        fan_temp_c=np.full(fan_count, power.ambient_temp_c, dtype=float),
        thermal_scale=np.ones(fan_count, dtype=float),
    )


def fault_motion_guard(geom: RingGeometry, fault: FaultSpec | None, topology: RingActuatorTopology | None = None) -> dict[str, float]:
    if fault is None:
        return {
            'fault_guard_scale': 1.0,
            'fault_response_scale': 1.0,
            'fault_available_scale': 1.0,
            'fault_asymmetry': 0.0,
            'dead_align_scale': 1.0,
            'dead_cross_scale': 1.0,
            'dead_align_speed_floor_mps': 0.55,
            'plenum_power_trim': 1.0,
            'plenum_revector_trim': 1.0,
            'plenum_align_speed_floor_mps': 0.0,
            'plenum_brake_trim': 1.0,
        }

    topology = default_ring_topology(geom.n_segments) if topology is None else topology
    scale = topology.segment_effectiveness_scales(fault)

    theta = segment_angles_rad(geom.n_segments)
    available_scale = float(np.clip(np.mean(scale), 0.0, 1.0))
    weight_sum = float(np.sum(scale))
    if weight_sum > 1e-9:
        asymmetry = float(math.hypot(float(np.sum(scale * np.cos(theta))), float(np.sum(scale * np.sin(theta)))) / weight_sum)
    else:
        asymmetry = 1.0

    dead_guard = 1.0
    dead_align_scale = 1.0
    dead_cross_scale = 1.0
    dead_align_speed_floor_mps = 0.55
    if fault.dead_fan_group is not None:
        idxs = list(topology.fan_segments(int(fault.dead_fan_group)))
        local_scale = float(np.mean(scale[idxs])) if idxs else 1.0
        lost_scale = float(np.clip(1.0 - local_scale, 0.0, 1.0))
        dead_guard = 0.60 + 0.40 * local_scale
        dead_align_scale = 1.0 + 0.45 * lost_scale
        dead_cross_scale = 1.0 - 0.24 * lost_scale
        dead_align_speed_floor_mps = 0.55 + 0.22 * lost_scale

    plenum_guard = 1.0
    plenum_power_trim = 1.0
    plenum_revector_trim = 1.0
    plenum_align_speed_floor_mps = 0.0
    plenum_brake_trim = 1.0
    if fault.plenum_sector_idx is not None:
        idxs = list(topology.plenum_segments(int(fault.plenum_sector_idx)))
        local_scale = float(np.mean(scale[idxs])) if idxs else 1.0
        lost_scale = float(np.clip(1.0 - local_scale, 0.0, 1.0))
        plenum_guard = 0.82 + 0.18 * local_scale
        plenum_power_trim = 1.0 - 0.40 * lost_scale
        plenum_revector_trim = 1.0 - 0.22 * lost_scale
        plenum_align_speed_floor_mps = 0.48 + 0.18 * lost_scale
        plenum_brake_trim = 1.0 - 0.18 * lost_scale

    stuck_guard = 1.0
    if fault.stuck_flap_idx is not None:
        alpha_limit_rad = max(1e-6, math.radians(geom.alpha_max_deg))
        stuck_severity = min(1.0, abs(math.radians(float(fault.stuck_flap_alpha_deg))) / alpha_limit_rad)
        stuck_guard = 1.0 - 0.18 * stuck_severity

    authority_guard = min(available_scale, dead_guard, plenum_guard, stuck_guard, 1.0 - 2.8 * asymmetry)
    response_guard = min(dead_guard, 0.78 + 0.22 * available_scale, 1.0 - 4.5 * asymmetry)
    return {
        'fault_guard_scale': float(np.clip(authority_guard, 0.55, 1.0)),
        'fault_response_scale': float(np.clip(response_guard, 0.40, 1.0)),
        'fault_available_scale': float(available_scale),
        'fault_asymmetry': float(np.clip(asymmetry, 0.0, 1.0)),
        'dead_align_scale': float(np.clip(dead_align_scale, 1.0, 1.5)),
        'dead_cross_scale': float(np.clip(dead_cross_scale, 0.70, 1.0)),
        'dead_align_speed_floor_mps': float(np.clip(dead_align_speed_floor_mps, 0.55, 0.85)),
        'plenum_power_trim': float(np.clip(plenum_power_trim, 0.84, 1.0)),
        'plenum_revector_trim': float(np.clip(plenum_revector_trim, 0.88, 1.0)),
        'plenum_align_speed_floor_mps': float(np.clip(plenum_align_speed_floor_mps, 0.0, 0.65)),
        'plenum_brake_trim': float(np.clip(plenum_brake_trim, 0.88, 1.0)),
    }


def guidance_force_budget(power_state: PowerSystemState, state: AllocatorState, geom: RingGeometry, power: PowerSystemParams, fxy_max_n: float, fault: FaultSpec | None = None, topology: RingActuatorTopology | None = None) -> tuple[float, dict[str, float]]:
    alpha_limit_rad = max(1e-6, math.radians(geom.alpha_max_deg))
    flap_use = float(np.max(np.abs(state.alpha_rad)) / alpha_limit_rad) if state.alpha_rad.size else 0.0
    flap_guard = 1.0 - 0.60 * smoothstep01((flap_use - 0.60) / 0.22)
    flap_guard = float(np.clip(flap_guard, 0.42, 1.0))

    if power.continuous_power_w > 1e-6:
        continuous_ratio_raw = float(power_state.power_w) / power.continuous_power_w
    else:
        continuous_ratio_raw = 0.0
    burst_capacity = burst_capacity_j(power)
    burst_reserve_j = float(getattr(power_state, 'burst_reserve_j', burst_capacity))
    burst_reserve_j = float(np.clip(burst_reserve_j, 0.0, burst_capacity)) if burst_capacity > 0.0 else 0.0
    burst_reserve_frac = burst_reserve_j / burst_capacity if burst_capacity > 1e-9 else 1.0
    continuous_ratio = sustained_power_ratio(power, float(power_state.power_w), burst_reserve_j)
    power_guard = 1.0 - 0.62 * smoothstep01((continuous_ratio - 0.80) / 0.18)
    power_guard = float(np.clip(power_guard, 0.40, 1.0))

    thermal_guard = float(np.clip(np.mean(power_state.thermal_scale), max(power.min_thermal_scale, 0.55), 1.0))
    supply_guard = float(np.clip(power_state.thrust_scale, 0.60, 1.0))
    fault_guard = fault_motion_guard(geom, fault, topology=topology)
    combined_guard = min(flap_guard, power_guard, thermal_guard, supply_guard, fault_guard['fault_guard_scale'])
    budget_ratio = float(np.clip(combined_guard, 0.25, 1.0))
    budget_n = float(max(0.25 * fxy_max_n, budget_ratio * fxy_max_n))
    return budget_n, {
        'guard_scale': float(combined_guard),
        'budget_ratio': float(budget_ratio),
        'flap_guard_scale': float(flap_guard),
        'power_guard_scale': float(power_guard),
        'thermal_guard_scale': float(thermal_guard),
        'supply_guard_scale': float(supply_guard),
        'fault_guard_scale': float(fault_guard['fault_guard_scale']),
        'fault_response_scale': float(fault_guard['fault_response_scale']),
        'fault_available_scale': float(fault_guard['fault_available_scale']),
        'fault_asymmetry_pct': float(100.0 * fault_guard['fault_asymmetry']),
        'dead_align_scale': float(fault_guard['dead_align_scale']),
        'dead_cross_scale': float(fault_guard['dead_cross_scale']),
        'dead_align_speed_floor_mps': float(fault_guard['dead_align_speed_floor_mps']),
        'plenum_power_trim': float(fault_guard['plenum_power_trim']),
        'plenum_revector_trim': float(fault_guard['plenum_revector_trim']),
        'plenum_align_speed_floor_mps': float(fault_guard['plenum_align_speed_floor_mps']),
        'plenum_brake_trim': float(fault_guard['plenum_brake_trim']),
        'continuous_power_ratio': float(continuous_ratio),
        'continuous_power_raw_ratio': float(continuous_ratio_raw),
        'burst_reserve_ratio': float(burst_reserve_frac),
        'flap_usage_ratio': float(flap_use),
    }


def apply_power_system(
    thrust_cmd_n,
    thrust_pre_power_n,
    alpha_cmd_rad,
    alpha_actual_rad,
    ft_actual_n,
    sim: SimParams,
    geom: RingGeometry,
    power: PowerSystemParams,
    power_state: PowerSystemState,
    dt_s: float,
    topology: RingActuatorTopology | None = None,
):
    thrust_cmd = np.asarray(thrust_cmd_n, dtype=float)
    thrust_pre_power = np.asarray(thrust_pre_power_n, dtype=float)
    alpha_cmd = np.asarray(alpha_cmd_rad, dtype=float)
    alpha_actual = np.asarray(alpha_actual_rad, dtype=float)
    ft_actual = np.asarray(ft_actual_n, dtype=float)

    hover_per_seg_n = max(1e-6, sim.mass_kg * sim.gravity / max(1, geom.n_segments))
    alpha_limit_rad = max(1e-6, math.radians(geom.alpha_max_deg))
    tangential_scale_n = max(1e-6, 0.35 * hover_per_seg_n)
    active_topology = default_ring_topology(geom.n_segments) if topology is None else topology
    fan_count = active_topology.fan_count

    flap_activity = float(np.mean(np.abs(alpha_actual)) / alpha_limit_rad) if alpha_actual.size else 0.0
    flap_tracking = float(np.mean(np.abs(alpha_cmd - alpha_actual)) / alpha_limit_rad) if alpha_actual.size else 0.0
    tangential_activity = float(np.mean(np.abs(ft_actual)) / tangential_scale_n) if ft_actual.size else 0.0
    load_multiplier = (
        1.0
        + power.flap_power_scale * flap_activity
        + 0.05 * flap_tracking
        + power.tangential_power_scale * tangential_activity
    )

    active_topology = default_ring_topology(geom.n_segments) if topology is None else topology
    fan_cmd_mean = np.asarray(active_topology.segment_values_to_fan_means(thrust_cmd), dtype=float)
    fan_pre_power_mean = np.asarray(active_topology.segment_values_to_fan_means(thrust_pre_power), dtype=float)
    if fan_cmd_mean.size != fan_count:
        fan_cmd_mean = np.resize(fan_cmd_mean, fan_count)
    if fan_pre_power_mean.size != fan_count:
        fan_pre_power_mean = np.resize(fan_pre_power_mean, fan_count)

    prev_fan_mean = np.asarray(power_state.fan_mean_n, dtype=float)
    if prev_fan_mean.size != fan_count:
        prev_fan_mean = np.zeros(fan_count, dtype=float)
    prev_fan_temp = np.asarray(power_state.fan_temp_c, dtype=float)
    if prev_fan_temp.size != fan_count:
        prev_fan_temp = np.full(fan_count, power.ambient_temp_c, dtype=float)
    prev_thermal_scale = np.asarray(power_state.thermal_scale, dtype=float)
    if prev_thermal_scale.size != fan_count:
        prev_thermal_scale = np.ones(fan_count, dtype=float)

    soc_frac = max(0.0, min(1.0, float(power_state.soc_frac)))
    span_v = max(1e-6, power.battery_full_v - power.battery_empty_v)
    open_circuit_v = power.battery_empty_v + span_v * soc_frac
    supply_scale = power.min_supply_scale + (1.0 - power.min_supply_scale) * smoothstep01((float(power_state.voltage_v) - power.battery_empty_v) / span_v)
    supply_scale = max(power.min_supply_scale, min(1.0, float(supply_scale)))

    fan_target_mean = np.clip(fan_pre_power_mean, 0.0, None) * supply_scale * prev_thermal_scale
    spool_k = 1.0 if power.fan_spool_tau_s <= 1e-6 else min(1.0, dt_s / power.fan_spool_tau_s)
    fan_actual_mean = prev_fan_mean + (fan_target_mean - prev_fan_mean) * spool_k

    burst_capacity = burst_capacity_j(power)
    burst_reserve_j = float(getattr(power_state, 'burst_reserve_j', burst_capacity))
    burst_reserve_j = float(np.clip(burst_reserve_j, 0.0, burst_capacity)) if burst_capacity > 0.0 else 0.0

    fan_actual_mean, thrust_actual, power_w, peak_clip_scale = clip_main_power_to_budget(
        fan_actual_mean,
        thrust_pre_power,
        load_multiplier,
        power.peak_power_w,
        power,
        active_topology,
        hover_per_seg_n,
    )
    continuous_limit_w = max(power.continuous_power_w, 1e-6)
    continuous_power_raw_pct = 100.0 * power_w / continuous_limit_w
    burst_clip_pct = 0.0
    burst_active_time_s = 0.0
    if power_w > power.continuous_power_w and dt_s > 0.0:
        burst_active_time_s = float(dt_s)
        burst_needed_j = (power_w - power.continuous_power_w) * dt_s
        if burst_capacity <= 1e-9:
            allowed_power_w = power.continuous_power_w
        else:
            available_burst_j = max(0.0, burst_reserve_j)
            if burst_needed_j <= available_burst_j + 1e-9:
                burst_reserve_j = max(0.0, available_burst_j - burst_needed_j)
                allowed_power_w = power_w
            else:
                allowed_power_w = power.continuous_power_w + available_burst_j / dt_s
                burst_reserve_j = 0.0
        if allowed_power_w + 1e-9 < power_w:
            burst_clip_pct = 100.0 * max(0.0, power_w - allowed_power_w) / max(power_w, 1e-6)
            fan_actual_mean, thrust_actual, power_w, _burst_scale = clip_main_power_to_budget(
                fan_actual_mean,
                thrust_pre_power,
                load_multiplier,
                min(power.peak_power_w, allowed_power_w),
                power,
                active_topology,
                hover_per_seg_n,
            )
    elif burst_capacity > 1e-9 and power.burst_recharge_tau_s > 1e-6 and power_w < power.continuous_power_w:
        headroom_j = (power.continuous_power_w - power_w) * dt_s
        recharge_cap_j = burst_capacity - burst_reserve_j
        recharge_tau_j = burst_capacity * dt_s / power.burst_recharge_tau_s
        burst_reserve_j += min(recharge_cap_j, headroom_j, recharge_tau_j)

    burst_reserve_j = float(np.clip(burst_reserve_j, 0.0, burst_capacity)) if burst_capacity > 0.0 else 0.0
    sustained_power_pct = 100.0 * sustained_power_ratio(power, power_w, burst_reserve_j)
    burst_reserve_pct = 100.0 * burst_reserve_j / burst_capacity if burst_capacity > 1e-9 else 100.0

    fan_power_share_w = max(0.0, power_w - power.aux_power_w) / max(1, fan_count)
    overload_ratio = max(0.0, power_w / max(power.continuous_power_w, 1e-6) - 1.0)
    heating_w = fan_power_share_w * (power.fan_heat_fraction + power.overload_heat_gain * overload_ratio)
    cooling_w = power.fan_cooling_w_per_c * np.maximum(prev_fan_temp - power.ambient_temp_c, 0.0)
    fan_temp_next = prev_fan_temp + (heating_w - cooling_w) * dt_s / max(power.fan_thermal_capacity_j_per_c, 1e-6)
    thermal_scale_next = temperature_to_thermal_scale(fan_temp_next, power)

    current_a = power_w / max(open_circuit_v, 1.0)
    loaded_voltage_v = max(power.battery_empty_v * 0.5, open_circuit_v - current_a * power.internal_resistance_ohm)
    supply_scale_next = power.min_supply_scale + (1.0 - power.min_supply_scale) * smoothstep01((loaded_voltage_v - power.battery_empty_v) / span_v)
    supply_scale_next = max(power.min_supply_scale, min(1.0, float(supply_scale_next)))

    energy_used_wh = float(power_state.energy_used_wh) + power_w * dt_s / 3600.0
    capacity_wh = max(1e-6, power.capacity_kwh * 1000.0)
    soc_frac = max(0.0, 1.0 - energy_used_wh / capacity_wh)
    open_circuit_v = power.battery_empty_v + span_v * soc_frac
    loaded_voltage_v = max(power.battery_empty_v * 0.5, open_circuit_v - current_a * power.internal_resistance_ohm)

    fan_response_pct = 100.0
    active_fans = fan_cmd_mean > 1e-6
    if np.any(active_fans):
        fan_response_pct = float(100.0 * np.mean(fan_actual_mean[active_fans] / fan_cmd_mean[active_fans]))

    telemetry = {
        "battery_v": float(loaded_voltage_v),
        "battery_a": float(current_a),
        "battery_soc": float(100.0 * soc_frac),
        "power_w": float(power_w),
        "energy_wh": float(energy_used_wh),
        "thrust_scale_pct": float(100.0 * min(supply_scale_next, float(np.mean(thermal_scale_next)))) ,
        "continuous_power_pct": float(sustained_power_pct),
        "continuous_power_raw_pct": float(continuous_power_raw_pct),
        "sustained_power_pct": float(sustained_power_pct),
        "burst_reserve_pct": float(burst_reserve_pct),
        "burst_clip_pct": float(burst_clip_pct),
        "burst_active_time_s": float(burst_active_time_s),
        "power_margin_kw": float((power.continuous_power_w - power_w) / 1000.0),
        "thermal_scale_pct": float(100.0 * np.mean(thermal_scale_next)),
        "fan_temp_max_c": float(np.max(fan_temp_next)) if fan_temp_next.size else float(power.ambient_temp_c),
        "fan_temp_mean_c": float(np.mean(fan_temp_next)) if fan_temp_next.size else float(power.ambient_temp_c),
        "fan_response_pct": float(fan_response_pct),
        "fan_cmd_16": [float(v) for v in fan_cmd_mean],
        "fan_actual_16": [float(v) for v in fan_actual_mean],
        "fan_temp_c_16": [float(v) for v in fan_temp_next],
        "fan_thermal_scale_16": [float(100.0 * v) for v in thermal_scale_next],
        "alpha_cmd_deg_32": [float(v) for v in np.degrees(alpha_cmd)],
        "alpha_cmd_rms": float(np.sqrt(np.mean(np.degrees(alpha_cmd) ** 2))) if alpha_cmd.size else 0.0,
    }
    updated_state = PowerSystemState(
        soc_frac=soc_frac,
        voltage_v=float(loaded_voltage_v),
        current_a=float(current_a),
        power_w=float(power_w),
        energy_used_wh=float(energy_used_wh),
        thrust_scale=float(min(supply_scale_next, float(np.mean(thermal_scale_next)))),
        burst_reserve_j=float(burst_reserve_j),
        fan_mean_n=np.asarray(fan_actual_mean, dtype=float),
        fan_temp_c=np.asarray(fan_temp_next, dtype=float),
        thermal_scale=np.asarray(thermal_scale_next, dtype=float),
    )
    return thrust_actual, updated_state, telemetry

