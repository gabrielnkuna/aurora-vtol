from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class EstimatedVehicleState:
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    yaw_deg: float
    yaw_rate_deg_s: float
    battery_soc_pct: float | None = None
    bus_voltage_v: float | None = None
    continuous_power_ratio: float | None = None
    thermal_scale_pct: float | None = None
    fault_available_scale: float | None = None
    fault_asymmetry_pct: float | None = None


@dataclass(frozen=True)
class GuidanceTarget:
    goal_x_m: float
    goal_y_m: float
    goal_z_m: float = 0.0
    desired_vx_mps: float = 0.0
    desired_vy_mps: float = 0.0
    desired_vz_mps: float = 0.0
    hold_yaw_deg: float = 0.0
    mode: str = "track-position"
    force_scale: float = 1.0


@dataclass(frozen=True)
class RedirectTarget:
    desired_ux: float
    desired_uy: float
    desired_speed_mps: float
    reference_speed_mps: float
    cross_gain: float = 1.0
    hold_yaw_deg: float = 0.0
    phase: str = "redirect"
    force_scale: float = 1.0


@dataclass(frozen=True)
class ActuatorHealthState:
    lateral_budget_n: float
    guard_scale: float = 1.0
    response_scale: float = 1.0
    continuous_power_ratio: float = 0.0
    thermal_scale_pct: float = 100.0
    supply_scale_pct: float = 100.0
    fault_available_scale: float = 1.0
    fault_asymmetry_pct: float = 0.0


@dataclass(frozen=True)
class DesiredWrench:
    fx_n: float
    fy_n: float
    fz_n: float
    mz_nm: float = 0.0
    source: str = "vehicle-controller"


@dataclass(frozen=True)
class AllocatorCommand:
    fan_thrust_cmd_n_16: tuple[float, ...]
    vane_angle_cmd_rad_32: tuple[float, ...]
    tangential_cmd_n_32: tuple[float, ...]
    note: str = ""
