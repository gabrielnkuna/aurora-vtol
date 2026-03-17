from __future__ import annotations

import math
from dataclasses import dataclass

from .dynamics import AllocatorState, ActuatorLimits, PlenumModel
from .faults import FaultSpec
from .model import RingGeometry, segment_angles_rad
from .power_system import PowerSystemParams, init_hover_power_state, smoothstep01
from .sim_runtime import SimParams, SimState
from ..effectiveness import effectiveness_table_for_topology
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
