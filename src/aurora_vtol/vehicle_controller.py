from __future__ import annotations
from dataclasses import dataclass
import math

from .icd import ActuatorHealthState, DesiredWrench, EstimatedVehicleState, GuidanceTarget, RedirectTarget


def _clip_force_xy(fx_n: float, fy_n: float, fxy_max_n: float) -> tuple[float, float, float]:
    mag = math.hypot(fx_n, fy_n)
    limit = max(1e-9, float(fxy_max_n))
    if mag <= limit:
        return float(fx_n), float(fy_n), 1.0
    scale = limit / mag
    return float(fx_n * scale), float(fy_n * scale), float(scale)


@dataclass(frozen=True)
class XYVehicleControllerGains:
    pos_k_n_per_m: float = 120.0
    vel_k_n_per_mps: float = 900.0


@dataclass(frozen=True)
class TrackXYControlResult:
    desired_wrench: DesiredWrench
    raw_fx_n: float
    raw_fy_n: float
    clip_scale: float


def track_xy_position(
    state: EstimatedVehicleState,
    target: GuidanceTarget,
    gains: XYVehicleControllerGains,
    health: ActuatorHealthState,
    fz_n: float,
    mz_nm: float = 0.0,
    source: str = "vehicle-controller.track-xy",
) -> TrackXYControlResult:
    goal_dx_m = float(target.goal_x_m - state.x_m)
    goal_dy_m = float(target.goal_y_m - state.y_m)
    raw_fx_n = float(target.force_scale) * (
        gains.pos_k_n_per_m * goal_dx_m
        + gains.vel_k_n_per_mps * float(target.desired_vx_mps - state.vx_mps)
    )
    raw_fy_n = float(target.force_scale) * (
        gains.pos_k_n_per_m * goal_dy_m
        + gains.vel_k_n_per_mps * float(target.desired_vy_mps - state.vy_mps)
    )
    fx_n, fy_n, clip_scale = _clip_force_xy(raw_fx_n, raw_fy_n, health.lateral_budget_n)
    return TrackXYControlResult(
        desired_wrench=DesiredWrench(
            fx_n=fx_n,
            fy_n=fy_n,
            fz_n=float(fz_n),
            mz_nm=float(mz_nm),
            source=source,
        ),
        raw_fx_n=raw_fx_n,
        raw_fy_n=raw_fy_n,
        clip_scale=clip_scale,
    )


@dataclass(frozen=True)
class RedirectControlResult:
    desired_wrench: DesiredWrench
    raw_fx_n: float
    raw_fy_n: float
    clip_scale: float
    along_speed_mps: float
    cross_speed_mps: float
    command_ux: float
    command_uy: float


def _zero_directional_wrench(
    *,
    fz_n: float,
    mz_nm: float,
    source: str,
) -> DesiredWrench:
    return DesiredWrench(
        fx_n=0.0,
        fy_n=0.0,
        fz_n=float(fz_n),
        mz_nm=float(mz_nm),
        source=source,
    )


def track_redirect_velocity(
    state: EstimatedVehicleState,
    target: RedirectTarget,
    health: ActuatorHealthState,
    fxy_command_n: float,
    mz_nm: float = 0.0,
    source: str = "vehicle-controller.redirect",
) -> RedirectControlResult:
    desired_mag = math.hypot(target.desired_ux, target.desired_uy)
    if desired_mag < 1e-6:
        return RedirectControlResult(
            desired_wrench=_zero_directional_wrench(
                fz_n=0.0,
                mz_nm=mz_nm,
                source=source,
            ),
            raw_fx_n=0.0,
            raw_fy_n=0.0,
            clip_scale=1.0,
            along_speed_mps=0.0,
            cross_speed_mps=0.0,
            command_ux=0.0,
            command_uy=0.0,
        )
    else:
        desired_ux = float(target.desired_ux / desired_mag)
        desired_uy = float(target.desired_uy / desired_mag)

    cross_ux = -desired_uy
    cross_uy = desired_ux
    along_speed = float(state.vx_mps * desired_ux + state.vy_mps * desired_uy)
    cross_speed = float(state.vx_mps * cross_ux + state.vy_mps * cross_uy)
    speed_ref = max(1.0, float(target.reference_speed_mps))

    along_cmd = max(-0.45, min(0.90, (float(target.desired_speed_mps) - along_speed) / speed_ref))
    cross_cmd = max(-1.0, min(1.0, (-cross_speed) / speed_ref))

    raw_x = along_cmd * desired_ux + float(target.cross_gain) * cross_cmd * cross_ux
    raw_y = along_cmd * desired_uy + float(target.cross_gain) * cross_cmd * cross_uy
    raw_mag = math.hypot(raw_x, raw_y)
    if raw_mag < 1e-6:
        raw_x, raw_y = 0.0, 0.0
    if raw_mag > 1.0:
        raw_x /= raw_mag
        raw_y /= raw_mag

    raw_fx_n = float(fxy_command_n) * float(target.force_scale) * raw_x
    raw_fy_n = float(fxy_command_n) * float(target.force_scale) * raw_y
    fx_n, fy_n, clip_scale = _clip_force_xy(raw_fx_n, raw_fy_n, health.lateral_budget_n)
    return RedirectControlResult(
        desired_wrench=DesiredWrench(
            fx_n=fx_n,
            fy_n=fy_n,
            fz_n=0.0,
            mz_nm=float(mz_nm),
            source=source,
        ),
        raw_fx_n=raw_fx_n,
        raw_fy_n=raw_fy_n,
        clip_scale=clip_scale,
        along_speed_mps=along_speed,
        cross_speed_mps=cross_speed,
        command_ux=float(raw_x),
        command_uy=float(raw_y),
    )


@dataclass(frozen=True)
class DirectionalForceControlResult:
    desired_wrench: DesiredWrench
    raw_fx_n: float
    raw_fy_n: float
    clip_scale: float
    command_ux: float
    command_uy: float


def command_directional_force(
    command_ux: float,
    command_uy: float,
    health: ActuatorHealthState,
    fxy_command_n: float,
    force_scale: float = 1.0,
    fz_n: float = 0.0,
    mz_nm: float = 0.0,
    source: str = "vehicle-controller.directional",
) -> DirectionalForceControlResult:
    mag = math.hypot(command_ux, command_uy)
    if mag < 1e-6:
        return DirectionalForceControlResult(
            desired_wrench=_zero_directional_wrench(
                fz_n=fz_n,
                mz_nm=mz_nm,
                source=source,
            ),
            raw_fx_n=0.0,
            raw_fy_n=0.0,
            clip_scale=1.0,
            command_ux=0.0,
            command_uy=0.0,
        )
    else:
        ux = float(command_ux / mag)
        uy = float(command_uy / mag)
    raw_fx_n = float(fxy_command_n) * float(force_scale) * ux
    raw_fy_n = float(fxy_command_n) * float(force_scale) * uy
    fx_n, fy_n, clip_scale = _clip_force_xy(raw_fx_n, raw_fy_n, health.lateral_budget_n)
    return DirectionalForceControlResult(
        desired_wrench=DesiredWrench(
            fx_n=fx_n,
            fy_n=fy_n,
            fz_n=float(fz_n),
            mz_nm=float(mz_nm),
            source=source,
        ),
        raw_fx_n=raw_fx_n,
        raw_fy_n=raw_fy_n,
        clip_scale=clip_scale,
        command_ux=ux,
        command_uy=uy,
    )


def track_step_snap_brake(
    state: EstimatedVehicleState,
    target_ux: float,
    target_uy: float,
    nominal_brake_ux: float,
    nominal_brake_uy: float,
    steer_mix: float,
    health: ActuatorHealthState,
    fxy_command_n: float,
    brake_gain: float,
    fz_n: float,
    mz_nm: float = 0.0,
    source: str = "vehicle-controller.step-snap-brake",
) -> DirectionalForceControlResult:
    brake_mag = math.hypot(-state.vx_mps, -state.vy_mps)
    if brake_mag < 1e-6:
        brake_ux, brake_uy = float(nominal_brake_ux), float(nominal_brake_uy)
    else:
        brake_ux = float((-state.vx_mps) / brake_mag)
        brake_uy = float((-state.vy_mps) / brake_mag)
    cmd_x = (1.0 - float(steer_mix)) * brake_ux + float(steer_mix) * float(target_ux)
    cmd_y = (1.0 - float(steer_mix)) * brake_uy + float(steer_mix) * float(target_uy)
    return command_directional_force(
        cmd_x,
        cmd_y,
        health,
        fxy_command_n=fxy_command_n,
        force_scale=float(brake_gain),
        fz_n=fz_n,
        mz_nm=mz_nm,
        source=source,
    )


def track_step_snap_reverse(
    state: EstimatedVehicleState,
    target_ux: float,
    target_uy: float,
    reference_speed_mps: float,
    desired_along_speed_mps: float,
    cross_gain: float,
    health: ActuatorHealthState,
    fxy_command_n: float,
    force_scale: float,
    fz_n: float,
    mz_nm: float = 0.0,
    source: str = "vehicle-controller.step-snap-reverse",
) -> DirectionalForceControlResult:
    cross_ux = -float(target_uy)
    cross_uy = float(target_ux)
    along_speed = float(state.vx_mps * target_ux + state.vy_mps * target_uy)
    cross_speed = float(state.vx_mps * cross_ux + state.vy_mps * cross_uy)
    speed_ref = max(1.0, float(reference_speed_mps))
    along_cmd = max(-0.55, min(1.0, (float(desired_along_speed_mps) - along_speed) / speed_ref))
    cross_cmd = max(-1.0, min(1.0, (-cross_speed) / speed_ref))
    raw_x = along_cmd * float(target_ux) + float(cross_gain) * cross_cmd * cross_ux
    raw_y = along_cmd * float(target_uy) + float(cross_gain) * cross_cmd * cross_uy
    return command_directional_force(
        raw_x,
        raw_y,
        health,
        fxy_command_n=fxy_command_n,
        force_scale=float(force_scale),
        fz_n=fz_n,
        mz_nm=mz_nm,
        source=source,
    )
