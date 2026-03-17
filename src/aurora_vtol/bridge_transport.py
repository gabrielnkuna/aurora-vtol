from __future__ import annotations

import math
import time

from aurora_vtol.bridge_artifacts import BridgeSetpoint


def _import_mavutil():
    try:
        from pymavlink import mavutil
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'pymavlink is required for live MAVLink bridge streaming. Use --jsonl-out/--dry-run or install pymavlink in the active environment.'
        ) from exc
    return mavutil


def _position_target_type_mask(mavutil) -> int:
    return (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )


def _send_bridge_heartbeat(mv, mavutil) -> None:
    mv.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        0,
    )


def _stream_script_to_conn(
    conn,
    mavutil,
    script: list[BridgeSetpoint],
    target_system: int,
    target_component: int,
    heartbeat_hz: float = 1.0,
    realtime: bool = True,
) -> None:
    mv = conn.mav
    type_mask = _position_target_type_mask(mavutil)
    start_wall = time.time()
    last_hb_wall = -1e9
    first_time_s = script[0].time_s

    for sp in script:
        if realtime:
            target_wall = start_wall + (sp.time_s - first_time_s)
            while True:
                now = time.time()
                if now >= target_wall:
                    break
                time.sleep(min(0.05, target_wall - now))

        now = time.time()
        if heartbeat_hz > 0.0 and (now - last_hb_wall) >= (1.0 / heartbeat_hz):
            _send_bridge_heartbeat(mv, mavutil)
            last_hb_wall = now

        mv.set_position_target_local_ned_send(
            int(sp.time_s * 1000.0),
            target_system,
            target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            sp.north_m,
            sp.east_m,
            sp.down_m,
            sp.v_north_mps,
            sp.v_east_mps,
            sp.v_down_mps,
            0.0,
            0.0,
            0.0,
            math.radians(sp.yaw_deg),
            0.0,
        )


def stream_script_mavlink(
    script: list[BridgeSetpoint],
    out: str,
    target_system: int,
    target_component: int,
    source_system: int,
    source_component: int,
    heartbeat_hz: float = 1.0,
    realtime: bool = True,
) -> None:
    mavutil = _import_mavutil()
    conn = mavutil.mavlink_connection(
        out,
        source_system=source_system,
        source_component=source_component,
    )
    try:
        _stream_script_to_conn(
            conn,
            mavutil,
            script,
            target_system=target_system,
            target_component=target_component,
            heartbeat_hz=heartbeat_hz,
            realtime=realtime,
        )
    finally:
        conn.close()
