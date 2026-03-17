from __future__ import annotations

import math
import time

from aurora_vtol.bridge_artifacts import (
    BridgeSetpoint,
    build_qgc_wpl_mission,
    build_setpoint_script,
    inspect_script,
    load_script_jsonl,
    load_script_jsonl_metadata,
    load_trace_json,
    script_metadata_from_trace_meta,
    summarize_qgc_wpl_mission,
    summarize_script,
    summarize_setpoints,
    write_qgc_wpl_mission,
    write_script_jsonl,
    write_script_svg,
)


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


def _resolve_target_ids(conn, heartbeat, target_system: int, target_component: int) -> tuple[int, int]:
    resolved_system = int(target_system or getattr(conn, 'target_system', 0) or heartbeat.get_srcSystem())
    resolved_component = int(target_component or getattr(conn, 'target_component', 0) or heartbeat.get_srcComponent())
    return resolved_system, resolved_component


def _wait_command_ack(conn, command_id: int, timeout_s: float):
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        msg = conn.recv_match(type='COMMAND_ACK', blocking=True, timeout=min(1.0, deadline - time.time()))
        if msg is None:
            continue
        if int(getattr(msg, 'command', -1)) == int(command_id):
            return msg
    return None


def _enum_name(mavutil, group: str, value: int) -> str:
    entry = getattr(mavutil.mavlink, 'enums', {}).get(group, {}).get(int(value))
    if entry is None:
        return str(value)
    return str(getattr(entry, 'name', value))


def _ack_result_name(mavutil, result: int) -> str:
    return _enum_name(mavutil, 'MAV_RESULT', result)


def _require_command_accepted(conn, mavutil, command_id: int, timeout_s: float, context: str):
    ack = _wait_command_ack(conn, command_id=command_id, timeout_s=timeout_s)
    if ack is None:
        return None
    accepted = {
        int(mavutil.mavlink.MAV_RESULT_ACCEPTED),
        int(getattr(mavutil.mavlink, 'MAV_RESULT_IN_PROGRESS', mavutil.mavlink.MAV_RESULT_ACCEPTED)),
    }
    result = int(getattr(ack, 'result', -1))
    if result not in accepted:
        raise RuntimeError(f'{context} rejected by autopilot: {_ack_result_name(mavutil, result)}')
    return {
        'command': int(getattr(ack, 'command', command_id)),
        'result': result,
        'result_name': _ack_result_name(mavutil, result),
    }


def _wait_flightmode(conn, expected_mode: str, timeout_s: float) -> None:
    expected = expected_mode.strip().upper()
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        current = str(getattr(conn, 'flightmode', '') or '').upper()
        if current == expected:
            return
        conn.recv_match(type='HEARTBEAT', blocking=True, timeout=min(1.0, deadline - time.time()))
    raise RuntimeError(f"timed out waiting for flight mode {expected_mode!r}; last mode={getattr(conn, 'flightmode', None)!r}")


def _wait_armed_state(conn, armed: bool, timeout_s: float) -> None:
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        if bool(conn.motors_armed()) == armed:
            return
        conn.recv_match(type='HEARTBEAT', blocking=True, timeout=min(1.0, deadline - time.time()))
    state = 'armed' if armed else 'disarmed'
    raise RuntimeError(f'timed out waiting for motors to become {state}')


def _relative_alt_from_msg(msg) -> float | None:
    msg_type = msg.get_type()
    if msg_type == 'GLOBAL_POSITION_INT':
        return float(getattr(msg, 'relative_alt', 0.0)) * 1e-3
    if msg_type == 'LOCAL_POSITION_NED':
        return -float(getattr(msg, 'z', 0.0))
    if msg_type == 'VFR_HUD':
        return float(getattr(msg, 'alt', 0.0))
    return None


def _request_message_interval(conn, mavutil, target_system: int, target_component: int, message_id: int, rate_hz: float) -> None:
    if rate_hz <= 0.0:
        return
    interval_us = int(max(1.0, 1_000_000.0 / rate_hz))
    conn.mav.command_long_send(
        target_system,
        target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(message_id),
        float(interval_us),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _request_takeoff_telemetry(conn, mavutil, target_system: int, target_component: int, rate_hz: float) -> None:
    for message_id in (
        int(getattr(mavutil.mavlink, 'MAVLINK_MSG_ID_GLOBAL_POSITION_INT', 33)),
        int(getattr(mavutil.mavlink, 'MAVLINK_MSG_ID_LOCAL_POSITION_NED', 32)),
        int(getattr(mavutil.mavlink, 'MAVLINK_MSG_ID_VFR_HUD', 74)),
    ):
        try:
            _request_message_interval(conn, mavutil, target_system, target_component, message_id=message_id, rate_hz=rate_hz)
        except Exception:
            pass


def _wait_relative_alt_m(conn, target_alt_m: float, timeout_s: float) -> float:
    deadline = time.time() + max(0.0, timeout_s)
    threshold_m = max(0.5, 0.9 * max(0.0, target_alt_m))
    last_alt_m = 0.0
    while time.time() < deadline:
        msg = conn.recv_match(blocking=True, timeout=min(1.0, deadline - time.time()))
        if msg is None:
            continue
        alt_m = _relative_alt_from_msg(msg)
        if alt_m is None:
            continue
        last_alt_m = alt_m
        if last_alt_m >= threshold_m:
            return last_alt_m
    raise RuntimeError(f'timed out waiting to reach {target_alt_m:.2f} m takeoff altitude; last_alt_m={last_alt_m:.2f}')


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


def stream_script_ardupilot_sitl(
    script: list[BridgeSetpoint],
    connect: str,
    target_system: int = 0,
    target_component: int = 0,
    source_system: int = 245,
    source_component: int = 190,
    heartbeat_hz: float = 1.0,
    realtime: bool = True,
    wait_heartbeat_s: float = 30.0,
    mode: str = 'GUIDED',
    mode_timeout_s: float = 10.0,
    arm: bool = True,
    arm_timeout_s: float = 15.0,
    takeoff_alt_m: float = 0.0,
    takeoff_timeout_s: float = 45.0,
    takeoff_message_rate_hz: float = 5.0,
    takeoff_strict: bool = False,
    land_mode: str = '',
    land_timeout_s: float = 20.0,
    disarm: bool = False,
    disarm_timeout_s: float = 15.0,
) -> dict:
    mavutil = _import_mavutil()
    conn = mavutil.mavlink_connection(
        connect,
        source_system=source_system,
        source_component=source_component,
    )

    try:
        heartbeat = conn.wait_heartbeat(timeout=wait_heartbeat_s)
        if heartbeat is None:
            raise RuntimeError(f'timed out waiting for heartbeat on {connect}')

        resolved_system, resolved_component = _resolve_target_ids(
            conn,
            heartbeat,
            target_system=target_system,
            target_component=target_component,
        )

        heartbeat_type = int(getattr(heartbeat, 'type', -1))
        heartbeat_autopilot = int(getattr(heartbeat, 'autopilot', -1))
        result = {
            'connect': connect,
            'target_system': resolved_system,
            'target_component': resolved_component,
            'source_system': source_system,
            'source_component': source_component,
            'heartbeat': {
                'system': heartbeat.get_srcSystem(),
                'component': heartbeat.get_srcComponent(),
                'type': heartbeat_type,
                'type_name': _enum_name(mavutil, 'MAV_TYPE', heartbeat_type),
                'autopilot': heartbeat_autopilot,
                'autopilot_name': _enum_name(mavutil, 'MAV_AUTOPILOT', heartbeat_autopilot),
            },
            'mode': None,
            'armed': False,
            'takeoff_alt_m': float(takeoff_alt_m),
            'takeoff_reached_alt_m': None,
            'takeoff_status': None,
            'land_mode': land_mode or None,
            'disarmed': False,
            'warnings': [],
        }
        if result['heartbeat']['type_name'] == 'MAV_TYPE_FIXED_WING':
            result['warnings'].append(
                'heartbeat reports FIXED_WING; GUIDED takeoff and LOCAL_NED setpoints are multirotor-style and may be rejected or ignored. Use Copter SITL or run with --takeoff-alt-m 0 for link-level testing.'
            )

        if mode:
            conn.set_mode_apm(mode)
            _wait_flightmode(conn, expected_mode=mode, timeout_s=mode_timeout_s)
            result['mode'] = str(getattr(conn, 'flightmode', mode))

        if arm:
            conn.arducopter_arm()
            _wait_armed_state(conn, armed=True, timeout_s=arm_timeout_s)
            result['armed'] = True

        if takeoff_alt_m > 0.0:
            _request_takeoff_telemetry(
                conn,
                mavutil,
                target_system=resolved_system,
                target_component=resolved_component,
                rate_hz=takeoff_message_rate_hz,
            )
            try:
                conn.mav.command_long_send(
                    resolved_system,
                    resolved_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(takeoff_alt_m),
                )
                result['takeoff_ack'] = _require_command_accepted(
                    conn,
                    mavutil,
                    command_id=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    timeout_s=min(5.0, takeoff_timeout_s),
                    context='guided takeoff',
                )
                result['takeoff_reached_alt_m'] = _wait_relative_alt_m(
                    conn,
                    target_alt_m=float(takeoff_alt_m),
                    timeout_s=takeoff_timeout_s,
                )
                result['takeoff_status'] = 'reached-alt'
            except RuntimeError as exc:
                result['takeoff_status'] = 'warning'
                result['warnings'].append(str(exc))
                if takeoff_strict:
                    raise

        _stream_script_to_conn(
            conn,
            mavutil,
            script,
            target_system=resolved_system,
            target_component=resolved_component,
            heartbeat_hz=heartbeat_hz,
            realtime=realtime,
        )

        if land_mode:
            conn.set_mode_apm(land_mode)
            _wait_flightmode(conn, expected_mode=land_mode, timeout_s=land_timeout_s)
            result['land_mode'] = str(getattr(conn, 'flightmode', land_mode))

        if disarm:
            conn.arducopter_disarm()
            _wait_armed_state(conn, armed=False, timeout_s=disarm_timeout_s)
            result['disarmed'] = True

        return result
    finally:
        conn.close()


