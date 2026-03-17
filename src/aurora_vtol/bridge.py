from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from pathlib import Path


@dataclass(frozen=True)
class BridgeSetpoint:
    time_s: float
    north_m: float
    east_m: float
    down_m: float
    v_north_mps: float
    v_east_mps: float
    v_down_mps: float
    yaw_deg: float
    phase: str


def _list_or_default(hist: dict, key: str, count: int, default=0.0):
    values = hist.get(key, [])
    if not values:
        return [default] * count
    if len(values) >= count:
        return list(values[:count])
    out = list(values)
    out.extend([values[-1]] * (count - len(out)))
    return out


def load_trace_json(path: str | Path) -> tuple[dict, dict]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data.get('meta', {}), data.get('hist', {})


def script_metadata_from_trace_meta(meta: dict) -> dict:
    return {
        'trace_version': meta.get('version'),
        'profile': meta.get('profile'),
        'mission': meta.get('mission'),
        'hardware_assumptions': meta.get('hardware_assumptions'),
    }


def build_setpoint_script(meta: dict, hist: dict, hold_final_s: float = 2.0, rate_hz: float | None = None) -> list[BridgeSetpoint]:
    t = [float(v) for v in hist.get('t', [])]
    if not t:
        raise ValueError('trace contains no hist.t values')

    n = len(t)
    east = [float(v) for v in _list_or_default(hist, 'x', n, 0.0)]
    north = [float(v) for v in _list_or_default(hist, 'y', n, 0.0)]
    up = [float(v) for v in _list_or_default(hist, 'z', n, 0.0)]
    v_east = [float(v) for v in _list_or_default(hist, 'vx', n, 0.0)]
    v_north = [float(v) for v in _list_or_default(hist, 'vy', n, 0.0)]
    v_up = [float(v) for v in _list_or_default(hist, 'vz', n, 0.0)]
    yaw = [float(v) for v in _list_or_default(hist, 'yaw_deg', n, 0.0)]
    phases = _list_or_default(hist, 'mission_phase', n, 'track')

    if rate_hz is not None and rate_hz > 0.0 and n > 1:
        native_dt = max(1e-6, (t[-1] - t[0]) / max(1, n - 1))
        native_rate_hz = 1.0 / native_dt
        stride = max(1, int(round(native_rate_hz / rate_hz)))
        indexes = list(range(0, n, stride))
        if indexes[-1] != n - 1:
            indexes.append(n - 1)
    else:
        indexes = list(range(n))

    script = [
        BridgeSetpoint(
            time_s=float(t[i]),
            north_m=north[i],
            east_m=east[i],
            down_m=-up[i],
            v_north_mps=v_north[i],
            v_east_mps=v_east[i],
            v_down_mps=-v_up[i],
            yaw_deg=yaw[i],
            phase=str(phases[i]),
        )
        for i in indexes
    ]

    if hold_final_s > 0.0:
        final = script[-1]
        script.append(
            BridgeSetpoint(
                time_s=final.time_s + float(hold_final_s),
                north_m=final.north_m,
                east_m=final.east_m,
                down_m=final.down_m,
                v_north_mps=0.0,
                v_east_mps=0.0,
                v_down_mps=0.0,
                yaw_deg=final.yaw_deg,
                phase='hold-final',
            )
        )

    return script


def summarize_script(meta: dict, script: list[BridgeSetpoint]) -> dict:
    if not script:
        raise ValueError('bridge script is empty')

    mission = meta.get('mission', {})
    planner = meta.get('planner', {})
    phases: dict[str, int] = {}
    for sp in script:
        phases[sp.phase] = phases.get(sp.phase, 0) + 1

    return {
        'samples': len(script),
        'duration_s': script[-1].time_s,
        'start': {
            'north_m': script[0].north_m,
            'east_m': script[0].east_m,
            'up_m': -script[0].down_m,
        },
        'final': {
            'north_m': script[-1].north_m,
            'east_m': script[-1].east_m,
            'up_m': -script[-1].down_m,
            'yaw_deg': script[-1].yaw_deg,
        },
        'mission': mission,
        'planner': planner,
        'hardware_assumptions': meta.get('hardware_assumptions'),
        'phase_counts': phases,
    }


def summarize_setpoints(script: list[BridgeSetpoint], metadata: dict | None = None) -> dict:
    if not script:
        raise ValueError('bridge script is empty')
    return {
        'samples': len(script),
        'duration_s': script[-1].time_s - script[0].time_s,
        'start': asdict(script[0]),
        'final': asdict(script[-1]),
        'phase_counts': _phase_counts(script),
        'metadata': metadata or {},
    }


def write_script_jsonl(path: str | Path, script: list[BridgeSetpoint], metadata: dict | None = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        if metadata:
            f.write(json.dumps({'kind': 'bridge_meta', 'metadata': metadata}) + '\n')
        for sp in script:
            f.write(json.dumps(asdict(sp)) + '\n')


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


def load_script_jsonl(path: str | Path) -> list[BridgeSetpoint]:
    script_path = Path(path)
    script: list[BridgeSetpoint] = []
    for lineno, line in enumerate(script_path.read_text(encoding='utf-8').splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get('kind', '')).lower() == 'bridge_meta':
            continue
        try:
            script.append(
                BridgeSetpoint(
                    time_s=float(row['time_s']),
                    north_m=float(row['north_m']),
                    east_m=float(row['east_m']),
                    down_m=float(row['down_m']),
                    v_north_mps=float(row['v_north_mps']),
                    v_east_mps=float(row['v_east_mps']),
                    v_down_mps=float(row['v_down_mps']),
                    yaw_deg=float(row['yaw_deg']),
                    phase=str(row.get('phase', 'track')),
                )
            )
        except KeyError as exc:
            raise ValueError(f'script JSONL missing field {exc} on line {lineno}') from exc
    if not script:
        raise ValueError('bridge script JSONL is empty')
    return script


def load_script_jsonl_metadata(path: str | Path) -> dict | None:
    script_path = Path(path)
    for line in script_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get('kind', '')).lower() == 'bridge_meta':
            metadata = row.get('metadata')
            return metadata if isinstance(metadata, dict) else {}
    return None


def _phase_counts(script: list[BridgeSetpoint]) -> dict[str, int]:
    phases: dict[str, int] = {}
    for sp in script:
        phases[sp.phase] = phases.get(sp.phase, 0) + 1
    return phases


def inspect_script(script: list[BridgeSetpoint], metadata: dict | None = None) -> dict:
    if not script:
        raise ValueError('bridge script is empty')

    issues: list[str] = []
    notes: list[str] = []
    dts: list[float] = []
    dts_checked: list[float] = []
    horiz_steps: list[float] = []
    yaw_steps: list[float] = []
    horiz_speeds: list[float] = []
    vert_speeds: list[float] = []
    path_length_m = 0.0

    for sp in script:
        horiz_speeds.append(math.hypot(sp.v_north_mps, sp.v_east_mps))
        vert_speeds.append(abs(sp.v_down_mps))

    for i in range(1, len(script)):
        prev = script[i - 1]
        cur = script[i]
        dt = cur.time_s - prev.time_s
        dts.append(dt)
        if dt <= 0.0:
            issues.append(f'non-monotonic time at sample {i}: dt={dt:.6f}')
        elif cur.phase != 'hold-final':
            dts_checked.append(dt)
        dn = cur.north_m - prev.north_m
        de = cur.east_m - prev.east_m
        step_m = math.hypot(dn, de)
        horiz_steps.append(step_m)
        path_length_m += step_m
        yaw_delta = abs(((cur.yaw_deg - prev.yaw_deg + 180.0) % 360.0) - 180.0)
        yaw_steps.append(yaw_delta)

    if len(script) < 2:
        issues.append('script has fewer than 2 samples')

    positive_checked_dts = [dt for dt in dts_checked if dt > 0.0]
    positive_all_dts = [dt for dt in dts if dt > 0.0]
    median_dt = sorted(positive_checked_dts)[len(positive_checked_dts) // 2] if positive_checked_dts else 0.0
    if median_dt <= 0.0 and positive_all_dts:
        median_dt = sorted(positive_all_dts)[len(positive_all_dts) // 2]

    if positive_checked_dts and median_dt > 0.0 and max(positive_checked_dts) > 2.5 * median_dt:
        issues.append(f'large timing gap detected: max_dt={max(positive_checked_dts):.3f}s median_dt={median_dt:.3f}s')

    terminal_hold_gap_s = None
    final = script[-1]
    if final.phase == 'hold-final' and dts:
        terminal_hold_gap_s = dts[-1]
        notes.append('terminal hold-final gap excluded from timing-gap warning')

    final_horiz_speed = math.hypot(final.v_north_mps, final.v_east_mps)
    final_vert_speed = abs(final.v_down_mps)
    if final.phase == 'hold-final':
        if final_horiz_speed > 0.05 or final_vert_speed > 0.05:
            issues.append('hold-final sample still has non-zero velocity')
    elif final_horiz_speed > 0.2 or final_vert_speed > 0.2:
        issues.append('final sample is not settled; no hold-final phase detected')

    return {
        'samples': len(script),
        'duration_s': script[-1].time_s - script[0].time_s,
        'phase_counts': _phase_counts(script),
        'timing': {
            'dt_min_s': min(dts) if dts else None,
            'dt_max_s': max(dts) if dts else None,
            'dt_max_checked_s': max(positive_checked_dts) if positive_checked_dts else None,
            'dt_median_s': median_dt if (dts or dts_checked) else None,
            'terminal_hold_gap_s': terminal_hold_gap_s,
        },
        'motion': {
            'path_length_m': path_length_m,
            'max_step_m': max(horiz_steps) if horiz_steps else 0.0,
            'max_horizontal_speed_mps': max(horiz_speeds) if horiz_speeds else 0.0,
            'max_vertical_speed_mps': max(vert_speeds) if vert_speeds else 0.0,
            'max_yaw_step_deg': max(yaw_steps) if yaw_steps else 0.0,
        },
        'start': asdict(script[0]),
        'final': asdict(final),
        'metadata': metadata or {},
        'notes': notes,
        'issues': issues,
        'ok': not issues,
    }


def write_script_svg(path: str | Path, script: list[BridgeSetpoint], title: str = 'Bridge Script') -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    east_vals = [sp.east_m for sp in script]
    north_vals = [sp.north_m for sp in script]
    min_e, max_e = min(east_vals), max(east_vals)
    min_n, max_n = min(north_vals), max(north_vals)
    width = 960
    height = 760
    margin = 50
    span_e = max(1.0, max_e - min_e)
    span_n = max(1.0, max_n - min_n)
    scale = min((width - 2 * margin) / span_e, (height - 2 * margin) / span_n)

    def project(sp: BridgeSetpoint) -> tuple[float, float]:
        x = margin + (sp.east_m - min_e) * scale
        y = height - margin - (sp.north_m - min_n) * scale
        return x, y

    phase_colors = {
        'bypass': '#f28e2b',
        'cruise': '#4e79a7',
        'approach': '#59a14f',
        'hold': '#e15759',
        'hold-final': '#9c755f',
        'safety': '#af7aa1',
        'track': '#4e79a7',
    }

    segments: list[tuple[str, list[tuple[float, float]]]] = []
    current_phase = script[0].phase
    current_points = [project(script[0])]
    for sp in script[1:]:
        pt = project(sp)
        if sp.phase != current_phase and len(current_points) >= 1:
            current_points.append(pt)
            segments.append((current_phase, current_points))
            current_phase = sp.phase
            current_points = [pt]
        else:
            current_points.append(pt)
    if current_points:
        segments.append((current_phase, current_points))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f3ea"/>',
        f'<text x="{margin}" y="30" font-family="monospace" font-size="20" fill="#1f1f1f">{title}</text>',
        f'<text x="{margin}" y="52" font-family="monospace" font-size="12" fill="#444">east [{min_e:.1f}, {max_e:.1f}] m   north [{min_n:.1f}, {max_n:.1f}] m</text>',
        f'<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="none" stroke="#b8b1a9" stroke-width="1"/>',
    ]

    for phase, pts in segments:
        if len(pts) < 2:
            continue
        color = phase_colors.get(phase, '#333333')
        points_attr = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        svg_lines.append(f'<polyline points="{points_attr}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')

    sx, sy = project(script[0])
    ex, ey = project(script[-1])
    svg_lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#2ca02c"/>')
    svg_lines.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="#d62728"/>')
    svg_lines.append(f'<text x="{sx + 8:.1f}" y="{sy - 8:.1f}" font-family="monospace" font-size="12" fill="#2ca02c">start</text>')
    svg_lines.append(f'<text x="{ex + 8:.1f}" y="{ey - 8:.1f}" font-family="monospace" font-size="12" fill="#d62728">final</text>')

    legend_x = width - 210
    legend_y = margin + 10
    svg_lines.append(f'<rect x="{legend_x - 12}" y="{legend_y - 18}" width="180" height="{24 + 18 * len(phase_colors)}" fill="#fffaf2" stroke="#c8c0b6"/>')
    row_y = legend_y
    for phase, color in phase_colors.items():
        svg_lines.append(f'<line x1="{legend_x}" y1="{row_y}" x2="{legend_x + 22}" y2="{row_y}" stroke="{color}" stroke-width="3"/>')
        svg_lines.append(f'<text x="{legend_x + 30}" y="{row_y + 4}" font-family="monospace" font-size="12" fill="#333">{phase}</text>')
        row_y += 18

    svg_lines.append('</svg>')
    out_path.write_text('\n'.join(svg_lines) + '\n', encoding='utf-8')



def _xy_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _local_xy_to_latlon(home_lat_deg: float, home_lon_deg: float, east_m: float, north_m: float) -> tuple[float, float]:
    earth_radius_m = 6378137.0
    lat_deg = float(home_lat_deg) + math.degrees(float(north_m) / earth_radius_m)
    lon_scale = max(1e-9, math.cos(math.radians(lat_deg)))
    lon_deg = float(home_lon_deg) + math.degrees(float(east_m) / (earth_radius_m * lon_scale))
    return lat_deg, lon_deg


def _sample_trace_route_points(hist: dict, spacing_m: float) -> list[tuple[float, float]]:
    east_vals = [float(v) for v in hist.get('x', [])]
    north_vals = [float(v) for v in hist.get('y', [])]
    phases = [str(v) for v in hist.get('mission_phase', [])]
    if not east_vals or not north_vals or len(east_vals) != len(north_vals):
        raise ValueError('trace requires hist.x and hist.y to build a Mission Planner route')

    route: list[tuple[float, float]] = [(east_vals[0], north_vals[0])]
    last_phase = phases[0] if phases else ''
    for idx in range(1, len(east_vals)):
        point = (east_vals[idx], north_vals[idx])
        phase_changed = bool(phases) and phases[idx] != last_phase
        far_enough = _xy_distance_m(route[-1], point) >= max(1.0, spacing_m)
        if phase_changed or far_enough or idx == len(east_vals) - 1:
            if _xy_distance_m(route[-1], point) >= 0.5 or idx == len(east_vals) - 1:
                route.append(point)
            last_phase = phases[idx] if phases else last_phase
    return route



def _densify_route_points(route: list[tuple[float, float]], spacing_m: float) -> list[tuple[float, float]]:
    if not route:
        return []
    if len(route) == 1:
        return list(route)
    spacing = max(1.0, float(spacing_m))
    dense: list[tuple[float, float]] = [route[0]]
    for start, end in zip(route[:-1], route[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1e-6:
            continue
        steps = max(1, int(math.ceil(seg_len / spacing)))
        for step in range(1, steps + 1):
            t = min(1.0, step / steps)
            point = (float(start[0]) + dx * t, float(start[1]) + dy * t)
            if _xy_distance_m(dense[-1], point) >= 0.5 or step == steps:
                dense.append(point)
    return dense

def _extract_route_points(meta: dict, hist: dict, spacing_m: float, prefer_trace_points: bool = False) -> list[tuple[float, float]]:
    planner_waypoints = meta.get('planner', {}).get('waypoints', []) or []
    planner_route: list[tuple[float, float]] = []
    for wp in planner_waypoints:
        try:
            planner_route.append((float(wp['x_m']), float(wp['y_m'])))
        except (KeyError, TypeError, ValueError):
            continue

    sampled_route: list[tuple[float, float]] = []
    try:
        sampled_route = _sample_trace_route_points(hist, spacing_m=spacing_m)
    except ValueError:
        sampled_route = []

    if prefer_trace_points and sampled_route:
        route = sampled_route
    elif planner_route:
        route = _densify_route_points(planner_route, spacing_m=spacing_m)
    else:
        route = sampled_route or planner_route

    deduped: list[tuple[float, float]] = []
    for point in route:
        if not deduped or _xy_distance_m(deduped[-1], point) >= 0.5:
            deduped.append(point)
    if len(deduped) == 1:
        deduped.append(deduped[0])
    return deduped


def build_qgc_wpl_mission(
    meta: dict,
    hist: dict,
    home_lat_deg: float,
    home_lon_deg: float,
    home_alt_m: float = 0.0,
    takeoff_alt_m: float = 0.0,
    cruise_alt_m: float = 0.0,
    waypoint_spacing_m: float = 10.0,
    land: bool = True,
    prefer_trace_points: bool = False,
    approach_distance_m: float = 18.0,
    descent_alt_m: float = 0.0,
    use_spline: bool = True,
    speed_mps: float = 0.0,
    acceptance_radius_m: float = 4.0,
) -> list[dict]:
    mission_meta = meta.get('mission', {})
    route = _extract_route_points(meta, hist, spacing_m=waypoint_spacing_m, prefer_trace_points=prefer_trace_points)
    start_x_m = float(mission_meta.get('start', {}).get('x_m', route[0][0]))
    start_y_m = float(mission_meta.get('start', {}).get('y_m', route[0][1]))
    dest_x_m = float(mission_meta.get('destination', {}).get('x_m', route[-1][0]))
    dest_y_m = float(mission_meta.get('destination', {}).get('y_m', route[-1][1]))
    dest_z_m = float(mission_meta.get('destination', {}).get('z_m', hist.get('z', [0.0])[-1] if hist.get('z') else 0.0))
    max_hist_z_m = max([float(v) for v in hist.get('z', [0.0])] or [0.0])
    resolved_cruise_alt_m = max(0.0, float(cruise_alt_m or mission_meta.get('cruise_alt_m', 0.0) or max_hist_z_m or takeoff_alt_m))
    resolved_takeoff_alt_m = max(0.0, float(takeoff_alt_m or resolved_cruise_alt_m or max_hist_z_m))
    start_ref = (start_x_m, start_y_m)
    if route and _xy_distance_m(route[0], start_ref) <= 0.5:
        route_nav = route[1:]
    else:
        route_nav = list(route)

    final_point = (dest_x_m, dest_y_m)
    while route_nav and _xy_distance_m(route_nav[-1], final_point) < max(2.0, 0.5 * max(1.0, approach_distance_m)):
        route_nav.pop()
    if not route_nav or _xy_distance_m(route_nav[-1], final_point) > 0.5:
        route_nav.append(final_point)

    items: list[dict] = []

    def add_item(frame: int, command: int, lat_deg: float, lon_deg: float, alt_m: float, param1: float = 0.0, param2: float = 0.0, param3: float = 0.0, param4: float = 0.0):
        items.append(
            {
                'seq': len(items),
                'current': 1 if not items else 0,
                'frame': int(frame),
                'command': int(command),
                'param1': float(param1),
                'param2': float(param2),
                'param3': float(param3),
                'param4': float(param4),
                'x': float(lat_deg),
                'y': float(lon_deg),
                'z': float(alt_m),
                'autocontinue': 1,
            }
        )

    add_item(frame=0, command=16, lat_deg=home_lat_deg, lon_deg=home_lon_deg, alt_m=home_alt_m)

    if resolved_takeoff_alt_m > 0.0:
        add_item(frame=3, command=22, lat_deg=home_lat_deg, lon_deg=home_lon_deg, alt_m=resolved_takeoff_alt_m)

    desired_speed_mps = float(speed_mps or mission_meta.get('max_speed_mps', 0.0) or 0.0)
    if desired_speed_mps > 0.0:
        add_item(frame=2, command=178, lat_deg=0.0, lon_deg=0.0, alt_m=0.0, param1=1.0, param2=desired_speed_mps)

    route_command = 82 if use_spline else 16
    for east_m, north_m in route_nav[:-1]:
        lat_deg, lon_deg = _local_xy_to_latlon(home_lat_deg, home_lon_deg, east_m, north_m)
        add_item(
            frame=3,
            command=route_command,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_m=resolved_cruise_alt_m,
            param2=acceptance_radius_m,
        )

    final_lat_deg, final_lon_deg = _local_xy_to_latlon(home_lat_deg, home_lon_deg, final_point[0], final_point[1])
    if land:
        final_approach_alt_m = max(
            dest_z_m + 2.0,
            float(descent_alt_m) if descent_alt_m > 0.0 else min(resolved_cruise_alt_m, max(4.0, 0.35 * resolved_cruise_alt_m)),
        )
    else:
        final_approach_alt_m = resolved_cruise_alt_m if resolved_cruise_alt_m > 0.0 else max(dest_z_m, resolved_takeoff_alt_m)

    prev_point = route_nav[-2] if len(route_nav) >= 2 else start_ref
    leg_dx = final_point[0] - prev_point[0]
    leg_dy = final_point[1] - prev_point[1]
    leg_dist = math.hypot(leg_dx, leg_dy)
    if land and leg_dist > 5.0 and approach_distance_m > 1.0:
        backoff_m = min(float(approach_distance_m), max(4.0, 0.6 * leg_dist))
        scale = max(0.0, (leg_dist - backoff_m) / leg_dist)
        approach_point = (prev_point[0] + leg_dx * scale, prev_point[1] + leg_dy * scale)
        approach_lat_deg, approach_lon_deg = _local_xy_to_latlon(home_lat_deg, home_lon_deg, approach_point[0], approach_point[1])
        add_item(frame=3, command=16, lat_deg=approach_lat_deg, lon_deg=approach_lon_deg, alt_m=final_approach_alt_m, param2=acceptance_radius_m)

    add_item(frame=3, command=16, lat_deg=final_lat_deg, lon_deg=final_lon_deg, alt_m=final_approach_alt_m, param2=acceptance_radius_m)

    if land:
        add_item(frame=3, command=21, lat_deg=final_lat_deg, lon_deg=final_lon_deg, alt_m=dest_z_m)
    elif abs(dest_z_m - final_approach_alt_m) > 0.1:
        add_item(frame=3, command=16, lat_deg=final_lat_deg, lon_deg=final_lon_deg, alt_m=dest_z_m)

    return items


def summarize_qgc_wpl_mission(items: list[dict], out_path: str | Path, home_lat_deg: float, home_lon_deg: float) -> dict:
    if not items:
        raise ValueError('mission item list is empty')
    final = items[-1]
    return {
        'format': 'QGC WPL 110',
        'out': str(out_path),
        'items': len(items),
        'home': {
            'lat_deg': float(home_lat_deg),
            'lon_deg': float(home_lon_deg),
        },
        'final': {
            'command': int(final['command']),
            'lat_deg': float(final['x']),
            'lon_deg': float(final['y']),
            'alt_m': float(final['z']),
        },
    }


def write_qgc_wpl_mission(path: str | Path, items: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['QGC WPL 110']
    for item in items:
        lines.append(
            '	'.join(
                [
                    str(int(item['seq'])),
                    str(int(item['current'])),
                    str(int(item['frame'])),
                    str(int(item['command'])),
                    f"{float(item['param1']):.6f}",
                    f"{float(item['param2']):.6f}",
                    f"{float(item['param3']):.6f}",
                    f"{float(item['param4']):.6f}",
                    f"{float(item['x']):.8f}",
                    f"{float(item['y']):.8f}",
                    f"{float(item['z']):.6f}",
                    str(int(item['autocontinue'])),
                ]
            )
        )
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
