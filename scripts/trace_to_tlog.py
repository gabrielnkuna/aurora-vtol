#!/usr/bin/env python3
"""Convert an allocator trace into a Mission Planner-friendly MAVLink replay."""

import argparse
import json
import math
import socket
import time

from pymavlink import mavutil


EARTH_RADIUS_M = 6378137.0
REQUEST_HOME_POSITION = 242
FLAP_SAMPLE_INDEXES = [0, 4, 8, 12, 16, 20, 24, 28]
FULL_ACTUATOR_STRIDE = 5
FAKE_PARAMS = [
    ("SYSID_THISMAV", 1.0),
    ("FRAME_CLASS", 1.0),
    ("FRAME_TYPE", 1.0),
    ("ARMING_CHECK", 0.0),
    ("RTL_ALT", 1500.0),
    ("WP_YAW_BEHAVIOR", 0.0),
    ("AHRS_EKF_TYPE", 3.0),
    ("BATT_MONITOR", 0.0),
    ("GPS_TYPE", 1.0),
]
PARAM_INDEX = {name: idx for idx, (name, _value) in enumerate(FAKE_PARAMS)}


class SocketWriter:
    def __init__(self, sock):
        self.sock = sock

    def write(self, data):
        self.sock.sendall(data)
        return len(data)

    def flush(self):
        return None


class DirectTCPServer:
    def __init__(self, host, port, source_system, source_component):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        print(f'listening -> tcp://{host}:{port}')
        self.sock, addr = self.server.accept()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        print(f'client connected -> {addr}')
        self.writer = SocketWriter(self.sock)
        self.mav = mavutil.mavlink.MAVLink(self.writer, srcSystem=source_system, srcComponent=source_component)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.server.close()
        except Exception:
            pass


def build_connection(out_path, source_system, source_component):
    if out_path.endswith('.tlog') or out_path.endswith('.bin'):
        f = open(out_path, 'wb')
        mv = mavutil.mavlink.MAVLink(f, srcSystem=source_system, srcComponent=source_component)
        return {'mode': 'file', 'mv': mv, 'conn': None, 'closer': f}

    if out_path.startswith('listen:'):
        target = out_path.split(':', 1)[1]
        host, port_text = target.rsplit(':', 1)
        server = DirectTCPServer(host, int(port_text), source_system, source_component)
        return {'mode': 'server', 'mv': server.mav, 'conn': server, 'closer': server}

    m = mavutil.mavlink_connection(
        out_path,
        source_system=source_system,
        source_component=source_component,
    )
    return {'mode': 'network', 'mv': m.mav, 'conn': m, 'closer': None}


def offset_global_position(home_lat_deg, home_lon_deg, east_m, north_m):
    lat = home_lat_deg + math.degrees(north_m / EARTH_RADIUS_M)
    cos_lat = max(math.cos(math.radians(home_lat_deg)), 1e-6)
    lon = home_lon_deg + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    return lat, lon


def wrap_angle_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0


def smoothstep01(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def list_or_default(hist, key, count, default=0.0):
    values = hist.get(key, [])
    if not values:
        return [default] * count
    if len(values) >= count:
        return [float(v) for v in values[:count]]
    out = [float(v) for v in values]
    out.extend([float(values[-1])] * (count - len(out)))
    return out


def hist_scalar(hist, key, idx, default=None):
    values = hist.get(key, [])
    if idx >= len(values):
        return default
    value = values[idx]
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def max_abs(values):
    return max((abs(float(v)) for v in values), default=0.0)


def angle_blend_deg(current, target, alpha):
    delta = wrap_angle_deg(target - current)
    return wrap_angle_deg(current + alpha * delta)


def derivative(values, times):
    out = [0.0] * len(values)
    for i in range(1, len(values)):
        dt = max(1e-6, float(times[i]) - float(times[i - 1]))
        out[i] = (float(values[i]) - float(values[i - 1])) / dt
    if len(out) > 1:
        out[0] = out[1]
    return out


def synthesize_successful_flight(hist, meta, args):
    t = list_or_default(hist, 't', len(hist.get('t', [])), 0.0)
    n = len(t)
    east = list_or_default(hist, 'x', n, 0.0)
    north = list_or_default(hist, 'y', n, 0.0)
    v_east = list_or_default(hist, 'vx', n, 0.0)
    v_north = list_or_default(hist, 'vy', n, 0.0)
    raw_up = list_or_default(hist, 'z', n, 0.0)
    raw_vup = list_or_default(hist, 'vz', n, 0.0)
    raw_yaw = list_or_default(hist, 'yaw_deg', n, 0.0)
    fx_cmd = list_or_default(hist, 'fx_cmd', n, 0.0)
    fy_cmd = list_or_default(hist, 'fy_cmd', n, 0.0)
    speed = list_or_default(hist, 'speed', n, 0.0)

    total_t = float(t[-1]) if t else 0.0
    mass = float(meta.get('sim', {}).get('mass_kg', 800.0))
    gravity = float(meta.get('sim', {}).get('gravity', 9.81))

    use_raw_alt = max_abs(raw_up) > 0.5 or max_abs(raw_vup) > 0.2
    if args.flight_profile == 'raw':
        up = raw_up
        v_up = raw_vup
        used_raw_alt = True
    else:
        if args.flight_profile == 'auto' and use_raw_alt:
            up = raw_up
            v_up = raw_vup
            used_raw_alt = True
        else:
            takeoff_s = min(args.takeoff_s, total_t) if total_t > 0.0 else args.takeoff_s
            up = []
            for ti in t:
                if takeoff_s <= 1e-6:
                    up.append(args.cruise_alt_m)
                else:
                    up.append(args.cruise_alt_m * smoothstep01(ti / takeoff_s))
            v_up = derivative(up, t)
            used_raw_alt = False

    yaw_span = (max(raw_yaw) - min(raw_yaw)) if raw_yaw else 0.0
    yaw_mode = args.yaw_source
    if yaw_mode == 'auto':
        yaw_mode = 'trace' if yaw_span > 5.0 else 'fixed'

    yaw = []
    prev_yaw = raw_yaw[0] if raw_yaw else 0.0
    alpha = min(1.0, (float(meta.get('sim', {}).get('dt_s', 0.02)) / 0.25))
    for i in range(n):
        if yaw_mode == 'trace':
            target = raw_yaw[i]
        elif yaw_mode == 'track':
            if math.hypot(v_east[i], v_north[i]) > 0.15:
                target = math.degrees(math.atan2(v_east[i], v_north[i])) % 360.0
            else:
                target = prev_yaw
        else:
            target = raw_yaw[0] if raw_yaw else 0.0
        prev_yaw = angle_blend_deg(prev_yaw, target, alpha)
        yaw.append(prev_yaw % 360.0)

    if max_abs(fx_cmd) > 1e-6 or max_abs(fy_cmd) > 1e-6:
        a_east = [fx / mass for fx in fx_cmd]
        a_north = [fy / mass for fy in fy_cmd]
    else:
        a_east = derivative(v_east, t)
        a_north = derivative(v_north, t)

    roll = []
    pitch = []
    prev_roll = 0.0
    prev_pitch = 0.0
    roll_limit = math.radians(args.max_roll_deg)
    pitch_limit = math.radians(args.max_pitch_deg)
    for i in range(n):
        psi = math.radians(yaw[i])
        forward_acc = a_north[i] * math.cos(psi) + a_east[i] * math.sin(psi)
        right_acc = a_east[i] * math.cos(psi) - a_north[i] * math.sin(psi)
        target_roll = max(-roll_limit, min(roll_limit, math.atan2(right_acc, gravity)))
        target_pitch = max(-pitch_limit, min(pitch_limit, -math.atan2(forward_acc, gravity)))
        prev_roll += alpha * (target_roll - prev_roll)
        prev_pitch += alpha * (target_pitch - prev_pitch)
        roll.append(prev_roll)
        pitch.append(prev_pitch)

    return {
        't': t,
        'east': east,
        'north': north,
        'up': up,
        'v_east': v_east,
        'v_north': v_north,
        'v_up': v_up,
        'yaw_deg': yaw,
        'roll_rad': roll,
        'pitch_rad': pitch,
        'speed': speed,
        'used_raw_alt': used_raw_alt,
    }


def send_heartbeat(mv, armed=True):
    base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
    base_mode |= mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED
    base_mode |= mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
    if armed:
        base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    custom_mode = 4 if armed else 0
    mv.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        base_mode,
        custom_mode,
        mavutil.mavlink.MAV_STATE_ACTIVE if armed else mavutil.mavlink.MAV_STATE_STANDBY,
    )


def send_sys_status(mv, battery_v=None, battery_a=None, battery_remaining=None):
    sensors = (
        mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_YAW_POSITION
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_Z_ALTITUDE_CONTROL
    )
    voltage_v = 24.0 if battery_v is None else float(battery_v)
    current_a = 1.2 if battery_a is None else float(battery_a)
    remaining_pct = 92.0 if battery_remaining is None else float(battery_remaining)
    mv.sys_status_send(
        sensors,
        sensors,
        sensors,
        0,
        min(65535, int(max(0.0, voltage_v) * 1000.0)),
        min(32767, int(max(0.0, current_a) * 100.0)),
        int(max(0.0, min(100.0, remaining_pct))),
        0,
        0,
        0,
        0,
        0,
        0,
    )


def send_ekf_status(mv):
    flags = (
        mavutil.mavlink.EKF_ATTITUDE
        | mavutil.mavlink.EKF_VELOCITY_HORIZ
        | mavutil.mavlink.EKF_VELOCITY_VERT
        | mavutil.mavlink.EKF_POS_HORIZ_REL
        | mavutil.mavlink.EKF_POS_HORIZ_ABS
        | mavutil.mavlink.EKF_POS_VERT_ABS
        | mavutil.mavlink.EKF_CONST_POS_MODE
        | mavutil.mavlink.EKF_PRED_POS_HORIZ_REL
        | mavutil.mavlink.EKF_PRED_POS_HORIZ_ABS
    )
    mv.ekf_status_report_send(flags, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02)


def send_home_position(mv, time_usec, home_lat_deg, home_lon_deg, home_alt_m):
    q = [1.0, 0.0, 0.0, 0.0]
    lat = int(home_lat_deg * 1e7)
    lon = int(home_lon_deg * 1e7)
    alt_mm = int(home_alt_m * 1000.0)
    mv.home_position_send(
        lat,
        lon,
        alt_mm,
        0.0,
        0.0,
        0.0,
        q,
        0.0,
        0.0,
        0.0,
        int(time_usec),
    )
    try:
        mv.gps_global_origin_send(lat, lon, alt_mm, int(time_usec))
    except TypeError:
        mv.gps_global_origin_send(lat, lon, alt_mm)


def send_gps_raw(mv, time_usec, lat_deg, lon_deg, alt_m, groundspeed_mps, course_deg):
    lat = int(lat_deg * 1e7)
    lon = int(lon_deg * 1e7)
    alt_mm = int(alt_m * 1000.0)
    vel_cms = int(max(groundspeed_mps, 0.0) * 100.0)
    cog_cdeg = int((course_deg % 360.0) * 100.0)
    base = [time_usec, 3, lat, lon, alt_mm, 100, 100, vel_cms, cog_cdeg, 12]
    try:
        mv.gps_raw_int_send(*base, alt_mm, 3000, 3000, 0, 0, 0)
    except TypeError:
        mv.gps_raw_int_send(*base)


def send_named_value(mv, boot_ms, name, value):
    if value is None:
        return
    try:
        v = float(value)
    except Exception:
        return
    mv.named_value_float_send(int(boot_ms), name.encode('ascii')[:10], v)


def send_indexed_named_values(mv, boot_ms, prefix, values, width=2):
    for value_idx, value in enumerate(values):
        send_named_value(mv, boot_ms, f'{prefix}{value_idx:0{width}d}', value)


def should_send_full_actuators(hist, idx):
    sample_count = len(hist.get('t', []))
    if sample_count <= 1:
        return True
    if idx == 0 or idx + 1 >= sample_count:
        return True
    return (idx % FULL_ACTUATOR_STRIDE) == 0


def send_trace_telemetry(mv, boot_ms, hist, idx, profile):
    scalar_specs = {
        'speed': ('speed', 1.0),
        'fx_cmd': ('fx_cmd', 1.0),
        'fy_cmd': ('fy_cmd', 1.0),
        'fz_cmd': ('fz_cmd', 1.0),
        'dist_to_obstacle': ('dist', 1.0),
        'nearest_obstacle_m': ('obs_m', 1.0),
        'dist_to_goal': ('goal_m', 1.0),
        'alpha_deg_rms': ('a_rms', 1.0),
        'alpha_cmd_rms': ('a_cmd', 1.0),
        'ft_tan_rms': ('ft_rms', 1.0),
        'mz_est': ('mz_est', 1.0),
        'vx': ('vx', 1.0),
        'vy': ('vy', 1.0),
        'z': ('z', 1.0),
        'vz': ('vz', 1.0),
        'yaw_rate_deg_s': ('yaw_rt', 1.0),
        'fx_ach': ('fx_ach', 1.0),
        'fy_ach': ('fy_ach', 1.0),
        'fz_ach': ('fz_ach', 1.0),
        'fx_err': ('fx_err', 1.0),
        'fy_err': ('fy_err', 1.0),
        'fz_err': ('fz_err', 1.0),
        'battery_v': ('bat_v', 1.0),
        'battery_a': ('bat_a', 1.0),
        'battery_soc': ('soc_pct', 1.0),
        'power_w': ('pwr_kw', 0.001),
        'energy_wh': ('eng_wh', 1.0),
        'thrust_scale_pct': ('thr_pct', 1.0),
        'continuous_power_pct': ('cont_pct', 1.0),
        'power_margin_kw': ('pwr_marg', 1.0),
        'thermal_scale_pct': ('therm_pct', 1.0),
        'fan_temp_max_c': ('temp_max', 1.0),
        'fan_temp_mean_c': ('temp_avg', 1.0),
        'fan_response_pct': ('rsp_pct', 1.0),
    }
    for key, (label, scale) in scalar_specs.items():
        value = hist_scalar(hist, key, idx, None)
        if value is not None:
            send_named_value(mv, boot_ms, label, value * scale)

    send_named_value(mv, boot_ms, 'roll_deg', math.degrees(profile['roll_rad'][idx]))
    send_named_value(mv, boot_ms, 'pitchdeg', math.degrees(profile['pitch_rad'][idx]))
    send_named_value(mv, boot_ms, 'alt_synth', profile['up'][idx])

    alpha32 = hist.get('alpha_deg_32', [])
    if idx < len(alpha32) and alpha32[idx]:
        vals = alpha32[idx]
        for sample_idx, flap_idx in enumerate(FLAP_SAMPLE_INDEXES):
            if flap_idx < len(vals):
                send_named_value(mv, boot_ms, f'a{sample_idx}', vals[flap_idx])
        if should_send_full_actuators(hist, idx):
            send_indexed_named_values(mv, boot_ms, 'a', vals)

    alpha_cmd32 = hist.get('alpha_cmd_deg_32', [])
    if idx < len(alpha_cmd32) and alpha_cmd32[idx]:
        vals = alpha_cmd32[idx]
        for sample_idx, flap_idx in enumerate(FLAP_SAMPLE_INDEXES):
            if flap_idx < len(vals):
                send_named_value(mv, boot_ms, f'c{sample_idx}', vals[flap_idx])
        if should_send_full_actuators(hist, idx):
            send_indexed_named_values(mv, boot_ms, 'c', vals)

    fan16 = hist.get('fan_thrust_16', [])
    if idx < len(fan16) and fan16[idx]:
        vals = [float(x) for x in fan16[idx]]
        mean = sum(vals) / len(vals)
        send_named_value(mv, boot_ms, 'fan_avg', mean)
        if mean > 1e-9:
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = math.sqrt(var)
            maxdev = max(abs(x - mean) for x in vals)
            send_named_value(mv, boot_ms, 'fan_stdp', 100.0 * std / mean)
            send_named_value(mv, boot_ms, 'fan_maxp', 100.0 * maxdev / mean)
        if should_send_full_actuators(hist, idx):
            send_indexed_named_values(mv, boot_ms, 'f', vals)

    fan_cmd16 = hist.get('fan_cmd_16', [])
    if idx < len(fan_cmd16) and fan_cmd16[idx]:
        vals = [float(x) for x in fan_cmd16[idx]]
        mean = sum(vals) / len(vals)
        send_named_value(mv, boot_ms, 'cmd_avg', mean)
        if mean > 1e-9:
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = math.sqrt(var)
            maxdev = max(abs(x - mean) for x in vals)
            send_named_value(mv, boot_ms, 'cmd_stdp', 100.0 * std / mean)
            send_named_value(mv, boot_ms, 'cmd_maxp', 100.0 * maxdev / mean)
        if should_send_full_actuators(hist, idx):
            send_indexed_named_values(mv, boot_ms, 'fc', vals)


def send_param_value(mv, name, value, index):
    mv.param_value_send(
        name.encode('ascii')[:16].ljust(16, b'\x00'),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        len(FAKE_PARAMS),
        index,
    )


def msg_src_ids(msg):
    src_sys = 255
    src_comp = 0
    try:
        src_sys = int(msg.get_srcSystem())
    except Exception:
        pass
    try:
        src_comp = int(msg.get_srcComponent())
    except Exception:
        pass
    return src_sys, src_comp


def send_home_mission_item(mv, target_system, target_component, home_lat_deg, home_lon_deg, home_alt_m):
    alt = float(home_alt_m)
    mv.mission_item_send(
        target_system,
        target_component,
        0,
        mavutil.mavlink.MAV_FRAME_GLOBAL,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        float(home_lat_deg),
        float(home_lon_deg),
        alt,
    )


def send_home_mission_item_int(mv, target_system, target_component, home_lat_deg, home_lon_deg, home_alt_m):
    lat = int(home_lat_deg * 1e7)
    lon = int(home_lon_deg * 1e7)
    alt = float(home_alt_m)
    mv.mission_item_int_send(
        target_system,
        target_component,
        0,
        mavutil.mavlink.MAV_FRAME_GLOBAL,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        lat,
        lon,
        alt,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )


def process_message(msg, mv, home_lat_deg=0.0, home_lon_deg=0.0, home_alt_m=0.0):
    mtype = msg.get_type()
    if mtype == 'PARAM_REQUEST_LIST':
        for idx, (name, value) in enumerate(FAKE_PARAMS):
            send_param_value(mv, name, value, idx)
    elif mtype == 'PARAM_REQUEST_READ':
        param_index = getattr(msg, 'param_index', -1)
        param_id = getattr(msg, 'param_id', b'')
        if isinstance(param_id, bytes):
            param_name = param_id.split(b'\x00', 1)[0].decode('ascii', 'ignore')
        else:
            param_name = str(param_id).split('\x00', 1)[0]
        if 0 <= param_index < len(FAKE_PARAMS):
            name, value = FAKE_PARAMS[param_index]
            send_param_value(mv, name, value, param_index)
        elif param_name in PARAM_INDEX:
            idx = PARAM_INDEX[param_name]
            name, value = FAKE_PARAMS[idx]
            send_param_value(mv, name, value, idx)
    elif mtype in ('MISSION_REQUEST_LIST', 'WAYPOINT_REQUEST_LIST'):
        src_sys, src_comp = msg_src_ids(msg)
        try:
            mv.mission_count_send(src_sys, src_comp, 1, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
        except TypeError:
            mv.mission_count_send(src_sys, src_comp, 1)
    elif mtype in ('MISSION_REQUEST', 'WAYPOINT_REQUEST'):
        if int(getattr(msg, 'seq', -1)) == 0:
            src_sys, src_comp = msg_src_ids(msg)
            send_home_mission_item(mv, src_sys, src_comp, home_lat_deg, home_lon_deg, home_alt_m)
    elif mtype == 'MISSION_REQUEST_INT':
        if int(getattr(msg, 'seq', -1)) == 0:
            src_sys, src_comp = msg_src_ids(msg)
            send_home_mission_item_int(mv, src_sys, src_comp, home_lat_deg, home_lon_deg, home_alt_m)
    elif mtype == 'COMMAND_LONG':
        command = int(getattr(msg, 'command', -1))
        if command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            mv.command_ack_send(command, mavutil.mavlink.MAV_RESULT_ACCEPTED)
        elif command == mavutil.mavlink.MAV_CMD_DO_SET_HOME:
            send_home_position(mv, int(time.time() * 1_000_000), home_lat_deg, home_lon_deg, home_alt_m)
            mv.command_ack_send(command, mavutil.mavlink.MAV_RESULT_ACCEPTED)
        elif command in (mavutil.mavlink.MAV_CMD_GET_HOME_POSITION, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE):
            requested = int(getattr(msg, 'param1', 0))
            if command == mavutil.mavlink.MAV_CMD_GET_HOME_POSITION or requested == REQUEST_HOME_POSITION:
                send_home_position(mv, int(time.time() * 1_000_000), home_lat_deg, home_lon_deg, home_alt_m)
                mv.command_ack_send(command, mavutil.mavlink.MAV_RESULT_ACCEPTED)
    elif mtype == 'COMMAND_INT':
        command = int(getattr(msg, 'command', -1))
        if command == mavutil.mavlink.MAV_CMD_DO_SET_HOME:
            send_home_position(mv, int(time.time() * 1_000_000), home_lat_deg, home_lon_deg, home_alt_m)
            mv.command_ack_send(command, mavutil.mavlink.MAV_RESULT_ACCEPTED)
    elif mtype == 'REQUEST_DATA_STREAM':
        return


def handle_requests(connection, mv, home_lat_deg=0.0, home_lon_deg=0.0, home_alt_m=0.0):
    if connection is None:
        return

    if hasattr(connection, 'recv_match'):
        while True:
            msg = connection.recv_match(blocking=False)
            if msg is None:
                return
            process_message(msg, mv, home_lat_deg, home_lon_deg, home_alt_m)

    if isinstance(connection, DirectTCPServer):
        while True:
            try:
                data = connection.sock.recv(4096)
            except BlockingIOError:
                return
            if not data:
                return
            for byte in data:
                msg = mv.parse_char(bytes([byte]))
                if msg is not None:
                    process_message(msg, mv, home_lat_deg, home_lon_deg, home_alt_m)


def send_state(mv, boot_ms, unix_usec, home_lat, home_lon, home_alt, north_m, east_m, up_m, v_north_mps, v_east_mps, v_up_mps, roll_rad, pitch_rad, yaw_deg, armed, battery_v=None, battery_a=None, battery_remaining=None):
    send_sys_status(mv, battery_v=battery_v, battery_a=battery_a, battery_remaining=battery_remaining)
    send_ekf_status(mv)
    send_home_position(mv, unix_usec, home_lat, home_lon, home_alt)
    mv.extended_sys_state_send(
        mavutil.mavlink.MAV_VTOL_STATE_UNDEFINED,
        mavutil.mavlink.MAV_LANDED_STATE_IN_AIR if armed else mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
    )
    yaw_rad = math.radians(yaw_deg)
    lat_deg, lon_deg = offset_global_position(home_lat, home_lon, east_m, north_m)
    alt_m = home_alt + up_m
    groundspeed = math.hypot(v_east_mps, v_north_mps)
    course_deg = math.degrees(math.atan2(v_east_mps, v_north_mps)) % 360.0 if groundspeed > 1e-6 else yaw_deg
    heading_cdeg = int((yaw_deg % 360.0) * 100.0)
    mv.local_position_ned_send(boot_ms, north_m, east_m, -up_m, v_north_mps, v_east_mps, -v_up_mps)
    mv.attitude_send(boot_ms, roll_rad, pitch_rad, yaw_rad, 0.0, 0.0, 0.0)
    mv.global_position_int_send(
        boot_ms,
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        int(alt_m * 1000.0),
        int(up_m * 1000.0),
        int(v_north_mps * 100.0),
        int(v_east_mps * 100.0),
        int(-v_up_mps * 100.0),
        heading_cdeg,
    )
    mv.vfr_hud_send(float(groundspeed), float(groundspeed), int(round(yaw_deg)) % 360, 0, float(alt_m), float(v_up_mps))
    send_gps_raw(mv, unix_usec, lat_deg, lon_deg, alt_m, groundspeed, course_deg)


def main():
    p = argparse.ArgumentParser(description='Stream or dump trace as MAVLink')
    p.add_argument('--trace', required=True, help='JSON trace file with hist.* arrays')
    p.add_argument('--out', default='listen:0.0.0.0:5770', help='MAVLink target (tcp/udp, listen:host:port, or .tlog filepath)')
    p.add_argument('--rate', type=float, default=None, help='override playback rate (Hz)')
    p.add_argument('--home-lat', type=float, default=0.0, help='home latitude in degrees for GLOBAL_POSITION_INT')
    p.add_argument('--home-lon', type=float, default=0.0, help='home longitude in degrees for GLOBAL_POSITION_INT')
    p.add_argument('--home-alt-m', type=float, default=0.0, help='home altitude above mean sea level in meters')
    p.add_argument('--source-system', type=int, default=1, help='MAVLink source system id')
    p.add_argument('--source-component', type=int, default=1, help='MAVLink source component id')
    p.add_argument('--heartbeat-hz', type=float, default=4.0, help='heartbeat rate during playback')
    p.add_argument('--hold-last-s', type=float, default=1.0, help='hold the final sample for a short time before disconnecting')
    p.add_argument('--land-s', type=float, default=8.0, help='smoothly descend from the final altitude to home altitude before disconnecting')
    p.add_argument('--post-land-s', type=float, default=8.0, help='hold a final grounded state before disconnecting')
    p.add_argument('--pre-disarm-s', type=float, default=2.0, help='hold a zero-velocity landed state before disarming so MP settles its HUD')
    p.add_argument('--linger', action=argparse.BooleanOptionalAction, default=True, help='keep serving the final landed state until interrupted instead of disconnecting at the end')
    p.add_argument('--disarmed', action='store_true', help='publish heartbeat as disarmed instead of armed')
    p.add_argument('--loop', action='store_true', help='loop playback indefinitely')
    p.add_argument('--repeat', type=int, default=1, help='number of playback passes to run before settling; ignored when --loop')
    p.add_argument('--pre-roll-s', type=float, default=0.0, help='hold the initial state before each playback pass so MP graphs can be armed')
    p.add_argument('--flight-profile', choices=['auto', 'raw', 'successful'], default='auto', help='how aggressively to synthesize a real flight profile from the trace')
    p.add_argument('--end-mode', choices=['auto', 'home', 'final'], default='auto', help='where to settle after the trace: home for synthetic demo flights, final for real destination traces')
    p.add_argument('--cruise-alt-m', type=float, default=12.0, help='synthetic cruise altitude when the trace has no real altitude motion')
    p.add_argument('--takeoff-s', type=float, default=2.0, help='synthetic takeoff duration')
    p.add_argument('--yaw-source', choices=['auto', 'trace', 'track', 'fixed'], default='fixed', help='heading source when yaw is flat in the trace')
    p.add_argument('--max-roll-deg', type=float, default=25.0, help='maximum synthesized bank angle')
    p.add_argument('--max-pitch-deg', type=float, default=15.0, help='maximum synthesized pitch angle')
    args = p.parse_args()

    with open(args.trace, 'r', encoding='utf-8') as f:
        d = json.load(f)
    hist = d.get('hist', {})
    meta = d.get('meta', {})
    t = hist.get('t', [])
    if not t:
        raise SystemExit('trace contains no hist.t values')

    profile = synthesize_successful_flight(hist, meta, args)

    bundle = build_connection(args.out, args.source_system, args.source_component)
    mv = bundle['mv']
    connection = bundle['conn']
    closer = bundle['closer']

    print(f'output -> {args.out}')
    total_passes = None if args.loop else max(1, int(args.repeat))
    pass_duration_s = float(profile['t'][-1]) if args.rate is None else max(0.0, (len(profile['t']) - 1) / max(args.rate, 1e-6))
    idle_dt = max(0.05, 1.0 / max(args.heartbeat_hz, 1.0))
    time_cursor_s = 0.0
    completed_passes = 0

    while total_passes is None or completed_passes < total_passes:
        last_heartbeat = -1e9
        initial_idx = 0
        initial_battery_v = hist_scalar(hist, 'battery_v', initial_idx, None)
        initial_battery_a = hist_scalar(hist, 'battery_a', initial_idx, None)
        initial_battery_remaining = hist_scalar(hist, 'battery_soc', initial_idx, None)

        if args.pre_roll_s > 0.0:
            pre_roll_start = time.time()
            pre_roll_deadline = pre_roll_start + args.pre_roll_s
            while True:
                handle_requests(connection, mv, args.home_lat, args.home_lon, args.home_alt_m)
                now = time.time()
                if now >= pre_roll_deadline:
                    break
                pre_roll_t = min(args.pre_roll_s, max(0.0, now - pre_roll_start))
                timeline_t = time_cursor_s + pre_roll_t
                boot_ms = int(timeline_t * 1000.0)
                if args.heartbeat_hz > 0.0 and (timeline_t - last_heartbeat) >= (1.0 / args.heartbeat_hz):
                    send_heartbeat(mv, armed=not args.disarmed)
                    last_heartbeat = timeline_t
                send_state(
                    mv,
                    boot_ms,
                    int(now * 1_000_000),
                    args.home_lat,
                    args.home_lon,
                    args.home_alt_m,
                    profile['north'][initial_idx],
                    profile['east'][initial_idx],
                    profile['up'][initial_idx],
                    0.0,
                    0.0,
                    0.0,
                    profile['roll_rad'][initial_idx],
                    profile['pitch_rad'][initial_idx],
                    profile['yaw_deg'][initial_idx],
                    not args.disarmed,
                    battery_v=initial_battery_v,
                    battery_a=initial_battery_a,
                    battery_remaining=initial_battery_remaining,
                )
                send_trace_telemetry(mv, boot_ms, hist, initial_idx, profile)
                time.sleep(idle_dt)
            time_cursor_s += args.pre_roll_s

        start_time = time.time()
        for idx in range(len(profile['t'])):
            handle_requests(connection, mv, args.home_lat, args.home_lon, args.home_alt_m)
            sim_t = float(profile['t'][idx])
            target = start_time + sim_t if args.rate is None else start_time + idx / args.rate
            now = time.time()
            if now < target:
                time.sleep(target - now)
                now = time.time()

            timeline_t = time_cursor_s + sim_t
            boot_ms = int(timeline_t * 1000.0)
            unix_usec = int(now * 1_000_000)
            armed = not args.disarmed
            battery_v = hist_scalar(hist, 'battery_v', idx, None)
            battery_a = hist_scalar(hist, 'battery_a', idx, None)
            battery_remaining = hist_scalar(hist, 'battery_soc', idx, None)
            if args.heartbeat_hz > 0.0 and (timeline_t - last_heartbeat) >= (1.0 / args.heartbeat_hz):
                send_heartbeat(mv, armed=armed)
                last_heartbeat = timeline_t

            send_state(
                mv,
                boot_ms,
                unix_usec,
                args.home_lat,
                args.home_lon,
                args.home_alt_m,
                profile['north'][idx],
                profile['east'][idx],
                profile['up'][idx],
                profile['v_north'][idx],
                profile['v_east'][idx],
                profile['v_up'][idx],
                profile['roll_rad'][idx],
                profile['pitch_rad'][idx],
                profile['yaw_deg'][idx],
                armed,
                battery_v=battery_v,
                battery_a=battery_a,
                battery_remaining=battery_remaining,
            )
            send_trace_telemetry(mv, boot_ms, hist, idx, profile)

        time_cursor_s += pass_duration_s
        completed_passes += 1

    if profile['t']:
        final_idx = len(profile['t']) - 1
        final_t = time_cursor_s
        final_boot_ms = int(final_t * 1000.0)
        sample_dt = max(0.1, 1.0 / max(args.heartbeat_hz, 1.0))
        final_battery_v = hist_scalar(hist, 'battery_v', final_idx, None)
        final_battery_a = hist_scalar(hist, 'battery_a', final_idx, None)
        final_battery_remaining = hist_scalar(hist, 'battery_soc', final_idx, None)

        if args.hold_last_s > 0.0:
            end_deadline = time.time() + args.hold_last_s
            while time.time() < end_deadline:
                handle_requests(connection, mv, args.home_lat, args.home_lon, args.home_alt_m)
                send_heartbeat(mv, armed=True)
                send_state(
                    mv,
                    final_boot_ms,
                    int(time.time() * 1_000_000),
                    args.home_lat,
                    args.home_lon,
                    args.home_alt_m,
                    profile['north'][final_idx],
                    profile['east'][final_idx],
                    profile['up'][final_idx],
                    profile['v_north'][final_idx],
                    profile['v_east'][final_idx],
                    0.0,
                    profile['roll_rad'][final_idx],
                    profile['pitch_rad'][final_idx],
                    profile['yaw_deg'][final_idx],
                    True,
                    battery_v=final_battery_v,
                    battery_a=final_battery_a,
                    battery_remaining=final_battery_remaining,
                )
                send_trace_telemetry(mv, final_boot_ms, hist, final_idx, profile)
                time.sleep(sample_dt)

        end_mode = args.end_mode
        if end_mode == 'auto':
            end_mode = 'final' if profile.get('used_raw_alt') else 'home'
        settle_north_m = profile['north'][final_idx] if end_mode == 'final' else 0.0
        settle_east_m = profile['east'][final_idx] if end_mode == 'final' else 0.0
        settle_up_m = profile['up'][final_idx] if end_mode == 'final' else 0.0
        settle_yaw_deg = profile['yaw_deg'][final_idx]
        landed_text = b'LANDED DEST' if end_mode == 'final' else b'LANDED HOME'

        phase_boot_ms = final_boot_ms
        phase_wall = time.time()

        def next_phase_boot_ms():
            nonlocal phase_boot_ms, phase_wall
            now = time.time()
            phase_boot_ms += max(1, int((now - phase_wall) * 1000.0))
            phase_wall = now
            return phase_boot_ms

        if end_mode == 'home' and args.land_s > 0.0 and profile['up'][final_idx] > 0.01:
            steps = max(2, int(args.land_s / sample_dt))
            last_up_m = profile['up'][final_idx]
            last_north_m = profile['north'][final_idx]
            last_east_m = profile['east'][final_idx]
            for step in range(steps):
                handle_requests(connection, mv, args.home_lat, args.home_lon, args.home_alt_m)
                frac = (step + 1) / steps
                eased = smoothstep01(frac)
                up_m = profile['up'][final_idx] * (1.0 - eased)
                north_m = profile['north'][final_idx] * (1.0 - eased)
                east_m = profile['east'][final_idx] * (1.0 - eased)
                v_up_mps = (up_m - last_up_m) / sample_dt
                v_north_mps = (north_m - last_north_m) / sample_dt
                v_east_mps = (east_m - last_east_m) / sample_dt
                last_up_m = up_m
                last_north_m = north_m
                last_east_m = east_m
                boot_ms = next_phase_boot_ms()
                send_heartbeat(mv, armed=True)
                send_state(
                    mv,
                    boot_ms,
                    int(time.time() * 1_000_000),
                    args.home_lat,
                    args.home_lon,
                    args.home_alt_m,
                    north_m,
                    east_m,
                    up_m,
                    v_north_mps,
                    v_east_mps,
                    v_up_mps,
                    0.0,
                    0.0,
                    profile['yaw_deg'][final_idx],
                    True,
                    battery_v=final_battery_v,
                    battery_a=final_battery_a,
                    battery_remaining=final_battery_remaining,
                )
                send_trace_telemetry(mv, boot_ms, hist, final_idx, profile)
                time.sleep(sample_dt)

        pre_disarm_deadline = time.time() + max(0.0, args.pre_disarm_s)
        while time.time() < pre_disarm_deadline:
            handle_requests(connection, mv, args.home_lat, args.home_lon, args.home_alt_m)
            boot_ms = next_phase_boot_ms()
            send_heartbeat(mv, armed=True)
            send_state(
                mv,
                boot_ms,
                int(time.time() * 1_000_000),
                args.home_lat,
                args.home_lon,
                args.home_alt_m,
                settle_north_m,
                settle_east_m,
                settle_up_m,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                settle_yaw_deg,
                True,
                battery_v=final_battery_v,
                battery_a=final_battery_a,
                battery_remaining=final_battery_remaining,
            )
            send_named_value(mv, boot_ms, 'speed', 0.0)
            send_named_value(mv, boot_ms, 'vx', 0.0)
            send_named_value(mv, boot_ms, 'vy', 0.0)
            send_named_value(mv, boot_ms, 'z', settle_up_m)
            send_named_value(mv, boot_ms, 'vz', 0.0)
            send_named_value(mv, boot_ms, 'yaw_rt', 0.0)
            send_named_value(mv, boot_ms, 'alt_synth', settle_up_m)
            send_named_value(mv, boot_ms, 'roll_deg', 0.0)
            send_named_value(mv, boot_ms, 'pitchdeg', 0.0)
            time.sleep(sample_dt)

        post_land_deadline = time.time() + max(0.0, args.post_land_s)
        while True:
            now = time.time()
            now_usec = int(now * 1_000_000)
            boot_ms = next_phase_boot_ms()
            handle_requests(connection, mv, args.home_lat, args.home_lon, args.home_alt_m)
            send_heartbeat(mv, armed=False)
            send_state(
                mv,
                boot_ms,
                now_usec,
                args.home_lat,
                args.home_lon,
                args.home_alt_m,
                settle_north_m,
                settle_east_m,
                settle_up_m,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                settle_yaw_deg,
                False,
                battery_v=final_battery_v,
                battery_a=final_battery_a,
                battery_remaining=final_battery_remaining,
            )
            send_named_value(mv, boot_ms, 'speed', 0.0)
            send_named_value(mv, boot_ms, 'vx', 0.0)
            send_named_value(mv, boot_ms, 'vy', 0.0)
            send_named_value(mv, boot_ms, 'z', settle_up_m)
            send_named_value(mv, boot_ms, 'vz', 0.0)
            send_named_value(mv, boot_ms, 'yaw_rt', 0.0)
            send_named_value(mv, boot_ms, 'alt_synth', settle_up_m)
            send_named_value(mv, boot_ms, 'roll_deg', 0.0)
            send_named_value(mv, boot_ms, 'pitchdeg', 0.0)
            mv.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, landed_text)
            if not args.linger and now >= post_land_deadline:
                break
            time.sleep(sample_dt)

    if closer is not None:
        closer.close()
    print('finished')


if __name__ == '__main__':
    main()