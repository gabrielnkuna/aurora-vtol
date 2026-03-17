from __future__ import annotations

import unittest
from unittest.mock import patch

from aurora_vtol.bridge_artifacts import BridgeSetpoint
from aurora_vtol.bridge_sitl import stream_script_ardupilot_sitl
from aurora_vtol.bridge_transport import stream_script_mavlink


class _FakeMav:
    def __init__(self) -> None:
        self.heartbeats = []
        self.position_targets = []
        self.command_longs = []

    def heartbeat_send(self, *args):
        self.heartbeats.append(args)

    def set_position_target_local_ned_send(self, *args):
        self.position_targets.append(args)

    def command_long_send(self, *args):
        self.command_longs.append(args)


class _FakeHeartbeat:
    type = 2
    autopilot = 3

    def get_srcSystem(self):
        return 42

    def get_srcComponent(self):
        return 24


class _FakeConn:
    def __init__(self) -> None:
        self.mav = _FakeMav()
        self.target_system = 0
        self.target_component = 0
        self.flightmode = ''
        self.closed = False
        self._armed = False

    def close(self):
        self.closed = True

    def wait_heartbeat(self, timeout=None):
        return _FakeHeartbeat()

    def recv_match(self, *args, **kwargs):
        return None

    def motors_armed(self):
        return self._armed

    def set_mode_apm(self, mode):
        self.flightmode = mode

    def arducopter_arm(self):
        self._armed = True

    def arducopter_disarm(self):
        self._armed = False


class _FakeMavlink:
    POSITION_TARGET_TYPEMASK_AX_IGNORE = 1
    POSITION_TARGET_TYPEMASK_AY_IGNORE = 2
    POSITION_TARGET_TYPEMASK_AZ_IGNORE = 4
    POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE = 8
    MAV_FRAME_LOCAL_NED = 1
    MAV_TYPE_GCS = 6
    MAV_AUTOPILOT_INVALID = 8
    MAV_TYPE_FIXED_WING = 1
    MAV_CMD_SET_MESSAGE_INTERVAL = 511
    MAV_CMD_NAV_TAKEOFF = 22
    MAV_RESULT_ACCEPTED = 0
    MAV_RESULT_IN_PROGRESS = 5
    enums = {
        'MAV_TYPE': {2: type('E', (), {'name': 'MAV_TYPE_QUADROTOR'})(), 1: type('E', (), {'name': 'MAV_TYPE_FIXED_WING'})()},
        'MAV_AUTOPILOT': {3: type('E', (), {'name': 'MAV_AUTOPILOT_ARDUPILOTMEGA'})()},
        'MAV_RESULT': {0: type('E', (), {'name': 'MAV_RESULT_ACCEPTED'})(), 5: type('E', (), {'name': 'MAV_RESULT_IN_PROGRESS'})()},
    }


class _FakeMavutil:
    mavlink = _FakeMavlink()

    def __init__(self, conn):
        self._conn = conn

    def mavlink_connection(self, *args, **kwargs):
        return self._conn


class BridgeTransportTests(unittest.TestCase):
    def test_stream_script_mavlink_sends_setpoints_and_closes(self):
        conn = _FakeConn()
        mavutil = _FakeMavutil(conn)
        script = [
            BridgeSetpoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'track'),
            BridgeSetpoint(1.0, 1.0, 2.0, -0.5, 0.1, 0.2, -0.1, 5.0, 'hold-final'),
        ]
        with patch('aurora_vtol.bridge_transport._import_mavutil', return_value=mavutil):
            stream_script_mavlink(script, out='udp:127.0.0.1:14550', target_system=1, target_component=2, source_system=245, source_component=190, realtime=False)
        self.assertTrue(conn.closed)
        self.assertEqual(len(conn.mav.position_targets), 2)
        self.assertGreaterEqual(len(conn.mav.heartbeats), 1)

    def test_stream_script_ardupilot_sitl_basic_flow(self):
        conn = _FakeConn()
        mavutil = _FakeMavutil(conn)
        script = [
            BridgeSetpoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'track'),
            BridgeSetpoint(0.5, 1.0, 0.5, -0.2, 0.0, 0.0, 0.0, 0.0, 'hold-final'),
        ]
        with patch('aurora_vtol.bridge_sitl._import_mavutil', return_value=mavutil):
            result = stream_script_ardupilot_sitl(script, connect='tcp:127.0.0.1:5760', mode='', arm=False, takeoff_alt_m=0.0, land_mode='', disarm=False, realtime=False)
        self.assertTrue(conn.closed)
        self.assertEqual(result['target_system'], 42)
        self.assertEqual(result['target_component'], 24)
        self.assertEqual(len(conn.mav.position_targets), 2)
        self.assertEqual(result['warnings'], [])


if __name__ == '__main__':
    unittest.main()
