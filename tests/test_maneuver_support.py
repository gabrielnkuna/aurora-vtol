import unittest

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.maneuver_support import build_stateful_maneuver_setup, build_turn_geometry, heading_error_deg, unit_or_default
from aurora_vtol.allocator.model import RingGeometry
from aurora_vtol.allocator.sim_runtime import SimParams


class ManeuverSupportTests(unittest.TestCase):
    def test_build_stateful_maneuver_setup_initializes_defaults(self):
        setup = build_stateful_maneuver_setup(
            total_s=0.1,
            yaw_hold_deg=15.0,
            geom=RingGeometry(),
            sim=SimParams(),
            lim=ActuatorLimits(),
            pl=PlenumModel(),
            fault=FaultSpec(),
        )
        self.assertGreaterEqual(setup.steps, 1)
        self.assertEqual(len(setup.theta_rad), setup.geom.n_segments)
        self.assertEqual(setup.st.yaw_deg, 15.0)
        self.assertEqual(setup.topology.segment_count, setup.geom.n_segments)

    def test_build_turn_geometry_reports_units_and_turn_ratio(self):
        turn = build_turn_geometry(0.0, 90.0)
        self.assertAlmostEqual(turn.start_unit[0], 1.0)
        self.assertAlmostEqual(turn.target_unit[1], 1.0)
        self.assertAlmostEqual(turn.separation_deg, 90.0)
        self.assertAlmostEqual(turn.turn_ratio, 0.5)

    def test_heading_error_deg_wraps(self):
        self.assertEqual(heading_error_deg(350.0, 10.0), 20.0)

    def test_unit_or_default_uses_fallback_for_near_zero(self):
        self.assertEqual(unit_or_default(0.0, 0.0, (1.0, 0.0)), (1.0, 0.0))


if __name__ == '__main__':
    unittest.main()
