import unittest

from aurora_vtol.allocator.sim import SimParams
from aurora_vtol.allocator.sim_runtime import SimState, clip_force_xy, rate_limit_xy_force, step_vehicle


class SimRuntimeTests(unittest.TestCase):
    def test_sim_reexports_simparams(self):
        sim = SimParams()
        self.assertGreater(sim.dt_s, 0.0)

    def test_clip_force_xy_respects_limit(self):
        fx, fy = clip_force_xy(3.0, 4.0, 2.5)
        self.assertAlmostEqual((fx * fx + fy * fy) ** 0.5, 2.5, places=6)

    def test_rate_limit_xy_force_limits_step(self):
        fx, fy = rate_limit_xy_force(0.0, 0.0, 10.0, 0.0, 5.0, 0.2)
        self.assertAlmostEqual(fx, 1.0)
        self.assertAlmostEqual(fy, 0.0)

    def test_step_vehicle_advances_state(self):
        state = SimState()
        step_vehicle(state, 1000.0, 0.0, 8000.0, 0.0, SimParams())
        self.assertGreater(state.x_m, 0.0)


if __name__ == '__main__':
    unittest.main()
