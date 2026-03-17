import unittest

from aurora_vtol.allocator.maneuver_scenarios import run_demo, run_repel_test_v4, run_step_redirect_v3, run_step_test_v3


class ManeuverScenarioModuleTests(unittest.TestCase):
    def test_run_demo_smoke(self):
        hist = run_demo(dir_deg=45.0, fxy_n=1200.0, duration_s=0.06)
        self.assertGreaterEqual(len(hist['t']), 1)
        self.assertIn('fan_thrust_16', hist)

    def test_run_step_smoke(self):
        meta, hist = run_step_test_v3(step_time_s=0.02, total_s=0.1)
        self.assertIn('version', meta)
        self.assertGreaterEqual(len(hist['t']), 1)

    def test_run_step_redirect_smoke(self):
        meta, hist = run_step_redirect_v3(step_time_s=0.02, redirect_time_s=0.02, total_s=0.1)
        self.assertIn('version', meta)
        self.assertGreaterEqual(len(hist['t']), 1)

    def test_run_repel_smoke(self):
        meta, hist = run_repel_test_v4(total_s=0.1)
        self.assertIn('version', meta['meta'])
        self.assertGreaterEqual(len(hist['t']), 1)


if __name__ == '__main__':
    unittest.main()
