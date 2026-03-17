import unittest

from aurora_vtol.allocator.coordinate_mission import run_coordinate_mission_v5
from aurora_vtol.allocator.sim import run_coordinate_mission_v5 as run_coordinate_mission_v5_compat


class CoordinateMissionModuleTests(unittest.TestCase):
    def test_coordinate_mission_smoke(self):
        meta, hist = run_coordinate_mission_v5(total_s=0.1)
        self.assertEqual(meta['version'], 'v5-coordinates')
        self.assertGreaterEqual(len(hist['t']), 1)

    def test_sim_compat_reexport(self):
        self.assertIs(run_coordinate_mission_v5_compat, run_coordinate_mission_v5)


if __name__ == '__main__':
    unittest.main()
