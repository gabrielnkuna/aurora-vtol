from __future__ import annotations

import math
import unittest

from aurora_vtol.allocator.mission_planning import MissionObstacle, mission_safety_force, plan_route_waypoints


class MissionPlanningTests(unittest.TestCase):
    def test_plan_route_waypoints_without_obstacles_is_direct(self) -> None:
        route = plan_route_waypoints(0.0, 0.0, 20.0, 0.0, [], clearance_m=4.0)
        self.assertEqual(route, [(0.0, 0.0), (20.0, 0.0)])

    def test_plan_route_waypoints_adds_bypass_points(self) -> None:
        obstacle = MissionObstacle(x_m=10.0, y_m=0.0, radius_m=3.0, influence_m=18.0)
        route = plan_route_waypoints(0.0, 0.0, 20.0, 0.0, [obstacle], clearance_m=4.0)
        self.assertGreaterEqual(len(route), 4)
        bypass_points = route[1:-1]
        self.assertTrue(any(abs(y) > 0.5 for _x, y in bypass_points))

    def test_mission_safety_force_points_away_from_obstacle(self) -> None:
        obstacle = MissionObstacle(x_m=0.0, y_m=0.0, radius_m=2.0, influence_m=12.0, fxy_max_n=1200.0)
        fx, fy, nearest_clearance_m, active_count, max_threat = mission_safety_force(3.0, 0.0, [obstacle])
        self.assertGreater(fx, 0.0)
        self.assertAlmostEqual(fy, 0.0, places=6)
        self.assertAlmostEqual(nearest_clearance_m, 1.0, places=6)
        self.assertEqual(active_count, 1)
        self.assertGreater(max_threat, 0.0)


if __name__ == '__main__':
    unittest.main()
