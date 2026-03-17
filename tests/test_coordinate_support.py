import unittest

from aurora_vtol.allocator.coordinate_support import (
    advance_route_goal,
    build_coordinate_history,
    build_coordinate_output,
    compute_desired_speed_mps,
    compute_target_altitude_m,
    resolve_coordinate_arrival_state,
    resolve_route_goal,
)
from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.power_system import PowerSystemParams
from aurora_vtol.allocator.sim_runtime import SimParams


class CoordinateSupportTests(unittest.TestCase):
    def test_advance_route_goal_promotes_subgoal_inside_radius(self):
        route_xy = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        goal_idx = advance_route_goal(5.0, 0.0, route_xy, 1, planner_clearance_m=4.0)
        self.assertEqual(goal_idx, 2)

    def test_compute_desired_speed_mps_slows_near_final_goal(self):
        route_goal = resolve_route_goal(0.0, 0.0, [(0.0, 0.0), (2.0, 0.0)], 1)
        speed = compute_desired_speed_mps(
            max_speed_mps=10.0,
            slow_radius_m=10.0,
            planner_clearance_m=4.0,
            dist_to_goal_m=2.0,
            route_goal=route_goal,
        )
        self.assertLess(speed, 10.0)
        self.assertGreaterEqual(speed, 0.0)

    def test_compute_target_altitude_m_descends_at_final_goal(self):
        z_target_m = compute_target_altitude_m(
            transit_alt_m=12.0,
            dest_z_m=2.0,
            descent_radius_m=18.0,
            arrival_radius_m=1.5,
            dist_to_goal_m=1.0,
            is_final_goal=True,
        )
        self.assertEqual(z_target_m, 2.0)

    def test_resolve_coordinate_arrival_state_transitions_to_hold(self):
        route_goal = resolve_route_goal(0.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], 1)
        arrival = resolve_coordinate_arrival_state(
            route_goal=route_goal,
            dist_to_goal_m=1.0,
            arrival_radius_m=1.5,
            dest_z_m=0.0,
            z_m=0.0,
            speed_mps=0.2,
            t_s=3.0,
            hold_start_s=None,
            arrival_time_s=None,
            fx_cmd=12.0,
            fy_cmd=4.0,
            command_fx_prev=12.0,
            command_fy_prev=4.0,
            descent_radius_m=18.0,
            z_target_m=0.0,
            active_safety=0.0,
        )
        self.assertTrue(arrival.arrived_now)
        self.assertEqual(arrival.phase, "hold")
        self.assertEqual(arrival.fx_cmd, 0.0)
        self.assertEqual(arrival.fy_cmd, 0.0)
        self.assertEqual(arrival.arrival_time_s, 3.0)

    def test_build_coordinate_output_includes_hardware_assumptions(self):
        hist = build_coordinate_history()
        hist["t"].append(0.0)
        hist["x"].append(1.0)
        hist["y"].append(2.0)
        hist["z"].append(3.0)
        hist["vx"].append(0.0)
        hist["vy"].append(0.0)
        hist["vz"].append(0.0)
        hist["speed"].append(0.0)
        hist["yaw_deg"].append(0.0)
        hist["yaw_rate_deg_s"].append(0.0)
        hist["mz_est"].append(0.0)
        hist["alpha_deg_rms"].append(0.0)
        hist["ft_tan_rms"].append(0.0)
        hist["nearest_obstacle_m"].append(None)
        out = build_coordinate_output(
            hist=hist,
            start_x_m=0.0,
            start_y_m=0.0,
            start_z_m=0.0,
            dest_x_m=1.0,
            dest_y_m=2.0,
            dest_z_m=3.0,
            total_s=1.0,
            yaw_hold_deg=0.0,
            mz_nm=0.0,
            cruise_alt_m=12.0,
            hold_s=2.0,
            max_speed_mps=8.0,
            fxy_max_n=3500.0,
            arrival_radius_m=1.5,
            slow_radius_m=14.0,
            descent_radius_m=18.0,
            pos_k_n_per_m=120.0,
            vel_k_n_per_mps=900.0,
            planner_clearance_m=4.0,
            command_rate_n_s=9000.0,
            route_xy=[(0.0, 0.0), (1.0, 2.0)],
            obstacles=[],
            sim=SimParams(),
            lim=ActuatorLimits(),
            pl=PlenumModel(),
            power=PowerSystemParams(),
            fault=FaultSpec(),
            arrival_time_s=0.0,
            hardware_assumptions={"topology": {"name": "demo"}},
        )
        self.assertEqual(out["hardware_assumptions"]["topology"]["name"], "demo")
        self.assertEqual(out["final"]["x_m"], 1.0)


if __name__ == '__main__':
    unittest.main()
