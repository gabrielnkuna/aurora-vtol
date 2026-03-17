import unittest

from aurora_vtol.allocator.coordinate_execution import append_coordinate_step_history, execute_coordinate_step
from aurora_vtol.allocator.coordinate_support import CoordinateGuidanceCommand, build_coordinate_history, resolve_route_goal
from aurora_vtol.allocator.dynamics import AllocatorState, ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.model import RingGeometry, segment_angles_rad
from aurora_vtol.allocator.power_system import PowerSystemParams, init_hover_power_state
from aurora_vtol.allocator.sim_runtime import SimParams, SimState
from aurora_vtol.effectiveness import effectiveness_table_for_topology
from aurora_vtol.topology import default_ring_topology


class CoordinateExecutionTests(unittest.TestCase):
    def test_execute_coordinate_step_smoke(self):
        geom = RingGeometry()
        sim = SimParams()
        power = PowerSystemParams()
        st = SimState()
        allocator_state = AllocatorState.init(geom.n_segments)
        topology = default_ring_topology(geom.n_segments)
        effectiveness = effectiveness_table_for_topology(topology)
        power_state = init_hover_power_state(power, geom, sim)

        execution = execute_coordinate_step(
            st=st,
            allocator_state=allocator_state,
            geom=geom,
            sim=sim,
            lim=ActuatorLimits(),
            pl=PlenumModel(),
            power=power,
            power_state=power_state,
            fault=FaultSpec(),
            topology=topology,
            effectiveness=effectiveness,
            theta_rad=segment_angles_rad(geom.n_segments),
            fx_cmd=0.0,
            fy_cmd=0.0,
            fz_cmd=sim.mass_kg * sim.gravity,
            mz_nm=0.0,
            z_target_m=0.0,
        )

        self.assertEqual(len(execution.alpha_actual_rad), geom.n_segments)
        self.assertEqual(execution.net_force_n.shape, (3,))
        self.assertEqual(len(execution.telemetry["fan_actual_16"]), topology.fan_count)
        self.assertGreaterEqual(execution.speed_mps, 0.0)

    def test_append_coordinate_step_history_records_step(self):
        geom = RingGeometry()
        sim = SimParams()
        power = PowerSystemParams()
        st = SimState()
        allocator_state = AllocatorState.init(geom.n_segments)
        topology = default_ring_topology(geom.n_segments)
        effectiveness = effectiveness_table_for_topology(topology)
        power_state = init_hover_power_state(power, geom, sim)
        execution = execute_coordinate_step(
            st=st,
            allocator_state=allocator_state,
            geom=geom,
            sim=sim,
            lim=ActuatorLimits(),
            pl=PlenumModel(),
            power=power,
            power_state=power_state,
            fault=FaultSpec(),
            topology=topology,
            effectiveness=effectiveness,
            theta_rad=segment_angles_rad(geom.n_segments),
            fx_cmd=0.0,
            fy_cmd=0.0,
            fz_cmd=sim.mass_kg * sim.gravity,
            mz_nm=0.0,
            z_target_m=0.0,
        )
        hist = build_coordinate_history()
        route_goal = resolve_route_goal(0.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], 1)
        guidance = CoordinateGuidanceCommand(
            desired_speed_mps=0.0,
            z_target_m=0.0,
            speed_guard_scale=1.0,
            gain_guard_scale=1.0,
            goal_force_scale=1.0,
            goal_budget_n=100.0,
            fx_cmd=0.0,
            fy_cmd=0.0,
            desired_vx_mps=0.0,
            desired_vy_mps=0.0,
        )
        guard = {
            "guard_scale": 1.0,
            "flap_guard_scale": 1.0,
            "power_guard_scale": 1.0,
            "thermal_guard_scale": 1.0,
            "supply_guard_scale": 1.0,
            "fault_guard_scale": 1.0,
            "fault_response_scale": 1.0,
            "fault_available_scale": 1.0,
            "fault_asymmetry_pct": 0.0,
            "budget_ratio": 1.0,
            "continuous_power_ratio": 1.0,
            "flap_usage_ratio": 0.0,
        }

        append_coordinate_step_history(
            hist=hist,
            t_s=0.0,
            st=st,
            route_goal=route_goal,
            dist_to_goal_m=1.0,
            nearest_obstacle_m=None,
            fx_safety=0.0,
            fy_safety=0.0,
            z_target_m=0.0,
            phase="cruise",
            guard=guard,
            fxy_budget_n=100.0,
            guidance=guidance,
            fx_cmd=0.0,
            fy_cmd=0.0,
            fz_cmd=sim.mass_kg * sim.gravity,
            execution=execution,
        )

        self.assertEqual(len(hist["t"]), 1)
        self.assertEqual(len(hist["fan_thrust_16"]), 1)
        self.assertEqual(hist["mission_phase"][0], "cruise")


if __name__ == '__main__':
    unittest.main()
