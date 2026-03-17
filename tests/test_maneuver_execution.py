import unittest

from aurora_vtol.allocator.dynamics import AllocatorState, ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.maneuver_execution import execute_maneuver_step
from aurora_vtol.allocator.maneuver_scenarios import run_demo, run_repel_test_v4, run_step_redirect_v3, run_step_test_v3
from aurora_vtol.allocator.model import RingGeometry, segment_angles_rad
from aurora_vtol.allocator.power_system import PowerSystemParams, init_hover_power_state
from aurora_vtol.allocator.sim_runtime import SimParams, SimState
from aurora_vtol.effectiveness import effectiveness_table_for_topology
from aurora_vtol.topology import default_ring_topology


class ManeuverExecutionTests(unittest.TestCase):
    def test_execute_maneuver_step_smoke(self):
        geom = RingGeometry()
        sim = SimParams()
        power = PowerSystemParams()
        st = SimState()
        allocator_state = AllocatorState.init(geom.n_segments)
        topology = default_ring_topology(geom.n_segments)
        effectiveness = effectiveness_table_for_topology(topology)
        power_state = init_hover_power_state(power, geom, sim)

        execution = execute_maneuver_step(
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
        )
        self.assertEqual(len(execution.alpha_actual_rad), geom.n_segments)
        self.assertEqual(execution.net_force_n.shape, (3,))
        self.assertEqual(len(execution.telemetry['fan_actual_16']), topology.fan_count)
        self.assertGreaterEqual(execution.speed_mps, 0.0)


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
