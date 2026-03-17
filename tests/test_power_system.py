import unittest

import numpy as np

import aurora_vtol.allocator.mission_planning as mission_planning
from aurora_vtol.allocator.mission_planning import MissionObstacle
from aurora_vtol.allocator.model import RingGeometry
from aurora_vtol.allocator.power_system import PowerSystemParams, apply_power_system, init_hover_power_state
from aurora_vtol.allocator.sim import SimParams
from aurora_vtol.topology import RingActuatorTopology


class PowerSystemModuleTests(unittest.TestCase):
    def test_mission_planning_stays_trimmed(self):
        self.assertTrue(hasattr(MissionObstacle, '__dataclass_fields__'))
        self.assertNotIn('PowerSystemParams', vars(mission_planning))

    def test_init_hover_power_state_honors_explicit_topology(self):
        geom = RingGeometry(n_segments=32)
        sim = SimParams()
        topology = RingActuatorTopology(
            segment_count=32,
            fan_to_segments=tuple(tuple(range(i * 4, (i + 1) * 4)) for i in range(8)),
            plenum_to_segments=tuple((i,) for i in range(32)),
        )
        state = init_hover_power_state(PowerSystemParams(), geom, sim, topology=topology)
        self.assertEqual(len(state.fan_mean_n), 8)

    def test_apply_power_system_honors_explicit_topology(self):
        geom = RingGeometry(n_segments=32)
        sim = SimParams()
        power = PowerSystemParams()
        topology = RingActuatorTopology(
            segment_count=32,
            fan_to_segments=tuple(tuple(range(i * 4, (i + 1) * 4)) for i in range(8)),
            plenum_to_segments=tuple((i,) for i in range(32)),
        )
        state = init_hover_power_state(power, geom, sim, topology=topology)
        hover = sim.mass_kg * sim.gravity / geom.n_segments
        thrust, updated_state, telemetry = apply_power_system(
            np.full(geom.n_segments, hover, dtype=float),
            np.full(geom.n_segments, hover, dtype=float),
            np.zeros(geom.n_segments, dtype=float),
            np.zeros(geom.n_segments, dtype=float),
            np.zeros(geom.n_segments, dtype=float),
            sim,
            geom,
            power,
            state,
            sim.dt_s,
            topology=topology,
        )
        self.assertEqual(thrust.shape[0], geom.n_segments)
        self.assertEqual(len(updated_state.fan_mean_n), 8)
        self.assertEqual(len(telemetry['fan_cmd_16']), 8)
        self.assertEqual(len(telemetry['fan_actual_16']), 8)


if __name__ == '__main__':
    unittest.main()
