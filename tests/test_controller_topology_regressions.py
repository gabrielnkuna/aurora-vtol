from __future__ import annotations

import sys
import unittest

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aurora_vtol.icd import ActuatorHealthState, EstimatedVehicleState, RedirectTarget
from aurora_vtol.topology import RingActuatorTopology
from aurora_vtol.vehicle_controller import command_directional_force, track_redirect_velocity
from aurora_vtol.allocator.faults import FaultSpec, apply_faults_to_thrust


class ControllerTopologyRegressionTests(unittest.TestCase):
    def test_command_directional_force_zero_command_stays_zero(self) -> None:
        result = command_directional_force(
            0.0,
            0.0,
            ActuatorHealthState(lateral_budget_n=2500.0),
            fxy_command_n=1800.0,
            force_scale=1.0,
            fz_n=42.0,
            mz_nm=9.0,
        )
        self.assertEqual(result.desired_wrench.fx_n, 0.0)
        self.assertEqual(result.desired_wrench.fy_n, 0.0)
        self.assertEqual(result.desired_wrench.fz_n, 42.0)
        self.assertEqual(result.desired_wrench.mz_nm, 9.0)
        self.assertEqual(result.command_ux, 0.0)
        self.assertEqual(result.command_uy, 0.0)

    def test_track_redirect_velocity_zero_direction_stays_zero(self) -> None:
        result = track_redirect_velocity(
            state=EstimatedVehicleState(
                x_m=0.0,
                y_m=0.0,
                z_m=0.0,
                vx_mps=1.5,
                vy_mps=-0.5,
                vz_mps=0.0,
                yaw_deg=0.0,
                yaw_rate_deg_s=0.0,
            ),
            target=RedirectTarget(
                desired_ux=0.0,
                desired_uy=0.0,
                desired_speed_mps=4.0,
                reference_speed_mps=4.0,
            ),
            health=ActuatorHealthState(lateral_budget_n=3000.0),
            fxy_command_n=2200.0,
        )
        self.assertEqual(result.desired_wrench.fx_n, 0.0)
        self.assertEqual(result.desired_wrench.fy_n, 0.0)
        self.assertEqual(result.command_ux, 0.0)
        self.assertEqual(result.command_uy, 0.0)

    def test_topology_segment_count_mismatch_raises(self) -> None:
        topology = RingActuatorTopology.aurora_ring_32()
        with self.assertRaises(ValueError):
            topology.smooth_segment_values([1.0] * 30)
        with self.assertRaises(ValueError):
            topology.segment_values_to_fan_means([1.0] * 30)
        with self.assertRaises(ValueError):
            topology.distribute_fan_means_to_segments([1.0] * 16, [1.0] * 30)

    def test_topology_fan_count_mismatch_raises(self) -> None:
        topology = RingActuatorTopology.aurora_ring_32()
        with self.assertRaises(ValueError):
            topology.distribute_fan_means_to_segments([1.0] * 15, [1.0] * 32)

    def test_apply_faults_to_thrust_uses_explicit_topology(self) -> None:
        topology = RingActuatorTopology(
            segment_count=4,
            fan_to_segments=((0,), (1, 2, 3)),
            plenum_to_segments=((0,), (1,), (2,), (3,)),
            fan_nominal_sigma_segments=0.0,
            fan_fault_sigma_segments=0.0,
            plenum_fault_sigma_segments=0.0,
        )
        thrust = apply_faults_to_thrust(
            np.array([10.0, 10.0, 10.0, 10.0], dtype=float),
            FaultSpec(dead_fan_group=1, dead_fan_scale=0.0),
            topology=topology,
        )
        self.assertListEqual(thrust.tolist(), [10.0, 0.0, 0.0, 0.0])

    def test_apply_faults_to_thrust_shape_mismatch_raises(self) -> None:
        topology = RingActuatorTopology.even_pairs(4)
        with self.assertRaises(ValueError):
            apply_faults_to_thrust(
                np.array([10.0, 10.0, 10.0], dtype=float),
                FaultSpec(dead_fan_group=0, dead_fan_scale=0.0),
                topology=topology,
            )


if __name__ == "__main__":
    unittest.main()
