import unittest

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.maneuver_support import (
    append_stateful_maneuver_history,
    build_maneuver_health,
    build_maneuver_state,
    build_stateful_maneuver_setup,
    build_step_redirect_guard_profile,
    build_step_redirect_shaping,
    build_step_snap_guard_profile,
    build_step_snap_shaping,
    build_turn_geometry,
    compute_step_redirect_phase_command,
    compute_step_snap_phase_command,
    heading_error_deg,
    unit_or_default,
)
from aurora_vtol.allocator.model import RingGeometry
from aurora_vtol.allocator.power_system import PowerSystemState
from aurora_vtol.allocator.sim_runtime import SimParams, SimState


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

    def test_build_step_snap_shaping_clips_inputs_and_builds_ramp(self):
        turn = build_turn_geometry(0.0, 90.0)
        shaping = build_step_snap_shaping(
            turn=turn,
            redirect_hold_frac=-1.0,
            redirect_steer_scale=2.0,
            fault_profile={'dead_align_scale': 1.0},
            dt_s=0.1,
        )
        self.assertGreaterEqual(shaping.hold_frac, 0.55)
        self.assertLessEqual(shaping.steer_scale, 1.5)
        self.assertGreaterEqual(shaping.phase_c_ramp_steps, 1)
        self.assertEqual(shaping.redirect_sep_deg, 90.0)

    def test_build_step_redirect_shaping_builds_transition_windows(self):
        turn = build_turn_geometry(0.0, 90.0)
        shaping = build_step_redirect_shaping(turn=turn, step_idx=10, redirect_end_idx=18, dt_s=0.1)
        self.assertEqual(shaping.redirect_steps, 8)
        self.assertGreaterEqual(shaping.settle_steps, 1)
        self.assertEqual(shaping.redirect_sep_deg, 90.0)

    def test_heading_error_deg_wraps(self):
        self.assertEqual(heading_error_deg(350.0, 10.0), 20.0)

    def test_unit_or_default_uses_fallback_for_near_zero(self):
        self.assertEqual(unit_or_default(0.0, 0.0, (1.0, 0.0)), (1.0, 0.0))

    def test_build_maneuver_state_and_health_reflect_guard_inputs(self):
        st = SimState(x_m=1.0, y_m=2.0, z_m=3.0, vx_mps=4.0, vy_mps=5.0, vz_mps=6.0, yaw_deg=7.0, yaw_rate_deg_s=8.0)
        power_state = PowerSystemState(soc_frac=0.9, voltage_v=57.0, thermal_scale=0.95, burst_reserve_j=100.0)
        guard = {
            'continuous_power_ratio': 0.91,
            'thermal_guard_scale': 0.88,
            'fault_available_scale': 0.77,
            'fault_asymmetry_pct': 12.0,
            'guard_scale': 0.66,
            'fault_response_scale': 0.55,
            'supply_guard_scale': 0.44,
        }

        maneuver_state = build_maneuver_state(st, power_state, guard)
        maneuver_health = build_maneuver_health(fxy_budget_n=1234.0, guard=guard)

        self.assertEqual(maneuver_state.x_m, 1.0)
        self.assertEqual(maneuver_state.bus_voltage_v, 57.0)
        self.assertEqual(maneuver_state.thermal_scale_pct, 88.0)
        self.assertEqual(maneuver_health.lateral_budget_n, 1234.0)
        self.assertEqual(maneuver_health.guard_scale, 0.66)
        self.assertEqual(maneuver_health.supply_scale_pct, 44.0)

    def test_build_step_snap_guard_profile_applies_fault_and_plenum_scaling(self):
        guard = {
            'budget_ratio': 0.8,
            'continuous_power_ratio': 0.97,
            'power_guard_scale': 0.9,
            'fault_guard_scale': 0.85,
            'dead_align_scale': 1.1,
            'dead_cross_scale': 0.92,
            'dead_align_speed_floor_mps': 0.4,
            'plenum_power_trim': 0.95,
            'plenum_revector_trim': 0.9,
            'plenum_align_speed_floor_mps': 0.3,
            'plenum_brake_trim': 0.88,
        }
        profile = build_step_snap_guard_profile(initial_budget_n=2000.0, guard=guard, power_ratio_filt=1.0)
        self.assertLess(profile.fxy_budget_n, 2000.0)
        self.assertLess(profile.speed_guard_scale, 1.0)
        self.assertLess(profile.gain_guard_scale, 1.0)
        self.assertGreaterEqual(profile.power_priority_scale, 0.66)
        self.assertEqual(profile.dead_align_scale, 1.1)
        self.assertEqual(profile.plenum_brake_trim, 0.88)

    def test_build_step_redirect_guard_profile_applies_budget_scaling(self):
        guard = {
            'budget_ratio': 0.7,
            'continuous_power_ratio': 0.95,
            'power_guard_scale': 0.92,
            'fault_guard_scale': 0.9,
        }
        profile = build_step_redirect_guard_profile(initial_budget_n=1800.0, guard=guard, power_ratio_filt=0.98)
        self.assertLess(profile.fxy_budget_n, 1800.0)
        self.assertLess(profile.speed_guard_scale, 1.0)
        self.assertLess(profile.gain_guard_scale, 1.0)
        self.assertGreaterEqual(profile.power_priority_scale, 0.68)
        self.assertLess(profile.power_ratio_filt, 0.98)

    def test_compute_step_snap_phase_command_returns_cruise_before_step(self):
        guard = {
            'continuous_power_ratio': 0.91,
            'thermal_guard_scale': 0.88,
            'fault_available_scale': 0.77,
            'fault_asymmetry_pct': 12.0,
            'guard_scale': 0.66,
            'fault_response_scale': 0.55,
            'supply_guard_scale': 0.44,
        }
        st = SimState()
        power_state = PowerSystemState(soc_frac=0.9, voltage_v=57.0, thermal_scale=0.95, burst_reserve_j=100.0)
        maneuver_state = build_maneuver_state(st, power_state, guard)
        maneuver_health = build_maneuver_health(fxy_budget_n=1500.0, guard=guard)
        turn = build_turn_geometry(0.0, 90.0)
        shaping = build_step_snap_shaping(turn=turn, redirect_hold_frac=-1.0, redirect_steer_scale=1.0, fault_profile={'dead_align_scale': 1.0}, dt_s=0.1)
        snap_guard = build_step_snap_guard_profile(
            initial_budget_n=1500.0,
            guard={
                'budget_ratio': 0.8, 'continuous_power_ratio': 0.91, 'power_guard_scale': 1.0, 'fault_guard_scale': 1.0,
                'dead_align_scale': 1.0, 'dead_cross_scale': 1.0, 'dead_align_speed_floor_mps': 0.0,
                'plenum_power_trim': 1.0, 'plenum_revector_trim': 1.0, 'plenum_align_speed_floor_mps': 0.0, 'plenum_brake_trim': 1.0,
            },
            power_ratio_filt=1.0,
        )
        command = compute_step_snap_phase_command(
            k=0, step_idx=5, snap_end_idx=10, dir_deg_a=0.0, dir_deg_b=90.0, step_speed_ref=0.0,
            speed_stop_thr_mps=0.2, fxy_n=1000.0, fz_cmd=10.0, mz_nm=0.0, brake_gain=1.2,
            maneuver_state=maneuver_state, maneuver_health=maneuver_health, snap_shaping=shaping,
            turn=turn, guard_profile=snap_guard, st=st,
        )
        self.assertEqual(command.phase, 'A')
        self.assertEqual(command.dir_deg, 0.0)
        self.assertGreater(command.fx_raw, 0.0)
        self.assertAlmostEqual(command.fy_raw, 0.0, places=6)

    def test_compute_step_redirect_phase_command_returns_cruise_before_step(self):
        guard = {
            'continuous_power_ratio': 0.91,
            'thermal_guard_scale': 0.88,
            'fault_available_scale': 0.77,
            'fault_asymmetry_pct': 12.0,
            'guard_scale': 0.66,
            'fault_response_scale': 0.55,
            'supply_guard_scale': 0.44,
        }
        st = SimState()
        power_state = PowerSystemState(soc_frac=0.9, voltage_v=57.0, thermal_scale=0.95, burst_reserve_j=100.0)
        maneuver_state = build_maneuver_state(st, power_state, guard)
        maneuver_health = build_maneuver_health(fxy_budget_n=1500.0, guard=guard)
        turn = build_turn_geometry(0.0, 90.0)
        shaping = build_step_redirect_shaping(turn=turn, step_idx=5, redirect_end_idx=10, dt_s=0.1)
        redirect_guard = build_step_redirect_guard_profile(
            initial_budget_n=1500.0,
            guard={'budget_ratio': 0.8, 'continuous_power_ratio': 0.91, 'power_guard_scale': 1.0, 'fault_guard_scale': 1.0},
            power_ratio_filt=1.0,
        )
        command = compute_step_redirect_phase_command(
            k=0, step_idx=5, redirect_end_idx=10, dir_deg_a=0.0, yaw_hold_deg=0.0, step_speed_ref=0.0,
            fxy_n=1000.0, mz_nm=0.0, redirect_speed_scale=0.88, redirect_cross_gain=1.0,
            maneuver_state=maneuver_state, maneuver_health=maneuver_health, redirect_shaping=shaping,
            turn=turn, guard_profile=redirect_guard, st=st,
        )
        self.assertEqual(command.phase, 'A')
        self.assertEqual(command.dir_deg, 0.0)
        self.assertGreater(command.fx_raw, 0.0)
        self.assertAlmostEqual(command.fy_raw, 0.0, places=6)

    def test_append_stateful_maneuver_history_updates_common_fields(self):
        st = SimState(x_m=1.0, y_m=2.0, z_m=3.0, vx_mps=4.0, vy_mps=5.0, vz_mps=6.0, yaw_deg=7.0, yaw_rate_deg_s=8.0)
        hist = {
            't': [], 'x': [], 'y': [], 'z': [], 'vx': [], 'vy': [], 'vz': [], 'speed': [],
            'yaw_deg': [], 'yaw_rate_deg_s': [], 'mz_est': [], 'alpha_deg_rms': [], 'ft_tan_rms': [],
            'alpha_deg_32': [], 'ft_tan_32': [], 'fan_thrust_16': [], 'fx_cmd': [], 'fy_cmd': [],
            'cmd_phase': [], 'power_total_kw': [], 'continuous_power_pct': []
        }
        telemetry = {
            'fan_cmd_16': [0.9, 1.9],
            'fan_actual_16': [1.0, 2.0],
            'fan_temp_c_16': [30.0, 31.0],
            'fan_thermal_scale_16': [1.0, 1.0],
            'alpha_cmd_deg_32': [0.0, 1.0],
            'alpha_cmd_rms': 0.5,
            'battery_v': 57.0,
            'battery_a': 100.0,
            'battery_soc': 90.0,
            'power_w': 12300.0,
            'energy_wh': 12.0,
            'thrust_scale_pct': 95.0,
            'continuous_power_pct': 45.6,
            'continuous_power_raw_pct': 47.0,
            'sustained_power_pct': 44.0,
            'burst_reserve_pct': 80.0,
            'burst_clip_pct': 0.0,
            'burst_active_time_s': 0.1,
            'power_margin_kw': 3.2,
            'thermal_scale_pct': 96.0,
            'fan_temp_max_c': 31.0,
            'fan_temp_mean_c': 30.5,
            'fan_response_pct': 98.0,
            'power_total_kw': 12.3,
        }

        append_stateful_maneuver_history(
            hist,
            t=0.1,
            st=st,
            speed=6.4,
            mz_est=9.9,
            alpha_actual_rad=[0.0, 0.1],
            ft_tan_per_seg_n=[1.0, 2.0],
            telemetry=telemetry,
            fx_cmd=10.0,
            fy_cmd=20.0,
            fz_cmd=30.0,
            net_force_n=[11.0, 22.0, 33.0],
            extras={'cmd_phase': 'A'},
        )

        self.assertEqual(hist['t'], [0.1])
        self.assertEqual(hist['fan_thrust_16'][0], [1.0, 2.0])
        self.assertEqual(hist['fx_cmd'], [10.0])
        self.assertEqual(hist['cmd_phase'], ['A'])
        self.assertEqual(hist['power_w'], [12300.0])
        self.assertEqual(hist['continuous_power_pct'], [45.6])
        self.assertEqual(len(hist['alpha_deg_32'][0]), 2)


if __name__ == '__main__':
    unittest.main()
