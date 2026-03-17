import unittest

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.maneuver_analysis import render_maneuver_pack_markdown, tune_maneuver_profile


class ManeuverAnalysisTests(unittest.TestCase):
    def test_render_maneuver_pack_markdown_includes_burst_columns(self):
        rendered = render_maneuver_pack_markdown({
            'profile': 'step-snap-eco',
            'angles_deg': [45.0, 90.0],
            'fault_cases': ['nominal'],
            'maneuver_safe': True,
            'worst_status': 'pass',
            'results': [{
                'status': 'pass',
                'fault_case': 'nominal',
                'dir_b_deg': 90.0,
                'selected_aggressiveness': 0.5,
                't_to_speed_below_thr_s': 0.4,
                't_reversal_s': 0.8,
                't90_dir_s': 1.2,
                'yaw_hold_error_mean_abs_deg': 0.2,
                'flap_limit_usage_peak_pct': 50.0,
                'fan_tracking_rms_pct_mean_cmd': 4.0,
                'continuous_power_p95_pct': 99.8,
                'continuous_power_raw_p95_pct': 102.1,
                'burst_reserve_min_pct': 88.0,
                'burst_clip_time_s': 0.0,
            }],
        })
        self.assertIn('Raw P95 (%)', rendered)
        self.assertIn('Burst Min (%)', rendered)
        self.assertIn('nominal', rendered)

    def test_tune_maneuver_profile_step_smoke(self):
        meta, hist, assessment, tuning = tune_maneuver_profile(
            'step',
            {
                'dir_a_deg': 0.0,
                'dir_b_deg': 45.0,
                'fxy_n': 400.0,
                'step_time_s': 0.1,
                'total_s': 0.3,
                'yaw_hold_deg': 0.0,
                'mz_nm': 0.0,
                'lim': ActuatorLimits(),
                'pl': PlenumModel(),
                'fault': FaultSpec(),
            },
            max_iterations=1,
            min_aggressiveness=0.5,
        )
        self.assertIn('step_metrics', meta)
        self.assertIn('t', hist)
        self.assertIn('status', assessment)
        self.assertEqual(tuning['profile'], 'step')
        self.assertIn('selected', tuning)


if __name__ == '__main__':
    unittest.main()
