import unittest

from aurora_vtol.allocator.sim import PowerSystemParams
from aurora_vtol.reporting import (
    infer_requirements_table_format,
    render_fault_envelope_table,
    render_requirements_table,
    summarize_power_params,
)


class ReportingTests(unittest.TestCase):
    def test_summarize_power_params_returns_kw_fields(self):
        power = PowerSystemParams(
            battery_full_v=60.0,
            battery_empty_v=48.0,
            capacity_kwh=12.5,
            internal_resistance_ohm=0.002,
            hover_power_w=105000.0,
            continuous_power_w=130000.0,
            peak_power_w=160000.0,
            aux_power_w=5000.0,
        )
        summary = summarize_power_params(power)
        self.assertEqual(summary['battery_full_v'], 60.0)
        self.assertEqual(summary['hover_power_kw'], 105.0)
        self.assertEqual(summary['continuous_power_kw'], 130.0)
        self.assertEqual(summary['peak_power_kw'], 160.0)
        self.assertEqual(summary['aux_power_kw'], 5.0)

    def test_infer_requirements_table_format_uses_suffix(self):
        self.assertEqual(infer_requirements_table_format('out.csv', 'auto'), 'csv')
        self.assertEqual(infer_requirements_table_format('out.md', 'auto'), 'markdown')
        self.assertEqual(infer_requirements_table_format('out.txt', 'auto'), 'text')
        self.assertEqual(infer_requirements_table_format('out.txt', 'markdown'), 'markdown')

    def test_render_requirements_table_markdown_flattens_power_model(self):
        rendered = render_requirements_table([
            {
                'power_model': {
                    'battery_full_v': 58.8,
                    'battery_empty_v': 48.0,
                    'capacity_kwh': 15.0,
                    'internal_resistance_ohm': 0.0015,
                    'hover_power_kw': 105.0,
                    'peak_power_kw': 165.0,
                    'aux_power_kw': 5.0,
                },
                'feasible': True,
                'required_continuous_power_kw': 125.0,
                'best_selected_aggressiveness': 0.5,
                'best_report_status': 'pass',
                'best_continuous_power_p95_pct': 99.8,
                'best_final_goal_error_m': 1.2,
            }
        ], format_name='markdown', preset='medium')
        self.assertIn('# medium power requirements', rendered)
        self.assertIn('125', rendered)
        self.assertIn('pass', rendered)
        self.assertIn('true', rendered)

    def test_render_fault_envelope_table_joins_family_lists(self):
        rendered = render_fault_envelope_table([
            {
                'case': 'dead-fan-0+plenum-sector-0',
                'order': 2,
                'families': ['fan', 'plenum'],
                'feasible': True,
                'report_status': 'pass',
                'selected_aggressiveness': 0.4,
                'continuous_power_p95_pct': 99.9,
                'flap_limit_usage_peak_pct': 60.0,
                'xy_tracking_rms_n': 20.0,
                'final_goal_error_m': 1.1,
                'arrival_time_s': 35.0,
                'severity_score': 10.0,
            }
        ], format_name='text', title='fault envelope')
        self.assertIn('fan,plenum', rendered)
        self.assertIn('fault envelope', rendered)


if __name__ == '__main__':
    unittest.main()
