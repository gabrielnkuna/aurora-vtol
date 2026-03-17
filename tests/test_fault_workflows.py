import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aurora_vtol.allocator.power_system import PowerSystemParams
from aurora_vtol.fault_workflows import _fault_envelope_manifest, build_fault_envelope_report, load_fault_threshold_cases, resolve_fault_case, run_fault_envelope_report, run_fault_threshold_pack_report, run_fault_threshold_report, select_fault_cases


class FaultWorkflowTests(unittest.TestCase):
    def test_resolve_fault_case_normalizes_composites(self):
        normalized, spec = resolve_fault_case('plenum-sector-1+dead-fan-0+dead-fan-0')
        self.assertEqual(normalized, 'dead-fan-0+plenum-sector-1')
        self.assertEqual(spec['components'], ['dead-fan-0', 'plenum-sector-1'])

    def test_select_fault_cases_uses_set_and_dedupes(self):
        self.assertIn('dead-fan-0+stuck-flap-8', select_fault_cases([], 'stress'))
        self.assertEqual(select_fault_cases(['nominal', 'nominal'], 'nominal'), ['nominal'])

    def test_fault_envelope_manifest_carries_required_status(self):
        manifest = _fault_envelope_manifest(preset='medium', power=PowerSystemParams(), required_status='pass', power_target_pct=100.0, flap_target_pct=90.0, tune_iterations=2, tune_min_aggressiveness=0.25, fan_indices=[0], flap_indices=[0], plenum_indices=[0], fan_scales=[0.0], flap_angles=[20.0], plenum_scales=[0.7], top_per_family=1, include_pairs=False, include_triples=False)
        self.assertEqual(manifest['targets']['required_status'], 'pass')

    def test_build_fault_envelope_report_smoke_uses_required_status(self):
        fake = {'entry': {'preset': 'smoke', 'case': 'dead-fan-00-x0', 'order': 1, 'families': ['fan'], 'fault_case': {'case': 'dead-fan-00-x0'}, 'feasible': True, 'selected_aggressiveness': 0.8, 'selected_score': 1.0, 'report_status': 'pass', 'warnings': [], 'continuous_power_p95_pct': 99.0, 'power_margin_min_kw': 0.0, 'flap_limit_usage_peak_pct': 40.0, 'xy_tracking_rms_n': 10.0, 'final_goal_error_m': 0.2, 'arrival_time_s': 1.2, 'severity_score': 1.0}, 'meta': {'trace': 'ok'}, 'hist': {'t': [0.0]}, 'parts': [{'family': 'fan', 'label': 'dead-fan-00-x0', 'description': 'fan group 0 scale 0.00', 'kwargs': {'dead_fan_group': 0, 'dead_fan_scale': 0.0}}]}
        with patch('aurora_vtol.fault_workflows._evaluate_fault_envelope_case', return_value=fake):
            report, ranked = build_fault_envelope_report(preset='smoke', preset_cfg={'dest_x_m': 5.0, 'dest_y_m': 0.0, 'dest_z_m': 0.0, 'total_s': 5.0, 'cruise_alt_m': 1.0, 'max_speed_mps': 2.0, 'arrival_radius_m': 1.0, 'slow_radius_m': 2.0, 'descent_radius_m': 2.0, 'hold_s': 0.0, 'fxy_max_n': 500.0, 'pos_k_n_per_m': 50.0, 'vel_k_n_per_mps': 200.0}, obstacles=[], required_status='pass', power_target_pct=100.0, flap_target_pct=90.0, tune_iterations=1, tune_min_aggressiveness=0.5, battery_full_v=None, battery_empty_v=None, capacity_kwh=None, internal_resistance_ohm=None, hover_power_kw=None, continuous_power_kw=None, peak_power_kw=None, aux_power_kw=None, dead_fan_scale=[0.0], stuck_flap_alpha_deg=[], plenum_sector_scale=[], fan_group=[0], flap_idx=[], plenum_sector_idx=[], flap_step=32, plenum_step=32, top_per_family=1, include_pairs=False, include_triples=False, top=1)
        self.assertEqual(report['targets']['required_status'], 'pass')
        self.assertGreaterEqual(len(ranked), 1)

    def test_load_fault_threshold_cases_collects_top_and_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'summary.json'
            p.write_text(json.dumps({'preset': 'medium', 'worst_single_by_family': {'fan': {'case': 'dead-fan-00-x0', 'fault_case': {'case': 'dead-fan-00-x0'}, 'families': ['fan'], 'order': 1, 'severity_score': 5.0}}, 'top_cases': [{'case': 'plenum-sector-00-x0p7', 'fault_case': {'case': 'plenum-sector-00-x0p7'}, 'families': ['plenum'], 'order': 1, 'severity_score': 4.0}]}), encoding='utf-8')
            _report, cases = load_fault_threshold_cases(str(p), case_names=[], top_cases=1, include_family_worst=True)
        self.assertEqual([case['case'] for case in cases], ['dead-fan-00-x0', 'plenum-sector-00-x0p7'])

    def test_run_fault_threshold_report_uses_preset_context_resolver(self):
        fake_report = {'search_space': {'cases': 1}, 'best': {'feasible': True, 'meets_threshold': True, 'power_model': {'battery_full_v': 58.8, 'capacity_kwh': 15.0, 'internal_resistance_ohm': 0.0015, 'hover_power_kw': 105.0, 'peak_power_kw': 165.0, 'continuous_power_kw': 125.0}, 'selected_aggressiveness': 0.5, 'report_status': 'pass', 'continuous_power_p95_pct': 99.5, 'final_goal_error_m': 1.0, 'arrival_time_s': 6.0}, 'requirements_table': [{'power_model': {'battery_full_v': 58.8, 'battery_empty_v': 48.0, 'capacity_kwh': 15.0, 'internal_resistance_ohm': 0.0015, 'hover_power_kw': 105.0, 'peak_power_kw': 165.0, 'aux_power_kw': 5.0}, 'feasible': True, 'meets_threshold': True, 'required_status': 'pass', 'required_continuous_power_kw': 125.0, 'best_selected_aggressiveness': 0.5, 'best_report_status': 'pass', 'best_continuous_power_p95_pct': 99.5, 'best_final_goal_error_m': 1.0, 'best_arrival_time_s': 6.0}], 'results': []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'summary.json'
            p.write_text(json.dumps({'preset': 'medium', 'power_model': {'battery_full_v': 58.8, 'battery_empty_v': 48.0, 'capacity_kwh': 15.0, 'internal_resistance_ohm': 0.0015, 'hover_power_kw': 105.0, 'continuous_power_kw': 125.0, 'peak_power_kw': 165.0, 'aux_power_kw': 5.0}, 'targets': {'required_status': 'pass', 'continuous_power_p95_pct': 100.0, 'flap_limit_usage_peak_pct': 90.0, 'tune_iterations': 1, 'tune_min_aggressiveness': 0.25}, 'worst_single_by_family': {}, 'top_cases': [{'case': 'dead-fan-00-x0', 'fault_case': {'case': 'dead-fan-00-x0', 'dead_fan_group': 0, 'dead_fan_scale': 0.0}, 'families': ['fan'], 'order': 1, 'severity_score': 5.0}]}), encoding='utf-8')
            with patch('aurora_vtol.fault_workflows.build_power_sweep_report_for_fault', return_value=(fake_report, None)) as mock_build:
                report = run_fault_threshold_report(summary=str(p), case_names=[], top_cases=1, include_family_worst=False, required_status='feasible', power_target_pct=None, flap_target_pct=None, tune_iterations=None, tune_min_aggressiveness=None, battery_full_v=[], battery_empty_v=[], capacity_kwh=[], internal_resistance_ohm=[], hover_power_kw=[], continuous_power_kw=[], peak_power_kw=[], aux_power_kw=[], top=1, resume=True, max_new_cases=0, out_dir=str(Path(tmp) / 'out'), preset_context_resolver=lambda preset: ({'dest_x_m': 4.0, 'dest_y_m': 0.0, 'dest_z_m': 0.0, 'total_s': 5.0, 'cruise_alt_m': 1.0, 'max_speed_mps': 2.0, 'arrival_radius_m': 1.0, 'slow_radius_m': 2.0, 'descent_radius_m': 2.0, 'hold_s': 0.0, 'fxy_max_n': 500.0, 'pos_k_n_per_m': 50.0, 'vel_k_n_per_mps': 200.0}, []))
        self.assertEqual(report['targets']['required_status'], 'pass')
        kwargs = mock_build.call_args.kwargs
        self.assertIn('preset_cfg', kwargs)
        self.assertIn('obstacles', kwargs)
        self.assertEqual(kwargs['preset_cfg']['dest_x_m'], 4.0)

    def test_run_fault_envelope_report_writes_artifacts(self):
        fake_report = {'preset': 'medium', 'top_cases': [{'case': 'dead-fan-00-x0'}]}
        fake_ranked = [{'entry': {'case': 'dead-fan-00-x0'}, 'meta': {'trace': 'ok'}, 'hist': {'t': [0.0]}}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('aurora_vtol.fault_workflows.build_fault_envelope_report', return_value=(fake_report, fake_ranked)) as mock_build:
                with patch('aurora_vtol.fault_workflows.save_trace_json') as mock_save:
                    report = run_fault_envelope_report(
                        preset='medium',
                        power_target_pct=100.0,
                        flap_target_pct=90.0,
                        tune_iterations=2,
                        tune_min_aggressiveness=0.25,
                        battery_full_v=None,
                        battery_empty_v=None,
                        capacity_kwh=None,
                        internal_resistance_ohm=None,
                        hover_power_kw=None,
                        continuous_power_kw=None,
                        peak_power_kw=None,
                        aux_power_kw=None,
                        dead_fan_scale=[0.0],
                        stuck_flap_alpha_deg=[],
                        plenum_sector_scale=[],
                        fan_group=[0],
                        flap_idx=[],
                        plenum_sector_idx=[],
                        flap_step=32,
                        plenum_step=32,
                        top_per_family=1,
                        include_pairs=False,
                        include_triples=False,
                        resume=True,
                        max_new_cases=0,
                        top=1,
                        save_traces=1,
                        out_dir=tmp,
                        preset_context_resolver=lambda preset: ({'dest_x_m': 4.0}, []),
                    )
                    self.assertTrue(Path(report['artifacts']['json']).exists())
                    self.assertEqual(len(report['artifacts']['traces']), 1)
                    self.assertIn('preset_cfg', mock_build.call_args.kwargs)
                    self.assertIn('obstacles', mock_build.call_args.kwargs)
                    mock_save.assert_called_once()

    def test_run_fault_threshold_pack_report_aggregates_summaries(self):
        fake_threshold = {
            'summary_rows': [{
                'case': 'dead-fan-00-x0',
                'source': 'top:1',
                'order': 1,
                'families': 'fan',
                'best_feasible': True,
                'meets_threshold': True,
                'required_continuous_power_kw': 125.0,
                'battery_full_v': 58.8,
                'capacity_kwh': 15.0,
                'internal_resistance_ohm': 0.0015,
                'hover_power_kw': 105.0,
                'peak_power_kw': 165.0,
                'selected_aggressiveness': 0.5,
                'report_status': 'pass',
                'continuous_power_p95_pct': 99.5,
                'final_goal_error_m': 1.0,
                'arrival_time_s': 6.0,
                'severity_score': 5.0,
            }],
            'search_space': {'cached_cases': 1, 'new_cases': 0, 'complete': True},
            'pending_cases': [],
            'summary_artifacts': {'json': 'x.json'},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for preset in ('medium', 'long'):
                summary_dir = root / f'fault_envelope_{preset}'
                summary_dir.mkdir()
                (summary_dir / 'summary.json').write_text(json.dumps({'preset': preset}), encoding='utf-8')
            summaries = [str(root / 'fault_envelope_medium' / 'summary.json'), str(root / 'fault_envelope_long' / 'summary.json')]
            with patch('aurora_vtol.fault_workflows.run_fault_threshold_report', return_value=fake_threshold) as mock_run:
                report = run_fault_threshold_pack_report(
                    summary_paths=summaries,
                    preset_names=[],
                    case_names=[],
                    top_cases=1,
                    include_family_worst=False,
                    required_status='pass',
                    power_target_pct=None,
                    flap_target_pct=None,
                    tune_iterations=None,
                    tune_min_aggressiveness=None,
                    battery_full_v=[],
                    battery_empty_v=[],
                    capacity_kwh=[],
                    internal_resistance_ohm=[],
                    hover_power_kw=[],
                    continuous_power_kw=[],
                    peak_power_kw=[],
                    aux_power_kw=[],
                    top=1,
                    resume=True,
                    max_new_cases=0,
                    max_new_cases_total=0,
                    out_dir=str(root / 'out'),
                    preset_context_resolver=lambda preset: ({'dest_x_m': 4.0}, []),
                )
                self.assertEqual(report['search_space']['summaries'], 2)
                self.assertEqual(report['search_space']['cached_cases'], 2)
                self.assertEqual(len(report['summary_rows']), 2)
                self.assertTrue(Path(report['summary_artifacts']['json']).exists())
                self.assertEqual(mock_run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
