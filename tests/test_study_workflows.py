import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.power_system import PowerSystemParams
from aurora_vtol.study_workflows import (
    _normalize_required_status,
    _status_meets_threshold,
    build_coordinate_mission_kwargs,
    build_power_params,
    build_power_sweep_report_for_fault,
    write_power_sweep_outputs,
)


class StudyWorkflowTests(unittest.TestCase):
    def test_build_power_params_converts_kw_to_w(self):
        power = build_power_params(
            battery_full_v=60.0,
            capacity_kwh=12.0,
            hover_power_kw=105.0,
            continuous_power_kw=130.0,
            peak_power_kw=165.0,
            aux_power_kw=5.0,
        )
        self.assertEqual(power.battery_full_v, 60.0)
        self.assertEqual(power.hover_power_w, 105000.0)
        self.assertEqual(power.continuous_power_w, 130000.0)
        self.assertEqual(power.peak_power_w, 165000.0)
        self.assertEqual(power.aux_power_w, 5000.0)

    def test_build_coordinate_mission_kwargs_preserves_inputs(self):
        power = PowerSystemParams()
        fault = FaultSpec()
        lim = ActuatorLimits()
        pl = PlenumModel()
        kwargs = build_coordinate_mission_kwargs(
            dest_x_m=5.0,
            dest_y_m=1.0,
            dest_z_m=0.0,
            start_x_m=0.0,
            start_y_m=0.0,
            start_z_m=0.0,
            total_s=5.0,
            yaw_hold_deg=0.0,
            mz_nm=0.0,
            cruise_alt_m=1.0,
            max_speed_mps=2.0,
            arrival_radius_m=1.0,
            slow_radius_m=2.0,
            descent_radius_m=2.0,
            hold_s=0.0,
            fxy_max_n=500.0,
            pos_k_n_per_m=50.0,
            vel_k_n_per_mps=200.0,
            obstacles=[],
            lim=lim,
            pl=pl,
            power=power,
            fault=fault,
        )
        self.assertEqual(kwargs['dest_x_m'], 5.0)
        self.assertIs(kwargs['power'], power)
        self.assertIs(kwargs['fault'], fault)
        self.assertIs(kwargs['lim'], lim)
        self.assertIs(kwargs['pl'], pl)

    def test_threshold_helpers(self):
        self.assertEqual(_normalize_required_status('PASS'), 'pass')
        self.assertTrue(_status_meets_threshold(feasible=True, status='pass', required_status='caution'))
        self.assertFalse(_status_meets_threshold(feasible=True, status='risk', required_status='pass'))
        self.assertFalse(_status_meets_threshold(feasible=False, status='pass', required_status='feasible'))

    def test_build_power_sweep_report_for_fault_smoke(self):
        preset_cfg = {
            'dest_x_m': 4.0,
            'dest_y_m': 0.0,
            'dest_z_m': 0.0,
            'total_s': 5.0,
            'cruise_alt_m': 1.0,
            'max_speed_mps': 2.0,
            'arrival_radius_m': 1.0,
            'slow_radius_m': 2.0,
            'descent_radius_m': 2.0,
            'hold_s': 0.0,
            'fxy_max_n': 500.0,
            'pos_k_n_per_m': 50.0,
            'vel_k_n_per_mps': 200.0,
        }
        report, best = build_power_sweep_report_for_fault(
            preset='smoke',
            preset_cfg=preset_cfg,
            obstacles=[],
            fault_label='nominal',
            fault=FaultSpec(),
            fault_summary={'case': 'nominal'},
            required_status='feasible',
            power_target_pct=120.0,
            flap_target_pct=100.0,
            tune_iterations=1,
            tune_min_aggressiveness=0.5,
            battery_full_v=[],
            battery_empty_v=[],
            capacity_kwh=[],
            internal_resistance_ohm=[],
            hover_power_kw=[],
            continuous_power_kw=[200.0],
            peak_power_kw=[],
            aux_power_kw=[],
            top=1,
        )
        self.assertEqual(report['preset'], 'smoke')
        self.assertTrue(report['results'])
        self.assertTrue(report['requirements_table'])
        self.assertIsNotNone(best)

    def test_write_power_sweep_outputs_writes_json_and_requirements(self):
        report = {
            'requirements_table': [{
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
                'best_continuous_power_p95_pct': 99.9,
                'best_final_goal_error_m': 1.1,
            }]
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report.json'
            req = Path(tmp) / 'requirements.csv'
            updated = write_power_sweep_outputs(report, preset='medium', out=str(out), requirements_out=str(req), requirements_format='auto')
            self.assertEqual(updated['requirements_format'], 'csv')
            self.assertTrue(out.exists())
            self.assertTrue(req.exists())
            loaded = json.loads(out.read_text(encoding='utf-8'))
            self.assertIn('requirements_table', loaded)
            self.assertIn('battery_full_v', req.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
