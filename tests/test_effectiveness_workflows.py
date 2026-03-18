import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.effectiveness_workflows import (
    DEFAULT_GEOMETRY_SEED_SPEC,
    build_effectiveness_candidate_template,
    build_effectiveness_comparison_report,
    build_effectiveness_report,
    infer_effectiveness_summary_format,
    render_effectiveness_candidate_provenance_note,
    render_effectiveness_comparison_report,
    render_effectiveness_report,
    write_effectiveness_candidate_template_outputs,
    write_effectiveness_comparison_outputs,
    write_effectiveness_report_outputs,
)


class EffectivenessWorkflowTests(unittest.TestCase):
    def test_build_effectiveness_report_from_geometry_seed(self):
        report, table, spec = build_effectiveness_report(spec_path=DEFAULT_GEOMETRY_SEED_SPEC)
        self.assertEqual(report['source_kind'], 'geometry-seed')
        self.assertIsNotNone(spec)
        self.assertEqual(report['table_summary']['segment_count'], 32)
        self.assertAlmostEqual(report['fan_weight_summary']['column_sum_min'], 1.0, places=6)
        self.assertTrue(any('provisional' in warning.lower() for warning in report['warnings']))
        self.assertEqual(table.segment_count, 32)

    def test_build_effectiveness_report_from_materialized_table(self):
        report, table, spec = build_effectiveness_report(
            table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json')
        )
        self.assertEqual(report['source_kind'], 'table')
        self.assertIsNone(spec)
        self.assertEqual(table.fan_count, 16)
        self.assertIsNone(report['spec_summary'])

    def test_build_effectiveness_candidate_template_has_placeholder_provenance(self):
        payload = build_effectiveness_candidate_template()
        self.assertEqual(payload['segment_count'], 32)
        self.assertIn('replace', payload['provenance'].lower())

    def test_render_effectiveness_candidate_provenance_note_has_sections(self):
        note = render_effectiveness_candidate_provenance_note(
            spec_name='candidate-spec',
            spec_path='data/effectiveness_specs/candidate.json',
        )
        self.assertIn('# Aurora VTOL effectiveness candidate provenance note', note)
        self.assertIn('## Source evidence', note)
        self.assertIn('candidate-spec', note)

    def test_write_effectiveness_candidate_template_outputs_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'candidate_template'
            report = write_effectiveness_candidate_template_outputs(out_dir=str(out_dir))
            self.assertTrue((out_dir / 'candidate_spec.json').exists())
            self.assertTrue((out_dir / 'provenance_template.md').exists())
            payload = json.loads((out_dir / 'candidate_spec.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['segment_count'], 32)
            self.assertIn('template_spec_name', report)
            self.assertIn('warnings', report)

    def test_build_effectiveness_comparison_report_requires_candidate(self):
        with self.assertRaises(ValueError):
            build_effectiveness_comparison_report()

    def test_build_effectiveness_comparison_report_with_candidate_table(self):
        report, baseline_table, baseline_spec, candidate_table, candidate_spec = build_effectiveness_comparison_report(
            candidate_table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json')
        )
        self.assertTrue(report['compatibility']['comparable'])
        self.assertIsNotNone(report['delta_summary'])
        self.assertGreater(report['delta_summary']['fan_weight_max_abs_delta'], 0.0)
        self.assertIsNotNone(baseline_spec)
        self.assertIsNone(candidate_spec)
        self.assertEqual(baseline_table.segment_count, candidate_table.segment_count)

    def test_infer_effectiveness_summary_format_uses_suffix(self):
        self.assertEqual(infer_effectiveness_summary_format('out.json', 'auto'), 'json')
        self.assertEqual(infer_effectiveness_summary_format('out.md', 'auto'), 'markdown')
        self.assertEqual(infer_effectiveness_summary_format('out.txt', 'auto'), 'text')

    def test_render_effectiveness_report_markdown_has_sections(self):
        report, _table, _spec = build_effectiveness_report(spec_path=DEFAULT_GEOMETRY_SEED_SPEC)
        rendered = render_effectiveness_report(report, format_name='markdown')
        self.assertIn('# effectiveness report', rendered)
        self.assertIn('## Fan Weight Summary', rendered)
        self.assertIn('## Warnings', rendered)

    def test_render_effectiveness_comparison_report_markdown_has_sections(self):
        report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec = build_effectiveness_comparison_report(
            candidate_table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json')
        )
        rendered = render_effectiveness_comparison_report(report, format_name='markdown')
        self.assertIn('# effectiveness comparison', rendered)
        self.assertIn('## Baseline Source', rendered)
        self.assertIn('## Delta Summary', rendered)

    def test_write_effectiveness_comparison_outputs_writes_artifacts(self):
        report, baseline_table, baseline_spec, candidate_table, candidate_spec = build_effectiveness_comparison_report(
            candidate_table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json')
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'effectiveness_compare'
            summary_out = out_dir / 'compare.txt'
            updated = write_effectiveness_comparison_outputs(
                report,
                baseline_table,
                candidate_table,
                baseline_spec=baseline_spec,
                candidate_spec=candidate_spec,
                out_dir=str(out_dir),
                summary_out=str(summary_out),
                summary_format='auto',
            )
            self.assertTrue((out_dir / 'summary.json').exists())
            self.assertTrue((out_dir / 'summary.md').exists())
            self.assertTrue((out_dir / 'baseline_table.json').exists())
            self.assertTrue((out_dir / 'candidate_table.json').exists())
            self.assertTrue(summary_out.exists())
            loaded = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertIn('compatibility', loaded)
            self.assertEqual(updated['summary_format'], 'text')

    def test_write_effectiveness_report_outputs_writes_artifacts(self):
        report, table, spec = build_effectiveness_report(spec_path=DEFAULT_GEOMETRY_SEED_SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'effectiveness'
            summary_out = out_dir / 'summary.txt'
            updated = write_effectiveness_report_outputs(
                report,
                table,
                spec=spec,
                out_dir=str(out_dir),
                summary_out=str(summary_out),
                summary_format='auto',
            )
            self.assertTrue((out_dir / 'summary.json').exists())
            self.assertTrue((out_dir / 'summary.md').exists())
            self.assertTrue((out_dir / 'materialized_table.json').exists())
            self.assertTrue((out_dir / 'source_spec.json').exists())
            self.assertTrue(summary_out.exists())
            loaded = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertIn('artifacts', loaded)
            table_json = json.loads((out_dir / 'materialized_table.json').read_text(encoding='utf-8'))
            self.assertEqual(table_json['segment_count'], 32)
            self.assertEqual(updated['summary_format'], 'text')


if __name__ == '__main__':
    unittest.main()
