import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.effectiveness_workflows import (
    DEFAULT_GEOMETRY_SEED_SPEC,
    build_effectiveness_report,
    infer_effectiveness_summary_format,
    render_effectiveness_report,
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
