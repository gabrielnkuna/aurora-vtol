import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.effectiveness_workflows import (
    DEFAULT_GEOMETRY_SEED_SPEC,
    build_effectiveness_adoption_report,
    build_effectiveness_candidate_template,
    build_effectiveness_comparison_report,
    build_effectiveness_report,
    build_effectiveness_validation_report,
    infer_effectiveness_summary_format,
    render_effectiveness_adoption_report,
    render_effectiveness_candidate_provenance_note,
    render_effectiveness_comparison_report,
    render_effectiveness_report,
    render_effectiveness_validation_report,
    write_effectiveness_adoption_outputs,
    write_effectiveness_candidate_template_outputs,
    write_effectiveness_comparison_outputs,
    write_effectiveness_report_outputs,
    write_effectiveness_validation_outputs,
)





def _write_candidate_table(path: Path, *, axial_scale_delta: float = -0.05) -> Path:
    payload = json.loads(Path('data/effectiveness/aurora_ring32_provisional_v1.json').read_text(encoding='utf-8'))
    payload['table_name'] = 'aurora-vtol-ring32-reviewed-candidate-v1-table'
    payload['provenance'] = 'bench-derived reviewed candidate with measured vane effectiveness changes'
    payload['axial_scale_by_segment'][0] = float(payload['axial_scale_by_segment'][0]) + axial_scale_delta
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return path


def _write_candidate_note(path: Path, *, spec_name: str = 'aurora-vtol-ring32-reviewed-candidate-v1-table', validation_state: str = 'reviewed') -> Path:
    note = (
        f"# Aurora VTOL effectiveness candidate provenance note\n\n"
        f"## Candidate identity\n\n"
        f"- spec_name: `{spec_name}`\n"
        f"- spec_path: `data/effectiveness/reviewed_candidate.json`\n"
        f"- owner: `aurora-team`\n"
        f"- date_utc: `2026-03-18`\n\n"
        f"## Source evidence\n\n"
        f"- source_type: `bench`\n"
        f"- source_reference: `bench-run-042`\n"
        f"- extraction_method: `identified from actuator-ring thrust stand replay`\n"
        f"- source_revision: `r1`\n\n"
        f"## Assumptions\n\n"
        f"- fan footprint assumptions: `measured on ring stand with nominal plenum`\n"
        f"- plenum coupling assumptions: `held equal to current nominal plenum model`\n"
        f"- component scale assumptions: `segment 0 axial scale reduced from measured response`\n\n"
        f"## Validation status\n\n"
        f"- validation_state: `{validation_state}`\n"
        f"- reviewer: `controls-review`\n"
        f"- known_limitations: `single-article bench data`\n\n"
        f"## Change summary vs baseline\n\n"
        f"- changed relative to geometry-seeded baseline: `segment 0 axial effectiveness reduced`\n"
        f"- expected impact on allocator behavior: `slightly lower local axial authority`\n"
    )
    path.write_text(note, encoding='utf-8')
    return path


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

    def test_build_effectiveness_validation_report_flags_template_candidate(self):
        report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec = build_effectiveness_validation_report(
            candidate_spec_path=Path('data/effectiveness_specs/aurora_ring32_candidate_template_v1.json')
        )
        self.assertEqual(report['status'], 'risk')
        self.assertTrue(report['blocking_issues'])
        self.assertTrue(any('template' in item.lower() or 'placeholder' in item.lower() for item in report['blocking_issues']))





    def test_build_effectiveness_adoption_report_rejects_missing_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, candidate_note = build_effectiveness_adoption_report(
                candidate_table_path=candidate_path,
            )
            self.assertEqual(report['adoption_status'], 'rejected')
            self.assertIsNone(candidate_note)
            self.assertTrue(any('provenance note is required' in item.lower() for item in report['blocking_issues']))

    def test_build_effectiveness_adoption_report_rejects_template_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = Path(tmp) / 'candidate_note.md'
            note_path.write_text(Path('docs/effectiveness_candidate_provenance_template.md').read_text(encoding='utf-8'), encoding='utf-8')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, candidate_note = build_effectiveness_adoption_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
            )
            self.assertEqual(report['adoption_status'], 'rejected')
            self.assertIsNotNone(candidate_note)
            self.assertTrue(any('placeholder or missing fields' in item.lower() for item in report['blocking_issues']))

    def test_build_effectiveness_adoption_report_needs_review_for_non_material_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, _candidate_note = build_effectiveness_adoption_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
                material_delta_tolerance=1.0,
            )
            self.assertEqual(report['adoption_status'], 'needs-review')
            self.assertFalse(report['material_delta_significant'])

    def test_build_effectiveness_adoption_report_marks_adoptable_with_reviewed_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, candidate_note = build_effectiveness_adoption_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
            )
            self.assertEqual(report['adoption_status'], 'adoptable')
            self.assertTrue(report['material_delta_significant'])
            self.assertIsNotNone(candidate_note)

    def test_render_effectiveness_adoption_report_markdown_has_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, _candidate_note = build_effectiveness_adoption_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
            )
            rendered = render_effectiveness_adoption_report(report, format_name='markdown')
            self.assertIn('# effectiveness adoption assessment', rendered)
            self.assertIn('## Blocking Issues', rendered)
            self.assertIn('## Review Notes', rendered)

    def test_write_effectiveness_adoption_outputs_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_adoption_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
            )
            out_dir = Path(tmp) / 'effectiveness_adoption'
            summary_out = out_dir / 'adoption.txt'
            updated = write_effectiveness_adoption_outputs(
                report,
                baseline_table,
                candidate_table,
                baseline_spec=baseline_spec,
                candidate_spec=candidate_spec,
                candidate_note=candidate_note,
                out_dir=str(out_dir),
                summary_out=str(summary_out),
                summary_format='auto',
            )
            self.assertTrue((out_dir / 'summary.json').exists())
            self.assertTrue((out_dir / 'summary.md').exists())
            self.assertTrue((out_dir / 'candidate_note_fields.json').exists())
            self.assertTrue((out_dir / 'candidate_note.md').exists())
            self.assertTrue(summary_out.exists())
            loaded = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(loaded['adoption_status'], 'adoptable')
            self.assertEqual(updated['summary_format'], 'text')

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

    def test_render_effectiveness_validation_report_markdown_has_sections(self):
        report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec = build_effectiveness_validation_report(
            candidate_spec_path=Path('data/effectiveness_specs/aurora_ring32_candidate_template_v1.json')
        )
        rendered = render_effectiveness_validation_report(report, format_name='markdown')
        self.assertIn('# effectiveness validation', rendered)
        self.assertIn('## Blocking Issues', rendered)
        self.assertIn('## Passed Checks', rendered)

    def test_write_effectiveness_validation_outputs_writes_artifacts(self):
        report, baseline_table, baseline_spec, candidate_table, candidate_spec = build_effectiveness_validation_report(
            candidate_spec_path=Path('data/effectiveness_specs/aurora_ring32_candidate_template_v1.json')
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'effectiveness_validate'
            summary_out = out_dir / 'validate.txt'
            updated = write_effectiveness_validation_outputs(
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
            self.assertTrue((out_dir / 'candidate_table.json').exists())
            self.assertTrue(summary_out.exists())
            loaded = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(loaded['status'], 'risk')
            self.assertEqual(updated['summary_format'], 'text')

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
