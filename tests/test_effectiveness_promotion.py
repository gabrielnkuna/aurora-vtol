import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.effectiveness_workflows import (
    build_effectiveness_promotion_report,
    render_effectiveness_promotion_report,
    write_effectiveness_promotion_outputs,
)


def _write_candidate_table(path: Path, *, axial_scale_delta: float = -0.05) -> Path:
    payload = json.loads(Path('data/effectiveness/aurora_ring32_provisional_v1.json').read_text(encoding='utf-8'))
    payload['table_name'] = 'aurora-vtol-ring32-reviewed-candidate-v1-table'
    payload['provenance'] = 'bench-derived reviewed candidate with measured vane effectiveness changes'
    payload['axial_scale_by_segment'][0] = float(payload['axial_scale_by_segment'][0]) + axial_scale_delta
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
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


class EffectivenessPromotionWorkflowTests(unittest.TestCase):
    def test_build_effectiveness_promotion_report_blocks_non_adoptable_candidate(self):
        report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, _candidate_note = build_effectiveness_promotion_report(
            candidate_spec_path=Path('data/effectiveness_specs/aurora_ring32_candidate_template_v1.json'),
            candidate_note_path=Path('docs/effectiveness_candidate_provenance_template.md'),
        )
        self.assertEqual(report['promotion_status'], 'blocked')
        self.assertTrue(any('adoption status' in item.lower() for item in report['blocking_issues']))

    def test_build_effectiveness_promotion_report_blocks_kind_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, _candidate_note = build_effectiveness_promotion_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
            )
            self.assertEqual(report['promotion_status'], 'blocked')
            self.assertTrue(any('source kind does not match' in item.lower() for item in report['blocking_issues']))

    def test_write_effectiveness_promotion_outputs_skips_staged_replacement_when_blocked(self):
        report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_promotion_report(
            candidate_spec_path=Path('data/effectiveness_specs/aurora_ring32_candidate_template_v1.json'),
            candidate_note_path=Path('docs/effectiveness_candidate_provenance_template.md'),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'effectiveness_promotion_blocked'
            updated = write_effectiveness_promotion_outputs(
                report,
                baseline_table,
                candidate_table,
                baseline_spec=baseline_spec,
                candidate_spec=candidate_spec,
                candidate_note=candidate_note,
                out_dir=str(out_dir),
            )
            self.assertNotIn('staged_replacement', updated.get('artifacts', {}))
            self.assertFalse((out_dir / 'staged').exists())

    def test_build_effectiveness_promotion_report_stages_adoptable_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, _candidate_note = build_effectiveness_promotion_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
                baseline_table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json'),
            )
            self.assertEqual(report['promotion_status'], 'staged')
            self.assertEqual(report['target_kind'], 'table')
            self.assertTrue(report['staged_replacement_basename'])

    def test_render_effectiveness_promotion_report_markdown_has_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, _baseline_table, _baseline_spec, _candidate_table, _candidate_spec, _candidate_note = build_effectiveness_promotion_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
                baseline_table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json'),
            )
            rendered = render_effectiveness_promotion_report(report, format_name='markdown')
            self.assertIn('# effectiveness promotion assessment', rendered)
            self.assertIn('## Blocking Issues', rendered)
            self.assertIn('## Passed Checks', rendered)

    def test_write_effectiveness_promotion_outputs_writes_staged_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = _write_candidate_table(Path(tmp) / 'candidate.json')
            note_path = _write_candidate_note(Path(tmp) / 'candidate_note.md')
            report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_promotion_report(
                candidate_table_path=candidate_path,
                candidate_note_path=note_path,
                baseline_table_path=Path('data/effectiveness/aurora_ring32_provisional_v1.json'),
            )
            out_dir = Path(tmp) / 'effectiveness_promotion'
            summary_out = out_dir / 'promotion.txt'
            updated = write_effectiveness_promotion_outputs(
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
            self.assertTrue((out_dir / 'promotion_manifest.json').exists())
            self.assertTrue((out_dir / 'candidate_note.md').exists())
            staged_path = Path(updated['artifacts']['staged_replacement'])
            self.assertTrue(staged_path.exists())
            self.assertTrue(summary_out.exists())
            loaded = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(loaded['promotion_status'], 'staged')
            self.assertEqual(updated['summary_format'], 'text')


if __name__ == '__main__':
    unittest.main()
