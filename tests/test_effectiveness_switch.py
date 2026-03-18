import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.effectiveness_workflows import (
    build_effectiveness_promotion_report,
    build_effectiveness_switch_report,
    render_effectiveness_switch_report,
    write_effectiveness_promotion_outputs,
    write_effectiveness_switch_outputs,
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


def _write_staged_promotion_pack(root: Path) -> tuple[Path, Path]:
    baseline_target = root / 'baseline_table.json'
    baseline_target.write_text(
        Path('data/effectiveness/aurora_ring32_provisional_v1.json').read_text(encoding='utf-8'),
        encoding='utf-8',
    )
    candidate_path = _write_candidate_table(root / 'candidate.json')
    note_path = _write_candidate_note(root / 'candidate_note.md')
    report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_promotion_report(
        candidate_table_path=candidate_path,
        candidate_note_path=note_path,
        baseline_table_path=baseline_target,
    )
    out_dir = root / 'promotion'
    updated = write_effectiveness_promotion_outputs(
        report,
        baseline_table,
        candidate_table,
        baseline_spec=baseline_spec,
        candidate_spec=candidate_spec,
        candidate_note=candidate_note,
        out_dir=str(out_dir),
    )
    return Path(updated['artifacts']['promotion_manifest']), baseline_target


def _write_blocked_promotion_pack(root: Path) -> Path:
    report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_promotion_report(
        candidate_spec_path=Path('data/effectiveness_specs/aurora_ring32_candidate_template_v1.json'),
        candidate_note_path=Path('docs/effectiveness_candidate_provenance_template.md'),
    )
    updated = write_effectiveness_promotion_outputs(
        report,
        baseline_table,
        candidate_table,
        baseline_spec=baseline_spec,
        candidate_spec=candidate_spec,
        candidate_note=candidate_note,
        out_dir=str(root / 'promotion_blocked'),
    )
    return Path(updated['artifacts']['promotion_manifest'])


class EffectivenessSwitchWorkflowTests(unittest.TestCase):
    def test_build_effectiveness_switch_report_blocks_non_staged_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_blocked_promotion_pack(Path(tmp))
            report, _context = build_effectiveness_switch_report(
                promotion_manifest_path=manifest_path,
            )
            self.assertEqual(report['switch_status'], 'blocked')
            self.assertTrue(any('status is blocked' in item.lower() for item in report['blocking_issues']))

    def test_build_effectiveness_switch_report_marks_ready_for_staged_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, baseline_target = _write_staged_promotion_pack(Path(tmp))
            report, _context = build_effectiveness_switch_report(
                promotion_manifest_path=manifest_path,
            )
            self.assertEqual(report['switch_status'], 'ready')
            self.assertEqual(report['target_path'], str(baseline_target))
            self.assertFalse(report['target_overridden'])

    def test_render_effectiveness_switch_report_markdown_has_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, _baseline_target = _write_staged_promotion_pack(Path(tmp))
            report, _context = build_effectiveness_switch_report(
                promotion_manifest_path=manifest_path,
            )
            rendered = render_effectiveness_switch_report(report, format_name='markdown')
            self.assertIn('# effectiveness baseline switch assessment', rendered)
            self.assertIn('## Blocking Issues', rendered)
            self.assertIn('## Passed Checks', rendered)

    def test_write_effectiveness_switch_outputs_requires_out_dir_when_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, _baseline_target = _write_staged_promotion_pack(Path(tmp))
            report, context = build_effectiveness_switch_report(
                promotion_manifest_path=manifest_path,
            )
            with self.assertRaises(ValueError):
                write_effectiveness_switch_outputs(report, context, apply=True)

    def test_write_effectiveness_switch_outputs_applies_and_writes_rollback_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, baseline_target = _write_staged_promotion_pack(Path(tmp))
            report, context = build_effectiveness_switch_report(
                promotion_manifest_path=manifest_path,
            )
            out_dir = Path(tmp) / 'switch'
            original_payload = json.loads(baseline_target.read_text(encoding='utf-8'))
            updated = write_effectiveness_switch_outputs(
                report,
                context,
                out_dir=str(out_dir),
                apply=True,
            )
            self.assertEqual(updated['switch_status'], 'applied')
            self.assertTrue((out_dir / 'summary.json').exists())
            self.assertTrue((out_dir / 'summary.md').exists())
            self.assertTrue((out_dir / 'switch_manifest.json').exists())
            self.assertTrue((out_dir / 'input_promotion_manifest.json').exists())
            self.assertTrue((out_dir / 'candidate_note.md').exists())
            backup_path = Path(updated['artifacts']['previous_target_backup'])
            applied_snapshot_path = Path(updated['artifacts']['applied_target_snapshot'])
            self.assertTrue(backup_path.exists())
            self.assertTrue(applied_snapshot_path.exists())
            backup_payload = json.loads(backup_path.read_text(encoding='utf-8'))
            applied_payload = json.loads(baseline_target.read_text(encoding='utf-8'))
            self.assertEqual(backup_payload['table_name'], original_payload['table_name'])
            self.assertEqual(applied_payload['table_name'], 'aurora-vtol-ring32-reviewed-candidate-v1-table')

    def test_write_effectiveness_switch_outputs_skips_rollback_artifacts_when_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_blocked_promotion_pack(Path(tmp))
            report, context = build_effectiveness_switch_report(
                promotion_manifest_path=manifest_path,
            )
            out_dir = Path(tmp) / 'switch_blocked'
            updated = write_effectiveness_switch_outputs(
                report,
                context,
                out_dir=str(out_dir),
                apply=True,
            )
            self.assertEqual(updated['switch_status'], 'blocked')
            self.assertNotIn('previous_target_backup', updated.get('artifacts', {}))
            self.assertFalse((out_dir / 'backup').exists())


if __name__ == '__main__':
    unittest.main()
