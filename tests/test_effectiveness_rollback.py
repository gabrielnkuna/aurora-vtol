import json
import tempfile
import unittest
from pathlib import Path

from aurora_vtol.effectiveness_workflows import (
    build_effectiveness_promotion_report,
    build_effectiveness_rollback_report,
    build_effectiveness_switch_report,
    render_effectiveness_rollback_report,
    write_effectiveness_promotion_outputs,
    write_effectiveness_rollback_outputs,
    write_effectiveness_switch_outputs,
)


def _write_candidate_table(path: Path, *, axial_scale_delta: float = -0.05) -> Path:
    payload = json.loads(Path("data/effectiveness/aurora_ring32_provisional_v1.json").read_text(encoding="utf-8"))
    payload["table_name"] = "aurora-vtol-ring32-reviewed-candidate-v1-table"
    payload["provenance"] = "bench-derived reviewed candidate with measured vane effectiveness changes"
    payload["axial_scale_by_segment"][0] = float(payload["axial_scale_by_segment"][0]) + axial_scale_delta
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_candidate_note(path: Path, *, spec_name: str = "aurora-vtol-ring32-reviewed-candidate-v1-table", validation_state: str = "reviewed") -> Path:
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
    path.write_text(note, encoding="utf-8")
    return path


def _write_staged_promotion_pack(root: Path) -> tuple[Path, Path]:
    baseline_target = root / "baseline_table.json"
    baseline_target.write_text(
        Path("data/effectiveness/aurora_ring32_provisional_v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_path = _write_candidate_table(root / "candidate.json")
    note_path = _write_candidate_note(root / "candidate_note.md")
    report, baseline_table, baseline_spec, candidate_table, candidate_spec, candidate_note = build_effectiveness_promotion_report(
        candidate_table_path=candidate_path,
        candidate_note_path=note_path,
        baseline_table_path=baseline_target,
    )
    out_dir = root / "promotion"
    updated = write_effectiveness_promotion_outputs(
        report,
        baseline_table,
        candidate_table,
        baseline_spec=baseline_spec,
        candidate_spec=candidate_spec,
        candidate_note=candidate_note,
        out_dir=str(out_dir),
    )
    return Path(updated["artifacts"]["promotion_manifest"]), baseline_target


def _write_applied_switch_pack(root: Path) -> tuple[Path, Path, dict, dict]:
    manifest_path, baseline_target = _write_staged_promotion_pack(root)
    original_payload = json.loads(baseline_target.read_text(encoding="utf-8"))
    report, context = build_effectiveness_switch_report(promotion_manifest_path=manifest_path)
    updated = write_effectiveness_switch_outputs(
        report,
        context,
        out_dir=str(root / "switch"),
        apply=True,
    )
    switched_payload = json.loads(baseline_target.read_text(encoding="utf-8"))
    return Path(updated["artifacts"]["switch_manifest"]), baseline_target, original_payload, switched_payload


def _write_ready_switch_pack(root: Path) -> Path:
    manifest_path, _baseline_target = _write_staged_promotion_pack(root)
    report, context = build_effectiveness_switch_report(promotion_manifest_path=manifest_path)
    updated = write_effectiveness_switch_outputs(
        report,
        context,
        out_dir=str(root / "switch_ready"),
        apply=False,
    )
    return Path(updated["artifacts"]["switch_manifest"])


class EffectivenessRollbackWorkflowTests(unittest.TestCase):
    def test_build_effectiveness_rollback_report_blocks_non_applied_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch_manifest = _write_ready_switch_pack(Path(tmp))
            report, _context = build_effectiveness_rollback_report(switch_manifest_path=switch_manifest)
            self.assertEqual(report["rollback_status"], "blocked")
            self.assertTrue(any("status is ready" in item.lower() for item in report["blocking_issues"]))

    def test_build_effectiveness_rollback_report_marks_ready_for_applied_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch_manifest, baseline_target, _original_payload, _switched_payload = _write_applied_switch_pack(Path(tmp))
            report, _context = build_effectiveness_rollback_report(switch_manifest_path=switch_manifest)
            self.assertEqual(report["rollback_status"], "ready")
            self.assertEqual(report["target_path"], str(baseline_target))
            self.assertFalse(report["target_overridden"])

    def test_render_effectiveness_rollback_report_markdown_has_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch_manifest, _baseline_target, _original_payload, _switched_payload = _write_applied_switch_pack(Path(tmp))
            report, _context = build_effectiveness_rollback_report(switch_manifest_path=switch_manifest)
            rendered = render_effectiveness_rollback_report(report, format_name="markdown")
            self.assertIn("# effectiveness baseline rollback assessment", rendered)
            self.assertIn("## Blocking Issues", rendered)
            self.assertIn("## Passed Checks", rendered)

    def test_write_effectiveness_rollback_outputs_requires_out_dir_when_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch_manifest, _baseline_target, _original_payload, _switched_payload = _write_applied_switch_pack(Path(tmp))
            report, context = build_effectiveness_rollback_report(switch_manifest_path=switch_manifest)
            with self.assertRaises(ValueError):
                write_effectiveness_rollback_outputs(report, context, apply=True)

    def test_write_effectiveness_rollback_outputs_applies_and_writes_forward_recovery_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch_manifest, baseline_target, original_payload, switched_payload = _write_applied_switch_pack(Path(tmp))
            report, context = build_effectiveness_rollback_report(switch_manifest_path=switch_manifest)
            out_dir = Path(tmp) / "rollback"
            updated = write_effectiveness_rollback_outputs(
                report,
                context,
                out_dir=str(out_dir),
                apply=True,
            )
            self.assertEqual(updated["rollback_status"], "applied")
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "summary.md").exists())
            self.assertTrue((out_dir / "rollback_manifest.json").exists())
            self.assertTrue((out_dir / "input_switch_manifest.json").exists())
            self.assertTrue((out_dir / "candidate_note.md").exists())
            pre_backup_path = Path(updated["artifacts"]["pre_rollback_target_backup"])
            restored_snapshot_path = Path(updated["artifacts"]["restored_target_snapshot"])
            self.assertTrue(pre_backup_path.exists())
            self.assertTrue(restored_snapshot_path.exists())
            pre_backup_payload = json.loads(pre_backup_path.read_text(encoding="utf-8"))
            restored_payload = json.loads(baseline_target.read_text(encoding="utf-8"))
            self.assertEqual(pre_backup_payload["table_name"], switched_payload["table_name"])
            self.assertEqual(restored_payload["table_name"], original_payload["table_name"])

    def test_write_effectiveness_rollback_outputs_skips_forward_recovery_artifacts_when_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch_manifest = _write_ready_switch_pack(Path(tmp))
            report, context = build_effectiveness_rollback_report(switch_manifest_path=switch_manifest)
            out_dir = Path(tmp) / "rollback_blocked"
            updated = write_effectiveness_rollback_outputs(
                report,
                context,
                out_dir=str(out_dir),
                apply=True,
            )
            self.assertEqual(updated["rollback_status"], "blocked")
            self.assertNotIn("pre_rollback_target_backup", updated.get("artifacts", {}))
            self.assertFalse((out_dir / "pre_rollback").exists())


if __name__ == "__main__":
    unittest.main()
