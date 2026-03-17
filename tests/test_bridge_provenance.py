from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aurora_vtol.bridge import (
    BridgeSetpoint,
    inspect_script,
    load_script_jsonl,
    load_script_jsonl_metadata,
    script_metadata_from_trace_meta,
    summarize_script,
    summarize_setpoints,
    write_script_jsonl,
)


class BridgeProvenanceTests(unittest.TestCase):
    def test_script_metadata_from_trace_meta_preserves_hardware_assumptions(self) -> None:
        meta = {
            "version": "v3",
            "profile": "step-snap",
            "hardware_assumptions": {
                "topology": {"name": "aurora-ring-32"},
                "effectiveness": {"table_name": "aurora-table"},
            },
        }
        summary = script_metadata_from_trace_meta(meta)
        self.assertEqual(summary["trace_version"], "v3")
        self.assertEqual(summary["profile"], "step-snap")
        self.assertEqual(summary["hardware_assumptions"]["topology"]["name"], "aurora-ring-32")

    def test_write_and_load_script_jsonl_round_trips_metadata(self) -> None:
        script = [
            BridgeSetpoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "track"),
            BridgeSetpoint(1.0, 1.0, 2.0, -0.5, 0.1, 0.2, -0.1, 5.0, "hold-final"),
        ]
        metadata = {
            "hardware_assumptions": {
                "topology": {"name": "aurora-ring-32"},
                "effectiveness": {"table_name": "aurora-ring32"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "script.jsonl"
            write_script_jsonl(path, script, metadata=metadata)
            loaded_metadata = load_script_jsonl_metadata(path)
            loaded_script = load_script_jsonl(path)
        self.assertEqual(loaded_metadata, metadata)
        self.assertEqual(len(loaded_script), 2)
        self.assertEqual(loaded_script[-1].phase, "hold-final")

    def test_inspect_and_summary_surface_metadata(self) -> None:
        script = [
            BridgeSetpoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "track"),
            BridgeSetpoint(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "hold-final"),
        ]
        metadata = {
            "hardware_assumptions": {
                "topology": {"name": "aurora-ring-32"},
                "effectiveness": {"table_name": "aurora-ring32"},
            }
        }
        report = inspect_script(script, metadata=metadata)
        summary = summarize_setpoints(script, metadata=metadata)
        trace_summary = summarize_script({"hardware_assumptions": metadata["hardware_assumptions"]}, script)
        self.assertEqual(report["metadata"], metadata)
        self.assertEqual(summary["metadata"], metadata)
        self.assertEqual(trace_summary["hardware_assumptions"], metadata["hardware_assumptions"])


if __name__ == "__main__":
    unittest.main()
