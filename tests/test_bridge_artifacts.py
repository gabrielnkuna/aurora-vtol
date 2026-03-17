from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aurora_vtol.bridge import BridgeSetpoint, build_qgc_wpl_mission, summarize_qgc_wpl_mission, write_qgc_wpl_mission, write_script_svg


class BridgeArtifactTests(unittest.TestCase):
    def test_write_script_svg_emits_svg_file(self) -> None:
        script = [
            BridgeSetpoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "track"),
            BridgeSetpoint(1.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, "hold-final"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "route.svg"
            write_script_svg(path, script, title="Bridge Test")
            content = path.read_text(encoding="utf-8")
        self.assertIn("<svg", content)
        self.assertIn("Bridge Test", content)
        self.assertIn("hold-final", content)

    def test_build_and_write_qgc_wpl_mission(self) -> None:
        meta = {
            "mission": {
                "start": {"x_m": 0.0, "y_m": 0.0},
                "destination": {"x_m": 20.0, "y_m": 10.0, "z_m": 0.0},
                "cruise_alt_m": 12.0,
                "max_speed_mps": 6.0,
            },
            "planner": {
                "waypoints": [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 10.0, "y_m": 5.0},
                    {"x_m": 20.0, "y_m": 10.0},
                ]
            },
        }
        hist = {"x": [0.0, 10.0, 20.0], "y": [0.0, 5.0, 10.0], "z": [0.0, 12.0, 0.0], "mission_phase": ["track", "track", "hold-final"]}
        items = build_qgc_wpl_mission(meta, hist, home_lat_deg=-26.2041, home_lon_deg=28.0473, land=True)
        summary = summarize_qgc_wpl_mission(items, out_path="mission.wpl", home_lat_deg=-26.2041, home_lon_deg=28.0473)
        self.assertEqual(summary["format"], "QGC WPL 110")
        self.assertGreaterEqual(summary["items"], 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mission.wpl"
            write_qgc_wpl_mission(path, items)
            content = path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("QGC WPL 110"))
        self.assertIn("	", content)


if __name__ == "__main__":
    unittest.main()
