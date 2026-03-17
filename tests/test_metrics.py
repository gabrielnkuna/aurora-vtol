import unittest

from aurora_vtol.allocator.metrics import time_to_positive_target_projection_s, time_to_track_alignment_s, track_angle_series_deg


class MetricsTests(unittest.TestCase):
    def test_track_angle_series_deg_returns_expected_angles(self):
        angles = track_angle_series_deg([1.0, 0.0, -1.0], [0.0, 1.0, 0.0])
        self.assertEqual(list(angles), [0.0, 90.0, 180.0])

    def test_time_to_track_alignment_s_finds_first_valid_alignment(self):
        t_s = [0.0, 1.0, 2.0, 3.0]
        vx = [1.0, 0.5, 0.0, 0.0]
        vy = [0.0, 0.5, 1.0, 1.0]
        speed = [1.0, 0.8, 0.4, 0.9]
        result = time_to_track_alignment_s(t_s, vx, vy, speed, start_idx=1, target_deg=90.0)
        self.assertEqual(result, 2.0)

    def test_time_to_positive_target_projection_s_finds_first_positive_projection(self):
        t_s = [0.0, 1.0, 2.0, 3.0]
        vx = [-1.0, -0.2, 0.1, 0.5]
        vy = [0.0, 0.0, 0.0, 0.0]
        result = time_to_positive_target_projection_s(t_s, vx, vy, start_idx=1, target_deg=0.0)
        self.assertEqual(result, 1.0)


if __name__ == '__main__':
    unittest.main()
