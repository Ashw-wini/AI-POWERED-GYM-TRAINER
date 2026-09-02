"""Unit tests for reusable pose-angle calculations."""

import unittest

from src.features.angles import calculate_angle


class CalculateAngleTests(unittest.TestCase):
    def test_right_angle(self) -> None:
        angle = calculate_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=6)

    def test_zero_length_vector_is_not_a_valid_angle(self) -> None:
        self.assertIsNone(calculate_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
