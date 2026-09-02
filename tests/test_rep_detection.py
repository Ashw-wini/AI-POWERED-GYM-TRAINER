"""Synthetic tests for squat repetition detection."""

import unittest

from src.reps.detection import DetectionConfig, FeatureSample, detect_repetitions


def build_samples(signal: list[float]) -> list[FeatureSample]:
    """Create deterministic feature samples where both knees track one signal."""
    return [
        FeatureSample(
            frame_index=index,
            timestamp_ms=index * 33,
            left_knee_angle_deg=angle + 1.0,
            right_knee_angle_deg=angle,
            row={"frame_index": str(index)},
        )
        for index, angle in enumerate(signal)
    ]


class RepDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DetectionConfig(
            smoothing_window=3,
            min_rep_frames=10,
            min_rep_samples=8,
            min_phase_samples=2,
        )

    def test_detects_two_complete_squats_with_small_jitter(self) -> None:
        standing = [170.0, 171.0, 169.0, 170.0, 170.0, 169.0, 171.0, 170.0]
        squat = [160.0, 148.0, 132.0, 112.0, 92.0, 80.0, 78.0, 79.0, 91.0, 112.0, 134.0, 150.0, 162.0, 170.0]
        samples = build_samples(standing + squat + standing + squat + standing)

        result = detect_repetitions(samples, self.config)

        self.assertEqual(len(result.repetitions), 2)
        self.assertLess(result.repetitions[0].start_frame, result.repetitions[0].bottom_frame)
        self.assertLess(result.repetitions[0].bottom_frame, result.repetitions[0].end_frame)

    def test_ignores_a_short_false_repetition(self) -> None:
        signal = [170.0] * 8 + [135.0, 110.0, 90.0, 150.0, 165.0] + [170.0] * 8

        result = detect_repetitions(build_samples(signal), self.config)

        self.assertEqual(len(result.repetitions), 0)


if __name__ == "__main__":
    unittest.main()
