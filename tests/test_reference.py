"""Tests for temporal normalization and trainer-reference aggregation."""

import unittest
from pathlib import Path

import numpy as np

from src.references.reference import (
    TrainerRep,
    aggregate_normalized_repetitions,
    interpolate_sequence,
    normalize_rep_sequence,
)


class ReferenceSequenceTests(unittest.TestCase):
    def test_interpolate_sequence_to_requested_length(self) -> None:
        values = np.asarray([[0.0], [10.0]])
        timestamps = np.asarray([0, 1000])

        normalized = interpolate_sequence(values, timestamps, target_length=5)

        np.testing.assert_allclose(normalized[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_generates_mean_and_standard_deviation_reference(self) -> None:
        first = np.asarray([[0.0, 10.0], [10.0, 20.0]])
        second = np.asarray([[2.0, 14.0], [12.0, 24.0]])

        reference = aggregate_normalized_repetitions([first, second], ("knee", "hip"))

        np.testing.assert_allclose(reference.mean, [[1.0, 12.0], [11.0, 22.0]])
        np.testing.assert_allclose(reference.standard_deviation, [[1.0, 2.0], [1.0, 2.0]])

    def test_handles_one_or_many_repetitions(self) -> None:
        rep = TrainerRep(
            source_path=Path("rep_001.csv"),
            frame_indices=np.asarray([0, 10, 20]),
            timestamps_ms=np.asarray([0, 300, 900]),
            feature_names=("knee",),
            values=np.asarray([[170.0], [80.0], [170.0]]),
            bottom_sample_index=1,
        )

        normalized = normalize_rep_sequence(rep, normalization_length=5, align_bottom=True)
        one_rep_reference = aggregate_normalized_repetitions([normalized], rep.feature_names)
        many_rep_reference = aggregate_normalized_repetitions(
            [normalized, normalized, normalized], rep.feature_names
        )

        self.assertEqual(one_rep_reference.mean.shape, (5, 1))
        self.assertEqual(many_rep_reference.mean.shape, (5, 1))
        np.testing.assert_allclose(one_rep_reference.standard_deviation, 0.0)
        np.testing.assert_allclose(many_rep_reference.standard_deviation, 0.0)
