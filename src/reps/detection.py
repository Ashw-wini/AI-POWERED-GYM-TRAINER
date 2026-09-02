"""Noise-tolerant, state-based squat repetition detection."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


FEATURE_INPUT_COLUMNS = {
    "frame_index",
    "timestamp_ms",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
}
PRIMARY_SIGNAL_NAME = "minimum of left and right knee angle"


class RepDetectionError(Exception):
    """Raised when a feature CSV cannot be used for repetition detection."""


class MovementPhase(str, Enum):
    """The ordered phases of a complete squat repetition."""

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


@dataclass(frozen=True)
class FeatureSample:
    """One valid feature row, retaining its original CSV data for rep exports."""

    frame_index: int
    timestamp_ms: int
    left_knee_angle_deg: float
    right_knee_angle_deg: float
    row: dict[str, str]


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds for a general squat cycle, not a fixed repetition count."""

    smoothing_window: int = 5
    standing_percentile: float = 90.0
    bottom_percentile: float = 10.0
    standing_fraction: float = 0.15
    descending_fraction: float = 0.35
    bottom_fraction: float = 0.65
    min_phase_samples: int = 2
    min_rep_frames: int = 15
    min_rep_samples: int = 6
    min_rep_amplitude_deg: float = 20.0
    max_frame_gap: int = 60


@dataclass(frozen=True)
class DetectionThresholds:
    """Data-derived signal levels used by the state machine."""

    standing_angle_deg: float
    standing_threshold_deg: float
    descending_threshold_deg: float
    bottom_threshold_deg: float
    motion_range_deg: float


@dataclass(frozen=True)
class DetectedRep:
    """One complete standing-to-standing squat repetition."""

    rep_number: int
    start_sample_index: int
    end_sample_index: int
    bottom_sample_index: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_ms: int
    bottom_frame: int
    bottom_knee_angle_deg: float

    def as_summary_record(self) -> dict[str, float | int]:
        """Return the portable summary fields for JSON and CSV output."""
        return {
            "rep_number": self.rep_number,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_ms": self.duration_ms,
            "bottom_frame": self.bottom_frame,
            "bottom_knee_angle_deg": self.bottom_knee_angle_deg,
        }


@dataclass(frozen=True)
class DetectionResult:
    """Detected repetitions, debug signals, and warnings from one input CSV."""

    repetitions: tuple[DetectedRep, ...]
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray
    thresholds: DetectionThresholds | None
    warning_counts: dict[str, int]


def parse_finite_float(value: str | None) -> float | None:
    """Return a finite number from a CSV cell, or None for missing/bad data."""
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def load_feature_samples(
    input_path: Path,
) -> tuple[list[FeatureSample], tuple[str, ...], Counter[str]]:
    """Read valid feature rows without changing the Milestone 2 CSV format."""
    if not input_path.is_file():
        raise RepDetectionError(f"Feature CSV was not found: {input_path}")

    samples: list[FeatureSample] = []
    warnings: Counter[str] = Counter()
    try:
        with input_path.open("r", newline="", encoding="utf-8") as source_file:
            reader = csv.DictReader(source_file)
            if reader.fieldnames is None or not FEATURE_INPUT_COLUMNS.issubset(reader.fieldnames):
                raise RepDetectionError(
                    "The input CSV does not contain the required Milestone 2 feature columns."
                )
            fieldnames = tuple(reader.fieldnames)

            for row in reader:
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_ms = int(row["timestamp_ms"])
                except (TypeError, ValueError):
                    warnings["malformed_frame_identifier_rows"] += 1
                    continue

                left_knee = parse_finite_float(row.get("left_knee_angle_deg"))
                right_knee = parse_finite_float(row.get("right_knee_angle_deg"))
                if left_knee is None or right_knee is None:
                    warnings["invalid_knee_angle_rows"] += 1
                    continue

                samples.append(
                    FeatureSample(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        left_knee_angle_deg=left_knee,
                        right_knee_angle_deg=right_knee,
                        row=row,
                    )
                )
    except OSError as error:
        raise RepDetectionError(f"Could not read feature CSV '{input_path}': {error}") from error

    if not samples:
        raise RepDetectionError(f"The feature CSV contains no usable knee-angle rows: {input_path}")

    sorted_samples = sorted(samples, key=lambda sample: (sample.frame_index, sample.timestamp_ms))
    if sorted_samples != samples:
        warnings["out_of_order_feature_rows"] += 1
    return sorted_samples, fieldnames, warnings


def select_primary_signal(samples: list[FeatureSample]) -> np.ndarray:
    """Use the more-flexed knee angle each frame as the squat-depth signal."""
    return np.asarray(
        [min(sample.left_knee_angle_deg, sample.right_knee_angle_deg) for sample in samples],
        dtype=float,
    )


def rolling_median(values: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth small landmark jitter without shifting squat-bottom timing heavily."""
    if window_size < 1:
        raise ValueError("smoothing_window must be at least 1")
    if window_size % 2 == 0:
        raise ValueError("smoothing_window must be odd")

    half_window = window_size // 2
    result = np.empty_like(values, dtype=float)
    for index in range(values.size):
        start = max(0, index - half_window)
        end = min(values.size, index + half_window + 1)
        result[index] = float(np.median(values[start:end]))
    return result


def derive_thresholds(
    smoothed_signal: np.ndarray, config: DetectionConfig
) -> DetectionThresholds | None:
    """Derive hysteresis levels from the observed standing and bottom angles."""
    if smoothed_signal.size < config.min_rep_samples:
        return None

    standing_angle = float(np.percentile(smoothed_signal, config.standing_percentile))
    bottom_angle = float(np.percentile(smoothed_signal, config.bottom_percentile))
    motion_range = standing_angle - bottom_angle
    if motion_range < config.min_rep_amplitude_deg:
        return None

    return DetectionThresholds(
        standing_angle_deg=standing_angle,
        standing_threshold_deg=standing_angle - config.standing_fraction * motion_range,
        descending_threshold_deg=standing_angle - config.descending_fraction * motion_range,
        bottom_threshold_deg=standing_angle - config.bottom_fraction * motion_range,
        motion_range_deg=motion_range,
    )


def is_valid_config(config: DetectionConfig) -> bool:
    """Check configuration ordering before applying it to a source sequence."""
    return (
        config.smoothing_window >= 1
        and config.smoothing_window % 2 == 1
        and 0 <= config.bottom_percentile < config.standing_percentile <= 100
        and 0 < config.standing_fraction < config.descending_fraction < config.bottom_fraction < 1
        and config.min_phase_samples >= 1
        and config.min_rep_frames >= 1
        and config.min_rep_samples >= 1
        and config.min_rep_amplitude_deg > 0
        and config.max_frame_gap >= 1
    )


def detect_repetitions(
    samples: list[FeatureSample], config: DetectionConfig = DetectionConfig()
) -> DetectionResult:
    """Detect complete standing→descending→bottom→ascending→standing cycles."""
    if not is_valid_config(config):
        raise ValueError("Rep-detection configuration is invalid.")

    warnings: Counter[str] = Counter()
    raw_signal = select_primary_signal(samples)
    smoothed_signal = rolling_median(raw_signal, config.smoothing_window)
    thresholds = derive_thresholds(smoothed_signal, config)
    if thresholds is None:
        warnings["insufficient_squat_motion"] += 1
        return DetectionResult((), raw_signal, smoothed_signal, None, dict(warnings))

    repetitions: list[DetectedRep] = []
    phase = MovementPhase.STANDING
    last_standing_index: int | None = None
    rep_start_index: int | None = None
    bottom_index: int | None = None
    descending_samples = 0
    incomplete_motion_before_baseline = False

    for index, value in enumerate(smoothed_signal):
        if index > 0 and samples[index].frame_index - samples[index - 1].frame_index > config.max_frame_gap:
            warnings["large_frame_gaps"] += 1
            phase = MovementPhase.STANDING
            rep_start_index = None
            bottom_index = None
            descending_samples = 0
            last_standing_index = index if value >= thresholds.standing_threshold_deg else None

        if phase is MovementPhase.STANDING:
            if value >= thresholds.standing_threshold_deg:
                last_standing_index = index
                descending_samples = 0
            elif last_standing_index is None and value <= thresholds.descending_threshold_deg:
                incomplete_motion_before_baseline = True
            elif last_standing_index is not None and value <= thresholds.descending_threshold_deg:
                descending_samples += 1
                if descending_samples >= config.min_phase_samples:
                    phase = MovementPhase.DESCENDING
                    rep_start_index = last_standing_index
                    bottom_index = index
                    descending_samples = 0
            else:
                descending_samples = 0
            continue

        if phase is MovementPhase.DESCENDING:
            if bottom_index is None or value < smoothed_signal[bottom_index]:
                bottom_index = index
            if value <= thresholds.bottom_threshold_deg:
                phase = MovementPhase.BOTTOM
            elif value >= thresholds.standing_threshold_deg:
                warnings["aborted_descents"] += 1
                phase = MovementPhase.STANDING
                last_standing_index = index
                rep_start_index = None
                bottom_index = None
            continue

        if phase is MovementPhase.BOTTOM:
            if bottom_index is None or value < smoothed_signal[bottom_index]:
                bottom_index = index
            if value >= thresholds.descending_threshold_deg:
                phase = MovementPhase.ASCENDING
            continue

        if phase is MovementPhase.ASCENDING:
            if bottom_index is None or value < smoothed_signal[bottom_index]:
                bottom_index = index
            if value <= thresholds.bottom_threshold_deg:
                phase = MovementPhase.BOTTOM
                continue
            if value < thresholds.standing_threshold_deg:
                continue

            if rep_start_index is None or bottom_index is None:
                warnings["invalid_rep_state"] += 1
            else:
                start_sample = samples[rep_start_index]
                end_sample = samples[index]
                duration_frames = end_sample.frame_index - start_sample.frame_index
                sample_count = index - rep_start_index + 1
                amplitude = thresholds.standing_angle_deg - smoothed_signal[bottom_index]
                if duration_frames < config.min_rep_frames or sample_count < config.min_rep_samples:
                    warnings["rejected_short_repetitions"] += 1
                elif amplitude < config.min_rep_amplitude_deg:
                    warnings["rejected_shallow_repetitions"] += 1
                else:
                    repetitions.append(
                        DetectedRep(
                            rep_number=len(repetitions) + 1,
                            start_sample_index=rep_start_index,
                            end_sample_index=index,
                            bottom_sample_index=bottom_index,
                            start_frame=start_sample.frame_index,
                            end_frame=end_sample.frame_index,
                            duration_frames=duration_frames,
                            duration_ms=end_sample.timestamp_ms - start_sample.timestamp_ms,
                            bottom_frame=samples[bottom_index].frame_index,
                            bottom_knee_angle_deg=float(smoothed_signal[bottom_index]),
                        )
                    )

            phase = MovementPhase.STANDING
            last_standing_index = index
            rep_start_index = None
            bottom_index = None
            descending_samples = 0

    if phase is not MovementPhase.STANDING:
        warnings["incomplete_motion_at_end"] += 1
    if incomplete_motion_before_baseline:
        warnings["incomplete_motion_before_first_standing"] += 1

    return DetectionResult(
        repetitions=tuple(repetitions),
        raw_signal=raw_signal,
        smoothed_signal=smoothed_signal,
        thresholds=thresholds,
        warning_counts=dict(warnings),
    )
