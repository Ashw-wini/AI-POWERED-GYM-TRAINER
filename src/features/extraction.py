"""Convert MediaPipe landmark CSV data into normalized squat movement features."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .angles import calculate_angle, calculate_torso_angle
from .normalization import BodyReference, build_body_reference, normalize_landmarks


REQUIRED_LANDMARKS = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
)
POSITION_LANDMARKS = ("LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP")
INPUT_COLUMNS = {
    "frame_index",
    "timestamp_ms",
    "landmark_name",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
}
FEATURE_COLUMNS = (
    "frame_index",
    "timestamp_ms",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "torso_angle_deg",
    "norm_left_knee_x",
    "norm_left_knee_y",
    "norm_left_knee_z",
    "norm_right_knee_x",
    "norm_right_knee_y",
    "norm_right_knee_z",
    "norm_left_hip_x",
    "norm_left_hip_y",
    "norm_left_hip_z",
    "norm_right_hip_x",
    "norm_right_hip_y",
    "norm_right_hip_z",
)


class FeatureExtractionError(Exception):
    """Raised when landmark data cannot be converted into squat features."""


@dataclass(frozen=True)
class Landmark:
    """One MediaPipe landmark and its detection confidence."""

    coordinates: np.ndarray
    visibility: float
    presence: float


@dataclass
class LandmarkFrame:
    """All available landmarks for one source-video frame."""

    frame_index: int
    timestamp_ms: int
    landmarks: dict[str, Landmark] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionSummary:
    """A compact report of one feature-extraction run."""

    input_frames: int
    usable_frames: int
    output_path: Path
    warning_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create normalized squat features from a MediaPipe landmark CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Landmark CSV created by Milestone 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/trainer_squat_features.csv"),
        help="Feature CSV destination (default: data/processed/features/trainer_squat_features.csv).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum required MediaPipe visibility and presence score (default: 0.5).",
    )
    arguments = parser.parse_args()
    if not 0.0 <= arguments.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1.")
    return arguments


def parse_finite_float(value: str | None) -> float | None:
    """Convert a CSV cell to a finite float, returning None for bad data."""
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def load_landmark_frames(input_path: Path) -> tuple[list[LandmarkFrame], Counter[str]]:
    """Read landmark rows and retain malformed rows as warnings rather than crashes."""
    if not input_path.is_file():
        raise FeatureExtractionError(f"Landmark CSV was not found: {input_path}")

    frames: dict[int, LandmarkFrame] = {}
    warnings: Counter[str] = Counter()
    try:
        with input_path.open("r", newline="", encoding="utf-8") as source_file:
            reader = csv.DictReader(source_file)
            if reader.fieldnames is None or not INPUT_COLUMNS.issubset(reader.fieldnames):
                raise FeatureExtractionError(
                    "The input CSV does not have the required Milestone 1 landmark columns."
                )

            for line_number, row in enumerate(reader, start=2):
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_ms = int(row["timestamp_ms"])
                except (TypeError, ValueError):
                    warnings["malformed_frame_identifier_rows"] += 1
                    continue

                frame = frames.setdefault(
                    frame_index,
                    LandmarkFrame(frame_index=frame_index, timestamp_ms=timestamp_ms),
                )
                landmark_name = row.get("landmark_name", "")
                coordinates = [parse_finite_float(row.get(axis)) for axis in ("x", "y", "z")]
                visibility = parse_finite_float(row.get("visibility"))
                presence = parse_finite_float(row.get("presence"))
                if (
                    not landmark_name
                    or any(value is None for value in coordinates)
                    or visibility is None
                    or presence is None
                ):
                    warnings["malformed_landmark_rows"] += 1
                    continue

                if landmark_name in frame.landmarks:
                    warnings["duplicate_landmark_rows"] += 1
                frame.landmarks[landmark_name] = Landmark(
                    coordinates=np.asarray(coordinates, dtype=float),
                    visibility=visibility,
                    presence=presence,
                )
    except OSError as error:
        raise FeatureExtractionError(f"Could not read landmark CSV '{input_path}': {error}") from error

    if not frames:
        raise FeatureExtractionError(f"The landmark CSV contains no readable frame data: {input_path}")
    return list(frames.values()), warnings


def get_reliable_points(
    frame: LandmarkFrame, min_confidence: float
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Return all required points or a reason why the frame is unsafe to use."""
    points: dict[str, np.ndarray] = {}
    for landmark_name in REQUIRED_LANDMARKS:
        landmark = frame.landmarks.get(landmark_name)
        if landmark is None:
            return None, "frames_missing_required_landmarks"
        if landmark.visibility < min_confidence or landmark.presence < min_confidence:
            return None, "frames_low_confidence"
        points[landmark_name] = landmark.coordinates
    return points, None


def build_feature_row(
    frame: LandmarkFrame, points: dict[str, np.ndarray], reference: BodyReference
) -> dict[str, float | int] | None:
    """Calculate angles and normalized positions for one validated landmark frame."""
    normalized = normalize_landmarks(points, reference)
    if normalized is None:
        return None

    angles = {
        "left_knee_angle_deg": calculate_angle(
            points["LEFT_HIP"], points["LEFT_KNEE"], points["LEFT_ANKLE"]
        ),
        "right_knee_angle_deg": calculate_angle(
            points["RIGHT_HIP"], points["RIGHT_KNEE"], points["RIGHT_ANKLE"]
        ),
        "left_hip_angle_deg": calculate_angle(
            points["LEFT_SHOULDER"], points["LEFT_HIP"], points["LEFT_KNEE"]
        ),
        "right_hip_angle_deg": calculate_angle(
            points["RIGHT_SHOULDER"], points["RIGHT_HIP"], points["RIGHT_KNEE"]
        ),
        "torso_angle_deg": calculate_torso_angle(
            reference.hip_center, reference.shoulder_center
        ),
    }
    if any(angle is None for angle in angles.values()):
        return None

    feature_row: dict[str, float | int] = {
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        **angles,
    }
    for landmark_name in POSITION_LANDMARKS:
        point = normalized[landmark_name]
        prefix = f"norm_{landmark_name.lower()}"
        feature_row[f"{prefix}_x"] = float(point[0])
        feature_row[f"{prefix}_y"] = float(point[1])
        feature_row[f"{prefix}_z"] = float(point[2])
    return feature_row


def write_feature_rows(output_path: Path, rows: list[dict[str, float | int]]) -> None:
    """Safely write feature rows only after at least one usable frame exists."""
    if not rows:
        raise FeatureExtractionError("No usable frames were available to write feature data.")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_output.replace(output_path)
    except OSError as error:
        raise FeatureExtractionError(f"Could not write feature CSV '{output_path}': {error}") from error


def extract_squat_features(
    input_path: Path, output_path: Path, min_confidence: float = 0.5
) -> ExtractionSummary:
    """Create normalized angle and position features from an input landmark CSV."""
    frames, warnings = load_landmark_frames(input_path)
    rows: list[dict[str, float | int]] = []

    for frame in frames:
        points, unusable_reason = get_reliable_points(frame, min_confidence)
        if points is None:
            warnings[unusable_reason or "frames_with_invalid_landmarks"] += 1
            continue

        reference = build_body_reference(points)
        if reference is None:
            warnings["frames_invalid_body_reference"] += 1
            continue

        feature_row = build_feature_row(frame, points, reference)
        if feature_row is None:
            warnings["frames_invalid_feature_geometry"] += 1
            continue
        rows.append(feature_row)

    write_feature_rows(output_path, rows)
    return ExtractionSummary(
        input_frames=len(frames),
        usable_frames=len(rows),
        output_path=output_path,
        warning_counts=dict(warnings),
    )


def main() -> int:
    arguments = parse_args()
    try:
        summary = extract_squat_features(
            arguments.input, arguments.output, arguments.min_confidence
        )
    except FeatureExtractionError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while extracting features: {error}", file=sys.stderr)
        return 1

    print(f"Input frames: {summary.input_frames}")
    print(f"Usable frames: {summary.usable_frames}")
    print(f"Feature CSV saved to: {summary.output_path}")
    if summary.warning_counts:
        print("Warnings:")
        for warning, count in sorted(summary.warning_counts.items()):
            print(f"- {warning}: {count}")
    else:
        print("Warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
