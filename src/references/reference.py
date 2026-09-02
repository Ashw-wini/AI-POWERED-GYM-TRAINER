"""Build a temporally normalized mean-and-variation squat reference sequence."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IDENTITY_COLUMNS = ("frame_index", "timestamp_ms")


class ReferenceBuildError(Exception):
    """Raised when trainer repetition data cannot form a reference sequence."""


@dataclass(frozen=True)
class TrainerRep:
    """One detected rep retaining all normalized squat-feature measurements."""

    source_path: Path
    frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    feature_names: tuple[str, ...]
    values: np.ndarray
    bottom_sample_index: int | None


@dataclass(frozen=True)
class ReferenceSequence:
    """Mean and population-standard-deviation movement values at every time point."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    standard_deviation: np.ndarray
    normalized_length: int


@dataclass(frozen=True)
class ReferenceBuildSummary:
    """The outputs and data quality notes from one reference-build run."""

    reps_used: int
    normalized_length: int
    feature_names: tuple[str, ...]
    reference_csv: Path
    metadata_json: Path
    warning_counts: dict[str, int]


def parse_finite_float(value: str | None) -> float | None:
    """Convert a CSV cell to a finite floating-point value."""
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def load_bottom_frame_metadata(reps_directory: Path) -> dict[int, int]:
    """Read Milestone 3 summary metadata, returning a rep-number to bottom-frame map."""
    summary_path = reps_directory / "rep_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as summary_file:
            records = json.load(summary_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceBuildError(f"Could not read repetition summary '{summary_path}': {error}") from error

    if not isinstance(records, list):
        raise ReferenceBuildError(f"Repetition summary must contain a list: {summary_path}")

    bottom_frames: dict[int, int] = {}
    for record in records:
        try:
            rep_number = int(record["rep_number"])
            bottom_frame = int(record["bottom_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        bottom_frames[rep_number] = bottom_frame
    return bottom_frames


def rep_number_from_path(rep_path: Path) -> int | None:
    """Return the numeric suffix of a Milestone 3 `rep_###.csv` filename."""
    try:
        return int(rep_path.stem.removeprefix("rep_"))
    except ValueError:
        return None


def load_rep_csv(
    rep_path: Path, bottom_frame: int | None
) -> tuple[TrainerRep | None, Counter[str]]:
    """Load one rep file and reject malformed or non-numeric feature rows safely."""
    warnings: Counter[str] = Counter()
    try:
        with rep_path.open("r", newline="", encoding="utf-8") as source_file:
            reader = csv.DictReader(source_file)
            if reader.fieldnames is None or not set(IDENTITY_COLUMNS).issubset(reader.fieldnames):
                warnings["reps_missing_identity_columns"] += 1
                return None, warnings

            feature_names = tuple(
                field for field in reader.fieldnames if field not in IDENTITY_COLUMNS
            )
            if not feature_names:
                warnings["reps_without_feature_columns"] += 1
                return None, warnings

            frame_indices: list[int] = []
            timestamps_ms: list[int] = []
            values: list[list[float]] = []
            for row in reader:
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_ms = int(row["timestamp_ms"])
                except (TypeError, ValueError):
                    warnings["malformed_rep_identifier_rows"] += 1
                    continue

                feature_values = [parse_finite_float(row.get(name)) for name in feature_names]
                if any(value is None for value in feature_values):
                    warnings["malformed_rep_feature_rows"] += 1
                    continue
                frame_indices.append(frame_index)
                timestamps_ms.append(timestamp_ms)
                values.append([float(value) for value in feature_values if value is not None])
    except OSError as error:
        raise ReferenceBuildError(f"Could not read trainer rep '{rep_path}': {error}") from error

    if len(values) < 2:
        warnings["reps_with_fewer_than_two_valid_samples"] += 1
        return None, warnings

    frame_array = np.asarray(frame_indices, dtype=int)
    timestamp_array = np.asarray(timestamps_ms, dtype=int)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(timestamp_array, kind="stable")
    if not np.array_equal(order, np.arange(order.size)):
        warnings["out_of_order_rep_rows"] += 1
        frame_array = frame_array[order]
        timestamp_array = timestamp_array[order]
        value_array = value_array[order]

    if np.any(np.diff(timestamp_array) <= 0):
        warnings["reps_with_non_increasing_timestamps"] += 1
        return None, warnings

    bottom_sample_index: int | None = None
    if bottom_frame is not None:
        matching_indices = np.flatnonzero(frame_array == bottom_frame)
        if matching_indices.size:
            bottom_sample_index = int(matching_indices[0])
        else:
            warnings["bottom_frame_not_found_in_rep"] += 1

    return (
        TrainerRep(
            source_path=rep_path,
            frame_indices=frame_array,
            timestamps_ms=timestamp_array,
            feature_names=feature_names,
            values=value_array,
            bottom_sample_index=bottom_sample_index,
        ),
        warnings,
    )


def load_trainer_repetitions(
    reps_directory: Path,
) -> tuple[list[TrainerRep], Counter[str]]:
    """Load every per-rep CSV produced by Milestone 3 with a matching schema."""
    if not reps_directory.is_dir():
        raise ReferenceBuildError(f"Trainer reps folder was not found: {reps_directory}")

    rep_paths = sorted(
        path
        for path in reps_directory.glob("rep_*.csv")
        if rep_number_from_path(path) is not None
    )
    if not rep_paths:
        raise ReferenceBuildError(f"No per-repetition CSV files were found in: {reps_directory}")

    bottom_frames = load_bottom_frame_metadata(reps_directory)
    repetitions: list[TrainerRep] = []
    warnings: Counter[str] = Counter()
    expected_features: tuple[str, ...] | None = None
    for rep_path in rep_paths:
        rep_number = rep_number_from_path(rep_path)
        trainer_rep, rep_warnings = load_rep_csv(
            rep_path, bottom_frames.get(rep_number) if rep_number is not None else None
        )
        warnings.update(rep_warnings)
        if trainer_rep is None:
            continue
        if expected_features is None:
            expected_features = trainer_rep.feature_names
        elif trainer_rep.feature_names != expected_features:
            warnings["reps_with_inconsistent_feature_schema"] += 1
            continue
        repetitions.append(trainer_rep)

    if not repetitions:
        raise ReferenceBuildError("No valid trainer repetitions are available for reference generation.")
    return repetitions, warnings


def interpolate_sequence(values: np.ndarray, source_time: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a multi-feature sequence at evenly spaced normalized time points."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("A sequence needs at least two samples to be normalized.")
    if source_time.ndim != 1 or source_time.size != values.shape[0]:
        raise ValueError("Source timestamps must match the sequence sample count.")
    if target_length < 2:
        raise ValueError("normalization_length must be at least 2.")

    duration = float(source_time[-1] - source_time[0])
    if duration <= 0:
        raise ValueError("Sequence timestamps must increase.")
    source_progress = (source_time - source_time[0]) / duration
    target_progress = np.linspace(0.0, 1.0, target_length)
    return np.column_stack(
        [np.interp(target_progress, source_progress, values[:, column]) for column in range(values.shape[1])]
    )


def normalize_rep_sequence(
    repetition: TrainerRep, normalization_length: int, align_bottom: bool
) -> np.ndarray:
    """Normalize one rep to a fixed cycle length, optionally aligning its bottom phase."""
    if not align_bottom:
        return interpolate_sequence(repetition.values, repetition.timestamps_ms, normalization_length)

    bottom_index = repetition.bottom_sample_index
    if bottom_index is None or bottom_index <= 0 or bottom_index >= repetition.values.shape[0] - 1:
        raise ValueError("Bottom alignment needs a bottom sample inside the repetition.")

    normalized_bottom_index = normalization_length // 2
    descent_length = normalized_bottom_index + 1
    ascent_length = normalization_length - normalized_bottom_index
    descent = interpolate_sequence(
        repetition.values[: bottom_index + 1],
        repetition.timestamps_ms[: bottom_index + 1],
        descent_length,
    )
    ascent = interpolate_sequence(
        repetition.values[bottom_index:],
        repetition.timestamps_ms[bottom_index:],
        ascent_length,
    )
    return np.vstack((descent, ascent[1:]))


def aggregate_normalized_repetitions(
    normalized_sequences: list[np.ndarray], feature_names: tuple[str, ...]
) -> ReferenceSequence:
    """Calculate mean and population standard deviation across any number of reps."""
    if not normalized_sequences:
        raise ValueError("At least one normalized repetition is required.")
    sequence_stack = np.stack(normalized_sequences, axis=0)
    if sequence_stack.ndim != 3 or sequence_stack.shape[2] != len(feature_names):
        raise ValueError("Normalized sequence shapes do not match the feature schema.")
    return ReferenceSequence(
        feature_names=feature_names,
        mean=np.mean(sequence_stack, axis=0),
        standard_deviation=np.std(sequence_stack, axis=0, ddof=0),
        normalized_length=sequence_stack.shape[1],
    )


def write_reference_files(
    reference: ReferenceSequence,
    repetitions: list[TrainerRep],
    output_directory: Path,
    align_bottom: bool,
    warning_counts: Counter[str],
) -> tuple[Path, Path]:
    """Write the wide mean/std reference CSV and portable metadata JSON."""
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        reference_csv = output_directory / "squat_trainer_reference.csv"
        csv_columns = ["normalized_index", "normalized_time_percent"]
        for feature_name in reference.feature_names:
            csv_columns.extend((f"{feature_name}_mean", f"{feature_name}_std"))

        temporary_csv = reference_csv.with_suffix(".csv.tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=csv_columns)
            writer.writeheader()
            for index in range(reference.normalized_length):
                row: dict[str, float | int] = {
                    "normalized_index": index,
                    "normalized_time_percent": 100.0 * index / (reference.normalized_length - 1),
                }
                for feature_index, feature_name in enumerate(reference.feature_names):
                    row[f"{feature_name}_mean"] = float(reference.mean[index, feature_index])
                    row[f"{feature_name}_std"] = float(
                        reference.standard_deviation[index, feature_index]
                    )
                writer.writerow(row)
        temporary_csv.replace(reference_csv)

        metadata_json = output_directory / "squat_trainer_reference_metadata.json"
        metadata = {
            "exercise": "squat",
            "number_of_trainer_reps_used": len(repetitions),
            "normalization_length": reference.normalized_length,
            "features_used": list(reference.feature_names),
            "source_files": [str(repetition.source_path) for repetition in repetitions],
            "alignment_method": (
                "bottom-aligned piecewise linear interpolation" if align_bottom else "linear interpolation"
            ),
            "bottom_alignment_index": reference.normalized_length // 2 if align_bottom else None,
            "variability_measure": "population standard deviation",
            "reference_csv": str(reference_csv),
            "warning_counts": dict(sorted(warning_counts.items())),
        }
        temporary_json = metadata_json.with_suffix(".json.tmp")
        with temporary_json.open("w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, indent=2)
        temporary_json.replace(metadata_json)
    except OSError as error:
        raise ReferenceBuildError(f"Could not save trainer reference: {error}") from error

    return reference_csv, metadata_json


def build_trainer_reference(
    reps_directory: Path,
    output_directory: Path,
    normalization_length: int = 100,
) -> ReferenceBuildSummary:
    """Build one reusable squat reference from all valid trainer repetition files."""
    if normalization_length < 2:
        raise ReferenceBuildError("normalization_length must be at least 2.")

    repetitions, warnings = load_trainer_repetitions(reps_directory)
    align_bottom = all(rep.bottom_sample_index is not None for rep in repetitions)
    if not align_bottom:
        warnings["bottom_alignment_unavailable"] += 1

    normalized_sequences: list[np.ndarray] = []
    usable_repetitions: list[TrainerRep] = []
    for repetition in repetitions:
        try:
            normalized_sequences.append(
                normalize_rep_sequence(repetition, normalization_length, align_bottom)
            )
            usable_repetitions.append(repetition)
        except ValueError:
            warnings["reps_that_could_not_be_normalized"] += 1

    if not normalized_sequences:
        raise ReferenceBuildError("No trainer repetitions could be normalized into a reference sequence.")

    reference = aggregate_normalized_repetitions(
        normalized_sequences, usable_repetitions[0].feature_names
    )
    reference_csv, metadata_json = write_reference_files(
        reference, usable_repetitions, output_directory, align_bottom, warnings
    )
    return ReferenceBuildSummary(
        reps_used=len(usable_repetitions),
        normalized_length=normalization_length,
        feature_names=reference.feature_names,
        reference_csv=reference_csv,
        metadata_json=metadata_json,
        warning_counts=dict(warnings),
    )
