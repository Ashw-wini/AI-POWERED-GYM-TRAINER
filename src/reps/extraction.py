"""Split a normalized squat feature sequence into individual repetition files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .detection import (
    DetectedRep,
    DetectionConfig,
    RepDetectionError,
    detect_repetitions,
    load_feature_samples,
)


SUMMARY_COLUMNS = (
    "rep_number",
    "start_frame",
    "end_frame",
    "duration_frames",
    "duration_ms",
    "bottom_frame",
    "bottom_knee_angle_deg",
)


@dataclass(frozen=True)
class RepExtractionSummary:
    """The files and counts produced by one rep-extraction run."""

    input_frames: int
    detected_repetitions: int
    output_directory: Path
    summary_csv: Path
    summary_json: Path
    warning_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect individual squats from a normalized feature CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Milestone 2 feature CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/reps"),
        help="Folder for per-rep CSVs and summaries (default: data/processed/reps).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Odd rolling-median window for jitter reduction (default: 5).",
    )
    parser.add_argument(
        "--min-rep-frames",
        type=int,
        default=15,
        help="Reject a repetition shorter than this many source frames (default: 15).",
    )
    arguments = parser.parse_args()
    if arguments.smoothing_window < 1 or arguments.smoothing_window % 2 == 0:
        parser.error("--smoothing-window must be a positive odd number.")
    if arguments.min_rep_frames < 1:
        parser.error("--min-rep-frames must be at least 1.")
    return arguments


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Write a CSV via a temporary file so partial output is never treated as valid."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def save_detection_output(
    output_directory: Path,
    fieldnames: tuple[str, ...],
    samples: list,
    repetitions: tuple[DetectedRep, ...],
) -> tuple[Path, Path]:
    """Save each rep's source features plus portable CSV and JSON summaries."""
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        for repetition in repetitions:
            rep_path = output_directory / f"rep_{repetition.rep_number:03d}.csv"
            rep_rows = [
                sample.row
                for sample in samples[
                    repetition.start_sample_index : repetition.end_sample_index + 1
                ]
            ]
            write_csv(rep_path, fieldnames, rep_rows)

        summary_records = [repetition.as_summary_record() for repetition in repetitions]
        summary_csv = output_directory / "rep_summary.csv"
        write_csv(summary_csv, SUMMARY_COLUMNS, summary_records)

        summary_json = output_directory / "rep_summary.json"
        temporary_json = summary_json.with_suffix(".json.tmp")
        with temporary_json.open("w", encoding="utf-8") as output_file:
            json.dump(summary_records, output_file, indent=2)
        temporary_json.replace(summary_json)
    except OSError as error:
        raise RepDetectionError(f"Could not save repetition output: {error}") from error

    return summary_csv, summary_json


def extract_repetitions(
    input_path: Path,
    output_directory: Path,
    config: DetectionConfig = DetectionConfig(),
) -> RepExtractionSummary:
    """Detect repetitions and save their original normalized feature rows."""
    samples, fieldnames, input_warnings = load_feature_samples(input_path)
    detection_result = detect_repetitions(samples, config)
    summary_csv, summary_json = save_detection_output(
        output_directory, fieldnames, samples, detection_result.repetitions
    )
    warning_counts: Counter[str] = Counter(input_warnings)
    warning_counts.update(detection_result.warning_counts)
    return RepExtractionSummary(
        input_frames=len(samples),
        detected_repetitions=len(detection_result.repetitions),
        output_directory=output_directory,
        summary_csv=summary_csv,
        summary_json=summary_json,
        warning_counts=dict(warning_counts),
    )


def main() -> int:
    arguments = parse_args()
    config = DetectionConfig(
        smoothing_window=arguments.smoothing_window,
        min_rep_frames=arguments.min_rep_frames,
    )
    try:
        summary = extract_repetitions(arguments.input, arguments.output_dir, config)
    except (RepDetectionError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while detecting repetitions: {error}", file=sys.stderr)
        return 1

    print(f"Input feature frames: {summary.input_frames}")
    print(f"Detected repetitions: {summary.detected_repetitions}")
    print(f"Rep files and summaries saved to: {summary.output_directory}")
    print(f"Summary CSV: {summary.summary_csv}")
    print(f"Summary JSON: {summary.summary_json}")
    if summary.warning_counts:
        print("Warnings:")
        for warning, count in sorted(summary.warning_counts.items()):
            print(f"- {warning}: {count}")
    else:
        print("Warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
