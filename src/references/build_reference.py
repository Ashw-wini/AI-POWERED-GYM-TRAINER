"""Command-line entry point for creating a correct-form squat reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .reference import ReferenceBuildError, build_trainer_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized correct-form squat reference from trainer repetitions."
    )
    parser.add_argument(
        "--reps-dir",
        type=Path,
        default=Path("data/processed/reps"),
        help="Folder containing Milestone 3 rep_###.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/references"),
        help="Reference output folder (default: data/references).",
    )
    parser.add_argument(
        "--normalization-length",
        type=int,
        default=100,
        help="Number of points in each normalized reference sequence (default: 100).",
    )
    arguments = parser.parse_args()
    if arguments.normalization_length < 2:
        parser.error("--normalization-length must be at least 2.")
    return arguments


def main() -> int:
    arguments = parse_args()
    try:
        summary = build_trainer_reference(
            arguments.reps_dir, arguments.output_dir, arguments.normalization_length
        )
    except ReferenceBuildError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while building the trainer reference: {error}", file=sys.stderr)
        return 1

    print(f"Trainer reps used: {summary.reps_used}")
    print(f"Normalized sequence length: {summary.normalized_length}")
    print(f"Features included: {len(summary.feature_names)}")
    print(f"Reference CSV: {summary.reference_csv}")
    print(f"Metadata JSON: {summary.metadata_json}")
    if summary.warning_counts:
        print("Warnings:")
        for warning, count in sorted(summary.warning_counts.items()):
            print(f"- {warning}: {count}")
    else:
        print("Warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
