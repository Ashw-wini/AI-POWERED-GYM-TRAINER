"""Render trainer-reference mean curves and variability bands for squat form."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np


PANEL_FEATURES = (
    ("Knee angle", ("left_knee_angle_deg", "right_knee_angle_deg")),
    ("Hip angle", ("left_hip_angle_deg", "right_hip_angle_deg")),
    ("Torso angle", ("torso_angle_deg",)),
)
SERIES_STYLES = {
    "left": {"color": "#1f5aa6", "label": "Left"},
    "right": {"color": "#c47a16", "label": "Right"},
    "torso": {"color": "#5c5c5c", "label": "Torso"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the mean trainer squat movement and its variation band."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/references/squat_trainer_reference.csv"),
        help="Reference CSV produced by build_reference.py.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/references/squat_trainer_reference_metadata.json"),
        help="Reference metadata JSON produced by build_reference.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/references/squat_trainer_reference_plot.png"),
        help="PNG destination for the reference chart.",
    )
    return parser.parse_args()


def load_reference(reference_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the wide mean/std reference CSV into named numeric series."""
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference CSV was not found: {reference_path}")
    with reference_path.open("r", newline="", encoding="utf-8") as source_file:
        rows = list(csv.DictReader(source_file))
    if not rows:
        raise ValueError("Reference CSV contains no normalized sequence rows.")

    fields = rows[0].keys()
    time = np.asarray([float(row["normalized_time_percent"]) for row in rows], dtype=float)
    series = {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in fields
        if field not in {"normalized_index", "normalized_time_percent"}
    }
    return time, series


def create_reference_plot(reference_path: Path, metadata_path: Path, output_path: Path) -> None:
    """Create three aligned panels for knee, hip, and torso reference movement."""
    time, series = load_reference(reference_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Reference metadata was not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    figure.suptitle("Correct-form trainer squat reference")
    trainer_rep_count = metadata["number_of_trainer_reps_used"]
    figure.text(
        0.5,
        0.955,
        f"Mean ± population standard deviation across {trainer_rep_count} trainer rep(s)",
        ha="center",
    )
    if trainer_rep_count == 1:
        figure.text(
            0.5,
            0.935,
            "Variation bands are 0° because only one trainer rep is available.",
            ha="center",
        )

    for axis, (title, features) in zip(axes, PANEL_FEATURES, strict=True):
        for feature in features:
            mean_key = f"{feature}_mean"
            std_key = f"{feature}_std"
            if mean_key not in series or std_key not in series:
                continue
            if feature.startswith("left_"):
                style = SERIES_STYLES["left"]
            elif feature.startswith("right_"):
                style = SERIES_STYLES["right"]
            else:
                style = SERIES_STYLES["torso"]
            mean = series[mean_key]
            std = series[std_key]
            axis.plot(time, mean, color=style["color"], linewidth=2, label=style["label"])
            axis.fill_between(time, mean - std, mean + std, color=style["color"], alpha=0.18)

        axis.set_title(title, loc="left")
        axis.set_ylabel("Degrees")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Normalized squat cycle (%)")
    axes[-1].set_xlim(0, 100)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    arguments = parse_args()
    try:
        create_reference_plot(arguments.reference, arguments.metadata, arguments.output)
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: could not create trainer reference plot: {error}")
        return 1
    print(f"Saved trainer reference plot to: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
