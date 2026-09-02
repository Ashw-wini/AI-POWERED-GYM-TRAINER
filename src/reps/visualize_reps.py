"""Plot the primary knee-angle signal with detected squat repetitions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep Matplotlib's cache within the workspace on restricted Windows profiles.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np

from .detection import DetectionConfig, detect_repetitions, load_feature_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a knee-angle debug plot with detected squat repetitions."
    )
    parser.add_argument("--input", type=Path, required=True, help="Milestone 2 feature CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reps/trainer_squat_rep_debug.png"),
        help="PNG destination (default: data/processed/reps/trainer_squat_rep_debug.png).",
    )
    parser.add_argument(
        "--smoothing-window", type=int, default=5, help="Odd median window (default: 5)."
    )
    return parser.parse_args()


def break_lines_at_missing_frames(frames: list[int], values: np.ndarray) -> tuple[list[float], list[float]]:
    """Insert plot breaks so dropped feature frames are not shown as measured motion."""
    plot_frames: list[float] = []
    plot_values: list[float] = []
    for index, (frame, value) in enumerate(zip(frames, values, strict=True)):
        if index > 0 and frame - frames[index - 1] > 1:
            plot_frames.append(float("nan"))
            plot_values.append(float("nan"))
        plot_frames.append(float(frame))
        plot_values.append(float(value))
    return plot_frames, plot_values


def create_debug_plot(input_path: Path, output_path: Path, smoothing_window: int = 5) -> int:
    """Render the selected signal, adaptive thresholds, and complete rep boundaries."""
    samples, _, _ = load_feature_samples(input_path)
    result = detect_repetitions(samples, DetectionConfig(smoothing_window=smoothing_window))
    frames = [sample.frame_index for sample in samples]
    plot_frames, raw_plot_values = break_lines_at_missing_frames(frames, result.raw_signal)
    _, smoothed_plot_values = break_lines_at_missing_frames(frames, result.smoothed_signal)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(plot_frames, raw_plot_values, color="0.7", linewidth=1.2, label="Raw knee signal")
    axis.plot(
        plot_frames,
        smoothed_plot_values,
        color="#1f77b4",
        linewidth=2,
        label="Smoothed knee signal",
    )

    for repetition in result.repetitions:
        axis.axvspan(repetition.start_frame, repetition.end_frame, color="#2ca02c", alpha=0.12)
        axis.axvline(repetition.start_frame, color="#2ca02c", linestyle="--", linewidth=1)
        axis.axvline(repetition.end_frame, color="#2ca02c", linestyle="--", linewidth=1)
        axis.scatter(
            repetition.bottom_frame,
            repetition.bottom_knee_angle_deg,
            color="#d62728",
            marker="v",
            zorder=3,
            label="Bottom" if repetition.rep_number == 1 else None,
        )
        axis.text(
            (repetition.start_frame + repetition.end_frame) / 2,
            axis.get_ylim()[1],
            f"Rep {repetition.rep_number}",
            ha="center",
            va="top",
        )

    if result.thresholds is not None:
        axis.axhline(
            result.thresholds.standing_threshold_deg,
            color="#9467bd",
            linestyle=":",
            label="Standing threshold",
        )
        axis.axhline(
            result.thresholds.bottom_threshold_deg,
            color="#ff7f0e",
            linestyle=":",
            label="Bottom threshold",
        )

    axis.set_title("Squat repetition detection debug view")
    axis.set_xlabel("Source video frame")
    axis.set_ylabel("Primary knee angle (degrees)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return len(result.repetitions)


def main() -> int:
    arguments = parse_args()
    if arguments.smoothing_window < 1 or arguments.smoothing_window % 2 == 0:
        raise SystemExit("Error: --smoothing-window must be a positive odd number.")
    try:
        rep_count = create_debug_plot(arguments.input, arguments.output, arguments.smoothing_window)
    except Exception as error:
        raise SystemExit(f"Error: could not create debug plot: {error}") from error
    print(f"Saved debug plot for {rep_count} detected repetitions to: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
