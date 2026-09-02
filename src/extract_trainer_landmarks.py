"""Extract MediaPipe Pose Landmarker data from a trainer squat video.

This is Milestone 1 of the AI gym trainer project. It deliberately records raw
pose landmarks only; normalisation, feature extraction, and movement comparison
will be added in later milestones.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# MediaPipe imports Matplotlib on some Windows installations. Keep its cache in
# the project so a restricted user profile does not produce startup warnings.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


EXPECTED_LANDMARK_COUNT = 33
LANDMARK_NAMES = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)
CSV_COLUMNS = (
    "frame_index",
    "timestamp_ms",
    "landmark_index",
    "landmark_name",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
)


class LandmarkExtractionError(Exception):
    """Raised when a video cannot be processed into a usable landmark CSV."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 33 MediaPipe pose landmarks from every detected video frame."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to a trainer squat video, for example data/trainer_videos/squat.mp4.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a MediaPipe Pose Landmarker .task model file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path, for example data/landmarks/trainer_squat.csv.",
    )
    return parser.parse_args()


def validate_input_paths(video_path: Path, model_path: Path) -> None:
    if not video_path.is_file():
        raise LandmarkExtractionError(f"Trainer video was not found: {video_path}")
    if not model_path.is_file():
        raise LandmarkExtractionError(
            f"Pose Landmarker model was not found: {model_path}\n"
            "Download a .task model as described in the README."
        )
    if model_path.suffix.lower() != ".task":
        raise LandmarkExtractionError(
            f"Expected a .task Pose Landmarker model, received: {model_path.name}"
        )


def create_landmarker(model_path: Path) -> vision.PoseLandmarker:
    try:
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)
    except Exception as error:
        raise LandmarkExtractionError(
            f"Could not initialise MediaPipe Pose Landmarker with '{model_path}': {error}"
        ) from error


def extract_landmarks(video_path: Path, model_path: Path, output_path: Path) -> tuple[int, int]:
    """Write one CSV row per landmark and return (processed_frames, detected_frames)."""
    validate_input_paths(video_path, model_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise LandmarkExtractionError(
            f"OpenCV could not open the trainer video: {video_path}\n"
            "Check that the file is a readable video format and is not in use."
        )

    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    if frames_per_second <= 0:
        capture.release()
        raise LandmarkExtractionError(
            "The trainer video does not report a valid frame rate, so timestamps cannot be created."
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        capture.release()
        raise LandmarkExtractionError(
            f"Could not create the output folder '{output_path.parent}': {error}"
        ) from error

    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    processed_frames = 0
    detected_frames = 0
    output_saved = False

    try:
        with create_landmarker(model_path) as landmarker, temporary_output.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break

                timestamp_ms = round((processed_frames / frames_per_second) * 1000)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                try:
                    result = landmarker.detect_for_video(image, timestamp_ms)
                except Exception as error:
                    raise LandmarkExtractionError(
                        f"Pose detection failed on frame {processed_frames}: {error}"
                    ) from error

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    if len(landmarks) != EXPECTED_LANDMARK_COUNT:
                        raise LandmarkExtractionError(
                            f"Pose detection returned {len(landmarks)} landmarks on frame "
                            f"{processed_frames}; expected {EXPECTED_LANDMARK_COUNT}."
                        )

                    for landmark_index, landmark in enumerate(landmarks):
                        writer.writerow(
                            {
                                "frame_index": processed_frames,
                                "timestamp_ms": timestamp_ms,
                                "landmark_index": landmark_index,
                                "landmark_name": LANDMARK_NAMES[landmark_index],
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": getattr(landmark, "visibility", ""),
                                "presence": getattr(landmark, "presence", ""),
                            }
                        )
                    detected_frames += 1

                processed_frames += 1
        if processed_frames == 0:
            raise LandmarkExtractionError(f"The trainer video contains no readable frames: {video_path}")
        if detected_frames == 0:
            raise LandmarkExtractionError(
                "No person was detected in the video. Use a clear, well-lit trainer squat video "
                "where the full body is visible."
            )

        temporary_output.replace(output_path)
        output_saved = True
    except OSError as error:
        raise LandmarkExtractionError(f"Could not write CSV output to '{output_path}': {error}") from error
    finally:
        capture.release()
        if not output_saved:
            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                pass

    return processed_frames, detected_frames


def main() -> int:
    args = parse_args()
    try:
        processed_frames, detected_frames = extract_landmarks(
            args.video, args.model, args.output
        )
    except LandmarkExtractionError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while extracting landmarks: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {detected_frames * EXPECTED_LANDMARK_COUNT} landmarks from "
        f"{detected_frames}/{processed_frames} detected frames to: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
