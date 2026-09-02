# AI Gym Trainer — Milestones 1–3

This project currently extracts raw body-pose data from a professional trainer's
squat video, converts it into normalized squat features, and identifies complete
squat repetitions. It uses MediaPipe Pose Landmarker to detect the 33 standard
body landmarks in each frame and writes landmark, feature, and per-repetition
CSV files. Movement comparison, scoring, and live feedback are intentionally
out of scope for now.

## Project structure

```text
GYM TRAINER/
├── data/
│   ├── landmarks/                 # Generated CSV landmark data
│   ├── processed/features/        # Normalized squat-feature CSV data
│   ├── processed/reps/            # Per-rep features, summaries, debug plot
│   └── trainer_videos/            # Put trainer squat videos here
├── models/                        # Put the downloaded .task model here
├── src/
│   └── extract_trainer_landmarks.py
│   └── features/
│       ├── angles.py               # Reusable three-point angle helpers
│       ├── normalization.py        # Hip-centered, torso-scaled coordinates
│       └── extraction.py           # Landmark CSV to squat-feature CSV pipeline
│   └── reps/
│       ├── detection.py            # Smoothed state-machine rep detection
│       ├── extraction.py           # Per-rep and summary file creation
│       └── visualize_reps.py       # Knee-angle detection debug plot
├── tests/
│   └── test_angles.py
│   └── test_rep_detection.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.9–3.12
- A trainer squat video where one person's whole body is visible (for example,
  `data/trainer_videos/trainer_squat.mp4`)
- The **Pose Landmarker Full** model from the [MediaPipe Pose Landmarker model
  bundle](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task).
  Save the downloaded file as `models/pose_landmarker_full.task`.

## Install

From the project directory, install the dependencies using your Python
environment:

```powershell
# If this project contains the local runtime installed during setup:
.\.python\python.exe -m pip install -r requirements.txt

# Or, with a normal Python installation, create and activate a virtual
# environment first (optional but recommended):
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Extract trainer landmarks

Put a squat video in `data/trainer_videos/`, then run:

```powershell
.\.python\python.exe src\extract_trainer_landmarks.py `
  --video data/trainer_videos/trainer_squat.mp4 `
  --model models/pose_landmarker_full.task `
  --output data/landmarks/trainer_squat_landmarks.csv
```

If you used a normal virtual environment instead, replace
`.\.python\python.exe` with `py` (or `python`) after activating it.

The output contains one row for each of the 33 landmarks in every frame where a
person is detected. The fields are:

- `frame_index` and `timestamp_ms`: the source video location
- `landmark_index` and `landmark_name`: the MediaPipe landmark identity
- `x`, `y`, `z`: raw normalized landmark coordinates from MediaPipe
- `visibility` and `presence`: MediaPipe confidence values

Frames without a detected person are skipped; they have no CSV rows. The script
exits with a clear error if the input video or model is missing, the video cannot
be read, pose detection fails, no person is detected, or the CSV cannot be saved.

## Generate normalized squat features

After landmark extraction, create one feature row for every usable video frame:

```powershell
.\.python\python.exe -m src.features.extraction `
  --input data/landmarks/trainer_squat_landmarks.csv `
  --output data/processed/features/trainer_squat_features.csv
```

The extractor skips frames if any landmark required for squat features is missing
or has either a visibility or presence score below `0.5`. It uses the midpoint of
both hips as the body origin and the hip-to-shoulder-center distance as the body
scale. Each usable row includes:

- Left and right knee angles
- Left and right hip angles
- Torso lean from vertical
- Hip-center- and torso-length-normalized `x`, `y`, and `z` positions for both
  knees and hips

To use a different confidence threshold, append `--min-confidence 0.6` (the
value must be between `0` and `1`).

## Detect squat repetitions

The repetition detector uses the lower of the left and right knee angles each
frame as the primary flexion signal. It reduces small jitter with a rolling
median, derives standing and bottom thresholds from the observed movement, and
confirms a complete standing → descending → bottom → ascending → standing cycle.

```powershell
.\.python\python.exe -m src.reps.extraction `
  --input data/processed/features/trainer_squat_features.csv `
  --output-dir data/processed/reps
```

The command writes `rep_001.csv`, `rep_002.csv`, and so on for each complete
rep it finds. It also creates `rep_summary.csv` and `rep_summary.json` with each
rep's start frame, end frame, duration, and bottom frame. Brief, shallow, or
incomplete movements are not saved as repetitions.

Create a visual check of the chosen knee-angle signal and the resulting rep
boundaries:

```powershell
.\.python\python.exe -m src.reps.visualize_reps `
  --input data/processed/features/trainer_squat_features.csv `
  --output data/processed/reps/trainer_squat_rep_debug.png
```

## Test the angle utility

```powershell
.\.python\python.exe -m unittest tests.test_angles tests.test_rep_detection
```

## Next milestones

1. Compare user and trainer movement sequences with DTW.
2. Add real-time scoring and posture feedback.
