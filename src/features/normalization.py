"""Pose-reference and body-scale helpers for squat features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


MIN_TORSO_LENGTH = 1e-8
REFERENCE_LANDMARKS = (
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
)


@dataclass(frozen=True)
class BodyReference:
    """The origin and scale used to make a pose camera- and size-relative."""

    hip_center: np.ndarray
    shoulder_center: np.ndarray
    torso_length: float


def midpoint(first_point: Sequence[float], second_point: Sequence[float]) -> np.ndarray | None:
    """Return the midpoint of matching finite coordinate arrays."""
    try:
        first = np.asarray(first_point, dtype=float).reshape(-1)
        second = np.asarray(second_point, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if first.shape != second.shape or first.size < 2:
        return None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    return (first + second) / 2.0


def build_body_reference(landmarks: Mapping[str, Sequence[float]]) -> BodyReference | None:
    """Build a hip-centered body reference scaled by shoulder-to-hip distance."""
    if any(name not in landmarks for name in REFERENCE_LANDMARKS):
        return None

    hip_center = midpoint(landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"])
    shoulder_center = midpoint(landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"])
    if hip_center is None or shoulder_center is None:
        return None

    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    if not np.isfinite(torso_length) or torso_length <= MIN_TORSO_LENGTH:
        return None

    return BodyReference(
        hip_center=hip_center,
        shoulder_center=shoulder_center,
        torso_length=torso_length,
    )


def normalize_point(point: Sequence[float], reference: BodyReference) -> np.ndarray | None:
    """Express a landmark relative to hip center in torso-length units."""
    try:
        coordinates = np.asarray(point, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if (
        coordinates.shape != reference.hip_center.shape
        or not np.isfinite(coordinates).all()
        or reference.torso_length <= MIN_TORSO_LENGTH
    ):
        return None
    return (coordinates - reference.hip_center) / reference.torso_length


def normalize_landmarks(
    landmarks: Mapping[str, Sequence[float]], reference: BodyReference
) -> dict[str, np.ndarray] | None:
    """Normalize a set of landmarks using a previously validated body reference."""
    normalized: dict[str, np.ndarray] = {}
    for name, point in landmarks.items():
        normalized_point = normalize_point(point, reference)
        if normalized_point is None:
            return None
        normalized[name] = normalized_point
    return normalized
