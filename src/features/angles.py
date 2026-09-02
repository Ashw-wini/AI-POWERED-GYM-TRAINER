"""Reusable geometric helpers for pose and joint-angle calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


MIN_VECTOR_LENGTH = 1e-8


def calculate_angle(
    first_point: Sequence[float], vertex: Sequence[float], third_point: Sequence[float]
) -> float | None:
    """Return the angle in degrees at ``vertex`` formed by three landmarks.

    The function accepts two- or three-dimensional points and returns ``None``
    when a point is invalid or either direction vector has zero length.
    """
    try:
        first = np.asarray(first_point, dtype=float).reshape(-1)
        middle = np.asarray(vertex, dtype=float).reshape(-1)
        third = np.asarray(third_point, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if (
        first.size < 2
        or first.shape != middle.shape
        or middle.shape != third.shape
        or not np.isfinite(first).all()
        or not np.isfinite(middle).all()
        or not np.isfinite(third).all()
    ):
        return None

    first_vector = first - middle
    third_vector = third - middle
    first_length = np.linalg.norm(first_vector)
    third_length = np.linalg.norm(third_vector)
    if first_length <= MIN_VECTOR_LENGTH or third_length <= MIN_VECTOR_LENGTH:
        return None

    cosine = np.dot(first_vector, third_vector) / (first_length * third_length)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def calculate_torso_angle(
    hip_center: Sequence[float], shoulder_center: Sequence[float]
) -> float | None:
    """Return torso lean from upright in degrees (0° is vertically upright)."""
    try:
        hip = np.asarray(hip_center, dtype=float).reshape(-1)
        shoulder = np.asarray(shoulder_center, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if hip.size < 2 or hip.shape != shoulder.shape or not np.isfinite(hip).all():
        return None

    upright_reference = hip.copy()
    upright_reference[1] -= 1.0
    return calculate_angle(upright_reference, hip, shoulder)
