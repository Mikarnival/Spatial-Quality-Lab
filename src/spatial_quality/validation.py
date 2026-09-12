from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ValidationReport:
    input_points: int
    finite_points: int
    nan_points: int
    infinite_points: int
    duplicate_points: int
    extreme_coordinate_points: int


def validate_points(
    points: NDArray[np.float64],
    coordinate_limit: float = 100.0,
) -> ValidationReport:
    """Inspect a point cloud for common data-quality problems."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Points must have shape (N, 3)."
        )

    nan_mask = np.isnan(points).any(axis=1)
    infinite_mask = np.isinf(points).any(axis=1)
    finite_mask = np.isfinite(points).all(axis=1)

    extreme_mask = np.zeros(len(points), dtype=bool)
    extreme_mask[finite_mask] = (
        np.abs(points[finite_mask]) > coordinate_limit
    ).any(axis=1)

    normal_finite_points = points[
        finite_mask & ~extreme_mask
    ]

    unique_points = np.unique(
        normal_finite_points,
        axis=0,
    )

    duplicate_count = (
        len(normal_finite_points) - len(unique_points)
    )

    return ValidationReport(
        input_points=len(points),
        finite_points=int(finite_mask.sum()),
        nan_points=int(nan_mask.sum()),
        infinite_points=int(infinite_mask.sum()),
        duplicate_points=duplicate_count,
        extreme_coordinate_points=int(extreme_mask.sum()),
    )


def clean_points(
    points: NDArray[np.float64],
    coordinate_limit: float = 100.0,
) -> NDArray[np.float64]:
    """Remove non-finite, extreme and duplicate points."""
    finite_mask = np.isfinite(points).all(axis=1)

    extreme_mask = np.zeros(len(points), dtype=bool)
    extreme_mask[finite_mask] = (
        np.abs(points[finite_mask]) > coordinate_limit
    ).any(axis=1)

    valid_points = points[
        finite_mask & ~extreme_mask
    ]

    _, first_indices = np.unique(
        valid_points,
        axis=0,
        return_index=True,
    )

    return valid_points[np.sort(first_indices)]


def get_extreme_points(
    points: NDArray[np.float64],
    coordinate_limit: float = 100.0,
) -> NDArray[np.float64]:
    """Return finite points outside the accepted coordinate range."""
    finite_mask = np.isfinite(points).all(axis=1)

    extreme_mask = np.zeros(len(points), dtype=bool)
    extreme_mask[finite_mask] = (
        np.abs(points[finite_mask]) > coordinate_limit
    ).any(axis=1)

    return points[extreme_mask]