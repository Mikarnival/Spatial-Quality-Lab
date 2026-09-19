from dataclasses import dataclass

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlaneModel:
    """A normalized plane model in the form ax + by + cz + d = 0."""

    normal: NDArray[np.float64]
    offset: float
    inlier_indices: NDArray[np.int64]


@dataclass(frozen=True)
class PlaneQualityReport:
    """Measurements and acceptance result for one planar surface."""

    input_points: int
    inlier_points: int
    inlier_ratio: float
    measured_angle_degrees: float
    mean_distance: float
    maximum_distance: float
    distance_rmse: float
    passed: bool
    plane: PlaneModel
    distances: NDArray[np.float64]


def _validate_points(points: NDArray[np.float64]) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3).")
    if len(points) == 0:
        raise ValueError("Points must not be empty.")
    if not np.isfinite(points).all():
        raise ValueError("Points must contain only finite values.")


def _normalize_vector(
    vector: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")

    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError(f"{name} must not be a zero vector.")

    return vector / length


def fit_plane_ransac(
    points: NDArray[np.float64],
    distance_threshold: float = 0.01,
    sample_size: int = 3,
    iterations: int = 1000,
) -> PlaneModel:
    """Fit the largest supported plane using Open3D RANSAC."""
    _validate_points(points)
    if distance_threshold <= 0.0:
        raise ValueError("Distance threshold must be greater than zero.")
    if sample_size < 3:
        raise ValueError("Sample size must be at least three.")
    if len(points) < sample_size:
        raise ValueError("Not enough points for the requested sample size.")
    if iterations < 1:
        raise ValueError("Iterations must be at least one.")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    coefficients, inlier_indices = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=sample_size,
        num_iterations=iterations,
    )

    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    normal_length = float(np.linalg.norm(coefficient_array[:3]))
    if normal_length == 0.0:
        raise ValueError("RANSAC returned a plane with a zero normal.")

    return PlaneModel(
        normal=coefficient_array[:3] / normal_length,
        offset=float(coefficient_array[3] / normal_length),
        inlier_indices=np.asarray(inlier_indices, dtype=np.int64),
    )


def calculate_point_to_plane_distances(
    points: NDArray[np.float64],
    plane: PlaneModel,
) -> NDArray[np.float64]:
    """Return the perpendicular distance from every point to a plane."""
    _validate_points(points)
    normal = _normalize_vector(plane.normal, "Plane normal")
    normal_length = float(np.linalg.norm(plane.normal))
    normalized_offset = plane.offset / normal_length

    return np.abs(points @ normal + normalized_offset)


def calculate_normal_angle(
    measured_normal: NDArray[np.float64],
    expected_normal: NDArray[np.float64],
) -> float:
    """Return the unsigned angle between two plane normals in degrees."""
    measured = _normalize_vector(measured_normal, "Measured normal")
    expected = _normalize_vector(expected_normal, "Expected normal")
    cosine = float(np.clip(abs(measured @ expected), 0.0, 1.0))

    return float(np.rad2deg(np.arccos(cosine)))


def calculate_plane_rmse(
    distances: NDArray[np.float64],
) -> float:
    """Return the root mean square of point-to-plane distances."""
    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("Distances must be a non-empty one-dimensional array.")
    if not np.isfinite(distances).all():
        raise ValueError("Distances must contain only finite values.")

    return float(np.sqrt(np.mean(distances**2)))


def create_distance_colors(
    distances: NDArray[np.float64],
    maximum_accepted_distance: float,
    inlier_indices: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Map distances to green, yellow, and red quality colors."""
    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("Distances must be a non-empty one-dimensional array.")
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("Distances must be finite and non-negative.")
    if maximum_accepted_distance <= 0.0:
        raise ValueError("Maximum accepted distance must be greater than zero.")

    ratio = np.clip(distances / maximum_accepted_distance, 0.0, 2.0)
    colors = np.zeros((len(distances), 3), dtype=np.float64)
    colors[:, 0] = np.minimum(ratio, 1.0)
    colors[:, 1] = np.clip(2.0 - ratio, 0.0, 1.0)

    if inlier_indices is not None:
        inlier_mask = np.zeros(len(distances), dtype=bool)
        inlier_mask[inlier_indices] = True
        colors[~inlier_mask] = np.array([1.0, 0.0, 0.0])

    return colors


def inspect_plane_quality(
    points: NDArray[np.float64],
    expected_normal: NDArray[np.float64],
    maximum_angle_degrees: float = 2.0,
    maximum_rmse: float = 0.01,
    minimum_inlier_ratio: float = 0.9,
    distance_threshold: float = 0.01,
    iterations: int = 1000,
) -> PlaneQualityReport:
    """Fit a plane, measure its quality, and apply acceptance thresholds."""
    if maximum_angle_degrees < 0.0:
        raise ValueError("Maximum angle must not be negative.")
    if maximum_rmse < 0.0:
        raise ValueError("Maximum RMSE must not be negative.")
    if not 0.0 <= minimum_inlier_ratio <= 1.0:
        raise ValueError("Minimum inlier ratio must be between zero and one.")

    plane = fit_plane_ransac(
        points,
        distance_threshold=distance_threshold,
        iterations=iterations,
    )
    distances = calculate_point_to_plane_distances(points, plane)
    inlier_distances = distances[plane.inlier_indices]
    inlier_points = len(plane.inlier_indices)
    inlier_ratio = inlier_points / len(points)
    angle = calculate_normal_angle(plane.normal, expected_normal)
    mean_distance = float(np.mean(inlier_distances))
    maximum_distance = float(np.max(inlier_distances))
    distance_rmse = calculate_plane_rmse(inlier_distances)
    passed = (
        angle <= maximum_angle_degrees
        and distance_rmse <= maximum_rmse
        and inlier_ratio >= minimum_inlier_ratio
    )

    return PlaneQualityReport(
        input_points=len(points),
        inlier_points=inlier_points,
        inlier_ratio=inlier_ratio,
        measured_angle_degrees=angle,
        mean_distance=mean_distance,
        maximum_distance=maximum_distance,
        distance_rmse=distance_rmse,
        passed=passed,
        plane=plane,
        distances=distances,
    )
