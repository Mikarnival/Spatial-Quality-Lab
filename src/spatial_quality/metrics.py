import numpy as np
from numpy.typing import NDArray


def calculate_centroid(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the geometric centre of a point cloud."""
    if len(points) == 0:
        raise ValueError(
            "Cannot calculate the centroid of an empty point cloud."
        )

    return np.mean(points, axis=0)


def calculate_centroid_error(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
) -> float:
    """Calculate the distance between two point-cloud centroids."""
    reference_centroid = calculate_centroid(reference)
    candidate_centroid = calculate_centroid(candidate)

    return float(
        np.linalg.norm(
            reference_centroid - candidate_centroid
        )
    )


def calculate_rmse(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
) -> float:
    """Calculate RMSE between corresponding 3D points."""
    if reference.shape != candidate.shape:
        raise ValueError(
            "Reference and candidate must have the same shape."
        )

    if len(reference) == 0:
        raise ValueError(
            "Cannot calculate RMSE for empty point clouds."
        )

    squared_distances = np.sum(
        (reference - candidate) ** 2,
        axis=1,
    )

    return float(
        np.sqrt(np.mean(squared_distances))
    )