import numpy as np
from numpy.typing import NDArray


def euclidean_distance(
    point_a: NDArray[np.float64],
    point_b: NDArray[np.float64],
) -> float:
    """Calculate the Euclidean distance between two 3D points."""
    if point_a.shape != (3,) or point_b.shape != (3,):
        raise ValueError("Each point must contain exactly three coordinates.")

    return float(np.linalg.norm(point_b - point_a))