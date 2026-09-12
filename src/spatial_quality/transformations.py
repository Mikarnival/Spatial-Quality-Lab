import numpy as np
from numpy.typing import NDArray


def create_z_rotation_transform(
    angle_degrees: float,
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Create a 4x4 transformation around the z-axis."""
    if translation.shape != (3,):
        raise ValueError(
            "Translation must contain exactly three values."
        )

    angle_radians = np.deg2rad(angle_degrees)

    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)

    transformation = np.array(
        [
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return transformation


def transform_points(
    points: NDArray[np.float64],
    transformation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a homogeneous transformation to 3D points."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Points must have shape (N, 3)."
        )

    if transformation.shape != (4, 4):
        raise ValueError(
            "Transformation must have shape (4, 4)."
        )

    homogeneous_points = np.column_stack(
        [
            points,
            np.ones(len(points), dtype=np.float64),
        ]
    )

    transformed_homogeneous = (
        transformation @ homogeneous_points.T
    ).T

    return transformed_homogeneous[:, :3]


def invert_transformation(
    transformation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the inverse of a 4x4 transformation."""
    if transformation.shape != (4, 4):
        raise ValueError(
            "Transformation must have shape (4, 4)."
        )

    return np.linalg.inv(transformation)