from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from numpy.typing import NDArray


def read_point_cloud_points(
    path: str | Path,
) -> NDArray[np.float64]:
    """Read the points from a PLY point cloud."""
    input_path = Path(path)
    if input_path.suffix.lower() != ".ply":
        raise ValueError("Input point cloud must use the .ply extension.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Point cloud does not exist: {input_path}")

    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("PLY file must contain at least one 3D point.")

    return points.copy()


def write_point_cloud_points(
    path: str | Path,
    points: NDArray[np.float64],
    colors: NDArray[np.float64] | None = None,
) -> Path:
    """Write points and optional RGB colors to a PLY point cloud."""
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Points must have non-empty shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("Points must contain only finite values.")
    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError("Colors must have the same shape as points.")
        if not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
            raise ValueError("Colors must contain finite values from zero to one.")

    output_path = Path(path)
    if output_path.suffix.lower() != ".ply":
        raise ValueError("Output point cloud must use the .ply extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)

    succeeded = o3d.io.write_point_cloud(
        str(output_path),
        cloud,
        write_ascii=True,
    )
    if not succeeded:
        raise OSError(f"Could not write point cloud: {output_path}")

    return output_path
