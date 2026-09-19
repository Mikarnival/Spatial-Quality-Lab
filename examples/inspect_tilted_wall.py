import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from spatial_quality.plane_quality import (
    PlaneQualityReport,
    create_distance_colors,
    inspect_plane_quality,
)
from spatial_quality.point_cloud_io import write_point_cloud_points
from spatial_quality.room_generator import create_room_points


def extract_back_wall(
    room_points: NDArray[np.float64],
    wall_y: float = 2.0,
) -> NDArray[np.float64]:
    """Extract the back wall while excluding its shared floor edge."""
    mask = np.isclose(room_points[:, 1], wall_y) & (room_points[:, 2] > 0.0)
    wall_points = room_points[mask]
    if len(wall_points) == 0:
        raise ValueError("No back-wall points were found.")

    return wall_points


def create_tilted_wall(
    wall_points: NDArray[np.float64],
    angle_degrees: float,
    noise_standard_deviation: float = 0.0,
    random_seed: int = 7,
    wall_y: float = 2.0,
) -> NDArray[np.float64]:
    """Tilt a wall around its bottom x-axis and add optional noise."""
    if noise_standard_deviation < 0.0:
        raise ValueError("Noise standard deviation must not be negative.")

    angle_radians = np.deg2rad(angle_degrees)
    height = wall_points[:, 2]
    tilted = wall_points.copy()
    tilted[:, 1] = wall_y + height * np.sin(angle_radians)
    tilted[:, 2] = height * np.cos(angle_radians)

    if noise_standard_deviation > 0.0:
        random = np.random.default_rng(random_seed)
        tilted += random.normal(
            scale=noise_standard_deviation,
            size=tilted.shape,
        )

    return tilted


def create_cloud(
    points: NDArray[np.float64],
    colors: NDArray[np.float64],
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def print_report(report: PlaneQualityReport) -> None:
    print()
    print("TILTED WALL QUALITY REPORT")
    print("=" * 42)
    print(f"Input points:          {report.input_points}")
    print(f"RANSAC inliers:        {report.inlier_points}")
    print(f"Inlier ratio:          {report.inlier_ratio:.2%}")
    print(f"Measured tilt:         {report.measured_angle_degrees:.3f} deg")
    print(f"Mean plane distance:   {report.mean_distance:.6f} m")
    print(f"Maximum inlier error:  {report.maximum_distance:.6f} m")
    print(f"Plane RMSE:            {report.distance_rmse:.6f} m")
    print(f"Result:                {'PASS' if report.passed else 'FAIL'}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a synthetic tilted wall.")
    parser.add_argument("--angle", type=float, default=5.0)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--maximum-angle", type=float, default=2.0)
    parser.add_argument("--maximum-rmse", type=float, default=0.01)
    parser.add_argument("--distance-threshold", type=float, default=0.015)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    reference_wall = extract_back_wall(create_room_points())
    measured_wall = create_tilted_wall(
        reference_wall,
        angle_degrees=arguments.angle,
        noise_standard_deviation=arguments.noise,
    )
    report = inspect_plane_quality(
        measured_wall,
        expected_normal=np.array([0.0, 1.0, 0.0]),
        maximum_angle_degrees=arguments.maximum_angle,
        maximum_rmse=arguments.maximum_rmse,
        distance_threshold=arguments.distance_threshold,
    )
    print_report(report)

    colors = create_distance_colors(
        report.distances,
        maximum_accepted_distance=arguments.maximum_rmse,
        inlier_indices=report.plane.inlier_indices,
    )

    if arguments.export_dir is not None:
        write_point_cloud_points(
            arguments.export_dir / "reference_wall.ply",
            reference_wall,
        )
        write_point_cloud_points(
            arguments.export_dir / "measured_wall.ply",
            measured_wall,
        )
        inspected_path = write_point_cloud_points(
            arguments.export_dir / "inspected_wall.ply",
            measured_wall,
            colors,
        )
        print(f"Exported inspected point cloud: {inspected_path}")

    if arguments.no_viewer:
        return

    reference_colors = np.full(reference_wall.shape, 0.65, dtype=np.float64)
    reference_cloud = create_cloud(reference_wall, reference_colors)
    inspected_cloud = create_cloud(measured_wall, colors)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    print("Green: close to plane; yellow: near limit; red: over limit/outlier")
    o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
        [reference_cloud, inspected_cloud, coordinate_frame],
        window_name="Tilted Wall Quality Inspection",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
