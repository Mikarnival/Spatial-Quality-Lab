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
from spatial_quality.point_cloud_io import (
    read_point_cloud_points,
    write_point_cloud_points,
)
from spatial_quality.validation import clean_points, validate_points


def print_report(
    input_points: NDArray[np.float64],
    report: PlaneQualityReport,
) -> None:
    validation = validate_points(input_points)
    print()
    print("PLY WALL QUALITY REPORT")
    print("=" * 42)
    print(f"Input points:          {validation.input_points}")
    print(f"Finite points:         {validation.finite_points}")
    print(f"Duplicate points:      {validation.duplicate_points}")
    print(f"RANSAC inliers:        {report.inlier_points}")
    print(f"Inlier ratio:          {report.inlier_ratio:.2%}")
    print(f"Measured tilt:         {report.measured_angle_degrees:.3f} deg")
    print(f"Mean plane distance:   {report.mean_distance:.6f} m")
    print(f"Maximum inlier error:  {report.maximum_distance:.6f} m")
    print(f"Plane RMSE:            {report.distance_rmse:.6f} m")
    print(f"Result:                {'PASS' if report.passed else 'FAIL'}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read, inspect, and export a wall point cloud in PLY format."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-normal",
        nargs=3,
        type=float,
        default=[0.0, 1.0, 0.0],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--maximum-angle", type=float, default=2.0)
    parser.add_argument("--maximum-rmse", type=float, default=0.01)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.9)
    parser.add_argument("--distance-threshold", type=float, default=0.01)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    input_points = read_point_cloud_points(arguments.input)
    cleaned_points = clean_points(input_points)
    report = inspect_plane_quality(
        cleaned_points,
        expected_normal=np.asarray(arguments.expected_normal, dtype=np.float64),
        maximum_angle_degrees=arguments.maximum_angle,
        maximum_rmse=arguments.maximum_rmse,
        minimum_inlier_ratio=arguments.minimum_inlier_ratio,
        distance_threshold=arguments.distance_threshold,
    )
    print_report(input_points, report)

    colors = create_distance_colors(
        report.distances,
        maximum_accepted_distance=arguments.maximum_rmse,
        inlier_indices=report.plane.inlier_indices,
    )
    output_path = arguments.output or arguments.input.with_name(
        f"{arguments.input.stem}_inspected.ply"
    )
    write_point_cloud_points(output_path, cleaned_points, colors)
    print(f"Exported inspected point cloud: {output_path}")

    if arguments.no_viewer:
        return

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cleaned_points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
        [cloud, coordinate_frame],
        window_name="PLY Wall Quality Inspection",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
