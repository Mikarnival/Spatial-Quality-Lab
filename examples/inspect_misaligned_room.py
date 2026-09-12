import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from spatial_quality.metrics import (
    calculate_centroid_error,
    calculate_rmse,
)
from spatial_quality.room_generator import create_room_points
from spatial_quality.transformations import (
    create_z_rotation_transform,
    invert_transformation,
    transform_points,
)
from spatial_quality.validation import (
    clean_points,
    get_extreme_points,
    validate_points,
)


def inject_corruption(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Add several deliberate defects to a point cloud."""
    duplicate_points = points[:25].copy()

    corrupt_points = np.array(
        [
            [np.nan, 0.0, 0.0],
            [0.0, np.inf, 0.0],
            [10000.0, 10000.0, 10000.0],
        ],
        dtype=np.float64,
    )

    return np.vstack(
        [
            points,
            duplicate_points,
            corrupt_points,
        ]
    )


def create_open3d_cloud(
    points: NDArray[np.float64],
    color: tuple[float, float, float],
) -> o3d.geometry.PointCloud:
    """Convert a NumPy point array into an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(color)

    return cloud


def print_report(
    input_points: NDArray[np.float64],
    reference_points: NDArray[np.float64],
    cleaned_scan: NDArray[np.float64],
    recovered_points: NDArray[np.float64],
) -> None:
    report = validate_points(input_points)

    before_centroid_error = calculate_centroid_error(
        reference_points,
        cleaned_scan,
    )
    before_rmse = calculate_rmse(
        reference_points,
        cleaned_scan,
    )

    after_centroid_error = calculate_centroid_error(
        reference_points,
        recovered_points,
    )
    after_rmse = calculate_rmse(
        reference_points,
        recovered_points,
    )

    passed = (
        after_centroid_error < 1e-9
        and after_rmse < 1e-9
    )

    print()
    print("SPATIAL INSPECTION REPORT")
    print("=" * 42)
    print(f"Input points:               {report.input_points}")
    print(f"Finite points:              {report.finite_points}")
    print(f"NaN points:                 {report.nan_points}")
    print(f"Infinite points:            {report.infinite_points}")
    print(f"Duplicate points:           {report.duplicate_points}")
    print(
        "Extreme-coordinate points: "
        f"{report.extreme_coordinate_points}"
    )

    print()
    print("Before alignment")
    print(
        f"Centroid error:             "
        f"{before_centroid_error:.6f} m"
    )
    print(
        f"Alignment RMSE:             "
        f"{before_rmse:.6f} m"
    )

    print()
    print("After alignment")
    print(
        f"Centroid error:             "
        f"{after_centroid_error:.12f} m"
    )
    print(
        f"Alignment RMSE:             "
        f"{after_rmse:.12f} m"
    )

    print()
    print(f"Result: {'PASS' if passed else 'FAIL'}")


def main() -> None:
    reference_points = create_room_points()

    transformation = create_z_rotation_transform(
        angle_degrees=20.0,
        translation=np.array(
            [1.5, -0.8, 0.2],
            dtype=np.float64,
        ),
    )

    misaligned_points = transform_points(
        reference_points,
        transformation,
    )

    corrupt_scan = inject_corruption(
        misaligned_points,
    )

    extreme_points = get_extreme_points(
        corrupt_scan,
    )

    cleaned_scan = clean_points(
        corrupt_scan,
    )

    inverse_transformation = invert_transformation(
        transformation,
    )

    recovered_points = transform_points(
        cleaned_scan,
        inverse_transformation,
    )

    print_report(
        input_points=corrupt_scan,
        reference_points=reference_points,
        cleaned_scan=cleaned_scan,
        recovered_points=recovered_points,
    )

    reference_cloud = create_open3d_cloud(
        reference_points,
        color=(0.65, 0.65, 0.65),
    )

    misaligned_cloud = create_open3d_cloud(
        cleaned_scan,
        color=(0.1, 0.4, 1.0),
    )

    extreme_cloud = create_open3d_cloud(
        extreme_points,
        color=(1.0, 0.0, 0.0),
    )

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0
        )
    )

    print()
    print("Viewer 1")
    print("Grey: reference room")
    print("Blue: misaligned scan")
    print("Red: extreme-coordinate points")
    print("Close the window to continue.")

    o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
        [
            reference_cloud,
            misaligned_cloud,
            extreme_cloud,
            coordinate_frame,
        ],
        window_name="Misaligned Room Inspection",
        width=1200,
        height=800,
    )

    recovered_cloud = create_open3d_cloud(
        recovered_points,
        color=(0.1, 0.9, 0.3),
    )

    print()
    print("Viewer 2")
    print("Grey: reference room")
    print("Green: recovered scan")
    print("Close the window to finish.")

    o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
        [
            reference_cloud,
            recovered_cloud,
            coordinate_frame,
        ],
        window_name="Recovered Room Inspection",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()