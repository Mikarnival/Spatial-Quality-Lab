import numpy as np
import pytest

from spatial_quality.plane_quality import (
    PlaneModel,
    calculate_normal_angle,
    calculate_plane_rmse,
    calculate_point_to_plane_distances,
    create_distance_colors,
    fit_plane_ransac,
    inspect_plane_quality,
)


def create_wall(angle_degrees: float = 0.0) -> np.ndarray:
    x_grid, z_grid = np.meshgrid(
        np.linspace(-2.0, 2.0, 12),
        np.linspace(0.1, 2.5, 10),
    )
    angle = np.deg2rad(angle_degrees)
    return np.column_stack(
        [
            x_grid.ravel(),
            2.0 + z_grid.ravel() * np.sin(angle),
            z_grid.ravel() * np.cos(angle),
        ]
    )


def test_points_on_plane_have_zero_distance() -> None:
    points = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0],
            [-4.0, 2.0, 1.0],
        ]
    )
    plane = PlaneModel(
        normal=np.array([0.0, 2.0, 0.0]),
        offset=-4.0,
        inlier_indices=np.arange(3),
    )

    distances = calculate_point_to_plane_distances(points, plane)

    assert distances == pytest.approx(np.zeros(3))


def test_opposite_normals_describe_zero_plane_angle() -> None:
    angle = calculate_normal_angle(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )

    assert angle == pytest.approx(0.0)


def test_plane_rmse_uses_all_distances() -> None:
    result = calculate_plane_rmse(np.array([0.0, 3.0, 4.0]))

    assert result == pytest.approx(np.sqrt(25.0 / 3.0))


def test_ransac_measures_five_degree_wall() -> None:
    model = fit_plane_ransac(create_wall(5.0))

    angle = calculate_normal_angle(
        model.normal,
        np.array([0.0, 1.0, 0.0]),
    )

    assert angle == pytest.approx(5.0, abs=0.05)
    assert len(model.inlier_indices) == 120


def test_excessive_tilt_fails_quality_check() -> None:
    report = inspect_plane_quality(
        create_wall(5.0),
        expected_normal=np.array([0.0, 1.0, 0.0]),
        maximum_angle_degrees=2.0,
        maximum_rmse=0.01,
    )

    assert report.measured_angle_degrees == pytest.approx(5.0, abs=0.05)
    assert report.passed is False


def test_distance_colors_mark_good_limit_and_failed_points() -> None:
    colors = create_distance_colors(
        np.array([0.0, 0.01, 0.02]),
        maximum_accepted_distance=0.01,
    )

    assert colors[0] == pytest.approx([0.0, 1.0, 0.0])
    assert colors[1] == pytest.approx([1.0, 1.0, 0.0])
    assert colors[2] == pytest.approx([1.0, 0.0, 0.0])


def test_rejects_invalid_point_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_plane_ransac(np.array([[1.0, 2.0]]))
