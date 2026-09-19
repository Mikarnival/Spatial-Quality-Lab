from pathlib import Path

import numpy as np
import pytest

from spatial_quality.point_cloud_io import (
    read_point_cloud_points,
    write_point_cloud_points,
)


def test_ply_round_trip_preserves_points_and_colors(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [-1.0, 0.5, 7.0],
        ]
    )
    colors = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    path = tmp_path / "wall.ply"

    written_path = write_point_cloud_points(path, points, colors)
    loaded_points = read_point_cloud_points(path)

    assert written_path == path
    assert loaded_points == pytest.approx(points)


def test_writer_rejects_color_count_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same shape"):
        write_point_cloud_points(
            tmp_path / "wall.ply",
            np.zeros((3, 3)),
            np.zeros((2, 3)),
        )
