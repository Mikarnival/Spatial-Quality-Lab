import numpy as np
import pytest

from spatial_quality.coordinates import euclidean_distance


def test_calculates_distance_between_two_3d_points() -> None:
    point_a = np.array([0.0, 0.0, 0.0])
    point_b = np.array([3.0, 4.0, 0.0])

    distance = euclidean_distance(point_a, point_b)

    assert distance == pytest.approx(5.0)


def test_distance_is_independent_of_point_order() -> None:
    point_a = np.array([1.0, 2.0, 3.0])
    point_b = np.array([4.0, 6.0, 3.0])

    distance_ab = euclidean_distance(point_a, point_b)
    distance_ba = euclidean_distance(point_b, point_a)

    assert distance_ab == pytest.approx(distance_ba)


def test_rejects_point_with_missing_coordinate() -> None:
    invalid_point = np.array([1.0, 2.0])
    valid_point = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="exactly three coordinates",
    ):
        euclidean_distance(invalid_point, valid_point)