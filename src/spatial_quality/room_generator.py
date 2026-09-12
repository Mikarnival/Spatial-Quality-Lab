import numpy as np
from numpy.typing import NDArray


def _create_surface(
    first_values: NDArray[np.float64],
    second_values: NDArray[np.float64],
    fixed_axis: int,
    fixed_value: float,
) -> NDArray[np.float64]:
    """Create points on one flat surface."""
    first_grid, second_grid = np.meshgrid(
        first_values,
        second_values,
    )

    points = np.zeros(
        (first_grid.size, 3),
        dtype=np.float64,
    )

    variable_axes = [
        axis
        for axis in range(3)
        if axis != fixed_axis
    ]

    points[:, variable_axes[0]] = first_grid.ravel()
    points[:, variable_axes[1]] = second_grid.ravel()
    points[:, fixed_axis] = fixed_value

    return points


def create_room_points(
    step: float = 0.15,
) -> NDArray[np.float64]:
    """Create a small room containing walls, a floor and a table."""
    x_values = np.arange(-3.0, 3.0 + step, step)
    y_values = np.arange(-2.0, 2.0 + step, step)
    z_values = np.arange(0.0, 2.5 + step, step)

    floor = _create_surface(
        x_values,
        y_values,
        fixed_axis=2,
        fixed_value=0.0,
    )

    back_wall = _create_surface(
        x_values,
        z_values,
        fixed_axis=1,
        fixed_value=2.0,
    )

    left_wall = _create_surface(
        y_values,
        z_values,
        fixed_axis=0,
        fixed_value=-3.0,
    )

    table_x = np.arange(-1.0, 1.0 + step, step)
    table_y = np.arange(-0.5, 0.5 + step, step)

    table_top = _create_surface(
        table_x,
        table_y,
        fixed_axis=2,
        fixed_value=0.9,
    )

    all_points = np.vstack(
        [
            floor,
            back_wall,
            left_wall,
            table_top,
        ]
    )

    return np.unique(all_points, axis=0)