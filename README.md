# Spatial Quality Lab

Spatial Quality Lab is a Python project for learning and testing spatial
data-processing systems.

The project focuses on quality engineering for:

- 3D coordinates
- point clouds
- meshes
- camera models
- spatial transformations
- 3D reconstruction
- Gaussian Splatting

## Milestones

### Milestone 1: 3D Coordinates and Transformations

Goals:

- represent 3D points with NumPy
- calculate distances, centroids, and bounding boxes
- apply translation, scaling, and rotation
- use homogeneous transformation matrices
- test mathematical properties and invalid inputs
- understand floating-point tolerance

### Milestone 2: PLY Point Cloud Pipeline

Implemented capabilities:

- load and export PLY point clouds
- fit planar surfaces with Open3D RANSAC
- measure surface angle and point-to-plane error
- apply configurable quality thresholds
- export inspection results with distance colors

### Planned Milestone 3: Performance Regression

Measure runtime and memory usage and run regression checks in CI.

## Experiments

### Room alignment

The first experiment generates a room, deliberately misaligns and corrupts
its point cloud, cleans it, and applies the known inverse transformation:

```powershell
.\.venv\Scripts\python.exe examples\inspect_misaligned_room.py
```

### Tilted-wall quality inspection

The tilted-wall experiment extracts the back wall from the generated room,
tilts it around its bottom edge, adds repeatable measurement noise, fits a
plane with Open3D RANSAC, and evaluates the surface against quality limits:

```powershell
.\.venv\Scripts\python.exe examples\inspect_tilted_wall.py
```

The default wall is tilted by 5 degrees. The default acceptance limit is 2
degrees, so the report should fail because of its angle. Point colors show
distance from the fitted plane:

- green: close to the fitted plane
- yellow: at the accepted distance
- red: beyond the accepted distance or rejected by RANSAC

Run the experiment without opening an Open3D window and export its point
clouds as PLY files:

```powershell
.\.venv\Scripts\python.exe examples\inspect_tilted_wall.py `
    --angle 5 `
    --noise 0.005 `
    --no-viewer `
    --export-dir output\tilted-wall
```

Useful options include `--maximum-angle`, `--maximum-rmse`, and
`--distance-threshold`.

## Plane-quality model

`spatial_quality.plane_quality` contains the reusable inspection logic:

- `fit_plane_ransac` fits the largest supported plane.
- `calculate_point_to_plane_distances` measures perpendicular distances.
- `calculate_normal_angle` compares measured and expected orientations.
- `inspect_plane_quality` creates a `PlaneQualityReport` and applies limits.

A report passes only when all three conditions pass:

1. the normal angle is at or below `maximum_angle_degrees`;
2. the inlier point-to-plane RMSE is at or below `maximum_rmse`;
3. the RANSAC inlier ratio is at or above `minimum_inlier_ratio`.

Plane distance statistics use RANSAC inliers. The report retains distances
for every input point so that rejected points can still be colored and
exported.

## PLY wall pipeline

`inspect_ply_wall.py` expects a PLY file containing one target wall or another
single planar surface. It reads the points, removes invalid and duplicate
points, fits the plane, prints a quality report, and exports a colored PLY:

```powershell
.\.venv\Scripts\python.exe examples\inspect_ply_wall.py `
    output\tilted-wall\measured_wall.ply `
    --output output\tilted-wall\measured_wall_inspected.ply
```

Use `--no-viewer` for a command-line-only run. The default expected normal is
the positive y-axis. For a floor, specify the z-axis instead:

```powershell
--expected-normal 0 0 1
```

The low-level PLY functions are available in
`spatial_quality.point_cloud_io` for reuse by later pipelines.

## Tests and code quality

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
```
