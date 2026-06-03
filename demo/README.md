# Demo

This directory contains a self-contained demo setup for the `point_cloud_object_detection` node. It replays sample `.pcd` files, serves a Triton model repository, starts the detection node, and opens RViz plus an `rqt` parameter GUI.

## Components

- `docker-compose.yml`: starts the full demo stack.
- `data/`: sample point clouds published on `/demo/points`.
- `models/`: Triton model repository mounted into the Triton server and the detection container.
- `point_cloud_object_detection.demo.params.yml`: runtime parameters for the detection node used by the demo.
- `rviz/config.rviz`: RViz configuration for the demo topics.
- `pcd-publisher/`: small publisher image that loops over the sample `.pcd` files.

## Services

- `triton-server`: serves the model repository from `./models`.
- `point-cloud-object-detection`: runs the packaged detection node against the Triton server.
- `pcd-publisher`: publishes the sample point clouds to `/demo/points`.
- `rviz`: visualizes `/demo/points`, `/demo/objects`, `/demo/detection_area`, `/demo/no_detection_zone`, `/demo/no_detection_zone_points`, and `/demo/model_bounds`.
- `ros-parameter-gui`: starts `rqt_reconfigure` for interactive parameter changes.

## Run

From this directory:

```bash
xhost +local:
docker compose up
```

The detection node consumes `/demo/points` and publishes detections on `/demo/objects`. It also publishes polygons on `/demo/detection_area`, `/demo/no_detection_zone`, and `/demo/model_bounds`, no-detection-zone points on `/demo/no_detection_zone_points`, and grid maps on `/demo/density_grid_map`, `/demo/occupancy_grid_map`, `/demo/combined_grid_map`, and `/demo/static_grid_map`.

## Interacting Through `rqt`

The `ros-parameter-gui` service starts `rqt` with the `rqt_reconfigure` plugin. Use it to inspect and adjust the running parameters of the `point_cloud_object_detection` node while the demo is active.

In the `rqt` window:

- Select the `point_cloud_object_detection` node in the parameter tree.
- Modify dynamic parameters and observe the result in RViz.
