# point_cloud_object_detection

<p align="center">
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
</p>

This repository provides a ROS 2 point-cloud object detection node for automated driving perception stacks. The node subscribes to a `sensor_msgs/msg/PointCloud2`, sends preprocessed point data to a Triton-served detection model, and publishes detected objects as `perception_msgs/msg/ObjectList`.

The detector itself does not host the neural network. A [Triton Inference Server](https://github.com/triton-inference-server/server) with a compatible exported model repository must be available at runtime.

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).


## 🚀 Quick Start

Run the ready-made demo setup from [`demo`](demo), which starts Triton, the object detection node, a PCD publisher, RViz, and an `rqt` parameter GUI.

1. Launch the demo Docker Compose setup.
    ```bash
    cd demo
    xhost +local:
    docker compose up
    ```
1. Inspect the published point cloud, object list, detection area, no-detection zone, model bounds, and optional grid-map outputs in RViz.
1. Stop the demo and clean up.
    > *Ctrl+C*
    ```bash
    docker compose down
    xhost -local:
    ```

## 💻 Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://gitlab.ika.rwth-aachen.de/fb-fi/its-modules/perception/point_cloud_object_detection.git
    ```
1. Initialize the [`.openads-dev-environment`](https://github.com/openads-project/openads-dev-environment) submodule containing development environment configuration.
    ```bash
    cd point_cloud_object_detection
    git submodule update --init --recursive
    ```
1. Open the repository in [Visual Studio Code](https://code.visualstudio.com).
    ```bash
    code .
    ```
1. Install the recommended VS Code extensions.
    > *Ctrl+Shift+P / Extensions: Show Recommended Extensions / Install Workspace Recommended Extensions (Cloud Download Icon)*
1. Reopen the repository in a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers).
    > *Ctrl+Shift+P / Dev Containers: Rebuild and Reopen in Container*

### Build

> *Ctrl+Shift+B*

```bash
colcon build
```

### Run Tests

> *Ctrl+Shift+P / Tasks: Run Test Task*

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=1
colcon test
colcon test-result --verbose
```


## 📝 Documentation

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://fb-fi.pages.ika.rwth-aachen.de/its-modules/perception/point_cloud_object_detection).

| Package | Description |
| --- | --- |
| [point_cloud_object_detection](point_cloud_object_detection/README.md) | Detects objects in point clouds |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

Development and maintenance of this repository are supported by the Institute for Automotive Engineering (ika) at RWTH Aachen University.
