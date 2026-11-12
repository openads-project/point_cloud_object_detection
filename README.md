# point_cloud_object_detection

<p align="center">
  <a href="https://openads-project.github.io"><img src="https://img.shields.io/badge/OpenADS-45ccc6"/></a>
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/releases/latest"><img src="https://img.shields.io/github/v/release/openads-project/point_cloud_object_detection"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/blob/main/LICENSE"><img src="https://img.shields.io/github/license/openads-project/point_cloud_object_detection"/></a>
  <br>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/docker-ros.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/compose-oci.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/compose-oci.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/helm-oci.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/helm-oci.yml/badge.svg"/></a>
  <a href="https://openads-project.github.io/point_cloud_object_detection"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/consistency.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

**ROS 2 Object Detection in Point Clouds for Automated Driving**

This repository provides a ROS 2 point cloud object detection node for automated driving perception stacks. The node subscribes to a `sensor_msgs/msg/PointCloud2`, sends preprocessed point data to a Triton-served detection model, and publishes detected objects as `perception_msgs/msg/ObjectList`. Additionally, the node optionally provides four auxiliary grid maps published as `nav_msgs/msg/OccupancyGrid`.

The detector itself does not host the neural network. A [Triton Inference Server](https://github.com/triton-inference-server/server) with a compatible exported model repository must be available at runtime.

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://openads-project.github.io/), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).


  <video src="https://github.com/user-attachments/assets/89f1dbc1-1e82-403a-96e9-3422ca66a526" width="720" style="max-width: 100%;">
  </video>

## 🚀 Quick Start

The [demo](demo) provides an example setup for the `point_cloud_object_detection` node. It can download and replay the [DrivIng dataset](https://doi.org/10.7910/DVN/VBZKDY) with the help of the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package. Alternatively, it can replay ten PCD files contained in the demo's PCD publisher image. See [Additional dataset demos](demo/README.md) for another example and information on how to process your own data.

> [!NOTE]
> Running the demo requires an NVIDIA GPU with compute capability 8.0 or higher, and a host NVIDIA driver compatible with CUDA 13.1 or newer. 8 GB of VRAM is recommended.

1. Allow local Docker containers to connect to the X server for RViz visualization.

   ```bash
   xhost +local:
   ```

2. Choose the data to run the demo on by setting `DATASET` in [`demo/.env`](demo/.env).

   ```dotenv
   DATASET=pcd
   ```

   - `pcd` replays and processes the ten PCD files provided by the PCD publisher image. This is by far the fastest way to see example output of the detection node.

   - `driving` uses [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) to download, convert, replay, and process the [DrivIng dataset](https://doi.org/10.7910/DVN/VBZKDY).

        ⚠️ Depending on your bandwidth, it may take **several hours** to start the first run due to the initial download and conversion of the selected sequence. Check the current progress with  
        `docker compose logs -f autonomy-datasets`

        ⚠️ Make sure to comply with the [DrivIng dataset’s license](https://creativecommons.org/licenses/by-nc-nd/4.0/).

        ℹ️ See [point_cloud_object_detection.driving.params.yml](demo/config/point_cloud_object_detection.driving.params.yml) for further configuration options.

        ℹ️ Since a subset of the [DrivIng dataset](https://doi.org/10.7910/DVN/VBZKDY) was used in the training of the [demo model](demo/assets/models/README.md), interpret its apparent performance in the demo with care. Apply the model to your own data to estimate the model performance.

   - `nvidia_physicalai_av_dataset` uses [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) to download, convert, replay, and process the [NVIDIA PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). This dataset requires a Hugging Face access token, which must be exported in the shell you start the demo from:

        ```bash
        export HF_TOKEN=your_token_here
        ```

        ℹ️ See [Additional dataset demos](demo/README.md) for how to get a token and accept the dataset terms.

3. Start the demo.

   ```bash
   cd demo
   docker compose up -d
   ```

   ℹ️ `demo/.env` selects the matching Docker Compose profile, so no `--profile` option is needed.

   ℹ️ For the datasets, `demo/.env` can also enable an evaluation of the detector that shows true positives, false positives and false negatives in RViz and writes a JSON report. See [Evaluating the detector](demo/README.md#evaluating-the-detector).

4. Stop the demo once you're done. Do this before changing `DATASET`, so that all services of the current run are stopped.

   ```bash
   docker compose down
   ```

5. Disable the connection to the X server after you're done with the demo:

   ```bash
   xhost -local:
   ```

#### Interacting Through `rqt`

The `ros-parameter-gui` service starts `rqt` with the `rqt_reconfigure` plugin. You may use it to inspect and adjust the parameters of the `point_cloud_object_detection` node while the demo is active. Click **Refresh** in rqt after startup if not all nodes are listed.

### 📊 Results

The metrics are computed by the `object_detection_3d` evaluation (version 1.0.0) of [autonomy_evaluation](https://github.com/thinking-cars/autonomy_evaluation/blob/main/docs/IMPLEMENTATION.md). The detector does not estimate velocities, so the velocity error is not computed and is left out of the detection score. AP is averaged over center-distance thresholds of 0.5, 1, 2 and 4 m. Classes without true positives count with an error of 1 in the mean errors, and classes without labels are not evaluated.

| Dataset | Split | Scenes | Samples | Score ↑ | mAP ↑ | mATE (m) ↓ | mASE ↓ | mAOE (rad) ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [DrivIng](https://doi.org/10.7910/DVN/VBZKDY) | night | 99 | 19,559 | 0.271 | 0.232 | 0.564 | 0.430 | 1.112 |

<details>
<summary>Per-class results on DrivIng (night)</summary>

| Class | Labels | Detections | AP ↑ | ATE (m) ↓ | ASE ↓ | AOE (rad) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 114,815 | 71,658 | 0.551 | 0.161 | 0.175 | 0.283 |
| motorcycle | 1,568 | 986 | 0.523 | 0.198 | 0.104 | 2.735 |
| pedestrian | 9,160 | 9,044 | 0.321 | 0.145 | 0.289 | 1.774 |
| bicycle | 3,886 | 1,589 | 0.205 | 0.170 | 0.327 | 1.600 |
| bus | 331 | 176 | 0.179 | 1.057 | 0.275 | 0.171 |
| utility | 3,107 | 1,043 | 0.076 | 0.780 | 0.267 | 0.332 |
| micro | 268 | 0 | 0.000 | – | – | – |
| animal | 145 | 5 | 0.000 | – | – | – |
| vru | 0 | 5 | – | – | – | – |

</details>

## 💻 Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://github.com/openads-project/point_cloud_object_detection.git
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

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://openads-project.github.io/point_cloud_object_detection).

| Package | Description |
| --- | --- |
| [point_cloud_object_detection](point_cloud_object_detection/README.md) | Provides a C++ ROS 2 node for point cloud object detection. |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

Trained models, including learned weights and model architectures, are licensed under the [AI Pubs Research-use RAIL-M License v0.1](demo/assets/models/point-cloud-object-detection/MODEL_LICENSE).

## 🙏 Acknowledgements

Development and maintenance of this repository are supported by the following projects. We acknowledge the funding of the respective institutions.

| Project | Funding Institution | Grant Number |
| --- | --- | --- |
| [AIGGREGATE](https://aiggregate.eu/) | 🇪🇺 European Union | 101202457 |
| [autotech.agil](https://www.autotechagil.de/) | 🇩🇪 Federal Ministry for Research, Technology and Space (BMFTR) | 1IS22088A |
| [UNICAR*agil*](https://www.unicaragil.de/en/) | 🇩🇪 Federal Ministry for Research, Technology and Space (BMFTR) | 16EMO0284K |

<p>
  <img src="https://www.drought.uni-freiburg.de/stressres/images/bmftr-logo/image" height=70>
  <img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_funded_en.jpg" height=70>
</p>

<sup><sub>Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor CINEA can be held responsible for them.</sub></sup>
