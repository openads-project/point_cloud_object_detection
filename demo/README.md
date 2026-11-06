# Additional dataset demos

[autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) makes it easy to run the demo with other supported datasets.

For example, setting `DATASET=nvidia_physicalai_av_dataset` in [`.env`](.env) uses the [NVIDIA PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). It downloads selected scenes, converts them to ROS bags, and replays them through the detector.

Feel free to configure additional demos with one of the datasets available through [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets). To do so, add the dataset name to the `profiles` of the `autonomy-datasets` service in [`docker-compose.yml`](docker-compose.yml), and provide `config/autonomy_datasets.<dataset>.params.yml`, `config/point_cloud_object_detection.<dataset>.params.yml`, and `rviz/config.<dataset>.rviz`.

## Example setup for the [NVIDIA PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

1. Create a read-only [Hugging Face access token](https://huggingface.co/settings/tokens/new?tokenType=read).
2. Accept the dataset terms on the [NVIDIA dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
3. Export the token in the shell you start the demo from:

   ```bash
   export HF_TOKEN=your_token_here
   ```

4. Select the dataset in [`demo/.env`](.env):

   ```dotenv
   DATASET=nvidia_physicalai_av_dataset
   ```

5. Allow the containers to access the local X server and start the demo:

   ```bash
   xhost +local:
   cd demo
   docker compose up -d
   ```

   ⚠️ The first scene may take at least several minutes to download and convert. 
   
   ℹ️ Change the country with `nvidia_filter_countries` in [`config/autonomy_datasets.nvidia_physicalai_av_dataset.params.yml`](config/autonomy_datasets.nvidia_physicalai_av_dataset.params.yml). Generated data is stored under `demo/data/datasets`, or under `DATASET_DIR` if that variable is set, and can be reused in later runs.

6. Shutdown of the demo:

    ```bash
    docker compose down
    xhost -local:
    ```

## Custom PCD files

To process your own PCD files, follow [these instructions](services/pcd-publisher/README.md) and run the demo with `DATASET=pcd` in [`.env`](.env).
