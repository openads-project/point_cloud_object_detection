# Additional dataset demos

[autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) makes it easy to run the demo with other supported datasets.

For example, the `nvidia` profile uses the [NVIDIA PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). It downloads selected scenes, converts them to ROS bags, and replays them through the detector.

Feel free to configure additional demos with one of the datasets available through [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets).

## Example setup for the [NVIDIA PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

1. Create a read-only [Hugging Face access token](https://huggingface.co/settings/tokens/new?tokenType=read).
2. Accept the dataset terms on the [NVIDIA dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
3. Store the token in `demo/.env`:

   ```dotenv
   HF_TOKEN=your_token_here
   ```

4. Allow the containers to access the local X server and start the profile:

   ```bash
   xhost +local:
   cd demo
   docker compose --profile nvidia up -d
   ```

   ⚠️ The first scene may take at least several minutes to download and convert. 
   
   ℹ️ Change the country with `nvidia_filter_countries` in [`config/autonomy_datasets.nvidia.params.yml`](config/autonomy_datasets.nvidia.params.yml). Generated data is stored under `demo/data/datasets` and can be reused in later runs.

5. Shutdown of the demo:

    ```bash
    docker compose --profile nvidia down
    xhost -local:
    ```

## Custom PCD files

To process your own PCD files, follow [these instructions](services/pcd-publisher/README.md).
