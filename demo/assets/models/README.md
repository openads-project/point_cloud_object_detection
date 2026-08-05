# Model image

Build and optionally publish the Triton model repository image mounted by the
demo services at `/models`.

Place the model repository in
[`point-cloud-object-detection`](point-cloud-object-detection), then run from
the repository root:

```bash
MODEL_VERSION=v1.0.0
MODEL_IMAGE="ghcr.io/openads-project/point_cloud_object_detection/model:${MODEL_VERSION}"

docker build \
  --build-arg IMAGE_VERSION="${MODEL_VERSION}" \
  --tag "${MODEL_IMAGE}" \
  demo/assets/models
```

The default image name is already configured in
[`demo/docker-compose.yml`](../../docker-compose.yml). To use another local
image, set `MODEL_IMAGE` when running Docker Compose.

To publish the image, log in with your GitHub username and a `write:packages`
token as the password, then push:

```bash
docker login ghcr.io
docker push "${MODEL_IMAGE}"
```
