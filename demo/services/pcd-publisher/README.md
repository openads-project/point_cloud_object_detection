# PCD publisher image

Build, minify, and publish the PCD demo image. It contains the
publisher at `/opt/demo/pcd_publisher.py` and the PCDs under `/pcds`.

Place the PCD files in [`pcds`](pcds), choose an image name, and set that name
for the `pcd-publisher` service in
[`demo/docker-compose.yml`](../../docker-compose.yml). Users with registry
access may publish the image to GHCR; others can use it locally through Docker
Compose.

## Preparing PCDs

PCDs must contain `x`, `y`, `z`, and either `intensity` or `reflectivity`, and
use `DATA ascii` or uncompressed `DATA binary`. Points should already be
transformed to `base_link` (origin at the rear-axle center on the ground).

Configure the
[detector parameters](../../config/point_cloud_object_detection.pcd.params.yml):
set `input.point_feature_field` to `intensity` or `reflectivity` to match the
PCDs, set
`preprocessing.point_feature.value_threshold` to roughly its 99th percentile
for normalization, and adjust other parameters as needed.

Run from the repository root:

```bash
PCD_PUBLISHER_VERSION=v1.0.0
PCD_PUBLISHER_IMAGE="ghcr.io/openads-project/point_cloud_object_detection-pcd-publisher:${PCD_PUBLISHER_VERSION}"
PCD_PUBLISHER_SLIM_IMAGE="${PCD_PUBLISHER_IMAGE}-slim"

docker build \
  --build-arg IMAGE_VERSION="${PCD_PUBLISHER_VERSION}" \
  --target runtime \
  --tag "${PCD_PUBLISHER_IMAGE}" \
  demo/services/pcd-publisher
```

```bash
docker run --rm \
  --network host \
  --env DOCKER_API_VERSION="$(docker version --format '{{.Server.APIVersion}}')" \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  mintoolkit/mint:1.41.8 slim \
  --target "${PCD_PUBLISHER_IMAGE}" \
  --tag "${PCD_PUBLISHER_SLIM_IMAGE}" \
  --continue-after=30 \
  --show-clogs \
  --http-probe=false \
  --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  --include-new=false \
  --include-path /pcds
```

Log in with your GitHub username and a `write:packages` token as the password,
then push:

```bash
docker login ghcr.io
docker push "${PCD_PUBLISHER_SLIM_IMAGE}"
```
