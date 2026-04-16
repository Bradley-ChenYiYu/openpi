#!/bin/bash
container_name="openpi"
image_name="openpi_ros2"
script_path=$(realpath "$0")
docker_path=$(dirname "$script_path")
# Determine workspace root: prefer git repo root if available,
# otherwise use the parent of the docker directory's parent
if workspace_git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
    workspace_path="$workspace_git_root"
else
    workspace_path=$(dirname "$(dirname "$docker_path")")
fi
echo "Script path: $script_path"
echo "Workspace path: $workspace_path"
workspace_name=$(basename "$workspace_path")
container_workspace="/$workspace_name"
echo "Workspace name: $workspace_name"
echo "Container workspace: $container_workspace"

# Check if the image is built
if [ -z "$(docker images -q $image_name)" ]; then
    echo "Image $image_name not found. Building the image..."
    docker build -t $image_name $docker_path
    # docker build -t "$image_name" -f "$docker_path/Dockerfile_deploy" "$docker_path"
else
    echo "Image $image_name found."
fi

if [ -z "$(docker ps -a -q -f name=$container_name)" ]; then
    echo 'container not exist'
    # --network host \
    # --privileged \
    # -v /tmp/.X11-unix:/tmp/.X11-unix \
    # -e DISPLAY=$DISPLAY \
    docker run -it -d --init \
        --name $container_name \
        --network host \
        --privileged \
        -v "$workspace_path":"$container_workspace" \
        -w "$container_workspace" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /dev:/dev \
        -e DISPLAY=$DISPLAY \
        $image_name
else
    echo 'container exist'
    docker start $container_name
fi