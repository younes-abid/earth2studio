#!/bin/bash

# Run the Docker container in interactive mode with GPU support
docker run -it --rm \
    --name earth2studio_container \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=8g \
    -p 8888:8888 \
    -v "$(pwd)/examples:/app/examples" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/scripts:/app/scripts" \
    -v "$(pwd)/notebooks:/app/notebooks" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v /home/younes.abid/git/physicsnemo:/workspace/physicsnemo \
    -v "/:/app/host" \
    earth2studio:latest