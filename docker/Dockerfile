#FROM --platform=linux/amd64 pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#FROM nvcr.io/nvidia/pytorch:23.05-py3
#FROM nvcr.io/nvidia/pytorch:23.06-py3
#FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    make \
    curl \
    rsync \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Disable constraint files in the container
ENV PIP_CONSTRAINT=
RUN for PIP_CONF in "/etc/xdg/pip/pip.conf" "/root/.pip/pip.conf" "/usr/pip.conf" "/root/.config/pip/pip.conf" "/etc/pip.conf"; do \
        if [ -f "$PIP_CONF" ]; then \
          sed -i 's/^no-cache-dir = true/#no-cache-dir = true/' "$PIP_CONF"; \
        fi; \
    done

# Copy the repository into the container
COPY . /app

# Set PYTHONPATH to include /app
ENV PYTHONPATH=/app

# Install Earth2Studio from the current repository
RUN uv pip install --system --break-system-packages ".[dlwp,corrdiff]"

# Install Earth2Studio with extra dependencies from the GitHub repository
# RUN uv pip install --system --break-system-packages "earth2studio[dlwp]@git+https://github.com/NVIDIA/earth2studio.git@0.7.0"

RUN pip install -r requirements_docker.txt
CMD ["/bin/bash"]