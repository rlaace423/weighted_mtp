FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into the system python
# We use --system because we are in a container and don't need a venv
RUN uv sync --frozen --system

# The code will be mounted at runtime, so we don't copy it here.
# This allows for fast iteration without rebuilding the image.
