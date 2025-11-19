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

# Install dependencies into virtual environment
# uv sync creates .venv by default
RUN uv sync --frozen

# Add .venv/bin to PATH so installed packages are available
# This allows running python/torch commands directly without 'uv run'
ENV PATH="/app/.venv/bin:$PATH"

# The code will be mounted at runtime via VESSL git import
# This allows fast iteration without rebuilding the image
