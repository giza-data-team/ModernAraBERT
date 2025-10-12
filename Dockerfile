# Multi-stage Dockerfile for ModernAraBERT
# Base image with CUDA support for GPU acceleration

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY README.md LICENSE CITATION.cff ./

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/results

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for Jupyter (optional)
EXPOSE 8888

# Default command
CMD ["/bin/bash"]

# Optional: Development stage with additional tools
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest \
    black \
    isort \
    flake8 \
    mypy

# Copy notebooks
COPY notebooks/ /app/notebooks/

# Copy tests
COPY tests/ /app/tests/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

