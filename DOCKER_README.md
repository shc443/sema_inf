# SEMA Docker Deployment Guide

**Environment-agnostic SEMA inference using Docker**

No more setup issues. No more dependency conflicts. Just consistent, reproducible results.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Google Colab Usage](#google-colab-usage)
- [Local Development](#local-development)
- [Docker Hub Distribution](#docker-hub-distribution)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **NVIDIA Docker** (for GPU): [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Input data**: Excel files with `VOC1` and `VOC2` columns

### Option 1: Use Pre-built Image (Fastest)

```bash
# Pull pre-built image from Docker Hub
docker pull YOUR_DOCKERHUB_USERNAME/sema-inference:latest

# Place your Excel files in data/input/
mkdir -p data/input data/output
cp your_files.xlsx data/input/

# Run inference (GPU)
docker run --rm --gpus all \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  YOUR_DOCKERHUB_USERNAME/sema-inference:latest

# Results will be in data/output/
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/shc443/sema_inf
cd sema_inf

# Build Docker image
docker build -t sema-inference:latest .

# Run inference
docker run --rm --gpus all \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  sema-inference:latest
```

---

## 📊 Google Colab Usage

### Method 1: Pre-built Image (Recommended)

1. Open [`sema_docker_colab.ipynb`](./sema_docker_colab.ipynb) in Google Colab
2. Update `DOCKER_IMAGE` variable with your Docker Hub image
3. Run all cells in **Option 1** section
4. Upload your Excel files when prompted
5. Download results when processing completes

### Method 2: Build from Source

1. Open [`sema_docker_colab.ipynb`](./sema_docker_colab.ipynb) in Google Colab
2. Run all cells in **Option 2** section
3. Wait 10-15 minutes for initial build
4. Upload Excel files and process

**Note**: Google Colab may have restrictions on Docker installation. If it fails, use a pre-built image from Docker Hub or run locally.

---

## 🛠️ Local Development

### Build Image

```bash
# Standard build
docker build -t sema-inference:latest .

# Build with custom tag
docker build -t sema-inference:v1.0.0 .

# Build without cache (clean build)
docker build --no-cache -t sema-inference:latest .
```

### Run with Docker Compose (Recommended for Development)

```bash
# Start service
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Advanced Usage

#### CPU-Only Mode

```bash
docker run --rm \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  sema-inference:latest
```

#### Interactive Shell for Debugging

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  sema-inference:latest bash

# Inside container
python3.10 run_simple.py
```

#### Custom Model Path

```bash
docker run --rm --gpus all \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  -v $(pwd)/custom_model:/workspace/model \
  sema-inference:latest
```

#### Jupyter Notebook in Container

```bash
docker run --rm -it --gpus all \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  sema-inference:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

---

## 🐳 Docker Hub Distribution

### 1. Build and Tag

```bash
# Build image
docker build -t sema-inference:latest .

# Tag for Docker Hub
docker tag sema-inference:latest YOUR_DOCKERHUB_USERNAME/sema-inference:latest
docker tag sema-inference:latest YOUR_DOCKERHUB_USERNAME/sema-inference:v1.0.0
```

### 2. Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push images
docker push YOUR_DOCKERHUB_USERNAME/sema-inference:latest
docker push YOUR_DOCKERHUB_USERNAME/sema-inference:v1.0.0
```

### 3. Share with Team

Share the pull command:

```bash
docker pull YOUR_DOCKERHUB_USERNAME/sema-inference:latest
```

### 4. Automate with GitHub Actions (Optional)

Create `.github/workflows/docker-build.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            YOUR_DOCKERHUB_USERNAME/sema-inference:latest
            YOUR_DOCKERHUB_USERNAME/sema-inference:${{ github.ref_name }}
```

---

## 🔧 Troubleshooting

### Build Issues

**Error: Failed to fetch packages**
```bash
# Clear Docker cache and rebuild
docker builder prune -a
docker build --no-cache -t sema-inference:latest .
```

**Error: CUDA version mismatch**
```bash
# Use CPU-only base image
# Edit Dockerfile: FROM python:3.10-slim instead of nvidia/cuda
```

### Runtime Issues

**Error: GPU not found**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**Error: Permission denied on volumes**
```bash
# Fix permissions
sudo chown -R $(whoami):$(whoami) data/

# Or run as root (not recommended)
docker run --rm --gpus all --user root \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  sema-inference:latest
```

**Error: Out of memory**
```bash
# Reduce batch size in colab_cli.py
# Or increase Docker memory limit
docker run --rm --gpus all --memory=16g \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  sema-inference:latest
```

### Colab Issues

**Error: Docker installation fails in Colab**
- Use pre-built image from Docker Hub instead
- Colab may have changed their Docker policies
- Alternative: Use Colab Pro with more permissions

**Error: GPU not available in Colab**
- Change runtime type: Runtime → Change runtime type → GPU (T4)
- Remove `--gpus all` flag (will run on CPU, much slower)

**Error: Session timeout during build**
- Use pre-built image (Option 1 in notebook)
- Build locally and push to Docker Hub
- Colab has 12-hour session limits

### Model Download Issues

**Error: HuggingFace Hub connection timeout**
```bash
# Pre-download models locally
huggingface-cli download shc443/sema2025 --local-dir ./model

# Mount model directory
docker run --rm --gpus all \
  -v $(pwd)/model:/workspace/model \
  -v $(pwd)/data/input:/workspace/data/input \
  -v $(pwd)/data/output:/workspace/data/output \
  sema-inference:latest
```

### Data Issues

**Error: No valid data after filtering**
- Check Excel files have `VOC1` and `VOC2` columns
- Ensure text is Korean language
- Check for minimum 4 characters per entry

---

## 📝 System Requirements

### Minimum (CPU Mode)
- 8GB RAM
- 20GB disk space
- Docker 20.10+

### Recommended (GPU Mode)
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (T4, V100, A100)
- 30GB disk space
- Docker 20.10+
- NVIDIA Container Toolkit 1.0+

### Tested Environments
- ✅ Ubuntu 20.04/22.04
- ✅ macOS (Intel & Apple Silicon via Rosetta)
- ✅ Windows 11 with WSL2
- ✅ Google Colab (with limitations)
- ✅ AWS EC2 g4dn.xlarge
- ✅ Google Cloud Platform with GPU

---

## 🎯 Key Features

✅ **Zero Setup**: No Python/Java/KoNLPy installation needed
✅ **Reproducible**: Same results across all environments
✅ **GPU Accelerated**: Full CUDA support for faster inference
✅ **Isolated**: No conflicts with existing Python installations
✅ **Portable**: Run anywhere Docker is available
✅ **Production Ready**: Multi-stage build, security best practices

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Google Colab Docker Guide](https://research.google.com/colaboratory/faq.html)
- [SEMA Repository](https://github.com/shc443/sema_inf)

---

## 🤝 Support

Issues? Questions?

1. Check [Troubleshooting](#troubleshooting) section
2. Review Docker logs: `docker logs <container-id>`
3. Open GitHub issue with error details
4. Include: OS, Docker version, GPU info (if applicable)

---

**Built with** ❤️ **for consistent ML deployment**
