# syntax=docker/dockerfile:1

#=== Build stage: Install dependencies ===#
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment for KoNLPy (Java 11 fixes SIGSEGV crashes)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"

# Create virtual environment
RUN python3.10 -m venv /opt/venv

# Upgrade pip and install wheel
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
WORKDIR /tmp
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Pre-install KoNLPy and test it
RUN /opt/venv/bin/pip install --no-cache-dir konlpy && \
    /opt/venv/bin/python -c "from konlpy.tag import Kkma; kkma = Kkma(); print('KoNLPy OK')"


#=== Runtime stage: Minimal production image ===#
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-privileged user for security
RUN groupadd -r sema && useradd --no-log-init -r -g sema sema

# Set working directory
WORKDIR /workspace

# Copy application code
COPY --chown=sema:sema . .

# Create data directories
RUN mkdir -p data/input data/output data/train model cache && \
    chown -R sema:sema data model cache

# Switch to non-privileged user
USER sema

# Expose port for Jupyter (optional, for debugging)
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.10 -c "import torch; import transformers; import konlpy" || exit 1

# Default command: run the simple runner
CMD ["python3.10", "run_simple.py"]
