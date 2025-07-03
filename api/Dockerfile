# Multi-platform optimized Dockerfile for Apple Silicon, NVIDIA GPUs, and CPU
# Automatically detects build platform and optimizes accordingly
FROM ubuntu:22.04

# Build arguments for platform detection and optimization
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG ENABLE_GPU=auto

# Set environment variables for performance optimization
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1) Install base system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      # Build essentials
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
      # Platform detection tools
      lsb-release \
      # Audio and video processing
      ffmpeg \
      libavcodec-dev \
      libavformat-dev \
      libavutil-dev \
      libswscale-dev \
      libswresample-dev \
      # Audio libraries for TTS
      libsndfile1-dev \
      libportaudio2 \
      portaudio19-dev \
      libasound2-dev \
      # Neural network acceleration libraries
      libblas-dev \
      liblapack-dev \
      libopenblas-dev \
      # Python and networking
      python3-pip \
      python3-dev \
      python3-venv \
      curl \
      git \
      wget \
      # Database
      libpq-dev \
      # Rust for some dependencies
      rustc \
      cargo \
      # Additional optimization libraries
      libomp-dev \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# 2) Platform-specific setup
RUN echo "Build platform: $BUILDPLATFORM" \
 && echo "Target platform: $TARGETPLATFORM" \
 && echo "GPU setting: $ENABLE_GPU"

# 2.1) Install NVIDIA CUDA toolkit for x86_64 platforms when GPU is enabled
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ "$ENABLE_GPU" != "false" ]; then \
      echo "Installing CUDA toolkit for x86_64/NVIDIA GPU support..." \
      && apt-get update \
      && apt-get install -y --no-install-recommends \
           gnupg \
           software-properties-common \
      && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
      && dpkg -i cuda-keyring_1.0-1_all.deb \
      && apt-get update \
      && apt-get install -y --no-install-recommends \
           cuda-toolkit-12-1 \
           nvidia-cuda-toolkit \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean \
      && rm cuda-keyring_1.0-1_all.deb \
      && echo "CUDA toolkit installed for x86_64" \
      && echo "Verifying CUDA installation..." \
      && nvcc --version || echo "Warning: CUDA compiler not found in PATH"; \
    else \
      echo "Skipping CUDA installation (not x86_64 or GPU disabled)"; \
    fi

# 2.2) Install Apple Silicon optimizations for ARM64 platforms
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      echo "Setting up ARM64/Apple Silicon optimizations..." \
      && apt-get update \
      && apt-get install -y --no-install-recommends \
           # ARM64 specific optimizations
           libblas64-dev \
           liblapack64-dev \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean \
      && echo "ARM64 optimizations installed"; \
    else \
      echo "Skipping ARM64 optimizations (not ARM64 platform)"; \
    fi

# 3) Upgrade pip and install foundational packages
RUN pip3 install --upgrade pip setuptools wheel

# 4) Install numpy and scipy with platform optimizations
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    cython

# 5) Set platform-specific compilation flags
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      # Apple Silicon / ARM64 optimizations - conservative flags to avoid conflicts
      export CFLAGS="-O2 -fPIC -mcpu=native" \
      && export CXXFLAGS="-O2 -fPIC -mcpu=native" \
      && export LDFLAGS="-Wl,-O1 -Wl,--as-needed" \
      && echo "Set ARM64 compilation flags"; \
    else \
      # x86_64 optimizations
      export CFLAGS="-O3 -fPIC -march=native -fno-finite-math-only" \
      && export CXXFLAGS="-O3 -fPIC -march=native -fno-finite-math-only" \
      && export LDFLAGS="-Wl,-O1 -Wl,--as-needed" \
      && echo "Set x86_64 compilation flags"; \
    fi

# Store the flags in environment for later use
ENV MAKEFLAGS="-j$(nproc)"

# 6) Install PyTorch with platform-specific optimizations
COPY install_pytorch.sh /tmp/install_pytorch.sh
RUN chmod +x /tmp/install_pytorch.sh \
 && /tmp/install_pytorch.sh "$TARGETPLATFORM" "$ENABLE_GPU" \
 && rm /tmp/install_pytorch.sh

# 7) Create working directory and copy requirements
WORKDIR /app
COPY requirements.txt .

# 8) Set platform-specific environment variables for llama-cpp-python
COPY setup_llama_env.sh /tmp/setup_llama_env.sh
RUN chmod +x /tmp/setup_llama_env.sh \
 && /tmp/setup_llama_env.sh "$TARGETPLATFORM" "$ENABLE_GPU" \
 && rm /tmp/setup_llama_env.sh

# 9) Install Python dependencies with optimizations
RUN pip3 install --no-cache-dir --ignore-installed -r requirements.txt

# 10) Install additional TTS/STT optimization dependencies
RUN pip3 install --no-cache-dir \
    # Advanced Whisper for better STT performance
    openai-whisper \
    # Faster whisper implementation
    faster-whisper \
    # Audio processing optimizations
    librosa \
    soundfile \
    resampy \
    # Deep learning optimizations
    accelerate \
    optimum \
    # Memory optimization
    psutil

# 11) Install platform-specific optimizations
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      # Apple Silicon specific packages
      pip3 install --no-cache-dir coremltools || echo "CoreML not available, skipping"; \
    else \
      # x86_64 specific packages
      echo "x86_64 platform - no additional packages needed"; \
    fi

# 12) Install spaCy model for NLP processing
RUN python3 -m spacy download en_core_web_sm

# 13) Copy application code
COPY . .

# 14) Create directories for persistent volumes with optimized permissions
RUN mkdir -p /app/chroma_db /app/storage /app/outputs /app/cache \
 && chmod 755 /app/chroma_db /app/storage /app/outputs /app/cache \
 && chown -R 1000:1000 /app

# 15) Set platform-specific runtime environment variables
COPY set_runtime_env.sh /tmp/set_runtime_env.sh
RUN chmod +x /tmp/set_runtime_env.sh \
 && /tmp/set_runtime_env.sh "$TARGETPLATFORM" "$ENABLE_GPU" > /app/platform_env.sh \
 && rm /tmp/set_runtime_env.sh

# 16) Create cache directories
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/cache/whisper \
 && chmod 755 /app/cache/huggingface /app/cache/torch /app/cache/whisper

# 17) Expose port
EXPOSE 5000

# 18) Add healthcheck for better container management
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 19) Create startup script that sources platform environment
RUN echo '#!/bin/bash' > /app/start.sh \
 && echo 'echo "🚀 Starting Titans API..."' >> /app/start.sh \
 && echo 'source /app/platform_env.sh' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo '# GPU Detection and Validation' >> /app/start.sh \
 && echo 'if [ "$ENABLE_GPU" = "true" ] || [ "$ENABLE_GPU" = "auto" ]; then' >> /app/start.sh \
 && echo '  echo "🔍 Checking GPU availability..."' >> /app/start.sh \
 && echo '  if command -v nvidia-smi >/dev/null 2>&1; then' >> /app/start.sh \
 && echo '    echo "✓ nvidia-smi found"' >> /app/start.sh \
 && echo '    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || echo "⚠️  nvidia-smi failed to query GPU"' >> /app/start.sh \
 && echo '  else' >> /app/start.sh \
 && echo '    echo "⚠️  nvidia-smi not found - GPU support may be limited"' >> /app/start.sh \
 && echo '  fi' >> /app/start.sh \
 && echo '  if [ -n "$CUDA_VISIBLE_DEVICES" ]; then' >> /app/start.sh \
 && echo '    echo "🎯 CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"' >> /app/start.sh \
 && echo '  fi' >> /app/start.sh \
 && echo 'else' >> /app/start.sh \
 && echo '  echo "💻 GPU support disabled - running in CPU mode"' >> /app/start.sh \
 && echo 'fi' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo 'echo "🎬 Starting application..."' >> /app/start.sh \
 && echo 'exec python3 app.py' >> /app/start.sh \
 && chmod +x /app/start.sh

# 20) Use startup script for better platform detection
CMD ["/app/start.sh"]
