# Multi-platform Dockerfile with CUDA support when available
# Works on Linux (with/without NVIDIA), Windows Docker Desktop, and macOS
ARG ENABLE_GPU=auto
ARG BASE_IMAGE=ubuntu:22.04

# Use CUDA image when GPU is explicitly enabled, otherwise use Ubuntu
FROM ${BASE_IMAGE}

# Build arguments for GPU optimization
ARG ENABLE_GPU=auto

# Set environment variables for performance optimization
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set CUDA environment variables only if we might have CUDA
ENV CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 1) Install base system dependencies including TTS requirements
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      # Build essentials
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
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
      # Text-to-speech system dependencies
      espeak-ng \
      espeak-ng-data \
      # Neural network acceleration libraries
      libblas-dev \
      liblapack-dev \
      libopenblas-dev \
      # Python and networking
      python3.10 \
      python3.10-dev \
      python3-pip \
      python3-venv \
      curl \
      git \
      wget \
      # Database
      libpq-dev \
      # Rust for some dependencies (required for deepfilternet)
      rustc \
      cargo \
      # Additional optimization libraries
      libomp-dev \
      libgomp1 \
      # System utilities for debugging
      lsof \
      htop \
      vim \
      # Cross-platform compatibility tools
      dos2unix \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# 2) Create python3 symlink
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.10 /usr/bin/python

# 3) Upgrade pip and install foundational packages
RUN python3 -m pip install --upgrade pip setuptools wheel

    # 4) Install PyTorch - with CUDA support when available, CPU otherwise
    RUN if [ -f /usr/local/cuda/bin/nvcc ] || [ "$ENABLE_GPU" = "true" ]; then \
            echo "Installing PyTorch with CUDA support..." && \
            pip3 install --no-cache-dir \
                torch==2.1.2+cu121 \
                torchaudio==2.1.2+cu121 \
                --index-url https://download.pytorch.org/whl/cu121; \
        else \
            echo "Installing PyTorch CPU-only version..." && \
            pip3 install --no-cache-dir \
                torch==2.1.2 \
                torchaudio==2.1.2 \
                --index-url https://download.pytorch.org/whl/cpu; \
        fi

# 5) Install core dependencies in order
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.11.4 \
    cython

# 6) Install audio processing dependencies
RUN pip3 install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    resampy==0.4.2

# 7) Install AI/ML dependencies
RUN pip3 install --no-cache-dir \
    transformers==4.35.2 \
    accelerate \
    optimum \
    huggingface-hub \
    sentence-transformers

# 8) Create working directory and copy requirements
WORKDIR /app
COPY requirements.txt .

# 9) Install requirements.txt dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# 10) Install TTS-specific dependencies with error handling for Docker environment
RUN pip3 install --no-cache-dir openai-whisper || echo "Warning: openai-whisper installation failed" \
 && pip3 install --no-cache-dir faster-whisper || echo "Warning: faster-whisper installation failed" \
 && pip3 install --no-cache-dir resemblyzer || echo "Warning: resemblyzer installation failed" \
 && pip3 install --no-cache-dir deepfilternet || echo "Warning: deepfilternet installation failed" \
 && pip3 install --no-cache-dir TTS || echo "Warning: TTS installation failed" \
 && pip3 install --no-cache-dir f5-tts || echo "Warning: F5-TTS installation failed"

# 11) Install spaCy model for NLP processing
RUN python3 -m spacy download en_core_web_sm || echo "Warning: spaCy model download failed"

# 12) Copy application code
COPY . .

# 13) Create directories for persistent volumes
RUN mkdir -p /app/chroma_db /app/storage /app/outputs /app/cache \
 && chmod 755 /app/chroma_db /app/storage /app/outputs /app/cache

# 14) Create cache directories for models
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/cache/whisper \
 && chmod 755 /app/cache/huggingface /app/cache/torch /app/cache/whisper

# 15) Set environment variables for model caching
ENV HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    WHISPER_CACHE=/app/cache/whisper \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    HF_DATASETS_CACHE=/app/cache/huggingface

# 16) Expose port
EXPOSE 5000

# 17) Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 18) Create startup script with multi-platform support
RUN echo '#!/bin/bash' > /app/start.sh \
 && echo 'echo "🚀 Starting Titans API (Cross-Platform Docker)..."' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo '# Detect platform and environment' >> /app/start.sh \
 && echo 'if [ -f /proc/version ]; then' >> /app/start.sh \
 && echo '  if grep -qi microsoft /proc/version; then' >> /app/start.sh \
 && echo '    echo "🪟 Running on Windows Docker Desktop (WSL2)"' >> /app/start.sh \
 && echo '  else' >> /app/start.sh \
 && echo '    echo "🐧 Running on Linux Docker"' >> /app/start.sh \
 && echo '  fi' >> /app/start.sh \
 && echo 'elif uname -a | grep -qi darwin; then' >> /app/start.sh \
 && echo '  echo "🍎 Running on macOS Docker Desktop"' >> /app/start.sh \
 && echo 'else' >> /app/start.sh \
 && echo '  echo "🐋 Running in Docker container"' >> /app/start.sh \
 && echo 'fi' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo '# Verify critical dependencies' >> /app/start.sh \
 && echo 'echo "🔍 Verifying dependencies..."' >> /app/start.sh \
 && echo 'python3 -c "import torch; print(f\"✓ PyTorch: {torch.__version__}\"); print(f\"✓ CUDA available: {torch.cuda.is_available()}\")" || echo "❌ PyTorch/CUDA issue"' >> /app/start.sh \
 && echo 'python3 -c "import numpy; print(f\"✓ NumPy: {numpy.__version__}\")" || echo "❌ NumPy import failed"' >> /app/start.sh \
 && echo 'python3 -c "import librosa; print(f\"✓ Librosa: {librosa.__version__}\")" || echo "❌ Librosa import failed"' >> /app/start.sh \
 && echo 'python3 -c "import soundfile; print(f\"✓ SoundFile: {soundfile.__version__}\")" || echo "❌ SoundFile import failed"' >> /app/start.sh \
 && echo 'command -v espeak-ng >/dev/null 2>&1 && echo "✓ espeak-ng available" || echo "❌ espeak-ng not found"' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo '# GPU Detection (works on Linux, Windows, and macOS)' >> /app/start.sh \
 && echo 'echo "🔍 Checking GPU availability..."' >> /app/start.sh \
 && echo 'if command -v nvidia-smi >/dev/null 2>&1; then' >> /app/start.sh \
 && echo '  echo "✓ nvidia-smi found"' >> /app/start.sh \
 && echo '  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || echo "⚠️  nvidia-smi query failed"' >> /app/start.sh \
 && echo '  python3 -c "import torch; print(f\"CUDA devices: {torch.cuda.device_count()}\")" 2>/dev/null || echo "⚠️  PyTorch CUDA detection failed"' >> /app/start.sh \
 && echo 'else' >> /app/start.sh \
 && echo '  echo "ℹ️  nvidia-smi not found - running in CPU mode"' >> /app/start.sh \
 && echo '  echo "ℹ️  This is normal on macOS and systems without NVIDIA GPUs"' >> /app/start.sh \
 && echo 'fi' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo '# Platform-specific information' >> /app/start.sh \
 && echo 'if grep -qi microsoft /proc/version 2>/dev/null; then' >> /app/start.sh \
 && echo '  echo "ℹ️  Windows Docker Desktop detected - ensure WSL2 integration is enabled"' >> /app/start.sh \
 && echo '  echo "ℹ️  For GPU support on Windows, ensure NVIDIA Container Toolkit is installed"' >> /app/start.sh \
 && echo 'elif uname -a | grep -qi darwin 2>/dev/null; then' >> /app/start.sh \
 && echo '  echo "ℹ️  macOS Docker Desktop detected - GPU acceleration not available in Docker"' >> /app/start.sh \
 && echo '  echo "ℹ️  For best performance on macOS, consider running natively outside Docker"' >> /app/start.sh \
 && echo 'fi' >> /app/start.sh \
 && echo '' >> /app/start.sh \
 && echo 'echo "🎬 Starting application..."' >> /app/start.sh \
 && echo 'exec python3 app.py' >> /app/start.sh \
 && chmod +x /app/start.sh \
 && dos2unix /app/start.sh

# 19) Use startup script
CMD ["/app/start.sh"]
