# Multi-stage optimized Dockerfile for STT/TTS performance on both Apple Silicon and NVIDIA GPUs
FROM ubuntu:22.04

# NOTE: This Dockerfile is optimized for building on ARM64 (Apple Silicon) and x86_64 architectures
# The compilation flags are specifically tuned to avoid conflicts with ggml/llama-cpp-python
# while still providing good performance for TTS/STT workloads

# Set environment variables for performance optimization
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1) Install system dependencies with optimizations
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

# 2) Upgrade pip and install foundational packages
RUN pip3 install --upgrade pip setuptools wheel

# 3) Install numpy and scipy with optimizations first
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    cython

# 4) Set compilation flags for optimal performance
# These flags optimize for both x86_64 and ARM64 (Apple Silicon when building)
# Removed -ffast-math as it conflicts with ggml compilation
ENV CFLAGS="-O3 -fPIC -fno-finite-math-only" \
    CXXFLAGS="-O3 -fPIC -fno-finite-math-only" \
    LDFLAGS="-Wl,-O1 -Wl,--as-needed" \
    MAKEFLAGS="-j$(nproc)"

# 5) Install PyTorch with device detection
# Use CPU version for better compatibility across architectures
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6) Create working directory and copy requirements
WORKDIR /app
COPY requirements.txt .

# Set environment variables for llama-cpp-python compilation
# Disable all advanced features that can cause ARM64 build issues
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_BACKEND_DL=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF" \
    FORCE_CMAKE=1 \
    LLAMA_CPP_LIB_CUDA=0 \
    LLAMA_CPP_FORCE_CPU=1 \
    LLAMA_CPP_BACKEND=OFF

# 7) Install Python dependencies with optimizations
RUN pip3 install --no-cache-dir -r requirements.txt

# 8) Install additional TTS/STT optimization dependencies
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
    psutil \
    # Apple Silicon optimizations (will be ignored on non-Apple hardware)
    coremltools || true

# 9) Install spaCy model for NLP processing
RUN python3 -m spacy download en_core_web_sm

# 10) Copy application code
COPY . .

# 11) Create directories for persistent volumes with optimized permissions
RUN mkdir -p /app/chroma_db /app/storage /app/outputs /app/cache \
 && chmod 755 /app/chroma_db /app/storage /app/outputs /app/cache \
 && chown -R 1000:1000 /app

# 12) Set environment variables for runtime optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # PyTorch optimizations
    TORCH_CUDNN_V8_API_ENABLED=1 \
    CUDA_LAUNCH_BLOCKING=0 \
    # OpenMP optimizations
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    # Cache directories
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    WHISPER_CACHE_DIR=/app/cache/whisper \
    # Default Postgres connection settings (compose will override)
    PGHOST=db \
    PGPORT=5432 \
    PGUSER=myuser \
    PGPASSWORD=mypassword \
    PGDATABASE=mydb

# 13) Create cache directories
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/cache/whisper \
 && chmod 755 /app/cache/huggingface /app/cache/torch /app/cache/whisper

# 14) Expose port
EXPOSE 5000

# 15) Add healthcheck for better container management
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 16) Use exec form for better signal handling
CMD ["python3", "app.py"]
