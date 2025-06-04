# Use Python 3.10 slim image as base
FROM python:3.10.16-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    portaudio19-dev \
    espeak-ng \
    libpq-dev \
    ffmpeg \
    git \
    curl \
    pkg-config \
    libssl-dev \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for deepfilternet)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first for better caching
COPY requirements.txt .

# Create a modified requirements.txt for ARM64 compatibility
RUN if [ "$(uname -m)" = "aarch64" ]; then \
    # For ARM64, we'll skip llama-cpp-python initially and install it separately with specific flags
    grep -v "llama-cpp-python" requirements.txt > requirements_no_llama.txt; \
    else \
    cp requirements.txt requirements_no_llama.txt; \
    fi

# Upgrade pip and install core dependencies first
RUN pip install --upgrade pip

# Install remaining Python dependencies (excluding llama-cpp-python for ARM64)
RUN pip install -r requirements_no_llama.txt

# Install llama-cpp-python with ARM64-specific configuration if needed
RUN if [ "$(uname -m)" = "aarch64" ]; then \
    # For ARM64, install with specific CMAKE flags to avoid NEON issues
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_HBM=OFF -DGGML_CPU_ALL_VARIANTS=OFF" \
    pip install llama-cpp-python==0.3.9 --no-cache-dir --force-reinstall; \
    else \
    # For x86_64, install normally
    pip install llama-cpp-python==0.3.9; \
    fi

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p storage/generated_audio models chroma_db

# Create a non-root user for security
RUN useradd -m -u 1000 titans && chown -R titans:titans /app
USER titans

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Command to run the application
CMD ["python", "app.py"]
