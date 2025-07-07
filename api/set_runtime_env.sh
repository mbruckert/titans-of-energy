#!/bin/bash
# Runtime environment setup script for multi-platform support
# Usage: set_runtime_env.sh <TARGETPLATFORM> <ENABLE_GPU>
# Outputs environment variables to stdout

TARGETPLATFORM=$1
ENABLE_GPU=$2

echo "# Platform-specific runtime environment variables"
echo "# Generated for platform: $TARGETPLATFORM, GPU: $ENABLE_GPU"
echo ""

# Base environment variables for all platforms
cat << 'EOF'
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HOME=/app/cache/huggingface
export TORCH_HOME=/app/cache/torch
export WHISPER_CACHE_DIR=/app/cache/whisper
export PGHOST=db
export PGPORT=5432
export PGUSER=myuser
export PGPASSWORD=mypassword
export PGDATABASE=mydb
EOF

if [ "$TARGETPLATFORM" = "linux/arm64" ]; then
    echo ""
    echo "# Apple Silicon (ARM64) optimizations"
    cat << 'EOF'
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export ACCELERATE_USE_MPS_DEVICE=1
# Optimize thread count for Apple Silicon
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Enable Metal Performance Shaders where possible
export PYTORCH_MPS_ENABLED=1
EOF

elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then
    if [ "$ENABLE_GPU" = "false" ]; then
        echo ""
        echo "# x86_64 CPU-only optimizations"
        cat << 'EOF'
# CPU-specific optimizations
export TORCH_CUDNN_V8_API_ENABLED=0
export CUDA_VISIBLE_DEVICES=""
# Optimize for CPU workloads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
EOF
    else
        echo ""
        echo "# x86_64 NVIDIA GPU optimizations"
        cat << 'EOF'
# NVIDIA GPU optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_ALLOW_TF32_MATMUL_OVERRIDE=1
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Enable CUDNN optimizations
export TORCH_CUDNN_BENCHMARK=1
export TORCH_CUDNN_DETERMINISTIC=0
EOF
    fi
else
    echo ""
    echo "# Unknown platform - conservative settings"
    cat << 'EOF'
# Conservative settings for unknown platform
export TORCH_CUDNN_V8_API_ENABLED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
EOF
fi

echo ""
echo "# End of platform-specific environment variables" 