#!/bin/bash
# llama-cpp-python environment setup script for multi-platform support
# Usage: setup_llama_env.sh <TARGETPLATFORM> <ENABLE_GPU>

TARGETPLATFORM=$1
ENABLE_GPU=$2

echo "Setting up llama-cpp-python environment for platform: $TARGETPLATFORM, GPU: $ENABLE_GPU"

if [ "$TARGETPLATFORM" = "linux/arm64" ]; then
    echo "Setting up llama-cpp-python for Apple Silicon (ARM64)..."
    # Apple Silicon / ARM64 - optimize for Metal Performance Shaders when available
    export CMAKE_ARGS="-DGGML_METAL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF"
    export FORCE_CMAKE=1
    export LLAMA_CPP_LIB_CUDA=0
    export LLAMA_CPP_METAL=1
    echo "CMAKE_ARGS=\"-DGGML_METAL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF\"" >> /etc/environment
    echo "FORCE_CMAKE=1" >> /etc/environment
    echo "LLAMA_CPP_LIB_CUDA=0" >> /etc/environment
    echo "LLAMA_CPP_METAL=1" >> /etc/environment
    echo "llama-cpp-python configured for Apple Silicon with Metal support"
    
elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then
    if [ "$ENABLE_GPU" = "false" ]; then
        echo "Setting up llama-cpp-python for x86_64 CPU-only..."
        export CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_AVX=ON -DGGML_AVX2=ON"
        export FORCE_CMAKE=1
        export LLAMA_CPP_LIB_CUDA=0
        echo "CMAKE_ARGS=\"-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_AVX=ON -DGGML_AVX2=ON\"" >> /etc/environment
        echo "FORCE_CMAKE=1" >> /etc/environment
        echo "LLAMA_CPP_LIB_CUDA=0" >> /etc/environment
        echo "llama-cpp-python configured for x86_64 CPU-only"
    else
        echo "Setting up llama-cpp-python for x86_64 with CUDA..."
        export CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF"
        export FORCE_CMAKE=1
        export LLAMA_CPP_LIB_CUDA=1
        export CUDA_DOCKER_ARCH=all
        echo "CMAKE_ARGS=\"-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF\"" >> /etc/environment
        echo "FORCE_CMAKE=1" >> /etc/environment
        echo "LLAMA_CPP_LIB_CUDA=1" >> /etc/environment
        echo "CUDA_DOCKER_ARCH=all" >> /etc/environment
        echo "llama-cpp-python configured for x86_64 with CUDA support"
    fi
else
    echo "Unknown platform: $TARGETPLATFORM, using conservative CPU settings..."
    export CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF"
    export FORCE_CMAKE=1
    export LLAMA_CPP_LIB_CUDA=0
    echo "CMAKE_ARGS=\"-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF\"" >> /etc/environment
    echo "FORCE_CMAKE=1" >> /etc/environment
    echo "LLAMA_CPP_LIB_CUDA=0" >> /etc/environment
    echo "llama-cpp-python configured for unknown platform (conservative settings)"
fi

echo "llama-cpp-python environment setup complete" 