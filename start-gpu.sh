#!/bin/bash

# Titans of Energy - GPU Docker Startup Script
# This script helps diagnose GPU setup and start the application with NVIDIA GPU support

echo "🚀 Titans of Energy - GPU Setup and Startup"
echo "=============================================="

# Check if NVIDIA Docker is available
echo "🔍 Checking NVIDIA Docker support..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed or not in PATH"
    exit 1
fi

# Test NVIDIA Docker runtime
echo "🔧 Testing NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Docker runtime is working correctly"
    DOCKER_METHOD="modern"
elif docker run --rm --runtime=nvidia nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Docker runtime is working with legacy syntax"
    DOCKER_METHOD="legacy"
else
    echo "❌ NVIDIA Docker runtime is not working"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "1. Install NVIDIA Container Toolkit:"
    echo "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y nvidia-container-toolkit"
    echo ""
    echo "2. Configure Docker daemon:"
    echo "   sudo nvidia-ctk runtime configure --runtime=docker"
    echo "   sudo systemctl restart docker"
    echo ""
    echo "3. Verify GPU access:"
    echo "   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi"
    echo ""
    exit 1
fi

# Show GPU information
echo ""
echo "💡 GPU Information:"
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "🚀 Starting Titans of Energy with GPU support..."

# Choose the appropriate docker-compose command
if [ "$DOCKER_METHOD" = "legacy" ]; then
    echo "🔧 Using legacy GPU runtime syntax..."
    exec docker-compose -f docker-compose.yaml -f docker-compose.gpu-legacy.yaml up --build "$@"
else
    echo "🔧 Using modern GPU runtime syntax..."
    exec docker-compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build "$@"
fi 