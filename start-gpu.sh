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

# Check Docker permissions
echo "🔧 Checking Docker permissions..."
if ! docker info &> /dev/null; then
    if sudo docker info &> /dev/null; then
        echo "⚠️  Docker requires sudo. Consider adding your user to the docker group:"
        echo "   sudo usermod -aG docker $USER"
        echo "   newgrp docker  # or logout and login again"
        echo ""
        DOCKER_CMD="sudo docker"
        COMPOSE_CMD="sudo docker-compose"
    else
        echo "❌ Docker is not running or accessible"
        exit 1
    fi
else
    DOCKER_CMD="docker"
    COMPOSE_CMD="docker-compose"
fi

# Detect OS for package manager specific instructions
if [ -f /etc/arch-release ]; then
    OS_TYPE="arch"
elif [ -f /etc/debian_version ]; then
    OS_TYPE="debian"
elif [ -f /etc/redhat-release ]; then
    OS_TYPE="redhat"
else
    OS_TYPE="unknown"
fi

# Test NVIDIA Docker runtime with correct image tags
echo "🔧 Testing NVIDIA Docker runtime..."

# Try different CUDA image variants that actually exist
CUDA_IMAGES=(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04"
    "nvidia/cuda:12.1.0-devel-ubuntu22.04"
    "nvidia/cuda:12.1-devel-ubuntu22.04"
    "nvidia/cuda:12.0.1-devel-ubuntu22.04"
    "nvidia/cuda:11.8-devel-ubuntu22.04"
)

WORKING_IMAGE=""
DOCKER_METHOD=""

for image in "${CUDA_IMAGES[@]}"; do
    echo "   Testing image: $image"
    if $DOCKER_CMD run --rm --gpus all $image nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA Docker runtime is working correctly with $image"
        DOCKER_METHOD="modern"
        WORKING_IMAGE=$image
        break
    elif $DOCKER_CMD run --rm --runtime=nvidia $image nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA Docker runtime is working with legacy syntax using $image"
        DOCKER_METHOD="legacy"
        WORKING_IMAGE=$image
        break
    fi
done

if [ -z "$WORKING_IMAGE" ]; then
    echo "❌ NVIDIA Docker runtime is not working with any CUDA image"
    echo ""
    echo "🔧 Troubleshooting steps for $OS_TYPE:"
    
    if [ "$OS_TYPE" = "arch" ]; then
        echo "1. Install NVIDIA Container Toolkit (Arch Linux):"
        echo "   yay -S nvidia-container-toolkit"
        echo "   # OR using pacman (if available in extra repos):"
        echo "   sudo pacman -S nvidia-container-toolkit"
        echo ""
        echo "2. Alternative - Install from AUR:"
        echo "   git clone https://aur.archlinux.org/nvidia-container-toolkit.git"
        echo "   cd nvidia-container-toolkit"
        echo "   makepkg -si"
        echo ""
    elif [ "$OS_TYPE" = "debian" ]; then
        echo "1. Install NVIDIA Container Toolkit (Debian/Ubuntu):"
        echo "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "   sudo apt-get update"
        echo "   sudo apt-get install -y nvidia-container-toolkit"
        echo ""
    else
        echo "1. Install NVIDIA Container Toolkit (Generic):"
        echo "   Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
    fi
    
    echo "2. Configure Docker daemon:"
    echo "   sudo nvidia-ctk runtime configure --runtime=docker"
    echo "   sudo systemctl restart docker"
    echo ""
    echo "3. Add user to docker group (optional, to avoid sudo):"
    echo "   sudo usermod -aG docker $USER"
    echo "   newgrp docker"
    echo ""
    echo "4. Verify GPU access with available images:"
    for image in "${CUDA_IMAGES[@]}"; do
        echo "   $DOCKER_CMD run --rm --gpus all $image nvidia-smi"
    done
    echo ""
    echo "5. Check NVIDIA driver installation:"
    echo "   nvidia-smi"
    echo ""
    exit 1
fi

# Show GPU information using the working image
echo ""
echo "💡 GPU Information:"
$DOCKER_CMD run --rm --gpus all $WORKING_IMAGE nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || {
    echo "⚠️  Could not get detailed GPU info, but basic CUDA test passed"
    $DOCKER_CMD run --rm --gpus all $WORKING_IMAGE nvidia-smi 2>/dev/null || echo "GPU info unavailable"
}

echo ""
echo "🚀 Starting Titans of Energy with GPU support..."

# Choose the appropriate docker-compose command
if [ "$DOCKER_METHOD" = "legacy" ]; then
    echo "🔧 Using legacy GPU runtime syntax..."
    exec $COMPOSE_CMD -f docker-compose.yaml -f docker-compose.gpu-legacy.yaml up --build "$@"
else
    echo "🔧 Using modern GPU runtime syntax..."
    exec $COMPOSE_CMD -f docker-compose.yaml -f docker-compose.gpu.yaml up --build "$@"
fi 