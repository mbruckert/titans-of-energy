# Docker Setup Guide for Titans of Energy

## Overview

This guide covers running Titans of Energy in Docker with NVIDIA GPU support on:
- **Linux** with NVIDIA GPUs
- **Windows** with Docker Desktop and NVIDIA GPUs (WSL2 backend)

> **Note for macOS Users**: If you're on macOS, it's recommended to run the application natively (outside Docker) for optimal Apple Silicon/MPS support. The Docker setup is optimized for Linux and Windows with NVIDIA GPUs.

## Prerequisites

### For Linux with NVIDIA GPU:
1. **Docker Engine** 20.10.0+ 
2. **Docker Compose** 2.0.0+
3. **NVIDIA Container Toolkit** (for GPU support)
4. **NVIDIA drivers** compatible with CUDA 12.1

### For Windows with NVIDIA GPU:
1. **Docker Desktop** 4.0.0+ with WSL2 backend enabled
2. **Windows 11** or **Windows 10** version 2004+ (Build 19041+)
3. **WSL2** enabled and updated
4. **NVIDIA drivers** for Windows (version 460.82+)
5. **NVIDIA Container Toolkit** for WSL2

## Installation Instructions

### Linux Setup (Ubuntu/Debian):

#### Install NVIDIA Container Toolkit:
```bash
# Add the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Windows Setup:

#### 1. Install WSL2 and Docker Desktop:
```powershell
# Enable WSL2 (run as Administrator)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart Windows, then set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu from Microsoft Store
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
```

#### 2. Enable WSL2 Integration in Docker Desktop:
- Open Docker Desktop
- Go to Settings → General → Enable "Use the WSL 2 based engine"
- Go to Settings → Resources → WSL Integration
- Enable integration with your WSL2 distro

#### 3. Install NVIDIA Container Toolkit in WSL2:
```bash
# Open WSL2 Ubuntu terminal
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon (in WSL2)
sudo mkdir -p /etc/docker
echo '{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}' | sudo tee /etc/docker/daemon.json

# Restart Docker Desktop from Windows
```

## Setup Instructions

### 1. Clone Repository

**Linux:**
```bash
git clone <repository-url>
cd titans-of-energy
```

**Windows (in WSL2 terminal):**
```bash
# Clone to WSL2 filesystem for better performance
git clone <repository-url>
cd titans-of-energy
```

### 2. Create Environment File

Create a `.env` file in the root directory:

**Linux/Windows (same content):**
```bash
# Database Configuration
DB_HOST=db
DB_USER=titans_user
DB_PASSWORD=titans_password
DB_NAME=titans_db
DB_PORT=5432

# OpenAI API Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face API Key (optional)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# GPU Configuration for Docker
ENABLE_GPU=true
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=all

# Platform detection (auto-detected)
DOCKER_PLATFORM=auto
```

### 3. Build and Run

**For macOS or systems without NVIDIA GPU (CPU-only):**
```bash
docker-compose up --build
```

**For Linux/Windows with NVIDIA GPU support:**
```bash
docker-compose -f docker-compose.gpu.yaml up --build
```

**Windows-specific notes:**
- If running from Windows PowerShell (not WSL2):
  ```powershell
  docker-compose -f docker-compose.gpu.yaml up --build
  ```
- For CPU-only on Windows: `docker-compose up --build`

**macOS note:** The main `docker-compose.yaml` uses CPU-only PyTorch for maximum compatibility on macOS Docker Desktop.

### 4. Verify Installation

The startup script will automatically detect your platform and verify dependencies:

**Linux output:**
```
🐧 Running on Linux Docker
✓ PyTorch: 2.1.2+cu121
✓ CUDA available: True
```

**Windows output (GPU mode):**
```
🪟 Running on Windows Docker Desktop (WSL2)
✓ PyTorch: 2.1.2+cu121
✓ CUDA available: True
ℹ️  Windows Docker Desktop detected - ensure WSL2 integration is enabled
ℹ️  For GPU support on Windows, ensure NVIDIA Container Toolkit is installed
```

**macOS output (CPU mode):**
```
🍎 Running on macOS Docker Desktop
✓ PyTorch: 2.1.2+cpu
✓ CUDA available: False
ℹ️  macOS Docker Desktop detected - GPU acceleration not available in Docker
ℹ️  For best performance on macOS, consider running natively outside Docker
```

Look for these verification messages in the logs:
```
✓ NumPy: 1.24.3
✓ Librosa: 0.10.1
✓ SoundFile: 0.12.1
✓ espeak-ng available
✓ nvidia-smi found
CUDA devices: 1
```

## Troubleshooting

### Windows-Specific Issues:

#### 1. WSL2 Not Detected
```bash
# Check WSL2 version
wsl --list --verbose

# Ensure Ubuntu is version 2
wsl --set-version Ubuntu 2
```

#### 2. Docker Desktop Integration Issues
- Restart Docker Desktop
- Ensure WSL2 integration is enabled for your distro
- Try running from WSL2 terminal instead of Windows PowerShell

#### 3. GPU Not Accessible in Windows
```bash
# Test in WSL2 terminal
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

#### 4. File Permission Issues on Windows
If you encounter permission issues:
```bash
# Ensure you're working in WSL2 filesystem
cd /home/$USER/
git clone <repository-url>
cd titans-of-energy
```

### Linux-Specific Issues:

#### 1. Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

#### 2. NVIDIA Driver Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Common Issues (Both Platforms):

#### 1. Import Errors in Docker
- These are usually resolved during the Docker build process
- Check build logs for installation failures
- Try rebuilding: `docker-compose -f docker-compose.gpu.yaml up --build --force-recreate`

#### 2. CUDA Out of Memory
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Reduce model size or use CPU mode for development
ENABLE_GPU=false docker-compose up --build
```

#### 3. Port Already in Use
```bash
# Check what's using the port
# Linux:
sudo netstat -tulpn | grep :5000
# Windows:
netstat -ano | findstr :5000

# Stop conflicting services or change ports in docker-compose.yaml
```

## Performance Considerations

### Windows Performance Tips:
- Keep project files in WSL2 filesystem (`/home/$USER/`) for better performance
- Allocate sufficient resources to WSL2 in `.wslconfig`:
```ini
[wsl2]
memory=8GB
processors=4
```

### Storage Locations:
- **Linux**: Docker volumes stored in `/var/lib/docker/volumes/`
- **Windows**: Docker volumes stored in WSL2 distro

### Development Workflow:
- **Linux**: Direct file editing and Docker restart
- **Windows**: Edit files in WSL2 or use VS Code with WSL extension

## API Endpoints

Once running, the API will be available at:
- **Backend**: http://localhost:5000
- **Frontend**: http://localhost:3000

Works identically on both Linux and Windows.

## Stopping Services

**Linux/Windows:**
```bash
# Stop services
docker-compose -f docker-compose.gpu.yaml down

# Stop and remove volumes (careful - deletes data!)
docker-compose -f docker-compose.gpu.yaml down -v
```

## Platform-Specific Notes

### Linux Advantages:
- Direct native performance
- Simpler setup
- Better Docker integration

### Windows Advantages:
- Docker Desktop GUI
- Integrated with Windows development workflow
- WSL2 provides Linux compatibility

### Performance Comparison:
- **Linux**: Best performance, direct hardware access
- **Windows + WSL2**: ~95% of Linux performance, excellent compatibility
- **macOS native**: Recommended for Apple Silicon users (outside Docker)

## Support

- **Windows Docker issues**: Check WSL2 integration and NVIDIA toolkit installation
- **Linux Docker issues**: Verify NVIDIA Container Toolkit and driver compatibility
- **Application issues**: Check logs with platform detection output
- **Cross-platform development**: Use WSL2 for Windows, native terminals for Linux 