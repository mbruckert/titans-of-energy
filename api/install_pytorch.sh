#!/bin/bash
# PyTorch installation script for multi-platform support
# Usage: install_pytorch.sh <TARGETPLATFORM> <ENABLE_GPU>

TARGETPLATFORM=$1
ENABLE_GPU=$2

echo "Installing PyTorch for platform: $TARGETPLATFORM, GPU: $ENABLE_GPU"

if [ "$TARGETPLATFORM" = "linux/arm64" ]; then
    echo "Installing PyTorch for Apple Silicon (ARM64) with MPS support..."
    # For Apple Silicon / ARM64 - install standard version with MPS support
    pip3 install --no-cache-dir torch torchaudio
    echo "PyTorch installed for ARM64 with MPS support"
    
elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then
    if [ "$ENABLE_GPU" = "false" ]; then
        echo "Installing CPU-only PyTorch for x86_64..."
        pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        echo "CPU-only PyTorch installed for x86_64"
    else
        echo "Installing CUDA-enabled PyTorch for x86_64..."
        # For x86_64 with NVIDIA GPU support
        pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
        echo "CUDA-enabled PyTorch installed for x86_64"
    fi
else
    echo "Unknown platform: $TARGETPLATFORM, installing CPU-only PyTorch..."
    pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "CPU-only PyTorch installed (fallback)"
fi

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else \"N/A\"}')" 