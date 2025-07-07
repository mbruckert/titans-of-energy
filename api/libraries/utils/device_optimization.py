"""
Device optimization utilities for GPU and hardware detection.

This module provides hardware detection and optimization configurations
for different devices including NVIDIA GPUs, Apple Silicon, and CPU-only systems.
"""

import os
import platform
import subprocess
import multiprocessing
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class DeviceType(Enum):
    """Supported device types."""
    NVIDIA_GPU = "nvidia_gpu"
    APPLE_SILICON = "apple_silicon"
    CPU_ONLY = "cpu_only"
    AMD_GPU = "amd_gpu"


class GPUInfo:
    """GPU information container."""

    def __init__(self, name: str = "", memory_gb: float = 0.0, compute_capability: str = "", driver_version: str = ""):
        self.name = name
        self.memory_gb = memory_gb
        self.compute_capability = compute_capability
        self.driver_version = driver_version
        self.is_available = memory_gb > 0


def detect_device() -> Tuple[DeviceType, Dict[str, Any]]:
    """
    Detect the best available device and return optimization parameters.

    Returns:
        Tuple of (device_type, device_info)
    """
    # Check for Apple Silicon first
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return DeviceType.APPLE_SILICON, _get_apple_silicon_info()

    # Check for NVIDIA GPU
    nvidia_info = _detect_nvidia_gpu()
    if nvidia_info.is_available:
        return DeviceType.NVIDIA_GPU, _get_nvidia_info(nvidia_info)

    # Check for AMD GPU (basic detection)
    if _detect_amd_gpu():
        return DeviceType.AMD_GPU, _get_amd_info()

    # Fallback to CPU
    return DeviceType.CPU_ONLY, _get_cpu_info()


def _detect_nvidia_gpu() -> GPUInfo:
    """Detect NVIDIA GPU using nvidia-smi."""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if lines:
                # Parse first GPU info
                parts = [p.strip() for p in lines[0].split(',')]
                if len(parts) >= 3:
                    name = parts[0]
                    memory_mb = float(parts[1])
                    memory_gb = memory_mb / 1024
                    driver_version = parts[2]

                    # Get compute capability if possible
                    compute_capability = _get_compute_capability(name)

                    return GPUInfo(name, memory_gb, compute_capability, driver_version)

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    # Try PyTorch CUDA detection as fallback
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                name = torch.cuda.get_device_name(0)
                # Get memory in GB
                memory_bytes = torch.cuda.get_device_properties(0).total_memory
                memory_gb = memory_bytes / (1024**3)

                compute_capability = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"

                return GPUInfo(name, memory_gb, compute_capability, "")
    except ImportError:
        pass

    return GPUInfo()


def _get_compute_capability(gpu_name: str) -> str:
    """Get compute capability based on GPU name."""
    gpu_name_lower = gpu_name.lower()

    # NVIDIA L4 and other modern GPUs
    if "l4" in gpu_name_lower:
        return "8.9"
    elif "a100" in gpu_name_lower:
        return "8.0"
    elif "v100" in gpu_name_lower:
        return "7.0"
    elif "rtx 40" in gpu_name_lower or "4090" in gpu_name_lower or "4080" in gpu_name_lower:
        return "8.9"
    elif "rtx 30" in gpu_name_lower or "3090" in gpu_name_lower or "3080" in gpu_name_lower:
        return "8.6"
    elif "rtx 20" in gpu_name_lower or "2080" in gpu_name_lower or "2070" in gpu_name_lower:
        return "7.5"
    elif "gtx 16" in gpu_name_lower or "1660" in gpu_name_lower:
        return "7.5"
    elif "gtx 10" in gpu_name_lower or "1080" in gpu_name_lower or "1070" in gpu_name_lower:
        return "6.1"

    return "unknown"


def _detect_amd_gpu() -> bool:
    """Basic AMD GPU detection via ROCm CLI or PyTorch."""
    # 1) Try ROCm CLI
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2) Fall back to PyTorch ROCm backend detection
    try:
        import torch
        # torch.cuda.is_available() returns True under ROCm
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_nvidia_info(gpu_info: GPUInfo) -> Dict[str, Any]:
    """Get NVIDIA GPU optimization parameters."""
    cpu_count = multiprocessing.cpu_count()

    # Determine if this is a high-end GPU (like L4)
    is_high_end = (
        gpu_info.memory_gb >= 20 or  # L4 has 24GB
        "l4" in gpu_info.name.lower() or
        "a100" in gpu_info.name.lower() or
        "4090" in gpu_info.name.lower()
    )

    return {
        "device_name": gpu_info.name,
        "memory_gb": gpu_info.memory_gb,
        "compute_capability": gpu_info.compute_capability,
        "driver_version": gpu_info.driver_version,
        "is_high_end": is_high_end,
        "cuda_device": "0",
        "torch_device": "cuda:0",
        "optimization_level": "high" if is_high_end else "medium",

        # TTS optimizations
        "tts_batch_size": 8 if is_high_end else 4,
        "tts_use_gpu": True,
        "tts_precision": "float16",
        "tts_compile": is_high_end,  # Use torch.compile for high-end GPUs

        # LLM optimizations
        "llm_gpu_layers": -1,  # Use all GPU layers
        "llm_batch_size": 2048 if is_high_end else 1024,
        "llm_context_length": 8192 if is_high_end else 4096,
        "llm_threads": min(cpu_count, 8),
        "llm_use_flash_attention": True,
        "llm_use_tensor_cores": True,

        # STT optimizations
        "stt_device": "cuda",
        "stt_batch_size": 16 if is_high_end else 8,
        "stt_precision": "float16",

        # Audio processing optimizations
        "audio_processing_device": "cuda",
        "audio_parallel_workers": 4 if is_high_end else 2,

        # Memory optimizations
        "enable_memory_efficient_attention": True,
        "gradient_checkpointing": not is_high_end,  # Disable for high-end GPUs
        "mixed_precision": True,
    }


def _get_apple_silicon_info() -> Dict[str, Any]:
    """Get Apple Silicon optimization parameters."""
    cpu_count = multiprocessing.cpu_count()

    # Detect specific Apple Silicon chip
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5
        )
        cpu_brand = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"
    except:
        cpu_brand = "Apple Silicon"

    # Determine performance tier - M4 Max/Ultra are the most powerful Apple Silicon chips to date
    is_high_end = any(chip in cpu_brand.lower() for chip in [
                      "m1 max", "m1 ultra", "m2 max", "m2 ultra", "m3 max", "m3 ultra", "m4 max", "m4 ultra"])
    is_pro = any(chip in cpu_brand.lower()
                 for chip in ["m1 pro", "m2 pro", "m3 pro", "m4 pro"]) or is_high_end

    return {
        "device_name": cpu_brand,
        "cpu_count": cpu_count,
        "is_high_end": is_high_end,
        "is_pro": is_pro,
        "torch_device": "mps" if _check_mps_available() else "cpu",
        "optimization_level": "high" if is_high_end else ("medium" if is_pro else "basic"),

        # TTS optimizations - MAXIMUM SPEED for M4 Max real-time inference
        "tts_batch_size": 16 if ("m4 max" in cpu_brand.lower()) else (8 if is_high_end else 2),  # DOUBLE the batch size
        "tts_use_gpu": _check_mps_available(),
        "tts_precision": "float32",  # MPS works better with float32
        "tts_compile": True if ("m4 max" in cpu_brand.lower()) else False,  # Enable torch.compile for M4 Max
        "tts_use_mps": _check_mps_available(),  # Explicit MPS flag for TTS
        "tts_mps_fallback": True,  # Enable MPS fallback to CPU
        "tts_streaming": True,  # Enable streaming inference for real-time
        "tts_chunk_size": 256,  # SMALLER chunks for faster response
        "tts_fast_preprocessing": True,  # Skip heavy audio preprocessing for speed
        "tts_model_warming": True,  # Keep models warm for instant inference
        "tts_aggressive_caching": True,  # Aggressive response caching
        "tts_skip_validation": True,  # Skip all audio validation for speed
        "tts_parallel_processing": True,  # Enable parallel processing
        "tts_memory_preallocation": True,  # Pre-allocate memory buffers

        # LLM optimizations - MAXIMUM M4 Max performance
        "llm_gpu_layers": 0,  # Use CPU for GGUF on Apple Silicon
        "llm_batch_size": 4096 if ("m4 max" in cpu_brand.lower()) else (2048 if is_high_end else 1024),  # LARGER batches
        "llm_context_length": 16384 if ("m4 max" in cpu_brand.lower()) else (8192 if is_high_end else 4096),  # LARGER context
        # M4 Max with 16 cores - USE ALL OF THEM!
        "llm_threads": min(cpu_count, 14 if ("m4 max" in cpu_brand.lower()) else 6),  # AGGRESSIVE threading
        "llm_use_metal": True,  # Use Metal for llama.cpp
        "llm_use_mlock": False,  # Disable mlock on macOS

        # STT optimizations - M4 Max has 40-core GPU for better parallel processing
        "stt_device": "mps" if _check_mps_available() else "cpu",
        "stt_batch_size": 12 if ("m4 max" in cpu_brand.lower()) else (8 if is_high_end else 4),
        "stt_precision": "float32",

        # Audio processing optimizations
        # CPU is often faster for audio processing on Apple Silicon
        "audio_processing_device": "cpu",
        "audio_parallel_workers": 2,

        # Memory optimizations
        "enable_memory_efficient_attention": True,
        "gradient_checkpointing": True,
        "mixed_precision": False,  # MPS has issues with mixed precision
    }


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU-only optimization parameters."""
    cpu_count = multiprocessing.cpu_count()

    # Detect CPU information
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpu_info = f.read()
                cpu_name = "Unknown CPU"
                for line in cpu_info.split('\n'):
                    if line.startswith("model name"):
                        cpu_name = line.split(':')[1].strip()
                        break
        else:
            cpu_name = platform.processor() or "Unknown CPU"
    except:
        cpu_name = "Unknown CPU"

    return {
        "device_name": cpu_name,
        "cpu_count": cpu_count,
        "torch_device": "cpu",
        "optimization_level": "basic",

        # TTS optimizations
        "tts_batch_size": 1,
        "tts_use_gpu": False,
        "tts_precision": "float32",
        "tts_compile": False,

        # LLM optimizations
        "llm_gpu_layers": 0,
        "llm_batch_size": 512,
        "llm_context_length": 2048,
        "llm_threads": min(cpu_count, 8),
        "llm_use_flash_attention": False,

        # STT optimizations
        "stt_device": "cpu",
        "stt_batch_size": 1,
        "stt_precision": "float32",

        # Audio processing optimizations
        "audio_processing_device": "cpu",
        "audio_parallel_workers": 1,

        # Memory optimizations
        "enable_memory_efficient_attention": False,
        "gradient_checkpointing": True,
        "mixed_precision": False,
    }


def _get_amd_info() -> Dict[str, Any]:
    """Get AMD GPU optimization parameters."""
    cpu_count = multiprocessing.cpu_count()

    return {
        "device_name": "AMD GPU",
        "torch_device": "cuda",  # ROCm uses CUDA API
        "optimization_level": "medium",

        # TTS optimizations
        "tts_batch_size": 4,
        "tts_use_gpu": True,
        "tts_precision": "float32",  # ROCm works better with float32
        "tts_compile": False,

        # LLM optimizations
        "llm_gpu_layers": -1,
        "llm_batch_size": 1024,
        "llm_context_length": 4096,
        "llm_threads": min(cpu_count, 8),
        "llm_use_flash_attention": False,  # Limited support on ROCm

        # STT optimizations
        "stt_device": "cuda",
        "stt_batch_size": 8,
        "stt_precision": "float32",

        # Audio processing optimizations
        "audio_processing_device": "cuda",
        "audio_parallel_workers": 2,

        # Memory optimizations
        "enable_memory_efficient_attention": True,
        "gradient_checkpointing": True,
        "mixed_precision": False,
    }


def _check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def get_optimized_config(base_config: Dict[str, Any], device_type: DeviceType, device_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get optimized configuration based on device capabilities.

    Args:
        base_config: Base configuration dictionary
        device_type: Detected device type
        device_info: Device information from detect_device()

    Returns:
        Optimized configuration dictionary
    """
    optimized_config = base_config.copy()

    # Apply device-specific optimizations
    if device_type == DeviceType.NVIDIA_GPU:
        optimized_config.update({
            "cuda_device": device_info["cuda_device"],
            "torch_device": device_info["torch_device"],
            "batch_size": device_info.get("tts_batch_size", 4),
            "precision": device_info.get("tts_precision", "float16"),
            "use_gpu": device_info.get("tts_use_gpu", True),
            "compile_model": device_info.get("tts_compile", False),
        })

        # Set CUDA environment variables for optimal performance
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        if device_info.get("is_high_end", False):
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

    elif device_type == DeviceType.APPLE_SILICON:
        optimized_config.update({
            "torch_device": device_info["torch_device"],
            "batch_size": device_info.get("tts_batch_size", 2),
            "precision": device_info.get("tts_precision", "float32"),
            "use_gpu": device_info.get("tts_use_gpu", False),
            "compile_model": False,
        })

        # Set MPS environment variables for MAXIMUM performance
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        # AGGRESSIVE MPS optimizations for TTS workloads
        os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "garbage_collection"
        # MAXIMUM Apple Silicon performance settings
        os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"  # Use Metal directly
        os.environ["METAL_FORCE_INTEL_GPU"] = "0"  # Force discrete GPU
        os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"  # Prefer Metal over CPU
        # AGGRESSIVE threading for M4 Max
        cpu_cores = device_info.get('cpu_count', 8)
        max_threads = str(min(cpu_cores, 14))
        os.environ["OMP_NUM_THREADS"] = max_threads
        os.environ["MKL_NUM_THREADS"] = max_threads
        os.environ["NUMEXPR_NUM_THREADS"] = max_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads
        print(f"🚀 Set MAXIMUM threading environment variables: {max_threads} threads")

    else:  # CPU or AMD
        optimized_config.update({
            "torch_device": device_info["torch_device"],
            "batch_size": device_info.get("tts_batch_size", 1),
            "precision": device_info.get("tts_precision", "float32"),
            "use_gpu": device_info.get("tts_use_gpu", False),
            "compile_model": False,
        })

    return optimized_config


def print_device_info(device_type: DeviceType, device_info: Dict[str, Any]) -> None:
    """Print device information and optimization status."""
    print(f"\n🔧 Device Optimization Status:")
    print(f"   Device Type: {device_type.value}")
    print(f"   Device Name: {device_info.get('device_name', 'Unknown')}")
    print(
        f"   Optimization Level: {device_info.get('optimization_level', 'basic')}")

    if device_type == DeviceType.NVIDIA_GPU:
        print(f"   GPU Memory: {device_info.get('memory_gb', 0):.1f} GB")
        print(
            f"   Compute Capability: {device_info.get('compute_capability', 'unknown')}")
        print(
            f"   High-End GPU: {'Yes' if device_info.get('is_high_end', False) else 'No'}")
    elif device_type == DeviceType.APPLE_SILICON:
        print(f"   CPU Cores: {device_info.get('cpu_count', 0)}")
        print(
            f"   MPS Available: {'Yes' if device_info.get('torch_device') == 'mps' else 'No'}")
        print(
            f"   Performance Tier: {'High-End' if device_info.get('is_high_end', False) else ('Pro' if device_info.get('is_pro', False) else 'Standard')}")

    print(f"   Torch Device: {device_info.get('torch_device', 'cpu')}")
    print()


# Global device detection (cached)
_device_type = None
_device_info = None


def get_device_info() -> Tuple[DeviceType, Dict[str, Any]]:
    """Get cached device information."""
    global _device_type, _device_info
    if _device_type is None or _device_info is None:
        _device_type, _device_info = detect_device()
    return _device_type, _device_info
