"""
TTS (Text-to-Speech) inference module with comprehensive hardware optimization.

Apple Silicon MPS Optimizations:
- Automatic MPS device detection and utilization for F5-TTS, XTTS, and Zonos
- MPS-specific memory management and fallback mechanisms
- Optimized thread counts and environment variables for Apple Silicon
- Inference mode usage for better MPS performance
- Robust error handling with automatic CPU fallback
- Environment variable optimizations for MPS workloads
"""

import os
import sys
import subprocess
import tempfile
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
import torchaudio
import soundfile as sf
import numpy as np
import base64
import io
from torch.serialization import add_safe_globals

# Import device optimization utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from device_optimization import get_device_info, get_optimized_config, print_device_info, DeviceType
    DEVICE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Device optimization not available. Using default configurations.")
    DEVICE_OPTIMIZATION_AVAILABLE = False
    DeviceType = None

# Global device info cache
_device_type = None
_device_info = None

# Global model caches with AGGRESSIVE management for maximum speed
_f5tts_model = None
_xtts_model = None
_model_load_times = {}
_model_memory_usage = {}

# AGGRESSIVE caching for maximum speed
_audio_generation_cache = {}  # Cache generated audio
_model_warm_cache = {}  # Keep models warm and ready
_precompiled_models = {}  # Store compiled models for reuse

# Apple Silicon specific optimizations
_mps_available = False
_apple_neural_engine_available = False

# NVIDIA GPU specific optimizations
_cuda_memory_fraction = 0.8
_mixed_precision_enabled = False

# Model-specific optimization flags
_xtts_mixed_precision_enabled = False  # XTTS has numerical issues with FP16

# Global variables for model caching and performance tracking
_f5tts_load_time = None
_xtts_load_time = None
_f5tts_memory_usage = None
_xtts_memory_usage = None

# Zonos persistent worker management
_zonos_workers = {}  # Dictionary to store persistent workers by character/model
_zonos_worker_lock = None


def normalize_tts_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize TTS configuration to ensure proper data types for all models.
    This fixes issues where numeric values might be stored as strings.
    """
    if not config:
        return config
    
    # Create a copy to avoid modifying the original
    normalized = config.copy()
    
    # XTTS-specific numeric conversions
    xtts_numeric_fields = {
        'repetition_penalty': float,
        'top_k': int,
        'top_p': float,
        'speed': float,
    }
    
    # Zonos-specific numeric conversions
    zonos_numeric_fields = {
        'seed': int,
        'cfg_scale': float,
        'speaking_rate': int,
        'frequency_max': int,
        'pitch_standard_deviation': int,
        'e1': float,
        'e2': float,
        'e3': float,
        'e4': float,
        'e5': float,
        'e6': float,
        'e7': float,
        'e8': float,
    }
    
    # Audio preprocessing numeric conversions
    preprocessing_numeric_fields = {
        'top_db': float,
        'fade_length_ms': int,
        'batch_size': int,
    }
    
    # Apply conversions for all relevant fields
    all_numeric_fields = {**xtts_numeric_fields, **zonos_numeric_fields, **preprocessing_numeric_fields}
    
    for field, conversion_func in all_numeric_fields.items():
        if field in normalized and normalized[field] is not None:
            try:
                # Convert to proper type if it's not already the correct type
                if isinstance(normalized[field], str) or (conversion_func == float and isinstance(normalized[field], int)) or (conversion_func == int and isinstance(normalized[field], float)):
                    normalized[field] = conversion_func(normalized[field])
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {field}={normalized[field]} to {conversion_func.__name__}: {e}")
    
    # Handle boolean conversions
    boolean_fields = ['preprocess_audio', 'clean_audio', 'remove_silence', 'enhance_audio', 'skip_all_processing', 
                     'bass_boost', 'treble_boost', 'compression', 'enable_text_splitting', 'use_mixed_precision',
                     'memory_efficient', 'use_mps', 'mps_fallback', 'use_neural_engine', 'optimize_for_latency',
                     'enable_flash_attention', 'compile_model', 'use_cuda_graphs', 'fast_mode']
    
    for field in boolean_fields:
        if field in normalized and normalized[field] is not None:
            if isinstance(normalized[field], str):
                normalized[field] = normalized[field].lower() in ('true', '1', 'yes', 'on')
            elif not isinstance(normalized[field], bool):
                normalized[field] = bool(normalized[field])
    
    return normalized


def _get_device_optimization():
    """Get cached device optimization info."""
    global _device_type, _device_info, _mps_available, _apple_neural_engine_available
    global _mixed_precision_enabled, _xtts_mixed_precision_enabled

    if DEVICE_OPTIMIZATION_AVAILABLE and (_device_type is None or _device_info is None):
        _device_type, _device_info = get_device_info()
        print_device_info(_device_type, _device_info)

        # Initialize device-specific features
        if _device_type == DeviceType.APPLE_SILICON:
            _mps_available = torch.backends.mps.is_available(
            ) if hasattr(torch.backends, 'mps') else False
            # Check for Apple Neural Engine availability (approximate)
            _apple_neural_engine_available = _device_info.get(
                'is_high_end', False) or _device_info.get('is_pro', False)
            print(
                f"🍎 Apple Silicon Features: MPS={_mps_available}, Neural Engine={_apple_neural_engine_available}")
            
            # Set MPS-specific optimizations
            if _mps_available:
                # Enable MPS fallback for unsupported operations
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                # Set MPS memory fraction for better memory management
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                print("🍎 MPS environment optimizations enabled")

        elif _device_type == DeviceType.NVIDIA_GPU:
            _mixed_precision_enabled = _device_info.get(
                'mixed_precision', True)
            # XTTS has numerical stability issues with FP16, so disable mixed precision for it
            _xtts_mixed_precision_enabled = False
            # Set CUDA memory fraction based on GPU memory
            gpu_memory = _device_info.get('memory_gb', 8)
            _cuda_memory_fraction = min(
                0.9, max(0.7, (gpu_memory - 2) / gpu_memory))
            print(
                f"🎯 NVIDIA GPU Features: Mixed Precision={_mixed_precision_enabled}, Memory Fraction={_cuda_memory_fraction:.2f}")
            print(
                f"🎯 XTTS Mixed Precision: {_xtts_mixed_precision_enabled} (disabled for numerical stability)")


    return _device_type, _device_info


def _optimize_for_apple_silicon(model, device_info: Dict[str, Any]):
    """Apply aggressive Apple Silicon optimizations for real-time TTS."""
    try:
        # Enable MPS optimizations if available
        if _mps_available and hasattr(model, 'to'):
            try:
                model = model.to('mps')
                print("✓ Enabled MPS acceleration for Apple Silicon")
                
                # Ensure model is in eval mode for inference
                if hasattr(model, 'eval'):
                    model.eval()
                    
                # Aggressive MPS warm-up for M4 Max
                try:
                    # Larger warm-up for better MPS performance
                    dummy_tensor = torch.randn(1, 512, device='mps', dtype=torch.float32)
                    for _ in range(3):  # Multiple warm-up passes
                        _ = dummy_tensor * 2 + 1
                    del dummy_tensor
                    print("✓ Aggressive MPS warm-up completed")
                except Exception as warmup_error:
                    print(f"Warning: MPS warm-up failed: {warmup_error}")
                    
            except Exception as mps_error:
                print(f"Warning: Failed to move model to MPS, falling back to CPU: {mps_error}")
                if hasattr(model, 'to'):
                    model = model.to('cpu')

        # Apply aggressive memory optimizations for M4 Max
        if _mps_available:
            try:
                if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                    torch.backends.mps.set_per_process_memory_fraction(0.9)  # Use more memory for speed
                # Enable MPS backend optimizations
                if hasattr(torch.backends.mps, 'enabled'):
                    torch.backends.mps.enabled = True
                print("✓ Aggressive MPS memory optimizations applied")
            except Exception as mem_error:
                print(f"Warning: MPS memory optimization failed: {mem_error}")

        # Enable torch.compile for M4 Max (latest PyTorch has better MPS support)
        if device_info.get('tts_compile', False) and hasattr(torch, 'compile'):
            try:
                print("🚀 Attempting torch.compile for M4 Max real-time inference...")
                model = torch.compile(model, mode="reduce-overhead", dynamic=False)
                print("✓ torch.compile enabled for M4 Max - expect faster inference after warm-up")
            except Exception as compile_error:
                print(f"Warning: torch.compile failed on M4 Max: {compile_error}")

        # Enable optimized attention if available
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("✓ Scaled dot product attention available for Apple Silicon")
        except Exception as attention_error:
            print(f"Warning: Attention optimization check failed: {attention_error}")

        # Set MAXIMUM thread counts for M4 Max ULTRA performance
        try:
            # M4 Max - USE ALL THE CORES! MAXIMUM AGGRESSION!
            cpu_cores = device_info.get('cpu_count', 16)
            if "m4 max" in device_info.get('device_name', '').lower():
                optimal_threads = min(cpu_cores, 16)  # Use ALL 16 cores!
                inter_op_threads = min(8, cpu_cores // 2)  # Maximum inter-op threading
            else:
                optimal_threads = min(cpu_cores, 12)
                inter_op_threads = min(6, cpu_cores // 2)
                
            torch.set_num_threads(optimal_threads)
            torch.set_num_interop_threads(inter_op_threads)
            print(f"🚀 MAXIMUM THREADING for M4 Max: {optimal_threads} threads, {inter_op_threads} inter-op")
            
            # AGGRESSIVE performance environment variables for M4 Max
            if "m4 max" in device_info.get('device_name', '').lower():
                env_vars = {
                    "MKL_NUM_THREADS": str(optimal_threads),
                    "NUMEXPR_NUM_THREADS": str(optimal_threads),
                    "OMP_NUM_THREADS": str(optimal_threads),
                    "VECLIB_MAXIMUM_THREADS": str(optimal_threads),
                    "ACCELERATE_NEW_LAPACK": "1",  # Use new LAPACK for better performance
                    "ACCELERATE_LAPACK_ILP64": "1",  # Use 64-bit integers for larger problems
                }
                for key, value in env_vars.items():
                    os.environ[key] = value
                print("🚀 MAXIMUM M4 Max performance environment variables set")
                
        except Exception as thread_error:
            print(f"Warning: Thread optimization failed: {thread_error}")

        return model

    except Exception as e:
        print(f"Warning: Apple Silicon optimizations failed: {e}")
        return model


def _optimize_for_nvidia_gpu(model, device_info: Dict[str, Any], model_name: str = ""):
    """Apply NVIDIA GPU specific optimizations with model-specific handling."""
    try:
        # Move model to GPU
        if hasattr(model, 'to'):
            model = model.to('cuda')
            print("✓ Moved model to CUDA")

        # Enable mixed precision if supported and appropriate for the model
        should_use_mixed_precision = _mixed_precision_enabled
        if model_name.lower() == "xtts":
            should_use_mixed_precision = _xtts_mixed_precision_enabled
            if not should_use_mixed_precision:
                print(
                    "✓ Using FP32 for XTTS (mixed precision disabled for numerical stability)")

        if should_use_mixed_precision and hasattr(model, 'half'):
            try:
                model = model.half()
                print("✓ Enabled mixed precision (FP16)")
            except Exception as e:
                print(f"Warning: Mixed precision failed, using FP32: {e}")

        # Apply torch.compile for high-end GPUs (but be careful with XTTS)
        if device_info.get('is_high_end', False) and hasattr(torch, 'compile'):
            # XTTS has known issues with torch.compile, so be more cautious
            if model_name.lower() == "xtts":
                print("⚠️  Skipping torch.compile for XTTS (known compatibility issues)")
            else:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("✓ Applied torch.compile optimization")
                except Exception as e:
                    print(f"Warning: torch.compile failed: {e}")

        # Set CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(_cuda_memory_fraction)
            torch.cuda.empty_cache()
            print(f"✓ Set CUDA memory fraction to {_cuda_memory_fraction:.2f}")

        # Enable CUDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        return model

    except Exception as e:
        print(f"Warning: Some NVIDIA GPU optimizations failed: {e}")
        return model


def _get_f5tts_model(force_init: bool = False):
    """Get or initialize F5TTS model with comprehensive device optimization."""
    global _f5tts_model, _model_load_times, _model_memory_usage

    # Check if model is already loaded and return it without re-initialization
    if _f5tts_model is not None and not force_init:
        print("✓ F5TTS model already loaded in memory, reusing...")
        return _f5tts_model

    if _f5tts_model is None or force_init:
        try:
            from f5_tts.api import F5TTS

            # Get device optimization info
            device_type, device_info = _get_device_optimization()

            print(
                f"Initializing F5TTS model with {device_type.value} optimization...")
            start_time = time.perf_counter()

            # Determine optimal device
            if DEVICE_OPTIMIZATION_AVAILABLE:
                if device_type == DeviceType.NVIDIA_GPU:
                    device_str = "cuda"
                elif device_type == DeviceType.APPLE_SILICON and _mps_available:
                    device_str = "mps"
                else:
                    device_str = "cpu"
            else:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize F5TTS with device-specific settings
            # Note: F5TTS API only accepts device parameter in constructor
            model_kwargs = {"device": device_str}

            _f5tts_model = F5TTS(**model_kwargs)

            # Apply device-specific optimizations
            if device_type == DeviceType.APPLE_SILICON:
                _f5tts_model = _optimize_for_apple_silicon(
                    _f5tts_model, device_info)
            elif device_type == DeviceType.NVIDIA_GPU:
                _f5tts_model = _optimize_for_nvidia_gpu(
                    _f5tts_model, device_info, "f5tts")

            # Record performance metrics
            load_time = time.perf_counter() - start_time
            _model_load_times['f5tts'] = load_time

            # Estimate memory usage
            if torch.cuda.is_available():
                _model_memory_usage['f5tts'] = torch.cuda.memory_allocated(
                ) / 1024**3

            print(
                f"✓ F5TTS model initialized in {load_time:.2f}s on {device_str}")
            if 'f5tts' in _model_memory_usage:
                print(
                    f"✓ GPU Memory usage: {_model_memory_usage['f5tts']:.2f} GB")

        except ImportError as e:
            print(f"Error importing F5TTS: {e}")
            raise ImportError(
                "F5TTS not available. Please install with: pip install f5-tts")
        except Exception as e:
            print(f"Error initializing F5TTS model: {e}")
            raise

    return _f5tts_model


def _get_xtts_model(force_init: bool = False):
    """Get or initialize XTTS model with comprehensive device optimization."""
    global _xtts_model, _model_load_times, _model_memory_usage

    # Check if model is already loaded and return it without re-initialization
    if _xtts_model is not None and not force_init:
        print("✓ XTTS model already loaded in memory, reusing...")
        return _xtts_model

    if _xtts_model is None or force_init:
        try:
            from TTS.api import TTS

            # Get device optimization info
            device_type, device_info = _get_device_optimization()

            print(
                f"Initializing XTTS model with {device_type.value} optimization...")
            start_time = time.perf_counter()

            # Determine optimal device
            if DEVICE_OPTIMIZATION_AVAILABLE:
                if device_type == DeviceType.NVIDIA_GPU:
                    device_str = "cuda"
                elif device_type == DeviceType.APPLE_SILICON and _mps_available:
                    device_str = "mps"
                else:
                    device_str = "cpu"
            else:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"

            # Set PyTorch configuration for XTTS
            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)
            torch.load = patched_load

            try:
                # Initialize with device-specific optimizations
                _xtts_model = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2")
                _xtts_model = _xtts_model.to(device_str)

                # Apply device-specific optimizations
                if device_type == DeviceType.APPLE_SILICON:
                    _xtts_model = _optimize_for_apple_silicon(
                        _xtts_model, device_info)
                elif device_type == DeviceType.NVIDIA_GPU:
                    _xtts_model = _optimize_for_nvidia_gpu(
                        _xtts_model, device_info, "xtts")

                # Record performance metrics
                load_time = time.perf_counter() - start_time
                _model_load_times['xtts'] = load_time

                # Estimate memory usage
                if torch.cuda.is_available():
                    _model_memory_usage['xtts'] = torch.cuda.memory_allocated(
                    ) / 1024**3

                print(
                    f"✓ XTTS model initialized in {load_time:.2f}s on {device_str}")
                if 'xtts' in _model_memory_usage:
                    print(
                        f"✓ GPU Memory usage: {_model_memory_usage['xtts']:.2f} GB")

            finally:
                # Restore original torch.load
                torch.load = original_load

        except ImportError as e:
            print(f"Error importing TTS: {e}")
            raise ImportError(
                "TTS not available. Please install with: pip install TTS")
        except Exception as e:
            print(f"Error initializing XTTS model: {e}")
            raise

    return _xtts_model


def generate_audio(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    config: Optional[Dict[str, Any]] = None,
    auto_download: bool = True,
    fast_mode: bool = False
) -> str:
    """
    Generate cloned audio using specified TTS model with comprehensive hardware optimization.

    Args:
        model: Model name/path ("F5TTS", "tts_models/multilingual/multi-dataset/xtts_v2", 
               "Zyphra/Zonos-v0.1-transformer")
        ref_audio: Path to reference audio file
        ref_text: Reference text (what was said in ref_audio)
        gen_text: Text to generate with cloned voice
        config: Additional configuration parameters
        auto_download: Whether to automatically download model if not available
        fast_mode: Enable real-time optimizations (skip heavy preprocessing, use aggressive caching)

    Returns:
        Path to generated audio file
    """
    print(f"🎵 TTS Audio Generation Request:")
    print(f"   • Model: {model}")
    print(f"   • Reference Audio: {ref_audio}")
    print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
    print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
    print(f"   • Auto Download: {auto_download}")
    print(f"   • Fast Mode: {fast_mode}")
    print(f"   • Input Config: {config}")
    
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

    # AGGRESSIVE CACHING for maximum speed - check if we've generated this exact audio before
    cache_key = f"{model}_{hash(ref_audio)}_{hash(ref_text)}_{hash(gen_text)}_{fast_mode}"
    if fast_mode and cache_key in _audio_generation_cache:
        cached_file = _audio_generation_cache[cache_key]
        if os.path.exists(cached_file):
            print(f"🚀 CACHE HIT! Returning previously generated audio: {cached_file}")
            return cached_file
        else:
            # Remove stale cache entry
            del _audio_generation_cache[cache_key]

    # Get device optimization info
    device_type, device_info = _get_device_optimization()

    # Set default config values with comprehensive device optimization
    default_config = {
        'output_dir': 'outputs',
        'cuda_device': '0',
        'language': 'en',
        'coqui_tos_agreed': True,
        'torch_force_no_weights_only_load': True,
        'fast_mode': fast_mode  # Pass fast mode to config
    }

    # Apply device-specific optimizations
    if DEVICE_OPTIMIZATION_AVAILABLE:
        default_config = get_optimized_config(
            default_config, device_type, device_info)

        # Add comprehensive device-specific TTS optimizations
        if device_type == DeviceType.NVIDIA_GPU:
            default_config.update({
                'batch_size': device_info.get('tts_batch_size', 4),
                'use_mixed_precision': _mixed_precision_enabled,
                'enable_flash_attention': device_info.get('llm_use_flash_attention', True),
                'compile_model': device_info.get('tts_compile', device_info.get('is_high_end', False)),
                'memory_efficient': True,
                'use_cuda_graphs': device_info.get('is_high_end', False),
            })
        elif device_type == DeviceType.APPLE_SILICON:
            default_config.update({
                'batch_size': min(device_info.get('tts_batch_size', 2), 4 if device_info.get('is_high_end', False) else 2),
                'use_mps': _mps_available,
                'mps_fallback': True,
                'memory_efficient': True,
                'use_neural_engine': _apple_neural_engine_available,
                'optimize_for_latency': device_info.get('is_high_end', False),
            })

    if config:
        default_config.update(config)
    config = default_config
    
    # Normalize config to ensure proper data types
    config = normalize_tts_config(config)

    print(f"🔧 Final TTS Configuration:")
    for key, value in config.items():
        if key in ['ref_text', 'gen_text']:
            # Truncate long text for readability
            display_value = value[:100] + '...' if len(str(value)) > 100 else value
            print(f"   • {key}: {display_value}")
        else:
            print(f"   • {key}: {value}")

    # Check if model is already loaded, otherwise ensure it's available
    model_lower = model.lower()
    model_already_loaded = False

    if model_lower == "f5tts":
        model_already_loaded = is_model_loaded("f5tts")
    elif model_lower == "xtts":
        model_already_loaded = is_model_loaded("xtts")

    if model_already_loaded:
        print(
            f"✓ Model {model} already loaded in memory, skipping availability check...")
    elif auto_download:
        print(f"📥 Checking availability for model: {model}")
        success = ensure_model_available(model, config.get('cache_dir'))
        if not success:
            print(
                f"Warning: Model {model} may not be fully available. Attempting to proceed...")

    # Ensure output directory exists
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based output filename
    timestamp = int(time.time())
    output_file = output_dir / f"generated_audio_{timestamp}.wav"

    print(f"🎯 Routing to model implementation: {model_lower}")

    # Route to appropriate model implementation
    result_file = None
    if model_lower == "f5tts":
        result_file = _generate_f5tts(ref_audio, ref_text, gen_text, str(output_file), config)
    elif model_lower == "xtts":
        result_file = _generate_xtts(ref_audio, ref_text, gen_text, str(output_file), config)
    elif model_lower == "zonos":
        result_file = _generate_zonos("Zyphra/Zonos-v0.1-transformer", ref_audio, ref_text, gen_text, str(output_file), config)
    else:
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {get_supported_models()}")
    
    # AGGRESSIVE CACHING - store the result for future use
    if fast_mode and result_file and os.path.exists(result_file):
        _audio_generation_cache[cache_key] = result_file
        print(f"🚀 CACHED generated audio for future use: {cache_key}")
        
        # Limit cache size to prevent memory issues (keep last 100 generations)
        if len(_audio_generation_cache) > 100:
            oldest_key = next(iter(_audio_generation_cache))
            old_file = _audio_generation_cache.pop(oldest_key)
            print(f"🧹 Cache cleanup: removed {oldest_key}")
    
    return result_file


def ensure_model_available(model: str, cache_dir: Optional[str] = None) -> bool:
    """
    Ensure that the specified model is available on disk (downloads/installs if necessary).
    This does NOT load the model into memory - use preload_models_smart() for that.

    Args:
        model: Model name to check/download
        cache_dir: Directory to cache models

    Returns:
        True if model is available on disk, False otherwise
    """
    # Import here to avoid circular imports
    try:
        from .preprocess import download_voice_models, check_single_model_availability
    except ImportError:
        print("Warning: Could not import preprocess module for model downloading")
        return False

    # Check if the specific model is already available
    if check_single_model_availability(model):
        print(f"Model {model} is already available")
        return True

    # Determine which model to download
    model_lower = model.lower()
    if model_lower == "f5tts":
        print(f"Downloading model: {model}")
        results = download_voice_models(["F5TTS"], cache_dir=cache_dir)
        return results.get("F5TTS", False)

    elif model_lower == "xtts":
        print(f"Downloading model: {model}")
        results = download_voice_models(
            ["tts_models/multilingual/multi-dataset/xtts_v2"], cache_dir=cache_dir)
        return results.get("tts_models/multilingual/multi-dataset/xtts_v2", False)

    elif model_lower == "zonos":
        print(f"Downloading model: {model}")
        results = download_voice_models(
            ["Zyphra/Zonos-v0.1-transformer"], cache_dir=cache_dir)
        return results.get("Zyphra/Zonos-v0.1-transformer", False)
    else:
        print(f"Unknown model type: {model}")
        return False


def _generate_f5tts(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using F5-TTS API with comprehensive hardware optimization."""
    try:
        print(f"🎤 F5-TTS Generation Parameters:")
        print(f"   • Reference Audio: {ref_audio}")
        print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
        print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
        print(f"   • Output File: {output_file}")
        print(f"   • Config: {config}")
        
        print(f"Generating F5TTS audio for text: {gen_text[:100]}...")
        print(f"Using reference: {ref_audio}")
        print(f"Reference text: {ref_text[:100]}...")

        # Get the F5TTS model (should be cached in memory)
        f5tts_model = _get_f5tts_model()
        device_type, device_info = _get_device_optimization()

        start_time = time.perf_counter()

        # ULTRA FAST mode: Skip ALL validation and preprocessing for MAXIMUM SPEED
        if config.get('fast_mode', False):
            print("🚀 ULTRA FAST MODE: Skipping ALL validation - MAXIMUM SPEED!")
        else:
            # Only do validation in non-fast mode
            # Preprocess and validate audio/text lengths to prevent tensor mismatch
            import librosa
            try:
                # Load reference audio to check duration
                ref_audio_data, sr = librosa.load(ref_audio, sr=22050)
                ref_duration = len(ref_audio_data) / sr

                # Estimate text durations (rough approximation: ~150 words per minute, ~5 chars per word)
                ref_text_duration = len(ref_text) / \
                    (150 * 5 / 60)  # Convert to seconds
                gen_text_duration = len(gen_text) / (150 * 5 / 60)

                print(f"Reference audio duration: {ref_duration:.2f}s")
                print(
                    f"Reference text estimated duration: {ref_text_duration:.2f}s")
                print(
                    f"Generation text estimated duration: {gen_text_duration:.2f}s")

                # If there's a significant mismatch, we might get tensor errors
                # Adjust or warn about potential issues
                if abs(ref_duration - ref_text_duration) > 3.0:  # More than 3 seconds difference
                    print(
                        f"⚠️  Warning: Reference audio ({ref_duration:.1f}s) and text ({ref_text_duration:.1f}s) duration mismatch may cause tensor issues")
            except Exception as e:
                print(f"Warning: Could not validate audio/text lengths: {e}")

        # Prepare generation parameters - ULTRA optimized for maximum speed
        is_fast_mode = config.get('fast_mode', False)
        generation_params = {
            'ref_file': ref_audio,
            'ref_text': ref_text,
            'gen_text': gen_text,
            'file_wave': output_file,
            'seed': 42 if is_fast_mode else config.get('seed', None),  # Always use fixed seed in fast mode
            'remove_silence': False if is_fast_mode else config.get('remove_silence', True),  # Never remove silence in fast mode
        }
        
        # ULTRA FAST mode optimizations - skip everything possible
        if is_fast_mode:
            print("🚀 ULTRA FAST MODE optimizations enabled for F5-TTS")
            generation_params.update({
                'speed': 1.0,  # Slightly faster playback for speed
            })

        print(f"🎯 F5-TTS Final Generation Parameters:")
        for key, value in generation_params.items():
            if key in ['ref_text', 'gen_text']:
                # Truncate long text for readability
                display_value = value[:100] + '...' if len(str(value)) > 100 else value
                print(f"   • {key}: {display_value}")
            else:
                print(f"   • {key}: {value}")

        # Use device-specific optimizations with retry logic for tensor mismatches
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if device_type == DeviceType.NVIDIA_GPU and _mixed_precision_enabled:
                    print(f"🎯 Using NVIDIA GPU with mixed precision for F5-TTS")
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        wav, sr, spec = f5tts_model.infer(**generation_params)
                elif device_type == DeviceType.APPLE_SILICON and _mps_available:
                    # Use MPS optimizations for Apple Silicon
                    print(f"🍎 Using Apple Silicon MPS optimization for F5-TTS")
                    with torch.inference_mode():  # Use inference mode for better MPS performance
                        try:
                            wav, sr, spec = f5tts_model.infer(**generation_params)
                        except Exception as mps_error:
                            # Fallback to CPU if MPS fails
                            if "mps" in str(mps_error).lower() or "metal" in str(mps_error).lower():
                                print(f"⚠️  MPS inference failed, falling back to CPU: {mps_error}")
                                # Temporarily move model to CPU for this inference
                                original_device = next(f5tts_model.parameters()).device if hasattr(f5tts_model, 'parameters') else 'mps'
                                if hasattr(f5tts_model, 'to'):
                                    f5tts_model = f5tts_model.to('cpu')
                                wav, sr, spec = f5tts_model.infer(**generation_params)
                                # Move back to original device
                                if hasattr(f5tts_model, 'to'):
                                    f5tts_model = f5tts_model.to(original_device)
                            else:
                                raise mps_error
                else:
                    print(f"💻 Using CPU/default device for F5-TTS")
                    wav, sr, spec = f5tts_model.infer(**generation_params)
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                if "Sizes of tensors must match" in error_msg and attempt < max_retries:
                    print(
                        f"⚠️  Tensor size mismatch (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                    print("🔄 Trying with adjusted parameters...")

                    # Try with different parameters to fix tensor mismatch
                    if attempt == 0:
                        # First retry: disable silence removal which can cause length mismatches
                        generation_params['remove_silence'] = False
                        print("   • Disabled silence removal")
                    elif attempt == 1:
                        # Second retry: try with a different seed to change internal processing
                        generation_params['seed'] = 42
                        print("   • Using fixed seed")
                else:
                    # Either not a tensor mismatch error, or we've exhausted retries
                    raise e

        runtime = time.perf_counter() - start_time
        print(f"✓ F5-TTS generation completed in {runtime:.3f}s")

        # Validate the output file exists
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"F5-TTS did not generate expected output file: {output_file}")

        print(f"✓ F5-TTS output saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"✗ F5-TTS generation failed: {str(e)}")
        raise RuntimeError(f"F5-TTS generation failed: {str(e)}")


def _generate_xtts(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using XTTS-v2 API with comprehensive hardware optimization."""
    try:
        print(f"🎤 XTTS-v2 Generation Parameters:")
        print(f"   • Reference Audio: {ref_audio}")
        print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
        print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
        print(f"   • Output File: {output_file}")
        print(f"   • Config: {config}")
        
        print(f"Generating XTTS audio for text: {gen_text[:100]}...")
        print(f"Using reference: {ref_audio}")

        # Get the XTTS model (should be cached in memory)
        xtts_model = _get_xtts_model()
        device_type, device_info = _get_device_optimization()

        start_time = time.perf_counter()

        # Prepare generation parameters with model-specific settings
        generation_params = {
            'text': gen_text,
            'speaker_wav': ref_audio,
            'language': config.get('language', 'en'),
            'file_path': output_file
        }
        
        # Add XTTS-specific parameters if available
        if 'repetition_penalty' in config:
            generation_params['repetition_penalty'] = config['repetition_penalty']
        if 'top_k' in config:
            generation_params['top_k'] = config['top_k']
        if 'top_p' in config:
            generation_params['top_p'] = config['top_p']
        if 'speed' in config:
            generation_params['speed'] = config['speed']
        if 'enable_text_splitting' in config:
            generation_params['enable_text_splitting'] = config['enable_text_splitting']

        print(f"🎯 XTTS-v2 Final Generation Parameters:")
        for key, value in generation_params.items():
            if key == 'text':
                # Truncate long text for readability
                display_value = value[:100] + '...' if len(str(value)) > 100 else value
                print(f"   • {key}: {display_value}")
            else:
                print(f"   • {key}: {value}")

        # Use device-specific optimizations (but not mixed precision for XTTS due to numerical instability)
        try:
            if device_type == DeviceType.NVIDIA_GPU and _xtts_mixed_precision_enabled:
                print(f"🎯 Using NVIDIA GPU with mixed precision for XTTS-v2")
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    xtts_model.tts_to_file(**generation_params)
            elif device_type == DeviceType.APPLE_SILICON and _mps_available:
                # Use MPS optimizations for Apple Silicon with XTTS
                print(f"🍎 Using Apple Silicon MPS optimization for XTTS-v2")
                with torch.inference_mode():  # Use inference mode for better MPS performance
                    try:
                        xtts_model.tts_to_file(**generation_params)
                    except Exception as mps_error:
                        # Fallback to CPU if MPS fails with XTTS
                        if "mps" in str(mps_error).lower() or "metal" in str(mps_error).lower():
                            print(f"⚠️  MPS inference failed for XTTS, falling back to CPU: {mps_error}")
                            # Temporarily move model to CPU for this inference
                            original_device = next(xtts_model.synthesizer.tts_model.parameters()).device if hasattr(xtts_model, 'synthesizer') and hasattr(xtts_model.synthesizer, 'tts_model') else 'mps'
                            if hasattr(xtts_model, 'to'):
                                xtts_model = xtts_model.to('cpu')
                            xtts_model.tts_to_file(**generation_params)
                            # Move back to original device
                            if hasattr(xtts_model, 'to'):
                                xtts_model = xtts_model.to(original_device)
                        else:
                            raise mps_error
            else:
                # Use FP32 for XTTS to avoid numerical instability issues
                print(f"💻 Using CPU/FP32 for XTTS-v2 (numerical stability)")
                xtts_model.tts_to_file(**generation_params)
        except RuntimeError as e:
            error_msg = str(e)
            if "device-side assert triggered" in error_msg or "probability tensor" in error_msg:
                print(f"⚠️  XTTS numerical instability detected: {error_msg}")
                print("🔄 Attempting recovery by clearing CUDA cache and retrying...")

                # Clear CUDA cache and try again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Retry with explicit FP32 (no autocast)
                try:
                    print("🔄 Retrying XTTS generation with FP32 precision...")
                    xtts_model.tts_to_file(**generation_params)
                    print("✓ XTTS recovery successful with FP32")
                except Exception as retry_error:
                    print(f"✗ XTTS recovery failed: {retry_error}")
                    raise RuntimeError(
                        f"XTTS generation failed even after recovery attempt: {retry_error}")
            else:
                # Re-raise if it's a different type of error
                raise e

        runtime = time.perf_counter() - start_time
        print(f"✓ XTTS-v2 generation completed in {runtime:.3f}s")

        # Validate the output file exists
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"XTTS-v2 did not generate expected output file: {output_file}")

        print(f"✓ XTTS-v2 output saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"✗ XTTS-v2 generation failed: {str(e)}")
        raise RuntimeError(f"XTTS-v2 generation failed: {str(e)}")


def _generate_zonos(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """
    Generate audio using Zonos TTS with persistent worker optimization.
    Falls back to single-shot mode if persistent worker fails.
    """
    print(f"🎤 Zonos Generation:")
    print(f"   • Model: {model}")
    print(f"   • Reference Audio: {ref_audio}")
    print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
    print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
    print(f"   • Output File: {output_file}")
    print(f"   • Config: {config}")

    # Try persistent worker first for better performance
    use_persistent = config.get('use_persistent_worker', True)
    device = config.get('torch_device', 'auto')
    
    if use_persistent:
        try:
            print("🚀 Attempting Zonos generation with persistent worker...")
            return _generate_zonos_persistent(model, ref_audio, ref_text, gen_text, output_file, config, device)
        except Exception as e:
            print(f"⚠️ Persistent worker failed: {e}")
            print("🔄 Falling back to single-shot mode...")
    
    # Fallback to single-shot subprocess mode
    return _generate_zonos_subprocess(model, ref_audio, ref_text, gen_text, output_file, config)


def _generate_zonos_persistent(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any],
    device: str = "auto"
) -> str:
    """Generate audio using persistent Zonos worker."""
    try:
        # Get or create persistent worker
        worker = _get_or_create_zonos_worker(model, device)
        
        # Prepare request
        request = {
            "model": model,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "gen_text": gen_text,
            "output_file": output_file,
            "config": config,
            "device": device
        }
        
        # Send request and get response
        response = _send_request_to_zonos_worker(worker, request)
        
        if not response.get("success", False):
            error_msg = response.get("error", "Unknown error")
            raise RuntimeError(f"Zonos worker error: {error_msg}")
        
        # Check if output file was created
        if not os.path.exists(output_file):
            raise RuntimeError(f"Zonos worker completed but output file not found: {output_file}")
        
        generation_time = response.get("generation_time", 0)
        load_time = response.get("load_time", 0)
        cached = response.get("cached", False)
        
        print(f"✓ Zonos persistent generation completed in {generation_time:.2f}s")
        if cached:
            print(f"✓ Used cached model (load time: 0s)")
        else:
            print(f"✓ Model loaded in {load_time:.2f}s")
        
        return output_file
        
    except Exception as e:
        # If persistent worker fails, clean it up and raise error
        worker_key = _get_zonos_worker_key(model, device)
        if worker_key in _zonos_workers:
            try:
                worker = _zonos_workers[worker_key]
                worker.terminate()
                del _zonos_workers[worker_key]
                print(f"🧹 Cleaned up failed Zonos worker: {worker_key}")
            except:
                pass
        raise e


def _generate_zonos_subprocess(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using single-shot Zonos subprocess (original method)."""
    import subprocess
    import json

    # Get device configuration
    device = config.get('torch_device', 'auto')

    # Get conda environment path
    zonos_env_python = _get_zonos_env_python()

    # Path to worker script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(current_dir, '..', '..', 'zonos_worker.py')

    print(f"🚀 Starting Zonos worker process...")
    print(f"   • Python: {zonos_env_python}")
    print(f"   • Script: {worker_script}")
    print(f"   • Device: {device}")

    try:
        # Run subprocess with timeout
        result = subprocess.run([
            zonos_env_python, worker_script,
            '--model', model,
            '--ref_audio', ref_audio,
            '--ref_text', ref_text,
            '--gen_text', gen_text,
            '--output_file', output_file,
            '--config', json.dumps(config),
            '--device', device
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        if result.returncode != 0:
            # Check for specific error patterns
            stderr_output = result.stderr
            stdout_output = result.stdout
            
            if "espeak" in stderr_output.lower():
                raise RuntimeError("espeak-ng is required for Zonos but not available. Please install it with: brew install espeak-ng")
            
            raise RuntimeError(f"Zonos worker process failed (exit code {result.returncode}):\nSTDERR: {stderr_output}\nSTDOUT: {stdout_output}")

        # Check if output file was created
        if not os.path.exists(output_file):
            raise RuntimeError(f"Zonos generation completed but output file not found: {output_file}")

        print("✓ Zonos single-shot generation completed successfully")
        return output_file

    except subprocess.TimeoutExpired:
        raise RuntimeError("Zonos generation timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Zonos generation failed: {str(e)}")


def get_supported_models():
    """Get list of supported TTS model options."""
    return [
        "f5tts",
        "xtts", 
        "zonos"
    ]


def validate_model(model: str) -> bool:
    """Validate if model is supported."""
    supported = get_supported_models()
    return any(model.lower() in supported_model.lower() for supported_model in supported)


def check_model_dependencies(model: str) -> Dict[str, bool]:
    """Check if required dependencies are available for the model."""
    dependencies = {
        "torch": False,
        "torchaudio": False,
        "soundfile": False
    }

    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass

    try:
        import torchaudio
        dependencies["torchaudio"] = True
    except ImportError:
        pass

    if "soundfile" in sys.modules:
        dependencies["soundfile"] = True

    model_lower = model.lower()
    if model_lower == "xtts":
        try:
            import TTS
            dependencies["TTS"] = True
        except ImportError:
            dependencies["TTS"] = False

    if model_lower == "zonos":
        try:
            import zonos
            dependencies["zonos"] = True
        except ImportError:
            dependencies["zonos"] = False

    if model_lower == "f5tts":
        # Check if F5-TTS CLI is available
        try:
            result = subprocess.run(
                ["f5-tts_infer-cli", "--help"],
                capture_output=True,
                text=True
            )
            dependencies["f5-tts"] = result.returncode == 0
        except FileNotFoundError:
            dependencies["f5-tts"] = False

    return dependencies


def generate_cloned_audio_base64(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = True  # Enable fast mode by default for real-time performance
) -> str:
    """
    Generate cloned audio and return as base64 encoded string with real-time optimizations.

    Args:
        model: Model name ("f5tts", "xtts", "zonos")
        ref_audio: Path to reference audio file
        ref_text: Reference text
        gen_text: Text to generate
        config: Additional configuration
        fast_mode: Enable real-time optimizations (default: True for speed)

    Returns:
        Base64 encoded audio data
    """
    print(f"🎵 TTS Base64 Audio Generation Request:")
    print(f"   • Model: {model}")
    print(f"   • Reference Audio: {ref_audio}")
    print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
    print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
    print(f"   • Fast Mode: {fast_mode}")
    print(f"   • Config: {config}")
    
    try:
        # Generate audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_output_path = temp_file.name

        try:
            # Generate audio using the standard API with fast mode
            output_file = generate_audio(
                model=model,
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                config=config,
                fast_mode=fast_mode
            )

            # Convert to base64
            return _audio_file_to_base64(output_file)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    except Exception as e:
        print(f"Error in audio generation: {str(e)}")
        print("Generating silent audio as fallback...")
        return generate_silent_audio_base64(duration=2.0, sample_rate=22050)


def _audio_file_to_base64(audio_file: str) -> str:
    """Convert audio file to base64 encoded string."""
    try:
        # Read audio file
        audio_data, sample_rate = sf.read(audio_file)

        # Ensure audio is 1D
        if audio_data.ndim > 1:
            if audio_data.shape[0] == 1:
                audio_data = audio_data.squeeze(0)
            elif audio_data.shape[1] == 1:
                audio_data = audio_data.squeeze(1)
            else:
                # Take first channel if stereo
                audio_data = audio_data[:,
                                        0] if audio_data.shape[1] < audio_data.shape[0] else audio_data[0, :]

        # Normalize audio to prevent clipping
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Convert to base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate,
                 format='WAV', subtype='PCM_16')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

        print(
            f"Audio successfully converted to base64 (length: {len(audio_base64)})")
        return audio_base64

    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        return generate_silent_audio_base64()


def generate_silent_audio_base64(duration: float = 2.0, sample_rate: int = 22050) -> str:
    """
    Generate silent audio as a fallback when audio generation fails.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        Base64 encoded silent audio
    """
    try:
        # Generate silent audio
        num_samples = int(duration * sample_rate)
        silent_audio = np.zeros(num_samples, dtype=np.float32)

        # Convert to base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, silent_audio, sample_rate,
                 format='WAV', subtype='PCM_16')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

        print(
            f"Silent audio generated (duration: {duration}s, sample_rate: {sample_rate})")
        return audio_base64

    except Exception as e:
        print(f"Error generating silent audio: {str(e)}")
        # Return empty string as last resort
        return ""


def preload_models_smart(models: List[str] = None, force_reload: bool = False):
    """
    Intelligently preload TTS models with comprehensive device optimization.

    Args:
        models: List of model names to preload. If None, preloads all available models.
        force_reload: Whether to force reload models even if already loaded.
    """
    if models is None:
        models = ["f5tts", "xtts"]

    device_type, device_info = _get_device_optimization()
    print(
        f"🚀 Smart preloading TTS models with {device_type.value} optimization: {models}")

    for model in models:
        model_lower = model.lower()
        try:
            # Check if model is already loaded
            if not force_reload and is_model_loaded(model_lower):
                print(
                    f"✓ TTS model {model_lower.upper()} already loaded in memory, skipping preload...")
                continue

            if model_lower == "f5tts":
                print("🔄 Preloading F5TTS model into memory...")
                _get_f5tts_model(force_init=force_reload)
                print("✓ F5TTS model preloaded and cached in memory!")
            elif model_lower == "xtts":
                print("🔄 Preloading XTTS model into memory...")
                _get_xtts_model(force_init=force_reload)
                print("✓ XTTS model preloaded and cached in memory!")
        except Exception as e:
            print(f"✗ Failed to preload {model}: {e}")

    print("🎯 Smart model preloading completed!")

    # Print performance summary
    if _model_load_times:
        print("\n📊 Model Performance Summary:")
        for model_name, load_time in _model_load_times.items():
            memory_info = f", Memory: {_model_memory_usage.get(model_name, 0):.2f} GB" if model_name in _model_memory_usage else ""
            print(
                f"   • {model_name.upper()}: Load time {load_time:.2f}s{memory_info}")


def unload_models():
    """Unload all TTS models from memory to free up resources."""
    global _f5tts_model, _xtts_model, _f5tts_load_time, _xtts_load_time, _f5tts_memory_usage, _xtts_memory_usage

    print("🧹 Unloading all TTS models from memory...")

    # Unload F5TTS model
    if _f5tts_model is not None:
        print("🗑️  Unloading F5TTS model...")
        del _f5tts_model
        _f5tts_model = None
        _f5tts_load_time = None
        _f5tts_memory_usage = None

    # Unload XTTS model
    if _xtts_model is not None:
        print("🗑️  Unloading XTTS model...")
        del _xtts_model
        _xtts_model = None
        _xtts_load_time = None
        _xtts_memory_usage = None

    # Clean up persistent Zonos workers
    try:
        _cleanup_zonos_workers()
    except Exception as e:
        print(f"Warning: Error cleaning up Zonos workers: {e}")

    # Clear GPU cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ CUDA cache cleared and synchronized")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                print("✓ MPS cache cleared")
            except:
                pass  # MPS cache clearing might not be available
    except ImportError:
        pass  # torch not available

    # Force garbage collection
    import gc
    gc.collect()

    print("✓ All TTS models unloaded successfully")


def get_loaded_models() -> Dict[str, bool]:
    """
    Check which models are currently loaded in memory with performance info.

    Returns:
        Dictionary with model names as keys and loaded status/info as values.
    """
    models_info = {
        "f5tts": _f5tts_model is not None,
        "xtts": _xtts_model is not None
    }

    # Add performance information if available
    if _model_load_times or _model_memory_usage:
        models_info["performance"] = {
            "load_times": _model_load_times.copy(),
            "memory_usage": _model_memory_usage.copy()
        }

    return models_info


def get_device_performance_info() -> Dict[str, Any]:
    """
    Get comprehensive device and performance information.

    Returns:
        Dictionary with device and optimization information
    """
    device_type, device_info = _get_device_optimization()

    performance_info = {
        "device_type": device_type.value if device_type else "unknown",
        "device_info": device_info.copy() if device_info else {},
        "optimization_features": {
            "mps_available": _mps_available,
            "apple_neural_engine": _apple_neural_engine_available,
            "mixed_precision_enabled": _mixed_precision_enabled,
            "cuda_memory_fraction": _cuda_memory_fraction,
        },
        "loaded_models": get_loaded_models(),
        "torch_info": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "compile_available": hasattr(torch, 'compile'),
        }
    }

    # Add CUDA-specific info
    if torch.cuda.is_available():
        performance_info["cuda_info"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3,
        }

    return performance_info


def is_model_loaded(model_name: str) -> bool:
    """
    Check if a specific TTS model is currently loaded.

    Args:
        model_name: Name of the model to check ("f5tts", "xtts", "zonos")

    Returns:
        True if the model is loaded, False otherwise
    """
    model_lower = model_name.lower()

    if model_lower == "f5tts":
        return _f5tts_model is not None
    elif model_lower == "xtts":
        return _xtts_model is not None
    else:
        return False


def get_loaded_model_names() -> List[str]:
    """
    Get list of currently loaded TTS model names.

    Returns:
        List of loaded model names
    """
    loaded = []
    if _f5tts_model is not None:
        loaded.append("f5tts")
    if _xtts_model is not None:
        loaded.append("xtts")
    return loaded


def generate_realtime_audio_base64(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate audio with maximum speed optimizations for real-time chat applications.
    This function is optimized specifically for M4 Max and similar high-end hardware.

    Args:
        model: Model name ("f5tts", "xtts", "zonos")
        ref_audio: Path to reference audio file
        ref_text: Reference text
        gen_text: Text to generate
        config: Additional configuration

    Returns:
        Base64 encoded audio data
    """
    print("🚀 Real-time audio generation mode activated")
    print(f"🎵 Real-time TTS Generation Request:")
    print(f"   • Model: {model}")
    print(f"   • Reference Audio: {ref_audio}")
    print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
    print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
    print(f"   • Input Config: {config}")
    
    # Aggressive real-time configuration
    realtime_config = {
        'fast_mode': True,
        'remove_silence': False,  # Skip for speed
        'seed': 42,  # Fixed seed for consistency
        'torch_force_no_weights_only_load': True,
    }
    
    if config:
        realtime_config.update(config)
    
    # Force enable all real-time optimizations
    realtime_config.update({
        'fast_mode': True,
        'tts_streaming': True,
        'tts_fast_preprocessing': True,
    })
    
    print(f"🚀 Real-time Optimized Config:")
    for key, value in realtime_config.items():
        if key in ['ref_text', 'gen_text']:
            # Truncate long text for readability
            display_value = value[:100] + '...' if len(str(value)) > 100 else value
            print(f"   • {key}: {display_value}")
        else:
            print(f"   • {key}: {value}")
    
    return generate_cloned_audio_base64(
        model=model,
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        config=realtime_config,
        fast_mode=True
    )


def _get_conda_base_path():
    """
    Get the conda base installation path.
    Uses the same logic as the app.py setup function.
    """
    conda_base = None
    
    # First try to get conda base from CONDA_EXE environment variable
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe and os.path.exists(conda_exe):
        # Extract base path from conda executable
        conda_base = os.path.dirname(os.path.dirname(conda_exe))
    
    # If that doesn't work, try to find conda in common locations
    if not conda_base:
        possible_conda_paths = [
            '/opt/miniconda3',
            '/opt/anaconda3',
            '/usr/local/miniconda3',
            '/usr/local/anaconda3',
            os.path.expanduser('~/miniconda3'),
            os.path.expanduser('~/anaconda3')
        ]
        
        for path in possible_conda_paths:
            if os.path.exists(os.path.join(path, 'bin', 'conda')):
                conda_base = path
                break
    
    # Last resort: try to get from current CONDA_PREFIX by going up directories
    if not conda_base:
        current_prefix = os.environ.get('CONDA_PREFIX')
        if current_prefix:
            # Check if we're in a conda environment (has /envs/ in path)
            if '/envs/' in current_prefix:
                # Extract base conda path (everything before /envs/)
                potential_base = current_prefix.split('/envs/')[0]
                if os.path.exists(os.path.join(potential_base, 'bin', 'conda')):
                    conda_base = potential_base
            else:
                # We might be in the base environment
                if os.path.exists(os.path.join(current_prefix, 'bin', 'conda')):
                    conda_base = current_prefix
    
    return conda_base


def _get_zonos_env_python():
    """Get the path to the Zonos conda environment Python executable."""
    conda_base = _get_conda_base_path()
    if not conda_base:
        raise RuntimeError("Could not find conda installation")
    
    conda_env_name = "tts_zonos"
    zonos_env_python = os.path.join(conda_base, 'envs', conda_env_name, 'bin', 'python')
    
    if not os.path.exists(zonos_env_python):
        raise RuntimeError(f"Zonos conda environment not found at {zonos_env_python}")
    
    return zonos_env_python


def _init_zonos_worker_lock():
    """Initialize the lock for Zonos worker management."""
    global _zonos_worker_lock
    if _zonos_worker_lock is None:
        import threading
        _zonos_worker_lock = threading.Lock()


def _get_zonos_worker_key(model: str, device: str = "auto") -> str:
    """Generate a key for Zonos worker identification."""
    return f"zonos_{model}_{device}"


def _start_persistent_zonos_worker(model: str, device: str = "auto") -> subprocess.Popen:
    """Start a persistent Zonos worker process."""
    import subprocess
    
    # Get conda environment path
    zonos_env_python = _get_zonos_env_python()
    
    # Path to worker script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(current_dir, '..', '..', 'zonos_worker.py')
    
    print(f"🚀 Starting persistent Zonos worker for model: {model}")
    print(f"   • Python: {zonos_env_python}")
    print(f"   • Script: {worker_script}")
    print(f"   • Device: {device}")
    
    # Verify script exists
    if not os.path.exists(worker_script):
        raise RuntimeError(f"Zonos worker script not found: {worker_script}")
    
    # Start worker in persistent mode
    worker = subprocess.Popen([
        zonos_env_python, worker_script, 
        '--persistent', '--device', device
    ], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1  # Line buffered
    )
    
    print(f"🔄 Waiting for READY signal from worker (PID: {worker.pid})...")
    
    # Wait for READY signal with timeout
    import select
    import time
    
    start_time = time.time()
    timeout = 30  # 30 seconds for startup
    
    while True:
        # Check if worker is still alive
        if worker.poll() is not None:
            stderr_output = worker.stderr.read()
            stdout_output = worker.stdout.read()
            raise RuntimeError(f"Zonos worker died during startup (exit code: {worker.returncode}).\nSTDOUT: {stdout_output}\nSTDERR: {stderr_output}")
        
        # Check for timeout
        if time.time() - start_time > timeout:
            worker.terminate()
            raise RuntimeError(f"Zonos worker startup timeout after {timeout} seconds")
        
        # Try to read READY signal
        try:
            if hasattr(select, 'select'):
                # Unix/Linux/macOS
                ready, _, _ = select.select([worker.stdout], [], [], 1.0)  # 1 second timeout
                if ready:
                    ready_line = worker.stdout.readline().strip()
                    if ready_line:
                        break
            else:
                # Windows fallback
                ready_line = worker.stdout.readline().strip()
                if ready_line:
                    break
                time.sleep(0.1)
        except Exception as read_error:
            print(f"⚠️ Error reading startup signal: {read_error}")
            time.sleep(0.1)
    
    if ready_line != "READY":
        stderr_output = worker.stderr.read()
        worker.terminate()
        raise RuntimeError(f"Zonos worker failed to start. Expected 'READY', got: '{ready_line}'. Error: {stderr_output}")
    
    print(f"✓ Persistent Zonos worker started for model: {model} (PID: {worker.pid})")
    return worker


def _get_or_create_zonos_worker(model: str, device: str = "auto") -> subprocess.Popen:
    """Get existing or create new persistent Zonos worker."""
    _init_zonos_worker_lock()
    
    with _zonos_worker_lock:
        worker_key = _get_zonos_worker_key(model, device)
        
        # Check if worker exists and is still alive
        if worker_key in _zonos_workers:
            worker = _zonos_workers[worker_key]
            if worker.poll() is None:  # Process is still running
                return worker
            else:
                # Worker died, remove it
                print(f"⚠️ Zonos worker {worker_key} died, removing from cache")
                del _zonos_workers[worker_key]
        
        # Create new worker
        worker = _start_persistent_zonos_worker(model, device)
        _zonos_workers[worker_key] = worker
        return worker


def _send_request_to_zonos_worker(worker: subprocess.Popen, request: dict) -> dict:
    """Send a request to a persistent Zonos worker and get response."""
    try:
        print(f"📤 Sending request to Zonos worker...")
        print(f"   • Model: {request.get('model', 'unknown')}")
        print(f"   • Device: {request.get('device', 'unknown')}")
        print(f"   • Output: {request.get('output_file', 'unknown')}")
        
        # Send request
        request_json = json.dumps(request) + '\n'
        worker.stdin.write(request_json)
        worker.stdin.flush()
        print(f"✓ Request sent to worker")
        
        # Read response with timeout
        import select
        import time
        
        start_time = time.time()
        timeout = 300  # 5 minutes
        
        while True:
            # Check if worker is still alive
            if worker.poll() is not None:
                stderr_output = worker.stderr.read()
                raise RuntimeError(f"Zonos worker died while processing request. Error: {stderr_output}")
            
            # Check for timeout
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Zonos worker response timeout after {timeout} seconds")
            
            # Try to read response (non-blocking on Unix systems)
            try:
                if hasattr(select, 'select'):
                    # Unix/Linux/macOS
                    ready, _, _ = select.select([worker.stdout], [], [], 1.0)  # 1 second timeout
                    if ready:
                        response_line = worker.stdout.readline().strip()
                        if response_line:
                            break
                else:
                    # Windows fallback
                    response_line = worker.stdout.readline().strip()
                    if response_line:
                        break
                    time.sleep(0.1)
            except Exception as read_error:
                print(f"⚠️ Error reading from worker: {read_error}")
                time.sleep(0.1)
        
        if not response_line:
            raise RuntimeError("No response from Zonos worker")
        
        print(f"📥 Received response from worker: {response_line[:200]}{'...' if len(response_line) > 200 else ''}")
        
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from Zonos worker: {e}. Response: {response_line[:500]}")
        
        if response.get("success", False):
            print(f"✓ Worker completed successfully")
        else:
            print(f"✗ Worker reported error: {response.get('error', 'unknown')}")
        
        return response
        
    except Exception as e:
        # If communication fails, the worker might be dead
        print(f"❌ Communication with Zonos worker failed: {e}")
        raise RuntimeError(f"Failed to communicate with Zonos worker: {e}")


def _cleanup_zonos_workers():
    """Clean up all persistent Zonos workers."""
    _init_zonos_worker_lock()
    
    with _zonos_worker_lock:
        for worker_key, worker in _zonos_workers.items():
            try:
                if worker.poll() is None:  # Still running
                    worker.stdin.write("EXIT\n")
                    worker.stdin.flush()
                    worker.wait(timeout=5)  # Wait up to 5 seconds
            except:
                worker.terminate()  # Force terminate if needed
            
        _zonos_workers.clear()
        print("✓ All persistent Zonos workers cleaned up")


def preload_zonos_worker(model: str = "Zyphra/Zonos-v0.1-transformer", device: str = "auto") -> bool:
    """
    Preload a persistent Zonos worker for faster subsequent generations.
    
    Args:
        model: Zonos model name to preload
        device: Device to use (auto, cpu, cuda, mps)
    
    Returns:
        bool: True if worker was successfully preloaded, False otherwise
    """
    try:
        print(f"🚀 Preloading Zonos worker for model: {model}")
        worker = _get_or_create_zonos_worker(model, device)
        print(f"✓ Zonos worker preloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to preload Zonos worker: {e}")
        return False


def get_zonos_worker_status() -> Dict[str, Any]:
    """
    Get status of all persistent Zonos workers.
    
    Returns:
        dict: Status information about active workers
    """
    _init_zonos_worker_lock()
    
    with _zonos_worker_lock:
        status = {
            "active_workers": len(_zonos_workers),
            "workers": {}
        }
        
        for worker_key, worker in _zonos_workers.items():
            is_alive = worker.poll() is None
            status["workers"][worker_key] = {
                "alive": is_alive,
                "pid": worker.pid if is_alive else None
            }
        
        return status


def calculate_audio_similarity(reference_audio_path: str, generated_audio_path: str) -> Optional[float]:
    """
    Calculate similarity score between reference audio and generated audio using voice embeddings.
    
    Args:
        reference_audio_path: Path to the reference audio file
        generated_audio_path: Path to the generated audio file
    
    Returns:
        Similarity score between 0 and 1, or None if calculation fails
    """
    try:
        # Import resemblyzer for voice similarity calculation
        from resemblyzer import VoiceEncoder, preprocess_wav
        import numpy as np
        
        print(f"🔍 Calculating audio similarity:")
        print(f"   • Reference: {reference_audio_path}")
        print(f"   • Generated: {generated_audio_path}")
        
        # Verify both files exist
        if not os.path.exists(reference_audio_path):
            print(f"❌ Reference audio file not found: {reference_audio_path}")
            return None
            
        if not os.path.exists(generated_audio_path):
            print(f"❌ Generated audio file not found: {generated_audio_path}")
            return None
        
        # Initialize the voice encoder
        encoder = VoiceEncoder()
        
        # Process the reference audio
        ref_wav = preprocess_wav(reference_audio_path)
        ref_embedding = encoder.embed_utterance(ref_wav)
        
        # Process the generated audio
        gen_wav = preprocess_wav(generated_audio_path)
        gen_embedding = encoder.embed_utterance(gen_wav)
        
        # Calculate cosine similarity between embeddings
        similarity = np.dot(ref_embedding, gen_embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(gen_embedding)
        )
        
        # Ensure the similarity score is within [0, 1] range
        similarity = max(0.0, min(1.0, float(similarity)))
        
        print(f"✓ Audio similarity calculated: {similarity:.4f}")
        return similarity
        
    except ImportError:
        print("⚠️  Resemblyzer not available for audio similarity calculation")
        print("Install with: pip install resemblyzer")
        return None
    except Exception as e:
        print(f"❌ Error calculating audio similarity: {str(e)}")
        return None


def generate_audio_with_similarity(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    config: Optional[Dict[str, Any]] = None,
    auto_download: bool = True,
    fast_mode: bool = False,
    calculate_similarity: bool = True
) -> Dict[str, Any]:
    """
    Generate cloned audio and calculate similarity score with the reference audio.
    
    Args:
        model: Model name/path
        ref_audio: Path to reference audio file
        ref_text: Reference text
        gen_text: Text to generate with cloned voice
        config: Additional configuration parameters
        auto_download: Whether to automatically download model if not available
        fast_mode: Enable real-time optimizations
        calculate_similarity: Whether to calculate similarity score
    
    Returns:
        Dictionary containing:
        - 'audio_path': Path to generated audio file
        - 'similarity_score': Similarity score (0-1) or None if calculation failed
        - 'generation_time': Time taken to generate audio in seconds
    """
    print(f"🎵 TTS Audio Generation with Similarity Analysis:")
    print(f"   • Model: {model}")
    print(f"   • Reference Audio: {ref_audio}")
    print(f"   • Calculate Similarity: {calculate_similarity}")
    
    start_time = time.perf_counter()
    
    # Generate the audio using existing function
    try:
        audio_path = generate_audio(
            model=model,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            config=config,
            auto_download=auto_download,
            fast_mode=fast_mode
        )
        
        generation_time = time.perf_counter() - start_time
        
        # Calculate similarity if requested and audio was generated successfully
        similarity_score = None
        if calculate_similarity and audio_path and os.path.exists(audio_path):
            similarity_start = time.perf_counter()
            similarity_score = calculate_audio_similarity(ref_audio, audio_path)
            similarity_time = time.perf_counter() - similarity_start
            print(f"✓ Similarity calculation completed in {similarity_time:.3f}s")
        
        result = {
            'audio_path': audio_path,
            'similarity_score': similarity_score,
            'generation_time': generation_time
        }
        
        print(f"✓ Audio generation with similarity analysis completed")
        print(f"   • Generation Time: {generation_time:.3f}s")
        if similarity_score is not None:
            print(f"   • Similarity Score: {similarity_score:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Audio generation with similarity failed: {str(e)}")
        return {
            'audio_path': None,
            'similarity_score': None,
            'generation_time': time.perf_counter() - start_time
        }
