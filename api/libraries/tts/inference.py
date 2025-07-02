import os
import sys
import subprocess
import tempfile
import time
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

# Global device info cache
_device_type = None
_device_info = None

# Global model caches with enhanced management
_f5tts_model = None
_xtts_model = None
_model_load_times = {}
_model_memory_usage = {}

# Apple Silicon specific optimizations
_mps_available = False
_apple_neural_engine_available = False

# NVIDIA GPU specific optimizations
_cuda_memory_fraction = 0.8
_mixed_precision_enabled = False

# Model-specific optimization flags
_xtts_mixed_precision_enabled = False  # XTTS has numerical issues with FP16


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
    """Apply Apple Silicon specific optimizations."""
    try:
        # Enable MPS optimizations if available
        if _mps_available and hasattr(model, 'to'):
            model = model.to('mps')
            print("✓ Enabled MPS acceleration for Apple Silicon")

        # Apply memory optimizations
        if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
            torch.backends.mps.set_per_process_memory_fraction(0.8)

        # Enable optimized attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.mps.enabled = True
            print("✓ Enabled optimized attention for Apple Silicon")

        # Set optimal thread counts
        torch.set_num_threads(min(device_info.get('cpu_count', 8), 8))

        return model

    except Exception as e:
        print(f"Warning: Some Apple Silicon optimizations failed: {e}")
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
    auto_download: bool = True
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

    Returns:
        Path to generated audio file
    """
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

    # Get device optimization info
    device_type, device_info = _get_device_optimization()

    # Set default config values with comprehensive device optimization
    default_config = {
        'output_dir': 'outputs',
        'cuda_device': '0',
        'language': 'en',
        'coqui_tos_agreed': True,
        'torch_force_no_weights_only_load': True
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

    # Route to appropriate model implementation
    if model_lower == "f5tts":
        return _generate_f5tts(ref_audio, ref_text, gen_text, str(output_file), config)
    elif model_lower == "xtts":
        return _generate_xtts(ref_audio, ref_text, gen_text, str(output_file), config)
    elif model_lower == "zonos":
        return _generate_zonos("Zyphra/Zonos-v0.1-transformer", ref_audio, ref_text, gen_text, str(output_file), config)
    else:
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {get_supported_models()}")


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
        print(f"Generating F5TTS audio for text: {gen_text[:100]}...")
        print(f"Using reference: {ref_audio}")
        print(f"Reference text: {ref_text[:100]}...")

        # Get the F5TTS model (should be cached in memory)
        f5tts_model = _get_f5tts_model()
        device_type, device_info = _get_device_optimization()

        start_time = time.perf_counter()

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

        # Prepare generation parameters - use only basic F5TTS API parameters
        generation_params = {
            'ref_file': ref_audio,
            'ref_text': ref_text,
            'gen_text': gen_text,
            'file_wave': output_file,
            'seed': config.get('seed', None),
            'remove_silence': config.get('remove_silence', True),
        }

        # Use autocast for mixed precision on NVIDIA GPUs with retry logic for tensor mismatches
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if device_type == DeviceType.NVIDIA_GPU and _mixed_precision_enabled:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        wav, sr, spec = f5tts_model.infer(**generation_params)
                else:
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
        print(f"Generating XTTS audio for text: {gen_text[:100]}...")
        print(f"Using reference: {ref_audio}")

        # Get the XTTS model (should be cached in memory)
        xtts_model = _get_xtts_model()
        device_type, device_info = _get_device_optimization()

        start_time = time.perf_counter()

        # Prepare generation parameters
        generation_params = {
            'text': gen_text,
            'speaker_wav': ref_audio,
            'language': config.get('language', 'en'),
            'file_path': output_file
        }

        # Use autocast for mixed precision on NVIDIA GPUs (but not for XTTS due to numerical instability)
        try:
            if device_type == DeviceType.NVIDIA_GPU and _xtts_mixed_precision_enabled:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    xtts_model.tts_to_file(**generation_params)
            else:
                # Use FP32 for XTTS to avoid numerical instability issues
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
    """Generate audio using Zonos with comprehensive hardware optimization."""
    try:
        # Get device optimization info
        device_type, device_info = _get_device_optimization()

        # Set up environment with comprehensive optimizations
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(config.get('cuda_device', '0')),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1" if config.get('torch_force_no_weights_only_load', True) else "0"
        }

        # Add device-specific environment variables
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                env.update({
                    "TORCH_CUDNN_V8_API_ENABLED": "1",
                    "CUDA_LAUNCH_BLOCKING": "0",
                    "TORCH_CUDNN_BENCHMARK": "1",
                })
                if device_info.get('is_high_end', False):
                    env.update({
                        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                        "TORCH_ALLOW_TF32_MATMUL_OVERRIDE": "1",
                    })
            elif device_type == DeviceType.APPLE_SILICON:
                env.update({
                    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                    "OMP_NUM_THREADS": str(min(device_info.get('cpu_count', 8), 8)),
                })

        # Import Zonos modules
        try:
            from zonos.model import Zonos
            from zonos.conditioning import make_cond_dict
            from zonos.utils import DEFAULT_DEVICE as device
        except ImportError as e:
            raise ImportError(f"Zonos modules not available: {e}")

        print(
            f"Loading Zonos model: {model} with {device_type.value if DEVICE_OPTIMIZATION_AVAILABLE else 'default'} optimization")
        start_time = time.perf_counter()

        # Determine optimal device for Zonos
        zonos_device = device
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                zonos_device = torch.device("cuda:0")
            elif device_type == DeviceType.APPLE_SILICON and _mps_available:
                zonos_device = torch.device("mps")
            else:
                zonos_device = torch.device("cpu")

        # Load Zonos model with device optimization
        zonos_model = Zonos.from_pretrained(model, device=zonos_device)

        # Apply comprehensive model optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                # Apply NVIDIA-specific optimizations
                if _mixed_precision_enabled:
                    try:
                        zonos_model = zonos_model.half()
                        print("✓ Applied mixed precision to Zonos model")
                    except Exception as e:
                        print(
                            f"Warning: Mixed precision failed for Zonos: {e}")

                if device_info.get('tts_compile', False) and hasattr(torch, 'compile'):
                    try:
                        zonos_model = torch.compile(
                            zonos_model, mode="reduce-overhead")
                        print("✓ Applied torch.compile optimization to Zonos model")
                    except Exception as e:
                        print(
                            f"Warning: Could not apply torch.compile to Zonos: {e}")

            elif device_type == DeviceType.APPLE_SILICON:
                # Apply Apple Silicon optimizations
                try:
                    # Optimize for Apple Silicon
                    torch.set_num_threads(
                        min(device_info.get('cpu_count', 8), 8))
                    print("✓ Applied Apple Silicon thread optimization to Zonos")
                except Exception as e:
                    print(
                        f"Warning: Apple Silicon optimization failed for Zonos: {e}")

        # Load reference audio and create speaker embedding
        wav, sampling_rate = torchaudio.load(ref_audio)
        wav = wav.to(zonos_device)

        # Create speaker embedding with device optimization
        if device_type == DeviceType.NVIDIA_GPU and _mixed_precision_enabled:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                speaker = zonos_model.make_speaker_embedding(
                    wav, sampling_rate)
        else:
            speaker = zonos_model.make_speaker_embedding(wav, sampling_rate)

        # Generate speech using Zonos
        cond_dict = make_cond_dict(
            text=gen_text,
            speaker=speaker,
            language=config.get('language', 'en-us')
        )
        conditioning = zonos_model.prepare_conditioning(cond_dict)

        # Generate audio codes and decode with optimization
        with torch.inference_mode():  # Use inference mode for better performance
            if device_type == DeviceType.NVIDIA_GPU and _mixed_precision_enabled:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    codes = zonos_model.generate(conditioning)
                    wavs = zonos_model.autoencoder.decode(codes).cpu()
            else:
                codes = zonos_model.generate(conditioning)
                wavs = zonos_model.autoencoder.decode(codes).cpu()

        # Save generated audio
        torchaudio.save(output_file, wavs[0],
                        zonos_model.autoencoder.sampling_rate)

        runtime = time.perf_counter() - start_time
        print(f"✓ Zonos generation completed in {runtime:.3f}s")
        print(f"✓ Voice cloning successful! Output saved to: {output_file}")

        return output_file

    except RuntimeError as e:
        if "espeak not installed" in str(e):
            error_msg = (
                "Error: espeak-ng is required for Zonos TTS.\n"
                "Please install it using one of these commands:\n"
                "Ubuntu/Debian: sudo apt install espeak-ng\n"
                "MacOS: brew install espeak-ng"
            )
            print(error_msg)
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError(f"Zonos generation failed: {str(e)}")

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
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate cloned audio and return as base64 encoded string.

    Args:
        model: Model name ("f5tts", "xtts", "zonos")
        ref_audio: Path to reference audio file
        ref_text: Reference text
        gen_text: Text to generate
        config: Additional configuration

    Returns:
        Base64 encoded audio data
    """
    try:
        # Generate audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_output_path = temp_file.name

        try:
            # Generate audio using the standard API
            output_file = generate_audio(
                model=model,
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                config=config
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
    """
    Unload all TTS models from memory with comprehensive cleanup.
    """
    global _f5tts_model, _xtts_model, _model_load_times, _model_memory_usage

    print("🧹 Unloading TTS models from memory...")

    if _f5tts_model is not None:
        del _f5tts_model
        _f5tts_model = None
        print("✓ F5TTS model unloaded")

    if _xtts_model is not None:
        del _xtts_model
        _xtts_model = None
        print("✓ XTTS model unloaded")

    # Clear performance metrics
    _model_load_times.clear()
    _model_memory_usage.clear()

    # Comprehensive memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ CUDA cache cleared and synchronized")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Clear MPS cache if available
        try:
            torch.mps.empty_cache()
            print("✓ MPS cache cleared")
        except:
            pass

    # Force garbage collection
    import gc
    gc.collect()
    print("✓ Python garbage collection completed")

    print("🎯 Model unloading completed!")


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
