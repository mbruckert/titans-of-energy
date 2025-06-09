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

# Global model caches
_f5tts_model = None
_xtts_model = None


def _get_device_optimization():
    """Get cached device optimization info."""
    global _device_type, _device_info
    if DEVICE_OPTIMIZATION_AVAILABLE and (_device_type is None or _device_info is None):
        _device_type, _device_info = get_device_info()
        print_device_info(_device_type, _device_info)
    return _device_type, _device_info


def _get_f5tts_model(force_init: bool = False):
    """Get or initialize F5TTS model with device optimization."""
    global _f5tts_model
    if _f5tts_model is None or force_init:
        try:
            from f5_tts.api import F5TTS

            # Get device optimization info
            device_type, device_info = _get_device_optimization()

            # Determine device
            if DEVICE_OPTIMIZATION_AVAILABLE:
                if device_type == DeviceType.NVIDIA_GPU:
                    device_str = "cuda"
                elif device_type == DeviceType.APPLE_SILICON and device_info.get('torch_device') == 'mps':
                    device_str = "mps"
                else:
                    device_str = "cpu"
            else:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Initializing F5TTS model on device: {device_str}")
            _f5tts_model = F5TTS(device=device_str)
            print("F5TTS model initialized successfully!")

        except ImportError as e:
            print(f"Error importing F5TTS: {e}")
            raise ImportError(
                "F5TTS not available. Please install with: pip install f5-tts")
        except Exception as e:
            print(f"Error initializing F5TTS model: {e}")
            raise

    return _f5tts_model


def _get_xtts_model(force_init: bool = False):
    """Get or initialize XTTS model with device optimization."""
    global _xtts_model
    if _xtts_model is None or force_init:
        try:
            from TTS.api import TTS

            # Get device optimization info
            device_type, device_info = _get_device_optimization()

            # Determine device
            if DEVICE_OPTIMIZATION_AVAILABLE:
                if device_type == DeviceType.NVIDIA_GPU:
                    device_str = "cuda"
                elif device_type == DeviceType.APPLE_SILICON and device_info.get('torch_device') == 'mps':
                    device_str = "mps"
                else:
                    device_str = "cpu"
            else:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Initializing XTTS model on device: {device_str}")

            # Set PyTorch to use weights_only=False for XTTS (trusted model)
            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)
            torch.load = patched_load

            try:
                _xtts_model = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2").to(device_str)
                print("XTTS model initialized successfully!")
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
    Generate cloned audio using specified TTS model with hardware optimization.

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

    # Set default config values with device optimization
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

        # Add device-specific TTS optimizations
        if device_type == DeviceType.NVIDIA_GPU:
            default_config.update({
                'batch_size': device_info.get('tts_batch_size', 4),
                'use_mixed_precision': device_info.get('mixed_precision', True),
                'enable_flash_attention': device_info.get('llm_use_flash_attention', True),
                'compile_model': device_info.get('tts_compile', False),
            })
        elif device_type == DeviceType.APPLE_SILICON:
            default_config.update({
                'batch_size': device_info.get('tts_batch_size', 2),
                'use_mps': device_info.get('torch_device') == 'mps',
                'mps_fallback': True,
            })

    if config:
        default_config.update(config)
    config = default_config

    # Auto-download model if requested
    if auto_download:
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
    model_lower = model.lower()
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
    Ensure that the specified model is available for inference.
    Downloads it if necessary.

    Args:
        model: Model name to check/download
        cache_dir: Directory to cache models

    Returns:
        True if model is available, False otherwise
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
    """Generate audio using F5-TTS API with hardware optimization."""
    try:
        print(f"Generating F5TTS audio for text: {gen_text[:100]}...")
        print(f"Using reference: {ref_audio}")
        print(f"Reference text: {ref_text[:100]}...")

        # Get the F5TTS model
        f5tts_model = _get_f5tts_model()

        start_time = time.perf_counter()

        # Generate audio using F5TTS API
        wav, sr, spec = f5tts_model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=output_file,
            seed=config.get('seed', None),
            remove_silence=config.get('remove_silence', True),
        )

        runtime = time.perf_counter() - start_time
        print(f"F5-TTS generation completed in {runtime:.3f} seconds")

        # Validate the output file exists
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"F5-TTS did not generate expected output file: {output_file}")

        print(f"F5-TTS output saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"F5-TTS generation failed: {str(e)}")
        raise RuntimeError(f"F5-TTS generation failed: {str(e)}")


def _generate_xtts(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using XTTS-v2 API with hardware optimization."""
    try:
        print(f"Generating XTTS audio for text: {gen_text[:100]}...")
        print(f"Using reference: {ref_audio}")

        # Get the XTTS model
        xtts_model = _get_xtts_model()

        start_time = time.perf_counter()

        # Generate audio using XTTS API
        xtts_model.tts_to_file(
            text=gen_text,
            speaker_wav=ref_audio,
            language=config.get('language', 'en'),
            file_path=output_file
        )

        runtime = time.perf_counter() - start_time
        print(f"XTTS-v2 generation completed in {runtime:.3f} seconds")

        # Validate the output file exists
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"XTTS-v2 did not generate expected output file: {output_file}")

        print(f"XTTS-v2 output saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"XTTS-v2 generation failed: {str(e)}")
        raise RuntimeError(f"XTTS-v2 generation failed: {str(e)}")


def _generate_zonos(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using Zonos with hardware optimization."""
    try:
        # Get device optimization info
        device_type, device_info = _get_device_optimization()

        # Set up environment with optimizations
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
                })
                if device_info.get('is_high_end', False):
                    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
            elif device_type == DeviceType.APPLE_SILICON:
                env.update({
                    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                })

        # Import Zonos modules (these need to be available in the environment)
        try:
            from zonos.model import Zonos
            from zonos.conditioning import make_cond_dict
            from zonos.utils import DEFAULT_DEVICE as device
        except ImportError as e:
            raise ImportError(f"Zonos modules not available: {e}")

        print(
            f"Loading Zonos model: {model} with {device_type.value if DEVICE_OPTIMIZATION_AVAILABLE else 'default'} optimization")
        start_time = time.perf_counter()

        # Determine device for Zonos
        zonos_device = device
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                zonos_device = torch.device("cuda:0")
            elif device_type == DeviceType.APPLE_SILICON and device_info.get('torch_device') == 'mps':
                zonos_device = torch.device("mps")
            else:
                zonos_device = torch.device("cpu")

        # Load Zonos model with device optimization
        zonos_model = Zonos.from_pretrained(model, device=zonos_device)

        # Apply model optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE and device_type == DeviceType.NVIDIA_GPU:
            if device_info.get('tts_compile', False):
                try:
                    zonos_model = torch.compile(
                        zonos_model, mode="reduce-overhead")
                    print("Applied torch.compile optimization to Zonos model")
                except Exception as e:
                    print(f"Warning: Could not apply torch.compile: {e}")

        # Load reference audio and create speaker embedding
        wav, sampling_rate = torchaudio.load(ref_audio)
        wav = wav.to(zonos_device)
        speaker = zonos_model.make_speaker_embedding(wav, sampling_rate)

        # Generate speech using Zonos
        cond_dict = make_cond_dict(
            text=gen_text,
            speaker=speaker,
            language=config.get('language', 'en-us')
        )
        conditioning = zonos_model.prepare_conditioning(cond_dict)

        # Generate audio codes and decode
        with torch.inference_mode():  # Use inference mode for better performance
            codes = zonos_model.generate(conditioning)
            wavs = zonos_model.autoencoder.decode(codes).cpu()

        # Save generated audio
        torchaudio.save(output_file, wavs[0],
                        zonos_model.autoencoder.sampling_rate)

        runtime = time.perf_counter() - start_time
        print(f"Zonos generation completed in {runtime:.3f} seconds")
        print(f"Voice cloning successful! Output saved to: {output_file}")

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


def preload_models(models: List[str] = None, force_reload: bool = False):
    """
    Preload specified TTS models to improve inference speed.
    Call this from your /load-character endpoint to initialize models in memory.

    Args:
        models: List of model names to preload. If None, preloads all available models.
        force_reload: Whether to force reload models even if already loaded.
    """
    if models is None:
        models = ["f5tts", "xtts"]

    print(f"Preloading TTS models: {models}")

    for model in models:
        model_lower = model.lower()
        try:
            if model_lower == "f5tts":
                print("Preloading F5TTS model...")
                _get_f5tts_model(force_init=force_reload)
                print("F5TTS model preloaded successfully!")
            elif model_lower == "xtts":
                print("Preloading XTTS model...")
                _get_xtts_model(force_init=force_reload)
                print("XTTS model preloaded successfully!")
        except Exception as e:
            print(f"Failed to preload {model}: {e}")

    print("Model preloading completed!")


def unload_models():
    """
    Unload all TTS models from memory to free up resources.
    Call this when switching characters or when models are no longer needed.
    """
    global _f5tts_model, _xtts_model

    print("Unloading TTS models from memory...")

    if _f5tts_model is not None:
        del _f5tts_model
        _f5tts_model = None
        print("F5TTS model unloaded")

    if _xtts_model is not None:
        del _xtts_model
        _xtts_model = None
        print("XTTS model unloaded")

    # Force garbage collection to free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")

    print("Model unloading completed!")


def get_loaded_models() -> Dict[str, bool]:
    """
    Check which models are currently loaded in memory.

    Returns:
        Dictionary with model names as keys and loaded status as values.
    """
    return {
        "f5tts": _f5tts_model is not None,
        "xtts": _xtts_model is not None
    }


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


def preload_models_smart(models: List[str] = None, force_reload: bool = False):
    """
    Intelligently preload TTS models, avoiding reloading if already loaded.

    Args:
        models: List of model names to preload. If None, preloads all available models.
        force_reload: Whether to force reload models even if already loaded.
    """
    if models is None:
        models = ["f5tts", "xtts"]

    print(f"Smart preloading TTS models: {models}")

    for model in models:
        model_lower = model.lower()
        try:
            # Check if model is already loaded
            if not force_reload and is_model_loaded(model_lower):
                print(f"TTS model {model_lower} already loaded, skipping...")
                continue

            if model_lower == "f5tts":
                print("Preloading F5TTS model...")
                _get_f5tts_model(force_init=force_reload)
                print("F5TTS model preloaded successfully!")
            elif model_lower == "xtts":
                print("Preloading XTTS model...")
                _get_xtts_model(force_init=force_reload)
                print("XTTS model preloaded successfully!")
        except Exception as e:
            print(f"Failed to preload {model}: {e}")

    print("Smart model preloading completed!")
