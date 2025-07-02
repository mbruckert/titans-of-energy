import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import librosa
import soundfile as sf
import resampy
import torch
from df.enhance import enhance, init_df
import time

# Import device optimization utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from device_optimization import get_device_info, DeviceType
    DEVICE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Device optimization not available for audio preprocessing.")
    DEVICE_OPTIMIZATION_AVAILABLE = False

# Global device info cache
_device_type = None
_device_info = None


def _get_device_optimization():
    """Get cached device optimization info."""
    global _device_type, _device_info
    if DEVICE_OPTIMIZATION_AVAILABLE and (_device_type is None or _device_info is None):
        _device_type, _device_info = get_device_info()
    return _device_type, _device_info


def download_voice_models(
    models: List[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False
) -> Dict[str, bool]:
    """
    Download voice cloning models if they aren't already available.

    Args:
        models: List of model names to download. If None, downloads all supported models.
        cache_dir: Directory to cache models (uses default if None)
        force_download: Whether to force re-download even if models exist

    Returns:
        Dictionary with model names as keys and success status as values
    """
    if models is None:
        models = ["f5tts", "xtts", "zonos"]

    results = {}

    for model in models:
        try:
            model_lower = model.lower()
            if model_lower == "f5tts" or model.upper() == "F5TTS":
                results[model] = _download_f5tts(cache_dir, force_download)
            elif model_lower == "xtts" or model.upper() == "XTTS" or "xtts" in model.lower():
                # Map XTTS to full model name for processing
                full_model_name = "tts_models/multilingual/multi-dataset/xtts_v2" if model_lower == "xtts" or model.upper() == "XTTS" else model
                results[model] = _download_xtts(
                    full_model_name, cache_dir, force_download)
            elif model_lower == "zonos" or "zonos" in model.lower():
                # Use full model name for Zonos
                full_model_name = "Zyphra/Zonos-v0.1-transformer" if model_lower == "zonos" else model
                results[model] = _download_zonos(
                    full_model_name, cache_dir, force_download)
            else:
                print(f"Warning: Unknown model type: {model}")
                results[model] = False
        except Exception as e:
            print(f"Error downloading {model}: {str(e)}")
            results[model] = False

    return results


def _download_f5tts(cache_dir: Optional[str] = None, force_download: bool = False) -> bool:
    """Download F5-TTS model and dependencies."""
    try:
        # Check if F5-TTS is already installed and working
        if not force_download:
            try:
                # Create clean environment for subprocess calls
                clean_env = os.environ.copy()
                # Fix F5-TTS installation issue
                clean_env['PYTHONHASHSEED'] = 'random'

                # First try a quick import check
                result = subprocess.run(
                    [sys.executable, "-c",
                        "import f5_tts; print('F5-TTS importable')"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=clean_env
                )
                if result.returncode == 0:
                    print("F5-TTS Python package is available")
                    # Now try the CLI with a longer timeout since it may need to initialize
                    try:
                        cli_result = subprocess.run(
                            ["f5-tts_infer-cli", "--help"],
                            capture_output=True,
                            text=True,
                            timeout=30,  # Increased timeout for CLI initialization
                            env=clean_env
                        )
                        if cli_result.returncode == 0:
                            print("F5-TTS is already installed and accessible")
                            return True
                        else:
                            print(
                                "F5-TTS CLI not responding properly, will reinstall")
                    except subprocess.TimeoutExpired:
                        print(
                            "F5-TTS CLI timed out during verification, will reinstall")
                else:
                    print("F5-TTS not found or not responding, will install")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("F5-TTS not found or not responding, will install")
            except Exception as e:
                print(f"F5-TTS check failed: {e}, will install")

        print("Installing F5-TTS...")

        # Install F5-TTS via pip
        cmd = [sys.executable, "-m", "pip", "install", "f5-tts"]
        if force_download:
            cmd.append("--force-reinstall")

        # Create clean environment for installation
        clean_env = os.environ.copy()
        clean_env['PYTHONHASHSEED'] = 'random'  # Fix F5-TTS installation issue

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            env=clean_env
        )

        if result.returncode == 0:
            print("F5-TTS installed successfully")

            # Verify installation with more lenient approach
            # First check if the package can be imported
            import_result = subprocess.run(
                [sys.executable, "-c",
                    "import f5_tts; print('F5-TTS import successful')"],
                capture_output=True,
                text=True,
                timeout=10,
                env=clean_env
            )

            if import_result.returncode == 0:
                print("F5-TTS Python package verification successful")

                # Try CLI verification with longer timeout
                try:
                    verify_result = subprocess.run(
                        ["f5-tts_infer-cli", "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,  # Increased timeout for CLI
                        env=clean_env
                    )

                    if verify_result.returncode == 0:
                        print("F5-TTS CLI verification successful")
                        return True
                    else:
                        print(
                            "F5-TTS CLI verification failed, but package is importable")
                        # Consider this a partial success - the package is installed
                        return True
                except subprocess.TimeoutExpired:
                    print(
                        "F5-TTS CLI verification timed out, but package is importable")
                    # The CLI might be slow to initialize, but if the package imports, it's likely working
                    return True
            else:
                print("F5-TTS package import failed")
                return False
        else:
            print(f"F5-TTS installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error installing F5-TTS: {str(e)}")
        return False


def _download_xtts(model: str, cache_dir: Optional[str] = None, force_download: bool = False) -> bool:
    """Download XTTS-v2 model."""
    try:
        # Check if TTS is installed
        try:
            import TTS
            from TTS.api import TTS as TTS_API
        except ImportError:
            print("Installing Coqui TTS...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "TTS"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                print(f"TTS installation failed: {result.stderr}")
                return False

            # Try importing again
            import TTS
            from TTS.api import TTS as TTS_API

        # Set PyTorch to use weights_only=False for XTTS (trusted model)
        import torch
        original_load = torch.load

        def patched_load(*args, **kwargs):
            # Set weights_only=False for XTTS model loading
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = patched_load

        # Map shortened model name to full name if needed
        if model.upper() == "XTTS":
            full_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        elif "xtts" in model.lower() and "/" in model:
            full_model_name = model  # Already a full model name
        else:
            full_model_name = model

        # Initialize TTS to trigger model download
        print(f"Downloading/verifying XTTS-v2 model: {full_model_name}")

        try:
            # This will download the model if it doesn't exist
            tts = TTS_API(model_name=full_model_name, progress_bar=True)
            print("XTTS-v2 model ready")
            return True

        except Exception as e:
            print(f"Error initializing XTTS-v2: {str(e)}")
            return False
        finally:
            # Restore original torch.load
            torch.load = original_load

    except Exception as e:
        print(f"Error setting up XTTS-v2: {str(e)}")
        return False


def _download_zonos(model: str, cache_dir: Optional[str] = None, force_download: bool = False) -> bool:
    """Download Zonos model."""
    try:
        # Try to import zonos
        try:
            from zonos.model import Zonos
        except ImportError:
            print("Zonos not found. Attempting to install...")

            # Try installing zonos (this might not work if it's not on PyPI)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "zonos"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                print(
                    "Zonos installation via pip failed. You may need to install it manually.")
                print(
                    "Please check the Zonos documentation for installation instructions.")
                return False

            # Try importing again
            try:
                from zonos.model import Zonos
            except ImportError:
                print("Failed to import Zonos after installation")
                return False

        # Check if espeak-ng is available (required for Zonos)
        try:
            subprocess.run(["espeak-ng", "--help"],
                           capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(
                "Warning: espeak-ng not found. Zonos requires espeak-ng to be installed.")
            print(
                "Install with: sudo apt install espeak-ng (Ubuntu/Debian) or brew install espeak-ng (macOS)")
            return False

        # Try to load the model to verify it's accessible
        print(f"Downloading/verifying Zonos model: {model}")

        try:
            # Set a reasonable cache directory
            if cache_dir:
                os.environ['HF_HOME'] = cache_dir

            # This will download the model if it doesn't exist
            zonos_model = Zonos.from_pretrained(model)
            print("Zonos model ready")
            return True

        except Exception as e:
            print(f"Error loading Zonos model: {str(e)}")
            return False

    except Exception as e:
        print(f"Error setting up Zonos: {str(e)}")
        return False


def check_voice_model_availability() -> Dict[str, Dict[str, Any]]:
    """
    Check availability and status of voice cloning models.

    Returns:
        Dictionary with model status information
    """
    models = {
        "F5TTS": {"available": False, "path": None, "dependencies": []},
        "XTTS-v2": {"available": False, "path": None, "dependencies": []},
        "Zonos": {"available": False, "path": None, "dependencies": []}
    }

    # Check F5-TTS
    try:
        result = subprocess.run(
            ["f5-tts_infer-cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        models["F5TTS"]["available"] = result.returncode == 0
        if models["F5TTS"]["available"]:
            models["F5TTS"]["dependencies"].append("f5-tts CLI")
    except:
        pass

    # Check XTTS-v2
    try:
        import TTS
        models["XTTS-v2"]["available"] = True
        models["XTTS-v2"]["dependencies"].append("TTS")

        # Try to get model path
        try:
            from TTS.api import TTS as TTS_API
            tts = TTS_API(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            models["XTTS-v2"]["path"] = "Downloaded via TTS API"
        except:
            models["XTTS-v2"]["available"] = False
    except ImportError:
        pass

    # Check Zonos
    try:
        from zonos.model import Zonos
        models["Zonos"]["available"] = True
        models["Zonos"]["dependencies"].append("zonos")

        # Check espeak-ng
        try:
            subprocess.run(["espeak-ng", "--help"],
                           capture_output=True, check=True)
            models["Zonos"]["dependencies"].append("espeak-ng")
        except:
            models["Zonos"]["dependencies"].append("espeak-ng (MISSING)")
            models["Zonos"]["available"] = False
    except ImportError:
        pass

    return models


def check_single_model_availability(model: str) -> bool:
    """
    Check availability of a single model without triggering initialization of other models.

    Args:
        model: Model name to check ("f5tts", "xtts", "zonos")

    Returns:
        True if model is available, False otherwise
    """
    model_lower = model.lower()

    if model_lower == "f5tts":
        try:
            # First try importing the package (faster and more reliable)
            import_result = subprocess.run(
                [sys.executable, "-c",
                    "import f5_tts; print('F5-TTS available')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if import_result.returncode == 0:
                return True

            # If import fails, try CLI as fallback with shorter timeout
            cli_result = subprocess.run(
                ["f5-tts_infer-cli", "--help"],
                capture_output=True,
                text=True,
                timeout=15  # Reasonable timeout for quick check
            )
            return cli_result.returncode == 0
        except:
            return False

    elif model_lower == "xtts":
        try:
            import TTS
            return True
        except ImportError:
            return False

    elif model_lower == "zonos":
        try:
            from zonos.model import Zonos
            return True
        except ImportError:
            return False
    else:
        return False


def generate_reference_audio(
    audio_file: str,
    output_file: Optional[str] = None,
    clean_audio: bool = True,
    remove_silence: bool = True,
    enhance_audio: bool = True,
    skip_all_processing: bool = False,
    preprocessing_order: List[str] = ['clean', 'remove_silence', 'enhance'],
    top_db: float = 40.0,
    fade_length_ms: int = 50,
    bass_boost: bool = True,
    treble_boost: bool = True,
    compression: bool = True
) -> str:
    """
    Clean, enhance, and process reference audio for voice cloning.

    Args:
        audio_file: Path to input audio file
        output_file: Path for output file (if None, uses temp file)
        clean_audio: Whether to apply noise reduction
        remove_silence: Whether to remove silence segments
        enhance_audio: Whether to enhance audio quality
        skip_all_processing: If True, skips all processing and just copies the file
        preprocessing_order: Order of processing steps
        top_db: Silence threshold in dB for silence removal
        fade_length_ms: Fade length in milliseconds for smooth transitions
        bass_boost: Whether to apply bass boost
        treble_boost: Whether to apply treble boost
        compression: Whether to apply dynamic range compression

    Returns:
        Path to processed audio file

    Note:
        To disable all audio processing, set skip_all_processing=True or 
        set all of clean_audio, remove_silence, and enhance_audio to False.
        Alternatively, use preprocess_audio=False in voice_cloning_settings 
        when creating a character to bypass this function entirely.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Input audio file not found: {audio_file}")

    # Create output file path if not provided
    if output_file is None:
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(
            temp_dir, f"processed_{Path(audio_file).name}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # If skip_all_processing is True, just copy the file
    if skip_all_processing:
        if audio_file != output_file:
            sf.copy_audio(audio_file, output_file)
        return output_file

    # If all processing options are disabled, just copy the file
    if not any([clean_audio, remove_silence, enhance_audio]):
        if audio_file != output_file:
            sf.copy_audio(audio_file, output_file)
        return output_file

    current_file = audio_file
    temp_files = []

    try:
        # Process according to specified order
        for step in preprocessing_order:
            if step == 'clean' and clean_audio:
                current_file = _clean_audio(current_file, temp_files)
            elif step == 'remove_silence' and remove_silence:
                current_file = _remove_silence(
                    current_file, temp_files, top_db, fade_length_ms)
            elif step == 'enhance' and enhance_audio:
                current_file = _enhance_audio(
                    current_file, temp_files, bass_boost, treble_boost, compression)

        # Copy final result to output file
        if current_file != output_file:
            sf.copy_audio(current_file, output_file)

        return output_file

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file) and temp_file != output_file:
                    os.remove(temp_file)
            except:
                pass


def _clean_audio(input_file: str, temp_files: List[str]) -> str:
    """Clean audio using DeepFilterNet for noise reduction with comprehensive hardware optimization."""
    temp_file = tempfile.mktemp(suffix='.wav')
    temp_files.append(temp_file)

    try:
        # Get device optimization info
        device_type, device_info = _get_device_optimization()

        # Initialize the DeepFilterNet model with device optimization
        model, state, _ = init_df()

        # Apply comprehensive device-specific optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                # NVIDIA GPU optimizations
                try:
                    device = torch.device("cuda:0")
                    model = model.to(device)

                    # Enable mixed precision if supported
                    if device_info.get('mixed_precision', True):
                        try:
                            model = model.half()
                            print(
                                f"✓ Using mixed precision (FP16) for DeepFilterNet on {device_info.get('device_name', 'GPU')}")
                        except Exception as e:
                            print(
                                f"Warning: Mixed precision failed for DeepFilterNet: {e}")

                    # Apply torch.compile for high-end GPUs (but be cautious with mixed precision)
                    if device_info.get('is_high_end', False) and hasattr(torch, 'compile'):
                        # torch.compile can be problematic with DeepFilterNet + mixed precision
                        if device_info.get('mixed_precision', True):
                            print(
                                "⚠️  Skipping torch.compile for DeepFilterNet with mixed precision (known compatibility issues)")
                        else:
                            try:
                                model = torch.compile(
                                    model, mode="reduce-overhead")
                                print(
                                    "✓ Applied torch.compile optimization to DeepFilterNet")
                            except Exception as e:
                                print(
                                    f"Warning: torch.compile failed for DeepFilterNet: {e}")

                    # Set CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True

                    print(
                        f"✓ Using NVIDIA GPU acceleration for audio cleaning on {device_info.get('device_name', 'GPU')}")

                except Exception as e:
                    print(f"Warning: Could not move DeepFilterNet to GPU: {e}")

            elif device_type == DeviceType.APPLE_SILICON:
                # Apple Silicon optimizations
                try:
                    if torch.backends.mps.is_available():
                        device = torch.device("mps")
                        model = model.to(device)

                        # Apply MPS optimizations
                        if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                            torch.backends.mps.set_per_process_memory_fraction(
                                0.8)

                        # Set optimal thread counts for Apple Silicon
                        optimal_threads = min(
                            device_info.get('cpu_count', 8), 8)
                        torch.set_num_threads(optimal_threads)

                        print(
                            f"✓ Using MPS acceleration for audio cleaning on {device_info.get('device_name', 'Apple Silicon')}")
                        print(f"✓ Optimized thread count: {optimal_threads}")
                    else:
                        # CPU fallback with optimized thread count
                        optimal_threads = min(
                            device_info.get('cpu_count', 8), 8)
                        torch.set_num_threads(optimal_threads)
                        print(
                            f"✓ Using optimized CPU processing for audio cleaning with {optimal_threads} threads")

                except Exception as e:
                    print(
                        f"Warning: Could not optimize DeepFilterNet for Apple Silicon: {e}")
            else:
                # CPU optimizations
                cpu_count = device_info.get('cpu_count', 4)
                optimal_threads = min(cpu_count, 8)
                torch.set_num_threads(optimal_threads)
                print(
                    f"✓ Using CPU processing for audio cleaning with {optimal_threads} threads")

        # Read and process audio
        audio_data, sample_rate = sf.read(input_file, always_2d=True)

        # Determine target precision early
        target_dtype = torch.float32
        if DEVICE_OPTIMIZATION_AVAILABLE and device_type == DeviceType.NVIDIA_GPU:
            mixed_precision_enabled = device_info.get('mixed_precision', True)
            if mixed_precision_enabled and hasattr(model, 'dtype') and model.dtype == torch.float16:
                target_dtype = torch.float16
                print("✓ Target precision: FP16 to match DeepFilterNet model")

        # Resample if necessary
        if sample_rate != state.sr():
            print(f"Resampling audio from {sample_rate}Hz to {state.sr()}Hz")
            audio_data = resampy.resample(audio_data, sample_rate, state.sr())
            sample_rate = state.sr()

        # Convert to target precision and create tensor
        if target_dtype == torch.float16:
            audio_data = audio_data.astype(
                np.float32).T  # Create as float32 first
            audio_tensor = torch.from_numpy(audio_data)
        else:
            audio_data = audio_data.astype(np.float32).T
            audio_tensor = torch.from_numpy(audio_data)

        # Move tensor to same device as model and convert to target precision
        if hasattr(model, 'device'):
            audio_tensor = audio_tensor.to(model.device)

            # Convert to target precision after moving to device
            if target_dtype == torch.float16:
                audio_tensor = audio_tensor.to(dtype=target_dtype)
                print("✓ Audio tensor converted to FP16 after device placement")

        # Enhance audio with device-specific optimization
        # Note: DeepFilterNet can be sensitive to mixed precision with complex numbers
        start_time = time.perf_counter()

        with torch.no_grad():  # Use no_grad for inference to save memory
            try:
                enhanced_audio = enhance(model, state, audio_tensor)
            except RuntimeError as e:
                error_msg = str(e)
                if ("type torch.float16" in error_msg and "torch.float32" in error_msg) or \
                   ("backend='inductor'" in error_msg and ("Half" in error_msg or "Float" in error_msg)):
                    print(
                        f"⚠️  Precision/compilation mismatch detected: {error_msg}")
                    print(
                        "🔄 Attempting to fix by ensuring tensor precision consistency...")

                    # Try to fix precision mismatch
                    if hasattr(model, 'dtype'):
                        audio_tensor = audio_tensor.to(dtype=model.dtype)
                        print(f"✓ Converted audio tensor to {model.dtype}")

                        try:
                            enhanced_audio = enhance(
                                model, state, audio_tensor)
                            print(
                                "✓ Audio enhancement successful after precision fix")
                        except RuntimeError as retry_error:
                            print(
                                f"⚠️  Still failing after precision fix: {retry_error}")
                            print("🔄 Falling back to FP32 processing...")
                            # Fall back to FP32 processing
                            audio_tensor = audio_tensor.to(dtype=torch.float32)
                            # If model was compiled with mixed precision, try to reset it
                            if hasattr(model, 'dtype') and model.dtype == torch.float16:
                                # Convert model back to FP32 for this operation
                                model = model.float()
                                print(
                                    "✓ Converted model to FP32 for fallback processing")
                            enhanced_audio = enhance(
                                model, state, audio_tensor)
                            print(
                                "✓ Audio enhancement successful with FP32 fallback")
                    else:
                        raise e
                else:
                    raise e

        processing_time = time.perf_counter() - start_time
        print(f"✓ Audio cleaning completed in {processing_time:.3f}s")

        # Convert back to numpy array
        enhanced_audio = enhanced_audio.detach().cpu().numpy()
        enhanced_audio = np.squeeze(enhanced_audio.T)

        # Save enhanced audio
        sf.write(temp_file, enhanced_audio, sample_rate)

        return temp_file

    except Exception as e:
        print(f"Warning: Audio cleaning failed: {e}")
        return input_file


def _remove_silence(
    input_file: str,
    temp_files: List[str],
    top_db: float = 40.0,
    fade_length_ms: int = 50
) -> str:
    """Remove silence from audio with smooth transitions."""
    temp_file = tempfile.mktemp(suffix='.wav')
    temp_files.append(temp_file)

    try:
        # Load audio
        signal, sample_rate = librosa.load(input_file, sr=None)

        # Calculate fade length in samples
        fade_samples = int(fade_length_ms * sample_rate / 1000)

        # Find non-silent intervals
        intervals = librosa.effects.split(
            signal,
            top_db=top_db,
            frame_length=4096,
            hop_length=1024
        )

        if len(intervals) == 0:
            return input_file

        # Initialize output array
        processed_signal = np.zeros_like(signal)
        current_pos = 0

        # Process each segment with crossfading
        for i, (start, end) in enumerate(intervals):
            segment = signal[start:end]

            if i == 0:
                # First segment - no crossfading needed
                processed_signal[:len(segment)] = segment
                current_pos = len(segment)
                continue

            # Apply crossfading between segments
            if current_pos >= fade_samples and len(segment) >= fade_samples:
                prev_segment = processed_signal[current_pos -
                                                fade_samples:current_pos]
                next_segment = segment[:fade_samples]

                # Create crossfade
                fade_out = np.linspace(1.0, 0.0, fade_samples)
                fade_in = np.linspace(0.0, 1.0, fade_samples)
                crossfaded = prev_segment * fade_out + next_segment * fade_in

                # Apply crossfaded section
                processed_signal[current_pos -
                                 fade_samples:current_pos] = crossfaded

                # Add remaining segment
                segment_start = current_pos
                segment_end = current_pos + len(segment) - fade_samples
                processed_signal[segment_start:segment_end] = segment[fade_samples:]
                current_pos = segment_end
            else:
                # If segment too short for crossfading, just append
                segment_end = current_pos + len(segment)
                processed_signal[current_pos:segment_end] = segment
                current_pos = segment_end

        # Trim trailing zeros
        processed_signal = processed_signal[:current_pos]

        # Save processed audio
        sf.write(temp_file, processed_signal, sample_rate)

        return temp_file

    except Exception as e:
        print(f"Warning: Silence removal failed: {e}")
        return input_file


def _enhance_audio(
    input_file: str,
    temp_files: List[str],
    bass_boost: bool = True,
    treble_boost: bool = True,
    compression: bool = True
) -> str:
    """Enhance audio quality with frequency boosts and compression."""
    temp_file = tempfile.mktemp(suffix='.wav')
    temp_files.append(temp_file)

    try:
        # Load audio
        signal, sample_rate = librosa.load(input_file, sr=None)
        enhanced_signal = signal.copy()

        if bass_boost:
            # Bass boost around 200 Hz
            enhanced_signal = _add_frequency_boost(
                enhanced_signal,
                sample_rate,
                freq_range=(150, 250),
                gain=3.0
            )

        if treble_boost:
            # Treble boost above 5000 Hz
            enhanced_signal = _add_frequency_boost(
                enhanced_signal,
                sample_rate,
                freq_range=(5000, sample_rate//2),
                gain=2.0
            )

        if compression:
            enhanced_signal = _dynamic_range_compression(
                enhanced_signal,
                threshold=-24.0,
                ratio=4.0
            )

        # Save enhanced audio
        sf.write(temp_file, enhanced_signal, sample_rate)

        return temp_file

    except Exception as e:
        print(f"Warning: Audio enhancement failed: {e}")
        return input_file


def _add_frequency_boost(
    signal: np.ndarray,
    sample_rate: int,
    freq_range: Tuple[int, int],
    gain: float
) -> np.ndarray:
    """Apply frequency-specific boost to signal."""
    fft_out = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

    # Create mask for frequency range
    mask = (np.abs(freqs) >= freq_range[0]) & (np.abs(freqs) <= freq_range[1])
    fft_out[mask] *= gain

    return np.real(np.fft.ifft(fft_out))


def _dynamic_range_compression(
    signal: np.ndarray,
    threshold: float = -24.0,
    ratio: float = 4.0
) -> np.ndarray:
    """Apply dynamic range compression to signal."""
    # Convert to dB
    db_signal = 20 * np.log10(np.abs(signal) + 1e-8)

    # Apply compression above threshold
    mask = db_signal > threshold
    compressed_db = db_signal.copy()
    compressed_db[mask] = threshold + (db_signal[mask] - threshold) / ratio

    # Convert back to linear scale
    compressed_signal = np.sign(signal) * np.power(10, compressed_db / 20)

    return compressed_signal


# Utility function for copying audio files
def copy_audio(source: str, destination: str):
    """Copy audio file from source to destination."""
    audio_data, sample_rate = sf.read(source)
    sf.write(destination, audio_data, sample_rate)


# Add to soundfile module if not available
if not hasattr(sf, 'copy_audio'):
    sf.copy_audio = copy_audio
