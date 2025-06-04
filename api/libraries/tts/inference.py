import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torchaudio
import soundfile as sf
from torch.serialization import add_safe_globals


def generate_audio(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    config: Optional[Dict[str, Any]] = None,
    auto_download: bool = True
) -> str:
    """
    Generate cloned audio using specified TTS model.

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

    # Set default config values
    default_config = {
        'output_dir': 'outputs',
        'cuda_device': '0',
        'language': 'en',
        'coqui_tos_agreed': True,
        'torch_force_no_weights_only_load': True
    }

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
    """Generate audio using F5-TTS."""
    try:
        # Set up environment
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(config.get('cuda_device', '0'))
        }

        # Use CLI parameters directly instead of config file
        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS_v1_Base",
            "--ref_audio", ref_audio,
            "--ref_text", ref_text,
            "--gen_text", gen_text,
            "--output_dir", str(Path(output_file).parent)
        ]

        print(f"Running F5-TTS: {' '.join(cmd)}")
        print(f"Reference audio: {ref_audio}")
        print(f"Reference text: {ref_text}")
        print(f"Generation text: {gen_text}")
        print(f"Output directory: {Path(output_file).parent}")

        start_time = time.perf_counter()

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )

        runtime = time.perf_counter() - start_time
        print(f"F5-TTS generation completed in {runtime:.3f} seconds")
        print(f"Output: {result.stdout}")

        # F5-TTS generates infer_cli_basic.wav by default
        output_dir = Path(output_file).parent
        f5tts_default_output = output_dir / "infer_cli_basic.wav"

        if f5tts_default_output.exists():
            # Rename the F5-TTS output to our expected filename
            if str(f5tts_default_output) != output_file:
                import shutil
                shutil.move(str(f5tts_default_output), output_file)
                print(
                    f"Renamed F5-TTS output from {f5tts_default_output} to {output_file}")
        elif os.path.exists(output_file):
            # File already exists with correct name
            print(f"F5-TTS output found at expected location: {output_file}")
        else:
            # Look for any recently created wav files as fallback
            print("F5-TTS default output not found, searching for generated files...")
            for file in output_dir.glob("*.wav"):
                if file.stat().st_mtime >= start_time - 5:  # Allow 5 second buffer
                    print(f"Found recently created file: {file}")
                    if str(file) != output_file:
                        import shutil
                        shutil.move(str(file), output_file)
                        print(f"Moved {file} to {output_file}")
                    break
            else:
                raise RuntimeError(
                    f"F5-TTS did not generate expected output file. Expected: {output_file}")

        # Verify the final output exists
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"F5-TTS output file not found after processing: {output_file}")

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"F5-TTS generation failed:")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")
        raise RuntimeError(f"F5-TTS generation failed: {e.stderr}")

    except Exception as e:
        raise RuntimeError(f"F5-TTS generation failed: {str(e)}")


def _generate_xtts(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using XTTS-v2."""
    try:
        # Set PyTorch to use weights_only=False for XTTS (trusted model)
        import torch
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = patched_load

        # Set up environment
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(config.get('cuda_device', '0')),
            "COQUI_TOS_AGREED": "1" if config.get('coqui_tos_agreed', True) else "0",
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': "1" if config.get('torch_force_no_weights_only_load', True) else "0"
        }

        # Build TTS command
        cmd = [
            "tts",
            "--text", gen_text,
            "--reference_wav", ref_audio,
            "--speaker_wav", ref_audio,
            "--out_path", output_file,
            "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
            "--language_idx", config.get('language', 'en')
        ]

        print(f"Running XTTS-v2: {' '.join(cmd)}")
        start_time = time.perf_counter()

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )

        runtime = time.perf_counter() - start_time
        print(f"XTTS-v2 generation completed in {runtime:.3f} seconds")
        print(f"Output: {result.stdout}")

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"XTTS-v2 generation failed:")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")
        raise RuntimeError(f"XTTS-v2 generation failed: {e.stderr}")

    except Exception as e:
        raise RuntimeError(f"XTTS-v2 generation failed: {str(e)}")
    finally:
        # Restore original torch.load if it was patched
        if 'original_load' in locals():
            torch.load = original_load


def _generate_zonos(
    model: str,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_file: str,
    config: Dict[str, Any]
) -> str:
    """Generate audio using Zonos."""
    try:
        # Set up environment
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(config.get('cuda_device', '0')),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1" if config.get('torch_force_no_weights_only_load', True) else "0"
        }

        # Import Zonos modules (these need to be available in the environment)
        try:
            from zonos.model import Zonos
            from zonos.conditioning import make_cond_dict
            from zonos.utils import DEFAULT_DEVICE as device
        except ImportError as e:
            raise ImportError(f"Zonos modules not available: {e}")

        print(f"Loading Zonos model: {model}")
        start_time = time.perf_counter()

        # Load Zonos model
        zonos_model = Zonos.from_pretrained(model, device=device)

        # Load reference audio and create speaker embedding
        wav, sampling_rate = torchaudio.load(ref_audio)
        speaker = zonos_model.make_speaker_embedding(wav, sampling_rate)

        # Generate speech using Zonos
        cond_dict = make_cond_dict(
            text=gen_text,
            speaker=speaker,
            language=config.get('language', 'en-us')
        )
        conditioning = zonos_model.prepare_conditioning(cond_dict)

        # Generate audio codes and decode
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
