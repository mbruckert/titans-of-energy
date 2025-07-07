from libraries.knowledgebase.preprocess import (
    process_documents_for_collection,
    ensure_character_collections_compatible,
    handle_character_embedding_model_change,
    delete_character_collections,
    rename_character_collections
)
from libraries.knowledgebase.retrieval import (
    query_collection,
    check_collection_compatibility,
    get_collection_diagnostics
)
from libraries.llm.inference import generate_styled_text, get_style_data, load_model, ModelType, preload_llm_model, unload_all_cached_models, get_cached_models_info
from libraries.llm.preprocess import download_model as download_model_func
from libraries.tts.preprocess import generate_reference_audio, download_voice_models
from libraries.tts.inference import generate_audio, ensure_model_available, unload_models, get_loaded_models, preload_models_smart, is_model_loaded, generate_audio_with_similarity
from libraries.utils.embedding_models import EmbeddingModelManager, embedding_manager
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
import json
import uuid
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Optional
import time
import torch
import subprocess

# Import library modules
import sys
sys.path.append('./libraries')

# Import device optimization utilities
sys.path.append('./libraries/utils')
try:
    from device_optimization import get_device_info, print_device_info, DeviceType
    DEVICE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Device optimization not available. Using default configurations.")
    DEVICE_OPTIMIZATION_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

# Create subdirectory for generated audio files
GENERATED_AUDIO_DIR = STORAGE_DIR / "generated_audio"
GENERATED_AUDIO_DIR.mkdir(exist_ok=True)

# Track startup time for performance metrics
_startup_time = time.time()

# Initialize device optimization on startup
if DEVICE_OPTIMIZATION_AVAILABLE:
    print("\n" + "="*60)
    print("🚀 TITANS API - Hardware Optimization Enabled")
    print("="*60)
    device_type, device_info = get_device_info()
    print_device_info(device_type, device_info)

    # Print optimization summary
    if device_type == DeviceType.NVIDIA_GPU:
        print(f"🎯 NVIDIA GPU Optimizations Active:")
        print(f"   • GPU Memory: {device_info.get('memory_gb', 0):.1f} GB")
        print(
            f"   • Compute Capability: {device_info.get('compute_capability', 'unknown')}")
        print(f"   • TTS Batch Size: {device_info.get('tts_batch_size', 4)}")
        print(f"   • LLM GPU Layers: {device_info.get('llm_gpu_layers', -1)}")
        print(
            f"   • Mixed Precision: {'Enabled' if device_info.get('mixed_precision', True) else 'Disabled'}")
        print(
            f"   • Flash Attention: {'Enabled' if device_info.get('llm_use_flash_attention', True) else 'Disabled'}")
        if device_info.get('is_high_end', False):
            print(f"   • High-End GPU Features: torch.compile, larger batches")
    elif device_type == DeviceType.APPLE_SILICON:
        print(f"🍎 Apple Silicon Optimizations Active:")
        print(f"   • Chip: {device_info.get('device_name', 'Apple Silicon')}")
        print(f"   • CPU Cores: {device_info.get('cpu_count', 0)}")
        print(
            f"   • MPS Available: {'Yes' if device_info.get('torch_device') == 'mps' else 'No'}")
        print(
            f"   • Performance Tier: {'High-End' if device_info.get('is_high_end', False) else ('Pro' if device_info.get('is_pro', False) else 'Standard')}")
        print(f"   • Optimized for Metal Performance Shaders")
    else:
        print(f"💻 CPU Optimizations Active:")
        print(f"   • Device: {device_info.get('device_name', 'CPU')}")
        print(f"   • Threads: {device_info.get('llm_threads', 8)}")
        print(f"   • Conservative settings for stability")

    print("="*60)
else:
    print("\n" + "="*60)
    print("🚀 TITANS API - Standard Configuration")
    print("="*60)
    print("⚠️  Hardware optimization not available - using default settings")
    print("="*60)

# Check for Hugging Face authentication
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if HUGGINGFACE_API_KEY:
    try:
        from huggingface_hub import login
        login(token=HUGGINGFACE_API_KEY, add_to_git_credential=True)
        print("✓ Authenticated with Hugging Face")
    except Exception as e:
        print(f"⚠ Warning: Failed to authenticate with Hugging Face: {e}")
else:
    print("ℹ No Hugging Face API key found. Public models will still work.")


def setup_zonos_environment():
    """
    Automatically setup the Zonos conda environment if it doesn't exist.
    This ensures no manual setup is required for Zonos TTS functionality.
    """
    print("\n" + "="*60)
    print("🔧 ZONOS ENVIRONMENT SETUP")
    print("="*60)
    
    try:
        # Get conda base path - need to find the actual conda installation, not the current env
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
        
        if not conda_base:
            print("⚠️  Warning: Could not find conda installation")
            print("   Zonos TTS will not be available")
            print("   Please install conda/miniconda to use Zonos")
            return False
        
        conda_env_name = "tts_zonos"
        conda_env_path = os.path.join(conda_base, 'envs', conda_env_name)
        conda_executable = os.path.join(conda_base, 'bin', 'conda')
        zonos_env_python = os.path.join(conda_env_path, 'bin', 'python')
        
        print(f"📍 Conda base: {conda_base}")
        print(f"📍 Target environment: {conda_env_path}")
        
        # Check if environment already exists
        if os.path.exists(zonos_env_python):
            print("✓ Zonos conda environment already exists")
            
            # Verify Zonos installation
            try:
                result = subprocess.run([
                    zonos_env_python, '-c', 
                    'import zonos.model; import torch; import torchaudio; print("OK")'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and "OK" in result.stdout:
                    print("✓ Zonos dependencies verified")
                    print("🎯 Zonos environment ready!")
                    return True
                else:
                    print("⚠️  Zonos dependencies missing, reinstalling...")
            except subprocess.TimeoutExpired:
                print("⚠️  Zonos verification timed out, reinstalling...")
            except Exception as e:
                print(f"⚠️  Zonos verification failed: {e}, reinstalling...")
        else:
            print("📦 Creating new Zonos conda environment...")
            
        # If we get here, either the environment doesn't exist or verification failed
        # Check if environment directory exists but is broken
        if os.path.exists(conda_env_path):
            print("🗑️  Removing existing broken environment...")
            try:
                subprocess.run([
                    conda_executable, 'env', 'remove', '-n', conda_env_name, '-y'
                ], capture_output=True, text=True, timeout=120)
            except:
                # If conda remove fails, try manual removal
                try:
                    shutil.rmtree(conda_env_path)
                except:
                    pass
        
        # Create new environment
        print("🔄 Setting up Zonos conda environment...")
        print("   This may take a few minutes on first run...")
        print("📦 Creating conda environment with Python 3.10...")
        result = subprocess.run([
            conda_executable, 'create', '-n', conda_env_name, 'python=3.10', '-y'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Failed to create conda environment: {result.stderr}")
            return False
        
        print("✓ Conda environment created successfully")
        
        # Install required packages
        print("📦 Installing PyTorch and dependencies...")
        
        # Determine the appropriate PyTorch installation command based on system
        if DEVICE_OPTIMIZATION_AVAILABLE:
            device_type, device_info = get_device_info()
            if device_type == DeviceType.APPLE_SILICON:
                # Apple Silicon - use conda-forge for better compatibility
                torch_cmd = [
                    conda_executable, 'install', '-n', conda_env_name, '-c', 'conda-forge',
                    'pytorch', 'torchaudio', 'numpy', 'soundfile', '-y'
                ]
            elif device_type == DeviceType.NVIDIA_GPU:
                # NVIDIA GPU - use PyTorch with CUDA
                torch_cmd = [
                    conda_executable, 'install', '-n', conda_env_name, '-c', 'pytorch', '-c', 'nvidia',
                    'pytorch', 'torchaudio', 'pytorch-cuda=11.8', '-y'
                ]
            else:
                # CPU only
                torch_cmd = [
                    conda_executable, 'install', '-n', conda_env_name, '-c', 'pytorch',
                    'pytorch', 'torchaudio', 'cpuonly', '-y'
                ]
        else:
            # Default installation
            torch_cmd = [
                conda_executable, 'install', '-n', conda_env_name, '-c', 'pytorch',
                'pytorch', 'torchaudio', '-y'
            ]
        
        result = subprocess.run(torch_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"❌ Failed to install PyTorch: {result.stderr}")
            return False
        
        print("✓ PyTorch installed successfully")
        
        # Install additional dependencies via pip
        print("📦 Installing additional dependencies...")
        pip_packages = [
            'soundfile',
            'numpy',
            'librosa'
        ]
        
        for package in pip_packages:
            try:
                result = subprocess.run([
                    zonos_env_python, '-m', 'pip', 'install', package
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"✓ Installed {package}")
                else:
                    print(f"⚠️  Warning: Failed to install {package} via pip")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Warning: Timeout installing {package}")
            except Exception as e:
                print(f"⚠️  Warning: Error installing {package}: {e}")
        
        # Install Zonos from GitHub
        print("📦 Installing Zonos from GitHub...")
        result = subprocess.run([
            zonos_env_python, '-m', 'pip', 'install', 
            'git+https://github.com/Zyphra/Zonos.git'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Failed to install Zonos: {result.stderr}")
            print("🔄 Trying alternative installation method...")
            
            # Try cloning and installing manually
            try:
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    clone_result = subprocess.run([
                        'git', 'clone', 'https://github.com/Zyphra/Zonos.git', 
                        os.path.join(temp_dir, 'Zonos')
                    ], capture_output=True, text=True, timeout=120)
                    
                    if clone_result.returncode == 0:
                        install_result = subprocess.run([
                            zonos_env_python, '-m', 'pip', 'install', '-e', 
                            os.path.join(temp_dir, 'Zonos')
                        ], capture_output=True, text=True, timeout=300)
                        
                        if install_result.returncode != 0:
                            print(f"❌ Alternative installation also failed: {install_result.stderr}")
                            return False
                    else:
                        print(f"❌ Failed to clone Zonos repository: {clone_result.stderr}")
                        return False
            except Exception as e:
                print(f"❌ Alternative installation failed: {e}")
                return False
        
        print("✓ Zonos installed successfully")
        
        # Verify installation
        print("🔍 Verifying Zonos installation...")
        try:
            result = subprocess.run([
                zonos_env_python, '-c', 
                'import zonos.model; import torch; import torchaudio; print("Zonos installation verified")'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✓ Zonos installation verified successfully")
                print("🎯 Zonos environment setup complete!")
                
                # Check for espeak-ng system installation
                espeak_check = subprocess.run(['which', 'espeak-ng'], capture_output=True, text=True)
                if espeak_check.returncode == 0:
                    print(f"✓ espeak-ng found at: {espeak_check.stdout.strip()}")
                else:
                    print("⚠️  espeak-ng not found in system PATH")
                    print("   Please install it with: brew install espeak-ng (macOS)")
                    print("   Zonos will attempt to locate it automatically")
                
                return True
            else:
                print(f"❌ Zonos verification failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Zonos verification timed out")
            return False
        except Exception as e:
            print(f"❌ Zonos verification error: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Zonos environment setup failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    
    finally:
        print("="*60)


# Setup Zonos environment on startup
setup_zonos_environment()

# Database configuration


def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


# Global model cache
model_cache = {}


def resolve_model_path(model_id: str) -> tuple[str, str]:
    """
    Resolve model ID to actual model path and type.

    Args:
        model_id: Model identifier (e.g., "google-gemma-3-4b-it-qat-q4_0-gguf" or "custom-model-name")

    Returns:
        Tuple of (model_path, model_type)
    """
    # Define supported models with their paths
    model_configs = {
        "google-gemma-3-4b-it-qat-q4_0-gguf": {
            "path": "./models/gemma-3-4b-it-q4_0.gguf",
            "type": "gguf"
        },
        "llama-3.2-3b": {
            "path": "meta-llama/Llama-3.2-3B",
            "type": "huggingface"
        },
        "gpt-4o": {
            "path": "gpt-4o",
            "type": "openai_api"
        },
        "gpt-4o-mini": {
            "path": "gpt-4o-mini",
            "type": "openai_api"
        }
    }

    # Check if model_id is in our supported models
    if model_id in model_configs:
        config = model_configs[model_id]
        return config["path"], config["type"]

    # Check for custom models in the models directory
    if model_id.startswith('custom-'):
        import os
        import glob
        import json
        models_dir = "./models"
        
        # Look for files matching the custom model pattern
        custom_name = model_id.replace('custom-', '', 1)
        
        # Check for GGUF files first
        gguf_pattern = os.path.join(models_dir, f"{custom_name}*.gguf")
        gguf_files = glob.glob(gguf_pattern)
        if gguf_files:
            return gguf_files[0], "gguf"
        
        # Check for directories (regular HF models)
        possible_dirs = [
            os.path.join(models_dir, custom_name),
            os.path.join(models_dir, custom_name.replace('-', '_')),
            os.path.join(models_dir, custom_name.replace(' ', '_')),
            # Also check for the original directory name pattern (e.g., Llama_3.2)
            os.path.join(models_dir, custom_name.replace('-', '_').replace('_', '.')),
            os.path.join(models_dir, custom_name.replace('-', '.')),
        ]
        
        # Also check all directories in models folder to find a match
        if os.path.exists(models_dir):
            for dir_name in os.listdir(models_dir):
                dir_path = os.path.join(models_dir, dir_name)
                if os.path.isdir(dir_path):
                    # Check if this directory name could match our custom_name
                    normalized_dir = dir_name.lower().replace('_', '-').replace('.', '-')
                    normalized_custom = custom_name.lower().replace('_', '-').replace('.', '-')
                    if normalized_dir == normalized_custom:
                        possible_dirs.append(dir_path)
                        print(f"Debug: Found matching directory: {dir_path}")
                        
        print(f"Debug: Looking for custom model '{custom_name}' in directories: {possible_dirs}")
        print(f"Debug: Available directories in {models_dir}: {[d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))] if os.path.exists(models_dir) else 'models_dir does not exist'}")
        
        print(f"Debug: All possible directories to check: {possible_dirs}")
        
        for model_dir in possible_dirs:
            print(f"Debug: Checking directory: {model_dir}")
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                print(f"Debug: Found directory: {model_dir}")
                # Check if there's a .huggingface_repo file that stores the original repo
                repo_file = os.path.join(model_dir, '.huggingface_repo')
                if os.path.exists(repo_file):
                    try:
                        with open(repo_file, 'r') as f:
                            original_repo = f.read().strip()
                        print(f"Found original repo for {model_id}: {original_repo}")
                        # Return the original repository name since it should be cached by transformers
                        print(f"Using original repo name for transformers cache: {original_repo}")
                        return original_repo, "huggingface"
                    except Exception as e:
                        print(f"Error reading repo file for {model_id}: {e}")
                        # Fallback to local directory
                        print(f"Fallback: using local directory path: {model_dir}")
                        return model_dir, "huggingface"
                
                # Check if this directory has a config.json with model info
                config_path = os.path.join(model_dir, 'config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        # Check if this is a transformers model
                        if 'model_type' in config or 'architectures' in config:
                            # Use the directory path as fallback
                            print(f"Using directory path for {model_id}: {model_dir}")
                            return model_dir, "huggingface"
                    except Exception as e:
                        print(f"Error reading config for {model_id}: {e}")
                        # Fallback to directory path
                        return model_dir, "huggingface"
                else:
                    # Directory exists but no config - might be a GGUF or other format
                    return model_dir, "huggingface"

    # Fallback: try to determine type based on the model_id
    if model_id.endswith('.gguf'):
        return model_id, "gguf"
    elif any(api_model in model_id.lower() for api_model in ['gpt-', 'claude-', 'openai']):
        return model_id, "openai_api"
    else:
        return model_id, "huggingface"


# STT (Speech-to-Text) optimizations - Global Whisper model cache
_whisper_model = None
_whisper_model_size = None
_stt_device = None
_stt_load_time = None
_stt_memory_usage = None


def _get_optimal_whisper_model_size(device_type, device_info):
    """Determine optimal Whisper model size based on device capability."""
    if not DEVICE_OPTIMIZATION_AVAILABLE:
        return "small"  # Conservative default

    if device_type == DeviceType.NVIDIA_GPU:
        gpu_memory = device_info.get('memory_gb', 8)
        if gpu_memory >= 16 and device_info.get('is_high_end', False):
            return "small"  # Best quality for high-end GPUs
        elif gpu_memory >= 12:
            return "medium"  # Good balance
        elif gpu_memory >= 8:
            return "small"  # Memory efficient
        else:
            return "tiny"  # Ultra memory efficient

    elif device_type == DeviceType.APPLE_SILICON:
        if device_info.get('is_high_end', False):  # M1/M2/M3/M4 Max/Ultra
            return "medium"  # Apple Silicon can handle medium well
        elif device_info.get('is_pro', False):     # M1/M2/M3/M4 Pro
            return "small"   # Good balance for Pro chips
        else:
            return "small"   # Conservative for base chips
    else:
        # CPU-only
        cpu_count = device_info.get('cpu_count', 4)
        if cpu_count >= 16:
            return "small"
        elif cpu_count >= 8:
            return "tiny"
        else:
            return "tiny"


def _get_stt_device(device_type, device_info):
    """Determine optimal device for STT processing."""
    if not DEVICE_OPTIMIZATION_AVAILABLE:
        return "cpu"

    if device_type == DeviceType.NVIDIA_GPU:
        return "cuda"
    elif device_type == DeviceType.APPLE_SILICON:
        # Try MPS first for Apple Silicon, fallback to CPU if not available
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"  # Fallback to CPU with optimizations
    else:
        return "cpu"


def _get_whisper_model():
    """Get or initialize Whisper model with comprehensive device optimization."""
    global _whisper_model, _whisper_model_size, _stt_device, _stt_load_time, _stt_memory_usage

    if DEVICE_OPTIMIZATION_AVAILABLE:
        device_type, device_info = get_device_info()
    else:
        device_type, device_info = None, {}

    # Determine optimal model size and device
    optimal_size = _get_optimal_whisper_model_size(device_type, device_info)
    optimal_device = _get_stt_device(device_type, device_info)

    # Check if we need to reload the model
    need_reload = (
        _whisper_model is None or
        _whisper_model_size != optimal_size or
        _stt_device != optimal_device
    )

    if need_reload:
        try:
            import whisper

            print(
                f"🎙️  Loading Whisper model '{optimal_size}' for {device_type.value if device_type else 'default'} on {optimal_device}")
            start_time = time.perf_counter()

            # Clear previous model if exists
            if _whisper_model is not None:
                del _whisper_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load new model with device optimization
            _whisper_model = whisper.load_model(
                optimal_size, device=optimal_device)

            # Apply device-specific optimizations
            if DEVICE_OPTIMIZATION_AVAILABLE:
                if device_type == DeviceType.NVIDIA_GPU:
                    print("🎯 Applying NVIDIA GPU optimizations to Whisper...")

                    # Enable mixed precision if supported
                    if device_info.get('mixed_precision', True):
                        try:
                            _whisper_model = _whisper_model.half()
                            print("✓ Enabled mixed precision (FP16) for Whisper")
                        except Exception as e:
                            print(
                                f"Warning: Mixed precision failed for Whisper: {e}")

                    # Apply torch.compile for high-end GPUs
                    if device_info.get('is_high_end', False) and hasattr(torch, 'compile'):
                        try:
                            _whisper_model = torch.compile(
                                _whisper_model, mode="reduce-overhead")
                            print("✓ Applied torch.compile optimization to Whisper")
                        except Exception as e:
                            print(
                                f"Warning: torch.compile failed for Whisper: {e}")

                    # Set CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True

                elif device_type == DeviceType.APPLE_SILICON:
                    print("🍎 Applying Apple Silicon optimizations to Whisper...")

                    # Set optimal thread counts for Apple Silicon
                    optimal_threads = min(device_info.get('cpu_count', 8), 8)
                    torch.set_num_threads(optimal_threads)

                    # Enable Metal Performance Shaders optimizations where possible
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        try:
                            # Enable MPS backend optimizations
                            torch.backends.mps.enabled = True
                            print("✓ Enabled MPS backend for compatible operations")
                            
                            # Set MPS-specific optimizations for Whisper
                            if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                                torch.backends.mps.set_per_process_memory_fraction(0.8)
                                print("✓ Set MPS memory fraction for Whisper")
                        except Exception as e:
                            print(
                                f"Note: Some MPS optimizations not available for Whisper: {e}")

                    print(f"✓ Optimized thread count: {optimal_threads}")

                    # Enable Accelerate framework optimizations if available
                    try:
                        import accelerate
                        print(
                            "✓ Accelerate framework available for additional optimizations")
                    except ImportError:
                        pass

                else:
                    print("💻 Applying CPU optimizations to Whisper...")

                    # Set optimal thread counts for CPU
                    cpu_count = device_info.get('cpu_count', 4)
                    optimal_threads = min(cpu_count, 8)
                    torch.set_num_threads(optimal_threads)
                    print(f"✓ Optimized thread count: {optimal_threads}")

            # Record performance metrics
            load_time = time.perf_counter() - start_time
            _stt_load_time = load_time
            _whisper_model_size = optimal_size
            _stt_device = optimal_device

            # Estimate memory usage
            if torch.cuda.is_available() and optimal_device == "cuda":
                _stt_memory_usage = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"✓ Whisper model '{optimal_size}' loaded in {load_time:.2f}s on {optimal_device}")
                print(f"✓ GPU Memory usage: {_stt_memory_usage:.2f} GB")
            else:
                print(
                    f"✓ Whisper model '{optimal_size}' loaded in {load_time:.2f}s on {optimal_device}")

        except ImportError as e:
            print(f"Error importing Whisper: {e}")
            raise ImportError(
                "Whisper not available. Please install with: pip install openai-whisper")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            # Fallback to smaller model
            try:
                print("Attempting fallback to 'tiny' model...")
                _whisper_model = whisper.load_model("tiny", device="cpu")
                _whisper_model_size = "tiny"
                _stt_device = "cpu"
                print("✓ Fallback Whisper model loaded successfully")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise

    return _whisper_model


def _transcribe_audio_optimized(audio_path: str) -> str:
    """Transcribe audio using optimized Whisper model."""
    try:
        # Get optimized Whisper model
        whisper_model = _get_whisper_model()

        if DEVICE_OPTIMIZATION_AVAILABLE:
            device_type, device_info = get_device_info()
        else:
            device_type, device_info = None, {}

        print(
            f"🎙️  Transcribing audio with Whisper '{_whisper_model_size}' on {_stt_device}")
        start_time = time.perf_counter()

        # Transcribe with device-specific optimizations
        transcribe_options = {
            "language": "en",
            "task": "transcribe",
        }

        # Add device-specific transcription options
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                # NVIDIA GPU specific options
                transcribe_options.update({
                    "fp16": device_info.get('mixed_precision', True),
                    "beam_size": 5 if device_info.get('is_high_end', False) else 3,
                    "patience": 2.0,
                })
            elif device_type == DeviceType.APPLE_SILICON:
                # Apple Silicon specific options
                transcribe_options.update({
                    "fp16": False,  # Apple Silicon works better with FP32 for Whisper
                    "beam_size": 3 if device_info.get('is_high_end', False) else 1,
                    "patience": 1.0,
                })
            else:
                # CPU specific options
                transcribe_options.update({
                    "fp16": False,
                    "beam_size": 1,  # Conservative for CPU
                    "patience": 1.0,
                })

        # Perform transcription with device-specific optimizations
        try:
            if device_type == DeviceType.NVIDIA_GPU and device_info.get('mixed_precision', True):
                # Use autocast for mixed precision on NVIDIA GPUs
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    result = whisper_model.transcribe(
                        audio_path, **transcribe_options)
            elif device_type == DeviceType.APPLE_SILICON and _stt_device == "mps":
                # Use MPS optimizations for Apple Silicon
                with torch.inference_mode():  # Use inference mode for better MPS performance
                    result = whisper_model.transcribe(audio_path, **transcribe_options)
            else:
                result = whisper_model.transcribe(audio_path, **transcribe_options)
        except Exception as transcribe_error:
            # Handle MPS-specific errors and fallback to CPU
            if _stt_device == "mps" and ("mps" in str(transcribe_error).lower() or "metal" in str(transcribe_error).lower()):
                print(f"⚠️  MPS transcription failed, falling back to CPU: {transcribe_error}")
                # Reload model on CPU for this transcription
                import whisper
                cpu_model = whisper.load_model(_whisper_model_size, device="cpu")
                result = cpu_model.transcribe(audio_path, **transcribe_options)
                del cpu_model  # Clean up
            else:
                raise transcribe_error

        transcript = result["text"].strip()
        transcription_time = time.perf_counter() - start_time

        print(f"✓ Audio transcription completed in {transcription_time:.3f}s")
        print(f"✓ Transcript length: {len(transcript)} characters")

        return transcript

    except Exception as e:
        print(f"Error in optimized audio transcription: {e}")
        raise


def _unload_stt_model():
    """Unload STT model from memory to free up resources."""
    global _whisper_model, _whisper_model_size, _stt_device, _stt_load_time, _stt_memory_usage

    if _whisper_model is not None:
        print("🧹 Unloading Whisper STT model from memory...")
        del _whisper_model
        _whisper_model = None
        _whisper_model_size = None
        _stt_device = None
        _stt_load_time = None
        _stt_memory_usage = None

        # Clear GPU cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ CUDA cache cleared")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                print("✓ MPS cache cleared")
            except:
                pass

        # Force garbage collection
        import gc
        gc.collect()
        print("✓ Whisper STT model unloaded successfully")


def get_stt_performance_info() -> Dict[str, Any]:
    """Get STT model performance information."""
    return {
        "model_loaded": _whisper_model is not None,
        "model_size": _whisper_model_size,
        "device": _stt_device,
        "load_time": _stt_load_time,
        "memory_usage": _stt_memory_usage,
    }


def get_file_url(file_path: str, file_type: str) -> Optional[str]:
    """
    Convert a file path to a URL for serving files.
    Returns None if file_path is None or file doesn't exist.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"get_file_url: File not found or path is None: {file_path}")
        return None

    # Convert absolute path to relative path from storage directory
    try:
        relative_path = Path(file_path).relative_to(STORAGE_DIR)
        if file_type == 'audio':
            url = url_for('serve_audio', filename=str(
                relative_path), _external=True)
            print(
                f"get_file_url: Generated audio URL: {url} for file: {file_path}")
            return url
        elif file_type == 'image':
            return url_for('serve_image', filename=str(relative_path), _external=True)
    except ValueError as e:
        # File is not in storage directory
        print(
            f"get_file_url: File not in storage directory: {file_path}, error: {e}")
        return None

    return None


def get_image_base64(file_path: str) -> Optional[str]:
    """
    Convert an image file to base64 string.
    Returns None if file_path is None or file doesn't exist.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"get_image_base64: File not found or path is None: {file_path}")
        return None

    try:
        import base64
        with open(file_path, 'rb') as image_file:
            # Read the image file
            image_data = image_file.read()
            # Encode to base64
            base64_string = base64.b64encode(image_data).decode('utf-8')

            # Determine MIME type based on file extension
            file_extension = Path(file_path).suffix.lower()
            if file_extension in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif file_extension == '.png':
                mime_type = 'image/png'
            elif file_extension == '.gif':
                mime_type = 'image/gif'
            elif file_extension == '.webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'  # Default fallback

            # Return as data URL
            return f"data:{mime_type};base64,{base64_string}"

    except Exception as e:
        print(f"get_image_base64: Error converting image to base64: {e}")
        return None


def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Create characters table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            image_path VARCHAR(500),
            llm_model VARCHAR(255),
            llm_config JSONB,
            knowledge_base_path VARCHAR(500),
            voice_cloning_audio_path VARCHAR(500),
            voice_cloning_reference_text TEXT,
            voice_cloning_settings JSONB,
            style_tuning_data_path VARCHAR(500),
            wakeword VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create chat_history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            character_id INTEGER REFERENCES characters(id) ON DELETE CASCADE,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            audio_base64 TEXT,
            knowledge_context TEXT,
            knowledge_references JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add voice_cloning_reference_text column if it doesn't exist (migration)
    try:
        cur.execute("""
            ALTER TABLE characters 
            ADD COLUMN IF NOT EXISTS voice_cloning_reference_text TEXT
        """)
        print("Added voice_cloning_reference_text column (if not exists)")
    except Exception as e:
        print(f"Migration note: {e}")

    # Add knowledge_references column if it doesn't exist (migration)
    try:
        cur.execute("""
            ALTER TABLE chat_history 
            ADD COLUMN IF NOT EXISTS knowledge_references JSONB
        """)
        print("Added knowledge_references column (if not exists)")
    except Exception as e:
        print(f"Migration note: {e}")

    # Add wakeword column if it doesn't exist (migration)
    try:
        cur.execute("""
            ALTER TABLE characters 
            ADD COLUMN IF NOT EXISTS wakeword VARCHAR(255)
        """)
        print("Added wakeword column (if not exists)")
    except Exception as e:
        print(f"Migration note: {e}")

    # Add thinking_audio_base64 column if it doesn't exist (migration)
    try:
        cur.execute("""
            ALTER TABLE characters 
            ADD COLUMN IF NOT EXISTS thinking_audio_base64 JSONB
        """)
        print("Added thinking_audio_base64 column (if not exists)")
    except Exception as e:
        print(f"Migration note: {e}")

    # Add embedding configuration columns if they don't exist (migration)
    try:
        cur.execute("""
            ALTER TABLE characters 
            ADD COLUMN IF NOT EXISTS knowledge_base_embedding_config JSONB,
            ADD COLUMN IF NOT EXISTS style_tuning_embedding_config JSONB
        """)
        print("Added embedding config columns (if not exists)")
    except Exception as e:
        print(f"Migration note: {e}")
    
    # Add similarity_score column to chat_history if it doesn't exist (migration)
    try:
        cur.execute("""
            ALTER TABLE chat_history 
            ADD COLUMN IF NOT EXISTS similarity_score REAL
        """)
        print("Added similarity_score column to chat_history (if not exists)")
    except Exception as e:
        print(f"Migration note: {e}")

    conn.commit()
    cur.close()
    conn.close()


def generate_thinking_audio(character_name: str, voice_cloning_settings: dict, voice_cloning_audio_path: str, voice_reference_text: str) -> Optional[dict]:
    """
    Generate thinking audio phrases for a character using their TTS model.
    Returns a dictionary of phrase -> base64_audio mappings.
    """
    thinking_phrases = [
        "That's a great question, let me think about that for a second",
        "Thanks for asking, let me think that over for a second", 
        "An interesting inquiry, allow me a moment to reflect",
        "Give me just a moment to consider your question"
    ]
    
    if not voice_cloning_settings or not voice_cloning_audio_path:
        print(f"Warning: No voice cloning settings for {character_name}, skipping thinking audio generation")
        return None
    
    # Check if the voice cloning audio file exists
    import os
    if not os.path.exists(voice_cloning_audio_path):
        print(f"Warning: Voice cloning audio file not found: {voice_cloning_audio_path}")
        return None
    
    try:
        # Import TTS generation function
        from libraries.tts.inference import generate_cloned_audio_base64
        
        thinking_audio = {}
        model = voice_cloning_settings.get('model', 'f5tts')
        
        print(f"🤔 Generating thinking audio for {character_name} using {model}...")
        print(f"🎤 Using reference audio: {voice_cloning_audio_path}")
        print(f"📝 Using reference text: {voice_reference_text[:50]}...")
        
        # Limit to 2 phrases for faster character creation (can be expanded later)
        limited_phrases = thinking_phrases[:2]
        
        for i, phrase in enumerate(limited_phrases):
            try:
                print(f"🎤 Generating phrase {i+1}/{len(limited_phrases)}: '{phrase[:30]}...'")
                
                # Generate audio for this phrase with fast mode enabled
                audio_base64 = generate_cloned_audio_base64(
                    model=model,
                    ref_audio=voice_cloning_audio_path,
                    ref_text=voice_reference_text,
                    gen_text=phrase,
                    config=voice_cloning_settings,
                    fast_mode=True
                )
                
                if audio_base64:
                    thinking_audio[f"phrase_{i+1}"] = audio_base64
                    print(f"✅ Generated thinking audio {i+1}")
                else:
                    print(f"⚠️ Failed to generate audio for phrase {i+1}")
                    
            except Exception as e:
                print(f"❌ Error generating thinking audio for phrase {i+1}: {e}")
                continue
        
        if thinking_audio:
            print(f"✅ Successfully generated {len(thinking_audio)} thinking audio clips for {character_name}")
            return thinking_audio
        else:
            print(f"⚠️ Failed to generate any thinking audio for {character_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating thinking audio for {character_name}: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return None


@app.route('/')
def hello_world():
    return {'message': 'Titans API - Ready for character interactions!', 'status': 'success'}


@app.route('/create-character', methods=['POST'])
def create_character():
    """
    Create a new character with all associated data and models.
    Expects form data with files and JSON configuration.
    """
    try:
        # Get form data
        name = request.form.get('name')
        if not name:
            return jsonify({"error": "Character name is required"}), 400

        # Get configuration data
        llm_model = request.form.get('llm_model')
        llm_config = json.loads(request.form.get('llm_config', '{}'))
        voice_cloning_settings = json.loads(
            request.form.get('voice_cloning_settings', '{}'))
        # Normalize voice cloning settings to ensure proper data types
        voice_cloning_settings = normalize_voice_cloning_settings(voice_cloning_settings)
        wakeword = request.form.get('wakeword', f"hey {name.lower()}")

        # Get embedding configurations
        knowledge_base_embedding_config = json.loads(
            request.form.get('knowledge_base_embedding_config', '{}'))
        style_tuning_embedding_config = json.loads(
            request.form.get('style_tuning_embedding_config', '{}'))

        # Use default embedding config if not provided
        default_embedding_config = {
            'model_type': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'config': {'device': 'auto'}
        }
        
        if not knowledge_base_embedding_config:
            knowledge_base_embedding_config = default_embedding_config
        
        if not style_tuning_embedding_config:
            style_tuning_embedding_config = default_embedding_config

        # Extract reference text from voice cloning settings
        voice_reference_text = voice_cloning_settings.get('ref_text', '')

        # Create character directory
        character_dir = STORAGE_DIR / name.replace(' ', '_').lower()
        character_dir.mkdir(exist_ok=True)

        # Handle file uploads
        image_path = None
        knowledge_base_path = None
        voice_cloning_audio_path = None
        style_tuning_data_path = None

        # Save character image
        if 'character_image' in request.files:
            image_file = request.files['character_image']
            if image_file.filename:
                image_path = str(
                    character_dir / f"image_{image_file.filename}")
                image_file.save(image_path)

        # Save and process knowledge base data (support multiple files)
        kb_files = request.files.getlist('knowledge_base_file')
        if kb_files and any(f.filename for f in kb_files):
            try:
                collection_name = f"{name.lower().replace(' ', '')}-knowledge"

                # Create temporary directories for processing
                kb_docs_dir = character_dir / "kb_docs"
                kb_archive_dir = character_dir / "kb_archive"
                kb_docs_dir.mkdir(exist_ok=True)
                kb_archive_dir.mkdir(exist_ok=True)

                # Process all knowledge base files
                kb_file_paths = []
                for i, kb_file in enumerate(kb_files):
                    if kb_file.filename:
                        # Save each file
                        kb_file_path = str(character_dir / f"knowledge_base_{i+1}_{kb_file.filename}")
                        kb_file.save(kb_file_path)
                        kb_file_paths.append(kb_file_path)

                        # Copy the file to the docs directory for processing
                        temp_kb_path = kb_docs_dir / kb_file.filename
                        shutil.copy2(kb_file_path, temp_kb_path)

                # Store the first file path for backward compatibility (or create a manifest)
                if kb_file_paths:
                    knowledge_base_path = kb_file_paths[0]  # Store first file path
                    # Create a manifest file listing all uploaded files
                    manifest_path = character_dir / "knowledge_base_manifest.json"
                    with open(manifest_path, 'w') as f:
                        json.dump({
                            "files": [os.path.basename(path) for path in kb_file_paths],
                            "count": len(kb_file_paths),
                            "created_at": time.time()
                        }, f)

                # Process all documents in the directory
                process_documents_for_collection(
                    str(kb_docs_dir), str(kb_archive_dir), collection_name, knowledge_base_embedding_config, force_recreate=True)
                
                print(f"Successfully processed {len(kb_file_paths)} knowledge base files for {name}")
                
                # Ensure collection is compatible with embedding model
                compatibility_result = ensure_character_collections_compatible(
                    name,
                    knowledge_base_embedding_config=knowledge_base_embedding_config,
                    character_dir=str(character_dir)
                )
                
                if compatibility_result["knowledge_base"]["checked"]:
                    kb_result = compatibility_result["knowledge_base"]
                    if kb_result.get("action") == "recreated":
                        print(f"✅ Knowledge base collection automatically recreated with new embedding model")
                    elif kb_result.get("action") == "failed":
                        print(f"⚠️  Warning: Could not recreate knowledge base collection with new embedding model")
                    elif kb_result.get("action") == "manual_required":
                        print(f"⚠️  Warning: Knowledge base collection needs manual recreation")
            except Exception as e:
                print(f"Warning: Knowledge base processing failed: {e}")

        # Save and preprocess voice cloning audio
        if 'voice_cloning_audio' in request.files:
            voice_file = request.files['voice_cloning_audio']
            if voice_file.filename:
                raw_audio_path = str(
                    character_dir / f"voice_raw_{voice_file.filename}")
                voice_file.save(raw_audio_path)

                # Check if audio preprocessing is enabled (default: True)
                preprocess_audio = voice_cloning_settings.get(
                    'preprocess_audio', True)

                if preprocess_audio:
                    # Preprocess the audio for voice cloning
                    try:
                        # Filter out parameters that don't belong to generate_reference_audio
                        # These are TTS model configuration parameters, not audio processing parameters
                        tts_only_params = {'model', 'cache_dir', 'preprocess_audio', 'ref_text', 'reference_text',
                                           'language', 'output_dir', 'cuda_device', 'coqui_tos_agreed',
                                           'torch_force_no_weights_only_load', 'auto_download', 'gen_text',
                                           'generative_text', 'repetition_penalty', 'top_k', 'top_p', 'speed',
                                           'enable_text_splitting', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8',
                                           'seed', 'cfg_scale', 'speaking_rate', 'frequency_max', 'pitch_standard_deviation'}
                        audio_processing_params = {
                            k: v for k, v in voice_cloning_settings.items()
                            if k not in tts_only_params
                        }
                        
                        # Add device optimization for Apple Silicon
                        if DEVICE_OPTIMIZATION_AVAILABLE:
                            device_type, device_info = get_device_info()
                            if device_type == DeviceType.APPLE_SILICON:
                                # Enable safe mode for Apple Silicon to avoid MPS issues with DeepFilterNet
                                print("🍎 Apple Silicon detected - enabling safe mode for audio preprocessing")
                                audio_processing_params['safe_mode'] = True
                        
                        voice_cloning_audio_path = generate_reference_audio(
                            raw_audio_path,
                            output_file=str(
                                character_dir / "voice_processed.wav"),
                            **audio_processing_params
                        )
                        print(f"Audio preprocessing completed for {name}")
                    except Exception as e:
                        print(f"Warning: Voice preprocessing failed: {e}")
                        print("Attempting fallback with safe mode...")
                        try:
                            # Retry with safe mode enabled
                            audio_processing_params['safe_mode'] = True
                            voice_cloning_audio_path = generate_reference_audio(
                                raw_audio_path,
                                output_file=str(
                                    character_dir / "voice_processed.wav"),
                                **audio_processing_params
                            )
                            print(f"Audio preprocessing completed for {name} with safe mode fallback")
                        except Exception as fallback_error:
                            print(f"Warning: Voice preprocessing failed even with safe mode: {fallback_error}")
                            voice_cloning_audio_path = raw_audio_path
                else:
                    # Use raw audio without preprocessing
                    voice_cloning_audio_path = raw_audio_path
                    print(
                        f"Audio preprocessing skipped for {name} - using raw audio")

        # Generate thinking audio after voice cloning setup is complete
        thinking_audio_base64 = None
        # Add environment variable to control thinking audio generation
        generate_thinking_audio_enabled = os.getenv('GENERATE_THINKING_AUDIO', 'true').lower() == 'true'
        
        if voice_cloning_audio_path and voice_cloning_settings and generate_thinking_audio_enabled:
            try:
                print(f"🤔 Starting thinking audio generation for {name}...")
                thinking_audio_base64 = generate_thinking_audio(
                    name, voice_cloning_settings, voice_cloning_audio_path, voice_reference_text
                )
                if thinking_audio_base64:
                    print(f"✅ Thinking audio generation completed for {name}")
                else:
                    print(f"⚠️ Thinking audio generation returned None for {name}")
            except Exception as thinking_error:
                print(f"❌ Thinking audio generation failed for {name}: {thinking_error}")
                # Continue with character creation even if thinking audio fails
                thinking_audio_base64 = None
        elif not generate_thinking_audio_enabled:
            print(f"⏭️ Thinking audio generation disabled via environment variable for {name}")
        else:
            print(f"⏭️ Skipping thinking audio generation for {name} (no voice cloning settings)")

        # Save style tuning data
        if 'style_tuning_file' in request.files:
            style_file = request.files['style_tuning_file']
            if style_file.filename:
                style_tuning_data_path = str(
                    character_dir / f"style_tuning_{style_file.filename}")
                style_file.save(style_tuning_data_path)

                # Process style tuning data into vector database
                try:
                    collection_name = f"{name.lower().replace(' ', '')}-style"

                    # Create temporary directories for processing
                    style_docs_dir = character_dir / "style_docs"
                    style_archive_dir = character_dir / "style_archive"
                    style_docs_dir.mkdir(exist_ok=True)
                    style_archive_dir.mkdir(exist_ok=True)

                    # Copy the file to the docs directory for processing
                    temp_style_path = style_docs_dir / style_file.filename
                    shutil.copy2(style_tuning_data_path, temp_style_path)

                    # Process the documents
                    process_documents_for_collection(
                        str(style_docs_dir), str(style_archive_dir), collection_name, style_tuning_embedding_config, force_recreate=True)
                    
                    # Ensure collection is compatible with embedding model
                    compatibility_result = ensure_character_collections_compatible(
                        name,
                        style_tuning_embedding_config=style_tuning_embedding_config,
                        character_dir=str(character_dir)
                    )
                    
                    if compatibility_result["style_tuning"]["checked"]:
                        style_result = compatibility_result["style_tuning"]
                        if style_result.get("action") == "recreated":
                            print(f"✅ Style tuning collection automatically recreated with new embedding model")
                        elif style_result.get("action") == "failed":
                            print(f"⚠️  Warning: Could not recreate style tuning collection with new embedding model")
                        elif style_result.get("action") == "manual_required":
                            print(f"⚠️  Warning: Style tuning collection needs manual recreation")
                except Exception as e:
                    print(f"Warning: Style tuning processing failed: {e}")

        # Save to database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO characters 
            (name, image_path, llm_model, llm_config, knowledge_base_path, 
             voice_cloning_audio_path, voice_cloning_reference_text, voice_cloning_settings, 
             style_tuning_data_path, wakeword, thinking_audio_base64,
             knowledge_base_embedding_config, style_tuning_embedding_config)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            RETURNING id
        """, (
            name, image_path, llm_model, json.dumps(
                llm_config), knowledge_base_path,
            voice_cloning_audio_path, voice_reference_text,
            json.dumps(voice_cloning_settings),
            style_tuning_data_path, wakeword, 
            json.dumps(thinking_audio_base64) if thinking_audio_base64 else None,
            json.dumps(knowledge_base_embedding_config),
            json.dumps(style_tuning_embedding_config)
        ))

        character_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "Character created successfully",
            "character_id": character_id,
            "status": "success"
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-characters', methods=['GET'])
def get_characters():
    """Get list of all characters from database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, name, image_path, llm_model, wakeword, created_at 
            FROM characters 
            ORDER BY created_at DESC
        """)
        characters = cur.fetchall()
        cur.close()
        conn.close()

        # Convert to JSON serializable format and add image base64
        result = []
        for char in characters:
            char_dict = dict(char)
            # Convert image path to base64
            char_dict['image_base64'] = get_image_base64(
                char_dict['image_path'])
            # Remove the raw file path for security
            char_dict.pop('image_path', None)
            result.append(char_dict)

        return jsonify({"characters": result, "status": "success"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-character/<int:character_id>', methods=['GET'])
def get_character(character_id):
    """Get detailed information about a specific character."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()
        cur.close()
        conn.close()

        if not character:
            return jsonify({"error": "Character not found"}), 404

        # Convert to dict and add image base64
        char_dict = dict(character)
        char_dict['image_base64'] = get_image_base64(char_dict['image_path'])

        # Keep file path info for edit functionality (but don't expose actual paths)
        char_dict['has_image'] = bool(char_dict['image_path'])
        char_dict['has_knowledge_base'] = bool(
            char_dict['knowledge_base_path'])
        char_dict['has_voice_cloning'] = bool(
            char_dict['voice_cloning_audio_path'])
        char_dict['has_style_tuning'] = bool(
            char_dict['style_tuning_data_path'])

        # Add embedding configurations for editing
        char_dict['knowledge_base_embedding_config'] = char_dict.get('knowledge_base_embedding_config') or {
            'model_type': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'config': {'device': 'auto'}
        }
        char_dict['style_tuning_embedding_config'] = char_dict.get('style_tuning_embedding_config') or {
            'model_type': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'config': {'device': 'auto'}
        }

        # Remove sensitive file paths for security but keep the boolean flags
        char_dict.pop('image_path', None)
        char_dict.pop('knowledge_base_path', None)
        char_dict.pop('voice_cloning_audio_path', None)
        char_dict.pop('style_tuning_data_path', None)

        return jsonify({"character": char_dict, "status": "success"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-character-wakeword/<int:character_id>', methods=['GET'])
def get_character_wakeword(character_id):
    """Get the wakeword for a specific character."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT wakeword FROM characters WHERE id = %s", (character_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if not result:
            return jsonify({"error": "Character not found"}), 404

        wakeword = result['wakeword'] or f"hey character"
        
        return jsonify({
            "wakeword": wakeword,
            "character_id": character_id,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-character-thinking-audio/<int:character_id>', methods=['GET'])
def get_character_thinking_audio(character_id):
    """Get a random thinking audio for a specific character."""
    try:
        import random
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT thinking_audio_base64 FROM characters WHERE id = %s", (character_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if not result:
            return jsonify({"error": "Character not found"}), 404

        thinking_audio_data = result['thinking_audio_base64']
        
        if not thinking_audio_data:
            return jsonify({"error": "No thinking audio available for this character"}), 404
        
        # Parse JSON if it's a string
        if isinstance(thinking_audio_data, str):
            thinking_audio_data = json.loads(thinking_audio_data)
        
        # Select a random thinking audio
        audio_keys = list(thinking_audio_data.keys())
        if not audio_keys:
            return jsonify({"error": "No thinking audio available for this character"}), 404
        
        selected_key = random.choice(audio_keys)
        selected_audio = thinking_audio_data[selected_key]
        
        return jsonify({
            "audio_base64": selected_audio,
            "phrase_id": selected_key,
            "character_id": character_id,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/load-character', methods=['POST'])
def load_character():
    """
    Preload all models associated with a character and unload unused models.
    Expects: {"character_id": int}
    """
    try:
        data = request.get_json()
        character_id = data.get('character_id')

        if not character_id:
            return jsonify({"error": "character_id is required"}), 400

        # Get character data
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()
        cur.close()
        conn.close()

        if not character:
            return jsonify({"error": "Character not found"}), 404

        character_name = character['name']

        print(f"Loading character: {character_name}")
        print("Cleaning up unused models...")

        # Get current loaded models info before cleanup
        from libraries.tts.inference import get_loaded_models
        from libraries.llm.inference import get_cached_models_info

        current_tts_models = get_loaded_models()
        current_llm_models = get_cached_models_info()

        print(
            f"Currently loaded TTS models: {list(current_tts_models.keys())}")
        print(
            f"Currently loaded LLM models: {list(current_llm_models.keys())}")

        # Determine what models this character needs
        needed_tts_model = None
        needed_llm_cache_key = f"{character_name}_llm"

        if character['voice_cloning_settings']:
            voice_settings = character['voice_cloning_settings']
            needed_tts_model = voice_settings.get('model', 'f5tts')

        # Unload TTS models that aren't needed
        from libraries.tts.inference import unload_models as unload_tts_models
        if needed_tts_model:
            # Check if the needed model is already loaded
            if is_model_loaded(needed_tts_model):
                print(
                    f"TTS model {needed_tts_model} already loaded, keeping it")
            else:
                # Unload current models and load the needed one
                if any(current_tts_models.values()):
                    print(
                        f"Unloading existing TTS models to load {needed_tts_model}")
                    unload_tts_models()
        else:
            # No TTS needed, unload all TTS models
            if any(current_tts_models.values()):
                print("No TTS needed for this character, unloading all TTS models")
                unload_tts_models()

        # Unload LLM models that aren't needed for this character
        from libraries.llm.inference import unload_cached_model
        for cache_key in list(current_llm_models.keys()):
            if cache_key != needed_llm_cache_key:
                print(f"Unloading unused LLM model: {cache_key}")
                unload_cached_model(cache_key)
                # Also remove from legacy model_cache
                if cache_key in model_cache:
                    del model_cache[cache_key]

        # Load LLM model for this character
        if character['llm_model'] and character['llm_config']:
            try:
                llm_config = character['llm_config']

                # Resolve model path and type
                model_path, model_type_str = resolve_model_path(
                    character['llm_model'])

                # Determine model type - prioritize model path resolution over config
                if model_type_str == "gguf":
                    model_type = ModelType.GGUF
                elif model_type_str == "openai_api":
                    model_type = ModelType.OPENAI_API
                elif model_type_str == "huggingface":
                    model_type = ModelType.HUGGINGFACE
                elif 'api_key' in llm_config:
                    # Only use API if model type is ambiguous and api_key is present
                    model_type = ModelType.OPENAI_API
                else:
                    model_type = ModelType.HUGGINGFACE

                print(
                    f"Preloading LLM model '{character['llm_model']}' (resolved to: {model_path}) for {character_name}...")

                # Preload the model into cache
                model = preload_llm_model(
                    model_type=model_type,
                    model_config={
                        'model_path': model_path,
                        **llm_config
                    },
                    cache_key=needed_llm_cache_key
                )

                # Keep backward compatibility with existing model_cache
                model_cache[needed_llm_cache_key] = model
                print(f"✓ LLM model preloaded and ready for {character_name}")

            except Exception as e:
                print(f"Failed to preload LLM model for {character_name}: {e}")
                print(f"  Text generation may be slower due to model re-initialization")

        # Load TTS model for this character
        if character['voice_cloning_settings'] and needed_tts_model:
            try:
                # Only load if not already loaded
                if not is_model_loaded(needed_tts_model):
                    print(
                        f"Preloading TTS model '{needed_tts_model}' for {character_name}...")

                    # First ensure the model is downloaded/available
                    success = ensure_model_available(needed_tts_model)
                    if success:
                        # Special handling for Zonos - preload persistent worker
                        if needed_tts_model.lower() == 'zonos':
                            try:
                                from libraries.tts.inference import preload_zonos_worker
                                voice_settings = character['voice_cloning_settings']
                                zonos_model = voice_settings.get('zonos_model', 'Zyphra/Zonos-v0.1-transformer')
                                device = voice_settings.get('torch_device', 'auto')
                                
                                print(f"🚀 Preloading Zonos worker for model: {zonos_model}")
                                zonos_success = preload_zonos_worker(zonos_model, device)
                                
                                if zonos_success:
                                    print(f"✓ Zonos worker preloaded successfully for {character_name}")
                                    model_cache[f"{character_name}_zonos_ready"] = True
                                else:
                                    print(f"⚠ Warning: Zonos worker preload failed for {character_name}")
                            except Exception as zonos_error:
                                print(f"⚠ Warning: Zonos worker preload error for {character_name}: {zonos_error}")
                        else:
                            # Use smart preloading for other TTS models
                            print(
                                f"Loading TTS model '{needed_tts_model}' into memory...")
                            preload_models_smart([needed_tts_model])

                        # Mark TTS model as ready for this character
                        model_cache[f"{character_name}_tts_ready"] = True
                        print(
                            f"✓ TTS model {needed_tts_model} preloaded and ready for {character_name}")
                    else:
                        print(
                            f"⚠ Warning: TTS model {needed_tts_model} preparation failed for {character_name}")
                        print(
                            f"  Audio generation may be slower due to model re-initialization")
                else:
                    print(
                        f"✓ TTS model {needed_tts_model} already loaded and ready for {character_name}")
                    model_cache[f"{character_name}_tts_ready"] = True

            except Exception as e:
                print(
                    f"✗ Failed to prepare TTS model for {character_name}: {e}")
                print(
                    f"  Character loading will continue, but audio generation may be slower")

        # Get final loaded models info
        try:
            final_tts_models = get_loaded_models()
            final_llm_models = get_cached_models_info()

            print(f"Final loaded TTS models: {list(final_tts_models.keys())}")
            print(f"Final loaded LLM models: {list(final_llm_models.keys())}")

            # Convert model info to JSON-serializable format
            tts_models_serializable = {}
            for key, value in final_tts_models.items():
                if value is not None:
                    tts_models_serializable[key] = str(type(value).__name__)
                else:
                    tts_models_serializable[key] = None

            llm_models_serializable = {}
            for key, value in final_llm_models.items():
                if value is not None:
                    llm_models_serializable[key] = str(type(value).__name__)
                else:
                    llm_models_serializable[key] = None

            return jsonify({
                "message": f"Models loaded for character {character_name}",
                "character_id": character_id,
                "character_name": character_name,
                "loaded_models": {
                    "tts": tts_models_serializable,
                    "llm": llm_models_serializable
                },
                "status": "success"
            }), 200

        except Exception as model_info_error:
            print(f"Error getting model info: {model_info_error}")
            # Return success without detailed model info
            return jsonify({
                "message": f"Models loaded for character {character_name}",
                "character_id": character_id,
                "character_name": character_name,
                "status": "success"
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-question-text', methods=['POST'])
def ask_question_text():
    """
    Process text question and return styled text response with generated audio as base64.
    Expects: {"character_id": int, "question": str}
    """
    try:
        data = request.get_json()
        character_id = data.get('character_id')
        question = data.get('question')

        if not character_id or not question:
            return jsonify({"error": "character_id and question are required"}), 400

        print(f"🚀 Processing request with optimized performance")

        # Get character data
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()

        if not character:
            cur.close()
            conn.close()
            return jsonify({"error": "Character not found"}), 404

        character_name = character['name']

        # Get embedding configurations from character record
        kb_embedding_config = character.get('knowledge_base_embedding_config')
        style_embedding_config = character.get('style_tuning_embedding_config')
        
        # Use default embedding config if not set (for backward compatibility)
        default_embedding_config = {
            'model_type': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'config': {'device': 'auto'}
        }
        
        if not kb_embedding_config:
            kb_embedding_config = default_embedding_config
            print(f"🔍 Using default embedding config for knowledge base")
        
        if not style_embedding_config:
            style_embedding_config = default_embedding_config
            print(f"🔍 Using default embedding config for style tuning")

        # Search knowledge base - ALWAYS provide 3 references regardless of fast mode
        knowledge_context = ""
        knowledge_references = []
        try:
            kb_collection = f"{character_name.lower().replace(' ', '')}-knowledge"
            print(f"🔍 Searching knowledge base collection: {kb_collection}")
            kb_result = query_collection(
                kb_collection, question, n_results=3, return_structured=True, embedding_config=kb_embedding_config)

            if isinstance(kb_result, dict) and "references" in kb_result:
                knowledge_context = kb_result.get("context", "")
                knowledge_references = kb_result.get("references", [])
                print(f"🔍 Knowledge base search successful - found {len(knowledge_references)} references")
                print(f"🔍 Knowledge context length: {len(knowledge_context)} characters")
                if knowledge_context:
                    print(f"🔍 Knowledge context preview: {knowledge_context[:200]}...")
            else:
                # Fallback to string format for backward compatibility
                knowledge_context = str(kb_result)
                print(f"🔍 Knowledge base search returned string format: {len(knowledge_context)} characters")
        except Exception as e:
            print(f"Knowledge base search failed: {e}")
            print(f"🔍 Collection name attempted: {kb_collection}")
            print(f"🔍 Question: {question}")
        
        # Get style examples - Use optimal number for speed while maintaining quality
        style_examples = []
        try:
            style_examples = get_style_data(
                question, character_name, num_examples=2, embedding_config=style_embedding_config)
        except Exception as e:
            print(f"Style data retrieval failed: {e}")

        # Generate styled text response
        styled_response = ""
        if character['llm_model'] and character['llm_config']:
            try:
                # Resolve model path for text generation
                model_path, _ = resolve_model_path(character['llm_model'])
                styled_response = generate_styled_text(
                    question,
                    style_examples,
                    knowledge_context,
                    model_path,
                    character['llm_config'],
                    character_name=character_name
                )
            except Exception as e:
                print(f"Text generation failed: {e}")
                styled_response = "I'm sorry, I'm having trouble generating a response right now."

        # Generate audio response as base64 with optimized performance
        audio_base64 = None
        audio_generation_start = time.time()
        if character['voice_cloning_audio_path'] and character['voice_cloning_settings']:
            try:
                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                # Use the stored reference text from the character
                ref_text = character.get('voice_cloning_reference_text', '')
                if not ref_text:
                    # Fallback if no reference text was stored
                    ref_text = "Hello, how can I help you?"
                    print(
                        "Warning: No reference text found for character, using fallback")

                print(f"🎤 API TTS Request for Character '{character_name}':")
                print(f"   • TTS Model: {tts_model}")
                print(f"   • Reference Audio: {character['voice_cloning_audio_path']}")
                print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
                print(f"   • Generation Text: {styled_response[:100]}{'...' if len(styled_response) > 100 else ''}")
                print(f"   • Voice Settings: {voice_settings}")

                # Initialize similarity_score variable
                similarity_score = None

                # Always use similarity calculation method to ensure we get similarity scores
                print("🎵 Using TTS generation with similarity calculation")
                
                # Generate audio with similarity calculation
                generation_result = generate_audio_with_similarity(
                    model=tts_model,
                    ref_audio=character['voice_cloning_audio_path'],
                    ref_text=ref_text,
                    gen_text=styled_response,
                    config=voice_settings,
                    fast_mode=True,
                    calculate_similarity=True
                )
                
                # Extract audio path and similarity score
                if generation_result and generation_result.get('audio_path'):
                    # Convert audio file to base64
                    from libraries.tts.inference import _audio_file_to_base64
                    audio_base64 = _audio_file_to_base64(generation_result['audio_path'])
                    similarity_score = generation_result.get('similarity_score')
                    print(f"🎯 Similarity score calculated: {similarity_score}")
                else:
                    audio_base64 = None
                    similarity_score = None

            except Exception as e:
                print(f"Audio generation failed: {e}")
                audio_base64 = None
                similarity_score = None

        # Store chat history in database
        try:
            cur.execute("""
                INSERT INTO chat_history 
                (character_id, user_message, bot_response, audio_base64, knowledge_context, knowledge_references, similarity_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                character_id,
                question,
                styled_response,
                audio_base64,
                knowledge_context,
                json.dumps(knowledge_references),  # Store as JSON
                similarity_score  # Add similarity score to database
            ))
            conn.commit()
        except Exception as e:
            print(f"Failed to store chat history: {e}")
            # Try without similarity_score and knowledge_references columns for backward compatibility
            try:
                cur.execute("""
                    INSERT INTO chat_history 
                    (character_id, user_message, bot_response, audio_base64, knowledge_context)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    character_id,
                    question,
                    styled_response,
                    audio_base64,
                    knowledge_context
                ))
                conn.commit()
            except Exception as e2:
                print(f"Failed to store chat history (fallback): {e2}")

        cur.close()
        conn.close()

        # Calculate audio generation time for performance monitoring
        audio_generation_time = time.time() - audio_generation_start if 'audio_generation_start' in locals() else 0
        
        # Add performance indicators to response
        response_data = {
            "question": question,
            "text_response": styled_response,
            "audio_base64": audio_base64,
            "knowledge_context": knowledge_context,
            "knowledge_references": knowledge_references,
            "character_name": character_name,
            "status": "success",
            "audio_generation_time": round(audio_generation_time, 3),
            "similarity_score": similarity_score
        }
        
        print(f"🚀 Optimized response delivered for {character_name} - Audio gen: {audio_generation_time:.3f}s")
        
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe-audio', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio to text using optimized Whisper with comprehensive device optimization.
    Expects: multipart/form-data with 'audio_file' or 'audio'
    Returns: {"transcript": str, "status": "success"}
    """
    try:
        # Accept both 'audio_file' and 'audio' parameter names for compatibility
        audio_file = request.files.get('audio_file') or request.files.get('audio')

        if not audio_file:
            return jsonify({"error": "audio_file or audio is required"}), 400

        # Save uploaded audio file temporarily
        temp_audio_path = None
        try:
            import tempfile
            import os

            # Create temporary file
            temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.webm')
            os.close(temp_fd)  # Close the file descriptor

            # Save the uploaded audio
            audio_file.save(temp_audio_path)

            # Transcribe audio using optimized function
            transcript = _transcribe_audio_optimized(temp_audio_path)

            if not transcript:
                return jsonify({"error": "No speech detected in audio"}), 400

        except Exception as e:
            print(f"Audio processing failed: {e}")
            return jsonify({"error": "Failed to process audio file"}), 500
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

        return jsonify({
            "transcript": transcript,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-question-audio', methods=['POST'])
def ask_question_audio():
    """
    Process audio question by transcribing it to text and returning styled text response with generated audio as base64.
    Expects: multipart/form-data with 'character_id' and 'audio_file'
    """
    try:
        character_id = request.form.get('character_id')
        audio_file = request.files.get('audio_file')

        if not character_id or not audio_file:
            return jsonify({"error": "character_id and audio_file are required"}), 400

        character_id = int(character_id)

        # Get character data
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()

        if not character:
            cur.close()
            conn.close()
            return jsonify({"error": "Character not found"}), 404

        character_name = character['name']

        # Save uploaded audio file temporarily
        temp_audio_path = None
        try:
            import tempfile
            import os

            # Create temporary file
            temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.webm')
            os.close(temp_fd)  # Close the file descriptor

            # Save the uploaded audio
            audio_file.save(temp_audio_path)

            # Transcribe audio to text using Whisper
            transcript = _transcribe_audio_optimized(temp_audio_path)

        except Exception as e:
            print(f"Audio processing failed: {e}")
            return jsonify({"error": "Failed to process audio file"}), 500
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

        # Now process the transcribed text like the text endpoint
        question = transcript

        # Get embedding configurations from character record
        kb_embedding_config = character.get('knowledge_base_embedding_config')
        style_embedding_config = character.get('style_tuning_embedding_config')
        
        # Use default embedding config if not set (for backward compatibility)
        default_embedding_config = {
            'model_type': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'config': {'device': 'auto'}
        }
        
        if not kb_embedding_config:
            kb_embedding_config = default_embedding_config
        
        if not style_embedding_config:
            style_embedding_config = default_embedding_config

        # Search knowledge base
        knowledge_context = ""
        knowledge_references = []
        try:
            kb_collection = f"{character_name.lower().replace(' ', '')}-knowledge"
            kb_result = query_collection(
                kb_collection, question, n_results=3, return_structured=True, embedding_config=kb_embedding_config)

            if isinstance(kb_result, dict) and "references" in kb_result:
                knowledge_context = kb_result.get("context", "")
                knowledge_references = kb_result.get("references", [])
            else:
                # Fallback to string format for backward compatibility
                knowledge_context = str(kb_result)
        except Exception as e:
            print(f"Knowledge base search failed: {e}")

        # Get style examples
        style_examples = []
        try:
            style_examples = get_style_data(
                question, character_name, num_examples=2, embedding_config=style_embedding_config)
        except Exception as e:
            print(f"Style data retrieval failed: {e}")

        # Generate styled text response
        styled_response = ""
        if character['llm_model'] and character['llm_config']:
            try:
                # Resolve model path for text generation
                model_path, _ = resolve_model_path(character['llm_model'])
                styled_response = generate_styled_text(
                    question,
                    style_examples,
                    knowledge_context,
                    model_path,
                    character['llm_config'],
                    character_name=character_name
                )
            except Exception as e:
                print(f"Text generation failed: {e}")
                styled_response = "I'm sorry, I'm having trouble generating a response right now."

        # Generate audio response as base64 with optimized performance
        audio_base64 = None
        if character['voice_cloning_audio_path'] and character['voice_cloning_settings']:
            try:
                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                # Use the stored reference text from the character
                ref_text = character.get('voice_cloning_reference_text', '')
                if not ref_text:
                    # Fallback if no reference text was stored
                    ref_text = "Hello, how can I help you?"
                    print(
                        "Warning: No reference text found for character, using fallback")

                print(f"🎤 API Audio TTS Request for Character '{character_name}':")
                print(f"   • TTS Model: {tts_model}")
                print(f"   • Reference Audio: {character['voice_cloning_audio_path']}")
                print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
                print(f"   • Generation Text: {styled_response[:100]}{'...' if len(styled_response) > 100 else ''}")
                print(f"   • Voice Settings: {voice_settings}")

                # Initialize similarity_score variable
                similarity_score = None

                # Always use similarity calculation method to ensure we get similarity scores
                print("🎵 Using TTS generation with similarity calculation")
                
                # Generate audio with similarity calculation
                generation_result = generate_audio_with_similarity(
                    model=tts_model,
                    ref_audio=character['voice_cloning_audio_path'],
                    ref_text=ref_text,
                    gen_text=styled_response,
                    config=voice_settings,
                    fast_mode=True,
                    calculate_similarity=True
                )
                
                # Extract audio path and similarity score
                if generation_result and generation_result.get('audio_path'):
                    # Convert audio file to base64
                    from libraries.tts.inference import _audio_file_to_base64
                    audio_base64 = _audio_file_to_base64(generation_result['audio_path'])
                    similarity_score = generation_result.get('similarity_score')
                    print(f"🎯 Similarity score calculated: {similarity_score}")
                else:
                    audio_base64 = None
                    similarity_score = None

            except Exception as e:
                print(f"Audio generation failed: {e}")
                audio_base64 = None
                similarity_score = None

        # Store chat history in database
        try:
            cur.execute("""
                INSERT INTO chat_history 
                (character_id, user_message, bot_response, audio_base64, knowledge_context, knowledge_references, similarity_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                character_id,
                question,
                styled_response,
                audio_base64,
                knowledge_context,
                json.dumps(knowledge_references),  # Store as JSON
                similarity_score  # Add similarity score to database
            ))
            conn.commit()
        except Exception as e:
            print(f"Failed to store chat history: {e}")
            # Try without similarity_score and knowledge_references columns for backward compatibility
            try:
                cur.execute("""
                    INSERT INTO chat_history 
                    (character_id, user_message, bot_response, audio_base64, knowledge_context)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    character_id,
                    question,
                    styled_response,
                    audio_base64,
                    knowledge_context
                ))
                conn.commit()
            except Exception as e2:
                print(f"Failed to store chat history (fallback): {e2}")

        cur.close()
        conn.close()

        return jsonify({
            "transcript": transcript,
            "question": question,
            "text_response": styled_response,
            "audio_base64": audio_base64,
            "knowledge_context": knowledge_context,
            "knowledge_references": knowledge_references,
            "character_name": character_name,
            "status": "success",
            "similarity_score": similarity_score
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download-model', methods=['POST'])
def download_model_endpoint():
    """
    Download a model using the preprocess module.
    Expects POST requests with 'model_name', 'model_type', and optional 'custom_name' parameters.
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        model_type = data.get('model_type')
        custom_name = data.get('custom_name')  # Optional custom name for the model

        if not model_name or not model_type:
            return jsonify({"error": "model_name and model_type are required"}), 400

        print(f"🔽 Download request received:")
        print(f"  Model name: {model_name}")
        print(f"  Model type: {model_type}")
        print(f"  Custom name: {custom_name}")

        # Download the model
        try:
            model_path = download_model_func(model_name, model_type, custom_name=custom_name)
            print(f"✅ Download completed successfully: {model_path}")
            return jsonify({
                "model_path": model_path, 
                "custom_name": custom_name,
                "status": "success"
            }), 200
        except Exception as e:
            print(f"❌ Download failed: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"❌ Request processing failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get-llm-models', methods=['GET'])
def get_llm_models():
    """
    Get list of available LLM models with their availability status.
    """
    try:
        from libraries.llm.inference import list_available_models
        import os
        import glob

        # Get list of available models
        available_models = list_available_models()

        # Define the models we support in the frontend
        supported_models = [
            {"id": "google-gemma-3-4b-it-qat-q4_0-gguf", "name": "Gemma3 4B", "type": "gguf",
                "path": "./models/gemma-3-4b-it-q4_0.gguf",
                "repo": "google/gemma-3-4b-it-qat-q4_0-gguf:gemma-3-4b-it-q4_0.gguf"},
            {"id": "llama-3.2-3b", "name": "Llama 3.2 3B",
                "type": "huggingface", "repo": "meta-llama/Llama-3.2-3B"},
            {"id": "gpt-4o", "name": "GPT-4o",
                "type": "openai_api", "repo": "gpt-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o-mini",
                "type": "openai_api", "repo": "gpt-4o-mini"},
        ]

        # Check availability for each model
        model_info = []
        for model in supported_models:
            model_data = {
                "id": model["id"],
                "name": model["name"],
                "type": model["type"],
                "repo": model.get("repo", ""),
                "path": model.get("path", ""),
                "requiresKey": model["type"] == "openai_api",
                "available": False,
                "downloaded": False
            }

            if model["type"] == "openai_api":
                # API models are always "available" but need keys
                model_data["available"] = True
                model_data["downloaded"] = True
            elif model["type"] == "gguf":
                # Check if GGUF file exists locally
                gguf_path = model.get("path", "")
                if gguf_path and os.path.exists(gguf_path):
                    model_data["available"] = True
                    model_data["downloaded"] = True
            elif model["type"] == "huggingface":
                # Check if model is downloaded locally
                models_dir = "./models"  # Adjust path as needed
                if os.path.exists(models_dir):
                    # Check for GGUF file
                    model_filename = f"{model['repo'].replace('/', '_')}.gguf"
                    model_path = os.path.join(models_dir, model_filename)
                    if os.path.exists(model_path):
                        model_data["available"] = True
                        model_data["downloaded"] = True
                    else:
                        # Check for directory with model files
                        model_dir = os.path.join(
                            models_dir, model['repo'].replace('/', '_'))
                        if os.path.exists(model_dir) and os.listdir(model_dir):
                            model_data["available"] = True
                            model_data["downloaded"] = True

            model_info.append(model_data)

        # Detect custom models in the models directory
        models_dir = "./models"
        if os.path.exists(models_dir):
            # Track already included models to avoid duplicates
            included_model_paths = set()
            for model in model_info:
                if model.get("path"):
                    included_model_paths.add(os.path.abspath(model["path"]))
            
            # Find all GGUF files in the models directory
            gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
            
            for gguf_file in gguf_files:
                abs_path = os.path.abspath(gguf_file)
                if abs_path not in included_model_paths:
                    filename = os.path.basename(gguf_file)
                    # Generate a custom model ID and name
                    custom_id = f"custom-{filename.replace('.gguf', '').replace('_', '-')}"
                    custom_name = filename.replace('.gguf', '').replace('_', ' ').title()
                    
                    custom_model = {
                        "id": custom_id,
                        "name": f"{custom_name} (Custom)",
                        "type": "gguf",
                        "repo": "",
                        "path": gguf_file,
                        "requiresKey": False,
                        "available": True,
                        "downloaded": True,
                        "custom": True
                    }
                    model_info.append(custom_model)
            
            # Find all directories that might contain HuggingFace models
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains model files
                    has_model_files = False
                    for file in os.listdir(item_path):
                        if file.endswith(('.bin', '.safetensors', '.gguf', '.json')):
                            has_model_files = True
                            break
                    
                    if has_model_files:
                        # Check if this is already included as a supported model
                        already_included = False
                        for existing_model in model_info:
                            if existing_model.get("path") and os.path.abspath(existing_model["path"]) == os.path.abspath(item_path):
                                already_included = True
                                break
                            # Also check if the directory name matches a repo pattern
                            if existing_model.get("repo") and existing_model["repo"].replace('/', '_') == item:
                                already_included = True
                                break
                        
                        if not already_included:
                            # This is a custom model directory
                            custom_id = f"custom-{item.replace('_', '-')}"
                            custom_name = item.replace('_', ' ').title()
                            
                            custom_model = {
                                "id": custom_id,
                                "name": f"{custom_name} (Custom)",
                                "type": "huggingface",
                                "repo": "",
                                "path": item_path,
                                "requiresKey": False,
                                "available": True,
                                "downloaded": True,
                                "custom": True
                            }
                            model_info.append(custom_model)

        return jsonify({
            "models": model_info,
            "status": "success"
        }), 200

    except Exception as e:
        print(f"Error in get_llm_models: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/get-tts-models', methods=['GET'])
def get_tts_models():
    """
    Get list of available TTS model options.
    """
    try:
        from libraries.tts.inference import get_supported_models
        from libraries.tts.preprocess import check_voice_model_availability

        # Get supported model options
        supported_models = get_supported_models()

        # Get availability status
        availability = check_voice_model_availability()

        # Map simple model names to availability info
        model_info = []
        for model in supported_models:
            if model == "f5tts":
                status = availability.get("F5TTS", {})
            elif model == "xtts":
                status = availability.get("XTTS-v2", {})
            elif model == "zonos":
                status = availability.get("Zonos", {})
            else:
                status = {"available": False, "dependencies": []}

            model_info.append({
                "name": model,
                "available": status.get("available", False),
                "dependencies": status.get("dependencies", [])
            })

        return jsonify({
            "models": model_info,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/serve-audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from the storage directory."""
    try:
        file_path = STORAGE_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "Audio file not found"}), 404

        return send_file(str(file_path), as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/serve-image/<path:filename>')
def serve_image(filename):
    """Serve image files from the storage directory."""
    try:
        file_path = STORAGE_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "Image file not found"}), 404

        return send_file(str(file_path), as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-loaded-models', methods=['GET'])
def get_loaded_models_endpoint():
    """
    Get information about which TTS models are currently loaded in memory.
    """
    try:
        from libraries.tts.inference import get_loaded_models
        loaded_models = get_loaded_models()

        return jsonify({
            "loaded_models": loaded_models,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/unload-models', methods=['POST'])
def unload_models_endpoint():
    """
    Unload all TTS models from memory to free up resources.
    """
    try:
        from libraries.tts.inference import unload_models
        unload_models()

        return jsonify({
            "message": "All TTS models unloaded from memory",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-loaded-llm-models', methods=['GET'])
def get_loaded_llm_models_endpoint():
    """
    Get information about which LLM models are currently loaded in memory.
    """
    try:
        from libraries.llm.inference import get_cached_models_info
        cached_models = get_cached_models_info()

        return jsonify({
            "cached_models": cached_models,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/unload-llm-models', methods=['POST'])
def unload_llm_models_endpoint():
    """
    Unload all LLM models from memory to free up resources.
    """
    try:
        from libraries.llm.inference import unload_all_cached_models
        unload_all_cached_models()

        return jsonify({
            "message": "All LLM models unloaded from memory",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/unload-all-models', methods=['POST'])
def unload_all_models_endpoint():
    """
    Unload all models (both TTS and LLM) from memory to free up resources.
    """
    try:
        from libraries.tts.inference import unload_models
        from libraries.llm.inference import unload_all_cached_models

        # Unload TTS models
        unload_models()

        # Unload LLM models
        unload_all_cached_models()

        return jsonify({
            "message": "All models (TTS and LLM) unloaded from memory",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-chat-history/<int:character_id>', methods=['GET'])
def get_chat_history(character_id):
    """
    Get chat history for a specific character.
    Optional query parameters: limit (default 50), offset (default 0)
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Validate limit
        if limit > 100:
            limit = 100

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if character exists
        cur.execute("SELECT name FROM characters WHERE id = %s",
                    (character_id,))
        character = cur.fetchone()
        if not character:
            cur.close()
            conn.close()
            return jsonify({"error": "Character not found"}), 404

        # Get chat history
        cur.execute("""
            SELECT id, user_message, bot_response, audio_base64, knowledge_context, knowledge_references, similarity_score, created_at
            FROM chat_history 
            WHERE character_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s OFFSET %s
        """, (character_id, limit, offset))

        history = cur.fetchall()
        cur.close()
        conn.close()

        # Convert to JSON serializable format
        result = []
        for entry in history:
            entry_dict = dict(entry)
            # Convert datetime to string
            entry_dict['created_at'] = entry_dict['created_at'].isoformat()

            # Parse knowledge_references from JSON if available
            if entry_dict.get('knowledge_references'):
                try:
                    # If it's already a dict/list (from JSONB), keep it as is
                    if isinstance(entry_dict['knowledge_references'], (dict, list)):
                        pass  # Already parsed
                    else:
                        # If it's a string, parse it
                        entry_dict['knowledge_references'] = json.loads(
                            entry_dict['knowledge_references'])
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Warning: Failed to parse knowledge_references: {e}")
                    entry_dict['knowledge_references'] = []
            else:
                entry_dict['knowledge_references'] = []

            result.append(entry_dict)

        return jsonify({
            "character_id": character_id,
            "character_name": character['name'],
            "chat_history": result,
            "total_messages": len(result),
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/clear-chat-history/<int:character_id>', methods=['DELETE'])
def clear_chat_history(character_id):
    """Clear all chat history for a specific character."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if character exists
        cur.execute("SELECT name FROM characters WHERE id = %s",
                    (character_id,))
        character = cur.fetchone()
        if not character:
            cur.close()
            conn.close()
            return jsonify({"error": "Character not found"}), 404

        # Delete chat history
        cur.execute(
            "DELETE FROM chat_history WHERE character_id = %s", (character_id,))
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": f"Cleared {deleted_count} messages from chat history",
            "character_id": character_id,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete-character/<int:character_id>', methods=['DELETE'])
def delete_character(character_id):
    """Delete a character and all associated data."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get character data first
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()
        if not character:
            cur.close()
            conn.close()
            return jsonify({"error": "Character not found"}), 404

        character_name = character['name']

        # Delete associated collections
        try:
            collection_deletion_result = delete_character_collections(character_name)
            if collection_deletion_result["knowledge_base"]["deleted"]:
                print(f"✅ Deleted knowledge base collection for {character_name}")
            if collection_deletion_result["style_tuning"]["deleted"]:
                print(f"✅ Deleted style tuning collection for {character_name}")
        except Exception as e:
            print(f"Warning: Failed to delete character collections: {e}")

        # Delete associated files
        try:
            character_dir = STORAGE_DIR / \
                character_name.replace(' ', '_').lower()
            if character_dir.exists():
                shutil.rmtree(character_dir)
                print(f"Deleted character directory: {character_dir}")
        except Exception as e:
            print(f"Warning: Failed to delete character files: {e}")

        # Delete from database (chat history will be deleted automatically due to CASCADE)
        cur.execute("DELETE FROM characters WHERE id = %s", (character_id,))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": f"Character '{character_name}' deleted successfully",
            "character_id": character_id,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update-character/<int:character_id>', methods=['PUT'])
def update_character(character_id):
    """
    Update a character's information and associated data.
    Expects form data with files and JSON configuration.
    Only processes files when they have actually changed.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get existing character data
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()
        if not character:
            cur.close()
            conn.close()
            return jsonify({"error": "Character not found"}), 404

        old_name = character['name']
        old_character_dir = STORAGE_DIR / old_name.replace(' ', '_').lower()

        # Get form data
        new_name = request.form.get('name', old_name)
        llm_model = request.form.get('llm_model', character['llm_model'])
        llm_config = json.loads(request.form.get(
            'llm_config', json.dumps(character['llm_config'] or {})))
        voice_cloning_settings = json.loads(request.form.get(
            'voice_cloning_settings', json.dumps(character['voice_cloning_settings'] or {})))
        # Normalize voice cloning settings to ensure proper data types
        voice_cloning_settings = normalize_voice_cloning_settings(voice_cloning_settings)
        wakeword = request.form.get('wakeword', character.get('wakeword', f"hey {new_name.lower()}"))

        # Get embedding configurations
        knowledge_base_embedding_config = json.loads(
            request.form.get('knowledge_base_embedding_config', '{}'))
        style_tuning_embedding_config = json.loads(
            request.form.get('style_tuning_embedding_config', '{}'))

        # Use default embedding config if not provided or use existing character configs
        default_embedding_config = {
            'model_type': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'config': {'device': 'auto'}
        }
        
        if not knowledge_base_embedding_config:
            knowledge_base_embedding_config = character.get('knowledge_base_embedding_config') or default_embedding_config
        
        if not style_tuning_embedding_config:
            style_tuning_embedding_config = character.get('style_tuning_embedding_config') or default_embedding_config

        # Extract reference text from form data
        voice_reference_text = request.form.get(
            'voice_cloning_reference_text', character.get('voice_cloning_reference_text', ''))

        # Create new character directory if name changed
        new_character_dir = STORAGE_DIR / new_name.replace(' ', '_').lower()
        if new_name != old_name:
            if old_character_dir.exists():
                shutil.move(str(old_character_dir), str(new_character_dir))
                print(
                    f"Renamed character directory from {old_character_dir} to {new_character_dir}")
            else:
                new_character_dir.mkdir(exist_ok=True)
            
            # Handle collection renaming when character name changes
            try:
                rename_result = rename_character_collections(old_name, new_name)
                if rename_result["knowledge_base"]["renamed"]:
                    print(f"✅ Renamed knowledge base collection for character name change")
                if rename_result["style_tuning"]["renamed"]:
                    print(f"✅ Renamed style tuning collection for character name change")
            except Exception as e:
                print(f"⚠️  Warning: Could not rename collections for character name change: {e}")
        else:
            new_character_dir.mkdir(exist_ok=True)

        # Initialize paths with existing values
        image_path = character['image_path']
        knowledge_base_path = character['knowledge_base_path']
        voice_cloning_audio_path = character['voice_cloning_audio_path']
        style_tuning_data_path = character['style_tuning_data_path']

        # Handle file uploads (only update if new files are provided)
        if 'character_image' in request.files:
            image_file = request.files['character_image']
            if image_file.filename:
                # Delete old image if it exists
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)

                image_path = str(new_character_dir /
                                 f"image_{image_file.filename}")
                image_file.save(image_path)
                print(f"Updated character image for {new_name}")

        # Update knowledge base ONLY if new files are provided
        kb_files = request.files.getlist('knowledge_base_file')
        if kb_files and any(f.filename for f in kb_files):
            print(f"🔄 New knowledge base files detected for {new_name} - processing...")
            try:
                # Delete old knowledge base files if they exist
                if knowledge_base_path and os.path.exists(knowledge_base_path):
                    os.remove(knowledge_base_path)
                
                # Also clean up any existing knowledge base files and manifest
                kb_pattern = new_character_dir / "knowledge_base_*"
                import glob
                for old_kb_file in glob.glob(str(kb_pattern)):
                    if os.path.exists(old_kb_file):
                        os.remove(old_kb_file)
                
                manifest_path = new_character_dir / "knowledge_base_manifest.json"
                if manifest_path.exists():
                    os.remove(manifest_path)

                collection_name = f"{new_name.lower().replace(' ', '')}-knowledge"

                # Create temporary directories for processing
                kb_docs_dir = new_character_dir / "kb_docs"
                kb_archive_dir = new_character_dir / "kb_archive"
                
                # Clean up existing processing directories
                if kb_docs_dir.exists():
                    shutil.rmtree(kb_docs_dir)
                if kb_archive_dir.exists():
                    shutil.rmtree(kb_archive_dir)
                    
                kb_docs_dir.mkdir(exist_ok=True)
                kb_archive_dir.mkdir(exist_ok=True)

                # Process all knowledge base files
                kb_file_paths = []
                for i, kb_file in enumerate(kb_files):
                    if kb_file.filename:
                        # Save each file
                        kb_file_path = str(new_character_dir / f"knowledge_base_{i+1}_{kb_file.filename}")
                        kb_file.save(kb_file_path)
                        kb_file_paths.append(kb_file_path)

                        # Copy the file to the docs directory for processing
                        temp_kb_path = kb_docs_dir / kb_file.filename
                        shutil.copy2(kb_file_path, temp_kb_path)

                # Store the first file path for backward compatibility (or create a manifest)
                if kb_file_paths:
                    knowledge_base_path = kb_file_paths[0]  # Store first file path
                    # Create a manifest file listing all uploaded files
                    with open(manifest_path, 'w') as f:
                        json.dump({
                            "files": [os.path.basename(path) for path in kb_file_paths],
                            "count": len(kb_file_paths),
                            "created_at": time.time()
                        }, f)

                # Process all documents in the directory (this will replace the existing collection)
                process_documents_for_collection(
                    str(kb_docs_dir), str(kb_archive_dir), collection_name, knowledge_base_embedding_config, force_recreate=True)
                
                print(f"✅ Successfully updated and processed {len(kb_file_paths)} knowledge base files for {new_name}")
                
                # Ensure collection is compatible with new embedding model
                compatibility_result = ensure_character_collections_compatible(
                    new_name,
                    knowledge_base_embedding_config=knowledge_base_embedding_config,
                    character_dir=str(new_character_dir)
                )
                
                if compatibility_result["knowledge_base"]["checked"]:
                    kb_result = compatibility_result["knowledge_base"]
                    if kb_result.get("action") == "recreated":
                        print(f"✅ Knowledge base collection automatically recreated with new embedding model")
                    elif kb_result.get("action") == "failed":
                        print(f"⚠️  Warning: Could not recreate knowledge base collection with new embedding model")
                    elif kb_result.get("action") == "manual_required":
                        print(f"⚠️  Warning: Knowledge base collection needs manual recreation")
            except Exception as e:
                print(f"❌ Warning: Knowledge base processing failed: {e}")
        else:
            print(f"ℹ️  No new knowledge base files provided for {new_name} - keeping existing knowledge base")
            
            # Check if embedding model changed even without new files
            old_kb_config = character.get('knowledge_base_embedding_config')
            if old_kb_config != knowledge_base_embedding_config:
                print(f"🔄 Knowledge base embedding model changed - checking compatibility...")
                
                try:
                    # Ensure kb_docs directory exists and has source files for recreation
                    kb_docs_dir = new_character_dir / "kb_docs"
                    kb_archive_dir = new_character_dir / "kb_archive"
                    
                    # Create directories if they don't exist
                    kb_docs_dir.mkdir(exist_ok=True)
                    kb_archive_dir.mkdir(exist_ok=True)
                    
                    # Check if kb_docs is empty and try to populate it from existing files
                    if not any(kb_docs_dir.iterdir()):
                        print(f"📂 kb_docs directory is empty, populating from existing knowledge base files...")
                        
                        # Look for existing knowledge base files in character directory
                        kb_pattern = new_character_dir / "knowledge_base_*"
                        import glob
                        existing_kb_files = glob.glob(str(kb_pattern))
                        
                        if existing_kb_files:
                            for kb_file_path in existing_kb_files:
                                if os.path.exists(kb_file_path):
                                    filename = os.path.basename(kb_file_path)
                                    # Remove the "knowledge_base_N_" prefix to get original filename
                                    original_filename = filename.split('_', 2)[-1] if '_' in filename else filename
                                    dest_path = kb_docs_dir / original_filename
                                    shutil.copy2(kb_file_path, dest_path)
                                    print(f"📄 Copied {filename} to kb_docs for recreation")
                        else:
                            print(f"⚠️  No existing knowledge base files found for recreation")
                    
                    embedding_change_result = handle_character_embedding_model_change(
                        new_name,
                        old_kb_config,
                        knowledge_base_embedding_config,
                        "knowledge",
                        str(new_character_dir)
                    )
                    
                    if embedding_change_result["success"]:
                        if embedding_change_result["action"] == "recreated":
                            print(f"✅ Knowledge base collection automatically recreated with new embedding model")
                        elif embedding_change_result["action"] == "none":
                            print(f"ℹ️  Knowledge base embedding model unchanged")
                    else:
                        print(f"⚠️  Warning: Could not handle knowledge base embedding model change: {embedding_change_result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"⚠️  Warning: Error checking knowledge base embedding model change: {e}")

        # Check if preprocessing settings have changed
        old_voice_settings = character.get('voice_cloning_settings') or {}
        preprocessing_changed = False
        
        # Compare preprocessing-related settings with proper defaults
        preprocessing_keys = ['preprocess_audio', 'clean_audio', 'remove_silence', 'enhance_audio', 
                             'skip_all_processing', 'preprocessing_order', 'top_db', 'fade_length_ms',
                             'bass_boost', 'treble_boost', 'compression']
        
        # Define defaults to match what the frontend sends
        preprocessing_defaults = {
            'preprocess_audio': True,
            'clean_audio': True,
            'remove_silence': True,
            'enhance_audio': True,
            'skip_all_processing': False,
            'preprocessing_order': ['clean', 'remove_silence', 'enhance'],
            'top_db': 40.0,
            'fade_length_ms': 50,
            'bass_boost': True,
            'treble_boost': True,
            'compression': True
        }
        
        for key in preprocessing_keys:
            old_value = old_voice_settings.get(key, preprocessing_defaults.get(key))
            new_value = voice_cloning_settings.get(key, preprocessing_defaults.get(key))
            
            # Special handling for preprocessing_order list comparison
            if key == 'preprocessing_order':
                if old_value != new_value:
                    preprocessing_changed = True
                    print(f"🔄 Preprocessing order changed: {old_value} → {new_value}")
                    break
            else:
                if old_value != new_value:
                    preprocessing_changed = True
                    print(f"🔄 Preprocessing setting '{key}' changed: {old_value} → {new_value}")
                    break

        # Update voice cloning audio if new file is provided OR if preprocessing settings changed
        if 'voice_cloning_audio' in request.files:
            voice_file = request.files['voice_cloning_audio']
            if voice_file.filename:
                print(f"🔄 New voice cloning audio detected for {new_name} - processing...")
                
                # Delete old voice files if they exist
                if voice_cloning_audio_path and os.path.exists(voice_cloning_audio_path):
                    os.remove(voice_cloning_audio_path)

                raw_audio_path = str(new_character_dir /
                                     f"voice_raw_{voice_file.filename}")
                voice_file.save(raw_audio_path)

                # Check if audio preprocessing is enabled
                preprocess_audio = voice_cloning_settings.get(
                    'preprocess_audio', True)

                if preprocess_audio:
                    try:
                        # Filter out TTS-only parameters
                        tts_only_params = {'model', 'cache_dir', 'preprocess_audio', 'ref_text', 'reference_text',
                                           'language', 'output_dir', 'cuda_device', 'coqui_tos_agreed',
                                           'torch_force_no_weights_only_load', 'auto_download', 'gen_text',
                                           'generative_text', 'repetition_penalty', 'top_k', 'top_p', 'speed',
                                           'enable_text_splitting', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8',
                                           'seed', 'cfg_scale', 'speaking_rate', 'frequency_max', 'pitch_standard_deviation'}
                        audio_processing_params = {
                            k: v for k, v in voice_cloning_settings.items()
                            if k not in tts_only_params
                        }
                        
                        # Add device optimization for Apple Silicon
                        if DEVICE_OPTIMIZATION_AVAILABLE:
                            device_type, device_info = get_device_info()
                            if device_type == DeviceType.APPLE_SILICON:
                                # Enable safe mode for Apple Silicon to avoid MPS issues with DeepFilterNet
                                print("🍎 Apple Silicon detected - enabling safe mode for audio preprocessing")
                                audio_processing_params['safe_mode'] = True
                        
                        voice_cloning_audio_path = generate_reference_audio(
                            raw_audio_path,
                            output_file=str(
                                new_character_dir / "voice_processed.wav"),
                            **audio_processing_params
                        )
                        print(f"✅ Audio preprocessing completed for {new_name}")
                    except Exception as e:
                        print(f"❌ Warning: Voice preprocessing failed: {e}")
                        print("🔄 Attempting fallback with safe mode...")
                        try:
                            # Retry with safe mode enabled
                            audio_processing_params['safe_mode'] = True
                            voice_cloning_audio_path = generate_reference_audio(
                                raw_audio_path,
                                output_file=str(
                                    new_character_dir / "voice_processed.wav"),
                                **audio_processing_params
                            )
                            print(f"✅ Audio preprocessing completed for {new_name} with safe mode fallback")
                        except Exception as fallback_error:
                            print(f"❌ Warning: Voice preprocessing failed even with safe mode: {fallback_error}")
                            voice_cloning_audio_path = raw_audio_path
                else:
                    voice_cloning_audio_path = raw_audio_path
                    print(f"ℹ️  Audio preprocessing disabled for {new_name} - using raw audio")
            else:
                print(f"ℹ️  No new voice cloning audio provided for {new_name} - keeping existing audio")
        elif preprocessing_changed and voice_cloning_audio_path and os.path.exists(voice_cloning_audio_path):
            print(f"🔄 Voice preprocessing settings changed for {new_name} - reprocessing existing audio...")
            
            # Find the raw audio file to reprocess
            raw_audio_path = None
            character_dir_path = STORAGE_DIR / new_name.replace(' ', '_').lower()
            
            # Look for raw audio file
            for file_pattern in ['voice_raw_*', 'voice_processed.wav']:
                import glob
                matching_files = glob.glob(str(character_dir_path / file_pattern))
                if matching_files:
                    # Use the first raw file found, or the processed file as fallback
                    if 'raw' in file_pattern:
                        raw_audio_path = matching_files[0]
                        break
                    elif not raw_audio_path:  # Use processed as fallback if no raw found
                        raw_audio_path = matching_files[0]
            
            if raw_audio_path and os.path.exists(raw_audio_path):
                # Check if audio preprocessing is enabled
                preprocess_audio = voice_cloning_settings.get('preprocess_audio', True)

                if preprocess_audio:
                    try:
                        # Filter out TTS-only parameters
                        tts_only_params = {'model', 'cache_dir', 'preprocess_audio', 'ref_text', 'reference_text',
                                           'language', 'output_dir', 'cuda_device', 'coqui_tos_agreed',
                                           'torch_force_no_weights_only_load', 'auto_download', 'gen_text',
                                           'generative_text', 'repetition_penalty', 'top_k', 'top_p', 'speed',
                                           'enable_text_splitting', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8',
                                           'seed', 'cfg_scale', 'speaking_rate', 'frequency_max', 'pitch_standard_deviation'}
                        audio_processing_params = {
                            k: v for k, v in voice_cloning_settings.items()
                            if k not in tts_only_params
                        }
                        
                        # Add device optimization for Apple Silicon
                        if DEVICE_OPTIMIZATION_AVAILABLE:
                            device_type, device_info = get_device_info()
                            if device_type == DeviceType.APPLE_SILICON:
                                print("🍎 Apple Silicon detected - enabling safe mode for audio preprocessing")
                                audio_processing_params['safe_mode'] = True
                        
                        voice_cloning_audio_path = generate_reference_audio(
                            raw_audio_path,
                            output_file=str(character_dir_path / "voice_processed.wav"),
                            **audio_processing_params
                        )
                        print(f"✅ Audio reprocessing completed for {new_name}")
                    except Exception as e:
                        print(f"❌ Warning: Voice reprocessing failed: {e}")
                        print("🔄 Attempting fallback with safe mode...")
                        try:
                            # Retry with safe mode enabled
                            audio_processing_params['safe_mode'] = True
                            voice_cloning_audio_path = generate_reference_audio(
                                raw_audio_path,
                                output_file=str(character_dir_path / "voice_processed.wav"),
                                **audio_processing_params
                            )
                            print(f"✅ Audio reprocessing completed for {new_name} with safe mode fallback")
                        except Exception as fallback_error:
                            print(f"❌ Warning: Voice reprocessing failed even with safe mode: {fallback_error}")
                            # Keep existing audio path
                else:
                    # Copy raw audio as processed since preprocessing is disabled
                    processed_path = str(character_dir_path / "voice_processed.wav")
                    if raw_audio_path != processed_path:
                        shutil.copy2(raw_audio_path, processed_path)
                    voice_cloning_audio_path = processed_path
                    print(f"ℹ️  Audio preprocessing disabled for {new_name} - using raw audio")
            else:
                print(f"⚠️  Could not find raw audio file to reprocess for {new_name}")
        else:
            print(f"ℹ️  No new voice cloning audio provided for {new_name} - keeping existing audio")

        # Update style tuning data ONLY if new file is provided
        if 'style_tuning_file' in request.files:
            style_file = request.files['style_tuning_file']
            if style_file.filename:
                print(f"🔄 New style tuning file detected for {new_name} - processing...")
                
                # Delete old style tuning file if it exists
                if style_tuning_data_path and os.path.exists(style_tuning_data_path):
                    os.remove(style_tuning_data_path)

                style_tuning_data_path = str(
                    new_character_dir / f"style_tuning_{style_file.filename}")
                style_file.save(style_tuning_data_path)

                # Process style tuning data
                try:
                    collection_name = f"{new_name.lower().replace(' ', '')}-style"

                    # Create temporary directories for processing
                    style_docs_dir = new_character_dir / "style_docs"
                    style_archive_dir = new_character_dir / "style_archive"
                    
                    # Clean up existing processing directories
                    if style_docs_dir.exists():
                        shutil.rmtree(style_docs_dir)
                    if style_archive_dir.exists():
                        shutil.rmtree(style_archive_dir)
                        
                    style_docs_dir.mkdir(exist_ok=True)
                    style_archive_dir.mkdir(exist_ok=True)

                    # Copy the file to the docs directory for processing
                    temp_style_path = style_docs_dir / style_file.filename
                    shutil.copy2(style_tuning_data_path, temp_style_path)

                    # Process the documents (this will replace the existing collection)
                    process_documents_for_collection(
                        str(style_docs_dir), str(style_archive_dir), collection_name, style_tuning_embedding_config, force_recreate=True)
                    
                    print(f"✅ Successfully updated and processed style tuning data for {new_name}")
                    
                    # Ensure collection is compatible with new embedding model
                    compatibility_result = ensure_character_collections_compatible(
                        new_name,
                        style_tuning_embedding_config=style_tuning_embedding_config,
                        character_dir=str(new_character_dir)
                    )
                    
                    if compatibility_result["style_tuning"]["checked"]:
                        style_result = compatibility_result["style_tuning"]
                        if style_result.get("action") == "recreated":
                            print(f"✅ Style tuning collection automatically recreated with new embedding model")
                        elif style_result.get("action") == "failed":
                            print(f"⚠️  Warning: Could not recreate style tuning collection with new embedding model")
                        elif style_result.get("action") == "manual_required":
                            print(f"⚠️  Warning: Style tuning collection needs manual recreation")
                except Exception as e:
                    print(f"❌ Warning: Style tuning processing failed: {e}")
            else:
                print(f"ℹ️  No new style tuning file provided for {new_name} - keeping existing style data")
                
                # Check if embedding model changed even without new files
                old_style_config = character.get('style_tuning_embedding_config')
                if old_style_config != style_tuning_embedding_config:
                    print(f"🔄 Style tuning embedding model changed - checking compatibility...")
                    
                    try:
                        # Ensure style_docs directory exists and has source files for recreation
                        style_docs_dir = new_character_dir / "style_docs"
                        style_archive_dir = new_character_dir / "style_archive"
                        
                        # Create directories if they don't exist
                        style_docs_dir.mkdir(exist_ok=True)
                        style_archive_dir.mkdir(exist_ok=True)
                        
                        # Check if style_docs is empty and try to populate it from existing files
                        if not any(style_docs_dir.iterdir()):
                            print(f"📂 style_docs directory is empty, populating from existing style tuning files...")
                            
                            # Look for existing style tuning files in character directory
                            style_pattern = new_character_dir / "style_tuning_*"
                            import glob
                            existing_style_files = glob.glob(str(style_pattern))
                            
                            if existing_style_files:
                                for style_file_path in existing_style_files:
                                    if os.path.exists(style_file_path):
                                        filename = os.path.basename(style_file_path)
                                        # Remove the "style_tuning_" prefix to get original filename
                                        original_filename = filename.split('_', 2)[-1] if '_' in filename else filename
                                        dest_path = style_docs_dir / original_filename
                                        shutil.copy2(style_file_path, dest_path)
                                        print(f"📄 Copied {filename} to style_docs for recreation")
                            else:
                                print(f"⚠️  No existing style tuning files found for recreation")
                        
                        embedding_change_result = handle_character_embedding_model_change(
                            new_name,
                            old_style_config,
                            style_tuning_embedding_config,
                            "style",
                            str(new_character_dir)
                        )
                        
                        if embedding_change_result["success"]:
                            if embedding_change_result["action"] == "recreated":
                                print(f"✅ Style tuning collection automatically recreated with new embedding model")
                            elif embedding_change_result["action"] == "none":
                                print(f"ℹ️  Style tuning embedding model unchanged")
                        else:
                            print(f"⚠️  Warning: Could not handle style tuning embedding model change: {embedding_change_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"⚠️  Warning: Error checking style tuning embedding model change: {e}")
        else:
            print(f"ℹ️  No new style tuning file provided for {new_name} - keeping existing style data")

        # Check if thinking audio needs to be regenerated
        should_regenerate_thinking_audio = False
        thinking_regeneration_reasons = []
        
        # Check if voice cloning model changed
        old_voice_model = old_voice_settings.get('model', 'f5tts')
        new_voice_model = voice_cloning_settings.get('model', 'f5tts')
        if old_voice_model != new_voice_model:
            should_regenerate_thinking_audio = True
            thinking_regeneration_reasons.append(f"TTS model changed: {old_voice_model} → {new_voice_model}")
        
        # Check if new voice file was uploaded
        if 'voice_cloning_audio' in request.files and request.files['voice_cloning_audio'].filename:
            should_regenerate_thinking_audio = True
            thinking_regeneration_reasons.append("New voice audio uploaded")
        
        # Check if preprocessing settings changed (and audio was reprocessed)
        if preprocessing_changed:
            should_regenerate_thinking_audio = True
            thinking_regeneration_reasons.append("Voice preprocessing settings changed")
        
        # Check if reference text changed
        old_reference_text = character.get('voice_cloning_reference_text', '')
        new_reference_text = request.form.get('voice_cloning_reference_text', old_reference_text)
        if old_reference_text != new_reference_text:
            should_regenerate_thinking_audio = True
            thinking_regeneration_reasons.append("Reference text changed")
        
        # Regenerate thinking audio if needed
        if should_regenerate_thinking_audio and voice_cloning_audio_path and voice_cloning_settings:
            print(f"🤔 Regenerating thinking audio for {new_name}...")
            print(f"   Reasons: {', '.join(thinking_regeneration_reasons)}")
            
            # Check if thinking audio generation is enabled
            generate_thinking_audio_enabled = os.getenv('GENERATE_THINKING_AUDIO', 'true').lower() == 'true'
            
            if generate_thinking_audio_enabled:
                try:
                    thinking_audio_base64 = generate_thinking_audio(
                        new_name, voice_cloning_settings, voice_cloning_audio_path, new_reference_text
                    )
                    
                    if thinking_audio_base64:
                        # Update the character's thinking audio in the database
                        conn_temp = get_db_connection()
                        cur_temp = conn_temp.cursor()
                        cur_temp.execute(
                            "UPDATE characters SET thinking_audio_base64 = %s WHERE id = %s",
                            (json.dumps(thinking_audio_base64), character_id)
                        )
                        conn_temp.commit()
                        cur_temp.close()
                        conn_temp.close()
                        print(f"✅ Successfully regenerated thinking audio for {new_name}")
                    else:
                        print(f"⚠️  Thinking audio regeneration returned None for {new_name}")
                except Exception as thinking_error:
                    print(f"❌ Thinking audio regeneration failed for {new_name}: {thinking_error}")
                    # Continue with character update even if thinking audio fails
            else:
                print(f"⏭️ Thinking audio generation disabled via environment variable for {new_name}")
        elif should_regenerate_thinking_audio:
            print(f"⚠️  Would regenerate thinking audio for {new_name}, but missing voice cloning settings or audio path")

        # Update paths if character directory was renamed
        if new_name != old_name:
            if image_path:
                image_path = image_path.replace(old_name.replace(
                    ' ', '_').lower(), new_name.replace(' ', '_').lower())
            if knowledge_base_path:
                knowledge_base_path = knowledge_base_path.replace(
                    old_name.replace(' ', '_').lower(), new_name.replace(' ', '_').lower())
            if voice_cloning_audio_path:
                voice_cloning_audio_path = voice_cloning_audio_path.replace(
                    old_name.replace(' ', '_').lower(), new_name.replace(' ', '_').lower())
            if style_tuning_data_path:
                style_tuning_data_path = style_tuning_data_path.replace(
                    old_name.replace(' ', '_').lower(), new_name.replace(' ', '_').lower())

        # Update database
        cur.execute("""
            UPDATE characters 
            SET name = %s, image_path = %s, llm_model = %s, llm_config = %s, 
                knowledge_base_path = %s, voice_cloning_audio_path = %s, 
                voice_cloning_reference_text = %s, voice_cloning_settings = %s, 
                style_tuning_data_path = %s, wakeword = %s, updated_at = CURRENT_TIMESTAMP,
                knowledge_base_embedding_config = %s, style_tuning_embedding_config = %s
            WHERE id = %s
        """, (
            new_name, image_path, llm_model, json.dumps(llm_config),
            knowledge_base_path, voice_cloning_audio_path, voice_reference_text,
            json.dumps(voice_cloning_settings), style_tuning_data_path, wakeword,
            json.dumps(knowledge_base_embedding_config), json.dumps(style_tuning_embedding_config),
            character_id
        ))

        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Character '{new_name}' updated successfully")
        return jsonify({
            "message": f"Character updated successfully",
            "character_id": character_id,
            "character_name": new_name,
            "status": "success"
        }), 200

    except Exception as e:
        print(f"❌ Error updating character: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get-system-performance', methods=['GET'])
def get_system_performance():
    """
    Get comprehensive system performance information including STT and TTS models.
    """
    try:
        # Get TTS performance info
        from libraries.tts.inference import get_device_performance_info
        tts_info = get_device_performance_info()

        # Get STT performance info
        stt_info = get_stt_performance_info()

        # Get general system info
        system_info = {
            "device_optimization_available": DEVICE_OPTIMIZATION_AVAILABLE,
            "startup_time": time.time() - _startup_time if '_startup_time' in globals() else None,
        }

        if DEVICE_OPTIMIZATION_AVAILABLE:
            device_type, device_info = get_device_info()
            system_info.update({
                "device_type": device_type.value,
                "device_info": device_info,
            })

        return jsonify({
            "system": system_info,
            "tts": tts_info,
            "stt": stt_info,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/preload-models', methods=['POST'])
def preload_models_endpoint():
    """
    Preload models for optimal performance.
    Expects: {"models": ["stt", "tts"], "tts_models": ["f5tts", "xtts"]}
    """
    try:
        data = request.get_json()
        models_to_load = data.get('models', ['stt', 'tts'])
        tts_models = data.get('tts_models', ['f5tts'])

        results = {
            "loaded": [],
            "failed": [],
            "status": "success"
        }

        # Preload STT model
        if 'stt' in models_to_load:
            try:
                print("🎙️  Preloading STT (Whisper) model...")
                _get_whisper_model()  # This will load the optimal model
                results["loaded"].append("stt")
                print("✓ STT model preloaded successfully")
            except Exception as e:
                print(f"✗ Failed to preload STT model: {e}")
                results["failed"].append({"model": "stt", "error": str(e)})

        # Preload TTS models
        if 'tts' in models_to_load:
            try:
                print("🔊 Preloading TTS models...")
                from libraries.tts.inference import preload_models_smart
                preload_models_smart(tts_models, force_reload=False)
                results["loaded"].append("tts")
                print("✓ TTS models preloaded successfully")
            except Exception as e:
                print(f"✗ Failed to preload TTS models: {e}")
                results["failed"].append({"model": "tts", "error": str(e)})

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/unload-all-models-comprehensive', methods=['POST'])
def unload_all_models_comprehensive():
    """
    Unload all models (STT, TTS, and LLM) from memory with comprehensive cleanup.
    """
    try:
        print("🧹 Starting comprehensive model cleanup...")

        # Unload STT model
        _unload_stt_model()

        # Unload TTS models
        from libraries.tts.inference import unload_models
        unload_models()

        # Unload LLM models
        from libraries.llm.inference import unload_all_cached_models
        unload_all_cached_models()

        # Clear global model cache
        global model_cache
        model_cache.clear()

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ Final CUDA cache cleared and synchronized")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                print("✓ Final MPS cache cleared")
            except:
                pass

        # Force garbage collection
        import gc
        gc.collect()
        print("✓ Final garbage collection completed")

        print("🎯 Comprehensive model cleanup completed!")

        return jsonify({
            "message": "All models (STT, TTS, and LLM) unloaded from memory",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/optimize-for-character', methods=['POST'])
def optimize_for_character():
    """
    Optimize system for a specific character by preloading only necessary models.
    Expects: {"character_id": int, "preload_stt": bool}
    """
    try:
        data = request.get_json()
        character_id = data.get('character_id')
        preload_stt = data.get('preload_stt', True)

        if not character_id:
            return jsonify({"error": "character_id is required"}), 400

        # Get character data
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM characters WHERE id = %s", (character_id,))
        character = cur.fetchone()
        cur.close()
        conn.close()

        if not character:
            return jsonify({"error": "Character not found"}), 404

        character_name = character['name']
        print(f"🎯 Optimizing system for character: {character_name}")

        # First, unload all models to start fresh
        print("🧹 Clearing existing models...")
        _unload_stt_model()
        from libraries.tts.inference import unload_models
        unload_models()

        # Preload STT model if requested
        if preload_stt:
            try:
                print("🎙️  Preloading STT model for character...")
                _get_whisper_model()
                print("✓ STT model optimized for character")
            except Exception as e:
                print(f"Warning: STT preload failed: {e}")

        # Preload character-specific models directly
        try:
            print(f"🔄 Loading models for character {character_name}...")

            # Load LLM model for this character
            if character['llm_model'] and character['llm_config']:
                from libraries.llm.inference import preload_llm_model, ModelType

                llm_config = character['llm_config']
                llm_cache_key = f"{character_name}_llm"

                # Determine model type - prioritize model path over config
                model_path, model_type_str = resolve_model_path(character['llm_model'])
                if model_type_str == "gguf":
                    model_type = ModelType.GGUF
                elif model_type_str == "openai_api":
                    model_type = ModelType.OPENAI_API
                elif model_type_str == "huggingface":
                    model_type = ModelType.HUGGINGFACE
                elif 'api_key' in llm_config:
                    # Only use API if model type is ambiguous and api_key is present
                    model_type = ModelType.OPENAI_API
                else:
                    model_type = ModelType.HUGGINGFACE

                model = preload_llm_model(
                    model_type=model_type,
                    model_config={
                        'model_path': model_path,
                        **llm_config
                    },
                    cache_key=llm_cache_key
                )
                model_cache[llm_cache_key] = model
                print(f"✓ LLM model preloaded for {character_name}")

            # Load TTS model for this character
            if character['voice_cloning_settings']:
                from libraries.tts.inference import preload_models_smart, ensure_model_available

                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                success = ensure_model_available(tts_model)
                if success:
                    preload_models_smart([tts_model])
                    print(
                        f"✓ TTS model {tts_model} preloaded for {character_name}")

        except Exception as e:
            print(f"Warning: Character model loading failed: {e}")

        return jsonify({
            "message": f"System optimized for character {character_name}",
            "character_id": character_id,
            "character_name": character_name,
            "optimizations": {
                "stt_preloaded": preload_stt,
                "tts_preloaded": True,
                "llm_preloaded": True,
            },
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/check-character-name', methods=['POST'])
def check_character_name():
    """
    Check if a character name already exists in the database.
    Expects: {"name": str}
    Returns: {"exists": bool, "status": "success"}
    """
    try:
        data = request.get_json()
        name = data.get('name')
        
        if not name:
            return jsonify({"error": "name is required"}), 400
        
        # Normalize the name for comparison (case-insensitive)
        normalized_name = name.strip().lower()
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM characters WHERE LOWER(name) = %s", (normalized_name,))
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        exists = count > 0
        
        return jsonify({
            "exists": exists,
            "name": name,
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-embedding-models', methods=['GET'])
def get_embedding_models():
    """Get available embedding models."""
    try:
        available_models = embedding_manager.get_available_models()
        
        return jsonify({
            "success": True,
            "models": available_models
        })
        
    except Exception as e:
        print(f"Error getting embedding models: {e}")
        return jsonify({"error": "Failed to get embedding models"}), 500


@app.route('/test-embedding-config', methods=['POST'])
def test_embedding_config():
    """Test an embedding configuration."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        model_name = data.get('model_name')
        config = data.get('config', {})
        
        if not model_type or not model_name:
            return jsonify({"error": "model_type and model_name are required"}), 400
        
        # Test the configuration by creating and testing the model
        test_model_id = f"test_{uuid.uuid4().hex[:8]}"
        
        try:
            # Try to load the model
            embedding_model = embedding_manager.load_model(
                model_id=test_model_id,
                model_type=model_type,
                model_name=model_name,
                config=config
            )
            
            # Test with a simple text
            test_text = "This is a test sentence for embedding."
            embedding = embedding_model.embed_text(test_text)
            
            # Clean up test model
            embedding_manager.unload_model(test_model_id)
            
            return jsonify({
                "success": True,
                "message": "Embedding configuration is valid",
                "embedding_dimensions": len(embedding)
            })
            
        except Exception as model_error:
            # Clean up test model if it was partially created
            embedding_manager.unload_model(test_model_id)
            
            return jsonify({
                "success": False,
                "error": f"Configuration test failed: {str(model_error)}"
            }), 400
        
    except Exception as e:
        print(f"Error testing embedding config: {e}")
        return jsonify({"error": "Failed to test embedding configuration"}), 500


@app.route('/get-loaded-embedding-models', methods=['GET'])
def get_loaded_embedding_models():
    """Get information about currently loaded embedding models."""
    try:
        loaded_models = embedding_manager.list_models()
        
        return jsonify({
            "success": True,
            "models": loaded_models
        })
        
    except Exception as e:
        print(f"Error getting loaded embedding models: {e}")
        return jsonify({"error": "Failed to get loaded embedding models"}), 500


@app.route('/unload-embedding-model', methods=['POST'])
def unload_embedding_model():
    """Unload a specific embedding model."""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({"error": "model_id is required"}), 400
        
        success = embedding_manager.unload_model(model_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Embedding model '{model_id}' unloaded successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Embedding model '{model_id}' not found"
            }), 404
        
    except Exception as e:
        print(f"Error unloading embedding model: {e}")
        return jsonify({"error": "Failed to unload embedding model"}), 500


@app.route('/unload-all-embedding-models', methods=['POST'])
def unload_all_embedding_models():
    """Unload all embedding models."""
    try:
        embedding_manager.unload_all_models()
        
        return jsonify({
            "success": True,
            "message": "All embedding models unloaded successfully"
        })
        
    except Exception as e:
        print(f"Error unloading all embedding models: {e}")
        return jsonify({"error": "Failed to unload embedding models"}), 500


@app.route('/preload-zonos-worker', methods=['POST'])
def preload_zonos_worker_endpoint():
    """
    Preload a specific Zonos worker for faster subsequent generations.
    Expects: {"model": str, "device": str}
    """
    try:
        data = request.get_json()
        model = data.get('model', 'Zyphra/Zonos-v0.1-transformer')
        device = data.get('device', 'auto')

        if not model:
            return jsonify({"error": "model is required"}), 400

        print(f"🚀 Preloading Zonos worker for model: {model}")

        # Import the preload function
        from libraries.tts.inference import preload_zonos_worker

        # Preload the worker
        success = preload_zonos_worker(model, device)

        if success:
            return jsonify({
                "message": f"Zonos worker preloaded successfully for model: {model}",
                "model": model,
                "device": device,
                "status": "success"
            }), 200
        else:
            return jsonify({
                "error": f"Failed to preload Zonos worker for model: {model}",
                "model": model,
                "device": device,
                "status": "failed"
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-zonos-worker-status', methods=['GET'])
def get_zonos_worker_status_endpoint():
    """
    Get status of all persistent Zonos workers.
    """
    try:
        from libraries.tts.inference import get_zonos_worker_status
        
        status = get_zonos_worker_status()
        
        return jsonify({
            "status": status,
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/cleanup-zonos-workers', methods=['POST'])
def cleanup_zonos_workers_endpoint():
    """
    Clean up all persistent Zonos workers.
    """
    try:
        from libraries.tts.inference import _cleanup_zonos_workers
        
        _cleanup_zonos_workers()
        
        return jsonify({
            "message": "All Zonos workers cleaned up successfully",
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/check-collection-compatibility', methods=['POST'])
def check_collection_compatibility_endpoint():
    """
    Check if a collection is compatible with the given embedding configuration.
    
    Expected JSON body:
    {
        "collection_name": "character_name",
        "embedding_config": {
            "model_type": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2",
            "model_id": "default",
            "config": {}
        }
    }
    
    Returns:
        JSON response with compatibility information
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        collection_name = data.get('collection_name')
        embedding_config = data.get('embedding_config')
        
        if not collection_name:
            return jsonify({
                "success": False,
                "error": "collection_name is required"
            }), 400
        
        if not embedding_config:
            return jsonify({
                "success": False,
                "error": "embedding_config is required"
            }), 400
        
        # Check compatibility
        compatibility_result = check_collection_compatibility(collection_name, embedding_config)
        
        return jsonify({
            "success": True,
            "compatibility": compatibility_result
        })
        
    except Exception as e:
        print(f"Error checking collection compatibility: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/get-collection-diagnostics', methods=['POST'])
def get_collection_diagnostics_endpoint():
    """
    Get comprehensive diagnostic information about a collection.
    
    Expected JSON body:
    {
        "collection_name": "character_name"
    }
    
    Returns:
        JSON response with diagnostic information
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        collection_name = data.get('collection_name')
        
        if not collection_name:
            return jsonify({
                "success": False,
                "error": "collection_name is required"
            }), 400
        
        # Get diagnostics
        diagnostics = get_collection_diagnostics(collection_name)
        
        return jsonify({
            "success": True,
            "diagnostics": diagnostics
        })
        
    except Exception as e:
        print(f"Error getting collection diagnostics: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def normalize_voice_cloning_settings(settings: dict) -> dict:
    """
    Normalize voice cloning settings to ensure proper data types for TTS models.
    This fixes issues where numeric values might be stored as strings.
    """
    if not settings:
        return settings
    
    # Create a copy to avoid modifying the original
    normalized = settings.copy()
    
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
    
    # Handle nested settings dictionaries (XTTS, F5, Zonos settings)
    for settings_key in ['xtts_settings', 'f5_settings', 'zonos_settings']:
        if settings_key in normalized and isinstance(normalized[settings_key], dict):
            nested_settings = normalized[settings_key].copy()
            
            if settings_key == 'xtts_settings':
                target_fields = xtts_numeric_fields
            elif settings_key == 'zonos_settings':
                target_fields = zonos_numeric_fields
            else:  # f5_settings
                target_fields = {}  # F5 doesn't have specific numeric fields yet
            
            for field, conversion_func in target_fields.items():
                if field in nested_settings and nested_settings[field] is not None:
                    try:
                        if isinstance(nested_settings[field], str) or (conversion_func == float and isinstance(nested_settings[field], int)) or (conversion_func == int and isinstance(nested_settings[field], float)):
                            nested_settings[field] = conversion_func(nested_settings[field])
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert {settings_key}.{field}={nested_settings[field]} to {conversion_func.__name__}: {e}")
            
            normalized[settings_key] = nested_settings
    
    # Handle boolean conversions
    boolean_fields = ['preprocess_audio', 'clean_audio', 'remove_silence', 'enhance_audio', 'skip_all_processing', 
                     'bass_boost', 'treble_boost', 'compression', 'enable_text_splitting']
    
    for field in boolean_fields:
        if field in normalized and normalized[field] is not None:
            if isinstance(normalized[field], str):
                normalized[field] = normalized[field].lower() in ('true', '1', 'yes', 'on')
            elif not isinstance(normalized[field], bool):
                normalized[field] = bool(normalized[field])
    
    # Handle nested boolean fields
    for settings_key in ['xtts_settings', 'f5_settings', 'zonos_settings']:
        if settings_key in normalized and isinstance(normalized[settings_key], dict):
            nested_settings = normalized[settings_key]
            for field in boolean_fields:
                if field in nested_settings and nested_settings[field] is not None:
                    if isinstance(nested_settings[field], str):
                        nested_settings[field] = nested_settings[field].lower() in ('true', '1', 'yes', 'on')
                    elif not isinstance(nested_settings[field], bool):
                        nested_settings[field] = bool(nested_settings[field])
    
    return normalized


if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
