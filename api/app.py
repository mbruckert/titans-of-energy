from libraries.knowledgebase.preprocess import process_documents_for_collection
from libraries.knowledgebase.retrieval import query_collection
from libraries.llm.inference import generate_styled_text, get_style_data, load_model, ModelType, preload_llm_model, unload_all_cached_models, get_cached_models_info
from libraries.llm.preprocess import download_model as download_model_func
from libraries.tts.preprocess import generate_reference_audio, download_voice_models
from libraries.tts.inference import generate_audio, ensure_model_available, unload_models, get_loaded_models, preload_models_smart, is_model_loaded
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
        model_id: Model identifier (e.g., "google-gemma-3-4b-it-qat-q4_0-gguf")

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

    conn.commit()
    cur.close()
    conn.close()


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
                    str(kb_docs_dir), str(kb_archive_dir), collection_name)
                
                print(f"Successfully processed {len(kb_file_paths)} knowledge base files for {name}")
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
                        str(style_docs_dir), str(style_archive_dir), collection_name)
                except Exception as e:
                    print(f"Warning: Style tuning processing failed: {e}")

        # Save to database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO characters 
            (name, image_path, llm_model, llm_config, knowledge_base_path, 
             voice_cloning_audio_path, voice_cloning_reference_text, voice_cloning_settings, style_tuning_data_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
            RETURNING id
        """, (
            name, image_path, llm_model, json.dumps(
                llm_config), knowledge_base_path,
            voice_cloning_audio_path, voice_reference_text,
            json.dumps(voice_cloning_settings),
            style_tuning_data_path
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
            SELECT id, name, image_path, llm_model, created_at 
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

        # Remove sensitive file paths for security but keep the boolean flags
        char_dict.pop('image_path', None)
        char_dict.pop('knowledge_base_path', None)
        char_dict.pop('voice_cloning_audio_path', None)
        char_dict.pop('style_tuning_data_path', None)

        return jsonify({"character": char_dict, "status": "success"}), 200

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

                # Determine model type
                if 'api_key' in llm_config:
                    model_type = ModelType.OPENAI_API
                elif model_type_str == "gguf":
                    model_type = ModelType.GGUF
                elif model_type_str == "openai_api":
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
                        # Use smart preloading to avoid unnecessary reloads
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

        # Search knowledge base - ALWAYS provide 3 references regardless of fast mode
        knowledge_context = ""
        knowledge_references = []
        try:
            kb_collection = f"{character_name.lower().replace(' ', '')}-knowledge"
            kb_result = query_collection(
                kb_collection, question, n_results=3, return_structured=True)

            if isinstance(kb_result, dict) and "references" in kb_result:
                knowledge_context = kb_result.get("context", "")
                knowledge_references = kb_result.get("references", [])
            else:
                # Fallback to string format for backward compatibility
                knowledge_context = str(kb_result)
        except Exception as e:
            print(f"Knowledge base search failed: {e}")
        
        # Get style examples - Use optimal number for speed while maintaining quality
        style_examples = []
        try:
            style_examples = get_style_data(
                question, character_name, num_examples=2)
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
                # Check if we should use real-time optimized generation for high-end hardware
                if DEVICE_OPTIMIZATION_AVAILABLE:
                    device_type, device_info = get_device_info()
                    use_realtime = (device_type == DeviceType.APPLE_SILICON and 
                                   device_info.get('is_high_end', False))
                else:
                    use_realtime = False

                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                # Use the stored reference text from the character
                ref_text = character.get('voice_cloning_reference_text', '')
                if not ref_text:
                    # Fallback if no reference text was stored
                    ref_text = "Hello, how can I help you?"
                    print(
                        "Warning: No reference text found for character, using fallback")

                if use_realtime:
                    # Use real-time optimized generation for M4 Max and high-end hardware
                    from libraries.tts.inference import generate_realtime_audio_base64
                    print("🚀 Using real-time optimized TTS generation")
                    
                    audio_base64 = generate_realtime_audio_base64(
                        model=tts_model,
                        ref_audio=character['voice_cloning_audio_path'],
                        ref_text=ref_text,
                        gen_text=styled_response,
                        config=voice_settings
                    )
                else:
                    # Generate audio as base64 with optimized performance
                    from libraries.tts.inference import generate_cloned_audio_base64
                    
                    audio_base64 = generate_cloned_audio_base64(
                        model=tts_model,
                        ref_audio=character['voice_cloning_audio_path'],
                        ref_text=ref_text,
                        gen_text=styled_response,
                        config=voice_settings
                    )

            except Exception as e:
                print(f"Audio generation failed: {e}")
                audio_base64 = None

        # Store chat history in database
        try:
            cur.execute("""
                INSERT INTO chat_history 
                (character_id, user_message, bot_response, audio_base64, knowledge_context, knowledge_references)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                character_id,
                question,
                styled_response,
                audio_base64,
                knowledge_context,
                json.dumps(knowledge_references)  # Store as JSON
            ))
            conn.commit()
        except Exception as e:
            print(f"Failed to store chat history: {e}")
            # Try without knowledge_references column for backward compatibility
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
            "audio_generation_time": round(audio_generation_time, 3)
        }
        
        print(f"🚀 Optimized response delivered for {character_name} - Audio gen: {audio_generation_time:.3f}s")
        
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe-audio', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio to text using optimized Whisper with comprehensive device optimization.
    Expects: multipart/form-data with 'audio_file'
    Returns: {"transcript": str, "status": "success"}
    """
    try:
        audio_file = request.files.get('audio_file')

        if not audio_file:
            return jsonify({"error": "audio_file is required"}), 400

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

        # Search knowledge base
        knowledge_context = ""
        knowledge_references = []
        try:
            kb_collection = f"{character_name.lower().replace(' ', '')}-knowledge"
            kb_result = query_collection(
                kb_collection, question, n_results=3, return_structured=True)

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
                question, character_name, num_examples=2)
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
                # Check if we should use real-time optimized generation for high-end hardware
                if DEVICE_OPTIMIZATION_AVAILABLE:
                    device_type, device_info = get_device_info()
                    use_realtime = (device_type == DeviceType.APPLE_SILICON and 
                                   device_info.get('is_high_end', False))
                else:
                    use_realtime = False

                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                # Use the stored reference text from the character
                ref_text = character.get('voice_cloning_reference_text', '')
                if not ref_text:
                    # Fallback if no reference text was stored
                    ref_text = "Hello, how can I help you?"
                    print(
                        "Warning: No reference text found for character, using fallback")

                if use_realtime:
                    # Use real-time optimized generation for M4 Max and high-end hardware
                    from libraries.tts.inference import generate_realtime_audio_base64
                    print("🚀 Using real-time optimized TTS generation")
                    
                    audio_base64 = generate_realtime_audio_base64(
                        model=tts_model,
                        ref_audio=character['voice_cloning_audio_path'],
                        ref_text=ref_text,
                        gen_text=styled_response,
                        config=voice_settings
                    )
                else:
                    # Generate audio as base64 with optimized performance
                    from libraries.tts.inference import generate_cloned_audio_base64
                    
                    audio_base64 = generate_cloned_audio_base64(
                        model=tts_model,
                        ref_audio=character['voice_cloning_audio_path'],
                        ref_text=ref_text,
                        gen_text=styled_response,
                        config=voice_settings
                    )

            except Exception as e:
                print(f"Audio generation failed: {e}")
                audio_base64 = None

        # Store chat history in database
        try:
            cur.execute("""
                INSERT INTO chat_history 
                (character_id, user_message, bot_response, audio_base64, knowledge_context, knowledge_references)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                character_id,
                question,
                styled_response,
                audio_base64,
                knowledge_context,
                json.dumps(knowledge_references)  # Store as JSON
            ))
            conn.commit()
        except Exception as e:
            print(f"Failed to store chat history: {e}")
            # Try without knowledge_references column for backward compatibility
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
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download-model', methods=['POST'])
def download_model_endpoint():
    """
    Download a model using the preprocess module.
    Expects POST requests with 'model_name' and 'model_type' parameters.
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        model_type = data.get('model_type')

        if not model_name or not model_type:
            return jsonify({"error": "model_name and model_type are required"}), 400

        # Download the model
        try:
            model_path = download_model_func(model_name, model_type)
            return jsonify({"model_path": model_path, "status": "success"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-llm-models', methods=['GET'])
def get_llm_models():
    """
    Get list of available LLM models with their availability status.
    """
    try:
        from libraries.llm.inference import list_available_models
        import os

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

        return jsonify({
            "models": model_info,
            "status": "success"
        }), 200

    except Exception as e:
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
            SELECT id, user_message, bot_response, audio_base64, knowledge_context, knowledge_references, created_at
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
                    entry_dict['knowledge_references'] = json.loads(
                        entry_dict['knowledge_references'])
                except (json.JSONDecodeError, TypeError):
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

        # Extract reference text from voice cloning settings
        voice_reference_text = voice_cloning_settings.get(
            'reference_text', character.get('voice_cloning_reference_text', ''))

        # Create new character directory if name changed
        new_character_dir = STORAGE_DIR / new_name.replace(' ', '_').lower()
        if new_name != old_name:
            if old_character_dir.exists():
                shutil.move(str(old_character_dir), str(new_character_dir))
                print(
                    f"Renamed character directory from {old_character_dir} to {new_character_dir}")
            else:
                new_character_dir.mkdir(exist_ok=True)
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

        # Update knowledge base if new files provided (support multiple files)
        kb_files = request.files.getlist('knowledge_base_file')
        if kb_files and any(f.filename for f in kb_files):
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

                # Process all documents in the directory
                process_documents_for_collection(
                    str(kb_docs_dir), str(kb_archive_dir), collection_name)
                
                print(f"Successfully updated and processed {len(kb_file_paths)} knowledge base files for {new_name}")
            except Exception as e:
                print(f"Warning: Knowledge base processing failed: {e}")

        # Update voice cloning audio if new file provided
        if 'voice_cloning_audio' in request.files:
            voice_file = request.files['voice_cloning_audio']
            if voice_file.filename:
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
                        print(f"Audio preprocessing completed for {new_name}")
                    except Exception as e:
                        print(f"Warning: Voice preprocessing failed: {e}")
                        print("Attempting fallback with safe mode...")
                        try:
                            # Retry with safe mode enabled
                            audio_processing_params['safe_mode'] = True
                            voice_cloning_audio_path = generate_reference_audio(
                                raw_audio_path,
                                output_file=str(
                                    new_character_dir / "voice_processed.wav"),
                                **audio_processing_params
                            )
                            print(f"Audio preprocessing completed for {new_name} with safe mode fallback")
                        except Exception as fallback_error:
                            print(f"Warning: Voice preprocessing failed even with safe mode: {fallback_error}")
                            voice_cloning_audio_path = raw_audio_path
                else:
                    voice_cloning_audio_path = raw_audio_path

        # Update style tuning data if new file provided
        if 'style_tuning_file' in request.files:
            style_file = request.files['style_tuning_file']
            if style_file.filename:
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
                    style_docs_dir.mkdir(exist_ok=True)
                    style_archive_dir.mkdir(exist_ok=True)

                    # Copy the file to the docs directory for processing
                    temp_style_path = style_docs_dir / style_file.filename
                    shutil.copy2(style_tuning_data_path, temp_style_path)

                    # Process the documents
                    process_documents_for_collection(
                        str(style_docs_dir), str(style_archive_dir), collection_name)
                except Exception as e:
                    print(f"Warning: Style tuning processing failed: {e}")

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
                style_tuning_data_path = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (
            new_name, image_path, llm_model, json.dumps(llm_config),
            knowledge_base_path, voice_cloning_audio_path, voice_reference_text,
            json.dumps(
                voice_cloning_settings), style_tuning_data_path, character_id
        ))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": f"Character updated successfully",
            "character_id": character_id,
            "character_name": new_name,
            "status": "success"
        }), 200

    except Exception as e:
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

                # Determine model type
                if 'api_key' in llm_config:
                    model_type = ModelType.OPENAI_API
                elif character['llm_model'].endswith('.gguf'):
                    model_type = ModelType.GGUF
                else:
                    model_type = ModelType.HUGGINGFACE

                model = preload_llm_model(
                    model_type=model_type,
                    model_config={
                        'model_path': character['llm_model'],
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


if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
