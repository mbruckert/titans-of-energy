from libraries.knowledgebase.preprocess import process_documents_for_collection
from libraries.knowledgebase.retrieval import query_collection
from libraries.llm.inference import generate_styled_text, get_style_data, load_model, ModelType, preload_llm_model, unload_all_cached_models, get_cached_models_info
from libraries.llm.preprocess import download_model as download_model_func
from libraries.tts.preprocess import generate_reference_audio, download_voice_models
from libraries.tts.inference import generate_audio, ensure_model_available, preload_models, unload_models, get_loaded_models, preload_models_smart, is_model_loaded
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
        voice_reference_text = voice_cloning_settings.get('reference_text', '')

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

        # Save and process knowledge base data
        if 'knowledge_base_file' in request.files:
            kb_file = request.files['knowledge_base_file']
            if kb_file.filename:
                knowledge_base_path = str(
                    character_dir / f"knowledge_base_{kb_file.filename}")
                kb_file.save(knowledge_base_path)

                # Process knowledge base documents
                try:
                    collection_name = f"{name.lower().replace(' ', '')}-knowledge"

                    # Create temporary directories for processing
                    kb_docs_dir = character_dir / "kb_docs"
                    kb_archive_dir = character_dir / "kb_archive"
                    kb_docs_dir.mkdir(exist_ok=True)
                    kb_archive_dir.mkdir(exist_ok=True)

                    # Copy the file to the docs directory for processing
                    temp_kb_path = kb_docs_dir / kb_file.filename
                    shutil.copy2(knowledge_base_path, temp_kb_path)

                    # Process the documents
                    process_documents_for_collection(
                        str(kb_docs_dir), str(kb_archive_dir), collection_name)
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
                        tts_only_params = {'model', 'cache_dir', 'preprocess_audio', 'reference_text',
                                           'language', 'output_dir', 'cuda_device', 'coqui_tos_agreed',
                                           'torch_force_no_weights_only_load', 'auto_download'}
                        audio_processing_params = {
                            k: v for k, v in voice_cloning_settings.items()
                            if k not in tts_only_params
                        }
                        voice_cloning_audio_path = generate_reference_audio(
                            raw_audio_path,
                            output_file=str(
                                character_dir / "voice_processed.wav"),
                            **audio_processing_params
                        )
                        print(f"Audio preprocessing completed for {name}")
                    except Exception as e:
                        print(f"Warning: Voice preprocessing failed: {e}")
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
        # Remove sensitive file paths for security
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

                # Determine model type
                if 'api_key' in llm_config:
                    model_type = ModelType.OPENAI_API
                elif character['llm_model'].endswith('.gguf'):
                    model_type = ModelType.GGUF
                else:
                    model_type = ModelType.HUGGINGFACE

                print(
                    f"Preloading LLM model '{character['llm_model']}' for {character_name}...")

                # Preload the model into cache
                model = preload_llm_model(
                    model_type=model_type,
                    model_config={
                        'model_path': character['llm_model'],
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

        # Search knowledge base
        knowledge_context = ""
        try:
            kb_collection = f"{character_name.lower().replace(' ', '')}-knowledge"
            knowledge_context = query_collection(
                kb_collection, question, n_results=3)
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
                styled_response = generate_styled_text(
                    question,
                    style_examples,
                    knowledge_context,
                    character['llm_model'],
                    character['llm_config'],
                    character_name=character_name
                )
            except Exception as e:
                print(f"Text generation failed: {e}")
                styled_response = "I'm sorry, I'm having trouble generating a response right now."

        # Generate audio response as base64
        audio_base64 = None
        if character['voice_cloning_audio_path'] and character['voice_cloning_settings']:
            try:
                from libraries.tts.inference import generate_cloned_audio_base64

                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                # Use the stored reference text from the character
                ref_text = character.get('voice_cloning_reference_text', '')
                if not ref_text:
                    # Fallback if no reference text was stored
                    ref_text = "Hello, how can I help you?"
                    print(
                        "Warning: No reference text found for character, using fallback")

                # Generate audio as base64
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
        except Exception as e:
            print(f"Failed to store chat history: {e}")

        cur.close()
        conn.close()

        return jsonify({
            "question": question,
            "text_response": styled_response,
            "audio_base64": audio_base64,
            "knowledge_context": knowledge_context,
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
            SELECT id, user_message, bot_response, audio_base64, knowledge_context, created_at
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


if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
