from libraries.knowledgebase.preprocess import process_documents_for_collection
from libraries.knowledgebase.retrieval import query_collection
from libraries.llm.inference import generate_styled_text, get_style_data, load_model, ModelType
from libraries.llm.preprocess import download_model as download_model_func
from libraries.stt.transcription import listen_and_transcribe, transcribe_audio_file
from libraries.tts.preprocess import generate_reference_audio, download_voice_models
from libraries.tts.inference import generate_audio, ensure_model_available
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


load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

# Create subdirectory for generated audio files
GENERATED_AUDIO_DIR = STORAGE_DIR / "generated_audio"
GENERATED_AUDIO_DIR.mkdir(exist_ok=True)

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
            stt_settings JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        stt_settings = json.loads(request.form.get('stt_settings', '{}'))

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
             voice_cloning_audio_path, voice_cloning_reference_text, voice_cloning_settings, style_tuning_data_path, stt_settings)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            RETURNING id
        """, (
            name, image_path, llm_model, json.dumps(
                llm_config), knowledge_base_path,
            voice_cloning_audio_path, voice_reference_text,
            json.dumps(voice_cloning_settings),
            style_tuning_data_path, json.dumps(stt_settings)
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

        # Convert to JSON serializable format and add image URLs
        result = []
        for char in characters:
            char_dict = dict(char)
            # Convert image path to URL
            char_dict['image_url'] = get_file_url(
                char_dict['image_path'], 'image')
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

        # Convert to dict and add image URL
        char_dict = dict(character)
        char_dict['image_url'] = get_file_url(char_dict['image_path'], 'image')
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
    Preload all models associated with a character.
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

        # Load LLM model
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

                model = load_model(model_type, {
                    'model_path': character['llm_model'],
                    **llm_config
                })

                model_cache[f"{character_name}_llm"] = model
                print(f"Loaded LLM model for {character_name}")

            except Exception as e:
                print(f"Failed to load LLM model for {character_name}: {e}")

        # Ensure TTS model is available
        if character['voice_cloning_settings']:
            try:
                voice_settings = character['voice_cloning_settings']
                tts_model = voice_settings.get('model', 'f5tts')

                print(
                    f"Preparing TTS model '{tts_model}' for {character_name}...")
                success = ensure_model_available(tts_model)
                if success:
                    # Mark TTS model as ready for this character
                    model_cache[f"{character_name}_tts_ready"] = True
                    print(
                        f"✓ TTS model {tts_model} ready for {character_name}")
                else:
                    print(
                        f"⚠ Warning: TTS model {tts_model} preparation failed for {character_name}")
                    print(
                        f"  Audio generation may be slower due to model re-initialization")

            except Exception as e:
                print(
                    f"✗ Failed to prepare TTS model for {character_name}: {e}")
                print(
                    f"  Character loading will continue, but audio generation may be slower")

        return jsonify({
            "message": f"Models loaded for character {character_name}",
            "character_id": character_id,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-question-text', methods=['POST'])
def ask_question_text():
    """
    Process text question and return styled text response with generated audio.
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
        cur.close()
        conn.close()

        if not character:
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
                    character['llm_config']
                )
            except Exception as e:
                print(f"Text generation failed: {e}")
                styled_response = "I'm sorry, I'm having trouble generating a response right now."

        # Generate audio response
        audio_path = None
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

                # Check if TTS model is already prepared for this character
                tts_ready = model_cache.get(
                    f"{character_name}_tts_ready", False)

                # Set output directory to be within storage so files can be served
                voice_settings_copy = voice_settings.copy()
                voice_settings_copy['output_dir'] = str(
                    STORAGE_DIR / "generated_audio")

                audio_path = generate_audio(
                    model=tts_model,
                    ref_audio=character['voice_cloning_audio_path'],
                    ref_text=ref_text,
                    gen_text=styled_response,
                    config=voice_settings_copy,
                    auto_download=not tts_ready  # Skip download if already prepared
                )

            except Exception as e:
                print(f"Audio generation failed: {e}")

        return jsonify({
            "question": question,
            "text_response": styled_response,
            "audio_url": get_file_url(audio_path, 'audio'),
            "knowledge_context": knowledge_context,
            "character_name": character_name,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-question-audio', methods=['POST'])
def ask_question_audio():
    """
    Process pre-recorded audio file and return styled text response with generated audio.
    Expects form data with 'character_id' and 'audio_file'
    """
    try:
        character_id = request.form.get('character_id')

        if not character_id:
            return jsonify({"error": "character_id is required"}), 400

        if 'audio_file' not in request.files:
            return jsonify({"error": "audio_file is required"}), 400

        audio_file = request.files['audio_file']
        if not audio_file.filename:
            return jsonify({"error": "No audio file provided"}), 400

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

        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            temp_audio_path = tmp_file.name

        # Transcribe audio to text
        question = ""
        try:
            stt_settings = character.get('stt_settings', {})
            stt_model = stt_settings.get('model', 'whisper')

            question = transcribe_audio_file(
                temp_audio_path, stt_model, stt_settings)

        except Exception as e:
            print(f"Speech transcription failed: {e}")
            return jsonify({"error": "Failed to transcribe audio"}), 500
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        if not question.strip():
            return jsonify({"error": "No speech detected in audio"}), 400

        # Now process like text question
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
                    character['llm_config']
                )
            except Exception as e:
                print(f"Text generation failed: {e}")
                styled_response = "I'm sorry, I'm having trouble generating a response right now."

        # Generate audio response
        audio_path = None
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

                # Check if TTS model is already prepared for this character
                tts_ready = model_cache.get(
                    f"{character_name}_tts_ready", False)

                # Set output directory to be within storage so files can be served
                voice_settings_copy = voice_settings.copy()
                voice_settings_copy['output_dir'] = str(
                    STORAGE_DIR / "generated_audio")

                audio_path = generate_audio(
                    model=tts_model,
                    ref_audio=character['voice_cloning_audio_path'],
                    ref_text=ref_text,
                    gen_text=styled_response,
                    config=voice_settings_copy,
                    auto_download=not tts_ready  # Skip download if already prepared
                )

            except Exception as e:
                print(f"Audio generation failed: {e}")

        return jsonify({
            "transcribed_question": question,
            "text_response": styled_response,
            "audio_url": get_file_url(audio_path, 'audio'),
            "knowledge_context": knowledge_context,
            "character_name": character_name,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-question-audio-live', methods=['POST'])
def ask_question_audio_live():
    """
    Process audio question with live recording and return styled text response with generated audio.
    Expects JSON data with 'character_id'
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

        # Transcribe audio to text using live recording
        question = ""
        try:
            stt_settings = character.get('stt_settings', {})
            stt_model = stt_settings.get('model', 'whisper')

            question = listen_and_transcribe(stt_model, stt_settings)

        except Exception as e:
            print(f"Speech transcription failed: {e}")
            return jsonify({"error": "Failed to transcribe audio"}), 500

        if not question.strip():
            return jsonify({"error": "No speech detected in audio"}), 400

        # Now process like text question
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
                    character['llm_config']
                )
            except Exception as e:
                print(f"Text generation failed: {e}")
                styled_response = "I'm sorry, I'm having trouble generating a response right now."

        # Generate audio response
        audio_path = None
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

                # Check if TTS model is already prepared for this character
                tts_ready = model_cache.get(
                    f"{character_name}_tts_ready", False)

                # Set output directory to be within storage so files can be served
                voice_settings_copy = voice_settings.copy()
                voice_settings_copy['output_dir'] = str(
                    STORAGE_DIR / "generated_audio")

                audio_path = generate_audio(
                    model=tts_model,
                    ref_audio=character['voice_cloning_audio_path'],
                    ref_text=ref_text,
                    gen_text=styled_response,
                    config=voice_settings_copy,
                    auto_download=not tts_ready  # Skip download if already prepared
                )

            except Exception as e:
                print(f"Audio generation failed: {e}")

        return jsonify({
            "transcribed_question": question,
            "text_response": styled_response,
            "audio_url": get_file_url(audio_path, 'audio'),
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


if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
