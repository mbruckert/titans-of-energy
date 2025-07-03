# Titans API Documentation

A comprehensive AI character interaction API that supports text-to-speech, speech-to-text, knowledge base integration, and style-based text generation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [GPU Setup](#gpu-setup)
- [API Endpoints](#api-endpoints)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [Model Support](#model-support)
- [Troubleshooting](#troubleshooting)

## Overview

The Titans API allows you to create AI characters with:

- **Voice Cloning**: Generate speech using reference audio
- **Knowledge Base**: Upload documents for context-aware responses
- **Style Tuning**: Train characters to respond in specific styles
- **Speech Recognition**: Process audio input and convert to text
- **LLM Integration**: Support for local and API-based language models

## Installation

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

2. **Install spaCy model:**

```bash
python -m spacy download en_core_web_sm
```

3. **Install system dependencies (for audio processing):**

**Ubuntu/Debian:**

```bash
sudo apt install espeak-ng portaudio19-dev
```

**macOS:**

```bash
brew install espeak-ng portaudio
```

4. **Optional: Set up Hugging Face authentication (for private/gated models):**

```bash
# Get your token from https://huggingface.co/settings/tokens
huggingface-cli login
# Or set HUGGINGFACE_API_KEY in your .env file
```

## Environment Setup

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DB_NAME=titans_db
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432

# Optional Model Paths
GGUF_MODEL_PATH=./models/your-model.gguf
MODELS_DIR=./models
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=text-embedding-ada-002

# Optional API Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_OPENAI_MODEL=gpt-3.5-turbo

# Optional Hugging Face Configuration
HUGGINGFACE_API_KEY=your_huggingface_token_here
```

**Note about Hugging Face Authentication:**

- Required for private or gated models (like Llama, Gemma, etc.)
- Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Choose "Read" access for downloading models
- Public models work without authentication

## GPU Setup

### NVIDIA GPU Support (Linux)

For optimal performance with NVIDIA GPUs, follow these steps:

#### 1. Install NVIDIA Container Runtime

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-runtime
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# Restart Docker
sudo systemctl restart docker
```

#### 2. Verify GPU Access

```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 3. Docker Compose with GPU

**Option A: Modern GPU syntax (Docker Compose 1.28+)**

```bash
docker-compose -f docker-compose.yaml -f docker-compose.gpu.yaml up
```

**Option B: Legacy GPU syntax (if Option A fails)**

```bash
docker-compose -f docker-compose.yaml -f docker-compose.gpu-legacy.yaml up
```

#### 4. Manual Docker Runtime Configuration

If you get "could not select device driver nvidia" error, manually configure Docker:

1. Edit `/etc/docker/daemon.json`:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

2. Restart Docker:

```bash
sudo systemctl restart docker
```

#### 5. Environment Variables

Set these environment variables for GPU optimization:

```env
# Force GPU usage
ENABLE_GPU=true

# NVIDIA specific
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=all
```

### Apple Silicon (macOS)

Apple Silicon Macs are automatically detected and optimized when running natively (not in Docker). GPU acceleration uses Metal Performance Shaders (MPS) when available.

### CPU-Only Mode

If you don't have a GPU or want to run in CPU mode:

```env
ENABLE_GPU=false
```

### Troubleshooting GPU Issues

**Error: "could not select device driver nvidia"**

1. Install nvidia-container-runtime: `sudo apt-get install nvidia-container-runtime`
2. Restart Docker: `sudo systemctl restart docker`
3. Try legacy syntax: `docker-compose -f docker-compose.yaml -f docker-compose.gpu-legacy.yaml up`

**Error: "nvidia-smi not found"**

1. Install NVIDIA drivers: `sudo apt-get install nvidia-driver-XXX` (replace XXX with your driver version)
2. Reboot your system
3. Verify with: `nvidia-smi`

**Error: "CUDA out of memory"**

1. Reduce batch sizes in model configurations
2. Use smaller models (e.g., "tiny" instead of "base" for Whisper)
3. Set `CUDA_VISIBLE_DEVICES=0` to use only one GPU

## API Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running.

**Response:**

```json
{
  "message": "Titans API - Ready for character interactions!",
  "status": "success"
}
```

### 2. Create Character

**POST** `/create-character`

Create a new AI character with all associated models and data.

**Content-Type:** `multipart/form-data`

**Form Parameters:**

- `name` (string, required): Character name
- `llm_model` (string, optional): LLM model name/path
- `llm_config` (JSON string, optional): LLM configuration
- `voice_cloning_settings` (JSON string, optional): Voice cloning configuration
- `stt_settings` (JSON string, optional): Speech-to-text configuration

**File Uploads:**

- `character_image` (file, optional): Character image (PNG, JPG)
- `knowledge_base_file` (file, optional): Knowledge base document (TXT, JSON)
- `voice_cloning_audio` (file, optional): Reference audio for voice cloning (WAV, MP3)
- `style_tuning_file` (file, optional): Style examples (JSON)

**Example LLM Config:**

```json
{
  "api_key": "your_api_key",
  "base_url": "https://api.openai.com/v1",
  "model_name": "gpt-3.5-turbo",
  "max_tokens": 200,
  "temperature": 0.7,
  "system_prompt": "You are a helpful assistant."
}
```

**Example Voice Cloning Settings:**

```json
{
  "model": "f5tts",
  "reference_text": "Hello, this is my voice sample.",
  "preprocess_audio": true,
  "language": "en"
}
```

**Example STT Settings:**

```json
{
  "model": "whisper",
  "model_size": "base",
  "language": "en"
}
```

**Response:**

```json
{
  "message": "Character created successfully",
  "character_id": 1,
  "status": "success"
}
```

### 3. Get Characters

**GET** `/get-characters`

Retrieve list of all characters.

**Response:**

```json
{
  "characters": [
    {
      "id": 1,
      "name": "Character Name",
      "image_url": "http://localhost:5000/serve-image/character_name/image.jpg",
      "llm_model": "gpt-3.5-turbo",
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "status": "success"
}
```

### 4. Get Character Details

**GET** `/get-character/<character_id>`

Get detailed information about a specific character.

**Response:**

```json
{
  "character": {
    "id": 1,
    "name": "Character Name",
    "image_url": "http://localhost:5000/serve-image/character_name/image.jpg",
    "llm_model": "gpt-3.5-turbo",
    "llm_config": {...},
    "voice_cloning_settings": {...},
    "stt_settings": {...},
    "created_at": "2024-01-01T00:00:00"
  },
  "status": "success"
}
```

### 5. Load Character

**POST** `/load-character`

Preload all models associated with a character for faster inference.

**Request Body:**

```json
{
  "character_id": 1
}
```

**Response:**

```json
{
  "message": "Models loaded for character Character Name",
  "character_id": 1,
  "status": "success"
}
```

### 6. Ask Question (Text)

**POST** `/ask-question-text`

Send a text question to a character and receive text + audio response.

**Request Body:**

```json
{
  "character_id": 1,
  "question": "What is artificial intelligence?"
}
```

**Response:**

```json
{
  "question": "What is artificial intelligence?",
  "text_response": "Artificial intelligence is...",
  "audio_url": "http://localhost:5000/serve-audio/character_name/generated_audio_123456.wav",
  "knowledge_context": "Document 1: AI is a field of computer science...",
  "character_name": "Character Name",
  "status": "success"
}
```

### 7. Ask Question (Audio)

**POST** `/ask-question-audio`

Send an audio question to a character and receive text + audio response.

**Content-Type:** `multipart/form-data`

**Form Parameters:**

- `character_id` (string, required): Character ID
- `audio_file` (file, required): Audio file containing the question

**Response:**

```json
{
  "transcribed_question": "What is artificial intelligence?",
  "text_response": "Artificial intelligence is...",
  "audio_url": "http://localhost:5000/serve-audio/character_name/generated_audio_123456.wav",
  "knowledge_context": "Document 1: AI is a field of computer science...",
  "character_name": "Character Name",
  "status": "success"
}
```

### 8. Download Model

**POST** `/download-model`

Download a model for local use.

**Request Body:**

```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "model_type": "huggingface"
}
```

**Supported Model Types:**

- `huggingface`: Hugging Face models
- `openai`: OpenAI models (no download needed)

**Response:**

```json
{
  "model_path": "/path/to/downloaded/model",
  "status": "success"
}
```

### 9. Get TTS Models

**GET** `/get-tts-models`

Get list of available TTS models and their status.

**Response:**

```json
{
  "models": [
    {
      "name": "f5tts",
      "available": true,
      "dependencies": ["f5-tts CLI"]
    },
    {
      "name": "xtts",
      "available": false,
      "dependencies": ["TTS"]
    }
  ],
  "status": "success"
}
```

### 10. Serve Files

**GET** `/serve-audio/<path:filename>`
**GET** `/serve-image/<path:filename>`

Serve generated audio files and character images.

## Configuration Options

### LLM Configuration

**For OpenAI-compatible APIs:**

```json
{
  "api_key": "your_api_key",
  "base_url": "https://api.openai.com/v1",
  "model_name": "gpt-3.5-turbo",
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "system_prompt": "You are a helpful assistant.",
  "stop_tokens": ["\nUser:", "\nAssistant:"]
}
```

**For GGUF Models:**

```json
{
  "model_path": "./models/model.gguf",
  "context_length": 4096,
  "gpu_layers": -1,
  "batch_size": 1024,
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**For Hugging Face Models:**

```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "device": "auto",
  "torch_dtype": "float16",
  "max_tokens": 200,
  "temperature": 0.7,
  "trust_remote_code": false
}
```

### Voice Cloning Configuration

**F5-TTS:**

```json
{
  "model": "f5tts",
  "reference_text": "This is the text that was spoken in the reference audio",
  "preprocess_audio": true,
  "language": "en",
  "cuda_device": "0"
}
```

**XTTS-v2:**

```json
{
  "model": "xtts",
  "reference_text": "Reference audio transcription",
  "preprocess_audio": true,
  "language": "en",
  "coqui_tos_agreed": true
}
```

**Zonos:**

```json
{
  "model": "zonos",
  "reference_text": "Reference audio transcription",
  "preprocess_audio": true,
  "language": "en-us"
}
```

### Audio Preprocessing Configuration

```json
{
  "clean_audio": true,
  "remove_silence": true,
  "enhance_audio": true,
  "top_db": 40.0,
  "fade_length_ms": 50,
  "bass_boost": true,
  "treble_boost": true,
  "compression": true
}
```

### Speech-to-Text Configuration

**Whisper:**

```json
{
  "model": "whisper",
  "model_size": "base",
  "language": "en",
  "threshold": 0.05,
  "silence": 2.0,
  "max_time": 30,
  "min_duration": 0.5
}
```

**Wav2Vec2:**

```json
{
  "model": "wav2vec",
  "model_name": "facebook/wav2vec2-large-960h",
  "threshold": 0.05,
  "silence": 2.0,
  "max_time": 30
}
```

## Usage Examples

### Creating a Character with Python

```python
import requests

# Character data
files = {
    'character_image': open('character.jpg', 'rb'),
    'knowledge_base_file': open('knowledge.txt', 'rb'),
    'voice_cloning_audio': open('voice_sample.wav', 'rb'),
    'style_tuning_file': open('style_examples.json', 'rb')
}

data = {
    'name': 'AI Assistant',
    'llm_model': 'gpt-3.5-turbo',
    'llm_config': json.dumps({
        'api_key': 'your_key',
        'base_url': 'https://api.openai.com/v1',
        'max_tokens': 200,
        'temperature': 0.7
    }),
    'voice_cloning_settings': json.dumps({
        'model': 'f5tts',
        'reference_text': 'Hello, this is my voice.',
        'preprocess_audio': True
    }),
    'stt_settings': json.dumps({
        'model': 'whisper',
        'model_size': 'base'
    })
}

response = requests.post('http://localhost:5000/create-character',
                        files=files, data=data)
print(response.json())
```

### Asking a Text Question

```python
import requests

response = requests.post('http://localhost:5000/ask-question-text',
                        json={
                            'character_id': 1,
                            'question': 'What is machine learning?'
                        })
result = response.json()
print(f"Response: {result['text_response']}")
print(f"Audio URL: {result['audio_url']}")
```

### Style Tuning Data Format

Create a JSON file with question-answer pairs:

```json
[
  {
    "question": "How are you today?",
    "response": "I'm doing wonderfully, thank you for asking! How about you?"
  },
  {
    "question": "What's your favorite color?",
    "response": "I find myself drawn to deep blues - they remind me of the ocean depths."
  }
]
```

## Model Support

### Text-to-Speech Models

| Model   | Description                | Requirements                 |
| ------- | -------------------------- | ---------------------------- |
| F5-TTS  | High-quality voice cloning | `f5-tts` package             |
| XTTS-v2 | Multilingual TTS           | `TTS` package                |
| Zonos   | Advanced voice synthesis   | `zonos` package, `espeak-ng` |

### Speech-to-Text Models

| Model    | Description                  | Requirements     |
| -------- | ---------------------------- | ---------------- |
| Whisper  | OpenAI's speech recognition  | `openai-whisper` |
| Wav2Vec2 | Facebook's speech model      | `transformers`   |
| HuBERT   | Self-supervised speech model | `transformers`   |

### Language Models

| Type         | Description            | Requirements            |
| ------------ | ---------------------- | ----------------------- |
| OpenAI API   | GPT models via API     | OpenAI API key          |
| GGUF         | Quantized local models | `llama-cpp-python`      |
| Hugging Face | Transformer models     | `transformers`, `torch` |

## Troubleshooting

### Common Issues

**1. Audio Processing Errors:**

- Install system audio dependencies: `portaudio19-dev`, `espeak-ng`
- Check audio file format (WAV recommended)

**2. Model Loading Failures:**

- Verify model paths and permissions
- Check available disk space for model downloads
- Ensure CUDA is available for GPU acceleration

**3. Database Connection Issues:**

- Verify PostgreSQL is running
- Check database credentials in `.env`
- Ensure database exists and user has permissions

**4. Memory Issues:**

- Reduce model batch sizes
- Use smaller models for testing
- Monitor system memory usage

**5. Hugging Face Authentication Issues:**

- Verify your token has "Read" permissions
- Check token validity at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- For gated models, ensure you've accepted the model's license agreement
- Try logging in via CLI: `huggingface-cli login`

### Performance Optimization

**For Apple Silicon Macs:**

- Use Metal Performance Shaders (MPS) when available
- Reduce batch sizes for GGUF models
- Enable memory mapping optimizations

**For CUDA Systems:**

- Set appropriate GPU layers for GGUF models
- Use mixed precision for Hugging Face models
- Monitor GPU memory usage

### API Rate Limits

- OpenAI API: Respect rate limits based on your plan
- Local models: Limited by hardware capabilities
- File uploads: Maximum size depends on server configuration

## Support

For issues and questions:

1. Check the troubleshooting section
2. Verify your environment setup
3. Review the configuration examples
4. Check system requirements and dependencies

The API provides detailed error messages to help diagnose issues. Enable debug mode for additional logging information.
