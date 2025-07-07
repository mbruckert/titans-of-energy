# Titans of Energy

A comprehensive AI platform for creating interactive virtual characters with voice cloning, speech-to-text, knowledge base integration, and style-tuned language models. Create and interact with historical figures like Oppenheimer through voice conversation.

## 🚀 Key Features

### Character Creation & Management
- **Visual Character Builder**: Web-based interface for creating new characters
- **Multi-modal Input**: Upload character images, voice samples, knowledge documents, and style examples
- **Character Profiles**: Persistent character storage with configuration management
- **Wakeword Customization**: Set custom wake phrases for each character
- **Character Optimization**: Automatic model preloading for faster response times

### Voice & Audio
- **Voice Cloning**: Multiple TTS models including F5-TTS, XTTS-v2, and Zonos
- **Speech-to-Text**: Whisper, Wav2Vec2, and HuBERT models for voice recognition
- **Audio Preprocessing**: Automatic noise reduction, silence removal, and audio enhancement
- **Voice Interaction CLI**: Real-time voice conversations with wake-word detection
- **Audio Similarity Scoring**: Voice quality validation using Resemblyzer
- **Thinking Audio**: Character-specific audio clips for natural conversation flow

### AI & Knowledge
- **Knowledge Base Integration**: Vector database with ChromaDB for contextual responses
- **Multiple Embedding Models**: Support for OpenAI, Sentence Transformers, and custom models
- **LLM Support**: OpenAI API, GGUF models, and Hugging Face transformers
- **Style Tuning**: Train characters to speak in specific styles and personalities
- **Conversation Memory**: Persistent chat history and context management
- **Embedding Model Compatibility**: Automatic detection and migration for model changes

### Platform Support
- **Cross-Platform**: Native support for Windows, macOS, and Linux
- **GPU Acceleration**: NVIDIA CUDA and Apple Silicon (MPS) optimization
- **Docker Support**: Containerized deployment with GPU support
- **Device Optimization**: Automatic hardware detection and model optimization
- **Performance Monitoring**: Model loading times and memory usage tracking

## 📋 Prerequisites

### For Docker (Recommended)
- Docker and Docker Compose
- For GPU support: NVIDIA drivers and NVIDIA Container Toolkit (Linux/Windows)
- 8GB+ RAM recommended
- 10GB+ storage for models

### For Native Installation
- Python 3.8+ 
- Node.js 16+ and npm
- PostgreSQL database
- Conda/Miniconda (recommended)
- For GPU: NVIDIA drivers or Apple Silicon
- For Zonos TTS: espeak-ng system package

### System Dependencies
**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install espeak-ng postgresql postgresql-contrib
```

**macOS:**
```bash
brew install espeak postgresql
```

**Windows:**
- Install PostgreSQL from official website
- espeak-ng will be handled by the application

## 🐳 Docker Installation (Recommended)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd titans-of-energy

# Create environment file
cp env.example .env
```

### 2. Configure Environment
Edit `.env` file with your API keys:
```bash
# Required API Keys (at least one recommended)
OPENAI_API_KEY=your_actual_openai_api_key_here
HUGGINGFACE_API_KEY=your_actual_huggingface_api_key_here

# Database settings (pre-configured for Docker)
DB_HOST=db
DB_USER=titans_user
DB_PASSWORD=titans_password
DB_NAME=titans_db

# Embedding Configuration (optional - defaults to sentence transformers)
EMBEDDING_MODEL=text-embedding-ada-002
```

### 3. Start Services

**CPU-only (works on all platforms):**
```bash
docker-compose up --build
```

**With NVIDIA GPU (Linux/Windows):**
```bash
docker-compose -f docker-compose.gpu.yaml up --build
```

**Services will be available at:**
- Frontend: http://localhost:3000
- API: http://localhost:5000
- Database: Internal (PostgreSQL)

For detailed Docker setup including Windows/WSL2 and troubleshooting, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

## 💻 Native Installation

### 1. Database Setup
Install and start PostgreSQL, then create the database:
```sql
CREATE DATABASE titans_db;
CREATE USER titans_user WITH PASSWORD 'titans_password';
GRANT ALL PRIVILEGES ON DATABASE titans_db TO titans_user;
```

### 2. Backend Setup
```bash
# Create conda environment (recommended)
conda create -n titans-energy python=3.10
conda activate titans-energy

# Navigate to API directory
cd api

# Install Python dependencies
pip install -r requirements.txt

# Install spacy language model (required for knowledge base)
python -m spacy download en_core_web_sm

# Set environment variables (create .env file in api directory)
cat > .env << EOF
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
DB_HOST=localhost
DB_USER=titans_user
DB_PASSWORD=titans_password
DB_NAME=titans_db
CHROMA_DB_PATH=chroma_db
MODELS_DIR=./models
EOF

# Start the API server
python app.py
```

### 3. Frontend Setup
```bash
# In a new terminal, navigate to frontend
cd ui/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. CLI Dependencies (Optional)
For the character interaction CLI:
```bash
# Install CLI-specific dependencies
pip install -r requirements-cli.txt
```

### 5. TTS Model Dependencies (Automatic Setup)
The application automatically handles TTS model dependencies:

**F5-TTS & XTTS-v2:**
- Dependencies are automatically installed through Docker (see lines 128-131 in Dockerfile)
- For native installation, they're handled in the Dockerfile configuration
- **No manual installation required by users**

**Zonos (Experimental):**
- Automatically sets up a separate conda environment (`tts_zonos`) on first startup
- Downloads and installs Zonos from GitHub automatically
- Only requirement: ensure `espeak-ng` is installed system-wide (see System Dependencies above)
- **Environment creation and model installation is fully automated**

The system will automatically:
1. Detect missing TTS dependencies on startup
2. Set up required environments
3. Download and configure models
4. Handle version compatibility

**Note**: Users do not need to manually install F5-TTS, TTS, or Zonos packages.

## 🎤 Character Interaction CLI

The CLI provides real-time voice interaction with characters using wake-word detection.

### Basic Usage
```bash
# Start voice interaction with character ID 1
python character_interaction.py --character_id 1

# Use different speech-to-text models
python character_interaction.py --model whisper --whisper_size base --character_id 1
python character_interaction.py --model wav2vec2 --character_id 1
python character_interaction.py --model hubert --character_id 1
```

### Advanced Options
```bash
# Customize voice detection sensitivity
python character_interaction.py --threshold 0.03 --character_id 1

# Adjust wake-word detection sensitivity
python character_interaction.py --wakeword_threshold 0.01 --character_id 1

# Adjust silence detection (seconds before stopping)
python character_interaction.py --silence 1.5 --character_id 1

# Set maximum session time
python character_interaction.py --max_time 600 --character_id 1

# Use different API endpoint
python character_interaction.py --api_endpoint http://localhost:5000 --character_id 1
```

### CLI Workflow
1. **Wake-word Detection**: Say the character's wake phrase (e.g., "Hey Oppenheimer")
2. **Voice Recording**: Speak your question when prompted
3. **AI Processing**: Character processes your question using their knowledge base
4. **Voice Response**: Character responds in their cloned voice
5. **Thinking Audio**: Optional thinking sounds during processing

### CLI Command Reference
```bash
python character_interaction.py [OPTIONS]

Required:
  --character_id INT       Character ID to interact with

Speech-to-Text:
  --model {whisper,wav2vec2,hubert}  STT model (default: whisper)
  --whisper_size {tiny,base,small,medium,large,large-v2}  Model size

Audio Settings:
  --threshold FLOAT        Volume threshold to start recording (default: 0.05)
  --wakeword_threshold FLOAT  Wake-word detection threshold (default: 0.01)
  --silence FLOAT          Seconds of silence before stopping (default: 2.0)

Session Settings:
  --max_time INT           Max session time in seconds (default: 300)
  --output STR             Output folder for recordings (default: .)
  --api_endpoint STR       API base URL (default: http://localhost:5000)
```

## 🎭 Creating Characters

### Web Interface (Recommended)
1. Open http://localhost:3000
2. Click "Create New Character"
3. Follow the step-by-step wizard:
   - **Basic Info**: Name, image, wake-word
   - **Model Selection**: Choose LLM, download models, configure settings
   - **Knowledge Base**: Upload text documents (.txt, .pdf, .docx)
   - **Voice Cloning**: Upload voice sample and configure TTS model
   - **Style Tuning**: Upload conversation examples (JSON format)

### Supported File Types
- **Images**: JPG, PNG, GIF
- **Audio**: WAV, MP3, FLAC
- **Documents**: TXT, PDF, DOCX, MD
- **Style Data**: JSON files with question-answer pairs

### Voice Cloning Models
- **F5-TTS**: High quality, fast inference (recommended)
- **XTTS-v2**: Multilingual support, slower but versatile
- **Zonos**: Experimental model with emotion control

### LLM Model Types
- **OpenAI API**: GPT-3.5-turbo, GPT-4, custom endpoints
- **GGUF Models**: Quantized models for local inference
- **Hugging Face**: Transformer models (Llama, Gemma, etc.)

### Embedding Models
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small/large
- **Sentence Transformers**: all-MiniLM-L6-v2, BGE models, MPNet
- **Custom Models**: Support for custom embedding models

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings/LLM | Optional |
| `HUGGINGFACE_API_KEY` | Hugging Face API key | Optional |
| `DB_HOST` | Database host | `localhost` |
| `DB_USER` | Database username | `titans_user` |
| `DB_PASSWORD` | Database password | `titans_password` |
| `DB_NAME` | Database name | `titans_db` |
| `CHROMA_DB_PATH` | ChromaDB storage path | `chroma_db` |
| `MODELS_DIR` | Local models directory | `./models` |
| `EMBEDDING_MODEL` | Default embedding model | `text-embedding-ada-002` |

### Hardware Optimization
The system automatically detects and optimizes for:
- **NVIDIA GPUs**: CUDA acceleration, mixed precision, flash attention
- **Apple Silicon**: MPS optimization for M1/M2/M3/M4 Macs, Neural Engine support
- **CPU**: Multi-threading optimization, memory efficiency
- **Device-Specific Settings**: Batch sizes, context lengths, compilation options

### Embedding Model Configuration
```json
{
  "model_type": "sentence_transformers",
  "model_name": "all-MiniLM-L6-v2",
  "config": {
    "device": "auto",
    "batch_size": 32
  }
}
```

### Voice Cloning Configuration
```json
{
  "model": "f5tts",
  "preprocess_audio": true,
  "clean_audio": true,
  "remove_silence": true,
  "enhance_audio": true,
  "top_db": 40.0,
  "fade_length_ms": 50
}
```

## 📚 API Endpoints

### Character Management
- `POST /create-character` - Create new character
- `GET /get-characters` - List all characters
- `GET /get-character/<id>` - Get character details
- `PUT /update-character/<id>` - Update character
- `DELETE /delete-character/<id>` - Delete character
- `GET /get-character-wakeword/<id>` - Get character's wake phrase
- `GET /get-character-thinking-audio/<id>` - Get thinking audio clips

### Interaction
- `POST /ask-question-text` - Text-based conversation
- `POST /ask-question-audio` - Audio-based conversation
- `POST /transcribe-audio` - Speech-to-text transcription
- `GET /get-chat-history/<id>` - Get conversation history
- `DELETE /clear-chat-history/<id>` - Clear conversation history

### Model Management
- `GET /get-tts-models` - Available TTS models
- `GET /get-llm-models` - Available LLM models
- `POST /download-model` - Download models
- `POST /load-character` - Preload character models
- `POST /optimize-for-character` - Optimize system for character
- `POST /unload-all-models` - Unload all cached models

### System & Performance
- `GET /get-system-performance` - System status and metrics
- `GET /get-loaded-models` - Currently loaded models
- `GET /get-embedding-models` - Available embedding models
- `POST /test-embedding-config` - Test embedding configuration

### Advanced Features
- `POST /check-collection-compatibility` - Check embedding compatibility
- `POST /get-collection-diagnostics` - Detailed collection information
- `POST /preload-zonos-worker` - Preload Zonos TTS worker
- `GET /get-zonos-worker-status` - Zonos worker status

## 🔧 Troubleshooting

### Common Issues

**"Import errors" in native installation:**
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Install spacy model: `python -m spacy download en_core_web_sm`
- Check Python version (3.8+ required)
- For TTS models: `pip install f5-tts TTS`

**Database connection errors:**
- Verify PostgreSQL is running
- Check database credentials in `.env` file
- Ensure database and user exist

**GPU not detected:**
- Update NVIDIA drivers
- Install NVIDIA Container Toolkit (for Docker)
- Check CUDA compatibility
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**Audio issues in CLI:**
- Install system audio dependencies
- Check microphone permissions
- Verify sounddevice installation: `python -c "import sounddevice; print(sounddevice.query_devices())"`

**Character creation fails:**
- Check API logs for specific errors
- Verify all required files are uploaded
- Ensure sufficient disk space for model downloads
- Check API keys are valid

**Embedding model compatibility issues:**
- Collections are automatically migrated when embedding models change
- Check `/get-collection-diagnostics` endpoint for detailed info
- Use `/check-collection-compatibility` to verify compatibility

**Voice cloning quality issues:**
- Ensure reference audio is clear and 10-30 seconds long
- Use audio preprocessing options (enabled by default)
- Try different TTS models for your use case
- Check similarity scores in API responses

### Performance Tips
- Use GPU acceleration when available
- Preload character models with `/load-character` endpoint
- Use smaller Whisper models (base/small) for faster transcription
- Close unused characters to free memory
- Use `/optimize-for-character` for best performance
- Monitor system performance with `/get-system-performance`

### Zonos TTS Specific Issues
- Ensure espeak-ng is installed system-wide
- Zonos uses a separate Python worker process
- Check worker status with `/get-zonos-worker-status`
- Use `/preload-zonos-worker` for faster startup

## 📁 Project Structure

```
titans-of-energy/
├── api/                 # Backend API server
│   ├── app.py          # Main Flask application
│   ├── libraries/      # Core AI libraries
│   │   ├── tts/        # Text-to-speech models
│   │   │   ├── inference.py    # TTS generation
│   │   │   └── preprocess.py   # Audio preprocessing
│   │   ├── llm/        # Language models
│   │   │   ├── inference.py    # LLM generation
│   │   │   └── preprocess.py   # Model downloading
│   │   ├── knowledgebase/ # Vector database
│   │   │   ├── preprocess.py   # Document processing
│   │   │   └── retrieval.py    # Knowledge retrieval
│   │   └── utils/      # Utilities
│   │       ├── device_optimization.py # Hardware optimization
│   │       └── embedding_models.py    # Embedding management
│   ├── zonos_worker.py # Zonos TTS worker process
│   └── requirements.txt
├── ui/frontend/        # React web interface
│   └── src/pages/      # Character creation pages
│       ├── CharacterSelection.tsx
│       ├── ModelSelection.tsx
│       ├── KnowledgeBaseUpload.tsx
│       ├── VoiceCloningUpload.tsx
│       └── StyleTuningConfig.tsx
├── character_interaction.py # Voice interaction CLI
├── docker-compose.yaml # Docker configuration
├── DOCKER_SETUP.md     # Detailed Docker guide
└── README.md           # This file
```