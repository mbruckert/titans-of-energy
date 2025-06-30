# Titans of Energy

A Robust Pipeline to train new virtual clones of historical characters, starting with Oppenheimer.

## Project Overview

- Ability to talk with Oppenheimer
  - Speech-to-text to take voice input and convert to text
  - Knowledge base of facts about Oppenheimer
  - Stylized LLM that speaks in the style (structure, vocab, etc.) of Oppenheimer
  - Toggleable LLMs
  - Voice cloning for output that sounds like Oppenheimer
- Pipeline to train new characters
  - Provide the pipeline with style, voice, and knowledge data -> trains a new person
  - Frontend/CLI to make this process easy

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for CLI usage)
- Git

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd titans-of-energy

# Create your environment file from the example
cp env.example .env
```

### 2. Configure Environment Variables

Edit the `.env` file with your API keys and configuration:

```bash
# Required API Keys
OPENAI_API_KEY=your_actual_openai_api_key_here
HUGGINGFACE_API_KEY=your_actual_huggingface_api_key_here

# Database settings are pre-configured for Docker
# Model paths can be customized if needed
```

**Important**: Replace `your_openai_api_key_here` and `your_huggingface_api_key_here` with your actual API keys.

### 3. Start the Application with Docker

```bash
# Build and start all services (API, Database, Frontend)
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

This will start:

- **API Server**: http://localhost:5000
- **Frontend**: http://localhost:3000
- **PostgreSQL Database**: Internal (port 5432)

### 4. Verify Services

```bash
# Check if all services are running
docker-compose ps

# View logs
docker-compose logs -f
```

## Speech-to-Text CLI Usage

### Install CLI Dependencies

```bash
# Install Python dependencies for the CLI
pip install -r requirements-cli.txt
```

### CLI Features

The unified speech-to-text CLI supports multiple models:

- **Whisper** (OpenAI) - Various sizes: tiny, base, small, medium, large, large-v2
- **Wav2Vec2** (Facebook) - Large 960h model
- **HuBERT** (Facebook) - Large LS960 fine-tuned model

### Basic Usage

```bash
# Use Whisper (default) with character ID 1
python character_interaction.py --character_id 1

# Use different Whisper model size
python character_interaction.py --model whisper --whisper_size base --character_id 1

# Use Wav2Vec2 model
python character_interaction.py --model wav2vec2 --character_id 1

# Use HuBERT model
python character_interaction.py --model hubert --character_id 1
```

### Advanced CLI Options

```bash
# Customize voice detection sensitivity
python character_interaction.py --threshold 0.03 --character_id 1

# Adjust silence detection (seconds before stopping)
python character_interaction.py --silence 1.5 --character_id 1

# Set maximum session time
python character_interaction.py --max_time 600 --character_id 1

# Use different API endpoint
python character_interaction.py --api_endpoint http://localhost:5000 --character_id 1
```

### CLI Workflow

1. **Start the CLI**: Run with your desired model and character ID
2. **Voice Detection**: Speak when you see "🎧 Listening..."
3. **Recording**: CLI automatically starts/stops based on voice activity
4. **Transcription**: Your speech is converted to text using the selected model
5. **API Call**: Text is sent to the character API for processing
6. **Audio Response**: Character's voice response is played back

### CLI Command Reference

```bash
python character_interaction.py [OPTIONS]

Options:
  --model {whisper,wav2vec2,hubert}  Choose STT model (default: whisper)
  --whisper_size {tiny,base,small,medium,large,large-v2}  Whisper model size
  --threshold FLOAT                  Volume threshold to start recording (default: 0.05)
  --silence FLOAT                    Seconds of silence before stopping (default: 2.0)
  --max_time INT                     Max session time in seconds (default: 300)
  --output STR                       Output folder for recordings (default: .)
  --api_endpoint STR                 Base URL for API (default: http://localhost:5000)
  --character_id INT                 Character ID to use for responses (required)
```

## Architecture

### Services

- **API (`./api/`)**: Python Flask backend with ML models
- **Frontend (`./ui/frontend/`)**: React/TypeScript web interface
- **Database**: PostgreSQL for persistent data
- **CLI**: Unified speech-to-text interface

### Persistent Storage

The following directories persist between container restarts:

- `chroma_data/`: ChromaDB vector database
- `storage_data/`: File storage (audio, images, etc.)
- `outputs_data/`: Generated content
- `db_data/`: PostgreSQL database

## Environment Variables Reference

| Variable              | Description                       | Default                  |
| --------------------- | --------------------------------- | ------------------------ |
| `OPENAI_API_KEY`      | OpenAI API key for embeddings/LLM | Required                 |
| `HUGGINGFACE_API_KEY` | Hugging Face API key              | Required                 |
| `DB_HOST`             | Database host                     | `db`                     |
| `DB_PORT`             | Database port                     | `5432`                   |
| `DB_USER`             | Database username                 | `titans_user`            |
| `DB_PASSWORD`         | Database password                 | `titans_password`        |
| `DB_NAME`             | Database name                     | `titans_db`              |
| `CHROMA_DB_PATH`      | ChromaDB storage path             | `chroma_db`              |
| `MODELS_DIR`          | Local models directory            | `./models`               |
| `EMBEDDING_MODEL`     | OpenAI embedding model            | `text-embedding-ada-002` |

## Additional Resources

- Check `api/README.md` for detailed API documentation
- See `api/demo.ipynb` for usage examples
- Review `api/setup.sh` for manual setup instructions
