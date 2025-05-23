# Titans of Energy API

A Flask-based API for generating styled text responses and cloned audio using J. Robert Oppenheimer's persona. This system combines knowledge base querying, few-shot learning for text styling, and voice cloning capabilities.

## Features

- **Knowledge Base Processing**: Automatically process and embed documents for contextual retrieval
- **Few-Shot Text Styling**: Generate responses in Oppenheimer's voice using GGUF models
- **Voice Cloning**: Generate audio using F5-TTS with reference voice samples
- **ChromaDB Integration**: Persistent vector storage for efficient similarity search

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM available
- **Storage**: ~3GB for models and dependencies

### Environment Setup

```bash
# Create and activate conda environment
conda create -n titans-of-energy
conda activate titans-of-energy

# Verify Python version
python --version
```

## Installation

### 1. Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Install spaCy English model
python -m spacy download en_core_web_sm
```

### 2. Download Required Models

#### GGUF Model (Required)

```bash
# Create models directory
mkdir -p models

# Install Hugging Face CLI if not already installed
pip install huggingface_hub

# Download the GGUF model
huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-q4_0.gguf --local-dir ./models
```

**Note**: The GGUF model file is required and must be located at `./models/gemma-3-4b-it-q4_0.gguf`

### 3. Environment Variables

Create a `.env` file in the API directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Starting the API

```bash
python main.py
```

The API will start on `http://localhost:5001`

### API Endpoints

#### 1. Preprocess Knowledge Base

**POST** `/preprocess`

Processes documents in `./data/knowledge/` and generates embeddings for both knowledge base and styling examples.

```bash
curl -X POST http://localhost:5001/preprocess
```

**Response:**

```json
{
  "message": "Preprocessing complete",
  "status": "success"
}
```

#### 2. Generate Styled Response with Audio

**POST** `/generate`

Generates a styled text response and corresponding audio in Oppenheimer's voice.

```bash
curl -X POST http://localhost:5001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was your role in the Manhattan Project?",
    "ref_file": "data/voice/reference.wav",
    "ref_text": "Reference text that matches the audio",
    "output_file": "output/response.wav"
  }'
```

**Response:**

```json
{
  "message": "Styled text and audio generated",
  "status": "success",
  "styled_text": "Generated response in Oppenheimer's style...",
  "audio_base64": "base64_encoded_audio_data",
  "timing": {
    "context_query_time": 0.45,
    "text_styling_time": 2.31,
    "audio_generation_time": 8.92,
    "total_time": 11.68
  }
}
```
