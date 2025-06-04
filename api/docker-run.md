# Docker Setup for Titans API

This guide explains how to run the Titans API using Docker and Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (required)
- Hugging Face API key (optional, for private models)

## Architecture-Specific Notes

### ARM64 Systems (Apple Silicon, ARM servers)

If you're running on ARM64 architecture and encounter build issues with `llama-cpp-python`, use the ARM64-specific setup:

```bash
# For ARM64 systems, use the ARM64 profile
docker-compose --profile arm64 up --build
```

Or manually specify the ARM64 Dockerfile:

```bash
docker build -f Dockerfile.arm64 -t titans-api .
```

**Note**: The ARM64 version excludes `llama-cpp-python` to avoid compilation issues. Local GGUF model support will be limited, but OpenAI API and Hugging Face models will work normally.

## Quick Start

1. **Set up environment variables:**

   ```bash
   # Create a .env file in the project root
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "HUGGINGFACE_API_KEY=your_huggingface_token_here" >> .env
   ```

2. **Build and run with Docker Compose:**

   **For x86_64 systems:**

   ```bash
   docker-compose up --build
   ```

   **For ARM64 systems (Apple Silicon, etc.):**

   ```bash
   docker-compose --profile arm64 up --build
   ```

3. **Access the API:**
   - API: http://localhost:5000
   - PostgreSQL: localhost:5432

## Environment Variables

### Required

- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM inference

### Optional

- `HUGGINGFACE_API_KEY`: For accessing private/gated Hugging Face models
- `DB_NAME`: Database name (default: titans_db)
- `DB_USER`: Database user (default: titans_user)
- `DB_PASSWORD`: Database password (default: titans_password)
- `DB_HOST`: Database host (default: postgres)
- `DB_PORT`: Database port (default: 5432)

## Docker Commands

### Build the image

**For x86_64:**

```bash
docker build -t titans-api .
```

**For ARM64:**

```bash
docker build -f Dockerfile.arm64 -t titans-api .
```

### Run with Docker Compose (recommended)

**Standard (x86_64):**

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f titans-api

# Stop services
docker-compose down
```

**ARM64 systems:**

```bash
# Start services with ARM64 profile
docker-compose --profile arm64 up -d

# View logs
docker-compose logs -f titans-api-arm64

# Stop services
docker-compose --profile arm64 down
```

### Run standalone (requires external PostgreSQL)

```bash
docker run -d \
  --name titans-api \
  -p 5000:5000 \
  -e OPENAI_API_KEY=your_key_here \
  -e DB_HOST=your_postgres_host \
  -e DB_NAME=titans_db \
  -e DB_USER=titans_user \
  -e DB_PASSWORD=titans_password \
  -v titans_storage:/app/storage \
  -v titans_models:/app/models \
  -v titans_chroma:/app/chroma_db \
  titans-api
```

## Data Persistence

The Docker setup uses named volumes to persist data:

- `titans_storage`: Generated audio files and character data
- `titans_models`: Downloaded AI models
- `titans_chroma`: Vector database files
- `postgres_data`: PostgreSQL database files

## Accessing Services

### API Endpoints

- Health check: `GET http://localhost:5000/`
- Create character: `POST http://localhost:5000/create-character`
- List characters: `GET http://localhost:5000/get-characters`
- Ask question (text): `POST http://localhost:5000/ask-question-text`
- Ask question (audio): `POST http://localhost:5000/ask-question-audio`

### Database Access

```bash
# Connect to PostgreSQL
docker exec -it titans-postgres psql -U titans_user -d titans_db
```

## Troubleshooting

### Common Issues

1. **Build fails with llama-cpp-python errors on ARM64:**

   - **Solution**: Use the ARM64-specific build:
     ```bash
     docker-compose --profile arm64 up --build
     ```
   - **Alternative**: Build with the ARM64 Dockerfile:
     ```bash
     docker build -f Dockerfile.arm64 -t titans-api .
     ```
   - **Note**: This excludes local GGUF model support but maintains full API functionality

2. **ARM NEON instruction errors:**

   - This is a known issue with llama-cpp-python on ARM64
   - The main Dockerfile attempts to fix this with CMAKE flags
   - If issues persist, use `Dockerfile.arm64` which skips llama-cpp-python entirely

3. **Build fails with Rust errors:**

   - The Dockerfile installs Rust automatically for deepfilternet
   - If issues persist, try building without cache: `docker-compose build --no-cache`

4. **Out of memory during build:**

   - Increase Docker memory limit in Docker Desktop settings
   - Consider using a multi-stage build for production

5. **Permission errors:**

   - The container runs as user `titans` (UID 1000)
   - Ensure volume permissions are correct

6. **Database connection fails:**
   - Wait for PostgreSQL to be ready (health check should handle this)
   - Check logs: `docker-compose logs postgres`

### Architecture Detection

To check your system architecture:

```bash
# Check your system architecture
uname -m

# x86_64 = Intel/AMD 64-bit (use standard Dockerfile)
# aarch64 = ARM 64-bit (use Dockerfile.arm64)
```

### Debugging

```bash
# Enter the running container (standard)
docker exec -it titans-api bash

# Enter the running container (ARM64)
docker exec -it titans-api-arm64 bash

# Check logs (standard)
docker-compose logs titans-api

# Check logs (ARM64)
docker-compose logs titans-api-arm64

# Restart a specific service
docker-compose restart titans-api
```

## Production Considerations

1. **Security:**

   - Use secrets management for API keys
   - Run behind a reverse proxy (nginx, traefik)
   - Enable HTTPS

2. **Performance:**

   - Consider using a multi-stage build to reduce image size
   - Use external PostgreSQL for better performance
   - Mount model directories for faster startup

3. **Monitoring:**
   - Add health checks and monitoring
   - Set up log aggregation
   - Monitor resource usage

## Model Management

### Pre-loading Models

To speed up first-time usage, you can pre-download models:

```bash
# Mount a local models directory
docker run -v ./models:/app/models titans-api python -c "
from libraries.tts.preprocess import download_voice_models
download_voice_models()
"
```

### Custom Models

- **x86_64**: Place GGUF models in the `models/` directory (will be persisted in `titans_models` volume)
- **ARM64**: GGUF models are not supported due to llama-cpp-python exclusion. Use OpenAI API or Hugging Face models instead

### ARM64 Limitations

When using the ARM64 build (`Dockerfile.arm64`):

- ✅ OpenAI API models (GPT-3.5, GPT-4, etc.)
- ✅ Hugging Face models
- ✅ TTS (F5-TTS, XTTS, etc.)
- ✅ STT (Whisper, etc.)
- ✅ Vector databases and embeddings
- ❌ Local GGUF models (llama-cpp-python not available)

## Backup and Restore

### Backup

```bash
# Backup database
docker exec titans-postgres pg_dump -U titans_user titans_db > backup.sql

# Backup volumes
docker run --rm -v titans_storage:/data -v $(pwd):/backup alpine tar czf /backup/storage-backup.tar.gz -C /data .
```

### Restore

```bash
# Restore database
docker exec -i titans-postgres psql -U titans_user titans_db < backup.sql

# Restore volumes
docker run --rm -v titans_storage:/data -v $(pwd):/backup alpine tar xzf /backup/storage-backup.tar.gz -C /data
```
