# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy English model
RUN python -m spacy download en_core_web_sm

# Install F5TTS
RUN pip install f5-tts

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/knowledge data/archive data/style chroma_db

# Expose the port the app runs on
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "main.py"]
