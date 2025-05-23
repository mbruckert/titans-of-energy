# macOS Setup Guide for Fewshot Styling (GGUF Only)

This guide provides macOS-specific instructions for setting up the fewshot styling system using only GGUF models for optimal performance.

## Prerequisites

### 1. Python Environment

```bash
# Ensure you have Python 3.8+ installed
python3 --version

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

#### For Apple Silicon Macs (M1/M2/M3):

```bash
# Install llama-cpp-python with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Install remaining dependencies
pip install -r requirements.txt
```

#### For Intel Macs:

```bash
# Standard installation
pip install -r requirements.txt
```

## Model Setup (Required)

### Download the GGUF Model

```bash
# Create models directory
mkdir -p models

# Install Hugging Face CLI if not already installed
pip install huggingface_hub

# Download the GGUF model (Required - no fallback available)
huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-qat-q4_0.gguf --local-dir ./models
```

**Note**: The GGUF model file is required. The system will not work without it.

## Performance Optimizations

### Apple Silicon Optimizations

- **Metal GPU Acceleration**: Automatically enabled for GGUF models
- **Memory Mapping**: Enabled for efficient model loading
- **Optimal Threading**: Auto-detected based on your CPU cores

### Memory Requirements

- **GGUF Model**: ~2.5GB RAM (quantized)

## Troubleshooting

### Common Issues

#### 1. Model File Not Found

```bash
# Ensure the model file exists
ls -la ./models/gemma-3-4b-it-qat-q4_0.gguf

# If missing, download it:
huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-qat-q4_0.gguf --local-dir ./models
```

#### 2. llama-cpp-python Installation Issues

```bash
# Reinstall with Metal support (Apple Silicon)
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

# For Intel Macs
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir
```

#### 3. Memory Issues

- Close other applications to free up RAM
- Ensure you have at least 4GB of available RAM
- Keep your Mac plugged in for maximum performance

### Performance Tips

1. **Close unnecessary applications** to free up memory
2. **Use Activity Monitor** to check memory usage
3. **Keep your Mac plugged in** for maximum performance
4. **Ensure adequate cooling** for sustained performance

## Verification

Test the installation:

```python
from fewshot_styling import generate_styled_text

# Test generation
result = generate_styled_text(
    question="What was your role in the Manhattan Project?",
    context="Historical context about Oppenheimer's work",
    max_new_tokens=100
)
print(result)
```

## Expected Performance

### Apple Silicon (M1/M2/M3):

- **GGUF Model with Metal**: ~15-25 tokens/second

### Intel Mac:

- **GGUF Model (CPU)**: ~3-8 tokens/second

Performance varies based on context length, available memory, and thermal conditions.

## Advantages of GGUF-Only Approach

1. **Smaller Memory Footprint**: ~2.5GB vs 8GB+ for full models
2. **Faster Loading**: Optimized binary format loads quickly
3. **Better Performance**: Quantized models run faster on consumer hardware
4. **Metal Acceleration**: Full GPU acceleration on Apple Silicon
5. **Consistent Behavior**: Single model type eliminates variability
