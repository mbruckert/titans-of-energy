# Style Tuning CLI

A command-line tool for generating text in the style of J. Robert Oppenheimer using either style vector manipulation or few-shot learning. The tool supports both local transformer models and Ollama models for few-shot generation.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd titans-of-energy/clis/style_tuning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install Ollama:
   If you want to use the Ollama integration, you'll need to install Ollama first. Visit [Ollama's official website](https://ollama.ai) for installation instructions.

## Requirements

- Python 3.x
- (Optional) CUDA-compatible GPU or CPU
- (Optional) Ollama installed locally for Ollama-based generation

## Usage

The CLI supports three main commands:

1. `generate`: Generate styled text using various methods
2. `list-models`: List installed Ollama models
3. `pull-model`: Pull new models from Ollama library

### Generate Command

```bash
python cli.py generate --question "Your question here" --qa-file path/to/qa.json --method [style_vectors|few_shot|ollama_few_shot]
```

#### Required Arguments for Generate

- `--question`: The question to generate a response for
- `--qa-file`: Path to the JSON file containing question-answer pairs

#### Optional Arguments for Generate

- `--method`: Generation method to use (default: "style_vectors")
  - `style_vectors`: Uses style vector manipulation
  - `few_shot`: Uses few-shot learning with transformers
  - `ollama_few_shot`: Uses few-shot learning with Ollama
- `--neutral-file`: Path to neutral JSON file (required for style_vectors method)
- `--model-name`: Transformers model name (default: "google/gemma-3-4b-it")
- `--ollama-model`: Ollama model name (default: "google/gemma-3-4b-it", used with ollama_few_shot method)
- `--layer-index`: Layer index for style vector (default: 20)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.7)
- `--scale`: Style vector scaling factor (default: 1.0)

### List Models Command

List all locally installed Ollama models:

```bash
python cli.py list-models
```

This command will display:

- Model name
- Model size
- Model family
- Parameter size
- Quantization level

### Pull Model Command

Pull a new model from the Ollama library:

```bash
python cli.py pull-model <model_name> [--no-stream]
```

Arguments:

- `model_name`: Name of the model to pull (e.g., "llama2", "mistral", etc.)
- `--no-stream`: Optional flag to disable progress streaming

### Examples

1. Generate text using Style Vectors:

```bash
python cli.py generate --question "What is the nature of the universe?" --qa-file examples.json --method style_vectors --neutral-file neutral.json
```

2. Generate text using Few-Shot Learning with Transformers:

```bash
python cli.py generate --question "What is the nature of the universe?" --qa-file examples.json --method few_shot
```

3. Generate text using Few-Shot Learning with Ollama:

```bash
python cli.py generate --question "What is the nature of the universe?" --qa-file examples.json --method ollama_few_shot --ollama-model llama2
```

4. List installed Ollama models:

```bash
python cli.py list-models
```

5. Pull a new Ollama model:

```bash
python cli.py pull-model llama2
```

### Input File Format

#### QA File Format (JSON)

```json
[
    {
        "question": "What is the nature of the universe?",
        "response": "The universe is a vast, mysterious entity..."
    },
    ...
]
```

#### Neutral File Format (JSON, for style_vectors method)

```json
[
    {
        "question": "What is the nature of the universe?",
        "response": "The universe is a large space containing all matter and energy..."
    },
    ...
]
```

## Notes

- The style_vectors method requires both a QA file and a neutral file
- The few_shot methods only require a QA file
- For Ollama-based generation, make sure Ollama is running locally and the specified model is downloaded
- Temperature values closer to 0.0 will produce more focused and deterministic responses
- Temperature values closer to 1.0 will produce more creative and varied responses
- The scale parameter controls the intensity of the style vector influence (only applicable to style_vectors method)
- When pulling models, the download progress will be displayed in real-time unless --no-stream is specified
