"""
LLM inference module supporting multiple model types.

This module handles loading and text generation for:
- GGUF models via llama-cpp-python
- Hugging Face transformers models  
- OpenAI-compatible API providers
"""

import multiprocessing
import os
import platform
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_GGUF_MODEL_PATH = os.getenv(
    "GGUF_MODEL_PATH", "./models/gemma-3-4b-it-q4_0.gguf")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-3.5-turbo")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hardware detection
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"
CPU_COUNT = multiprocessing.cpu_count()
OPTIMAL_THREADS = min(
    CPU_COUNT, 8) if not IS_APPLE_SILICON else min(CPU_COUNT, 6)

# Initialize clients for style retrieval
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
chroma_client = PersistentClient(path=CHROMA_DB_PATH)


def _authenticate_huggingface():
    """
    Authenticate with Hugging Face using API token if available.

    Returns:
        bool: True if authentication successful or not needed, False if failed
    """
    if not HUGGINGFACE_API_KEY:
        return True

    try:
        from huggingface_hub import login
        login(token=HUGGINGFACE_API_KEY, add_to_git_credential=True)
        return True
    except ImportError:
        print("Warning: huggingface_hub not available for authentication")
        return False
    except Exception as e:
        print(f"Warning: Failed to authenticate with Hugging Face: {e}")
        return False


class ModelType(Enum):
    """Supported model types."""
    GGUF = "gguf"
    HUGGINGFACE = "huggingface"
    OPENAI_API = "openai_api"


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


def get_style_data(query: str, character_name: str, num_examples: int = 3) -> List[str]:
    """
    Get style examples from the vector database most similar to the query.

    Args:
        query: The query question to find similar examples for
        character_name: Name of the character to get style examples for
        num_examples: Number of examples to retrieve

    Returns:
        List of question-answer pair strings

    Raises:
        Exception: If style retrieval fails
    """
    if not openai_client:
        raise Exception("OpenAI API key required for style data retrieval")

    try:
        # Create collection name
        collection_name = f"{character_name.lower().replace(' ', '')}-style"

        # Get collection
        collection = chroma_client.get_collection(collection_name)

        # Generate query embedding
        response = openai_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = response.data[0].embedding

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_examples,
            include=['documents', 'metadatas']
        )

        # Format results as strings
        style_examples = []
        if results['ids'] and results['ids'][0]:
            for i, doc in enumerate(results['documents'][0]):
                style_examples.append(doc)

        return style_examples

    except Exception as e:
        raise Exception(
            f"Failed to retrieve style data for {character_name}: {e}")


def generate_styled_text(
    query: str,
    examples: List[str],
    knowledge: str,
    model: str,
    model_config: Dict[str, Any]
) -> str:
    """
    Generate styled text using either OpenAI-compatible API or local models.

    Args:
        query: The question/prompt to generate a response for
        examples: List of style example strings
        knowledge: Additional knowledge/context
        model: Model name/path to use
        model_config: Configuration for the model

    Returns:
        Generated response text

    Raises:
        Exception: If generation fails
        ValueError: If model configuration is invalid
    """
    # Build the prompt
    prompt_parts = []

    # Add system prompt if provided
    if 'system_prompt' in model_config:
        prompt_parts.append(model_config['system_prompt'])

    # Add style examples
    if examples:
        prompt_parts.append("Here are some examples of the desired style:")
        for example in examples:
            prompt_parts.append(example)
        prompt_parts.append("")  # Empty line

    # Add knowledge context
    if knowledge.strip():
        prompt_parts.append("Additional context:")
        prompt_parts.append(knowledge)
        prompt_parts.append("")  # Empty line

    # Add the query
    prompt_parts.append(f"User: {query}")
    prompt_parts.append("Assistant: ")

    prompt = "\n".join(prompt_parts)

    # Determine model type based on configuration
    if 'api_key' in model_config and 'base_url' in model_config:
        # OpenAI-compatible API
        return _generate_with_api(prompt, model, model_config)
    else:
        # Local model
        return _generate_with_local_model(prompt, model, model_config)


def _generate_with_api(prompt: str, model: str, model_config: Dict[str, Any]) -> str:
    """Generate text using OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for API inference")

    api_key = model_config.get('api_key')
    base_url = model_config.get('base_url', 'https://api.openai.com/v1')

    if not api_key:
        raise ValueError("api_key required in model_config for API inference")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Convert prompt to messages format
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=model_config.get('max_tokens', 200),
            temperature=model_config.get('temperature', 0.7),
            top_p=model_config.get('top_p', 0.9)
        )

        generated_text = response.choices[0].message.content

        # Clean up stop tokens
        stop_tokens = model_config.get('stop_tokens', [
            "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
            "User:", "Assistant:", "System:"
        ])

        for token in stop_tokens:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]

        return generated_text.strip()

    except Exception as e:
        raise Exception(f"API generation failed: {e}")


def _generate_with_local_model(prompt: str, model: str, model_config: Dict[str, Any]) -> str:
    """Generate text using local model (GGUF or Hugging Face)."""
    model_path = model

    # Check if model exists in models directory
    if not os.path.isabs(model_path):
        # Relative path, check in models directory
        models_dir_path = os.path.join(MODELS_DIR, model)
        if os.path.exists(models_dir_path):
            model_path = models_dir_path
        elif os.path.exists(model):
            model_path = model
        else:
            raise FileNotFoundError(
                f"Model not found: {model}. Please download it first.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")

    # Determine model type
    if model_path.endswith('.gguf'):
        return _generate_with_gguf(prompt, model_path, model_config)
    elif os.path.isdir(model_path):
        # Check if it's a Hugging Face model
        if any(f.endswith('.json') for f in os.listdir(model_path)):
            return _generate_with_huggingface(prompt, model_path, model_config)

    raise ValueError(f"Unsupported model format: {model_path}")


def _generate_with_gguf(prompt: str, model_path: str, model_config: Dict[str, Any]) -> str:
    """Generate text using GGUF model."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python required for GGUF inference.\n"
            "Install with: pip install llama-cpp-python"
        )

    try:
        # Configure GGUF model
        gguf_config = {
            "model_path": model_path,
            "n_ctx": model_config.get("context_length", 4096),
            "n_gpu_layers": model_config.get("gpu_layers", -1),
            "n_batch": model_config.get("batch_size", 1024),
            "n_ubatch": model_config.get("batch_size", 1024) // 2,
            "rope_frequency_base": 10000,
            "use_mlock": True,
            "use_mmap": True,
            "n_threads": OPTIMAL_THREADS,
            "n_threads_batch": OPTIMAL_THREADS,
            "verbose": False,
            "flash_attn": True,
            "offload_kqv": True,
        }

        # Apple Silicon optimizations
        if IS_APPLE_SILICON:
            batch_size = model_config.get("batch_size", 1024)
            gguf_config.update({
                "n_batch": min(batch_size * 2, 2048),
                "n_ubatch": min(batch_size, 1024),
                "use_mlock": False,
            })

        # Load model
        llama_model = Llama(**gguf_config)

        # Generate
        response = llama_model(
            prompt,
            max_tokens=model_config.get('max_tokens', 200),
            temperature=model_config.get('temperature', 0.7),
            top_p=model_config.get('top_p', 0.9),
            stop=model_config.get('stop_tokens', [
                "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
                "User:", "Assistant:", "System:"
            ]),
            echo=False
        )

        generated_text = response['choices'][0]['text'].strip()

        # Additional cleanup for stop tokens
        stop_tokens = model_config.get('stop_tokens', [
            "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
            "User:", "Assistant:", "System:"
        ])

        for token in stop_tokens:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]

        return generated_text.strip()

    except Exception as e:
        raise Exception(f"GGUF generation failed: {e}")


def _generate_with_huggingface(prompt: str, model_path: str, model_config: Dict[str, Any]) -> str:
    """Generate text using Hugging Face model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch required for Hugging Face inference.\n"
            "Install with: pip install transformers torch"
        )

    try:
        # Authenticate with Hugging Face before loading
        _authenticate_huggingface()

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, model_config.get(
                "torch_dtype", "float16")),
            device_map=model_config.get("device", "auto"),
            low_cpu_mem_usage=True
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generation parameters
        max_new_tokens = model_config.get("max_tokens", 200)
        temperature = model_config.get("temperature", 0.7)
        top_p = model_config.get("top_p", 0.9)
        do_sample = temperature > 0

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=model_config.get("repetition_penalty", 1.1)
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up stop tokens
        stop_tokens = model_config.get('stop_tokens', [
            "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
            "User:", "Assistant:", "System:"
        ])

        for token in stop_tokens:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]

        return generated_text.strip()

    except Exception as e:
        raise Exception(f"Hugging Face generation failed: {e}")


class BaseModel(ABC):
    """Abstract base class for all model types."""

    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model_type = model_config.get("type")
        self.model = None

    @abstractmethod
    def load(self) -> Any:
        """Load the model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free memory."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


class GGUFModel(BaseModel):
    """GGUF model implementation using llama-cpp-python."""

    def load(self) -> Any:
        """Load GGUF model with optimized configuration."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF model support.\n"
                "Install it with: pip install llama-cpp-python\n"
                "For Apple Silicon Macs: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
            )

        model_path = self.config.get("model_path", DEFAULT_GGUF_MODEL_PATH)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model file not found at {model_path}\n"
                "Please download it first."
            )

        print(f"Loading GGUF model: {model_path}")
        print(f"Detected system: {platform.system()} {platform.machine()}")
        print(f"Using {OPTIMAL_THREADS} threads for optimal performance")

        # Get model configuration
        model_config = self._get_model_config(model_path)

        try:
            self.model = Llama(**model_config)
            print("GGUF model loaded successfully with optimized configuration")
            return self.model
        except Exception as e:
            raise ModelLoadError(f"Failed to load GGUF model: {e}")

    def _get_model_config(self, model_path: str) -> Dict[str, Any]:
        """Get optimized GGUF model configuration."""
        base_config = {
            "model_path": model_path,
            "n_ctx": self.config.get("context_length", 4096),
            "n_gpu_layers": self.config.get("gpu_layers", -1),
            "n_batch": self.config.get("batch_size", 1024),
            "n_ubatch": self.config.get("batch_size", 1024) // 2,
            "rope_frequency_base": 10000,
            "use_mlock": True,
            "use_mmap": True,
            "n_threads": OPTIMAL_THREADS,
            "n_threads_batch": OPTIMAL_THREADS,
            "verbose": False,
            "flash_attn": True,
            "offload_kqv": True,
        }

        # Apple Silicon specific optimizations
        if IS_APPLE_SILICON:
            batch_size = self.config.get("batch_size", 1024)
            base_config.update({
                "n_batch": min(batch_size * 2, 2048),
                "n_ubatch": min(batch_size, 1024),
                "use_mlock": False,
            })

        return base_config

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using GGUF model."""
        if not self.is_loaded():
            raise ModelLoadError("Model not loaded. Call load() first.")

        max_tokens = kwargs.get("max_tokens", 200)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        stop_tokens = kwargs.get("stop_tokens", [
            "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
            "User:", "Assistant:", "System:"
        ])

        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_tokens,
                echo=False
            )

            generated_text = response['choices'][0]['text'].strip()

            # Additional cleanup for stop tokens
            for token in stop_tokens:
                if token in generated_text:
                    generated_text = generated_text.split(token)[0]

            return generated_text.strip()

        except Exception as e:
            raise Exception(f"GGUF text generation failed: {e}")

    def unload(self) -> None:
        """Unload GGUF model."""
        if self.model is not None:
            del self.model
            self.model = None


class HuggingFaceModel(BaseModel):
    """Hugging Face transformers model implementation."""

    def load(self) -> Any:
        """Load Hugging Face model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for Hugging Face model support.\n"
                "Install with: pip install transformers torch"
            )

        model_name = self.config.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for Hugging Face models")

        print(f"Loading Hugging Face model: {model_name}")

        # Authenticate with Hugging Face before loading
        _authenticate_huggingface()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.get("cache_dir", "./hf_cache"),
                trust_remote_code=self.config.get("trust_remote_code", False)
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            device = self.config.get("device", "auto")
            torch_dtype = getattr(
                torch, self.config.get("torch_dtype", "float16"))

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.config.get("cache_dir", "./hf_cache"),
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=self.config.get("trust_remote_code", False),
                low_cpu_mem_usage=True
            )

            print(
                f"Hugging Face model loaded successfully on device: {self.model.device}")
            return self.model

        except Exception as e:
            raise ModelLoadError(f"Failed to load Hugging Face model: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face model."""
        if not self.is_loaded():
            raise ModelLoadError("Model not loaded. Call load() first.")

        try:
            import torch

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generation parameters
            max_new_tokens = kwargs.get("max_tokens", 200)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)
            do_sample = temperature > 0

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=kwargs.get("repetition_penalty", 1.1)
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True)

            # Clean up stop tokens
            stop_tokens = kwargs.get("stop_tokens", [
                "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
                "User:", "Assistant:", "System:"
            ])

            for token in stop_tokens:
                if token in generated_text:
                    generated_text = generated_text.split(token)[0]

            return generated_text.strip()

        except Exception as e:
            raise Exception(f"Hugging Face text generation failed: {e}")

    def unload(self) -> None:
        """Unload Hugging Face model."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer


class OpenAIAPIModel(BaseModel):
    """OpenAI-compatible API model implementation."""

    def load(self) -> Any:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for API model support.\n"
                "Install with: pip install openai"
            )

        api_key = self.config.get("api_key", OPENAI_API_KEY)
        if not api_key:
            raise ValueError(
                "API key is required for OpenAI-compatible models")

        base_url = self.config.get("base_url", OPENAI_BASE_URL)

        try:
            self.model = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

            print(f"OpenAI-compatible API client initialized: {base_url}")
            return self.model

        except Exception as e:
            raise ModelLoadError(f"Failed to initialize OpenAI client: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI-compatible API."""
        if not self.is_loaded():
            raise ModelLoadError("Model not loaded. Call load() first.")

        try:
            model_name = self.config.get("model_name", DEFAULT_OPENAI_MODEL)
            max_tokens = kwargs.get("max_tokens", 200)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)

            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]

            response = self.model.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            generated_text = response.choices[0].message.content

            # Clean up stop tokens
            stop_tokens = kwargs.get("stop_tokens", [
                "\nUser:", "\nAssistant:", "<STOP>", "\nSystem:",
                "User:", "Assistant:", "System:"
            ])

            for token in stop_tokens:
                if token in generated_text:
                    generated_text = generated_text.split(token)[0]

            return generated_text.strip()

        except Exception as e:
            raise Exception(f"OpenAI API text generation failed: {e}")

    def unload(self) -> None:
        """Unload API client."""
        self.model = None


# Global model instance
_current_model: Optional[BaseModel] = None


def create_model(model_type: ModelType, model_config: Dict[str, Any]) -> BaseModel:
    """
    Create a model instance based on type.

    Args:
        model_type: Type of model to create
        model_config: Configuration for the model

    Returns:
        Model instance

    Raises:
        ValueError: If model type is not supported
    """
    model_config["type"] = model_type

    if model_type == ModelType.GGUF:
        return GGUFModel(model_config)
    elif model_type == ModelType.HUGGINGFACE:
        return HuggingFaceModel(model_config)
    elif model_type == ModelType.OPENAI_API:
        return OpenAIAPIModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_model(
    model_type: Union[str, ModelType],
    model_config: Optional[Dict[str, Any]] = None,
    force_reload: bool = False
) -> BaseModel:
    """
    Load a model of specified type.

    Args:
        model_type: Type of model to load
        model_config: Configuration for the model
        force_reload: Whether to force reload if model already loaded

    Returns:
        Loaded model instance

    Raises:
        ModelLoadError: If model loading fails
    """
    global _current_model

    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())

    if model_config is None:
        model_config = {}

    # Return existing model if same type and not forcing reload
    if (not force_reload and _current_model is not None and
            _current_model.model_type == model_type):
        return _current_model

    # Unload existing model
    if _current_model is not None:
        _current_model.unload()

    # Create and load new model
    _current_model = create_model(model_type, model_config)
    _current_model.load()

    return _current_model


def generate_text(
    prompt: str,
    model: Optional[BaseModel] = None,
    **kwargs
) -> str:
    """
    Generate text using the specified or current model.

    Args:
        prompt: Input prompt for text generation
        model: Model instance (uses global if None)
        **kwargs: Generation parameters

    Returns:
        Generated text string

    Raises:
        ModelLoadError: If no model is loaded
    """
    if model is None:
        if _current_model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        model = _current_model

    return model.generate(prompt, **kwargs)


def get_current_model() -> Optional[BaseModel]:
    """Get the currently loaded model."""
    return _current_model


def unload_current_model() -> None:
    """Unload the current model and free memory."""
    global _current_model

    if _current_model is not None:
        _current_model.unload()
        _current_model = None
        print("Model unloaded successfully")


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.

    Returns:
        Dictionary containing model information
    """
    info = {
        "model_loaded": _current_model is not None,
        "model_type": _current_model.model_type.value if _current_model else None,
        "system_info": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu_count": CPU_COUNT,
            "optimal_threads": OPTIMAL_THREADS,
            "is_apple_silicon": IS_APPLE_SILICON
        }
    }

    if _current_model:
        info["model_config"] = _current_model.config

    return info


# Backward compatibility functions
def load_gguf_model(model_path: Optional[str] = None, force_reload: bool = False, **kwargs) -> BaseModel:
    """Load GGUF model (backward compatibility)."""
    config = {"model_path": model_path or DEFAULT_GGUF_MODEL_PATH}
    config.update(kwargs)
    return load_model(ModelType.GGUF, config, force_reload)


def load_huggingface_model(model_name: str, **kwargs) -> BaseModel:
    """Load Hugging Face model."""
    config = {"model_name": model_name}
    config.update(kwargs)
    return load_model(ModelType.HUGGINGFACE, config)


def load_openai_model(model_name: Optional[str] = None, api_key: Optional[str] = None,
                      base_url: Optional[str] = None, **kwargs) -> BaseModel:
    """Load OpenAI-compatible API model."""
    config = {
        "model_name": model_name or DEFAULT_OPENAI_MODEL,
        "api_key": api_key or OPENAI_API_KEY,
        "base_url": base_url or OPENAI_BASE_URL
    }
    config.update(kwargs)
    return load_model(ModelType.OPENAI_API, config)


def list_available_models(models_dir: str = None) -> Dict[str, List[str]]:
    """
    List available models by type.

    Args:
        models_dir: Directory to search for local models

    Returns:
        Dictionary with model types as keys and lists of available models
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    available = {
        "gguf": [],
        "huggingface": [],
        "openai_api": []
    }

    # Find GGUF models
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.lower().endswith('.gguf'):
                available["gguf"].append(os.path.join(models_dir, file))

    # Popular HuggingFace models (could be extended or made configurable)
    available["huggingface"] = [
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill",
        "microsoft/DialoGPT-small",
        "google/flan-t5-base",
        "google/flan-t5-large"
    ]

    # OpenAI-compatible models
    available["openai_api"] = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo-preview"
    ]

    return available
