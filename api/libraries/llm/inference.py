"""
LLM inference module supporting multiple model types with hardware optimization.

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

# Import device optimization utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from device_optimization import get_device_info, get_optimized_config, print_device_info, DeviceType
    DEVICE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Device optimization not available. Using default configurations.")
    DEVICE_OPTIMIZATION_AVAILABLE = False

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

# Hardware detection with device optimization
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"
CPU_COUNT = multiprocessing.cpu_count()
OPTIMAL_THREADS = min(
    CPU_COUNT, 8) if not IS_APPLE_SILICON else min(CPU_COUNT, 6)

# Global device info cache
_device_type = None
_device_info = None


def _get_device_optimization():
    """Get cached device optimization info."""
    global _device_type, _device_info
    if DEVICE_OPTIMIZATION_AVAILABLE and (_device_type is None or _device_info is None):
        _device_type, _device_info = get_device_info()
        print_device_info(_device_type, _device_info)
    return _device_type, _device_info


# Initialize clients for style retrieval
chroma_client = PersistentClient(path=CHROMA_DB_PATH)

# Import unified embedding models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from embedding_models import get_embedding_model, EmbeddingModelType, embedding_manager


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


def get_style_data(query: str, character_name: str, embedding_config: Dict[str, Any], num_examples: int = 3) -> List[str]:
    """
    Get style examples from the vector database most similar to the query.

    Args:
        query: The query question to find similar examples for
        character_name: Name of the character to get style examples for
        embedding_config: Configuration for embedding model
        num_examples: Number of examples to retrieve

    Returns:
        List of question-answer pair strings

    Raises:
        Exception: If style retrieval fails
    """
    try:
        # Create collection name
        collection_name = f"{character_name.lower().replace(' ', '')}-style"

        # Get collection
        collection = chroma_client.get_collection(collection_name)

        # Generate query embedding using configured model
        model_id = embedding_config.get("model_id", "default")
        model_type = embedding_config.get("model_type", "sentence_transformers")
        model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        
        # Get or create embedding model
        embedding_model = get_embedding_model(
            model_id=model_id,
            model_type=model_type,
            model_name=model_name,
            config=embedding_config.get("config", {})
        )
        
        query_embedding = [embedding_model.embed_text(query)]

        # Query the collection
        results = collection.query(
            query_embeddings=query_embedding,
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
    model_config: Dict[str, Any],
    character_name: Optional[str] = None
) -> str:
    """
    Generate styled text using either OpenAI-compatible API or local models.
    Now supports using cached models for better performance.

    Args:
        query: The question/prompt to generate a response for
        examples: List of style example strings
        knowledge: Additional knowledge/context
        model: Model name/path to use
        model_config: Configuration for the model
        character_name: Character name for cache lookup (optional)

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
        print(f"🔍 Knowledge context added to prompt ({len(knowledge)} characters)")
        print(f"🔍 Knowledge preview: {knowledge[:200]}...")
    else:
        print(f"🔍 No knowledge context provided (empty or whitespace)")

    # Add the query
    prompt_parts.append(f"User: {query}")
    prompt_parts.append("Assistant:")

    prompt = "\n".join(prompt_parts)
    print(f"🔍 Final prompt length: {len(prompt)} characters")
    print(f"🔍 Final prompt preview: {prompt[:500]}...")

    # Try to use cached model first if character_name is provided
    if character_name:
        cache_key = f"{character_name}_llm"
        cached_model = get_cached_model(cache_key)
        if cached_model:
            print(f"Using cached LLM model for {character_name}")
            return cached_model.generate(prompt, **model_config)

    # Determine model type and generate
    # First check if it's a local model based on file path/extension
    if model.endswith('.gguf') or os.path.exists(model) or os.path.exists(os.path.join(MODELS_DIR, model)):
        return _generate_with_local_model(prompt, model, model_config)
    elif 'api_key' in model_config:
        return _generate_with_api(prompt, model, model_config)
    elif '/' in model and not model.startswith('./') and not model.startswith('/'):
        # This looks like a HuggingFace repository name (e.g., "meta-llama/Llama-3.2-1B")
        print(f"Detected HuggingFace repository name: {model}")
        return _generate_with_huggingface(prompt, model, model_config)
    else:
        # Default to local model if no clear indication
        return _generate_with_local_model(prompt, model, model_config)


def generate_styled_text_with_cached_model(
    query: str,
    examples: List[str],
    knowledge: str,
    cache_key: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate styled text using a specific cached model.

    Args:
        query: The question/prompt to generate a response for
        examples: List of style example strings
        knowledge: Additional knowledge/context
        cache_key: Cache key for the model to use
        generation_config: Generation parameters (temperature, max_tokens, etc.)

    Returns:
        Generated response text

    Raises:
        ModelLoadError: If cached model not found
        Exception: If generation fails
    """
    # Get cached model
    model = get_cached_model(cache_key)
    if model is None:
        raise ModelLoadError(f"No cached model found with key: {cache_key}")

    # Build the prompt
    prompt_parts = []

    # Add system prompt if provided in generation config
    if generation_config and 'system_prompt' in generation_config:
        prompt_parts.append(generation_config['system_prompt'])

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
    prompt_parts.append("Assistant:")

    prompt = "\n".join(prompt_parts)

    # Generate using cached model
    generation_params = generation_config or {}
    return model.generate(prompt, **generation_params)


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
    """Generate text using GGUF model with hardware optimization."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python required for GGUF inference.\n"
            "Install with: pip install llama-cpp-python"
        )

    try:
        # Get device optimization info
        device_type, device_info = _get_device_optimization()

        # Configure GGUF model with device optimization
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

        # Apply device-specific optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                gguf_config.update({
                    "n_gpu_layers": device_info.get("llm_gpu_layers", -1),
                    "n_batch": device_info.get("llm_batch_size", 2048),
                    "n_ubatch": device_info.get("llm_batch_size", 2048) // 2,
                    "n_ctx": device_info.get("llm_context_length", 8192),
                    "n_threads": device_info.get("llm_threads", 8),
                    "flash_attn": device_info.get("llm_use_flash_attention", True),
                    "use_mlock": True,
                    "use_mmap": True,
                    "offload_kqv": True,
                })
                print(
                    f"Applied NVIDIA GPU optimizations for {device_info.get('device_name', 'GPU')}")

            elif device_type == DeviceType.APPLE_SILICON:
                batch_size = device_info.get("llm_batch_size", 1024)
                gguf_config.update({
                    "n_gpu_layers": 0,  # Use CPU for Apple Silicon
                    "n_batch": min(batch_size * 2, 2048),
                    "n_ubatch": min(batch_size, 1024),
                    "n_ctx": device_info.get("llm_context_length", 8192),
                    "n_threads": device_info.get("llm_threads", 6),
                    "use_mlock": False,  # Disable mlock on macOS
                    "use_mmap": True,
                    "flash_attn": False,  # Disable flash attention on Apple Silicon
                })
                print(
                    f"Applied Apple Silicon optimizations for {device_info.get('device_name', 'Apple Silicon')}")

            else:  # CPU or other
                gguf_config.update({
                    "n_gpu_layers": 0,
                    "n_batch": device_info.get("llm_batch_size", 512),
                    "n_ubatch": device_info.get("llm_batch_size", 512) // 2,
                    "n_ctx": device_info.get("llm_context_length", 2048),
                    "n_threads": device_info.get("llm_threads", 8),
                    "flash_attn": False,
                })
                print(f"Applied CPU optimizations")

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

        # Clean up response
        for stop_token in model_config.get('stop_tokens', []):
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0].strip()

        return generated_text

    except Exception as e:
        raise RuntimeError(f"GGUF generation failed: {str(e)}")


def _generate_with_huggingface(prompt: str, model_path: str, model_config: Dict[str, Any]) -> str:
    """Generate text using Hugging Face model with hardware optimization."""
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

        # Add debugging for model path
        print(f"🔍 Loading HuggingFace model from: {model_path}")
        print(f"🔍 Model path exists: {os.path.exists(model_path)}")
        print(f"🔍 Is directory: {os.path.isdir(model_path)}")
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            print(f"🔍 Files in directory: {files}")
            # Check for essential files
            essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            for file in essential_files:
                if file in files:
                    print(f"✓ Found {file}")
                else:
                    print(f"❌ Missing {file}")

        # Get device optimization info
        device_type, device_info = _get_device_optimization()

        # Determine device and dtype based on optimization
        device = "auto"
        torch_dtype = torch.float16

        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                device = device_info.get("torch_device", "cuda:0")
                torch_dtype = torch.float16 if device_info.get(
                    "mixed_precision", True) else torch.float32
                print(
                    f"Using NVIDIA GPU optimization: device={device}, dtype={torch_dtype}")
            elif device_type == DeviceType.APPLE_SILICON:
                device = device_info.get("torch_device", "mps") if device_info.get(
                    "torch_device") == "mps" else "cpu"
                torch_dtype = torch.float32  # MPS works better with float32
                print(
                    f"Using Apple Silicon optimization: device={device}, dtype={torch_dtype}")
            else:
                device = "cpu"
                torch_dtype = torch.float32
                print(
                    f"Using CPU optimization: device={device}, dtype={torch_dtype}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device,
            "low_cpu_mem_usage": True
        }

        # Add device-specific optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE and device_type == DeviceType.NVIDIA_GPU:
            if device_info.get("enable_memory_efficient_attention", True):
                model_kwargs["attn_implementation"] = "flash_attention_2"
            if not device_info.get("gradient_checkpointing", True):
                model_kwargs["use_cache"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs)

        # Apply model optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU and device_info.get("tts_compile", False):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("Applied torch.compile optimization to Hugging Face model")
                except Exception as e:
                    print(f"Warning: Could not apply torch.compile: {e}")

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Extract system prompt and user message for proper chat formatting
        system_prompt = ""
        user_message = ""
        
        if 'system_prompt' in model_config and model_config['system_prompt']:
            system_prompt = model_config['system_prompt']
            print(f"🔍 Found system prompt in config: {system_prompt}")
            
            # Remove system prompt from the beginning of the prompt to get the rest
            if prompt.startswith(system_prompt):
                remaining_prompt = prompt[len(system_prompt):].strip()
                print(f"🔍 Remaining prompt after removing system: {remaining_prompt[:200]}...")
            else:
                remaining_prompt = prompt
                print(f"🔍 System prompt not at beginning, using full prompt: {prompt[:200]}...")
        else:
            remaining_prompt = prompt
            print(f"🔍 No system prompt found, using full prompt: {prompt[:200]}...")
        
        # Extract the actual user question from the end of the prompt
        # The format is: [style examples] [knowledge context] User: <question> Assistant:
        lines = remaining_prompt.split('\n')
        user_line = None
        for i, line in enumerate(lines):
            if line.startswith('User: '):
                user_line = line
                user_message = line[6:]  # Remove "User: " prefix
                break
        
        if not user_message:
            # Fallback: use the remaining prompt as user message
            user_message = remaining_prompt
            print(f"🔍 Could not find 'User: ' line, using remaining prompt as user message")
        
        print(f"🔍 Extracted user message: {user_message}")
        
        # Build the complete user message including context
        # We need to include style examples and knowledge context in the user message
        # since the chat template only supports system + user format
        user_message_parts = []
        
        # Add style examples and knowledge context to user message
        context_parts = []
        lines = remaining_prompt.split('\n')
        
        # Find everything before "User: " line
        for line in lines:
            if line.startswith('User: '):
                break
            if line.strip() and not line.startswith('Assistant:'):
                context_parts.append(line)
        
        # Combine context with the actual user question
        if context_parts:
            full_user_message = '\n'.join(context_parts) + '\n\n' + user_message
        else:
            full_user_message = user_message
        
        print(f"🔍 Final system prompt: {system_prompt}")
        print(f"🔍 Final user message: {full_user_message[:200]}...")
        
        # Use the model's chat template if available
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": full_user_message})
            
            print(f"🔍 Messages for chat template: {[{'role': m['role'], 'content': m['content'][:100] + '...' if len(m['content']) > 100 else m['content']} for m in messages]}")
            
            try:
                # Apply the chat template
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print(f"🔧 Using chat template for model {model_path}")
                print(f"🔧 Formatted prompt: {formatted_prompt[:300]}...")
            except Exception as e:
                print(f"Warning: Could not apply chat template: {e}")
                formatted_prompt = prompt
        else:
            print(f"🔧 No chat template available, using custom Llama format")
            # Create a Llama-style chat format manually
            if system_prompt:
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{full_user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{full_user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            print(f"🔧 Custom formatted prompt: {formatted_prompt[:300]}...")

        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Handle pad_token_id for Llama models (often None)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode response
        generated_text = tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        ).strip()

        return generated_text

    except Exception as e:
        raise RuntimeError(f"Hugging Face generation failed: {str(e)}")


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
    """GGUF model implementation using llama-cpp-python with hardware optimization."""

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

        # Get model configuration with device optimization
        model_config = self._get_model_config(model_path)

        try:
            self.model = Llama(**model_config)
            print("GGUF model loaded successfully with optimized configuration")
            return self.model
        except Exception as e:
            raise ModelLoadError(f"Failed to load GGUF model: {e}")

    def _get_model_config(self, model_path: str) -> Dict[str, Any]:
        """Get optimized GGUF model configuration with device optimization."""
        # Get device optimization info
        device_type, device_info = _get_device_optimization()

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

        # Apply device-specific optimizations
        if DEVICE_OPTIMIZATION_AVAILABLE:
            if device_type == DeviceType.NVIDIA_GPU:
                base_config.update({
                    "n_gpu_layers": device_info.get("llm_gpu_layers", -1),
                    "n_batch": device_info.get("llm_batch_size", 2048),
                    "n_ubatch": device_info.get("llm_batch_size", 2048) // 2,
                    "n_ctx": device_info.get("llm_context_length", 8192),
                    "n_threads": device_info.get("llm_threads", 8),
                    "flash_attn": device_info.get("llm_use_flash_attention", True),
                    "use_mlock": True,
                    "use_mmap": True,
                    "offload_kqv": True,
                })
                print(
                    f"Applied NVIDIA GPU optimizations for {device_info.get('device_name', 'GPU')}")

            elif device_type == DeviceType.APPLE_SILICON:
                batch_size = device_info.get("llm_batch_size", 1024)
                base_config.update({
                    "n_gpu_layers": 0,  # Use CPU for Apple Silicon GGUF
                    "n_batch": min(batch_size * 2, 2048),
                    "n_ubatch": min(batch_size, 1024),
                    "n_ctx": device_info.get("llm_context_length", 8192),
                    "n_threads": device_info.get("llm_threads", 6),
                    "use_mlock": False,  # Disable mlock on macOS
                    "use_mmap": True,
                    "flash_attn": False,  # Disable flash attention on Apple Silicon
                })
                print(
                    f"Applied Apple Silicon optimizations for {device_info.get('device_name', 'Apple Silicon')}")

            else:  # CPU or other
                base_config.update({
                    "n_gpu_layers": 0,
                    "n_batch": device_info.get("llm_batch_size", 512),
                    "n_ubatch": device_info.get("llm_batch_size", 512) // 2,
                    "n_ctx": device_info.get("llm_context_length", 2048),
                    "n_threads": device_info.get("llm_threads", 8),
                    "flash_attn": False,
                })
                print(f"Applied CPU optimizations")

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
    """Hugging Face transformers model implementation with hardware optimization."""

    def load(self) -> Any:
        """Load Hugging Face model with hardware optimization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for Hugging Face model support.\n"
                "Install them with: pip install transformers torch"
            )

        # Authenticate with Hugging Face before loading
        _authenticate_huggingface()

        model_name = self.config.get("model_path", "microsoft/DialoGPT-medium")
        print(f"Loading Hugging Face model: {model_name}")

        # Get device optimization info
        device_type, device_info = _get_device_optimization()

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

            # Determine device and dtype based on optimization
            device = "auto"
            torch_dtype = torch.float16

            if DEVICE_OPTIMIZATION_AVAILABLE:
                if device_type == DeviceType.NVIDIA_GPU:
                    device = device_info.get("torch_device", "cuda:0")
                    torch_dtype = torch.float16 if device_info.get(
                        "mixed_precision", True) else torch.float32
                    print(
                        f"Using NVIDIA GPU optimization: device={device}, dtype={torch_dtype}")
                elif device_type == DeviceType.APPLE_SILICON:
                    device = device_info.get("torch_device", "mps") if device_info.get(
                        "torch_device") == "mps" else "cpu"
                    torch_dtype = torch.float32  # MPS works better with float32
                    print(
                        f"Using Apple Silicon optimization: device={device}, dtype={torch_dtype}")
                else:
                    device = "cpu"
                    torch_dtype = torch.float32
                    print(
                        f"Using CPU optimization: device={device}, dtype={torch_dtype}")

            # Load model with optimizations
            model_kwargs = {
                "cache_dir": self.config.get("cache_dir", "./hf_cache"),
                "torch_dtype": torch_dtype,
                "device_map": device,
                "trust_remote_code": self.config.get("trust_remote_code", False),
                "low_cpu_mem_usage": True
            }

            # Add device-specific optimizations
            if DEVICE_OPTIMIZATION_AVAILABLE and device_type == DeviceType.NVIDIA_GPU:
                if device_info.get("enable_memory_efficient_attention", True):
                    try:
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                    except:
                        pass  # Flash attention not available for this model
                if not device_info.get("gradient_checkpointing", True):
                    model_kwargs["use_cache"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs)

            # Apply model optimizations
            if DEVICE_OPTIMIZATION_AVAILABLE and device_type == DeviceType.NVIDIA_GPU:
                if device_info.get("tts_compile", False):
                    try:
                        self.model = torch.compile(
                            self.model, mode="reduce-overhead")
                        print(
                            "Applied torch.compile optimization to Hugging Face model")
                    except Exception as e:
                        print(f"Warning: Could not apply torch.compile: {e}")

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

            # Generation parameters with device optimization
            max_new_tokens = kwargs.get("max_tokens", 200)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)
            do_sample = temperature > 0

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if do_sample else None,
                "top_p": top_p if do_sample else None,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
            }

            # Add device-specific generation optimizations
            if DEVICE_OPTIMIZATION_AVAILABLE and self.model.device.type == "cuda":
                if device_info.get("enable_memory_efficient_attention", True):
                    generation_kwargs["use_cache"] = True

            # Generate
            with torch.no_grad():
                if DEVICE_OPTIMIZATION_AVAILABLE and self.model.device.type == "cuda" and device_info.get("mixed_precision", True):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generation_kwargs)

            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            ).strip()

            return generated_text

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


# Global model cache - support multiple models loaded simultaneously
_current_model: Optional[BaseModel] = None  # Keep for backward compatibility
_model_cache: Dict[str, BaseModel] = {}  # New cache for multiple models


def create_model(model_type: ModelType, model_config: Dict[str, Any]) -> BaseModel:
    """
    Create a model instance of the specified type.

    Args:
        model_type: Type of model to create
        model_config: Configuration for the model

    Returns:
        Model instance (not yet loaded)

    Raises:
        ValueError: If model type is not supported
    """
    # Add type to config for reference
    model_config["type"] = model_type

    if model_type == ModelType.GGUF:
        return GGUFModel(model_config)
    elif model_type == ModelType.HUGGINGFACE:
        return HuggingFaceModel(model_config)
    elif model_type == ModelType.OPENAI_API:
        return OpenAIAPIModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _generate_cache_key(model_type: ModelType, model_config: Dict[str, Any]) -> str:
    """Generate a unique cache key for a model configuration."""
    if model_type == ModelType.GGUF:
        return f"gguf:{model_config.get('model_path', '')}"
    elif model_type == ModelType.HUGGINGFACE:
        return f"hf:{model_config.get('model_name', '')}"
    elif model_type == ModelType.OPENAI_API:
        return f"openai:{model_config.get('model_name', '')}"
    else:
        return f"{model_type.value}:{str(model_config)}"


def preload_llm_model(
    model_type: Union[str, ModelType],
    model_config: Dict[str, Any],
    cache_key: Optional[str] = None,
    force_reload: bool = False
) -> BaseModel:
    """
    Preload an LLM model into memory cache for faster inference.

    Args:
        model_type: Type of model to load
        model_config: Configuration for the model
        cache_key: Custom cache key (auto-generated if None)
        force_reload: Whether to force reload if model already cached

    Returns:
        Loaded model instance

    Raises:
        ModelLoadError: If model loading fails
    """
    global _model_cache

    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())

    # Generate cache key if not provided
    if cache_key is None:
        cache_key = _generate_cache_key(model_type, model_config)

    # Return cached model if exists and not forcing reload
    if not force_reload and cache_key in _model_cache:
        print(f"Using cached LLM model: {cache_key}")
        return _model_cache[cache_key]

    # Unload existing model with same cache key
    if cache_key in _model_cache:
        print(f"Unloading existing LLM model: {cache_key}")
        _model_cache[cache_key].unload()
        del _model_cache[cache_key]

    # Create and load new model
    print(f"Loading LLM model into cache: {cache_key}")
    model = create_model(model_type, model_config)
    model.load()

    # Cache the model
    _model_cache[cache_key] = model
    print(f"✓ LLM model cached successfully: {cache_key}")

    return model


def get_cached_model(cache_key: str) -> Optional[BaseModel]:
    """
    Get a cached model by its cache key.

    Args:
        cache_key: Cache key for the model

    Returns:
        Cached model instance or None if not found
    """
    return _model_cache.get(cache_key)


def unload_cached_model(cache_key: str) -> bool:
    """
    Unload a specific cached model.

    Args:
        cache_key: Cache key for the model to unload

    Returns:
        True if model was unloaded, False if not found
    """
    global _model_cache

    if cache_key in _model_cache:
        print(f"Unloading cached LLM model: {cache_key}")
        _model_cache[cache_key].unload()
        del _model_cache[cache_key]
        return True

    return False


def unload_all_cached_models() -> None:
    """Unload all cached models and free memory."""
    global _model_cache

    print("Unloading all cached LLM models...")

    for cache_key, model in _model_cache.items():
        print(f"Unloading: {cache_key}")
        model.unload()

    _model_cache.clear()

    # Also unload the legacy current model
    global _current_model
    if _current_model is not None:
        _current_model.unload()
        _current_model = None

    print("All LLM models unloaded from cache")


def get_cached_models_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all cached models.

    Returns:
        Dictionary with cache keys and model information
    """
    info = {}

    for cache_key, model in _model_cache.items():
        info[cache_key] = {
            "model_type": model.model_type.value,
            "is_loaded": model.is_loaded(),
            "config": model.config
        }

    return info


def load_model(
    model_type: Union[str, ModelType],
    model_config: Optional[Dict[str, Any]] = None,
    force_reload: bool = False
) -> BaseModel:
    """
    Load a model of specified type (legacy function for backward compatibility).

    This function maintains the old behavior of having a single "current" model
    while also supporting the new caching system.

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
    Get information about the currently loaded model and hardware optimization.

    Returns:
        Dictionary containing model information
    """
    # Get device optimization info
    device_type, device_info = _get_device_optimization()

    info = {
        "model_loaded": _current_model is not None,
        "model_type": _current_model.model_type.value if _current_model else None,
        "system_info": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu_count": CPU_COUNT,
            "optimal_threads": OPTIMAL_THREADS,
            "is_apple_silicon": IS_APPLE_SILICON
        },
        "device_optimization": {
            "available": DEVICE_OPTIMIZATION_AVAILABLE,
            "device_type": device_type.value if DEVICE_OPTIMIZATION_AVAILABLE else None,
            "device_info": device_info if DEVICE_OPTIMIZATION_AVAILABLE else None
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
