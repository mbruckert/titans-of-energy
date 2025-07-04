"""
Unified embedding model management for both OpenAI and open source models.

This module provides a consistent interface for different embedding models
including OpenAI's text-embedding-ada-002 and various sentence-transformers models.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class EmbeddingModelType(Enum):
    """Supported embedding model types."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the embedding model."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free memory."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key") if config else None
        self.client = None
    
    def load(self) -> None:
        """Load OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI embeddings")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "loaded"  # Indicate model is ready
        print(f"✓ OpenAI embedding model loaded: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}")
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"  📊 Processing OpenAI embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI batch embedding failed: {e}")
    
    def unload(self) -> None:
        """Unload OpenAI client."""
        self.client = None
        self.model = None

class SentenceTransformersEmbeddingModel(EmbeddingModel):
    """Sentence Transformers embedding model implementation."""
    
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "description": "Lightweight and fast (80MB)",
            "dimensions": 384,
            "performance": "Good for most tasks"
        },
        "all-mpnet-base-v2": {
            "description": "Higher quality but slower (420MB)",
            "dimensions": 768,
            "performance": "Best overall quality"
        },
        "BAAI/bge-small-en-v1.5": {
            "description": "Strong retrieval and fast (130MB)",
            "dimensions": 384,
            "performance": "Optimized for retrieval"
        },
        "BAAI/bge-base-en-v1.5": {
            "description": "High accuracy but slower (440MB)",
            "dimensions": 768,
            "performance": "Best accuracy, slower"
        },
        "BAAI/bge-large-en-v1.5": {
            "description": "Highest quality, very slow (1.3GB)",
            "dimensions": 1024,
            "performance": "Best quality, resource intensive"
        }
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.device = config.get("device", "auto") if config else "auto"
    
    def load(self) -> None:
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package required for open source embeddings")
        
        print(f"Loading sentence-transformers model: {self.model_name}")
        
        # Handle device selection
        device = self.device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"✓ Sentence-transformers model loaded: {self.model_name} on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformers model: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using sentence-transformers."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            raise RuntimeError(f"Sentence-transformers embedding failed: {e}")
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts using sentence-transformers."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"  📊 Processing sentence-transformers batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
                
                batch_embeddings = self.model.encode(batch).tolist()
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            raise RuntimeError(f"Sentence-transformers batch embedding failed: {e}")
    
    def unload(self) -> None:
        """Unload sentence-transformers model."""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Clear GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

class EmbeddingModelManager:
    """Manager for different embedding models."""
    
    def __init__(self):
        self.models: Dict[str, EmbeddingModel] = {}
    
    def create_model(self, model_type: Union[str, EmbeddingModelType], 
                    model_name: str, config: Optional[Dict[str, Any]] = None) -> EmbeddingModel:
        """Create an embedding model instance."""
        if isinstance(model_type, str):
            model_type = EmbeddingModelType(model_type.lower())
        
        if model_type == EmbeddingModelType.OPENAI:
            return OpenAIEmbeddingModel(model_name, config)
        elif model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            return SentenceTransformersEmbeddingModel(model_name, config)
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
    
    def load_model(self, model_id: str, model_type: Union[str, EmbeddingModelType], 
                  model_name: str, config: Optional[Dict[str, Any]] = None) -> EmbeddingModel:
        """Load and cache an embedding model."""
        if model_id in self.models:
            return self.models[model_id]
        
        model = self.create_model(model_type, model_name, config)
        model.load()
        self.models[model_id] = model
        return model
    
    def get_model(self, model_id: str) -> Optional[EmbeddingModel]:
        """Get a cached embedding model."""
        return self.models.get(model_id)
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model."""
        if model_id in self.models:
            self.models[model_id].unload()
            del self.models[model_id]
            return True
        return False
    
    def unload_all_models(self) -> None:
        """Unload all cached models."""
        for model_id, model in self.models.items():
            model.unload()
        self.models.clear()
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all cached models."""
        return {
            model_id: {
                "model_name": model.model_name,
                "is_loaded": model.is_loaded(),
                "config": model.config
            }
            for model_id, model in self.models.items()
        }
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        return {
            "openai": {
                "text-embedding-ada-002": {
                    "description": "OpenAI's standard embedding model",
                    "dimensions": 1536,
                    "performance": "High quality, requires API key"
                },
                "text-embedding-3-small": {
                    "description": "OpenAI's newer small embedding model",
                    "dimensions": 1536,
                    "performance": "Good quality, faster, requires API key"
                },
                "text-embedding-3-large": {
                    "description": "OpenAI's newer large embedding model",
                    "dimensions": 3072,
                    "performance": "Best quality, requires API key"
                }
            },
            "sentence_transformers": SentenceTransformersEmbeddingModel.AVAILABLE_MODELS
        }

# Global embedding model manager instance
embedding_manager = EmbeddingModelManager()

def get_embedding_model(model_id: str, model_type: Union[str, EmbeddingModelType], 
                       model_name: str, config: Optional[Dict[str, Any]] = None) -> EmbeddingModel:
    """Get or create an embedding model."""
    return embedding_manager.load_model(model_id, model_type, model_name, config)

def embed_text(text: str, model_id: str) -> List[float]:
    """Generate embedding for a single text using specified model."""
    model = embedding_manager.get_model(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not found. Load it first.")
    return model.embed_text(text)

def embed_texts_batch(texts: List[str], model_id: str) -> List[List[float]]:
    """Generate embeddings for multiple texts using specified model."""
    model = embedding_manager.get_model(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not found. Load it first.")
    return model.embed_texts_batch(texts) 