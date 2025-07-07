"""
Knowledge base retrieval module for querying ChromaDB collections.

This module handles querying ChromaDB collections using configurable embedding models
and formatting results for context retrieval.
Supports both OpenAI and open source embedding models with automatic embedding
model compatibility checking.
"""

import json
import os
from typing import Dict, List, Optional, Union, Any

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

# Import unified embedding models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from embedding_models import get_embedding_model, EmbeddingModelType, embedding_manager

# Import preprocessing utilities for compatibility checking
from .preprocess import (
    get_embedding_model_signature, 
    get_collection_embedding_info, 
    is_embedding_model_compatible,
    backup_and_recreate_collection,
    list_collection_backups
)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_db"

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)


def query_collection(
    collection_name: str,
    query: str,
    embedding_config: Dict[str, Any],
    n_results: int = 2,
    include_metadata: bool = True,
    verbose: bool = False,
    return_structured: bool = False
) -> Union[str, Dict]:
    """
    Query a ChromaDB collection and return formatted context or structured data.
    Automatically handles embedding model compatibility issues.

    Args:
        collection_name: Name of the ChromaDB collection to query
        query: Query string to search for
        embedding_config: Configuration for embedding model
        n_results: Number of results to return (default: 2)
        include_metadata: Whether to include metadata in results (default: True)
        verbose: Whether to print debug information (default: False)
        return_structured: Whether to return structured data instead of formatted string (default: False)

    Returns:
        Formatted context string or structured dictionary containing query results

    Raises:
        Exception: If collection doesn't exist or query fails
    """
    try:
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            error_msg = f"Collection '{collection_name}' does not exist"
            print(error_msg)
            if return_structured:
                return {"error": error_msg, "references": []}
            else:
                return f"Query failed: {error_msg}"

        # Check embedding model compatibility
        if not is_embedding_model_compatible(collection_name, embedding_config):
            error_msg = f"Embedding model incompatible with collection '{collection_name}'"
            print(f"❌ {error_msg}")
            
            # Get collection info for better error message
            collection_info = get_collection_embedding_info(collection_name)
            if collection_info:
                current_signature = get_embedding_model_signature(embedding_config)
                stored_signature = collection_info.get("embedding_model_signature", "unknown")
                detailed_error = f"{error_msg}. Current model: {current_signature}, Collection model: {stored_signature}"
            else:
                detailed_error = f"{error_msg}. Collection has no embedding model metadata."
            
            print(f"💡 Suggestion: Recreate the knowledge base with the new embedding model")
            
            if return_structured:
                return {
                    "error": detailed_error,
                    "references": [],
                    "suggestion": "Recreate the knowledge base with the new embedding model",
                    "collection_info": collection_info
                }
            else:
                return f"Query failed: {detailed_error}"

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

        if verbose:
            print(f"Query: {query}")
            print(f"Query embedding vector length: {len(query_embedding[0])}")
            
            # Print collection info
            collection_info = get_collection_embedding_info(collection_name)
            if collection_info:
                print(f"Collection embedding model: {collection_info.get('embedding_model_type')}:{collection_info.get('embedding_model_name')}")
                print(f"Collection embedding dimensions: {collection_info.get('embedding_dimensions')}")

        # Perform similarity search
        include_fields = ['documents', 'distances']
        if include_metadata:
            include_fields.append('metadatas')

        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=include_fields
            )
        except Exception as query_error:
            # Handle dimension mismatch errors specifically
            if "dimension" in str(query_error).lower() or "shape" in str(query_error).lower():
                error_msg = f"Embedding dimension mismatch in collection '{collection_name}': {query_error}"
                print(f"❌ {error_msg}")
                print(f"💡 This usually means the collection was created with a different embedding model")
                
                collection_info = get_collection_embedding_info(collection_name)
                if collection_info:
                    print(f"   Collection model: {collection_info.get('embedding_model_type')}:{collection_info.get('embedding_model_name')} ({collection_info.get('embedding_dimensions')}D)")
                    current_signature = get_embedding_model_signature(embedding_config)
                    print(f"   Current model: {current_signature}")
                
                if return_structured:
                    return {
                        "error": error_msg,
                        "references": [],
                        "suggestion": "Recreate the knowledge base with the current embedding model",
                        "collection_info": collection_info
                    }
                else:
                    return f"Query failed: {error_msg}"
            else:
                raise query_error

        # Return structured data if requested
        if return_structured:
            return _format_structured_results(results, include_metadata, verbose)
        else:
            # Format results as string (backward compatibility)
            return _format_query_results(results, include_metadata, verbose)

    except Exception as e:
        error_msg = f"Error querying collection '{collection_name}': {e}"
        print(error_msg)
        if return_structured:
            return {"error": error_msg, "references": []}
        else:
            return f"Query failed: {error_msg}"


def _format_query_results(
    results: Dict,
    include_metadata: bool = True,
    verbose: bool = False
) -> str:
    """
    Format ChromaDB query results into a readable context string.

    Args:
        results: Raw results from ChromaDB query
        include_metadata: Whether to include metadata in formatting
        verbose: Whether to include debug information

    Returns:
        Formatted context string
    """
    if not results.get('documents') or not results['documents'][0]:
        return "No relevant context found."

    context_parts = []
    documents = results['documents'][0]
    metadatas = results.get('metadatas', [None])[
        0] if include_metadata else None

    if verbose:
        print(f"Found {len(documents)} relevant documents")

    for i, doc in enumerate(documents):
        # Add document content
        context_parts.append(f"Document {i+1}:")
        context_parts.append(doc)
        context_parts.append("")  # Empty line for readability

        # Add metadata if available and requested
        if include_metadata and metadatas and i < len(metadatas):
            metadata = metadatas[i]
            if metadata:
                context_parts.append("Metadata:")
                context_parts.append(json.dumps(metadata, indent=2))
                context_parts.append("")  # Empty line for readability

    return "\n".join(context_parts).strip()


def _format_structured_results(
    results: Dict,
    include_metadata: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Format ChromaDB query results into structured data for UI display.

    Args:
        results: Raw results from ChromaDB query
        include_metadata: Whether to include metadata in formatting
        verbose: Whether to include debug information

    Returns:
        Dictionary with formatted context and references
    """
    if not results.get('documents') or not results['documents'][0]:
        return {
            "context": "No relevant context found.",
            "references": []
        }

    documents = results['documents'][0]
    metadatas = results.get('metadatas', [None])[
        0] if include_metadata else None
    distances = results.get('distances', [None])[
        0] if 'distances' in results else None

    if verbose:
        print(f"Found {len(documents)} relevant documents")

    # Build context string
    context_parts = []
    references = []

    for i, doc in enumerate(documents):
        # Add to context
        context_parts.append(f"Source {i+1}: {doc}")

        # Build reference data
        ref_data = {
            "id": i + 1,
            "content": doc,
            "relevance_score": None
        }

        # Add metadata if available
        if metadatas and i < len(metadatas) and metadatas[i]:
            metadata = metadatas[i]
            ref_data.update({
                "source": metadata.get("source", "Unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "keywords": metadata.get("keywords", "").split(", ") if metadata.get("keywords") else [],
                "entities": metadata.get("entities", "").split(", ") if metadata.get("entities") else [],
                "type": metadata.get("type", "text")
            })

        # Add similarity score if available
        if distances and i < len(distances) and distances[i] is not None:
            # Convert distance to similarity percentage (lower distance = higher similarity)
            similarity = max(0, min(100, (1 - distances[i]) * 100))
            ref_data["relevance_score"] = round(similarity, 1)

        references.append(ref_data)

    return {
        "context": "\n\n".join(context_parts),
        "references": references,
        "total_sources": len(references)
    }


def list_collections() -> List[str]:
    """
    List all available collections in the ChromaDB instance.

    Returns:
        List of collection names
    """
    try:
        collections = chroma_client.list_collections()
        return [collection.name for collection in collections]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []


def get_collection_info(collection_name: str) -> Optional[Dict]:
    """
    Get information about a specific collection including embedding model details.

    Args:
        collection_name: Name of the collection

    Returns:
        Dictionary containing collection information or None if not found
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Get basic collection info
        basic_info = {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
        
        # Add embedding model info
        embedding_info = get_collection_embedding_info(collection_name)
        if embedding_info:
            basic_info.update(embedding_info)
        
        return basic_info
    except Exception as e:
        print(f"Error getting collection info for '{collection_name}': {e}")
        return None


def search_with_filters(
    collection_name: str,
    query: str,
    embedding_config: Dict[str, Any],
    where_filter: Optional[Dict] = None,
    n_results: int = 2
) -> str:
    """
    Query collection with metadata filters.
    Automatically handles embedding model compatibility issues.

    Args:
        collection_name: Name of the ChromaDB collection
        query: Query string to search for
        embedding_config: Configuration for embedding model
        where_filter: Metadata filter conditions
        n_results: Number of results to return

    Returns:
        Formatted context string containing filtered results
    """
    try:
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            error_msg = f"Collection '{collection_name}' does not exist"
            print(error_msg)
            return f"Filtered search failed: {error_msg}"

        # Check embedding model compatibility
        if not is_embedding_model_compatible(collection_name, embedding_config):
            error_msg = f"Embedding model incompatible with collection '{collection_name}'"
            print(f"❌ {error_msg}")
            return f"Filtered search failed: {error_msg}"
        
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

        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas']
            )
        except Exception as query_error:
            # Handle dimension mismatch errors specifically
            if "dimension" in str(query_error).lower() or "shape" in str(query_error).lower():
                error_msg = f"Embedding dimension mismatch in collection '{collection_name}': {query_error}"
                print(f"❌ {error_msg}")
                return f"Filtered search failed: {error_msg}"
            else:
                raise query_error

        return _format_query_results(results, include_metadata=True)

    except Exception as e:
        error_msg = f"Error in filtered search: {e}"
        print(error_msg)
        return f"Filtered search failed: {error_msg}"


def check_collection_compatibility(collection_name: str, embedding_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a collection is compatible with the given embedding configuration.
    
    Args:
        collection_name: Name of the ChromaDB collection
        embedding_config: Configuration for embedding model
    
    Returns:
        Dictionary with compatibility information
    """
    try:
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            return {
                "exists": False,
                "compatible": False,
                "error": f"Collection '{collection_name}' does not exist"
            }
        
        # Get collection embedding info
        collection_info = get_collection_embedding_info(collection_name)
        current_signature = get_embedding_model_signature(embedding_config)
        
        if not collection_info:
            return {
                "exists": True,
                "compatible": False,
                "error": "Collection has no embedding model metadata",
                "suggestion": "This appears to be a legacy collection. Consider recreating it with embedding model metadata.",
                "current_model": current_signature,
                "collection_model": "unknown"
            }
        
        stored_signature = collection_info.get("embedding_model_signature")
        is_compatible = current_signature == stored_signature
        
        result = {
            "exists": True,
            "compatible": is_compatible,
            "current_model": current_signature,
            "collection_model": stored_signature,
            "collection_info": collection_info,
            "count": collection.count()
        }
        
        if not is_compatible:
            result["error"] = "Embedding model mismatch"
            result["suggestion"] = "Recreate the knowledge base with the current embedding model"
        
        return result
        
    except Exception as e:
        return {
            "exists": False,
            "compatible": False,
            "error": f"Error checking collection compatibility: {e}"
        }


def get_collection_diagnostics(collection_name: str) -> Dict[str, Any]:
    """
    Get comprehensive diagnostic information about a collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
    
    Returns:
        Dictionary with diagnostic information
    """
    try:
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            return {
                "exists": False,
                "error": f"Collection '{collection_name}' does not exist"
            }
        
        # Get basic info
        diagnostics = {
            "exists": True,
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
        
        # Get embedding model info
        embedding_info = get_collection_embedding_info(collection_name)
        if embedding_info:
            diagnostics["embedding_info"] = embedding_info
        else:
            diagnostics["embedding_info"] = None
            diagnostics["warning"] = "Collection has no embedding model metadata (legacy collection)"
        
        # Get sample documents to check embedding dimensions
        if diagnostics["count"] > 0:
            try:
                sample = collection.get(limit=1, include=['embeddings'])
                if sample['embeddings'] and len(sample['embeddings']) > 0:
                    actual_dimensions = len(sample['embeddings'][0])
                    diagnostics["actual_embedding_dimensions"] = actual_dimensions
                    
                    # Compare with metadata
                    if embedding_info and embedding_info.get("embedding_dimensions"):
                        expected_dimensions = embedding_info["embedding_dimensions"]
                        if actual_dimensions != expected_dimensions:
                            diagnostics["dimension_mismatch"] = {
                                "expected": expected_dimensions,
                                "actual": actual_dimensions
                            }
                else:
                    diagnostics["warning"] = "Collection has no embeddings"
            except Exception as e:
                diagnostics["sample_error"] = f"Could not retrieve sample: {e}"
        
        # List available backups
        backups = list_collection_backups(collection_name)
        if backups:
            diagnostics["backups"] = backups
        
        return diagnostics
        
    except Exception as e:
        return {
            "exists": False,
            "error": f"Error getting collection diagnostics: {e}"
        }
