"""
Knowledge base retrieval module for querying ChromaDB collections.

This module handles querying ChromaDB collections using OpenAI embeddings
and formatting results for context retrieval.
"""

import json
import os
from typing import Dict, List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize OpenAI embedding function
openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)


def query_collection(
    collection_name: str,
    query: str,
    n_results: int = 2,
    include_metadata: bool = True,
    verbose: bool = False
) -> str:
    """
    Query a ChromaDB collection and return formatted context.

    Args:
        collection_name: Name of the ChromaDB collection to query
        query: Query string to search for
        n_results: Number of results to return (default: 2)
        include_metadata: Whether to include metadata in results (default: True)
        verbose: Whether to print debug information (default: False)

    Returns:
        Formatted context string containing query results

    Raises:
        Exception: If collection doesn't exist or query fails
    """
    try:
        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name)

        # Generate query embedding
        query_embedding = openai_embedding_function([query])

        if verbose:
            print(f"Query: {query}")
            print(f"Query embedding vector length: {len(query_embedding[0])}")

        # Perform similarity search
        include_fields = ['documents']
        if include_metadata:
            include_fields.append('metadatas')

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=include_fields
        )

        # Format results
        return _format_query_results(results, include_metadata, verbose)

    except Exception as e:
        error_msg = f"Error querying collection '{collection_name}': {e}"
        print(error_msg)
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
    Get information about a specific collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Dictionary containing collection information or None if not found
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    except Exception as e:
        print(f"Error getting collection info for '{collection_name}': {e}")
        return None


def search_with_filters(
    collection_name: str,
    query: str,
    where_filter: Optional[Dict] = None,
    n_results: int = 2
) -> str:
    """
    Query collection with metadata filters.

    Args:
        collection_name: Name of the ChromaDB collection
        query: Query string to search for
        where_filter: Metadata filter conditions
        n_results: Number of results to return

    Returns:
        Formatted context string containing filtered results
    """
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name)
        query_embedding = openai_embedding_function([query])

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas']
        )

        return _format_query_results(results, include_metadata=True)

    except Exception as e:
        error_msg = f"Error in filtered search: {e}"
        print(error_msg)
        return f"Filtered search failed: {error_msg}"
