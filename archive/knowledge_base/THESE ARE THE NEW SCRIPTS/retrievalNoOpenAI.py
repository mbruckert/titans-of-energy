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

# Library for free embedding models!********
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_db"
""" 
Below change the embedding model from sentence transformers

Choices:
    all-MiniLM-L6-v2 (Lightweight and fast)
    all-mpnet-base-v2 (Higher quality but slower)
    BAAI/bge-small-en-v1.5 (Strong Retrival and fast)
    BAAI/bge-base-en-v1.5 (High accuracy but very slow)
"""
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)

print(f"Loading local embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)



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
        query_embedding = [embed_query(query)]


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
    
def embed_query(text: str) -> List[float]:
    """
    Embed the query using the local embedding model.
    
    Args:
        text: The query string

    Returns:
        Embedding vector as list of floats
    """
    return embedding_model.encode(text).tolist()



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
        query_embedding = [embed_query(query)]


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
    
    
    
# Below is for testing, you can remove this when actually implementing
    
    
if __name__ == "__main__":
    # Prompt user for collection name
    collection_name = input("Enter ChromaDB collection name: ").strip().lower()
    print(f"Using collection: '{collection_name}'")

    print("\nEnter your query (type 'exit' or 'quit' to stop):")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_query:
            continue  # Ignore empty input

        results_raw = query_collection(
            collection_name=collection_name,
            query=user_query,
            n_results=3,  # or however many results you want
            include_metadata=True,
            verbose=False
        )

        if isinstance(results_raw, str):
            # In case of failure message
            print(results_raw)
            continue

        print("\nResponse:")
        documents = results_raw.get("documents", [[]])[0]
        metadatas = results_raw.get("metadatas", [[]])[0]

        if not documents:
            print("No matching documents found.")
            continue

        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            print(f"\nResult #{i}:")
            print(f"Document:\n{doc}")
            print("Metadata:")
            print(json.dumps(meta, indent=2))
            print("-" * 80)

