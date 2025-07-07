"""
Knowledge base preprocessing module for text processing and ChromaDB storage.

This module handles text cleaning, chunking, keyword extraction, entity recognition,
and embedding generation for documents to be stored in ChromaDB.
Supports both OpenAI and open source embedding models with automatic collection
management when embedding models change.
"""

import json
import os
import re
import shutil
import uuid
import warnings
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial

import spacy
from chromadb import PersistentClient
from chromadb.config import Settings
from dotenv import load_dotenv
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Import unified embedding models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from embedding_models import get_embedding_model, EmbeddingModelType, embedding_manager

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "chroma_db"
SPACY_MODEL = "en_core_web_sm"

# Performance configuration
MAX_WORKERS = min(8, multiprocessing.cpu_count())  # Limit to avoid API rate limits
BATCH_SIZE = 100  # Process embeddings in batches
CHUNK_BATCH_SIZE = 50  # Insert chunks in batches

# Set tokenizer parallelism for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Suppress spaCy warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*falling back to type probe function.*")

# Initialize models and clients
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    raise RuntimeError(
        f"spaCy model '{SPACY_MODEL}' not found. Install with: python -m spacy download {SPACY_MODEL}")

kw_model = KeyBERT()
chroma_client = PersistentClient(path=CHROMA_DB_PATH)

# Device detection for GPU acceleration
def get_processing_device():
    """Detect available processing device for optimization."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", "Apple Silicon"
        else:
            return "cpu", f"CPU ({multiprocessing.cpu_count()} cores)"
    except ImportError:
        return "cpu", f"CPU ({multiprocessing.cpu_count()} cores)"

DEVICE, DEVICE_NAME = get_processing_device()
print(f"🚀 Knowledge base processing using: {DEVICE_NAME} ({DEVICE})")


def get_embedding_model_signature(embedding_config: Dict[str, Any]) -> str:
    """
    Generate a unique signature for an embedding model configuration.
    
    Args:
        embedding_config: Configuration containing model type, name, and settings
    
    Returns:
        Unique string signature for the embedding model
    """
    model_type = embedding_config.get("model_type", "sentence_transformers")
    model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
    
    # Create a signature that uniquely identifies this embedding configuration
    signature = f"{model_type}:{model_name}"
    
    # Include relevant config parameters that affect embeddings
    config = embedding_config.get("config", {})
    relevant_params = ["device", "batch_size"]  # Add more if needed
    
    for param in relevant_params:
        if param in config:
            signature += f":{param}={config[param]}"
    
    return signature


def get_collection_embedding_info(collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Get embedding model information stored in collection metadata.
    
    Args:
        collection_name: Name of the ChromaDB collection
    
    Returns:
        Dictionary with embedding model info, or None if collection doesn't exist
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        metadata = collection.metadata or {}
        
        return {
            "embedding_model_signature": metadata.get("embedding_model_signature"),
            "embedding_model_type": metadata.get("embedding_model_type"),
            "embedding_model_name": metadata.get("embedding_model_name"),
            "embedding_dimensions": metadata.get("embedding_dimensions"),
            "created_at": metadata.get("created_at"),
            "last_updated": metadata.get("last_updated")
        }
    except Exception:
        return None


def is_embedding_model_compatible(collection_name: str, embedding_config: Dict[str, Any]) -> bool:
    """
    Check if the current embedding model is compatible with an existing collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        embedding_config: Current embedding configuration
    
    Returns:
        True if compatible, False if incompatible or collection doesn't exist
    """
    collection_info = get_collection_embedding_info(collection_name)
    
    if not collection_info:
        return False  # Collection doesn't exist
    
    current_signature = get_embedding_model_signature(embedding_config)
    stored_signature = collection_info.get("embedding_model_signature")
    
    if not stored_signature:
        # Legacy collection without embedding info - assume incompatible
        print(f"⚠️  Collection '{collection_name}' has no embedding model metadata - assuming incompatible")
        return False
    
    is_compatible = current_signature == stored_signature
    
    if not is_compatible:
        print(f"🔄 Embedding model mismatch for collection '{collection_name}':")
        print(f"   Current:  {current_signature}")
        print(f"   Stored:   {stored_signature}")
    
    return is_compatible


def backup_and_recreate_collection(
    collection_name: str,
    embedding_config: Dict[str, Any],
    source_documents_path: Optional[str] = None,
    archive_path: Optional[str] = None
) -> bool:
    """
    Backup an existing collection and prepare for recreation with new embedding model.
    
    Args:
        collection_name: Name of the ChromaDB collection
        embedding_config: New embedding configuration
        source_documents_path: Path to source documents (if available)
        archive_path: Path to archive processed documents (if available)
    
    Returns:
        True if backup was successful, False if collection didn't exist
    """
    try:
        # Check if collection exists
        collection = chroma_client.get_collection(name=collection_name)
        
        # Get all documents from the existing collection
        results = collection.get(include=['documents', 'metadatas'])
        
        if not results['ids']:
            print(f"📭 Collection '{collection_name}' is empty - no backup needed")
            # Delete empty collection
            chroma_client.delete_collection(collection_name)
            return True
        
        # Create backup filename with timestamp
        import time
        timestamp = int(time.time())
        backup_filename = f"{collection_name}_backup_{timestamp}.json"
        backup_path = os.path.join(CHROMA_DB_PATH, backup_filename)
        
        # Prepare backup data
        backup_data = {
            "collection_name": collection_name,
            "backup_timestamp": timestamp,
            "original_embedding_config": get_collection_embedding_info(collection_name),
            "new_embedding_config": embedding_config,
            "documents": results['documents'],
            "metadatas": results['metadatas'],
            "ids": results['ids'],
            "count": len(results['ids'])
        }
        
        # Save backup
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Backed up {len(results['ids'])} documents from '{collection_name}' to {backup_filename}")
        
        # Delete the old collection
        chroma_client.delete_collection(collection_name)
        print(f"🗑️  Deleted old collection '{collection_name}'")
        
        return True
        
    except Exception as e:
        if "does not exist" in str(e).lower():
            return False  # Collection didn't exist
        else:
            print(f"❌ Error backing up collection '{collection_name}': {e}")
            raise


def clean_text(text: str) -> str:
    """
    Clean input text by removing page numbers, footnotes, and normalizing whitespace.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text string
    """
    # Remove page markers and footnotes
    text = re.sub(r"\[\d+\]|\(Page \d+\)", "", text)
    text = re.sub(r"\^\d+|Footnote \d+:", "", text)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> List[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text: Text to split into sentences

    Returns:
        List of sentence strings
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract keywords using KeyBERT and TF-IDF with fallback handling.

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return

    Returns:
        List of extracted keywords
    """
    # KeyBERT extraction
    keybert_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english"
    )

    # TF-IDF extraction with error handling
    tfidf_keywords = []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        sorted_indices = tfidf_scores.argsort()[::-1]
        tfidf_keywords = [feature_names[i]
                          for i in sorted_indices[:max_keywords]]
    except ValueError as e:
        print(f"TF-IDF extraction failed, using fallback: {str(e)[:100]}...")
        # Fallback: extract non-stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        tfidf_keywords = [
            word for word in words if word not in ENGLISH_STOP_WORDS][:max_keywords]

    # Combine and deduplicate
    combined_keywords = list(
        set([kw[0] for kw in keybert_keywords] + tfidf_keywords))
    return combined_keywords[:max_keywords]


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities using spaCy.

    Args:
        text: Text to extract entities from

    Returns:
        List of tuples containing (entity_text, entity_label)
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def embed_text(text: str, embedding_config: Dict[str, Any]) -> List[float]:
    """
    Generate embedding for text using configured embedding model.
    
    Args:
        text: Text to embed
        embedding_config: Configuration containing model type, name, and settings

    Returns:
        Embedding vector as list of floats
    """
    try:
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
        
        return embedding_model.embed_text(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def embed_texts_batch(texts: List[str], embedding_config: Dict[str, Any]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using configured embedding model.
    
    Args:
        texts: List of texts to embed
        embedding_config: Configuration containing model type, name, and settings

    Returns:
        List of embedding vectors
    """
    try:
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
        
        return embedding_model.embed_texts_batch(texts)
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        raise


def get_embedding_dimensions(embedding_config: Dict[str, Any]) -> int:
    """
    Get the embedding dimensions for a given embedding configuration.
    
    Args:
        embedding_config: Configuration containing model type, name, and settings
    
    Returns:
        Number of dimensions in the embedding vector
    """
    try:
        # Generate a test embedding to determine dimensions
        test_embedding = embed_text("test", embedding_config)
        return len(test_embedding)
    except Exception as e:
        print(f"Error determining embedding dimensions: {e}")
        # Return default dimensions based on model type/name
        model_type = embedding_config.get("model_type", "sentence_transformers")
        model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        
        # Default dimensions for known models
        default_dimensions = {
            "openai": {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            },
            "sentence_transformers": {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-large-en-v1.5": 1024
            }
        }
        
        return default_dimensions.get(model_type, {}).get(model_name, 384)


def process_chunk_metadata(chunk_data: Tuple[int, str, str]) -> Optional[Dict[str, Any]]:
    """
    Process metadata for a single chunk (for parallel processing).
    
    Args:
        chunk_data: Tuple of (index, chunk_text, source)
    
    Returns:
        Dictionary with chunk metadata (without embedding) or None if failed
    """
    i, chunk, source = chunk_data
    
    try:
        keywords = extract_keywords(chunk)
        entities = extract_entities(chunk)
        
        metadata = {
            "source": source,
            "chunk_id": i + 1,
            "keywords": ", ".join(keywords),
            "entities": ", ".join([f"{text} ({label})" for text, label in entities])
        }
        
        chunk_id = f"{source}_{i+1}_{uuid.uuid4().hex[:8]}"
        
        return {
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata,
            "index": i
        }
    except Exception as e:
        print(f"Error processing metadata for chunk {i+1} from {source}: {e}")
        return None


def process_document(file_path: str, source: str, embedding_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a document into chunks with metadata and embeddings using parallel processing.

    Args:
        file_path: Path to the document file
        source: Source identifier for the document
        embedding_config: Configuration for embedding model

    Returns:
        List of processed chunks with metadata
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = clean_text(f.read())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    chunks = preprocess_text(raw_text)
    
    # Filter out very short chunks
    valid_chunks = [(i, chunk, source) for i, chunk in enumerate(chunks) if len(chunk.strip()) >= 10]
    
    if not valid_chunks:
        print(f"No valid chunks found in {source}")
        return []
    
    print(f"  🔄 Processing {len(valid_chunks)} chunks from {source} with {MAX_WORKERS} workers...")
    
    # Process metadata in parallel
    processed_chunks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunk_results = list(executor.map(process_chunk_metadata, valid_chunks))
    
    # Filter out failed chunks
    chunk_results = [chunk for chunk in chunk_results if chunk is not None]
    
    if not chunk_results:
        print(f"No chunks successfully processed from {source}")
        return []
    
    # Extract texts for batch embedding
    texts = [chunk["text"] for chunk in chunk_results]
    
    print(f"  🧠 Generating embeddings for {len(texts)} chunks...")
    try:
        embeddings = embed_texts_batch(texts, embedding_config)
        
        # Combine metadata with embeddings
        for chunk, embedding in zip(chunk_results, embeddings):
            chunk["embedding"] = embedding
            processed_chunks.append(chunk)
            
    except Exception as e:
        print(f"Error generating embeddings for {source}: {e}")
        return []

    return processed_chunks


def insert_into_chroma(collection_name: str, chunk_data: List[Dict[str, Any]], embedding_config: Dict[str, Any]) -> None:
    """
    Insert processed chunks into ChromaDB collection using batch processing.
    Creates collection with embedding model metadata for future compatibility checking.

    Args:
        collection_name: Name of the ChromaDB collection
        chunk_data: List of processed chunks to insert
        embedding_config: Configuration for embedding model
    """
    if not chunk_data:
        print("No chunks to insert")
        return

    try:
        # Check if collection exists and is compatible
        collection_exists = True
        try:
            existing_collection = chroma_client.get_collection(collection_name)
        except Exception:
            collection_exists = False
        
        if collection_exists:
            if not is_embedding_model_compatible(collection_name, embedding_config):
                print(f"🔄 Embedding model incompatible - recreating collection '{collection_name}'")
                backup_and_recreate_collection(collection_name, embedding_config)
                collection_exists = False
        
        # Create collection with metadata if it doesn't exist
        if not collection_exists:
            import time
            
            # Prepare collection metadata
            embedding_dimensions = get_embedding_dimensions(embedding_config)
            collection_metadata = {
                "embedding_model_signature": get_embedding_model_signature(embedding_config),
                "embedding_model_type": embedding_config.get("model_type", "sentence_transformers"),
                "embedding_model_name": embedding_config.get("model_name", "all-MiniLM-L6-v2"),
                "embedding_dimensions": embedding_dimensions,
                "created_at": time.time(),
                "last_updated": time.time()
            }
            
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            
            print(f"✨ Created new collection '{collection_name}' with embedding model: {embedding_config.get('model_type')}:{embedding_config.get('model_name')} ({embedding_dimensions}D)")
        else:
            collection = chroma_client.get_collection(collection_name)
            print(f"📚 Using existing compatible collection '{collection_name}'")

        # Insert in batches for better performance
        total_chunks = len(chunk_data)
        for i in range(0, total_chunks, CHUNK_BATCH_SIZE):
            batch = chunk_data[i:i + CHUNK_BATCH_SIZE]
            batch_num = i // CHUNK_BATCH_SIZE + 1
            total_batches = (total_chunks + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE
            
            print(f"  💾 Inserting batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            collection.add(
                documents=[chunk["text"] for chunk in batch],
                embeddings=[chunk["embedding"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
                ids=[chunk["id"] for chunk in batch]
            )

        # Update last_updated timestamp
        try:
            import time
            current_metadata = collection.metadata or {}
            current_metadata["last_updated"] = time.time()
            collection.modify(metadata=current_metadata)
        except Exception as e:
            print(f"Warning: Could not update collection metadata: {e}")

        print(f"✅ Successfully inserted {total_chunks} chunks into collection '{collection_name}'")
    except Exception as e:
        print(f"Error inserting chunks into ChromaDB: {e}")
        raise


def move_processed_files(source_dir: str, target_dir: str) -> None:
    """
    Move processed files from source to target directory.

    Args:
        source_dir: Source directory path
        target_dir: Target directory path
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith((".txt", ".json")):
            try:
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)
                shutil.move(source_path, target_path)
                print(f"Moved {filename} to {target_dir}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")

def process_single_file(file_info: Tuple[str, str, str, Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a single file and return its chunks.
    
    Args:
        file_info: Tuple of (file_path, filename, file_type, embedding_config)
    
    Returns:
        Tuple of (filename, chunk_data)
    """
    file_path, filename, file_type, embedding_config = file_info
    
    try:
        if file_type == "txt":
            chunk_data = process_document(file_path, filename, embedding_config)
        elif file_type == "json":
            chunk_data = process_json_document(file_path, filename, embedding_config)
        else:
            return filename, []
            
        return filename, chunk_data if chunk_data else []
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, []


def process_documents_for_collection(new_docs_dir: str, archive_dir: str, collection_name: str, embedding_config: Dict[str, Any]) -> None:
    """
    Process all new text and JSON files and add them to ChromaDB collection using parallel processing.
    Automatically handles embedding model compatibility and collection recreation.

    Args:
        new_docs_dir: Directory containing new documents to process
        archive_dir: Directory to move processed documents to
        collection_name: Name of the ChromaDB collection
        embedding_config: Configuration for embedding model
    """
    if not os.path.exists(new_docs_dir):
        print(f"Source directory {new_docs_dir} does not exist")
        return

    txt_files = [f for f in os.listdir(new_docs_dir) if f.endswith(".txt")]
    json_files = [f for f in os.listdir(new_docs_dir) if f.endswith(".json")]

    total_files = len(txt_files) + len(json_files)

    if total_files == 0:
        print(f"No .txt or .json files found in {new_docs_dir}")
        return

    print(f"🚀 Processing {total_files} files using {DEVICE_NAME}...")
    print(f"📊 Embedding model: {embedding_config.get('model_type')}:{embedding_config.get('model_name')}")
    
    # Check embedding model compatibility
    if not is_embedding_model_compatible(collection_name, embedding_config):
        print(f"🔄 Collection '{collection_name}' needs to be recreated due to embedding model change")
    
    # Prepare file info for parallel processing
    file_infos = []
    for filename in txt_files:
        file_path = os.path.join(new_docs_dir, filename)
        file_infos.append((file_path, filename, "txt", embedding_config))
    
    for filename in json_files:
        file_path = os.path.join(new_docs_dir, filename)
        file_infos.append((file_path, filename, "json", embedding_config))
    
    # Process files in parallel (but limit concurrency to avoid overwhelming OpenAI API)
    all_chunks = []
    file_workers = min(3, total_files)  # Limit to 3 concurrent files to avoid API rate limits
    
    print(f"📁 Processing files with {file_workers} concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=file_workers) as executor:
        results = list(executor.map(process_single_file, file_infos))
    
    # Collect all chunks and insert in batches
    for filename, chunk_data in results:
        if chunk_data:
            print(f"📄 {filename}: {len(chunk_data)} chunks processed")
            all_chunks.extend(chunk_data)
        else:
            print(f"⚠️  {filename}: No valid chunks extracted")
    
    # Insert all chunks at once for better performance
    if all_chunks:
        print(f"\n💾 Inserting {len(all_chunks)} total chunks into collection '{collection_name}'...")
        insert_into_chroma(collection_name, all_chunks, embedding_config)
    else:
        print("⚠️  No chunks to insert")

    # Move processed files to archive
    move_processed_files(new_docs_dir, archive_dir)
    print("🎉 Processing complete!")


def process_json_document(file_path: str, source: str, embedding_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a JSON document containing Q&A pairs or other structured data using optimized processing.

    Args:
        file_path: Path to the JSON file
        source: Source identifier for the document
        embedding_config: Configuration for embedding model

    Returns:
        List of processed chunks with metadata
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return []

    # Extract all text chunks first
    text_chunks = []
    chunk_counter = 0

    def extract_json_texts(value, prefix=""):
        """Recursively extract text content from JSON values."""
        nonlocal chunk_counter

        if isinstance(value, dict):
            # Handle Q&A pairs
            if "question" in value and "answer" in value:
                question = str(value["question"]).strip()
                answer = str(value["answer"]).strip()

                if question and len(question) > 10:
                    chunk_counter += 1
                    text_chunks.append({
                        "text": question,
                        "type": "question",
                        "chunk_id": chunk_counter,
                        "field": prefix
                    })

                if answer and len(answer) > 10:
                    chunk_counter += 1
                    text_chunks.append({
                        "text": answer,
                        "type": "answer", 
                        "chunk_id": chunk_counter,
                        "field": prefix
                    })
            else:
                # Process other dictionary structures
                for key, val in value.items():
                    extract_json_texts(val, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            # Process list items
            for i, item in enumerate(value):
                extract_json_texts(item, f"{prefix}[{i}]" if prefix else f"[{i}]")
        elif isinstance(value, str) and len(value.strip()) > 10:
            # Process string values as text chunks
            text = clean_text(value.strip())
            if len(text) > 10:
                chunk_counter += 1
                text_chunks.append({
                    "text": text,
                    "type": "text",
                    "chunk_id": chunk_counter,
                    "field": prefix
                })

    # Extract all texts
    extract_json_texts(json_data)
    
    if not text_chunks:
        print(f"No valid text chunks found in {source}")
        return []
    
    print(f"  🔄 Processing {len(text_chunks)} JSON chunks from {source} with {MAX_WORKERS} workers...")
    
    # Process metadata in parallel
    chunk_metadata_data = [(chunk["chunk_id"], chunk["text"], source, chunk["type"], chunk["field"]) 
                           for chunk in text_chunks]
    
    def process_json_chunk_metadata(chunk_data):
        chunk_id, text, source, chunk_type, field = chunk_data
        try:
            keywords = extract_keywords(text)
            entities = extract_entities(text)
            
            metadata = {
                "source": source,
                "chunk_id": chunk_id,
                "type": chunk_type,
                "keywords": ", ".join(keywords),
                "entities": ", ".join([f"{text} ({label})" for text, label in entities])
            }
            
            if field:
                metadata["field"] = field
            
            unique_chunk_id = f"{source}_{chunk_type[0]}_{chunk_id}_{uuid.uuid4().hex[:8]}"
            
            return {
                "id": unique_chunk_id,
                "text": text,
                "metadata": metadata
            }
        except Exception as e:
            print(f"Error processing JSON chunk {chunk_id}: {e}")
            return None
    
    # Process metadata in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunk_results = list(executor.map(process_json_chunk_metadata, chunk_metadata_data))
    
    # Filter out failed chunks
    chunk_results = [chunk for chunk in chunk_results if chunk is not None]
    
    if not chunk_results:
        print(f"No chunks successfully processed from {source}")
        return []
    
    # Extract texts for batch embedding
    texts = [chunk["text"] for chunk in chunk_results]
    
    print(f"  🧠 Generating embeddings for {len(texts)} JSON chunks...")
    try:
        embeddings = embed_texts_batch(texts, embedding_config)
        
        # Combine metadata with embeddings
        processed_chunks = []
        for chunk, embedding in zip(chunk_results, embeddings):
            chunk["embedding"] = embedding
            processed_chunks.append(chunk)
            
        return processed_chunks
        
    except Exception as e:
        print(f"Error generating embeddings for JSON {source}: {e}")
        return []


def list_collection_backups(collection_name: str = None) -> List[Dict[str, Any]]:
    """
    List available collection backups.
    
    Args:
        collection_name: Optional collection name to filter backups
    
    Returns:
        List of backup information dictionaries
    """
    backups = []
    backup_dir = CHROMA_DB_PATH
    
    if not os.path.exists(backup_dir):
        return backups
    
    for filename in os.listdir(backup_dir):
        if filename.endswith('_backup_*.json'):
            try:
                backup_path = os.path.join(backup_dir, filename)
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                backup_info = {
                    "filename": filename,
                    "path": backup_path,
                    "collection_name": backup_data.get("collection_name"),
                    "backup_timestamp": backup_data.get("backup_timestamp"),
                    "count": backup_data.get("count", 0),
                    "original_embedding_config": backup_data.get("original_embedding_config"),
                    "new_embedding_config": backup_data.get("new_embedding_config")
                }
                
                if collection_name is None or backup_info["collection_name"] == collection_name:
                    backups.append(backup_info)
                    
            except Exception as e:
                print(f"Error reading backup file {filename}: {e}")
    
    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x.get("backup_timestamp", 0), reverse=True)
    return backups


def restore_collection_from_backup(backup_path: str, embedding_config: Dict[str, Any]) -> bool:
    """
    Restore a collection from a backup file with new embedding model.
    
    Args:
        backup_path: Path to the backup JSON file
        embedding_config: New embedding configuration to use
    
    Returns:
        True if restoration was successful
    """
    try:
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        collection_name = backup_data["collection_name"]
        documents = backup_data["documents"]
        metadatas = backup_data["metadatas"]
        ids = backup_data["ids"]
        
        print(f"🔄 Restoring collection '{collection_name}' from backup with new embedding model...")
        print(f"📊 Processing {len(documents)} documents with {embedding_config.get('model_type')}:{embedding_config.get('model_name')}")
        
        # Generate new embeddings for all documents
        print("🧠 Generating new embeddings...")
        new_embeddings = embed_texts_batch(documents, embedding_config)
        
        # Prepare chunk data
        chunk_data = []
        for i, (doc, metadata, doc_id, embedding) in enumerate(zip(documents, metadatas, ids, new_embeddings)):
            chunk_data.append({
                "text": doc,
                "metadata": metadata,
                "id": doc_id,
                "embedding": embedding
            })
        
        # Insert into ChromaDB
        insert_into_chroma(collection_name, chunk_data, embedding_config)
        
        print(f"✅ Successfully restored collection '{collection_name}' with new embedding model")
        return True
        
    except Exception as e:
        print(f"❌ Error restoring collection from backup: {e}")
        return False


def recreate_collection_with_new_embedding_model(
    collection_name: str,
    new_embedding_config: Dict[str, Any],
    source_documents_path: Optional[str] = None,
    archive_path: Optional[str] = None
) -> bool:
    """
    Recreate a collection with a new embedding model.
    
    Args:
        collection_name: Name of the collection to recreate
        new_embedding_config: New embedding model configuration
        source_documents_path: Path to source documents (if available)
        archive_path: Path to archive processed documents (if available)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🔄 Recreating collection '{collection_name}' with new embedding model...")
        
        # Get new embedding model signature
        new_signature = get_embedding_model_signature(new_embedding_config)
        print(f"   New embedding model: {new_signature}")
        
        # Check if collection exists and get its info
        collection_info = get_collection_embedding_info(collection_name)
        if collection_info:
            old_signature = collection_info.get("embedding_model_signature", "unknown")
            print(f"   Old embedding model: {old_signature}")
            
            # If signatures match, no need to recreate
            if old_signature == new_signature:
                print(f"✅ Collection '{collection_name}' already uses the correct embedding model")
                return True
        
        # Create backup before recreating
        backup_success = backup_and_recreate_collection(
            collection_name, 
            new_embedding_config, 
            source_documents_path, 
            archive_path
        )
        
        if backup_success:
            print(f"✅ Successfully recreated collection '{collection_name}' with new embedding model")
            return True
        else:
            print(f"❌ Failed to recreate collection '{collection_name}'")
            return False
            
    except Exception as e:
        print(f"❌ Error recreating collection '{collection_name}': {e}")
        return False


def ensure_collection_compatible_with_embedding_model(
    collection_name: str,
    embedding_config: Dict[str, Any],
    source_documents_path: Optional[str] = None,
    archive_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ensure a collection is compatible with the given embedding model.
    Recreates the collection if needed.
    
    Args:
        collection_name: Name of the collection
        embedding_config: Embedding model configuration
        source_documents_path: Path to source documents for recreation
        archive_path: Path to archive for recreation
    
    Returns:
        Dictionary with compatibility status and actions taken
    """
    try:
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
            collection_exists = True
        except Exception:
            collection_exists = False
        
        if not collection_exists:
            return {
                "exists": False,
                "compatible": False,
                "action": "none",
                "message": f"Collection '{collection_name}' does not exist"
            }
        
        # Check compatibility
        if is_embedding_model_compatible(collection_name, embedding_config):
            return {
                "exists": True,
                "compatible": True,
                "action": "none",
                "message": f"Collection '{collection_name}' is already compatible"
            }
        
        # Collection exists but is incompatible - recreate it
        print(f"🔄 Collection '{collection_name}' is incompatible with current embedding model")
        
        if source_documents_path and os.path.exists(source_documents_path):
            # Recreate using source documents
            success = recreate_collection_with_new_embedding_model(
                collection_name,
                embedding_config,
                source_documents_path,
                archive_path
            )
            
            if success:
                return {
                    "exists": True,
                    "compatible": True,
                    "action": "recreated",
                    "message": f"Collection '{collection_name}' recreated with new embedding model"
                }
            else:
                return {
                    "exists": True,
                    "compatible": False,
                    "action": "failed",
                    "message": f"Failed to recreate collection '{collection_name}'"
                }
        else:
            # No source documents available - suggest manual recreation
            return {
                "exists": True,
                "compatible": False,
                "action": "manual_required",
                "message": f"Collection '{collection_name}' needs manual recreation - no source documents available"
            }
            
    except Exception as e:
        return {
            "exists": False,
            "compatible": False,
            "action": "error",
            "message": f"Error checking collection compatibility: {e}"
        }


def get_character_collection_names(character_name: str) -> Dict[str, str]:
    """
    Get the standard collection names for a character.
    
    Args:
        character_name: Name of the character
    
    Returns:
        Dictionary with collection names for different purposes
    """
    normalized_name = character_name.lower().replace(' ', '')
    
    return {
        "knowledge": f"{normalized_name}-knowledge",
        "style": f"{normalized_name}-style"
    }


def ensure_character_collections_compatible(
    character_name: str,
    knowledge_base_embedding_config: Optional[Dict[str, Any]] = None,
    style_tuning_embedding_config: Optional[Dict[str, Any]] = None,
    character_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ensure all collections for a character are compatible with their embedding models.
    
    Args:
        character_name: Name of the character
        knowledge_base_embedding_config: Embedding config for knowledge base
        style_tuning_embedding_config: Embedding config for style tuning
        character_dir: Path to character directory (for finding source documents)
    
    Returns:
        Dictionary with compatibility status for all collections
    """
    results = {
        "character_name": character_name,
        "knowledge_base": {"checked": False},
        "style_tuning": {"checked": False}
    }
    
    collection_names = get_character_collection_names(character_name)
    
    # Check knowledge base collection
    if knowledge_base_embedding_config:
        kb_docs_path = None
        kb_archive_path = None
        
        if character_dir:
            kb_docs_path = os.path.join(character_dir, "kb_docs")
            kb_archive_path = os.path.join(character_dir, "kb_archive")
        
        kb_result = ensure_collection_compatible_with_embedding_model(
            collection_names["knowledge"],
            knowledge_base_embedding_config,
            kb_docs_path,
            kb_archive_path
        )
        
        results["knowledge_base"] = {
            "checked": True,
            "collection_name": collection_names["knowledge"],
            **kb_result
        }
    
    # Check style tuning collection
    if style_tuning_embedding_config:
        style_docs_path = None
        style_archive_path = None
        
        if character_dir:
            style_docs_path = os.path.join(character_dir, "style_docs")
            style_archive_path = os.path.join(character_dir, "style_archive")
        
        style_result = ensure_collection_compatible_with_embedding_model(
            collection_names["style"],
            style_tuning_embedding_config,
            style_docs_path,
            style_archive_path
        )
        
        results["style_tuning"] = {
            "checked": True,
            "collection_name": collection_names["style"],
            **style_result
        }
    
    return results


def handle_character_embedding_model_change(
    character_name: str,
    old_embedding_config: Optional[Dict[str, Any]],
    new_embedding_config: Dict[str, Any],
    collection_type: str,  # "knowledge" or "style"
    character_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle embedding model change for a character's collection.
    
    Args:
        character_name: Name of the character
        old_embedding_config: Previous embedding configuration
        new_embedding_config: New embedding configuration
        collection_type: Type of collection ("knowledge" or "style")
        character_dir: Path to character directory
    
    Returns:
        Dictionary with the result of the operation
    """
    try:
        collection_names = get_character_collection_names(character_name)
        collection_name = collection_names.get(collection_type)
        
        if not collection_name:
            return {
                "success": False,
                "error": f"Invalid collection type: {collection_type}"
            }
        
        # Check if embedding model actually changed
        if old_embedding_config:
            old_signature = get_embedding_model_signature(old_embedding_config)
            new_signature = get_embedding_model_signature(new_embedding_config)
            
            if old_signature == new_signature:
                return {
                    "success": True,
                    "action": "none",
                    "message": "Embedding model unchanged - no action needed"
                }
        
        # Determine source and archive paths
        source_path = None
        archive_path = None
        
        if character_dir:
            if collection_type == "knowledge":
                source_path = os.path.join(character_dir, "kb_docs")
                archive_path = os.path.join(character_dir, "kb_archive")
            elif collection_type == "style":
                source_path = os.path.join(character_dir, "style_docs")
                archive_path = os.path.join(character_dir, "style_archive")
        
        # Ensure collection compatibility
        result = ensure_collection_compatible_with_embedding_model(
            collection_name,
            new_embedding_config,
            source_path,
            archive_path
        )
        
        return {
            "success": result.get("compatible", False),
            "collection_name": collection_name,
            "action": result.get("action", "unknown"),
            "message": result.get("message", "Unknown result")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error handling embedding model change: {e}"
        }


def delete_character_collections(character_name: str) -> Dict[str, Any]:
    """
    Delete all collections associated with a character.
    
    Args:
        character_name: Name of the character
    
    Returns:
        Dictionary with deletion results
    """
    results = {
        "character_name": character_name,
        "knowledge_base": {"deleted": False},
        "style_tuning": {"deleted": False}
    }
    
    collection_names = get_character_collection_names(character_name)
    
    # Delete knowledge base collection
    try:
        chroma_client.delete_collection(collection_names["knowledge"])
        results["knowledge_base"]["deleted"] = True
        results["knowledge_base"]["message"] = f"Deleted collection '{collection_names['knowledge']}'"
        print(f"✅ Deleted knowledge base collection: {collection_names['knowledge']}")
    except Exception as e:
        results["knowledge_base"]["error"] = str(e)
        print(f"⚠️  Could not delete knowledge base collection '{collection_names['knowledge']}': {e}")
    
    # Delete style tuning collection
    try:
        chroma_client.delete_collection(collection_names["style"])
        results["style_tuning"]["deleted"] = True
        results["style_tuning"]["message"] = f"Deleted collection '{collection_names['style']}'"
        print(f"✅ Deleted style tuning collection: {collection_names['style']}")
    except Exception as e:
        results["style_tuning"]["error"] = str(e)
        print(f"⚠️  Could not delete style tuning collection '{collection_names['style']}': {e}")
    
    return results


def rename_character_collections(old_character_name: str, new_character_name: str) -> Dict[str, Any]:
    """
    Rename collections when a character's name changes.
    
    Args:
        old_character_name: Previous character name
        new_character_name: New character name
    
    Returns:
        Dictionary with renaming results
    """
    if old_character_name == new_character_name:
        return {
            "success": True,
            "action": "none",
            "message": "Character name unchanged - no collection renaming needed"
        }
    
    old_collection_names = get_character_collection_names(old_character_name)
    new_collection_names = get_character_collection_names(new_character_name)
    
    results = {
        "old_character_name": old_character_name,
        "new_character_name": new_character_name,
        "knowledge_base": {"renamed": False},
        "style_tuning": {"renamed": False}
    }
    
    # Rename knowledge base collection
    try:
        old_kb_collection = chroma_client.get_collection(old_collection_names["knowledge"])
        
        # Get all data from old collection
        all_data = old_kb_collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        if all_data['ids']:
            # Create new collection with new name
            new_kb_collection = chroma_client.create_collection(new_collection_names["knowledge"])
            
            # Copy all data to new collection
            new_kb_collection.add(
                documents=all_data['documents'],
                metadatas=all_data['metadatas'],
                embeddings=all_data['embeddings'],
                ids=all_data['ids']
            )
            
            # Delete old collection
            chroma_client.delete_collection(old_collection_names["knowledge"])
            
            results["knowledge_base"]["renamed"] = True
            results["knowledge_base"]["message"] = f"Renamed '{old_collection_names['knowledge']}' to '{new_collection_names['knowledge']}'"
            print(f"✅ Renamed knowledge base collection: {old_collection_names['knowledge']} → {new_collection_names['knowledge']}")
        else:
            results["knowledge_base"]["message"] = f"Old knowledge base collection '{old_collection_names['knowledge']}' was empty"
            print(f"ℹ️  Old knowledge base collection was empty: {old_collection_names['knowledge']}")
            
    except Exception as e:
        results["knowledge_base"]["error"] = str(e)
        print(f"⚠️  Could not rename knowledge base collection: {e}")
    
    # Rename style tuning collection
    try:
        old_style_collection = chroma_client.get_collection(old_collection_names["style"])
        
        # Get all data from old collection
        all_data = old_style_collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        if all_data['ids']:
            # Create new collection with new name
            new_style_collection = chroma_client.create_collection(new_collection_names["style"])
            
            # Copy all data to new collection
            new_style_collection.add(
                documents=all_data['documents'],
                metadatas=all_data['metadatas'],
                embeddings=all_data['embeddings'],
                ids=all_data['ids']
            )
            
            # Delete old collection
            chroma_client.delete_collection(old_collection_names["style"])
            
            results["style_tuning"]["renamed"] = True
            results["style_tuning"]["message"] = f"Renamed '{old_collection_names['style']}' to '{new_collection_names['style']}'"
            print(f"✅ Renamed style tuning collection: {old_collection_names['style']} → {new_collection_names['style']}")
        else:
            results["style_tuning"]["message"] = f"Old style tuning collection '{old_collection_names['style']}' was empty"
            print(f"ℹ️  Old style tuning collection was empty: {old_collection_names['style']}")
            
    except Exception as e:
        results["style_tuning"]["error"] = str(e)
        print(f"⚠️  Could not rename style tuning collection: {e}")
    
    return results
