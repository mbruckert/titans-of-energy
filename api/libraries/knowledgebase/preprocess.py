"""
Knowledge base preprocessing module for text processing and ChromaDB storage.

This module handles text cleaning, chunking, keyword extraction, entity recognition,
and embedding generation for documents to be stored in ChromaDB.
"""

import json
import os
import re
import shutil
import uuid
import warnings
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial

import spacy
from chromadb import PersistentClient
from chromadb.config import Settings
from dotenv import load_dotenv
from keybert import KeyBERT
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "text-embedding-ada-002"
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
openai_client = OpenAI(api_key=OPENAI_API_KEY)
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


def embed_text(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI's embedding model.
    
    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    try:
        response = openai_client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call for better performance.
    
    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    try:
        # OpenAI API has a limit on batch size, so we process in chunks
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            print(f"  📊 Processing embedding batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} texts)")
            
            response = openai_client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        raise


def process_chunk_metadata(chunk_data: Tuple[int, str, str]) -> Dict[str, Any]:
    """
    Process metadata for a single chunk (for parallel processing).
    
    Args:
        chunk_data: Tuple of (index, chunk_text, source)
    
    Returns:
        Dictionary with chunk metadata (without embedding)
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


def process_document(file_path: str, source: str) -> List[Dict[str, Any]]:
    """
    Process a document into chunks with metadata and embeddings using parallel processing.

    Args:
        file_path: Path to the document file
        source: Source identifier for the document

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
        embeddings = embed_texts_batch(texts)
        
        # Combine metadata with embeddings
        for chunk, embedding in zip(chunk_results, embeddings):
            chunk["embedding"] = embedding
            processed_chunks.append(chunk)
            
    except Exception as e:
        print(f"Error generating embeddings for {source}: {e}")
        return []

    return processed_chunks


def insert_into_chroma(collection_name: str, chunk_data: List[Dict[str, Any]]) -> None:
    """
    Insert processed chunks into ChromaDB collection using batch processing.

    Args:
        collection_name: Name of the ChromaDB collection
        chunk_data: List of processed chunks to insert
    """
    if not chunk_data:
        print("No chunks to insert")
        return

    try:
        collection = chroma_client.get_or_create_collection(collection_name)

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

def process_single_file(file_info: Tuple[str, str, str]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a single file and return its chunks.
    
    Args:
        file_info: Tuple of (file_path, filename, file_type)
    
    Returns:
        Tuple of (filename, chunk_data)
    """
    file_path, filename, file_type = file_info
    
    try:
        if file_type == "txt":
            chunk_data = process_document(file_path, filename)
        elif file_type == "json":
            chunk_data = process_json_document(file_path, filename)
        else:
            return filename, []
            
        return filename, chunk_data if chunk_data else []
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, []


def process_documents_for_collection(new_docs_dir: str, archive_dir: str, collection_name: str) -> None:
    """
    Process all new text and JSON files and add them to ChromaDB collection using parallel processing.

    Args:
        new_docs_dir: Directory containing new documents to process
        archive_dir: Directory to move processed documents to
        collection_name: Name of the ChromaDB collection
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
    
    # Prepare file info for parallel processing
    file_infos = []
    for filename in txt_files:
        file_path = os.path.join(new_docs_dir, filename)
        file_infos.append((file_path, filename, "txt"))
    
    for filename in json_files:
        file_path = os.path.join(new_docs_dir, filename)
        file_infos.append((file_path, filename, "json"))
    
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
        insert_into_chroma(collection_name, all_chunks)
    else:
        print("⚠️  No chunks to insert")

    # Move processed files to archive
    move_processed_files(new_docs_dir, archive_dir)
    print("🎉 Processing complete!")


def process_json_document(file_path: str, source: str) -> List[Dict[str, Any]]:
    """
    Process a JSON document containing Q&A pairs or other structured data using optimized processing.

    Args:
        file_path: Path to the JSON file
        source: Source identifier for the document

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
        embeddings = embed_texts_batch(texts)
        
        # Combine metadata with embeddings
        processed_chunks = []
        for chunk, embedding in zip(chunk_results, embeddings):
            chunk["embedding"] = embedding
            processed_chunks.append(chunk)
            
        return processed_chunks
        
    except Exception as e:
        print(f"Error generating embeddings for JSON {source}: {e}")
        return []
