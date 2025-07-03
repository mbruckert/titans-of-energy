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

import spacy
from chromadb import PersistentClient
from chromadb.config import Settings
from dotenv import load_dotenv
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Library for free embedding models!********
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "chroma_db"

""" 
Below change the embedding model from sentence transformers

Choices:
    all-MiniLM-L6-v2 (Lightweight and fast)
    all-mpnet-base-v2 (Higher quality but slower)
    BAAI/bge-small-en-v1.5 (Strong Retrival and fast)
    BAAI/bge-base-en-v1.5 (High accuracy but very slow)
"""
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

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

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


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


# Free Embedding**************
def embed_text(text: str) -> List[float]:
    """
    Generate embedding for a single text using the local embedding model.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding
    """
    return embedding_model.encode(text).tolist()





def process_document(file_path: str, source: str) -> List[Dict[str, Any]]:
    """
    Process a document into chunks with metadata and embeddings.

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
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        # Skip very short chunks
        if len(chunk.strip()) < 10:
            continue

        try:
            keywords = extract_keywords(chunk)
            entities = extract_entities(chunk)
            embedding = embed_text(chunk)

            metadata = {
                "source": source,
                "chunk_id": i + 1,
                "keywords": ", ".join(keywords),
                "entities": ", ".join([f"{text} ({label})" for text, label in entities])
            }

            chunk_id = f"{source}_{i+1}_{uuid.uuid4().hex[:8]}"

            processed_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata
            })
        except Exception as e:
            print(f"Error processing chunk {i+1} from {source}: {e}")
            continue

    return processed_chunks


def insert_into_chroma(collection_name: str, chunks: List[Dict[str, Any]]):
    """
    Insert processed chunks into a ChromaDB collection.

    Args:
        collection_name: Name of the collection to insert into
        chunks: List of dicts with keys: id, text, embedding, metadata
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )

    print(f"Inserted {len(chunks)} chunks into collection '{collection_name}'.")



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

def process_documents_for_collection(new_docs_dir: str, archive_dir: str, collection_name: str) -> None:
    """
    Process all new text and JSON files and add them to ChromaDB collection.

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

    print(f"Processing {total_files} files...")

    # Process text files
    for filename in txt_files:
        file_path = os.path.join(new_docs_dir, filename)
        print(f"Processing {filename}...")

        try:
            chunk_data = process_document(file_path, filename)
            if chunk_data:
                insert_into_chroma(collection_name, chunk_data)
            else:
                print(f"No valid chunks extracted from {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Process JSON files
    for filename in json_files:
        file_path = os.path.join(new_docs_dir, filename)
        print(f"Processing {filename}...")

        try:
            chunk_data = process_json_document(file_path, filename)
            if chunk_data:
                insert_into_chroma(collection_name, chunk_data)
            else:
                print(f"No valid chunks extracted from {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Move processed files to archive
    move_processed_files(new_docs_dir, archive_dir)
    print("Processing complete!")
    
    
def process_json_document(file_path: str, source: str) -> List[Dict[str, Any]]:
    """
    Process a JSON document containing Q&A pairs or other structured data.

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

    processed_chunks = []
    chunk_counter = 0

    def process_json_value(value, prefix=""):
        """Recursively process JSON values to extract text content."""
        nonlocal chunk_counter

        if isinstance(value, dict):
            # Handle Q&A pairs
            if "question" in value and "answer" in value:
                question = str(value["question"]).strip()
                answer = str(value["answer"]).strip()

                if question and len(question) > 10:
                    chunk_counter += 1
                    try:
                        keywords = extract_keywords(question)
                        entities = extract_entities(question)
                        embedding = embed_text(question)

                        metadata = {
                            "source": source,
                            "chunk_id": chunk_counter,
                            "type": "question",
                            "keywords": ", ".join(keywords),
                            "entities": ", ".join([f"{text} ({label})" for text, label in entities])
                        }

                        chunk_id = f"{source}_q_{chunk_counter}_{uuid.uuid4().hex[:8]}"
                        processed_chunks.append({
                            "id": chunk_id,
                            "text": question,
                            "embedding": embedding,
                            "metadata": metadata
                        })
                    except Exception as e:
                        print(
                            f"Error processing question chunk {chunk_counter}: {e}")

                if answer and len(answer) > 10:
                    chunk_counter += 1
                    try:
                        keywords = extract_keywords(answer)
                        entities = extract_entities(answer)
                        embedding = embed_text(answer)

                        metadata = {
                            "source": source,
                            "chunk_id": chunk_counter,
                            "type": "answer",
                            "keywords": ", ".join(keywords),
                            "entities": ", ".join([f"{text} ({label})" for text, label in entities])
                        }

                        chunk_id = f"{source}_a_{chunk_counter}_{uuid.uuid4().hex[:8]}"
                        processed_chunks.append({
                            "id": chunk_id,
                            "text": answer,
                            "embedding": embedding,
                            "metadata": metadata
                        })
                    except Exception as e:
                        print(
                            f"Error processing answer chunk {chunk_counter}: {e}")
            else:
                # Process other dictionary structures
                for key, val in value.items():
                    process_json_value(
                        val, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            # Process list items
            for i, item in enumerate(value):
                process_json_value(
                    item, f"{prefix}[{i}]" if prefix else f"[{i}]")
        elif isinstance(value, str) and len(value.strip()) > 10:
            # Process string values as text chunks
            text = clean_text(value.strip())
            if len(text) > 10:
                chunk_counter += 1
                try:
                    keywords = extract_keywords(text)
                    entities = extract_entities(text)
                    embedding = embed_text(text)

                    metadata = {
                        "source": source,
                        "chunk_id": chunk_counter,
                        "type": "text",
                        "field": prefix,
                        "keywords": ", ".join(keywords),
                        "entities": ", ".join([f"{text} ({label})" for text, label in entities])
                    }

                    chunk_id = f"{source}_t_{chunk_counter}_{uuid.uuid4().hex[:8]}"
                    processed_chunks.append({
                        "id": chunk_id,
                        "text": text,
                        "embedding": embedding,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"Error processing text chunk {chunk_counter}: {e}")

    # Start processing the JSON data
    process_json_value(json_data)

    return processed_chunks
    
    
    
# Below is for testing, you can remove this when actually implementing
    
    
if __name__ == "__main__":
    # Folder with new files to be processed
    new_documents_directory = "newDocumentsFolder"

    # Folder to archive files after processing
    documents_directory = "documentsFolder"

    # Ensure the folders exist
    os.makedirs(new_documents_directory, exist_ok=True)
    os.makedirs(documents_directory, exist_ok=True)

    # Ask the user for which Chroma collection to insert into
    collection = input("Enter ChromaDB collection name: ").strip().lower()

    # Run processing
    process_documents_for_collection(new_documents_directory, documents_directory, collection)