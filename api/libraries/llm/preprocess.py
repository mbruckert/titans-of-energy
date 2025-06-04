"""
LLM preprocessing module for model downloading and style database generation.

This module handles downloading models and generating style embeddings for characters.
"""

import json
import os
from typing import Dict, List, Any

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = PersistentClient(path=CHROMA_DB_PATH)


def _authenticate_huggingface():
    """
    Authenticate with Hugging Face using API token if available.

    Returns:
        bool: True if authentication successful or not needed, False if failed
    """
    if not HUGGINGFACE_API_KEY:
        print("No Hugging Face API key found. Public models will still work.")
        return True

    try:
        from huggingface_hub import login
        login(token=HUGGINGFACE_API_KEY, add_to_git_credential=True)
        print("Successfully authenticated with Hugging Face")
        return True
    except ImportError:
        print("huggingface_hub not available for authentication")
        return False
    except Exception as e:
        print(f"Failed to authenticate with Hugging Face: {e}")
        return False


def download_model(model_name: str, model_type: str) -> str:
    """
    Download a model to the models folder if it's a Hugging Face model.

    Args:
        model_name: Name/path of the model. Supports:
                   - HF repo: "repo/model" or "repo/model:filename.gguf"
                   - Direct URL: "https://huggingface.co/repo/model/blob/main/file.gguf"
        model_type: Type of model ('huggingface', 'openai')

    Returns:
        Local path to the model or original model_name for OpenAI models

    Raises:
        ValueError: If model_type is not supported
        Exception: If download fails
    """
    if model_type.lower() == 'openai':
        # No download needed for OpenAI models
        print(f"OpenAI model specified: {model_name} - no download required")
        return model_name

    if model_type.lower() != 'huggingface':
        raise ValueError(
            f"Unsupported model type: {model_type}. Use 'huggingface' or 'openai'")

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Check if it's a direct URL
    if model_name.startswith('http'):
        return _download_from_url(model_name)

    # Parse model name - check if specific file is requested
    if ':' in model_name:
        repo_id, filename = model_name.split(':', 1)
        model_path = os.path.join(
            MODELS_DIR, f"{repo_id.replace('/', '_')}_{filename}")
        is_specific_file = True
    else:
        repo_id = model_name
        filename = None
        model_path = os.path.join(MODELS_DIR, model_name.replace('/', '_'))
        is_specific_file = False

    # Check if model already exists locally
    if os.path.exists(model_path):
        if is_specific_file or (os.path.isdir(model_path) and os.listdir(model_path)):
            print(f"Model already exists: {model_path}")
            return model_path

    print(f"Downloading Hugging Face model: {model_name}")

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading Hugging Face models.\n"
            "Install with: pip install huggingface_hub"
        )

    # Authenticate with Hugging Face before downloading
    auth_success = _authenticate_huggingface()
    if not auth_success:
        print("Warning: Hugging Face authentication failed. Proceeding with public access only.")

    try:
        if is_specific_file:
            # Download specific file (likely GGUF)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=MODELS_DIR
            )
            print(f"Downloaded file to: {downloaded_path}")
            return downloaded_path
        else:
            # Download entire repository (regular model)
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=model_path
            )
            print(f"Downloaded model to: {downloaded_path}")
            return downloaded_path

    except Exception as e:
        raise Exception(f"Failed to download model {model_name}: {e}")


def _download_from_url(url: str) -> str:
    """
    Download a GGUF file from a direct Hugging Face URL.

    Args:
        url: Direct URL to the GGUF file

    Returns:
        Local path to the downloaded file
    """
    import re
    import requests
    from urllib.parse import urlparse

    # Parse Hugging Face URL to extract repo and filename
    # Example: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/blob/main/gemma-3-4b-it-q4_0.gguf
    hf_pattern = r'https://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+)'
    match = re.match(hf_pattern, url)

    if not match:
        raise ValueError(f"Invalid Hugging Face URL format: {url}")

    repo_id = match.group(1)
    branch = match.group(2)  # Usually 'main'
    filename = match.group(3)

    print(f"Parsed URL:")
    print(f"  Repo ID: {repo_id}")
    print(f"  Branch: {branch}")
    print(f"  Filename: {filename}")

    # Create local filename
    safe_repo_name = repo_id.replace('/', '_')
    local_filename = f"{safe_repo_name}_{filename}"
    local_path = os.path.join(MODELS_DIR, local_filename)

    # Check if already downloaded
    if os.path.exists(local_path):
        print(f"Model already exists: {local_path}")
        return local_path

    print(f"Downloading from Hugging Face...")
    print(f"Saving to: {local_path}")

    try:
        # Use huggingface_hub for better reliability and progress tracking
        from huggingface_hub import hf_hub_download

        print(
            f"Starting download of {filename} (this may take several minutes for large models)...")

        # Download to a temporary location first
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=branch,
            cache_dir=None,  # Use default cache
            local_files_only=False
        )

        # Copy to our preferred location with our naming convention
        import shutil
        if os.path.exists(local_path):
            os.remove(local_path)
        shutil.copy2(downloaded_path, local_path)

        print(f"✅ Downloaded successfully to: {local_path}")
        return local_path

    except Exception as e:
        print(f"HF download failed: {e}")
        print("Trying direct HTTP download as fallback...")

        # Convert blob URL to direct download URL for fallback
        if '/blob/' in url:
            download_url = url.replace('/blob/', '/resolve/')
        else:
            download_url = url

        # Fallback to direct HTTP download
        try:
            print(f"Downloading from: {download_url}")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            print(f"File size: {total_size / (1024*1024*1024):.2f} GB")

            with open(local_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024*1024)
                            mb_total = total_size / (1024*1024)
                            print(
                                f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

            print(f"\n✅ Downloaded successfully to: {local_path}")
            return local_path

        except Exception as e2:
            raise Exception(
                f"Both HF and direct download failed. HF error: {e}, Direct error: {e2}")


def generate_style_db(character_name: str, path_to_style_data: str) -> str:
    """
    Generate embeddings for character style data and store in ChromaDB.

    Args:
        character_name: Name of the character
        path_to_style_data: Path to JSON file containing style examples

    Returns:
        Collection name used for the style database

    Raises:
        FileNotFoundError: If style data file doesn't exist
        Exception: If database generation fails
    """
    if not os.path.exists(path_to_style_data):
        raise FileNotFoundError(
            f"Style data file not found: {path_to_style_data}")

    # Load style data
    try:
        with open(path_to_style_data, 'r', encoding='utf-8') as f:
            style_data = json.load(f)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON in style data file: {e}")

    if not isinstance(style_data, list):
        raise ValueError("Style data must be a list of question-answer pairs")

    # Create collection name
    collection_name = f"{character_name.lower().replace(' ', '')}-style"

    print(f"Generating style database for {character_name}...")
    print(f"Collection name: {collection_name}")

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass  # Collection might not exist

    # Create new collection
    collection = chroma_client.create_collection(collection_name)

    # Prepare data for embedding
    documents = []
    questions = []
    ids = []

    for i, item in enumerate(style_data):
        if not isinstance(item, dict) or 'question' not in item or 'response' not in item:
            print(
                f"Skipping invalid style item {i}: missing 'question' or 'response'")
            continue

        # Create combined text for better similarity matching
        combined_text = f"Q: {item['question']}\nA: {item['response']}"
        documents.append(combined_text)
        questions.append(item['question'])
        ids.append(f"{character_name.lower().replace(' ', '_')}_style_{i}")

    if not documents:
        raise ValueError("No valid style examples found in the data")

    print(f"Generating embeddings for {len(documents)} style examples...")

    # Generate embeddings
    embeddings = []
    batch_size = 10

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_embeddings = []

        for doc in batch_docs:
            try:
                response = openai_client.embeddings.create(
                    input=[doc],
                    model=EMBEDDING_MODEL
                )
                batch_embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Failed to generate embedding for document {i}: {e}")
                # Use zero vector as fallback
                batch_embeddings.append([0.0] * 1536)  # Ada-002 dimension

        embeddings.extend(batch_embeddings)
        print(
            f"Generated embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

    # Add documents with embeddings to collection
    try:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=[{
                "question": q,
                "character": character_name,
                "type": "style_example"
            } for q in questions],
            ids=ids
        )

        print(
            f"Successfully stored {len(documents)} style examples for {character_name}")
        return collection_name

    except Exception as e:
        raise Exception(f"Failed to store style data in ChromaDB: {e}")
