# --- Environment Setup ---

import os
import re
import json
import shutil
import warnings

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Please set the OPENAI_API_KEY in your .env file."

# Below is used only if there is a problem with the GPU, uncomment in only that situation
# Disable GPU (force CPU usage only)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress specific spaCY warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*falling back to type probe function.*")

import spacy
import uuid
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

from openai import OpenAI
from chromadb import PersistentClient
from chromadb.config import Settings


# --- Initialize Models and Clients ---

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Load the KeyBERT model for keyword extraction
kw_model = KeyBERT()

# Initialize OpenAI client
openai = OpenAI(api_key=OPENAI_API_KEY)

# Initialize a persistent ChromaDB client (local storage at ./chroma_db)
client = PersistentClient(path="chroma_db")

# Step 1: Clean the input text by removing page numbers, footnotes, and excessive whitespace.
def clean_text(text):
    text = re.sub(r"\[\d+\]|\(Page \d+\)", "", text)      # Remove page markers like [1] or (Page 2)
    text = re.sub(r"\^\d+|Footnote \d+:", "", text)       # Remove footnotes
    return re.sub(r"\s+", " ", text).strip()              # Normalize whitespace

# Step 2: Split the text into individual sentences using spaCy.
def preprocess_text(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Step 3: Extract keywords using both KeyBERT and TF-IDF, then deduplicate.
def extract_keywords(text, max_keywords=5):
    keybert_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english"
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    sorted_indices = tfidf_scores.argsort()[::-1]
    tfidf_keywords = [feature_names[i] for i in sorted_indices[:max_keywords]]

    # Combine
    combined_keywords = list(set([kw[0] for kw in keybert_keywords] + tfidf_keywords))
    return combined_keywords[:max_keywords]

# Step 4: Extract named entities (e.g., people, places, dates) using spaCy.
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Step 5: Generate an embedding for the text using OpenAI's embedding model.
def embed_text(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Step 6: Process one document into chunks with metadata and embeddings.
def process_document(file_path, source):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = clean_text(f.read())

    chunks = preprocess_text(raw_text)
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        keywords = extract_keywords(chunk)
        entities = extract_entities(chunk)
        embedding = embed_text(chunk)

        # ChromaDb needs metadata to be strings so we convert them here
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

    return processed_chunks

# Step 7: Insert a list of processed text chunks into ChromaDB.
def insert_into_chroma(collection_name, chunk_data):
    # Create or retrieve the specified collection
    collection = client.get_or_create_collection(collection_name)

    # Add to ChromaDB
    collection.add(
        documents=[chunk["text"] for chunk in chunk_data],
        embeddings=[chunk["embedding"] for chunk in chunk_data],
        metadatas=[chunk["metadata"] for chunk in chunk_data],
        ids=[chunk["id"] for chunk in chunk_data]
    )

    print(f"Inserted {len(chunk_data)} chunks into ChromaDB collection '{collection_name}'.")


# Step 8: Move processed .txt files from source to documentsFolder.
def move_processed_files(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            shutil.move(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
            print(f"Moved {filename} to {target_dir}")


# ------Actual Processing------

# Process all new text files in the folder and add them to the ChromaDB collection.
def process_documents_for_collection(new_docs_dir, archive_dir, collection_name):
    for filename in os.listdir(new_docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(new_docs_dir, filename)
            print(f"Processing {filename}...")
            try:
                chunk_data = process_document(file_path, filename)
                insert_into_chroma(collection_name, chunk_data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    move_processed_files(new_docs_dir, archive_dir)

if __name__ == "__main__":
    # Folder with new files to be processed
    new_documents_directory = "newDocumentsFolder"

    # Folder to archive files after processing
    documents_directory = "documentsFolder"

    # Ask the user for which Chroma collection to insert into
    collection = input("Enter ChromaDB collection name: ").strip().lower()

    # Run processing
    process_documents_for_collection(new_documents_directory, documents_directory, collection)
