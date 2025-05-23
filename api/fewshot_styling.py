import json
import chromadb
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv
import os
import platform

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."

# Initialize OpenAI client
openai = OpenAI(api_key=OPENAI_API_KEY)

# Preload the GGUF model
# Only supporting GGUF format for optimal performance
model = None

# Detect macOS and Apple Silicon for optimizations
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"

try:
    from llama_cpp import Llama

    # Load the specific GGUF model
    model_path = "./models/gemma-3-4b-it-q4_0.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GGUF model file not found at {model_path}\n"
            "Please download it first using:\n"
            "mkdir -p models\n"
            "huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-qat-q4_0.gguf --local-dir ./models"
        )

    print(f"Loading GGUF model: {model_path}")

    # Optimize for macOS/Apple Silicon
    if IS_APPLE_SILICON:
        print("Detected Apple Silicon - enabling Metal GPU acceleration")
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # Use Metal GPU on Apple Silicon
            verbose=False,
            use_mlock=True,  # Keep model in memory
            use_mmap=True,   # Memory mapping for efficiency
            n_threads=None   # Auto-detect optimal thread count
        )
    else:
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,  # CPU only for non-Apple Silicon
            verbose=False,
            use_mlock=True,
            use_mmap=True,
            n_threads=None
        )

    print("GGUF model loaded successfully")

except ImportError:
    raise ImportError(
        "llama-cpp-python is required for GGUF model support.\n"
        "Install it with: pip install llama-cpp-python\n"
        "For Apple Silicon Macs: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
    )
except Exception as e:
    raise Exception(f"Failed to load GGUF model: {e}")

# ----------- Data Loading -----------


def load_data_files(qa_file):
    """Load QA data from JSON file."""
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)
    return qa_data

# ----------- Main Generation Function -----------


def generate_embeddings(qa_file: str, collection_name: str = "oppenheimer-qa") -> None:
    """
    Generate and store embeddings for QA examples in ChromaDB.

    Args:
        qa_file (str): Path to the JSON file containing QA examples
        collection_name (str): Name of the ChromaDB collection to use
    """
    # Load QA data
    qa_data = load_data_files(qa_file)

    # Setup ChromaDB with persistence
    client = PersistentClient(path="chroma_db")

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection might not exist, which is fine

    # Create new collection
    coll = client.create_collection(collection_name)

    # Generate embeddings using OpenAI
    docs = [item['response'] for item in qa_data]
    qns = [item['question'] for item in qa_data]
    ids = [f"qa_{i}" for i in range(len(docs))]

    # Generate embeddings for all documents
    embeddings = []
    for doc in docs:
        response = openai.embeddings.create(
            input=[doc],
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)

    # Add documents with embeddings
    coll.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=[{"question": q} for q in qns],
        ids=ids
    )


def generate_styled_text(
    question: str,
    context: str,
    collection_name: str = "oppenheimer-qa",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    num_examples: int = 3
) -> str:
    """
    Generate styled text using few-shot learning approach with GGUF model.

    Args:
        question (str): The question to generate a response for
        context (str): Additional context to help answer the question
        collection_name (str): Name of the ChromaDB collection to use
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Sampling temperature
        num_examples (int): Number of examples to use for few-shot learning

    Returns:
        str: Generated styled text
    """
    # Setup ChromaDB with persistence
    client = PersistentClient(path="chroma_db")
    try:
        coll = client.get_collection(collection_name)
    except Exception:
        raise Exception(
            f"Collection {collection_name} not found. Please run generate_embeddings first.")

    # Generate embedding for the question
    response = openai.embeddings.create(
        input=[question],
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding

    # Get relevant examples using the embedding
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=num_examples
    )

    # Handle case where no results are found
    if not res['ids'] or not res['ids'][0]:
        examples = []
    else:
        examples = []
        for i in range(len(res['ids'][0])):
            examples.append({
                'question': res['metadatas'][0][i]['question'],
                'response': res['documents'][0][i]
            })

    # Build prompt with examples
    context_str = ''.join(
        [f"Q: {e['question']}\nA: {e['response']}\n\n" for e in examples])
    prompt = (
        "System: For the purposes of this interaction, you are J Robert Oppenheimer. Answering questions about your life and work to a general audience in a museum setting."
        "Answer in a paragraph at most if required by the question, but tend towards a shorter, more conversational style.\n\n"
        "Here are some examples of your style:\n\n"
        f"{context_str}\n\n"
        f"Here is some additional context to help you answer the question:\n\n"
        f"{context}\n\n"
        f"User: {question}\nAssistant: "
    )

    print(prompt)

    # Generate response using GGUF model
    response = model(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop=["\nUser:", "\nAssistant:", "<STOP>", "\nSystem:"],
        echo=False
    )
    gen = response['choices'][0]['text'].strip()

    # Trim at stop tokens
    stop_tokens = ["\nUser:", "\nAssistant:", "<STOP>",
                   "\nSystem:", "User:", "Assistant:", "System:"]
    for tok in stop_tokens:
        if tok in gen:
            gen = gen.split(tok)[0]

    return gen.strip()
