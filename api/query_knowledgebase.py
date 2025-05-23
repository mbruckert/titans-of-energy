import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # matches your previous code
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient('./chroma_db')


def query_collection(collection_name, query, n_results=2):
    collection = chroma_client.get_or_create_collection(name=collection_name)

    query_embedding = openai_ef([query])
    print(f"Query embedding vector length: {len(query_embedding[0])}")

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'metadatas']
    )

    # Format the results into a context string
    context_parts = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            context_parts.append(f"Document {i+1}: {doc}\n")
            if metadata:
                context_parts.append(f"Metadata: {json.dumps(metadata)}\n")

    return "\n".join(context_parts) if context_parts else "No relevant context found."
