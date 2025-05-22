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
    
    print(f"Query results keys: {results.keys()}")
    print(f"Number of documents returned: {len(results['documents'][0])}")
    
    return results


if __name__ == "__main__":
    # Prompt user for collection name exactly like your previous script
    collection_name = input("Enter ChromaDB collection name: ").strip().lower()
    print(f"Using collection: '{collection_name}'")

    
    print("Enter your query (type 'exit' or 'quit' to stop):")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["quit", "exit"]:
            break
        if not user_query:
            continue  # skip empty queries

        results = query_collection(collection_name, user_query)

        print("Response:")
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            print(f"\nResult #{i}:")
            print(f"Document:\n{doc}")
            print(f"Metadata:\n{json.dumps(meta, indent=2)}")
            print("-" * 80)
