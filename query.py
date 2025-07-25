
# Import ChromaDB for vector database operations
import chromadb
from chromadb.config import Settings

# Default number of results to return
TOP_K_DEFAULT = 5
# Directory where ChromaDB vector store is stored
DB_DIR = "chroma_store"
# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings())
collection = client.get_or_create_collection("helpdesk_knowledge")

# Query the helpdesk knowledge base for relevant document chunks
def query_helpdesk(user_input: str, top_k: int = TOP_K_DEFAULT, category: str | None = None):
    """
    Query the ChromaDB vector store for the most relevant document chunks.
    Args:
        user_input (str): The user's helpdesk question.
        top_k (int): Number of top results to return.
        category (str|None): Optional category filter.
    Returns:
        list[tuple[str, dict]]: List of (document chunk, metadata) tuples.
    """
    # Build filter for category if provided
    where = {"category": {"$eq": category}} if category else None
    try:
    # Query the vector store
        results = collection.query(
            query_texts=[user_input],
            n_results=top_k,
            where=where  # None means no filter
        )

        # Extract document chunks and metadata
        docs  = results["documents"][0]   # list[str]
        metas = results["metadatas"][0]   # list[dict]
        return list(zip(docs, metas))

    except Exception as e:
        # Log or handle errors gracefully
        print(f"Error querying helpdesk knowledge base: {e}")
        return []