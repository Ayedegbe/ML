import chromadb
from chromadb.config import Settings                   
TOP_K_DEFAULT = 5
DB_DIR = "chroma_store"
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings())
collection = client.get_or_create_collection("helpdesk_knowledge")

def query_helpdesk(user_input: str, top_k: int = TOP_K_DEFAULT, category: str | None = None):
   
    where = {"category": {"$eq": category}} if category else None

    results = collection.query(
        query_texts=[user_input],
        n_results=top_k,
        where=where            # None means no filter
    )

    docs  = results["documents"][0]   # list[str]
    metas = results["metadatas"][0]   # list[dict]
    return list(zip(docs, metas))

