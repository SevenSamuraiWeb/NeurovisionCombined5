import chromadb

from config import settings

client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))

collection = client.get_or_create_collection(
    name="image_embeddings",
    metadata={"hnsw:space": "cosine"}
)

def get_collection():
    return collection
