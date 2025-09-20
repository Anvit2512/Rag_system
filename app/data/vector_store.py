import chromadb
from chromadb.utils import embedding_functions
import os


class VectorStore:
    def __init__(self, vdb_dir: str, embed_model_id: str, hf_token: str):
        os.makedirs(vdb_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=vdb_dir)
        # Use HF Inference for embeddings (server-side). Chroma integrates sentence-transformers locally;
        # we'll inject embeddings manually when ingesting, so collection can be vanilla.
        self.collection = self.client.get_or_create_collection(name="manuals")
        self.embed_model_id = embed_model_id
        self.hf_token = hf_token

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]]):
        self.collection.add(documents=texts, ids=ids, embeddings=embeddings)

    def query(self, query_embedding: list[float], n_results: int = 6):
        res = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return res # {ids, distances, documents}