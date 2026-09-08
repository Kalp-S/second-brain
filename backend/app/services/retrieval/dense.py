import os
from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from backend.app.core.config import settings

COLLECTION_NAME = "second_brain_chunks"

class DenseRetriever:
    """Manages local ONNX embeddings and Qdrant vector search."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or str(settings.QDRANT_PATH)
        self.embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        self.client = QdrantClient(path=self.storage_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE
                )
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.embedding_model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        embeddings = list(self.embedding_model.embed([f"query: {query}"]))
        return embeddings[0].tolist()

    def index_chunks(self, points: List[Dict[str, Any]]) -> None:
        """
        Points is a list of dicts:
        {
          "id": str,
          "vector": List[float],
          "payload": {
             "child_id": str,
             "parent_id": str,
             "document_id": str,
             "doc_title": str,
             "header_path": str,
             "content": str
          }
        }
        """
        if not points:
            return

        qdrant_points = [
            qmodels.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"]
            )
            for p in points
        ]
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=qdrant_points,
            wait=True
        )

    def delete_by_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        query_vector = self.embed_query(query)
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points

        results = []
        for rank, hit in enumerate(hits, start=1):
            results.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "rank": rank,
                "payload": hit.payload or {}
            })
        return results
