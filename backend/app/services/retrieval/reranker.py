from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from backend.app.core.config import settings

class CrossEncoderReranker:
    """
    Applies cross-attention reranking over candidate chunks.
    While bi-encoder dense vectors score query and document independently,
    the cross-encoder passes the query and passage together through transformer
    attention layers, scoring deep semantic alignment and filtering false positives.
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = settings.RERANK_TOP_K
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        passages = []
        for c in candidates:
            payload = c.get("payload", {})
            text = f"{payload.get('doc_title', '')} - {payload.get('header_path', '')}\n{payload.get('content', '')}"
            passages.append({
                "id": c["id"],
                "text": text,
                "meta": c
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked_items: List[Dict[str, Any]] = []
        for rank, item in enumerate(results[:top_k], start=1):
            original_meta = item.get("meta", {})
            reranked_items.append({
                "id": str(item["id"]),
                "score": round(float(item["score"]), 4),
                "rank": rank,
                "provenance": original_meta.get("provenance", {}),
                "payload": original_meta.get("payload", {})
            })

        return reranked_items
