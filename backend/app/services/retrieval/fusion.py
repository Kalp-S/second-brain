from typing import Any


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) to merge candidate rankings
    from heterogeneous retrieval systems (Dense Vector + Sparse Lexical).

    Formula: RRF_Score(d) = sum_m [ w_m / (k + rank_m(d)) ]
    where k is a smoothing constant (typically 60) that prevents top ranks
    from disproportionately dominating.
    """

    def __init__(self, k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self, dense_hits: list[dict[str, Any]], sparse_hits: list[dict[str, Any]], top_k: int = 20
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}
        provenance: dict[str, dict[str, Any]] = {}

        # Process Dense hits
        for item in dense_hits:
            item_id = str(item["id"])
            rank = item["rank"]
            rrf_contrib = self.dense_weight / (self.k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_contrib

            if item_id not in metadata:
                metadata[item_id] = item.get("payload", {})
            if item_id not in provenance:
                provenance[item_id] = {}
            provenance[item_id]["dense_rank"] = rank
            provenance[item_id]["dense_score"] = item.get("score")

        # Process Sparse BM25 hits
        for item in sparse_hits:
            item_id = str(item["id"])
            rank = item["rank"]
            rrf_contrib = self.sparse_weight / (self.k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_contrib

            if item_id not in metadata:
                metadata[item_id] = item.get("payload", {})
            if item_id not in provenance:
                provenance[item_id] = {}
            provenance[item_id]["sparse_rank"] = rank
            provenance[item_id]["sparse_score"] = item.get("score")

        # Sort by fused RRF score descending
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_results: list[dict[str, Any]] = []
        for rank, (item_id, rrf_score) in enumerate(sorted_items[:top_k], start=1):
            fused_results.append(
                {
                    "id": item_id,
                    "score": round(rrf_score, 6),
                    "rank": rank,
                    "provenance": provenance.get(item_id, {}),
                    "payload": metadata.get(item_id, {}),
                }
            )

        return fused_results
