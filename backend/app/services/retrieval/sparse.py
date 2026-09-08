import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

class SparseBM25Retriever:
    """
    BM25 Okapi lexical retriever.
    Complements dense semantic vector search by capturing exact keywords,
    acronyms, symbol names, and rare technical terminology.
    """

    def __init__(self):
        self.corpus: List[List[str]] = []
        self.chunk_records: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric characters while keeping words/identifiers
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        return tokens

    def index_chunks(self, records: List[Dict[str, Any]]) -> None:
        """
        records is a list of dicts:
        {
            "id": str,
            "child_id": str,
            "parent_id": str,
            "document_id": str,
            "doc_title": str,
            "header_path": str,
            "content": str
        }
        """
        for r in records:
            # Remove any existing record with the same child_id to prevent duplicates
            self.remove_by_id(r.get("child_id") or r.get("id"))
            tokens = self.tokenize(f"{r.get('doc_title', '')} {r.get('header_path', '')} {r.get('content', '')}")
            self.corpus.append(tokens)
            self.chunk_records.append(r)

        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    def remove_by_id(self, chunk_id: str) -> None:
        indices_to_remove = [
            i for i, r in enumerate(self.chunk_records)
            if (r.get("child_id") == chunk_id or r.get("id") == chunk_id)
        ]
        if indices_to_remove:
            for idx in reversed(indices_to_remove):
                self.chunk_records.pop(idx)
                self.corpus.pop(idx)
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
            else:
                self.bm25 = None

    def remove_by_document(self, document_id: str) -> None:
        indices_to_remove = [
            i for i, r in enumerate(self.chunk_records)
            if r.get("document_id") == document_id
        ]
        if indices_to_remove:
            for idx in reversed(indices_to_remove):
                self.chunk_records.pop(idx)
                self.corpus.pop(idx)
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
            else:
                self.bm25 = None

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Sort candidate indices by score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        rank = 1
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            results.append({
                "id": self.chunk_records[idx].get("child_id") or self.chunk_records[idx].get("id"),
                "score": score,
                "rank": rank,
                "payload": self.chunk_records[idx]
            })
            rank += 1
            if rank > top_k:
                break

        return results
