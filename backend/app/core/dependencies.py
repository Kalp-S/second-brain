from typing import Optional
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.sparse import SparseBM25Retriever
from backend.app.services.retrieval.reranker import CrossEncoderReranker
from backend.app.services.retrieval.pipeline import AdvancedRAGRetriever
from backend.app.services.ingestion.pipeline import IngestionPipeline
from backend.app.services.llm.provider import get_llm_provider, BaseLLMProvider

_dense_retriever: Optional[DenseRetriever] = None
_sparse_retriever: Optional[SparseBM25Retriever] = None
_reranker: Optional[CrossEncoderReranker] = None
_rag_retriever: Optional[AdvancedRAGRetriever] = None
_ingestion_pipeline: Optional[IngestionPipeline] = None
_llm_provider: Optional[BaseLLMProvider] = None

def get_dense_retriever() -> DenseRetriever:
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = DenseRetriever()
    return _dense_retriever

def get_sparse_retriever() -> SparseBM25Retriever:
    global _sparse_retriever
    if _sparse_retriever is None:
        _sparse_retriever = SparseBM25Retriever()
    return _sparse_retriever

def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker

def get_rag_retriever() -> AdvancedRAGRetriever:
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = AdvancedRAGRetriever(
            dense_retriever=get_dense_retriever(),
            sparse_retriever=get_sparse_retriever(),
            reranker=get_reranker()
        )
    return _rag_retriever

def get_ingestion_pipeline() -> IngestionPipeline:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(
            dense_retriever=get_dense_retriever(),
            sparse_retriever=get_sparse_retriever()
        )
    return _ingestion_pipeline

def get_llm() -> BaseLLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider
