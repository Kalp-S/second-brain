from backend.app.services.ingestion.pipeline import IngestionPipeline
from backend.app.services.llm.provider import BaseLLMProvider, get_llm_provider
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.pipeline import AdvancedRAGRetriever
from backend.app.services.retrieval.reranker import CrossEncoderReranker
from backend.app.services.retrieval.sparse import SparseBM25Retriever

_dense_retriever: DenseRetriever | None = None
_sparse_retriever: SparseBM25Retriever | None = None
_reranker: CrossEncoderReranker | None = None
_rag_retriever: AdvancedRAGRetriever | None = None
_ingestion_pipeline: IngestionPipeline | None = None
_llm_provider: BaseLLMProvider | None = None


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
            reranker=get_reranker(),
        )
    return _rag_retriever


def get_ingestion_pipeline() -> IngestionPipeline:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(
            dense_retriever=get_dense_retriever(), sparse_retriever=get_sparse_retriever()
        )
    return _ingestion_pipeline


def get_llm() -> BaseLLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def reset_dependencies() -> None:
    """Reset singletons, primarily used in isolated unit and integration tests."""
    global \
        _dense_retriever, \
        _sparse_retriever, \
        _reranker, \
        _rag_retriever, \
        _ingestion_pipeline, \
        _llm_provider
    _dense_retriever = None
    _sparse_retriever = None
    _reranker = None
    _rag_retriever = None
    _ingestion_pipeline = None
    _llm_provider = None
