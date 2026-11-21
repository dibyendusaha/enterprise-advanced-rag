import pickle
import re
from typing import Sequence
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from src.config import (
    CHROMA_DIR,
    BM25_INDEX_PATH,
    EMBEDDING_MODEL,
    RERANKER_LLM_MODEL,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    HYBRID_WEIGHTS,
    RERANK_TOP_N,
    MIN_RELEVANCE_SCORE,
)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())


class NativeBM25Retriever(BaseRetriever):
    """Standalone LangChain BaseRetriever backed directly by rank_bm25.BM25Okapi."""
    documents: list[Document]
    k: int = SPARSE_TOP_K
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, __context) -> None:
        tokenized_corpus = [_tokenize(doc.page_content) for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )
        return [
            self.documents[idx]
            for idx in ranked_indices[: self.k]
            if scores[idx] > 0.0
        ]


class ChunkRelevanceScore(BaseModel):
    chunk_index: int = Field(description="The 0-based index of the candidate chunk.")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (completely irrelevant) to 1.0 (directly answers the query)."
    )


class BatchRerankOutput(BaseModel):
    scores: list[ChunkRelevanceScore]


class OpenAIReranker(BaseDocumentCompressor):
    """100% OpenAI-based listwise/pointwise document reranker using Structured Outputs."""
    model_name: str = RERANKER_LLM_MODEL
    top_n: int = RERANK_TOP_N
    min_score: float = MIN_RELEVANCE_SCORE

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        llm = ChatOpenAI(model=self.model_name, temperature=0.0).with_structured_output(
            BatchRerankOutput,
            method="json_schema",
            strict=True,
        )

        formatted_candidates = []
        for idx, doc in enumerate(documents):
            src = doc.metadata.get("source", "Unknown")
            formatted_candidates.append(f"[Index {idx} | Source: {src}]\n{doc.page_content}")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precision search reranker for an enterprise policy assistant. "
             "If the user's query asks multiple sub-questions spanning different policies, score a passage "
             "high (0.75 - 1.0) if it directly answers ANY sub-question in the query. "
             "Score 0.0 if the passage is unrelated. Return a score for every candidate index."),
            ("human", "QUERY: {query}\n\nCANDIDATE PASSAGES:\n{candidates}")
        ])

        raw_result = (prompt | llm).invoke({
            "query": query,
            "candidates": "\n\n---\n\n".join(formatted_candidates),
        })

        result = (
            raw_result
            if isinstance(raw_result, BatchRerankOutput)
            else BatchRerankOutput.model_validate(raw_result)
        )

        score_map = {item.chunk_index: item.relevance_score for item in result.scores}

        scored_docs = []
        for idx, doc in enumerate(documents):
            score = score_map.get(idx, 0.0)
            if score >= self.min_score:
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": round(score, 3)},
                )
                scored_docs.append((score, doc_copy))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[: self.top_n]]


def _weighted_reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    rrf_k: int = 60,
) -> list[Document]:
    """
    Combines ranked lists from Sparse (BM25) and Dense (Chroma) retrievers
    using Weighted Reciprocal Rank Fusion (RRF), deduplicating by `chunk_id`.
    """
    rrf_scores: dict[str, float] = {}
    doc_registry: dict[str, Document] = {}

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            doc_key = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source')}::{hash(doc.page_content)}"
            doc_registry[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + weight * (1.0 / (rrf_k + rank))

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    return [doc_registry[k] for k in sorted_keys]


class EnterpriseHybridRerankRetriever:
    def __init__(self):
        if not CHROMA_DIR.exists() or not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                "Vector store or BM25 index missing. Run `python -m src.ingest` first."
            )

        # 1. Dense Retriever (Standalone langchain-chroma + langchain-openai)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="enterprise_policies",
        )
        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": DENSE_TOP_K},
        )

        # 2. Sparse Retriever (Standalone NativeBM25Retriever over rank_bm25)
        with open(BM25_INDEX_PATH, "rb") as f:
            chunks = pickle.load(f)

        self.bm25_retriever = NativeBM25Retriever(
            documents=chunks,
            k=SPARSE_TOP_K,
        )

        # 3. OpenAI Structured-Output Reranker
        self.reranker = OpenAIReranker(
            model_name=RERANKER_LLM_MODEL,
            top_n=RERANK_TOP_N,
            min_score=MIN_RELEVANCE_SCORE,
        )

    def retrieve_with_diagnostics(self, query: str) -> dict:
        """
        Executes Sparse (BM25) + Dense (Chroma) -> RRF Hybrid Fusion -> OpenAI Reranking
        """
        sparse_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)

        hybrid_candidates = _weighted_reciprocal_rank_fusion(
            ranked_lists=[sparse_docs, dense_docs],
            weights=HYBRID_WEIGHTS,
        )

        reranked_docs = list(self.reranker.compress_documents(hybrid_candidates, query=query))

        return {
            "hybrid_candidate_count": len(hybrid_candidates),
            "hybrid_candidates": hybrid_candidates,
            "reranked_docs": reranked_docs,
        }