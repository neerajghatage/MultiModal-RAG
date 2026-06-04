"""
Cross-encoder reranker for RAG retrieval results.

Uses fastembed's reranking models to reorder candidates by true
query-document relevance after initial hybrid retrieval.

This replaces the simple cosine-score ordering with a cross-encoder
that jointly attends to query + document for more accurate relevance.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.

    Uses fastembed's TextCrossEncoder with BAAI/bge-reranker-v2-m3
    (or similar) for high-quality relevance scoring.
    """

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: FastEmbed-compatible reranker model.
                        Options:
                        - "Xenova/ms-marco-MiniLM-L-6-v2" (fast, good quality)
                        - "BAAI/bge-reranker-v2-m3" (best quality, slower)
                        - "jinaai/jina-reranker-v1-turbo-en" (fast)
        """
        self.model_name = model_name
        self._reranker = None
        self._init_reranker()

    def _init_reranker(self):
        """Initialize the cross-encoder model (pre-cached at Docker build time)."""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._reranker = TextCrossEncoder(model_name=self.model_name)
            logger.info(f"Cross-encoder reranker initialized: {self.model_name}")
        except ImportError:
            logger.warning(
                "fastembed not installed — reranking disabled. "
                "Install with: pip install fastembed"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the reranker is ready to use."""
        return self._reranker is not None

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder relevance scoring.

        Args:
            query: The user query
            results: List of retrieved results (must have 'content' field)
            top_k: Number of results to return after reranking
            score_threshold: Minimum reranker score to include

        Returns:
            Reranked results (top_k), with 'rerank_score' field added
        """
        if not self._reranker or not results:
            return results[:top_k]

        # Build query-document pairs for the cross-encoder
        documents = []
        for r in results:
            # Use the best available text representation
            doc_text = (
                r.get("summary_medium")
                or r.get("summary_short")
                or r.get("content", "")
            )
            # Truncate very long documents (cross-encoders have token limits)
            if len(doc_text) > 1500:
                doc_text = doc_text[:1500]
            documents.append(doc_text)

        # Score all query-document pairs
        try:
            scores = list(self._reranker.rerank(query, documents))
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return results[:top_k]

        # Attach scores and sort
        scored_results = []
        for result, score_entry in zip(results, scores):
            # fastembed rerank returns objects with .score attribute or float
            if hasattr(score_entry, "score"):
                score = float(score_entry.score)
            else:
                score = float(score_entry)

            if score >= score_threshold:
                result_copy = dict(result)
                result_copy["rerank_score"] = score
                scored_results.append(result_copy)

        # Sort by rerank score descending
        scored_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        logger.debug(
            f"Reranked {len(results)} → {min(top_k, len(scored_results))} results. "
            f"Top score: {scored_results[0]['rerank_score']:.3f}" if scored_results else ""
        )

        return scored_results[:top_k]
