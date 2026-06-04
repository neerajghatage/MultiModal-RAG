"""
LangChain-based Vector Store backed by Qdrant.

Features:
  - Hybrid search: Dense (text-embedding-3-large) + Sparse (BM25 via fastembed)
  - Reciprocal Rank Fusion (RRF) for score merging
  - Parent-child (small-to-big) retrieval
  - Relationship-aware retrieval with cross-document linking
  - Batch upsert with rich metadata payloads
"""

from typing import List, Dict, Any, Optional
import logging
import hashlib
import time
import math

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as LCQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    HnswConfigDiff, PayloadSchemaType,
    TextIndexParams, TokenizerType,
    SparseVectorParams, SparseIndexParams,
    SparseVector as QdrantSparseVector,
    Prefetch, Fusion,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── RRF constant ──────────────────────────────────────────────
RRF_K = 60  # Standard RRF constant


class VectorStore:
    """
    Qdrant vector store with hybrid search (dense + sparse).

    Uses:
    - LangChain embeddings for dense vectors
    - FastEmbed BM25 for sparse vectors
    - RRF fusion for hybrid ranking
    - Parent-child chunk expansion for context delivery
    """

    def __init__(self, qdrant_url: str, embeddings, collection_name: str = "confidential_docs",
                 blob_store=None, enable_sparse: bool = True):
        """
        Args:
            qdrant_url:      Qdrant server URL
            embeddings:      A LangChain Embeddings instance (from Config.get_embeddings())
            collection_name: Qdrant collection name
            blob_store:      Optional ImageBlobStore for uploading images
            enable_sparse:   Whether to enable BM25 sparse vectors for hybrid search
        """
        self.qdrant_url = qdrant_url
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.blob_store = blob_store
        self.enable_sparse = enable_sparse

        # Raw client — used for upsert, scroll, health, stats
        self.client = QdrantClient(url=qdrant_url, prefer_grpc=False)

        # Detect dimensions: prefer existing collection config, then model name heuristic
        self.embedding_dimensions = self._detect_dimensions(embeddings)

        # Sparse encoder (BM25 via fastembed)
        self._sparse_encoder = None
        if enable_sparse:
            self._init_sparse_encoder()

        # LangChain wrapper — used for similarity search
        self._lc_store: Optional[LCQdrant] = None
        self._init_lc_store()

        logger.info(f"VectorStore connected to {qdrant_url}, collection={collection_name}, sparse={enable_sparse}")

    def _init_sparse_encoder(self):
        """Initialize BM25 sparse encoder via fastembed."""
        try:
            from fastembed import SparseTextEmbedding
            self._sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("BM25 sparse encoder initialized (Qdrant/bm25)")
        except ImportError:
            logger.warning("fastembed not installed — sparse search disabled. "
                           "Install with: pip install fastembed")
            self.enable_sparse = False
        except Exception as e:
            logger.warning(f"Failed to initialize sparse encoder: {e}")
            self.enable_sparse = False

    def _detect_dimensions(self, embeddings) -> int:
        """Detect embedding dimensions from existing collection or model name."""
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_config = info.config.params.vectors
            if hasattr(vectors_config, "size"):
                return vectors_config.size
        except Exception:
            pass
        # Fallback: infer from model name / deployment name
        model_name = (
            getattr(embeddings, "deployment", "")
            or getattr(embeddings, "azure_deployment", "")
            or getattr(embeddings, "model", "")
        )
        return 3072 if "large" in (model_name or "").lower() else 1536

    def _init_lc_store(self):
        """Initialise the LangChain QdrantVectorStore wrapper (if collection exists)."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist yet — will be created on first upsert
            self._lc_store = None
            return

        try:
            self._lc_store = LCQdrant(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                content_payload_key="content",
                metadata_payload_key="metadata",
            )
        except Exception as e:
            logger.error(f"Failed to initialise LangChain QdrantVectorStore: {e}")
            self._lc_store = None

    # ── Collection management ─────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        try:
            collections = self.client.get_collections()
            return {"status": "healthy", "collections": len(collections.collections)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def create_collection(self, recreate: bool = False) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return False
        except Exception:
            pass

        # Configure sparse vectors if enabled
        sparse_vectors_config = None
        if self.enable_sparse:
            sparse_vectors_config = {
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dimensions, distance=Distance.COSINE),
            sparse_vectors_config=sparse_vectors_config,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        )
        self._create_payload_indexes()
        logger.info(f"Collection '{self.collection_name}' created (sparse={'enabled' if self.enable_sparse else 'disabled'})")
        self._init_lc_store()
        return True

    def _create_payload_indexes(self):
        """Create payload indexes for fast filtered retrieval."""
        keyword_fields = ["type", "file_id", "filename", "source_type", "chunk_type", "parent_chunk_id"]
        for field in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug(f"Index on '{field}' skipped: {e}")

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="element_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="hierarchy_path",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=30,
                ),
            )
        except Exception:
            pass

    # ── Batch upsert (custom — preserves rich metadata payloads) ──

    def upsert_elements(
        self,
        elements: List[Dict[str, Any]],
        batch_size: int = 100,
        start_batch: int = 0,
        embed_max_retries: int = 8,
        embed_retry_base_wait: int = 65,
        embed_retry_max_wait: int = 180,
        second_pass_retry: bool = True,
        second_pass_max_retries: int = 12,
    ) -> Dict[str, int]:
        """
        Embed + upsert elements preserving full metadata payloads.
        Generates both dense (LangChain) and sparse (BM25) vectors.
        """
        if not elements:
            return {"success": 0, "failed": 0, "successful": 0, "embeddings_generated": 0}

        import re

        logger.info(f"Upserting {len(elements)} elements to Qdrant …")
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch} (skipping first {start_batch * batch_size} elements)")
        success = failed = embeddings_generated = 0
        failed_batches: List[tuple[int, List[Dict[str, Any]]]] = []

        def _retry_wait(exc: Exception, attempt: int, max_retries: int, batch_idx: int) -> Optional[int]:
            if attempt >= max_retries - 1:
                return None
            e_str = str(exc)
            retry_after = None
            if "429" in e_str or "RateLimitReached" in e_str:
                match = re.search(r"retry after (\d+) seconds", e_str, re.IGNORECASE)
                retry_after = int(match.group(1)) + 5 if match else embed_retry_base_wait
            backoff_wait = min((2 ** attempt) * max(1, embed_retry_base_wait), max(1, embed_retry_max_wait))
            wait = retry_after if retry_after is not None else backoff_wait
            logger.warning(
                f"Embedding error at batch {batch_idx} (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {wait}s ..."
            )
            return wait

        def _embed_with_retries(texts: List[str], batch_idx: int, max_retries: int) -> Optional[List[List[float]]]:
            for attempt in range(max_retries):
                try:
                    return self.embeddings.embed_documents(texts)
                except Exception as exc:
                    wait = _retry_wait(exc, attempt, max_retries, batch_idx)
                    if wait is None:
                        logger.error(
                            f"Embedding error at batch {batch_idx} after {max_retries} attempts: {exc}"
                        )
                        return None
                    time.sleep(wait)
            return None

        def _build_points(
            batch: List[Dict[str, Any]],
            batch_embeddings: List[List[float]],
            batch_sparse: Optional[List[Any]],
        ) -> List[PointStruct]:
            points: List[PointStruct] = []
            for idx, (elem, vec) in enumerate(zip(batch, batch_embeddings)):
                pid = int(hashlib.sha256(elem["element_id"].encode()).hexdigest()[:16], 16) % (10**15)

                image_url = elem.get("image_url", "")
                if not image_url and elem.get("type") == "image" and self.blob_store:
                    b64 = elem.get("image_base64") or elem.get("image_data") or ""
                    if b64 and b64[:5] in ("iVBOR", "/9j/A", "AAAA+", "R0lGO"):
                        fmt = elem.get("format", "png")
                        url = self.blob_store.upload_image(elem["element_id"], b64, fmt)
                        if url:
                            image_url = url

                payload = {
                    "element_id": elem["element_id"],
                    "file_id": elem["file_id"],
                    "filename": elem["filename"],
                    "type": elem["type"],
                    "chunk_type": elem.get("chunk_type", "parent"),
                    "parent_chunk_id": elem.get("parent_chunk_id", ""),
                    "source_type": elem.get("source_type"),
                    "hierarchy_path": elem.get("hierarchy_path", ""),
                    "section_title": elem.get("section_title", ""),
                    "content": (elem.get("content", "") or "")[:4000],
                    "summary_short": elem.get("summary_short", ""),
                    "summary_medium": elem.get("summary_medium", ""),
                    "element_index": elem.get("element_index", 0),
                    "page_number": elem.get("page_number"),
                    "image_url": image_url,
                    "related_text_ids": elem.get("related_text_ids", []),
                    "related_table_ids": elem.get("related_table_ids", []),
                    "related_image_ids": elem.get("related_image_ids", []),
                    "all_sibling_ids": elem.get("all_sibling_ids", [])[:10],
                    "loaded_at": elem.get("loaded_at", ""),
                }

                if batch_sparse and idx < len(batch_sparse):
                    sparse_vec = batch_sparse[idx]
                    point = PointStruct(
                        id=pid,
                        vector={
                            "": vec,
                            "bm25": QdrantSparseVector(
                                indices=sparse_vec.indices.tolist(),
                                values=sparse_vec.values.tolist(),
                            ),
                        },
                        payload=payload,
                    )
                else:
                    point = PointStruct(id=pid, vector=vec, payload=payload)

                points.append(point)
            return points

        all_batches = list(range(0, len(elements), batch_size))
        for batch_idx, i in enumerate(tqdm(all_batches, desc="Embed+Upsert")):
            if batch_idx < start_batch:
                continue

            batch = elements[i : i + batch_size]

            # 1. Embed via LangChain (dense) with configurable retry behavior
            texts = [self._prepare_text(el) for el in batch]
            batch_embeddings = _embed_with_retries(texts, batch_idx, max(1, embed_max_retries))

            if batch_embeddings is None:
                failed_batches.append((batch_idx, batch))
                continue
            embeddings_generated += len(batch_embeddings)

            # 2. Generate sparse vectors (BM25) if enabled
            batch_sparse = None
            if self.enable_sparse and self._sparse_encoder:
                try:
                    sparse_results = list(self._sparse_encoder.embed(texts))
                    batch_sparse = sparse_results
                except Exception as e:
                    logger.warning(f"Sparse encoding failed for batch {batch_idx}: {e}")
                    batch_sparse = None

            # 3. Build Qdrant points with rich payloads
            points = _build_points(batch, batch_embeddings, batch_sparse)

            # 4. Upsert
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                success += len(points)
            except Exception as e:
                logger.error(f"Upsert error at batch {i // batch_size}: {e}")
                failed_batches.append((batch_idx, batch))

        # 5. Second pass recovery: retry failed batches item-by-item.
        if failed_batches and second_pass_retry:
            logger.info(
                f"Second-pass recovery for {len(failed_batches)} failed batches "
                f"({sum(len(b) for _, b in failed_batches)} elements)"
            )
            for orig_batch_idx, batch in failed_batches:
                for elem in batch:
                    texts = [self._prepare_text(elem)]
                    item_embeddings = _embed_with_retries(
                        texts,
                        orig_batch_idx,
                        max(1, second_pass_max_retries),
                    )
                    if item_embeddings is None:
                        failed += 1
                        continue
                    embeddings_generated += 1

                    item_sparse = None
                    if self.enable_sparse and self._sparse_encoder:
                        try:
                            item_sparse = list(self._sparse_encoder.embed(texts))
                        except Exception as e:
                            logger.warning(f"Sparse encoding failed in second pass: {e}")
                            item_sparse = None

                    item_points = _build_points([elem], item_embeddings, item_sparse)
                    try:
                        self.client.upsert(collection_name=self.collection_name, points=item_points)
                        success += 1
                    except Exception as e:
                        logger.error(
                            f"Second-pass upsert failed for element {elem.get('element_id', '<unknown>')}: {e}"
                        )
                        failed += 1
        elif failed_batches:
            failed = sum(len(batch) for _, batch in failed_batches)

        # Re-init LC store after first successful upsert
        if self._lc_store is None:
            self._init_lc_store()

        logger.info(f"Upsert done — success: {success}, failed: {failed}")
        return {"success": success, "successful": success, "failed": failed, "embeddings_generated": embeddings_generated}

    # ── Search (Hybrid: Dense + Sparse with RRF) ──────────────

    def search(self, query: str, top_k: int = 5, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search: dense + sparse (BM25) with Reciprocal Rank Fusion.

        If sparse is unavailable, falls back to dense-only search.
        """
        if self.enable_sparse and self._sparse_encoder:
            return self._hybrid_search(query, top_k, filter_by)
        return self._dense_search(query, top_k, filter_by)

    def _dense_search(self, query: str, top_k: int = 5, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Dense-only similarity search using LangChain embeddings."""
        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

        qfilter = self._build_filter(filter_by)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant dense search failed: {e}")
            return []

        return self._format_results(results.points)

    def _hybrid_search(self, query: str, top_k: int = 5, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search using Qdrant's native prefetch + RRF fusion.

        Strategy:
        - Prefetch top-20 from dense vectors
        - Prefetch top-20 from BM25 sparse vectors
        - Fuse with RRF
        - Return top_k results
        """
        # Embed query (dense)
        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query for hybrid search: {e}")
            return []

        # Embed query (sparse / BM25)
        sparse_vector = None
        try:
            sparse_results = list(self._sparse_encoder.query_embed(query))
            if sparse_results:
                sv = sparse_results[0]
                indices = sv.indices.tolist()
                values = sv.values.tolist()
                # Only use sparse if it has valid non-empty entries
                if indices and values and len(indices) == len(values):
                    sparse_vector = QdrantSparseVector(
                        indices=indices,
                        values=values,
                    )
        except Exception as e:
            logger.warning(f"Sparse query embedding failed, falling back to dense-only: {e}")
            return self._dense_search(query, top_k, filter_by)

        if sparse_vector is None:
            return self._dense_search(query, top_k, filter_by)

        qfilter = self._build_filter(filter_by)
        prefetch_limit = max(top_k * 4, 20)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using=None,  # default (unnamed) dense vector
                        limit=prefetch_limit,
                        filter=qfilter,
                    ),
                    Prefetch(
                        query=sparse_vector,
                        using="bm25",
                        limit=prefetch_limit,
                        filter=qfilter,
                    ),
                ],
                query=Fusion.RRF,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to dense: {e}")
            return self._dense_search(query, top_k, filter_by)

        return self._format_results(results.points)

    def _build_filter(self, filter_by: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
        """Build a Qdrant filter from a dict or Filter object."""
        if not filter_by:
            return None
        if isinstance(filter_by, Filter):
            return filter_by
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_by.items()]
        return Filter(must=conditions) if conditions else None

    def _format_results(self, points) -> List[Dict[str, Any]]:
        """Format Qdrant search results into standard dicts."""
        formatted = []
        for point in points:
            p = point.payload or {}
            formatted.append({
                "score": point.score,
                "element_id": p.get("element_id"),
                "file_id": p.get("file_id"),
                "filename": p.get("filename"),
                "type": p.get("type"),
                "chunk_type": p.get("chunk_type", "parent"),
                "parent_chunk_id": p.get("parent_chunk_id", ""),
                "hierarchy_path": p.get("hierarchy_path"),
                "section_title": p.get("section_title", ""),
                "content": p.get("content", ""),
                "summary_short": p.get("summary_short"),
                "summary_medium": p.get("summary_medium"),
                "page_number": p.get("page_number"),
                "image_url": p.get("image_url", ""),
                "related_text_ids": p.get("related_text_ids", []),
                "related_table_ids": p.get("related_table_ids", []),
                "related_image_ids": p.get("related_image_ids", []),
                "payload": p,
            })
        return formatted

    # ── Parent-child expansion ─────────────────────────────────

    def expand_to_parents(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For child chunks, fetch their parent chunk for richer context.

        Returns the expanded results where child chunks are replaced by
        (or augmented with) their parent's full content. If the matched
        child content is not contained within the parent (e.g., parent was
        truncated), the child's text is appended to ensure no matched
        content is lost.
        """
        expanded = []
        seen_parents = set()

        for r in results:
            chunk_type = r.get("chunk_type", "parent")
            parent_id = r.get("parent_chunk_id", "")

            if chunk_type == "child" and parent_id:
                # Fetch parent chunk (deduplicate)
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent = self._fetch_by_element_id(parent_id)
                if parent:
                    # Use parent content but keep child's score
                    parent["score"] = r.get("score", 0.0)
                    parent["_expanded_from"] = r.get("element_id")
                    # If matched child content is not in parent (parent was
                    # truncated), append child text so LLM sees it.
                    child_content = r.get("content", "")
                    parent_content = parent.get("content", "")
                    if child_content and child_content[:80] not in parent_content:
                        parent["content"] = f"{parent_content}\n\n[Matched excerpt]\n{child_content}"
                    expanded.append(parent)
                else:
                    # Parent not found, use child as-is
                    expanded.append(r)
            else:
                # Already a parent or no parent_id
                if r.get("element_id") not in seen_parents:
                    expanded.append(r)
                    seen_parents.add(r.get("element_id", ""))

        return expanded

    def _fetch_by_element_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single element by its element_id."""
        try:
            pts, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="element_id", match=MatchValue(value=element_id))]),
                limit=1,
                with_payload=True,
            )
            if pts:
                p = pts[0].payload
                return {
                    "score": 0.0,
                    "element_id": p.get("element_id"),
                    "file_id": p.get("file_id"),
                    "filename": p.get("filename"),
                    "type": p.get("type"),
                    "chunk_type": p.get("chunk_type", "parent"),
                    "parent_chunk_id": p.get("parent_chunk_id", ""),
                    "hierarchy_path": p.get("hierarchy_path"),
                    "section_title": p.get("section_title", ""),
                    "content": p.get("content", ""),
                    "summary_short": p.get("summary_short"),
                    "summary_medium": p.get("summary_medium"),
                    "page_number": p.get("page_number"),
                    "image_url": p.get("image_url", ""),
                    "related_text_ids": p.get("related_text_ids", []),
                    "related_table_ids": p.get("related_table_ids", []),
                    "related_image_ids": p.get("related_image_ids", []),
                    "payload": p,
                }
        except Exception as e:
            logger.warning(f"Could not fetch element {element_id}: {e}")
        return None

    def search_with_relationships(self, query: str, top_k: int = 5, include_related: bool = True,
                                   filter_by: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search + parent expansion + fetch related elements.

        Pipeline:
        1. Hybrid search (dense + sparse RRF) → top candidates
        2. Expand children to parents (small-to-big)
        3. Fetch related elements (images, tables) via cross-links
        """
        # Over-fetch for reranking downstream and to compensate for
        # parent deduplication when child chunks collapse during expansion.
        search_k = top_k * 3
        main_results = self.search(query, top_k=search_k, filter_by=filter_by)

        # Expand child chunks to their parents for richer context
        main_results = self.expand_to_parents(main_results)

        # Trim to requested top_k after expansion (some children may merge)
        main_results = main_results[:top_k]

        if not include_related:
            return {"main_results": main_results, "related_elements": []}

        related_ids: set = set()
        for r in main_results:
            related_ids.update(r.get("related_text_ids", []))
            related_ids.update(r.get("related_table_ids", []))
            related_ids.update(r.get("related_image_ids", []))

        related_elements = self._fetch_by_element_ids(list(related_ids)) if related_ids else []
        return {
            "main_results": main_results,
            "related_elements": related_elements,
            "total_results": len(main_results) + len(related_elements),
        }

    def _fetch_by_element_ids(self, element_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch elements by their element_ids using indexed keyword filter."""
        results = []
        # Batch into groups to reduce round-trips
        batch_size = 10
        for i in range(0, min(len(element_ids), 20), batch_size):
            batch_ids = element_ids[i : i + batch_size]
            for eid in batch_ids:
                result = self._fetch_by_element_id(eid)
                if result:
                    results.append(result)
        return results

    # ── LangChain retriever (for use in LCEL chains) ──────────

    def as_retriever(self, top_k: int = 5, **kwargs):
        """Return a LangChain Retriever for LCEL chain composition."""
        if self._lc_store is None:
            raise RuntimeError("Collection not initialised — call create_collection() or upsert first")
        return self._lc_store.as_retriever(search_kwargs={"k": top_k, **kwargs})

    # ── Statistics ────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": getattr(info, "points_count", 0),
                "vectors_count": getattr(info, "vectors_count", 0),
                "status": getattr(info, "status", "unknown"),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _prepare_text(element: Dict[str, Any]) -> str:
        """Build embedding text with hierarchy context for better retrieval."""
        parts = []

        hierarchy = element.get("hierarchy_path", "")
        if hierarchy:
            parts.append(f"[{hierarchy}]")

        if element.get("summary_medium"):
            parts.append(element["summary_medium"])
        elif element.get("summary_short"):
            parts.append(element["summary_short"])
        else:
            content = element.get("content", "")
            parts.append(content[:2500])

        text = " ".join(parts)
        return text[:3000] if len(text) > 3000 else text
