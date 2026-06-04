"""
FastAPI REST API for the RAG system.

Architecture aligned with RAG API:
    - JWT Bearer authentication via Azure AD (Entra ID)
    - Configurable base URL, host, and port
    - HTTPS support in production
    - Health check endpoint with structured response
    - CORS with configurable origins

Endpoints:
    POST /ingest          — Ingest documents from a folder
    POST /query           — Single question → answer
    POST /chat            — Multi-turn conversational query
    GET  /health          — System health check (no auth required)
    GET  /stats           — Collection statistics
    DELETE /history       — Clear conversation memory

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
    python main.py                  (self-host mode with configurable port)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from .config import Config
from .auth import validate_token
from .loader import MultiFileLoader
from .context_preserver import HierarchicalContextPreserver
from .summarizer import IntelligentSummarizer
from .vector_store import VectorStore
from .rag_chain import RAGChain
from .blob_storage import ImageBlobStore
from .reranker import CrossEncoderReranker
from .session_store import SessionStore

logger = logging.getLogger(__name__)

# ── Global state (initialised at startup) ─────────────────────
_vs: VectorStore = None
_chain: RAGChain = None
_summarizer: IntelligentSummarizer = None
_session_store: SessionStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise RAG components once on startup."""
    global _vs, _chain, _summarizer, _session_store

    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    Config.validate()

    embeddings = Config.get_embeddings()
    llm = Config.get_chat_llm(streaming=True)
    raw_client = Config.get_raw_openai_client()
    model_name = Config.get_chat_model_name()

    blob_store = None
    if Config.AZURE_BLOB_CONNECTION_STRING:
        try:
            blob_store = ImageBlobStore(
                connection_string=Config.AZURE_BLOB_CONNECTION_STRING,
                container_name=Config.AZURE_BLOB_CONTAINER_NAME,
            )
        except Exception as e:
            logger.warning(f"Blob storage init failed (images will be unavailable): {e}")

    _vs = VectorStore(
        qdrant_url=Config.QDRANT_URL,
        embeddings=embeddings,
        collection_name=Config.QDRANT_COLLECTION_NAME,
        blob_store=blob_store,
        enable_sparse=Config.ENABLE_SPARSE,
    )
    _summarizer = IntelligentSummarizer(client=raw_client, model=model_name)

    # Initialize reranker
    reranker = None
    if Config.ENABLE_RERANKER:
        reranker = CrossEncoderReranker(model_name=Config.RERANKER_MODEL)

    _chain = RAGChain(vector_store=_vs, llm=llm, reranker=reranker, embeddings=embeddings)

    # Initialize Redis session store
    _session_store = SessionStore()

    logger.info("RAG API ready")
    yield
    logger.info("RAG API shutting down")


app = FastAPI(
    title="Confidential RAG API",
    version="1.0.0",
    description="Industry-grade RAG system powered by LangChain + Qdrant + Azure OpenAI",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Session ID helper ─────────────────────────────────────────

def _get_session_id(claims: Optional[dict]) -> str:
    """Derive session_id from JWT claims (oid) or fallback for local dev."""
    if claims and claims.get("oid"):
        return claims["oid"]
    return "dev"


# ── Request / Response models ─────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(10, ge=1, le=50, description="Number of documents to retrieve")
    include_related: bool = Field(True, description="Fetch related elements")
    stream: bool = Field(False, description="Stream response tokens")
    agentic: bool = Field(False, description="Use multi-agent agentic RAG for complex questions")


class ImageResult(BaseModel):
    element_id: str
    filename: str
    description: str
    image_url: str
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    retrieved_count: int
    latency_ms: float
    images: List[ImageResult] = []
    sub_queries: Optional[List[str]] = None
    iterations: Optional[int] = None
    agent_trace: Optional[List[str]] = None


class IngestRequest(BaseModel):
    folder: str = Field("data/", description="Folder path to scan")
    recreate: bool = Field(False, description="Drop and recreate the collection")
    batch_size: int = Field(100, ge=1, le=500)
    skip_summarize: bool = Field(False, description="Skip LLM summarization step")


class IngestResponse(BaseModel):
    documents_loaded: int
    elements_created: int
    upsert_result: Dict[str, int]
    stats: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    qdrant: Dict[str, Any]
    collection: Dict[str, Any]
    redis: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check (no auth required — used by load balancers/probes)."""
    return HealthResponse(
        status="healthy",
        qdrant=_vs.health_check(),
        collection=_vs.get_statistics(),
        redis=_session_store.health_check() if _session_store else {},
    )


@app.get("/image/{element_id}")
async def get_image(element_id: str, claims: dict = Depends(validate_token)):
    """Proxy endpoint to serve images from blob storage."""
    # Try blob storage first
    if _vs.blob_store:
        result = _vs.blob_store.download_image(element_id)
        if result:
            image_bytes, content_type = result
            return Response(content=image_bytes, media_type=content_type)

    # Fallback: try to get base64 from Qdrant payload
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        points = _vs.client.scroll(
            collection_name=_vs.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="element_id", match=MatchValue(value=element_id))
            ]),
            limit=1,
            with_payload=True,
        )[0]
        if points:
            payload = points[0].payload
            b64 = payload.get("image_base64") or payload.get("image_data", "")
            if b64:
                import base64
                image_bytes = base64.b64decode(b64)
                return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        logger.warning(f"Fallback image fetch failed for {element_id}: {e}")

    raise HTTPException(status_code=404, detail=f"Image not found: {element_id}")


@app.get("/stats")
async def stats(claims: dict = Depends(validate_token)):
    """Return collection statistics."""
    return _vs.get_statistics()


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, claims: dict = Depends(validate_token)):
    """Single question → answer (with per-user session memory)."""
    session_id = _get_session_id(claims)

    if req.stream:
        return StreamingResponse(
            _stream_answer(req, session_id),
            media_type="text/event-stream",
        )

    start = time.perf_counter()

    # Load session state from Redis
    chat_history = _session_store.load_history(session_id)
    conversation_summary = _session_store.load_summary(session_id)

    if req.agentic:
        result = _chain.agentic_query(
            user_query=req.question,
            top_k=req.top_k,
            chat_history=chat_history,
            conversation_summary=conversation_summary,
        )
    else:
        result = _chain.query(
            user_query=req.question,
            top_k=req.top_k,
            include_related=req.include_related,
            chat_history=chat_history,
            conversation_summary=conversation_summary,
        )

    # Save updated session state back to Redis
    _session_store.save_history(session_id, result["chat_history"])
    _session_store.save_summary(session_id, result["conversation_summary"])

    latency = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        retrieved_count=result["retrieved_count"],
        latency_ms=round(latency, 1),
        images=result.get("images", []),
        sub_queries=result.get("sub_queries"),
        iterations=result.get("iterations"),
        agent_trace=result.get("agent_trace"),
    )


async def _stream_answer(req: QueryRequest, session_id: str):
    """SSE stream for token-by-token responses."""
    chat_history = _session_store.load_history(session_id)
    conversation_summary = _session_store.load_summary(session_id)

    def on_complete(updated_history, updated_summary):
        _session_store.save_history(session_id, updated_history)
        _session_store.save_summary(session_id, updated_summary)

    for chunk in _chain.query_stream(
        user_query=req.question,
        top_k=req.top_k,
        include_related=req.include_related,
        chat_history=chat_history,
        conversation_summary=conversation_summary,
        on_complete=on_complete,
    ):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest, claims: dict = Depends(validate_token)):
    """Multi-turn conversational query (per-user session via Redis)."""
    session_id = _get_session_id(claims)
    start = time.perf_counter()

    # Load session state from Redis
    chat_history = _session_store.load_history(session_id)
    conversation_summary = _session_store.load_summary(session_id)

    if req.agentic:
        result = _chain.agentic_query(
            user_query=req.question,
            top_k=req.top_k,
            chat_history=chat_history,
            conversation_summary=conversation_summary,
        )
    else:
        result = _chain.query(
            user_query=req.question,
            top_k=req.top_k,
            include_related=req.include_related,
            chat_history=chat_history,
            conversation_summary=conversation_summary,
        )

    # Save updated session state back to Redis
    _session_store.save_history(session_id, result["chat_history"])
    _session_store.save_summary(session_id, result["conversation_summary"])

    latency = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        retrieved_count=result["retrieved_count"],
        latency_ms=round(latency, 1),
        images=result.get("images", []),
        sub_queries=result.get("sub_queries"),
        iterations=result.get("iterations"),
        agent_trace=result.get("agent_trace"),
    )


@app.delete("/history")
async def clear_history(claims: dict = Depends(validate_token)):
    """Clear conversation memory for the current user."""
    session_id = _get_session_id(claims)
    _session_store.clear(session_id)
    return {"status": "history cleared", "session_id": session_id}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, claims: dict = Depends(validate_token)):
    """Ingest documents from a folder into Qdrant."""
    folder = Path(req.folder)
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    # 1. Load files
    loader = MultiFileLoader()
    documents = loader.load_all_documents(folder)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents loaded from folder")

    # 2. Context linking
    preserver = HierarchicalContextPreserver()
    for doc in documents:
        preserver.add_document_elements(doc)
    preserver.resolve_cross_document_links()
    elements = preserver.all_elements

    # 2.5. Upload images to blob BEFORE summarization (summarizer strips image_base64)
    if _vs.blob_store:
        import base64
        for el in elements:
            if el.get("type") == "image" and not el.get("image_url"):
                b64 = el.get("image_base64") or el.get("image_data") or ""
                if b64 and len(b64) > 100 and b64[:5] in ("iVBOR", "/9j/A", "AAAA+", "R0lGO"):
                    fmt = el.get("format", "png")
                    url = _vs.blob_store.upload_image(el["element_id"], b64, fmt)
                    if url:
                        el["image_url"] = url

    # 3. Summarize
    if not req.skip_summarize:
        elements = _summarizer.process_elements(elements)

    # 4. Upsert
    _vs.create_collection(recreate=req.recreate)
    upsert_result = _vs.upsert_elements(elements, batch_size=req.batch_size)

    return IngestResponse(
        documents_loaded=len(documents),
        elements_created=len(elements),
        upsert_result=upsert_result,
        stats=_vs.get_statistics(),
    )


# ── Self-host entry point (like RAG API Program.cs) ──

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8002"))
    ssl_keyfile = os.getenv("SSL_KEYFILE", "")
    ssl_certfile = os.getenv("SSL_CERTFILE", "")

    base_url = os.getenv("API_BASE_URL", f"http{'s' if ssl_certfile else ''}://{host}:{port}")
    logger.info(f"Starting RAG API at {base_url}")

    uvicorn_kwargs = {
        "app": "api:app",
        "host": host,
        "port": port,
        "log_level": Config.LOG_LEVEL.lower(),
    }

    # HTTPS support (like Product.BasePath with https:// in RAG API)
    if ssl_certfile and ssl_keyfile:
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile

    uvicorn.run(**uvicorn_kwargs)
