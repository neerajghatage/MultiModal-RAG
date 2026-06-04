"""
Configuration for the LangChain-based RAG system.
Supports both Azure OpenAI and standard OpenAI via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for RAG system."""

    # ── Qdrant ────────────────────────────────────────────────
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_docs")

    # ── Standard OpenAI (used when AZURE_OPENAI_ENDPOINT is not set) ──
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS = 3072

    # ── Azure OpenAI ──────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

    # ── Project paths ─────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).parent.parent  # src/ -> project root
    DATA_FOLDER = PROJECT_ROOT / "data"
    NOTEBOOKS_FOLDER = PROJECT_ROOT / "notebooks"
    QDRANT_STORAGE = PROJECT_ROOT / "qdrant_storage"

    # ── Azure Blob Storage (for images) ────────────────────────
    AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "rag-images")

    # ── Misc ──────────────────────────────────────────────────
    PROJECT_NAME = os.getenv("PROJECT_NAME", "DOC_RAG")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ── Redis (session store) ─────────────────────────────────
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    SESSION_TTL = int(os.getenv("SESSION_TTL", "3600"))  # seconds

    # ── Reranker ──────────────────────────────────────────────
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
    ENABLE_SPARSE = os.getenv("ENABLE_SPARSE", "true").lower() == "true"

    # ── Convenience properties ────────────────────────────────
    @classmethod
    def is_azure(cls) -> bool:
        return bool(cls.AZURE_OPENAI_ENDPOINT)

    @classmethod
    def effective_api_key(cls) -> str:
        """Return the API key that should be used (Azure takes priority)."""
        if cls.is_azure():
            return cls.AZURE_OPENAI_API_KEY
        return cls.OPENAI_API_KEY

    # ── LangChain model factories ─────────────────────────────
    @classmethod
    def get_chat_llm(cls, temperature: float = 0.3, streaming: bool = False):
        """Return a LangChain chat model configured from env vars."""
        if cls.is_azure():
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_endpoint=cls.AZURE_OPENAI_ENDPOINT,
                api_key=cls.AZURE_OPENAI_API_KEY,
                api_version=cls.AZURE_OPENAI_API_VERSION,
                azure_deployment=cls.AZURE_OPENAI_CHAT_DEPLOYMENT,
                temperature=temperature,
                streaming=streaming,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=cls.OPENAI_API_KEY,
                model=cls.OPENAI_MODEL,
                temperature=temperature,
                streaming=streaming,
            )

    @classmethod
    def get_embeddings(cls):
        """Return a LangChain embeddings model configured from env vars."""
        if cls.is_azure():
            from langchain_openai import AzureOpenAIEmbeddings
            return AzureOpenAIEmbeddings(
                azure_endpoint=cls.AZURE_OPENAI_ENDPOINT,
                api_key=cls.AZURE_OPENAI_API_KEY,
                api_version=cls.AZURE_OPENAI_API_VERSION,
                azure_deployment=cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                max_retries=0,  # Disable SDK retries — vector_store handles retries with backoff
            )
        else:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                api_key=cls.OPENAI_API_KEY,
                model=cls.EMBEDDING_MODEL,
            )

    @classmethod
    def get_raw_openai_client(cls):
        """Return a raw openai SDK client (for summarizer / vision calls)."""
        if cls.is_azure():
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=cls.AZURE_OPENAI_API_KEY,
                azure_endpoint=cls.AZURE_OPENAI_ENDPOINT,
                api_version=cls.AZURE_OPENAI_API_VERSION,
            )
        else:
            from openai import OpenAI
            return OpenAI(api_key=cls.OPENAI_API_KEY)

    @classmethod
    def get_chat_model_name(cls) -> str:
        if cls.is_azure():
            return cls.AZURE_OPENAI_CHAT_DEPLOYMENT
        return cls.OPENAI_MODEL

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        print("\n" + "=" * 60)
        print("VALIDATING CONFIGURATION")
        print("=" * 60)

        if cls.is_azure():
            if not cls.AZURE_OPENAI_API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY not set in .env file")
            print(f"  Mode:             Azure OpenAI")
            print(f"  Endpoint:         {cls.AZURE_OPENAI_ENDPOINT}")
            print(f"  API Version:      {cls.AZURE_OPENAI_API_VERSION}")
            print(f"  Chat Deployment:  {cls.AZURE_OPENAI_CHAT_DEPLOYMENT}")
            print(f"  Embed Deployment: {cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
        else:
            if not cls.OPENAI_API_KEY:
                raise ValueError("Neither AZURE_OPENAI_ENDPOINT nor OPENAI_API_KEY is set")
            print(f"  Mode:       Standard OpenAI")
            print(f"  Model:      {cls.OPENAI_MODEL}")
            print(f"  Embedding:  {cls.EMBEDDING_MODEL}")

        print(f"  Qdrant URL: {cls.QDRANT_URL}")
        print(f"  Data:       {cls.DATA_FOLDER}")
        if cls.AZURE_BLOB_CONNECTION_STRING:
            print(f"  Blob Store: {cls.AZURE_BLOB_CONTAINER_NAME} (configured)")
        else:
            print(f"  Blob Store: not configured (images will use data-uri fallback)")
        print("\n  Configuration validated successfully!")
        print("=" * 60 + "\n")
