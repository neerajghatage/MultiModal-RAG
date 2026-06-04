"""
Confidential Multimodal RAG System — LangChain Edition
"""

__version__ = "1.0.0"
__author__ = "Neeraj Ghatage"

from .config import Config
from .loader import MultiFileLoader
from .context_preserver import HierarchicalContextPreserver
from .summarizer import IntelligentSummarizer
from .vector_store import VectorStore
from .rag_chain import RAGChain
