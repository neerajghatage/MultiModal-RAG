"""
Metadata self-query filtering for RAG retrieval.

Parses user queries to extract structured filter conditions
(document type, section, hierarchy) that narrow Qdrant search
BEFORE vector similarity runs — reducing noise significantly.
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Filter schema for DocRAG docs ────────────────────────────

FILTER_EXTRACTION_PROMPT = """\
You are a query analyzer for DocRAG oil & gas measurement software documentation.

Given a user query, extract any metadata filters that would narrow the search.
Return ONLY a JSON object with these optional fields (omit fields that don't apply):

- "type": one of ["text", "image", "table", "procedure"] — the document type the user wants
- "section_keywords": list of section/topic keywords to match in hierarchy_path (e.g. ["meter run", "AGA-3"])

Rules:
- If the user asks "how to" or "steps to" → type: "text"
- If the user asks for a "screenshot", "picture", "dialog" → type: "image"
- If the user asks for a "table", "list of values", "codes" → type: "table"
- Extract specific DocRAG features/sections mentioned (e.g. "meter run", "station", "rollup", "exception")
- If no filters can be extracted, return empty object: {}

Examples:
Query: "Show me the meter run configuration dialog"
{"type": "image", "section_keywords": ["meter run", "configuration"]}

Query: "How do I set up AGA-3 orifice plate calculations?"
{"type": "text", "section_keywords": ["AGA-3", "orifice"]}

Query: "What error codes can appear in exceptions?"
{"type": "table", "section_keywords": ["exception", "error"]}

Query: "Tell me about DocRAG"
{}

User query: {query}
JSON:"""


class QueryFilterExtractor:
    """Extracts Qdrant filter conditions from natural language queries."""

    def __init__(self, llm):
        self.llm = llm

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Parse user query → Qdrant filter dict.

        Returns a dict that can be passed to VectorStore.search(filter_by=...).
        Returns empty dict if no filters can be extracted.
        """
        try:
            prompt = FILTER_EXTRACTION_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            filters = json.loads(content)
            if not isinstance(filters, dict):
                return {}

            # Build Qdrant-compatible filter
            qdrant_filter = self._build_qdrant_filter(filters)
            if qdrant_filter:
                logger.info(f"Self-query filters extracted: {qdrant_filter}")
            return qdrant_filter

        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Filter extraction failed (non-critical): {e}")
            return {}

    def _build_qdrant_filter(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed JSON to Qdrant filter_by dict."""
        filters = {}

        doc_type = parsed.get("type")
        if doc_type and doc_type in ("text", "image", "table", "procedure"):
            filters["type"] = doc_type

        return filters

    def get_section_keywords(self, query: str) -> list:
        """
        Extract section keywords for hierarchy_path text search.
        Returns keywords that can be used for scroll/filter.
        """
        try:
            prompt = FILTER_EXTRACTION_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            return parsed.get("section_keywords", [])
        except Exception:
            return []
