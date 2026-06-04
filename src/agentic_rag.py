"""
LangGraph Multi-Agent Agentic RAG Pipeline for documentation.

Architecture:
    Router Agent → Research Agent → Grader Agent → Synthesis Agent

    - Router:    Classifies query complexity (simple → direct, complex → pipeline)
    - Research:  Decomposes query, performs exhaustive multi-query retrieval
    - Grader:    Evaluates each document's relevance, identifies gaps, loops back
    - Synthesis: Generates final answer from vetted context with citations

Uses LangGraph StateGraph for orchestration with typed state passing.
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from operator import add

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# State Schema
# ═══════════════════════════════════════════════════════════════

class AgenticState(TypedDict):
    """Shared state flowing through the agent graph."""
    # Input
    original_query: str
    top_k: int

    # Router output
    is_complex: bool

    # Research Agent output
    sub_queries: Annotated[List[str], add]
    retrieved_docs: List[Dict[str, Any]]  # deduplicated pool
    related_elements: List[Dict[str, Any]]

    # Grader Agent output
    relevant_docs: List[Dict[str, Any]]  # docs that passed grading
    context_gaps: List[str]  # identified gaps needing follow-up
    grader_loop_count: int

    # Synthesis Agent output
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    images: List[Dict[str, Any]]

    # Trace (for debugging/observability)
    agent_trace: Annotated[List[str], add]


# ═══════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """\
You are a query complexity classifier for DocRAG oil & gas measurement documentation.

Classify the user's question as SIMPLE or COMPLEX:

SIMPLE queries:
- Ask about a single concept, feature, or definition
- Can be answered from one document/section
- Examples: "What is exception code 5?", "Where is the meter run dialog?"

COMPLEX queries:
- Require combining information from multiple documents
- Ask about relationships between features
- Involve prerequisites, dependencies, or multi-step workflows
- Ask "does X require Y" or "how does X relate to Y"
- Examples: "Does meter characteristics and flow data are necessary to apply source using source assignment?", "What are all the steps to configure AGA-3 with orifice calculations?"

User question: {question}

Respond with ONLY one word: SIMPLE or COMPLEX"""


RESEARCH_DECOMPOSE_PROMPT = """\
You are a search query optimizer for DocRAG oil & gas measurement software documentation.

documentation covers:
- Meter configuration (characteristics, data resolution, meter types)
- Source assignments (how measurement sources connect to meters)
- Quality analyses and recalculation workflows
- Exception codes (error conditions with numbered codes)
- Admin options, toolbox operations, import/export
- Reports, rollups, locations, billing, stations

Given the user's complex question, decompose it into 3-5 focused sub-queries that will TOGETHER retrieve ALL information needed. Each sub-query targets a DIFFERENT document or section.

Rules:
1. Each sub-query should target a different aspect/document
2. Include at least one query about prerequisites/requirements
3. Include at least one query about the "how it works" mechanism
4. Include one query targeting error conditions or edge cases
5. Keep each sub-query under 15 words
6. Return ONLY sub-queries, one per line, no numbering, no explanation

User question: {question}

Sub-queries:"""


GRADER_PROMPT = """\
You are a relevance grader for documentation retrieval.

User's original question: {question}

Evaluate the following retrieved document. Determine if it contains information that would help answer the user's question.

Document source: {filename}
Document section: {section}
Document content:
{content}

Respond with ONLY one of:
- HIGH: Directly answers part of the question or provides critical context
- LOW: Tangentially related but not useful for answering
- NONE: Completely irrelevant"""


GAP_ANALYSIS_PROMPT = """\
You are analyzing whether retrieved documentation is sufficient to answer a user's question about DocRAG software.

User question: {question}

Documents retrieved (summarized):
{context_summary}

Analyze the retrieved information:
1. What aspects of the question ARE covered by these documents?
2. What aspects are MISSING or insufficiently covered?

If the documents are SUFFICIENT to fully answer the question, respond:
SUFFICIENT

If there are GAPS, respond:
GAPS
- <describe specific missing information as a search query, max 12 words>
- <another gap as search query if needed>

Maximum 2 gap queries. Only identify genuine gaps, not minor details."""


SYNTHESIS_PROMPT = """\
You are a knowledgeable client applicationlication expert and documentation assistant.

DocRAG is an enterprise measurement software used in oil and gas for flow measurement, data management, and regulatory reporting.

Answer the user's question based ONLY on the provided vetted documentation context. This context has been carefully retrieved and verified as relevant.

When answering:
1. Be accurate — use ONLY information from the provided context
2. Be specific — reference exact menu paths, dialog names, field names
3. Include visual guidance when context mentions related images/screenshots
4. Provide step-by-step instructions for how-to questions
5. Cite sources — reference document names for details
6. If context doesn't fully answer the question, say so explicitly

Context from documentation:
{context}

---

Question: {question}

Provide a comprehensive, well-structured answer:"""


# ═══════════════════════════════════════════════════════════════
# Agent Nodes
# ═══════════════════════════════════════════════════════════════

class RouterAgent:
    """Classifies query complexity to decide pipeline path."""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgenticState) -> dict:
        query = state["original_query"]

        try:
            prompt = ROUTER_PROMPT.format(question=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            classification = response.content.strip().upper()
            is_complex = "COMPLEX" in classification
        except Exception as e:
            logger.warning(f"Router failed, defaulting to complex: {e}")
            is_complex = True

        route = "COMPLEX" if is_complex else "SIMPLE"
        logger.info(f"Router Agent: query classified as {route}")

        return {
            "is_complex": is_complex,
            "agent_trace": [f"Router: classified as {route}"],
        }


class ResearchAgent:
    """Decomposes queries and performs exhaustive multi-query retrieval."""

    def __init__(self, llm, vector_store, sub_query_top_k: int = 6):
        self.llm = llm
        self.vector_store = vector_store
        self.sub_query_top_k = sub_query_top_k

    def __call__(self, state: AgenticState) -> dict:
        query = state["original_query"]
        top_k = state.get("top_k", 10)

        # Check if this is a follow-up from Grader (gap-filling)
        context_gaps = state.get("context_gaps", [])
        if context_gaps:
            sub_queries = context_gaps
            logger.info(f"Research Agent: filling {len(sub_queries)} gaps from Grader")
        else:
            sub_queries = self._decompose_query(query)
            if not sub_queries:
                sub_queries = [query]
            logger.info(f"Research Agent: decomposed into {len(sub_queries)} sub-queries")

        # Retrieve for each sub-query
        all_results: Dict[str, Dict] = {}
        all_related: Dict[str, Dict] = {}

        # Preserve existing results from previous iterations
        for doc in state.get("retrieved_docs", []):
            eid = doc.get("element_id", "")
            if eid:
                all_results[eid] = doc
        for rel in state.get("related_elements", []):
            eid = rel.get("element_id", "")
            if eid:
                all_related[eid] = rel

        for sq in sub_queries:
            try:
                search_results = self.vector_store.search_with_relationships(
                    query=sq,
                    top_k=self.sub_query_top_k,
                    include_related=True,
                )
                for r in search_results["main_results"]:
                    eid = r.get("element_id", "")
                    if eid and (eid not in all_results or r.get("score", 0) > all_results[eid].get("score", 0)):
                        all_results[eid] = r
                for r in search_results.get("related_elements", []):
                    eid = r.get("element_id", "")
                    if eid:
                        all_related[eid] = r
            except Exception as e:
                logger.warning(f"Research retrieval failed for '{sq[:40]}': {e}")

        retrieved = list(all_results.values())
        related = list(all_related.values())

        # Sort by score descending
        retrieved.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(f"Research Agent: {len(retrieved)} docs retrieved, {len(related)} related")

        return {
            "sub_queries": sub_queries,
            "retrieved_docs": retrieved,
            "related_elements": related[:15],
            "context_gaps": [],  # Clear gaps after filling
            "agent_trace": [f"Research: {len(sub_queries)} queries → {len(retrieved)} docs"],
        }

    def _decompose_query(self, query: str) -> List[str]:
        """Use LLM to decompose query into sub-queries."""
        try:
            prompt = RESEARCH_DECOMPOSE_PROMPT.format(question=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            lines = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            sub_queries = [l for l in lines if len(l) > 5]
            return sub_queries[:5]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]


class GraderAgent:
    """Evaluates document relevance and identifies context gaps."""

    def __init__(self, llm, max_loops: int = 1):
        self.llm = llm
        self.max_loops = max_loops

    def __call__(self, state: AgenticState) -> dict:
        query = state["original_query"]
        retrieved_docs = state.get("retrieved_docs", [])
        loop_count = state.get("grader_loop_count", 0)

        if not retrieved_docs:
            return {
                "relevant_docs": [],
                "context_gaps": [],
                "grader_loop_count": loop_count + 1,
                "agent_trace": ["Grader: no docs to evaluate"],
            }

        # Grade each document (top 15 to cap LLM calls)
        relevant_docs = []
        for doc in retrieved_docs[:15]:
            grade = self._grade_document(query, doc)
            if grade == "HIGH":
                relevant_docs.append(doc)

        logger.info(f"Grader Agent: {len(relevant_docs)}/{min(len(retrieved_docs), 15)} docs passed (HIGH relevance)")

        # Gap analysis — only if we haven't looped too many times
        context_gaps = []
        if loop_count < self.max_loops:
            context_gaps = self._analyze_gaps(query, relevant_docs)
            if context_gaps:
                logger.info(f"Grader Agent: identified {len(context_gaps)} gaps, requesting follow-up")

        return {
            "relevant_docs": relevant_docs,
            "context_gaps": context_gaps,
            "grader_loop_count": loop_count + 1,
            "agent_trace": [f"Grader: {len(relevant_docs)} relevant, {len(context_gaps)} gaps"],
        }

    def _grade_document(self, query: str, doc: Dict[str, Any]) -> str:
        """Grade a single document for relevance."""
        content = doc.get("summary_medium") or doc.get("summary_short") or doc.get("content", "")
        if not content.strip():
            return "NONE"

        try:
            prompt = GRADER_PROMPT.format(
                question=query,
                filename=doc.get("filename", "Unknown"),
                section=doc.get("hierarchy_path", "") or doc.get("section_title", ""),
                content=content[:500],
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            grade = response.content.strip().upper()

            if "HIGH" in grade:
                return "HIGH"
            elif "LOW" in grade:
                return "LOW"
            return "NONE"
        except Exception as e:
            logger.warning(f"Grading failed for doc: {e}")
            # On failure, keep the doc (conservative)
            return "HIGH"

    def _analyze_gaps(self, query: str, relevant_docs: List[Dict]) -> List[str]:
        """Check if the relevant docs are sufficient or have gaps."""
        if not relevant_docs:
            return [query]  # No relevant docs at all, retry original

        # Build summary of what we have
        summary_parts = []
        for doc in relevant_docs[:8]:
            fname = doc.get("filename", "?")
            section = doc.get("section_title", "") or doc.get("hierarchy_path", "")
            content = (doc.get("summary_short") or doc.get("content", ""))[:100]
            summary_parts.append(f"- [{fname}] {section}: {content}")
        context_summary = "\n".join(summary_parts)

        try:
            prompt = GAP_ANALYSIS_PROMPT.format(
                question=query,
                context_summary=context_summary,
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            if text.startswith("SUFFICIENT"):
                return []

            # Extract gap queries
            gaps = []
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("- ") and len(line) > 5:
                    gap_query = line[2:].strip()
                    if gap_query:
                        gaps.append(gap_query)

            return gaps[:2]
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return []


class SynthesisAgent:
    """Generates final answer from vetted, graded context."""

    def __init__(self, llm, reranker=None, max_context_length: int = 8000):
        self.llm = llm
        self.reranker = reranker
        self.max_context_length = max_context_length

    def __call__(self, state: AgenticState) -> dict:
        query = state["original_query"]
        relevant_docs = state.get("relevant_docs", [])
        related_elements = state.get("related_elements", [])
        top_k = state.get("top_k", 10)

        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant information in the documentation to answer your question.",
                "confidence": 0.0,
                "sources": [],
                "images": [],
                "agent_trace": ["Synthesis: no relevant docs, returning empty answer"],
            }

        # Rerank the relevant docs for optimal ordering
        if self.reranker and self.reranker.is_available:
            relevant_docs = self.reranker.rerank(
                query=query,
                results=relevant_docs,
                top_k=top_k,
            )

        # Assemble context
        context = self._assemble_context(relevant_docs, related_elements)

        # Generate answer
        try:
            prompt = SYNTHESIS_PROMPT.format(context=context, question=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"Synthesis generation failed: {e}")
            answer = "An error occurred while generating the answer. Please try again."

        # Compute confidence
        if relevant_docs:
            avg_score = sum(r.get("score", 0) for r in relevant_docs) / len(relevant_docs)
            confidence = min(avg_score * 1.2, 1.0)
        else:
            confidence = 0.0

        # Collect sources and images
        sources = self._format_sources(relevant_docs, related_elements)
        images = self._collect_images(relevant_docs, related_elements)

        logger.info(f"Synthesis Agent: generated answer ({len(answer)} chars), confidence={confidence:.2f}")

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "images": images,
            "agent_trace": [f"Synthesis: {len(relevant_docs)} docs → {len(answer)} char answer"],
        }

    def _assemble_context(self, docs: List[Dict], related: List[Dict]) -> str:
        """Build context string from graded docs."""
        parts = []
        length = 0

        for idx, doc in enumerate(docs, 1):
            if length >= self.max_context_length:
                break
            part = self._format_doc(doc, idx)
            if length + len(part) > self.max_context_length:
                break
            parts.append(part)
            length += len(part)

        # Add related content if space allows
        if related and length < self.max_context_length:
            parts.append("\n--- Related Content ---\n")
            for el in related:
                if length >= self.max_context_length:
                    break
                part = f"[Related {el.get('type', 'content')}] From: {el.get('filename', '?')}\n"
                content = el.get("summary_short") or el.get("content", "")
                part += content[:200]
                if length + len(part) > self.max_context_length:
                    break
                parts.append(part)
                length += len(part)

        return "\n\n".join(parts)

    @staticmethod
    def _format_doc(doc: Dict[str, Any], idx: int) -> str:
        lines = [f"[Document {idx}]", f"Source: {doc.get('filename', 'Unknown')}"]
        hierarchy = doc.get("hierarchy_path", "")
        if hierarchy:
            lines.append(f"Location: {hierarchy}")
        content = doc.get("summary_medium") or doc.get("summary_short") or doc.get("content", "")
        if content:
            lines.append(f"\nContent:\n{content}")
        return "\n".join(lines)

    @staticmethod
    def _format_sources(docs, related) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for r in docs:
            fid = r.get("file_id", "")
            if fid not in seen:
                sources.append({
                    "filename": r.get("filename", "Unknown"),
                    "type": r.get("type", "unknown"),
                    "hierarchy_path": r.get("hierarchy_path", ""),
                    "page_number": r.get("page_number"),
                    "relevance_score": r.get("score", 0.0),
                })
                seen.add(fid)
        for el in related:
            eid = el.get("element_id", "")
            if eid not in seen:
                sources.append({
                    "filename": el.get("filename", "Unknown"),
                    "type": el.get("type", "unknown"),
                    "relationship": "related",
                })
                seen.add(eid)
        return sources

    @staticmethod
    def _collect_images(docs, related) -> List[Dict[str, Any]]:
        images = []
        seen = set()
        for r in list(docs) + list(related):
            if r.get("type") != "image":
                continue
            eid = r.get("element_id", "")
            if eid in seen:
                continue
            seen.add(eid)
            url = r.get("image_url", "")
            if not url:
                continue
            images.append({
                "element_id": eid,
                "filename": r.get("filename", "Unknown"),
                "description": r.get("summary_short") or r.get("content", "")[:200],
                "image_url": url,
                "page_number": r.get("page_number"),
            })
        return images


# ═══════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════

def build_agentic_graph(llm, vector_store, reranker=None, max_context_length: int = 8000):
    """
    Build the LangGraph multi-agent RAG pipeline.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({"original_query": "...", "top_k": 10})
    """
    # Instantiate agents
    router = RouterAgent(llm)
    researcher = ResearchAgent(llm, vector_store, sub_query_top_k=6)
    grader = GraderAgent(llm, max_loops=1)
    synthesizer = SynthesisAgent(llm, reranker=reranker, max_context_length=max_context_length)

    # Build the state graph
    workflow = StateGraph(AgenticState)

    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("research", researcher)
    workflow.add_node("grader", grader)
    workflow.add_node("synthesis", synthesizer)

    # Define edges
    workflow.set_entry_point("router")

    # Router → Research (always for agentic path; simple queries bypass the graph entirely)
    workflow.add_edge("router", "research")

    # Research → Grader
    workflow.add_edge("research", "grader")

    # Grader → conditional: if gaps AND loop allowed → Research, else → Synthesis
    def grader_routing(state: AgenticState) -> str:
        gaps = state.get("context_gaps", [])
        loop_count = state.get("grader_loop_count", 0)
        if gaps and loop_count <= 1:
            return "research"  # Loop back for gap-filling
        return "synthesis"

    workflow.add_conditional_edges("grader", grader_routing, {
        "research": "research",
        "synthesis": "synthesis",
    })

    # Synthesis → END
    workflow.add_edge("synthesis", END)

    # Compile
    graph = workflow.compile()
    logger.info("LangGraph agentic RAG pipeline compiled successfully")

    return graph
