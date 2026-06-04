"""
LangChain LCEL RAG Chain for documentation.

Features:
  - ChatPromptTemplate for managed prompts
  - RunnablePassthrough for chain composition
  - Conversation memory with rolling summary for better context
  - Streaming support built-in
  - Source tracking and image collection
"""

from typing import List, Dict, Any, Optional, Iterator
import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from .agentic_rag import build_agentic_graph

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a knowledgeable client applicationlication expert and documentation assistant.

DocRAG is an enterprise measurement software system used in the oil and gas industry for gas and liquid flow measurement, data management, and regulatory reporting.

Your role is to help users understand and use DocRAG by answering questions based ONLY on the provided documentation context. You have access to the complete Documentation documentation, including feature guides, configuration instructions, screenshots, and reference material.

When answering:

1. **Be accurate**: Use ONLY information from the provided context. Do not invent features, menu paths, or configuration options.

2. **Be specific**: Reference exact menu paths, dialog names, field names, and button labels as they appear in the documentation.

3. **Embed images inline**: When the context includes [IMAGE] entries with element IDs, embed them in your answer using EXACTLY this format: {{IMG:element_id}} (with double curly braces). Place the marker on its own line between paragraphs, only where the image directly illustrates the surrounding text. Only reference images whose element_id appears in the context. Do NOT include every image — only those that add real value to the explanation.

4. **Provide step-by-step instructions**: For how-to questions, give numbered steps the user can follow. Place relevant screenshots after the step they illustrate.

5. **Cover related areas**: If a question touches multiple DocRAG features (e.g., meters, locations, rollups, exceptions, reports), explain the connections.

6. **Do NOT cite source filenames**: Never include references like "(Per filename.htm)" or "(Source: ...)" in your answer. The user does not need to see document filenames.

7. **Acknowledge gaps**: If the provided context doesn't fully answer the question, say so explicitly rather than guessing.

If conversation history is provided, use it to maintain context and give coherent follow-up answers."""


class RAGChain:
    """
    LangChain LCEL RAG chain with:
    - Relationship-aware retrieval (custom VectorStore)
    - Conversation memory with rolling summary
    - Streaming support
    - Source tracking and citation
    """

    def __init__(self, vector_store, llm, top_k: int = 10, max_context_length: int = 8000,
                 reranker=None, embeddings=None):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.max_context_length = max_context_length
        self.reranker = reranker
        self.embeddings = embeddings

        # Max history turns to keep before trimming
        self.max_history_turns = 10

        # Build the LCEL chain
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{preamble}Context from documentation:\n\n{context}\n\n---\n\nQuestion: {question}\n\nPlease provide a helpful answer based on the context above."),
        ])

        self._chain = (
            RunnablePassthrough.assign(
                context=lambda x: x["context"],
                chat_history=lambda x: x.get("chat_history", []),
                preamble=lambda x: x.get("preamble", ""),
            )
            | self._prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"RAGChain initialized (top_k={top_k})")

    # ── Conversation summary ─────────────────────────────────

    def _build_preamble(self, conversation_summary: str = "") -> str:
        """Build a preamble with conversation summary for context continuity."""
        if not conversation_summary:
            return ""
        return (
            f"[Conversation context so far: {conversation_summary}]\n\n"
        )

    def _update_conversation_summary(self, user_query: str, answer: str, current_summary: str = "") -> str:
        """Update the rolling conversation summary after each exchange. Returns new summary."""
        try:
            answer_brief = answer[:300] + "..." if len(answer) > 300 else answer

            if not current_summary:
                return (
                    f"User asked about: {user_query[:150]}. "
                    f"Key points from answer: {answer_brief[:200]}"
                )
            else:
                summary_prompt = (
                    f"Current conversation summary:\n{current_summary}\n\n"
                    f"Latest exchange:\nUser: {user_query[:200]}\nAssistant: {answer_brief}\n\n"
                    "Update the summary in 2-3 sentences, keeping only the most important "
                    "topics and context needed for follow-up questions. Be concise."
                )
                try:
                    result = self.llm.invoke([HumanMessage(content=summary_prompt)])
                    return result.content.strip()
                except Exception:
                    return (
                        f"{current_summary[:300]} | "
                        f"Then asked: {user_query[:100]}."
                    )
        except Exception as e:
            logger.warning(f"Summary update failed: {e}")
            return current_summary

    # ── Main query ────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        top_k: int = None,
        include_related: bool = True,
        max_context_length: int = None,
        chat_history: List = None,
        conversation_summary: str = "",
    ) -> Dict[str, Any]:
        k = top_k or self.top_k
        max_ctx = max_context_length or self.max_context_length
        history = chat_history if chat_history is not None else []

        # Step 1: Retrieve
        search_results = self.vector_store.search_with_relationships(
            query=user_query, top_k=k, include_related=include_related,
        )
        main_results = search_results["main_results"]
        related_elements = search_results["related_elements"]

        if not main_results:
            return {
                "answer": "I couldn't find any relevant information in the documentation to answer your question.",
                "sources": [], "confidence": 0.0, "retrieved_count": 0,
                "chat_history": history, "conversation_summary": conversation_summary,
            }

        # Step 2: Assemble context
        context = self._assemble_context(main_results, related_elements, max_ctx)

        # Step 3: Build preamble with conversation summary
        preamble = self._build_preamble(conversation_summary)

        # Step 4: Generate answer via LCEL chain
        answer = self._chain.invoke({
            "context": context,
            "question": user_query,
            "chat_history": history,
            "preamble": preamble,
        })

        # Step 5: Update memory and summary
        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=answer))
        if len(history) > self.max_history_turns * 2:
            history = history[-(self.max_history_turns * 2):]

        new_summary = self._update_conversation_summary(user_query, answer, conversation_summary)

        # Step 6: Sources + confidence
        sources = self._format_sources(main_results, related_elements)
        avg_score = sum(r["score"] for r in main_results) / len(main_results)
        confidence = min(avg_score * 1.2, 1.0)

        # Step 7: Collect images
        images = self._collect_images(main_results, related_elements)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "retrieved_count": len(main_results) + len(related_elements),
            "images": images,
            "main_results": main_results,
            "related_elements": related_elements,
            "chat_history": history,
            "conversation_summary": new_summary,
        }

    # ── Streaming query ───────────────────────────────────────

    def query_stream(
        self,
        user_query: str,
        top_k: int = None,
        include_related: bool = True,
        max_context_length: int = None,
        chat_history: List = None,
        conversation_summary: str = "",
        on_complete=None,
    ) -> Iterator[str]:
        """Stream answer tokens. Calls on_complete(history, summary) when done."""
        k = top_k or self.top_k
        max_ctx = max_context_length or self.max_context_length
        history = chat_history if chat_history is not None else []

        search_results = self.vector_store.search_with_relationships(
            query=user_query, top_k=k, include_related=include_related,
        )
        main_results = search_results["main_results"]
        related_elements = search_results["related_elements"]

        if not main_results:
            yield "I couldn't find any relevant information in the documentation to answer your question."
            return

        context = self._assemble_context(main_results, related_elements, max_ctx)
        preamble = self._build_preamble(conversation_summary)

        full_answer = ""
        for chunk in self._chain.stream({
            "context": context,
            "question": user_query,
            "chat_history": history,
            "preamble": preamble,
        }):
            full_answer += chunk
            yield chunk

        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=full_answer))
        if len(history) > self.max_history_turns * 2:
            history = history[-(self.max_history_turns * 2):]

        new_summary = self._update_conversation_summary(user_query, full_answer, conversation_summary)

        if on_complete:
            on_complete(history, new_summary)

    # ── Agentic query (LangGraph multi-agent) ─────────────────

    def agentic_query(
        self,
        user_query: str,
        top_k: int = None,
        chat_history: List = None,
        conversation_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Multi-agent agentic RAG using LangGraph pipeline.

        Agents: Router → Research → Grader → Synthesis
        The Grader can loop back to Research once to fill context gaps.
        """
        k = top_k or self.top_k
        history = chat_history if chat_history is not None else []

        # Build the graph (lightweight — agents are just callables)
        graph = build_agentic_graph(
            llm=self.llm,
            vector_store=self.vector_store,
            reranker=self.reranker,
            max_context_length=self.max_context_length,
        )

        # Invoke the graph
        initial_state = {
            "original_query": user_query,
            "top_k": k,
            "is_complex": False,
            "sub_queries": [],
            "retrieved_docs": [],
            "related_elements": [],
            "relevant_docs": [],
            "context_gaps": [],
            "grader_loop_count": 0,
            "answer": "",
            "confidence": 0.0,
            "sources": [],
            "images": [],
            "agent_trace": [],
        }

        result = graph.invoke(initial_state)

        # Update conversation memory
        answer = result.get("answer", "")
        new_summary = conversation_summary
        if answer:
            history.append(HumanMessage(content=user_query))
            history.append(AIMessage(content=answer))
            if len(history) > self.max_history_turns * 2:
                history = history[-(self.max_history_turns * 2):]
            new_summary = self._update_conversation_summary(user_query, answer, conversation_summary)

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "retrieved_count": len(result.get("relevant_docs", [])) + len(result.get("related_elements", [])),
            "images": result.get("images", []),
            "sub_queries": result.get("sub_queries", []),
            "iterations": result.get("grader_loop_count", 1),
            "agent_trace": result.get("agent_trace", []),
            "chat_history": history,
            "conversation_summary": new_summary,
        }

    # ── Context assembly ─────────────────────────────────────

    def _assemble_context(
        self,
        main_results: List[Dict[str, Any]],
        related_elements: List[Dict[str, Any]],
        max_length: int,
    ) -> str:
        parts = []
        length = 0

        for idx, r in enumerate(main_results, 1):
            if length >= max_length:
                break
            part = self._format_result(r, idx)
            if length + len(part) > max_length:
                break
            parts.append(part)
            length += len(part)

        if related_elements and length < max_length:
            parts.append("\n--- Related Content ---\n")
            for el in related_elements:
                if length >= max_length:
                    break
                part = self._format_related(el)
                if length + len(part) > max_length:
                    break
                parts.append(part)
                length += len(part)

        return "\n\n".join(parts)

    @staticmethod
    def _format_result(result: Dict[str, Any], idx: int) -> str:
        lines = [f"[Document {idx}]", f"Source: {result.get('filename', 'Unknown')}"]
        hierarchy = result.get("hierarchy_path", "")
        if hierarchy:
            lines.append(f"Location: {hierarchy}")
        elem_type = result.get('type', 'unknown')
        lines.append(f"Type: {elem_type}")

        # For images, provide element_id so LLM can embed them inline
        if elem_type == "image":
            eid = result.get("element_id", "")
            desc = result.get("summary_short") or result.get("content", "")[:200]
            lines.append(f'[IMAGE element_id="{eid}" description="{desc}"]')
        else:
            content = result.get("summary_medium") or result.get("summary_short") or result.get("content", "")
            if content:
                lines.append(f"\nContent:\n{content}")

        rel_info = []
        if result.get("related_text_ids"):
            rel_info.append("text")
        if result.get("related_image_ids"):
            rel_info.append("images")
        if result.get("related_table_ids"):
            rel_info.append("tables")
        if rel_info:
            lines.append(f"Related: {', '.join(rel_info)}")
        return "\n".join(lines)

    @staticmethod
    def _format_related(element: Dict[str, Any]) -> str:
        elem_type = element.get('type', 'unknown')
        lines = [
            f"[Related {elem_type.title()}]",
            f"From: {element.get('filename', 'Unknown')}",
        ]
        # For images, provide element_id for inline embedding
        if elem_type == "image":
            eid = element.get("element_id", "")
            desc = element.get("summary_short") or element.get("content", "")[:200]
            lines.append(f'[IMAGE element_id="{eid}" description="{desc}"]')
        else:
            content = element.get("summary_short") or element.get("summary_medium") or element.get("content", "")
            if content:
                lines.append(content)
        return "\n".join(lines)

    # ── Source formatting ─────────────────────────────────────

    @staticmethod
    def _format_sources(main_results, related_elements) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for r in main_results:
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
        for el in related_elements:
            eid = el.get("element_id", "")
            if eid not in seen:
                sources.append({
                    "filename": el.get("filename", "Unknown"),
                    "type": el.get("type", "unknown"),
                    "relationship": "related",
                })
                seen.add(eid)
        return sources

    # ── Image collection ──────────────────────────────────────

    @staticmethod
    def _collect_images(main_results, related_elements) -> List[Dict[str, Any]]:
        images = []
        seen = set()
        for r in list(main_results) + list(related_elements):
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

    # ── Display helper ────────────────────────────────────────

    @staticmethod
    def format_response(result: Dict[str, Any]) -> str:
        lines = ["=" * 60, "ANSWER", "=" * 60, result["answer"], ""]
        confidence = result.get("confidence", 0.0)
        lines.append(f"Confidence: {confidence:.1%}\n")
        sources = result.get("sources", [])
        if sources:
            lines += ["=" * 60, "SOURCES", "=" * 60]
            for i, s in enumerate(sources, 1):
                lines.append(f"{i}. {s.get('filename', 'Unknown')} ({s.get('type', 'unknown')})")
                if s.get("hierarchy_path"):
                    lines.append(f"   Path: {s['hierarchy_path']}")
                if s.get("page_number"):
                    lines.append(f"   Page: {s['page_number']}")
                if "relevance_score" in s:
                    lines.append(f"   Relevance: {s['relevance_score']:.2%}")
                lines.append("")
        return "\n".join(lines)

    # ── Memory management (for CLI usage) ───────────────────

    def clear_history(self):
        """No-op — history is now externally managed via SessionStore."""
        pass
