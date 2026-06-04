"""
CLI entry point for the RAG system.

Usage:
    python main.py ingest --folder data/
    python main.py query "What is the gas flow calculation?"
    python main.py chat
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

from src.config import Config
from src.loader import MultiFileLoader
from src.context_preserver import HierarchicalContextPreserver
from src.summarizer import IntelligentSummarizer
from src.vector_store import VectorStore
from src.rag_chain import RAGChain
from src.blob_storage import ImageBlobStore


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_components(collection_name=None, skip_blob=False):
    """Instantiate all RAG components from Config."""
    Config.validate()

    embeddings = Config.get_embeddings()
    llm = Config.get_chat_llm(streaming=True)
    raw_client = Config.get_raw_openai_client()
    model_name = Config.get_chat_model_name()

    blob_store = None
    if not skip_blob and Config.AZURE_BLOB_CONNECTION_STRING:
        blob_store = ImageBlobStore(
            connection_string=Config.AZURE_BLOB_CONNECTION_STRING,
            container_name=Config.AZURE_BLOB_CONTAINER_NAME,
        )

    vs = VectorStore(
        qdrant_url=Config.QDRANT_URL,
        embeddings=embeddings,
        collection_name=collection_name or Config.QDRANT_COLLECTION_NAME,
        blob_store=blob_store,
        enable_sparse=Config.ENABLE_SPARSE,
    )
    summarizer = IntelligentSummarizer(client=raw_client, model=model_name)

    # Initialize reranker
    reranker = None
    if Config.ENABLE_RERANKER:
        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name=Config.RERANKER_MODEL)

    chain = RAGChain(vector_store=vs, llm=llm, reranker=reranker, embeddings=embeddings)

    return vs, summarizer, chain


# ── Commands ──────────────────────────────────────────────────

def _cache_path(folder: Path, collection: str) -> Path:
    """Return the summary cache file path for a given folder + collection."""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    safe_folder = folder.resolve().as_posix().replace("/", "_").replace(":", "")
    safe_col = collection or "default"
    return cache_dir / f"summaries_{safe_folder}_{safe_col}.json"


def _save_summary_cache(elements: list, cache_file: Path):
    """Persist summarized elements to JSON, stripping heavy image_base64."""
    slim = []
    for el in elements:
        e = {k: v for k, v in el.items() if k != "image_base64"}
        slim.append(e)
    cache_file.write_text(json.dumps(slim, default=str), encoding="utf-8")
    print(f"Summary cache saved → {cache_file}  ({len(slim)} elements)")


def _load_summary_cache(cache_file: Path) -> list | None:
    """Load cached summaries if available."""
    if not cache_file.exists():
        return None
    try:
        elements = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"Summary cache loaded ← {cache_file}  ({len(elements)} elements)")
        return elements
    except Exception as e:
        logging.getLogger(__name__).warning(f"Cache load failed: {e}")
        return None


def _summarize_with_resume(summarizer, elements: list, cache_file: Path,
                           images_only: bool = False) -> list:
    """Summarize elements with per-element progress saving.

    If the process stops mid-way, the next run resumes from the last
    unsummarized element instead of restarting from scratch.
    """
    progress_file = cache_file.with_suffix(".progress.json")

    # Load any prior progress
    done_map: dict[str, dict] = {}
    if progress_file.exists():
        try:
            saved = json.loads(progress_file.read_text(encoding="utf-8"))
            done_map = {e["element_id"]: e for e in saved}
            print(f"Resuming summarization — {len(done_map)} elements already done")
        except Exception:
            done_map = {}

    pending = [el for el in elements if el.get("element_id") not in done_map]
    if not pending:
        print("All elements already summarized (from progress cache)")
        merged = []
        for el in elements:
            merged.append(done_map.get(el["element_id"], el))
        return merged

    print(f"Summarizing {len(pending)} remaining elements "
          f"({len(done_map)} already cached)...")

    SAVE_EVERY = 25
    newly_done = 0

    for el in pending:
        summarized = summarizer.process_elements([el], images_only=images_only)
        result = summarized[0] if summarized else el
        slim = {k: v for k, v in result.items() if k != "image_base64"}
        done_map[el["element_id"]] = slim
        newly_done += 1

        if newly_done % SAVE_EVERY == 0:
            _flush_progress(done_map, progress_file)
            print(f"  … progress saved ({len(done_map)} / {len(elements)})")

    # Final flush
    _flush_progress(done_map, progress_file)

    # Build ordered output matching original element order
    merged = []
    for el in elements:
        merged.append(done_map.get(el["element_id"], el))

    # Promote progress file to full cache and clean up
    _save_summary_cache(merged, cache_file)
    progress_file.unlink(missing_ok=True)
    return merged


def _flush_progress(done_map: dict, progress_file: Path):
    """Write current progress to disk."""
    progress_file.write_text(
        json.dumps(list(done_map.values()), default=str),
        encoding="utf-8",
    )


def cmd_ingest(args):
    """Load documents, summarize, embed, and upsert to Qdrant."""
    vs, summarizer, _ = build_components(
        collection_name=getattr(args, 'collection', None),
        skip_blob=getattr(args, 'no_blob', False),
    )

    folder = Path(args.folder)
    collection = getattr(args, 'collection', None) or Config.QDRANT_COLLECTION_NAME
    cache_file = _cache_path(folder, collection)
    print(f"\nIngesting documents from: {folder}\n")

    # Try loading from summary cache first
    if not args.no_cache and not args.skip_summarize:
        cached = _load_summary_cache(cache_file)
        if cached is not None:
            elements = cached
            print(f"Using cached summaries — skipping load + summarize")

            # Re-upload images from disk — image_base64 is stripped from cache,
            # so we must re-read raw bytes from the original filepath.
            if vs.blob_store:
                img_count = 0
                for el in elements:
                    if el.get("type") == "image" and not el.get("image_url"):
                        filepath = el.get("filepath", "")
                        if filepath and Path(filepath).exists():
                            try:
                                img_bytes = Path(filepath).read_bytes()
                                b64 = base64.b64encode(img_bytes).decode()
                                fmt = el.get("format") or Path(filepath).suffix.lstrip(".") or "png"
                                url = vs.blob_store.upload_image(el["element_id"], b64, fmt)
                                if url:
                                    el["image_url"] = url
                                    img_count += 1
                            except Exception as _e:
                                logging.getLogger(__name__).warning(
                                    f"Re-upload failed for {filepath}: {_e}"
                                )
                if img_count:
                    print(f"Re-uploaded {img_count} images to blob storage")
                else:
                    print("No images needed re-uploading (already have URLs or no filepath)")

            vs.create_collection(recreate=args.recreate)
            start_batch = getattr(args, 'start_batch', 0)
            result = vs.upsert_elements(
                elements,
                batch_size=args.batch_size,
                start_batch=start_batch,
                embed_max_retries=args.embed_max_retries,
                embed_retry_base_wait=args.embed_retry_base_wait,
                embed_retry_max_wait=args.embed_retry_max_wait,
                second_pass_retry=not args.no_second_pass_retry,
                second_pass_max_retries=args.embed_second_pass_retries,
            )
            print(f"\nUpsert complete: {result}")
            print(f"Stats: {vs.get_statistics()}")
            return

    # 1. Load
    loader = MultiFileLoader()
    documents = loader.load_all_documents(folder)
    print(f"Loaded {len(documents)} documents")

    if not documents:
        print("No documents found — exiting.")
        return

    # 2. Context linking
    preserver = HierarchicalContextPreserver()
    for doc in documents:
        preserver.add_document_elements(doc)
    preserver.resolve_cross_document_links()
    elements = preserver.all_elements
    print(f"Created {len(elements)} elements with relationships")

    # 2.5. Upload images to blob BEFORE summarization (base64 gets stripped later)
    if vs.blob_store:
        img_count = 0
        for el in elements:
            if el.get("type") == "image":
                b64 = el.get("image_base64") or el.get("image_data") or ""
                if b64 and len(b64) > 100 and b64[:5] in ("iVBOR", "/9j/A", "AAAA+", "R0lGO"):
                    fmt = el.get("format", "png")
                    url = vs.blob_store.upload_image(el["element_id"], b64, fmt)
                    if url:
                        el["image_url"] = url
                        img_count += 1
        if img_count:
            print(f"Uploaded {img_count} images to blob storage")

    # 3. Summarize (with per-element resume support)
    if not args.skip_summarize:
        images_only = getattr(args, 'images_only_summarize', False)
        elements = _summarize_with_resume(
            summarizer, elements, cache_file, images_only=images_only,
        )
        print(f"Summarized {len(elements)} elements")

    # 4. Create collection + upsert
    vs.create_collection(recreate=args.recreate)
    start_batch = getattr(args, 'start_batch', 0)
    result = vs.upsert_elements(
        elements,
        batch_size=args.batch_size,
        start_batch=start_batch,
        embed_max_retries=args.embed_max_retries,
        embed_retry_base_wait=args.embed_retry_base_wait,
        embed_retry_max_wait=args.embed_retry_max_wait,
        second_pass_retry=not args.no_second_pass_retry,
        second_pass_max_retries=args.embed_second_pass_retries,
    )
    print(f"\nUpsert complete: {result}")
    print(f"Stats: {vs.get_statistics()}")


def cmd_query(args):
    """Run a single query and print the result."""
    _, _, chain = build_components()

    result = chain.query(args.question, top_k=args.top_k)
    print(chain.format_response(result))


def cmd_chat(args):
    """Interactive multi-turn chat loop."""
    _, _, chain = build_components()

    print("\n" + "=" * 60)
    print("  RAG Chat (type 'quit' to exit, 'clear' to reset memory)")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            chain.clear_history()
            print("  [Memory cleared]\n")
            continue

        # Stream answer
        print("\nAssistant: ", end="", flush=True)
        for chunk in chain.query_stream(question, top_k=args.top_k):
            print(chunk, end="", flush=True)
        print("\n")


# ── Arg parser ────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Confidential RAG System (LangChain)")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into Qdrant")
    p_ingest.add_argument("--folder", default="data/", help="Folder to scan")
    p_ingest.add_argument("--collection", default=None, help="Collection name (default: from config)")
    p_ingest.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    p_ingest.add_argument("--batch-size", type=int, default=100, help="Upsert batch size")
    p_ingest.add_argument("--skip-summarize", action="store_true", help="Skip all LLM summarization")
    p_ingest.add_argument("--images-only-summarize", action="store_true", help="Only VLM-describe images, skip text/code/table (~$62 vs $2600)")
    p_ingest.add_argument("--start-batch", type=int, default=0, help="Resume upsert from this batch index (skip earlier batches)")
    p_ingest.add_argument("--no-cache", action="store_true", help="Ignore summary cache and re-run summarization")
    p_ingest.add_argument("--no-blob", action="store_true", help="Skip Azure Blob Storage uploads (images already uploaded)")
    p_ingest.add_argument("--embed-max-retries", type=int, default=8, help="Retries per embedding batch before marking batch failed")
    p_ingest.add_argument("--embed-retry-base-wait", type=int, default=65, help="Base wait seconds between embedding retries")
    p_ingest.add_argument("--embed-retry-max-wait", type=int, default=180, help="Max wait seconds between embedding retries")
    p_ingest.add_argument("--embed-second-pass-retries", type=int, default=12, help="Retries per element during second-pass recovery")
    p_ingest.add_argument("--no-second-pass-retry", action="store_true", help="Disable second-pass single-element recovery for failed batches")

    # query
    p_query = sub.add_parser("query", help="Run a single query")
    p_query.add_argument("question", help="Question to ask")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of docs to retrieve")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat")
    p_chat.add_argument("--top-k", type=int, default=5, help="Number of docs to retrieve")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat}[args.command](args)


if __name__ == "__main__":
    main()
