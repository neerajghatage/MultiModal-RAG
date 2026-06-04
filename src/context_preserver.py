"""
Hierarchical Context Preserver.
Links related content (text, tables, images) from the same document and
across documents using HTML image references for explicit text-image linking.
"""

from typing import List, Dict, Any
import logging
import os
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class HierarchicalContextPreserver:
    """Links related content elements within and across documents.

    For MadCap Flare HTML help docs, this also resolves <img src> references
    so that text chunks from an HTML page are explicitly linked to the image
    elements loaded from the Images folder.
    """

    def __init__(self):
        self.file_to_elements: Dict[str, List[str]] = defaultdict(list)
        self.element_relationships: Dict[str, Dict] = {}
        self._all_elements: List[Dict[str, Any]] = []
        # Maps resolved image file paths → element_ids (populated as images are processed)
        self._image_path_to_element_id: Dict[str, str] = {}
        # Deferred links: HTML file_id → list of resolved image paths referenced
        self._html_image_refs: Dict[str, List[str]] = defaultdict(list)
        logger.info("HierarchicalContextPreserver initialized")

    @property
    def all_elements(self) -> List[Dict[str, Any]]:
        return list(self._all_elements)

    def add_document_elements(self, document: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        if document is None:
            document = {
                "file_id": kwargs.get("file_id"),
                "filename": kwargs.get("filename"),
                "text_chunks": kwargs.get("text_chunks", []),
                "tables": kwargs.get("tables", []),
                "images": kwargs.get("images", []),
            }

        file_id = document.get("file_id")
        if not file_id:
            logger.warning("Document missing file_id, skipping")
            return []

        # Collect all element IDs within this document
        all_ids = (
            [c["element_id"] for c in document.get("text_chunks", [])]
            + [t["element_id"] for t in document.get("tables", [])]
            + [i["element_id"] for i in document.get("images", [])]
        )

        enriched: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(document.get("text_chunks", [])):
            enriched.append(self._enrich(chunk, file_id, document, idx, all_ids))
            self.file_to_elements[file_id].append(chunk["element_id"])

        for idx, table in enumerate(document.get("tables", [])):
            enriched.append(self._enrich(table, file_id, document, idx, all_ids, "table"))
            self.file_to_elements[file_id].append(table["element_id"])

        for idx, image in enumerate(document.get("images", [])):
            el = self._enrich(image, file_id, document, idx, all_ids, "image")
            enriched.append(el)
            self.file_to_elements[file_id].append(image["element_id"])
            # Register image path → element_id for cross-document linking
            filepath = document.get("filepath", "")
            if filepath:
                norm = os.path.normcase(os.path.normpath(filepath))
                self._image_path_to_element_id[norm] = image["element_id"]

        # Track HTML image references for cross-document linking
        if document.get("type") == "html" and document.get("referenced_images"):
            for ref in document["referenced_images"]:
                resolved = ref.get("resolved_path", "")
                if resolved:
                    norm = os.path.normcase(os.path.normpath(resolved))
                    self._html_image_refs[file_id].append(norm)

        logger.info(f"Created {len(enriched)} linked elements from {document.get('filename')}")
        self._all_elements.extend(enriched)
        return enriched

    def resolve_cross_document_links(self):
        """After all documents are loaded, resolve HTML→image cross-references.

        This updates text elements from HTML files with explicit related_image_ids
        pointing to image elements loaded from the Images folder.
        """
        resolved_count = 0
        for file_id, image_paths in self._html_image_refs.items():
            # Find matching image element IDs
            cross_image_ids = []
            for img_path in image_paths:
                eid = self._image_path_to_element_id.get(img_path)
                if eid:
                    cross_image_ids.append(eid)

            if not cross_image_ids:
                continue

            # Update all text elements from this HTML file
            for el in self._all_elements:
                if el.get("file_id") == file_id and el.get("type") in ("text",):
                    existing = set(el.get("related_image_ids", []))
                    existing.update(cross_image_ids)
                    el["related_image_ids"] = list(existing)[:10]
                    resolved_count += 1

            # Also update image elements with back-references to this HTML's text
            text_ids = self.file_to_elements.get(file_id, [])
            text_element_ids = [tid for tid in text_ids
                                if any(e.get("element_id") == tid and e.get("type") == "text"
                                       for e in self._all_elements)]
            for img_eid in cross_image_ids:
                for el in self._all_elements:
                    if el.get("element_id") == img_eid:
                        existing = set(el.get("related_text_ids", []))
                        existing.update(text_element_ids[:5])
                        el["related_text_ids"] = list(existing)[:10]

        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} cross-document HTML→image links "
                        f"({len(self._html_image_refs)} HTML files, "
                        f"{len(self._image_path_to_element_id)} images indexed)")

    def _enrich(
        self, element: Dict[str, Any], file_id: str,
        document: Dict[str, Any], index: int,
        all_ids: List[str], element_type: str = None,
    ) -> Dict[str, Any]:
        eid = element.get("element_id")
        etype = element_type or element.get("type", "text")

        related_text = [i for i in all_ids if "text" in i and i != eid]
        related_tables = [i for i in all_ids if "table" in i and i != eid]
        related_images = [i for i in all_ids if "image" in i and i != eid]

        enriched = {
            "element_id": eid,
            "content": element.get("content", ""),
            "type": etype,
            "chunk_type": element.get("chunk_type", "parent"),
            "parent_chunk_id": element.get("parent_chunk_id"),
            "file_id": file_id,
            "filename": document.get("filename"),
            "filepath": document.get("filepath"),
            "source_type": document.get("type"),
            "element_index": index,
            "page_number": element.get("metadata", {}).get("page_number") or element.get("page"),
            "related_text_ids": related_text[:5],
            "related_table_ids": related_tables[:5],
            "related_image_ids": related_images[:5],
            "all_sibling_ids": [i for i in all_ids if i != eid][:10],
            "hierarchy_path": element.get("hierarchy_path") or self._build_hierarchy(document, element),
            "section_title": element.get("section_title", ""),
            "heading_level": element.get("heading_level", 0),
            "original_metadata": element.get("metadata", {}),
            "loaded_at": document.get("loaded_at"),
        }

        # Preserve image data for blob upload during upsert
        if etype == "image":
            b64 = element.get("image_base64") or element.get("image_data") or ""
            if b64:
                enriched["image_base64"] = b64
                enriched["format"] = element.get("format") or element.get("metadata", {}).get("format", "png")

        self.element_relationships[eid] = {
            "related_text": related_text,
            "related_tables": related_tables,
            "related_images": related_images,
            "file_id": file_id,
        }
        return enriched

    @staticmethod
    def _build_hierarchy(document: Dict[str, Any], element: Dict[str, Any]) -> str:
        parts = [document.get("filename", "Unknown")]
        # Extract feature area from filepath (e.g., "version 10/Admin Options")
        filepath = document.get("filepath", "")
        if "version 10" in filepath:
            # Extract the path relative to version 10
            idx = filepath.find("version 10")
            rel = filepath[idx:]
            folder = os.path.dirname(rel)
            if folder and folder != "version 10":
                parts.insert(0, folder.replace("\\", "/"))
        page = element.get("metadata", {}).get("page_number") or element.get("page")
        if page:
            parts.append(f"Page {page}")
        section = element.get("metadata", {}).get("section")
        if section:
            parts.append(section)
        return " > ".join(parts)
