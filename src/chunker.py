"""
Heading-aware HTML chunker for MadCap Flare WebHelp documentation.

Features:
  - Parses data-mc-toc-path for breadcrumb hierarchy
  - Splits on H1/H2/H3/H4 headings (section boundaries)
  - Keeps procedures (<ol>) as atomic units
  - Creates parent + child chunks for small-to-big retrieval
  - Injects breadcrumb prefix into chunk content for self-contained embedding
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import re
import hashlib

from bs4 import BeautifulSoup, Tag, NavigableString

logger = logging.getLogger(__name__)

# Heading tags that define section boundaries
HEADING_TAGS = {"h1", "h2", "h3", "h4"}

# Block-level tags that indicate a container should be descended into
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol",
              "div", "p", "section", "article", "dl", "pre", "blockquote"}

# Navigation / chrome classes to skip during extraction
SKIP_CLASSES = {"nocontent", "breadcrumbs", "footer", "keywords",
                "side-content", "sidenav-wrapper", "title-bar"}

# Maximum child chunk size in characters
MAX_CHILD_CHUNK_SIZE = 500
# Maximum parent chunk size
MAX_PARENT_CHUNK_SIZE = 3000


class HeadingAwareChunker:
    """
    MadCap Flare-aware HTML chunker that preserves document structure.

    Produces:
    - Parent chunks: full heading section content (used for context delivery)
    - Child chunks: small paragraphs within a section (used for precise retrieval)

    Each chunk carries:
    - hierarchy_path: full breadcrumb from TOC + folder + heading hierarchy
    - parent_chunk_id: links children back to their parent
    - chunk_type: "parent" or "child"
    """

    def __init__(self, max_child_size: int = MAX_CHILD_CHUNK_SIZE,
                 max_parent_size: int = MAX_PARENT_CHUNK_SIZE):
        self.max_child_size = max_child_size
        self.max_parent_size = max_parent_size

    def chunk_html(self, filepath: str, raw_html: str, file_id: str) -> List[Dict[str, Any]]:
        """
        Parse HTML and return structured chunks with hierarchy metadata.

        Returns list of chunk dicts with keys:
            element_id, content, type, chunk_type, parent_chunk_id,
            hierarchy_path, heading_level, section_title
        """
        soup = BeautifulSoup(raw_html, "html.parser")

        # Extract metadata from MadCap attributes
        toc_path = self._extract_toc_path(soup)
        page_title = self._extract_title(soup)
        folder_path = self._extract_folder_hierarchy(filepath)

        # Build the base breadcrumb from TOC path or folder structure
        base_breadcrumb = toc_path or folder_path or ""

        # Find the main content area
        main_content = self._find_main_content(soup)
        if main_content is None:
            # Fallback: use entire body
            main_content = soup.find("body") or soup

        # Extract sections by heading hierarchy
        sections = self._extract_sections(main_content, page_title)

        # Generate parent + child chunks
        chunks = []
        for section in sections:
            section_chunks = self._build_section_chunks(
                section, file_id, base_breadcrumb
            )
            chunks.extend(section_chunks)

        # If no sections found (very simple page), create a single chunk
        if not chunks:
            text = main_content.get_text(separator="\n", strip=True)
            if text.strip():
                breadcrumb = f"{base_breadcrumb} > {page_title}" if base_breadcrumb else page_title
                chunk_id = f"{file_id}_section_0"
                chunks.append({
                    "element_id": chunk_id,
                    "content": text[:self.max_parent_size],
                    "type": "text",
                    "chunk_type": "parent",
                    "parent_chunk_id": None,
                    "hierarchy_path": breadcrumb,
                    "heading_level": 0,
                    "section_title": page_title,
                })

        # Filter out empty/trivial chunks that waste embedding budget.
        # Chunks under 50 chars rarely carry enough semantic meaning for
        # useful embeddings and add noise to search results.
        chunks = [c for c in chunks if len(c.get("content", "").strip()) >= 50]

        # Extract image references for cross-linking
        referenced_images = self._extract_image_refs(soup, filepath)

        return chunks, referenced_images

    def _extract_toc_path(self, soup: BeautifulSoup) -> str:
        """Extract data-mc-toc-path from the <html> tag."""
        html_tag = soup.find("html")
        if html_tag and html_tag.get("data-mc-toc-path"):
            return html_tag["data-mc-toc-path"]
        return ""

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from <title> tag."""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        # Fallback: first h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def _extract_folder_hierarchy(self, filepath: str) -> str:
        """Build hierarchy from folder path relative to version 10."""
        if not filepath:
            return ""
        normalized = filepath.replace("\\", "/")
        marker = "version 10/"
        idx = normalized.find(marker)
        if idx == -1:
            return ""
        rel = normalized[idx + len(marker):]
        # Remove filename, keep folder path
        folder = os.path.dirname(rel)
        if folder:
            return folder.replace("/", " > ")
        return ""

    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content div (MadCap uses id='mc-main-content' or role='main')."""
        main = soup.find(id="mc-main-content")
        if main:
            return main
        main = soup.find(attrs={"role": "main"})
        if main:
            return main
        main = soup.find(attrs={"data-mc-content-body": "True"})
        if main:
            return main
        return None

    def _is_exception_toggle_heading(self, tag: Tag) -> bool:
        """Check if an H4 tag is a MadCap MCToggler exception entry (not a standalone section).

        Exception directory pages use H4 with class "exception" and contain MCToggler
        links followed by hidden drop-down divs. These should be treated as dropdown
        items within the parent section (e.g., "1000 - 1099") rather than standalone sections.
        """
        if tag.name and tag.name.lower() == "h4":
            classes = tag.get("class", []) or []
            # Match "exception", "exceptionDeprecated", and any other exception* variant
            if any(c.lower().startswith("exception") for c in classes):
                return True
        return False

    def _is_dropdown_toggle(self, tag: Tag) -> bool:
        """Check if a tag is a MadCap dropdown/expanding toggle.

        Robust detection: any element containing an <a class="MCToggler"> with a
        data-mc-targets attribute. Covers DropDownLink, ExampleLink, notelink,
        BRDnotelink and any other toggle wrapper class. Excludes plain inline
        cross-reference links (InlineLink, GlossaryPageLink) which are not togglers.
        """
        if not isinstance(tag, Tag):
            return False
        # The toggle anchor must be the tag itself or a DIRECT child, otherwise a
        # generic wrapper div that merely *contains* a toggle somewhere deep would
        # be misclassified as a toggle (collapsing all its content).
        if tag.name == "a" and "MCToggler" in (tag.get("class", []) or []):
            return bool(tag.get("data-mc-targets"))
        link = tag.find("a", class_="MCToggler", recursive=False)
        if link is not None and link.get("data-mc-targets"):
            return True
        return False

    def _is_dropdown_content(self, tag: Tag) -> bool:
        """Check if a tag is MadCap dropdown content.

        Robust detection via data-mc-target-name attribute. Covers both
        <div class="drop-down" data-mc-target-name="X"> and bare
        <div style="display:none" data-mc-target-name="X"> (no class) forms.
        """
        if not isinstance(tag, Tag) or tag.name != "div":
            return False
        return tag.get("data-mc-target-name") is not None

    def _is_procedure_div(self, tag: Tag) -> bool:
        """Check if a tag is a MadCap procedure block."""
        if not isinstance(tag, Tag) or tag.name != "div":
            return False
        return "procedure" in " ".join(tag.get("class", []) or [])

    def _has_block_children(self, tag: Tag) -> bool:
        """Whether a container has block-level children worth descending into."""
        for c in tag.children:
            if isinstance(c, Tag) and c.name and c.name.lower() in BLOCK_TAGS:
                return True
        return False

    def _collect_nodes(self, container: Tag) -> List[Tag]:
        """
        Flatten a container into an ordered stream of meaningful structural nodes,
        descending through generic wrapper divs (e.g. <div class="BRD">).

        Special nodes (headings, dropdown toggles, dropdown content, procedures,
        tables, lists, paragraphs, images) are yielded as-is. Generic container
        divs/sections are unwrapped so their inner structure is not lost to a
        flat get_text() call.
        """
        nodes: List[Tag] = []
        for child in container.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower() if child.name else ""
            if name in ("script", "style", "noscript"):
                continue
            classes = child.get("class", []) or []
            if any(c in SKIP_CLASSES for c in classes):
                continue

            # Special structural nodes — yield as-is, do not descend
            if (name in HEADING_TAGS
                    or self._is_dropdown_toggle(child)
                    or self._is_dropdown_content(child)
                    or self._is_procedure_div(child)
                    or name in ("table", "ul", "ol", "img", "p", "pre", "blockquote", "dl")):
                nodes.append(child)
                continue

            # Generic container — descend to recover inner structure
            if name in ("div", "section", "article", "main", "span") and self._has_block_children(child):
                nodes.extend(self._collect_nodes(child))
                continue

            # Leaf element — yield
            nodes.append(child)
        return nodes

    def _extract_node(self, node: Tag) -> str:
        """Extract structured text from a single content node (table-aware)."""
        name = node.name.lower() if node.name else ""
        if name == "table":
            return self._extract_table(node)
        if name == "ol":
            return self._extract_ordered_list(node)
        if name == "ul":
            return self._extract_unordered_list(node)
        if self._is_procedure_div(node):
            return self._extract_procedure(node)
        # Container/leaf that holds tables or lists — preserve their structure
        if node.find("table") or node.find("ol") or node.find("ul"):
            return self._extract_section_text(node)
        return self._extract_element_text(node)

    def _extract_sections(self, content: Tag, page_title: str) -> List[Dict[str, Any]]:
        """
        Split content into sections by heading tags.
        Each section has: heading_level, title, elements, drop_downs.

        The content is first flattened via _collect_nodes so that wrapper divs are
        unwrapped and no inner structure (headings, tables, dropdowns) is lost.

        Special handling for Exception Directory pages: H4 headings that are
        MCToggler exception entries are NOT treated as section boundaries.
        Instead, they are collected as drop_downs within their parent H2 section.
        """
        sections = []
        current_section = {
            "heading_level": 0,
            "title": page_title,
            "elements": [],
            "drop_downs": [],
        }

        nodes = self._collect_nodes(content)
        consumed_targets: set = set()
        n = len(nodes)
        i = 0
        while i < n:
            child = nodes[i]
            tag_name = child.name.lower() if child.name else ""

            # Special case: H4 exception toggle headings do NOT start new sections.
            if self._is_exception_toggle_heading(child):
                exception_title = child.get_text(strip=True)
                dd_content = ""
                # Look ahead for the matching dropdown content node
                if i + 1 < n and self._is_dropdown_content(nodes[i + 1]):
                    dd_content = self._extract_section_text(nodes[i + 1])
                    tname = nodes[i + 1].get("data-mc-target-name")
                    if tname:
                        consumed_targets.add(tname)
                    i += 1

                if exception_title and dd_content.strip():
                    current_section["drop_downs"].append({
                        "title": exception_title,
                        "content": dd_content,
                    })
                elif exception_title:
                    current_section["elements"].append(exception_title)
                i += 1
                continue

            # Regular heading → start a new section
            if tag_name in HEADING_TAGS:
                if current_section["elements"] or current_section["drop_downs"]:
                    sections.append(current_section)
                level = int(tag_name[1])
                current_section = {
                    "heading_level": level,
                    "title": child.get_text(strip=True),
                    "elements": [],
                    "drop_downs": [],
                }
                i += 1
                continue

            # Dropdown toggle (DropDownLink / ExampleLink / notelink / etc.)
            if self._is_dropdown_toggle(child):
                dd_title, dd_content, target = self._parse_dropdown(child, content)
                if target:
                    consumed_targets.add(target)
                if dd_title and dd_content.strip():
                    current_section["drop_downs"].append({
                        "title": dd_title,
                        "content": dd_content,
                    })
                elif dd_title:
                    current_section["elements"].append(dd_title)
                i += 1
                continue

            # Dropdown content div (only if not already consumed by its toggle)
            if self._is_dropdown_content(child):
                tname = child.get("data-mc-target-name", "")
                if tname in consumed_targets:
                    i += 1
                    continue
                dd_text = self._extract_section_text(child)
                if dd_text.strip():
                    current_section["drop_downs"].append({
                        "title": tname,
                        "content": dd_text,
                    })
                i += 1
                continue

            # Regular content element (table-aware extraction)
            text = self._extract_node(child)
            if text.strip() and not self._is_nav_bar(text) and len(text.strip()) >= 5:
                current_section["elements"].append(text)
            i += 1

        if current_section["elements"] or current_section["drop_downs"]:
            sections.append(current_section)

        return sections

    def _parse_dropdown(self, toggle_tag: Tag, parent: Tag) -> Tuple[str, str, str]:
        """Parse a dropdown toggle and find its content.

        Returns (title, content, target_name). target_name is the data-mc-targets
        value so the caller can mark the matching content div as consumed.
        """
        link = toggle_tag.find("a", class_="MCToggler")
        if not link:
            return "", "", ""
        title = link.get_text(strip=True)
        # Remove the toggle icon text
        title = re.sub(r'^(Closed|Open)\s*', '', title)

        # Find the matching drop-down content div by target name
        targets = link.get("data-mc-targets", "")
        if targets:
            dd_div = parent.find("div", attrs={"data-mc-target-name": targets})
            if dd_div:
                return title, self._extract_section_text(dd_div), targets
        return title, "", targets

    def _extract_section_text(self, tag: Tag) -> str:
        """Extract text from a section, preserving procedure structure."""
        parts = []

        for child in tag.children:
            if not isinstance(child, Tag):
                if isinstance(child, NavigableString) and child.strip():
                    parts.append(child.strip())
                continue

            if child.name == "div" and "procedure" in " ".join(child.get("class", [])):
                # Procedure block — keep the numbered steps together
                proc_text = self._extract_procedure(child)
                if proc_text:
                    parts.append(proc_text)
            elif child.name == "ol":
                # Ordered list (procedure steps)
                proc_text = self._extract_ordered_list(child)
                if proc_text:
                    parts.append(proc_text)
            elif child.name == "ul":
                # Unordered list (may interleave explanatory content)
                ul_text = self._extract_unordered_list(child)
                if ul_text:
                    parts.append(ul_text)
            elif child.name == "table":
                # Table — extract as structured text
                table_text = self._extract_table(child)
                if table_text:
                    parts.append(table_text)
            else:
                text = child.get_text(strip=True)
                if text:
                    parts.append(text)

        return "\n\n".join(parts)

    def _extract_procedure(self, proc_div: Tag) -> str:
        """Extract a procedure block, preserving all steps and interspersed text.

        Procedure divs may contain multiple <ol> blocks separated by explanatory
        <p>/<ul>/<table> content. Iterate every direct child in document order so
        no steps or context are dropped.
        """
        parts = []
        for child in proc_div.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower() if child.name else ""
            classes = child.get("class", []) or []
            if any(c in SKIP_CLASSES for c in classes):
                continue
            if name == "ol":
                txt = self._extract_ordered_list(child)
            elif name == "ul":
                txt = self._extract_unordered_list(child)
            elif name == "table":
                txt = self._extract_table(child)
            elif self._is_procedure_div(child):
                txt = self._extract_procedure(child)
            else:
                txt = child.get_text(separator=" ", strip=True)
            if txt and txt.strip():
                parts.append(txt)
        return "\n".join(parts)

    def _extract_ordered_list(self, ol: Tag) -> str:
        """Extract an ordered list as numbered steps.

        MadCap procedure lists frequently interleave explanatory content
        (<p>, <ul>, <div class="example">, equations, notes) between the
        <li class="Step"> elements as DIRECT children of the <ol>. Iterate all
        direct children in document order so this content is not lost.
        """
        parts = []
        step_no = 0
        for child in ol.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower() if child.name else ""
            if name == "li":
                text = child.get_text(separator=" ", strip=True)
                if text:
                    step_no += 1
                    parts.append(f"{step_no}. {text}")
            elif name == "ul":
                items = [f"   • {li.get_text(strip=True)}"
                         for li in child.find_all("li", recursive=False)
                         if li.get_text(strip=True)]
                if items:
                    parts.append("\n".join(items))
            elif name == "ol":
                nested = self._extract_ordered_list(child)
                if nested.strip():
                    parts.append(nested)
            elif name == "table":
                tbl = self._extract_table(child)
                if tbl.strip():
                    parts.append(tbl)
            else:
                text = child.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _extract_unordered_list(self, ul: Tag) -> str:
        """Extract an unordered list, preserving interspersed content.

        MadCap lists often interleave explanatory <p>, <table>, nested lists and
        <div class="example"> between <li> items as DIRECT children of the <ul>.
        Iterate all direct children in document order so nothing is lost.
        """
        parts = []
        for child in ul.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower() if child.name else ""
            if name == "li":
                text = child.get_text(separator=" ", strip=True)
                if text:
                    parts.append(f"\u2022 {text}")
            elif name == "ul":
                nested = self._extract_unordered_list(child)
                if nested.strip():
                    parts.append(nested)
            elif name == "ol":
                nested = self._extract_ordered_list(child)
                if nested.strip():
                    parts.append(nested)
            elif name == "table":
                tbl = self._extract_table(child)
                if tbl.strip():
                    parts.append(tbl)
            else:
                text = child.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _extract_table(self, table: Tag) -> str:
        """Extract table as structured text."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True)
                     for td in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _extract_element_text(self, tag: Tag) -> str:
        """Extract text from a single element, handling nested structure."""
        if tag.name in ("script", "style", "noscript"):
            return ""
        # Skip navigation/chrome elements
        classes = tag.get("class", [])
        skip_classes = {"nocontent", "breadcrumbs", "footer", "keywords",
                        "side-content", "sidenav-wrapper", "title-bar"}
        if any(c in skip_classes for c in classes):
            return ""
        return tag.get_text(separator=" ", strip=True)

    def _build_section_chunks(
        self, section: Dict[str, Any], file_id: str, base_breadcrumb: str
    ) -> List[Dict[str, Any]]:
        """
        Build parent + child chunks for a section.

        Parent = full section text (for context delivery)
        Children = smaller sub-parts (for precise retrieval)

        For sections with many dropdowns (e.g., Exception Directory pages with
        100 exceptions), creates multiple sub-parent chunks so that no exception
        content is lost to truncation.
        """
        chunks = []
        title = section["title"]
        level = section["heading_level"]

        # Build full breadcrumb for this section
        if base_breadcrumb:
            breadcrumb = f"{base_breadcrumb} > {title}"
        else:
            breadcrumb = title

        # Collect all text for this section (elements + dropdowns)
        all_text_parts = list(section["elements"])
        dropdown_parts = []
        for dd in section["drop_downs"]:
            dd_header = f"**{dd['title']}**"
            dd_text = f"{dd_header}\n{dd['content']}"
            all_text_parts.append(dd_text)
            dropdown_parts.append(dd_text)

        full_text = "\n\n".join(all_text_parts)
        if not full_text.strip():
            return chunks

        # If the section has many dropdowns and content exceeds parent size,
        # create multiple sub-parent chunks to avoid truncation.
        if len(dropdown_parts) > 5 and len(full_text) > self.max_parent_size:
            chunks.extend(self._build_grouped_parent_chunks(
                section, file_id, breadcrumb, level, title, dropdown_parts
            ))
        elif len(full_text) > self.max_parent_size:
            # Long section without many dropdowns — split into sub-parents
            # so no content is lost when child expands to parent.
            chunks.extend(self._build_grouped_parent_chunks_from_parts(
                all_text_parts, file_id, breadcrumb, level, title
            ))
        else:
            # Standard single parent chunk (fits within limit)
            section_hash = hashlib.sha256(f"{file_id}_{title}".encode()).hexdigest()[:8]
            parent_id = f"{file_id}_parent_{section_hash}"

            chunks.append({
                "element_id": parent_id,
                "content": full_text,
                "type": "text",
                "chunk_type": "parent",
                "parent_chunk_id": None,
                "hierarchy_path": breadcrumb,
                "heading_level": level,
                "section_title": title,
            })

            # Only create child chunks when the parent is large enough to
            # benefit from splitting. If parent content fits within a single
            # child, the child would be identical — skip to avoid duplicates
            # that waste embedding budget and add no retrieval value.
            if len(full_text) > self.max_child_size:
                children = self._split_into_children(
                    all_text_parts, file_id, parent_id, breadcrumb, level, title
                )
                chunks.extend(children)

        return chunks

    def _build_grouped_parent_chunks(
        self, section: Dict[str, Any], file_id: str, breadcrumb: str,
        level: int, title: str, dropdown_parts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        For sections with many dropdowns (e.g., 100 exceptions), create grouped
        sub-parent chunks that each fit within max_parent_size.

        Each sub-parent contains multiple dropdowns grouped to stay within limits.
        Child chunks reference their respective sub-parent for expansion.
        """
        chunks = []
        group_idx = 0
        current_group = []
        current_size = 0
        # Also include section elements (intro text) in the first group
        intro_parts = list(section["elements"])

        for dd_text in dropdown_parts:
            entry_size = len(dd_text) + 2  # +2 for "\n\n" separator
            if current_group and (current_size + entry_size) > self.max_parent_size:
                # Flush current group as a sub-parent
                self._flush_dropdown_group(
                    chunks, current_group, intro_parts if group_idx == 0 else [],
                    file_id, breadcrumb, level, title, group_idx
                )
                group_idx += 1
                current_group = []
                current_size = 0

            current_group.append(dd_text)
            current_size += entry_size

        # Flush remaining
        if current_group:
            self._flush_dropdown_group(
                chunks, current_group, intro_parts if group_idx == 0 else [],
                file_id, breadcrumb, level, title, group_idx
            )

        return chunks

    def _flush_dropdown_group(
        self, chunks: List[Dict[str, Any]], group_parts: List[str],
        intro_parts: List[str], file_id: str, breadcrumb: str,
        level: int, title: str, group_idx: int
    ):
        """Create a sub-parent chunk and its children for a group of dropdowns."""
        section_hash = hashlib.sha256(f"{file_id}_{title}_{group_idx}".encode()).hexdigest()[:8]
        parent_id = f"{file_id}_parent_{section_hash}"

        all_parts = intro_parts + group_parts if intro_parts else group_parts
        parent_content = "\n\n".join(all_parts)
        if len(parent_content) > self.max_parent_size:
            parent_content = parent_content[:self.max_parent_size]

        chunks.append({
            "element_id": parent_id,
            "content": parent_content,
            "type": "text",
            "chunk_type": "parent",
            "parent_chunk_id": None,
            "hierarchy_path": breadcrumb,
            "heading_level": level,
            "section_title": title,
        })

        # Create child chunks for this group
        children = self._split_into_children(
            all_parts, file_id, parent_id, breadcrumb, level, title
        )
        chunks.extend(children)

    def _build_grouped_parent_chunks_from_parts(
        self, text_parts: List[str], file_id: str, breadcrumb: str,
        level: int, title: str
    ) -> List[Dict[str, Any]]:
        """
        Split a long non-dropdown section into multiple sub-parent chunks
        so no content is lost when child→parent expansion delivers context.

        Groups text_parts into sub-parents that each fit within max_parent_size.
        """
        chunks = []
        group_idx = 0
        current_group = []
        current_size = 0

        for part in text_parts:
            if not part.strip():
                continue
            entry_size = len(part) + 2  # +2 for "\n\n" separator
            if current_group and (current_size + entry_size) > self.max_parent_size:
                # Flush current group as a sub-parent
                self._flush_part_group(
                    chunks, current_group, file_id, breadcrumb, level, title, group_idx
                )
                group_idx += 1
                current_group = []
                current_size = 0

            current_group.append(part)
            current_size += entry_size

        # Flush remaining
        if current_group:
            self._flush_part_group(
                chunks, current_group, file_id, breadcrumb, level, title, group_idx
            )

        return chunks

    def _flush_part_group(
        self, chunks: List[Dict[str, Any]], group_parts: List[str],
        file_id: str, breadcrumb: str, level: int, title: str, group_idx: int
    ):
        """Create a sub-parent chunk and its children for a group of text parts."""
        section_hash = hashlib.sha256(f"{file_id}_{title}_{group_idx}".encode()).hexdigest()[:8]
        parent_id = f"{file_id}_parent_{section_hash}"

        parent_content = "\n\n".join(group_parts)
        if len(parent_content) > self.max_parent_size:
            parent_content = parent_content[:self.max_parent_size]

        chunks.append({
            "element_id": parent_id,
            "content": parent_content,
            "type": "text",
            "chunk_type": "parent",
            "parent_chunk_id": None,
            "hierarchy_path": breadcrumb,
            "heading_level": level,
            "section_title": title,
        })

        # Create child chunks for this group
        children = self._split_into_children(
            group_parts, file_id, parent_id, breadcrumb, level, title
        )
        chunks.extend(children)

    def _split_into_children(
        self, text_parts: List[str], file_id: str, parent_id: str,
        breadcrumb: str, level: int, section_title: str
    ) -> List[Dict[str, Any]]:
        """
        Split section content into small child chunks for precise retrieval.

        Rules:
        - Procedure steps are kept as one atomic child
        - Individual paragraphs become children if short enough
        - Long paragraphs are split at sentence boundaries
        - Each child gets a section_title prefix for self-contained embedding
        - Consecutive sentence-based children share 1-sentence overlap to
          prevent information loss at chunk boundaries
        """
        children = []
        child_idx = 0
        # Section title prefix ensures children are self-contained for
        # embedding quality and fallback (when parent expansion fails).
        title_prefix = f"[{section_title}] " if section_title else ""

        for part in text_parts:
            if not part.strip():
                continue

            # If it's a procedure (contains numbered steps), keep it atomic
            if self._is_procedure_text(part):
                # Use max_child_size * 2 for procedures — keeps them reasonably
                # atomic while still fitting within retrieval precision bounds.
                max_proc = self.max_child_size * 2
                if len(part) <= max_proc:
                    # Fits — keep the whole procedure as one atomic child
                    child_id = f"{parent_id}_child_{child_idx}"
                    children.append({
                        "element_id": child_id,
                        "content": f"{title_prefix}{part}",
                        "type": "text",
                        "chunk_type": "child",
                        "parent_chunk_id": parent_id,
                        "hierarchy_path": breadcrumb,
                        "heading_level": level,
                        "section_title": section_title,
                    })
                    child_idx += 1
                else:
                    # Too long to keep atomic — split at line boundaries into
                    # multiple children so NO content is lost to truncation.
                    for sub in self._split_long_block(part, max_proc):
                        child_id = f"{parent_id}_child_{child_idx}"
                        children.append({
                            "element_id": child_id,
                            "content": f"{title_prefix}{sub}",
                            "type": "text",
                            "chunk_type": "child",
                            "parent_chunk_id": parent_id,
                            "hierarchy_path": breadcrumb,
                            "heading_level": level,
                            "section_title": section_title,
                        })
                        child_idx += 1
            elif len(part) <= self.max_child_size:
                # Short paragraph — single child
                child_id = f"{parent_id}_child_{child_idx}"
                children.append({
                    "element_id": child_id,
                    "content": f"{title_prefix}{part}",
                    "type": "text",
                    "chunk_type": "child",
                    "parent_chunk_id": parent_id,
                    "hierarchy_path": breadcrumb,
                    "heading_level": level,
                    "section_title": section_title,
                })
                child_idx += 1
            else:
                # Long text — split at sentence boundaries with 1-sentence overlap
                sentences = self._split_sentences(part)
                current = []
                current_len = 0
                # Track last sentence of previous chunk for overlap
                prev_last_sentence = ""

                for sent in sentences:
                    # If a single "sentence" exceeds child limit (e.g., a table
                    # or SQL block with no sentence boundaries), force-split it.
                    if len(sent) > self.max_child_size:
                        # Flush accumulated content first
                        if current:
                            child_text = " ".join(current)
                            child_id = f"{parent_id}_child_{child_idx}"
                            children.append({
                                "element_id": child_id,
                                "content": f"{title_prefix}{child_text}",
                                "type": "text",
                                "chunk_type": "child",
                                "parent_chunk_id": parent_id,
                                "hierarchy_path": breadcrumb,
                                "heading_level": level,
                                "section_title": section_title,
                            })
                            child_idx += 1
                            prev_last_sentence = current[-1] if current else ""
                            current = []
                            current_len = 0
                        # Force-split the oversized sentence
                        for sub in self._split_long_block(sent, self.max_child_size):
                            child_id = f"{parent_id}_child_{child_idx}"
                            children.append({
                                "element_id": child_id,
                                "content": f"{title_prefix}{sub}",
                                "type": "text",
                                "chunk_type": "child",
                                "parent_chunk_id": parent_id,
                                "hierarchy_path": breadcrumb,
                                "heading_level": level,
                                "section_title": section_title,
                            })
                            child_idx += 1
                        prev_last_sentence = ""
                        continue
                    if current_len + len(sent) > self.max_child_size and current:
                        child_text = " ".join(current)
                        child_id = f"{parent_id}_child_{child_idx}"
                        children.append({
                            "element_id": child_id,
                            "content": f"{title_prefix}{child_text}",
                            "type": "text",
                            "chunk_type": "child",
                            "parent_chunk_id": parent_id,
                            "hierarchy_path": breadcrumb,
                            "heading_level": level,
                            "section_title": section_title,
                        })
                        child_idx += 1
                        # Overlap: carry last sentence into next chunk for
                        # boundary context continuity
                        prev_last_sentence = current[-1] if current else ""
                        current = []
                        current_len = 0
                        if prev_last_sentence and len(prev_last_sentence) < self.max_child_size // 3:
                            current.append(prev_last_sentence)
                            current_len = len(prev_last_sentence)
                    current.append(sent)
                    current_len += len(sent)

                if current:
                    child_text = " ".join(current)
                    child_id = f"{parent_id}_child_{child_idx}"
                    children.append({
                        "element_id": child_id,
                        "content": f"{title_prefix}{child_text}",
                        "type": "text",
                        "chunk_type": "child",
                        "parent_chunk_id": parent_id,
                        "hierarchy_path": breadcrumb,
                        "heading_level": level,
                        "section_title": section_title,
                    })
                    child_idx += 1

        return children

    @staticmethod
    def _is_nav_bar(text: str) -> bool:
        """Detect in-page tab/back navigation bars that carry no real content.

        These look like '<< Back | Relationship | General | Reports | ...' and
        pollute embeddings and the BM25 index with repeated tab labels.
        """
        t = text.strip()
        if not t:
            return False
        # Direct navigation markers
        if t.startswith("<< Back") or t.startswith(">> Next"):
            return True
        # MadCap breadcrumb prefix
        if t.startswith("You are here:"):
            return True
        # Tab navigation bars: "<< Back | Tab1 | Tab2 | Tab3 | ..."
        # Heuristic: a line with 3+ pipe separators and under 300 chars is a tab bar
        if "|" in t and t.count("|") >= 3 and len(t) < 300:
            return True
        return False

    @staticmethod
    def _is_procedure_text(text: str) -> bool:
        """Check if text contains numbered procedural steps."""
        lines = text.strip().split("\n")
        numbered_lines = sum(1 for l in lines if re.match(r'^\d+\.\s', l.strip()))
        return numbered_lines >= 2

    @staticmethod
    def _split_long_block(text: str, max_size: int) -> List[str]:
        """Split a long structured block into pieces <= max_size.

        Splits on line boundaries first (to keep steps/rows intact); if a single
        line still exceeds max_size it is hard-split. Guarantees no content is
        dropped.
        """
        pieces: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in text.split("\n"):
            line_len = len(line) + 1
            if current and current_len + line_len > max_size:
                pieces.append("\n".join(current))
                current = []
                current_len = 0
            if len(line) > max_size:
                # Hard-split an oversized single line
                for i in range(0, len(line), max_size):
                    pieces.append(line[i:i + max_size])
                continue
            current.append(line)
            current_len += line_len
        if current:
            pieces.append("\n".join(current))
        return [p for p in pieces if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences at period/question/exclamation boundaries."""
        # Split on sentence-ending punctuation followed by space or newline
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p for p in parts if p.strip()]

    def _extract_image_refs(self, soup: BeautifulSoup, filepath: str) -> List[Dict[str, str]]:
        """Extract image references from the HTML for cross-linking."""
        from pathlib import Path
        images = []
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src", "")
            if not src:
                continue
            # Skip UI chrome images (toggle icons, etc.)
            if "transparent.gif" in src or "Stylesheets" in src:
                continue
            img_path = str((Path(filepath).parent / src).resolve())
            alt = img_tag.get("alt", "")
            images.append({
                "src": src,
                "resolved_path": img_path,
                "alt": alt,
            })
        return images
