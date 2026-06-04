"""
Intelligent Summarization System.
Kept as custom code — multi-level summaries + GPT-4 Vision + code analysis
is domain-specific logic that LangChain's generic summarisers don't cover.

Uses the raw OpenAI SDK for vision and structured prompts.
"""

from typing import Dict, Any, List
import logging
import base64
import io

from openai import OpenAI, AzureOpenAI
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum decoded image size in bytes to bother describing (skip tiny spacers)
_MIN_IMAGE_BYTES = 500

# Known base64 header prefixes → MIME types
_B64_SIGNATURES = {
    b"\x89PNG":    "image/png",
    b"\xff\xd8":   "image/jpeg",
    b"GIF8":       "image/gif",
    b"RIFF":       "image/webp",   # (WebP starts with RIFF)
    b"BM":         "image/bmp",
}


def _validate_and_normalise_b64(raw_b64: str) -> str | None:
    """Return a clean PNG base64 string, or None if the image is invalid/tiny."""
    if not raw_b64 or len(raw_b64) < 20:
        return None

    # Strip any data-uri prefix that might have leaked in
    if raw_b64.startswith("data:"):
        parts = raw_b64.split(",", 1)
        raw_b64 = parts[1] if len(parts) == 2 else raw_b64

    # Try decoding
    try:
        img_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception:
        # Retry with padding fix
        try:
            padded = raw_b64 + "=" * (-len(raw_b64) % 4)
            img_bytes = base64.b64decode(padded, validate=True)
        except Exception:
            return None

    if len(img_bytes) < _MIN_IMAGE_BYTES:
        return None

    # Determine format from magic bytes
    mime = None
    for sig, m in _B64_SIGNATURES.items():
        if img_bytes[:len(sig)] == sig:
            mime = m
            break

    # If already a valid PNG/JPEG/GIF/WebP, use it directly
    if mime in ("image/png", "image/jpeg"):
        return raw_b64

    # For anything else (BMP, GIF, WMF, EMF, unknown) → convert to PNG via Pillow
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


class IntelligentSummarizer:
    """
    Summarize content intelligently:
    - Text: Multi-level summaries (short / medium / full)
    - Images: Describe using GPT-4 Vision
    - Code: Extract structure and purpose
    - Tables: Compress to key insights
    """

    def __init__(self, client, model: str = "gpt-4.1", vision_model: str = None):
        """
        Args:
            client:       A raw OpenAI or AzureOpenAI client (from Config.get_raw_openai_client())
            model:        Chat model / deployment name
            vision_model: Vision model / deployment name (defaults to model)
        """
        self.client = client
        self.model = model
        self.vision_model = vision_model or model
        self.text_length_threshold = 1000

        self.stats = {
            "text_summarized": 0, "images_described": 0,
            "code_processed": 0, "tables_summarized": 0,
            "total_api_calls": 0,
        }
        logger.info(f"IntelligentSummarizer initialized (model={model})")

    # ── Text ──────────────────────────────────────────────────

    def should_summarize_text(self, text: str) -> bool:
        return len(text) > self.text_length_threshold

    def summarize_text(self, text: str) -> Dict[str, Any]:
        if not self.should_summarize_text(text):
            return {
                "short": text[:50] + "..." if len(text) > 50 else text,
                "medium": text[:500] + "..." if len(text) > 500 else text,
                "full": text,
                "was_summarized": False,
            }
        try:
            short_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise summarizer. Create extremely concise summaries."},
                    {"role": "user", "content": f"Summarize in ONE sentence (max 50 chars):\n\n{text[:2000]}"},
                ],
                max_tokens=20, temperature=0.3,
            )
            medium_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical summarizer. Preserve important details."},
                    {"role": "user", "content": f"Summarize in 2-3 sentences (max 500 chars), preserving key technical details:\n\n{text[:4000]}"},
                ],
                max_tokens=150, temperature=0.3,
            )
            self.stats["text_summarized"] += 1
            self.stats["total_api_calls"] += 2
            return {
                "short": short_resp.choices[0].message.content.strip(),
                "medium": medium_resp.choices[0].message.content.strip(),
                "full": text,
                "was_summarized": True,
            }
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return {"short": text[:50], "medium": text[:500], "full": text, "was_summarized": False}

    # ── Image ─────────────────────────────────────────────────

    def describe_image(self, image_base64: str, context: str = "") -> Dict[str, str]:
        # Validate and normalise to clean PNG/JPEG base64
        clean_b64 = _validate_and_normalise_b64(image_base64)
        if clean_b64 is None:
            return {"short": "Image (skipped)", "full": "Image skipped — invalid or too small", "was_described": False}

        try:
            prompt = (
                "Describe this image in detail. Include:\n"
                "1. What the image shows\n2. Key visual elements\n"
                "3. Any text or labels visible\n4. Technical details (if diagram/chart)"
            )
            if context:
                prompt += f"\n\nContext: {context}"

            # Detect MIME from the (possibly re-encoded) bytes
            raw_bytes = base64.b64decode(clean_b64[:16] + "==")  # just peek at header
            mime = "image/png"
            if raw_bytes[:2] == b"\xff\xd8":
                mime = "image/jpeg"

            resp = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{clean_b64}"}},
                    ],
                }],
                max_tokens=500, temperature=0.3,
            )
            description = resp.choices[0].message.content.strip()

            short_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Create a 1-sentence description (max 50 chars)."},
                    {"role": "user", "content": f"Summarize: {description}"},
                ],
                max_tokens=20, temperature=0.3,
            )
            self.stats["images_described"] += 1
            self.stats["total_api_calls"] += 2
            return {"short": short_resp.choices[0].message.content.strip(), "full": description, "was_described": True}
        except Exception as e:
            logger.error(f"Error describing image: {e}")
            return {"short": "Image", "full": f"Image (description failed: {e})", "was_described": False}

    # ── Code ──────────────────────────────────────────────────

    def extract_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code analyzer. Be concise."},
                    {"role": "user", "content": (
                        f"Analyze this {language} code and provide:\n"
                        "1. Main purpose (1 sentence)\n2. Key functions/classes (list)\n"
                        f"3. Dependencies (list)\n\nCode:\n{code[:2000]}"
                    )},
                ],
                max_tokens=200, temperature=0.3,
            )
            self.stats["code_processed"] += 1
            self.stats["total_api_calls"] += 1
            return {
                "short": f"{language} code",
                "structure": resp.choices[0].message.content.strip(),
                "full": code, "language": language,
                "lines": len(code.split("\n")), "was_analyzed": True,
            }
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {
                "short": f"{language} code", "structure": "Analysis failed",
                "full": code, "language": language,
                "lines": len(code.split("\n")), "was_analyzed": False,
            }

    # ── Table ─────────────────────────────────────────────────

    def summarize_table(self, table_content: str) -> Dict[str, str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst. Summarize tables clearly."},
                    {"role": "user", "content": (
                        "Summarize this table in 2-3 sentences. Include:\n"
                        "1. What the table shows\n2. Key data points or patterns\n"
                        f"3. Any important values\n\nTable:\n{table_content[:2000]}"
                    )},
                ],
                max_tokens=150, temperature=0.3,
            )
            summary = resp.choices[0].message.content.strip()
            self.stats["tables_summarized"] += 1
            self.stats["total_api_calls"] += 1
            return {"short": summary.split(".")[0][:50], "medium": summary, "full": table_content, "was_summarized": True}
        except Exception as e:
            logger.error(f"Error summarizing table: {e}")
            return {"short": "Table", "medium": table_content[:500], "full": table_content, "was_summarized": False}

    # ── Generic element processor ─────────────────────────────

    def process_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        etype = element.get("type")
        content = element.get("content", "")

        if etype == "text":
            s = self.summarize_text(content)
            element["summary_short"] = s["short"]
            element["summary_medium"] = s["medium"]
            element["summary_full"] = s["full"]
            element["was_summarized"] = s["was_summarized"]

        elif etype == "image":
            # Collect the best available raw base64
            raw_b64 = element.get("image_base64") or element.get("image_data") or ""
            # If not set, check if content looks like base64
            if not raw_b64 and content and content[:5] in ("iVBOR", "/9j/A", "AAAA+", "R0lGO", "Qk0A+"):
                raw_b64 = content

            # Validate / normalise → clean PNG or JPEG base64 (or None)
            clean_b64 = _validate_and_normalise_b64(raw_b64)
            if clean_b64 is None:
                # Image is too small, corrupt, or unsupported — skip VLM
                element["content"] = f"Image: {element.get('filename', 'unknown')} (skipped — invalid or tiny)"
                element["summary_short"] = "Image (skipped)"
                element["summary_full"] = element["content"]
                element["was_described"] = False
                element.pop("image_base64", None)
            else:
                element["image_base64"] = clean_b64  # store validated version
                d = self.describe_image(clean_b64)
                element["content"] = d["full"]  # VLM description becomes searchable content
                element["summary_short"] = d["short"]
                element["summary_full"] = d["full"]
                element["was_described"] = d["was_described"]

        elif etype == "code":
            a = self.extract_code_structure(content, element.get("language", "unknown"))
            element["summary_short"] = a["short"]
            element["structure_analysis"] = a["structure"]
            element["was_analyzed"] = a["was_analyzed"]

        elif etype == "table":
            s = self.summarize_table(content)
            element["summary_short"] = s["short"]
            element["summary_medium"] = s["medium"]
            element["was_summarized"] = s["was_summarized"]

        return element

    def process_elements(self, elements: List[Dict[str, Any]], images_only: bool = False) -> List[Dict[str, Any]]:
        """Summarize elements.

        Args:
            elements:     List of document elements.
            images_only:  If True, only run VLM on image elements — text/code/table
                          are passed through unchanged (much faster, ~$62 vs $2,600).
        """
        mode = "images-only VLM" if images_only else "full"
        logger.info(f"Summarizing {len(elements)} elements (mode={mode}) …")
        processed = []
        for idx, el in enumerate(elements, 1):
            if idx % 100 == 0:
                logger.info(f"  {idx}/{len(elements)}")
            if images_only and el.get("type") != "image":
                processed.append(el)  # skip non-image — embed raw text
            else:
                processed.append(self.process_element(el))

        logger.info(
            f"Done — text:{self.stats['text_summarized']}  img:{self.stats['images_described']}  "
            f"code:{self.stats['code_processed']}  table:{self.stats['tables_summarized']}  "
            f"API calls:{self.stats['total_api_calls']}"
        )
        return processed
