"""
Azure Blob Storage client for uploading and serving RAG images.

Uploads image bytes to Azure Blob Storage during ingestion and
returns public/SAS URLs for serving images in the UI.
"""

import base64
import logging
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContentSettings

logger = logging.getLogger(__name__)


class ImageBlobStore:
    """Upload images to Azure Blob Storage and return accessible URLs."""

    def __init__(self, connection_string: str, container_name: str = "rag-images"):
        self.container_name = container_name
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._client.get_container_client(container_name)
        self._ensure_container()
        logger.info(f"ImageBlobStore connected to container '{container_name}'")

    def _ensure_container(self):
        """Create the container if it doesn't exist."""
        try:
            self._container.get_container_properties()
        except Exception:
            self._container.create_container()
            logger.info(f"Created blob container '{self.container_name}'")

    def upload_image(self, element_id: str, image_base64: str, fmt: str = "png") -> Optional[str]:
        """
        Upload a base64-encoded image to blob storage.

        Returns the blob URL on success, None on failure.
        """
        if not image_base64:
            return None

        blob_name = f"{element_id}.{fmt}"
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            logger.warning(f"Invalid base64 for {element_id}: {e}")
            return None

        content_type = f"image/{fmt}" if fmt != "jpg" else "image/jpeg"

        try:
            blob_client = self._container.get_blob_client(blob_name)
            blob_client.upload_blob(
                image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            return blob_client.url
        except Exception as e:
            logger.error(f"Failed to upload {blob_name}: {e}")
            return None

    def get_image_url(self, element_id: str, fmt: str = "png") -> Optional[str]:
        """Get the URL for an already-uploaded image."""
        blob_name = f"{element_id}.{fmt}"
        try:
            blob_client = self._container.get_blob_client(blob_name)
            blob_client.get_blob_properties()  # verify existence
            return blob_client.url
        except Exception:
            return None

    def download_image(self, element_id: str, fmt: str = "png") -> Optional[tuple]:
        """Download image bytes from blob storage.

        Returns (bytes, content_type) on success, None on failure.
        """
        blob_name = f"{element_id}.{fmt}"
        try:
            blob_client = self._container.get_blob_client(blob_name)
            download = blob_client.download_blob()
            content_type = f"image/{fmt}" if fmt != "jpg" else "image/jpeg"
            return download.readall(), content_type
        except Exception:
            # Try alternate formats
            for alt_fmt in ("png", "jpg", "gif"):
                if alt_fmt == fmt:
                    continue
                try:
                    alt_blob = f"{element_id}.{alt_fmt}"
                    blob_client = self._container.get_blob_client(alt_blob)
                    download = blob_client.download_blob()
                    ct = f"image/{alt_fmt}" if alt_fmt != "jpg" else "image/jpeg"
                    return download.readall(), ct
                except Exception:
                    continue
            return None

    def health_check(self) -> dict:
        """Check blob storage connectivity."""
        try:
            props = self._container.get_container_properties()
            return {"status": "healthy", "container": self.container_name, "last_modified": str(props.last_modified)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
