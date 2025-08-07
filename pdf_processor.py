"""
PDF text extraction processor using PyMuPDF and PyPDF2 as a fallback.
"""
import tempfile
import os
import logging
from typing import BinaryIO, Dict

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from app.config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    def extract_text(self, file_buffer: BinaryIO, filename: str) -> Dict:
        try:
            file_buffer.seek(0, os.SEEK_END)
            size = file_buffer.tell()
            file_buffer.seek(0)
            if size > Config.MAX_FILE_SIZE:
                raise ValueError(f"File size exceeds limit: {size / (1024 * 1024):.2f} MB")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_buffer.read())
                tmp_path = tmp_file.name

            try:
                text = self._extract_with_pymupdf(tmp_path)
                if not text.strip():
                    # fallback
                    file_buffer.seek(0)
                    text = self._extract_with_pypdf2(file_buffer)

                metadata = self._extract_metadata(tmp_path)

                return {
                    'text': text,
                    'filename': filename,
                    'page_count': metadata.get("page_count", 0),
                    'word_count': len(text.split()),
                    'title': metadata.get("title") or filename
                }
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise RuntimeError(f"Failed to process PDF: {e}")

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return ""

    def _extract_with_pypdf2(self, file_buffer: BinaryIO) -> str:
        try:
            reader = PdfReader(file_buffer)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""

    def _extract_metadata(self, pdf_path: str) -> Dict:
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            doc.close()
            return {
                "page_count": len(doc),
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", "")
            }
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            return {}
