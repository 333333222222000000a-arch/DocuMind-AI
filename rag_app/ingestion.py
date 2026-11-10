"""Safe upload persistence, PDF extraction, chunking, and corpus fingerprints."""

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    DUPLICATE_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    INDEX_SCHEMA_VERSION,
    MAX_PROMPT_TOKEN_BUDGET,
    MMR_LAMBDA,
    OCR_TEXT_THRESHOLD,
    RETRIEVAL_FETCH_K,
    TOP_K,
)


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str) -> str:
    """Return a single safe path component while retaining a useful PDF name."""
    basename = Path(str(name).replace("\\", "/")).name
    basename = _INVALID_FILENAME_CHARS.sub("_", basename).strip(" .")
    if not basename:
        basename = "document.pdf"
    if not basename.lower().endswith(".pdf"):
        basename += ".pdf"
    return basename


def save_uploaded_files(uploaded_files: Iterable, dest_dir: str) -> List[str]:
    """Save uploads under sanitized, unique names and return their full paths."""
    os.makedirs(dest_dir, exist_ok=True)
    saved_paths = []
    used_names = set()
    for uploaded_file in uploaded_files:
        filename = safe_filename(uploaded_file.name)
        stem, suffix = os.path.splitext(filename)
        candidate = filename
        serial = 1
        while candidate.casefold() in used_names or os.path.exists(os.path.join(dest_dir, candidate)):
            candidate = f"{stem}_{serial}{suffix}"
            serial += 1
        used_names.add(candidate.casefold())
        file_path = os.path.join(dest_dir, candidate)
        with open(file_path, "wb") as output:
            output.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def corpus_fingerprint(
    documents,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_separators=CHUNK_SEPARATORS,
    ocr_text_threshold: int = OCR_TEXT_THRESHOLD,
    index_schema_version: int = INDEX_SCHEMA_VERSION,
) -> str:
    """Hash document content, source labels, and index-relevant configuration."""
    digest = hashlib.sha256()
    digest.update(repr((
        embedding_model,
        chunk_size,
        chunk_overlap,
        tuple(chunk_separators),
        ocr_text_threshold,
        index_schema_version,
    )).encode("utf-8"))
    digest.update(b"\0")
    for document in documents:
        stable_metadata = {
            key: document.metadata.get(key)
            for key in ("source", "source_id", "chunk_id", "page", "page_number", "total_pages")
            if key in document.metadata
        }
        digest.update(json.dumps(stable_metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.page_content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def uploads_fingerprint(
    uploaded_files,
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_separators=CHUNK_SEPARATORS,
    ocr_text_threshold: int = OCR_TEXT_THRESHOLD,
    top_k: int = TOP_K,
    retrieval_fetch_k: int = RETRIEVAL_FETCH_K,
    mmr_lambda: float = MMR_LAMBDA,
    duplicate_similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    max_prompt_token_budget: int = MAX_PROMPT_TOKEN_BUDGET,
    index_schema_version: int = INDEX_SCHEMA_VERSION,
) -> str:
    """Hash upload bytes and all processing/retrieval settings for UI state."""
    digest = hashlib.sha256()
    digest.update(repr((
        embedding_model,
        chunk_size,
        chunk_overlap,
        tuple(chunk_separators),
        ocr_text_threshold,
        top_k,
        retrieval_fetch_k,
        mmr_lambda,
        duplicate_similarity_threshold,
        max_prompt_token_budget,
        index_schema_version,
    )).encode("utf-8"))
    digest.update(b"\0")
    for uploaded_file in uploaded_files:
        digest.update(str(uploaded_file.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes(uploaded_file.getbuffer()))
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def temporary_upload_directory():
    """Create an isolated upload directory and remove it after processing."""
    with tempfile.TemporaryDirectory(prefix="rag_chatbot_") as directory:
        yield directory


def load_documents_langchain(
    dir_path: str,
    *,
    source_names=None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_separators=CHUNK_SEPARATORS,
    ocr_text_threshold: int = OCR_TEXT_THRESHOLD,
    logger: Optional[Callable] = None,
    error_handler: Optional[Callable] = None,
):
    """Load PDFs, fall back to OCR for image-only files, and return source chunks."""
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    def log(message, level="INFO"):
        if logger:
            logger(message, level)

    docs = []
    for pdf_path in sorted(Path(dir_path).glob("*.pdf")):
        try:
            original_name = (source_names or {}).get(pdf_path.name, pdf_path.name)
            original_name = Path(str(original_name).replace("\\", "/")).name
            source_digest = hashlib.sha256()
            source_digest.update(original_name.encode("utf-8"))
            source_digest.update(b"\0")
            with pdf_path.open("rb") as pdf_file:
                for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                    source_digest.update(block)
            source_id = source_digest.hexdigest()

            pages = PyMuPDFLoader(str(pdf_path)).load()
            total_text = "".join(page.page_content for page in pages)
            if len(total_text.strip()) < ocr_text_threshold:
                log(f"Low text content detected in {pdf_path.name}, using OCR mode...")
                try:
                    pages = UnstructuredPDFLoader(
                        str(pdf_path), mode="elements", strategy="hi_res"
                    ).load()
                    log(f"OCR extraction completed for {pdf_path.name}")
                except Exception as ocr_error:
                    log(f"OCR failed for {pdf_path.name}: {ocr_error}", "WARNING")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=list(chunk_separators),
                length_function=len,
            )
            chunks = splitter.split_documents(pages)
            seen_page_text = set()
            loaded_chunk_count = 0
            for chunk_index, chunk in enumerate(chunks):
                metadata = {
                    key: value for key, value in chunk.metadata.items()
                    if key != "file_path"
                }
                page_identity = metadata.get("page_number", metadata.get("page", "unknown"))
                normalized_content = " ".join(
                    unicodedata.normalize("NFKC", chunk.page_content).casefold().split()
                )
                duplicate_key = (page_identity, normalized_content)
                if not normalized_content or duplicate_key in seen_page_text:
                    continue
                seen_page_text.add(duplicate_key)
                metadata["source"] = original_name
                metadata["source_id"] = source_id
                chunk_identity = hashlib.sha256(
                    f"{source_id}\0{page_identity}\0{chunk_index}\0".encode("utf-8")
                    + chunk.page_content.encode("utf-8")
                ).hexdigest()
                metadata["chunk_id"] = chunk_identity
                docs.append(Document(page_content=chunk.page_content, metadata=metadata))
                loaded_chunk_count += 1
            log(f"Created {loaded_chunk_count} chunks from {pdf_path.name}")
        except Exception as error:
            if error_handler:
                error_handler(f"Failed to load {pdf_path.name}: {error}")
            log(f"Error loading {pdf_path.name}: {error}", "ERROR")
    return docs
