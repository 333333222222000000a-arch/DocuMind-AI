"""FAISS indexing with a JSON document store and no pickle deserialization."""

import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    INDEX_SCHEMA_VERSION,
    DUPLICATE_SIMILARITY_THRESHOLD,
    MMR_LAMBDA,
    OCR_TEXT_THRESHOLD,
    RETRIEVAL_FETCH_K,
)
from .ingestion import corpus_fingerprint


def _artifact_digest(index_path: Path, records_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (index_path, records_path):
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_or_load_faiss(
    documents,
    index_dir=INDEX_DIR,
    force_rebuild: bool = False,
    *,
    embeddings=None,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_separators=CHUNK_SEPARATORS,
    ocr_text_threshold: int = OCR_TEXT_THRESHOLD,
    expected_fingerprint: Optional[str] = None,
    logger=None,
    ui=None,
):
    """Load only matching, integrity-checked app indexes; otherwise rebuild.

    The vector index is read through FAISS's native index reader. Documents and
    ID mappings use JSON; no pickle or LangChain ``load_local`` path is used.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_core.documents import Document
    import faiss

    if embeddings is None:
        from .embeddings import get_embeddings
        embeddings = get_embeddings(embedding_model)
    fingerprint = corpus_fingerprint(
        documents,
        embedding_model,
        chunk_size,
        chunk_overlap,
        chunk_separators,
        ocr_text_threshold,
        INDEX_SCHEMA_VERSION,
    )
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise ValueError("Provided fingerprint does not match the supplied documents")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Index fingerprint must be a SHA-256 hex digest")
    root = Path(index_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entry = root / fingerprint
    index_path = entry / "index.faiss"
    records_path = entry / "documents.json"
    manifest_path = entry / "manifest.json"

    if not force_rebuild and index_path.is_file() and records_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("format") == 1
                and manifest.get("schema_version") == INDEX_SCHEMA_VERSION
                and manifest.get("fingerprint") == fingerprint
                and manifest.get("embedding_model") == embedding_model
                and manifest.get("artifact_sha256") == _artifact_digest(index_path, records_path)
            ):
                records = json.loads(records_path.read_text(encoding="utf-8"))
                if len(records) == len(documents):
                    stored_docs = {
                        record["id"]: Document(
                            page_content=record["page_content"], metadata=record["metadata"]
                        )
                        for record in records
                    }
                    id_map = {int(i): record["id"] for i, record in enumerate(records)}
                    db = FAISS(
                        embedding_function=embeddings,
                        index=faiss.read_index(str(index_path)),
                        docstore=InMemoryDocstore(stored_docs),
                        index_to_docstore_id=id_map,
                    )
                    if logger:
                        logger("Loaded verified application FAISS index")
                    if ui:
                        ui.success("&#9989; Loaded existing FAISS index")
                    return db
        except Exception as error:
            if logger:
                logger(f"Stored FAISS index rejected; rebuilding: {error}", "WARNING")

    if logger:
        logger(f"Building FAISS index from {len(documents)} documents")
    spinner = ui.spinner("&#128296; Building FAISS index...") if ui else nullcontext()
    with spinner:
        db = FAISS.from_documents(documents, embeddings)
    entry.mkdir(parents=True, exist_ok=True)
    faiss.write_index(db.index, str(index_path))
    records = []
    for position, doc_id in sorted(db.index_to_docstore_id.items()):
        document = db.docstore.search(doc_id)
        records.append({
            "id": doc_id,
            "page_content": document.page_content,
            "metadata": document.metadata,
        })
    records_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "format": 1,
        "schema_version": INDEX_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "embedding_model": embedding_model,
        "artifact_sha256": _artifact_digest(index_path, records_path),
    }, indent=2), encoding="utf-8")
    if logger:
        logger("FAISS index built and saved")
    if ui:
        ui.success("&#9989; FAISS index built and saved")
    return db


def make_retriever(vector_store, top_k: int = 10):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": max(RETRIEVAL_FETCH_K, top_k),
            "lambda_mult": MMR_LAMBDA,
        },
    )


def deduplicate_ranked_documents(
    documents,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
):
    """Drop repeated/near-identical chunks from the same source page only.

    Same-content passages from different PDFs are retained because each has a
    distinct provenance and may need a separate citation.
    """
    kept = []
    fingerprints = {}
    for document in documents:
        metadata = document.metadata
        source_identity = metadata.get("source_id") or metadata.get("source") or "unknown"
        page_identity = metadata.get("page_number", metadata.get("page", "unknown"))
        identity = (source_identity, page_identity)
        normalized = " ".join(
            unicodedata.normalize("NFKC", document.page_content).casefold().split()
        )
        if not normalized:
            continue
        candidates = fingerprints.setdefault(identity, set())
        if normalized in candidates:
            continue
        if any(SequenceMatcher(None, normalized, previous).ratio() >= similarity_threshold
               for previous in candidates):
            continue
        candidates.add(normalized)
        kept.append(document)
    return kept
