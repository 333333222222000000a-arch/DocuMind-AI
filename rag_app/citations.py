"""Formatting helpers for source citations and retrieved context previews."""

import html
import re
import unicodedata


def display_page_number(metadata):
    """Return a display page, converting PyMuPDF's zero-based ``page`` value."""
    def parse_integer(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    value = metadata.get("page_number")
    if value is not None:
        page = parse_integer(value)
        if page is not None and page > 0:
            return page

    value = metadata.get("page")
    if value is not None:
        page_index = parse_integer(value)
        if page_index is not None and page_index >= 0:
            return page_index + 1
    return None


_display_page_number = display_page_number


def _markdown_escape(value):
    return re.sub(r"([\\`*_{}\[\]()#+\-!|>])", r"\\\1", str(value))


def _direction_isolate(value):
    """Isolate only text containing RTL characters to keep LTR output stable."""
    value = str(value)
    if re.search(r"[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]", value):
        return f"\u2068{value}\u2069"
    return value


def format_citations(source_docs, snippet_length: int = 200) -> str:
    if not source_docs:
        return ""
    text = "\n\n**&#128218; Sources:**\n"
    for index, document in enumerate(source_docs, 1):
        citation_id = document.metadata.get("citation_id", f"S{index}")
        source = document.metadata.get("source")
        source_label = _markdown_escape(source) if source else "filename unavailable"
        page_number = display_page_number(document.metadata)
        page_label = f"page {page_number}" if page_number is not None else "page not available"
        content = document.page_content
        preview = content[:snippet_length] + ("..." if len(content) > snippet_length else "")
        display_preview = " ".join(unicodedata.normalize("NFKC", preview).split())
        text += (
            f"\n**[{citation_id}]** {_direction_isolate(source_label)} — {page_label}\n"
            f"> {_direction_isolate(html.escape(display_preview))}\n"
        )
    return text


def format_debug_chunks(source_docs, chunk_length: int = 500) -> str:
    if not source_docs:
        return ""
    parts = [f"\n\n|||DEBUG_CHUNKS:{len(source_docs)}"]
    for index, document in enumerate(source_docs, 1):
        parts.append(f"\n|||CHUNK{index}:{document.page_content[:chunk_length].replace(chr(10), ' ')}")
    return "".join(parts)


def split_assistant_message(message: str) -> dict:
    """Separate answer, citations, diagnostics, and model label without loss.

    Historical app messages place the model footer before the citation section,
    so parsing the footer must preserve any text that follows its first line.
    """
    content = str(message)
    debug_chunks = []
    debug_marker = "|||DEBUG_CHUNKS:"
    if debug_marker in content:
        content, debug_text = content.split(debug_marker, 1)
        for chunk in debug_text.split("|||CHUNK"):
            if ":" in chunk:
                chunk_number, chunk_content = chunk.split(":", 1)
                debug_chunks.append(f"**Chunk {chunk_number}**: {chunk_content.strip()}")

    footer_text = ""
    footer_marker = "|||FOOTER:"
    if footer_marker in content:
        before_footer, footer_and_tail = content.split(footer_marker, 1)
        footer_line, separator, tail = footer_and_tail.partition("\n")
        footer_text = footer_line.strip()
        content = before_footer + (separator + tail if separator else "")

    sources_marker = "**&#128218; Sources:**"
    source_text = ""
    if sources_marker in content:
        content, source_text = content.split(sources_marker, 1)

    return {
        "answer": content.strip(),
        "sources": source_text.strip(),
        "debug_chunks": debug_chunks,
        "footer": footer_text,
    }
