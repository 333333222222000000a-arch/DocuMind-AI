import streamlit as st
import os
import logging
import datetime
import html

from rag_app.answering import answer_question
from rag_app.citations import (
    format_citations,
    format_debug_chunks,
    split_assistant_message,
)
from rag_app.config import (
    CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE, EMBEDDING_MODEL_NAME,
    GROQ_MODELS, INDEX_DIR, MAX_PROMPT_TOKEN_BUDGET, OCR_TEXT_THRESHOLD, PROMPT_TEMPLATE, TOP_K,
)
from rag_app.embeddings import get_embeddings
from rag_app.ingestion import (
    load_documents_langchain as _load_documents,
    save_uploaded_files,
    temporary_upload_directory,
    uploads_fingerprint,
)
from rag_app.providers import (
    GROQ_AVAILABLE, OPENAI_AVAILABLE, get_api_key, get_groq_llm_instance,
    get_openai_llm_instance, provider_error_status, provider_error_summary,
)
from rag_app.vector_store import build_or_load_faiss, make_retriever

# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message: str, level: str = "INFO"):
    """Log event to file and session state"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
        
    if "app_logs" not in st.session_state:
        st.session_state["app_logs"] = []
    
    # Use HTML entities for emojis in logs to avoid syntax errors
    emoji = '&#128308;' if level=='ERROR' else ('&#9888;' if level=='WARNING' else '&#128313;')
    st.session_state["app_logs"].insert(0, f"{emoji} {log_entry}")
    
    if len(st.session_state["app_logs"]) > 100:
        st.session_state["app_logs"] = st.session_state["app_logs"][:100]

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Shared application shell and responsive SaaS styling.
st.markdown("""
<style>
    :root { color-scheme: light; }
    .stApp, [data-testid="stAppViewContainer"] > .main { background: #f5f7fb; color: #182338; }
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] { color: #34445c; }
    [data-testid="stAppViewContainer"] .main textarea,
    [data-testid="stAppViewContainer"] .main input { background-color: #fff; color: #182338; }
    .block-container { max-width: 1320px; padding: 1.6rem clamp(1rem, 3vw, 2.8rem) 3rem; }
    [data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid #26344b; }
    [data-testid="stSidebar"] * { color: #dbe4f2; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f6f8fc; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebbd0; }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: .25rem; }
    [data-testid="stSidebar"] .stRadio label { background: transparent; border-radius: 9px; padding: .35rem .55rem; }
    [data-testid="stSidebar"] .stRadio label:has(input:checked) { background: #243653; }
    h1, h2, h3 { letter-spacing: -.02em; color: #17243a; }
    h1 { font-size: clamp(1.75rem, 3vw, 2.3rem) !important; margin: 0 0 .3rem; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.08rem !important; }
    p, li, label { line-height: 1.6; }
    .stButton button, .stDownloadButton button { min-height: 2.55rem; border-radius: 9px; font-weight: 600; }
    .stButton button[kind="primary"] { background: #365fce; border-color: #365fce; color: white; }
    [data-testid="stFileUploader"] { background: #fff; border: 1px dashed #aab8ce; border-radius: 13px; padding: .8rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e1e7f0; border-radius: 13px; padding: 1rem 1.05rem; box-shadow: 0 2px 8px rgba(20, 38, 68, .035); }
    [data-testid="stMetricLabel"] { color: #62718a; }
    [data-testid="stMetricValue"] { color: #17243a; font-size: 1.8rem; }
    [data-testid="stExpander"] { background: #fff; border: 1px solid #e1e7f0; border-radius: 11px; }
    [data-testid="stChatMessage"] { background: transparent; border: 0; padding: .25rem 0; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { unicode-bidi: plaintext; text-align: start; overflow-wrap: anywhere; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] { background: #eaf0ff; border: 1px solid #d9e3fb; border-radius: 12px; padding: .9rem 1rem; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] { background: #fff; border: 1px solid #e1e7f0; border-radius: 12px; padding: .95rem 1.05rem; box-shadow: 0 2px 8px rgba(20, 38, 68, .035); }
    .dm-brand { color: #f6f8fc; font-size: 1.2rem; font-weight: 750; letter-spacing: -.02em; padding: .4rem .1rem; }
    .dm-brand-mark { display: inline-grid; place-items: center; width: 2rem; height: 2rem; margin-right: .55rem; border-radius: 9px; color: white; background: #466fe0; font-weight: 800; }
    .dm-brand-tagline { color: #aebbd0; font-size: .76rem; letter-spacing: .01em; margin: -.15rem 0 .7rem .15rem; }
    .dm-header-tagline { color: #65758c; font-size: .82rem; margin: -.1rem 0 .45rem; }
    .dm-kicker { color: #657897; font-size: .78rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
    .dm-page-header { padding: .2rem 0 1.25rem; border-bottom: 1px solid #e3e8f0; margin-bottom: 1.25rem; }
    .dm-page-description { color: #687790; font-size: .97rem; margin: .2rem 0 0; }
    .dm-card { background: #fff; border: 1px solid #e1e7f0; border-radius: 13px; padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(20, 38, 68, .035); }
    .dm-card-title { color: #1c2b44; font-weight: 700; margin-bottom: .3rem; }
    .dm-card-copy { color: #687790; font-size: .91rem; line-height: 1.55; overflow-wrap: anywhere; }
    .dm-status { display: flex; align-items: center; gap: .5rem; color: #dbe4f2; font-size: .88rem; margin: .28rem 0; }
    .dm-dot { width: .55rem; height: .55rem; display: inline-block; border-radius: 50%; background: #92a0b4; }
    .dm-dot.ok { background: #42c58a; }
    .dm-dot.warn { background: #e9b949; }
    .dm-dot.off { background: #8c99ab; }
    .dm-chat-input { background: white; border: 1px solid #dfe6f0; border-radius: 13px; padding: .75rem; }
    @media (max-width: 760px) {
        .block-container { padding: 1rem .8rem 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="dm-brand"><span class="dm-brand-mark">D</span>DocuMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="dm-brand-tagline">AI-Powered Document Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Navigation")
    selected_page = st.radio(
        "Choose a section",
        ["Overview", "Chat", "Documents", "About", "Diagnostics"],
        key="main_navigation",
        label_visibility="collapsed",
    )
    st.markdown("---")
    groq_key = get_api_key("GROQ_API_KEY")
    openai_key = get_api_key("OPENAI_API_KEY")
    has_groq = bool(groq_key)
    has_openai = bool(openai_key)
    use_groq = st.checkbox("Prefer Groq", value=GROQ_AVAILABLE and has_groq)
    force_rebuild = st.button("Rebuild document index", help="Reprocess current PDFs and rebuild their index", use_container_width=True)
    st.markdown("#### Provider status")
    for name, available, configured in (("Groq", GROQ_AVAILABLE, has_groq), ("OpenAI", OPENAI_AVAILABLE, has_openai)):
        dot_class = "off" if not available or not configured else "ok"
        status = "Integration unavailable" if not available else ("Key configured · not verified" if configured else "Not configured")
        st.markdown(f'<div class="dm-status"><span class="dm-dot {dot_class}"></span><span><strong>{name}</strong> · {html.escape(status)}</span></div>', unsafe_allow_html=True)
    st.caption("A configured key does not confirm a live provider connection.")
    if st.session_state.get("messages"):
        history_lines = [f"DocuMind AI conversation · {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", ""]
        for role, content in st.session_state["messages"]:
            if role == "assistant":
                parsed = split_assistant_message(content)
                clean_content = parsed["answer"]
                if parsed["sources"]:
                    clean_content += "\n\n**&#128218; Sources:**\n" + parsed["sources"]
            else:
                clean_content = content
            history_lines.extend([f"{role.upper()}:", clean_content, ""])
        st.download_button(
            "Download chat history",
            "\n".join(history_lines),
            file_name=f"documind_chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    st.markdown("---")

page_descriptions = {
    "Overview": "A clear view of your document workspace and current session.",
    "Chat": "Ask questions grounded in your PDFs and inspect cited evidence.",
    "Documents": "Upload, review, and remove PDFs in the current session.",
    "About": "How DocuMind AI works, its technology, and project attribution.",
    "Diagnostics": "Provider configuration, retrieval settings, and application logs.",
}
st.markdown(f"""
<div class="dm-page-header">
  <div class="dm-kicker">DocuMind AI</div>
  <div class="dm-header-tagline">AI-Powered Document Intelligence</div>
  <h1>{html.escape(selected_page)}</h1>
  <div class="dm-page-description">{html.escape(page_descriptions[selected_page])}</div>
</div>
""", unsafe_allow_html=True)

# UI adapters preserve app.py's existing entry point while keeping pipeline
# implementations in the independently testable rag_app package.
def load_documents_langchain(dir_path, source_names=None):
    return _load_documents(
        dir_path,
        source_names=source_names,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunk_separators=CHUNK_SEPARATORS,
        ocr_text_threshold=OCR_TEXT_THRESHOLD,
        logger=log_event,
        error_handler=st.error,
    )


@st.cache_resource
def cached_embeddings(model_name):
    return get_embeddings(model_name)


def build_index(docs, index_dir, force_rebuild=False):
    return build_or_load_faiss(
        docs,
        index_dir,
        force_rebuild,
        embeddings=cached_embeddings(EMBEDDING_MODEL_NAME),
        embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunk_separators=CHUNK_SEPARATORS,
        ocr_text_threshold=OCR_TEXT_THRESHOLD,
        logger=log_event,
        ui=st,
    )

def clear_document_session():
    """Discard derived index/chat state when the uploaded PDF set changes."""
    st.session_state.pop("retriever", None)
    st.session_state.pop("processed_fingerprint", None)
    st.session_state.pop("indexed_document_count", None)
    st.session_state.pop("indexed_chunk_count", None)
    st.session_state["messages"] = []


def clear_document_session():
    """Discard derived index/chat state when the uploaded PDF set changes."""
    st.session_state.pop("retriever", None)
    st.session_state.pop("processed_fingerprint", None)
    st.session_state.pop("indexed_document_count", None)
    st.session_state.pop("indexed_chunk_count", None)
    st.session_state["messages"] = []


def navigate_to(page):
    st.session_state["main_navigation"] = page


# Keep the uploader mounted on every page so Streamlit preserves its widget state.
with st.expander("Add or remove PDF documents", expanded=selected_page == "Documents"):
    uploaded_files = st.file_uploader(
        "Drop PDF files here or browse to upload",
        accept_multiple_files=True,
        type=["pdf"],
        help="Upload one or more PDF documents for question answering.",
        on_change=clear_document_session,
        key="document_uploads",
    )

# Indexing remains the same pipeline and runs whenever the upload fingerprint changes.
if uploaded_files:
    current_upload_fingerprint = uploads_fingerprint(
        uploaded_files,
        embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunk_separators=CHUNK_SEPARATORS,
        ocr_text_threshold=OCR_TEXT_THRESHOLD,
        top_k=TOP_K,
    )
    try:
        if force_rebuild or st.session_state.get("processed_fingerprint") != current_upload_fingerprint:
            log_event(f"Processing {len(uploaded_files)} uploaded files")
            with temporary_upload_directory() as temp_dir:
                saved_paths = save_uploaded_files(uploaded_files, temp_dir)
                source_names = {
                    os.path.basename(path): os.path.basename(upload.name)
                    for path, upload in zip(saved_paths, uploaded_files)
                }
                with st.spinner("Extracting PDF text and preparing the document index…"):
                    docs = load_documents_langchain(temp_dir, source_names=source_names)
                    if not docs:
                        st.error("No readable content was found in the uploaded PDFs.")
                        st.stop()
                    db = build_index(docs, INDEX_DIR, force_rebuild)
            st.session_state["retriever"] = make_retriever(db, TOP_K)
            st.session_state["processed_fingerprint"] = current_upload_fingerprint
            st.session_state["indexed_document_count"] = len(uploaded_files)
            st.session_state["indexed_chunk_count"] = len(docs)
            st.rerun()
    except Exception as e:
        error_msg = provider_error_summary(e)
        st.error(f"Document processing failed: {error_msg}")
        log_event(f"Document processing error ({type(e).__name__}): {error_msg}", "ERROR")
else:
    uploaded_files = []

if selected_page == "Overview":
    current_files = uploaded_files or []
    current_fingerprint = uploads_fingerprint(
        current_files, embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        chunk_separators=CHUNK_SEPARATORS, ocr_text_threshold=OCR_TEXT_THRESHOLD,
        top_k=TOP_K,
    ) if current_files else None
    indexed_count = len(current_files) if current_files and st.session_state.get("processed_fingerprint") == current_fingerprint else 0
    chunk_count = st.session_state.get("indexed_chunk_count", 0) if indexed_count else 0
    question_count = sum(1 for role, _ in st.session_state.get("messages", []) if role == "user")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Uploaded PDFs", len(current_files))
    col2.metric("Indexed PDFs", indexed_count)
    col3.metric("Indexed chunks", chunk_count)
    col4.metric("Questions this session", question_count)

    st.markdown("### Get started")
    action1, action2 = st.columns(2)
    with action1:
        st.markdown('<div class="dm-card"><div class="dm-card-title">Add documents</div><div class="dm-card-copy">Upload PDFs in the panel above, review their status, and remove files from the uploader when needed.</div></div>', unsafe_allow_html=True)
        st.button("Go to Documents", on_click=navigate_to, args=("Documents",), type="primary", key="overview_documents")
    with action2:
        st.markdown('<div class="dm-card"><div class="dm-card-title">Ask a question</div><div class="dm-card-copy">Open Chat after indexing to ask about the documents in this session.</div></div>', unsafe_allow_html=True)
        st.button("Open Chat", on_click=navigate_to, args=("Chat",), key="overview_chat")

    st.markdown("### System configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        st.markdown(f'<div class="dm-card"><div class="dm-card-title">Embedding model</div><div class="dm-card-copy">{html.escape(EMBEDDING_MODEL_NAME.split("/")[-1])}</div></div>', unsafe_allow_html=True)
    with cfg2:
        provider_name = "Groq" if use_groq and GROQ_AVAILABLE and has_groq else ("OpenAI" if OPENAI_AVAILABLE and has_openai else "Not configured")
        st.markdown(f'<div class="dm-card"><div class="dm-card-title">Preferred answer provider</div><div class="dm-card-copy">{html.escape(provider_name)} · key configuration only, live connection not verified</div></div>', unsafe_allow_html=True)
    with cfg3:
        st.markdown(f'<div class="dm-card"><div class="dm-card-title">Retrieval settings</div><div class="dm-card-copy">Top-K {TOP_K} · chunk size {CHUNK_SIZE} · context budget {MAX_PROMPT_TOKEN_BUDGET:,}</div></div>', unsafe_allow_html=True)

elif selected_page == "Documents":
    st.markdown("### Document library")
    current_files = uploaded_files or []
    if not current_files:
        st.info("No PDFs uploaded yet. Add one or more files using the uploader above.")
    else:
        current_fingerprint = uploads_fingerprint(
            current_files, embedding_model=EMBEDDING_MODEL_NAME,
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            chunk_separators=CHUNK_SEPARATORS, ocr_text_threshold=OCR_TEXT_THRESHOLD,
            top_k=TOP_K,
        )
        is_indexed = st.session_state.get("processed_fingerprint") == current_fingerprint
        if is_indexed:
            st.success(f"Index ready · {st.session_state.get('indexed_chunk_count', 0)} chunks")
        else:
            st.info("Waiting for indexing to complete.")
        st.caption(f"{len(current_files)} PDF{'s' if len(current_files) != 1 else ''} in this session")
        for uploaded in current_files:
            filename_col, size_col, status_col = st.columns([5, 1.3, 1.7])
            filename_col.markdown(f"**{html.escape(os.path.basename(uploaded.name))}**")
            size_col.caption(f"{uploaded.size / (1024 * 1024):.2f} MB")
            status_col.caption("Indexed" if is_indexed else "Queued")
        st.caption("To remove a file, use its remove control in the uploader above. Removing it clears the active retriever and conversation.")

elif selected_page == "Chat":
    if not uploaded_files:
        st.info("Upload and index PDF documents before starting a chat.")
    elif "retriever" not in st.session_state:
        st.info("Your documents are not indexed yet. Wait for indexing, then open Chat again.")
    else:
        retriever = st.session_state["retriever"]
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        st.markdown("### Conversation")
        if not st.session_state["messages"]:
            st.markdown('<div class="dm-card"><div class="dm-card-title">Ask about your documents</div><div class="dm-card-copy">Answers are grounded in retrieved passages. Expand each source card to check the filename, page, and excerpt.</div></div>', unsafe_allow_html=True)

        for role, stored_message in st.session_state["messages"]:
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(stored_message)
                continue
            parsed_message = split_assistant_message(stored_message)
            with st.chat_message("assistant"):
                st.markdown(parsed_message["answer"])
                if parsed_message["sources"]:
                    source_lines = [line.strip() for line in parsed_message["sources"].splitlines() if line.strip()]
                    source_cards = []
                    current_source = None
                    for line in source_lines:
                        if line.startswith("**[") and "]**" in line:
                            current_source = [line.replace("**", ""), ""]
                        elif line.startswith(">") and current_source:
                            current_source[1] = line[1:].strip()
                            source_cards.append(tuple(current_source))
                            current_source = None
                    if source_cards:
                        st.markdown("#### Sources")
                        for source_title, excerpt in source_cards:
                            with st.expander(source_title, expanded=False):
                                st.markdown(f"> {excerpt}")
                if parsed_message["debug_chunks"]:
                    with st.expander("Technical diagnostics · retrieved passages", expanded=False):
                        for chunk in parsed_message["debug_chunks"]:
                            st.markdown(chunk)
                if parsed_message["footer"]:
                    st.caption(f"Generated with {parsed_message['footer'].removeprefix('Generated with:').strip()}")

        with st.form("chat_form", clear_on_submit=True, border=True):
            st.text_area("Ask a question about your documents", key="query_input", height=90,
                         placeholder="Ask in Arabic or English…")
            submit_col, clear_col = st.columns([1, 1])
            submitted = submit_col.form_submit_button("Send question", type="primary", use_container_width=True)
            cleared = clear_col.form_submit_button("Clear conversation", use_container_width=True)
        if cleared:
            st.session_state["messages"] = []
            st.rerun()
        if submitted and st.session_state.get("query_input", "").strip():
            latest_query = st.session_state["query_input"].strip()
            st.session_state["messages"].append(("user", latest_query))
            log_event("Processing user query")
            success = False
            models_to_try = []
            all_errors = []
            skip_remaining_groq = False
            if use_groq and GROQ_AVAILABLE and groq_key:
                models_to_try.extend((name, "groq") for name in GROQ_MODELS)
            if OPENAI_AVAILABLE and openai_key:
                models_to_try.append(("gpt-3.5-turbo", "openai"))
            if not models_to_try:
                st.error("No language model is configured. Set a Groq or OpenAI API key.")
            else:
                with st.spinner("Searching your documents and preparing an answer…"):
                    for model_name, provider in models_to_try:
                        if provider == "groq" and skip_remaining_groq:
                            continue
                        try:
                            current_llm = (
                                get_groq_llm_instance(groq_key, model_name)
                                if provider == "groq"
                                else get_openai_llm_instance(openai_key, model_name)
                            )
                            answer, source_docs = answer_question(retriever, current_llm, latest_query, logger=log_event)
                            log_event(f"Returned {len(source_docs)} citation-mapped source chunks for query")
                            full_answer = answer + f"|||FOOTER:Generated with: {model_name}"
                            if source_docs:
                                full_answer += format_citations(source_docs) + format_debug_chunks(source_docs)
                            else:
                                log_event("No sources cited: the answer abstained or no valid citation marker was returned.", "WARNING")
                            st.session_state["messages"].append(("assistant", full_answer))
                            log_event(f"Generated answer using {model_name}")
                            success = True
                            break
                        except Exception as e:
                            error_msg = provider_error_summary(e)
                            all_errors.append(f"{model_name}: {error_msg}")
                            log_event(f"Model {model_name} failed ({type(e).__name__}): {error_msg}", "WARNING")
                            if provider == "groq" and provider_error_status(e) in (413, 429):
                                skip_remaining_groq = True
                if not success:
                    if all_errors:
                        st.error("All configured models failed. Review the diagnostic details below.")
                        with st.expander("Provider error details"):
                            for error in all_errors:
                                st.code(error, language="text")
                    else:
                        st.error("No model could answer. Check provider configuration.")
            st.rerun()

elif selected_page == "About":
    st.markdown("### About DocuMind AI")
    st.write("DocuMind AI answers questions using uploaded PDFs. PDF text and embeddings are prepared locally. Selected excerpts are sent to the configured language model when a question is answered, and citations identify the source file and page.")
    about1, about2 = st.columns(2)
    with about1:
        st.markdown("#### Retrieval-augmented generation")
        st.markdown("1. Extract PDF text and page metadata.\n2. Split and embed the text.\n3. Retrieve relevant passages from FAISS.\n4. Generate an answer with Groq or OpenAI and cite the evidence.")
        st.markdown("#### Architecture")
        st.markdown("Streamlit → PDF ingestion and chunking → HuggingFace embeddings → FAISS retrieval → Groq / OpenAI → citations")
    with about2:
        st.markdown("#### Technology")
        st.write("Python · Streamlit · LangChain · HuggingFace Sentence Transformers · FAISS · Groq · OpenAI (optional)")
        st.markdown("#### Project credits")
        st.write("DocuMind AI is developed and extended from the open-source Universal PDF RAG Chatbot by Ratnesh Singh. The original MIT license and copyright notice remain in the project LICENSE file.")
        st.caption("The current ingestion pipeline supports PDF files.")
    with st.expander("Developer notes"):
        st.write("Upload PDFs in Documents, wait for indexing, then ask a question in Chat. Expand source cards to review evidence. Use Rebuild document index to reprocess the current document set.")

elif selected_page == "Diagnostics":
    st.markdown("### System status")
    provider_name = "Groq" if use_groq and GROQ_AVAILABLE and has_groq else ("OpenAI" if OPENAI_AVAILABLE and has_openai else "Not configured")
    diag1, diag2, diag3 = st.columns(3)
    diag1.metric("Preferred provider", provider_name)
    diag2.metric("Embedding model", EMBEDDING_MODEL_NAME.split("/")[-1])
    diag3.metric("Retrieval Top-K", TOP_K)
    status1, status2 = st.columns(2)
    with status1:
        st.markdown("#### Provider configuration")
        for name, available, configured in (("Groq", GROQ_AVAILABLE, has_groq), ("OpenAI", OPENAI_AVAILABLE, has_openai)):
            state = "Integration unavailable" if not available else ("Key configured · live connection not verified" if configured else "API key not configured")
            (st.success if available and configured else st.warning)(f"{name}: {state}")
    with status2:
        st.markdown("#### Retrieval configuration")
        st.write(f"Chunk size: {CHUNK_SIZE} characters · overlap: {CHUNK_OVERLAP}")
        st.write(f"Top-K: {TOP_K} · prompt context budget: {MAX_PROMPT_TOKEN_BUDGET:,}")
        st.write(f"Session index: {st.session_state.get('indexed_chunk_count', 0)} chunks")

    st.markdown("### Application logs")
    if st.button("Refresh logs", use_container_width=False):
        st.rerun()
    log_content = ""
    try:
        if os.path.exists("app.log"):
            with open("app.log", "r", encoding="utf-8", errors="replace") as log_file:
                log_content = log_file.read()
    except OSError as error:
        st.warning(f"Could not read app.log ({type(error).__name__}).")
    if not log_content:
        log_content = "\n".join(st.session_state.get("app_logs", [])) or "No logs available."
    st.download_button("Download diagnostics log", log_content, file_name="documind_diagnostics.txt", mime="text/plain")
    with st.expander("View recent logs"):
        st.code("\n".join(log_content.splitlines()[-200:]), language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9eacc4; padding: 20px;'>
    <p>DocuMind AI · AI-Powered Document Intelligence</p>
    <p>Python · Streamlit · LangChain · HuggingFace · FAISS · Groq / OpenAI</p>
</div>
""", unsafe_allow_html=True)
