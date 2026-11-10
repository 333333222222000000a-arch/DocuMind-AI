"""Application configuration shared by the UI and RAG services."""

INDEX_DIR = "faiss_index_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ("\n\n", "\n", ".", "!", "?", ",", " ", "")
OCR_TEXT_THRESHOLD = 100
TOP_K = 10
RETRIEVAL_FETCH_K = 20
MMR_LAMBDA = 0.5
DUPLICATE_SIMILARITY_THRESHOLD = 0.98
MAX_PROMPT_TOKEN_BUDGET = 4000
PROMPT_TOKEN_SAFETY_MARGIN = 256
MAX_OUTPUT_TOKENS = 512
INDEX_SCHEMA_VERSION = 2

GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)

PROMPT_TEMPLATE = """
You answer questions using only the supplied document context.

Context from the uploaded documents:
{context}

Question: {input}

Instructions:
- Use only facts stated in the Context. Do not add outside knowledge or infer missing specifications.
- If the Context does not contain the answer, begin with `NO_ANSWER:` and state that the document does not specify it.
- Cite each factual claim with the matching source marker such as [S1]. Never invent a source marker.
- If passages conflict, explain the conflict and cite each passage instead of choosing silently.
- Answer in the same language as the question. Be concise unless more detail is requested.
- Identify a conclusion combined from multiple passages as a synthesis, not as a direct quotation.

Answer:
""".strip()
