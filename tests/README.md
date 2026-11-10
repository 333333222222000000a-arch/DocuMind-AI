# DocuMind-AI test notes

Run the active deterministic test suite from the repository root:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py" -v

`data/rag_eval_cases.json` contains small synthetic, labeled retrieval cases in
Arabic and English. `rag_app.evaluation` computes Precision@K, Recall@K, MRR,
source/page citation precision and recall, and the unanswerable retrieval rate.
Retrieval evaluation is independent of LLM calls. Unit tests use deterministic
retrievers and mocked chains; they never contact Groq or OpenAI.

The numeric grounding check is deliberately narrow: it flags numeric answer
claims absent from the retrieved text. It is not a full semantic entailment
metric and must not be treated as proof that every claim is supported.

`phase1a_test_app_baseline.py` and `phase1a_baseline/README.md` preserve the
original Phase 1A tests and notes unchanged. The full Streamlit application, PDF parsing,
embeddings, native FAISS operations, and remote provider calls require the
project's local dependencies. The ordinary test suite does not call remote
providers.
